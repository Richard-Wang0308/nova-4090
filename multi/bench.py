"""
multi/bench.py — measure the best workers-per-GPU for THIS box, and record it.

    python3 -m multi.bench                 # tries 1 and 2 workers per GPU
    python3 -m multi.bench --max 4 --n 24

Writes multi/calibration.json, which topology.plan_workers() then uses instead
of its default. Two numbers are recorded:

    workers_per_gpu      the setting with the highest molecules/minute
    peak_worker_rss_mib  the largest RSS any worker reached, which feeds the
                         host-RAM cap in plan_workers

WHY THIS NEEDS MEASURING RATHER THAN REASONING
----------------------------------------------
The default of 2 comes from one observation on one box: boltz leaves the GPU
idle for ~22 s of every ~100 s call (checkpoint loading and manifest checking),
so a second worker has gaps to fill. How much a third or fourth adds depends on
the ratio of that fixed cost to per-molecule compute, which moves with card,
ligand size, and how much else is on the machine. Measuring takes a few minutes
and removes the guess.

RUN IT WHEN THE BOX IS OTHERWISE IDLE. Sharing a GPU with a live searcher does
not just add noise, it can take the machine out of RAM -- which is exactly how
the hunter on this box got OOM-killed mid-round while this package was being
written.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (BASE_DIR, os.path.join(BASE_DIR, "miner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multi import topology                      # noqa: E402
from multi.pool import BoltzPool                 # noqa: E402
from multi.selftest import SAMPLE                # noqa: E402

log = logging.getLogger("multi.bench")


def _rss_mib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm", "r", encoding="utf-8") as f:
            return int(f.read().split()[1]) * 4096 // (1024 * 1024)
    except Exception:
        return 0


def _molecules(n: int):
    """n molecules, repeating the probe set with a distinct name each time.

    De-duplication in BoltzPool.score is on SMILES, so the set is cycled only up
    to its length; asking for more than that is capped rather than silently
    scoring the same molecule twice.
    """
    return SAMPLE[:min(n, len(SAMPLE))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2,
                    help="highest workers-per-GPU to try (default 2)")
    ap.add_argument("--n", type=int, default=8,
                    help="molecules per trial (default 8, the probe set size)")
    ap.add_argument("--out", default=topology.CALIBRATION_PATH)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    from config.config_loader import load_config
    cfg = load_config()
    subnet = {
        "small_molecule_target": cfg["small_molecule_target"],
        "small_molecule_target_clip_interval": cfg["small_molecule_target_clip_interval"],
        "boltz_mode": cfg.get("boltz_mode", "max"),
        "boltz_metric": cfg.get("boltz_metric",
                                ["affinity_probability_binary", "affinity_pred_value"]),
        "combination_strategy": cfg.get("combination_strategy",
                                        "heavy_atom_normalization"),
    }
    mols = _molecules(args.n)
    n_gpus = max(1, len(topology.detect_gpus()))

    results = []
    peak_rss = 0
    for wpg in range(1, max(1, args.max) + 1):
        plan = topology.plan_workers(workers_per_gpu=wpg)
        if len(plan) < wpg * n_gpus:
            log.warning("workers_per_gpu=%d was capped to %d worker(s) by "
                        "VRAM/RAM; stopping the sweep here", wpg, len(plan))
            if not plan:
                break
        pool = BoltzPool(plan=plan, logger=log)
        try:
            pool.start()
            t0 = time.time()
            scores = pool.score(mols, subnet)
            dt = time.time() - t0
            for p in pool.procs.values():
                peak_rss = max(peak_rss, _rss_mib(p.pid))
            rate = 60.0 * len(scores) / dt if dt > 0 else 0.0
            results.append({"workers_per_gpu": wpg, "workers": len(plan),
                            "scored": len(scores), "seconds": round(dt, 1),
                            "molecules_per_min": round(rate, 2)})
            log.info("workers_per_gpu=%d (%d workers): %d scored in %.0fs "
                     "= %.2f molecules/min", wpg, len(plan), len(scores), dt, rate)
        finally:
            pool.shutdown()
            time.sleep(5)      # let VRAM actually come back before the next trial

    if not results:
        log.error("no trial completed; nothing written")
        return 1

    best = max(results, key=lambda r: r["molecules_per_min"])
    blob = {
        "workers_per_gpu": best["workers_per_gpu"],
        "peak_worker_rss_mib": peak_rss or None,
        "n_gpus": n_gpus,
        "molecules": len(mols),
        "trials": results,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)

    print()
    print(f"{'workers/gpu':>12}{'workers':>9}{'scored':>8}{'seconds':>9}{'mol/min':>10}")
    for r in results:
        mark = "  <- best" if r is best else ""
        print(f"{r['workers_per_gpu']:>12}{r['workers']:>9}{r['scored']:>8}"
              f"{r['seconds']:>9.0f}{r['molecules_per_min']:>10.2f}{mark}")
    if len(results) > 1:
        base = results[0]["molecules_per_min"]
        if base > 0:
            print(f"\nbest is {best['molecules_per_min'] / base:.2f}x the "
                  f"one-worker-per-GPU baseline")
    print(f"\npeak worker RSS: {peak_rss} MiB")
    print(f"written to {args.out} — plan_workers() will use it automatically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
