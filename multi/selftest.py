"""
multi/selftest.py — prove the pool works before trusting a mining run to it.

    python3 -m multi.selftest              # 4 molecules, real Boltz, ~2 min
    python3 -m multi.selftest --n 8

Checks, in order:
  1. GPUs are detected and a worker plan is produced.
  2. A worker process starts and constructs a real BoltzWrapper.
  3. Molecules come back scored, with the finite values the score formula needs.
  4. MultiGPUBoltz fills score_dict["molecule_scores"] in INPUT ORDER, which is
     the one property every call site depends on.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (BASE_DIR, os.path.join(BASE_DIR, "miner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Small, valid, structurally distinct ligands. Real molecules from this repo's
# own scored set, so a wrong answer is obvious.
SAMPLE = [
    ("probe-1", "Cc1cncc(-c2cc(Cl)cc(C3OCCO3)c2F)c1F"),
    ("probe-2", "CCNS(=O)(=O)c1ccc(-c2cc(C)n3ccnc(C#N)c23)c(C(F)(F)F)c1"),
    ("probe-3", "N#Cc1csc(-c2cncc(OC3CCOC3)n2)c1"),
    ("probe-4", "CCc1n[nH]c2ncc(-c3cc(C)cnc3C#N)cc12"),
    ("probe-5", "COC(=O)c1cc(C#N)cc(-c2ccc(Cl)c3cn[nH]c23)c1Cl"),
    ("probe-6", "O=S(=O)(c1cc(-c2c[nH]c3nc(Cl)cnc23)ccc1F)N1CCCC1"),
    ("probe-7", "Cc1cc(-c2cccc3[nH]c(Cl)nc23)ccc1S(=O)(=O)N1CCN(C)CC1"),
    ("probe-8", "Cc1c(F)cc2[nH]ncc2c1-c1cncc(-c2cncs2)c1"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        stream=sys.stdout)

    from config.config_loader import load_config
    from multi import MultiGPUBoltz, describe

    print(f"[1/4] topology: {describe()}")

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

    mols = SAMPLE[:max(1, min(args.n, len(SAMPLE)))]
    boltz = MultiGPUBoltz(workers_per_gpu=args.workers_per_gpu)
    print(f"[2/4] starting {boltz.pool.n_workers} worker(s)")

    vm = {0: {"smiles": [s for _, s in mols], "names": [n for n, _ in mols]}}
    sd = {0: {"target_scores": [[]], "antitarget_scores": [[]], "entropy": None,
              "entropy_boltz": None, "block_submitted": None, "push_time": ""}}

    t0 = time.time()
    boltz.score_molecules(vm, sd, subnet)
    dt = time.time() - t0

    scores = sd[0].get("molecule_scores", [[]])[0]
    finite = [s for s in scores if isinstance(s, float) and math.isfinite(s)]
    print(f"[3/4] scored {len(finite)}/{len(mols)} in {dt:.0f}s "
          f"({dt / max(len(mols), 1):.1f} s/molecule wall)")
    for (name, smi), s in zip(mols, scores):
        print(f"        {name:<9} {s if not isinstance(s, float) else f'{s:.6f}':>12}  {smi[:52]}")

    ok = True
    if len(scores) != len(mols):
        print(f"    FAIL: molecule_scores has {len(scores)} entries for {len(mols)} inputs")
        ok = False
    if not finite:
        print("    FAIL: nothing scored — check the worker traceback above")
        ok = False

    # Order is the contract orchestrator.py relies on explicitly.
    target = subnet["small_molecule_target"][0]
    fmap = boltz.final_boltz_scores.get(0, {}).get(target, {})
    in_order = [fmap.get(smi) for _, smi in mols]
    if in_order != list(scores):
        print("    FAIL: molecule_scores is not in input SMILES order")
        ok = False
    else:
        print("[4/4] molecule_scores is in input order and matches final_boltz_scores")

    boltz.shutdown()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
