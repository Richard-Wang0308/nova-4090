"""
sort_components.py — Rank every A component by "cost" using a surrogate
model, without calling Boltz at all (pure surrogate-based ranking).

Cost definition for a given A component (a_id):
  1. Fix A = a_id
  2. Randomly sample --sample_size (default 1000) B ids from the ENTIRE
     B library (before any validation — this is the expensive step we're
     avoiding doing on the full library)
  3. Build those --sample_size candidate molecules rxn:{rxn_id}:{a_id}:{b_id}
  4. Validate them (heavy atoms / banned atoms / rotatable bonds / RDKit
     parse) and DROP invalid ones
  5. Surrogate-predict scores for the remaining valid molecules
  6. cost(a_id) = mean of the top --top_k (default 10) surrogate scores
     among those valid molecules

All A components are then sorted by cost (descending) and written to
--output (default sort_out.txt). The file is (re)written every
--dump_every (default 500) newly-processed A components, so you can
tail -f / open the file mid-run to see live progress — plus a final
write once every component has been processed.

Usage:
    python sort_components.py --rxn_id 2
    python sort_components.py --rxn_id 2 --sample_size 1000 --top_k 10 --dump_every 500
    python sort_components.py --rxn_id 2 --limit 200          # quick test on first 200 A ids
"""
import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))  # so we can import component_exhaust

from config.config_loader import load_config
from molecules import MoleculeManager

# ── Reuse existing surrogate / validation logic ────────────────────────────
from component_exhaust import (
    SurrogateModel,
    build_and_validate,
    score_db_path,
    DB_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Cost computation for a single A component (sample-FIRST, validate-SECOND)
# ═══════════════════════════════════════════════════════════════════════════
def compute_cost_for_component(
    rxn_id: int,
    manager: MoleculeManager,
    config: Dict,
    surrogate: SurrogateModel,
    a_id,
    sample_size: int,
    top_k: int,
    rng: random.Random,
) -> Dict:
    """
    Fix A=a_id. Randomly sample `sample_size` B ids from the FULL B
    library BEFORE validating anything (this is the speed fix — we never
    touch the full library with RDKit). Validate only the sampled
    candidates, drop invalid ones, surrogate-score the survivors, and
    return cost = mean of the top `top_k` surrogate scores among them.
    """
    b_pool = list(manager.moles_B_id)
    n_total_b = len(b_pool)

    if n_total_b == 0:
        return {"a_id": a_id, "n_sampled": 0, "n_valid": 0, "cost": np.nan}

    # ── 1. Sample RAW candidate B ids first (no validation yet) ─────────
    k_sample = min(sample_size, n_total_b)
    sampled_b_ids = rng.sample(b_pool, k_sample)

    names = [f"rxn:{rxn_id}:{a_id}:{b_id}" for b_id in sampled_b_ids]

    # ── 2. Validate ONLY the sampled candidates ─────────────────────────
    valid_df = build_and_validate(names, config)
    n_valid = len(valid_df)

    if n_valid == 0:
        return {"a_id": a_id, "n_sampled": k_sample, "n_valid": 0, "cost": np.nan}

    # ── 3. Surrogate-score the valid survivors ──────────────────────────
    preds = surrogate.predict(valid_df["smiles"].tolist())

    # ── 4. cost = mean of top_k surrogate scores ─────────────────────────
    k = min(top_k, len(preds))
    if k == 0:
        cost = np.nan
    else:
        top_scores = np.sort(preds)[::-1][:k]
        cost = float(np.mean(top_scores))

    return {
        "a_id": a_id,
        "n_sampled": k_sample,
        "n_valid": n_valid,
        "cost": cost,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Write / overwrite sort_out.txt with current ranking
# ═══════════════════════════════════════════════════════════════════════════
def dump_results(
    results: List[Dict],
    output_path: str,
    rxn_id: int,
    elapsed: float,
    total: int,
    sample_size: int,
    top_k: int,
) -> None:
    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values("cost", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    header = (
        f"{'='*70}\n"
        f"A-COMPONENT COST RANKING — rxn_id={rxn_id}\n"
        f"Processed:    {len(results)} / {total} components\n"
        f"Elapsed:      {elapsed:.1f}s\n"
        f"Sample size:  random {sample_size} raw candidates tested per A "
        f"component (invalid ones dropped before scoring)\n"
        f"Top-k:        {top_k} (cost = mean of top-{top_k} surrogate scores "
        f"among the valid survivors)\n"
        f"Generated:    {pd.Timestamp.now()}\n"
        f"{'='*70}\n\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        f.write(df.to_string(index=False))
        f.write("\n")

    logger.info(
        f"[Dump] 💾 Wrote {len(results)}/{total} ranked A components → {output_path}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI + main
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank all A components by surrogate-based cost (no Boltz calls)."
    )
    parser.add_argument("--rxn_id", type=int, required=True)
    parser.add_argument(
        "--surrogate_top_n", type=int, default=4000,
        help="Top-N scored molecules used to train the surrogate (default 4000).",
    )
    parser.add_argument(
        "--surrogate_bottom_n", type=int, default=4000,
        help="Bottom-N scored molecules used to train the surrogate (default 4000).",
    )
    parser.add_argument(
        "--sample_size", type=int, default=1000,
        help="Random RAW candidates sampled (before validation) per A "
             "component (default 1000). Invalid ones are dropped after.",
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Cost = mean of the top-k surrogate scores among valid "
             "survivors (default 10).",
    )
    parser.add_argument(
        "--dump_every", type=int, default=500,
        help="Rewrite the output file every N newly-processed A components (default 500).",
    )
    parser.add_argument(
        "--output", type=str, default="sort_out.txt",
        help="Output file path (default sort_out.txt).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling reproducibility (default 42).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional: only process the first N A components (for quick testing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rxn_id = args.rxn_id

    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    if manager.is_three_component:
        raise NotImplementedError(
            f"rxn={rxn_id} is a 3-component reaction. This script currently "
            f"only supports ranking A components for 2-component (A:B) "
            f"reactions, since 'cost' is defined by sampling B alone. "
            f"For 3-component reactions you'd need to decide how to combine "
            f"B and C sampling — let me know if you want that extended."
        )

    db_path = score_db_path(rxn_id)
    if not os.path.exists(db_path):
        logger.error(f"❌ {db_path} does not exist — cannot train surrogate. Aborting.")
        return

    # ── 1. Train surrogate on top-N + bottom-N scored molecules ──────────
    surrogate = SurrogateModel()
    surrogate.train_from_db(
        rxn_id, db_path,
        top_n=args.surrogate_top_n,
        bottom_n=args.surrogate_bottom_n,
    )

    if not surrogate.is_trained:
        logger.error("❌ Surrogate training failed (too few samples?) — aborting.")
        return

    # ── 2. Enumerate all A components ─────────────────────────────────────
    a_ids = list(manager.moles_A_id)
    if args.limit is not None:
        a_ids = a_ids[: args.limit]
    total = len(a_ids)
    logger.info(f"[Sort] Found {total} A components to evaluate")

    rng = random.Random(args.seed)
    results: List[Dict] = []
    t_start = time.time()

    for i, a_id in enumerate(a_ids, start=1):
        t0 = time.time()
        res = compute_cost_for_component(
            rxn_id, manager, cfg, surrogate, a_id,
            sample_size=args.sample_size,
            top_k=args.top_k,
            rng=rng,
        )
        results.append(res)
        dt = time.time() - t0

        cost_str = f"{res['cost']:.4f}" if not np.isnan(res["cost"]) else "N/A"
        logger.info(
            f"[{i}/{total}] A={a_id} | sampled={res['n_sampled']:>5} "
            f"valid={res['n_valid']:>5} | cost={cost_str} ({dt:.2f}s)"
        )

        # ── Every `dump_every` newly-processed A components, rewrite file ──
        if i % args.dump_every == 0:
            dump_results(
                results, args.output, rxn_id,
                time.time() - t_start, total,
                args.sample_size, args.top_k,
            )

    # ── Final write (covers any remainder < dump_every) ──────────────────
    dump_results(
        results, args.output, rxn_id,
        time.time() - t_start, total,
        args.sample_size, args.top_k,
    )

    logger.info(
        f"✅ Done. Ranked {len(results)}/{total} A components → {args.output} "
        f"(total time: {time.time()-t_start:.1f}s)"
    )


if __name__ == "__main__":
    main()