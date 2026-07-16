"""
sort_components.py — Rank A components by surrogate-based cost for a selected
1-based index range.

IMPORTANT:
  --start_index and --end_index are POSITIONS in manager.moles_A_id,
  NOT molecule/component IDs.

Example:
    If manager.moles_A_id = [73951, 12003, 888991, ...]
    then:
        index 1 -> A id 73951
        index 2 -> A id 12003
        index 3 -> A id 888991

Cost definition for one A component:
  1. Fix A = a_id
  2. Randomly sample --sample_size B components from the full B library
     BEFORE validation
  3. Build candidate molecules:
         rxn:{rxn_id}:{a_id}:{b_id}
  4. Validate only those sampled molecules
  5. Drop invalid molecules
  6. Surrogate-score valid survivors
  7. cost(a_id) = average of top --top_k surrogate scores

Output:
  - Results are sorted by cost descending
  - Written to --output, default sort_out.txt
  - File is rewritten every --dump_every processed A components
  - Final file is written at the end

Usage:
    python3 neurons/sort_components.py --rxn_id 2 --start_index 1 --end_index 500

    python3 neurons/sort_components.py \\
        --rxn_id 2 \\
        --start_index 1001 \\
        --end_index 2000 \\
        --sample_size 1000 \\
        --top_k 10 \\
        --dump_every 500

Notes:
  - index range is inclusive: [start_index, end_index]
  - valid index range should be [1, number_of_A_components]
  - for your reaction 2, you said this is [1, 83307]
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))

from config.config_loader import load_config
from molecules import MoleculeManager

# Reuse surrogate + validation helpers from component_exhaust.py
from component_exhaust import (
    SurrogateModel,
    build_and_validate,
    score_db_path,
    DB_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: safe float formatting
# ═══════════════════════════════════════════════════════════════════════════
def fmt_float(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "N/A"
    try:
        if np.isnan(x):
            return "N/A"
    except TypeError:
        return "N/A"
    return f"{float(x):.{digits}f}"


# ═══════════════════════════════════════════════════════════════════════════
# Cost computation for one A component
# ═══════════════════════════════════════════════════════════════════════════
def compute_cost_for_component(
    rxn_id: int,
    manager: MoleculeManager,
    config: Dict,
    surrogate: SurrogateModel,
    a_id,
    component_index: int,
    sample_size: int,
    top_k: int,
    rng: random.Random,
) -> Dict:
    """
    Compute surrogate-based cost for one A component.

    component_index:
        1-based position in manager.moles_A_id.
        This is NOT the molecule/component ID.
    """
    b_pool = list(manager.moles_B_id)
    n_total_b = len(b_pool)

    if n_total_b == 0:
        return {
            "component_index": component_index,
            "a_id": a_id,
            "n_total_b": 0,
            "n_sampled": 0,
            "n_valid": 0,
            "top_k_used": 0,
            "cost": np.nan,
        }

    # ── 1. Randomly sample raw B ids first, before validation ────────────
    k_sample = min(sample_size, n_total_b)
    sampled_b_ids = rng.sample(b_pool, k_sample)

    # ── 2. Build reaction molecule names for sampled candidates only ────
    names = [f"rxn:{rxn_id}:{a_id}:{b_id}" for b_id in sampled_b_ids]

    # ── 3. Validate only sampled molecules; invalid molecules are erased ─
    valid_df = build_and_validate(names, config)
    n_valid = len(valid_df)

    if n_valid == 0:
        return {
            "component_index": component_index,
            "a_id": a_id,
            "n_total_b": n_total_b,
            "n_sampled": k_sample,
            "n_valid": 0,
            "top_k_used": 0,
            "cost": np.nan,
        }

    # ── 4. Surrogate-score valid molecules only ─────────────────────────
    preds = surrogate.predict(valid_df["smiles"].tolist())

    # ── 5. cost = average top-k surrogate scores ────────────────────────
    k = min(top_k, len(preds))
    if k == 0:
        cost = np.nan
    else:
        top_scores = np.sort(preds)[::-1][:k]
        cost = float(np.mean(top_scores))

    return {
        "component_index": component_index,
        "a_id": a_id,
        "n_total_b": n_total_b,
        "n_sampled": k_sample,
        "n_valid": n_valid,
        "top_k_used": k,
        "cost": cost,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dump sorted output
# ═══════════════════════════════════════════════════════════════════════════
def dump_results(
    results: List[Dict],
    output_path: str,
    rxn_id: int,
    elapsed: float,
    start_index: int,
    end_index: int,
    total_a_components: int,
    processed: int,
    selected_total: int,
    sample_size: int,
    top_k: int,
) -> None:
    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values("cost", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "rank_in_selected_range", range(1, len(df) + 1))

    header = (
        f"{'=' * 90}\n"
        f"A-COMPONENT COST RANKING — rxn_id={rxn_id}\n"
        f"{'=' * 90}\n"
        f"Index meaning: component_index is 1-based position in manager.moles_A_id, NOT molecule ID\n"
        f"Full A index range:      [1, {total_a_components}]\n"
        f"Selected index range:    [{start_index}, {end_index}] inclusive\n"
        f"Selected components:     {selected_total}\n"
        f"Processed selected:      {processed} / {selected_total}\n"
        f"Elapsed:                 {elapsed:.1f}s\n"
        f"Random raw B sample:     {sample_size} per A component before validation\n"
        f"Invalid molecules:       dropped before surrogate scoring\n"
        f"Cost definition:         mean(top-{top_k} surrogate scores among valid sampled molecules)\n"
        f"Generated:               {pd.Timestamp.now()}\n"
        f"{'=' * 90}\n\n"
    )

    columns = [
        "rank_in_selected_range",
        "component_index",
        "a_id",
        "cost",
        "n_valid",
        "n_sampled",
        "top_k_used",
        "n_total_b",
    ]

    with open(output_path, "w") as f:
        f.write(header)
        f.write(df[columns].to_string(index=False))
        f.write("\n")

    logger.info(
        f"[Dump] 💾 Wrote sorted results for {processed}/{selected_total} "
        f"selected A components → {output_path}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sort A components by surrogate-based cost within a selected "
            "1-based index range. Index is NOT molecule ID."
        )
    )

    parser.add_argument("--rxn_id", type=int, default=2)

    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
        help="1-based start index in manager.moles_A_id. Inclusive. NOT molecule ID.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        required=True,
        help="1-based end index in manager.moles_A_id. Inclusive. NOT molecule ID.",
    )

    parser.add_argument(
        "--surrogate_top_n",
        type=int,
        default=4000,
        help="Top-N scored molecules used to train surrogate. Default 4000.",
    )
    parser.add_argument(
        "--surrogate_bottom_n",
        type=int,
        default=4000,
        help="Bottom-N scored molecules used to train surrogate. Default 4000.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help=(
            "Number of raw B candidates sampled before validation per A component. "
            "Default 1000."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Cost = mean of top-k surrogate scores. Default 10.",
    )
    parser.add_argument(
        "--dump_every",
        type=int,
        default=500,
        help="Rewrite output every N processed A components. Default 500.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sort_out.txt",
        help="Output file path. Default sort_out.txt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default 42.",
    )
    parser.add_argument(
        "--shuffle_a",
        action="store_true",
        help=(
            "Optional: shuffle selected A components before processing. "
            "Output still keeps original component_index."
        ),
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    rxn_id = args.rxn_id

    if args.start_index < 1:
        raise ValueError("--start_index must be >= 1")
    if args.end_index < args.start_index:
        raise ValueError("--end_index must be >= --start_index")
    if args.sample_size < 1:
        raise ValueError("--sample_size must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")
    if args.dump_every < 1:
        raise ValueError("--dump_every must be >= 1")

    # ── Load config and molecule manager ────────────────────────────────
    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"

    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    if manager.is_three_component:
        raise NotImplementedError(
            f"rxn={rxn_id} is 3-component. This script currently supports "
            f"2-component reaction ranking only: fix A, randomly sample B."
        )

    # ── Load A ids and convert index range to Python slice ──────────────
    all_a_ids = list(manager.moles_A_id)
    total_a_components = len(all_a_ids)

    if args.end_index > total_a_components:
        raise ValueError(
            f"--end_index={args.end_index} is too large. "
            f"Available A component index range is [1, {total_a_components}]."
        )

    # 1-based inclusive index range -> 0-based Python slice
    selected_pairs = [
        (idx_1based, all_a_ids[idx_1based - 1])
        for idx_1based in range(args.start_index, args.end_index + 1)
    ]

    selected_total = len(selected_pairs)

    rng = random.Random(args.seed)
    if args.shuffle_a:
        rng.shuffle(selected_pairs)

    logger.info(
        f"[Init] rxn={rxn_id} | total A components={total_a_components} | "
        f"selected index range=[{args.start_index}, {args.end_index}] | "
        f"selected_total={selected_total}"
    )
    logger.info(
        f"[Init] First selected pair: index={selected_pairs[0][0]}, "
        f"A id={selected_pairs[0][1]}"
    )
    logger.info(
        f"[Init] Last selected pair: index={selected_pairs[-1][0]}, "
        f"A id={selected_pairs[-1][1]}"
    )

    # ── Train surrogate from score_results_{rxn_id}.sqlite ──────────────
    db_path = score_db_path(rxn_id)
    if not os.path.exists(db_path):
        logger.error(f"❌ Score DB does not exist: {db_path}")
        return

    surrogate = SurrogateModel()
    surrogate.train_from_db(
        rxn_id=rxn_id,
        db_path=db_path,
        top_n=args.surrogate_top_n,
        bottom_n=args.surrogate_bottom_n,
    )

    if not surrogate.is_trained:
        logger.error("❌ Surrogate training failed. Aborting.")
        return

    # ── Process selected A index range ──────────────────────────────────
    results: List[Dict] = []
    t_start = time.time()

    for processed_i, (component_index, a_id) in enumerate(selected_pairs, start=1):
        t0 = time.time()

        res = compute_cost_for_component(
            rxn_id=rxn_id,
            manager=manager,
            config=cfg,
            surrogate=surrogate,
            a_id=a_id,
            component_index=component_index,
            sample_size=args.sample_size,
            top_k=args.top_k,
            rng=rng,
        )
        results.append(res)

        dt = time.time() - t0

        logger.info(
            f"[{processed_i}/{selected_total}] "
            f"index={component_index} | A_id={a_id} | "
            f"sampled={res['n_sampled']} | valid={res['n_valid']} | "
            f"top_k_used={res['top_k_used']} | cost={fmt_float(res['cost'])} | "
            f"{dt:.2f}s"
        )

        # Write sorted partial output every dump_every components
        if processed_i % args.dump_every == 0:
            dump_results(
                results=results,
                output_path=args.output,
                rxn_id=rxn_id,
                elapsed=time.time() - t_start,
                start_index=args.start_index,
                end_index=args.end_index,
                total_a_components=total_a_components,
                processed=processed_i,
                selected_total=selected_total,
                sample_size=args.sample_size,
                top_k=args.top_k,
            )

    # Final write
    dump_results(
        results=results,
        output_path=args.output,
        rxn_id=rxn_id,
        elapsed=time.time() - t_start,
        start_index=args.start_index,
        end_index=args.end_index,
        total_a_components=total_a_components,
        processed=selected_total,
        selected_total=selected_total,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )

    logger.info(
        f"✅ Done. Sorted {selected_total} A components from index "
        f"[{args.start_index}, {args.end_index}] → {args.output}. "
        f"Total time: {time.time() - t_start:.1f}s"
    )


if __name__ == "__main__":
    main()