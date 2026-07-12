"""
mode3_input.py — Mode 3: Input.

Reads an external rxn{ID}.csv (molecule_name, final_score, epoch),
merges it into score_results_{ID}.sqlite using merge_scores_keep_max
(keep max(existing_score, new_score) per molecule_name, per user's
confirmed policy). Runs ONCE and exits (not a continuous loop).
"""

import os
import argparse
import pandas as pd
import numpy as np

import common
from common import logger, init_score_results_db, merge_scores_keep_max
from config.config_loader import load_config


def load_input_csv(csv_path: str, rxn_id: int) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, header=0)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'final_score' in df.columns and 'score' not in df.columns:
        df = df.rename(columns={'final_score': 'score'})
    df['molecule_name'] = df['molecule_name'].astype(str).str.strip().str.lstrip('\ufeff')

    prefix = f"rxn:{rxn_id}:"
    df = df[df['molecule_name'].str.startswith(prefix, na=False)].reset_index(drop=True)

    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df.loc[~np.isfinite(df['score']), 'score'] = np.nan
    df = df[df['score'].notna()]

    logger.info(f"[Input] Loaded {len(df)} rows for rxn={rxn_id} from {os.path.basename(csv_path)}")
    return df[['molecule_name', 'score']]


def run_input_merge(rxn_id: int, input_csv_path: str) -> None:
    init_score_results_db()
    df = load_input_csv(input_csv_path, rxn_id)
    if df.empty:
        logger.warning(f"[Input] No matching rows for rxn={rxn_id} — nothing to merge")
        return

    rows = list(zip(df['molecule_name'].tolist(), df['score'].tolist()))
    n_new, n_upd = merge_scores_keep_max(rows)

    logger.info(
        f"[Input] Merge complete for rxn={rxn_id}: "
        f"{n_new} new molecules inserted, {n_upd} existing molecules "
        f"updated (new score was higher), "
        f"{len(rows) - n_new - n_upd} unchanged (existing score was >= new)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# standalone entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Mode 3 — Input (merge external CSV)")
    parser.add_argument("--rxn_id", type=int, required=True)
    parser.add_argument(
        "--input_csv", type=str, default=None,
        help="Path to external rxn{ID}.csv. Defaults to data/rxn{ID}.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    common.configure_for_rxn(args.rxn_id)
    input_csv = args.input_csv or os.path.join(common.BASE_DIR, "data", f"rxn{args.rxn_id}.csv")
    run_input_merge(args.rxn_id, input_csv)


if __name__ == "__main__":
    main()
