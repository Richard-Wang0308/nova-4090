#!/usr/bin/env python3
"""
Top-k binder design selection by worst weighted metric rank (BoltzGen Algorithm 2).
Copied from nova/utils/minmax_weighted_rank.py for submit-time archive indexing.
"""

from typing import Optional

import bittensor as bt
import pandas as pd

METRIC_DEFS = {
    "design_iiptm": {"direction": "max"},
    "design_ptm": {"direction": "max"},
    "min_design_to_target_pae": {"direction": "min"},
    "plip_hbonds_refolded": {"direction": "max"},
    "plip_saltbridge_refolded": {"direction": "max"},
    "delta_sasa_refolded": {"direction": "max"},
}

PROTEIN_WEIGHTS = {
    "design_iiptm": 1.0,
    "design_ptm": 2.0,
    "min_design_to_target_pae": 1.0,
    "plip_hbonds_refolded": 2.0,
    "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 2.0,
}


def rank_binders(
    df: pd.DataFrame,
    weights: Optional[dict] = None,
    k: int = 50,
    max_liability_violations: Optional[int] = None,
    max_liability_score: Optional[float] = None,
) -> pd.DataFrame:
    if weights is None:
        weights = PROTEIN_WEIGHTS

    work = df.copy()

    n_before = len(work)
    if max_liability_violations is not None and "liability_num_violations" in work.columns:
        work = work[work["liability_num_violations"] <= max_liability_violations]
    if max_liability_score is not None and "liability_score" in work.columns:
        work = work[work["liability_score"] <= max_liability_score]
    n_after = len(work)
    if n_after < n_before:
        bt.logging.debug(
            f"Liability filter: {n_before} → {n_after} designs ({n_before - n_after} removed)"
        )

    if n_after == 0:
        bt.logging.warning("No designs survived liability filtering.")
        return pd.DataFrame()

    missing = [m for m in weights if m not in work.columns]
    if missing:
        bt.logging.warning(f"Missing columns for rank_binders: {missing}")
        return pd.DataFrame()

    rank_cols = {}
    for metric, w in weights.items():
        ascending = METRIC_DEFS[metric]["direction"] == "min"
        raw_rank = work[metric].rank(ascending=ascending, method="average")
        weighted_rank = raw_rank / w
        col_name = f"wrank_{metric}"
        rank_cols[col_name] = weighted_rank

    rank_df = pd.DataFrame(rank_cols, index=work.index)

    work["algo2_score"] = rank_df.max(axis=1)
    work["bottleneck_metric"] = rank_df.idxmax(axis=1).str.replace("wrank_", "", regex=False)

    for col in rank_df.columns:
        work[col] = rank_df[col]

    work = work.sort_values("algo2_score", ascending=True).reset_index(drop=True)
    work.index.name = "selection_rank"
    work.index += 1

    return work.head(k)
