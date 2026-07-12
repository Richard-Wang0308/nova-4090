"""
mode4_cross.py — Mode 4: Cross.

2-reactant (A:B): top-25 A component values × top-20 B component values
from current top molecules in the DB → build all 500 combinations →
validate → Boltz-score all → merge into DB (keep-max policy).

3-reactant (A:B:C): top-8 A × top-10 B × top-10 C = 800 combos.

Prints every step (anchors found, candidates built, dedup counts,
scoring progress, final results table) directly to terminal with
flush=True so nothing is delayed by output buffering.
"""

import os
import sys
import time
import asyncio
import argparse
import itertools
from typing import List

import pandas as pd
import numpy as np

import common
from common import (
    logger, MoleculeUtils, score_molecules_with_boltz_batched,
    get_all_scored_from_db, parse_molecule_name, build_molecule_name,
    merge_scores_keep_max,
)
from config.config_loader import load_config

TOP_A_2R = 25
TOP_B_2R = 20

TOP_A_3R = 8
TOP_B_3R = 10
TOP_C_3R = 10


def p(msg: str) -> None:
    """Print immediately to terminal, no buffering delay."""
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Top-component discovery
# ═══════════════════════════════════════════════════════════════════════════

def get_top_component_values(df: pd.DataFrame, role_idx: int, n: int) -> List[int]:
    seen = []
    for name in df["name"]:
        ids = parse_molecule_name(name)
        if ids is None or role_idx >= len(ids):
            continue
        val = ids[role_idx]
        if val not in seen:
            seen.append(val)
        if len(seen) >= n:
            break
    return seen


# ═══════════════════════════════════════════════════════════════════════════
# Results printer
# ═══════════════════════════════════════════════════════════════════════════

def print_results(scored_df: pd.DataFrame, prev_best_score: float = None) -> None:
    if scored_df.empty:
        p("\n[Cross][RESULTS] ⚠️  No new molecules were scored this pass.\n")
        return

    ranked = scored_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    p("\n" + "=" * 78)
    p(f"[Cross][RESULTS] {len(ranked)} newly scored molecule(s) this pass")
    p("=" * 78)
    p(f"{'Rank':<6}{'Score':<12}{'Molecule Name':<45}{'Flag'}")
    p("-" * 78)

    new_best_found = False
    for i, row in ranked.iterrows():
        flag = ""
        if prev_best_score is not None and row["score"] > prev_best_score:
            flag = "🏆 NEW BEST!"
            new_best_found = True
        p(f"{i+1:<6}{row['score']:<12.6f}{row['name']:<45}{flag}")

    p("-" * 78)
    p(f"Best this pass : {ranked.iloc[0]['name']}  (score={ranked.iloc[0]['score']:.6f})")
    if prev_best_score is not None:
        p(f"Previous best  : score={prev_best_score:.6f}")
    if new_best_found:
        p("🏆🏆🏆  NEW ALL-TIME BEST FOUND THIS PASS  🏆🏆🏆")
    p("=" * 78 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Build + validate + score a batch of candidate names
# ═══════════════════════════════════════════════════════════════════════════

async def _build_validate_score(
    state, manager, rxn_id: int, id_tuples: List[tuple], boltz_batch_size: int,
) -> pd.DataFrame:
    names = [build_molecule_name(rxn_id, ids) for ids in id_tuples]
    df = pd.DataFrame({"name": names}).drop_duplicates(subset=["name"])
    p(f"[Cross] Step 1/5: {len(df)} raw combinations built")

    df["smiles"] = df["name"].apply(MoleculeUtils.get_smiles_from_reaction_cached)
    df = df[df["smiles"].notna() & (df["smiles"] != "")].reset_index(drop=True)
    p(f"[Cross] Step 2/5: {len(df)} resolved to valid SMILES")

    if df.empty:
        p("[Cross] ⚠️  No valid SMILES resolved — aborting this pass.")
        return pd.DataFrame(columns=["name", "smiles", "score"])

    try:
        config = state['config']
        df = manager.validate_molecules(config, df)
        p(f"[Cross] Step 3/5: {len(df)} passed validation")
    except Exception as e:
        p(f"[Cross] ⚠️  validate_molecules skipped ({e})")

    if df.empty:
        p("[Cross] ⚠️  Nothing passed validation — aborting this pass.")
        return pd.DataFrame(columns=["name", "smiles", "score"])

    existing = get_all_scored_from_db()
    if not existing.empty:
        existing_names = set(existing["name"].tolist())
        pre = len(df)
        df  = df[~df["name"].isin(existing_names)].reset_index(drop=True)
        p(f"[Cross] Step 4/5: dedup vs DB → {pre} → {len(df)} new combos to score")

    if df.empty:
        p("[Cross] ℹ️  All combinations already scored previously — nothing new this pass.")
        return pd.DataFrame(columns=["name", "smiles", "score"])

    p(f"[Cross] Step 5/5: scoring {len(df)} molecules with Boltz (batch_size={boltz_batch_size})...")
    t0 = time.time()
    scored = await score_molecules_with_boltz_batched(
        state, df.to_dict("records"), batch_size=boltz_batch_size,
    )
    p(f"[Cross] Boltz scoring finished in {time.time()-t0:.1f}s")

    scored_df = pd.DataFrame([
        {"name": m["name"], "smiles": m.get("smiles", ""), "score": m.get("boltz_score")}
        for m in scored if m.get("boltz_score") is not None
    ])
    return scored_df


# ═══════════════════════════════════════════════════════════════════════════
# 2-reactant Cross
# ═══════════════════════════════════════════════════════════════════════════

async def run_cross_2reactant(state, manager, rxn_id: int, boltz_batch_size: int) -> int:
    current = get_all_scored_from_db()
    if current.empty:
        p(
            "\n⚠️  [Cross] No scored molecules exist yet in score_results_"
            f"{rxn_id}.sqlite — Cross mode needs an existing anchor.\n"
            "    → Run Mode 1 (DPEX_DJA) or Mode 3 (Input) first to seed the DB,\n"
            "    then re-run Mode 4.\n"
        )
        return 0

    prev_best_score = float(current.iloc[0]["score"])
    p(f"\n[Cross] Current DB best score: {prev_best_score:.6f}  ({current.iloc[0]['name']})")

    top_A_vals = get_top_component_values(current, role_idx=0, n=TOP_A_2R)
    top_B_vals = get_top_component_values(current, role_idx=1, n=TOP_B_2R)
    p(f"[Cross] top-{TOP_A_2R} A values : {top_A_vals}")
    p(f"[Cross] top-{TOP_B_2R} B values : {top_B_vals}")

    if not top_A_vals or not top_B_vals:
        p("[Cross] ⚠️  Not enough distinct A/B values found — skipping pass.")
        return 0

    id_tuples = [(a, b) for a in top_A_vals for b in top_B_vals]
    p(f"[Cross] {len(top_A_vals)} A × {len(top_B_vals)} B = {len(id_tuples)} combinations to try")

    scored_df = await _build_validate_score(state, manager, rxn_id, id_tuples, boltz_batch_size)
    print_results(scored_df, prev_best_score=prev_best_score)

    if scored_df.empty:
        return 0

    rows = list(zip(scored_df["name"].tolist(), scored_df["score"].tolist()))
    n_new, n_upd = merge_scores_keep_max(rows)
    p(f"[Cross] DB merge: {n_new} new molecules inserted, {n_upd} existing molecules updated")
    return len(scored_df)


# ═══════════════════════════════════════════════════════════════════════════
# 3-reactant Cross
# ═══════════════════════════════════════════════════════════════════════════

async def run_cross_3reactant(state, manager, rxn_id: int, boltz_batch_size: int) -> int:
    current = get_all_scored_from_db()
    if current.empty:
        p(
            "\n⚠️  [Cross] No scored molecules exist yet in score_results_"
            f"{rxn_id}.sqlite — Cross mode needs an existing anchor.\n"
            "    → Run Mode 1 (DPEX_DJA) or Mode 3 (Input) first to seed the DB,\n"
            "    then re-run Mode 4.\n"
        )
        return 0

    prev_best_score = float(current.iloc[0]["score"])
    p(f"\n[Cross] Current DB best score: {prev_best_score:.6f}  ({current.iloc[0]['name']})")

    top_A_vals = get_top_component_values(current, role_idx=0, n=TOP_A_3R)
    top_B_vals = get_top_component_values(current, role_idx=1, n=TOP_B_3R)
    top_C_vals = get_top_component_values(current, role_idx=2, n=TOP_C_3R)
    p(f"[Cross] top-{TOP_A_3R} A : {top_A_vals}")
    p(f"[Cross] top-{TOP_B_3R} B : {top_B_vals}")
    p(f"[Cross] top-{TOP_C_3R} C : {top_C_vals}")

    if not top_A_vals or not top_B_vals or not top_C_vals:
        p("[Cross] ⚠️  Not enough distinct A/B/C values found — skipping pass.")
        return 0

    id_tuples = [
        (a, b, c) for a in top_A_vals for b in top_B_vals for c in top_C_vals
    ]
    p(
        f"[Cross] {len(top_A_vals)} A × {len(top_B_vals)} B × {len(top_C_vals)} C "
        f"= {len(id_tuples)} combinations to try"
    )

    scored_df = await _build_validate_score(state, manager, rxn_id, id_tuples, boltz_batch_size)
    print_results(scored_df, prev_best_score=prev_best_score)

    if scored_df.empty:
        return 0

    rows = list(zip(scored_df["name"].tolist(), scored_df["score"].tolist()))
    n_new, n_upd = merge_scores_keep_max(rows)
    p(f"[Cross] DB merge: {n_new} new molecules inserted, {n_upd} existing molecules updated")
    return len(scored_df)


# ═══════════════════════════════════════════════════════════════════════════
# main loop
# ═══════════════════════════════════════════════════════════════════════════

async def run_cross_loop(state, rxn_id: int):
    manager = common.molecule_manager
    config  = state['config']
    boltz_batch_size = (
        config.get('boltz_batch_size', 10) if isinstance(config, dict)
        else getattr(config, 'boltz_batch_size', 10)
    )

    is_three = manager.is_three_component
    pass_num = 0
    try:
        while True:
            pass_num += 1
            p(f"\n{'='*60}\n[Cross] === PASS {pass_num} (rxn={rxn_id}) ===\n{'='*60}")
            t0 = time.time()
            if is_three:
                n_scored = await run_cross_3reactant(state, manager, rxn_id, boltz_batch_size)
            else:
                n_scored = await run_cross_2reactant(state, manager, rxn_id, boltz_batch_size)
            p(f"[Cross] PASS {pass_num} complete in {time.time()-t0:.1f}s ({n_scored} newly scored)")

            if n_scored == 0:
                p("[Cross] Sleeping 30s before next pass...")
                await asyncio.sleep(30)
            else:
                await asyncio.sleep(2)
    except KeyboardInterrupt:
        p(f"\n🛑 Stopping Cross loop (rxn={rxn_id})...")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# standalone entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    parser = argparse.ArgumentParser(description="Mode 4 — Cross Miner")
    parser.add_argument("--rxn_id", type=int, required=True)
    args = parser.parse_args()
    return args.rxn_id


async def main():
    rxn_id = parse_args()
    common.configure_for_rxn(rxn_id)
    config = load_config()
    common.initialize_solution(config, rxn_id)
    state = common.build_state(config)
    try:
        await run_cross_loop(state, rxn_id)
    except KeyboardInterrupt:
        logger.info(f"✅ rxn={rxn_id} stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
