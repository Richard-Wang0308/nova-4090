"""
mode2_exhaust.py — Mode 2: Exhaust search.

function_change_A / _B / _C: fix all-but-one reactant role, enumerate
EVERY component value for the free role (from molecule_manager's
component id list), build candidate names, resolve SMILES, validate,
surrogate-prescore, keep top-N, Boltz-score, write to DB.

Traversal:
  2-reactant: top-2 A anchors + top-2 B anchors (4 total), each running
              a 3-step alternating chain (B→A→B or A→B→A).
  3-reactant: single top molecule (A1,B1,C1), 3 round-robin chains
              (A→B→C, B→C→A, C→A→B), each a 3-step chain of 150-cand rounds.

ASSUMPTIONS (confirm with user if wrong):
  - "top 2 A/B" = top-2 DISTINCT component VALUES appearing in ranked
    top molecules (not top-2 rows) — confirmed by user.
  - Runs continuously in a loop, re-anchoring from the latest DB state
    after each full pass (like mode 1's continuous loop).
  - Dedup against everything already in scored_molecules DB before
    Boltz scoring.
"""

import os
import time
import asyncio
import argparse
import itertools
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import common
from common import (
    logger, MoleculeUtils, SurrogateModel,
    score_molecules_with_boltz_batched, get_all_scored_from_db,
    parse_molecule_name, build_molecule_name, write_scores_to_db,
    load_molecules_combined, load_training_csv_for_surrogate,
)
from config.config_loader import load_config

N_TOP_ANCHORS_2R = 2     # top-2 A and top-2 B component values (2-reactant)
KEEP_A_2R        = 200   # function_change_A/B keep-N for 2-reactant
KEEP_ABC_3R      = 150   # function_change_A/B/C keep-N for 3-reactant


# ═══════════════════════════════════════════════════════════════════════════
# function_change_A / B / C
# ═══════════════════════════════════════════════════════════════════════════

async def _score_and_store(state, candidate_names: List[str], boltz_batch_size: int) -> pd.DataFrame:
    """Resolve SMILES, dedup vs DB, Boltz-score, write to DB. Returns scored_df."""
    if not candidate_names:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    df = pd.DataFrame({"name": candidate_names}).drop_duplicates(subset=["name"])
    df["smiles"] = df["name"].apply(MoleculeUtils.get_smiles_from_reaction_cached)
    df = df[df["smiles"].notna() & (df["smiles"] != "")]
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    # dedup vs existing DB entries
    existing = get_all_scored_from_db()
    if not existing.empty:
        existing_names = set(existing["name"].tolist())
        pre = len(df)
        df  = df[~df["name"].isin(existing_names)].reset_index(drop=True)
        logger.info(f"[Exhaust] Dedup vs DB: {pre} → {len(df)}")

    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    scored = await score_molecules_with_boltz_batched(
        state, df.to_dict("records"), batch_size=boltz_batch_size,
    )
    scored_df = pd.DataFrame([
        {"name": m["name"], "smiles": m.get("smiles", ""), "score": m.get("boltz_score")}
        for m in scored if m.get("boltz_score") is not None
    ])
    return scored_df


def _enumerate_role(
    manager, rxn_id: int, role: str,
    fixed: Dict[str, int],
) -> List[str]:
    """
    Enumerate ALL component values for `role`, holding `fixed` roles constant.
    fixed = {'B': 94689} for 2-reactant, or {'B': 228623, 'C': 229120} for 3-reactant.
    """
    if role == "A":
        pool = manager.moles_A_id
    elif role == "B":
        pool = manager.moles_B_id
    else:
        pool = manager.moles_C_id

    names = []
    for val in pool:
        if role == "A":
            ids = (val, fixed["B"]) if "C" not in fixed else (val, fixed["B"], fixed["C"])
        elif role == "B":
            ids = (fixed["A"], val) if "C" not in fixed else (fixed["A"], val, fixed["C"])
        else:  # role == "C"
            ids = (fixed["A"], fixed["B"], val)
        names.append(build_molecule_name(rxn_id, ids))
    return names


async def function_change_role(
    state, manager, surrogate: SurrogateModel, rxn_id: int,
    role: str, fixed: Dict[str, int], keep_n: int, boltz_batch_size: int,
) -> pd.DataFrame:
    """
    Generic fix-all-but-one-role, enumerate, validate, surrogate-prescore,
    keep top keep_n, Boltz score, write to DB. Returns scored_df (>=0 rows).
    """
    t0 = time.time()
    all_names = _enumerate_role(manager, rxn_id, role, fixed)
    logger.info(f"[Exhaust] function_change_{role}(fixed={fixed}): enumerated {len(all_names)} candidates")

    df = pd.DataFrame({"name": all_names})
    df["smiles"] = df["name"].apply(MoleculeUtils.get_smiles_from_reaction_cached)
    df = df[df["smiles"].notna() & (df["smiles"] != "")].reset_index(drop=True)
    logger.info(f"[Exhaust] {len(df)} resolved to valid SMILES")

    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    # validate (heavy atoms, banned atoms, rotatable bonds via manager)
    try:
        config = state['config']
        df = manager.validate_molecules(config, df)
    except Exception as e:
        logger.debug(f"[Exhaust] validate_molecules skipped: {e}")

    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    # surrogate prescore → keep top keep_n
    if surrogate.is_trained:
        df = surrogate.filter_candidates(df, top_n=keep_n, smiles_col="smiles")
    else:
        df = df.sample(n=min(keep_n, len(df)), random_state=None).reset_index(drop=True)
        logger.info(f"[Exhaust] Surrogate not trained — random sample of {len(df)} used instead")

    logger.info(f"[Exhaust] function_change_{role}: {len(df)} candidates → Boltz")

    scored_df = await _score_and_store(state, df["name"].tolist(), boltz_batch_size)
    if not scored_df.empty:
        write_scores_to_db([
            {"name": r["name"], "boltz_score": r["score"]} for _, r in scored_df.iterrows()
        ])
        if surrogate.enabled:
            surrogate.add_training_data(scored_df["smiles"].tolist(), scored_df["score"].tolist())

    logger.info(
        f"[Exhaust] function_change_{role} done in {time.time()-t0:.1f}s → "
        f"{len(scored_df)} scored & written"
    )
    return scored_df.sort_values(by="score", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Top-component discovery
# ═══════════════════════════════════════════════════════════════════════════

def get_top_component_values(df: pd.DataFrame, role_idx: int, n: int) -> List[int]:
    """
    df must have a 'name' column ranked by score (descending).
    role_idx: 0 for A, 1 for B, 2 for C.
    Returns top-n DISTINCT component values seen at that role position,
    in order of first (best) appearance.
    """
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


def get_top_molecule_ids(df: pd.DataFrame) -> Optional[Tuple[int, ...]]:
    if df.empty:
        return None
    return parse_molecule_name(df.iloc[0]["name"])


# ═══════════════════════════════════════════════════════════════════════════
# 2-reactant Exhaust chain
# ═══════════════════════════════════════════════════════════════════════════

async def run_exhaust_2reactant(state, manager, surrogate, rxn_id: int, boltz_batch_size: int):
    current = get_all_scored_from_db()
    if current.empty:
        logger.warning("[Exhaust] No scored molecules yet — cannot select top anchors. Skipping pass.")
        return

    top_A_vals = get_top_component_values(current, role_idx=0, n=N_TOP_ANCHORS_2R)
    top_B_vals = get_top_component_values(current, role_idx=1, n=N_TOP_ANCHORS_2R)
    logger.info(f"[Exhaust] top-{N_TOP_ANCHORS_2R} A anchors={top_A_vals} | top-{N_TOP_ANCHORS_2R} B anchors={top_B_vals}")

    # ── Chains anchored on top-A values: B→A→B ─────────────────────────
    for a_val in top_A_vals:
        logger.info(f"[Exhaust] === Chain (anchor A={a_val}): change_B → change_A → change_B ===")
        # need a starting B — use best known B partner for this A, else first B in pool
        partner_row = current[current["name"].str.contains(f":{a_val}:", regex=False)]
        b_start = parse_molecule_name(partner_row.iloc[0]["name"])[1] if not partner_row.empty else manager.moles_B_id[0]

        r1 = await function_change_role(state, manager, surrogate, rxn_id, "B", {"A": a_val}, KEEP_A_2R, boltz_batch_size)
        if r1.empty:
            continue
        b_star = parse_molecule_name(r1.iloc[0]["name"])[1]

        r2 = await function_change_role(state, manager, surrogate, rxn_id, "A", {"B": b_star}, KEEP_A_2R, boltz_batch_size)
        if r2.empty:
            continue
        a_star = parse_molecule_name(r2.iloc[0]["name"])[0]

        await function_change_role(state, manager, surrogate, rxn_id, "B", {"A": a_star}, KEEP_A_2R, boltz_batch_size)

    # ── Chains anchored on top-B values: A→B→A ─────────────────────────
    for b_val in top_B_vals:
        logger.info(f"[Exhaust] === Chain (anchor B={b_val}): change_A → change_B → change_A ===")
        partner_row = current[current["name"].str.endswith(f":{b_val}")]
        a_start = parse_molecule_name(partner_row.iloc[0]["name"])[0] if not partner_row.empty else manager.moles_A_id[0]

        r1 = await function_change_role(state, manager, surrogate, rxn_id, "A", {"B": b_val}, KEEP_A_2R, boltz_batch_size)
        if r1.empty:
            continue
        a_star = parse_molecule_name(r1.iloc[0]["name"])[0]

        r2 = await function_change_role(state, manager, surrogate, rxn_id, "B", {"A": a_star}, KEEP_A_2R, boltz_batch_size)
        if r2.empty:
            continue
        b_star = parse_molecule_name(r2.iloc[0]["name"])[1]

        await function_change_role(state, manager, surrogate, rxn_id, "A", {"B": b_star}, KEEP_A_2R, boltz_batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# 3-reactant Exhaust chain
# ═══════════════════════════════════════════════════════════════════════════

async def run_exhaust_3reactant(state, manager, surrogate, rxn_id: int, boltz_batch_size: int):
    current = get_all_scored_from_db()
    if current.empty:
        logger.warning("[Exhaust] No scored molecules yet — cannot select top molecule. Skipping pass.")
        return

    top_ids = get_top_molecule_ids(current)
    if top_ids is None or len(top_ids) < 3:
        logger.warning("[Exhaust] Could not parse top molecule ids. Skipping pass.")
        return
    A1, B1, C1 = top_ids
    logger.info(f"[Exhaust] Top molecule anchor: A={A1} B={B1} C={C1}")

    chains = [
        ("A", "B", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
    ]

    fixed_base = {"A": A1, "B": B1, "C": C1}

    for chain_idx, (r1, r2, r3) in enumerate(chains, start=1):
        logger.info(f"[Exhaust] === Chain {chain_idx}: change_{r1} → change_{r2} → change_{r3} ===")
        fixed = {k: v for k, v in fixed_base.items() if k != r1}
        res1 = await function_change_role(state, manager, surrogate, rxn_id, r1, fixed, KEEP_ABC_3R, boltz_batch_size)
        if res1.empty:
            continue
        ids1 = parse_molecule_name(res1.iloc[0]["name"])
        role_order = ["A", "B", "C"]
        val1 = ids1[role_order.index(r1)]

        fixed2 = dict(fixed_base)
        fixed2[r1] = val1
        fixed2 = {k: v for k, v in fixed2.items() if k != r2}
        res2 = await function_change_role(state, manager, surrogate, rxn_id, r2, fixed2, KEEP_ABC_3R, boltz_batch_size)
        if res2.empty:
            continue
        ids2 = parse_molecule_name(res2.iloc[0]["name"])
        val2 = ids2[role_order.index(r2)]

        fixed3 = dict(fixed_base)
        fixed3[r1] = val1
        fixed3[r2] = val2
        fixed3 = {k: v for k, v in fixed3.items() if k != r3}
        await function_change_role(state, manager, surrogate, rxn_id, r3, fixed3, KEEP_ABC_3R, boltz_batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# main loop
# ═══════════════════════════════════════════════════════════════════════════

async def run_exhaust_loop(state, rxn_id: int):
    manager = common.molecule_manager
    config  = state['config']
    boltz_batch_size = (
        config.get('boltz_batch_size', 10) if isinstance(config, dict)
        else getattr(config, 'boltz_batch_size', 10)
    )

    surrogate = SurrogateModel(max_training_samples=5000)
    loaded_df = load_molecules_combined(rxn_id)
    if not loaded_df.empty:
        surrogate.add_anchor_data(loaded_df['smiles'].tolist(), loaded_df['score'].tolist())
    training_df = load_training_csv_for_surrogate(rxn_id)
    if not training_df.empty:
        surrogate.add_anchor_data(training_df['smiles'].tolist(), training_df['score'].tolist())
    if surrogate.total_train_size >= surrogate.min_train_size:
        surrogate.train(iteration=0, force=True)
    logger.info(f"[Exhaust] Surrogate ready={surrogate.is_trained} (train_size={surrogate.total_train_size})")

    is_three = manager.is_three_component
    pass_num = 0
    try:
        while True:
            pass_num += 1
            logger.info(f"\n{'='*60}\n[Exhaust] === PASS {pass_num} (rxn={rxn_id}) ===\n{'='*60}")
            t0 = time.time()
            if is_three:
                await run_exhaust_3reactant(state, manager, surrogate, rxn_id, boltz_batch_size)
            else:
                await run_exhaust_2reactant(state, manager, surrogate, rxn_id, boltz_batch_size)
            logger.info(f"[Exhaust] PASS {pass_num} complete in {time.time()-t0:.1f}s")
            await asyncio.sleep(2)
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Stopping Exhaust loop (rxn={rxn_id})...")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# standalone entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    parser = argparse.ArgumentParser(description="Mode 2 — Exhaust Miner")
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
        await run_exhaust_loop(state, rxn_id)
    except KeyboardInterrupt:
        logger.info(f"✅ rxn={rxn_id} stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
