"""
mode1_dpex_dja.py — Mode 1: original DPEX-DJA + Boltz loop, unchanged logic,
refactored to use shared infrastructure from common.py.
"""

import os
import time
import asyncio
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple

import common
from common import (
    logger, MoleculeUtils, SurrogateModel, ComponentRanker,
    select_tanimoto_diverse, load_molecules_combined,
    load_training_csv_for_surrogate, score_molecules_with_boltz_batched,
    SURROGATE_KEEP_RATIO, SURROGATE_MIN_TRAIN_SIZE,
)
from config.config_loader import load_config
from tools import (
    IterationParams, SynthonLibrary, generate_valid_random_molecules,
    build_component_weights,
)
from exploit import get_top_n_unexploited, run_exploit
from dpex_dja import (
    DPEXDJAState, dja_generate, tabu_generate, update_tabu,
    dpex_exchange, update_populations, set_ranker_weights,
)

GENERATE_MULTIPLIER = 5
BOLTZ_BUDGET         = 600


# ═══════════════════════════════════════════════════════════════════════════
# Pool progress helpers
# ═══════════════════════════════════════════════════════════════════════════

def _top_pool_stats(top_pool: pd.DataFrame, n: int) -> Tuple[float, float, Optional[str]]:
    if top_pool.empty or 'score' not in top_pool.columns:
        return 0.0, 0.0, None
    scores = pd.to_numeric(top_pool['score'], errors='coerce').dropna()
    if scores.empty:
        return 0.0, 0.0, None
    top_scores = pd.to_numeric(top_pool.head(n)['score'], errors='coerce').dropna()
    pool_avg  = float(top_scores.mean()) if not top_scores.empty else 0.0
    pool_max  = float(scores.max())
    best_name = str(top_pool.iloc[0].get('name', '')) or None
    return pool_avg, pool_max, best_name


def _iteration_mode_str(exploited_status, dpex, params, early_exploit_used=False, exploit_attempted=False) -> str:
    if exploited_status:
        return "EXPLOIT"
    if not dpex.pop_A:
        return "INIT(cold)"
    base = "DJA+TABU" if params.synthon_lib is not None else "DJA"
    if exploit_attempted:
        base = f"EXPLOIT(failed)→{base}"
    if early_exploit_used:
        base = f"{base}+early-exploit"
    return base


def _log_pool_progress(iteration, pool_avg, pool_max, best_name, prev_avg, prev_max,
                        best_max_ever, score_improvement_rate, n, mode=None) -> float:
    if pool_max > best_max_ever:
        best_max_ever = pool_max
    avg_delta = (pool_avg - prev_avg) if prev_avg is not None else None
    max_delta = (pool_max - prev_max) if prev_max is not None else None
    lines = ["[PoolProgress] " + (f"iter={iteration}" if iteration > 0 else "warm-start")]
    if mode:
        lines.append(f"  mode             : {mode}")
    lines += [f"  top-{n} avg      : {pool_avg:.6f}", f"  pool max         : {pool_max:.6f}"]
    if best_name:
        lines.append(f"  best molecule    : {best_name}")
    if avg_delta is not None:
        lines.append(f"  avg Δ            : {avg_delta:+.6f} ({score_improvement_rate:+.2%} rel)")
    if max_delta is not None:
        lines.append(f"  max Δ            : {max_delta:+.6f}")
    lines.append(f"  all-time max     : {best_max_ever:.6f}")
    logger.info("\n".join(lines))
    return best_max_ever


# ═══════════════════════════════════════════════════════════════════════════
# WARM START
# ═══════════════════════════════════════════════════════════════════════════

def warm_start(state, dpex, ranker, surrogate, params, top_pool, all_pool,
               num_molecules, tanimoto_max_threshold, rxn_id):
    config = state['config']
    logger.info(f"[WarmStart] rxn={rxn_id} | surrogate activates once total training data >= {SURROGATE_MIN_TRAIN_SIZE}")

    loaded_df = load_molecules_combined(rxn_id)

    if loaded_df.empty:
        logger.warning("[WarmStart] No rxn-specific data — cold start")
    else:
        logger.info(f"[WarmStart] Rxn-specific: {len(loaded_df)} molecules loaded")
        all_pool = loaded_df.rename(columns={'InChIKey': 'inchi'}).copy()
        ranker.update(loaded_df)
        ranker.warm_start_decay(n_historical_rounds=50)

        top_for_A  = loaded_df.head(dpex.N_A)
        dpex.pop_A = top_for_A.rename(columns={'InChIKey': 'inchi'}).to_dict('records')

        diverse_elites = select_tanimoto_diverse(
            loaded_df.reset_index(drop=True), n=dpex.N_B, threshold=0.85, smiles_col='smiles',
        )
        dpex.pop_B = diverse_elites.rename(columns={'InChIKey': 'inchi'}).to_dict('records')

        params.seen_molecules = set(loaded_df['name'].tolist())

        top_pool = select_tanimoto_diverse(
            all_pool.reset_index(drop=True), n=num_molecules + 50,
            threshold=tanimoto_max_threshold, smiles_col="smiles",
        ).reset_index(drop=True)

    logger.info(f"[WarmStart] Loading surrogate anchors (rxn={rxn_id})...")
    if not loaded_df.empty:
        surrogate.add_anchor_data(loaded_df['smiles'].tolist(), loaded_df['score'].tolist())

    training_df = load_training_csv_for_surrogate(rxn_id)
    if not training_df.empty:
        surrogate.add_anchor_data(training_df['smiles'].tolist(), training_df['score'].tolist())

    if surrogate.total_train_size < surrogate.min_train_size:
        logger.warning(
            f"[WarmStart] Insufficient data for surrogate "
            f"({surrogate.total_train_size} < {surrogate.min_train_size}) — hard-cap at {BOLTZ_BUDGET}"
        )
    else:
        surrogate.train(iteration=0, force=True)
        logger.info(f"[WarmStart] Surrogate pre-trained: trained={surrogate.is_trained}")

    if not loaded_df.empty:
        ws_avg, ws_max, ws_best = _top_pool_stats(top_pool, num_molecules)
        _log_pool_progress(0, ws_avg, ws_max, ws_best, None, None, ws_max, 0.0, num_molecules, mode="WARM-START")

    return top_pool, all_pool


# ═══════════════════════════════════════════════════════════════════════════
# find_solution
# ═══════════════════════════════════════════════════════════════════════════

async def find_solution(state: Dict[str, Any], rxn_id: int) -> None:
    config = state['config']

    def _cfg(key, default):
        return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)

    num_molecules          = _cfg('num_molecules', 100)
    tanimoto_max_threshold = _cfg('tanimoto_max_threshold', 0.9)
    boltz_batch_size       = _cfg('boltz_batch_size', 10)
    LIMIT_PER_REACTANT     = _cfg('limit_per_reactant', 600)

    surrogate       = SurrogateModel(max_training_samples=5000)
    use_surrogate   = True
    exploit_counter = 0
    ranker          = ComponentRanker(decay=0.90)
    plateau_counter = 0

    params = IterationParams(config=config)
    dpex   = DPEXDJAState()

    seed_df          = pd.DataFrame(columns=["name", "smiles"])
    top_pool         = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    all_pool         = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    tabued_molecules: set = set()
    iteration        = 0

    top_pool, all_pool = warm_start(
        state=state, dpex=dpex, ranker=ranker, surrogate=surrogate, params=params,
        top_pool=top_pool, all_pool=all_pool, num_molecules=num_molecules,
        tanimoto_max_threshold=tanimoto_max_threshold, rxn_id=rxn_id,
    )

    _, init_pool_max, _ = _top_pool_stats(top_pool, num_molecules)
    best_max
    _, init_pool_max, _ = _top_pool_stats(top_pool, num_molecules)
    best_max_ever = init_pool_max if not top_pool.empty else 0.0

    try:
        logger.info("[Solution] Building synthon library...")
        t0 = time.time()
        params.synthon_lib        = common.molecule_manager and SynthonLibrary(molecule_manager=common.molecule_manager)
        params.use_synthon_search = True
        logger.info(f"[Solution] Synthon library ready in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.warning(f"[Solution] Synthon library failed: {e}")
        params.synthon_lib        = None
        params.use_synthon_search = False

    manager = common.molecule_manager

    logger.info(
        f"🚀 DPEX-DJA loop | rxn={rxn_id} | "
        f"A={len(manager.moles_A_id)} B={len(manager.moles_B_id)} C={len(manager.moles_C_id)} | "
        f"pipeline: generate {GENERATE_MULTIPLIER}x → dedup → surrogate(keep {SURROGATE_KEEP_RATIO*100:.0f}% "
        f"if trained, else hard-cap {BOLTZ_BUDGET}) → Boltz [ALL modes]"
    )
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            iteration        += 1
            component_weights = None
            dpex.iteration    = iteration
            iter_start        = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"[Solution] --- Iteration {iteration} [rxn={rxn_id}] ---")

            # ── Reload only NEW scored molecules every 5 iterations ───
            if iteration % 5 == 0:
                loaded_df = load_molecules_combined(rxn_id)
                if not loaded_df.empty:
                    loaded_renamed = loaded_df.rename(columns={'InChIKey': 'inchi'})
                    existing_names = set(all_pool['name'].tolist())
                    new_only = loaded_renamed[~loaded_renamed['name'].isin(existing_names)]
                    if not new_only.empty:
                        all_pool = pd.concat([all_pool, new_only], ignore_index=True)
                        all_pool = all_pool.sort_values(
                            by='score', ascending=False, na_position='last'
                        ).drop_duplicates(subset=['inchi'], keep='first')
                        logger.info(f"[Solution] +{len(new_only)} new from disk (all_pool={len(all_pool)})")

                        if surrogate.enabled and 'score' in new_only.columns:
                            valid_new = new_only[new_only['score'].notna() & new_only['smiles'].notna()]
                            if not valid_new.empty:
                                surrogate.add_training_data(
                                    valid_new['smiles'].tolist(), valid_new['score'].tolist()
                                )
                                logger.info(
                                    f"[SURROGATE] +{len(valid_new)} new scores from disk "
                                    f"(total_train_size={surrogate.total_train_size})"
                                )

            # ── n_samples ───────────────────────────────────────────────
            n_base_samples = params.get_nsamples_from_iteration(iteration)
            n_samples      = n_base_samples * GENERATE_MULTIPLIER
            logger.info(
                f"[Solution] n_samples={n_samples} ({n_base_samples}×{GENERATE_MULTIPLIER}) "
                f"→ dedup → surrogate(keep {SURROGATE_KEEP_RATIO*100:.0f}% if trained, "
                f"else hard-cap {BOLTZ_BUDGET}) → Boltz"
            )

            # ── Component weights ──────────────────────────────────────
            if not top_pool.empty:
                component_weights = build_component_weights(top_pool.head(num_molecules), rxn_id)
            if component_weights is not None:
                component_weights = ranker.blend_component_weights(component_weights, manager)
            ranker.push_to_dja(manager)

            # ── Elite selection ────────────────────────────────────────
            elite_df = (
                MoleculeUtils.select_diverse_elites(top_pool, min(150, len(top_pool)))
                if not top_pool.empty else pd.DataFrame()
            )
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            # ── Exploit mode toggle ────────────────────────────────────
            if params.no_improvement_counter >= 2 and not params.use_exploit_mode:
                params.use_exploit_mode       = True
                params.no_improvement_counter = 0
                logger.info("[Solution] === EXPLOIT MODE ===")
            elif params.no_improvement_counter >= 2 or exploit_counter >= 4:
                params.use_exploit_mode       = False
                exploit_counter               = 0
                params.no_improvement_counter = 0

            # ── Augment pop_B ──────────────────────────────────────────
            if not top_pool.empty:
                cols = [c for c in ('name', 'smiles', 'score') if c in top_pool.columns]
                dpex.augment_pop_B(top_pool[cols].head(dpex.N_B).to_dict('records'))

            # ── Generation ────────────────────────────────────────────
            data               = pd.DataFrame(columns=["name", "smiles"])
            data_dja           = pd.DataFrame(columns=["name"])
            data_tabu          = pd.DataFrame(columns=["name"])
            data_tabu_moves: list = []
            data_early_exploit = pd.DataFrame(columns=["name", "smiles"])
            exploited_status   = False
            exploit_summary    = None
            exploit_attempted  = False
            current_mode       = "unknown"

            if not top_pool.empty and iteration <= 20:
                try:
                    unexploited_ee = get_top_n_unexploited(
                        top_pool.to_dict("records"), params.exploited_reactants, n=2
                    )
                    if unexploited_ee:
                        t0_ee = time.time()
                        early_results, _ = run_exploit(
                            manager=manager, config=config, top_molecules=unexploited_ee, top_n=1,
                            limit_per_reactant=150, avoid_names=params.seen_molecules,
                            exploited_reactants=set(),
                        )
                        if early_results:
                            data_early_exploit = pd.DataFrame(early_results)
                            logger.info(f"[Solution] Early exploit: {len(data_early_exploit)} in {time.time()-t0_ee:.1f}s")
                except Exception as e:
                    logger.debug(f"[Solution] Early exploit skipped: {e}")

            if params.use_exploit_mode and not top_pool.empty:
                exploit_attempted = True
                logger.info("[Solution] Exploit: structure-guided deep search...")
                try:
                    unexploited = get_top_n_unexploited(top_pool.to_dict("records"), params.exploited_reactants)
                    if unexploited:
                        t0 = time.time()
                        exploit_results, exploit_summary = run_exploit(
                            manager=manager, config=config, top_molecules=unexploited,
                            limit_per_reactant=LIMIT_PER_REACTANT, avoid_names=params.seen_molecules,
                            exploited_reactants=params.exploited_reactants,
                        )
                        logger.info(f"[Solution] Exploit: {len(exploit_results)} candidates in {time.time()-t0:.1f}s")
                        if exploit_results:
                            data             = pd.DataFrame(exploit_results)
                            exploited_status = True
                        else:
                            raise Exception("Exploit returned no molecules.")
                    else:
                        raise Exception("No unexploited molecules.")
                except Exception as e:
                    logger.warning(f"[Solution] Exploit skipped: {e}")
                exploit_counter += 1

            if not exploited_status:
                if not dpex.pop_A:
                    logger.info(f"[Solution] Cold init: generating {params.n_samples_start} random molecules")
                    raw  = generate_valid_random_molecules(
                        molecule_manager=manager, n_samples=params.n_samples_start,
                        seen_molecules=params.seen_molecules, component_weights=component_weights,
                    )
                    data = pd.DataFrame({"name": raw})
                else:
                    n_dja  = int(n_samples * 0.75)
                    n_tabu = n_samples - n_dja

                    logger.info(f"[Solution] DJA: {n_dja} candidates (pop_A={len(dpex.pop_A)})")
                    raw_dja = dja_generate(
                        state=dpex, molecule_manager=manager, n_samples=n_dja,
                        mutation_prob=params.mutation_prob,
                    )
                    if raw_dja:
                        data_dja = manager.validate_molecules(config, pd.DataFrame({"name": raw_dja}))
                        logger.info(f"[Solution] DJA: {len(data_dja)} validated")

                    if params.synthon_lib is not None and dpex.pop_B:
                        if params.score_improvement_rate > 0.05:
                            n_per_elite = 15
                        elif params.score_improvement_rate > 0.02:
                            n_per_elite = 20
                        elif params.score_improvement_rate > 0.005:
                            n_per_elite = 25
                        else:
                            n_per_elite = 50

                        logger.info(f"[Solution] Tabu: n_tabu≈{n_tabu} neighborhood={n_per_elite}")
                        raw_tabu = tabu_generate(
                            state=dpex, synthon_lib=params.synthon_lib, n_samples=n_tabu,
                            neighborhood_size=n_per_elite,
                        )
                        if params.score_improvement_rate <= 0.005:
                            tabued_molecules |= {x['name'] for x in dpex.pop_B}
                        if raw_tabu:
                            data_tabu = manager.validate_molecules(config, pd.DataFrame({"name": raw_tabu}))
                            if not data_dja.empty:
                                data_tabu = data_tabu[~data_tabu["name"].isin(data_dja["name"].tolist())]
                            logger.info(f"[Solution] Tabu: {len(data_tabu)} validated")

                    parts = [df for df in [data_dja, data_tabu, data_early_exploit] if not df.empty]
                    if parts:
                        data = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["name"])
                        if not seed_df.empty:
                            data = pd.concat([data, seed_df], ignore_index=True).drop_duplicates(subset=["name"])
                            seed_df = pd.DataFrame(columns=["name", "smiles"])

                    raw_rand = generate_valid_random_molecules(
                        molecule_manager=manager, n_samples=int(n_samples * 0.5),
                        seen_molecules=params.seen_molecules, component_weights=component_weights,
                    )
                    data = pd.concat(
                        [data, pd.DataFrame({"name": raw_rand})], ignore_index=True
                    ).drop_duplicates(subset=["name"])

            current_mode = _iteration_mode_str(
                exploited_status=exploited_status, dpex=dpex, params=params,
                early_exploit_used=not data_early_exploit.empty, exploit_attempted=exploit_attempted,
            )

            logger.info(f"[Solution] {len(data)} candidates generated ({current_mode}) in {time.time()-iter_start:.2f}s")

            if data.empty:
                logger.warning("[Solution] No candidates; skipping")
                await asyncio.sleep(5)
                continue

            if not seed_df.empty:
                data = pd.concat([data, seed_df], ignore_index=True).drop_duplicates(subset=["name"])
                seed_df = pd.DataFrame(columns=["name", "smiles"])

            if 'smiles' not in data.columns or data['smiles'].isna().all():
                data['smiles'] = data['name'].apply(MoleculeUtils.get_smiles_from_reaction_cached)
            data = data[data['smiles'].notna() & (data['smiles'] != '')]

            if data.empty:
                logger.warning("[Solution] No valid SMILES; skipping")
                await asyncio.sleep(5)
                continue

            # ── STEP 1 — Dedup ──────────────────────────────────────────
            pre_dedup = len(data)
            data      = data[~data["name"].isin(params.seen_molecules)].reset_index(drop=True)
            dup_ratio = (pre_dedup - len(data)) / max(1, pre_dedup)
            logger.info(f"[Solution] Dedup: {pre_dedup} → {len(data)} ({dup_ratio*100:.0f}% already seen)")

            if dup_ratio > 0.7:
                params.mutation_prob = min(0.90, params.mutation_prob * 1.5)
            elif dup_ratio > 0.5:
                params.mutation_prob = min(0.70, params.mutation_prob * 1.3)
            elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                params.mutation_prob = max(0.10, params.mutation_prob * 0.95)

            if data.empty:
                logger.error("[Solution] All duplicates after dedup; boosting diversity")
                params.mutation_prob = min(0.95, params.mutation_prob * 2.0)
                params.elite_prob    = max(0.10, params.elite_prob    * 0.5)
                await asyncio.sleep(5)
                continue

            # ── STEP 2 — Surrogate filter ──────────────────────────────
            surrogate_ready = (
                use_surrogate and surrogate.enabled and surrogate.is_trained
                and surrogate.total_train_size >= surrogate.min_train_size
            )

            if surrogate_ready:
                pre_sur = len(data)
                data    = surrogate.filter_candidates(data, keep_ratio=SURROGATE_KEEP_RATIO, smiles_col="smiles")
                logger.info(
                    f"[SURROGATE] iter={iteration} mode={current_mode} | {pre_sur} fresh → {len(data)} "
                    f"(train_size={surrogate.total_train_size})"
                )
            else:
                if len(data) > BOLTZ_BUDGET:
                    pre_cap = len(data)
                    data    = data.head(BOLTZ_BUDGET)
                    logger.info(
                        f"[Solution] Surrogate not ready "
                        f"(train_size={surrogate.total_train_size} < {surrogate.min_train_size}) — "
                        f"hard-cap {pre_cap} → {len(data)}"
                    )

            if data.empty:
                logger.warning("[Solution] No candidates after filter; skipping")
                await asyncio.sleep(5)
                continue

            # ── Boltz scoring ──────────────────────────────────────────
            logger.info(f"[Solution] Scoring {len(data)} molecules with Boltz...")
            t_score = time.time()
            scored_molecules = await score_molecules_with_boltz_batched(
                state, data.to_dict('records'), batch_size=boltz_batch_size,
            )
            logger.info(f"[Solution] Boltz done in {time.time()-t_score:.2f}s")

            scored_df = pd.DataFrame([
                {'name': m['name'], 'smiles': m.get('smiles', ''), 'score': m.get('boltz_score')}
                for m in scored_molecules if m.get('boltz_score') is not None
            ])

            if scored_df.empty:
                logger.warning("[Solution] No scores returned; skipping")
                await asyncio.sleep(5)
                continue

            ranker.update(scored_df)

            if surrogate.enabled:
                surrogate.add_training_data(scored_df['smiles'].tolist(), scored_df['score'].tolist())
                if surrogate.total_train_size >= surrogate.min_train_size:
                    t_train = time.time()
                    surrogate.train(iteration)
                    train_time = time.time() - t_train
                    if train_time > 10.0:
                        logger.warning(f"[SURROGATE] Training slow ({train_time:.2f}s) this round — skipping next {surrogate.train_interval} rounds")
                        # NOTE: fixed vs. original — no permanent disable, just backs off cadence
                        surrogate.train_interval = min(20, surrogate.train_interval * 2)

            dja_names  = set(data_dja["name"].tolist()) if not data_dja.empty else set()
            tabu_names = set(data_tabu["name"].tolist()) if not data_tabu.empty else set()
            scored_for_A = scored_df[scored_df["name"].isin(dja_names)] if dja_names else scored_df
            scored_for_B = (
                scored_df[scored_df["name"].isin(tabu_names)] if tabu_names
                else pd.DataFrame(columns=scored_df.columns)
            )

            update_populations(dpex, scored_for_A)
            if not scored_for_B.empty:
                dpex.augment_pop_B(scored_for_B.to_dict('records'))

            if data_tabu_moves:
                for move_name in data_tabu_moves:
                    update_tabu(dpex, move_name)

            if iteration % dpex.T_ex == 0:
                dpex_exchange(dpex)

            params.seen_molecules = params.seen_molecules | set(scored_df["name"].tolist())

            iter_prev_avg, iter_prev_max, _ = _top_pool_stats(top_pool, num_molecules)
            prev_avg = iter_prev_avg if not top_pool.empty else None
            prev_max = iter_prev_max if not top_pool.empty else None

            scored_df["inchi"] = scored_df["smiles"].apply(MoleculeUtils.generate_inchikey)
            scored_df = scored_df[scored_df["inchi"] != ""]

            if not scored_df.empty:
                all_pool = (
                    pd.concat([all_pool, scored_df], ignore_index=True)
                    if not all_pool.empty else scored_df.copy()
                )
                all_pool = all_pool.sort_values(
                    by='score', ascending=False, na_position='last'
                ).drop_duplicates(subset=['inchi'], keep='first')

            top_pool = select_tanimoto_diverse(
                all_pool.reset_index(drop=True), n=num_molecules + 50,
                threshold=tanimoto_max_threshold, smiles_col="smiles",
            ).reset_index(drop=True)

            pool_avg, pool_max, best_name = _top_pool_stats(top_pool, num_molecules)
            current_avg = pool_avg if not top_pool.empty else None
            if current_avg is not None and prev_avg is not None:
                params.score_improvement_rate = (current_avg - prev_avg) / max(abs(prev_avg), 1e-6)
            elif current_avg is not None:
                params.score_improvement_rate = 1.0

            if params.score_improvement_rate <= 0.0001:
                params.no_improvement_counter += 1
                plateau_counter               += 1
            else:
                params.no_improvement_counter = 0
                plateau_counter               = 0

            if plateau_counter >= 5:
                params.mutation_prob = min(0.85, params.mutation_prob * 2.0)
                logger.info(f"[Solution] ANTI-PLATEAU: mutation_prob → {params.mutation_prob:.2f}")
                plateau_counter = 0

            if (
                exploit_summary and 'exploited_reactant_ids' in exploit_summary
                and (params.score_improvement_rate <= 0.0001 or not exploited_status)
            ):
                params.exploited_reactants.update(exploit_summary['exploited_reactant_ids'])
                logger.info(f"[Solution] Exploited reactants total: {len(params.exploited_reactants)}")

            iter_time = time.time() - iter_start
            logger.info(
                f"Iter {iteration:4d} | {iter_time:6.1f}s | Mode: {current_mode:24s} | rxn={rxn_id} | "
                f"popA={len(dpex.pop_A):4d} popB={len(dpex.pop_B):4d} | "
                f"pool avg={pool_avg:.5f} max={pool_max:.5f} | Δ={params.score_improvement_rate:+.5f} | "
                f"no_improve={params.no_improvement_counter} | "
                f"surrogate={'ON' if surrogate_ready else 'OFF'} "
                f"({surrogate.total_train_size}/{surrogate.min_train_size} samples)"
            )

            best_max_ever = _log_pool_progress(
                iteration, pool_avg, pool_max, best_name, prev_avg, prev_max,
                best_max_ever, params.score_improvement_rate, num_molecules, mode=current_mode,
            )

            if not top_pool.empty:
                logger.info(f"   🏆 Best: {best_name} (score={pool_max:.6f})")

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        logger.info(f"\n🛑 Stopping DPEX-DJA loop (rxn={rxn_id})...")
        if not top_pool.empty:
            _, final_max, final_best = _top_pool_stats(top_pool, num_molecules)
            logger.info(f"Final best: {final_best} (score={final_max:.6f}) | all-time max={best_max_ever:.6f}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# standalone entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    parser = argparse.ArgumentParser(description="Mode 1 — DPEX-DJA Miner")
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
        await find_solution(state, rxn_id)
    except KeyboardInterrupt:
        logger.info(f"✅ rxn={rxn_id} stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
