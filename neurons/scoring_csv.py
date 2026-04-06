import os
import sys
import math
import logging
import asyncio
import time
import sqlite3
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

MOLECULES_CSV      = os.path.join(BASE_DIR, "data", "data.csv")
SCORE_RESULTS_DB   = os.path.join(BASE_DIR, "score_results.sqlite")

from config.config_loader import load_config
from combinatorial_db.reactions import get_smiles_from_reaction

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

# GPU / scoring
BOLTZ_BATCH_SIZE       = 10    # molecules per Boltz batch
BOLTZ_BUDGET_PER_BATCH = 100   # max molecules scored per batch iteration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def init_score_results_db() -> None:
    """
    Create score_results DB with schema:
        molecule_name  TEXT PRIMARY KEY
        score          REAL
        scored_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        available      BOOLEAN DEFAULT TRUE
    
    NOTE: No 'smiles' column - only molecule_name and score
    """
    conn = sqlite3.connect(SCORE_RESULTS_DB)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scored_molecules (
            molecule_name TEXT PRIMARY KEY,
            score         REAL,
            scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            available     BOOLEAN DEFAULT TRUE
        )
    """)

    # Migration: add any columns missing in older DBs
    existing = {row[1] for row in cur.execute("PRAGMA table_info(scored_molecules)")}
    if "score" not in existing:
        cur.execute("ALTER TABLE scored_molecules ADD COLUMN score REAL")
        logger.info("DB migrated: added column 'score'")
    if "scored_at" not in existing:
        cur.execute(
            "ALTER TABLE scored_molecules "
            "ADD COLUMN scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
        logger.info("DB migrated: added column 'scored_at'")
    if "available" not in existing:
        cur.execute(
            "ALTER TABLE scored_molecules "
            "ADD COLUMN available BOOLEAN DEFAULT TRUE"
        )
        logger.info("DB migrated: added column 'available'")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)")
    conn.commit()
    conn.close()
    logger.info("Score results database initialized (no smiles column)")


def write_scores_to_db(molecules: List[Dict[str, Any]]) -> None:
    """
    Persist scored molecules to SQLite.
    Writes: molecule_name, score, scored_at (explicit UTC now), available=TRUE.
    NO SMILES COLUMN.
    """
    if not molecules:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(SCORE_RESULTS_DB)
    cur  = conn.cursor()
    rows = []
    for m in molecules:
        name   = m.get("name")
        score  = m.get("score")
        if name is None or score is None:
            continue
        rows.append((name, float(score), now, True))

    if rows:
        cur.executemany(
            """INSERT OR REPLACE INTO scored_molecules
               (molecule_name, score, scored_at, available)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        logger.info(f"Wrote {len(rows)} scored molecules to DB (no smiles)")
    conn.close()


def batch_get_scores_from_db(mol_names: List[str]) -> Dict[str, float]:
    """
    Lightweight batch DB lookup: name → score.
    Used to skip already-scored molecules.
    """
    if not mol_names or not os.path.exists(SCORE_RESULTS_DB):
        return {}
    try:
        conn = sqlite3.connect(SCORE_RESULTS_DB)
        cur  = conn.cursor()
        ph   = ",".join("?" * len(mol_names))
        cur.execute(
            f"SELECT molecule_name, score FROM scored_molecules "
            f"WHERE molecule_name IN ({ph}) AND score IS NOT NULL",
            mol_names,
        )
        result = {name: float(sc) for name, sc in cur.fetchall()}
        conn.close()
        return result
    except Exception as e:
        logger.debug(f"batch_get_scores_from_db error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════
# MOLECULE LOADING FROM CSV (SORTED BY SCORE)
# ══════════════════════════════════════════════════════════════════════════

def load_molecules_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load molecules from CSV file and sort by final_score (descending).
    
    Expected CSV format:
        molecule_name,final_score,epoch,available
        rxn:2:68710:170235,0.1450203458468119,21711,True
        rxn:2:141397:170175,0.1443813461065292,21711,True
        ...
    
    Returns:
        DataFrame with columns: name, smiles
        Sorted by final_score (highest first)
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading molecules from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Detect molecule name column
        if "molecule_name" in df.columns:
            name_col = "molecule_name"
        elif "name" in df.columns:
            name_col = "name"
        else:
            # Assume first column is molecule name
            name_col = df.columns[0]
            logger.info(f"Using first column '{name_col}' as molecule name")
        
        # Detect score column
        score_col = None
        for col in ["final_score", "score", "boltz_score"]:
            if col in df.columns:
                score_col = col
                break
        
        if score_col:
            # Sort by score (descending - highest first)
            df = df.sort_values(by=score_col, ascending=False)
            logger.info(f"Sorted by '{score_col}' (descending)")
            logger.info(f"Score range: [{df[score_col].min():.6f}, {df[score_col].max():.6f}]")
        else:
            logger.warning("No score column found - processing in original order")
        
        molecules = df[name_col].tolist()
        
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()
    
    # Process molecules - generate SMILES
    records: List[Dict] = []
    failed_smiles = 0
    
    for mol_name in molecules:
        # Generate SMILES from reaction notation
        smiles = get_smiles_from_reaction(mol_name)
        if not smiles:
            failed_smiles += 1
            logger.warning(f"Failed to generate SMILES for: {mol_name}")
            continue
        
        records.append({
            "name":   mol_name,
            "smiles": smiles,
        })
    
    logger.info(
        f"Processed {len(molecules)} molecules: "
        f"{len(records)} valid | "
        f"failed_smiles={failed_smiles}"
    )
    
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# BATCHED BOLTZ2 SCORING
# ══════════════════════════════════════════════════════════════════════════

async def score_molecules_with_boltz_batched(
    candidates:  List[Dict[str, Any]],
    boltz:       Any,
    target_seq:  str,
    boltz_cfg:   Dict[str, Any],
    batch_size:  int = BOLTZ_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Score candidates with BoltzWrapper in batches of `batch_size`.

    Flow per batch:
      1. Check DB cache — skip already-scored molecules
      2. Call boltz.score_molecules_target() for the remainder
      3. Extract per-molecule scores (per_molecule_metric → target_scores → avg)
      4. Write newly scored molecules to DB immediately (scored_at + available set)
      5. Return all results (cached + newly scored)
    """
    if not candidates:
        return []

    total      = len(candidates)
    n_batches  = math.ceil(total / batch_size)
    dummy_hash = "0x" + "0" * 64
    all_results: List[Dict[str, Any]] = []

    logger.info(
        f"Boltz2 batched scoring: {total} molecules | "
        f"{n_batches} batches × {batch_size}"
    )

    # Pre-check DB cache for the whole list
    db_scores  = batch_get_scores_from_db([c["name"] for c in candidates])
    cache_hits = len(db_scores)
    if cache_hits:
        logger.info(f"  DB cache: {cache_hits}/{total} already scored, skipping GPU")

    for batch_idx in range(n_batches):
        batch = candidates[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # ── Split: cached vs needs scoring ───────────────────────────
        cached_in_batch   = []
        to_score_in_batch = []

        for mol in batch:
            if mol["name"] in db_scores:
                cached_in_batch.append({
                    **mol,
                    "score":        db_scores[mol["name"]],
                    "score_source": "db_cache",
                })
            else:
                to_score_in_batch.append(mol)

        logger.info(
            f"  Batch {batch_idx+1}/{n_batches}: "
            f"{len(cached_in_batch)} cached, "
            f"{len(to_score_in_batch)} need GPU scoring"
        )

        newly_scored: List[Dict[str, Any]] = []

        if to_score_in_batch:
            smiles_list = [m["smiles"] for m in to_score_in_batch]
            names_list  = [m["name"]   for m in to_score_in_batch]

            valid_molecules_by_uid = {
                0: {"smiles": smiles_list, "names": names_list}
            }
            score_dict = {
                0: {
                    "target_scores":     [[]],
                    "antitarget_scores": [[]],
                    "entropy":           None,
                    "entropy_boltz":     None,
                    "block_submitted":   None,
                    "push_time":         "",
                    "boltz_score":       None,
                }
            }
            subnet_config = {
                "weekly_target":        boltz_cfg.get("weekly_target", target_seq),
                "binding_pocket":       boltz_cfg.get("binding_pocket", None),
                "max_distance":         boltz_cfg.get("max_distance", None),
                "force":                boltz_cfg.get("force", False),
                "num_molecules_boltz":  len(smiles_list),
                "boltz_metric":         boltz_cfg.get(
                    "boltz_metric",
                    ["affinity_probability_binary", "affinity_pred_value"]
                ),
                "combination_strategy": boltz_cfg.get(
                    "combination_strategy", "heavy_atom_normalization"
                ),
                "sample_selection":     "first",
            }

            t0 = time.time()
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: boltz.score_molecules_target(
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config,
                        dummy_hash,
                    ),
                )
                elapsed = time.time() - t0

                # ── Extract scores ────────────────────────────────────
                pm_metric     = boltz.per_molecule_metric.get(0, {})
                pm_components = boltz.per_molecule_components.get(0, {})

                target_scores_raw  = score_dict[0].get("target_scores", [[]])
                target_scores_list: Optional[List[float]] = None
                if target_scores_raw and len(target_scores_raw[0]) > 0:
                    inner = target_scores_raw[0]
                    target_scores_list = inner if isinstance(inner, list) else [inner]

                batch_avg   = score_dict[0].get("boltz_score")
                batch_valid = 0

                for mol_idx, mol in enumerate(to_score_in_batch):
                    smiles     = mol["smiles"]
                    components = pm_components.get(smiles, {})

                    # Priority 1: per_molecule_metric
                    score = pm_metric.get(smiles)

                    # Priority 2: affinity_probability_binary from components
                    if score is None:
                        score = components.get("affinity_probability_binary")

                    # Priority 3: target_scores list (index-aligned)
                    if score is None and target_scores_list is not None:
                        if mol_idx < len(target_scores_list):
                            score = target_scores_list[mol_idx]
                        else:
                            try:
                                si = smiles_list.index(smiles)
                                if si < len(target_scores_list):
                                    score = target_scores_list[si]
                            except ValueError:
                                pass

                    # Priority 4: batch average fallback
                    if score is None and batch_avg is not None:
                        score = batch_avg

                    final_score = float(score) if score is not None else None
                    if final_score is not None:
                        batch_valid += 1

                    result = {
                        **mol,
                        "score":        final_score,
                        "score_source": "boltz",
                    }
                    newly_scored.append(result)

                logger.info(
                    f"    GPU batch {batch_idx+1}: "
                    f"{batch_valid}/{len(to_score_in_batch)} scored "
                    f"in {elapsed:.1f}s "
                    f"({elapsed/max(len(to_score_in_batch),1):.1f}s/mol)"
                )

                # ── Write to DB immediately (scored_at + available set) ─
                scored_ok = [r for r in newly_scored if r["score"] is not None]
                if scored_ok:
                    write_scores_to_db(scored_ok)
                    for r in scored_ok:
                        db_scores[r["name"]] = r["score"]

            except Exception as e:
                logger.error(f"  Batch {batch_idx+1} GPU scoring failed: {e}")
                logger.debug(traceback.format_exc())
                for mol in to_score_in_batch:
                    newly_scored.append({
                        **mol,
                        "score":        None,
                        "score_source": "error",
                    })

        # ── Merge batch results ──────────────────────────────────────
        all_results.extend(cached_in_batch)
        all_results.extend(newly_scored)

    # ── Final summary ────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("score") is not None]
    if valid:
        arr = np.array([r["score"] for r in valid])
        logger.info(
            f"Batched scoring complete: {len(valid)}/{len(all_results)} valid | "
            f"mean={arr.mean():.4f} | max={arr.max():.4f} | "
            f"from_cache={cache_hits} | newly_scored={len(valid)-cache_hits}"
        )
    else:
        logger.warning("No valid scores returned from Boltz2")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# MAIN SCORING PIPELINE
# ══════════════════════════════════════════════════════════════════════════

async def run_scoring_pipeline(
    molecules_df: pd.DataFrame,
    boltz: Any,
    target_seq: str,
    boltz_cfg: Dict[str, Any],
    batch_size: int = BOLTZ_BATCH_SIZE,
    budget_per_iteration: int = BOLTZ_BUDGET_PER_BATCH,
) -> pd.DataFrame:
    """
    Score all molecules from DataFrame in batches.
    
    Args:
        molecules_df: DataFrame with columns: name, smiles (already sorted by score)
        boltz: BoltzWrapper instance
        target_seq: Target protein sequence
        boltz_cfg: Boltz configuration dict
        batch_size: Molecules per Boltz batch
        budget_per_iteration: Max molecules to score per iteration
    
    Returns:
        DataFrame with scored molecules
    """
    if molecules_df.empty:
        logger.warning("No molecules to score")
        return pd.DataFrame()
    
    total_molecules = len(molecules_df)
    logger.info(f"Starting scoring pipeline for {total_molecules} molecules (sorted by score)")
    
    # Convert DataFrame to list of dicts
    candidates = molecules_df.to_dict("records")
    
    # Score in chunks if needed
    all_scored = []
    n_iterations = math.ceil(total_molecules / budget_per_iteration)
    
    for iteration in range(n_iterations):
        start_idx = iteration * budget_per_iteration
        end_idx = min((iteration + 1) * budget_per_iteration, total_molecules)
        chunk = candidates[start_idx:end_idx]
        
        logger.info(
            f"\nIteration {iteration+1}/{n_iterations}: "
            f"scoring molecules {start_idx+1}-{end_idx}"
        )
        
        scored_chunk = await score_molecules_with_boltz_batched(
            chunk,
            boltz,
            target_seq,
            boltz_cfg,
            batch_size=batch_size,
        )
        
        all_scored.extend(scored_chunk)
        
        # Small delay between iterations
        if iteration < n_iterations - 1:
            await asyncio.sleep(2)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_scored)
    
    # Summary statistics
    valid_scores = results_df[results_df["score"].notna()]
    if not valid_scores.empty:
        logger.info("\n" + "="*70)
        logger.info("SCORING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total molecules:     {len(results_df)}")
        logger.info(f"Successfully scored: {len(valid_scores)}")
        logger.info(f"Failed/cached:       {len(results_df) - len(valid_scores)}")
        logger.info(f"Score range:         [{valid_scores['score'].min():.4f}, {valid_scores['score'].max():.4f}]")
        logger.info(f"Score mean:          {valid_scores['score'].mean():.4f}")
        logger.info(f"Score median:        {valid_scores['score'].median():.4f}")
        logger.info("="*70)
    else:
        logger.warning("No molecules were successfully scored")
    
    return results_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

async def main():
    logger.info("Starting Molecule Scoring Pipeline (CSV → Sort → Boltz → DB)")
    logger.info("="*70)
    
    # ── Load config ──────────────────────────────────────────────────
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    target_seq = (
        getattr(config, "weekly_target", None) or
        (config.get("weekly_target", "") if isinstance(config, dict) else "")
    )

    if not target_seq:
        logger.error("No target protein sequence found in config")
        return

    logger.info(f"Target protein: {target_seq[:60]}...")
    
    # ── Initialize database ──────────────────────────────────────────
    init_score_results_db()
    
    # ── Load molecules from CSV (sorted by score) ────────────────────
    molecules_df = load_molecules_from_csv(MOLECULES_CSV)
    
    if molecules_df.empty:
        logger.error("No valid molecules loaded from CSV")
        return
    
    logger.info(f"Loaded {len(molecules_df)} valid molecules from CSV (sorted)")
    
    # ── Initialize BoltzWrapper ──────────────────────────────────────
    boltz_scoring_dir = os.path.join(BASE_DIR, "boltz-scoring")
    boltz_src_dir     = os.path.join(boltz_scoring_dir, "boltz", "src")
    for d in [boltz_scoring_dir, boltz_src_dir]:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)

    try:
        from boltz.wrapper import BoltzWrapper
        boltz = BoltzWrapper()
        logger.info("BoltzWrapper initialized on GPU")
    except Exception as e:
        logger.error(f"BoltzWrapper import/init failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ── Boltz configuration ──────────────────────────────────────────
    boltz_cfg = {
        "weekly_target":        target_seq,
        "binding_pocket":       getattr(config, "binding_pocket", None),
        "max_distance":         getattr(config, "max_distance", None),
        "force":                getattr(config, "force", False),
        "boltz_metric":         getattr(
            config, "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"]
        ),
        "combination_strategy": getattr(
            config, "combination_strategy", "heavy_atom_normalization"
        ),
        "sample_selection":     getattr(config, "sample_selection", "first"),
    }
    
    # ── Run scoring pipeline ─────────────────────────────────────────
    try:
        t0 = time.time()
        
        results_df = await run_scoring_pipeline(
            molecules_df,
            boltz,
            target_seq,
            boltz_cfg,
            batch_size=BOLTZ_BATCH_SIZE,
            budget_per_iteration=BOLTZ_BUDGET_PER_BATCH,
        )
        
        elapsed = time.time() - t0
        
        # ── Save results to CSV (optional) ───────────────────────────
        output_csv = os.path.join(BASE_DIR, "data", "scored_molecules.csv")
        if not results_df.empty:
            # Only save name and score (no smiles)
            output_df = results_df[["name", "score"]].copy()
            output_df.to_csv(output_csv, index=False)
            logger.info(f"\nResults saved to: {output_csv}")
        
        logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
        logger.info(f"Average time per molecule: {elapsed/len(molecules_df):.1f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
