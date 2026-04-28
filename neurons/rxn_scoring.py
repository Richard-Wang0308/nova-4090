import os
import sys
import math
import logging
import asyncio
import time
import sqlite3
import traceback
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
BOLTZ_BATCH_SIZE = 10
BOLTZ_BUDGET_PER_BATCH = 100

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════
def init_score_results_db(score_results_db: str) -> None:
    conn = sqlite3.connect(score_results_db)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scored_molecules (
            molecule_name TEXT PRIMARY KEY,
            score         REAL,
            scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            available     BOOLEAN DEFAULT TRUE
        )
        """
    )

    existing = {row[1] for row in cur.execute("PRAGMA table_info(scored_molecules)")}
    if "score" not in existing:
        cur.execute("ALTER TABLE scored_molecules ADD COLUMN score REAL")
    if "scored_at" not in existing:
        cur.execute(
            "ALTER TABLE scored_molecules "
            "ADD COLUMN scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
    if "available" not in existing:
        cur.execute(
            "ALTER TABLE scored_molecules " "ADD COLUMN available BOOLEAN DEFAULT TRUE"
        )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)")
    conn.commit()
    conn.close()
    logger.info(f"Score database ready: {score_results_db}")


def write_scores_to_db(score_results_db: str, molecules: List[Dict[str, Any]]) -> None:
    if not molecules:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(score_results_db)
    cur = conn.cursor()
    rows = []
    for m in molecules:
        name = m.get("name")
        score = m.get("score")
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
        logger.info(f"Wrote {len(rows)} molecules to DB")
    conn.close()


def batch_get_scores_from_db(score_results_db: str, mol_names: List[str]) -> Dict[str, float]:
    if not mol_names or not os.path.exists(score_results_db):
        return {}

    try:
        conn = sqlite3.connect(score_results_db)
        cur = conn.cursor()
        ph = ",".join("?" * len(mol_names))
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
# CSV LOADING (KEEP ORIGINAL ORDER)
# ══════════════════════════════════════════════════════════════════════════
def load_molecules_in_order(csv_path: str) -> pd.DataFrame:
    """
    Load molecules from CSV and preserve original row order.
    Expected columns:
      - Molecule_ID or molecule_name or name
      - SMILES or smiles
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()

    logger.info(f"Loading molecules from {csv_path} in original order")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        logger.warning("Input CSV is empty")
        return pd.DataFrame()

    name_col = None
    for col in ["Molecule_ID", "molecule_name", "name"]:
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        name_col = df.columns[0]
        logger.warning(f"No molecule name column found; using first column '{name_col}'")

    smiles_col = None
    for col in ["SMILES", "smiles"]:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        logger.error("No SMILES column found. Need one of: SMILES, smiles")
        return pd.DataFrame()

    # Keep original order by iterating rows directly.
    records: List[Dict[str, str]] = []
    skipped = 0
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        smiles = str(row.get(smiles_col, "")).strip()
        if not name or name.lower() == "nan" or not smiles or smiles.lower() == "nan":
            skipped += 1
            continue
        records.append({"name": name, "smiles": smiles})

    logger.info(
        f"Loaded {len(df)} rows | valid={len(records)} | skipped_empty={skipped} "
        "(order preserved)"
    )
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# BATCHED BOLTZ2 SCORING
# ══════════════════════════════════════════════════════════════════════════
async def score_molecules_with_boltz_batched(
    candidates: List[Dict[str, Any]],
    boltz: Any,
    target_seq: str,
    boltz_cfg: Dict[str, Any],
    score_results_db: str,
    batch_size: int = BOLTZ_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    total = len(candidates)
    n_batches = math.ceil(total / batch_size)
    dummy_hash = "0x" + "0" * 64
    all_results: List[Dict[str, Any]] = []

    logger.info(f"Scoring {total} molecules in {n_batches} batches")
    db_scores = batch_get_scores_from_db(score_results_db, [c["name"] for c in candidates])

    for batch_idx in range(n_batches):
        batch = candidates[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        cached_in_batch = []
        to_score_in_batch = []

        for mol in batch:
            if mol["name"] in db_scores:
                cached_in_batch.append(
                    {**mol, "score": db_scores[mol["name"]], "score_source": "db_cache"}
                )
            else:
                to_score_in_batch.append(mol)

        newly_scored: List[Dict[str, Any]] = []
        if to_score_in_batch:
            smiles_list = [m["smiles"] for m in to_score_in_batch]
            names_list = [m["name"] for m in to_score_in_batch]

            valid_molecules_by_uid = {0: {"smiles": smiles_list, "names": names_list}}
            score_dict = {
                0: {
                    "target_scores": [[]],
                    "antitarget_scores": [[]],
                    "entropy": None,
                    "entropy_boltz": None,
                    "block_submitted": None,
                    "push_time": "",
                    "boltz_score": None,
                }
            }
            subnet_config = {
                "weekly_target": boltz_cfg.get("weekly_target", target_seq),
                "binding_pocket": boltz_cfg.get("binding_pocket", None),
                "max_distance": boltz_cfg.get("max_distance", None),
                "force": boltz_cfg.get("force", False),
                "num_molecules_boltz": len(smiles_list),
                "boltz_metric": boltz_cfg.get(
                    "boltz_metric", ["affinity_probability_binary", "affinity_pred_value"]
                ),
                "combination_strategy": boltz_cfg.get(
                    "combination_strategy", "heavy_atom_normalization"
                ),
                "sample_selection": "first",
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

                pm_metric = boltz.per_molecule_metric.get(0, {})
                pm_components = boltz.per_molecule_components.get(0, {})
                target_scores_raw = score_dict[0].get("target_scores", [[]])

                target_scores_list: Optional[List[float]] = None
                if target_scores_raw and len(target_scores_raw[0]) > 0:
                    inner = target_scores_raw[0]
                    target_scores_list = inner if isinstance(inner, list) else [inner]

                batch_avg = score_dict[0].get("boltz_score")
                for mol_idx, mol in enumerate(to_score_in_batch):
                    smiles = mol["smiles"]
                    components = pm_components.get(smiles, {})

                    score = pm_metric.get(smiles)
                    if score is None:
                        score = components.get("affinity_probability_binary")
                    if score is None and target_scores_list is not None and mol_idx < len(
                        target_scores_list
                    ):
                        score = target_scores_list[mol_idx]
                    if score is None and batch_avg is not None:
                        score = batch_avg

                    final_score = float(score) if score is not None else None
                    newly_scored.append(
                        {**mol, "score": final_score, "score_source": "boltz"}
                    )

                logger.info(
                    f"Batch {batch_idx + 1}/{n_batches}: "
                    f"{len(to_score_in_batch)} GPU molecules in {elapsed:.1f}s"
                )

                scored_ok = [r for r in newly_scored if r["score"] is not None]
                if scored_ok:
                    write_scores_to_db(score_results_db, scored_ok)
                    for r in scored_ok:
                        db_scores[r["name"]] = r["score"]

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                logger.debug(traceback.format_exc())
                for mol in to_score_in_batch:
                    newly_scored.append({**mol, "score": None, "score_source": "error"})

        all_results.extend(cached_in_batch)
        all_results.extend(newly_scored)

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# MAIN SCORING PIPELINE
# ══════════════════════════════════════════════════════════════════════════
async def run_scoring_pipeline(
    molecules_df: pd.DataFrame,
    boltz: Any,
    target_seq: str,
    boltz_cfg: Dict[str, Any],
    score_results_db: str,
    batch_size: int = BOLTZ_BATCH_SIZE,
    budget_per_iteration: int = BOLTZ_BUDGET_PER_BATCH,
) -> pd.DataFrame:
    if molecules_df.empty:
        logger.warning("No molecules to score")
        return pd.DataFrame()

    candidates = molecules_df.to_dict("records")
    total = len(candidates)
    all_scored = []
    n_iterations = math.ceil(total / budget_per_iteration)

    logger.info(f"Starting ordered scoring for {total} molecules")
    for iteration in range(n_iterations):
        start_idx = iteration * budget_per_iteration
        end_idx = min((iteration + 1) * budget_per_iteration, total)
        chunk = candidates[start_idx:end_idx]
        logger.info(f"Iteration {iteration + 1}/{n_iterations}: rows {start_idx + 1}-{end_idx}")

        scored_chunk = await score_molecules_with_boltz_batched(
            chunk,
            boltz,
            target_seq,
            boltz_cfg,
            score_results_db=score_results_db,
            batch_size=batch_size,
        )
        all_scored.extend(scored_chunk)

        if iteration < n_iterations - 1:
            await asyncio.sleep(2)

    results_df = pd.DataFrame(all_scored)
    if not results_df.empty and "score" in results_df.columns:
        valid_scores = results_df[results_df["score"].notna()]
        logger.info(
            f"Complete: total={len(results_df)} | valid={len(valid_scores)} | "
            f"failed={len(results_df) - len(valid_scores)}"
        )
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score molecules from CSV in original row order using Boltz."
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.join(BASE_DIR, "data", "rxn2.csv"),
        help="Input CSV with Molecule_ID/name and SMILES columns",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(BASE_DIR, "data", "scored_molecules_ordered_2.csv"),
        help="Output CSV path for scored rows",
    )
    parser.add_argument(
        "--score-db",
        default=os.path.join(BASE_DIR, "score_results_2_2769.sqlite"),
        help="SQLite DB path used as score cache",
    )
    parser.add_argument("--batch-size", type=int, default=BOLTZ_BATCH_SIZE)
    parser.add_argument("--budget-per-iteration", type=int, default=BOLTZ_BUDGET_PER_BATCH)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logger.info("Starting ordered molecule scoring pipeline")
    logger.info("=" * 70)

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    target_seq = getattr(config, "weekly_target", None) or (
        config.get("weekly_target", "") if isinstance(config, dict) else ""
    )
    if not target_seq:
        logger.error("No target protein sequence found in config")
        return

    init_score_results_db(args.score_db)
    molecules_df = load_molecules_in_order(args.input_csv)
    if molecules_df.empty:
        logger.error("No valid molecules loaded")
        return

    boltz_scoring_dir = os.path.join(BASE_DIR, "boltz-scoring")
    boltz_src_dir = os.path.join(boltz_scoring_dir, "boltz", "src")
    for d in [boltz_scoring_dir, boltz_src_dir]:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)

    try:
        from boltz.wrapper import BoltzWrapper

        boltz = BoltzWrapper()
    except Exception as e:
        logger.error(f"BoltzWrapper import/init failed: {e}")
        logger.error(traceback.format_exc())
        return

    boltz_cfg = {
        "weekly_target": target_seq,
        "binding_pocket": getattr(config, "binding_pocket", None),
        "max_distance": getattr(config, "max_distance", None),
        "force": getattr(config, "force", False),
        "boltz_metric": getattr(
            config, "boltz_metric", ["affinity_probability_binary", "affinity_pred_value"]
        ),
        "combination_strategy": getattr(
            config, "combination_strategy", "heavy_atom_normalization"
        ),
        "sample_selection": getattr(config, "sample_selection", "first"),
    }

    try:
        t0 = time.time()
        results_df = await run_scoring_pipeline(
            molecules_df,
            boltz,
            target_seq,
            boltz_cfg,
            score_results_db=args.score_db,
            batch_size=args.batch_size,
            budget_per_iteration=args.budget_per_iteration,
        )
        elapsed = time.time() - t0

        if not results_df.empty:
            out_df = results_df[["name", "score", "score_source"]].copy()
            out_df.to_csv(args.output_csv, index=False)
            logger.info(f"Saved results: {args.output_csv}")
            logger.info(f"Total time: {elapsed/60:.1f} min")
            logger.info(f"Average sec/mol: {elapsed/max(len(results_df), 1):.2f}")
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
