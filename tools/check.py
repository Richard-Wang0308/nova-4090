import sqlite3
import os
import sys
import time
from typing import Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import bittensor as bt
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from combinatorial_db.reactions import get_smiles_from_reaction
from utils.molecules import molecule_unique_for_protein_hf, get_brenk_matches
# TODO: adjust import path to match where this helper actually lives
from utils import get_historical_submissions

# Read both from config rather than hardcoding them. Keeping this in sync by
# hand is what let MAX_SIMILARITY_TO_HISTORICAL sit at 0.9 while config said
# 0.7, marking molecules available=TRUE that the validator would reject.
from config.config_loader import load_config as _load_config

try:
    _cfg = _load_config()
    TARGET_PROTEIN = _cfg["small_molecule_target"][0]
    MAX_SIMILARITY_TO_HISTORICAL = float(_cfg["max_similarity_to_historical"])
except Exception as _e:  # keep the tool usable if config is unreadable
    TARGET_PROTEIN = "P40261"
    MAX_SIMILARITY_TO_HISTORICAL = 0.6
    print(f"[check] could not read config ({_e}); "
          f"falling back to {TARGET_PROTEIN} / {MAX_SIMILARITY_TO_HISTORICAL}")


def scored_molecules_table_exists(cursor: sqlite3.Cursor) -> bool:
    """Return True when the scored_molecules table exists in the connected DB."""
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'scored_molecules' LIMIT 1"
    )
    return cursor.fetchone() is not None


def load_historical_fingerprints(target_protein: str):
    """
    Load historical submissions for the target protein once and precompute
    Morgan fingerprints for fast Tanimoto similarity comparisons.

    Returns:
        A DataFrame with a 'fingerprint' column, or None if no historical
        submissions exist for this target.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    historical_df = get_historical_submissions(target_protein, "molecules")

    if historical_df is None or historical_df.empty:
        bt.logging.warning(
            f"No historical submissions found for target '{target_protein}'"
        )
        return None

    mols = [Chem.MolFromSmiles(smi) for smi in historical_df["SMILES"]]

    # Filter out any SMILES that failed to parse, keeping df/mols in sync
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    if len(valid_idx) != len(mols):
        bt.logging.warning(
            f"Dropped {len(mols) - len(valid_idx)} unparsable historical SMILES "
            f"for target '{target_protein}'"
        )
        historical_df = historical_df.iloc[valid_idx].reset_index(drop=True)
        mols = [mols[i] for i in valid_idx]

    if not mols:
        return None

    fps = morgan_gen.GetFingerprints(mols, numThreads=8)
    historical_df = historical_df.copy()
    historical_df["fingerprint"] = list(fps)

    return historical_df


def is_diverse_enough(
    mol_fp,
    historical_fps,
    max_similarity: float,
) -> bool:
    """
    Check whether a molecule's fingerprint is sufficiently diverse (not too
    similar) compared to all historical submission fingerprints.

    Args:
        historical_fps: Precomputed list of Morgan fingerprints (materialize
            once per run — do NOT rebuild from a DataFrame every call).

    Returns:
        True if diverse enough (no historical similarity >= max_similarity)
        False if too similar to at least one historical submission
    """
    if not historical_fps:
        return True

    similarities = DataStructs.BulkTanimotoSimilarity(mol_fp, historical_fps)
    return not any(sim >= max_similarity for sim in similarities)


def load_hf_inchikey_set(target_protein: str) -> set:
    """
    Load the HuggingFace InChIKey set once for this run.

    Warms molecule_unique_for_protein_hf's cache, then returns the set so
    callers can do a single MolFromSmiles → InChIKey lookup without re-parsing
    inside the HF helper on every row.
    """
    # Trigger cache load (metadata + CSV) via a trivial valid SMILES.
    molecule_unique_for_protein_hf(target_protein, "C")
    cache = getattr(molecule_unique_for_protein_hf, "_CACHE", None)
    if not cache:
        return set()
    inchikeys_set = cache[2]
    return inchikeys_set if inchikeys_set is not None else set()


def check_availability_fast(
    smiles: str,
    inchikeys_set: set,
    historical_fps,
    morgan_gen,
    max_similarity: float = MAX_SIMILARITY_TO_HISTORICAL,
) -> tuple:
    """
    Single-parse availability check.

    Returns:
        (available: bool, reason: str)
        reason is one of: "" | "bad_smiles" | "hf_duplicate" | "brenk" |
        "too_similar"
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "bad_smiles"

    # Exact match vs HF archive
    if Chem.MolToInchiKey(mol) in inchikeys_set:
        return False, "hf_duplicate"

    # BRENK structural alerts — one match invalidates the whole submission
    # on the validator side, so the molecule can never be submitted.
    if get_brenk_matches(mol):
        return False, "brenk"

    # Diversity vs historical submissions
    mol_fp = morgan_gen.GetFingerprint(mol)
    if not is_diverse_enough(mol_fp, historical_fps, max_similarity):
        return False, "too_similar"

    return True, ""


async def check_molecule_available(
    target_protein: str,
    molecule_name: str,
    smiles: str,
    historical_df,
    morgan_gen,
    max_similarity: float = MAX_SIMILARITY_TO_HISTORICAL,
    historical_fps=None,
    inchikeys_set: Optional[set] = None,
) -> bool:
    """
    Check if a molecule is "available" for the target protein:
      1. NOT already present in the HuggingFace dataset (exact match)
      2. NOT disallowed by the BRENK structural-alert filter
      3. NOT too similar (Tanimoto >= max_similarity) to any historical
         submission for the target protein

    Returns:
        True  -> molecule is available (unique AND diverse)
        False -> molecule already known OR too similar to a prior submission
    """
    if not target_protein:
        bt.logging.warning("No target protein provided")
        return False

    try:
        if historical_fps is None:
            if historical_df is None or getattr(historical_df, "empty", True):
                historical_fps = []
            else:
                historical_fps = list(historical_df["fingerprint"])

        if inchikeys_set is None:
            inchikeys_set = load_hf_inchikey_set(target_protein)

        available, reason = check_availability_fast(
            smiles, inchikeys_set, historical_fps, morgan_gen, max_similarity
        )
        if reason == "bad_smiles":
            bt.logging.warning(
                f"Could not parse SMILES for {molecule_name}: '{smiles}'"
            )
        elif reason == "brenk":
            bt.logging.debug(
                f"❌ Molecule {molecule_name} disallowed by the BRENK filter"
            )
        elif reason == "too_similar":
            bt.logging.debug(
                f"❌ Molecule {molecule_name} too similar to a historical "
                f"submission for target '{target_protein}'"
            )
        return available

    except Exception as e:
        bt.logging.error(f"Error checking availability: {e}")
        return False


def fix_available_column(db_path: str):
    """
    Fix 'available' column in scored_molecules table.
    - If column exists as INTEGER, recreate it as BOOLEAN
    - If column doesn't exist, add it as BOOLEAN
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not scored_molecules_table_exists(cursor):
            bt.logging.warning(
                f"⚠️  Table 'scored_molecules' not found in {db_path}; skipping column fix."
            )
            conn.close()
            return

        cursor.execute("PRAGMA table_info(scored_molecules)")
        columns = {column[1]: column[2] for column in cursor.fetchall()}

        if 'available' in columns:
            column_type = columns['available']
            bt.logging.info(f"Found existing 'available' column with type: {column_type}")

            if column_type.upper() != 'BOOLEAN':
                bt.logging.info("Converting INTEGER column to BOOLEAN...")

                cursor.execute("ALTER TABLE scored_molecules ADD COLUMN available_temp BOOLEAN")
                cursor.execute("UPDATE scored_molecules SET available_temp = CASE WHEN available = 1 THEN TRUE ELSE FALSE END")

                cursor.execute("PRAGMA table_info(scored_molecules)")
                all_columns = [col[1] for col in cursor.fetchall()]
                other_columns = [col for col in all_columns if col not in ['available', 'available_temp']]

                columns_list = ', '.join(other_columns)
                cursor.execute(f"""
                    CREATE TABLE scored_molecules_new (
                        {columns_list.replace('molecule_name', 'molecule_name TEXT')},
                        available BOOLEAN
                    )
                """)

                cursor.execute(f"""
                    INSERT INTO scored_molecules_new ({columns_list}, available)
                    SELECT {columns_list}, available_temp FROM scored_molecules
                """)

                cursor.execute("DROP TABLE scored_molecules")
                cursor.execute("ALTER TABLE scored_molecules_new RENAME TO scored_molecules")

                conn.commit()
                bt.logging.info("✅ Successfully converted 'available' column from INTEGER to BOOLEAN")
            else:
                bt.logging.info("✅ 'available' column is already BOOLEAN type")
        else:
            cursor.execute("ALTER TABLE scored_molecules ADD COLUMN available BOOLEAN")
            conn.commit()
            bt.logging.info("✅ Added 'available' column (BOOLEAN) to scored_molecules table")

        conn.close()
    except Exception as e:
        bt.logging.error(f"Error fixing available column: {e}")
        raise


async def update_available_values(db_path: str, target_protein: str, force_recalculate: bool = False):
    """
    Update available values for molecules in scored_molecules table.
    A molecule is 'available' (TRUE) only if:
      1. It is NOT already in the HuggingFace dataset for the target protein, AND
      2. It is NOT disallowed by the BRENK structural-alert filter, AND
      3. It is NOT too similar (Tanimoto >= MAX_SIMILARITY_TO_HISTORICAL) to any
         historical submission for the target protein.

    ONLY processes molecules where available is TRUE (skips FALSE and NULL),
    unless force_recalculate is True.
    """
    # Flush pending UPDATE rows this often (also drives progress logs).
    WRITE_BATCH_SIZE = 2000

    try:
        t0 = time.perf_counter()
        conn = sqlite3.connect(db_path)
        # Faster bulk writes: WAL + less fsync pressure.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        bt.logging.info(f"⏱️  DB connect: {time.perf_counter() - t0:.3f}s ({db_path})")
        cursor = conn.cursor()

        if not scored_molecules_table_exists(cursor):
            bt.logging.warning(
                f"⚠️  Table 'scored_molecules' not found in {db_path}; skipping availability update."
            )
            conn.close()
            return

        if force_recalculate:
            bt.logging.info("Force recalculate mode: Processing ALL molecules...")
            cursor.execute("SELECT molecule_name FROM scored_molecules")
        else:
            bt.logging.info("Incremental mode: Processing only molecules with available = TRUE...")
            cursor.execute("SELECT molecule_name FROM scored_molecules WHERE available = TRUE")

        molecules = cursor.fetchall()
        total = len(molecules)

        if total == 0:
            bt.logging.info("✅ No molecules to process. All molecules already have been processed or have FALSE values!")
            conn.close()
            return

        bt.logging.info(f"Found {total} molecules to process...")

        # --- Load historical submissions & fingerprints ONCE for this run ---
        bt.logging.info(f"Loading historical submissions for target '{target_protein}'...")
        t_hist = time.perf_counter()
        historical_df = load_historical_fingerprints(target_protein)
        # Materialize once — rebuilding list(df["fingerprint"]) per row was ~0.35ms each.
        historical_fps = (
            [] if historical_df is None else list(historical_df["fingerprint"])
        )
        bt.logging.info(
            f"⏱️  Historical submissions loaded in {time.perf_counter() - t_hist:.3f}s "
            f"({len(historical_fps)} fingerprints)"
        )

        # --- Load HF InChIKey set ONCE (skip per-row re-parse inside HF helper) ---
        t_hf = time.perf_counter()
        inchikeys_set = load_hf_inchikey_set(target_protein)
        bt.logging.info(
            f"⏱️  HF InChIKey set loaded in {time.perf_counter() - t_hf:.3f}s "
            f"({len(inchikeys_set)} keys)"
        )

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        success_count = 0
        error_count = 0
        updated_to_false = 0
        updated_to_true = 0
        skipped_hf = 0
        skipped_brenk = 0
        skipped_similarity = 0
        availability_checks = 0
        availability_batch_time = 0.0
        availability_total_time = 0.0
        pending_updates = []
        overall_start = time.perf_counter()

        def flush_updates():
            if not pending_updates:
                return
            cursor.executemany(
                "UPDATE scored_molecules SET available = ? WHERE molecule_name = ?",
                pending_updates,
            )
            conn.commit()
            pending_updates.clear()

        for idx, (molecule_name,) in enumerate(molecules, 1):
            try:
                smiles = get_smiles_from_reaction(molecule_name)

                if smiles is None:
                    bt.logging.warning(f"Could not generate SMILES for {molecule_name}")
                    available = False
                    updated_to_false += 1
                    error_count += 1
                else:
                    t_check = time.perf_counter()
                    available, reason = check_availability_fast(
                        smiles,
                        inchikeys_set,
                        historical_fps,
                        morgan_gen,
                        MAX_SIMILARITY_TO_HISTORICAL,
                    )
                    check_elapsed = time.perf_counter() - t_check
                    availability_checks += 1
                    availability_batch_time += check_elapsed
                    availability_total_time += check_elapsed

                    if reason == "bad_smiles":
                        bt.logging.warning(
                            f"Could not parse SMILES for {molecule_name}: '{smiles}'"
                        )
                        updated_to_false += 1
                    elif reason == "hf_duplicate":
                        skipped_hf += 1
                        updated_to_false += 1
                    elif reason == "brenk":
                        skipped_brenk += 1
                        updated_to_false += 1
                    elif reason == "too_similar":
                        skipped_similarity += 1
                        updated_to_false += 1
                    else:
                        updated_to_true += 1

                    success_count += 1

                pending_updates.append((available, molecule_name))

                if availability_checks > 0 and availability_checks % 1000 == 0:
                    bt.logging.info(
                        f"⏱️  Availability check: last 1000 rows took {availability_batch_time:.3f}s "
                        f"({availability_batch_time / 1000:.4f}s/row) | "
                        f"total so far: {availability_total_time:.3f}s "
                        f"over {availability_checks} checks"
                    )
                    availability_batch_time = 0.0

                if len(pending_updates) >= WRITE_BATCH_SIZE:
                    flush_updates()
                    bt.logging.info(
                        f"Progress: {idx}/{total} | Success: {success_count} | "
                        f"Errors: {error_count} | TRUE: {updated_to_true} | "
                        f"FALSE: {updated_to_false} (HF dup: {skipped_hf}, "
                        f"BRENK: {skipped_brenk}, "
                        f"too similar: {skipped_similarity})"
                    )

            except Exception as e:
                bt.logging.error(f"Error processing molecule {molecule_name}: {e}")
                error_count += 1
                pending_updates.append((False, molecule_name))
                updated_to_false += 1

        remainder = availability_checks % 1000
        if remainder > 0:
            bt.logging.info(
                f"⏱️  Availability check: last {remainder} rows took {availability_batch_time:.3f}s "
                f"({availability_batch_time / remainder:.4f}s/row) | "
                f"total: {availability_total_time:.3f}s "
                f"over {availability_checks} checks"
            )

        flush_updates()
        conn.close()

        overall_elapsed = time.perf_counter() - overall_start
        bt.logging.info("=" * 70)
        bt.logging.info(f"✅ Successfully processed {total} molecules")
        bt.logging.info(f"   Success: {success_count} | Errors: {error_count}")
        bt.logging.info(
            f"   Updated to TRUE: {updated_to_true} | Updated to FALSE: {updated_to_false} "
            f"(HF duplicate: {skipped_hf}, BRENK: {skipped_brenk}, "
            f"too similar: {skipped_similarity})"
        )
        bt.logging.info(
            f"⏱️  Timing summary: overall={overall_elapsed:.3f}s | "
            f"availability_total={availability_total_time:.3f}s "
            f"({availability_checks} checks"
            + (
                f", avg {availability_total_time / availability_checks:.4f}s/row)"
                if availability_checks
                else ")"
            )
        )
        bt.logging.info("=" * 70)

    except Exception as e:
        bt.logging.error(f"Error updating available values: {e}")
        raise


def get_statistics(db_path: str):
    """Get statistics about available values in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not scored_molecules_table_exists(cursor):
            conn.close()
            bt.logging.warning(
                f"⚠️  Table 'scored_molecules' not found in {db_path}; no statistics to report."
            )
            return {
                'total': 0,
                'null': 0,
                'true': 0,
                'false': 0,
                'column_exists': False,
                'table_exists': False,
            }

        cursor.execute("PRAGMA table_info(scored_molecules)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'available' not in columns:
            cursor.execute("SELECT COUNT(*) FROM scored_molecules")
            total = cursor.fetchone()[0]
            conn.close()

            bt.logging.info("=" * 60)
            bt.logging.info("DATABASE STATISTICS")
            bt.logging.info("=" * 60)
            bt.logging.info(f"Total molecules:        {total}")
            bt.logging.info("Available column:       Not yet created")
            bt.logging.info("=" * 60)

            return {
                'total': total,
                'null': 0,
                'true': 0,
                'false': 0,
                'column_exists': False
            }

        cursor.execute("SELECT COUNT(*) FROM scored_molecules")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM scored_molecules WHERE available IS NULL")
        null_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM scored_molecules WHERE available = TRUE")
        true_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM scored_molecules WHERE available = FALSE")
        false_count = cursor.fetchone()[0]

        conn.close()

        bt.logging.info("=" * 60)
        bt.logging.info("DATABASE STATISTICS")
        bt.logging.info("=" * 60)
        bt.logging.info(f"Total molecules:        {total}")
        if total == 0:
            bt.logging.info("Not yet processed:      0 (n/a — empty table)")
            bt.logging.info("Available (TRUE):       0 (n/a — empty table)")
            bt.logging.info("Not available (FALSE):  0 (n/a — empty table)")
        else:
            bt.logging.info(f"Not yet processed:      {null_count} ({null_count/total*100:.1f}%)")
            bt.logging.info(f"Available (TRUE):       {true_count} ({true_count/total*100:.1f}%)")
            bt.logging.info(f"Not available (FALSE):  {false_count} ({false_count/total*100:.1f}%)")
        bt.logging.info("=" * 60)

        return {
            'total': total,
            'null': null_count,
            'true': true_count,
            'false': false_count,
            'column_exists': True
        }

    except Exception as e:
        bt.logging.error(f"Error getting statistics: {e}")
        return None


async def main(
    force_recalculate: bool = False,
    skip_column_fix: bool = False,
    db_path: str = "score_results.sqlite",
):
    """
    Main function to fix column type and update values.

    Args:
        force_recalculate: If True, recalculate all molecules. If False, only process TRUE values.
        skip_column_fix: If True, skip the column type fix (useful for subsequent runs)
    """
    db_path = os.path.expanduser(db_path)
    if not os.path.isabs(db_path):
        db_path = os.path.join(BASE_DIR, db_path)
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        bt.logging.warning(
            f"⚠️  Database file does not exist: {db_path}. Skipping update."
        )
        return

    if not skip_column_fix:
        bt.logging.info("Fixing column type...")
        fix_available_column(db_path)
    else:
        bt.logging.info("Skipping column fix (already done)")

    bt.logging.info("\nChecking current database state...")
    get_statistics(db_path)

    bt.logging.info("\nUpdating available values (uniqueness + BRENK + diversity)...")
    await update_available_values(db_path, TARGET_PROTEIN, force_recalculate=force_recalculate)

    bt.logging.info("\nFinal database state:")
    get_statistics(db_path)


if __name__ == "__main__":
    import asyncio
    import argparse

    bt.logging.enable_info()

    parser = argparse.ArgumentParser(
        description="Fix/refresh the 'available' column in a score_results SQLite DB, "
                    "checking HuggingFace uniqueness, BRENK structural alerts and "
                    "historical diversity."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate availability for all molecules",
    )
    parser.add_argument(
        "--skip-fix",
        action="store_true",
        help="Skip available-column schema fix and only run updates",
    )
    parser.add_argument(
        "--db-path",
        default=os.path.join(BASE_DIR, "score_results_1.sqlite"),
        help="Path to target SQLite database",
    )
    args = parser.parse_args()

    if args.force:
        bt.logging.warning("⚠️  FORCE RECALCULATE MODE: Will process ALL molecules")

    asyncio.run(
        main(
            force_recalculate=args.force,
            skip_column_fix=args.skip_fix,
            db_path=args.db_path,
        )
    )