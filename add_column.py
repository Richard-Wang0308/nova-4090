import sqlite3
import os
from typing import Dict, Any
import bittensor as bt
from combinatorial_db.reactions import get_smiles_from_reaction
from utils.molecules import molecule_unique_for_protein_hf

# TODO: Set your target protein here
TARGET_PROTEIN = "Q9UQM7"  # Replace with actual target protein


def scored_molecules_table_exists(cursor: sqlite3.Cursor) -> bool:
    """Return True when the scored_molecules table exists in the connected DB."""
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'scored_molecules' LIMIT 1"
    )
    return cursor.fetchone() is not None


async def check_molecule_unique(target_protein: str, molecule_name: str, smiles: str) -> bool:
    """
    Check if molecule is unique for target protein (NOT in HuggingFace dataset).
    
    Args:
        target_protein: The target protein name
        molecule_name: The molecule identifier (rxn format)
        smiles: The SMILES string of the molecule
    
    Returns:
        True if molecule is NOT in HuggingFace (i.e., it's unique/new)
        False if molecule IS in HuggingFace (i.e., it's already known)
    """
    if not target_protein:
        bt.logging.warning("No target protein provided")
        return False
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(target_protein, smiles)
        
        if not is_unique_hf:
            # bt.logging.debug(f"❌ Molecule {molecule_name} already in HuggingFace dataset")
            return False
        
        # bt.logging.info(f"✅ Molecule {molecule_name} is NOT in HuggingFace (unique!)")
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
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
        
        # Check if column exists and get its type
        cursor.execute("PRAGMA table_info(scored_molecules)")
        columns = {column[1]: column[2] for column in cursor.fetchall()}
        
        if 'available' in columns:
            column_type = columns['available']
            bt.logging.info(f"Found existing 'available' column with type: {column_type}")
            
            if column_type.upper() != 'BOOLEAN':
                bt.logging.info("Converting INTEGER column to BOOLEAN...")
                
                # SQLite doesn't support ALTER COLUMN, so we need to:
                # 1. Create a temporary column
                # 2. Copy data
                # 3. Drop old column
                # 4. Rename temporary column
                
                cursor.execute("ALTER TABLE scored_molecules ADD COLUMN available_temp BOOLEAN")
                cursor.execute("UPDATE scored_molecules SET available_temp = CASE WHEN available = 1 THEN TRUE ELSE FALSE END")
                
                # Get all column names except 'available'
                cursor.execute("PRAGMA table_info(scored_molecules)")
                all_columns = [col[1] for col in cursor.fetchall()]
                other_columns = [col for col in all_columns if col not in ['available', 'available_temp']]
                
                # Create new table with correct schema
                columns_list = ', '.join(other_columns)
                cursor.execute(f"""
                    CREATE TABLE scored_molecules_new (
                        {columns_list.replace('molecule_name', 'molecule_name TEXT')},
                        available BOOLEAN
                    )
                """)
                
                # Copy data to new table
                cursor.execute(f"""
                    INSERT INTO scored_molecules_new ({columns_list}, available)
                    SELECT {columns_list}, available_temp FROM scored_molecules
                """)
                
                # Drop old table and rename new one
                cursor.execute("DROP TABLE scored_molecules")
                cursor.execute("ALTER TABLE scored_molecules_new RENAME TO scored_molecules")
                
                conn.commit()
                bt.logging.info("✅ Successfully converted 'available' column from INTEGER to BOOLEAN")
            else:
                bt.logging.info("✅ 'available' column is already BOOLEAN type")
        else:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE scored_molecules ADD COLUMN available BOOLEAN")
            conn.commit()
            bt.logging.info("✅ Added 'available' column (BOOLEAN) to scored_molecules table")
        
        conn.close()
    except Exception as e:
        bt.logging.error(f"Error fixing available column: {e}")
        raise


def fix_available_column_simple(db_path: str):
    """
    Simpler approach: Drop existing column and recreate as BOOLEAN.
    This is easier but will lose existing data in the column.
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
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(scored_molecules)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'available' in columns:
            bt.logging.info("Dropping existing 'available' column...")
            
            # Get all columns except 'available'
            cursor.execute("PRAGMA table_info(scored_molecules)")
            all_columns = [col[1] for col in cursor.fetchall() if col[1] != 'available']
            columns_list = ', '.join(all_columns)
            
            # Create new table without 'available' column
            cursor.execute(f"""
                CREATE TABLE scored_molecules_temp AS
                SELECT {columns_list} FROM scored_molecules
            """)
            
            # Drop old table
            cursor.execute("DROP TABLE scored_molecules")
            
            # Rename temp table
            cursor.execute("ALTER TABLE scored_molecules_temp RENAME TO scored_molecules")
            
            bt.logging.info("Dropped old 'available' column")
        
        # Add new BOOLEAN column
        cursor.execute("ALTER TABLE scored_molecules ADD COLUMN available BOOLEAN")
        conn.commit()
        bt.logging.info("✅ Added new 'available' column as BOOLEAN")
        
        conn.close()
    except Exception as e:
        bt.logging.error(f"Error fixing available column: {e}")
        raise


async def update_available_values(db_path: str, target_protein: str, force_recalculate: bool = False):
    """
    Update available values for molecules in scored_molecules table.
    ONLY processes molecules where available is TRUE (skips FALSE and NULL).
    
    Args:
        db_path: Path to the SQLite database
        target_protein: The target protein name to check against
        force_recalculate: If True, recalculate all molecules. If False, only process TRUE values.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if not scored_molecules_table_exists(cursor):
            bt.logging.warning(
                f"⚠️  Table 'scored_molecules' not found in {db_path}; skipping availability update."
            )
            conn.close()
            return
        
        # Get molecules that need processing
        if force_recalculate:
            bt.logging.info("Force recalculate mode: Processing ALL molecules...")
            cursor.execute("SELECT molecule_name FROM scored_molecules")
        else:
            # FIXED: Only process rows where available is TRUE
            bt.logging.info("Incremental mode: Processing only molecules with available = TRUE...")
            cursor.execute("SELECT molecule_name FROM scored_molecules WHERE available = TRUE")
        
        molecules = cursor.fetchall()
        
        total = len(molecules)
        
        if total == 0:
            bt.logging.info("✅ No molecules to process. All molecules already have been processed or have FALSE values!")
            conn.close()
            return
        
        bt.logging.info(f"Found {total} molecules to process...")
        
        success_count = 0
        error_count = 0
        updated_to_false = 0
        updated_to_true = 0
        
        for idx, (molecule_name,) in enumerate(molecules, 1):
            try:
                # Get SMILES from molecule_name
                smiles = get_smiles_from_reaction(molecule_name)
                
                if smiles is None:
                    bt.logging.warning(f"Could not generate SMILES for {molecule_name}")
                    available = False
                    updated_to_false += 1
                    error_count += 1
                else:
                    # Check if molecule is unique
                    available = await check_molecule_unique(target_protein, molecule_name, smiles)
                    
                    if available:
                        updated_to_true += 1
                    else:
                        updated_to_false += 1
                    
                    success_count += 1
                
                # Update database with TRUE/FALSE
                cursor.execute(
                    "UPDATE scored_molecules SET available = ? WHERE molecule_name = ?",
                    (available, molecule_name)
                )
                
                if idx % 100 == 0:
                    conn.commit()
                    bt.logging.info(
                        f"Progress: {idx}/{total} | Success: {success_count} | "
                        f"Errors: {error_count} | Updated to TRUE: {updated_to_true} | "
                        f"Updated to FALSE: {updated_to_false}"
                    )
                
            except Exception as e:
                bt.logging.error(f"Error processing molecule {molecule_name}: {e}")
                error_count += 1
                # Set to FALSE on error
                cursor.execute(
                    "UPDATE scored_molecules SET available = ? WHERE molecule_name = ?",
                    (False, molecule_name)
                )
                updated_to_false += 1
        
        conn.commit()
        conn.close()
        
        bt.logging.info("=" * 70)
        bt.logging.info(f"✅ Successfully processed {total} molecules")
        bt.logging.info(f"   Success: {success_count} | Errors: {error_count}")
        bt.logging.info(f"   Updated to TRUE: {updated_to_true} | Updated to FALSE: {updated_to_false}")
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
        
        # Check if 'available' column exists
        cursor.execute("PRAGMA table_info(scored_molecules)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'available' not in columns:
            # Column doesn't exist yet
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
        
        # Total rows
        cursor.execute("SELECT COUNT(*) FROM scored_molecules")
        total = cursor.fetchone()[0]
        
        # NULL values
        cursor.execute("SELECT COUNT(*) FROM scored_molecules WHERE available IS NULL")
        null_count = cursor.fetchone()[0]
        
        # TRUE values
        cursor.execute("SELECT COUNT(*) FROM scored_molecules WHERE available = TRUE")
        true_count = cursor.fetchone()[0]
        
        # FALSE values
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
    db_path = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.exists(db_path):
        bt.logging.warning(
            f"⚠️  Database file does not exist: {db_path}. Skipping update."
        )
        return
    
    # Fix available column first (only needed once)
    # This ensures the column exists before we try to get statistics
    if not skip_column_fix:
        bt.logging.info("Fixing column type...")
        # Method 1: Simple approach - drops and recreates column (loses existing data)
        # fix_available_column_simple(db_path)
        
        # Method 2: Complex approach - preserves existing data
        fix_available_column(db_path)
    else:
        bt.logging.info("Skipping column fix (already done)")
    
    # Show current statistics (after column is fixed)
    bt.logging.info("\nChecking current database state...")
    get_statistics(db_path)
    
    # Update available values (only TRUE values by default)
    bt.logging.info("\nUpdating available values...")
    await update_available_values(db_path, TARGET_PROTEIN, force_recalculate=force_recalculate)
    
    # Show final statistics
    bt.logging.info("\nFinal database state:")
    get_statistics(db_path)


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix/refresh the 'available' column in a score_results SQLite DB."
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
        default=os.path.join(os.path.dirname(__file__), "score_results_1.sqlite"),
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
#python3 add_column.py --db-path score_results_2.sqlite --force