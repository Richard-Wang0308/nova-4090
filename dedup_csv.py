import os
import asyncio
import sys
from typing import Set, List, Dict, Any
import pandas as pd
import bittensor as bt
from combinatorial_db.reactions import get_smiles_from_reaction
from utils.molecules import molecule_unique_for_protein_hf

# TODO: Set your target protein here
TARGET_PROTEIN = "P23975"  # Replace with actual target protein


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
            return False
        
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness for {molecule_name}: {e}")
        return False


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file with molecule data.
    
    Expected CSV format (with or without header):
        molecule_name,score,epoch
        rxn:5:182442:229331:223312,0.1928808007921491,21771
        rxn:5:198085:223312:229343,0.1826991908252239,21771
    
    Or just molecule names:
        rxn:5:182442:229331:223312
        rxn:5:198085:223312:229343
    
    Returns:
        DataFrame with at least 'molecule_name' column
    """
    try:
        # Try reading with header
        df = pd.read_csv(csv_path)
        
        # Check if first column looks like molecule names
        first_col = df.columns[0]
        if not str(df[first_col].iloc[0]).startswith('rxn:'):
            # First row is actual data, not header
            df = pd.read_csv(csv_path, header=None)
            df.columns = ['molecule_name'] + [f'col_{i}' for i in range(1, len(df.columns))]
        else:
            # Has proper header, ensure molecule_name column exists
            if 'molecule_name' not in df.columns:
                # Rename first column to molecule_name
                df.rename(columns={df.columns[0]: 'molecule_name'}, inplace=True)
        
        bt.logging.info(f"Loaded CSV with {len(df)} molecules")
        bt.logging.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        bt.logging.error(f"Error loading CSV: {e}")
        raise


def ensure_available_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'available' column exists in DataFrame.
    If it doesn't exist, add it with None values.
    
    Args:
        df: DataFrame with molecule data
    
    Returns:
        DataFrame with 'available' column
    """
    if 'available' not in df.columns:
        bt.logging.info("Adding 'available' column to DataFrame...")
        df['available'] = None
        bt.logging.info("✅ Added 'available' column")
    else:
        bt.logging.info(f"✅ 'available' column already exists")
    
    return df


async def update_available_values(df: pd.DataFrame, target_protein: str, batch_size: int = 100) -> pd.DataFrame:
    """
    Update available values for molecules in DataFrame.
    Processes molecules where available is None/NaN (not yet checked).
    
    Args:
        df: DataFrame with molecule data
        target_protein: The target protein name to check against
        batch_size: Number of molecules to process before logging progress
    
    Returns:
        Updated DataFrame with available column filled
    """
    try:
        # Get molecules that haven't been processed yet (available is None/NaN)
        unprocessed_mask = df['available'].isna()
        unprocessed_df = df[unprocessed_mask]
        
        total = len(unprocessed_df)
        
        if total == 0:
            bt.logging.info("✅ No molecules to process. All molecules have been checked!")
            return df
        
        bt.logging.info(f"Found {total} molecules to process...")
        
        success_count = 0
        error_count = 0
        unique_count = 0
        duplicate_count = 0
        
        # Process each unprocessed molecule
        for idx, (row_idx, row) in enumerate(unprocessed_df.iterrows(), 1):
            molecule_name = row['molecule_name']
            
            try:
                # Get SMILES from molecule_name
                smiles = get_smiles_from_reaction(molecule_name)
                
                if smiles is None:
                    bt.logging.warning(f"Could not generate SMILES for {molecule_name}")
                    available = False
                    error_count += 1
                else:
                    # Check if molecule is unique
                    available = await check_molecule_unique(target_protein, molecule_name, smiles)
                    
                    if available:
                        unique_count += 1
                    else:
                        duplicate_count += 1
                    
                    success_count += 1
                
                # Update DataFrame
                df.at[row_idx, 'available'] = available
                
                # Log progress
                if idx % batch_size == 0:
                    bt.logging.info(
                        f"Progress: {idx}/{total} ({idx/total*100:.1f}%) | "
                        f"Unique: {unique_count} | Duplicates: {duplicate_count} | Errors: {error_count}"
                    )
                
            except Exception as e:
                bt.logging.error(f"Error processing molecule {molecule_name}: {e}")
                error_count += 1
                # Set to False on error
                df.at[row_idx, 'available'] = False
        
        bt.logging.info("=" * 70)
        bt.logging.info(f"✅ Successfully processed {total} molecules")
        bt.logging.info(f"   Unique molecules: {unique_count} ({unique_count/total*100:.1f}%)")
        bt.logging.info(f"   Duplicate molecules: {duplicate_count} ({duplicate_count/total*100:.1f}%)")
        bt.logging.info(f"   Errors: {error_count} ({error_count/total*100:.1f}%)")
        bt.logging.info("=" * 70)
        
        return df
        
    except Exception as e:
        bt.logging.error(f"Error updating available values: {e}")
        raise


def remove_duplicates(df: pd.DataFrame, dry_run: bool = True) -> pd.DataFrame:
    """
    Remove duplicate molecules (where available = False) from DataFrame.
    
    Args:
        df: DataFrame with molecule data
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        DataFrame with duplicates removed (if not dry_run)
    """
    try:
        # Count duplicates (available == False)
        duplicate_mask = df['available'] == False
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count == 0:
            bt.logging.info("✅ No duplicates to remove!")
            return df
        
        if dry_run:
            bt.logging.info("=" * 70)
            bt.logging.info("DRY RUN MODE - No data will be deleted")
            bt.logging.info("=" * 70)
            bt.logging.info(f"Would delete {duplicate_count} duplicate molecules")
            
            # Show some examples
            examples = df[duplicate_mask]['molecule_name'].head(10).tolist()
            bt.logging.info("\nExample molecules that would be deleted:")
            for mol in examples:
                bt.logging.info(f"  - {mol}")
            
            if duplicate_count > 10:
                bt.logging.info(f"  ... and {duplicate_count - 10} more")
            
            return df
        else:
            bt.logging.warning(f"⚠️  DELETING {duplicate_count} duplicate molecules...")
            df_cleaned = df[~duplicate_mask].copy()
            bt.logging.info(f"✅ Successfully removed {duplicate_count} duplicate molecules")
            bt.logging.info(f"   Remaining molecules: {len(df_cleaned)}")
            return df_cleaned
        
    except Exception as e:
        bt.logging.error(f"Error removing duplicates: {e}")
        raise


def get_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about available values in the DataFrame."""
    try:
        total = len(df)
        
        if 'available' not in df.columns:
            bt.logging.info("=" * 70)
            bt.logging.info("DATAFRAME STATISTICS")
            bt.logging.info("=" * 70)
            bt.logging.info(f"Total molecules:        {total}")
            bt.logging.info("Available column:       Not yet created")
            bt.logging.info("=" * 70)
            
            return {
                'total': total,
                'null': total,
                'true': 0,
                'false': 0,
                'column_exists': False
            }
        
        # Count values
        null_count = df['available'].isna().sum()
        true_count = (df['available'] == True).sum()
        false_count = (df['available'] == False).sum()
        
        bt.logging.info("=" * 70)
        bt.logging.info("DATAFRAME STATISTICS")
        bt.logging.info("=" * 70)
        bt.logging.info(f"Total molecules:        {total}")
        bt.logging.info(f"Not yet processed:      {null_count} ({null_count/total*100:.1f}%)")
        bt.logging.info(f"Unique (TRUE):          {true_count} ({true_count/total*100:.1f}%)")
        bt.logging.info(f"Duplicates (FALSE):     {false_count} ({false_count/total*100:.1f}%)")
        bt.logging.info("=" * 70)
        
        return {
            'total': total,
            'null': int(null_count),
            'true': int(true_count),
            'false': int(false_count),
            'column_exists': True
        }
        
    except Exception as e:
        bt.logging.error(f"Error getting statistics: {e}")
        return None


def save_csv(df: pd.DataFrame, csv_path: str, backup: bool = True):
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        csv_path: Path to save CSV
        backup: If True, create backup of original file
    """
    try:
        # Create backup if requested and file exists
        if backup and os.path.exists(csv_path):
            backup_path = csv_path + '.backup'
            import shutil
            shutil.copy2(csv_path, backup_path)
            bt.logging.info(f"📦 Backup created: {backup_path}")
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        bt.logging.info(f"💾 Saved to: {csv_path}")
        
    except Exception as e:
        bt.logging.error(f"Error saving CSV: {e}")
        raise


async def main(mode: str = "check", batch_size: int = 100):
    """
    Main function to deduplicate CSV against target protein.
    
    Args:
        mode: Operation mode
            - "check": Only check and mark duplicates (default)
            - "remove": Check, mark, and remove duplicates
            - "remove-only": Only remove already marked duplicates
            - "stats": Only show statistics
        batch_size: Number of molecules to process before logging progress
    """
    csv_path = "data/data.csv"
    
    if not os.path.exists(csv_path):
        bt.logging.error(f"CSV file not found: {csv_path}")
        return
    
    bt.logging.info(f"Using CSV file: {csv_path}")
    bt.logging.info(f"Target protein: {TARGET_PROTEIN}")
    bt.logging.info(f"Mode: {mode}")
    bt.logging.info("=" * 70)
    
    # Load CSV
    bt.logging.info("\n📂 Loading CSV file...")
    df = load_csv(csv_path)
    
    # Ensure column exists
    df = ensure_available_column(df)
    
    # Show initial statistics
    bt.logging.info("\nInitial data state:")
    stats = get_statistics(df)
    
    if stats is None:
        return
    
    # Execute based on mode
    if mode == "stats":
        bt.logging.info("\n✅ Statistics only mode - done!")
        return
    
    elif mode == "remove-only":
        bt.logging.info("\n🗑️  Remove-only mode: Deleting marked duplicates...")
        df = remove_duplicates(df, dry_run=False)
        save_csv(df, csv_path, backup=True)
        
    elif mode == "check":
        bt.logging.info("\n🔍 Check mode: Marking duplicates (not deleting)...")
        df = await update_available_values(df, TARGET_PROTEIN, batch_size=batch_size)
        
        # Save with available column
        save_csv(df, csv_path, backup=True)
        
        bt.logging.info("\n📊 Showing what would be deleted (dry run):")
        remove_duplicates(df, dry_run=True)
        
    elif mode == "remove":
        bt.logging.info("\n🔍 Checking and marking duplicates...")
        df = await update_available_values(df, TARGET_PROTEIN, batch_size=batch_size)
        
        bt.logging.info("\n🗑️  Removing duplicates...")
        df = remove_duplicates(df, dry_run=False)
        
        # Save cleaned data
        save_csv(df, csv_path, backup=True)
    
    else:
        bt.logging.error(f"Unknown mode: {mode}")
        bt.logging.info("Valid modes: check, remove, remove-only, stats")
        return
    
    # Show final statistics
    bt.logging.info("\nFinal data state:")
    get_statistics(df)


if __name__ == "__main__":
    # Parse command line arguments
    mode = "check"  # Default mode
    batch_size = 100  # Default batch size
    
    if "--remove" in sys.argv:
        mode = "remove"
    elif "--remove-only" in sys.argv:
        mode = "remove-only"
    elif "--stats" in sys.argv:
        mode = "stats"
    
    if "--batch-size" in sys.argv:
        idx = sys.argv.index("--batch-size")
        if idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[idx + 1])
    
    bt.logging.info("""
╔══════════════════════════════════════════════════════════════╗
║         MOLECULE DEDUPLICATION SCRIPT (CSV VERSION)          ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python script.py                    # Check and mark duplicates (safe)
  python script.py --remove           # Check, mark, and remove duplicates
  python script.py --remove-only      # Only remove already marked duplicates
  python script.py --stats            # Show statistics only
  python script.py --batch-size 500   # Use custom batch size

Input:  data/data.csv
Output: data/data.csv (with 'available' column added)
Backup: data/data.csv.backup (created automatically)

""")
    
    asyncio.run(main(mode=mode, batch_size=batch_size))
