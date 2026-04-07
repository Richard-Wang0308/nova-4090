#!/usr/bin/env python3
"""
Script to remove rows from score_results.sqlite where molecule_name 
does not start with "rxn:5"
"""

import sqlite3
import sys
from pathlib import Path


def clean_database(db_path='score_results.sqlite', dry_run=False):
    """
    Remove rows where molecule_name does not start with 'rxn:5'
    
    Args:
        db_path: Path to the SQLite database file
        dry_run: If True, only show what would be deleted without actually deleting
    """
    # Check if database exists
    if not Path(db_path).exists():
        print(f"Error: Database file '{db_path}' not found!")
        sys.exit(1)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Count rows before deletion
        cursor.execute("SELECT COUNT(*) FROM scored_molecules")
        total_before = cursor.fetchone()[0]
        print(f"Total rows before: {total_before}")
        
        # Count rows that will be deleted
        cursor.execute("""
            SELECT COUNT(*) FROM scored_molecules 
            WHERE molecule_name NOT LIKE 'rxn:5%'
        """)
        to_delete = cursor.fetchone()[0]
        print(f"Rows to be deleted: {to_delete}")
        
        # Show sample of rows to be deleted
        if to_delete > 0:
            cursor.execute("""
                SELECT molecule_name, score, scored_at 
                FROM scored_molecules 
                WHERE molecule_name NOT LIKE 'rxn:5%'
                LIMIT 10
            """)
            samples = cursor.fetchall()
            print(f"\\nSample of rows to be deleted (up to 10):")
            for row in samples:
                print(f"  - {row[0]} (score: {row[1]}, date: {row[2]})")
        
        if to_delete == 0:
            print("\\n✓ No rows to delete. All molecule_names already start with 'rxn:5'")
            return
        
        # Perform deletion if not dry run
        if not dry_run:
            cursor.execute("""
                DELETE FROM scored_molecules 
                WHERE molecule_name NOT LIKE 'rxn:5%'
            """)
            conn.commit()
            
            # Count rows after deletion
            cursor.execute("SELECT COUNT(*) FROM scored_molecules")
            total_after = cursor.fetchone()[0]
            
            print(f"\\n✓ Deletion complete!")
            print(f"Total rows after: {total_after}")
            print(f"Rows deleted: {total_before - total_after}")
        else:
            print(f"\\n[DRY RUN] Would delete {to_delete} rows")
            print("Run without --dry-run flag to actually delete the rows")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove rows from score_results.sqlite where molecule_name does not start with "rxn:5"'
    )
    parser.add_argument(
        '--db', 
        default='score_results.sqlite',
        help='Path to the SQLite database file (default: score_results.sqlite)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    clean_database(args.db, args.dry_run)