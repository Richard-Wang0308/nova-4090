import sqlite3
import sys
from datetime import datetime

def merge_databases(db1_path, db2_path, output_path):
    """
    Merge two SQLite databases with scored_molecules table.
    
    Rules:
    - If rows are exactly same (molecule_name, score): keep one
    - If molecule_name is same but score differs: average the scores, use later scored_at
    - Otherwise: add all unique rows
    """
    
    # Connect to databases
    conn1 = sqlite3.connect(db1_path)
    conn2 = sqlite3.connect(db2_path)
    conn_out = sqlite3.connect(output_path)
    
    cursor1 = conn1.cursor()
    cursor2 = conn2.cursor()
    cursor_out = conn_out.cursor()
    
    # Create output table
    cursor_out.execute('''
        CREATE TABLE IF NOT EXISTS scored_molecules (
            molecule_name TEXT,
            score REAL,
            scored_at TEXT
        )
    ''')
    
    # Read all data from both databases
    cursor1.execute('SELECT molecule_name, score, scored_at FROM scored_molecules')
    data1 = cursor1.fetchall()
    
    cursor2.execute('SELECT molecule_name, score, scored_at FROM scored_molecules')
    data2 = cursor2.fetchall()
    
    print(f"Database 1: {len(data1)} rows")
    print(f"Database 2: {len(data2)} rows")
    
    # Create dictionaries to track molecules
    # Key: molecule_name, Value: list of (score, scored_at) tuples
    molecules = {}
    
    # Process first database
    for mol_name, score, scored_at in data1:
        if mol_name not in molecules:
            molecules[mol_name] = []
        molecules[mol_name].append((score, scored_at, 'db1'))
    
    # Process second database
    for mol_name, score, scored_at in data2:
        if mol_name not in molecules:
            molecules[mol_name] = []
        molecules[mol_name].append((score, scored_at, 'db2'))
    
    # Merge logic
    merged_data = []
    
    for mol_name, entries in molecules.items():
        # Remove exact duplicates (same score)
        unique_entries = {}
        for score, scored_at, source in entries:
            if score not in unique_entries:
                unique_entries[score] = (scored_at, source)
            else:
                # Keep the later timestamp for same score
                existing_time = unique_entries[score][0]
                if scored_at > existing_time:
                    unique_entries[score] = (scored_at, source)
        
        if len(unique_entries) == 1:
            # Only one unique score for this molecule
            score = list(unique_entries.keys())[0]
            scored_at = unique_entries[score][0]
            merged_data.append((mol_name, score, scored_at))
        else:
            # Multiple different scores - calculate average
            scores = list(unique_entries.keys())
            avg_score = sum(scores) / len(scores)
            
            # Get the latest timestamp
            latest_time = max(unique_entries[s][0] for s in scores)
            
            merged_data.append((mol_name, avg_score, latest_time))
    
    # Insert merged data
    cursor_out.executemany(
        'INSERT INTO scored_molecules (molecule_name, score, scored_at) VALUES (?, ?, ?)',
        merged_data
    )
    
    conn_out.commit()
    
    print(f"\nMerged result: {len(merged_data)} rows")
    print(f"Output saved to: {output_path}")
    
    # Show some statistics
    cursor_out.execute('SELECT COUNT(*) FROM scored_molecules')
    total = cursor_out.fetchone()[0]
    
    cursor_out.execute('SELECT AVG(score), MIN(score), MAX(score) FROM scored_molecules')
    avg, min_score, max_score = cursor_out.fetchone()
    
    print(f"\nStatistics:")
    print(f"  Total rows: {total}")
    print(f"  Average score: {avg:.6f}")
    print(f"  Min score: {min_score:.6f}")
    print(f"  Max score: {max_score:.6f}")
    
    # Close connections
    conn1.close()
    conn2.close()
    conn_out.close()
    
    print("\nMerge completed successfully!")


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python merge_dbs.py <db1_path> <db2_path> <output_path>")
    #     print("\nExample:")
    #     print("  python merge_dbs.py score_results1.sqlite score_results2.sqlite merged_results.sqlite")
    #     sys.exit(1)
    
    db1_path = "score_results1.sqlite"
    db2_path = "score_results2.sqlite"
    output_path = "score_results.sqlite"
    
    try:
        merge_databases(db1_path, db2_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)