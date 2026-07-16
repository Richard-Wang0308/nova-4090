"""
remove_null_scores.py — Delete rows with NULL score from scored_molecules
table in score_results_2.sqlite.
"""

import sqlite3

DB_PATH = "score_results_2.sqlite"
TABLE_NAME = "scored_molecules"

def remove_null_scores(db_path: str, table_name: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Count how many rows have NULL score before deleting
    cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE score IS NULL")
    null_count = cur.fetchone()[0]
    print(f"Found {null_count} rows with NULL score.")

    if null_count > 0:
        cur.execute(f"DELETE FROM {table_name} WHERE score IS NULL")
        conn.commit()
        print(f"✅ Deleted {null_count} rows with NULL score.")
    else:
        print("No NULL score rows found. Nothing to delete.")

    # Optional: reclaim disk space after deletion
    cur.execute("VACUUM")
    conn.commit()

    conn.close()


if __name__ == "__main__":
    remove_null_scores(DB_PATH, TABLE_NAME)
