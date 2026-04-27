import sqlite3
import csv
import os

# ── Config ──────────────────────────────────────────────
SQLITE_FILE = "score_results_1_2769.sqlite"   # <-- change this
OUTPUT_CSV  = "data.csv"   # <-- change this (optional)
TABLE_NAME  = "scored_molecules"
COLUMNS     = ["molecule_name", "score"]
SORT_ORDER  = "DESC"   # "DESC" = highest score first | "ASC" = lowest first
# ────────────────────────────────────────────────────────

def sqlite_to_csv(sqlite_path: str, csv_path: str) -> None:
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()

        # Fetch only the two columns, sorted by score
        query = f"""
            SELECT {', '.join(COLUMNS)}
            FROM {TABLE_NAME}
            ORDER BY score {SORT_ORDER}
        """
        cursor.execute(query)
        rows = cursor.fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(COLUMNS)   # header
        writer.writerows(rows)     # data rows

    print(f"✅ Done! {len(rows)} rows saved to '{csv_path}' (sorted by score {SORT_ORDER})")

if __name__ == "__main__":
    sqlite_to_csv(SQLITE_FILE, OUTPUT_CSV)