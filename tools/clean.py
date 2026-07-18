import sqlite3
import math
import glob
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def clean_inf_scores(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.create_function(
        "ISFINITE", 1,
        lambda x: 1 if (x is not None and math.isfinite(x)) else 0
    )
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM scored_molecules WHERE NOT ISFINITE(score)")
    bad = cur.fetchone()[0]
    print(f"{db_path}: found {bad} non-finite rows")

    if bad:
        cur.execute("DELETE FROM scored_molecules WHERE NOT ISFINITE(score)")
        conn.commit()
        print(f"{db_path}: deleted {bad} rows")

    conn.execute("VACUUM")
    conn.close()

# run against all rxn score DBs in the project root
for db in glob.glob(os.path.join(BASE_DIR, "score_results_*.sqlite")):
    clean_inf_scores(db)
