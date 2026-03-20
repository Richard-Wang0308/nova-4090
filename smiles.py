#!/usr/bin/env python3
"""
fix_db.py
---------
1. Removes junk columns: target_score, antitarget_scores, composite_score
2. Ensures smiles column exists
3. Backfills smiles for all rows using get_smiles_from_reaction()

Run once:  python fix_db.py
"""

import sqlite3
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")

from combinatorial_db.reactions import get_smiles_from_reaction

# ─────────────────────────────────────────────────────────────────────────────
def get_columns(cur, table="scored_molecules"):
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def drop_columns_sqlite(con, cur, table, cols_to_drop):
    """
    SQLite < 3.35 has no DROP COLUMN.
    We recreate the table keeping only the columns we want.
    """
    existing = get_columns(cur, table)
    keep = [c for c in existing if c not in cols_to_drop]

    print(f"  Keeping columns : {keep}")
    print(f"  Dropping columns: {[c for c in existing if c in cols_to_drop]}")

    keep_str = ", ".join(keep)

    cur.executescript(f"""
        BEGIN;
        CREATE TABLE scored_molecules_new AS
            SELECT {keep_str} FROM {table};
        DROP TABLE {table};
        ALTER TABLE scored_molecules_new RENAME TO {table};
        COMMIT;
    """)
    # restore NOT NULL / types via explicit recreation if needed
    print("  Table rebuilt successfully.")


def ensure_smiles_column(cur):
    cols = get_columns(cur)
    if "smiles" not in cols:
        cur.execute("ALTER TABLE scored_molecules ADD COLUMN smiles TEXT")
        print("  Added smiles column.")
    else:
        print("  smiles column already exists.")


def backfill_smiles(con, cur):
    cur.execute(
        "SELECT molecule_name FROM scored_molecules WHERE smiles IS NULL OR smiles = ''"
    )
    rows = cur.fetchall()
    total = len(rows)
    print(f"  Rows missing smiles: {total}")
    if total == 0:
        print("  Nothing to backfill.")
        return

    updated = 0
    failed  = 0

    for i, (name,) in enumerate(rows, 1):
        try:
            smiles = get_smiles_from_reaction(name)
            if smiles:
                cur.execute(
                    "UPDATE scored_molecules SET smiles = ? WHERE molecule_name = ?",
                    (smiles, name),
                )
                updated += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  WARN: failed for {name}: {e}")
            failed += 1

        if i % 200 == 0:
            con.commit()
            print(f"  Progress: {i}/{total} | updated={updated} | failed={failed}")

    con.commit()
    print(f"  Backfill done: total={total} | updated={updated} | failed={failed}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(SCORE_RESULTS_DB):
        print(f"ERROR: DB not found at {SCORE_RESULTS_DB}")
        sys.exit(1)

    con = sqlite3.connect(SCORE_RESULTS_DB)
    cur = con.cursor()

    print("\n[1/3] Removing junk columns...")
    junk = {"target_score", "antitarget_scores", "composite_score"}
    existing = set(get_columns(cur))
    to_drop = junk & existing
    if to_drop:
        drop_columns_sqlite(con, cur, "scored_molecules", to_drop)
    else:
        print("  No junk columns found — already clean.")

    print("\n[2/3] Ensuring smiles column exists...")
    ensure_smiles_column(cur)
    con.commit()

    print("\n[3/3] Backfilling smiles...")
    backfill_smiles(con, cur)

    con.close()
    print("\nDone. DB is clean.")


if __name__ == "__main__":
    main()
