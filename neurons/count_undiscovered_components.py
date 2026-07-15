"""
count_undiscovered_components.py — Count undiscovered A and B components
for a given reaction (default rxn_id=2), based on score_results_{rxn_id}.sqlite

"Discovered"   = at least one molecule containing that component id has
                 been scored and stored in the scored_molecules table.
"Undiscovered" = total possible ids for that component (from the full
                 combinatorial pool in combinatorial_db/molecules.sqlite)
                 MINUS the discovered ids.

Usage:
    python count_undiscovered_components.py --rxn_id 2
    python count_undiscovered_components.py --rxn_id 2 --output report.txt
"""
import os
import sys
import sqlite3
import argparse
from datetime import datetime
from typing import Set, Tuple

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

from config.config_loader import load_config
from molecules import MoleculeManager


def score_db_path(rxn_id: int) -> str:
    return os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")


def load_discovered_ids(db_path: str, rxn_id: int) -> Tuple[Set[int], Set[int]]:
    """
    Parse the scored_molecules table and extract the set of distinct
    A ids and B ids that appear in molecule_name, for the given rxn_id.

    molecule_name format: rxn:{rxn_id}:{A_id}:{B_id}
    """
    discovered_A: Set[int] = set()
    discovered_B: Set[int] = set()

    if not os.path.exists(db_path):
        print(f"⚠️  DB not found: {db_path}")
        return discovered_A, discovered_B

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT molecule_name FROM scored_molecules WHERE molecule_name LIKE ?",
        (f"rxn:{rxn_id}:%",),
    )
    rows = cur.fetchall()
    conn.close()

    for (name,) in rows:
        parts = name.split(":")
        # parts = ['rxn', '2', 'A_id', 'B_id']  for a 2-component reaction
        if len(parts) < 4:
            continue
        try:
            a_id = int(parts[2])
            b_id = int(parts[3])
        except ValueError:
            continue
        discovered_A.add(a_id)
        discovered_B.add(b_id)

    return discovered_A, discovered_B


def build_report(
    rxn_id: int,
    full_A_ids: Set[int],
    full_B_ids: Set[int],
    discovered_A: Set[int],
    discovered_B: Set[int],
) -> str:
    undiscovered_A = full_A_ids - discovered_A
    undiscovered_B = full_B_ids - discovered_B

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 60)
    lines.append(f" REACTION {rxn_id} — UNDISCOVERED COMPONENT REPORT")
    lines.append(f" Generated at: {timestamp}")
    lines.append("=" * 60)
    lines.append(f" Full pool sizes   →  A: {len(full_A_ids)}   B: {len(full_B_ids)}")
    lines.append(f" Discovered so far →  A: {len(discovered_A)}   B: {len(discovered_B)}")
    lines.append("-" * 60)
    lines.append(f" Component A : {len(undiscovered_A):>6} undiscovered / {len(full_A_ids)} total")
    lines.append(f" Component B : {len(undiscovered_B):>6} undiscovered / {len(full_B_ids)} total")
    lines.append("=" * 60)

    return "\n".join(lines), undiscovered_A, undiscovered_B


def main():
    parser = argparse.ArgumentParser(
        description="Count undiscovered A/B components for a given reaction"
    )
    parser.add_argument("--rxn_id", type=int, default=2, help="Reaction id (default: 2)")
    parser.add_argument(
        "--output", type=str, default="undiscovered_components_report.txt",
        help="Output file to write the report to "
             "(default: undiscovered_components_report.txt)",
    )
    parser.add_argument(
        "--list_ids", action="store_true",
        help="If set, also write the full list of undiscovered A/B ids to the file.",
    )
    args = parser.parse_args()
    rxn_id = args.rxn_id

    # ── Load the FULL combinatorial pool via MoleculeManager ─────────────
    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    full_A_ids: Set[int] = set(manager.moles_A_id)
    full_B_ids: Set[int] = set(manager.moles_B_id)

    # ── Load discovered ids from score_results_{rxn_id}.sqlite ───────────
    db_path = score_db_path(rxn_id)
    discovered_A, discovered_B = load_discovered_ids(db_path, rxn_id)

    # ── Build report text ────────────────────────────────────────────────
    report_text, undiscovered_A, undiscovered_B = build_report(
        rxn_id, full_A_ids, full_B_ids, discovered_A, discovered_B
    )

    # ── Print to console ──────────────────────────────────────────────────
    print(report_text)

    # ── Write to file ────────────────────────────────────────────────────
    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
        if args.list_ids:
            f.write("\n\n--- Undiscovered A ids ---\n")
            f.write(", ".join(str(x) for x in sorted(undiscovered_A)) + "\n")
            f.write("\n--- Undiscovered B ids ---\n")
            f.write(", ".join(str(x) for x in sorted(undiscovered_B)) + "\n")

    print(f"\n✅ Report written to: {output_path}")

    return {
        "A_total": len(full_A_ids),
        "A_discovered": len(discovered_A),
        "A_undiscovered": len(undiscovered_A),
        "B_total": len(full_B_ids),
        "B_discovered": len(discovered_B),
        "B_undiscovered": len(undiscovered_B),
        "output_path": output_path,
    }


if __name__ == "__main__":
    main()