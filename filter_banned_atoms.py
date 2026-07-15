#!/usr/bin/env python3
"""
Filter score_results_2.sqlite by converting molecule_name -> SMILES
and dropping molecules that contain banned atom types.

Keeps only molecules that pass the banned-atom check, writing them to filtered.sqlite.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from rdkit import Chem
from combinatorial_db.reactions import get_smiles_from_reaction

from utils.molecules import contains_atom_type


TABLE_NAME = "scored_molecules"
DEFAULT_BANNED = ["Se", "Na", "Fe", "Zn", "S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter scored molecules that contain banned atoms."
    )
    parser.add_argument(
        "--input",
        default="score_results_2.sqlite",
        help="Path to source SQLite database.",
    )
    parser.add_argument(
        "--output",
        default="filtered.sqlite",
        help="Path to output filtered SQLite database.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (reads molecule_requirements.banned_atom_types).",
    )
    return parser.parse_args()


def load_banned_atoms(config_path: Path) -> List[str]:
    if not config_path.exists():
        print(f"Config not found at {config_path}, using defaults: {DEFAULT_BANNED}")
        return DEFAULT_BANNED

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    banned = (
        cfg.get("molecule_requirements", {}).get("banned_atom_types")
        or DEFAULT_BANNED
    )
    return list(banned)


def validate_molecule_banned_atoms(
    molecule_name: str, smiles: str, config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Same logic as aggressive_surrogate.validate_molecule_banned_atoms."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for banned atom check"
        banned = config["banned_atom_types"]
        if not banned:
            return True, ""
        if contains_atom_type(mol, banned):
            return False, f"Contains banned atom types: {banned}"
        return True, ""
    except Exception as e:
        return False, f"Banned atom check error: {str(e)}"


def ensure_output_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            molecule_name TEXT PRIMARY KEY,
            score REAL,
            scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            available BOOLEAN
        )
        """
    )


def filter_db(input_path: Path, output_path: Path, banned: List[str]) -> None:
    config = {"banned_atom_types": banned}
    print(f"Banned atom types: {banned}")
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    src = sqlite3.connect(input_path)
    src_cur = src.cursor()

    src_cur.execute(
        f"SELECT molecule_name, score, scored_at, available FROM {TABLE_NAME}"
    )
    rows = src_cur.fetchall()
    total = len(rows)
    print(f"Loaded {total} molecules")

    if output_path.exists():
        output_path.unlink()

    dst = sqlite3.connect(output_path)
    dst_cur = dst.cursor()
    ensure_output_table(dst_cur)

    kept: List[Tuple[Any, ...]] = []
    stats = {
        "kept": 0,
        "banned": 0,
        "no_smiles": 0,
        "parse_error": 0,
    }

    for i, (molecule_name, score, scored_at, available) in enumerate(rows, 1):
        if i % 500 == 0 or i == total:
            print(
                f"  [{i}/{total}] kept={stats['kept']} banned={stats['banned']} "
                f"no_smiles={stats['no_smiles']} parse_error={stats['parse_error']}"
            )

        try:
            smiles = get_smiles_from_reaction(molecule_name)
        except Exception as e:
            stats["no_smiles"] += 1
            print(f"  SMILES error for {molecule_name}: {e}")
            continue

        if not smiles:
            stats["no_smiles"] += 1
            continue

        ok, reason = validate_molecule_banned_atoms(molecule_name, smiles, config)
        if not ok:
            if "Cannot parse" in reason or "error" in reason.lower():
                stats["parse_error"] += 1
            else:
                stats["banned"] += 1
            continue

        kept.append((molecule_name, score, scored_at, available))
        stats["kept"] += 1

    if kept:
        dst_cur.executemany(
            f"""
            INSERT INTO {TABLE_NAME}
                (molecule_name, score, scored_at, available)
            VALUES (?, ?, ?, ?)
            """,
            kept,
        )
        dst.commit()

    src.close()
    dst.close()

    print("\nDone.")
    print(f"  kept:        {stats['kept']}")
    print(f"  banned:      {stats['banned']}")
    print(f"  no_smiles:   {stats['no_smiles']}")
    print(f"  parse_error: {stats['parse_error']}")
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    if not input_path.exists():
        print(f"Input DB not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    banned = load_banned_atoms(config_path)
    filter_db(input_path, output_path, banned)


if __name__ == "__main__":
    main()
