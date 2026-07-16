#!/usr/bin/env python3
"""
Generic Murcko scaffold search against combinatorial_db/molecules.sqlite.

Uses MakeScaffoldGeneric so atom types are ignored (C/N/O/... -> carbon framework).
Best for grouping by ring topology / shape rather than exact chemistry
(e.g. benzene / pyridine / pyrimidine share one generic scaffold).

Example:
  .venv/bin/python scaffold_generic_murcko.py --smiles "c1ccccc1" --role 2
  .venv/bin/python scaffold_generic_murcko.py --mol-id 138879 --out generic_murcko_hits.csv
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "combinatorial_db" / "molecules.sqlite"
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def cap_attachments(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
        return Chem.RemoveHs(rw.GetMol())
    except Exception:
        return None


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return cap_attachments(mol)


def generic_scaffold_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic)
    except Exception:
        return None


def resolve_query(
    db: Path, smiles: Optional[str], mol_id: Optional[int]
) -> Tuple[int, str, str, object]:
    if (smiles is None) == (mol_id is None):
        raise SystemExit("Provide exactly one of --smiles or --mol-id")

    if mol_id is not None:
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT mol_id, smiles FROM molecules WHERE mol_id = ?", (mol_id,)
        ).fetchone()
        conn.close()
        if row is None:
            raise SystemExit(f"mol_id {mol_id} not found in {db}")
        mol_id, smiles = row

    mol = mol_from_smiles(smiles)
    if mol is None:
        raise SystemExit(f"Could not parse/cap query SMILES: {smiles}")
    scaffold = generic_scaffold_smiles(mol)
    if not scaffold:
        raise SystemExit(f"Could not extract generic scaffold from: {smiles}")
    return (mol_id if mol_id is not None else -1), smiles, scaffold, MORGAN.GetFingerprint(mol)


def iter_db(
    db: Path, role: Optional[int], limit: Optional[int]
) -> List[Tuple[int, str, int]]:
    conn = sqlite3.connect(str(db))
    if role is not None:
        sql = "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?"
        params: tuple = (role, role)
    else:
        sql = "SELECT mol_id, smiles, role_mask FROM molecules"
        params = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = params + (limit,)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generic (atom-type-agnostic) Murcko scaffold match over molecules.sqlite"
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--mol-id", type=int, default=None)
    p.add_argument("--role", type=int, choices=(1, 2, 4), default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None, help="Keep only top-N by Morgan Tanimoto")
    p.add_argument("--out", type=Path, default=Path("generic_murcko_hits.csv"))
    args = p.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    q_id, q_smi, q_scaffold, q_fp = resolve_query(args.db, args.smiles, args.mol_id)
    print(f"Query mol_id={q_id} smiles={q_smi}")
    print(f"Query generic Murcko scaffold: {q_scaffold}")

    hits = []
    scanned = skipped = 0
    for mol_id, smiles, role_mask in iter_db(args.db, args.role, args.limit):
        scanned += 1
        # Never report the query molecule as a hit.
        if q_id >= 0 and mol_id == q_id:
            continue
        if smiles == q_smi:
            continue
        mol = mol_from_smiles(smiles)
        if mol is None:
            skipped += 1
            continue
        scaffold = generic_scaffold_smiles(mol)
        if scaffold is None:
            skipped += 1
            continue
        if scaffold == q_scaffold:
            morgan = DataStructs.TanimotoSimilarity(q_fp, MORGAN.GetFingerprint(mol))
            hits.append(
                {
                    "morgan_tanimoto": round(float(morgan), 6),
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "role_mask": role_mask,
                    "generic_murcko": scaffold,
                }
            )

    hits.sort(key=lambda h: h["morgan_tanimoto"], reverse=True)
    if args.top is not None:
        hits = hits[: args.top]

    print(f"Scanned={scanned} skipped={skipped} hits={len(hits)} (sorted by Morgan Tanimoto)")
    for h in hits[:30]:
        print(f"  {h['morgan_tanimoto']:.3f}  mol_id={h['mol_id']} {h['smiles']}")
    if len(hits) > 30:
        print(f"  ... ({len(hits) - 30} more)")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["morgan_tanimoto", "mol_id", "smiles", "role_mask", "generic_murcko"],
        )
        w.writeheader()
        w.writerows(hits)
    print(f"Wrote {len(hits)} hits -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
