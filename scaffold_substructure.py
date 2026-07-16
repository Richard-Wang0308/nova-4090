#!/usr/bin/env python3
"""
Scaffold substructure search against combinatorial_db/molecules.sqlite.

Extracts the query Murcko scaffold, converts it to SMARTS, and finds molecules
that *contain* that scaffold as a substructure (including larger/fused cores).

Different from exact Murcko equality:
  - same Murcko  -> molecule core == query core
  - this search  -> molecule contains query core (may be elaborated/fused)

Example:
  .venv/bin/python scaffold_substructure.py --smiles "Nc1cn[nH]c(=O)c1" --role 4
  .venv/bin/python scaffold_substructure.py --mol-id 138879 --out scaffold_substruct_hits.csv
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


def query_scaffold_pattern(smiles: str) -> Tuple[str, Chem.Mol, str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise SystemExit(f"Could not parse/cap query SMILES: {smiles}")
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
        raise SystemExit(f"Empty Murcko scaffold for: {smiles}")
    scaffold_smi = Chem.MolToSmiles(scaffold_mol)
    scaffold_smarts = Chem.MolToSmarts(scaffold_mol)
    pattern = Chem.MolFromSmarts(scaffold_smarts)
    if pattern is None:
        raise SystemExit(f"Could not build SMARTS pattern from scaffold: {scaffold_smi}")
    return scaffold_smi, pattern, scaffold_smarts


def resolve_query_smiles(db: Path, smiles: Optional[str], mol_id: Optional[int]) -> Tuple[int, str]:
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
        return row[0], row[1]
    return -1, smiles


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
        description="Murcko scaffold substructure search over molecules.sqlite"
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--mol-id", type=int, default=None)
    p.add_argument("--role", type=int, choices=(1, 2, 4), default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=None, help="Keep only top-N by Morgan Tanimoto")
    p.add_argument("--out", type=Path, default=Path("scaffold_substruct_hits.csv"))
    args = p.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    q_id, q_smi = resolve_query_smiles(args.db, args.smiles, args.mol_id)
    q_mol = mol_from_smiles(q_smi)
    if q_mol is None:
        print(f"Could not parse/cap query SMILES: {q_smi}", file=sys.stderr)
        return 1
    q_fp = MORGAN.GetFingerprint(q_mol)

    scaffold_smi, pattern, scaffold_smarts = query_scaffold_pattern(q_smi)
    print(f"Query mol_id={q_id} smiles={q_smi}")
    print(f"Query Murcko scaffold: {scaffold_smi}")
    print(f"Scaffold SMARTS: {scaffold_smarts}")

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
        try:
            matched = mol.HasSubstructMatch(pattern)
        except Exception:
            skipped += 1
            continue
        if matched:
            # also report the molecule's own full Murcko for context
            try:
                own_murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                own_murcko = ""
            morgan = DataStructs.TanimotoSimilarity(q_fp, MORGAN.GetFingerprint(mol))
            hits.append(
                {
                    "morgan_tanimoto": round(float(morgan), 6),
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "role_mask": role_mask,
                    "own_murcko": own_murcko,
                    "contains_query_scaffold": scaffold_smi,
                }
            )

    hits.sort(key=lambda h: h["morgan_tanimoto"], reverse=True)
    if args.top is not None:
        hits = hits[: args.top]

    print(f"Scanned={scanned} skipped={skipped} hits={len(hits)} (sorted by Morgan Tanimoto)")
    for h in hits[:30]:
        same = " [exact Murcko]" if h["own_murcko"] == scaffold_smi else ""
        print(f"  {h['morgan_tanimoto']:.3f}  mol_id={h['mol_id']}{same} {h['smiles']}")
        if h["own_murcko"] and h["own_murcko"] != scaffold_smi:
            print(f"       own Murcko: {h['own_murcko']}")
    if len(hits) > 30:
        print(f"  ... ({len(hits) - 30} more)")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "morgan_tanimoto",
                "mol_id",
                "smiles",
                "role_mask",
                "own_murcko",
                "contains_query_scaffold",
            ],
        )
        w.writeheader()
        w.writerows(hits)
    print(f"Wrote {len(hits)} hits -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
