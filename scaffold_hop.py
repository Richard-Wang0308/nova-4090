#!/usr/bin/env python3
"""
Find scaffold hops in combinatorial_db/molecules.sqlite using 2D pharmacophore
fingerprints (RDKit Pharm2D) plus Morgan structural dissimilarity.

A hit is treated as a scaffold hop when:
  - pharmacophore Tanimoto >= --min-pharm
  - Morgan Tanimoto        <= --max-morgan
  - Murcko scaffolds differ (unless --allow-same-scaffold)

Modes:
  query     Find hops for one query SMILES / mol_id against the DB
  pairwise  Find hop pairs within a (optionally role-filtered) DB subset

Examples:
  .venv/bin/python scaffold_hop.py query --smiles "Nc1cn[nH]c(=O)c1" --role 2 --top 20
  .venv/bin/python scaffold_hop.py query --mol-id 138879 --role 2
  .venv/bin/python scaffold_hop.py pairwise --role 2 --limit 3000 --top 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import ChemicalFeatures, rdFingerprintGenerator
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Scaffolds import MurckoScaffold

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "combinatorial_db" / "molecules.sqlite"

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass
class MolRecord:
    mol_id: int
    smiles: str
    role_mask: int
    capped_smiles: str
    murcko: str
    pharm_fp: object
    morgan_fp: object


def build_sig_factory(
    min_point_count: int = 2,
    max_point_count: int = 3,
) -> SigFactory:
    """
    Same BaseFeatures / binning idea as the original snippet, with two robustness
    tweaks needed for real DB molecules:
      - catch-all distance bin (8, 100)
      - trianglePruneBins=False (avoids IndexError on odd 3-point distance combos)
    """
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    sig_factory = SigFactory(
        feat_factory,
        minPointCount=min_point_count,
        maxPointCount=max_point_count,
        trianglePruneBins=False,
    )
    sig_factory.SetBins([
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 100),
    ])
    sig_factory.Init()
    return sig_factory


def cap_attachments(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Replace combinatorial attachment points (*) with H so fingerprints/scaffolds work."""
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


def prepare_mol(smiles: str, sig_factory: SigFactory) -> Optional[Tuple[str, str, object, object]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    capped = cap_attachments(mol)
    if capped is None or capped.GetNumHeavyAtoms() < 2:
        return None
    try:
        pharm_fp = Generate.Gen2DFingerprint(capped, sig_factory)
        if pharm_fp.GetNumOnBits() == 0:
            return None
        morgan_fp = MORGAN_GENERATOR.GetFingerprint(capped)
        murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=capped)
        return Chem.MolToSmiles(capped), murcko, pharm_fp, morgan_fp
    except Exception:
        return None


def load_molecules(
    db_path: Path,
    sig_factory: SigFactory,
    role: Optional[int] = None,
    limit: Optional[int] = None,
    mol_ids: Optional[Sequence[int]] = None,
) -> List[MolRecord]:
    conn = sqlite3.connect(str(db_path))
    try:
        if mol_ids is not None:
            placeholders = ",".join("?" * len(mol_ids))
            sql = f"SELECT mol_id, smiles, role_mask FROM molecules WHERE mol_id IN ({placeholders})"
            params: Tuple = tuple(mol_ids)
        elif role is not None:
            sql = "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?"
            params = (role, role)
        else:
            sql = "SELECT mol_id, smiles, role_mask FROM molecules"
            params = ()

        if limit is not None:
            sql += " LIMIT ?"
            params = params + (limit,)

        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    records: List[MolRecord] = []
    skipped = 0
    t0 = time.time()
    for mol_id, smiles, role_mask in rows:
        prepared = prepare_mol(smiles, sig_factory)
        if prepared is None:
            skipped += 1
            continue
        capped_smiles, murcko, pharm_fp, morgan_fp = prepared
        records.append(
            MolRecord(
                mol_id=mol_id,
                smiles=smiles,
                role_mask=role_mask,
                capped_smiles=capped_smiles,
                murcko=murcko,
                pharm_fp=pharm_fp,
                morgan_fp=morgan_fp,
            )
        )
        if len(records) % 2000 == 0 and len(records):
            rate = len(records) / max(time.time() - t0, 1e-6)
            print(
                f"  fingerprints: {len(records)}/{len(rows)} "
                f"({rate:.0f}/s, skipped={skipped})",
                file=sys.stderr,
            )

    print(
        f"Loaded {len(records)} usable molecules "
        f"(skipped {skipped}) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return records


def is_scaffold_hop(
    pharm_sim: float,
    morgan_sim: float,
    murcko_a: str,
    murcko_b: str,
    min_pharm: float,
    max_morgan: float,
    require_different_scaffold: bool,
) -> bool:
    if pharm_sim < min_pharm or morgan_sim > max_morgan:
        return False
    if require_different_scaffold and murcko_a == murcko_b:
        return False
    return True


def search_query(
    query: MolRecord,
    library: Sequence[MolRecord],
    min_pharm: float,
    max_morgan: float,
    require_different_scaffold: bool,
    top: int,
) -> List[dict]:
    pharm_sims = DataStructs.BulkTanimotoSimilarity(
        query.pharm_fp, [r.pharm_fp for r in library]
    )
    morgan_sims = DataStructs.BulkTanimotoSimilarity(
        query.morgan_fp, [r.morgan_fp for r in library]
    )

    hits: List[dict] = []
    for rec, pharm_sim, morgan_sim in zip(library, pharm_sims, morgan_sims):
        if rec.mol_id == query.mol_id:
            continue
        if not is_scaffold_hop(
            pharm_sim,
            morgan_sim,
            query.murcko,
            rec.murcko,
            min_pharm,
            max_morgan,
            require_different_scaffold,
        ):
            continue
        hits.append(
            {
                "query_mol_id": query.mol_id,
                "query_smiles": query.smiles,
                "query_murcko": query.murcko,
                "hit_mol_id": rec.mol_id,
                "hit_smiles": rec.smiles,
                "hit_murcko": rec.murcko,
                "hit_role_mask": rec.role_mask,
                "pharm_tanimoto": round(float(pharm_sim), 6),
                "morgan_tanimoto": round(float(morgan_sim), 6),
                "hop_score": round(float(pharm_sim) - float(morgan_sim), 6),
            }
        )

    hits.sort(key=lambda h: (h["hop_score"], h["pharm_tanimoto"]), reverse=True)
    return hits[:top]


def search_pairwise(
    library: Sequence[MolRecord],
    min_pharm: float,
    max_morgan: float,
    require_different_scaffold: bool,
    top: int,
) -> List[dict]:
    """All-pairs within library. Keep O(n^2) tractable with --limit."""
    n = len(library)
    hits: List[dict] = []
    t0 = time.time()
    for i in range(n):
        a = library[i]
        pharm_sims = DataStructs.BulkTanimotoSimilarity(
            a.pharm_fp, [library[j].pharm_fp for j in range(i + 1, n)]
        )
        morgan_sims = DataStructs.BulkTanimotoSimilarity(
            a.morgan_fp, [library[j].morgan_fp for j in range(i + 1, n)]
        )
        for offset, (pharm_sim, morgan_sim) in enumerate(zip(pharm_sims, morgan_sims)):
            b = library[i + 1 + offset]
            if not is_scaffold_hop(
                pharm_sim,
                morgan_sim,
                a.murcko,
                b.murcko,
                min_pharm,
                max_morgan,
                require_different_scaffold,
            ):
                continue
            hits.append(
                {
                    "mol_id_a": a.mol_id,
                    "smiles_a": a.smiles,
                    "murcko_a": a.murcko,
                    "mol_id_b": b.mol_id,
                    "smiles_b": b.smiles,
                    "murcko_b": b.murcko,
                    "pharm_tanimoto": round(float(pharm_sim), 6),
                    "morgan_tanimoto": round(float(morgan_sim), 6),
                    "hop_score": round(float(pharm_sim) - float(morgan_sim), 6),
                }
            )
        if (i + 1) % 200 == 0:
            print(
                f"  pairwise {i + 1}/{n}, kept={len(hits)}, "
                f"elapsed={time.time() - t0:.1f}s",
                file=sys.stderr,
            )

    hits.sort(key=lambda h: (h["hop_score"], h["pharm_tanimoto"]), reverse=True)
    return hits[:top]


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        print(f"No hits; wrote empty file to {path}", file=sys.stderr)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}", file=sys.stderr)


def print_hits(rows: Sequence[dict], max_rows: int = 20) -> None:
    if not rows:
        print("No scaffold hops found with current thresholds.")
        return
    keys = list(rows[0].keys())
    show = rows[:max_rows]
    # compact terminal view
    for i, row in enumerate(show, 1):
        if "hit_mol_id" in row:
            print(
                f"{i:3d}. pharm={row['pharm_tanimoto']:.3f} "
                f"morgan={row['morgan_tanimoto']:.3f} "
                f"hop={row['hop_score']:.3f} | "
                f"q={row['query_mol_id']} -> hit={row['hit_mol_id']} | "
                f"{row['hit_smiles']}"
            )
            print(f"     murcko: {row['query_murcko']}  =>  {row['hit_murcko']}")
        else:
            print(
                f"{i:3d}. pharm={row['pharm_tanimoto']:.3f} "
                f"morgan={row['morgan_tanimoto']:.3f} "
                f"hop={row['hop_score']:.3f} | "
                f"{row['mol_id_a']} <-> {row['mol_id_b']}"
            )
            print(f"     {row['smiles_a']}")
            print(f"     {row['smiles_b']}")
            print(f"     murcko: {row['murcko_a']}  =>  {row['murcko_b']}")
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more; see --out)")


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="molecules.sqlite path")
    p.add_argument(
        "--role",
        type=int,
        choices=(1, 2, 4),
        default=None,
        help="Filter library by role_mask bit (1=A, 2=B, 4=C)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max molecules to load from DB (pairwise strongly recommended)",
    )
    p.add_argument("--min-pharm", type=float, default=0.45, help="Min Pharm2D Tanimoto")
    p.add_argument("--max-morgan", type=float, default=0.35, help="Max Morgan Tanimoto")
    p.add_argument(
        "--allow-same-scaffold",
        action="store_true",
        help="Do not require different Murcko scaffolds",
    )
    p.add_argument("--top", type=int, default=50, help="Max hits to keep/report")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path")


def cmd_query(args: argparse.Namespace) -> int:
    if (args.smiles is None) == (args.mol_id is None):
        print("Provide exactly one of --smiles or --mol-id", file=sys.stderr)
        return 2

    sig_factory = build_sig_factory()

    if args.mol_id is not None:
        conn = sqlite3.connect(str(args.db))
        row = conn.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE mol_id = ?",
            (args.mol_id,),
        ).fetchone()
        conn.close()
        if row is None:
            print(f"mol_id {args.mol_id} not found in {args.db}", file=sys.stderr)
            return 1
        q_id, q_smi, q_role = row
    else:
        q_id, q_smi, q_role = -1, args.smiles, -1

    prepared = prepare_mol(q_smi, sig_factory)
    if prepared is None:
        print(f"Could not fingerprint query SMILES: {q_smi}", file=sys.stderr)
        return 1
    capped, murcko, pharm_fp, morgan_fp = prepared
    query = MolRecord(
        mol_id=q_id,
        smiles=q_smi,
        role_mask=q_role,
        capped_smiles=capped,
        murcko=murcko,
        pharm_fp=pharm_fp,
        morgan_fp=morgan_fp,
    )
    print(
        f"Query mol_id={query.mol_id} murcko={query.murcko} smiles={query.smiles}",
        file=sys.stderr,
    )

    library = load_molecules(
        args.db, sig_factory, role=args.role, limit=args.limit
    )
    hits = search_query(
        query,
        library,
        min_pharm=args.min_pharm,
        max_morgan=args.max_morgan,
        require_different_scaffold=not args.allow_same_scaffold,
        top=args.top,
    )
    print_hits(hits)
    if args.out:
        write_csv(args.out, hits)
    return 0


def cmd_pairwise(args: argparse.Namespace) -> int:
    if args.limit is None:
        print(
            "pairwise mode needs --limit (all-pairs on full DB is intractable).",
            file=sys.stderr,
        )
        return 2
    sig_factory = build_sig_factory()
    library = load_molecules(
        args.db, sig_factory, role=args.role, limit=args.limit
    )
    hits = search_pairwise(
        library,
        min_pharm=args.min_pharm,
        max_morgan=args.max_morgan,
        require_different_scaffold=not args.allow_same_scaffold,
        top=args.top,
    )
    print_hits(hits)
    if args.out:
        write_csv(args.out, hits)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scaffold-hop search over molecules.sqlite via Pharm2D similarity."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Search DB for hops relative to one query molecule")
    add_shared_args(q)
    q.add_argument("--smiles", type=str, default=None, help="Query SMILES")
    q.add_argument("--mol-id", type=int, default=None, help="Query mol_id from DB")
    q.set_defaults(func=cmd_query)

    p = sub.add_parser("pairwise", help="Find hop pairs inside a limited DB subset")
    add_shared_args(p)
    p.set_defaults(func=cmd_pairwise)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
