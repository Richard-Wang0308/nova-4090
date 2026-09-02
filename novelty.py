#!/usr/bin/env python3
"""
novelty.py — one novelty guard, aligned with the validator, that does not go stale.

THE PROBLEM THIS SOLVES
-----------------------
A molecule is only worth submitting if it is far enough from every molecule in
the HuggingFace Submission-Archive: the validator requires
`max_tanimoto_to_historical < config['max_similarity_to_historical']` (0.6 here).

Two things were quietly breaking that:

1. THRESHOLD DRIFT. neurons/genetic.py, neurons/component_exhaust.py and
   neurons/late_stage_search.py hardcode 0.9, and miner/miner.py has no novelty
   check at all. Everything they score in the 0.7-0.9 band (or anywhere, for
   miner.py) can never be submitted, so the Boltz budget is spent on molecules
   that cannot earn anything.

2. ARCHIVE EROSION. The archive grows every epoch, but a long-running searcher
   loads it once at startup and never refreshes. A molecule that was novel when
   it was scored stops being novel once someone submits it — yours included.
   The score DB keeps the old high score, so the operator's "top-20 sum" keeps
   looking strong while the submittable set underneath it rots.

   Measured on this repo's score_results_2.sqlite: 78% of the top-300 scoring
   molecules are already in the archive at similarity >= 0.99, and 19 of the 20
   highest-scoring *available* molecules fail the 0.7 rule. The best genuinely
   submittable molecule sits at rank 242.

WHAT THIS GIVES YOU
-------------------
* NoveltyGuard: threshold read from config, fingerprints cached on disk, and a
  refresh() that re-pulls the archive so long runs stay honest.
* audit(): re-checks the whole score DB against today's archive, records each
  molecule's max_hist_sim, and marks the stale ones available=FALSE so
  submit.py and your own top-20 view finally agree with the validator.

USAGE
-----
    python3 novelty.py --rxn-id 2 --audit          # re-check DB, mark stale
    python3 novelty.py --rxn-id 2 --audit --dry-run
    python3 novelty.py --rxn-id 2 --report         # submittable top-k only
"""
from __future__ import annotations

import argparse
import os
import pickle
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

import score_store

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CACHE_DIR = os.path.join(BASE_DIR, "data", "novelty_cache")
DEFAULT_TTL_S = 3600  # re-pull the archive at most once an hour

NOVELTY_COLUMNS: Dict[str, str] = {
    "max_hist_sim": "REAL",          # distance to the nearest archived molecule
    "novelty_checked_at": "TIMESTAMP",
}


def config_threshold(config: Dict[str, Any]) -> float:
    """
    The validator's threshold, straight from config. Never hardcode this —
    that is exactly how the 0.9-vs-0.7 drift happened.
    """
    return float(config.get("max_similarity_to_historical", 0.6))


_INCHIKEY_CACHE: Dict[str, set] = {}


def archive_inchikeys(target: str) -> set:
    """InChIKeys of every archived submission."""
    if target not in _INCHIKEY_CACHE:
        try:
            from utils import get_historical_submissions
            df = get_historical_submissions(target, "molecules")
            col = next((c for c in ("InChI_Key", "inchikey", "InChIKey")
                        if df is not None and c in df.columns), None)
            _INCHIKEY_CACHE[target] = set(df[col]) if col else set()
        except Exception as e:
            print(f"[novelty] could not load archive InChIKeys: {e}")
            _INCHIKEY_CACHE[target] = set()
    return _INCHIKEY_CACHE[target]


def is_unique(smiles: str, target: str) -> bool:
    """
    The exact-duplicate half of the validator's gate: validity.py calls
    molecule_unique_for_protein_hf, which compares InChIKeys. A molecule must
    pass BOTH this and the Tanimoto rule in config_threshold().
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return False
    return Chem.MolToInchiKey(mol) not in archive_inchikeys(target)


class NoveltyGuard:
    """Archive-backed novelty test with a disk cache and explicit refresh."""

    def __init__(
        self,
        target: str,
        max_similarity: float,
        cache_dir: str = CACHE_DIR,
        ttl_s: int = DEFAULT_TTL_S,
    ):
        self.target = target
        self.max_similarity = float(max_similarity)
        self.cache_dir = cache_dir
        self.ttl_s = ttl_s
        self.fps: List[Any] = []
        self.loaded_at: float = 0.0
        self.n_bits = 2048
        os.makedirs(cache_dir, exist_ok=True)
        self.refresh(allow_cache=True)

    # -- archive loading ---------------------------------------------------

    @property
    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.target}_fps.pkl")

    # Bumped whenever the cache payload format changes, so a stale file on
    # disk is refetched rather than silently misread.
    CACHE_VERSION = 2

    def _load_from_cache(self) -> bool:
        path = self._cache_path
        if not os.path.exists(path):
            return False
        if time.time() - os.path.getmtime(path) > self.ttl_s:
            return False
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("version") != self.CACHE_VERSION:
                print("[novelty] cache format outdated; refetching")
                return False
            fps = blob["fps"]
            # Guard against a corrupt payload: every fingerprint must still be
            # the width the generator produces, or BulkTanimotoSimilarity will
            # raise "BitVects must be same length" deep in a later call.
            if fps and fps[0].GetNumBits() != self.n_bits:
                print(
                    f"[novelty] cached fingerprints are {fps[0].GetNumBits()} bits, "
                    f"expected {self.n_bits}; refetching"
                )
                return False
            self.fps = fps
            self.loaded_at = os.path.getmtime(path)
            print(f"[novelty] {len(self.fps)} archived fingerprints (cache)")
            return True
        except Exception as e:
            print(f"[novelty] cache unreadable ({e}); refetching")
            return False

    def _save_to_cache(self) -> None:
        # Pickle the fingerprint objects directly. ToBinary() paired with
        # CreateFromBinaryText() does NOT round-trip: a 2048-bit vector comes
        # back as 192 bits, which only surfaces as an error much later.
        # Atomic: several searchers run concurrently and share this cache file.
        # A partially-written pickle read by another process is unrecoverable,
        # so write to a private temp file and rename it into place.
        tmp = f"{self._cache_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump({"version": self.CACHE_VERSION, "fps": self.fps}, f)
            os.replace(tmp, self._cache_path)
        except Exception as e:
            print(f"[novelty] could not write cache: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def refresh(self, allow_cache: bool = False) -> int:
        """Re-pull the archive. Call this periodically inside long runs."""
        if allow_cache and self._load_from_cache():
            return len(self.fps)

        try:
            from utils import get_historical_submissions
            df = get_historical_submissions(self.target, "molecules")
        except Exception as e:
            print(f"[novelty] archive fetch failed: {e}")
            return len(self.fps)

        if df is None or df.empty:
            print(f"[novelty] archive empty for {self.target}")
            return len(self.fps)

        col = next((c for c in ("SMILES", "smiles", "Smiles", "canonical_smiles")
                    if c in df.columns), None)
        if col is None:
            print(f"[novelty] no SMILES column: {list(df.columns)}")
            return len(self.fps)

        mols = [Chem.MolFromSmiles(s) for s in df[col].dropna().astype(str)]
        mols = [m for m in mols if m is not None]
        if not mols:
            return len(self.fps)

        self.fps = list(MORGAN.GetFingerprints(mols, numThreads=8))
        self.loaded_at = time.time()
        self._save_to_cache()
        print(f"[novelty] {len(self.fps)} archived fingerprints (fresh)")
        return len(self.fps)

    def maybe_refresh(self) -> None:
        """Refresh if the in-memory archive is older than the TTL."""
        if time.time() - self.loaded_at > self.ttl_s:
            self.refresh(allow_cache=False)

    # -- queries -----------------------------------------------------------

    def max_similarity_to_archive(self, smiles: str) -> float:
        """1.0 means an exact archived molecule; unparseable is treated as 1.0."""
        if not self.fps:
            return 0.0
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return 1.0
        sims = DataStructs.BulkTanimotoSimilarity(MORGAN.GetFingerprint(mol), self.fps)
        return float(max(sims)) if sims else 0.0

    def is_novel(self, smiles: str) -> bool:
        return self.max_similarity_to_archive(smiles) < self.max_similarity

    def similarities(self, smiles_list: Sequence[str], progress_every: int = 0) -> np.ndarray:
        out = np.empty(len(smiles_list), dtype=float)
        for i, s in enumerate(smiles_list):
            out[i] = self.max_similarity_to_archive(s)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [novelty] {i+1}/{len(smiles_list)}", flush=True)
        return out

    def filter_frame(self, df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
        """Drop rows that the validator would reject."""
        if df.empty:
            return df
        sims = self.similarities(df[smiles_col].tolist())
        return df[sims < self.max_similarity].reset_index(drop=True)


# =============================================================================
# DB audit
# =============================================================================

def _ensure_novelty_columns(db_path: str) -> None:
    with score_store.connect(db_path) as conn:
        cur = conn.cursor()
        cols = {r[1] for r in cur.execute("PRAGMA table_info(scored_molecules)").fetchall()}
        for name, sqltype in NOVELTY_COLUMNS.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE scored_molecules ADD COLUMN {name} {sqltype}")
        conn.commit()


def audit(
    db_path: str,
    rxn_id: int,
    guard: NoveltyGuard,
    scan: int = 10000,
    mark_unavailable: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Re-check the highest-scoring molecules against today's archive.

    Records max_hist_sim for each and, unless dry_run, sets available=FALSE on
    the ones the validator would now reject. `available` previously only tracked
    molecules *you* had submitted; anything submitted by anyone else silently
    stayed at the top of your list.
    """
    _ensure_novelty_columns(db_path)

    with score_store.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT molecule_name, smiles, score FROM scored_molecules "
            "WHERE available=TRUE AND smiles IS NOT NULL AND smiles!='' "
            "AND molecule_name LIKE ? ORDER BY score DESC LIMIT ?",
            (f"rxn:{rxn_id}:%", scan),
        ).fetchall()

    if not rows:
        print("nothing to audit")
        return pd.DataFrame(columns=["name", "smiles", "score", "max_hist_sim"])

    print(f"auditing top {len(rows)} available molecules against "
          f"{len(guard.fps)} archived (threshold <{guard.max_similarity})")
    df = pd.DataFrame(rows, columns=["name", "smiles", "score"])
    df["max_hist_sim"] = guard.similarities(df["smiles"].tolist(), progress_every=1000)
    df["novel"] = df["max_hist_sim"] < guard.max_similarity

    stale = df[~df["novel"]]
    print(f"\n{len(stale)}/{len(df)} ({100*len(stale)/len(df):.1f}%) are no longer "
          f"submittable")
    print(f"  of which exact archive hits (sim>=0.99): "
          f"{int((df['max_hist_sim']>=0.99).sum())}")

    if not dry_run:
        with score_store.connect(db_path) as conn:
            conn.executemany(
                "UPDATE scored_molecules SET max_hist_sim=?, "
                "novelty_checked_at=CURRENT_TIMESTAMP WHERE molecule_name=?",
                [(float(s), n) for n, s in zip(df["name"], df["max_hist_sim"])],
            )
            if mark_unavailable and len(stale):
                conn.executemany(
                    "UPDATE scored_molecules SET available=FALSE WHERE molecule_name=?",
                    [(n,) for n in stale["name"]],
                )
            conn.commit()
        print(f"recorded max_hist_sim for {len(df)}"
              + (f"; marked {len(stale)} unavailable" if mark_unavailable else ""))
    else:
        print("(dry run — nothing written)")

    return df


def report(df: pd.DataFrame, k: int, threshold: float) -> None:
    if df.empty:
        return
    novel = df[df["max_hist_sim"] < threshold].reset_index(drop=True)
    naive = df.head(k)

    print("\n" + "=" * 74)
    print(f"WHAT YOU LOOK AT vs WHAT YOU CAN SUBMIT (k={k})")
    print("=" * 74)
    print(f"naive top-{k} sum (ignores novelty) : {naive['score'].sum():.5f}"
          f"   submittable: {int((naive['max_hist_sim']<threshold).sum())}/{k}")
    if len(novel) >= k:
        sel = novel.head(k)
        rank = int(df.index[df["name"] == sel.iloc[-1]["name"]][0]) + 1
        print(f"submittable top-{k} sum            : {sel['score'].sum():.5f}")
        print(f"gap                                : "
              f"{sel['score'].sum()-naive['score'].sum():+.5f}")
        print(f"the {k}th submittable molecule sits at rank {rank} of your list")
        print(f"\nsubmittable top-{k}:")
        for r in sel.itertuples(index=False):
            print(f"  {r.name:<28}{r.score:9.5f}   sim={r.max_hist_sim:.3f}")
    else:
        print(f"only {len(novel)} submittable in the scanned window — widen --scan "
              f"or mine for novelty")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rxn-id", type=int, required=True)
    p.add_argument("--scan", type=int, default=10000)
    p.add_argument("--audit", action="store_true", help="re-check DB and mark stale rows")
    p.add_argument("--report", action="store_true", help="print submittable top-k")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-mark", action="store_true", help="record sims, do not flip available")
    p.add_argument("--threshold", type=float, default=None,
                   help="override; default comes from config")
    p.add_argument("--ttl", type=int, default=DEFAULT_TTL_S)
    args = p.parse_args()

    from config.config_loader import load_config
    config = load_config()
    target = config["small_molecule_target"][0]
    thr = args.threshold if args.threshold is not None else config_threshold(config)
    k = int(config.get("num_molecules", 20))

    db_path = score_store.score_db_path(args.rxn_id)
    if not os.path.exists(db_path):
        raise SystemExit(f"score DB not found: {db_path}")

    guard = NoveltyGuard(target, thr, ttl_s=args.ttl)
    if not args.audit and not args.report:
        args.audit = True

    df = audit(db_path, args.rxn_id, guard, scan=args.scan,
               mark_unavailable=not args.no_mark, dry_run=args.dry_run)
    if args.report or args.audit:
        report(df, k, thr)


if __name__ == "__main__":
    main()
