#!/usr/bin/env python3
"""
nova_top20_miner_v2.py
======================

Unified SN68 NOVA small-molecule search miner for one fixed reaction per GPU/server.

Designed for Richard-Wang0308/nova-4090 repository layout.

Core design:
  - One persistent target-aware score database.
  - Exact fixed reaction (--rxn-id).
  - Candidate generation combines:
      * global component-prior sampling
      * elite crossover
      * synthon-neighbour local search
      * one-component anchored search
      * pair-interaction anchored search
  - Unified ensemble surrogate (RandomForest + ExtraTrees).
  - Tree-level uncertainty.
  - Acquisition is TOP-20 FRONTIER aware:
      * exploit candidates likely to beat current #20
      * uncertainty candidates that may hide extreme-tail winners
      * explicit exploration quota
  - All newly scored molecules immediately become parents/anchors/training data.
  - Target-aware SQLite prevents accidental score reuse across protein targets.
  - Final top 20 are revalidated against current HF/history/BRENK constraints
    and de-duplicated by InChIKey before export.
  - Writes a compatibility score_results_{rxn}.sqlite so existing submit.py
    can continue to work.

IMPORTANT:
  Run this script from the root of nova-4090, or place it in the repository root.

Example:
    python3 nova_top20_miner_v2.py --rxn-id 3

More aggressive 4090 run:
    python3 nova_top20_miner_v2.py \
        --rxn-id 3 \
        --boltz-budget 180 \
        --candidate-pool 25000 \
        --batch-size 10

This script intentionally optimizes the final 20-molecule portfolio, not a
single best molecule.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


# =============================================================================
# Paths / project imports
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MINER_DIR = BASE_DIR / "miner"
BOLTZ_DIR = BASE_DIR / "boltz"

for p in (str(BASE_DIR), str(MINER_DIR), str(BOLTZ_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DB_PATH = BASE_DIR / "combinatorial_db" / "molecules.sqlite"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "top20_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from config.config_loader import load_config
from molecules import MoleculeManager, MoleculeUtils
from tools import SynthonLibrary

try:
    from utils import (
        get_historical_submissions,
        molecule_unique_for_protein_hf,
    )
except Exception:
    get_historical_submissions = None
    molecule_unique_for_protein_hf = None

try:
    from utils.molecules import compute_maccs_entropy
except Exception:
    compute_maccs_entropy = None

from utils.molecules import get_brenk_matches

try:
    from boltz_wrapper import BoltzWrapper
except Exception:
    BoltzWrapper = None


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nova-top20-v2")


# =============================================================================
# Constants / fingerprints
# =============================================================================

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_mol_cache: Dict[str, Any] = {}
_fp_cache: Dict[str, np.ndarray] = {}
_fp_bv_cache: Dict[str, Any] = {}
_desc_cache: Dict[str, np.ndarray] = {}


def mol_from_smiles(smiles: str):
    if smiles not in _mol_cache:
        try:
            _mol_cache[smiles] = Chem.MolFromSmiles(smiles)
        except Exception:
            _mol_cache[smiles] = None
    return _mol_cache[smiles]


def morgan_array(smiles: str) -> Optional[np.ndarray]:
    cached = _fp_cache.get(smiles)
    if cached is not None:
        return cached
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    bv = MORGAN.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.float32)
    arr[list(bv.GetOnBits())] = 1.0
    _fp_cache[smiles] = arr
    return arr


def morgan_bv(smiles: str):
    cached = _fp_bv_cache.get(smiles)
    if cached is not None:
        return cached
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    bv = MORGAN.GetFingerprint(mol)
    _fp_bv_cache[smiles] = bv
    return bv


def descriptors(smiles: str) -> Optional[np.ndarray]:
    cached = _desc_cache.get(smiles)
    if cached is not None:
        return cached
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy = float(mol.GetNumHeavyAtoms())
    atoms = max(1.0, float(mol.GetNumAtoms()))
    aromatic = float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    vals = np.array(
        [
            Descriptors.MolWt(mol) / 1000.0,
            Descriptors.MolLogP(mol) / 10.0,
            Descriptors.TPSA(mol) / 250.0,
            float(Lipinski.NumHDonors(mol)) / 10.0,
            float(Lipinski.NumHAcceptors(mol)) / 20.0,
            float(Descriptors.NumRotatableBonds(mol)) / 15.0,
            heavy / 100.0,
            float(Lipinski.RingCount(mol)) / 10.0,
            aromatic / atoms,
            float(Lipinski.FractionCSP3(mol)),
        ],
        dtype=np.float32,
    )
    _desc_cache[smiles] = vals
    return vals


def brenk_ok(smiles: str) -> bool:
    """
    False when the validator's BRENK structural-alert filter would reject the
    molecule (one match invalidates an entire submission).
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False
    reasons = get_brenk_matches(mol)
    if reasons:
        log.debug("BRENK rejected %s: %s", smiles, "; ".join(reasons))
        return False
    return True


def inchikey(smiles: str) -> str:
    try:
        mol = mol_from_smiles(smiles)
        return Chem.MolToInchiKey(mol) if mol is not None else ""
    except Exception:
        return ""


def parse_components(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        parts = name.split(":")
        if len(parts) == 4:
            return int(parts[2]), int(parts[3]), None
        if len(parts) >= 5:
            return int(parts[2]), int(parts[3]), int(parts[4])
    except Exception:
        pass
    return None, None, None


def make_name(rxn: int, A: int, B: int, C: Optional[int]) -> str:
    if C is None:
        return f"rxn:{rxn}:{A}:{B}"
    return f"rxn:{rxn}:{A}:{B}:{C}"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified top-20-frontier miner for NOVA small molecules"
    )
    p.add_argument("--rxn-id", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--boltz-budget", type=int, default=150,
                   help="new molecules to Boltz-score per round")
    p.add_argument("--candidate-pool", type=int, default=18000,
                   help="approximate raw candidate target per round")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--min-train", type=int, default=500,
                   help="minimum scored examples before ensemble surrogate activates")
    p.add_argument("--train-cap", type=int, default=30000)
    p.add_argument("--parent-pool", type=int, default=250)
    p.add_argument("--elite-anchors", type=int, default=40)
    p.add_argument("--pair-anchors", type=int, default=50)
    p.add_argument("--neighbour-top-k", type=int, default=30)
    p.add_argument("--neighbour-min-sim", type=float, default=0.35)
    p.add_argument("--max-rounds", type=int, default=10**9)
    p.add_argument("--seed", type=int, default=68)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--disable-hf-filter", action="store_true")
    p.add_argument("--disable-history-filter", action="store_true")
    p.add_argument("--max-heavy-atoms", type=int, default=0,
                   help="0 means DO NOT impose the old local 40-heavy-atom cap")
    p.add_argument("--legacy-db", default="",
                   help="optional old score_results_N.sqlite to import ONCE; only use if it is for current target")
    return p.parse_args()


# =============================================================================
# Target-aware score DB
# =============================================================================

class ScoreStore:
    """
    Primary store = original score_results_{rxn}.sqlite.
    No separate top20_v2 score database is created.

    Existing rows are kept in the canonical scored_molecules table and extra
    columns are added in-place. Existing submit.py code remains compatible.

    Target safety:
      - metadata.active_target_key records the current target.
      - existing rows with no metadata are assumed to belong to the current target.
      - if the target changes later, old rows are archived INSIDE the same SQLite
        file and the live scored_molecules table is cleared for the new target.
    """
    def __init__(self, rxn_id: int, target_key: str, target_label: str):
        self.rxn_id = rxn_id
        self.target_key = target_key
        self.target_label = target_label
        self.path = BASE_DIR / f"score_results_{rxn_id}.sqlite"
        self.compat_path = self.path
        self._init()

    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_columns(self, conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scored_molecules (
                molecule_name TEXT PRIMARY KEY,
                score REAL NOT NULL,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available BOOLEAN DEFAULT TRUE,
                iteration INTEGER
            )
            """
        )
        cols = {r[1] for r in cur.execute("PRAGMA table_info(scored_molecules)").fetchall()}
        additions = {
            # miner/miner.py writes `iteration` and never `round`; keep both so the
            # two writers stay interoperable on a shared DB
            "iteration": "INTEGER",
            "target_key": "TEXT",
            "target_label": "TEXT",
            "rxn_id": "INTEGER",
            "smiles": "TEXT",
            "inchikey": "TEXT",
            "source": "TEXT",
            "round": "INTEGER",
        }
        for name, sqltype in additions.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE scored_molecules ADD COLUMN {name} {sqltype}")
                cols.add(name)

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.commit()
        return cols

    def _init(self):
        with self._connect() as conn:
            cols = self._ensure_columns(conn)

            row = conn.execute(
                "SELECT value FROM metadata WHERE key='active_target_key'"
            ).fetchone()
            active = row[0] if row else None
            count = conn.execute("SELECT COUNT(*) FROM scored_molecules").fetchone()[0]

            if active is None:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_key',?)",
                    (self.target_key,),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_label',?)",
                    (self.target_label,),
                )
                if count:
                    # legacy DBs may predate either `iteration` or `round`
                    round_clause = (
                        ", round=COALESCE(round,iteration)" if "iteration" in cols else ""
                    )
                    conn.execute(
                        f"""
                        UPDATE scored_molecules
                        SET target_key=COALESCE(target_key,?),
                            target_label=COALESCE(target_label,?),
                            rxn_id=COALESCE(rxn_id,?){round_clause}
                        """,
                        (self.target_key, self.target_label, self.rxn_id),
                    )
                conn.commit()

            elif active != self.target_key:
                stamp = int(time.time())
                archive = f"scored_molecules_archive_{stamp}"
                conn.execute(f"CREATE TABLE {archive} AS SELECT * FROM scored_molecules")
                conn.execute("DELETE FROM scored_molecules")
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_key',?)",
                    (self.target_key,),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_label',?)",
                    (self.target_label,),
                )
                conn.commit()
                log.warning(
                    "Target changed. Archived old rows to %s inside %s and reset live table.",
                    archive, self.path
                )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scored_score ON scored_molecules(score DESC)"
            )
            conn.commit()

    def names(self) -> Set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT molecule_name FROM scored_molecules").fetchall()
        return {r[0] for r in rows}

    def dataframe(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT molecule_name, smiles, inchikey, score,
                       COALESCE(source,'legacy') AS source,
                       COALESCE(round,iteration,0) AS round
                FROM scored_molecules
                ORDER BY score DESC
                """
            ).fetchall()

        df = pd.DataFrame(
            rows, columns=["name", "smiles", "inchikey", "score", "source", "round"]
        )
        if df.empty:
            return df

        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df[np.isfinite(df["score"])].dropna(subset=["score"]).reset_index(drop=True)

        missing = df["smiles"].isna() | (df["smiles"].astype(str) == "")
        if missing.any() and _GLOBAL_MANAGER is not None:
            names = df.loc[missing, "name"].tolist()
            raw = pd.DataFrame({"name": names})
            try:
                resolved = _GLOBAL_MANAGER.validate_molecules(
                    active_validation_config(_GLOBAL_MANAGER), raw
                )
                smap = dict(zip(resolved["name"], resolved["smiles"]))
                updates = []
                for idx in df.index[missing]:
                    name = df.at[idx, "name"]
                    s = smap.get(name)
                    if s:
                        ik = inchikey(s)
                        df.at[idx, "smiles"] = s
                        df.at[idx, "inchikey"] = ik
                        updates.append((s, ik, self.target_key, self.target_label,
                                        self.rxn_id, name))
                if updates:
                    with self._connect() as conn:
                        conn.executemany(
                            """
                            UPDATE scored_molecules
                            SET smiles=?, inchikey=?, target_key=?, target_label=?, rxn_id=?
                            WHERE molecule_name=?
                            """,
                            updates,
                        )
                        conn.commit()
            except Exception as e:
                log.warning("Could not backfill legacy SMILES: %s", e)

        df = df[df["smiles"].notna() & (df["smiles"].astype(str) != "")]
        return df.reset_index(drop=True)

    def write(self, records: Sequence[Dict[str, Any]], round_no: int):
        rows = []
        for r in records:
            score = r.get("boltz_score", r.get("score"))
            name = r.get("name")
            smiles = r.get("smiles")
            if not name or not smiles or score is None:
                continue
            try:
                score = float(score)
            except Exception:
                continue
            if not np.isfinite(score):
                continue
            rows.append(
                (
                    name, score, True, int(round_no),
                    self.target_key, self.target_label, self.rxn_id,
                    smiles, inchikey(smiles),
                    str(r.get("source", r.get("generation_method", "search"))),
                    int(round_no),
                )
            )

        if not rows:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO scored_molecules
                (molecule_name,score,available,iteration,target_key,target_label,
                 rxn_id,smiles,inchikey,source,round)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(molecule_name) DO UPDATE SET
                    score=excluded.score,
                    available=excluded.available,
                    iteration=excluded.iteration,
                    target_key=excluded.target_key,
                    target_label=excluded.target_label,
                    rxn_id=excluded.rxn_id,
                    smiles=excluded.smiles,
                    inchikey=excluded.inchikey,
                    source=excluded.source,
                    round=excluded.round,
                    scored_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
            conn.commit()

    def sync_compat_db(self):
        return

    def import_legacy_once(self, legacy_path: str, manager: MoleculeManager):
        if legacy_path:
            supplied = Path(legacy_path).resolve()
            if supplied != self.path.resolve():
                log.warning(
                    "--legacy-db=%s ignored because primary DB is now %s",
                    supplied, self.path
                )
            else:
                log.info("Using original DB directly: %s", self.path)


# =============================================================================
# Component / pair statistics
# =============================================================================

class ComponentStats:
    """
    Shrinkage estimates for A, B, C and pairwise interactions.
    The interaction priors are used as model features and generation weights.
    """
    def __init__(self):
        self.global_mean = 0.0
        self.global_std = 1.0
        self.single: Dict[str, Dict[int, Tuple[float, int, float]]] = {
            "A": {}, "B": {}, "C": {}
        }
        self.pairs: Dict[str, Dict[Tuple[int, int], Tuple[float, int, float]]] = {
            "AB": {}, "AC": {}, "BC": {}
        }

    @staticmethod
    def _aggregate(values: Dict[Any, List[float]], global_mean: float):
        out = {}
        for key, xs in values.items():
            arr = np.asarray(xs, dtype=float)
            n = len(arr)
            # upper-tail signal matters more than average for NOVA mining
            topq = float(np.quantile(arr, 0.80)) if n >= 2 else float(arr[0])
            mean = float(arr.mean())
            # Bayesian-ish shrinkage for rare components
            prior_n = 5.0
            shrunk = (n * (0.65 * mean + 0.35 * topq) + prior_n * global_mean) / (n + prior_n)
            out[key] = (shrunk, n, topq)
        return out

    def fit(self, df: pd.DataFrame):
        if df.empty:
            return
        scores = df["score"].astype(float).to_numpy()
        self.global_mean = float(np.mean(scores))
        self.global_std = float(np.std(scores) + 1e-6)

        vals = {"A": defaultdict(list), "B": defaultdict(list), "C": defaultdict(list)}
        pairs = {"AB": defaultdict(list), "AC": defaultdict(list), "BC": defaultdict(list)}

        for _, row in df.iterrows():
            A, B, C = parse_components(row["name"])
            s = float(row["score"])
            if A is not None:
                vals["A"][A].append(s)
            if B is not None:
                vals["B"][B].append(s)
            if C is not None:
                vals["C"][C].append(s)
            if A is not None and B is not None:
                pairs["AB"][(A, B)].append(s)
            if A is not None and C is not None:
                pairs["AC"][(A, C)].append(s)
            if B is not None and C is not None:
                pairs["BC"][(B, C)].append(s)

        for role in vals:
            self.single[role] = self._aggregate(vals[role], self.global_mean)
        for pair in pairs:
            self.pairs[pair] = self._aggregate(pairs[pair], self.global_mean)

    def _z(self, x: float) -> float:
        return (x - self.global_mean) / self.global_std

    def feature_vector(self, name: str) -> np.ndarray:
        A, B, C = parse_components(name)
        feats = []
        for role, key in (("A", A), ("B", B), ("C", C)):
            val, n, topq = self.single[role].get(
                key, (self.global_mean, 0, self.global_mean)
            )
            feats.extend([self._z(val), math.log1p(n) / 10.0, self._z(topq)])
        for pair, key in (
            ("AB", (A, B)),
            ("AC", (A, C)),
            ("BC", (B, C)),
        ):
            if None in key:
                val, n, topq = self.global_mean, 0, self.global_mean
            else:
                val, n, topq = self.pairs[pair].get(
                    key, (self.global_mean, 0, self.global_mean)
                )
            feats.extend([self._z(val), math.log1p(n) / 10.0, self._z(topq)])
        return np.asarray(feats, dtype=np.float32)

    def top_single(self, role: str, n: int) -> List[int]:
        items = sorted(
            self.single[role].items(),
            key=lambda kv: (kv[1][0] + 0.10 * kv[1][2], kv[1][1]),
            reverse=True,
        )
        return [k for k, _ in items[:n]]

    def top_pairs(self, pair: str, n: int) -> List[Tuple[int, int]]:
        items = sorted(
            self.pairs[pair].items(),
            key=lambda kv: (kv[1][0] + 0.10 * kv[1][2], kv[1][1]),
            reverse=True,
        )
        return [k for k, _ in items[:n]]

    def component_weights(self, ids: Sequence[int], role: str, epsilon: float = 0.25):
        if not ids:
            return None
        raw = []
        for mid in ids:
            val, n, topq = self.single[role].get(
                mid, (self.global_mean, 0, self.global_mean)
            )
            z = np.clip(self._z(0.65 * val + 0.35 * topq), -4, 4)
            raw.append(math.exp(0.7 * z))
        arr = np.asarray(raw, dtype=float)
        arr = arr / max(arr.sum(), 1e-12)
        uniform = np.ones_like(arr) / len(arr)
        arr = (1.0 - epsilon) * arr + epsilon * uniform
        return arr / arr.sum()


# =============================================================================
# Unified ensemble surrogate with uncertainty
# =============================================================================

class Top20Surrogate:
    def __init__(self, min_train: int, train_cap: int, seed: int):
        self.min_train = min_train
        self.train_cap = train_cap
        self.seed = seed
        self.stats = ComponentStats()

        self.rf = RandomForestRegressor(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
            bootstrap=True,
            max_samples=0.85,
        )
        self.et = ExtraTreesRegressor(
            n_estimators=160,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed + 1,
            bootstrap=True,
            max_samples=0.85,
        )
        self.trained = False
        self.frontier = -math.inf

    def _feature(self, name: str, smiles: str) -> Optional[np.ndarray]:
        fp = morgan_array(smiles)
        desc = descriptors(smiles)
        if fp is None or desc is None:
            return None
        comp = self.stats.feature_vector(name)
        return np.concatenate([fp, desc, comp]).astype(np.float32)

    def _training_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.train_cap:
            return df.copy()

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        n = self.train_cap
        n_elite = int(n * 0.40)
        n_recent = int(n * 0.25)
        n_random = n - n_elite - n_recent

        elite = df.head(n_elite)
        recent = df.sort_values("round", ascending=False).head(n_recent)
        remaining = df.drop(index=set(elite.index) | set(recent.index), errors="ignore")
        if len(remaining) > n_random:
            rand = remaining.sample(n=n_random, random_state=self.seed)
        else:
            rand = remaining

        out = pd.concat([elite, recent, rand], ignore_index=True)
        return out.drop_duplicates("name").head(n)

    def fit(self, df: pd.DataFrame):
        self.trained = False
        if df.empty or len(df) < self.min_train:
            return

        self.stats.fit(df)
        sorted_scores = df["score"].astype(float).sort_values(ascending=False)
        if len(sorted_scores) >= 20:
            self.frontier = float(sorted_scores.iloc[19])
        else:
            self.frontier = float(sorted_scores.iloc[-1])

        train = self._training_subset(df)
        X, y, weights = [], [], []

        p80 = float(np.quantile(train["score"], 0.80))
        p95 = float(np.quantile(train["score"], 0.95))

        for _, row in train.iterrows():
            feat = self._feature(row["name"], row["smiles"])
            if feat is None:
                continue
            s = float(row["score"])
            X.append(feat)
            y.append(s)

            # The model must learn the extreme tail / top-20 frontier accurately.
            w = 1.0
            if s >= self.frontier:
                w = 5.0
            elif s >= p95:
                w = 3.0
            elif s >= p80:
                w = 1.8
            weights.append(w)

        if len(X) < self.min_train:
            return

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        w = np.asarray(weights, dtype=float)

        t0 = time.time()
        self.rf.fit(X, y, sample_weight=w)
        self.et.fit(X, y, sample_weight=w)
        self.trained = True
        log.info(
            "Surrogate trained on %d | frontier(#20)=%.6f | %.1fs",
            len(X), self.frontier, time.time() - t0
        )

    @staticmethod
    def _tree_predictions(model, X: np.ndarray) -> np.ndarray:
        # shape: n_trees x n_samples
        return np.vstack([tree.predict(X) for tree in model.estimators_])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not self.trained:
            out = df.copy()
            out["mu"] = 0.0
            out["sigma"] = 1.0
            out["p_improve"] = 0.5
            out["ei"] = 0.0
            out["ucb"] = 0.0
            return out

        feats, keep_idx = [], []
        for idx, row in df.iterrows():
            feat = self._feature(row["name"], row["smiles"])
            if feat is not None:
                feats.append(feat)
                keep_idx.append(idx)

        if not feats:
            return df.head(0)

        X = np.asarray(feats, dtype=np.float32)
        rf_trees = self._tree_predictions(self.rf, X)
        et_trees = self._tree_predictions(self.et, X)
        all_trees = np.vstack([rf_trees, et_trees])

        mu = all_trees.mean(axis=0)
        sigma = all_trees.std(axis=0) + 1e-6

        threshold = self.frontier
        z = (mu - threshold) / sigma

        # Standard normal CDF without scipy.
        erf_vec = np.vectorize(math.erf)
        Phi = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        ei = (mu - threshold) * Phi + sigma * phi
        ucb = mu + 1.25 * sigma

        out = df.loc[keep_idx].copy()
        out["mu"] = mu
        out["sigma"] = sigma
        out["p_improve"] = Phi
        out["ei"] = ei
        out["ucb"] = ucb

        # Top-20 objective: probability + expected improvement dominate.
        # Normalize only for combining incomparable ranges.
        def norm(v):
            v = np.asarray(v, dtype=float)
            lo, hi = np.quantile(v, 0.05), np.quantile(v, 0.95)
            if hi <= lo + 1e-12:
                return np.zeros_like(v)
            return np.clip((v - lo) / (hi - lo), 0, 1)

        out["acq"] = (
            0.45 * norm(out["p_improve"])
            + 0.35 * norm(out["ei"])
            + 0.20 * norm(out["ucb"])
        )
        return out


# =============================================================================
# Historical validity
# =============================================================================

class HistoricalGuard:
    """Fast, fully cached historical/archive guard."""
    def __init__(self, target: str, max_similarity: float, disable_hf: bool, disable_history: bool):
        self.target = target
        self.max_similarity = max_similarity
        self.disable_hf = disable_hf
        self.disable_history = disable_history
        self.hist_fps: List[Any] = []
        self.hist_smiles: Set[str] = set()
        self.hist_inchikeys: Set[str] = set()
        self._load_once()

    def _load_once(self):
        if get_historical_submissions is None:
            log.warning("get_historical_submissions unavailable; archive cache disabled")
            return
        try:
            t0=time.time()
            df=get_historical_submissions(self.target, "molecules")
            if df is None or df.empty:
                log.info("HistoricalGuard: target archive is empty")
                return
            smiles_col=next((c for c in ("SMILES","smiles","Smiles","canonical_smiles") if c in df.columns), None)
            if smiles_col is None:
                log.warning("HistoricalGuard: no SMILES column in archive: %s", list(df.columns))
                return
            for s in df[smiles_col].dropna().astype(str).unique():
                mol=mol_from_smiles(s)
                if mol is None: continue
                try: canon=Chem.MolToSmiles(mol, canonical=True)
                except Exception: canon=s
                try: ik=Chem.MolToInchiKey(mol)
                except Exception: ik=""
                self.hist_smiles.add(canon)
                if ik: self.hist_inchikeys.add(ik)
                if not self.disable_history:
                    self.hist_fps.append(MORGAN.GetFingerprint(mol))
            log.info("HistoricalGuard cache ready: %d smiles | %d inchikeys | %d fps | %.2fs", len(self.hist_smiles), len(self.hist_inchikeys), len(self.hist_fps), time.time()-t0)
        except Exception as e:
            log.warning("Could not load historical submissions: %s", e)

    def exact_archive_ok(self, smiles: str) -> bool:
        if self.disable_hf: return True
        mol=mol_from_smiles(smiles)
        if mol is None: return False
        try:
            ik=Chem.MolToInchiKey(mol)
            if ik and ik in self.hist_inchikeys: return False
        except Exception: pass
        try:
            canon=Chem.MolToSmiles(mol, canonical=True)
            if canon in self.hist_smiles: return False
        except Exception: pass
        return True

    def hf_ok(self, smiles: str) -> bool:
        return self.exact_archive_ok(smiles)

    def history_ok(self, smiles: str) -> bool:
        if self.disable_history or not self.hist_fps: return True
        fp=morgan_bv(smiles)
        if fp is None: return False
        try:
            sims=DataStructs.BulkTanimotoSimilarity(fp, self.hist_fps)
            return (max(sims) if sims else 0.0) < self.max_similarity
        except Exception:
            return False

    def exact_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        keep=[idx for idx,row in df.iterrows() if self.exact_archive_ok(row["smiles"])]
        return df.loc[keep].reset_index(drop=True) if keep else df.head(0)

    def strict_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        t0=time.time(); rows=[]
        for _,row in df.iterrows():
            s=row["smiles"]
            if not self.exact_archive_ok(s): continue
            if not self.history_ok(s): continue
            rows.append(row)
        out=pd.DataFrame(rows).reset_index(drop=True) if rows else df.head(0)
        log.info("Strict history filter: %d -> %d candidates | %.2fs", len(df), len(out), time.time()-t0)
        return out

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strict_filter(df)


# =============================================================================
# Candidate generation
# =============================================================================

class CandidateGenerator:
    def __init__(
        self,
        rxn_id: int,
        manager: MoleculeManager,
        surrogate: Top20Surrogate,
        args,
    ):
        self.rxn_id = rxn_id
        self.manager = manager
        self.sub = manager.for_rxn(rxn_id)
        self.surrogate = surrogate
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.py_rng = random.Random(args.seed)
        self.synthon = SynthonLibrary(self.sub)

        self._ids = {
            "A": np.asarray(self.sub.moles_A_id, dtype=np.int64),
            "B": np.asarray(self.sub.moles_B_id, dtype=np.int64),
            "C": np.asarray(self.sub.moles_C_id, dtype=np.int64)
                 if self.sub.is_three_component else np.asarray([], dtype=np.int64),
        }
        self._prob_cache: Dict[Tuple[str, float], np.ndarray] = {}

    def refresh_sampling_cache(self):
        self._prob_cache.clear()

    def _role_probs(self, role: str, epsilon: float) -> Optional[np.ndarray]:
        key = (role, round(float(epsilon), 4))
        if key in self._prob_cache:
            return self._prob_cache[key]

        ids = self._ids[role]
        if len(ids) == 0:
            return None

        stat_map = self.surrogate.stats.single.get(role, {})
        if not stat_map:
            return None

        gm = self.surrogate.stats.global_mean
        gs = max(self.surrogate.stats.global_std, 1e-6)

        raw = np.empty(len(ids), dtype=np.float64)
        for i, mid in enumerate(ids):
            val, cnt, topq = stat_map.get(int(mid), (gm, 0, gm))
            z = np.clip(((0.65 * val + 0.35 * topq) - gm) / gs, -4.0, 4.0)
            raw[i] = math.exp(0.7 * z)

        total = raw.sum()
        if not np.isfinite(total) or total <= 0:
            return None

        p = raw / total
        uniform = np.full(len(ids), 1.0 / len(ids), dtype=np.float64)
        p = (1.0 - epsilon) * p + epsilon * uniform
        p /= p.sum()
        self._prob_cache[key] = p
        return p

    def _weighted_pick(self, ids: Sequence[int], role: str, n: int,
                       epsilon: float = 0.30) -> np.ndarray:
        arr = self._ids[role]
        if len(arr) == 0:
            return np.asarray([], dtype=np.int64)
        p = self._role_probs(role, epsilon)
        return self.rng.choice(arr, size=n, replace=True, p=p)

    def _bulk_components(self, n: int, epsilon: float = 0.35):
        A = self._weighted_pick(self._ids["A"], "A", n, epsilon)
        B = self._weighted_pick(self._ids["B"], "B", n, epsilon)
        C = (
            self._weighted_pick(self._ids["C"], "C", n, epsilon)
            if self.sub.is_three_component else None
        )
        return A, B, C

    def global_candidates(self, n: int) -> List[str]:
        A = self._weighted_pick(self.sub.moles_A_id, "A", n, epsilon=0.35)
        B = self._weighted_pick(self.sub.moles_B_id, "B", n, epsilon=0.35)
        if self.sub.is_three_component:
            C = self._weighted_pick(self.sub.moles_C_id, "C", n, epsilon=0.35)
            return [make_name(self.rxn_id, int(a), int(b), int(c))
                    for a, b, c in zip(A, B, C)]
        return [make_name(self.rxn_id, int(a), int(b), None) for a, b in zip(A, B)]

    def crossover(self, scored: pd.DataFrame, n: int) -> List[str]:
        if scored.empty or n <= 0:
            return []
        parents = scored.sort_values("score", ascending=False).head(self.args.parent_pool)
        names = parents["name"].tolist()
        if not names:
            return []

        mutA, mutB, mutC = self._bulk_components(n, epsilon=0.45)

        out = []
        for i in range(n):
            p1 = names[self.py_rng.randrange(len(names))]
            p2 = names[self.py_rng.randrange(len(names))]
            A1, B1, C1 = parse_components(p1)
            A2, B2, C2 = parse_components(p2)
            if A1 is None or A2 is None:
                continue

            A = A1 if self.py_rng.random() < 0.5 else A2
            B = B1 if self.py_rng.random() < 0.5 else B2
            C = (C1 if self.py_rng.random() < 0.5 else C2) if self.sub.is_three_component else None

            if self.py_rng.random() < 0.30:
                A = int(mutA[i])
            if self.py_rng.random() < 0.30:
                B = int(mutB[i])
            if self.sub.is_three_component and self.py_rng.random() < 0.30:
                C = int(mutC[i])

            out.append(make_name(self.rxn_id, int(A), int(B),
                                 int(C) if C is not None else None))
        return out

    def local_neighbour(self, scored: pd.DataFrame, n: int) -> List[str]:
        if scored.empty:
            return []
        seeds = scored.sort_values("score", ascending=False).head(
            min(self.args.elite_anchors, len(scored))
        )["name"].tolist()
        if not seeds:
            return []

        per_seed = max(2, int(math.ceil(n / len(seeds))))
        out = self.synthon.generate_similar_molecules(
            seeds,
            n_per_base=min(self.args.neighbour_top_k, per_seed),
            min_similarity=self.args.neighbour_min_sim,
        )
        if len(out) > n:
            self.py_rng.shuffle(out)
            out = out[:n]
        return out

    def single_anchor(self, n: int) -> List[str]:
        if n <= 0:
            return []
        stats = self.surrogate.stats
        top_A = stats.top_single("A", self.args.elite_anchors)
        top_B = stats.top_single("B", self.args.elite_anchors)
        top_C = stats.top_single("C", self.args.elite_anchors) if self.sub.is_three_component else []

        anchors = [("A", x) for x in top_A] + [("B", x) for x in top_B]
        if self.sub.is_three_component:
            anchors += [("C", x) for x in top_C]
        if not anchors:
            return []

        A_arr, B_arr, C_arr = self._bulk_components(n, epsilon=0.35)

        out = []
        for i in range(n):
            role, fixed = anchors[self.py_rng.randrange(len(anchors))]
            A = int(A_arr[i]); B = int(B_arr[i])
            C = int(C_arr[i]) if C_arr is not None else None
            if role == "A":
                A = int(fixed)
            elif role == "B":
                B = int(fixed)
            elif role == "C":
                C = int(fixed)
            out.append(make_name(self.rxn_id, A, B, C))
        return out

    def pair_anchor(self, n: int) -> List[str]:
        if n <= 0:
            return []
        stats = self.surrogate.stats
        pairs: List[Tuple[str, Tuple[int, int]]] = []
        pairs += [("AB", x) for x in stats.top_pairs("AB", self.args.pair_anchors)]
        if self.sub.is_three_component:
            pairs += [("AC", x) for x in stats.top_pairs("AC", self.args.pair_anchors)]
            pairs += [("BC", x) for x in stats.top_pairs("BC", self.args.pair_anchors)]
        if not pairs:
            return []

        A_arr, B_arr, C_arr = self._bulk_components(n, epsilon=0.35)

        out = []
        for i in range(n):
            kind, pair = pairs[self.py_rng.randrange(len(pairs))]
            A = int(A_arr[i]); B = int(B_arr[i])
            C = int(C_arr[i]) if C_arr is not None else None

            if kind == "AB":
                A, B = int(pair[0]), int(pair[1])
            elif kind == "AC":
                A, C = int(pair[0]), int(pair[1])
            elif kind == "BC":
                B, C = int(pair[0]), int(pair[1])

            out.append(make_name(self.rxn_id, A, B, C))
        return out

    def generate(self, scored: pd.DataFrame, seen: Set[str], n_total: int) -> pd.DataFrame:
        t_all = time.time()
        self.refresh_sampling_cache()
        # Distribution adapts once we actually have learned signal.
        if scored.empty or len(scored) < 200:
            proportions = {
                "global": 0.75,
                "cross": 0.15,
                "local": 0.10,
                "single": 0.0,
                "pair": 0.0,
            }
        else:
            proportions = {
                "global": 0.20,
                "cross": 0.20,
                "local": 0.20,
                "single": 0.18,
                "pair": 0.22,
            }

        names: List[str] = []

        t = time.time()
        g = self.global_candidates(int(n_total * proportions["global"]))
        log.info("generator/global: %d candidates | %.2fs", len(g), time.time() - t)
        names += g

        t = time.time()
        c = self.crossover(scored, int(n_total * proportions["cross"]))
        log.info("generator/crossover: %d candidates | %.2fs", len(c), time.time() - t)
        names += c

        t = time.time()
        l = self.local_neighbour(scored, int(n_total * proportions["local"]))
        log.info("generator/local-neighbour: %d candidates | %.2fs", len(l), time.time() - t)
        names += l

        t = time.time()
        sa = self.single_anchor(int(n_total * proportions["single"]))
        log.info("generator/single-anchor: %d candidates | %.2fs", len(sa), time.time() - t)
        names += sa

        t = time.time()
        pa = self.pair_anchor(int(n_total * proportions["pair"]))
        log.info("generator/pair-anchor: %d candidates | %.2fs", len(pa), time.time() - t)
        names += pa

        # Preserve order, remove already Boltz-scored.
        unique = []
        local_seen = set()
        for name in names:
            if name in seen or name in local_seen:
                continue
            local_seen.add(name)
            unique.append(name)

        if not unique:
            return pd.DataFrame(columns=["name", "smiles"])

        raw = pd.DataFrame({"name": unique})

        cfg = active_validation_config(self.manager)
        valid = self.manager.validate_molecules(cfg, raw)
        if valid.empty:
            return valid

        n_pre_brenk = len(valid)
        valid = valid[valid["smiles"].map(brenk_ok)]
        if len(valid) != n_pre_brenk:
            log.info("generator/brenk: dropped %d/%d candidates",
                     n_pre_brenk - len(valid), n_pre_brenk)
        if valid.empty:
            return valid

        valid["inchikey"] = valid["smiles"].map(inchikey)
        valid = valid[valid["inchikey"] != ""]
        valid = valid.drop_duplicates("inchikey", keep="first").reset_index(drop=True)
        log.info("generator/total: %d locally-valid unique | %.2fs",
                 len(valid), time.time() - t_all)
        return valid


# =============================================================================
# Validation config
# =============================================================================

_GLOBAL_CONFIG: Optional[Dict[str, Any]] = None
_GLOBAL_ARGS = None
_GLOBAL_MANAGER = None


def active_validation_config(manager: MoleculeManager) -> Dict[str, Any]:
    cfg = dict(_GLOBAL_CONFIG)
    cfg["allowed_reaction"] = f"rxn:{manager.rxn_id}"

    # Critical difference from old local miner:
    # NOVA's current validator config has no max-heavy-atom=40 rule.
    if _GLOBAL_ARGS.max_heavy_atoms and _GLOBAL_ARGS.max_heavy_atoms > 0:
        cfg["max_heavy_atoms"] = int(_GLOBAL_ARGS.max_heavy_atoms)
    else:
        cfg["max_heavy_atoms"] = 10**9
    return cfg


# =============================================================================
# Boltz scoring
# =============================================================================

async def boltz_score(
    boltz: BoltzWrapper,
    config: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    if not molecules:
        return []

    targets = config["small_molecule_target"]
    out: List[Dict[str, Any]] = []

    subnet_config = {
        "small_molecule_target": targets,
        "small_molecule_target_clip_interval": config["small_molecule_target_clip_interval"],
        "boltz_mode": config.get("boltz_mode", "max"),
        "boltz_metric": config.get(
            "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"],
        ),
        "combination_strategy": config.get(
            "combination_strategy", "heavy_atom_normalization"
        ),
    }

    for start in range(0, len(molecules), batch_size):
        batch = molecules[start:start + batch_size]
        valid_molecules_by_uid = {
            0: {
                "smiles": [x["smiles"] for x in batch],
                "names": [x["name"] for x in batch],
            }
        }
        score_dict = {
            0: {
                "target_scores": [[]],
                "antitarget_scores": [[]],
                "entropy": None,
                "entropy_boltz": None,
                "block_submitted": None,
                "push_time": "",
            }
        }

        def run():
            boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)

        t0 = time.time()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run)

        # score_dict[0]["molecule_scores"] follows the input SMILES order.
        scores_by_target = score_dict.get(0, {}).get("molecule_scores", [])
        scores = scores_by_target[0] if scores_by_target else []

        # Defensive fallback to wrapper map.
        if len(scores) != len(batch):
            target = targets[0]
            fmap = getattr(boltz, "final_boltz_scores", {}).get(0, {}).get(target, {})
            scores = [fmap.get(x["smiles"], -math.inf) for x in batch]

        batch_scored = []

        for rec, score in zip(batch, scores):
            try:
                score = float(score)
            except Exception:
                continue
            if not np.isfinite(score):
                continue

            r = dict(rec)
            r["boltz_score"] = score
            out.append(r)
            batch_scored.append((rec["name"], score))

        elapsed = time.time() - t0

        finite_scores = [score for _, score in batch_scored]

        log.info(
            "Boltz batch %d-%d: %d/%d finite | %.1fs",
            start + 1,
            min(start + len(batch), len(molecules)),
            len(batch_scored),
            len(batch),
            elapsed,
        )

        # Print every molecule score so live mining progress is visible.
        # Preserve original batch order for easy comparison with Boltz execution.
        for offset, (name, score) in enumerate(batch_scored, start=start + 1):
            log.info(
                "  BOLTZ [%d/%d] score=%.6f | %s",
                offset,
                len(molecules),
                score,
                name,
            )

        if finite_scores:
            log.info(
                "  BATCH STATS | best=%.6f | mean=%.6f | worst=%.6f",
                max(finite_scores),
                float(np.mean(finite_scores)),
                min(finite_scores),
            )

    return out


# =============================================================================
# Top-20 portfolio / acquisition
# =============================================================================

def choose_boltz_batch(
    ranked: pd.DataFrame,
    budget: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Budget allocation:
      60%: top frontier acquisition (P(improve)/EI/UCB)
      25%: uncertainty among plausibly competitive candidates
      15%: structural exploration

    This prevents a point-estimate surrogate from killing the extreme tail.
    """
    if ranked.empty:
        return ranked
    if len(ranked) <= budget:
        return ranked.copy()

    n_exploit = int(round(budget * 0.60))
    n_uncertain = int(round(budget * 0.25))
    n_explore = budget - n_exploit - n_uncertain

    picked: List[int] = []
    used: Set[int] = set()

    exploit = ranked.sort_values(
        ["acq", "p_improve", "ei", "mu"],
        ascending=False,
    )
    for idx in exploit.index:
        if len(picked) >= n_exploit:
            break
        picked.append(idx)
        used.add(idx)

    # Uncertainty should still be near a plausible frontier, not pure garbage.
    remaining = ranked.drop(index=list(used), errors="ignore")
    if not remaining.empty:
        mu_floor = float(remaining["mu"].quantile(0.45))
        uncertain = remaining[remaining["mu"] >= mu_floor].sort_values(
            ["sigma", "ucb"], ascending=False
        )
        for idx in uncertain.index:
            if len(picked) >= n_exploit + n_uncertain:
                break
            picked.append(idx)
            used.add(idx)

    # Explicit exploration: high novelty relative to selected molecules,
    # but novelty is NOT used as a validator reward.
    remaining = ranked.drop(index=list(used), errors="ignore")
    if not remaining.empty and n_explore > 0:
        selected_fps = [
            morgan_bv(ranked.loc[i, "smiles"]) for i in picked
            if morgan_bv(ranked.loc[i, "smiles"]) is not None
        ]
        candidates = list(remaining.index)
        rng.shuffle(candidates)

        explore_scores = []
        sample_for_div = candidates[: min(len(candidates), max(2000, n_explore * 20))]
        for idx in sample_for_div:
            fp = morgan_bv(ranked.loc[idx, "smiles"])
            if fp is None:
                continue
            maxsim = (
                max(DataStructs.BulkTanimotoSimilarity(fp, selected_fps))
                if selected_fps else 0.0
            )
            # Still favor candidates with decent UCB while exploring.
            score = (1.0 - maxsim) + 0.15 * float(ranked.loc[idx, "ucb"])
            explore_scores.append((score, idx))
        explore_scores.sort(reverse=True)
        for _, idx in explore_scores[:n_explore]:
            if idx not in used:
                picked.append(idx)
                used.add(idx)

    # Fill any gap with acquisition rank.
    if len(picked) < budget:
        for idx in exploit.index:
            if idx not in used:
                picked.append(idx)
                used.add(idx)
            if len(picked) >= budget:
                break

    return ranked.loc[picked[:budget]].copy().reset_index(drop=True)


def final_top20(
    store: ScoreStore,
    guard: HistoricalGuard,
    config: Dict[str, Any],
) -> pd.DataFrame:
    t0_final = time.time()
    df = store.dataframe()
    if df.empty:
        return df

    # Revalidate current archive availability.
    rows = []
    used_inchi = set()
    for _, row in df.sort_values("score", ascending=False).iterrows():
        if len(rows) >= max(100, int(config.get("num_molecules", 20)) * 5):
            break
        s = row["smiles"]
        ik = row["inchikey"] or inchikey(s)
        if not ik or ik in used_inchi:
            continue
        if not brenk_ok(s):
            continue
        if not guard.hf_ok(s):
            continue
        if not guard.history_ok(s):
            continue
        used_inchi.add(ik)
        rows.append(row)

    if not rows:
        return df.head(0)

    candidates = pd.DataFrame(rows).reset_index(drop=True)
    n = int(config.get("num_molecules", 20))

    # Default: top 20 scores. If entropy check fails, greedily diversify the tail.
    top = candidates.head(n).copy()
    if len(top) < n:
        return top

    if compute_maccs_entropy is not None:
        try:
            ent = float(compute_maccs_entropy(top["smiles"].tolist()))
            min_ent = float(config.get("min_entropy", 0.1))
            if ent < min_ent:
                log.warning("Top20 entropy %.4f < %.4f; running diversity repair", ent, min_ent)
                chosen = [candidates.iloc[0]]
                chosen_fps = [morgan_bv(chosen[0]["smiles"])]
                remaining = candidates.iloc[1:].copy()

                while len(chosen) < n and not remaining.empty:
                    best_idx = None
                    best_val = -math.inf
                    for idx, row in remaining.iterrows():
                        fp = morgan_bv(row["smiles"])
                        if fp is None:
                            continue
                        maxsim = max(DataStructs.BulkTanimotoSimilarity(fp, chosen_fps))
                        # Score remains primary; diversity only repairs invalid portfolio.
                        score_norm = float(row["score"] - candidates["score"].min()) / (
                            float(candidates["score"].max() - candidates["score"].min()) + 1e-9
                        )
                        val = 0.85 * score_norm + 0.15 * (1.0 - maxsim)
                        if val > best_val:
                            best_val = val
                            best_idx = idx
                    if best_idx is None:
                        break
                    chosen.append(remaining.loc[best_idx])
                    chosen_fps.append(morgan_bv(remaining.loc[best_idx, "smiles"]))
                    remaining = remaining.drop(index=best_idx)

                repaired = pd.DataFrame(chosen).reset_index(drop=True)
                if len(repaired) == n:
                    top = repaired
        except Exception as e:
            log.debug("Entropy check unavailable/failed: %s", e)

    top = top.reset_index(drop=True)
    log.info("final_top20 rebuild: %d molecules | %.2fs",
             len(top), time.time() - t0_final)
    return top


def export_top20(df: pd.DataFrame, rxn_id: int, target: str):
    if df.empty:
        return
    path = OUTPUT_DIR / f"TOP20_rxn{rxn_id}_{target}.csv"
    df.to_csv(path, index=False)

    names = df["name"].tolist()
    txt = OUTPUT_DIR / f"TOP20_rxn{rxn_id}_{target}.txt"
    txt.write_text(",".join(names), encoding="utf-8")

    log.info("=" * 78)
    log.info("CURRENT TOP %d | sum=%.6f | #20=%.6f",
             len(df), float(df["score"].sum()), float(df["score"].iloc[-1]))
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        log.info("%02d  %.6f  %s", i, float(row["score"]), row["name"])
    log.info("CSV: %s", path)
    log.info("Submission molecule string: %s", txt)
    log.info("=" * 78)


# =============================================================================
# Target identity
# =============================================================================

def target_identity(config: Dict[str, Any]) -> Tuple[str, str]:
    targets = config.get("small_molecule_target") or []
    clips = config.get("small_molecule_target_clip_interval") or []
    payload = json.dumps({"targets": targets, "clips": clips}, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()
    label = "_".join(map(str, targets)) if targets else "unknown"
    return key, label


# =============================================================================
# Main
# =============================================================================

async def main():
    global _GLOBAL_CONFIG, _GLOBAL_ARGS, _GLOBAL_MANAGER

    args = parse_args()
    _GLOBAL_ARGS = args

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config()
    _GLOBAL_CONFIG = config

    rxn_id = args.rxn_id
    cfg = dict(config)
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"

    # Do not inherit the local miner's silent 40-heavy-atom ceiling.
    if args.max_heavy_atoms > 0:
        cfg["max_heavy_atoms"] = args.max_heavy_atoms
    else:
        cfg["max_heavy_atoms"] = 10**9

    manager = MoleculeManager(config=cfg, db_path=str(DB_PATH))
    _GLOBAL_MANAGER = manager
    target_key, target_label = target_identity(config)
    target = config["small_molecule_target"][0]

    store = ScoreStore(rxn_id, target_key, target_label)
    store.import_legacy_once(args.legacy_db, manager)

    guard = HistoricalGuard(
        target=target,
        max_similarity=float(config.get("max_similarity_to_historical", 0.9)),
        disable_hf=args.disable_hf_filter,
        disable_history=args.disable_history_filter,
    )

    if BoltzWrapper is None:
        raise RuntimeError(
            "Could not import BoltzWrapper. Run this inside the nova-4090 repository "
            "with the same environment used by miner/miner.py."
        )
    boltz = BoltzWrapper()

    surrogate = Top20Surrogate(
        min_train=args.min_train,
        train_cap=args.train_cap,
        seed=args.seed,
    )
    generator = CandidateGenerator(rxn_id, manager, surrogate, args)
    rng = np.random.default_rng(args.seed)

    log.info("SN68 NOVA Top20 V2")
    log.info("target=%s | target_key=%s", target_label, target_key[:12])
    log.info("rxn=%d | A=%d B=%d C=%d",
             rxn_id,
             len(manager.moles_A_id),
             len(manager.moles_B_id),
             len(manager.moles_C_id or []))
    log.info("validator objective: exactly %d molecules", int(config.get("num_molecules", 20)))
    log.info("Boltz combination=%s metrics=%s",
             config.get("combination_strategy"), config.get("boltz_metric"))
    log.info("primary score DB=%s", store.path)

    for round_no in range(1, args.max_rounds + 1):
        t0 = time.time()
        scored = store.dataframe()
        seen = set(scored["name"].tolist()) if not scored.empty else set()

        # Fit one unified model on ALL current-target discoveries.
        surrogate.fit(scored)

        top20_before = final_top20(store, guard, config)
        threshold_before = (
            float(top20_before["score"].iloc[-1])
            if len(top20_before) >= 20 else -math.inf
        )
        sum_before = (
            float(top20_before["score"].sum())
            if not top20_before.empty else -math.inf
        )

        raw = generator.generate(scored, seen, args.candidate_pool)
        log.info(
            "[round %d] generated/locally-valid fresh=%d | already scored=%d",
            round_no, len(raw), len(seen)
        )
        if raw.empty:
            await asyncio.sleep(args.sleep)
            continue

        # Cheap exact archive identity check on the full pool.
        t_filter = time.time()
        filtered = guard.exact_filter(raw)
        log.info("[round %d] exact-archive-valid=%d/%d | %.2fs", round_no, len(filtered), len(raw), time.time()-t_filter)
        if filtered.empty:
            await asyncio.sleep(args.sleep)
            continue

        if surrogate.trained:
            ranked = surrogate.predict(filtered)
            # Historical-similarity filtering is expensive; apply it only to an acquisition shortlist.
            strict_pool_size = min(len(ranked), max(args.boltz_budget * 5, 500))
            prechosen = choose_boltz_batch(ranked, strict_pool_size, rng)
            strict = guard.strict_filter(prechosen)
            if len(strict) < args.boltz_budget:
                used_names = set(strict["name"]) if not strict.empty else set()
                remain = ranked[~ranked["name"].isin(used_names)].sort_values(
                    ["acq","p_improve","ei","mu"], ascending=False
                ).head(max(args.boltz_budget * 5, 500))
                extra = guard.strict_filter(remain)
                strict = pd.concat([strict, extra], ignore_index=True).drop_duplicates("name")
            chosen = choose_boltz_batch(strict, args.boltz_budget, rng)
        else:
            sample_n=min(len(filtered), max(args.boltz_budget * 5, 500))
            pre=filtered.sample(n=sample_n, random_state=args.seed+round_no).reset_index(drop=True)
            strict=guard.strict_filter(pre)
            if len(strict)>args.boltz_budget:
                chosen=strict.sample(n=args.boltz_budget, random_state=args.seed+1000+round_no).reset_index(drop=True)
            else:
                chosen=strict.copy()
            chosen["source"]="bootstrap"

        if chosen.empty:
            await asyncio.sleep(args.sleep)
            continue

        # Preserve search provenance for debugging.
        if "source" not in chosen.columns:
            chosen["source"] = "top20_acquisition"

        payload = chosen[["name", "smiles", "source"]].to_dict("records")
        scored_new = await boltz_score(
            boltz=boltz,
            config=config,
            molecules=payload,
            batch_size=args.batch_size,
        )
        store.write(scored_new, round_no=round_no)

        # Every discovery is immediately visible to next round's model/parents/anchors.
        top20_after = final_top20(store, guard, config)
        export_top20(top20_after, rxn_id, target)

        threshold_after = (
            float(top20_after["score"].iloc[-1])
            if len(top20_after) >= 20 else -math.inf
        )
        sum_after = (
            float(top20_after["score"].sum())
            if not top20_after.empty else -math.inf
        )

        improved = 0
        if np.isfinite(threshold_before) and np.isfinite(threshold_after):
            improved = sum(1 for r in scored_new if float(r["boltz_score"]) > threshold_before)

        log.info(
            "[round %d done] new=%d | beat old #20=%d | "
            "top20 sum %.6f -> %.6f | #20 %.6f -> %.6f | %.1fs",
            round_no,
            len(scored_new),
            improved,
            sum_before if np.isfinite(sum_before) else 0.0,
            sum_after if np.isfinite(sum_after) else 0.0,
            threshold_before if np.isfinite(threshold_before) else 0.0,
            threshold_after if np.isfinite(threshold_after) else 0.0,
            time.time() - t0,
        )

        await asyncio.sleep(args.sleep)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped by user")
