#!/usr/bin/env python3
"""
late_stage_search.py — Late-stage mining for a long-lived protein target.

When top chemotypes are already submitted (HF uniqueness + historical
Tanimoto >= 0.6), chasing absolute Boltz peaks is wasteful. This miner
optimizes for:

    available + diverse + competitive score

Pipeline per round (one fixed --rxn_id):
  1. Analyze elite reactants from data/rxn{N}.csv (high score, not overused)
  2. Generate candidates via:
       a. component exhaust around elite reactant anchors
       b. exploratory genetic crossover + high-ratio mutation
  3. Dedup vs score_results DB + rxn CSV (never re-Boltz known names)
  4. Validate property filters (heavy atoms / banned / rotatable bonds)
  5. Reject HF duplicates + historical near-duplicates BEFORE Boltz
  6. Surrogate + novelty rank → keep top Boltz budget
  7. Boltz score → write into score_results_{rxn}.sqlite
  8. Export second-shelf inventory CSV (informational only)

Does NOT modify or refill miner/submit.py. submit.py already reads the
score DBs / available molecules on its own.

Usage examples:
  python3 neurons/late_stage_search.py --rxn_id 1
  python3 neurons/late_stage_search.py --rxn_id 3 --boltz_budget 200
  python3 neurons/late_stage_search.py --rxn_id 2 --mutation_ratio 0.8
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# ── project root ──────────────────────────────────────────────────────────
NEURONS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(NEURONS_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if NEURONS_DIR not in sys.path:
    sys.path.insert(0, NEURONS_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
DATA_DIR = os.path.join(BASE_DIR, "data")
INVENTORY_DIR = os.path.join(DATA_DIR, "late_stage_inventory")

from config.config_loader import load_config
from utils import (
    get_historical_submissions,
    molecule_unique_for_protein_hf,
    get_heavy_atom_count,
    contains_atom_type,
)
from combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from molecules import MoleculeManager, MoleculeUtils
import score_store

# Target identity for the shared score DB (orchestrator.py's format), resolved
# from config in main()/run_rxn_round().
TARGET_KEY: Optional[str] = None
TARGET_LABEL: Optional[str] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("late_stage")

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)
_fp_cache: Dict[str, np.ndarray] = {}
_bv_cache: Dict[str, Any] = {}

# Defaults tuned for late-stage / saturated archives
MAX_SIMILARITY_TO_HISTORICAL = 0.9
DEFAULT_MUTATION_RATIO = 0.7
DEFAULT_MUTATION_MODE = "mixed"
DEFAULT_NEIGHBOUR_RATIO = 0.4          # lean random for exploration
DEFAULT_NEIGHBOUR_TOP_K = 40
DEFAULT_MIN_NEIGHBOUR_SIM = 0.35
DEFAULT_NOVELTY_WEIGHT = 0.40          # score_hat = (1-w)*pred + w*novelty
DEFAULT_SHELF_MIN_RATIO = 0.70
DEFAULT_SHELF_MAX_RATIO = 0.98
DEFAULT_BOLTZ_BUDGET = 150
DEFAULT_GENERATE_N = 4000
DEFAULT_EXHAUST_SAMPLE = 2500
DEFAULT_N_ELITE_ANCHORS = 6
DEFAULT_ROUNDS = 10**9                 # run until Ctrl+C unless capped
SURROGATE_TOP_N = 30000
SURROGATE_BOTTOM_N = 15000
SURROGATE_MIN_TRAIN = 200

BoltzWrapper = None


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprints
# ═══════════════════════════════════════════════════════════════════════════

def get_morgan_fp_array(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    if smiles in _fp_cache:
        return _fp_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    arr[fp.GetOnBits()] = 1
    _fp_cache[smiles] = arr
    if len(_fp_cache) > 80_000:
        for k in list(_fp_cache.keys())[:20_000]:
            del _fp_cache[k]
    return arr


def get_morgan_fp_bv(smiles: str):
    if smiles in _bv_cache:
        return _bv_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    _bv_cache[smiles] = fp
    if len(_bv_cache) > 80_000:
        for k in list(_bv_cache.keys())[:20_000]:
            del _bv_cache[k]
    return fp


# ═══════════════════════════════════════════════════════════════════════════
# Paths / score DB
# ═══════════════════════════════════════════════════════════════════════════

def score_db_path(rxn_id: int) -> str:
    return os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")


def rxn_csv_path(rxn_id: int) -> str:
    return os.path.join(DATA_DIR, f"rxn{rxn_id}.csv")


def init_score_results_db(db_path: str, rxn_id: Optional[int] = None) -> None:
    """Canonical orchestrator schema + target guard (see score_store)."""
    score_store.init_score_results_db(
        db_path,
        rxn_id=rxn_id,
        target_key=TARGET_KEY,
        target_label=TARGET_LABEL,
    )


def load_all_scored(db_path: str, rxn_id: int) -> pd.DataFrame:
    """
    Orchestrator's dataframe shape: [name, smiles, inchikey, score, source,
    round]. Callers may reuse the stored SMILES instead of rebuilding it.
    """
    return score_store.load_all_scored(db_path, rxn_id)


def load_scored_name_set(db_path: str, rxn_id: int) -> Set[str]:
    names: Set[str] = score_store.load_scored_name_set(db_path, rxn_id)
    csv_path = rxn_csv_path(rxn_id)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]
            if "molecule_name" in df.columns:
                prefix = f"rxn:{rxn_id}:"
                names |= {
                    str(n).strip()
                    for n in df["molecule_name"].tolist()
                    if str(n).startswith(prefix)
                }
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
    return names


def write_scores_to_db(
    db_path: str,
    records: List[Dict[str, Any]],
    rxn_id: Optional[int] = None,
    round_no: int = 0,
) -> int:
    """Upsert with orchestrator's full column set (smiles/inchikey included)."""
    return score_store.write_scores_to_db(
        db_path,
        records,
        rxn_id=rxn_id,
        round_no=round_no,
        target_key=TARGET_KEY,
        target_label=TARGET_LABEL,
        source="late_stage",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Historical diversity
# ═══════════════════════════════════════════════════════════════════════════

def load_historical_fingerprints(target_protein: str) -> Optional[pd.DataFrame]:
    historical_df = get_historical_submissions(target_protein, "molecules")
    if historical_df is None or historical_df.empty:
        logger.warning(f"No historical submissions for '{target_protein}'")
        return None

    mols = [Chem.MolFromSmiles(smi) for smi in historical_df["SMILES"]]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    if len(valid_idx) != len(mols):
        historical_df = historical_df.iloc[valid_idx].reset_index(drop=True)
        mols = [mols[i] for i in valid_idx]
    if not mols:
        return None

    fps = MORGAN_FP_GENERATOR.GetFingerprints(mols, numThreads=8)
    historical_df = historical_df.copy()
    historical_df["fingerprint"] = list(fps)
    return historical_df


def max_historical_similarity(
    smiles: str,
    historical_df: Optional[pd.DataFrame],
) -> float:
    if historical_df is None or historical_df.empty:
        return 0.0
    fp = get_morgan_fp_bv(smiles)
    if fp is None:
        return 1.0
    sims = DataStructs.BulkTanimotoSimilarity(
        fp, list(historical_df["fingerprint"])
    )
    return float(max(sims)) if sims else 0.0


def is_diverse_enough(
    smiles: str,
    historical_df: Optional[pd.DataFrame],
    max_similarity: float = MAX_SIMILARITY_TO_HISTORICAL,
) -> bool:
    return max_historical_similarity(smiles, historical_df) < max_similarity


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_smiles(smiles: str, config: Dict[str, Any]) -> bool:
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        n_heavy = get_heavy_atom_count(smiles)
    except Exception:
        return False
    if n_heavy < config.get("min_heavy_atoms", 10):
        return False
    if n_heavy > config.get("max_heavy_atoms", 40):
        return False
    banned = config.get("banned_atom_types")
    if banned and contains_atom_type(mol, banned):
        return False
    n_rot = Descriptors.NumRotatableBonds(mol)
    if n_rot < config.get("min_rotatable_bonds", 0):
        return False
    if n_rot > config.get("max_rotatable_bonds", 15):
        return False
    return True


def resolve_smiles(name: str) -> Optional[str]:
    try:
        smiles = MoleculeUtils.get_smiles_from_reaction_cached(name)
        if smiles:
            return smiles
    except Exception:
        pass
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Reaction coverage (logging / MoleculeManager bootstrap for one rxn)
# ═══════════════════════════════════════════════════════════════════════════

def parse_component_ids(name: str) -> Optional[Tuple[int, ...]]:
    parts = str(name).split(":")
    if len(parts) < 4 or parts[0] != "rxn":
        return None
    try:
        return tuple(int(p) for p in parts[2:])
    except ValueError:
        return None


def measure_rxn_coverage(rxn_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Fraction of reactant IDs that have never appeared in scored molecules."""
    cfg = dict(config)
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    pools = {
        "A": set(int(x) for x in manager.moles_A_id),
        "B": set(int(x) for x in manager.moles_B_id),
    }
    if getattr(manager, "is_three_component", False):
        pools["C"] = set(int(x) for x in manager.moles_C_id)

    discovered = {role: set() for role in pools}
    scored = load_all_scored(score_db_path(rxn_id), rxn_id)
    for name in scored["name"].tolist():
        ids = parse_component_ids(name)
        if not ids:
            continue
        roles = list(pools.keys())
        for role, cid in zip(roles, ids):
            discovered[role].add(cid)

    # Also count CSV-known components as "seen" for coverage
    csv_path = rxn_csv_path(rxn_id)
    if os.path.exists(csv_path):
        try:
            csv_df = pd.read_csv(csv_path)
            csv_df.columns = [c.strip().lower() for c in csv_df.columns]
            if "molecule_name" in csv_df.columns:
                for name in csv_df["molecule_name"].tolist():
                    ids = parse_component_ids(str(name))
                    if not ids:
                        continue
                    for role, cid in zip(pools.keys(), ids):
                        discovered[role].add(cid)
        except Exception:
            pass

    undiscovered_fracs = []
    role_stats = {}
    for role, pool in pools.items():
        und = pool - discovered[role]
        frac = (len(und) / len(pool)) if pool else 0.0
        undiscovered_fracs.append(frac)
        role_stats[role] = {
            "total": len(pool),
            "discovered": len(discovered[role] & pool),
            "undiscovered": len(und),
            "undiscovered_frac": frac,
        }

    best_score = float(scored["score"].max()) if not scored.empty else 0.0
    n_scored = len(scored)

    return {
        "rxn_id": rxn_id,
        "n_scored": n_scored,
        "best_score": best_score,
        "role_stats": role_stats,
        "undiscovered_frac": float(np.mean(undiscovered_fracs)),
        "manager": manager,
        "is_three": bool(getattr(manager, "is_three_component", False)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Elite reactant analysis (quality vs saturation)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_elite_reactants(
    rxn_id: int,
    n_anchors: int = DEFAULT_N_ELITE_ANCHORS,
    min_count: int = 2,
) -> List[Dict[str, Any]]:
    """
    Pick reactant anchors with strong max/avg score, while avoiding the most
    over-mined chemotypes (extreme count outliers).

    Selection per role:
      - half: top by raw max_score (keep true peaks)
      - half: top by score * mild underuse bonus (explore less-saturated elites)
    """
    csv_path = rxn_csv_path(rxn_id)
    if not os.path.exists(csv_path):
        logger.warning(f"No CSV for rxn {rxn_id}: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "molecule_name" not in df.columns or "final_score" not in df.columns:
        return []

    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df = df[np.isfinite(df["final_score"])].copy()
    df["ids"] = df["molecule_name"].apply(parse_component_ids)
    df = df[df["ids"].notna()]
    if df.empty:
        return []

    n_comp = len(df["ids"].iloc[0])
    role_names = ["A", "B", "C"][:n_comp]
    anchors: List[Dict[str, Any]] = []
    per_role_budget = max(2, n_anchors // max(1, len(role_names)))

    for role_idx, role in enumerate(role_names):
        stats = defaultdict(lambda: {"scores": [], "count": 0})
        for ids, score in zip(df["ids"], df["final_score"]):
            cid = ids[role_idx]
            stats[cid]["scores"].append(float(score))
            stats[cid]["count"] += 1

        rows = []
        for cid, st in stats.items():
            if st["count"] < min_count:
                continue
            scores = st["scores"]
            avg = float(np.mean(scores))
            mx = float(np.max(scores))
            # Mild underuse bonus: high-count peaks stay competitive, but
            # less-mined strong reactants get a small lift.
            underuse = 1.0 / (1.0 + 0.08 * np.log1p(st["count"]))
            quality = (0.65 * mx + 0.35 * avg) * underuse
            rows.append(
                {
                    "rxn_id": rxn_id,
                    "role": role,
                    "component_id": int(cid),
                    "count": st["count"],
                    "avg_score": avg,
                    "max_score": mx,
                    "quality": quality,
                }
            )
        if not rows:
            continue

        # Drop extreme saturation outliers (top ~5% by count) from the
        # underuse lane only; peak lane still can use them.
        counts = np.array([r["count"] for r in rows], dtype=float)
        count_cut = float(np.quantile(counts, 0.95)) if len(counts) >= 20 else float("inf")

        by_max = sorted(rows, key=lambda x: x["max_score"], reverse=True)
        by_quality = sorted(
            [r for r in rows if r["count"] <= count_cut],
            key=lambda x: x["quality"],
            reverse=True,
        )
        if not by_quality:
            by_quality = sorted(rows, key=lambda x: x["quality"], reverse=True)

        n_peak = max(1, per_role_budget // 2)
        n_explore = max(1, per_role_budget - n_peak)
        picked: Dict[int, Dict[str, Any]] = {}
        for r in by_max[:n_peak]:
            picked[r["component_id"]] = r
        for r in by_quality:
            if len(picked) >= per_role_budget:
                break
            picked.setdefault(r["component_id"], r)
        # Fill if still short
        for r in by_max:
            if len(picked) >= per_role_budget:
                break
            picked.setdefault(r["component_id"], r)

        anchors.extend(picked.values())

    anchors.sort(key=lambda x: (x["quality"], x["max_score"]), reverse=True)
    selected = anchors[:n_anchors]
    for a in selected:
        logger.info(
            f"[ELITE] rxn:{rxn_id} {a['role']}={a['component_id']} "
            f"max={a['max_score']:.4f} avg={a['avg_score']:.4f} "
            f"n={a['count']} quality={a['quality']:.4f}"
        )
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Reactant library + genetic exploration
# ═══════════════════════════════════════════════════════════════════════════

class ReactantLibrary:
    ROLE_BY_NAME_INDEX = {2: "A", 3: "B", 4: "C"}

    def __init__(
        self,
        rxn_id: int,
        db_path: str,
        neighbour_top_k: int = DEFAULT_NEIGHBOUR_TOP_K,
        min_neighbour_sim: float = DEFAULT_MIN_NEIGHBOUR_SIM,
    ):
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.neighbour_top_k = neighbour_top_k
        self.min_neighbour_sim = min_neighbour_sim
        self._smiles_by_role: Dict[str, Dict[int, str]] = {}
        self._ids_by_role: Dict[str, List[int]] = {}
        self._fp_index: Dict[str, Tuple[List[int], List[Any]]] = {}
        self._neighbour_cache: Dict[Tuple[str, int], List[int]] = {}
        self._load_pools()

    def _load_pools(self) -> None:
        info = get_reaction_info(self.rxn_id, self.db_path)
        if not info:
            raise ValueError(f"Cannot load reaction {self.rxn_id}")
        _smarts, role_a, role_b, role_c = info
        role_masks = {"A": role_a, "B": role_b}
        if role_c:
            role_masks["C"] = role_c
        for role, mask in role_masks.items():
            abs_db = os.path.abspath(self.db_path)
            with sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True) as conn:
                conn.execute("PRAGMA query_only = ON")
                rows = conn.execute(
                    "SELECT mol_id, smiles FROM molecules WHERE (role_mask & ?) = ?",
                    (mask, mask),
                ).fetchall()
            self._smiles_by_role[role] = {mid: smi for mid, smi in rows if smi}
            self._ids_by_role[role] = list(self._smiles_by_role[role].keys())
        logger.info(
            "Reactant pools rxn:%s → %s",
            self.rxn_id,
            ", ".join(f"{r}={len(self._ids_by_role[r])}" for r in sorted(self._ids_by_role)),
        )

    def role_for_name_index(self, name_index: int) -> Optional[str]:
        role = self.ROLE_BY_NAME_INDEX.get(name_index)
        return role if role in self._ids_by_role else None

    def random_component(self, role: str, exclude_id: Optional[int] = None) -> Optional[int]:
        pool = self._ids_by_role.get(role, [])
        if not pool:
            return None
        for _ in range(8):
            cid = random.choice(pool)
            if cid != exclude_id:
                return cid
        return None

    def _fingerprint_index(self, role: str) -> Tuple[List[int], List[Any]]:
        if role in self._fp_index:
            return self._fp_index[role]
        ids, fps = [], []
        for cid, smiles in self._smiles_by_role.get(role, {}).items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            ids.append(cid)
            fps.append(MORGAN_FP_GENERATOR.GetFingerprint(mol))
        self._fp_index[role] = (ids, fps)
        return self._fp_index[role]

    def neighbour_component(self, role: str, component_id: int) -> Optional[int]:
        key = (role, component_id)
        if key in self._neighbour_cache:
            nbs = self._neighbour_cache[key]
            return random.choice(nbs) if nbs else None

        smiles = self._smiles_by_role.get(role, {}).get(component_id)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        nbs: List[int] = []
        if mol is not None:
            ids, fps = self._fingerprint_index(role)
            if ids:
                qfp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                sims = DataStructs.BulkTanimotoSimilarity(qfp, fps)
                ranked = sorted(
                    (
                        (cid, sim)
                        for cid, sim in zip(ids, sims)
                        if cid != component_id and sim >= self.min_neighbour_sim
                    ),
                    key=lambda p: p[1],
                    reverse=True,
                )
                nbs = [cid for cid, _ in ranked[: self.neighbour_top_k]]
        self._neighbour_cache[key] = nbs
        return random.choice(nbs) if nbs else None

    def replacement_component(
        self,
        role: str,
        component_id: int,
        mode: str,
        neighbour_ratio: float,
    ) -> Tuple[Optional[int], str]:
        want_nb = mode == "neighbour" or (
            mode == "mixed" and random.random() < neighbour_ratio
        )
        if want_nb:
            nb = self.neighbour_component(role, component_id)
            if nb is not None:
                return nb, "neighbour"
            if mode == "neighbour":
                return None, "neighbour"
        return self.random_component(role, exclude_id=component_id), "random"


class GeneticExplorer:
    def __init__(
        self,
        rxn_id: int,
        library: Optional[ReactantLibrary],
        mutation_ratio: float,
        mutation_mode: str,
        neighbour_ratio: float,
    ):
        self.rxn_id = rxn_id
        self.library = library
        self.mutation_ratio = mutation_ratio if library is not None else 0.0
        self.mutation_mode = mutation_mode
        self.neighbour_ratio = neighbour_ratio
        self.seen: Set[str] = set()

    def _buildable(self, name: str) -> bool:
        smiles = resolve_smiles(name)
        return bool(smiles and Chem.MolFromSmiles(smiles))

    def crossover(self, a: str, b: str) -> Optional[str]:
        p1, p2 = a.split(":"), b.split(":")
        if len(p1) != len(p2) or len(p1) not in (4, 5):
            return None
        try:
            if int(p1[1]) != self.rxn_id or int(p2[1]) != self.rxn_id:
                return None
        except ValueError:
            return None
        swap = random.choice(list(range(2, len(p1))))
        out = p1.copy()
        out[swap] = p2[swap]
        name = ":".join(out)
        if name in self.seen or name in (a, b):
            return None
        if self._buildable(name):
            self.seen.add(name)
            return name
        return None

    def mutate(self, name: str) -> Optional[str]:
        if self.library is None:
            return None
        parts = name.split(":")
        if len(parts) not in (4, 5):
            return None
        indices = list(range(2, len(parts)))
        random.shuffle(indices)
        for idx in indices:
            role = self.library.role_for_name_index(idx)
            if role is None:
                continue
            try:
                cur = int(parts[idx])
            except ValueError:
                continue
            new_id, _kind = self.library.replacement_component(
                role, cur, self.mutation_mode, self.neighbour_ratio
            )
            if new_id is None or new_id == cur:
                continue
            mutant_parts = parts.copy()
            mutant_parts[idx] = str(new_id)
            mutant = ":".join(mutant_parts)
            if mutant in self.seen:
                continue
            if self._buildable(mutant):
                self.seen.add(mutant)
                return mutant
        return None

    def generate(self, parent_names: List[str], n: int) -> List[str]:
        if len(parent_names) < 2:
            return []
        out: List[str] = []
        attempts = 0
        max_attempts = max(500, n * 4)
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            p1, p2 = random.sample(parent_names, 2)
            child = self.crossover(p1, p2)
            if child is None:
                continue
            if self.mutation_ratio > 0 and random.random() < self.mutation_ratio:
                mutant = self.mutate(child)
                if mutant is not None:
                    child = mutant
            out.append(child)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Component exhaust sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_exhaust_candidates(
    rxn_id: int,
    manager: MoleculeManager,
    fixed_role: str,
    fixed_id: int,
    sample_size: int,
    exclude: Set[str],
    rng: np.random.Generator,
) -> List[str]:
    is_three = bool(getattr(manager, "is_three_component", False))
    all_roles = ["A", "B", "C"] if is_three else ["A", "B"]
    vary_roles = [r for r in all_roles if r != fixed_role]
    pool_map = {
        "A": [int(x) for x in manager.moles_A_id],
        "B": [int(x) for x in manager.moles_B_id],
        "C": [int(x) for x in getattr(manager, "moles_C_id", [])],
    }
    pools = [pool_map[r] for r in vary_roles]
    if any(len(p) == 0 for p in pools):
        return []

    sizes = [len(p) for p in pools]
    total = 1
    for s in sizes:
        total *= s

    names: List[str] = []
    seen_local: Set[str] = set()
    draws = min(sample_size * 3, max(sample_size, total))
    for _ in range(draws):
        if len(names) >= sample_size:
            break
        parts = {fixed_role: fixed_id}
        for role, pool in zip(vary_roles, pools):
            parts[role] = int(pool[int(rng.integers(0, len(pool)))])
        if is_three:
            name = f"rxn:{rxn_id}:{parts['A']}:{parts['B']}:{parts['C']}"
        else:
            name = f"rxn:{rxn_id}:{parts['A']}:{parts['B']}"
        if name in exclude or name in seen_local:
            continue
        seen_local.add(name)
        names.append(name)
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Novelty-aware surrogate
# ═══════════════════════════════════════════════════════════════════════════

class NoveltySurrogate:
    """
    RandomForest score predictor + novelty term vs historical fingerprints.

    Combined ranking score:
        (1 - novelty_weight) * normalized_pred + novelty_weight * novelty

    where novelty = 1 - max_tanimoto_to_historical
    """

    def __init__(self, novelty_weight: float = DEFAULT_NOVELTY_WEIGHT):
        self.novelty_weight = float(np.clip(novelty_weight, 0.0, 1.0))
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def train(self, rxn_id: int, db_path: str) -> None:
        frames = []
        db_df = load_all_scored(db_path, rxn_id)
        if not db_df.empty:
            # DB rows already carry SMILES under the canonical schema.
            frames.append(db_df[["name", "smiles", "score"]])

        csv_path = rxn_csv_path(rxn_id)
        if os.path.exists(csv_path):
            try:
                csv_df = pd.read_csv(csv_path)
                csv_df.columns = [c.strip().lower() for c in csv_df.columns]
                if {"molecule_name", "final_score"} <= set(csv_df.columns):
                    tmp = pd.DataFrame(
                        {
                            "name": csv_df["molecule_name"].astype(str),
                            "smiles": None,
                            "score": pd.to_numeric(
                                csv_df["final_score"], errors="coerce"
                            ),
                        }
                    )
                    tmp = tmp[np.isfinite(tmp["score"])]
                    frames.append(tmp)
            except Exception as e:
                logger.warning(f"CSV train load failed: {e}")

        if not frames:
            logger.warning("[SURROGATE] no training data")
            self.is_trained = False
            return

        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="name")
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        top = df.head(SURROGATE_TOP_N)
        bottom = df.tail(SURROGATE_BOTTOM_N)
        train_df = pd.concat([top, bottom]).drop_duplicates(subset="name")

        X, y = [], []
        for name, stored_smiles, score in zip(
            train_df["name"], train_df["smiles"], train_df["score"]
        ):
            smiles = stored_smiles if isinstance(stored_smiles, str) and stored_smiles \
                else resolve_smiles(str(name))
            if not smiles:
                continue
            fp = get_morgan_fp_array(smiles)
            if fp is None:
                continue
            X.append(fp)
            y.append(float(score))

        if len(X) < SURROGATE_MIN_TRAIN:
            logger.warning(
                f"[SURROGATE] only {len(X)} samples (<{SURROGATE_MIN_TRAIN}) — untrained"
            )
            self.is_trained = False
            return

        t0 = time.time()
        self.model.fit(np.asarray(X), np.asarray(y))
        self.is_trained = True
        logger.info(f"[SURROGATE] trained on {len(X)} samples in {time.time()-t0:.1f}s")

    def rank(
        self,
        candidates: pd.DataFrame,
        historical_df: Optional[pd.DataFrame],
        keep_n: int,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        df = candidates.copy()
        smiles_list = df["smiles"].tolist()

        if self.is_trained:
            fps = []
            for s in smiles_list:
                fp = get_morgan_fp_array(s)
                fps.append(fp if fp is not None else np.zeros(2048, dtype=np.uint8))
            preds = self.model.predict(np.asarray(fps))
        else:
            preds = np.random.random(len(df))

        novelties = np.array(
            [1.0 - max_historical_similarity(s, historical_df) for s in smiles_list],
            dtype=float,
        )

        # Normalize predictions to [0,1] within the batch for mixing
        p = np.asarray(preds, dtype=float)
        if np.nanmax(p) > np.nanmin(p):
            p_norm = (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p))
        else:
            p_norm = np.zeros_like(p)

        w = self.novelty_weight
        combined = (1.0 - w) * p_norm + w * novelties

        df["surrogate_score"] = p
        df["novelty"] = novelties
        df["rank_score"] = combined
        df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
        return df.head(keep_n).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Second-shelf inventory (export only — submit.py untouched)
# ═══════════════════════════════════════════════════════════════════════════

def export_second_shelf_inventory(
    rxn_id: int,
    historical_df: Optional[pd.DataFrame],
    target_protein: str,
    shelf_min_ratio: float,
    shelf_max_ratio: float,
) -> Path:
    """
    Export competitive, still-unique-looking molecules from the score DB.
    This is an informational inventory for the operator; submit.py is not
    modified and does not consume this file.
    """
    os.makedirs(INVENTORY_DIR, exist_ok=True)
    db_path = score_db_path(rxn_id)
    scored = load_all_scored(db_path, rxn_id)
    out_path = Path(INVENTORY_DIR) / f"rxn{rxn_id}_second_shelf.csv"

    if scored.empty:
        pd.DataFrame(
            columns=[
                "molecule_name",
                "score",
                "max_hist_sim",
                "hf_unique",
                "shelf_ok",
            ]
        ).to_csv(out_path, index=False)
        return out_path

    best = float(scored["score"].max())
    lo, hi = shelf_min_ratio * best, shelf_max_ratio * best
    band = scored[(scored["score"] >= lo) & (scored["score"] <= hi)].copy()
    band = band.sort_values("score", ascending=False).reset_index(drop=True)

    rows = []
    for name, stored_smiles, score in zip(
        band["name"], band["smiles"], band["score"]
    ):
        smiles = stored_smiles if isinstance(stored_smiles, str) and stored_smiles \
            else resolve_smiles(str(name))
        if not smiles:
            continue
        max_sim = max_historical_similarity(smiles, historical_df)
        try:
            hf_unique = bool(molecule_unique_for_protein_hf(target_protein, smiles))
        except Exception:
            hf_unique = False
        shelf_ok = (
            hf_unique
            and max_sim < MAX_SIMILARITY_TO_HISTORICAL
            and lo <= float(score) <= hi
        )
        rows.append(
            {
                "molecule_name": name,
                "score": float(score),
                "max_hist_sim": max_sim,
                "hf_unique": hf_unique,
                "shelf_ok": shelf_ok,
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ["shelf_ok", "score"], ascending=[False, False]
        ).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    n_ok = int(out_df["shelf_ok"].sum()) if not out_df.empty else 0
    logger.info(
        f"[SHELF] rxn:{rxn_id} band=[{lo:.4f},{hi:.4f}] "
        f"candidates={len(out_df)} submit_ready_looking={n_ok} → {out_path}"
    )
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Boltz
# ═══════════════════════════════════════════════════════════════════════════

def _import_boltz_wrapper() -> bool:
    global BoltzWrapper
    try:
        boltz_src = os.path.join(BASE_DIR, "boltz")
        if boltz_src not in sys.path:
            sys.path.insert(0, boltz_src)
        from boltz_wrapper import BoltzWrapper as BW

        BoltzWrapper = BW
        logger.info("BoltzWrapper imported")
        return True
    except Exception as e:
        logger.error(f"Failed to import BoltzWrapper: {e}")
        return False


async def boltz_score_batch(
    boltz,
    config: Dict[str, Any],
    target_proteins: List[str],
    molecules: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not molecules:
        return []
    primary = target_proteins[0]
    output_dir = os.path.join(boltz.output_dir, "boltz_results_inputs")
    processed = os.path.join(output_dir, "processed")
    for d in (
        os.path.join(processed, "structures"),
        os.path.join(processed, "records"),
        os.path.join(processed, "msa"),
        os.path.join(output_dir, "predictions"),
    ):
        os.makedirs(d, exist_ok=True)

    valid_molecules_by_uid = {
        0: {
            "smiles": [m["smiles"] for m in molecules],
            "names": [m["name"] for m in molecules],
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
    subnet_config = {
        "small_molecule_target": config["small_molecule_target"],
        "small_molecule_target_clip_interval": config[
            "small_molecule_target_clip_interval"
        ],
        "boltz_mode": config.get("boltz_mode", "max"),
        "boltz_metric": config.get(
            "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"],
        ),
        "combination_strategy": config.get(
            "combination_strategy", "heavy_atom_normalization"
        ),
    }

    def run_scoring():
        boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)

    t0 = time.time()
    await asyncio.get_event_loop().run_in_executor(None, run_scoring)
    logger.info(f"[Boltz] scored {len(molecules)} in {time.time()-t0:.1f}s")

    final_scores = getattr(boltz, "final_boltz_scores", {}).get(0, {})
    smiles_to_score = final_scores.get(primary, {}) if final_scores else {}
    for m in molecules:
        m["boltz_score"] = smiles_to_score.get(m["smiles"])
    return molecules


# ═══════════════════════════════════════════════════════════════════════════
# Candidate filtering pipeline
# ═══════════════════════════════════════════════════════════════════════════

def filter_candidates(
    names: List[str],
    config: Dict[str, Any],
    scored_names: Set[str],
    target_protein: str,
    historical_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Validate + dedup + HF + historical BEFORE any Boltz call."""
    rows = []
    stats = defaultdict(int)
    for name in names:
        stats["input"] += 1
        if name in scored_names:
            stats["already_scored"] += 1
            continue
        smiles = resolve_smiles(name)
        if not smiles:
            stats["no_smiles"] += 1
            continue
        if not validate_smiles(smiles, config):
            stats["invalid_props"] += 1
            continue
        try:
            if not molecule_unique_for_protein_hf(target_protein, smiles):
                stats["hf_dup"] += 1
                continue
        except Exception:
            stats["hf_error"] += 1
            continue
        if not is_diverse_enough(smiles, historical_df):
            stats["hist_sim"] += 1
            continue
        rows.append({"name": name, "smiles": smiles})
        stats["kept"] += 1

    logger.info(
        "[FILTER] in=%d already=%d no_smi=%d invalid=%d hf=%d hist=%d kept=%d",
        stats["input"],
        stats["already_scored"],
        stats["no_smiles"],
        stats["invalid_props"],
        stats["hf_dup"],
        stats["hist_sim"],
        stats["kept"],
    )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# One round for one reaction
# ═══════════════════════════════════════════════════════════════════════════

async def run_rxn_round(
    rxn_id: int,
    config: Dict[str, Any],
    boltz,
    historical_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    coverage: Optional[Dict[str, Any]] = None,
    round_no: int = 0,
) -> int:
    global TARGET_KEY, TARGET_LABEL
    if TARGET_KEY is None:
        TARGET_KEY, TARGET_LABEL = score_store.target_identity(config)

    target = config["small_molecule_target"][0]
    db_path = score_db_path(rxn_id)
    init_score_results_db(db_path, rxn_id=rxn_id)

    if coverage is None:
        coverage = measure_rxn_coverage(rxn_id, config)
    manager = coverage["manager"]

    scored_names = load_scored_name_set(db_path, rxn_id)
    logger.info(f"[RXN {rxn_id}] known scored/CSV names: {len(scored_names)}")

    # ── elite anchors ────────────────────────────────────────────────────
    elites = analyze_elite_reactants(rxn_id, n_anchors=args.n_elite_anchors)

    # ── generate exhaust candidates ──────────────────────────────────────
    rng = np.random.default_rng(args.seed + rxn_id + int(time.time()) % 10_000)
    raw_names: List[str] = []
    for anchor in elites:
        sampled = sample_exhaust_candidates(
            rxn_id=rxn_id,
            manager=manager,
            fixed_role=anchor["role"],
            fixed_id=anchor["component_id"],
            sample_size=max(50, args.exhaust_sample // max(1, len(elites))),
            exclude=scored_names,
            rng=rng,
        )
        raw_names.extend(sampled)
        logger.info(
            f"[EXHAUST] fixed {anchor['role']}={anchor['component_id']} "
            f"→ {len(sampled)} names"
        )

    # ── exploratory genetic ──────────────────────────────────────────────
    parent_names: List[str] = []
    csv_path = rxn_csv_path(rxn_id)
    if os.path.exists(csv_path):
        try:
            csv_df = pd.read_csv(csv_path)
            csv_df.columns = [c.strip().lower() for c in csv_df.columns]
            if "molecule_name" in csv_df.columns:
                if "final_score" in csv_df.columns:
                    csv_df["final_score"] = pd.to_numeric(
                        csv_df["final_score"], errors="coerce"
                    )
                    csv_df = csv_df.sort_values(
                        "final_score", ascending=False, na_position="last"
                    )
                # Use a deeper parent pool than classic top-200 for exploration
                parent_names = (
                    csv_df["molecule_name"].astype(str).head(500).tolist()
                )
        except Exception as e:
            logger.warning(f"parent CSV load failed: {e}")

    library = None
    try:
        library = ReactantLibrary(
            rxn_id,
            DB_PATH,
            neighbour_top_k=args.neighbour_top_k,
            min_neighbour_sim=args.min_neighbour_similarity,
        )
    except Exception as e:
        logger.warning(f"ReactantLibrary unavailable: {e}")

    explorer = GeneticExplorer(
        rxn_id=rxn_id,
        library=library,
        mutation_ratio=args.mutation_ratio,
        mutation_mode=args.mutation_mode,
        neighbour_ratio=args.neighbour_ratio,
    )
    genetic_names = explorer.generate(parent_names, n=args.generate_n)
    logger.info(f"[GENETIC] generated {len(genetic_names)} exploratory offspring")
    raw_names.extend(genetic_names)

    # Unique preserve order
    seen = set()
    unique_names = []
    for n in raw_names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    logger.info(f"[GEN] unique raw candidates: {len(unique_names)}")

    # ── pre-Boltz filters ────────────────────────────────────────────────
    filtered = filter_candidates(
        unique_names, config, scored_names, target, historical_df
    )
    if filtered.empty:
        logger.warning(f"[RXN {rxn_id}] no candidates survived pre-Boltz filters")
        export_second_shelf_inventory(
            rxn_id,
            historical_df,
            target,
            args.shelf_min_ratio,
            args.shelf_max_ratio,
        )
        return 0

    # ── novelty-aware surrogate rank ─────────────────────────────────────
    surrogate = NoveltySurrogate(novelty_weight=args.novelty_weight)
    surrogate.train(rxn_id, db_path)
    ranked = surrogate.rank(filtered, historical_df, keep_n=args.boltz_budget)
    logger.info(
        f"[RANK] sending {len(ranked)} / {len(filtered)} to Boltz "
        f"(novelty_weight={args.novelty_weight})"
    )

    # ── Boltz in small batches + write-through ───────────────────────────
    molecules = ranked[["name", "smiles"]].to_dict("records")
    written = 0
    batch_size = args.boltz_batch_size
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(molecules), batch_size), start=1):
        batch = molecules[i : i + batch_size]
        scored = await boltz_score_batch(
            boltz, config, config["small_molecule_target"], batch
        )
        ok = [m for m in scored if m.get("boltz_score") is not None]
        failed = [m for m in scored if m.get("boltz_score") is None]
        n = write_scores_to_db(db_path, ok, rxn_id=rxn_id, round_no=round_no)
        written += n

        ok_sorted = sorted(
            ok, key=lambda m: m["boltz_score"], reverse=True
        )
        print(
            f"\n{'='*70}\n"
            f"BATCH {batch_idx}/{total_batches}  "
            f"scored={len(ok)}  failed={len(failed)}  wrote={n}\n"
            f"{'='*70}"
        )
        if ok_sorted:
            print(f"{'molecule_name':<45} {'score':>12}")
            print("-" * 58)
            for m in ok_sorted:
                print(f"{m['name']:<45} {m['boltz_score']:12.6f}")
            best = ok_sorted[0]
            print("-" * 58)
            print(
                f"batch best: {best['name']}  "
                f"score={best['boltz_score']:.6f}"
            )
        if failed:
            print(f"failed ({len(failed)}):")
            for m in failed:
                print(f"  {m['name']}")
        print(f"{'='*70}\n", flush=True)

        logger.info(
            f"[Boltz batch {batch_idx}/{total_batches}] "
            f"wrote={n} scored={len(ok)} failed={len(failed)}"
            + (
                f" best={ok_sorted[0]['name']} "
                f"score={ok_sorted[0]['boltz_score']:.6f}"
                if ok_sorted
                else ""
            )
        )

    # ── second-shelf inventory export ────────────────────────────────────
    export_second_shelf_inventory(
        rxn_id,
        historical_df,
        target,
        args.shelf_min_ratio,
        args.shelf_max_ratio,
    )
    logger.info(f"[RXN {rxn_id}] round complete — {written} new scores written")
    return written


# ═══════════════════════════════════════════════════════════════════════════
# CLI / main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Late-stage Nova miner for one reaction: novelty + exhaust + exploration"
    )
    p.add_argument(
        "--rxn_id",
        type=int,
        required=True,
        help="Reaction id to mine (required, e.g. 1-5).",
    )
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--boltz_budget", type=int, default=DEFAULT_BOLTZ_BUDGET)
    p.add_argument("--boltz_batch_size", type=int, default=10)
    p.add_argument("--generate_n", type=int, default=DEFAULT_GENERATE_N)
    p.add_argument("--exhaust_sample", type=int, default=DEFAULT_EXHAUST_SAMPLE)
    p.add_argument("--n_elite_anchors", type=int, default=DEFAULT_N_ELITE_ANCHORS)
    p.add_argument("--mutation_ratio", type=float, default=DEFAULT_MUTATION_RATIO)
    p.add_argument(
        "--mutation_mode",
        choices=["neighbour", "random", "mixed"],
        default=DEFAULT_MUTATION_MODE,
    )
    p.add_argument("--neighbour_ratio", type=float, default=DEFAULT_NEIGHBOUR_RATIO)
    p.add_argument("--neighbour_top_k", type=int, default=DEFAULT_NEIGHBOUR_TOP_K)
    p.add_argument(
        "--min_neighbour_similarity",
        type=float,
        default=DEFAULT_MIN_NEIGHBOUR_SIM,
    )
    p.add_argument("--novelty_weight", type=float, default=DEFAULT_NOVELTY_WEIGHT)
    p.add_argument("--shelf_min_ratio", type=float, default=DEFAULT_SHELF_MIN_RATIO)
    p.add_argument("--shelf_max_ratio", type=float, default=DEFAULT_SHELF_MAX_RATIO)
    p.add_argument("--seed", type=int, default=68)
    p.add_argument(
        "--refresh_historical_every",
        type=int,
        default=3,
        help="Reload historical fingerprints every N rounds",
    )
    return p.parse_args()


async def main() -> None:
    global TARGET_KEY, TARGET_LABEL

    args = parse_args()
    if not (0.0 <= args.mutation_ratio <= 1.0):
        raise SystemExit("--mutation_ratio must be in [0,1]")
    if not (0.0 <= args.novelty_weight <= 1.0):
        raise SystemExit("--novelty_weight must be in [0,1]")

    config = load_config()
    target = config["small_molecule_target"][0]
    rxn_id = args.rxn_id

    # Tag rows with the same target identity orchestrator.py uses.
    TARGET_KEY, TARGET_LABEL = score_store.target_identity(config)

    logger.info("=" * 70)
    logger.info("LATE-STAGE SEARCH")
    logger.info(f"target={target}  rxn_id={rxn_id}")
    logger.info(
        f"mutation_ratio={args.mutation_ratio} mode={args.mutation_mode} "
        f"neighbour_ratio={args.neighbour_ratio}"
    )
    logger.info(
        f"novelty_weight={args.novelty_weight} boltz_budget={args.boltz_budget}"
    )
    logger.info(
        f"shelf=[{args.shelf_min_ratio:.2f},{args.shelf_max_ratio:.2f}]×best "
        f"(export only — submit.py unchanged)"
    )
    logger.info("=" * 70)

    if not _import_boltz_wrapper() or BoltzWrapper is None:
        logger.error("BoltzWrapper required — aborting")
        return
    boltz = BoltzWrapper()

    historical_df = await asyncio.to_thread(load_historical_fingerprints, target)
    logger.info(
        f"Historical submissions: "
        f"{0 if historical_df is None else len(historical_df)} "
        f"(reject Tanimoto >= {MAX_SIMILARITY_TO_HISTORICAL})"
    )

    coverage = measure_rxn_coverage(rxn_id, config)
    logger.info(
        f"[COVERAGE] rxn:{rxn_id} undisc={coverage['undiscovered_frac']:.3f} "
        f"scored={coverage['n_scored']} best={coverage['best_score']:.4f}"
    )

    round_no = 0
    try:
        while round_no < args.rounds:
            round_no += 1
            logger.info("\n" + "#" * 70)
            logger.info(
                f"ROUND {round_no} rxn:{rxn_id} @ "
                f"{datetime.now(timezone.utc).isoformat()}"
            )
            logger.info("#" * 70)

            if round_no == 1 or (
                args.refresh_historical_every > 0
                and round_no % args.refresh_historical_every == 0
            ):
                historical_df = await asyncio.to_thread(
                    load_historical_fingerprints, target
                )
                logger.info(
                    f"Refreshed historical: "
                    f"{0 if historical_df is None else len(historical_df)}"
                )
                # Refresh coverage/manager occasionally so pools stay current
                coverage = measure_rxn_coverage(rxn_id, config)

            await run_rxn_round(
                rxn_id=rxn_id,
                config=config,
                boltz=boltz,
                historical_df=historical_df,
                args=args,
                coverage=coverage,
                round_no=round_no,
            )

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())
