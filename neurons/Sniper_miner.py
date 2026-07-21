"""
miner.py — DPEX-DJA + Boltz, single fixed reaction (no multi-rxn).

Modified version:
  - Main progress / plateau / improvement signal is now the
    average score of the top 50 molecules.
  - Single best molecule score is still logged for reporting only.
  - Surrogate / Boltz / DPEX-DJA / exploit logic remains unchanged.

Pipeline per iteration:
  1. Generate n_base_samples * GENERATE_MULTIPLIER candidates
  2. Resolve SMILES
  3. Dedup against seen
  4. Surrogate filter if trained, otherwise hard-cap at BOLTZ_BUDGET
  5. Boltz score
  6. Update all_pool / top_pool
  7. Compute top-50 average score
  8. Use top-50 average for improvement and plateau logic
"""

import os
import sys
import time
import asyncio
import logging
import sqlite3
import argparse
import pandas as pd
import numpy as np
import bittensor as bt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

# ── These are set dynamically from --rxn_id argument in parse_args() ─────
RXN_ID = None
SCORE_RESULTS_DB = None
RXN_CSV = None

# ── local imports ─────────────────────────────────────────────────────────
from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    molecule_unique_for_protein_hf,
    contains_atom_type,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from molecules import MoleculeManager, MoleculeUtils
from tools import (
    IterationParams,
    SynthonLibrary,
    generate_valid_random_molecules,
    build_component_weights,
)
from exploit import get_top_n_unexploited, run_exploit
from dpex_dja import (
    DPEXDJAState,
    dja_generate,
    tabu_generate,
    update_tabu,
    dpex_exchange,
    update_populations,
    set_ranker_weights,
)

# ── BoltzWrapper lazy import ──────────────────────────────────────────────
BOLTZ_AVAILABLE = False
BoltzWrapper = None

# ── Surrogate pipeline constants ──────────────────────────────────────────
GENERATE_MULTIPLIER = 5

# Hard cap used only while surrogate is not ready.
BOLTZ_BUDGET = 600

# Surrogate keep ratio.
# NOTE: 0.10 means top 10%, not 20%.
SURROGATE_KEEP_RATIO = 0.30

# Surrogate activation threshold.
SURROGATE_MIN_TRAIN_SIZE = 4000

# ── Progress metric constants ─────────────────────────────────────────────
# Main optimization/progress signal is now average score of top 50 molecules.
TOP_AVG_K = 50

# ── fingerprint generators ────────────────────────────────────────────────
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)

_fp_cache = {}
_mol_cache = {}
_morgan_bv_cache = {}

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    global RXN_ID, SCORE_RESULTS_DB, RXN_CSV

    parser = argparse.ArgumentParser(
        description="DPEX-DJA Miner — single fixed reaction mode"
    )
    parser.add_argument(
        "--rxn_id",
        type=int,
        required=True,
        help="Reaction ID, e.g. 1-5.",
    )
    args = parser.parse_args()

    RXN_ID = args.rxn_id
    SCORE_RESULTS_DB = os.path.join(BASE_DIR, f"score_results_{RXN_ID}.sqlite")
    RXN_CSV = os.path.join(BASE_DIR, "data", f"rxn{RXN_ID}.csv")

    logger.info(f"✅ rxn_id            = {RXN_ID}")
    logger.info(f"✅ SCORE_RESULTS_DB  = {SCORE_RESULTS_DB}")
    logger.info(f"✅ RXN_CSV           = {RXN_CSV}")
    logger.info(
        f"✅ Pipeline = generate {GENERATE_MULTIPLIER}x "
        f"→ dedup → surrogate keep {SURROGATE_KEEP_RATIO * 100:.0f}% "
        f"once ≥{SURROGATE_MIN_TRAIN_SIZE} samples, else hard-cap "
        f"{BOLTZ_BUDGET} → Boltz"
    )
    logger.info(
        f"✅ Progress metric = average score of top {TOP_AVG_K} molecules"
    )

    return RXN_ID


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_mol(smiles: str):
    if smiles in _mol_cache:
        return _mol_cache[smiles]

    mol = Chem.MolFromSmiles(smiles)
    _mol_cache[smiles] = mol
    return mol


def get_morgan_fingerprint(smiles: str, n_bits: int = 2048):
    if smiles in _fp_cache:
        return _fp_cache[smiles]

    mol = get_mol(smiles)
    if mol is None:
        return None

    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    fp_array = np.zeros(n_bits, dtype=np.uint8)
    fp_array[fp.GetOnBits()] = 1

    _fp_cache[smiles] = fp_array

    if len(_fp_cache) > 50000:
        for k in list(_fp_cache.keys())[:12500]:
            del _fp_cache[k]

    return fp_array


def get_morgan_fp_bv(smiles: str):
    fp = _morgan_bv_cache.get(smiles)

    if fp is not None:
        return fp

    mol = get_mol(smiles)
    if mol is None:
        return None

    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    _morgan_bv_cache[smiles] = fp

    if len(_morgan_bv_cache) > 50000:
        for k in list(_morgan_bv_cache.keys())[:12500]:
            del _morgan_bv_cache[k]

    return fp


# ═══════════════════════════════════════════════════════════════════════════
# Tanimoto diversity selection
# ═══════════════════════════════════════════════════════════════════════════

def select_tanimoto_diverse(
    df: pd.DataFrame,
    n: int,
    threshold: float = 0.9,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.head(0)

    kept_indices = []
    kept_fps = []

    for idx, row in df.iterrows():
        smi = row.get(smiles_col)

        if not isinstance(smi, str) or not smi:
            continue

        fp = get_morgan_fp_bv(smi)

        if fp is None:
            continue

        if kept_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, kept_fps)
            if max(sims) >= threshold:
                continue

        kept_indices.append(idx)
        kept_fps.append(fp)

        if len(kept_indices) >= n:
            break

    return df.loc[kept_indices]


# ═══════════════════════════════════════════════════════════════════════════
# ComponentRanker
# ═══════════════════════════════════════════════════════════════════════════

class ComponentRanker:
    """EMA-based per-reactant quality ranker for single rxn."""

    def __init__(self, decay: float = 0.90):
        self.decay = decay
        self.q_A: dict = {}
        self.q_B: dict = {}
        self.q_C: dict = {}

    def _ema(self, store: dict, key: int, score: float):
        if key in store:
            old, cnt = store[key]
            store[key] = (
                self.decay * old + (1 - self.decay) * score,
                cnt + 1,
            )
        else:
            store[key] = (score, 1)

    def update(self, scored_df: pd.DataFrame):
        if scored_df.empty:
            return

        for _, row in scored_df.iterrows():
            score = row.get("score", 0.0)

            if pd.isna(score):
                continue

            parts = str(row["name"]).split(":")

            if len(parts) < 4:
                continue

            try:
                A = int(parts[2])
                B = int(parts[3])
                C = int(parts[4]) if len(parts) > 4 else None
            except (ValueError, IndexError):
                continue

            self._ema(self.q_A, A, score)
            self._ema(self.q_B, B, score)

            if C is not None:
                self._ema(self.q_C, C, score)

    def warm_start_decay(self, n_historical_rounds: int = 50):
        decay_factor = self.decay ** n_historical_rounds

        for q in [self.q_A, self.q_B, self.q_C]:
            for k in q:
                val, cnt = q[k]
                q[k] = (val * decay_factor, cnt)

    def compute_weights(self, pool: list, q: dict) -> Optional[np.ndarray]:
        if not q:
            return None

        w = np.array([
            max(0.01, q[mid][0]) if mid in q else 0.05
            for mid in pool
        ])

        w /= w.sum()
        return w

    def push_to_dja(self, manager: MoleculeManager):
        w_A = self.compute_weights(manager.moles_A_id, self.q_A)
        w_B = self.compute_weights(manager.moles_B_id, self.q_B)
        w_C = (
            self.compute_weights(manager.moles_C_id, self.q_C)
            if manager.is_three_component
            else None
        )

        set_ranker_weights(w_A, w_B, w_C, rxn_id=RXN_ID)
        set_ranker_weights(w_A, w_B, w_C, rxn_id=None)

    def blend_component_weights(
        self,
        component_weights: dict,
        manager: MoleculeManager,
    ) -> dict:
        if not component_weights:
            return component_weights

        blended = dict(component_weights)

        for role, pool, q in [
            ("A", manager.moles_A_id, self.q_A),
            ("B", manager.moles_B_id, self.q_B),
        ]:
            if role not in blended or not q:
                continue

            orig = blended[role]
            new_w = {}

            for mid in pool:
                o = orig.get(mid, 0.05)
                e = max(0.01, q[mid][0]) if mid in q else 0.05
                new_w[mid] = 0.6 * o + 0.4 * e

            total = sum(new_w.values())

            if total > 0:
                new_w = {k: v / total for k, v in new_w.items()}

            blended[role] = new_w

        return blended


# ═══════════════════════════════════════════════════════════════════════════
# SurrogateModel
# ═══════════════════════════════════════════════════════════════════════════

class SurrogateModel:
    """
    Random Forest surrogate — rxn-specific only, balanced sampling.

    The surrogate logic is unchanged.
    The main algorithmic change is outside this class:
    optimization progress now uses top-50 average score.
    """

    def __init__(self, max_training_samples: int = 5000):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,
        )

        self.is_trained = False
        self.anchor_X: list = []
        self.anchor_y: list = []
        self.X_train: list = []
        self.y_train: list = []
        self.min_train_size = SURROGATE_MIN_TRAIN_SIZE
        self.max_training_samples = max_training_samples
        self.last_train_iteration = 0
        self.train_interval = 1
        self.enabled = True

    def _safe_fp(self, smiles: str) -> np.ndarray:
        fp = get_morgan_fingerprint(smiles)
        return fp if fp is not None else np.zeros(2048, dtype=np.uint8)

    def add_anchor_data(self, smiles_list: list, scores: list):
        if not smiles_list:
            return

        scores_arr = np.array(scores, dtype=float)

        finite_mask = np.isfinite(scores_arr)
        n_dropped = int((~finite_mask).sum())

        if n_dropped:
            logger.warning(
                f"[SURROGATE] add_anchor_data: dropping {n_dropped} "
                f"non-finite score(s)"
            )

        if not finite_mask.all():
            smiles_list = [
                s for s, ok in zip(smiles_list, finite_mask) if ok
            ]
            scores_arr = scores_arr[finite_mask]

        n = len(scores_arr)

        if n == 0:
            logger.warning(
                "[SURROGATE] add_anchor_data: no finite scores left"
            )
            return

        n_top = max(1, n // 3)
        n_bottom = max(1, n // 3)
        n_rand = max(1, n // 10)

        top_idx = set(np.argsort(scores_arr)[-n_top:].tolist())
        bottom_idx = set(np.argsort(scores_arr)[:n_bottom].tolist())

        middle = list(set(range(n)) - top_idx - bottom_idx)

        rand_idx = (
            set(
                np.random.choice(
                    middle,
                    min(n_rand, len(middle)),
                    replace=False,
                ).tolist()
            )
            if middle
            else set()
        )

        keep = sorted(top_idx | bottom_idx | rand_idx)

        added = 0

        for i in keep:
            fp = get_morgan_fingerprint(smiles_list[i])

            if fp is not None:
                self.anchor_X.append(fp)
                self.anchor_y.append(float(scores_arr[i]))
                added += 1

        logger.info(
            f"[SURROGATE] Anchored {added} rxn-specific samples "
            f"(top={n_top} bottom={n_bottom} rand={n_rand} "
            f"from {n} finite total, {n_dropped} dropped)"
        )

    def add_training_data(self, smiles_list: list, scores: list):
        if not self.enabled:
            return

        if not smiles_list:
            return

        scores_arr = np.array(scores, dtype=float)

        finite_mask = np.isfinite(scores_arr)
        n_dropped = int((~finite_mask).sum())

        if n_dropped:
            logger.warning(
                f"[SURROGATE] add_training_data: dropping {n_dropped} "
                f"non-finite score(s)"
            )

        if not finite_mask.all():
            smiles_list = [
                s for s, ok in zip(smiles_list, finite_mask) if ok
            ]
            scores_arr = scores_arr[finite_mask]

        if len(scores_arr) == 0:
            return

        scores = scores_arr.tolist()

        if len(smiles_list) > 200:
            scores_array = np.array(scores)
            n = len(scores_array)

            n_top = min(120, n // 3)
            n_bot = min(60, n // 3)

            top_idx = list(np.argsort(scores_array)[-n_top:])
            bot_idx = list(np.argsort(scores_array)[:n_bot])

            keep_indices = sorted(set(top_idx + bot_idx))

            smiles_list = [smiles_list[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]

        for smiles, score in zip(smiles_list, scores):
            fp = get_morgan_fingerprint(smiles)

            if fp is not None:
                self.X_train.append(fp)
                self.y_train.append(score)

        if len(self.X_train) > self.max_training_samples:
            scores_array = np.array(self.y_train)

            top_count = int(self.max_training_samples * 0.40)
            bot_count = int(self.max_training_samples * 0.20)
            recent_count = int(self.max_training_samples * 0.40)

            top_indices = list(np.argsort(scores_array)[-top_count:])
            bot_indices = list(np.argsort(scores_array)[:bot_count])
            recent_indices = list(
                range(len(self.X_train) - recent_count, len(self.X_train))
            )

            keep_indices = sorted(
                set(top_indices + bot_indices + recent_indices)
            )

            self.X_train = [self.X_train[i] for i in keep_indices]
            self.y_train = [self.y_train[i] for i in keep_indices]

    def train(self, iteration: int = 0):
        if self.is_trained and (
            iteration - self.last_train_iteration
        ) < self.train_interval:
            return

        X_all = self.anchor_X + self.X_train
        y_all = self.anchor_y + self.y_train

        if len(X_all) < self.min_train_size:
            self.is_trained = False
            return

        try:
            t0 = time.time()

            self.model.fit(np.array(X_all), np.array(y_all))

            self.is_trained = True
            self.last_train_iteration = iteration

            logger.info(
                f"[SURROGATE] Trained in {time.time() - t0:.2f}s on "
                f"{len(self.anchor_X)} anchors + {len(self.X_train)} live "
                f"= {len(X_all)} total | rxn={RXN_ID}"
            )

        except Exception as e:
            logger.warning(f"Surrogate training failed: {e}")
            self.is_trained = False

    def predict(self, smiles_list: list) -> np.ndarray:
        if not self.is_trained:
            return np.array([0.0] * len(smiles_list))

        try:
            fps = [self._safe_fp(s) for s in smiles_list]
            return self.model.predict(np.array(fps))

        except Exception as e:
            logger.warning(f"Surrogate prediction failed: {e}")
            return np.array([0.0] * len(smiles_list))

    def filter_candidates(
        self,
        data: pd.DataFrame,
        keep_ratio: float = SURROGATE_KEEP_RATIO,
        smiles_col: str = "smiles",
        min_keep: int = 1,
    ) -> pd.DataFrame:
        if not self.is_trained or data.empty:
            return data

        n_keep = max(min_keep, int(round(len(data) * keep_ratio)))

        if n_keep >= len(data):
            return data

        pred = self.predict(data[smiles_col].tolist())

        data = data.copy()
        data["_pred"] = pred

        filtered = (
            data.sort_values("_pred", ascending=False)
            .head(n_keep)
            .drop(columns=["_pred"])
        )

        logger.info(
            f"[SURROGATE] {len(data)} → {len(filtered)} "
            f"(kept top {keep_ratio * 100:.0f}%, "
            f"dropped {len(data) - len(filtered)} before Boltz)"
        )

        return filtered.reset_index(drop=True)

    @property
    def total_train_size(self) -> int:
        return len(self.anchor_X) + len(self.X_train)
# ═══════════════════════════════════════════════════════════════════════════
# Molecule validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def validate_molecule_smiles(molecule_name: str, smiles: str) -> Tuple[bool, str]:
    if not smiles:
        return False, "No SMILES provided"

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return False, f"RDKit cannot parse SMILES: {smiles}"

        return True, ""

    except Exception as e:
        return False, f"RDKit parsing error: {str(e)}"


def validate_molecule_heavy_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any],
) -> Tuple[bool, str]:
    try:
        count = get_heavy_atom_count(smiles)
        min_atoms = 10
        max_atoms = 40

        if count < min_atoms:
            return False, f"Insufficient heavy atoms: {count} < {min_atoms}"

        if count > max_atoms:
            return False, f"Too many heavy atoms: {count} > {max_atoms}"

        return True, ""

    except Exception as e:
        return False, f"Heavy atom count error: {str(e)}"


def validate_molecule_banned_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any],
) -> Tuple[bool, str]:
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


def validate_molecule_rotatable_bonds(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any],
) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return False, "Cannot parse SMILES for rotatable bonds check"

        n_rot = Descriptors.NumRotatableBonds(mol)
        min_bonds = config["min_rotatable_bonds"]
        max_bonds = config["max_rotatable_bonds"]

        if n_rot < min_bonds or n_rot > max_bonds:
            return False, (
                f"Rotatable bonds out of range: {n_rot} "
                f"(expected {min_bonds}-{max_bonds})"
            )

        return True, ""

    except Exception as e:
        return False, f"Rotatable bonds check error: {str(e)}"


async def validate_molecule_huggingface_unique(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str,
) -> Tuple[bool, str]:
    if not state.get("current_challenge_targets"):
        return False, "No target proteins available"

    primary_target = state["current_challenge_targets"][0]

    try:
        if not molecule_unique_for_protein_hf(primary_target, smiles):
            return False, f"Already in HuggingFace for {primary_target}"

        return True, ""

    except Exception as e:
        return False, f"HuggingFace uniqueness check error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# Score results DB helpers
# ═══════════════════════════════════════════════════════════════════════════

def init_score_results_db(db_path: str = None) -> None:
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scored_molecules (
                molecule_name TEXT PRIMARY KEY,
                score         REAL NOT NULL,
                scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available     BOOLEAN DEFAULT TRUE
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)"
        )

        conn.commit()
        conn.close()

        logger.info(f"✅ Score DB ready: {db_path}")

    except Exception as e:
        logger.error(f"❌ Error initializing score DB: {e}")


def write_scores_to_db(
    molecules: List[Dict[str, Any]],
    db_path: str = None,
) -> None:
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not molecules:
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        skipped_non_finite = 0
        to_insert = []

        for m in molecules:
            name = m.get("name")
            score = m.get("boltz_score")

            if not name or score is None:
                continue

            try:
                score_f = float(score)
            except (TypeError, ValueError):
                skipped_non_finite += 1
                continue

            if not np.isfinite(score_f):
                skipped_non_finite += 1
                continue

            to_insert.append((name, score_f, True))

        if skipped_non_finite:
            logger.warning(
                f"⚠️ write_scores_to_db: skipped {skipped_non_finite} "
                f"non-finite score(s)"
            )

        if to_insert:
            cursor.executemany(
                "INSERT OR REPLACE INTO scored_molecules "
                "(molecule_name, score, available) VALUES (?, ?, ?)",
                to_insert,
            )
            conn.commit()

            logger.info(f"✅ Wrote {len(to_insert)} scores → {db_path}")

        conn.close()

    except Exception as e:
        logger.error(f"❌ Error writing scores: {e}")


def batch_get_scores_from_db(
    molecule_names: List[str],
    db_path: str = None,
) -> Dict[str, float]:
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not molecule_names:
        return {}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(molecule_names))

        cursor.execute(
            f"SELECT molecule_name, score FROM scored_molecules "
            f"WHERE molecule_name IN ({placeholders})",
            molecule_names,
        )

        results = cursor.fetchall()
        conn.close()

        return {name: float(score) for name, score in results}

    except Exception as e:
        logger.debug(f"Error batch getting scores: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_molecules_from_csv(csv_path: str, rxn_id: int) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logger.warning(f"⚠️ RXN CSV not found: {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]

        if "final_score" in df.columns and "score" not in df.columns:
            df = df.rename(columns={"final_score": "score"})

        df["molecule_name"] = (
            df["molecule_name"]
            .astype(str)
            .str.strip()
            .str.lstrip("\ufeff")
        )

        prefix = f"rxn:{rxn_id}:"

        df = df[
            df["molecule_name"].str.startswith(prefix, na=False)
        ].reset_index(drop=True)

        if df.empty:
            logger.warning(
                f"⚠️ {os.path.basename(csv_path)}: "
                f"no rows matching prefix '{prefix}'"
            )
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

        df["smiles"] = df["molecule_name"].apply(
            MoleculeUtils.get_smiles_from_reaction_cached
        )

        df = df[df["smiles"].notna() & (df["smiles"] != "")]

        df["InChIKey"] = df["smiles"].apply(
            MoleculeUtils.generate_inchikey
        )

        df = df[df["InChIKey"].notna() & (df["InChIKey"] != "")]

        result = df[
            ["molecule_name", "smiles", "InChIKey", "score"]
        ].copy()

        result = result.rename(columns={"molecule_name": "name"})

        result["score"] = pd.to_numeric(result["score"], errors="coerce")
        result.loc[~np.isfinite(result["score"]), "score"] = np.nan

        result = result.drop_duplicates(
            subset=["InChIKey"],
            keep="first",
        )

        result = result.sort_values(
            by="score",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        logger.info(
            f"✅ CSV [{os.path.basename(csv_path)}]: "
            f"{len(result)} molecules loaded"
        )

        return result

    except Exception as e:
        logger.error(f"Error loading CSV {csv_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_db(db_path: str, rxn_id: int) -> pd.DataFrame:
    if not os.path.exists(db_path):
        logger.info(f"ℹ️ Score DB not found yet: {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT molecule_name, score FROM scored_molecules "
            "WHERE molecule_name LIKE ?",
            (f"rxn:{rxn_id}:%",),
        )

        db_results = cursor.fetchall()
        conn.close()

        if not db_results:
            logger.info(f"ℹ️ Score DB empty for rxn={rxn_id}: {db_path}")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

        rows = []
        fail = 0
        non_finite = 0

        for mol_name, score in db_results:
            try:
                score_f = None

                if score is not None:
                    score_f = float(score)

                    if not np.isfinite(score_f):
                        non_finite += 1
                        continue

                smiles = MoleculeUtils.get_smiles_from_reaction_cached(
                    mol_name
                )

                if not smiles:
                    fail += 1
                    continue

                inchikey = MoleculeUtils.generate_inchikey(smiles)

                if not inchikey:
                    fail += 1
                    continue

                rows.append({
                    "name": mol_name,
                    "smiles": smiles,
                    "InChIKey": inchikey,
                    "score": score_f,
                })

            except Exception as e:
                logger.debug(f"Could not process {mol_name}: {e}")
                fail += 1

        result = pd.DataFrame(rows)

        if not result.empty:
            result = result.drop_duplicates(
                subset=["InChIKey"],
                keep="first",
            )

            result = result.sort_values(
                by="score",
                ascending=False,
                na_position="last",
            ).reset_index(drop=True)

        if non_finite:
            logger.warning(
                f"⚠️ DB [{os.path.basename(db_path)}]: "
                f"dropped {non_finite} non-finite score rows"
            )

        logger.info(
            f"✅ DB [{os.path.basename(db_path)}]: "
            f"{len(result)} molecules loaded "
            f"(fail={fail}, non_finite={non_finite})"
        )

        return result

    except Exception as e:
        logger.error(f"Error loading DB {db_path}: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(
    rxn_id: int,
    config: Dict[str, Any] = None,
) -> pd.DataFrame:
    csv_path = os.path.join(BASE_DIR, "data", f"rxn{rxn_id}.csv")
    db_path = os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")

    logger.info(
        f"🔄 Warm-seed load rxn={rxn_id}: "
        f"{os.path.basename(csv_path)} + {os.path.basename(db_path)}"
    )

    csv_df = load_molecules_from_csv(csv_path, rxn_id)
    db_df = load_molecules_from_db(db_path, rxn_id)

    if csv_df.empty and db_df.empty:
        logger.warning(f"⚠️ No warm-seed data for rxn={rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    if csv_df.empty:
        return db_df

    if db_df.empty:
        return csv_df

    combined = pd.concat([csv_df, db_df], ignore_index=True)

    combined = (
        combined
        .sort_values(
            by="score",
            ascending=False,
            na_position="last",
        )
        .drop_duplicates(subset=["InChIKey"], keep="first")
    )

    logger.info(
        f"✅ Warm-seed rxn={rxn_id}: "
        f"{len(csv_df)} CSV + {len(db_df)} DB = {len(combined)} unique"
    )

    return combined


def load_training_csv_for_surrogate(rxn_id: int) -> pd.DataFrame:
    csv_path = RXN_CSV

    if not os.path.exists(csv_path):
        logger.warning(f"⚠️ Training CSV not found: {csv_path}")
        return pd.DataFrame(columns=["smiles", "score"])

    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]

        if "final_score" in df.columns and "score" not in df.columns:
            df = df.rename(columns={"final_score": "score"})

        df["molecule_name"] = (
            df["molecule_name"]
            .astype(str)
            .str.strip()
            .str.lstrip("\ufeff")
        )

        prefix = f"rxn:{rxn_id}:"

        df = df[
            df["molecule_name"].str.startswith(prefix, na=False)
        ].reset_index(drop=True)

        if df.empty:
            logger.warning(
                f"⚠️ rxn{rxn_id}.csv: no rows for prefix '{prefix}'"
            )
            return pd.DataFrame(columns=["smiles", "score"])

        df = df[df["score"].notna()].reset_index(drop=True)

        df["smiles"] = df["molecule_name"].apply(
            MoleculeUtils.get_smiles_from_reaction_cached
        )

        df = df[df["smiles"].notna() & (df["smiles"] != "")]

        result = df[["smiles", "score"]].copy()

        result["score"] = pd.to_numeric(result["score"], errors="coerce")
        result.loc[~np.isfinite(result["score"]), "score"] = np.nan
        result = result[result["score"].notna()]

        result = result.drop_duplicates(subset=["smiles"]).reset_index(
            drop=True
        )

        logger.info(
            f"✅ Surrogate training [rxn{rxn_id}.csv]: "
            f"{len(result)} molecules loaded"
        )

        return result

    except Exception as e:
        logger.error(f"Error loading training CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["smiles", "score"])


# ═══════════════════════════════════════════════════════════════════════════
# Progress helpers — TOP-50 average based
# ═══════════════════════════════════════════════════════════════════════════

def _top_pool_stats(
    top_pool: pd.DataFrame,
    num_molecules: int = TOP_AVG_K,
) -> Tuple[float, float, Optional[str]]:
    """
    Return progress statistics for the current molecule pool.

    Primary optimization/progress metric:
        average score of the top TOP_AVG_K molecules.

    Returns:
        topk_avg:
            Mean score of top TOP_AVG_K molecules.
            If fewer than TOP_AVG_K valid molecules exist, use all valid ones.

        pool_max:
            Best single-molecule score.
            Reporting only.

        best_name:
            Name of the best single molecule.
            Reporting only.
    """
    if top_pool.empty or "score" not in top_pool.columns:
        return 0.0, 0.0, None

    pool = top_pool.copy()

    pool["score"] = pd.to_numeric(pool["score"], errors="coerce")
    pool = pool[np.isfinite(pool["score"])]
    pool = pool.dropna(subset=["score"])

    if pool.empty:
        return 0.0, 0.0, None

    pool = pool.sort_values(
        by="score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    k = min(TOP_AVG_K, len(pool))

    topk_avg = float(pool.head(k)["score"].mean())
    pool_max = float(pool["score"].max())
    best_name = str(pool.iloc[0].get("name", "")) or None

    return topk_avg, pool_max, best_name


def _iteration_mode_str(
    exploited_status: bool,
    dpex: DPEXDJAState,
    params: IterationParams,
    early_exploit_used: bool = False,
    exploit_attempted: bool = False,
) -> str:
    """Describe which generation strategy ran this iteration."""
    if exploited_status:
        return "EXPLOIT"

    if not dpex.pop_A:
        return "INIT(cold)"

    base = "DJA+TABU" if params.synthon_lib is not None else "DJA"

    if exploit_attempted:
        base = f"EXPLOIT(failed)→{base}"

    if early_exploit_used:
        base = f"{base}+early-exploit"

    return base


def _log_pool_progress(
    iteration: int,
    top50_avg: float,
    pool_max: float,
    best_name: Optional[str],
    prev_top50_avg: Optional[float],
    prev_max: Optional[float],
    best_top50_avg_ever: float,
    score_improvement_rate: float,
    num_molecules: int,
    mode: Optional[str] = None,
) -> float:
    """
    Log top-pool progress metrics.

    The main progress metric is top-50 average score.
    Single best score is retained for reporting only.
    """
    if top50_avg > best_top50_avg_ever:
        best_top50_avg_ever = top50_avg

    avg_delta = (
        top50_avg - prev_top50_avg
        if prev_top50_avg is not None
        else None
    )

    max_delta = (
        pool_max - prev_max
        if prev_max is not None
        else None
    )

    lines = [
        "[PoolProgress] "
        + (f"iter={iteration}" if iteration > 0 else "warm-start"),
    ]

    if mode:
        lines.append(f"  mode                    : {mode}")

    lines.extend([
        f"  top-{TOP_AVG_K} avg          : {top50_avg:.6f}",
        f"  best single score       : {pool_max:.6f}",
    ])

    if best_name:
        lines.append(f"  best molecule           : {best_name}")

    if avg_delta is not None:
        lines.append(
            f"  top-{TOP_AVG_K} avg Δ        : {avg_delta:+.6f} "
            f"({score_improvement_rate:+.2%} rel)"
        )

    if max_delta is not None:
        lines.append(f"  best single Δ           : {max_delta:+.6f}")

    lines.append(
        f"  all-time top-{TOP_AVG_K} avg : "
        f"{best_top50_avg_ever:.6f}"
    )

    logger.info("\n".join(lines))

    return best_top50_avg_ever
# ═══════════════════════════════════════════════════════════════════════════
# WARM START
# ═══════════════════════════════════════════════════════════════════════════

def warm_start(
    state: Dict[str, Any],
    dpex: DPEXDJAState,
    ranker: ComponentRanker,
    surrogate: SurrogateModel,
    params: IterationParams,
    top_pool: pd.DataFrame,
    all_pool: pd.DataFrame,
    num_molecules: int,
    tanimoto_max_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = state["config"]

    logger.info(
        f"\n{'=' * 60}\n"
        f"[WarmStart] rxn={RXN_ID} | "
        f"pop-seed : rxn{RXN_ID}.csv + score_results_{RXN_ID}.sqlite\n"
        f"[WarmStart] surrogate: rxn{RXN_ID}.csv + DB data\n"
        f"[WarmStart] surrogate activates once total training data "
        f">= {SURROGATE_MIN_TRAIN_SIZE}\n"
        f"[WarmStart] progress metric: top-{TOP_AVG_K} average score\n"
        f"{'=' * 60}"
    )

    loaded_df = load_molecules_combined(RXN_ID, config)

    if loaded_df.empty:
        logger.warning("[WarmStart] No rxn-specific data — cold start")

    else:
        logger.info(
            f"[WarmStart] Rxn-specific: {len(loaded_df)} molecules loaded"
        )

        all_pool = loaded_df.rename(columns={"InChIKey": "inchi"}).copy()

        ranker.update(loaded_df)
        ranker.warm_start_decay(n_historical_rounds=50)

        logger.info(
            f"[WarmStart] ComponentRanker: "
            f"{len(ranker.q_A)} A | {len(ranker.q_B)} B "
            f"(EMA decayed 50 rounds)"
        )

        top_for_A = loaded_df.head(dpex.N_A)

        dpex.pop_A = top_for_A.rename(
            columns={"InChIKey": "inchi"}
        ).to_dict("records")

        logger.info(
            f"[WarmStart] pop_A: {len(dpex.pop_A)} | "
            f"best={top_for_A['score'].max():.6f} | "
            f"avg={top_for_A['score'].mean():.6f}"
        )

        diverse_elites = select_tanimoto_diverse(
            loaded_df.reset_index(drop=True),
            n=dpex.N_B,
            threshold=0.85,
            smiles_col="smiles",
        )

        dpex.pop_B = diverse_elites.rename(
            columns={"InChIKey": "inchi"}
        ).to_dict("records")

        logger.info(
            f"[WarmStart] pop_B: {len(dpex.pop_B)} diverse elites"
        )

        params.seen_molecules = set(loaded_df["name"].tolist())

        logger.info(
            f"[WarmStart] seen_molecules: {len(params.seen_molecules)}"
        )

        top_pool = select_tanimoto_diverse(
            all_pool.reset_index(drop=True),
            n=num_molecules + 50,
            threshold=tanimoto_max_threshold,
            smiles_col="smiles",
        ).reset_index(drop=True)

    # ── Surrogate anchors ─────────────────────────────────────────────
    logger.info(
        f"[WarmStart] Loading surrogate anchors "
        f"(rxn={RXN_ID} specific only)..."
    )

    if not loaded_df.empty:
        surrogate.add_anchor_data(
            loaded_df["smiles"].tolist(),
            loaded_df["score"].tolist(),
        )

        logger.info(
            f"[WarmStart] Surrogate anchors from loaded_df: "
            f"{len(loaded_df)} molecules "
            f"(total anchors: {len(surrogate.anchor_X)})"
        )

    training_df = load_training_csv_for_surrogate(RXN_ID)

    if not training_df.empty:
        surrogate.add_anchor_data(
            training_df["smiles"].tolist(),
            training_df["score"].tolist(),
        )

        logger.info(
            f"[WarmStart] Surrogate anchors from rxn{RXN_ID}.csv: "
            f"+{len(training_df)} molecules "
            f"(total anchors: {len(surrogate.anchor_X)})"
        )

    if surrogate.total_train_size < surrogate.min_train_size:
        logger.warning(
            f"[WarmStart] Insufficient rxn-specific data for surrogate "
            f"({surrogate.total_train_size} < {surrogate.min_train_size}) "
            f"— hard-cap at {BOLTZ_BUDGET} until enough live Boltz scores"
        )

    if surrogate.total_train_size >= surrogate.min_train_size:
        surrogate.train(iteration=0)

        logger.info(
            f"[WarmStart] Surrogate pre-trained: "
            f"{surrogate.total_train_size} samples | "
            f"trained={surrogate.is_trained}"
        )

    if not loaded_df.empty:
        scores = loaded_df["score"].dropna()

        ws_top50_avg, ws_max, ws_best = _top_pool_stats(
            top_pool,
            TOP_AVG_K,
        )

        logger.info(
            f"\n[WarmStart] ✅ Complete!\n"
            f"  Score range        : {scores.min():.6f} → {scores.max():.6f}\n"
            f"  Score mean         : {scores.mean():.6f}\n"
            f"  top_pool           : {len(top_pool)}\n"
            f"  top-{TOP_AVG_K} avg     : {ws_top50_avg:.6f}\n"
            f"  best single score  : {ws_max:.6f}\n"
            f"  all_pool           : {len(all_pool)}\n"
            f"  pop_A              : {len(dpex.pop_A)}\n"
            f"  pop_B              : {len(dpex.pop_B)}\n"
            f"  seen               : {len(params.seen_molecules)}\n"
            f"  Surrogate          : trained={surrogate.is_trained} "
            f"total_train_size={surrogate.total_train_size} "
            f"(min={surrogate.min_train_size})\n"
            f"  Pipeline           : generate {GENERATE_MULTIPLIER}x "
            f"→ dedup → surrogate keep {SURROGATE_KEEP_RATIO * 100:.0f}% "
            f"if trained, else hard-cap {BOLTZ_BUDGET} → Boltz\n"
            f"  Progress metric    : top-{TOP_AVG_K} average score\n"
        )

        _log_pool_progress(
            0,
            ws_top50_avg,
            ws_max,
            ws_best,
            None,
            None,
            ws_top50_avg,
            0.0,
            num_molecules,
            mode="WARM-START",
        )

    else:
        logger.info(
            f"\n[WarmStart] ⚠️ Cold start.\n"
            f"  Surrogate: trained={surrogate.is_trained} "
            f"total_train_size={surrogate.total_train_size}\n"
            f"  Progress metric: top-{TOP_AVG_K} average score\n"
        )

    return top_pool, all_pool


# ═══════════════════════════════════════════════════════════════════════════
# BoltzWrapper import + scoring
# ═══════════════════════════════════════════════════════════════════════════

def _import_boltz_wrapper():
    """Import BoltzWrapper following DataGenerator pattern."""
    global BOLTZ_AVAILABLE, BoltzWrapper

    try:
        BOLTZ_SRC_DIR = os.path.join(BASE_DIR, "boltz")

        if BOLTZ_SRC_DIR not in sys.path:
            sys.path.insert(0, BOLTZ_SRC_DIR)

        from boltz_wrapper import BoltzWrapper as BW

        BoltzWrapper = BW
        BOLTZ_AVAILABLE = True

        logger.info("✅ BoltzWrapper imported successfully")
        return True

    except ImportError as e:
        logger.warning(f"⚠️ Failed to import BoltzWrapper: {e}")
        return False

    except Exception as e:
        logger.warning(f"⚠️ Error setting up BoltzWrapper: {e}")
        return False


async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Score molecules using BoltzWrapper in batches.
    """
    if state.get("boltz_wrapper") is None:
        logger.warning("BoltzWrapper not available, skipping scoring")
        return molecules

    if not molecules:
        return molecules

    logger.info(
        f"🔬 Processing {len(molecules)} molecules for scoring "
        f"in batches of {batch_size}..."
    )

    init_score_results_db()

    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]

        logger.info(
            f"📦 Batch {batch_idx + 1}/{total_batches}: "
            f"Scoring {len(batch)} molecules"
        )

        molecules_to_score = []
        molecules_with_db_scores = []
        molecules_in_hf = []

        target_proteins = state.get("current_challenge_targets", [])
        primary_target = target_proteins[0] if target_proteins else None

        molecule_names = [mol["name"] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)

        logger.info(
            f"   Found {len(db_scores)} molecules already in database"
        )

        for mol in batch:
            molecule_name = mol["name"]
            smiles = mol.get("smiles")

            if molecule_name in db_scores:
                mol["boltz_score"] = db_scores[molecule_name]
                mol["boltz_score_source"] = "database"
                molecules_with_db_scores.append(mol)

                logger.debug(
                    f"   ✓ {molecule_name}: score from DB = "
                    f"{db_scores[molecule_name]:.6f}"
                )

                continue

            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(
                        primary_target,
                        smiles,
                    )

                    if not is_unique_hf:
                        logger.debug(
                            f"   ⏭️ {molecule_name}: already in HuggingFace"
                        )
                        molecules_in_hf.append(mol)
                        continue

                except Exception as e:
                    logger.debug(
                        f"   Error checking HuggingFace for "
                        f"{molecule_name}: {e}"
                    )

            molecules_to_score.append(mol)

        logger.info(
            f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
            f"{len(molecules_in_hf)} in HuggingFace skipped, "
            f"{len(molecules_to_score)} need scoring"
        )

        newly_scored_molecules = []

        if molecules_to_score:
            logger.info(
                f"   Scoring {len(molecules_to_score)} new molecules "
                f"with Boltz..."
            )

            boltz = state["boltz_wrapper"]
            config = state["config"]

            target_proteins = state.get("current_challenge_targets", [])
            antitarget_proteins = state.get(
                "current_challenge_antitargets",
                [],
            )

            if not target_proteins:
                logger.warning("No target proteins available for scoring")
                all_scored_molecules.extend(molecules_with_db_scores)
                continue

            primary_target = target_proteins[0]

            try:
                output_dir = os.path.join(
                    boltz.output_dir,
                    "boltz_results_inputs",
                )

                if os.path.exists(output_dir):
                    try:
                        lightning_logs_dir = os.path.join(
                            output_dir,
                            "lightning_logs",
                        )

                        if os.path.exists(lightning_logs_dir):
                            import shutil

                            shutil.rmtree(
                                lightning_logs_dir,
                                ignore_errors=True,
                            )

                            logger.debug(
                                "Cleaned up old lightning_logs directory"
                            )

                    except Exception as cleanup_err:
                        logger.debug(
                            f"Could not clean up old logs: {cleanup_err}"
                        )

                processed_dir = os.path.join(output_dir, "processed")
                structures_dir = os.path.join(processed_dir, "structures")
                records_dir = os.path.join(processed_dir, "records")
                msa_dir = os.path.join(processed_dir, "msa")
                predictions_dir = os.path.join(output_dir, "predictions")

                os.makedirs(structures_dir, exist_ok=True)
                os.makedirs(records_dir, exist_ok=True)
                os.makedirs(msa_dir, exist_ok=True)
                os.makedirs(predictions_dir, exist_ok=True)

                valid_molecules_by_uid = {
                    0: {
                        "smiles": [
                            mol["smiles"] for mol in molecules_to_score
                        ],
                        "names": [
                            mol["name"] for mol in molecules_to_score
                        ],
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
                    "small_molecule_target": config[
                        "small_molecule_target"
                    ],
                    "small_molecule_target_clip_interval": config[
                        "small_molecule_target_clip_interval"
                    ],
                    "boltz_mode": getattr(config, "boltz_mode", "max"),
                    "boltz_metric": getattr(
                        config,
                        "boltz_metric",
                        [
                            "affinity_probability_binary",
                            "affinity_pred_value",
                        ],
                    ),
                    "combination_strategy": getattr(
                        config,
                        "combination_strategy",
                        "heavy_atom_normalization",
                    ),
                }

                logger.info(
                    f"   Running Boltz scoring for "
                    f"{len(molecules_to_score)} molecules..."
                )

                start_time = time.time()

                def run_scoring():
                    boltz.score_molecules(
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config,
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, run_scoring)

                elapsed = time.time() - start_time

                logger.info(
                    f"   ✅ Boltz scoring completed in {elapsed:.2f}s"
                )

                uid = 0
                smiles_to_score = {}

                final_scores = getattr(
                    boltz,
                    "final_boltz_scores",
                    {},
                ).get(uid, {})

                if primary_target and primary_target in final_scores:
                    smiles_to_score = final_scores[primary_target].copy()

                elif final_scores:
                    smiles_to_score = next(iter(final_scores.values())).copy()

                elif (
                    hasattr(boltz, "per_molecule_metric")
                    and uid in boltz.per_molecule_metric
                ):
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()

                if smiles_to_score:
                    logger.info(
                        f"   ✅ Loaded {len(smiles_to_score)} "
                        f"unique SMILES scores"
                    )

                target_scores_list = None
                target_scores = score_dict[uid].get("target_scores", [[]])

                if target_scores and len(target_scores[0]) > 0:
                    target_scores_list = (
                        target_scores[0]
                        if isinstance(target_scores[0], list)
                        else [target_scores[0]]
                    )

                avg_score = None

                if not smiles_to_score and not target_scores_list:
                    avg_score = score_dict[uid].get("boltz_score")

                for mol_idx, mol in enumerate(molecules_to_score):
                    smiles = mol["smiles"]
                    score = None

                    if smiles in smiles_to_score:
                        score = smiles_to_score[smiles]

                    elif target_scores_list and mol_idx < len(
                        target_scores_list
                    ):
                        score = target_scores_list[mol_idx]

                    elif target_scores_list:
                        try:
                            valid_idx = valid_molecules_by_uid[uid][
                                "smiles"
                            ].index(smiles)

                            if valid_idx < len(target_scores_list):
                                score = target_scores_list[valid_idx]

                        except (ValueError, IndexError):
                            pass

                    if score is None and avg_score is not None:
                        score = avg_score

                    mol["boltz_score"] = score

                    if score is not None:
                        newly_scored_molecules.append(mol)

                if newly_scored_molecules:
                    for mol in newly_scored_molecules:
                        logger.debug(
                            f"Molecule {mol['name']} scored "
                            f"{mol['boltz_score']}"
                        )

                    write_scores_to_db(newly_scored_molecules)

            except Exception as e:
                logger.error(f"❌ Error scoring batch with Boltz: {e}")
                import traceback

                logger.error(traceback.format_exc())

        batch_results = molecules_with_db_scores + newly_scored_molecules

        for mol in molecules_in_hf:
            mol["boltz_score"] = None
            mol["boltz_score_source"] = "huggingface_skipped"
            batch_results.append(mol)

        all_scored_molecules.extend(batch_results)

        logger.info(
            f"   ✅ Batch {batch_idx + 1} complete: "
            f"{len(molecules_with_db_scores)} from DB, "
            f"{len(newly_scored_molecules)} newly scored, "
            f"{len(molecules_in_hf)} skipped"
        )

        if batch_results:
            logger.info(f"   Batch {batch_idx + 1} molecule scores:")

            for mol in sorted(
                batch_results,
                key=lambda m: (
                    m.get("boltz_score")
                    if m.get("boltz_score") is not None
                    else float("-inf")
                ),
                reverse=True,
            ):
                name = mol.get("name", "unknown")
                score = mol.get("boltz_score")
                source = mol.get("boltz_score_source", "boltz")

                if score is not None:
                    logger.info(
                        f"      {name}: {score:.6f} [{source}]"
                    )
                else:
                    logger.info(f"      {name}: skipped [{source}]")

    scored_molecules = sorted(
        all_scored_molecules,
        key=lambda m: (
            m.get("boltz_score")
            if m.get("boltz_score") is not None
            else float("-inf")
        ),
        reverse=True,
    )

    logger.info(
        f"✅ Batch scoring complete: {len(scored_molecules)} "
        f"total molecules scored"
    )

    return scored_molecules
# ═══════════════════════════════════════════════════════════════════════════
# Global MoleculeManager
# ═══════════════════════════════════════════════════════════════════════════

molecule_manager: Optional[MoleculeManager] = None


def initialize_solution(config: dict):
    global molecule_manager

    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{RXN_ID}"

    molecule_manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    logger.info(
        f"✅ MoleculeManager locked to rxn={RXN_ID} | "
        f"A={len(molecule_manager.moles_A_id)} | "
        f"B={len(molecule_manager.moles_B_id)} | "
        f"C={len(molecule_manager.moles_C_id)}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# find_solution
# ═══════════════════════════════════════════════════════════════════════════

async def find_solution(state: Dict[str, Any]) -> None:
    """
    Unlimited DPEX-DJA loop.

    Main change:
      Progress, plateau, and improvement are driven by the average score
      of the top TOP_AVG_K molecules, not by the single best molecule.

    Per-iteration pipeline:
      1. generate n_base_samples * GENERATE_MULTIPLIER
      2. resolve SMILES
      3. dedup seen
      4. surrogate filter if ready, else hard cap
      5. Boltz score
      6. update all_pool / top_pool
      7. compute top-50 average
      8. use top-50 average for improvement rate
    """
    global molecule_manager

    config = state["config"]

    def _cfg(key, default):
        return (
            config.get(key, default)
            if isinstance(config, dict)
            else getattr(config, key, default)
        )

    num_molecules = _cfg("num_molecules", 100)
    tanimoto_max_threshold = _cfg("tanimoto_max_threshold", 0.9)
    boltz_batch_size = _cfg("boltz_batch_size", 10)
    LIMIT_PER_REACTANT = _cfg("limit_per_reactant", 600)

    surrogate = SurrogateModel(max_training_samples=5000)
    use_surrogate = True
    exploit_counter = 0
    ranker = ComponentRanker(decay=0.90)
    plateau_counter = 0

    params = IterationParams(config=config)
    dpex = DPEXDJAState()

    seed_df = pd.DataFrame(columns=["name", "smiles"])
    top_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    all_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    tabued_molecules: set = set()
    iteration = 0

    top_pool, all_pool = warm_start(
        state=state,
        dpex=dpex,
        ranker=ranker,
        surrogate=surrogate,
        params=params,
        top_pool=top_pool,
        all_pool=all_pool,
        num_molecules=num_molecules,
        tanimoto_max_threshold=tanimoto_max_threshold,
    )

    init_top50_avg, init_pool_max, _ = _top_pool_stats(
        top_pool,
        TOP_AVG_K,
    )

    # Main all-time progress tracker.
    best_top50_avg_ever = init_top50_avg if not top_pool.empty else 0.0

    # Reporting-only best single score.
    best_max_ever = init_pool_max if not top_pool.empty else 0.0

    try:
        logger.info("[Solution] Building synthon library...")
        t0 = time.time()

        params.synthon_lib = SynthonLibrary(
            molecule_manager=molecule_manager
        )
        params.use_synthon_search = True

        logger.info(
            f"[Solution] Synthon library ready in {time.time() - t0:.2f}s"
        )

    except Exception as e:
        logger.warning(f"[Solution] Synthon library failed: {e}")
        params.synthon_lib = None
        params.use_synthon_search = False

    logger.info(
        f"🚀 DPEX-DJA loop | rxn={RXN_ID} | "
        f"A={len(molecule_manager.moles_A_id)} "
        f"B={len(molecule_manager.moles_B_id)} "
        f"C={len(molecule_manager.moles_C_id)} | "
        f"pipeline: generate {GENERATE_MULTIPLIER}x "
        f"→ dedup → surrogate keep {SURROGATE_KEEP_RATIO * 100:.0f}% "
        f"if trained, else hard-cap {BOLTZ_BUDGET} → Boltz | "
        f"progress: top-{TOP_AVG_K} average"
    )
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            iteration += 1
            component_weights = None
            dpex.iteration = iteration
            iter_start = time.time()

            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"[Solution] --- Iteration {iteration} [rxn={RXN_ID}] ---"
            )

            # ── Reload only NEW scored molecules every 5 iterations ───
            if iteration % 5 == 0:
                loaded_df = load_molecules_combined(RXN_ID, config)

                if not loaded_df.empty:
                    loaded_renamed = loaded_df.rename(
                        columns={"InChIKey": "inchi"}
                    )

                    existing_names = set(all_pool["name"].tolist())

                    new_only = loaded_renamed[
                        ~loaded_renamed["name"].isin(existing_names)
                    ]

                    if not new_only.empty:
                        all_pool = pd.concat(
                            [all_pool, new_only],
                            ignore_index=True,
                        )

                        all_pool = (
                            all_pool
                            .sort_values(
                                by="score",
                                ascending=False,
                                na_position="last",
                            )
                            .drop_duplicates(
                                subset=["inchi"],
                                keep="first",
                            )
                        )

                        logger.info(
                            f"[Solution] +{len(new_only)} new from disk "
                            f"(all_pool={len(all_pool)})"
                        )

                        if surrogate.enabled and "score" in new_only.columns:
                            valid_new = new_only[
                                new_only["score"].notna()
                                & new_only["smiles"].notna()
                            ]

                            if not valid_new.empty:
                                surrogate.add_training_data(
                                    valid_new["smiles"].tolist(),
                                    valid_new["score"].tolist(),
                                )

                                logger.info(
                                    f"[SURROGATE] +{len(valid_new)} "
                                    f"new rxn-specific scores from disk "
                                    f"(total_train_size="
                                    f"{surrogate.total_train_size})"
                                )

            # ── n_samples: always GENERATE_MULTIPLIER × base ──────────
            n_base_samples = params.get_nsamples_from_iteration(iteration)
            n_samples = n_base_samples * GENERATE_MULTIPLIER

            logger.info(
                f"[Solution] n_samples={n_samples} "
                f"({n_base_samples}×{GENERATE_MULTIPLIER}) "
                f"→ dedup → surrogate keep "
                f"{SURROGATE_KEEP_RATIO * 100:.0f}% "
                f"if trained, else hard-cap {BOLTZ_BUDGET} → Boltz"
            )

            # ── Component weights ─────────────────────────────────────
            if not top_pool.empty:
                component_weights = build_component_weights(
                    top_pool.head(num_molecules),
                    RXN_ID,
                )

            if component_weights is not None:
                component_weights = ranker.blend_component_weights(
                    component_weights,
                    molecule_manager,
                )

            ranker.push_to_dja(molecule_manager)

            # ── Elite selection ───────────────────────────────────────
            elite_df = (
                MoleculeUtils.select_diverse_elites(
                    top_pool,
                    min(150, len(top_pool)),
                )
                if not top_pool.empty
                else pd.DataFrame()
            )

            elite_names = (
                elite_df["name"].tolist()
                if not elite_df.empty
                else None
            )

            # ── Exploit mode toggle ──────────────────────────────────
            if (
                params.no_improvement_counter >= 2
                and not params.use_exploit_mode
            ):
                params.use_exploit_mode = True
                params.no_improvement_counter = 0
                logger.info("[Solution] === EXPLOIT MODE ===")

            elif (
                params.no_improvement_counter >= 2
                or exploit_counter >= 4
            ):
                params.use_exploit_mode = False
                exploit_counter = 0
                params.no_improvement_counter = 0

            # ── Augment pop_B ────────────────────────────────────────
            if not top_pool.empty:
                cols = [
                    c for c in ("name", "smiles", "score")
                    if c in top_pool.columns
                ]

                dpex.augment_pop_B(
                    top_pool[cols].head(dpex.N_B).to_dict("records")
                )

            # ── Generation containers ────────────────────────────────
            data = pd.DataFrame(columns=["name", "smiles"])
            data_dja = pd.DataFrame(columns=["name"])
            data_tabu = pd.DataFrame(columns=["name"])
            data_tabu_moves: list = []
            data_early_exploit = pd.DataFrame(columns=["name", "smiles"])
            exploited_status = False
            exploit_summary = None
            exploit_attempted = False
            current_mode = "unknown"

            # ── Light early exploit during first 20 iterations ────────
            if not top_pool.empty and iteration <= 20:
                try:
                    unexploited_ee = get_top_n_unexploited(
                        top_pool.to_dict("records"),
                        params.exploited_reactants,
                        n=2,
                    )

                    if unexploited_ee:
                        t0_ee = time.time()

                        early_results, _ = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited_ee,
                            top_n=1,
                            limit_per_reactant=150,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=set(),
                        )

                        if early_results:
                            data_early_exploit = pd.DataFrame(
                                early_results
                            )

                            logger.info(
                                f"[Solution] Early exploit: "
                                f"{len(data_early_exploit)} in "
                                f"{time.time() - t0_ee:.1f}s"
                            )

                except Exception as e:
                    logger.debug(f"[Solution] Early exploit skipped: {e}")

            # ── Full exploit mode ────────────────────────────────────
            if params.use_exploit_mode and not top_pool.empty:
                exploit_attempted = True

                logger.info(
                    "[Solution] Exploit: structure-guided deep search..."
                )

                try:
                    unexploited = get_top_n_unexploited(
                        top_pool.to_dict("records"),
                        params.exploited_reactants,
                    )

                    if unexploited:
                        t0 = time.time()

                        exploit_results, exploit_summary = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=params.exploited_reactants,
                        )

                        logger.info(
                            f"[Solution] Exploit: {len(exploit_results)} "
                            f"candidates in {time.time() - t0:.1f}s"
                        )

                        if exploit_results:
                            data = pd.DataFrame(exploit_results)
                            exploited_status = True

                        else:
                            raise Exception(
                                "Exploit returned no molecules."
                            )

                    else:
                        raise Exception("No unexploited molecules.")

                except Exception as e:
                    logger.warning(f"[Solution] Exploit skipped: {e}")

                exploit_counter += 1

            if not exploited_status:
                if not dpex.pop_A:
                    # Cold fallback
                    logger.info(
                        f"[Solution] Cold init: generating "
                        f"{params.n_samples_start} random molecules"
                    )

                    raw = generate_valid_random_molecules(
                        molecule_manager=molecule_manager,
                        n_samples=params.n_samples_start,
                        seen_molecules=params.seen_molecules,
                        component_weights=component_weights,
                    )

                    data = pd.DataFrame({"name": raw})

                else:
                    # ── DJA ──────────────────────────────────────────
                    n_dja = int(n_samples * 0.75)
                    n_tabu = n_samples - n_dja

                    logger.info(
                        f"[Solution] DJA: {n_dja} candidates "
                        f"(pop_A={len(dpex.pop_A)})"
                    )

                    raw_dja = dja_generate(
                        state=dpex,
                        molecule_manager=molecule_manager,
                        n_samples=n_dja,
                        mutation_prob=params.mutation_prob,
                    )

                    if raw_dja:
                        data_dja = molecule_manager.validate_molecules(
                            config,
                            pd.DataFrame({"name": raw_dja}),
                        )

                        logger.info(
                            f"[Solution] DJA: {len(data_dja)} validated"
                        )

                    # ── Tabu ─────────────────────────────────────────
                    if params.synthon_lib is not None and dpex.pop_B:
                        if params.score_improvement_rate > 0.05:
                            n_per_elite = 15
                        elif params.score_improvement_rate > 0.02:
                            n_per_elite = 20
                        elif params.score_improvement_rate > 0.005:
                            n_per_elite = 25
                        else:
                            n_per_elite = 50

                        logger.info(
                            f"[Solution] Tabu: n_tabu≈{n_tabu} "
                            f"neighborhood={n_per_elite}"
                        )

                        raw_tabu = tabu_generate(
                            state=dpex,
                            synthon_lib=params.synthon_lib,
                            n_samples=n_tabu,
                            neighborhood_size=n_per_elite,
                        )

                        if params.score_improvement_rate <= 0.005:
                            tabued_molecules |= {
                                x["name"] for x in dpex.pop_B
                            }

                        if raw_tabu:
                            data_tabu = molecule_manager.validate_molecules(
                                config,
                                pd.DataFrame({"name": raw_tabu}),
                            )

                            if not data_dja.empty:
                                data_tabu = data_tabu[
                                    ~data_tabu["name"].isin(
                                        data_dja["name"].tolist()
                                    )
                                ]

                            logger.info(
                                f"[Solution] Tabu: "
                                f"{len(data_tabu)} validated"
                            )

                    # ── Merge DJA + Tabu + early exploit ─────────────
                    parts = [
                        df
                        for df in [
                            data_dja,
                            data_tabu,
                            data_early_exploit,
                        ]
                        if not df.empty
                    ]

                    if parts:
                        data = pd.concat(
                            parts,
                            ignore_index=True,
                        ).drop_duplicates(subset=["name"])

                        if not seed_df.empty:
                            data = pd.concat(
                                [data, seed_df],
                                ignore_index=True,
                            ).drop_duplicates(subset=["name"])

                            seed_df = pd.DataFrame(
                                columns=["name", "smiles"]
                            )

                    # ── Random top-up ────────────────────────────────
                    raw_rand = generate_valid_random_molecules(
                        molecule_manager=molecule_manager,
                        n_samples=int(n_samples * 0.5),
                        seen_molecules=params.seen_molecules,
                        component_weights=component_weights,
                    )

                    data = pd.concat(
                        [data, pd.DataFrame({"name": raw_rand})],
                        ignore_index=True,
                    ).drop_duplicates(subset=["name"])

            current_mode = _iteration_mode_str(
                exploited_status=exploited_status,
                dpex=dpex,
                params=params,
                early_exploit_used=not data_early_exploit.empty,
                exploit_attempted=exploit_attempted,
            )

            logger.info(
                f"[Solution] {len(data)} candidates generated "
                f"({current_mode}) in {time.time() - iter_start:.2f}s"
            )
            if data.empty:
                logger.warning("[Solution] No candidates; skipping")
                await asyncio.sleep(5)
                continue

            if not seed_df.empty:
                data = pd.concat(
                    [data, seed_df],
                    ignore_index=True,
                ).drop_duplicates(subset=["name"])

                seed_df = pd.DataFrame(columns=["name", "smiles"])

            # ── Resolve SMILES ───────────────────────────────────────
            if "smiles" not in data.columns or data["smiles"].isna().all():
                data["smiles"] = data["name"].apply(
                    MoleculeUtils.get_smiles_from_reaction_cached
                )

            data = data[data["smiles"].notna() & (data["smiles"] != "")]

            if data.empty:
                logger.warning("[Solution] No valid SMILES; skipping")
                await asyncio.sleep(5)
                continue

            # ══════════════════════════════════════════════════════════
            # STEP 1 — Dedup against seen molecules
            # ══════════════════════════════════════════════════════════
            pre_dedup = len(data)

            data = data[
                ~data["name"].isin(params.seen_molecules)
            ].reset_index(drop=True)

            dup_ratio = (pre_dedup - len(data)) / max(1, pre_dedup)

            logger.info(
                f"[Solution] Dedup: {pre_dedup} → {len(data)} "
                f"({dup_ratio * 100:.0f}% already seen)"
            )

            # Adaptive mutation based on duplicate ratio
            if dup_ratio > 0.7:
                params.mutation_prob = min(
                    0.90,
                    params.mutation_prob * 1.5,
                )
            elif dup_ratio > 0.5:
                params.mutation_prob = min(
                    0.70,
                    params.mutation_prob * 1.3,
                )
            elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                params.mutation_prob = max(
                    0.10,
                    params.mutation_prob * 0.95,
                )

            if data.empty:
                logger.error(
                    "[Solution] All duplicates after dedup; "
                    "boosting diversity"
                )

                params.mutation_prob = min(
                    0.95,
                    params.mutation_prob * 2.0,
                )
                params.elite_prob = max(
                    0.10,
                    params.elite_prob * 0.5,
                )

                await asyncio.sleep(5)
                continue

            # ══════════════════════════════════════════════════════════
            # STEP 2 — Surrogate filter after dedup
            # ══════════════════════════════════════════════════════════
            surrogate_ready = (
                use_surrogate
                and surrogate.enabled
                and surrogate.is_trained
                and surrogate.total_train_size >= surrogate.min_train_size
            )

            if surrogate_ready:
                pre_sur = len(data)

                data = surrogate.filter_candidates(
                    data,
                    keep_ratio=SURROGATE_KEEP_RATIO,
                    smiles_col="smiles",
                )

                logger.info(
                    f"[SURROGATE] iter={iteration} mode={current_mode} | "
                    f"{pre_sur} fresh → {len(data)} "
                    f"(kept top {SURROGATE_KEEP_RATIO * 100:.0f}%, "
                    f"train_size={surrogate.total_train_size})"
                )

            else:
                if len(data) > BOLTZ_BUDGET:
                    pre_cap = len(data)

                    data = data.head(BOLTZ_BUDGET)

                    logger.info(
                        f"[Solution] Surrogate not ready "
                        f"(train_size={surrogate.total_train_size} < "
                        f"{surrogate.min_train_size}) — hard-cap "
                        f"{pre_cap} → {len(data)} at {BOLTZ_BUDGET}"
                    )

            if data.empty:
                logger.warning(
                    "[Solution] No candidates after surrogate/filter; skipping"
                )
                await asyncio.sleep(5)
                continue

            # ── Boltz scoring ────────────────────────────────────────
            logger.info(
                f"[Solution] Scoring {len(data)} molecules with Boltz..."
            )

            t_score = time.time()

            scored_molecules = await score_molecules_with_boltz_batched(
                state,
                data.to_dict("records"),
                batch_size=boltz_batch_size,
            )

            logger.info(
                f"[Solution] Boltz done in {time.time() - t_score:.2f}s"
            )

            # ── Build scored_df ──────────────────────────────────────
            scored_df = pd.DataFrame([
                {
                    "name": m["name"],
                    "smiles": m.get("smiles", ""),
                    "score": m.get("boltz_score"),
                }
                for m in scored_molecules
                if m.get("boltz_score") is not None
            ])

            if scored_df.empty:
                logger.warning("[Solution] No scores returned; skipping")
                await asyncio.sleep(5)
                continue

            scored_df["score"] = pd.to_numeric(
                scored_df["score"],
                errors="coerce",
            )
            scored_df = scored_df[np.isfinite(scored_df["score"])]
            scored_df = scored_df.dropna(subset=["score"]).reset_index(
                drop=True
            )

            if scored_df.empty:
                logger.warning(
                    "[Solution] No finite scores returned; skipping"
                )
                await asyncio.sleep(5)
                continue

            # ── Update ComponentRanker ───────────────────────────────
            ranker.update(scored_df)

            # ── Update surrogate with new rxn-specific Boltz scores ──
            if surrogate.enabled:
                surrogate.add_training_data(
                    scored_df["smiles"].tolist(),
                    scored_df["score"].tolist(),
                )

                if surrogate.total_train_size >= surrogate.min_train_size:
                    t_train = time.time()

                    surrogate.train(iteration)

                    train_time = time.time() - t_train

                    if train_time > 10.0:
                        logger.warning(
                            f"[SURROGATE] Training slow "
                            f"({train_time:.2f}s) — disabling"
                        )

                        surrogate.enabled = False
                        use_surrogate = False

            # ── Update DPEX populations ──────────────────────────────
            dja_names = (
                set(data_dja["name"].tolist())
                if not data_dja.empty
                else set()
            )

            tabu_names = (
                set(data_tabu["name"].tolist())
                if not data_tabu.empty
                else set()
            )

            scored_for_A = (
                scored_df[scored_df["name"].isin(dja_names)]
                if dja_names
                else scored_df
            )

            scored_for_B = (
                scored_df[scored_df["name"].isin(tabu_names)]
                if tabu_names
                else pd.DataFrame(columns=scored_df.columns)
            )

            update_populations(dpex, scored_for_A)

            if not scored_for_B.empty:
                dpex.augment_pop_B(scored_for_B.to_dict("records"))

            if data_tabu_moves:
                for move_name in data_tabu_moves:
                    update_tabu(dpex, move_name)

            if iteration % dpex.T_ex == 0:
                dpex_exchange(dpex)

            logger.debug(
                f"[DPEX] pop_A={len(dpex.pop_A)} "
                f"pop_B={len(dpex.pop_B)}"
            )

            # ── Update seen molecules ────────────────────────────────
            params.seen_molecules = params.seen_molecules | set(
                scored_df["name"].tolist()
            )

            # ══════════════════════════════════════════════════════════
            # Pool update + TOP-50 average progress tracking
            # ══════════════════════════════════════════════════════════

            # Capture previous top-50 average before updating pool.
            iter_prev_top50_avg, iter_prev_max, _ = _top_pool_stats(
                top_pool,
                TOP_AVG_K,
            )

            prev_top50_avg = (
                iter_prev_top50_avg
                if not top_pool.empty
                else None
            )

            prev_max = (
                iter_prev_max
                if not top_pool.empty
                else None
            )

            # Add InChIKey and merge into all_pool.
            scored_df["inchi"] = scored_df["smiles"].apply(
                MoleculeUtils.generate_inchikey
            )

            scored_df = scored_df[scored_df["inchi"] != ""]

            if not scored_df.empty:
                all_pool = (
                    pd.concat(
                        [all_pool, scored_df],
                        ignore_index=True,
                    )
                    if not all_pool.empty
                    else scored_df.copy()
                )

                all_pool = (
                    all_pool
                    .sort_values(
                        by="score",
                        ascending=False,
                        na_position="last",
                    )
                    .drop_duplicates(
                        subset=["inchi"],
                        keep="first",
                    )
                )

            # Build diverse top_pool.
            top_pool = select_tanimoto_diverse(
                all_pool.reset_index(drop=True),
                n=num_molecules + 50,
                threshold=tanimoto_max_threshold,
                smiles_col="smiles",
            ).reset_index(drop=True)

            # Main metric: average score of top 50 molecules.
            top50_avg, pool_max, best_name = _top_pool_stats(
                top_pool,
                TOP_AVG_K,
            )

            current_top50_avg = (
                top50_avg
                if not top_pool.empty
                else None
            )

            if (
                current_top50_avg is not None
                and prev_top50_avg is not None
            ):
                params.score_improvement_rate = (
                    (current_top50_avg - prev_top50_avg)
                    / max(abs(prev_top50_avg), 1e-6)
                )

            elif current_top50_avg is not None:
                params.score_improvement_rate = 1.0

            # Plateau logic now uses top-50 average improvement.
            if params.score_improvement_rate <= 0.0001:
                params.no_improvement_counter += 1
                plateau_counter += 1

            else:
                params.no_improvement_counter = 0
                plateau_counter = 0

            # Anti-plateau mutation boost
            if plateau_counter >= 5:
                params.mutation_prob = min(
                    0.85,
                    params.mutation_prob * 2.0,
                )

                logger.info(
                    f"[Solution] ANTI-PLATEAU: mutation_prob → "
                    f"{params.mutation_prob:.2f}"
                )

                plateau_counter = 0

            # ── Update exploited reactants ────────────────────────────
            if (
                exploit_summary
                and "exploited_reactant_ids" in exploit_summary
                and (
                    params.score_improvement_rate <= 0.0001
                    or not exploited_status
                )
            ):
                params.exploited_reactants.update(
                    exploit_summary["exploited_reactant_ids"]
                )

                logger.info(
                    f"[Solution] Exploited reactants total: "
                    f"{len(params.exploited_reactants)}"
                )

            # ── Iteration summary ────────────────────────────────────
            iter_time = time.time() - iter_start

            logger.info(
                f"Iter {iteration:4d} | {iter_time:6.1f}s | "
                f"Mode: {current_mode:24s} | rxn={RXN_ID} | "
                f"popA={len(dpex.pop_A):4d} "
                f"popB={len(dpex.pop_B):4d} | "
                f"top{TOP_AVG_K}_avg={top50_avg:.5f} "
                f"best={pool_max:.5f} | "
                f"Δ_top{TOP_AVG_K}="
                f"{params.score_improvement_rate:+.5f} | "
                f"no_improve={params.no_improvement_counter} | "
                f"surrogate={'ON' if surrogate_ready else 'OFF'} "
                f"({surrogate.total_train_size}/"
                f"{surrogate.min_train_size} samples, rxn={RXN_ID})"
            )

            best_top50_avg_ever = _log_pool_progress(
                iteration,
                top50_avg,
                pool_max,
                best_name,
                prev_top50_avg,
                prev_max,
                best_top50_avg_ever,
                params.score_improvement_rate,
                num_molecules,
                mode=current_mode,
            )

            # Reporting-only all-time best single molecule score.
            if pool_max > best_max_ever:
                best_max_ever = pool_max

            if not top_pool.empty:
                logger.info(
                    f"   🏆 Top-{TOP_AVG_K} avg: {top50_avg:.6f} | "
                    f"Best single: {best_name} "
                    f"(score={pool_max:.6f})"
                )

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        logger.info(f"\n🛑 Stopping DPEX-DJA loop (rxn={RXN_ID})...")

        if not top_pool.empty:
            final_top50_avg, final_max, final_best = _top_pool_stats(
                top_pool,
                TOP_AVG_K,
            )

            logger.info(
                f"Final top-{TOP_AVG_K} avg: "
                f"{final_top50_avg:.6f} | "
                f"best single: {final_best} "
                f"(score={final_max:.6f}) | "
                f"all-time top-{TOP_AVG_K} avg="
                f"{best_top50_avg_ever:.6f} | "
                f"all-time single max={best_max_ever:.6f}"
            )

        raise
# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    rxn_id = parse_args()

    logger.info(
        f"🚀 Starting miner.py | rxn={rxn_id} | "
        f"DB=score_results_{rxn_id}.sqlite | "
        f"CSV=data/rxn{rxn_id}.csv | "
        f"progress=top-{TOP_AVG_K} average score"
    )

    try:
        config = load_config()
        logger.info("✅ Config loaded")

    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    initialize_solution(config)

    state: Dict[str, Any] = {
        "config": config,
        "current_challenge_targets": [],
        "current_challenge_targets_clip_interval": [],
        "current_challenge_antitargets": [],
        "boltz_wrapper": None,
    }

    state["current_challenge_targets"] = config["small_molecule_target"]
    state["current_challenge_targets_clip_interval"] = config[
        "small_molecule_target_clip_interval"
    ]

    logger.info(f"🎯 Target:      {state['current_challenge_targets'][0]}")
    logger.info(f"🚫 Antitargets: {state['current_challenge_antitargets']}")

    init_score_results_db()

    logger.info("🔬 Importing BoltzWrapper...")

    if _import_boltz_wrapper() and BoltzWrapper is not None:
        try:
            state["boltz_wrapper"] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized")

        except Exception as e:
            logger.error(f"❌ BoltzWrapper init failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            state["boltz_wrapper"] = None

    else:
        logger.warning(
            "⚠️ BoltzWrapper unavailable — scoring will be skipped"
        )

    try:
        await find_solution(state)

    except KeyboardInterrupt:
        logger.info(f"✅ rxn={rxn_id} stopped by user")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
