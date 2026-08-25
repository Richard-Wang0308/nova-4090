"""
miner.py — DPEX-DJA + Boltz, single fixed reaction (no multi-rxn).

Pipeline per iteration:
  1. Generate  n_base_samples * 5  candidates  (≈ 2000)
  2. Resolve SMILES
  3. Dedup against seen            → removes already-scored molecules FIRST
  4. Surrogate filter              → top 100 from fresh candidates only
  5. Boltz score                   → 100 molecules with heavy model
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
RXN_ID           = None
SCORE_RESULTS_DB = None
RXN_CSV          = None
TRAINING_CSV     = None

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

# ── BoltzWrapper (lazy import) ────────────────────────────────────────────
BOLTZ_AVAILABLE = False
BoltzWrapper    = None

# ── Surrogate pipeline constants ──────────────────────────────────────────
GENERATE_MULTIPLIER = 5    # n_base_samples * 5  ≈ 2000
BOLTZ_BUDGET        = 100  # hard cap sent to Boltz after surrogate filter

# ── fingerprint generators ────────────────────────────────────────────────
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)
_fp_cache        = {}
_mol_cache       = {}
_morgan_bv_cache = {}

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    global RXN_ID, SCORE_RESULTS_DB, RXN_CSV, TRAINING_CSV

    parser = argparse.ArgumentParser(
        description="DPEX-DJA Miner — single fixed reaction mode"
    )
    parser.add_argument(
        "--rxn_id", type=int, required=True,
        help="Reaction ID (e.g. 1-5).",
    )
    args = parser.parse_args()

    RXN_ID           = args.rxn_id
    SCORE_RESULTS_DB = os.path.join(BASE_DIR, f"score_results_{RXN_ID}.sqlite")
    RXN_CSV          = os.path.join(BASE_DIR, "data", f"rxn{RXN_ID}.csv")
    TRAINING_CSV     = os.path.join(BASE_DIR, "data", "mols.csv")

    logger.info(f"✅ rxn_id           = {RXN_ID}")
    logger.info(f"✅ SCORE_RESULTS_DB  = {SCORE_RESULTS_DB}")
    logger.info(f"✅ RXN_CSV           = {RXN_CSV}  (warm-seed population)")
    logger.info(f"✅ TRAINING_CSV      = {TRAINING_CSV}  (surrogate training)")
    logger.info(
        f"✅ Pipeline = generate {GENERATE_MULTIPLIER}x "
        f"→ dedup → surrogate→{BOLTZ_BUDGET} → Boltz"
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
    fp       = MORGAN_FP_GENERATOR.GetFingerprint(mol)
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
    kept_fps     = []
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
            old, cnt   = store[key]
            store[key] = (self.decay * old + (1 - self.decay) * score, cnt + 1)
        else:
            store[key] = (score, 1)

    def update(self, scored_df: pd.DataFrame):
        if scored_df.empty:
            return
        for _, row in scored_df.iterrows():
            score = row.get('score', 0.0)
            if pd.isna(score):
                continue
            parts = str(row['name']).split(':')
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
            if manager.is_three_component else None
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
            ('A', manager.moles_A_id, self.q_A),
            ('B', manager.moles_B_id, self.q_B),
        ]:
            if role not in blended or not q:
                continue
            orig  = blended[role]
            new_w = {}
            for mid in pool:
                o          = orig.get(mid, 0.05)
                e          = max(0.01, q[mid][0]) if mid in q else 0.05
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
    Random Forest surrogate for pre-filtering before Boltz scoring.
    Warm-start anchor data is never evicted.
    Filters AFTER dedup so it only scores genuinely new candidates. $CITE_2
    """

    def __init__(self, max_training_samples: int = 4000):
        self.model = RandomForestRegressor(
            n_estimators=70, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1, max_samples=0.8,
        )
        self.is_trained           = False
        self.anchor_X: list       = []
        self.anchor_y: list       = []
        self.X_train: list        = []
        self.y_train: list        = []
        self.min_train_size       = 80
        self.max_training_samples = max_training_samples
        self.last_train_iteration = 0
        self.train_interval       = 2
        self.enabled              = True

    def _safe_fp(self, smiles: str) -> np.ndarray:
        """
        Always returns a 2048-dim uint8 array.
        Explicit `is None` check — avoids numpy ambiguous bool error. $CITE_1
        """
        fp = get_morgan_fingerprint(smiles)
        return fp if fp is not None else np.zeros(2048, dtype=np.uint8)

    def add_anchor_data(self, smiles_list: list, scores: list):
        if not smiles_list:
            return
        scores_arr = np.array(scores)
        n          = len(scores_arr)
        top_idx    = set(np.argsort(scores_arr)[-max(1, n // 2):].tolist())
        rand_idx   = set(
            np.random.choice(n, max(1, n // 10), replace=False).tolist()
        )
        keep  = sorted(top_idx | rand_idx)
        added = 0
        for i in keep:
            fp = get_morgan_fingerprint(smiles_list[i])
            if fp is not None:
                self.anchor_X.append(fp)
                self.anchor_y.append(float(scores_arr[i]))
                added += 1
        bt.logging.info(
            f"[SURROGATE] Anchored {added} warm-start samples "
            f"(from {n} total, never evicted)"
        )

    def add_training_data(self, smiles_list: list, scores: list):
        if not self.enabled:
            return
        if len(smiles_list) > 600:
            scores_array = np.array(scores)
            top_indices  = np.argsort(scores_array)[-500:]
            mid_low      = np.argsort(scores_array)[
                :min(200, len(scores_array) // 2)
            ]
            sample_low   = list(
                np.random.choice(
                    mid_low, min(100, len(mid_low)), replace=False
                )
            ) if len(mid_low) > 0 else []
            keep_indices = sorted(set(list(top_indices) + sample_low))
            smiles_list  = [smiles_list[i] for i in keep_indices]
            scores       = [scores[i]      for i in keep_indices]

        for smiles, score in zip(smiles_list, scores):
            fp = get_morgan_fingerprint(smiles)
            if fp is not None:
                self.X_train.append(fp)
                self.y_train.append(score)

        if len(self.X_train) > self.max_training_samples:
            scores_array   = np.array(self.y_train)
            top_count      = int(self.max_training_samples * 0.5)
            recent_count   = int(self.max_training_samples * 0.5)
            top_indices    = np.argsort(scores_array)[-top_count:]
            recent_indices = list(range(
                len(self.X_train) - recent_count, len(self.X_train)
            ))
            keep_indices   = sorted(set(list(top_indices) + recent_indices))
            self.X_train   = [self.X_train[i] for i in keep_indices]
            self.y_train   = [self.y_train[i]  for i in keep_indices]

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
            self.is_trained           = True
            self.last_train_iteration = iteration
            bt.logging.info(
                f"[SURROGATE] Trained in {time.time()-t0:.2f}s on "
                f"{len(self.anchor_X)} anchors + {len(self.X_train)} live "
                f"= {len(X_all)} total"
            )
        except Exception as e:
            bt.logging.warning(f"Surrogate training failed: {e}")
            self.is_trained = False

    def predict(self, smiles_list: list) -> np.ndarray:
        if not self.is_trained:
            return np.array([0.0] * len(smiles_list))
        try:
            # ✅ _safe_fp: no `or` on ndarray — avoids ambiguous bool $CITE_1
            fps = [self._safe_fp(s) for s in smiles_list]
            return self.model.predict(np.array(fps))
        except Exception as e:
            bt.logging.warning(f"Surrogate prediction failed: {e}")
            return np.array([0.0] * len(smiles_list))

    def filter_candidates(
        self,
        data: pd.DataFrame,
        n_keep: int,
        smiles_col: str = "smiles",
    ) -> pd.DataFrame:
        """
        Keep top n_keep by predicted score.
        Must be called AFTER dedup so input contains only fresh molecules. $CITE_2 $CITE_3
        """
        if not self.is_trained or data.empty or len(data) <= n_keep:
            return data
        pred = self.predict(data[smiles_col].tolist())
        data = data.copy()
        data["_pred"] = pred
        filtered = (
            data.sort_values("_pred", ascending=False)
            .head(n_keep)
            .drop(columns=["_pred"])
        )
        bt.logging.info(
            f"[SURROGATE] {len(data)} → {len(filtered)} "
            f"(dropped {len(data)-len(filtered)} low-quality before Boltz)"
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
    molecule_name: str, smiles: str, config: Dict[str, Any]
) -> Tuple[bool, str]:
    try:
        count     = get_heavy_atom_count(smiles)
        min_atoms = config.get('min_heavy_atoms', 10)
        max_atoms = config.get('max_heavy_atoms', 40)
        if count < min_atoms:
            return False, f"Insufficient heavy atoms: {count} < {min_atoms}"
        if count > max_atoms:
            return False, f"Too many heavy atoms: {count} > {max_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Heavy atom count error: {str(e)}"


def validate_molecule_banned_atoms(
    molecule_name: str, smiles: str, config: Dict[str, Any]
) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for banned atom check"
        banned = config.get('banned_atom_types', [])
        if not banned:
            return True, ""
        if contains_atom_type(mol, banned):
            return False, f"Contains banned atom types: {banned}"
        return True, ""
    except Exception as e:
        return False, f"Banned atom check error: {str(e)}"


def validate_molecule_rotatable_bonds(
    molecule_name: str, smiles: str, config: Dict[str, Any]
) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for rotatable bonds check"
        n_rot     = Descriptors.NumRotatableBonds(mol)
        min_bonds = config.get('min_rotatable_bonds', 1)
        max_bonds = config.get('max_rotatable_bonds', 10)
        if n_rot < min_bonds or n_rot > max_bonds:
            return False, (
                f"Rotatable bonds out of range: {n_rot} "
                f"(expected {min_bonds}-{max_bonds})"
            )
        return True, ""
    except Exception as e:
        return False, f"Rotatable bonds check error: {str(e)}"


async def validate_molecule_huggingface_unique(
    state: Dict[str, Any], molecule_name: str, smiles: str
) -> Tuple[bool, str]:
    if not state.get('current_challenge_targets'):
        return False, "No target proteins available"
    primary_target = state['current_challenge_targets'][0]
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
        conn   = sqlite3.connect(db_path)
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
    molecules: List[Dict[str, Any]], db_path: str = None
) -> None:
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    if not molecules:
        return
    try:
        conn      = sqlite3.connect(db_path)
        cursor    = conn.cursor()
        to_insert = [
            (m['name'], float(m['boltz_score']), True)
            for m in molecules
            if m.get('name') and m.get('boltz_score') is not None
        ]
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
    molecule_names: List[str], db_path: str = None
) -> Dict[str, float]:
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    if not molecule_names:
        return {}
    try:
        conn         = sqlite3.connect(db_path)
        cursor       = conn.cursor()
        placeholders = ','.join('?' * len(molecule_names))
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
        logger.warning(f"⚠️  RXN CSV not found: {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'final_score' in df.columns and 'score' not in df.columns:
            df = df.rename(columns={'final_score': 'score'})
        df['molecule_name'] = (
            df['molecule_name'].astype(str).str.strip().str.lstrip('\ufeff')
        )
        prefix = f"rxn:{rxn_id}:"
        df     = df[
            df['molecule_name'].str.startswith(prefix, na=False)
        ].reset_index(drop=True)
        if df.empty:
            logger.warning(
                f"⚠️  {os.path.basename(csv_path)}: "
                f"no rows matching prefix '{prefix}'"
            )
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        df['smiles'] = df['molecule_name'].apply(
            MoleculeUtils.get_smiles_from_reaction_cached
        )
        df = df[df['smiles'].notna() & (df['smiles'] != '')]
        df['InChIKey'] = df['smiles'].apply(MoleculeUtils.generate_inchikey)
        df = df[df['InChIKey'].notna() & (df['InChIKey'] != '')]
        result = df[['molecule_name', 'smiles', 'InChIKey', 'score']].copy()
        result = result.rename(columns={'molecule_name': 'name'})
        result['score'] = pd.to_numeric(result['score'], errors='coerce')
        result = result.drop_duplicates(subset=['InChIKey'], keep='first')
        result = result.sort_values(
            by='score', ascending=False, na_position='last'
        ).reset_index(drop=True)
        logger.info(
            f"✅ CSV [{os.path.basename(csv_path)}]: "
            f"{len(result)} molecules loaded (no validation)"
        )
        return result
    except Exception as e:
        logger.error(f"Error loading CSV {csv_path}: {e}")
        import traceback; logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_db(db_path: str, rxn_id: int) -> pd.DataFrame:
    if not os.path.exists(db_path):
        logger.info(f"ℹ️  Score DB not found yet: {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT molecule_name, score FROM scored_molecules "
            "WHERE molecule_name LIKE ?",
            (f"rxn:{rxn_id}:%",)
        )
        db_results = cursor.fetchall()
        conn.close()
        if not db_results:
            logger.info(f"ℹ️  Score DB empty for rxn={rxn_id}: {db_path}")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        rows = []
        fail = 0
        for mol_name, score in db_results:
            try:
                smiles = MoleculeUtils.get_smiles_from_reaction_cached(mol_name)
                if not smiles:
                    fail += 1; continue
                inchikey = MoleculeUtils.generate_inchikey(smiles)
                if not inchikey:
                    fail += 1; continue
                rows.append({
                    'name':     mol_name,
                    'smiles':   smiles,
                    'InChIKey': inchikey,
                    'score':    float(score) if score is not None else None,
                })
            except Exception as e:
                logger.debug(f"Could not process {mol_name}: {e}")
                fail += 1
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.drop_duplicates(subset=['InChIKey'], keep='first')
            result = result.sort_values(
                by='score', ascending=False, na_position='last'
            ).reset_index(drop=True)
        logger.info(
            f"✅ DB [{os.path.basename(db_path)}]: "
            f"{len(result)} molecules loaded (fail={fail}, no validation)"
        )
        return result
    except Exception as e:
        logger.error(f"Error loading DB {db_path}: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(
    rxn_id: int, config: Dict[str, Any] = None
) -> pd.DataFrame:
    csv_path = os.path.join(BASE_DIR, "data", f"rxn{rxn_id}.csv")
    db_path  = os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")
    logger.info(
        f"🔄 Warm-seed load rxn={rxn_id}: "
        f"{os.path.basename(csv_path)} + {os.path.basename(db_path)}"
    )
    csv_df = load_molecules_from_csv(csv_path, rxn_id)
    db_df  = load_molecules_from_db(db_path,   rxn_id)
    if csv_df.empty and db_df.empty:
        logger.warning(f"⚠️  No warm-seed data for rxn={rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    if csv_df.empty:
        return db_df
    if db_df.empty:
        return csv_df
    combined = pd.concat([csv_df, db_df], ignore_index=True)
    combined = combined.sort_values(
        by='score', ascending=False, na_position='last'
    ).drop_duplicates(subset=['InChIKey'], keep='first')
    logger.info(
        f"✅ Warm-seed rxn={rxn_id}: "
        f"{len(csv_df)} CSV + {len(db_df)} DB = {len(combined)} unique"
    )
    return combined


def load_training_csv_for_surrogate() -> pd.DataFrame:
    csv_path = TRAINING_CSV
    if not os.path.exists(csv_path):
        logger.warning(f"⚠️  Training CSV not found: {csv_path}")
        return pd.DataFrame(columns=["smiles", "score"])
    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'final_score' in df.columns and 'score' not in df.columns:
            df = df.rename(columns={'final_score': 'score'})
        df['molecule_name'] = (
            df['molecule_name'].astype(str).str.strip().str.lstrip('\ufeff')
        )
        df = df[df['score'].notna()].reset_index(drop=True)
        df['smiles'] = df['molecule_name'].apply(
            MoleculeUtils.get_smiles_from_reaction_cached
        )
        df = df[df['smiles'].notna() & (df['smiles'] != '')]
        result = df[['smiles', 'score']].copy()
        result['score'] = pd.to_numeric(result['score'], errors='coerce')
        result = result[result['score'].notna()]
        result = result.drop_duplicates(subset=['smiles']).reset_index(drop=True)
        logger.info(
            f"✅ Surrogate training [mols.csv]: "
            f"{len(result)} molecules loaded (no validation)"
        )
        return result
    except Exception as e:
        logger.error(f"Error loading training CSV: {e}")
        import traceback; logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["smiles", "score"])


# ═══════════════════════════════════════════════════════════════════════════
# WARM START
# ═══════════════════════════════════════════════════════════════════════════

def warm_start(
    state:    Dict[str, Any],
    dpex:     DPEXDJAState,
    ranker:   ComponentRanker,
    surrogate: SurrogateModel,
    params:   IterationParams,
    top_pool: pd.DataFrame,
    all_pool: pd.DataFrame,
    num_molecules: int,
    tanimoto_max_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = state['config']
    bt.logging.info(
        f"\n{'='*60}\n"
        f"[WarmStart] rxn={RXN_ID} | "
        f"pop-seed: rxn{RXN_ID}.csv + score_results_{RXN_ID}.sqlite | "
        f"surrogate: mols.csv\n"
        f"{'='*60}"
    )

    loaded_df = load_molecules_combined(RXN_ID, config)

    if loaded_df.empty:
        bt.logging.warning("[WarmStart] No rxn-specific data — cold start")
    else:
        bt.logging.info(
            f"[WarmStart] Rxn-specific: {len(loaded_df)} molecules loaded"
        )
        all_pool = loaded_df.rename(columns={'InChIKey': 'inchi'}).copy()
        ranker.update(loaded_df)
        ranker.warm_start_decay(n_historical_rounds=50)
        bt.logging.info(
            f"[WarmStart] ComponentRanker: "
            f"{len(ranker.q_A)} A | {len(ranker.q_B)} B "
            f"(EMA decayed 50 rounds)"
        )
        top_for_A  = loaded_df.head(dpex.N_A)
        dpex.pop_A = top_for_A.rename(
            columns={'InChIKey': 'inchi'}
        ).to_dict('records')
        bt.logging.info(
            f"[WarmStart] pop_A: {len(dpex.pop_A)} | "
            f"best={top_for_A['score'].max():.6f} | "
            f"avg={top_for_A['score'].mean():.6f}"
        )
        diverse_elites = select_tanimoto_diverse(
            loaded_df.reset_index(drop=True),
            n=dpex.N_B, threshold=0.85, smiles_col='smiles',
        )
        dpex.pop_B = diverse_elites.rename(
            columns={'InChIKey': 'inchi'}
        ).to_dict('records')
        bt.logging.info(
            f"[WarmStart] pop_B: {len(dpex.pop_B)} diverse elites"
        )
        params.seen_molecules = set(loaded_df['name'].tolist())
        bt.logging.info(
            f"[WarmStart] seen_molecules: {len(params.seen_molecules)}"
        )
        top_pool = select_tanimoto_diverse(
            all_pool.reset_index(drop=True),
            n=num_molecules + 50,
            threshold=tanimoto_max_threshold,
            smiles_col="smiles",
        ).reset_index(drop=True)

    bt.logging.info("[WarmStart] Loading mols.csv for surrogate training...")
    training_df = load_training_csv_for_surrogate()

    if len(training_df) >= surrogate.min_train_size:
        surrogate.add_anchor_data(
            training_df['smiles'].tolist(),
            training_df['score'].tolist(),
        )
        surrogate.train(iteration=0)
        bt.logging.info(
            f"[WarmStart] Surrogate pre-trained: "
            f"{len(surrogate.anchor_X)} anchors | "
            f"trained={surrogate.is_trained}"
        )
    else:
        bt.logging.warning(
            f"[WarmStart] mols.csv too small "
            f"({len(training_df)} < {surrogate.min_train_size}) — "
            f"will train after first scored batch"
        )

    if not loaded_df.empty:
        scores = loaded_df['score'].dropna()
        bt.logging.info(
            f"\n[WarmStart] ✅ Complete!\n"
            f"  Score range  : {scores.min():.6f} → {scores.max():.6f}\n"
            f"  Score mean   : {scores.mean():.6f}\n"
            f"  top_pool     : {len(top_pool)}\n"
            f"  all_pool     : {len(all_pool)}\n"
            f"  pop_A        : {len(dpex.pop_A)}\n"
            f"  pop_B        : {len(dpex.pop_B)}\n"
            f"  seen         : {len(params.seen_molecules)}\n"
            f"  Surrogate    : trained={surrogate.is_trained} "
            f"anchors={len(surrogate.anchor_X)} (mols.csv)\n"
            f"  Pipeline     : generate {GENERATE_MULTIPLIER}x "
            f"→ dedup → surrogate→{BOLTZ_BUDGET} → Boltz\n"
        )
    else:
        bt.logging.info(
            f"\n[WarmStart] ⚠️  Cold start.\n"
            f"  Surrogate: trained={surrogate.is_trained} "
            f"anchors={len(surrogate.anchor_X)}\n"
        )

    return top_pool, all_pool


# ═══════════════════════════════════════════════════════════════════════════
# BoltzWrapper import + scoring
# ═══════════════════════════════════════════════════════════════════════════

def _import_boltz_wrapper() -> bool:
    global BOLTZ_AVAILABLE, BoltzWrapper
    try:
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR     = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        if not os.path.exists(BOLTZ_SCORING_DIR):
            logger.warning(
                f"⚠️  Boltz-scoring directory not found: {BOLTZ_SCORING_DIR}"
            )
            return False
        for p in [BOLTZ_SCORING_DIR, BOLTZ_SRC_DIR]:
            if p not in sys.path:
                sys.path.insert(0, p)
        boltz_utils = os.path.join(BOLTZ_SCORING_DIR, 'utils')
        if os.path.exists(boltz_utils) and boltz_utils not in sys.path:
            sys.path.insert(0, boltz_utils)
        from boltz.wrapper import BoltzWrapper as BW
        BoltzWrapper    = BW
        BOLTZ_AVAILABLE = True
        logger.info("✅ BoltzWrapper imported successfully")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Failed to import BoltzWrapper: {e}")
        return False


async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    if state.get('boltz_wrapper') is None:
        logger.warning("BoltzWrapper not available, skipping scoring")
        return molecules
    if not molecules:
        return molecules

    logger.info(
        f"🔬 Scoring {len(molecules)} molecules (batch_size={batch_size})..."
    )
    init_score_results_db()

    all_scored    = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        batch     = molecules[start_idx: start_idx + batch_size]

        logger.info(
            f"📦 Batch {batch_idx + 1}/{total_batches}: {len(batch)} molecules"
        )

        target_proteins     = state.get('current_challenge_targets', [])
        antitarget_proteins = state.get('current_challenge_antitargets', [])
        primary_target      = target_proteins[0] if target_proteins else None

        db_scores = batch_get_scores_from_db([m['name'] for m in batch])

        molecules_with_db  = []
        molecules_in_hf    = []
        molecules_to_score = []

        for mol in batch:
            if mol['name'] in db_scores:
                mol['boltz_score']        = db_scores[mol['name']]
                mol['boltz_score_source'] = 'database'
                molecules_with_db.append(mol)
                continue
            if primary_target and mol.get('smiles'):
                try:
                    if not molecule_unique_for_protein_hf(
                        primary_target, mol['smiles']
                    ):
                        molecules_in_hf.append(mol)
                        continue
                except Exception:
                    pass
            molecules_to_score.append(mol)

        logger.info(
            f"   {len(molecules_with_db)} DB cache | "
            f"{len(molecules_in_hf)} HF skip | "
            f"{len(molecules_to_score)} → Boltz"
        )

        newly_scored = []
        if molecules_to_score and target_proteins:
            try:
                boltz  = state['boltz_wrapper']
                config = state['config']

                output_dir = os.path.join(
                    boltz.output_dir, 'boltz_results_inputs'
                )
                for d in ['processed/structures', 'processed/records',
                          'processed/msa', 'predictions']:
                    os.makedirs(os.path.join(output_dir, d), exist_ok=True)
                try:
                    import shutil
                    ll = os.path.join(output_dir, 'lightning_logs')
                    if os.path.exists(ll):
                        shutil.rmtree(ll, ignore_errors=True)
                except Exception:
                    pass

                valid_molecules_by_uid = {
                    0: {
                        'smiles': [m['smiles'] for m in molecules_to_score],
                        'names':  [m['name']   for m in molecules_to_score],
                    }
                }
                score_dict = {
                    0: {
                        "target_scores": [[]], "antitarget_scores": [[]],
                        "entropy": None, "entropy_boltz": None,
                        "block_submitted": None, "push_time": "",
                    }
                }
                subnet_config = {
                    'weekly_target':        primary_target,
                    'num_antitargets':      len(antitarget_proteins),
                    'binding_pocket':       getattr(config, 'binding_pocket', None),
                    'max_distance':         getattr(config, 'max_distance', None),
                    'force':                getattr(config, 'force', False),
                    'num_molecules_boltz':  len(molecules_to_score),
                    'boltz_metric':         getattr(config, 'boltz_metric', [
                        'affinity_probability_binary', 'affinity_pred_value'
                    ]),
                    'combination_strategy': getattr(
                        config, 'combination_strategy',
                        'heavy_atom_normalization'
                    ),
                    'sample_selection':     getattr(
                        config, 'sample_selection', 'first'
                    ),
                }

                t0   = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: boltz.score_molecules_target(
                        valid_molecules_by_uid, score_dict,
                        subnet_config, "0x" + "0" * 64,
                    )
                )
                logger.info(f"   ✅ Boltz scored in {time.time()-t0:.2f}s")

                uid                = 0
                smiles_to_score    = boltz.per_molecule_metric.get(uid, {}).copy()
                ts                 = score_dict[uid].get('target_scores', [[]])
                target_scores_list = (
                    ts[0] if ts and len(ts[0]) > 0 else None
                )
                avg_score = (
                    score_dict[uid].get('boltz_score')
                    if not smiles_to_score and not target_scores_list
                    else None
                )

                for idx, mol in enumerate(molecules_to_score):
                    smi   = mol['smiles']
                    score = smiles_to_score.get(smi)
                    if score is None and target_scores_list:
                        score = (
                            target_scores_list[idx]
                            if idx < len(target_scores_list) else None
                        )
                    if score is None:
                        score = avg_score
                    mol['boltz_score'] = score
                    if score is not None:
                        newly_scored.append(mol)

                if newly_scored:
                    write_scores_to_db(newly_scored)

            except Exception as e:
                logger.error(f"❌ Boltz scoring error: {e}")
                import traceback; logger.error(traceback.format_exc())

        for mol in molecules_in_hf:
            mol['boltz_score']        = None
            mol['boltz_score_source'] = 'huggingface_skipped'

        all_scored.extend(molecules_with_db + newly_scored + molecules_in_hf)
        logger.info(
            f"   ✅ Batch {batch_idx+1} done: "
            f"{len(molecules_with_db)} DB + {len(newly_scored)} new + "
            f"{len(molecules_in_hf)} skipped"
        )

    return sorted(
        all_scored,
        key=lambda m: m.get('boltz_score') or float('-inf'),
        reverse=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Global MoleculeManager
# ═══════════════════════════════════════════════════════════════════════════

molecule_manager: Optional[MoleculeManager] = None


def initialize_solution(config: dict):
    global molecule_manager
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg['allowed_reaction'] = f"rxn:{RXN_ID}"
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

    Per-iteration pipeline (order matters!): $CITE_1 $CITE_2
      1. generate  n_base_samples * GENERATE_MULTIPLIER  (≈ 2000)
      2. resolve SMILES
      3. dedup seen        ← FIRST: remove already-scored molecules
      4. surrogate filter  ← SECOND: pick top BOLTZ_BUDGET from fresh pool
      5. Boltz score       ← always receives ~BOLTZ_BUDGET molecules
    """
    global molecule_manager

    config = state['config']

    def _cfg(key, default):
        return (
            config.get(key, default)
            if isinstance(config, dict)
            else getattr(config, key, default)
        )

    num_molecules          = _cfg('num_molecules', 100)
    tanimoto_max_threshold = _cfg('tanimoto_max_threshold', 0.9)
    boltz_batch_size       = _cfg('boltz_batch_size', 10)
    LIMIT_PER_REACTANT     = _cfg('limit_per_reactant', 600)

    surrogate       = SurrogateModel(max_training_samples=4000)
    use_surrogate   = True
    exploit_counter = 0
    ranker          = ComponentRanker(decay=0.90)
    plateau_counter = 0

    params = IterationParams(config=config)
    dpex   = DPEXDJAState()

    seed_df          = pd.DataFrame(columns=["name", "smiles"])
    top_pool         = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    all_pool         = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    tabued_molecules: set = set()
    iteration        = 0

    top_pool, all_pool = warm_start(
        state=state, dpex=dpex, ranker=ranker, surrogate=surrogate,
        params=params, top_pool=top_pool, all_pool=all_pool,
        num_molecules=num_molecules,
        tanimoto_max_threshold=tanimoto_max_threshold,
    )

    try:
        bt.logging.info("[Solution] Building synthon library...")
        t0 = time.time()
        params.synthon_lib        = SynthonLibrary(molecule_manager=molecule_manager)
        params.use_synthon_search = True
        bt.logging.info(
            f"[Solution] Synthon library ready in {time.time()-t0:.2f}s"
        )
    except Exception as e:
        bt.logging.warning(f"[Solution] Synthon library failed: {e}")
        params.synthon_lib        = None
        params.use_synthon_search = False

    bt.logging.info(
        f"🚀 DPEX-DJA loop | rxn={RXN_ID} | "
        f"A={len(molecule_manager.moles_A_id)} "
        f"B={len(molecule_manager.moles_B_id)} "
        f"C={len(molecule_manager.moles_C_id)} | "
        f"pipeline: generate {GENERATE_MULTIPLIER}x "
        f"→ dedup → surrogate→{BOLTZ_BUDGET} → Boltz"
    )
    bt.logging.info("Press Ctrl+C to stop")

    try:
        while True:
            iteration        += 1
            component_weights = None
            dpex.iteration    = iteration
            iter_start        = time.time()

            bt.logging.info(f"\n{'='*60}")
            bt.logging.info(
                f"[Solution] --- Iteration {iteration} [rxn={RXN_ID}] ---"
            )

            # ── Reload only NEW scored molecules every 5 iterations ───
            if iteration % 5 == 0:
                loaded_df = load_molecules_combined(RXN_ID, config)
                if not loaded_df.empty:
                    loaded_renamed = loaded_df.rename(columns={'InChIKey': 'inchi'})
                    existing_names = set(all_pool['name'].tolist())
                    new_only = loaded_renamed[
                        ~loaded_renamed['name'].isin(existing_names)
                    ]
                    if not new_only.empty:
                        all_pool = pd.concat(
                            [all_pool, new_only], ignore_index=True
                        )
                        all_pool = all_pool.sort_values(
                            by='score', ascending=False, na_position='last'
                        ).drop_duplicates(subset=['inchi'], keep='first')
                        bt.logging.info(
                            f"[Solution] +{len(new_only)} new from disk "
                            f"(all_pool={len(all_pool)})"
                        )

            # ── n_samples: always GENERATE_MULTIPLIER × base ──────────
            n_base_samples = params.get_nsamples_from_iteration(iteration)
            n_samples      = n_base_samples * GENERATE_MULTIPLIER
            bt.logging.info(
                f"[Solution] n_samples={n_samples} "
                f"({n_base_samples}×{GENERATE_MULTIPLIER}) "
                f"→ dedup → surrogate→{BOLTZ_BUDGET} → Boltz"
            )

            # ── Component weights ──────────────────────────────────────
            if not top_pool.empty:
                component_weights = build_component_weights(
                    top_pool.head(num_molecules), RXN_ID
                )
            if component_weights is not None:
                component_weights = ranker.blend_component_weights(
                    component_weights, molecule_manager
                )
            ranker.push_to_dja(molecule_manager)

            # ── Elite selection ────────────────────────────────────────
            elite_df = (
                MoleculeUtils.select_diverse_elites(
                    top_pool, min(150, len(top_pool))
                )
                if not top_pool.empty else pd.DataFrame()
            )
            elite_names = (
                elite_df["name"].tolist() if not elite_df.empty else None
            )

            # ── Exploit mode toggle ────────────────────────────────────
            if params.no_improvement_counter >= 2 and not params.use_exploit_mode:
                params.use_exploit_mode       = True
                params.no_improvement_counter = 0
                bt.logging.info("[Solution] === EXPLOIT MODE ===")
            elif params.no_improvement_counter >= 2 or exploit_counter >= 4:
                params.use_exploit_mode       = False
                exploit_counter               = 0
                params.no_improvement_counter = 0

            # ── Augment pop_B ──────────────────────────────────────────
            if not top_pool.empty:
                cols = [c for c in ('name', 'smiles', 'score')
                        if c in top_pool.columns]
                dpex.augment_pop_B(
                    top_pool[cols].head(dpex.N_B).to_dict('records')
                )

            # ── Generation ────────────────────────────────────────────
            data               = pd.DataFrame(columns=["name", "smiles"])
            data_dja           = pd.DataFrame(columns=["name"])
            data_tabu          = pd.DataFrame(columns=["name"])
            data_tabu_moves: list = []
            data_early_exploit = pd.DataFrame(columns=["name", "smiles"])
            exploited_status   = False
            exploit_summary    = None

            # ── Light early exploit (iter 1-20) ────────────────────────
            if not top_pool.empty and iteration <= 20:
                try:
                    unexploited_ee = get_top_n_unexploited(
                        top_pool.to_dict("records"),
                        params.exploited_reactants, n=2
                    )
                    if unexploited_ee:
                        t0_ee = time.time()
                        early_results, _ = run_exploit(
                            manager=molecule_manager, config=config,
                            top_molecules=unexploited_ee, top_n=1,
                            limit_per_reactant=150,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=set(),
                        )
                        if early_results:
                            data_early_exploit = pd.DataFrame(early_results)
                            bt.logging.info(
                                f"[Solution] Early exploit: "
                                f"{len(data_early_exploit)} in "
                                f"{time.time()-t0_ee:.1f}s"
                            )
                except Exception as e:
                    bt.logging.debug(f"[Solution] Early exploit skipped: {e}")

            # ── Full exploit mode ──────────────────────────────────────
            if params.use_exploit_mode and not top_pool.empty:
                bt.logging.info(
                    "[Solution] Exploit: structure-guided deep search..."
                )
                try:
                    unexploited = get_top_n_unexploited(
                        top_pool.to_dict("records"),
                        params.exploited_reactants
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
                        bt.logging.info(
                            f"[Solution] Exploit: {len(exploit_results)} "
                            f"candidates in {time.time()-t0:.1f}s"
                        )
                        if exploit_results:
                            data             = pd.DataFrame(exploit_results)
                            exploited_status = True
                        else:
                            raise Exception("Exploit returned no molecules.")
                    else:
                        raise Exception("No unexploited molecules.")
                except Exception as e:
                    bt.logging.warning(f"[Solution] Exploit skipped: {e}")
                exploit_counter += 1

            if not exploited_status:
                if not dpex.pop_A:
                    # Cold fallback
                    bt.logging.info(
                        f"[Solution] Cold init: generating "
                        f"{params.n_samples_start} random molecules"
                    )
                    raw  = generate_valid_random_molecules(
                        molecule_manager=molecule_manager,
                        n_samples=params.n_samples_start,
                        seen_molecules=params.seen_molecules,
                        component_weights=component_weights,
                    )
                    data = pd.DataFrame({"name": raw})

                else:
                    # ── DJA ───────────────────────────────────────────
                    n_dja  = int(n_samples * 0.75)
                    n_tabu = n_samples - n_dja

                    bt.logging.info(
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
                            config, pd.DataFrame({"name": raw_dja})
                        )
                        bt.logging.info(
                            f"[Solution] DJA: {len(data_dja)} validated"
                        )

                    # ── Tabu ──────────────────────────────────────────
                    if params.synthon_lib is not None and dpex.pop_B:
                        if params.score_improvement_rate > 0.05:
                            n_per_elite = 15
                        elif params.score_improvement_rate > 0.02:
                            n_per_elite = 20
                        elif params.score_improvement_rate > 0.005:
                            n_per_elite = 25
                        else:
                            n_per_elite = 50

                        bt.logging.info(
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
                                x['name'] for x in dpex.pop_B
                            }

                        if raw_tabu:
                            data_tabu = molecule_manager.validate_molecules(
                                config, pd.DataFrame({"name": raw_tabu})
                            )
                            if not data_dja.empty:
                                data_tabu = data_tabu[
                                    ~data_tabu["name"].isin(
                                        data_dja["name"].tolist()
                                    )
                                ]
                            bt.logging.info(
                                f"[Solution] Tabu: {len(data_tabu)} validated"
                            )

                    # ── Merge DJA + Tabu + early exploit ──────────────
                    parts = [
                        df for df in [data_dja, data_tabu, data_early_exploit]
                        if not df.empty
                    ]
                    if parts:
                        data = pd.concat(
                            parts, ignore_index=True
                        ).drop_duplicates(subset=["name"])
                        if not seed_df.empty:
                            data = pd.concat(
                                [data, seed_df], ignore_index=True
                            ).drop_duplicates(subset=["name"])
                            seed_df = pd.DataFrame(columns=["name", "smiles"])

                    # ── Random top-up to reach n_samples ──────────────
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

            # ── Post-generation ────────────────────────────────────────
            bt.logging.info(
                f"[Solution] {len(data)} candidates generated "
                f"in {time.time()-iter_start:.2f}s"
            )

            if data.empty:
                bt.logging.warning("[Solution] No candidates; skipping")
                await asyncio.sleep(5)
                continue

            if not seed_df.empty:
                data = pd.concat(
                    [data, seed_df], ignore_index=True
                ).drop_duplicates(subset=["name"])
                seed_df = pd.DataFrame(columns=["name", "smiles"])

            # ── Resolve SMILES ─────────────────────────────────────────
            if 'smiles' not in data.columns or data['smiles'].isna().all():
                data['smiles'] = data['name'].apply(
                    MoleculeUtils.get_smiles_from_reaction_cached
                )
            data = data[data['smiles'].notna() & (data['smiles'] != '')]

            if data.empty:
                bt.logging.warning("[Solution] No valid SMILES; skipping")
                await asyncio.sleep(5)
                continue

            # ══════════════════════════════════════════════════════════
            # ✅ STEP 1 — Dedup against seen  (MUST come BEFORE surrogate)
            #
            # seen_molecules contains 5000+ warm-start molecules.
            # If surrogate filtered FIRST, it would pick 100 from 2000
            # but many of those 100 are already seen → only ~31 reach
            # Boltz. Dedup first ensures surrogate picks 100 from a
            # pool of genuinely NEW candidates only. $CITE_1 $CITE_2
            # ══════════════════════════════════════════════════════════
            pre_dedup = len(data)
            data      = data[
                ~data["name"].isin(params.seen_molecules)
            ].reset_index(drop=True)

            dup_ratio = (pre_dedup - len(data)) / max(1, pre_dedup)
            bt.logging.info(
                f"[Solution] Dedup: {pre_dedup} → {len(data)} "
                f"({dup_ratio*100:.0f}% already seen)"
            )

            # Adaptive mutation based on dup ratio
            if dup_ratio > 0.7:
                params.mutation_prob = min(0.90, params.mutation_prob * 1.5)
            elif dup_ratio > 0.5:
                params.mutation_prob = min(0.70, params.mutation_prob * 1.3)
            elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                params.mutation_prob = max(0.10, params.mutation_prob * 0.95)

            if data.empty:
                bt.logging.error(
                    "[Solution] All duplicates after dedup; boosting diversity"
                )
                params.mutation_prob = min(0.95, params.mutation_prob * 2.0)
                params.elite_prob    = max(0.10, params.elite_prob    * 0.5)
                await asyncio.sleep(5)
                continue

            # ══════════════════════════════════════════════════════════
            # ✅ STEP 2 — Surrogate filter  (AFTER dedup, on fresh pool)
            #
            # Now data contains only unseen molecules.
            # Surrogate picks top BOLTZ_BUDGET from this fresh pool.
            # Exploit mode bypassed — exploit is already guided. $CITE_3 $CITE_4
            # ══════════════════════════════════════════════════════════
            if use_surrogate and surrogate.is_trained and not exploited_status:
                pre_sur = len(data)
                data    = surrogate.filter_candidates(
                    data,
                    n_keep=BOLTZ_BUDGET,
                    smiles_col="smiles",
                )
                bt.logging.info(
                    f"[SURROGATE] iter={iteration} | "
                    f"{pre_sur} fresh → {len(data)} "
                    f"(dropped {pre_sur - len(data)} low-quality before Boltz)"
                )
            elif not surrogate.is_trained:
                # Not yet trained — hard-cap to protect Boltz $CITE_1
                if len(data) > BOLTZ_BUDGET:
                    data = data.head(BOLTZ_BUDGET)
                    bt.logging.info(
                        f"[Solution] Surrogate not trained yet — "
                        f"hard-cap at {BOLTZ_BUDGET}"
                    )

            if data.empty:
                bt.logging.warning(
                    "[Solution] No candidates after filter; skipping"
                )
                await asyncio.sleep(5)
                continue

            # ── Boltz scoring ──────────────────────────────────────────
            bt.logging.info(
                f"[Solution] Scoring {len(data)} molecules with Boltz..."
            )
            t_score = time.time()
            scored_molecules = await score_molecules_with_boltz_batched(
                state,
                data.to_dict('records'),
                batch_size=boltz_batch_size,
            )
            bt.logging.info(
                f"[Solution] Boltz done in {time.time()-t_score:.2f}s"
            )

            # ── Build scored_df ────────────────────────────────────────
            scored_df = pd.DataFrame([
                {
                    'name':   m['name'],
                    'smiles': m.get('smiles', ''),
                    'score':  m.get('boltz_score'),
                }
                for m in scored_molecules
                if m.get('boltz_score') is not None
            ])

            if scored_df.empty:
                bt.logging.warning("[Solution] No scores returned; skipping")
                await asyncio.sleep(5)
                continue

            # ── Update ComponentRanker ─────────────────────────────────
            ranker.update(scored_df)

            # ── Update surrogate (live data, evictable) ────────────────
            if surrogate.enabled:
                surrogate.add_training_data(
                    scored_df['smiles'].tolist(),
                    scored_df['score'].tolist(),
                )
                if surrogate.total_train_size >= surrogate.min_train_size:
                    t_train = time.time()
                    surrogate.train(iteration)
                    train_time = time.time() - t_train
                    if train_time > 10.0:
                        bt.logging.warning(
                            f"[SURROGATE] Training slow "
                            f"({train_time:.2f}s) — disabling"
                        )
                        surrogate.enabled = False
                        use_surrogate     = False

            # ── Update DPEX populations ────────────────────────────────
            dja_names  = (
                set(data_dja["name"].tolist())
                if not data_dja.empty else set()
            )
            tabu_names = (
                set(data_tabu["name"].tolist())
                if not data_tabu.empty else set()
            )
            scored_for_A = (
                scored_df[scored_df["name"].isin(dja_names)]
                if dja_names else scored_df
            )
            scored_for_B = (
                scored_df[scored_df["name"].isin(tabu_names)]
                if tabu_names
                else pd.DataFrame(columns=scored_df.columns)
            )

            update_populations(dpex, scored_for_A)
            if not scored_for_B.empty:
                dpex.augment_pop_B(scored_for_B.to_dict('records'))

            if data_tabu_moves:
                for move_name in data_tabu_moves:
                    update_tabu(dpex, move_name)

            if iteration % dpex.T_ex == 0:
                dpex_exchange(dpex)

            bt.logging.debug(
                f"[DPEX] pop_A={len(dpex.pop_A)}  pop_B={len(dpex.pop_B)}"
            )

            # ── Update seen molecules ──────────────────────────────────
            params.seen_molecules = params.seen_molecules | set(
                scored_df["name"].tolist()
            )

            # ── InChIKey + pool update ─────────────────────────────────
            prev_avg = (
                top_pool.head(num_molecules)['score'].mean()
                if not top_pool.empty else None
            )

            scored_df["inchi"] = scored_df["smiles"].apply(
                MoleculeUtils.generate_inchikey
            )
            scored_df = scored_df[scored_df["inchi"] != ""]

            if not scored_df.empty:
                all_pool = (
                    pd.concat([all_pool, scored_df], ignore_index=True)
                    if not all_pool.empty else scored_df.copy()
                )
                all_pool = (
                    all_pool
                    .sort_values(by='score', ascending=False, na_position='last')
                    .drop_duplicates(subset=['inchi'], keep='first')
                )

            # ── Build top_pool with Tanimoto diversity ─────────────────
            top_pool = select_tanimoto_diverse(
                all_pool.reset_index(drop=True),
                n=num_molecules + 50,
                threshold=tanimoto_max_threshold,
                smiles_col="smiles",
            ).reset_index(drop=True)

            # ── Score improvement tracking ─────────────────────────────
            current_avg = (
                top_pool.head(num_molecules)['score'].mean()
                if not top_pool.empty else None
            )
            if current_avg is not None and prev_avg is not None:
                params.score_improvement_rate = (
                    (current_avg - prev_avg) / max(abs(prev_avg), 1e-6)
                )
            elif current_avg is not None:
                params.score_improvement_rate = 1.0

            if params.score_improvement_rate <= 0.0001:
                params.no_improvement_counter += 1
                plateau_counter               += 1
            else:
                params.no_improvement_counter = 0
                plateau_counter               = 0

            # ── Anti-plateau mutation boost ────────────────────────────
            if plateau_counter >= 5:
                params.mutation_prob = min(0.85, params.mutation_prob * 2.0)
                bt.logging.info(
                    f"[Solution] ANTI-PLATEAU: mutation_prob → "
                    f"{params.mutation_prob:.2f}"
                )
                plateau_counter = 0

            # ── Update exploited reactants ─────────────────────────────
            if (
                exploit_summary
                and 'exploited_reactant_ids' in exploit_summary
                and (
                    params.score_improvement_rate <= 0.0001
                    or not exploited_status
                )
            ):
                params.exploited_reactants.update(
                    exploit_summary['exploited_reactant_ids']
                )
                bt.logging.info(
                    f"[Solution] Exploited reactants total: "
                    f"{len(params.exploited_reactants)}"
                )

            # ── Iteration summary ──────────────────────────────────────
            iter_time = time.time() - iter_start
            pool_avg  = (
                top_pool.head(num_molecules)['score'].mean()
                if not top_pool.empty else 0.0
            )
            pool_max  = (
                top_pool['score'].max()
                if not top_pool.empty else 0.0
            )

            if exploited_status:
                mode_str = "EXPLOIT"
            elif not dpex.pop_A:
                mode_str = "INIT(cold)"
            elif params.synthon_lib is not None:
                mode_str = "DJA+TABU"
            else:
                mode_str = "DJA"

            bt.logging.info(
                f"Iter {iteration:4d} | {iter_time:6.1f}s | "
                f"Mode: {mode_str:12s} | rxn={RXN_ID} | "
                f"popA={len(dpex.pop_A):4d} popB={len(dpex.pop_B):4d} | "
                f"pool avg={pool_avg:.5f} max={pool_max:.5f} | "
                f"Δ={params.score_improvement_rate:+.5f} | "
                f"no_improve={params.no_improvement_counter} | "
                f"surrogate={'ON' if surrogate.is_trained else 'OFF'} "
                f"({surrogate.total_train_size} samples)"
            )

            if not top_pool.empty:
                best = top_pool.iloc[0]
                bt.logging.info(
                    f"   🏆 Best: {best['name']} "
                    f"(score={best['score']:.6f})"
                )

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        bt.logging.info(f"\n🛑 Stopping DPEX-DJA loop (rxn={RXN_ID})...")
        if not top_pool.empty:
            best = top_pool.iloc[0]
            bt.logging.info(
                f"Final best: {best['name']} "
                f"(score={best['score']:.6f})"
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
        f"CSV=data/rxn{rxn_id}.csv"
    )

    try:
        config = load_config()
        logger.info("✅ Config loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    initialize_solution(config)

    state: Dict[str, Any] = {
        'config':                        config,
        'current_challenge_targets':     [],
        'current_challenge_antitargets': [],
        'boltz_wrapper':                 None,
    }

    if hasattr(config, 'weekly_target'):
        state['current_challenge_targets'] = [config.weekly_target]
    elif isinstance(config, dict) and 'weekly_target' in config:
        state['current_challenge_targets'] = [config['weekly_target']]
    else:
        # config_loader does not produce weekly_target (it comes from the
        # validator's argparse), so this branch is what a miner actually takes.
        # Fall back to the configured small-molecule target, not a stale literal.
        try:
            _fallback = config['small_molecule_target'][0]
        except Exception:
            _fallback = 'P40261'
        logger.warning(
            f"⚠️  No weekly_target in config — using small_molecule_target "
            f"{_fallback}"
        )
        state['current_challenge_targets'] = [_fallback]

    if hasattr(config, 'antitargets'):
        state['current_challenge_antitargets'] = config.antitargets or []
    elif isinstance(config, dict):
        state['current_challenge_antitargets'] = config.get('antitargets', [])

    logger.info(f"🎯 Target:      {state['current_challenge_targets'][0]}")
    logger.info(f"🚫 Antitargets: {state['current_challenge_antitargets']}")

    init_score_results_db()

    logger.info("🔬 Importing BoltzWrapper...")
    if _import_boltz_wrapper() and BoltzWrapper is not None:
        try:
            state['boltz_wrapper'] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized")
        except Exception as e:
            logger.error(f"❌ BoltzWrapper init failed: {e}")
            import traceback; logger.error(traceback.format_exc())
            state['boltz_wrapper'] = None
    else:
        logger.warning("⚠️  BoltzWrapper unavailable — scoring will be skipped")

    try:
        await find_solution(state)
    except KeyboardInterrupt:
        logger.info(f"✅ rxn={rxn_id} stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback; logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())