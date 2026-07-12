"""
common.py — shared infrastructure for all 4 miner modes
(DPEX_DJA / Exhaust / Input / Cross).

Contains: fingerprint helpers, SurrogateModel, ComponentRanker,
score-results DB helpers, CSV/DB loaders, Boltz scoring wrapper,
MoleculeManager initializer, and molecule-name parsing helpers
shared by mode2 (Exhaust) and mode4 (Cross).
"""

import os
import sys
import time
import asyncio
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

# ── local imports (project modules) ────────────────────────────────────────
from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    molecule_unique_for_protein_hf,
    contains_atom_type,
)
from molecules import MoleculeManager, MoleculeUtils

# ── logging (configured once, shared across all modes) ────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("nova_miner")

# ── mutable "current rxn" context, set by configure_for_rxn() ─────────────
RXN_ID           = None
SCORE_RESULTS_DB = None
RXN_CSV          = None
TRAINING_CSV     = None


def configure_for_rxn(rxn_id: int) -> None:
    """
    Set the module-level rxn context. Must be called once at the start
    of every mode's main(), before using any DB/CSV path helpers below.
    """
    global RXN_ID, SCORE_RESULTS_DB, RXN_CSV, TRAINING_CSV
    RXN_ID           = rxn_id
    SCORE_RESULTS_DB = os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")
    RXN_CSV          = os.path.join(BASE_DIR, "data", f"rxn{rxn_id}.csv")
    TRAINING_CSV     = os.path.join(BASE_DIR, "data", "mols.csv")
    logger.info(f"✅ rxn_id           = {rxn_id}")
    logger.info(f"✅ SCORE_RESULTS_DB  = {SCORE_RESULTS_DB}")
    logger.info(f"✅ RXN_CSV           = {RXN_CSV}")
    logger.info(f"✅ TRAINING_CSV      = {TRAINING_CSV}")


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint helpers
# ═══════════════════════════════════════════════════════════════════════════

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)
_fp_cache        = {}
_mol_cache       = {}
_morgan_bv_cache = {}


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
# Molecule-name parsing helpers
# (shared by Exhaust / Cross — both need to read/write A:B[:C] ids)
# ═══════════════════════════════════════════════════════════════════════════

def parse_molecule_name(name: str) -> Optional[Tuple[int, ...]]:
    """
    'rxn:2:99232:94689'        -> (99232, 94689)
    'rxn:5:198417:228623:229120' -> (198417, 228623, 229120)
    Returns None if malformed.
    """
    parts = str(name).split(':')
    if len(parts) < 4:
        return None
    try:
        return tuple(int(p) for p in parts[2:])
    except ValueError:
        return None


def build_molecule_name(rxn_id: int, component_ids: Tuple[int, ...]) -> str:
    return "rxn:" + str(rxn_id) + ":" + ":".join(str(c) for c in component_ids)


# ═══════════════════════════════════════════════════════════════════════════
# ComponentRanker  (EMA per-reactant quality — used by mode1 only,
# kept here so mode1 file can import it)
# ═══════════════════════════════════════════════════════════════════════════

class ComponentRanker:
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
            ids = parse_molecule_name(row['name'])
            if ids is None or len(ids) < 2:
                continue
            A, B = ids[0], ids[1]
            C    = ids[2] if len(ids) > 2 else None
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
        from dpex_dja import set_ranker_weights
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
# (used by mode1 for gatekeeping, and by mode2/mode4 for prescoring)
# ═══════════════════════════════════════════════════════════════════════════

SURROGATE_KEEP_RATIO     = 0.20
SURROGATE_MIN_TRAIN_SIZE = 5000


class SurrogateModel:
    def __init__(self, max_training_samples: int = 5000):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,
        )
        self.is_trained           = False
        self.anchor_X: list       = []
        self.anchor_y: list       = []
        self.X_train: list        = []
        self.y_train: list        = []
        self.min_train_size       = SURROGATE_MIN_TRAIN_SIZE
        self.max_training_samples = max_training_samples
        self.last_train_iteration = 0
        self.train_interval       = 5
        self.enabled              = True

    def _safe_fp(self, smiles: str) -> np.ndarray:
        fp = get_morgan_fingerprint(smiles)
        return fp if fp is not None else np.zeros(2048, dtype=np.uint8)

    def add_anchor_data(self, smiles_list: list, scores: list):
        if not smiles_list:
            return
        scores_arr  = np.array(scores, dtype=float)
        finite_mask = np.isfinite(scores_arr)
        n_dropped   = int((~finite_mask).sum())
        if n_dropped:
            logger.warning(
                f"[SURROGATE] add_anchor_data: dropping {n_dropped} "
                f"non-finite score(s)"
            )
        if not finite_mask.all():
            smiles_list = [s for s, ok in zip(smiles_list, finite_mask) if ok]
            scores_arr  = scores_arr[finite_mask]

        n = len(scores_arr)
        if n == 0:
            return

        n_top    = max(1, n // 3)
        n_bottom = max(1, n // 3)
        n_rand   = max(1, n // 10)

        top_idx    = set(np.argsort(scores_arr)[-n_top:].tolist())
        bottom_idx = set(np.argsort(scores_arr)[:n_bottom].tolist())
        middle     = list(set(range(n)) - top_idx - bottom_idx)
        rand_idx   = set(
            np.random.choice(middle, min(n_rand, len(middle)), replace=False).tolist()
        ) if middle else set()

        keep  = sorted(top_idx | bottom_idx | rand_idx)
        added = 0
        for i in keep:
            fp = get_morgan_fingerprint(smiles_list[i])
            if fp is not None:
                self.anchor_X.append(fp)
                self.anchor_y.append(float(scores_arr[i]))
                added += 1
        logger.info(
            f"[SURROGATE] Anchored {added} samples "
            f"(top={n_top} bottom={n_bottom} rand={n_rand}, "
            f"{n_dropped} non-finite dropped)"
        )

    def add_training_data(self, smiles_list: list, scores: list):
        if not self.enabled or not smiles_list:
            return
        scores_arr  = np.array(scores, dtype=float)
        finite_mask = np.isfinite(scores_arr)
        n_dropped   = int((~finite_mask).sum())
        if not finite_mask.all():
            smiles_list = [s for s, ok in zip(smiles_list, finite_mask) if ok]
            scores_arr  = scores_arr[finite_mask]
        if len(scores_arr) == 0:
            return
        scores = scores_arr.tolist()

        if len(smiles_list) > 200:
            scores_array = np.array(scores)
            n            = len(scores_array)
            n_top        = min(120, n // 3)
            n_bot        = min(60,  n // 3)
            top_idx      = list(np.argsort(scores_array)[-n_top:])
            bot_idx      = list(np.argsort(scores_array)[:n_bot])
            keep_indices = sorted(set(top_idx + bot_idx))
            smiles_list  = [smiles_list[i] for i in keep_indices]
            scores       = [scores[i]      for i in keep_indices]

        for smiles, score in zip(smiles_list, scores):
            fp = get_morgan_fingerprint(smiles)
            if fp is not None:
                self.X_train.append(fp)
                self.y_train.append(score)

        if len(self.X_train) > self.max_training_samples:
            scores_array   = np.array(self.y_train)
            top_count      = int(self.max_training_samples * 0.40)
            bot_count      = int(self.max_training_samples * 0.20)
            recent_count   = int(self.max_training_samples * 0.40)
            top_indices    = list(np.argsort(scores_array)[-top_count:])
            bot_indices    = list(np.argsort(scores_array)[:bot_count])
            recent_indices = list(range(
                len(self.X_train) - recent_count, len(self.X_train)
            ))
            keep_indices   = sorted(set(top_indices + bot_indices + recent_indices))
            self.X_train   = [self.X_train[i] for i in keep_indices]
            self.y_train   = [self.y_train[i]  for i in keep_indices]

    def train(self, iteration: int = 0, force: bool = False):
        if (
            self.is_trained
            and not force
            and (iteration - self.last_train_iteration) < self.train_interval
        ):
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
            logger.info(
                f"[SURROGATE] Trained in {time.time()-t0:.2f}s on "
                f"{len(X_all)} total samples"
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
        keep_ratio: float = None,
        top_n: int = None,
        smiles_col: str = "smiles",
        min_keep: int = 1,
    ) -> pd.DataFrame:
        """
        Two modes:
          - keep_ratio: keep top X% (used by mode1)
          - top_n:      keep top N absolute (used by mode2/mode4)
        If surrogate isn't trained, falls back to returning the head
        of `data` unchanged (caller should have a hard-cap fallback).
        """
        if data.empty:
            return data
        if not self.is_trained:
            if top_n is not None:
                return data.head(top_n)
            return data

        if top_n is not None:
            n_keep = min(top_n, len(data))
        else:
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
            f"(top_n={top_n}, keep_ratio={keep_ratio})"
        )
        return filtered.reset_index(drop=True)

    @property
    def total_train_size(self) -> int:
        return len(self.anchor_X) + len(self.X_train)


# ═══════════════════════════════════════════════════════════════════════════
# Score-results DB helpers
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
    """INSERT OR REPLACE — unconditional overwrite. Used by mode1/mode2/mode4."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    if not molecules:
        return
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        skipped = 0
        to_insert = []
        for m in molecules:
            name  = m.get('name')
            score = m.get('boltz_score')
            if not name or score is None:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not np.isfinite(score_f):
                skipped += 1
                continue
            to_insert.append((name, score_f, True))
        if skipped:
            logger.warning(f"⚠️  skipped {skipped} non-finite score(s)")
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


def merge_scores_keep_max(
    rows: List[Tuple[str, float]], db_path: str = None
) -> Tuple[int, int]:
    """
    Merge policy for Mode 3 (Input) and Mode 4 (Cross):
    keep max(existing_score, new_score) per molecule_name.
    Returns (n_inserted_new, n_updated_existing).
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    if not rows:
        return 0, 0
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    n_new, n_upd = 0, 0
    for name, score in rows:
        if name is None or score is None or not np.isfinite(score):
            continue
        cursor.execute(
            "SELECT score FROM scored_molecules WHERE molecule_name = ?",
            (name,),
        )
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO scored_molecules "
                "(molecule_name, score, available) VALUES (?, ?, ?)",
                (name, float(score), True),
            )
            n_new += 1
        elif float(score) > row[0]:
            cursor.execute(
                "UPDATE scored_molecules SET score = ?, "
                "scored_at = CURRENT_TIMESTAMP WHERE molecule_name = ?",
                (float(score), name),
            )
            n_upd += 1
    conn.commit()
    conn.close()
    return n_new, n_upd


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


def get_all_scored_from_db(db_path: str = None, rxn_id: int = None) -> pd.DataFrame:
    """Load the FULL scored_molecules table (used by mode2/mode4 to find
    current top molecule(s) and top components)."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["name", "score"])
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query(
        "SELECT molecule_name AS name, score FROM scored_molecules "
        "WHERE available = 1 ORDER BY score DESC",
        conn,
    )
    conn.close()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CSV / DB loaders (warm-seed / Input mode)
# ═══════════════════════════════════════════════════════════════════════════

def load_molecules_from_csv(csv_path: str, rxn_id: int) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        logger.warning(f"⚠️  CSV not found: {csv_path}")
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
        df     = df[df['molecule_name'].str.startswith(prefix, na=False)].reset_index(drop=True)
        if df.empty:
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
        result.loc[~np.isfinite(result['score']), 'score'] = np.nan
        result = result.drop_duplicates(subset=['InChIKey'], keep='first')
        result = result.sort_values(by='score', ascending=False, na_position='last').reset_index(drop=True)
        logger.info(f"✅ CSV [{os.path.basename(csv_path)}]: {len(result)} molecules loaded")
        return result
    except Exception as e:
        logger.error(f"Error loading CSV {csv_path}: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_db(db_path: str, rxn_id: int) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT molecule_name, score FROM scored_molecules WHERE molecule_name LIKE ?",
            (f"rxn:{rxn_id}:%",)
        )
        db_results = cursor.fetchall()
        conn.close()
        if not db_results:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        rows = []
        for mol_name, score in db_results:
            try:
                score_f = float(score) if score is not None else None
                if score_f is not None and not np.isfinite(score_f):
                    continue
                smiles = MoleculeUtils.get_smiles_from_reaction_cached(mol_name)
                if not smiles:
                    continue
                inchikey = MoleculeUtils.generate_inchikey(smiles)
                if not inchikey:
                    continue
                rows.append({'name': mol_name, 'smiles': smiles, 'InChIKey': inchikey, 'score': score_f})
            except Exception:
                continue
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.drop_duplicates(subset=['InChIKey'], keep='first')
            result = result.sort_values(by='score', ascending=False, na_position='last').reset_index(drop=True)
        logger.info(f"✅ DB [{os.path.basename(db_path)}]: {len(result)} molecules loaded")
        return result
    except Exception as e:
        logger.error(f"Error loading DB {db_path}: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(rxn_id: int) -> pd.DataFrame:
    csv_path = os.path.join(BASE_DIR, "data", f"rxn{rxn_id}.csv")
    db_path  = os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")
    csv_df = load_molecules_from_csv(csv_path, rxn_id)
    db_df  = load_molecules_from_db(db_path, rxn_id)
    if csv_df.empty and db_df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    if csv_df.empty:
        return db_df
    if db_df.empty:
        return csv_df
    combined = pd.concat([csv_df, db_df], ignore_index=True)
    combined = combined.sort_values(
        by='score', ascending=False, na_position='last'
    ).drop_duplicates(subset=['InChIKey'], keep='first')
    return combined


def load_training_csv_for_surrogate(rxn_id: int) -> pd.DataFrame:
    csv_path = TRAINING_CSV
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["smiles", "score"])
    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'final_score' in df.columns and 'score' not in df.columns:
            df = df.rename(columns={'final_score': 'score'})
        df['molecule_name'] = (
            df['molecule_name'].astype(str).str.strip().str.lstrip('\ufeff')
        )
        prefix = f"rxn:{rxn_id}:"
        df     = df[df['molecule_name'].str.startswith(prefix, na=False)].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=["smiles", "score"])
        df = df[df['score'].notna()].reset_index(drop=True)
        df['smiles'] = df['molecule_name'].apply(
            MoleculeUtils.get_smiles_from_reaction_cached
        )
        df = df[df['smiles'].notna() & (df['smiles'] != '')]
        result = df[['smiles', 'score']].copy()
        result['score'] = pd.to_numeric(result['score'], errors='coerce')
        result.loc[~np.isfinite(result['score']), 'score'] = np.nan
        result = result[result['score'].notna()]
        result = result.drop_duplicates(subset=['smiles']).reset_index(drop=True)
        return result
    except Exception as e:
        logger.error(f"Error loading training CSV: {e}")
        return pd.DataFrame(columns=["smiles", "score"])


# ═══════════════════════════════════════════════════════════════════════════
# BoltzWrapper import + batched scoring
# ═══════════════════════════════════════════════════════════════════════════

BOLTZ_AVAILABLE = False
BoltzWrapper    = None


def import_boltz_wrapper():
    global BOLTZ_AVAILABLE, BoltzWrapper
    try:
        BOLTZ_SRC_DIR = os.path.join(BASE_DIR, "boltz")
        if BOLTZ_SRC_DIR not in sys.path:
            sys.path.insert(0, BOLTZ_SRC_DIR)
        from boltz_wrapper import BoltzWrapper as BW
        BoltzWrapper   = BW
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
    """Identical logic to original miner.py — DB-cache check → HF-uniqueness
    check → Boltz scoring for whatever remains → write-through to DB."""
    if state.get('boltz_wrapper') is None:
        logger.warning("BoltzWrapper not available, skipping scoring")
        return molecules
    if not molecules:
        return molecules

    logger.info(f"🔬 Scoring {len(molecules)} molecules in batches of {batch_size}...")
    init_score_results_db()

    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx   = min(start_idx + batch_size, len(molecules))
        batch     = molecules[start_idx:end_idx]

        molecules_to_score      = []
        molecules_with_db_scores = []
        molecules_in_hf          = []

        target_proteins = state.get('current_challenge_targets', [])
        primary_target  = target_proteins[0] if target_proteins else None

        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)

        for mol in batch:
            molecule_name = mol['name']
            smiles        = mol.get('smiles')
            if molecule_name in db_scores:
                mol['boltz_score']        = db_scores[molecule_name]
                mol['boltz_score_source'] = 'database'
                molecules_with_db_scores.append(mol)
                continue
            if primary_target and smiles:
                try:
                    if not molecule_unique_for_protein_hf(primary_target, smiles):
                        molecules_in_hf.append(mol)
                        continue
                except Exception:
                    pass
            molecules_to_score.append(mol)

        newly_scored_molecules = []
        if molecules_to_score:
            boltz  = state['boltz_wrapper']
            config = state['config']
            target_proteins = state.get('current_challenge_targets', [])
            if not target_proteins:
                all_scored_molecules.extend(molecules_with_db_scores)
                continue
            primary_target = target_proteins[0]
            try:
                output_dir = os.path.join(boltz.output_dir, 'boltz_results_inputs')
                processed_dir  = os.path.join(output_dir, 'processed')
                structures_dir = os.path.join(processed_dir, 'structures')
                records_dir    = os.path.join(processed_dir, 'records')
                msa_dir        = os.path.join(processed_dir, 'msa')
                predictions_dir = os.path.join(output_dir, 'predictions')
                for d in (structures_dir, records_dir, msa_dir, predictions_dir):
                    os.makedirs(d, exist_ok=True)

                valid_molecules_by_uid = {
                    0: {
                        'smiles': [mol['smiles'] for mol in molecules_to_score],
                        'names':  [mol['name']   for mol in molecules_to_score],
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
                    'small_molecule_target': config['small_molecule_target'],
                    'small_molecule_target_clip_interval': config['small_molecule_target_clip_interval'],
                    'boltz_mode': getattr(config, 'boltz_mode', 'max'),
                    'boltz_metric': getattr(config, 'boltz_metric',
                                             ['affinity_probability_binary', 'affinity_pred_value']),
                    'combination_strategy': getattr(config, 'combination_strategy',
                                                      'heavy_atom_normalization'),
                }

                def run_scoring():
                    boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)

                loop = asyncio.get_event_loop()
                t0 = time.time()
                await loop.run_in_executor(None, run_scoring)
                logger.info(f"   ✅ Boltz scoring done in {time.time()-t0:.2f}s")

                uid = 0
                smiles_to_score = {}
                final_scores = getattr(boltz, 'final_boltz_scores', {}).get(uid, {})
                if primary_target and primary_target in final_scores:
                    smiles_to_score = final_scores[primary_target].copy()
                elif final_scores:
                    smiles_to_score = next(iter(final_scores.values())).copy()
                elif hasattr(boltz, 'per_molecule_metric') and uid in boltz.per_molecule_metric:
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()

                target_scores_list = None
                target_scores = score_dict[uid].get('target_scores', [[]])
                if target_scores and len(target_scores[0]) > 0:
                    target_scores_list = (
                        target_scores[0] if isinstance(target_scores[0], list)
                        else [target_scores[0]]
                    )
                avg_score = None
                if not smiles_to_score and not target_scores_list:
                    avg_score = score_dict[uid].get('boltz_score')

                for mol_idx, mol in enumerate(molecules_to_score):
                    smiles = mol['smiles']
                    score  = None
                    if smiles in smiles_to_score:
                        score = smiles_to_score[smiles]
                    elif target_scores_list and mol_idx < len(target_scores_list):
                        score = target_scores_list[mol_idx]
                    if score is None and avg_score is not None:
                        score = avg_score
                    mol['boltz_score'] = score
                    if score is not None:
                        newly_scored_molecules.append(mol)

                if newly_scored_molecules:
                    write_scores_to_db(newly_scored_molecules)
            except Exception as e:
                logger.error(f"❌ Error scoring batch with Boltz: {e}")
                import traceback; logger.error(traceback.format_exc())

        batch_results = molecules_with_db_scores + newly_scored_molecules
        for mol in molecules_in_hf:
            mol['boltz_score']        = None
            mol['boltz_score_source'] = 'huggingface_skipped'
            batch_results.append(mol)
        all_scored_molecules.extend(batch_results)

    scored_molecules = sorted(
        all_scored_molecules,
        key=lambda m: m.get('boltz_score') if m.get('boltz_score') is not None else float('-inf'),
        reverse=True,
    )
    logger.info(f"✅ Batch scoring complete: {len(scored_molecules)} total")
    return scored_molecules


# ═══════════════════════════════════════════════════════════════════════════
# MoleculeManager init  (shared by all modes)
# ═══════════════════════════════════════════════════════════════════════════

molecule_manager: Optional[MoleculeManager] = None


def initialize_solution(config: dict, rxn_id: int):
    global molecule_manager
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg['allowed_reaction'] = f"rxn:{rxn_id}"
    molecule_manager = MoleculeManager(config=cfg, db_path=DB_PATH)
    logger.info(
        f"✅ MoleculeManager locked to rxn={rxn_id} | "
        f"A={len(molecule_manager.moles_A_id)} | "
        f"B={len(molecule_manager.moles_B_id)} | "
        f"C={len(molecule_manager.moles_C_id)}"
    )
    return molecule_manager


def build_state(config: dict) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        'config':                        config,
        'current_challenge_targets':     config["small_molecule_target"],
        'current_challenge_targets_clip_interval': config["small_molecule_target_clip_interval"],
        'current_challenge_antitargets': [],
        'boltz_wrapper':                 None,
    }
    init_score_results_db()
    if import_boltz_wrapper() and BoltzWrapper is not None:
        try:
            state['boltz_wrapper'] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized")
        except Exception as e:
            logger.error(f"❌ BoltzWrapper init failed: {e}")
    else:
        logger.warning("⚠️  BoltzWrapper unavailable — scoring will be skipped")
    return state
