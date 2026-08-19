"""
miner.py — Blueprint DPEX-DJA search adapted for small-molecule (SN68 Compound).

Updates vs Blueprint:
  1. Config from config/config.yaml via load_config()
  2. Single fixed reaction via --rxn_id (no multi-rxn / rxn weights)
  3. Boltz2 scoring (batch size 10) instead of PSICHIC
  4. Surrogate keep ratio = 0.2
  5. Entropy removed entirely
  6. Per-iteration: append log.txt + avg/max score graphs
  7. Persist scores to score_results_{rxn}.sqlite with iteration column
"""

from __future__ import annotations

import os
import sys
import time
import json
import asyncio
import logging
import sqlite3
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import bittensor as bt
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# ── paths ────────────────────────────────────────────────────────────────
COMPOUND_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(COMPOUND_DIR, ".."))
# BASE_DIR first for config/utils/boltz; COMPOUND_DIR last so it wins over
# the top-level nova-4090/tools/ package when importing local tools.py etc.
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, COMPOUND_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

RXN_ID: Optional[int] = None
SCORE_RESULTS_DB: Optional[str] = None
# Target identity for the shared score DB, stamped from config in main().
# Kept in orchestrator.py's format so both writers agree on target_key.
TARGET_KEY: Optional[str] = None
TARGET_LABEL: Optional[str] = None
LOG_PATH = os.path.join(BASE_DIR, "log.txt")
GRAPH_AVG_PATH = os.path.join(BASE_DIR, "pool_avg_score.png")
GRAPH_MAX_PATH = os.path.join(BASE_DIR, "pool_max_score.png")

# ── constants ────────────────────────────────────────────────────────────
LIMIT_PER_REACTANT = 600
BOLTZ_BATCH_SIZE = 10
SURROGATE_KEEP_RATIO = 0.2
SURROGATE_MIN_TRAIN_SIZE = 3000  # filtering requires training samples > this
SURROGATE_ACTIVE_AFTER_ITER = 5   # filtering requires iteration > this
TOP_AVG_K = 100  # progress signal: mean of top-K pool scores

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fp_cache: Dict[str, np.ndarray] = {}
_mol_cache: Dict[str, Any] = {}
_morgan_bv_cache: Dict[str, Any] = {}

from config.config_loader import load_config
from utils import get_brenk_matches
from molecules import MoleculeManager, MoleculeUtils
import score_store
from tools import (
    IterationParams,
    SynthonLibrary,
    generate_valid_random_molecules,
    cpu_random_candidates_with_similarity,
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

BOLTZ_AVAILABLE = False
BoltzWrapper = None
molecule_manager: Optional[MoleculeManager] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    global RXN_ID, SCORE_RESULTS_DB, GRAPH_AVG_PATH, GRAPH_MAX_PATH

    parser = argparse.ArgumentParser(
        description="Blueprint→SM DPEX-DJA miner (single fixed reaction)"
    )
    parser.add_argument("--rxn_id", type=int, required=True, help="Reaction ID 1-5")
    args = parser.parse_args()

    RXN_ID = args.rxn_id
    SCORE_RESULTS_DB = os.path.join(BASE_DIR, f"score_results_{RXN_ID}.sqlite")
    GRAPH_AVG_PATH = os.path.join(BASE_DIR, f"pool_avg_score_rxn{RXN_ID}.png")
    GRAPH_MAX_PATH = os.path.join(BASE_DIR, f"pool_max_score_rxn{RXN_ID}.png")

    logger.info(f"✅ rxn_id           = {RXN_ID}")
    logger.info(f"✅ SCORE_RESULTS_DB = {SCORE_RESULTS_DB}")
    logger.info(
        f"✅ SURROGATE keep={SURROGATE_KEEP_RATIO} | "
        f"active when iter>{SURROGATE_ACTIVE_AFTER_ITER} and "
        f"train_samples>{SURROGATE_MIN_TRAIN_SIZE}"
    )
    logger.info(f"✅ LOG              = {LOG_PATH}")
    return RXN_ID


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprints / diversity
# ═══════════════════════════════════════════════════════════════════════════

def get_mol(smiles: str):
    if smiles in _mol_cache:
        return _mol_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    _mol_cache[smiles] = mol
    return mol


def passes_brenk_filter(smiles: str) -> bool:
    """Reject molecules the validator's BRENK structural-alert filter kills."""
    mol = get_mol(smiles)
    if mol is None:
        return False
    reasons = get_brenk_matches(mol)
    if reasons:
        logger.debug(f"[BRENK] rejected {smiles}: {'; '.join(reasons)}")
        return False
    return True


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


def select_tanimoto_diverse(
    df: pd.DataFrame,
    n: int,
    threshold: float = 1.0,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """threshold=1.0 → effectively pure top-by-score (no diversity gate)."""
    if df.empty or n <= 0:
        return df.head(0)
    if threshold >= 1.0:
        return df.head(n)

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
# ComponentRanker (single-rxn)
# ═══════════════════════════════════════════════════════════════════════════

class ComponentRanker:
    def __init__(self, decay: float = 0.90):
        self.decay = decay
        self.q_A: dict = {}
        self.q_B: dict = {}
        self.q_C: dict = {}

    def _ema(self, store: dict, key: int, score: float):
        if key in store:
            old, cnt = store[key]
            store[key] = (self.decay * old + (1 - self.decay) * score, cnt + 1)
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
                A, B = int(parts[2]), int(parts[3])
                C = int(parts[4]) if len(parts) > 4 else None
            except (ValueError, IndexError):
                continue
            self._ema(self.q_A, A, score)
            self._ema(self.q_B, B, score)
            if C is not None:
                self._ema(self.q_C, C, score)

    def compute_weights(self, pool, q):
        if not q:
            return None
        w = np.array([max(0.01, q[mid][0]) if mid in q else 0.05 for mid in pool])
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

    def blend_component_weights(self, component_weights: dict, manager: MoleculeManager) -> dict:
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
# Surrogate (Blueprint RF, keep_ratio=0.2)
# ═══════════════════════════════════════════════════════════════════════════

class SurrogateModel:
    def __init__(self, max_training_samples: int = 10000):
        self.model = RandomForestRegressor(
            n_estimators=70,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,
        )
        self.is_trained = False
        self.X_train: list = []
        self.y_train: list = []
        # Train once we have more than SURROGATE_MIN_TRAIN_SIZE samples.
        self.min_train_size = SURROGATE_MIN_TRAIN_SIZE + 1
        self.max_training_samples = max_training_samples
        self.last_train_iteration = 0
        self.train_interval = 2
        self.enabled = True

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    def ready_to_filter(self, iteration: int) -> bool:
        """Hard gates: iter > 5 and training samples > 4000."""
        return (
            self.enabled
            and self.is_trained
            and self.train_size > SURROGATE_MIN_TRAIN_SIZE
            and iteration > SURROGATE_ACTIVE_AFTER_ITER
        )

    def add_training_data(self, smiles_list: list, scores: list):
        if not self.enabled:
            return
        if len(smiles_list) > 600:
            scores_array = np.array(scores)
            top_indices = np.argsort(scores_array)[-500:]
            mid_low = np.argsort(scores_array)[: min(200, len(scores_array) // 2)]
            sample_low = (
                list(np.random.choice(mid_low, min(100, len(mid_low)), replace=False))
                if len(mid_low) > 0
                else []
            )
            keep_indices = sorted(set(list(top_indices) + sample_low))
            smiles_list = [smiles_list[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]

        for smiles, score in zip(smiles_list, scores):
            if score is None or not np.isfinite(float(score)):
                continue
            fp = get_morgan_fingerprint(smiles)
            if fp is not None:
                self.X_train.append(fp)
                self.y_train.append(float(score))

        if len(self.X_train) > self.max_training_samples:
            scores_array = np.array(self.y_train)
            top_count = int(self.max_training_samples * 0.5)
            recent_count = int(self.max_training_samples * 0.5)
            top_indices = np.argsort(scores_array)[-top_count:]
            recent_indices = list(range(len(self.X_train) - recent_count, len(self.X_train)))
            keep_indices = sorted(set(list(top_indices) + recent_indices))
            self.X_train = [self.X_train[i] for i in keep_indices]
            self.y_train = [self.y_train[i] for i in keep_indices]

    def train(self, iteration: int = 0):
        if self.is_trained and (iteration - self.last_train_iteration) < self.train_interval:
            return
        if len(self.X_train) < self.min_train_size:
            self.is_trained = False
            return
        try:
            t0 = time.time()
            self.model.fit(np.array(self.X_train), np.array(self.y_train))
            self.is_trained = True
            self.last_train_iteration = iteration
            logger.info(
                f"[SURROGATE] Trained in {time.time() - t0:.2f}s on {len(self.X_train)} samples"
            )
        except Exception as e:
            logger.warning(f"Surrogate training failed: {e}")
            self.is_trained = False

    def predict(self, smiles_list: list) -> np.ndarray:
        if not self.is_trained:
            return np.array([0.0] * len(smiles_list))
        try:
            fps = []
            for smiles in smiles_list:
                fp = get_morgan_fingerprint(smiles)
                fps.append(fp if fp is not None else np.zeros(2048, dtype=np.uint8))
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
            f"[SURROGATE] Filtered {len(data)} → {len(filtered)} "
            f"(keep_ratio={keep_ratio})"
        )
        return filtered.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Score DB — delegated to score_store so the schema, upsert semantics and
# target guard stay identical to orchestrator.py's ScoreStore.
# ═══════════════════════════════════════════════════════════════════════════

def init_score_results_db(db_path: str = None) -> None:
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    score_store.init_score_results_db(
        db_path,
        rxn_id=RXN_ID,
        target_key=TARGET_KEY,
        target_label=TARGET_LABEL,
    )
    logger.info(f"✅ Score DB ready: {db_path}")


def write_scores_to_db(
    molecules: List[Dict[str, Any]],
    iteration: int,
    db_path: str = None,
) -> None:
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    if not molecules:
        return
    # `iteration` lands in both the iteration and round columns, matching how
    # orchestrator stamps round_no, so COALESCE(round,iteration) always resolves.
    n = score_store.write_scores_to_db(
        db_path,
        molecules,
        rxn_id=RXN_ID,
        round_no=iteration,
        target_key=TARGET_KEY,
        target_label=TARGET_LABEL,
        source="dpex_dja",
    )
    if n:
        logger.info(f"✅ Wrote {n} scores (iter={iteration}) → {db_path}")
    skipped = len(molecules) - n
    if skipped > 0:
        logger.warning(f"⚠️ Skipped {skipped} unusable/non-finite scores")


def batch_get_scores_from_db(
    molecule_names: List[str],
    db_path: str = None,
) -> Dict[str, float]:
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    return score_store.batch_get_scores_from_db(db_path, molecule_names)


def count_scored_in_db(db_path: str = None) -> int:
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    return score_store.count_scored(db_path)


# ═══════════════════════════════════════════════════════════════════════════
# Logging + graphs
# ═══════════════════════════════════════════════════════════════════════════

def append_iteration_log(
    iteration: int,
    iter_time: float,
    total_time: float,
    n_scored_iter: int,
    n_scored_total: int,
    mode: str,
    pop_a: int,
    pop_b: int,
    pool_avg: float,
    pool_max: float,
) -> None:
    line = (
        f"Iteration {iteration} | {iter_time:.1f}s | Total: {total_time:.0f}s | "
        f"{n_scored_iter} | Total: {n_scored_total} | Mode: {mode} | "
        f"popA={pop_a} popB={pop_b} | "
        f"Pool: avg={pool_avg:.4f} max={pool_max:.4f}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info(line.rstrip())
    print(line.rstrip())


def update_score_graphs(
    history_iters: List[int],
    history_avg: List[float],
    history_max: List[float],
) -> None:
    if not history_iters:
        return
    if not MATPLOTLIB_AVAILABLE:
        logger.warning(
            "matplotlib not installed — skip graphs "
            "(pip install matplotlib)"
        )
        return
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_iters, history_avg, color="#1f77b4", linewidth=1.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Pool avg score")
        ax.set_title(f"Pool average score vs iteration (rxn={RXN_ID})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(GRAPH_AVG_PATH, dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_iters, history_max, color="#d62728", linewidth=1.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Pool max score")
        ax.set_title(f"Pool max score vs iteration (rxn={RXN_ID})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(GRAPH_MAX_PATH, dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Graph update failed: {e}")


def _pool_stats(top_pool: pd.DataFrame) -> Tuple[float, float]:
    if top_pool.empty or "score" not in top_pool.columns:
        return 0.0, 0.0
    pool = top_pool.copy()
    pool["score"] = pd.to_numeric(pool["score"], errors="coerce")
    pool = pool[np.isfinite(pool["score"])].dropna(subset=["score"])
    if pool.empty:
        return 0.0, 0.0
    pool = pool.sort_values("score", ascending=False)
    k = min(TOP_AVG_K, len(pool))
    return float(pool.head(k)["score"].mean()), float(pool["score"].max())


# ═══════════════════════════════════════════════════════════════════════════
# Boltz
# ═══════════════════════════════════════════════════════════════════════════

def _import_boltz_wrapper() -> bool:
    global BOLTZ_AVAILABLE, BoltzWrapper
    try:
        boltz_src = os.path.join(BASE_DIR, "boltz")
        if boltz_src not in sys.path:
            sys.path.insert(0, boltz_src)
        from boltz_wrapper import BoltzWrapper as BW

        BoltzWrapper = BW
        BOLTZ_AVAILABLE = True
        logger.info("✅ BoltzWrapper imported")
        return True
    except Exception as e:
        logger.warning(f"⚠️ BoltzWrapper import failed: {e}")
        return False


async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    iteration: int,
    batch_size: int = BOLTZ_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Score with Boltz in batches of `batch_size`; write DB after each batch."""
    if state.get("boltz_wrapper") is None:
        logger.warning("BoltzWrapper unavailable — skipping scoring")
        return molecules
    if not molecules:
        return molecules

    init_score_results_db()
    all_scored: List[Dict[str, Any]] = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    config = state["config"]
    target_proteins = state.get("current_challenge_targets") or []
    primary_target = target_proteins[0] if target_proteins else None

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(molecules))
        batch = molecules[start:end]
        logger.info(
            f"📦 Batch {batch_idx + 1}/{total_batches}: scoring {len(batch)} molecules"
        )

        names = [m["name"] for m in batch]
        db_scores = batch_get_scores_from_db(names)

        from_db: List[Dict[str, Any]] = []
        to_score: List[Dict[str, Any]] = []

        for mol in batch:
            name = mol["name"]
            if name in db_scores:
                mol = dict(mol)
                mol["boltz_score"] = db_scores[name]
                mol["score"] = db_scores[name]
                mol["boltz_score_source"] = "database"
                from_db.append(mol)
                continue
            to_score.append(mol)

        newly: List[Dict[str, Any]] = []
        if to_score and primary_target:
            boltz = state["boltz_wrapper"]
            try:
                output_dir = os.path.join(boltz.output_dir, "boltz_results_inputs")
                os.makedirs(os.path.join(output_dir, "processed", "structures"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "processed", "records"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "processed", "msa"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

                valid_molecules_by_uid = {
                    0: {
                        "smiles": [m["smiles"] for m in to_score],
                        "names": [m["name"] for m in to_score],
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

                t0 = time.time()

                def run_scoring():
                    boltz.score_molecules(
                        valid_molecules_by_uid, score_dict, subnet_config
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, run_scoring)
                logger.info(f"   ✅ Boltz batch done in {time.time() - t0:.2f}s")

                smiles_to_score: Dict[str, float] = {}
                final_scores = getattr(boltz, "final_boltz_scores", {}).get(0, {})
                if primary_target and primary_target in final_scores:
                    smiles_to_score = final_scores[primary_target].copy()
                elif final_scores:
                    smiles_to_score = next(iter(final_scores.values())).copy()

                target_scores = score_dict[0].get("target_scores", [[]])
                target_list = None
                if target_scores and len(target_scores[0]) > 0:
                    target_list = (
                        target_scores[0]
                        if isinstance(target_scores[0], list)
                        else [target_scores[0]]
                    )

                for mol_idx, mol in enumerate(to_score):
                    smiles = mol["smiles"]
                    score = None
                    if smiles in smiles_to_score:
                        score = smiles_to_score[smiles]
                    elif target_list and mol_idx < len(target_list):
                        score = target_list[mol_idx]
                    mol = dict(mol)
                    mol["boltz_score"] = score
                    mol["score"] = score
                    if score is not None and np.isfinite(float(score)):
                        newly.append(mol)

                if newly:
                    write_scores_to_db(newly, iteration=iteration)

            except Exception as e:
                logger.error(f"❌ Boltz batch error: {e}")
                import traceback
                logger.error(traceback.format_exc())

        batch_results = from_db + newly
        all_scored.extend(batch_results)

        logger.info(
            f"   ✅ Batch {batch_idx + 1}/{total_batches} complete: "
            f"{len(from_db)} from DB, {len(newly)} newly scored"
        )
        if batch_results:
            logger.info(f"   Batch {batch_idx + 1} scored results:")
            print(f"   Batch {batch_idx + 1}/{total_batches} scored results:")
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
                    line = f"      {name}: {float(score):.6f} [{source}]"
                else:
                    line = f"      {name}: skipped [{source}]"
                logger.info(line)
                print(line)

    return all_scored


# ═══════════════════════════════════════════════════════════════════════════
# Init
# ═══════════════════════════════════════════════════════════════════════════

def initialize_solution(config: dict):
    global molecule_manager
    cfg = dict(config)
    cfg["allowed_reaction"] = f"rxn:{RXN_ID}"
    cfg["max_heavy_atoms"] = cfg.get("max_heavy_atoms", 40)
    molecule_manager = MoleculeManager(config=cfg, db_path=DB_PATH)
    logger.info(
        f"✅ MoleculeManager locked to rxn={RXN_ID} | "
        f"A={len(molecule_manager.moles_A_id)} "
        f"B={len(molecule_manager.moles_B_id)} "
        f"C={len(molecule_manager.moles_C_id) if molecule_manager.moles_C_id else 0}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main search loop
# ═══════════════════════════════════════════════════════════════════════════

async def find_solution(state: Dict[str, Any]) -> None:
    global molecule_manager

    config = state["config"]
    cfg = dict(config)
    cfg["allowed_reaction"] = f"rxn:{RXN_ID}"
    cfg["max_heavy_atoms"] = cfg.get("max_heavy_atoms", 40)

    n_workers = os.cpu_count() or 1
    surrogate = SurrogateModel(max_training_samples=10000)
    use_surrogate = True
    exploit_counter = 0
    ranker = ComponentRanker(decay=0.90)
    plateau_counter = 0

    params = IterationParams(config=cfg)
    dpex = DPEXDJAState()

    seed_df = pd.DataFrame(columns=["name", "smiles"])
    top_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    all_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score"])
    tabued_molecules: set = set()

    history_iters: List[int] = []
    history_avg: List[float] = []
    history_max: List[float] = []

    time_start = time.time()
    total_scored_session = 0
    iteration = 0

    try:
        params.synthon_lib = SynthonLibrary(molecule_manager=molecule_manager)
        params.use_synthon_search = True
        logger.info("[Solution] Synthon library ready")
    except Exception as e:
        logger.warning(f"[Solution] Synthon library failed: {e}")
        params.synthon_lib = None

    with ProcessPoolExecutor(max_workers=n_workers) as cpu_executor:
        while True:
            iteration += 1
            sur_filter = False
            component_weights = None
            dpex.iteration = iteration
            iteration_start = time.time()

            logger.info(f"[Solution] --- Iteration {iteration} [rxn={RXN_ID}] ---")

            n_base_samples = params.get_nsamples_from_iteration(iteration)
            # 7× generation only when surrogate can actually filter.
            # If surrogate is inactive: always base samples (no 7×).
            # If surrogate is active:
            #   - every 4th iter → base samples, no filter
            #   - other iters    → 7× samples, filter intent
            surrogate_active = use_surrogate and surrogate.ready_to_filter(iteration)
            if surrogate_active and iteration % 4 != 0:
                n_samples = n_base_samples * 7
                sur_filter = True
            else:
                n_samples = n_base_samples
                sur_filter = False

            logger.info(
                f"[Solution] n_samples={n_samples} "
                f"(base={n_base_samples}, surrogate_active={surrogate_active}, "
                f"sur_filter_intent={sur_filter}, "
                f"train_size={surrogate.train_size})"
            )

            if not top_pool.empty:
                component_weights = build_component_weights(
                    top_pool.head(TOP_AVG_K), RXN_ID
                )
            if component_weights is not None and iteration > 2:
                component_weights = ranker.blend_component_weights(
                    component_weights, molecule_manager
                )
            if iteration > 2:
                ranker.push_to_dja(molecule_manager)

            elite_df = (
                MoleculeUtils.select_diverse_elites(
                    top_pool, min(150, len(top_pool))
                )
                if not top_pool.empty
                else pd.DataFrame()
            )
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            if params.no_improvement_counter >= 2 and not params.use_exploit_mode:
                params.use_exploit_mode = True
                params.no_improvement_counter = 0
                logger.info("[Solution] === EXPLOIT MODE ===")
            elif params.no_improvement_counter >= 2 or exploit_counter >= 4:
                params.use_exploit_mode = False
                exploit_counter = 0
                params.no_improvement_counter = 0

            if not top_pool.empty:
                cols = [c for c in ("name", "smiles", "score") if c in top_pool.columns]
                dpex.augment_pop_B(top_pool[cols].head(dpex.N_B).to_dict("records"))

            data = pd.DataFrame(columns=["name", "smiles"])
            data_dja = pd.DataFrame(columns=["name"])
            data_tabu = pd.DataFrame(columns=["name"])
            data_tabu_moves: list = []
            data_early_exploit = pd.DataFrame(columns=["name", "smiles"])
            exploited_status = False
            exploit_summary = None

            # early exploit
            if not top_pool.empty and 2 < iteration <= 20:
                try:
                    unexploited_ee = get_top_n_unexploited(
                        top_pool.to_dict("records"),
                        params.exploited_reactants,
                        n=2,
                    )
                    if unexploited_ee:
                        early_results, _ = run_exploit(
                            manager=molecule_manager,
                            config=cfg,
                            top_molecules=unexploited_ee,
                            top_n=1,
                            limit_per_reactant=150,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=set(),
                        )
                        if early_results:
                            data_early_exploit = pd.DataFrame(early_results)
                except Exception as e:
                    logger.debug(f"Early exploit skipped: {e}")

            if params.use_exploit_mode:
                try:
                    unexploited = get_top_n_unexploited(
                        top_pool.to_dict("records"),
                        params.exploited_reactants,
                    )
                    if unexploited:
                        exploit_results, exploit_summary = run_exploit(
                            manager=molecule_manager,
                            config=cfg,
                            top_molecules=unexploited,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=params.exploited_reactants,
                        )
                        if exploit_results:
                            data = pd.DataFrame(exploit_results)
                            exploited_status = True
                        else:
                            raise Exception("Exploit returned no molecules.")
                    else:
                        raise Exception("No unexploited top molecules.")
                except Exception as e:
                    logger.warning(f"[Solution] Exploit skipped: {e}")
                exploit_counter += 1
                sur_filter = False

            if not exploited_status:
                if iteration == 1 or not dpex.pop_A:
                    data = generate_valid_random_molecules(
                        config=cfg,
                        manager=molecule_manager,
                        n_samples=params.n_samples_start,
                        mutation_prob=0,
                        elite_prob=0,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=params.seen_molecules,
                        elite_names=None,
                        component_weights=component_weights,
                    )
                else:
                    n_dja = int(n_samples * 0.75)
                    raw_dja = dja_generate(
                        state=dpex,
                        manager=molecule_manager,
                        n_samples=n_dja,
                        avoid=params.seen_molecules,
                    )
                    if not raw_dja.empty:
                        data_dja = molecule_manager.validate_molecules(cfg, raw_dja)

                    if params.synthon_lib is not None and dpex.pop_B:
                        global_best = (
                            top_pool["score"].max()
                            if not top_pool.empty
                            else float("-inf")
                        )
                        if params.score_improvement_rate > 0.05:
                            n_per_elite, n_elites = 15, 30
                        elif params.score_improvement_rate > 0.02:
                            n_per_elite, n_elites = 20, 40
                        elif params.score_improvement_rate > 0.005:
                            n_per_elite, n_elites = 25, 60
                        else:
                            n_per_elite, n_elites = 50, 100

                        raw_tabu, data_tabu_moves = tabu_generate(
                            state=dpex,
                            synthon_lib=params.synthon_lib,
                            manager=molecule_manager,
                            avoid=params.seen_molecules,
                            k_per_elite=n_per_elite,
                            k_elites=n_elites,
                            global_best_score=global_best,
                            tabued_molecules=tabued_molecules,
                        )
                        if params.score_improvement_rate <= 0.005:
                            tabued_molecules |= {x["name"] for x in dpex.pop_B}
                        if not raw_tabu.empty:
                            data_tabu = molecule_manager.validate_molecules(
                                cfg, raw_tabu
                            )
                            if not data_dja.empty:
                                data_tabu = data_tabu[
                                    ~data_tabu["name"].isin(data_dja["name"].tolist())
                                ]

                    parts = [
                        df
                        for df in [data_dja, data_tabu, data_early_exploit]
                        if not df.empty
                    ]
                    if parts:
                        data = pd.concat(parts, ignore_index=True).drop_duplicates(
                            subset=["name"]
                        )
                        if not seed_df.empty:
                            data = pd.concat(
                                [data, seed_df], ignore_index=True
                            ).drop_duplicates(subset=["name"])
                            seed_df = pd.DataFrame(columns=["name", "smiles"])

                    traditional_df = generate_valid_random_molecules(
                        config=cfg,
                        manager=molecule_manager,
                        n_samples=int(n_samples * 0.5),
                        mutation_prob=params.mutation_prob,
                        elite_prob=params.elite_prob,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=params.seen_molecules,
                        elite_names=elite_names,
                        component_weights=component_weights,
                    )
                    data = pd.concat(
                        [data, traditional_df], ignore_index=True
                    ).drop_duplicates(subset=["name"])

            if data.empty:
                logger.warning("[Solution] No candidates; sleeping")
                await asyncio.sleep(2)
                continue

            if "smiles" not in data.columns or data["smiles"].isna().all():
                data["smiles"] = data["name"].map(
                    MoleculeUtils.get_smiles_from_reaction_cached
                )
            data = data[data["smiles"].notna() & (data["smiles"] != "")].reset_index(
                drop=True
            )
            if data.empty:
                await asyncio.sleep(2)
                continue

            # BRENK structural alerts — the validator discards a whole
            # submission on a single match, so never spend Boltz time on them.
            n_pre_brenk = len(data)
            data = data[data["smiles"].map(passes_brenk_filter)].reset_index(drop=True)
            n_brenk = n_pre_brenk - len(data)
            if n_brenk:
                logger.info(f"[BRENK] dropped {n_brenk}/{n_pre_brenk} candidates")
            if data.empty:
                await asyncio.sleep(2)
                continue

            # dedup
            filtered = data[~data["name"].isin(params.seen_molecules)]
            dup_ratio = (len(data) - len(filtered)) / max(1, len(data))
            if dup_ratio > 0.7:
                params.mutation_prob = min(0.90, params.mutation_prob * 1.5)
            elif dup_ratio > 0.5:
                params.mutation_prob = min(0.70, params.mutation_prob * 1.3)
            elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                params.mutation_prob = max(0.10, params.mutation_prob * 0.95)
            data = filtered.reset_index(drop=True)
            if data.empty:
                params.mutation_prob = min(0.95, params.mutation_prob * 2.0)
                params.elite_prob = max(0.10, params.elite_prob * 0.5)
                await asyncio.sleep(2)
                continue

            # Surrogate keep 0.2
            # - size override only if surrogate is active
            # - exploit mode: always inactive
            if surrogate_active and len(data) > n_base_samples * 2:
                sur_filter = True
            if exploited_status:
                sur_filter = False

            if use_surrogate and sur_filter and surrogate_active:
                data = surrogate.filter_candidates(
                    data,
                    keep_ratio=SURROGATE_KEEP_RATIO,
                    smiles_col="smiles",
                )
                logger.info(
                    f"[SURROGATE] ACTIVE iter={iteration} "
                    f"train_size={surrogate.train_size} "
                    f"(>{SURROGATE_MIN_TRAIN_SIZE})"
                )
            elif sur_filter:
                logger.info(
                    f"[SURROGATE] SKIP filter iter={iteration} "
                    f"train_size={surrogate.train_size} "
                    f"trained={surrogate.is_trained} "
                    f"(need iter>{SURROGATE_ACTIVE_AFTER_ITER} and "
                    f"train_size>{SURROGATE_MIN_TRAIN_SIZE})"
                )

            # optional CPU neighborhood seeds (overlap with scoring)
            cpu_futures = []
            if (
                not top_pool.empty
                and iteration > 10
                and iteration % 2 == 0
                and not params.use_exploit_mode
            ):
                cpu_futures.append(
                    (
                        cpu_executor.submit(
                            cpu_random_candidates_with_similarity,
                            molecule_manager,
                            30,
                            cfg,
                            top_pool.head(100)[["name", "smiles"]],
                            params.seen_molecules,
                            0.65,
                        ),
                        "top100",
                    )
                )

            # Boltz score
            scored_molecules = await score_molecules_with_boltz_batched(
                state,
                data.to_dict("records"),
                iteration=iteration,
                batch_size=BOLTZ_BATCH_SIZE,
            )

            scored_df = pd.DataFrame(
                [
                    {
                        "name": m["name"],
                        "smiles": m.get("smiles", ""),
                        "score": m.get("boltz_score"),
                    }
                    for m in scored_molecules
                    if m.get("boltz_score") is not None
                ]
            )
            if not scored_df.empty:
                scored_df["score"] = pd.to_numeric(scored_df["score"], errors="coerce")
                scored_df = scored_df[np.isfinite(scored_df["score"])].dropna(
                    subset=["score"]
                )

            n_scored_iter = len(scored_df)
            total_scored_session += n_scored_iter
            n_scored_total = max(count_scored_in_db(), total_scored_session)

            if scored_df.empty:
                logger.warning("[Solution] No finite Boltz scores this iteration")
                # still log empty-ish progress
                pool_avg, pool_max = _pool_stats(top_pool)
                iter_time = time.time() - iteration_start
                total_time = time.time() - time_start
                mode_str = (
                    "EXPLOIT"
                    if exploited_status
                    else (
                        "INIT"
                        if iteration == 1 or not dpex.pop_A
                        else ("DJA+TABU" if params.synthon_lib else "DJA")
                    )
                )
                append_iteration_log(
                    iteration,
                    iter_time,
                    total_time,
                    0,
                    n_scored_total,
                    mode_str,
                    len(dpex.pop_A),
                    len(dpex.pop_B),
                    pool_avg,
                    pool_max,
                )
                history_iters.append(iteration)
                history_avg.append(pool_avg)
                history_max.append(pool_max)
                update_score_graphs(history_iters, history_avg, history_max)
                await asyncio.sleep(1)
                continue

            ranker.update(scored_df)
            if surrogate.enabled:
                surrogate.add_training_data(
                    scored_df["smiles"].tolist(), scored_df["score"].tolist()
                )
                if len(surrogate.X_train) >= surrogate.min_train_size:
                    t_train = time.time()
                    surrogate.train(iteration)
                    if time.time() - t_train > 10.0:
                        surrogate.enabled = False
                        use_surrogate = False

            if cpu_futures:
                for fut, name in cpu_futures:
                    try:
                        cpu_df = fut.result(timeout=0)
                        if not cpu_df.empty:
                            seed_df = (
                                pd.concat([seed_df, cpu_df], ignore_index=True)
                                if not seed_df.empty
                                else cpu_df.copy()
                            )
                    except Exception:
                        pass
                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["name"])

            dja_names = set(data_dja["name"]) if not data_dja.empty else set()
            tabu_names = set(data_tabu["name"]) if not data_tabu.empty else set()
            scored_for_A = (
                scored_df[scored_df["name"].isin(dja_names)] if dja_names else scored_df
            )
            scored_for_B = (
                scored_df[scored_df["name"].isin(tabu_names)]
                if tabu_names
                else pd.DataFrame(columns=scored_df.columns)
            )
            update_populations(dpex, scored_for_A, scored_for_B)
            if data_tabu_moves:
                update_tabu(dpex, data_tabu_moves)
            if iteration % dpex.T_ex == 0:
                dpex_exchange(dpex)

            scored_df["inchi"] = scored_df["smiles"].map(MoleculeUtils.generate_inchikey)
            params.seen_molecules |= set(scored_df["name"].tolist())

            prev_avg, _ = _pool_stats(top_pool)
            total_data = scored_df[["name", "smiles", "inchi", "score"]].copy()
            all_pool = (
                pd.concat([all_pool, total_data], ignore_index=True)
                if not all_pool.empty
                else total_data.copy()
            )
            all_pool = (
                all_pool.sort_values("score", ascending=False)
                .drop_duplicates(subset=["inchi"], keep="first")
            )
            top_pool = select_tanimoto_diverse(
                all_pool.reset_index(drop=True),
                n=TOP_AVG_K + 50,
                threshold=1.0,
                smiles_col="smiles",
            ).reset_index(drop=True)

            pool_avg, pool_max = _pool_stats(top_pool)
            if prev_avg > 0:
                params.score_improvement_rate = (pool_avg - prev_avg) / max(
                    abs(prev_avg), 1e-6
                )
            else:
                params.score_improvement_rate = 1.0

            if params.score_improvement_rate <= 0.0001:
                params.no_improvement_counter += 1
                plateau_counter += 1
            else:
                params.no_improvement_counter = 0
                plateau_counter = 0

            if plateau_counter >= 5:
                params.mutation_prob = min(0.85, params.mutation_prob * 2.0)
                plateau_counter = 0

            if (
                exploit_summary
                and "exploited_reactant_ids" in exploit_summary
                and (
                    params.score_improvement_rate <= 0.0001 or not exploited_status
                )
            ):
                params.exploited_reactants.update(
                    exploit_summary["exploited_reactant_ids"]
                )

            mode_str = (
                "EXPLOIT"
                if exploited_status
                else (
                    "INIT"
                    if iteration == 1 or not dpex.pop_A
                    else ("DJA+TABU" if params.synthon_lib else "DJA")
                )
            )
            iter_time = time.time() - iteration_start
            total_time = time.time() - time_start

            append_iteration_log(
                iteration,
                iter_time,
                total_time,
                n_scored_iter,
                n_scored_total,
                mode_str,
                len(dpex.pop_A),
                len(dpex.pop_B),
                pool_avg,
                pool_max,
            )
            history_iters.append(iteration)
            history_avg.append(pool_avg)
            history_max.append(pool_max)
            update_score_graphs(history_iters, history_avg, history_max)

            await asyncio.sleep(1)


async def main():
    global TARGET_KEY, TARGET_LABEL

    parse_args()
    try:
        config = load_config()
        logger.info("✅ Config loaded from config/config.yaml")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    # Resolve target identity before touching the DB so every row we write is
    # tagged the same way orchestrator.py tags its own.
    TARGET_KEY, TARGET_LABEL = score_store.target_identity(config)
    logger.info(f"✅ target={TARGET_LABEL} | target_key={TARGET_KEY[:12]}")

    initialize_solution(config)
    init_score_results_db()

    # fresh log header
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n===== start rxn={RXN_ID} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    state: Dict[str, Any] = {
        "config": config,
        "current_challenge_targets": config["small_molecule_target"],
        "current_challenge_targets_clip_interval": config[
            "small_molecule_target_clip_interval"
        ],
        "current_challenge_antitargets": [],
        "boltz_wrapper": None,
    }
    logger.info(f"🎯 Target: {state['current_challenge_targets']}")

    if _import_boltz_wrapper() and BoltzWrapper is not None:
        try:
            state["boltz_wrapper"] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized")
        except Exception as e:
            logger.error(f"❌ BoltzWrapper init failed: {e}")
            state["boltz_wrapper"] = None
    else:
        logger.warning("⚠️ Boltz unavailable")

    try:
        await find_solution(state)
    except KeyboardInterrupt:
        logger.info(f"✅ Stopped by user (rxn={RXN_ID})")
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
