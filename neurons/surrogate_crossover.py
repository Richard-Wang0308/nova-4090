import os
import sys
import random
import logging
import asyncio
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
STARTING_EPOCH = 23863

# ── These are set dynamically from --rxn_id argument in parse_args() ─────
RXN_ID           = None
SCORE_RESULTS_DB = None
RXN_CSV          = None   # data/rxn{rxn_id}.csv — warm-seed + surrogate training

from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    molecule_unique_for_protein_hf,
    contains_atom_type
)
from molecules_base import generate_inchikey
from combinatorial_db.reactions import get_smiles_from_reaction

BOLTZ_AVAILABLE = False
BoltzWrapper = None

# ── Surrogate pipeline constants ──────────────────────────────────────────
# Mirrors miner.py's DPEX-DJA surrogate design:
#   generate GENERATE_MULTIPLIER x candidates (default 1x → 1500/iteration)
#   → dedup vs known-scored → surrogate filter (keep top ratio, once
#   trained) → Boltz. Below the min-train-size threshold, fall back to a
#   hard BOLTZ_BUDGET cap so Boltz never gets flooded.
GENERATE_MULTIPLIER      = 1      # generate 1x the desired batch size
BOLTZ_BUDGET             = 600    # hard cap while surrogate is not trained
SURROGATE_KEEP_RATIO     = 0.1   # keep top 20% once surrogate is trained
SURROGATE_MIN_TRAIN_SIZE = 4000   # min samples before surrogate activates

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> int:
    global RXN_ID, SCORE_RESULTS_DB, RXN_CSV

    parser = argparse.ArgumentParser(
        description="Surrogate crossover miner — single fixed reaction mode"
    )
    parser.add_argument(
        "--rxn_id", type=int, required=True,
        help="Reaction ID (e.g. 1-5).",
    )
    args = parser.parse_args()

    RXN_ID           = args.rxn_id
    SCORE_RESULTS_DB = os.path.join(BASE_DIR, f"score_results_{RXN_ID}.sqlite")
    RXN_CSV          = os.path.join(BASE_DIR, "data", f"rxn{RXN_ID}.csv")

    logger.info(f"✅ rxn_id           = {RXN_ID}")
    logger.info(f"✅ SCORE_RESULTS_DB  = {SCORE_RESULTS_DB}")
    logger.info(f"✅ RXN_CSV           = {RXN_CSV}  (warm-seed + surrogate training)")
    return RXN_ID


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint helpers (for surrogate)
# ═══════════════════════════════════════════════════════════════════════════

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)
_fp_cache: Dict[str, np.ndarray] = {}
_mol_cache: Dict[str, Any] = {}


def get_mol(smiles: str):
    if smiles in _mol_cache:
        return _mol_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    _mol_cache[smiles] = mol
    return mol


def get_morgan_fingerprint(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
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


# ═══════════════════════════════════════════════════════════════════════════
# SurrogateModel  (ported from miner.py)
# ═══════════════════════════════════════════════════════════════════════════

class SurrogateModel:
    """
    Random Forest surrogate for rxn-specific molecule scoring.

      1. Anchors (never evicted) come from warm-start (CSV+DB combined pool).
      2. Live training data (evictable, capped) comes from fresh Boltz scores.
      3. Balanced sampling (top/bottom/random) teaches the good/bad boundary.
      4. Only activates (is_trained + gatekeeps Boltz) once total training
         data >= SURROGATE_MIN_TRAIN_SIZE. Before that, caller should hard-cap
         at BOLTZ_BUDGET instead.
      5. Non-finite (inf/-inf/nan) scores are dropped before ever reaching
         model.fit().
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
        self.is_trained           = False
        self.anchor_X: list        = []
        self.anchor_y: list        = []
        self.X_train: list         = []
        self.y_train: list         = []
        self.min_train_size        = SURROGATE_MIN_TRAIN_SIZE
        self.max_training_samples  = max_training_samples
        self.last_train_iteration  = 0
        self.train_interval        = 1
        self.enabled                = True

    def _safe_fp(self, smiles: str) -> np.ndarray:
        fp = get_morgan_fingerprint(smiles)
        return fp if fp is not None else np.zeros(2048, dtype=np.uint8)

    def add_anchor_data(self, smiles_list: list, scores: list):
        """Permanent anchors — balanced top-33% + bottom-33% + random-10%."""
        if not smiles_list:
            return

        scores_arr = np.array(scores, dtype=float)
        finite_mask = np.isfinite(scores_arr)
        n_dropped   = int((~finite_mask).sum())
        if n_dropped:
            logger.warning(
                f"[SURROGATE] add_anchor_data: dropping {n_dropped} "
                f"non-finite score(s) before anchoring"
            )
        if not finite_mask.all():
            smiles_list = [s for s, ok in zip(smiles_list, finite_mask) if ok]
            scores_arr  = scores_arr[finite_mask]

        n = len(scores_arr)
        if n == 0:
            logger.warning(
                "[SURROGATE] add_anchor_data: no finite scores left — skipping"
            )
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
            f"(top={n_top} bottom={n_bottom} rand={n_rand} "
            f"from {n} finite total, {n_dropped} non-finite dropped, never evicted)"
        )

    def add_training_data(self, smiles_list: list, scores: list):
        """Live (evictable) training data — balanced top+bottom+recent."""
        if not self.enabled or not smiles_list:
            return

        scores_arr = np.array(scores, dtype=float)
        finite_mask = np.isfinite(scores_arr)
        n_dropped   = int((~finite_mask).sum())
        if n_dropped:
            logger.warning(
                f"[SURROGATE] add_training_data: dropping {n_dropped} "
                f"non-finite score(s)"
            )
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

    def train(self, iteration: int = 0):
        if self.is_trained and (iteration - self.last_train_iteration) < self.train_interval:
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
                f"{len(self.anchor_X)} anchors + {len(self.X_train)} live "
                f"= {len(X_all)} total | min_train_size={self.min_train_size}"
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
        """Keep top keep_ratio fraction by predicted score. Ratio-based."""
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
            f"(kept top {keep_ratio*100:.0f}%, "
            f"dropped {len(data)-len(filtered)} low-quality before Boltz)"
        )
        return filtered.reset_index(drop=True)

    @property
    def total_train_size(self) -> int:
        return len(self.anchor_X) + len(self.X_train)


# ═══════════════════════════════════════════════════════════════════════════
# Validation functions (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def validate_molecule_smiles(molecule_name: str, smiles: str) -> Tuple[bool, str]:
    """Validate SMILES string with RDKit."""
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
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate heavy atom count."""
    try:
        heavy_atom_count = get_heavy_atom_count(smiles)
        min_atoms = 10
        max_atoms = 40

        if heavy_atom_count < min_atoms:
            return False, f"Insufficient heavy atoms: {heavy_atom_count} < {min_atoms}"
        if heavy_atom_count > max_atoms:
            return False, f"Too many heavy atoms: {heavy_atom_count} > {max_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Heavy atom count error: {str(e)}"


def validate_molecule_banned_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate molecule doesn't contain banned atom types."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for banned atom check"

        banned_atoms = config["banned_atom_types"]
        if not banned_atoms:
            return True, ""

        if contains_atom_type(mol, banned_atoms):
            return False, f"Contains banned atom types: {banned_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Banned atom check error: {str(e)}"


def validate_molecule_rotatable_bonds(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate rotatable bonds are within acceptable range."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for rotatable bonds check"

        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        min_bonds = config["min_rotatable_bonds"]
        max_bonds = config["max_rotatable_bonds"]

        if num_rotatable_bonds < min_bonds or num_rotatable_bonds > max_bonds:
            return False, f"Rotatable bonds out of range: {num_rotatable_bonds} (expected {min_bonds}-{max_bonds})"
        return True, ""
    except Exception as e:
        return False, f"Rotatable bonds check error: {str(e)}"


async def validate_molecule_huggingface_unique(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str
) -> Tuple[bool, str]:
    """Validate molecule is unique (NOT in HuggingFace dataset)."""
    if not state.get('current_challenge_targets'):
        return False, "No target proteins available"

    primary_target = state['current_challenge_targets'][0]

    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)

        if not is_unique_hf:
            return False, f"Molecule already in HuggingFace dataset for {primary_target}"
        return True, ""
    except Exception as e:
        return False, f"HuggingFace uniqueness check error: {str(e)}"


async def validate_molecule_complete(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any] = None
) -> Tuple[bool, List[str]]:
    """Perform complete validation on a molecule."""
    if config is None:
        config = state.get('config', {})

    errors = []

    # 1. SMILES validity
    is_valid, error_msg = validate_molecule_smiles(molecule_name, smiles)
    if not is_valid:
        errors.append(f"[SMILES] {error_msg}")
        return False, errors

    # 2. Heavy atom count
    is_valid, error_msg = validate_molecule_heavy_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[HEAVY_ATOMS] {error_msg}")

    # 3. Banned atoms
    is_valid, error_msg = validate_molecule_banned_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[BANNED_ATOMS] {error_msg}")

    # 4. Rotatable bonds
    is_valid, error_msg = validate_molecule_rotatable_bonds(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[ROTATABLE_BONDS] {error_msg}")

    # 5. HuggingFace uniqueness
    is_valid, error_msg = await validate_molecule_huggingface_unique(state, molecule_name, smiles)
    if not is_valid:
        errors.append(f"[HF_UNIQUE] {error_msg}")

    return len(errors) == 0, errors


class GeneticAlgorithmOperator:
    """Performs genetic algorithm operations on molecules (CROSSOVER ONLY)."""

    def __init__(self, rxn_id: int, db_path: str):
        """Initialize GA operator."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()

    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """Crossover two molecules by swapping random components."""
        try:
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')

            if (parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                logger.debug(f"Invalid format: must start with 'rxn'")
                return None

            if len(parts1) != len(parts2):
                logger.debug(f"Invalid format: different number of components")
                return None

            if len(parts1) not in [4, 5]:
                logger.debug(f"Invalid format: expected 4 or 5 parts, got {len(parts1)}")
                return None

            try:
                rxn_id_1 = int(parts1[1])
                rxn_id_2 = int(parts2[1])
                if rxn_id_1 != self.rxn_id or rxn_id_2 != self.rxn_id:
                    logger.debug(f"Wrong rxn_ids: {rxn_id_1}, {rxn_id_2}")
                    return None
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing rxn_ids: {e}")
                return None

            num_components = len(parts1) - 2
            component_indices = list(range(2, 2 + num_components))
            swap_idx = random.choice(component_indices)

            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)

            if offspring_name in self.generated_molecule_names:
                return None

            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        self.generated_molecule_names.add(offspring_name)
                        return offspring_name
                    else:
                        logger.debug(f"Invalid SMILES from RDKit: {offspring_smiles}")
                else:
                    logger.debug(f"No SMILES generated for offspring")
            except Exception as e:
                logger.debug(f"Error validating crossover: {e}")

            return None

        except Exception as e:
            logger.debug(f"Error in crossover_molecules: {e}")
            return None

    def apply_genetic_operations(
        self,
        top_molecules: List[str],
        num_crossovers: int = 5
    ) -> List[Dict[str, Any]]:
        """Apply genetic operations (CROSSOVER ONLY) to top molecules."""
        new_molecules = []
        self.generated_molecule_names.clear()

        crossovers_created = 0

        for i in range(num_crossovers):
            parent1 = random.choice(top_molecules)
            parent2 = random.choice(top_molecules)

            if parent1 != parent2:
                offspring = self.crossover_molecules(parent1, parent2)

                if offspring:
                    try:
                        smiles = get_smiles_from_reaction(offspring)
                        inchikey = generate_inchikey(smiles)

                        if smiles and inchikey:
                            new_molecules.append({
                                'name': offspring,
                                'smiles': smiles,
                                'InChIKey': inchikey,
                                'type': 'crossover'
                            })
                            crossovers_created += 1

                    except Exception as e:
                        logger.debug(f"Error processing offspring: {e}")

        return new_molecules


# ═══════════════════════════════════════════════════════════════════════════
# Score results DB helpers (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def init_score_results_db(db_path: str = None) -> None:
    """Initialize/create the score_results.sqlite database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scored_molecules (
                molecule_name TEXT PRIMARY KEY,
                score REAL NOT NULL,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available BOOLEAN DEFAULT TRUE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)

        conn.commit()
        conn.close()
        print(f"Initialized score_results database at {db_path}")
    except Exception as e:
        print(f"Error initializing score_results database: {e}")


def get_score_from_db(molecule_name: str, db_path: str = None) -> Optional[float]:
    """Get score for a molecule from the database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT score FROM scored_molecules WHERE molecule_name = ?", (molecule_name,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return float(result[0])
        return None
    except Exception as e:
        logger.debug(f"Error getting score from DB for {molecule_name}: {e}")
        return None


def write_scores_to_db(molecules: List[Dict[str, Any]], db_path: str = None) -> None:
    """Write scored molecules to the database.

    scored_at is always set explicitly so inserts stay non-NULL even when the
    table was recreated without DEFAULT CURRENT_TIMESTAMP (e.g. after merge).
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not molecules:
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        to_insert = []
        for mol in molecules:
            molecule_name = mol.get('name')
            score = mol.get('boltz_score')

            if molecule_name and score is not None:
                to_insert.append((molecule_name, float(score), now, True))

        if to_insert:
            cursor.executemany(
                "INSERT INTO scored_molecules (molecule_name, score, scored_at, available) VALUES (?, ?, ?, ?)",
                to_insert
            )
            conn.commit()
            print(f"✅ Wrote {len(to_insert)} scored molecules to database")

        conn.close()
    except Exception as e:
        print(f"Error writing scores to database: {e}")


def batch_get_scores_from_db(molecule_names: List[str], db_path: str = None) -> Dict[str, float]:
    """Get scores for multiple molecules from the database in batch."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not molecule_names:
        return {}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(molecule_names))
        cursor.execute(
            f"SELECT molecule_name, score FROM scored_molecules WHERE molecule_name IN ({placeholders})",
            molecule_names
        )
        results = cursor.fetchall()
        conn.close()

        return {name: float(score) for name, score in results}
    except Exception as e:
        logger.debug(f"Error batch getting scores from DB: {e}")
        return {}

def load_molecules_from_db_with_validation(
    db_path: str,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from SQLite database with validation from config.yaml."""
    if config is None:
        config = {}

    if not os.path.exists(db_path):
        logger.warning(f"Database file not found at {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    try:
        logger.info(
            f"Loading molecules from database {db_path} for rxn_id={rxn_id}"
        )

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT molecule_name, score FROM scored_molecules")
        db_results = cursor.fetchall()
        conn.close()

        if not db_results:
            logger.info("No molecules found in database")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

        result_rows = []
        successful_count = 0
        failed_count = 0
        banned_atom_count = 0
        heavy_atom_count = 0
        wrong_rxn_id_count = 0

        for molecule_name, score in db_results:
            try:
                if not molecule_name.startswith(f"rxn:{rxn_id}:"):
                    wrong_rxn_id_count += 1
                    continue

                smiles = get_smiles_from_reaction(molecule_name)
                logger.debug(f"Attempting to parse SMILES from DB for {molecule_name}: {smiles}")

                if not smiles:
                    logger.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.debug(f"Cannot parse SMILES for {molecule_name}")
                    failed_count += 1
                    continue

                banned_atoms = config["banned_atom_types"]
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    logger.debug(f"Molecule {molecule_name} contains banned atoms {banned_atoms}, skipping")
                    banned_atom_count += 1
                    continue

                min_heavy_atoms = 10
                max_heavy_atoms = 45
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has insufficient heavy atoms ({heavy_atom_count_val} < {min_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue
                if heavy_atom_count_val > max_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has too many heavy atoms ({heavy_atom_count_val} > {max_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue

                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    logger.debug(f"Could not generate InChIKey for {molecule_name}")
                    failed_count += 1
                    continue

                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': float(score) if score is not None else None,
                })
                successful_count += 1

            except Exception as e:
                logger.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue

        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')

            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                logger.info(
                    f"✅ Loaded {len(result_df)} molecules from database "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                    f"wrong rxn_id: {wrong_rxn_id_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        logger.info(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            logger.warning(
                f"No valid molecules loaded from database "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                f"wrong rxn_id: {wrong_rxn_id_count})"
            )

        return result_df

    except Exception as e:
        logger.error(f"Error loading molecules from database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_csv_with_validation(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from CSV file with validation from config.yaml."""
    if config is None:
        config = {}

    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    try:
        logger.info(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)

        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            logger.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            logger.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

        if df.empty:
            logger.info("No matching molecules found in CSV")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

        result_rows = []
        successful_count = 0
        failed_count = 0
        banned_atom_count = 0
        heavy_atom_count = 0

        for _, row in df.iterrows():
            molecule_name = row['molecule_name']

            try:
                smiles = get_smiles_from_reaction(molecule_name)

                if not smiles:
                    logger.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.debug(f"Cannot parse SMILES for {molecule_name}")
                    failed_count += 1
                    continue

                banned_atoms = config["banned_atom_types"]
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    logger.debug(f"Molecule {molecule_name} contains banned atoms {banned_atoms}, skipping")
                    banned_atom_count += 1
                    continue

                min_heavy_atoms = 10
                max_heavy_atoms = 45
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has insufficient heavy atoms ({heavy_atom_count_val} < {min_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue
                if heavy_atom_count_val > max_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has too many heavy atoms ({heavy_atom_count_val} > {max_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue

                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    logger.debug(f"Could not generate InChIKey for {molecule_name}")
                    failed_count += 1
                    continue

                final_score = row.get('final_score', None)
                if pd.isna(final_score):
                    final_score = None
                else:
                    final_score = float(final_score)

                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': final_score,
                })
                successful_count += 1

            except Exception as e:
                logger.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue

        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')

            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                logger.info(
                    f"✅ Loaded {len(result_df)} molecules from CSV "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        logger.info(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            logger.warning(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
            )

        return result_df

    except Exception as e:
        logger.error(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(
    csv_path: str,
    db_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Load molecules from both CSV and database, merge them, and deduplicate.
    When duplicates exist (by InChIKey), prefer the one with the higher score.
    """
    if config is None:
        config = {}

    logger.info(f"🔄 Loading molecules from CSV and database...")

    csv_df = load_molecules_from_csv_with_validation(
        csv_path, target_proteins, starting_epoch, rxn_id, config
    )

    db_df = load_molecules_from_db_with_validation(
        db_path, rxn_id, config
    )

    if csv_df.empty and db_df.empty:
        logger.warning("No molecules loaded from either CSV or database")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    if csv_df.empty:
        logger.info("No molecules from CSV, using database only")
        return db_df

    if db_df.empty:
        logger.info("No molecules from database, using CSV only")
        return csv_df

    csv_df['source'] = 'csv'
    db_df['source'] = 'database'

    combined_df = pd.concat([csv_df, db_df], ignore_index=True)

    combined_df = combined_df.sort_values(
        by=['score', 'source'],
        ascending=[False, True],
        na_position='last'
    )

    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')

    combined_df = combined_df.drop(columns=['source'])

    combined_df = combined_df.sort_values(by='score', ascending=False, na_position='last')

    csv_count = len(csv_df)
    db_count = len(db_df)
    combined_count = len(combined_df)
    duplicates_removed = csv_count + db_count - combined_count

    logger.info(
        f"✅ Combined loading complete: "
        f"{csv_count} from CSV, {db_count} from database, "
        f"{combined_count} unique molecules after deduplication "
        f"({duplicates_removed} duplicates removed)"
    )

    if combined_count > 0:
        scores = combined_df['score'].dropna()
        if len(scores) > 0:
            logger.info(
                f"   Combined score range: {scores.min():.6f} to {scores.max():.6f} "
                f"(top 3: {scores.head(3).tolist()})"
            )

    return combined_df


def load_training_csv_for_surrogate(rxn_id: int) -> pd.DataFrame:
    """
    Load rxn-specific rows from data/rxn{rxn_id}.csv for surrogate training.
    Unlike the warm-seed CSV load, this does not apply an epoch filter.
    """
    csv_path = RXN_CSV
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

        prefix = f"rxn:{rxn_id}:"
        df     = df[
            df['molecule_name'].str.startswith(prefix, na=False)
        ].reset_index(drop=True)

        if df.empty:
            logger.warning(
                f"⚠️  rxn{rxn_id}.csv: no rows for rxn={rxn_id} "
                f"(prefix='{prefix}') — surrogate will cold-start"
            )
            return pd.DataFrame(columns=["smiles", "score"])

        df = df[df['score'].notna()].reset_index(drop=True)
        df['smiles'] = df['molecule_name'].apply(get_smiles_from_reaction)
        df = df[df['smiles'].notna() & (df['smiles'] != '')]
        result = df[['smiles', 'score']].copy()
        result['score'] = pd.to_numeric(result['score'], errors='coerce')
        result.loc[~np.isfinite(result['score']), 'score'] = np.nan
        result = result[result['score'].notna()]
        result = result.drop_duplicates(subset=['smiles']).reset_index(drop=True)
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


async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Score molecules using BoltzWrapper in batches.
    """
    if state.get('boltz_wrapper') is None:
        logger.warning("BoltzWrapper not available, skipping scoring")
        return molecules

    if not molecules:
        return molecules

    logger.info(f"🔬 Processing {len(molecules)} molecules for scoring in batches of {batch_size}...")

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

        target_proteins = state.get('current_challenge_targets', [])
        primary_target = target_proteins[0] if target_proteins else None

        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)

        logger.info(f"   Found {len(db_scores)} molecules already in database")

        for mol in batch:
            molecule_name = mol['name']
            smiles = mol.get('smiles')

            if molecule_name in db_scores:
                mol['boltz_score'] = db_scores[molecule_name]
                mol['boltz_score_source'] = 'database'
                molecules_with_db_scores.append(mol)
                logger.debug(f"   ✓ {molecule_name}: score from DB = {db_scores[molecule_name]:.6f}")
                continue

            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                    if not is_unique_hf:
                        logger.debug(f"   ⏭️  {molecule_name}: already in HuggingFace, skipping")
                        molecules_in_hf.append(mol)
                        continue
                except Exception as e:
                    logger.debug(f"   Error checking HuggingFace for {molecule_name}: {e}")

            molecules_to_score.append(mol)

        logger.info(
            f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
            f"{len(molecules_in_hf)} in HuggingFace (skipped), "
            f"{len(molecules_to_score)} need scoring"
        )

        newly_scored_molecules = []
        if molecules_to_score:
            logger.info(f"   Scoring {len(molecules_to_score)} new molecules with Boltz...")

            boltz = state['boltz_wrapper']
            config = state['config']
            target_proteins = state.get('current_challenge_targets', [])
            antitarget_proteins = state.get('current_challenge_antitargets', [])

            if not target_proteins:
                logger.warning("No target proteins available for scoring")
                all_scored_molecules.extend(molecules_with_db_scores)
                continue

            primary_target = target_proteins[0]

            try:
                output_dir = os.path.join(boltz.output_dir, 'boltz_results_inputs')
                if os.path.exists(output_dir):
                    try:
                        lightning_logs_dir = os.path.join(output_dir, 'lightning_logs')
                        if os.path.exists(lightning_logs_dir):
                            import shutil
                            shutil.rmtree(lightning_logs_dir, ignore_errors=True)
                            logger.debug(f"Cleaned up old lightning_logs directory")
                    except Exception as cleanup_err:
                        logger.debug(f"Could not clean up old logs: {cleanup_err}")

                processed_dir = os.path.join(output_dir, 'processed')
                structures_dir = os.path.join(processed_dir, 'structures')
                records_dir = os.path.join(processed_dir, 'records')
                msa_dir = os.path.join(processed_dir, 'msa')
                predictions_dir = os.path.join(output_dir, 'predictions')

                os.makedirs(structures_dir, exist_ok=True)
                os.makedirs(records_dir, exist_ok=True)
                os.makedirs(msa_dir, exist_ok=True)
                os.makedirs(predictions_dir, exist_ok=True)

                valid_molecules_by_uid = {
                    0: {
                        'smiles': [mol['smiles'] for mol in molecules_to_score],
                        'names': [mol['name'] for mol in molecules_to_score]
                    }
                }

                score_dict = {
                    0: {
                        "target_scores": [[]],
                        "antitarget_scores": [[]],
                        "entropy": None,
                        "entropy_boltz": None,
                        "block_submitted": None,
                        "push_time": ""
                    }
                }

                num_molecules_to_score = len(molecules_to_score)

                subnet_config = {
                    'small_molecule_target': config['small_molecule_target'],
                    'small_molecule_target_clip_interval': config['small_molecule_target_clip_interval'],
                    'boltz_mode': getattr(config, 'boltz_mode', 'max'),
                    'boltz_metric': getattr(config, 'boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']),
                    'combination_strategy': getattr(config, 'combination_strategy', 'heavy_atom_normalization')                }

                final_block_hash = "0x" + "0" * 64

                logger.info(f"   Running Boltz scoring for {len(molecules_to_score)} molecules...")
                start_time = time.time()

                def run_scoring():
                    boltz.score_molecules(
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, run_scoring)

                elapsed = time.time() - start_time
                logger.info(f"   ✅ Boltz scoring completed in {elapsed:.2f} seconds")

                uid = 0
                smiles_to_score = {}
                final_scores = getattr(boltz, 'final_boltz_scores', {}).get(uid, {})
                if primary_target and primary_target in final_scores:
                    smiles_to_score = final_scores[primary_target].copy()
                elif final_scores:
                    smiles_to_score = next(iter(final_scores.values())).copy()
                elif hasattr(boltz, 'per_molecule_metric') and uid in boltz.per_molecule_metric:
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()
                if smiles_to_score:
                    logger.info(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores")

                target_scores_list = None
                target_scores = score_dict[uid].get('target_scores', [[]])
                if target_scores and len(target_scores[0]) > 0:
                    target_scores_list = target_scores[0] if isinstance(target_scores[0], list) else [target_scores[0]]

                avg_score = None
                if not smiles_to_score and not target_scores_list:
                    avg_score = score_dict[uid].get('boltz_score')

                for mol_idx, mol in enumerate(molecules_to_score):
                    smiles = mol['smiles']
                    score = None

                    if smiles in smiles_to_score:
                        score = smiles_to_score[smiles]
                    elif target_scores_list and mol_idx < len(target_scores_list):
                        score = target_scores_list[mol_idx]
                    elif target_scores_list:
                        try:
                            valid_idx = valid_molecules_by_uid[uid]['smiles'].index(smiles)
                            if valid_idx < len(target_scores_list):
                                score = target_scores_list[valid_idx]
                        except (ValueError, IndexError):
                            pass

                    if score is None and avg_score is not None:
                        score = avg_score

                    mol['boltz_score'] = score
                    if score is not None:
                        mol['boltz_score_source'] = 'boltz'
                        newly_scored_molecules.append(mol)

                if newly_scored_molecules:
                    for mol in newly_scored_molecules:
                        logger.debug(f"Molecule {mol['name']} scored {mol['boltz_score']}")
                    write_scores_to_db(newly_scored_molecules)

            except Exception as e:
                logger.error(f"❌ Error scoring batch with Boltz: {e}")
                import traceback
                logger.error(traceback.format_exc())

        batch_results = molecules_with_db_scores + newly_scored_molecules

        for mol in molecules_in_hf:
            mol['boltz_score'] = None
            mol['boltz_score_source'] = 'huggingface_skipped'
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
                    m.get('boltz_score')
                    if m.get('boltz_score') is not None
                    else float('-inf')
                ),
                reverse=True,
            ):
                name = mol.get('name', 'unknown')
                score = mol.get('boltz_score')
                source = mol.get('boltz_score_source', 'boltz')
                if score is not None:
                    logger.info(f"      {name}: {score:.6f} [{source}]")
                else:
                    logger.info(f"      {name}: skipped [{source}]")

    scored_molecules = sorted(
        all_scored_molecules,
        key=lambda m: m.get('boltz_score') if m.get('boltz_score') is not None else float('-inf'),
        reverse=True
    )

    logger.info(f"✅ Batch scoring complete: {len(scored_molecules)} total molecules scored")

    return scored_molecules


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
        logger.info(f"✅ BoltzWrapper imported successfully")
        return True

    except ImportError as e:
        logger.warning(f"⚠️  Failed to import BoltzWrapper: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error setting up BoltzWrapper: {e}")
        return False

async def generate_unique_molecules_from_top200(
    state: Dict[str, Any],
    top_200_df: pd.DataFrame,
    desired_count: int = 100,
    max_attempts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules using genetic algorithm from top 200 molecules.

    NOTE: `desired_count` here is the PRE-surrogate-filter target — i.e. the
    caller may request GENERATE_MULTIPLIER x more than it actually intends
    to send to Boltz, then apply the surrogate filter afterward. This
    function itself is unaware of the surrogate — it just generates/validates
    candidates the same way as before.
    """
    if top_200_df.empty:
        logger.warning("Top 200 DataFrame is empty")
        return []

    ga_operator = GeneticAlgorithmOperator(state['rxn_id'], DB_PATH)
    all_names = top_200_df['name'].tolist()

    pool_sizes = [100, 150, 200]
    current_pool_size_idx = 0
    current_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))

    # ✅ Scale max_attempts with desired_count so larger pre-filter
    # candidate pools (desired_count * GENERATE_MULTIPLIER) still have a
    # realistic chance of completing.
    if max_attempts is None:
        max_attempts = max(500, desired_count * 3)

    logger.info(
        f"🧬 Generating {desired_count} unique molecules with validation "
        f"(starting with top {current_pool_size}, max_attempts={max_attempts})..."
    )

    unique_molecules = []
    attempts = 0
    last_successful_attempt = 0

    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())

    validation_stats = {
        'total_generated': 0,
        'passed_validation': 0,
        'failed_smiles': 0,
        'failed_heavy_atoms': 0,
        'failed_banned_atoms': 0,
        'failed_rotatable_bonds': 0,
        'failed_hf_unique': 0,
        'failed_other': 0,
    }

    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1

        if attempts - last_successful_attempt >= 100 and current_pool_size_idx < len(pool_sizes) - 1:
            current_pool_size_idx += 1
            new_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
            if new_pool_size > current_pool_size:
                current_pool_size = new_pool_size
                logger.info(f"📈 Increasing pool size to top {current_pool_size}")
                last_successful_attempt = attempts

        current_pool_names = all_names[:current_pool_size]
        new_molecules = ga_operator.apply_genetic_operations(current_pool_names, num_crossovers=10)

        for mol in new_molecules:
            if len(unique_molecules) >= desired_count:
                break

            molecule_name = mol['name']
            smiles = mol.get('smiles')

            validation_stats['total_generated'] += 1

            if molecule_name in [m['name'] for m in unique_molecules]:
                continue

            if molecule_name in generated_molecules:
                logger.debug(f"   ⏭️  Molecule {molecule_name} already generated")
                continue

            inchikey = None
            try:
                inchikey = generate_inchikey(smiles) if smiles else None
                if inchikey and inchikey in generated_inchikeys:
                    logger.debug(f"   ⏭️  Molecule {molecule_name} (InChIKey: {inchikey}) already generated")
                    continue
            except Exception as e:
                logger.debug(f"   Could not generate InChIKey for {molecule_name}: {e}")

            is_valid, errors = await validate_molecule_complete(state, molecule_name, smiles, state['config'])

            if not is_valid:
                for error in errors:
                    logger.debug(f"   ❌ {molecule_name}: {error}")
                    if "[SMILES]" in error:
                        validation_stats['failed_smiles'] += 1
                    elif "[HEAVY_ATOMS]" in error:
                        validation_stats['failed_heavy_atoms'] += 1
                    elif "[BANNED_ATOMS]" in error:
                        validation_stats['failed_banned_atoms'] += 1
                    elif "[ROTATABLE_BONDS]" in error:
                        validation_stats['failed_rotatable_bonds'] += 1
                    elif "[HF_UNIQUE]" in error:
                        validation_stats['failed_hf_unique'] += 1
                    else:
                        validation_stats['failed_other'] += 1
                continue

            unique_molecules.append(mol)
            generated_molecules.add(molecule_name)
            if inchikey:
                generated_inchikeys.add(inchikey)
            last_successful_attempt = attempts
            validation_stats['passed_validation'] += 1

        if len(unique_molecules) >= desired_count:
            break

        await asyncio.sleep(0.1)

    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys

    logger.info(
        f"✅ Generated {len(unique_molecules)} valid molecules (attempts: {attempts})"
        f"\n   Validation stats:"
        f"\n   - Total generated: {validation_stats['total_generated']}"
        f"\n   - Passed validation: {validation_stats['passed_validation']}"
        f"\n   - Failed SMILES: {validation_stats['failed_smiles']}"
        f"\n   - Failed heavy atoms: {validation_stats['failed_heavy_atoms']}"
        f"\n   - Failed banned atoms: {validation_stats['failed_banned_atoms']}"
        f"\n   - Failed rotatable bonds: {validation_stats['failed_rotatable_bonds']}"
        f"\n   - Failed HF uniqueness: {validation_stats['failed_hf_unique']}"
        f"\n   - Failed other: {validation_stats['failed_other']}"
    )

    return unique_molecules


def warm_start_surrogate(surrogate: "SurrogateModel", molecules_df: pd.DataFrame) -> None:
    """
    Seed the surrogate with permanent anchors from the combined CSV+DB pool.
    Mirrors miner.py's warm_start() anchor-seeding logic. Called once, on
    round 1, before the surrogate is expected to gatekeep anything.
    """
    if molecules_df.empty:
        logger.warning("[SURROGATE] warm_start: molecules_df is empty — skipping anchor seed")
    else:
        valid = molecules_df[
            molecules_df['score'].notna() & molecules_df['smiles'].notna()
        ]
        if valid.empty:
            logger.warning("[SURROGATE] warm_start: no rows with valid score+smiles — skipping")
        else:
            surrogate.add_anchor_data(
                valid['smiles'].tolist(),
                valid['score'].tolist(),
            )
            logger.info(
                f"[SURROGATE] warm_start: seeded {len(valid)} rows from CSV+DB "
                f"(anchors so far: {surrogate.total_train_size})"
            )

    # Also merge in rxn{rxn_id}.csv without epoch filter (additive anchors).
    training_df = load_training_csv_for_surrogate(RXN_ID)
    if not training_df.empty:
        surrogate.add_anchor_data(
            training_df['smiles'].tolist(),
            training_df['score'].tolist(),
        )
        logger.info(
            f"[SURROGATE] warm_start: +{len(training_df)} anchors from "
            f"rxn{RXN_ID}.csv (total_train_size={surrogate.total_train_size})"
        )

    logger.info(
        f"[SURROGATE] warm_start: total_train_size={surrogate.total_train_size}, "
        f"min_train_size={surrogate.min_train_size}"
    )

    if surrogate.total_train_size >= surrogate.min_train_size:
        surrogate.train(iteration=0)
        logger.info(
            f"[SURROGATE] warm_start: pre-trained on warm-start data | "
            f"trained={surrogate.is_trained}"
        )
    else:
        logger.warning(
            f"[SURROGATE] warm_start: insufficient data "
            f"({surrogate.total_train_size} < {surrogate.min_train_size}) — "
            f"will hard-cap at {BOLTZ_BUDGET} until enough live scores accumulate"
        )


async def run_generation_and_scoring_loop(state: Dict[str, Any]) -> None:
    """
    Main loop that continuously generates and scores molecules until interrupted.

    Pipeline per round (surrogate-gated, mirrors miner.py):
      1. Reload molecules from CSV + DB (warm-seed pool)
      2. [round 1 only] Warm-start surrogate anchors from that pool
      3. Generate  desired_unique_count * GENERATE_MULTIPLIER  candidates
         via genetic crossover + full validation (SMILES/heavy atoms/
         banned atoms/rotatable bonds/HF uniqueness — already dedup'd
         against generated_molecules/generated_inchikeys)
      4. Surrogate filter  → keep top SURROGATE_KEEP_RATIO (20%) of the
         fresh candidates once surrogate.total_train_size >=
         SURROGATE_MIN_TRAIN_SIZE; otherwise hard-cap at BOLTZ_BUDGET
      5. Boltz score whatever survives step 4
      6. Feed newly-scored molecules back into the surrogate as live
         training data; retrain each round once enough data exists.
    """
    logger.info("🚀 Starting generation and scoring loop...")
    logger.info("Press Ctrl+C to stop")
    logger.info(
        f"✅ Pipeline = generate {GENERATE_MULTIPLIER}x → validate/dedup → "
        f"surrogate(keep {SURROGATE_KEEP_RATIO*100:.0f}%, active once "
        f">= {SURROGATE_MIN_TRAIN_SIZE} samples, else hard-cap "
        f"{BOLTZ_BUDGET}) → Boltz"
    )

    desired_unique_count = 1500
    batch_size = 10
    round_number = 0

    surrogate = SurrogateModel(max_training_samples=5000)
    surrogate_warm_started = False

    try:
        while True:
            round_number += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 Round {round_number}")
            logger.info(f"{'='*70}")

            # ── Reload molecules from CSV and database ────────────────
            logger.info("📂 Reloading molecules from CSV and database...")
            config = state['config']
            molecules_df = load_molecules_combined(
                RXN_CSV,
                SCORE_RESULTS_DB,
                state['current_challenge_targets'],
                STARTING_EPOCH,
                RXN_ID,
                config
            )

            if molecules_df.empty:
                logger.warning("⚠️  No valid molecules loaded from CSV or database, waiting...")
                await asyncio.sleep(10)
                continue

            # ── Warm-start surrogate anchors (round 1 only) ────────────
            if not surrogate_warm_started:
                warm_start_surrogate(surrogate, molecules_df)
                surrogate_warm_started = True

            # Get top 200 molecules (already sorted by score)
            top_200_df = molecules_df.head(200)

            # Update state with new molecules
            state['top_pool'] = molecules_df.copy()
            state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
            state['top_200_df'] = top_200_df

            logger.info(f"✅ Reloaded {len(molecules_df)} molecules from CSV and database (top 200: {len(top_200_df)})")

            # ══════════════════════════════════════════════════════════
            # STEP 1 — Generate GENERATE_MULTIPLIER x candidates
            # ══════════════════════════════════════════════════════════
            generate_target = desired_unique_count * GENERATE_MULTIPLIER
            logger.info(
                f"🧬 Generating {generate_target} candidates "
                f"({desired_unique_count}×{GENERATE_MULTIPLIER}) with validation..."
            )
            unique_molecules = await generate_unique_molecules_from_top200(
                state, top_200_df, generate_target
            )

            if not unique_molecules:
                logger.warning("Failed to generate unique molecules, waiting before retry...")
                await asyncio.sleep(10)
                continue

            logger.info(f"✅ Generated {len(unique_molecules)} valid unique candidates")

            # ══════════════════════════════════════════════════════════
            # STEP 2 — Surrogate filter (fresh candidates only — the
            # generator above already dedup'd against generated_molecules
            # / generated_inchikeys / seen_inchikeys, so this pool is
            # already "unseen").
            # ══════════════════════════════════════════════════════════
            candidates_df = pd.DataFrame(unique_molecules)

            surrogate_ready = (
                surrogate.enabled
                and surrogate.is_trained
                and surrogate.total_train_size >= surrogate.min_train_size
            )

            if surrogate_ready:
                pre_sur = len(candidates_df)
                candidates_df = surrogate.filter_candidates(
                    candidates_df,
                    keep_ratio=SURROGATE_KEEP_RATIO,
                    smiles_col="smiles",
                )
                logger.info(
                    f"[SURROGATE] round={round_number} | "
                    f"{pre_sur} fresh → {len(candidates_df)} "
                    f"(kept top {SURROGATE_KEEP_RATIO*100:.0f}%, "
                    f"train_size={surrogate.total_train_size})"
                )
            else:
                if len(candidates_df) > BOLTZ_BUDGET:
                    pre_cap = len(candidates_df)
                    candidates_df = candidates_df.head(BOLTZ_BUDGET)
                    logger.info(
                        f"[Solution] Surrogate not ready "
                        f"(train_size={surrogate.total_train_size} < "
                        f"{surrogate.min_train_size}) — hard-cap "
                        f"{pre_cap} → {len(candidates_df)} at {BOLTZ_BUDGET}"
                    )

            if candidates_df.empty:
                logger.warning("[Solution] No candidates survived filtering; skipping round")
                await asyncio.sleep(10)
                continue

            unique_molecules = candidates_df.to_dict('records')

            # ══════════════════════════════════════════════════════════
            # STEP 3 — Boltz scoring
            # ══════════════════════════════════════════════════════════
            total_batches = (len(unique_molecules) + batch_size - 1) // batch_size
            logger.info(
                f"🔬 Round {round_number}: Scoring {len(unique_molecules)} "
                f"molecules in {total_batches} batches of {batch_size}..."
            )

            all_scored_molecules = []
            best_molecule_so_far = None
            best_score_so_far = float('-inf')

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(unique_molecules))
                batch = unique_molecules[start_idx:end_idx]

                logger.info(
                    f"   📦 Round {round_number}, Batch {batch_idx + 1}/{total_batches}: "
                    f"Scoring {len(batch)} molecules"
                )

                scored_batch = await score_molecules_with_boltz_batched(
                    state, batch, batch_size=len(batch)
                )

                batch_with_scores = [m for m in scored_batch if m.get('boltz_score') is not None]
                all_scored_molecules.extend(batch_with_scores)

                for mol in batch_with_scores:
                    score = mol.get('boltz_score')
                    if score is not None and score > best_score_so_far:
                        best_score_so_far = score
                        best_molecule_so_far = mol
                        source = mol.get('boltz_score_source', 'unknown')
                        logger.info(
                            f"   🏆 New best in round {round_number}, batch {batch_idx + 1}: "
                            f"{mol['name']} (score: {score:.6f}, source: {source})"
                        )

            if all_scored_molecules:
                logger.info(f"📊 Round {round_number} calculated scores ({len(all_scored_molecules)} molecules):")
                for mol in sorted(
                    all_scored_molecules,
                    key=lambda m: m.get('boltz_score', float('-inf')),
                    reverse=True,
                ):
                    name = mol.get('name', 'unknown')
                    score = mol.get('boltz_score')
                    source = mol.get('boltz_score_source', 'boltz')
                    logger.info(f"   {name}: {score:.6f} [{source}]")

            # ══════════════════════════════════════════════════════════
            # STEP 4 — Feed fresh Boltz scores back into the surrogate
            # ══════════════════════════════════════════════════════════
            if surrogate.enabled and all_scored_molecules:
                fresh_smiles = [m['smiles'] for m in all_scored_molecules if m.get('smiles')]
                fresh_scores = [m['boltz_score'] for m in all_scored_molecules if m.get('smiles')]
                surrogate.add_training_data(fresh_smiles, fresh_scores)

                if surrogate.total_train_size >= surrogate.min_train_size:
                    t_train = time.time()
                    surrogate.train(iteration=round_number)
                    train_time = time.time() - t_train
                    if train_time > 10.0:
                        logger.warning(
                            f"[SURROGATE] Training slow ({train_time:.2f}s) — disabling"
                        )
                        surrogate.enabled = False

            # ── Summary for this round ──────────────────────────────────
            best_score_str = f"{best_score_so_far:.6f}" if best_molecule_so_far else 'N/A'
            surrogate_ready_now = (
                surrogate.enabled
                and surrogate.is_trained
                and surrogate.total_train_size >= surrogate.min_train_size
            )
            logger.info(
                f"\n✅ Round {round_number} complete:"
                f"\n   - Generated (pre-filter): {generate_target} requested"
                f"\n   - Sent to Boltz: {len(unique_molecules)} molecules"
                f"\n   - Scored: {len(all_scored_molecules)} molecules"
                f"\n   - Best molecule: {best_molecule_so_far['name'] if best_molecule_so_far else 'None'}"
                f"\n   - Best score: {best_score_str}"
                f"\n   - Surrogate: {'ON' if surrogate_ready_now else 'OFF'} "
                f"({surrogate.total_train_size}/{surrogate.min_train_size} samples)"
            )

            # Wait a bit before next round
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping generation and scoring loop...")
        raise


async def main():
    """Main entry point."""
    rxn_id = parse_args()

    logger.info(
        f"🚀 Starting surrogate_crossover.py | rxn={rxn_id} | "
        f"DB=score_results_{rxn_id}.sqlite | "
        f"CSV=data/rxn{rxn_id}.csv"
    )

    # Load config
    try:
        config = load_config()
        logger.info("✅ Config loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    # Initialize state
    state: Dict[str, Any] = {
        'config': config,
        'startup_complete': False,
        'current_challenge_targets': [],
        'current_challenge_targets_clip_interval': [],
        'current_challenge_antitargets': [],
        'rxn_id': RXN_ID,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),
        'generated_inchikeys': set(),
        'boltz_wrapper': None,
        'top_200_df': pd.DataFrame(),
    }

    state['current_challenge_targets'] = config["small_molecule_target"]
    state['current_challenge_targets_clip_interval'] = config["small_molecule_target_clip_interval"]

    logger.info(f"Target protein: {state['current_challenge_targets'][0]}")

    # Initialize score_results database
    logger.info("💾 Initializing score_results database...")
    init_score_results_db()
    logger.info(f"✅ Score results database initialized")

    # Log validation config
    logger.info(
        f"✅ Loaded validation config:"
        f"\n   - min_heavy_atoms: {config['min_heavy_atoms']}"
        f"\n   - min_rotatable_bonds: {config['min_rotatable_bonds']}"
        f"\n   - max_rotatable_bonds: {config['max_rotatable_bonds']}"
        f"\n   - banned_atom_types: {config['banned_atom_types']}"
    )

    logger.info(
        f"✅ Surrogate config:"
        f"\n   - GENERATE_MULTIPLIER: {GENERATE_MULTIPLIER}"
        f"\n   - SURROGATE_KEEP_RATIO: {SURROGATE_KEEP_RATIO}"
        f"\n   - SURROGATE_MIN_TRAIN_SIZE: {SURROGATE_MIN_TRAIN_SIZE}"
        f"\n   - BOLTZ_BUDGET (pre-training hard cap): {BOLTZ_BUDGET}"
    )

    # Import BoltzWrapper
    logger.info("🔬 Importing BoltzWrapper...")
    boltz_imported = _import_boltz_wrapper()

    # Initialize BoltzWrapper
    if boltz_imported and BoltzWrapper is not None:
        logger.info("🔬 Initializing BoltzWrapper...")
        try:
            state['boltz_wrapper'] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BoltzWrapper: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state['boltz_wrapper'] = None
    else:
        logger.warning("⚠️  BoltzWrapper not available, scoring will be skipped")
        state['boltz_wrapper'] = None

    # Run the main loop
    try:
        await run_generation_and_scoring_loop(state)
    except KeyboardInterrupt:
        logger.info("✅ Program stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
