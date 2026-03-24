import os
import sys
import math
import random
import logging
import asyncio
import time
import sqlite3
import traceback
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH            = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, "data", "mols.csv")
SCORE_RESULTS_DB   = os.path.join(BASE_DIR, "score_results.sqlite")
SURROGATE_PATH     = os.path.join(BASE_DIR, "surrogate_model.pkl")

from config.config_loader import load_config
from utils import get_heavy_atom_count, contains_atom_type, molecule_unique_for_protein_hf
from molecules_base import generate_inchikey
from combinatorial_db.reactions import get_smiles_from_reaction

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

HARDCODED_RXN_ID      = 2
STARTING_EPOCH        = 21632
SEED_SCORE_THRESHOLD  = 0.15
TOP_SEED_THRESHOLD    = 0.17

RXN_ROLE_MAP: Dict[int, Dict[str, Any]] = {
    1: {"name": "triazole",                     "roleA": 1,   "roleB": 2,    "roleC": None},
    2: {"name": "reductive_amination",           "roleA": 4,   "roleB": 8,    "roleC": None},
    3: {"name": "click_amide_cascade",           "roleA": 16,  "roleB": 32,   "roleC": 64},
    4: {"name": "suzuki_bromide",                "roleA": 128, "roleB": 512,  "roleC": None},
    5: {"name": "suzuki_bromide_then_chloride",  "roleA": 384, "roleB": 1024, "roleC": 1024},
}

# ── GPU / scoring ────────────────────────────────────────────────────────────
BOLTZ_BATCH_SIZE       = 10    # molecules per Boltz batch
BOLTZ_BUDGET_PER_ROUND = 100   # max molecules scored per round

# ── Candidate generation ─────────────────────────────────────────────────────
TIER1_NEIGHBOURS_PER_SEED = 60
TIER1_SIM_TOP_K           = 50
TIER1_SIM_MIN             = 0.15

TIER2_NEIGHBOURS_PER_SEED = 20
TIER2_SIM_TOP_K           = 30
TIER2_SIM_MIN             = 0.20

CHAMPION_TOP_A = 10
CHAMPION_TOP_B = 10

TIER1_GPU_FRACTION = 0.60
TIER2_GPU_FRACTION = 0.30
TIER3_GPU_FRACTION = 0.10

# ── Physchem thresholds ──────────────────────────────────────────────────────
HA_MIN = 13;  HA_MAX = 28
MW_MIN = 150; MW_MAX = 600
LOGP_MIN = -2.0; LOGP_MAX = 5.5
HBD_MAX = 5;  HBA_MAX = 10
TPSA_MAX = 140; ROTB_MAX = 10
QED_MIN = 0.35

# ── Diversity / output ───────────────────────────────────────────────────────
TANIMOTO_THRESHOLD   = 0.70
FINAL_DIVERSE_TOPK   = 50
GOOD_SCORE_THRESHOLD = 0.15

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def get_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return MORGAN_GEN.GetFingerprint(mol) if mol else None


def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def passes_physchem(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        return (
            HA_MIN   <= mol.GetNumHeavyAtoms()        <= HA_MAX   and
            MW_MIN   <= Descriptors.ExactMolWt(mol)   <= MW_MAX   and
            LOGP_MIN <= Descriptors.MolLogP(mol)      <= LOGP_MAX and
            Descriptors.NumHDonors(mol)               <= HBD_MAX  and
            Descriptors.NumHAcceptors(mol)            <= HBA_MAX  and
            CalcTPSA(mol)                             <= TPSA_MAX and
            Descriptors.NumRotatableBonds(mol)        <= ROTB_MAX and
            QED.qed(mol)                              >= QED_MIN
        )
    except Exception:
        return False


def parse_mol_name(mol_name: str) -> Optional[Tuple[int, int, Optional[int]]]:
    """'rxn:2:1045:332' → (1045, 332, None)"""
    try:
        parts = mol_name.split(":")
        if len(parts) == 4:
            return int(parts[2]), int(parts[3]), None
        elif len(parts) == 5:
            return int(parts[2]), int(parts[3]), int(parts[4])
    except (ValueError, IndexError):
        pass
    return None


def build_mol_name(rxn_id: int, a_id: int, b_id: int,
                   c_id: Optional[int] = None) -> str:
    if c_id is not None:
        return f"rxn:{rxn_id}:{a_id}:{b_id}:{c_id}"
    return f"rxn:{rxn_id}:{a_id}:{b_id}"


# ══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_components_for_reaction(rxn_id: int
    ) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Load building blocks using correct per-reaction role_mask bitmasks."""
    info = RXN_ROLE_MAP[rxn_id]

    def _fetch(mask: int) -> List[Tuple[int, str]]:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro&immutable=1", uri=True)
        cur  = conn.cursor()
        cur.execute(
            "SELECT mol_id, smiles FROM molecules WHERE (role_mask & ?) = ?",
            (mask, mask),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    mols_A = _fetch(info["roleA"])
    mols_B = _fetch(info["roleB"])
    mols_C = _fetch(info["roleC"]) if info["roleC"] is not None else []
    logger.info(
        f"Components rxn:{rxn_id} ({info['name']}): "
        f"A={len(mols_A)}, B={len(mols_B)}, C={len(mols_C)}"
    )
    return mols_A, mols_B, mols_C


def init_score_results_db() -> None:
    """
    Create score_results DB.

    Schema:
        molecule_name  TEXT PRIMARY KEY
        smiles         TEXT
        score          REAL
        scored_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        available      BOOLEAN DEFAULT TRUE

    Safe to call on an existing DB — ALTER TABLE migration guards ensure
    any missing columns are added without destroying existing data.
    """
    conn = sqlite3.connect(SCORE_RESULTS_DB)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scored_molecules (
            molecule_name TEXT PRIMARY KEY,
            smiles        TEXT,
            score         REAL,
            scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            available     BOOLEAN DEFAULT TRUE
        )
    """)

    # Migration: add any columns missing in older DBs
    existing = {row[1] for row in cur.execute("PRAGMA table_info(scored_molecules)")}
    if "score" not in existing:
        cur.execute("ALTER TABLE scored_molecules ADD COLUMN score REAL")
        logger.info("DB migrated: added column 'score'")
    if "smiles" not in existing:
        cur.execute("ALTER TABLE scored_molecules ADD COLUMN smiles TEXT")
        logger.info("DB migrated: added column 'smiles'")
    if "scored_at" not in existing:
        cur.execute(
            "ALTER TABLE scored_molecules "
            "ADD COLUMN scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
        logger.info("DB migrated: added column 'scored_at'")
    if "available" not in existing:
        cur.execute(
            "ALTER TABLE scored_molecules "
            "ADD COLUMN available BOOLEAN DEFAULT TRUE"
        )
        logger.info("DB migrated: added column 'available'")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)")
    conn.commit()
    conn.close()


def write_scores_to_db(molecules: List[Dict[str, Any]]) -> None:
    """
    Persist scored molecules to SQLite.

    Writes: molecule_name, smiles, score, scored_at (explicit UTC now), available=TRUE.
    scored_at is always set explicitly so INSERT OR REPLACE never leaves it NULL.
    Score key: mol["score"]  (pipeline convention for this file)
    """
    if not molecules:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(SCORE_RESULTS_DB)
    cur  = conn.cursor()
    rows = []
    for m in molecules:
        name   = m.get("name")
        score  = m.get("score")
        smiles = m.get("smiles", "")
        if name is None or score is None:
            continue
        rows.append((name, smiles, float(score), now, True))

    if rows:
        cur.executemany(
            """INSERT OR REPLACE INTO scored_molecules
               (molecule_name, smiles, score, scored_at, available)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        logger.info(f"Wrote {len(rows)} scored molecules to DB")
    conn.close()


def get_cached_scores(mol_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return already-scored molecules from DB (score IS NOT NULL)."""
    if not mol_names or not os.path.exists(SCORE_RESULTS_DB):
        return {}
    conn = sqlite3.connect(SCORE_RESULTS_DB)
    cur  = conn.cursor()
    ph   = ",".join("?" * len(mol_names))
    cur.execute(
        f"""SELECT molecule_name, smiles, score
            FROM scored_molecules
            WHERE molecule_name IN ({ph}) AND score IS NOT NULL""",
        mol_names,
    )
    result = {
        name: {"smiles": smi or "", "score": float(sc)}
        for name, smi, sc in cur.fetchall()
    }
    conn.close()
    return result


def batch_get_scores_from_db(mol_names: List[str]) -> Dict[str, float]:
    """
    Lightweight batch DB lookup: name → score.
    Used inside the scoring loop to skip already-scored molecules.
    """
    if not mol_names or not os.path.exists(SCORE_RESULTS_DB):
        return {}
    try:
        conn = sqlite3.connect(SCORE_RESULTS_DB)
        cur  = conn.cursor()
        ph   = ",".join("?" * len(mol_names))
        cur.execute(
            f"SELECT molecule_name, score FROM scored_molecules "
            f"WHERE molecule_name IN ({ph}) AND score IS NOT NULL",
            mol_names,
        )
        result = {name: float(sc) for name, sc in cur.fetchall()}
        conn.close()
        return result
    except Exception as e:
        logger.debug(f"batch_get_scores_from_db error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════
# SEED LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_seed_molecules(rxn_id: int, config: Any) -> pd.DataFrame:
    """
    Load high-scoring molecules from mols.csv AND score_results DB.
    Returns DataFrame sorted by score desc with columns:
      name, smiles, InChIKey, score, a_id, b_id, c_id
    """
    banned = getattr(config, "banned_atom_types", []) or []
    records: List[Dict] = []
    seen_keys: Set[str] = set()

    def _add(name: str, smiles: Optional[str], score: float):
        if not smiles:
            smiles = get_smiles_from_reaction(name)
        if not smiles:
            return
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        ha = mol.GetNumHeavyAtoms()
        if not (HA_MIN <= ha <= HA_MAX):
            return
        if banned and contains_atom_type(mol, banned):
            return
        ik = generate_inchikey(smiles)
        if not ik or ik in seen_keys:
            return
        parsed = parse_mol_name(name)
        if parsed is None:
            return
        a_id, b_id, c_id = parsed
        seen_keys.add(ik)
        records.append({
            "name":     name,
            "smiles":   smiles,
            "InChIKey": ik,
            "score":    float(score),
            "a_id":     a_id,
            "b_id":     b_id,
            "c_id":     c_id,
        })

    # ── Source 1: mols.csv ──────────────────────────────────────────
    if os.path.exists(REACTION_TRAIN_CSV):
        try:
            df = pd.read_csv(REACTION_TRAIN_CSV)
            if "epoch" in df.columns:
                df = df[df["epoch"] >= STARTING_EPOCH]
            # FIX: was df.get(...) which is wrong for pandas DataFrame
            if "molecule_name" in df.columns:
                df = df[df["molecule_name"].str.startswith(f"rxn:{rxn_id}:", na=False)]
            if "final_score" in df.columns:
                df = df[df["final_score"] >= SEED_SCORE_THRESHOLD]
            for _, row in df.iterrows():
                _add(row["molecule_name"], None, float(row.get("final_score", 0)))
            logger.info(f"mols.csv seeds: {len(records)}")
        except Exception as e:
            logger.warning(f"mols.csv load failed: {e}")

    csv_count = len(records)

    # ── Source 2: score_results DB ──────────────────────────────────
    if os.path.exists(SCORE_RESULTS_DB):
        try:
            conn = sqlite3.connect(SCORE_RESULTS_DB)
            cur  = conn.cursor()
            cur.execute(
                """SELECT molecule_name, smiles, score
                   FROM scored_molecules
                   WHERE molecule_name LIKE ?
                     AND score >= ?
                   ORDER BY score DESC""",
                (f"rxn:{rxn_id}:%", SEED_SCORE_THRESHOLD),
            )
            for name, smi, sc in cur.fetchall():
                _add(name, smi, float(sc))
            conn.close()
            logger.info(f"DB seeds: {len(records) - csv_count} additional")
        except Exception as e:
            logger.warning(f"DB seed load failed: {e}")

    if not records:
        logger.warning(
            f"No seeds found above {SEED_SCORE_THRESHOLD}. "
            f"Lower SEED_SCORE_THRESHOLD or check mols.csv."
        )
        return pd.DataFrame()

    df = (
        pd.DataFrame(records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    top_seeds = (df["score"] >= TOP_SEED_THRESHOLD).sum()
    logger.info(
        f"Seeds loaded: {len(df)} total | "
        f"top-tier (>={TOP_SEED_THRESHOLD}): {top_seeds} | "
        f"best: {df['score'].iloc[0]:.4f}"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════
# CHAMPION COMPONENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def analyse_champion_components(
    seeds_df: pd.DataFrame,
    top_n_a:  int = CHAMPION_TOP_A,
    top_n_b:  int = CHAMPION_TOP_B,
) -> Tuple[List[int], List[int]]:
    """
    Find which A and B component IDs appear most often in high-scoring seeds.
    Weight each seed's a_id and b_id by its score, return top-N by total weight.
    """
    a_weights: Dict[int, float] = defaultdict(float)
    b_weights: Dict[int, float] = defaultdict(float)

    for _, row in seeds_df.iterrows():
        score = float(row["score"])
        a_weights[int(row["a_id"])] += score
        b_weights[int(row["b_id"])] += score

    champion_a = [
        aid for aid, _ in
        sorted(a_weights.items(), key=lambda x: x[1], reverse=True)[:top_n_a]
    ]
    champion_b = [
        bid for bid, _ in
        sorted(b_weights.items(), key=lambda x: x[1], reverse=True)[:top_n_b]
    ]

    logger.info(
        f"Champion components: "
        f"A={champion_a[:5]}... ({len(champion_a)} total) | "
        f"B={champion_b[:5]}... ({len(champion_b)} total)"
    )
    return champion_a, champion_b


# ══════════════════════════════════════════════════════════════════════════
# COMPONENT FP INDEX
# ══════════════════════════════════════════════════════════════════════════

class ComponentIndex:
    """
    Morgan FP index for fast Tanimoto similarity search over building blocks.
    Built once at startup, reused every round.
    """

    def __init__(self, mols_A, mols_B, mols_C):
        logger.info("Building component FP index (one-time cost)...")
        self._id_to_smiles: Dict[int, str] = {}
        self.fps_A, self.ids_A = self._build(mols_A)
        self.fps_B, self.ids_B = self._build(mols_B)
        self.fps_C, self.ids_C = self._build(mols_C) if mols_C else ([], [])
        logger.info(
            f"Index ready: A={len(self.fps_A)}, "
            f"B={len(self.fps_B)}, C={len(self.fps_C)}"
        )

    def _build(self, mols):
        fps, ids = [], []
        for mol_id, smiles in mols:
            fp = get_fp(smiles)
            if fp:
                fps.append(fp)
                ids.append(mol_id)
                self._id_to_smiles[mol_id] = smiles
        return fps, ids

    def smiles(self, mol_id: int) -> Optional[str]:
        return self._id_to_smiles.get(mol_id)

    def similar(self, query_smiles: str, fps: list, ids: List[int],
                top_k: int, min_sim: float) -> List[int]:
        if not fps or not query_smiles:
            return []
        qfp = get_fp(query_smiles)
        if qfp is None:
            return []
        sims = DataStructs.BulkTanimotoSimilarity(qfp, fps)
        ranked = sorted(
            [(s, i) for s, i in zip(sims, ids) if s >= min_sim],
            reverse=True,
        )
        return [i for _, i in ranked[:top_k]]

    def similar_A(self, smi, top_k=TIER1_SIM_TOP_K, min_sim=TIER1_SIM_MIN):
        return self.similar(smi, self.fps_A, self.ids_A, top_k, min_sim)

    def similar_B(self, smi, top_k=TIER1_SIM_TOP_K, min_sim=TIER1_SIM_MIN):
        return self.similar(smi, self.fps_B, self.ids_B, top_k, min_sim)

    def random_A(self, n=1): return random.sample(self.ids_A, min(n, len(self.ids_A)))
    def random_B(self, n=1): return random.sample(self.ids_B, min(n, len(self.ids_B)))
    def random_C(self, n=1): return random.sample(self.ids_C, min(n, len(self.ids_C)))


# ══════════════════════════════════════════════════════════════════════════
# TIERED CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def _try_build_candidate(
    rxn_id:      int,
    a_id:        int,
    b_id:        int,
    c_id:        Optional[int],
    seen_names:  Set[str],
    seen_smiles: Set[str],
    banned:      List[str],
    target_seq:  Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build one candidate molecule. Returns dict or None if invalid/duplicate.
    Adds name to seen_names regardless (so we never retry the same combo).

    HuggingFace check: if target_seq is provided, molecules already in the
    HF dataset are rejected here — they will never reach the Boltz scorer.
    """
    name = build_mol_name(rxn_id, a_id, b_id, c_id)
    if name in seen_names:
        return None
    seen_names.add(name)

    product = get_smiles_from_reaction(name)
    if not product:
        return None
    canon = canonicalize(product)
    if not canon or canon in seen_smiles:
        return None
    if not passes_physchem(canon):
        return None
    mol = Chem.MolFromSmiles(canon)
    if banned and contains_atom_type(mol, banned):
        return None
    ik = generate_inchikey(canon)
    if not ik:
        return None

    # ── HuggingFace uniqueness check ─────────────────────────────────
    # Reject molecules already known to the HF dataset so we never waste
    # GPU time scoring them.
    if target_seq:
        try:
            if not molecule_unique_for_protein_hf(target_seq, canon):
                logger.debug(f"HF duplicate skipped: {name}")
                return None
        except Exception as e:
            logger.debug(f"HF check error for {name}: {e} — allowing through")

    seen_smiles.add(canon)
    return {"name": name, "smiles": canon, "InChIKey": ik}


def generate_tiered_candidates(
    seeds_df:   pd.DataFrame,
    champion_a: List[int],
    champion_b: List[int],
    index:      ComponentIndex,
    rxn_id:     int,
    seen_names: Set[str],
    config:     Any,
    target_seq: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate candidates in three tiers.

    Tier 1 — Deep exploitation of top seeds (score >= TOP_SEED_THRESHOLD)
    Tier 2 — Broad exploration of mid seeds
    Tier 3 — Champion cross-product + champion × similar

    target_seq is passed to _try_build_candidate for HF uniqueness filtering.
    """
    banned      = getattr(config, "banned_atom_types", []) or []
    seen_smiles: Set[str] = set()
    is_3comp    = bool(index.ids_C)

    tier1: List[Dict] = []
    tier2: List[Dict] = []
    tier3: List[Dict] = []

    def _add(lst, a, b, c=None):
        r = _try_build_candidate(
            rxn_id, a, b, c,
            seen_names, seen_smiles, banned,
            target_seq=target_seq,
        )
        if r:
            lst.append(r)

    # ── Tier 1: top seeds ────────────────────────────────────────────
    top_seeds = seeds_df[seeds_df["score"] >= TOP_SEED_THRESHOLD]
    logger.info(
        f"Tier 1: {len(top_seeds)} top seeds × {TIER1_NEIGHBOURS_PER_SEED} neighbours"
    )

    for _, seed in top_seeds.iterrows():
        a_id  = int(seed["a_id"])
        b_id  = int(seed["b_id"])
        sim_B = (
            index.similar_B(index.smiles(b_id) or "", TIER1_SIM_TOP_K, TIER1_SIM_MIN)
            or index.random_B(TIER1_SIM_TOP_K)
        )
        sim_A = (
            index.similar_A(index.smiles(a_id) or "", TIER1_SIM_TOP_K, TIER1_SIM_MIN)
            or index.random_A(TIER1_SIM_TOP_K)
        )
        sim_C = index.random_C(10) if is_3comp else []
        half  = TIER1_NEIGHBOURS_PER_SEED // 2

        for _ in range(half):
            _add(tier1, a_id, random.choice(sim_B),
                 random.choice(sim_C) if sim_C else None)
        for _ in range(half):
            _add(tier1, random.choice(sim_A), b_id,
                 random.choice(sim_C) if sim_C else None)

    # ── Tier 2: mid seeds ────────────────────────────────────────────
    mid_seeds = seeds_df[
        (seeds_df["score"] >= SEED_SCORE_THRESHOLD) &
        (seeds_df["score"] <  TOP_SEED_THRESHOLD)
    ]
    logger.info(
        f"Tier 2: {len(mid_seeds)} mid seeds × {TIER2_NEIGHBOURS_PER_SEED} neighbours"
    )

    for _, seed in mid_seeds.iterrows():
        a_id  = int(seed["a_id"])
        b_id  = int(seed["b_id"])
        sim_B = (
            index.similar_B(index.smiles(b_id) or "", TIER2_SIM_TOP_K, TIER2_SIM_MIN)
            or index.random_B(TIER2_SIM_TOP_K)
        )
        sim_A = (
            index.similar_A(index.smiles(a_id) or "", TIER2_SIM_TOP_K, TIER2_SIM_MIN)
            or index.random_A(TIER2_SIM_TOP_K)
        )
        sim_C = index.random_C(5) if is_3comp else []

        n_A = int(TIER2_NEIGHBOURS_PER_SEED * 0.40)
        n_B = int(TIER2_NEIGHBOURS_PER_SEED * 0.40)
        n_C = TIER2_NEIGHBOURS_PER_SEED - n_A - n_B

        for _ in range(n_A):
            _add(tier2, a_id, random.choice(sim_B),
                 random.choice(sim_C) if sim_C else None)
        for _ in range(n_B):
            _add(tier2, random.choice(sim_A), b_id,
                 random.choice(sim_C) if sim_C else None)
        for _ in range(n_C):
            _add(tier2, random.choice(sim_A), random.choice(sim_B),
                 random.choice(sim_C) if sim_C else None)

    # ── Tier 3: champion cross-product ───────────────────────────────
    logger.info(
        f"Tier 3: {len(champion_a)} champion A × {len(champion_b)} champion B "
        f"+ similar expansions"
    )
    for ca in champion_a:
        for cb in champion_b:
            _add(tier3, ca, cb)

    for ca in champion_a:
        ca_smi = index.smiles(ca) or ""
        sim_B  = index.similar_B(ca_smi, top_k=20, min_sim=0.15) or index.random_B(20)
        for sb in random.sample(sim_B, min(10, len(sim_B))):
            _add(tier3, ca, sb)

    for cb in champion_b:
        cb_smi = index.smiles(cb) or ""
        sim_A  = index.similar_A(cb_smi, top_k=20, min_sim=0.15) or index.random_A(20)
        for sa in random.sample(sim_A, min(10, len(sim_A))):
            _add(tier3, sa, cb)

    logger.info(
        f"Candidates generated: "
        f"tier1={len(tier1)}, tier2={len(tier2)}, tier3={len(tier3)} "
        f"(total={len(tier1)+len(tier2)+len(tier3)})"
    )
    return {"tier1": tier1, "tier2": tier2, "tier3": tier3}


# ══════════════════════════════════════════════════════════════════════════
# SURROGATE MODEL
# ══════════════════════════════════════════════════════════════════════════

class SurrogateModel:
    """
    RF + GBM ensemble on Morgan FP → score.
    Saved to disk after each update so it persists across runs.
    """

    def __init__(self, max_train: int = 15_000):
        self.rf  = RandomForestRegressor(
            n_estimators=150, max_depth=10, n_jobs=-1, random_state=42
        )
        self.gbm = GradientBoostingRegressor(
            n_estimators=150, max_depth=4,
            learning_rate=0.05, subsample=0.8, random_state=42
        )
        self.max_train = max_train
        self.is_fitted = False
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    @classmethod
    def load_or_create(cls, path: str = SURROGATE_PATH) -> "SurrogateModel":
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                logger.info(
                    f"Surrogate loaded from {path} "
                    f"({'fitted' if model.is_fitted else 'unfitted'}, "
                    f"n={len(model._y) if model._y is not None else 0})"
                )
                return model
            except Exception as e:
                logger.warning(f"Could not load surrogate: {e} — creating fresh")
        return cls()

    def save(self, path: str = SURROGATE_PATH) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.warning(f"Could not save surrogate: {e}")

    def fit_from_df(self, df: pd.DataFrame) -> None:
        """Fit from DataFrame with 'smiles' and 'score' columns."""
        pairs = [
            (get_fp(s), sc)
            for s, sc in zip(df["smiles"], df["score"])
            if get_fp(s) is not None and sc is not None
        ]
        if len(pairs) < 10:
            logger.warning(f"Too few samples to fit surrogate ({len(pairs)})")
            return
        self._X = np.array([list(fp) for fp, _ in pairs], dtype=np.float32)
        self._y = np.array([sc for _, sc in pairs], dtype=np.float32)
        self._refit()

    def update(self, new_records: List[Dict[str, Any]]) -> None:
        """Add newly scored molecules and refit."""
        pairs = [
            (get_fp(r["smiles"]), r["score"])
            for r in new_records
            if get_fp(r.get("smiles", "")) is not None
            and r.get("score") is not None
        ]
        if not pairs:
            return
        X_new = np.array([list(fp) for fp, _ in pairs], dtype=np.float32)
        y_new = np.array([sc for _, sc in pairs], dtype=np.float32)

        self._X = np.vstack([self._X, X_new]) if self._X is not None else X_new
        self._y = np.concatenate([self._y, y_new]) if self._y is not None else y_new

        if len(self._y) > self.max_train:
            top    = np.argsort(self._y)[-(self.max_train // 2):]
            recent = np.arange(
                max(0, len(self._y) - self.max_train // 2), len(self._y)
            )
            keep   = np.unique(np.concatenate([top, recent]))
            self._X, self._y = self._X[keep], self._y[keep]

        self._refit()

    def _refit(self):
        if self._X is None or len(self._y) < 10:
            return
        logger.info(
            f"Surrogate refit: n={len(self._y)}, "
            f"y=[{self._y.min():.4f}, {self._y.max():.4f}]"
        )
        self.rf.fit(self._X, self._y)
        self.gbm.fit(self._X, self._y)
        self.is_fitted = True

    def predict(self, candidates: List[Dict[str, Any]],
                use_ucb: bool = False, beta: float = 1.5) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(candidates))
        fps       = [get_fp(r["smiles"]) for r in candidates]
        scores    = np.full(len(candidates), -np.inf)
        valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
        if not valid_idx:
            return scores
        X = np.array([list(fps[i]) for i in valid_idx], dtype=np.float32)
        if use_ucb:
            tree_preds = np.array([t.predict(X) for t in self.rf.estimators_])
            preds = tree_preds.mean(0) + beta * tree_preds.std(0)
        else:
            preds = 0.5 * self.rf.predict(X) + 0.5 * self.gbm.predict(X)
        for rank_i, orig_i in enumerate(valid_idx):
            scores[orig_i] = preds[rank_i]
        return scores

    def top_k(self, candidates: List[Dict[str, Any]], k: int,
              use_ucb: bool = False) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if not self.is_fitted:
            logger.info("Surrogate not fitted — using all candidates")
            return candidates[:k]
        scores  = self.predict(candidates, use_ucb=use_ucb)
        indices = np.argsort(scores)[::-1][:k]
        result  = [candidates[i] for i in indices]
        logger.info(
            f"Surrogate top-{k}: "
            f"pred_best={scores[indices[0]]:.4f} "
            f"({'UCB' if use_ucb else 'mean'})"
        )
        return result


# ══════════════════════════════════════════════════════════════════════════
# BATCHED BOLTZ2 SCORING
# ══════════════════════════════════════════════════════════════════════════

async def score_molecules_with_boltz_batched(
    candidates:  List[Dict[str, Any]],
    boltz:       Any,
    target_seq:  str,
    boltz_cfg:   Dict[str, Any],
    batch_size:  int = BOLTZ_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Score candidates with BoltzWrapper in batches of `batch_size`.

    Flow per batch:
      1. Check DB cache — skip already-scored molecules
      2. Call boltz.score_molecules_target() for the remainder
      3. Extract per-molecule scores (per_molecule_metric → target_scores → avg)
      4. Write newly scored molecules to DB immediately (scored_at + available set)
      6. Return all results (cached + newly scored)

    Score key written to each result dict: "score"
    DB write key expected by write_scores_to_db:  mol["score"]
    """
    if not candidates:
        return []

    total      = len(candidates)
    n_batches  = math.ceil(total / batch_size)
    dummy_hash = "0x" + "0" * 64
    all_results: List[Dict[str, Any]] = []

    logger.info(
        f"Boltz2 batched scoring: {total} molecules | "
        f"{n_batches} batches × {batch_size}"
    )

    # Pre-check DB cache for the whole list
    db_scores  = batch_get_scores_from_db([c["name"] for c in candidates])
    cache_hits = len(db_scores)
    if cache_hits:
        logger.info(f"  DB cache: {cache_hits}/{total} already scored, skipping GPU")

    for batch_idx in range(n_batches):
        batch = candidates[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # ── Split: cached vs needs scoring ───────────────────────────
        cached_in_batch   = []
        to_score_in_batch = []

        for mol in batch:
            if mol["name"] in db_scores:
                cached_in_batch.append({
                    **mol,
                    "score":        db_scores[mol["name"]],
                    "score_source": "db_cache",
                })
            else:
                to_score_in_batch.append(mol)

        logger.info(
            f"  Batch {batch_idx+1}/{n_batches}: "
            f"{len(cached_in_batch)} cached, "
            f"{len(to_score_in_batch)} need GPU scoring"
        )

        newly_scored: List[Dict[str, Any]] = []

        if to_score_in_batch:
            smiles_list = [m["smiles"] for m in to_score_in_batch]
            names_list  = [m["name"]   for m in to_score_in_batch]

            valid_molecules_by_uid = {
                0: {"smiles": smiles_list, "names": names_list}
            }
            score_dict = {
                0: {
                    "target_scores":     [[]],
                    "antitarget_scores": [[]],
                    "entropy":           None,
                    "entropy_boltz":     None,
                    "block_submitted":   None,
                    "push_time":         "",
                    "boltz_score":       None,
                }
            }
            subnet_config = {
                "weekly_target":        boltz_cfg.get("weekly_target", target_seq),
                "binding_pocket":       boltz_cfg.get("binding_pocket", None),
                "max_distance":         boltz_cfg.get("max_distance", None),
                "force":                boltz_cfg.get("force", False),
                "num_molecules_boltz":  len(smiles_list),
                "boltz_metric":         boltz_cfg.get(
                    "boltz_metric",
                    ["affinity_probability_binary", "affinity_pred_value"]
                ),
                "combination_strategy": boltz_cfg.get(
                    "combination_strategy", "heavy_atom_normalization"
                ),
                "sample_selection":     "first",
            }

            t0 = time.time()
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: boltz.score_molecules_target(
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config,
                        dummy_hash,
                    ),
                )
                elapsed = time.time() - t0

                # ── Extract scores ────────────────────────────────────
                pm_metric     = boltz.per_molecule_metric.get(0, {})
                pm_components = boltz.per_molecule_components.get(0, {})

                target_scores_raw  = score_dict[0].get("target_scores", [[]])
                target_scores_list: Optional[List[float]] = None
                if target_scores_raw and len(target_scores_raw[0]) > 0:
                    inner = target_scores_raw[0]
                    target_scores_list = inner if isinstance(inner, list) else [inner]

                batch_avg   = score_dict[0].get("boltz_score")
                batch_valid = 0

                for mol_idx, mol in enumerate(to_score_in_batch):
                    smiles     = mol["smiles"]
                    components = pm_components.get(smiles, {})

                    # Priority 1: per_molecule_metric
                    score = pm_metric.get(smiles)

                    # Priority 2: affinity_probability_binary from components
                    if score is None:
                        score = components.get("affinity_probability_binary")

                    # Priority 3: target_scores list (index-aligned)
                    if score is None and target_scores_list is not None:
                        if mol_idx < len(target_scores_list):
                            score = target_scores_list[mol_idx]
                        else:
                            try:
                                si = smiles_list.index(smiles)
                                if si < len(target_scores_list):
                                    score = target_scores_list[si]
                            except ValueError:
                                pass

                    # Priority 4: batch average fallback
                    if score is None and batch_avg is not None:
                        score = batch_avg

                    final_score = float(score) if score is not None else None
                    if final_score is not None:
                        batch_valid += 1

                    result = {
                        **mol,
                        "score":            final_score,
                        "affinity_value":   (
                            float(components["affinity_pred_value"])
                            if components.get("affinity_pred_value") is not None
                            else None
                        ),
                        "confidence_score": (
                            float(components["confidence_score"])
                            if components.get("confidence_score") is not None
                            else None
                        ),
                        "iptm":             (
                            float(components["iptm"])
                            if components.get("iptm") is not None
                            else None
                        ),
                        "ligand_iptm":      (
                            float(components["ligand_iptm"])
                            if components.get("ligand_iptm") is not None
                            else None
                        ),
                        "score_source":     "boltz",
                    }
                    newly_scored.append(result)

                logger.info(
                    f"    GPU batch {batch_idx+1}: "
                    f"{batch_valid}/{len(to_score_in_batch)} scored "
                    f"in {elapsed:.1f}s "
                    f"({elapsed/max(len(to_score_in_batch),1):.1f}s/mol)"
                )

                # ── Write to DB immediately (scored_at + available set) ─
                scored_ok = [r for r in newly_scored if r["score"] is not None]
                if scored_ok:
                    write_scores_to_db(scored_ok)
                    for r in scored_ok:
                        db_scores[r["name"]] = r["score"]

            except Exception as e:
                logger.error(f"  Batch {batch_idx+1} GPU scoring failed: {e}")
                logger.debug(traceback.format_exc())
                for mol in to_score_in_batch:
                    newly_scored.append({
                        **mol,
                        "score":        None,
                        "score_source": "error",
                    })

        # ── Merge batch results ──────────────────────────────────────
        all_results.extend(cached_in_batch)
        all_results.extend(newly_scored)

    # ── Final summary ────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("score") is not None]
    if valid:
        arr = np.array([r["score"] for r in valid])
        logger.info(
            f"Batched scoring complete: {len(valid)}/{len(all_results)} valid | "
            f"mean={arr.mean():.4f} | max={arr.max():.4f} | "
            f"above_{GOOD_SCORE_THRESHOLD}={(arr >= GOOD_SCORE_THRESHOLD).sum()} | "
            f"from_cache={cache_hits} | newly_scored={len(valid)-cache_hits}"
        )
    else:
        logger.warning("No valid scores returned from Boltz2 this round")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# DIVERSITY FILTER
# ══════════════════════════════════════════════════════════════════════════

def diverse_top_k(
    scored: List[Dict[str, Any]],
    k:      int   = FINAL_DIVERSE_TOPK,
    thr:    float = TANIMOTO_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Greedy Tanimoto diversity filter."""
    sorted_recs   = sorted(
        scored,
        key=lambda r: r.get("score") or float("-inf"),
        reverse=True,
    )
    selected_fps  = []
    selected_recs = []

    for rec in sorted_recs:
        if len(selected_recs) >= k:
            break
        fp = get_fp(rec["smiles"])
        if fp is None:
            continue
        if not selected_fps:
            selected_fps.append(fp)
            selected_recs.append(rec)
            continue
        if max(DataStructs.BulkTanimotoSimilarity(fp, selected_fps)) < thr:
            selected_fps.append(fp)
            selected_recs.append(rec)

    logger.info(
        f"Diversity filter: {len(sorted_recs)} → {len(selected_recs)} "
        f"(thr={thr})"
    )
    return selected_recs


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE LOOP
# ══════════════════════════════════════════════════════════════════════════

async def run_pipeline(state: Dict[str, Any]) -> None:
    """
    Single-GPU seed-driven pipeline loop.

    GPU budget per round is split by tier:
      Tier 1 (top seeds):  60% of BOLTZ_BUDGET_PER_ROUND
      Tier 2 (mid seeds):  30% of BOLTZ_BUDGET_PER_ROUND
      Tier 3 (champion):   10% of BOLTZ_BUDGET_PER_ROUND
    """
    config     = state["config"]
    rxn_id     = state["rxn_id"]
    target_seq = state["target_seq"]
    surrogate  = state["surrogate"]
    index      = state["index"]
    seen_names = state["seen_names"]
    boltz      = state["boltz"]

    boltz_cfg = {
        "weekly_target":        target_seq,
        "binding_pocket":       getattr(config, "binding_pocket", None),
        "max_distance":         getattr(config, "max_distance", None),
        "force":                getattr(config, "force", False),
        "boltz_metric":         getattr(
            config, "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"]
        ),
        "combination_strategy": getattr(
            config, "combination_strategy", "heavy_atom_normalization"
        ),
        "sample_selection":     getattr(config, "sample_selection", "first"),
    }

    t1_budget = int(BOLTZ_BUDGET_PER_ROUND * TIER1_GPU_FRACTION)
    t2_budget = int(BOLTZ_BUDGET_PER_ROUND * TIER2_GPU_FRACTION)
    t3_budget = BOLTZ_BUDGET_PER_ROUND - t1_budget - t2_budget
    round_num = 0

    while True:
        round_num += 1
        t0 = time.time()
        logger.info(f"\n{'='*65}")
        logger.info(
            f"ROUND {round_num} | rxn:{rxn_id} | "
            f"GPU budget={BOLTZ_BUDGET_PER_ROUND} "
            f"(T1={t1_budget}, T2={t2_budget}, T3={t3_budget})"
        )
        logger.info(f"{'='*65}")

        # ── 1. Load seeds ────────────────────────────────────────────
        seeds_df = load_seed_molecules(rxn_id, config)
        if seeds_df.empty:
            logger.warning("No seeds. Waiting 30s...")
            await asyncio.sleep(30)
            continue

        # Bootstrap surrogate from seeds on first round
        if not surrogate.is_fitted and len(seeds_df) >= 10:
            surrogate.fit_from_df(seeds_df)
            surrogate.save()
            logger.info(f"Surrogate bootstrapped from {len(seeds_df)} seeds")

        # ── 2. Champion component analysis ──────────────────────────
        top_seeds_for_champ = seeds_df[seeds_df["score"] >= TOP_SEED_THRESHOLD]
        if top_seeds_for_champ.empty:
            top_seeds_for_champ = seeds_df.head(20)
        champion_a, champion_b = analyse_champion_components(top_seeds_for_champ)

        # ── 3. Generate tiered candidates ────────────────────────────
        # target_seq passed so HF duplicates are filtered during generation
        tiers = generate_tiered_candidates(
            seeds_df, champion_a, champion_b,
            index, rxn_id, seen_names, config,
            target_seq=target_seq,
        )

        # ── 4. Surrogate rank within each tier ───────────────────────
        use_ucb = round_num <= 3

        t1_to_score = surrogate.top_k(tiers["tier1"], t1_budget, use_ucb=use_ucb)
        t2_to_score = surrogate.top_k(tiers["tier2"], t2_budget, use_ucb=use_ucb)
        t3_to_score = surrogate.top_k(tiers["tier3"], t3_budget, use_ucb=use_ucb)

        to_score = t1_to_score + t2_to_score + t3_to_score
        logger.info(
            f"Sending to scorer: {len(to_score)} "
            f"(T1={len(t1_to_score)}, T2={len(t2_to_score)}, T3={len(t3_to_score)})"
        )

        # ── 5. Batched Boltz2 scoring ────────────────────────────────
        if to_score:
            all_results = await score_molecules_with_boltz_batched(
                to_score,
                boltz,
                target_seq,
                boltz_cfg,
                batch_size=BOLTZ_BATCH_SIZE,
            )
        else:
            all_results = []
            logger.info("No candidates to score this round")

        newly_scored = [
            r for r in all_results
            if r.get("score") is not None and r.get("score_source") == "boltz"
        ]

        # ── 6. Update surrogate ──────────────────────────────────────
        if newly_scored:
            surrogate.update(newly_scored)
            surrogate.save()

        # ── 7. Merge all scored for this round ───────────────────────
        all_scored = [r for r in all_results if r.get("score") is not None]
        all_scored.sort(
            key=lambda r: r.get("score") or float("-inf"), reverse=True
        )

        if not all_scored:
            logger.warning("No scored molecules this round")
            await asyncio.sleep(5)
            continue

        # ── 8. Diverse top-K ─────────────────────────────────────────
        top_diverse = diverse_top_k(all_scored, k=FINAL_DIVERSE_TOPK)
        above_good  = sum(
            1 for r in top_diverse
            if (r.get("score") or 0) >= GOOD_SCORE_THRESHOLD
        )

        elapsed = time.time() - t0
        best    = top_diverse[0] if top_diverse else None
        logger.info(
            f"\nRound {round_num} done in {elapsed/60:.1f} min | "
            f"newly GPU-scored: {len(newly_scored)} | "
            f"diverse top-{FINAL_DIVERSE_TOPK}: {len(top_diverse)} | "
            f"above {GOOD_SCORE_THRESHOLD}: {above_good}"
        )
        if best:
            logger.info(
                f"Best this round: {best['name']} | "
                f"score={best['score']:.4f}"
            )

        await asyncio.sleep(2)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

async def main():
    logger.info("Starting Compound Miner (1 GPU, seed-driven)")

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    target_seq = (
        getattr(config, "weekly_target", None) or
        (config.get("weekly_target", "") if isinstance(config, dict) else "")
    )
    antitarget_seqs = (
        getattr(config, "antitargets", None) or
        (config.get("antitargets", []) if isinstance(config, dict) else [])
    ) or []

    logger.info(f"Target      : {target_seq[:60]}...")
    logger.info(f"Antitargets : {len(antitarget_seqs)}")
    logger.info(
        f"Reaction    : rxn:{HARDCODED_RXN_ID} "
        f"({RXN_ROLE_MAP[HARDCODED_RXN_ID]['name']})"
    )
    logger.info(
        f"GPU budget  : {BOLTZ_BUDGET_PER_ROUND} mols/round "
        f"(~{BOLTZ_BUDGET_PER_ROUND * 3 / 60:.0f} min/round at 3s/mol)"
    )

    # ── DB init ──────────────────────────────────────────────────────
    init_score_results_db()

    # ── Load building blocks ─────────────────────────────────────────
    try:
        mols_A, mols_B, mols_C = load_components_for_reaction(HARDCODED_RXN_ID)
    except Exception as e:
        logger.error(f"Component load failed: {e}")
        return

    if not mols_A or not mols_B:
        logger.error(
            f"Empty component pools (A={len(mols_A)}, B={len(mols_B)}). "
            f"Check DB and role_mask."
        )
        return

    # ── Component FP index (built once) ──────────────────────────────
    index = ComponentIndex(mols_A, mols_B, mols_C)

    # ── Surrogate (load from disk if available) ───────────────────────
    surrogate = SurrogateModel.load_or_create(SURROGATE_PATH)

    # Pre-fit from existing DB scores if surrogate not already fitted
    if not surrogate.is_fitted and os.path.exists(SCORE_RESULTS_DB):
        try:
            conn = sqlite3.connect(SCORE_RESULTS_DB)
            cur  = conn.cursor()
            cur.execute(
                """SELECT smiles, score
                   FROM scored_molecules
                   WHERE molecule_name LIKE ?
                     AND score IS NOT NULL
                     AND smiles IS NOT NULL
                   ORDER BY score DESC
                   LIMIT 10000""",
                (f"rxn:{HARDCODED_RXN_ID}:%",),
            )
            rows = cur.fetchall()
            conn.close()

            if len(rows) >= 10:
                pre_df = pd.DataFrame(rows, columns=["smiles", "score"])
                surrogate.fit_from_df(pre_df)
                surrogate.save()
                logger.info(f"Surrogate pre-fitted from {len(rows)} existing DB scores")
            else:
                logger.info(
                    f"Only {len(rows)} DB scores — "
                    f"surrogate will bootstrap from seeds in round 1"
                )
        except Exception as e:
            logger.warning(f"Could not pre-fit surrogate from DB: {e}")

    # ── seen_names: pre-populate from DB ─────────────────────────────
    seen_names: Set[str] = set()
    if os.path.exists(SCORE_RESULTS_DB):
        try:
            conn = sqlite3.connect(SCORE_RESULTS_DB)
            cur  = conn.cursor()
            cur.execute(
                "SELECT molecule_name FROM scored_molecules "
                "WHERE molecule_name LIKE ?",
                (f"rxn:{HARDCODED_RXN_ID}:%",),
            )
            seen_names = {row[0] for row in cur.fetchall()}
            conn.close()
            logger.info(f"Pre-loaded {len(seen_names)} seen molecule names from DB")
        except Exception as e:
            logger.warning(f"Could not pre-load seen_names: {e}")

    # ── Import BoltzWrapper ───────────────────────────────────────────
    boltz_scoring_dir = os.path.join(BASE_DIR, "boltz-scoring")
    boltz_src_dir     = os.path.join(boltz_scoring_dir, "boltz", "src")
    for d in [boltz_scoring_dir, boltz_src_dir]:
        if os.path.exists(d) and d not in sys.path:
            sys.path.insert(0, d)

    try:
        from boltz.wrapper import BoltzWrapper
        boltz = BoltzWrapper()
        logger.info("BoltzWrapper initialised on GPU:0")
    except Exception as e:
        logger.error(f"BoltzWrapper import/init failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ── State dict ────────────────────────────────────────────────────
    state: Dict[str, Any] = {
        "config":          config,
        "rxn_id":          HARDCODED_RXN_ID,
        "target_seq":      target_seq,
        "antitarget_seqs": antitarget_seqs,
        "surrogate":       surrogate,
        "index":           index,
        "seen_names":      seen_names,
        "boltz":           boltz,
        "mols_A":          mols_A,
        "mols_B":          mols_B,
        "mols_C":          mols_C,
    }

    # ── Run pipeline ──────────────────────────────────────────────────
    try:
        await run_pipeline(state)
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C) — surrogate saved")
        surrogate.save()
    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}")
        logger.error(traceback.format_exc())
        surrogate.save()   # always save surrogate on crash


if __name__ == "__main__":
    asyncio.run(main())