"""
coordinate_ascent_search.py — Surrogate-guided coordinate-ascent search
for a single fixed reaction, built on the real MoleculeManager / Boltz
scoring stack from miner.py.

Design (per your spec):
  1. Load existing scores from score_results_{RXN_ID}.sqlite.
  2. Train a RandomForest surrogate on (Morgan fingerprint -> score).
  3. Coordinate ascent over reactant slots:
       - 2-component rxn:  fix B, sweep A (keep top 300) ->
                            fix best A, sweep B (keep top 200)
       - 3-component rxn:  fix B,C sweep A (keep top 150) ->
                            fix best A,C sweep B (keep top 150) ->
                            fix best A,B sweep C (keep top 150)
  4. At each step, ALL valid combinations for the swept slot are
     surrogate-scored first; only the kept-top-N are sent to real
     Boltz scoring.
  5. Final results are written to new_found.sqlite.
"""

import os
import sys
import time
import sqlite3
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from molecules import MoleculeManager, MoleculeUtils
from combinatorial_db.reactions import get_smiles_from_reaction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

RXN_ID = None
SCORE_RESULTS_DB = None
NEW_FOUND_DB = None
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

# ─────────────────────────────────────────────────────────────────────────
# Fingerprint generator — Morgan, radius 2, 2048 bits.
# Pattern verified against RDKit's official fingerprint generator API
# and tutorial. [[0]](#__0) [[2]](#__2) [[3]](#__3)
# ─────────────────────────────────────────────────────────────────────────
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)
_fp_cache: Dict[str, np.ndarray] = {}


def get_morgan_fingerprint(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    if smiles in _fp_cache:
        return _fp_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    arr[fp.GetOnBits()] = 1
    _fp_cache[smiles] = arr
    if len(_fp_cache) > 50000:
        for k in list(_fp_cache.keys())[:12500]:
            del _fp_cache[k]
    return arr


# ─────────────────────────────────────────────────────────────────────────
# Surrogate model — RandomForestRegressor on Morgan fingerprints.
# Constructor kwargs verified against the official sklearn reference. [[1]](#__1)
# ─────────────────────────────────────────────────────────────────────────
class SurrogateModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,
        )
        self.is_trained = False

    def fit(self, smiles_list: List[str], scores: List[float]) -> None:
        X, y = [], []
        for s, sc in zip(smiles_list, scores):
            if sc is None or not np.isfinite(sc):
                continue
            fp = get_morgan_fingerprint(s)
            if fp is not None:
                X.append(fp)
                y.append(float(sc))
        if len(X) < 20:
            logger.warning(
                f"[Surrogate] Only {len(X)} usable rows — skipping fit"
            )
            self.is_trained = False
            return
        t0 = time.time()
        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True
        logger.info(
            f"[Surrogate] Trained on {len(X)} rows in {time.time()-t0:.2f}s"
        )

    def update(self, smiles_list: List[str], scores: List[float]) -> None:
        """Incremental re-fit — RF has no partial_fit, so we refit fresh."""
        self.fit(smiles_list, scores)

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(smiles_list))
        fps = []
        for s in smiles_list:
            fp = get_morgan_fingerprint(s)
            fps.append(fp if fp is not None else np.zeros(2048, dtype=np.uint8))
        return self.model.predict(np.array(fps))


# ─────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────
def load_scored_molecules(rxn_id: int, db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        logger.warning(f"⚠️  {db_path} not found — starting with empty surrogate data")
        return pd.DataFrame(columns=["name", "smiles", "score"])
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT molecule_name, score FROM scored_molecules WHERE molecule_name LIKE ?",
        (f"rxn:{rxn_id}:%",),
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for name, score in rows:
        try:
            smiles = MoleculeUtils.get_smiles_from_reaction_cached(name)
            if smiles:
                out.append({"name": name, "smiles": smiles, "score": float(score)})
        except Exception:
            continue
    df = pd.DataFrame(out)
    logger.info(f"[DB] Loaded {len(df)} scored molecules from {db_path}")
    return df


def init_new_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scored_molecules (
            molecule_name TEXT PRIMARY KEY,
            score         REAL NOT NULL,
            scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            available     BOOLEAN DEFAULT TRUE
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"✅ new_found DB ready: {db_path}")


def write_results(results: List[Dict[str, Any]], db_path: str) -> None:
    if not results:
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = [
        (r["name"], float(r["score"]), True)
        for r in results
        if r.get("score") is not None and np.isfinite(r["score"])
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO scored_molecules (molecule_name, score, available) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    logger.info(f"✅ Wrote {len(rows)} rows → {db_path}")


# ─────────────────────────────────────────────────────────────────────────
# Boltz scoring wrapper (simplified — reuses BoltzWrapper.score_molecules)
# ─────────────────────────────────────────────────────────────────────────
async def score_with_boltz(
    boltz_wrapper,
    config: Dict[str, Any],
    target_protein: str,
    candidates: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """candidates: list of {'name':..., 'smiles':...}. Returns same list with 'score' filled in."""
    if boltz_wrapper is None or not candidates:
        return candidates

    valid_molecules_by_uid = {
        0: {
            "smiles": [c["smiles"] for c in candidates],
            "names": [c["name"] for c in candidates],
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
        "small_molecule_target_clip_interval": config["small_molecule_target_clip_interval"],
        "boltz_mode": getattr(config, "boltz_mode", "max"),
        "boltz_metric": getattr(
            config, "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"],
        ),
        "combination_strategy": getattr(
            config, "combination_strategy", "heavy_atom_normalization"
        ),
    }

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: boltz_wrapper.score_molecules(
            valid_molecules_by_uid, score_dict, subnet_config
        ),
    )

    final_scores = getattr(boltz_wrapper, "final_boltz_scores", {}).get(0, {})
    smiles_to_score = final_scores.get(target_protein, {}) if final_scores else {}

    for c in candidates:
        c["score"] = smiles_to_score.get(c["smiles"])
    return candidates


# ─────────────────────────────────────────────────────────────────────────
# Candidate generation for a given slot, holding others fixed
# ─────────────────────────────────────────────────────────────────────────
def build_candidates(
    manager: MoleculeManager,
    fixed: Dict[str, int],
    sweep_role: str,
    sweep_pool: List[int],
) -> List[Dict[str, str]]:
    """
    fixed: dict of role->id for the roles held constant, e.g. {'B': 7} or {'B':7,'C':3}
    sweep_role: 'A' | 'B' | 'C'
    sweep_pool: full list of ids to try for sweep_role
    """
    out = []
    for cand_id in sweep_pool:
        ids = dict(fixed)
        ids[sweep_role] = cand_id
        a = ids.get("A")
        b = ids.get("B")
        c = ids.get("C")
        if c is not None:
            name = f"rxn:{RXN_ID}:{a}:{b}:{c}"
        else:
            name = f"rxn:{RXN_ID}:{a}:{b}"
        smiles = MoleculeUtils.get_smiles_from_reaction_cached(name)
        if smiles:
            out.append({"name": name, "smiles": smiles})
    return out


# ─────────────────────────────────────────────────────────────────────────
# Core evaluate step: surrogate-prescore -> keep top N -> real Boltz score
# ─────────────────────────────────────────────────────────────────────────
async def evaluate_candidates(
    state: Dict[str, Any],
    manager: MoleculeManager,
    config: Dict[str, Any],
    surrogate: SurrogateModel,
    candidates: List[Dict[str, str]],
    keep_n: int,
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    df = pd.DataFrame(candidates)

    # validate SMILES/heavy-atom/banned-atom constraints via MoleculeManager
    df = manager.validate_molecules(config, df)
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    # surrogate prescore -> keep top N
    if surrogate.is_trained and len(df) > keep_n:
        preds = surrogate.predict(df["smiles"].tolist())
        df = df.copy()
        df["_pred"] = preds
        df = df.sort_values("_pred", ascending=False).head(keep_n).drop(columns=["_pred"])
        logger.info(f"[Surrogate] Prescored -> kept top {keep_n} of candidate pool")
    else:
        df = df.head(keep_n)

    # real Boltz scoring
    target = state["current_challenge_targets"][0]
    scored = await score_with_boltz(
        state["boltz_wrapper"], config, target, df.to_dict("records")
    )
    result = pd.DataFrame(scored)
    result = result[result["score"].notna()]
    return result


def parse_id(name: str, idx: int) -> int:
    return int(name.split(":")[idx])


# ─────────────────────────────────────────────────────────────────────────
# 2-component coordinate ascent
# ─────────────────────────────────────────────────────────────────────────
async def run_2component_search(
    state: Dict[str, Any],
    manager: MoleculeManager,
    config: Dict[str, Any],
    surrogate: SurrogateModel,
) -> pd.DataFrame:
    pool_A = manager.moles_A_id
    pool_B = manager.moles_B_id

    # Step 1: fix a starting B, sweep all A, keep top 300
    start_B = pool_B[0]
    cands = build_candidates(manager, {"B": start_B}, "A", pool_A)
    logger.info(f"[2-comp] Step 1: sweeping A ({len(cands)} candidates), fix B={start_B}")
    result_A = await evaluate_candidates(state, manager, config, surrogate, cands, keep_n=300)
    if result_A.empty:
        logger.warning("[2-comp] No valid results sweeping A — aborting")
        return result_A
    surrogate.update(result_A["smiles"].tolist(), result_A["score"].tolist())

    best_A_name = result_A.sort_values("score", ascending=False).iloc[0]["name"]
    best_A = parse_id(best_A_name, 2)
    logger.info(f"[2-comp] Best A = {best_A} (score={result_A['score'].max():.5f})")

    # Step 2: fix best A, sweep all B, keep top 200
    cands = build_candidates(manager, {"A": best_A}, "B", pool_B)
    logger.info(f"[2-comp] Step 2: sweeping B ({len(cands)} candidates), fix A={best_A}")
    result_B = await evaluate_candidates(state, manager, config, surrogate, cands, keep_n=200)
    if not result_B.empty:
        surrogate.update(result_B["smiles"].tolist(), result_B["score"].tolist())

    return pd.concat([result_A, result_B], ignore_index=True).drop_duplicates(subset=["name"])


# ─────────────────────────────────────────────────────────────────────────
# 3-component coordinate ascent
# ─────────────────────────────────────────────────────────────────────────
async def run_3component_search(
    state: Dict[str, Any],
    manager: MoleculeManager,
    config: Dict[str, Any],
    surrogate: SurrogateModel,
) -> pd.DataFrame:
    pool_A = manager.moles_A_id
    pool_B = manager.moles_B_id
    pool_C = manager.moles_C_id

    start_B, start_C = pool_B[0], pool_C[0]

    # Step 1: fix B,C, sweep A, keep top 150
    cands = build_candidates(manager, {"B": start_B, "C": start_C}, "A", pool_A)
    logger.info(f"[3-comp] Step 1: sweeping A ({len(cands)} candidates), fix B={start_B} C={start_C}")
    result_A = await evaluate_candidates(state, manager, config, surrogate, cands, keep_n=150)
    if result_A.empty:
        logger.warning("[3-comp] No valid results sweeping A — aborting")
        return result_A
    surrogate.update(result_A["smiles"].tolist(), result_A["score"].tolist())
    best_A = parse_id(result_A.sort_values("score", ascending=False).iloc[0]["name"], 2)
    logger.info(f"[3-comp] Best A = {best_A} (score={result_A['score'].max():.5f})")

    # Step 2: fix best A, C, sweep B, keep top 150
    cands = build_candidates(manager, {"A": best_A, "C": start_C}, "B", pool_B)
    logger.info(f"[3-comp] Step 2: sweeping B ({len(cands)} candidates), fix A={best_A} C={start_C}")
    result_B = await evaluate_candidates(state, manager, config, surrogate, cands, keep_n=150)
    if result_B.empty:
        logger.warning("[3-comp] No valid results sweeping B — stopping early")
        return pd.concat([result_A], ignore_index=True)
    surrogate.update(result_B["smiles"].tolist(), result_B["score"].tolist())
    best_B = parse_id(result_B.sort_values("score", ascending=False).iloc[0]["name"], 3)
    logger.info(f"[3-comp] Best B = {best_B} (score={result_B['score'].max():.5f})")

    # Step 3: fix best A, B, sweep C, keep top 150
    cands = build_candidates(manager, {"A": best_A, "B": best_B}, "C", pool_C)
    logger.info(f"[3-comp] Step 3: sweeping C ({len(cands)} candidates), fix A={best_A} B={best_B}")
    result_C = await evaluate_candidates(state, manager, config, surrogate, cands, keep_n=150)
    if not result_C.empty:
        surrogate.update(result_C["smiles"].tolist(), result_C["score"].tolist())

    return pd.concat(
        [result_A, result_B, result_C], ignore_index=True
    ).drop_duplicates(subset=["name"])


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
async def main(rxn_id: int):
    global RXN_ID, SCORE_RESULTS_DB, NEW_FOUND_DB
    RXN_ID = rxn_id
    SCORE_RESULTS_DB = os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")
    NEW_FOUND_DB = os.path.join(BASE_DIR, "new_found.sqlite")

    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    logger.info(
        f"✅ MoleculeManager rxn={rxn_id} | "
        f"A={len(manager.moles_A_id)} B={len(manager.moles_B_id)} "
        f"C={len(manager.moles_C_id)} | three_component={manager.is_three_component}"
    )

    state = {
        "config": cfg,
        "current_challenge_targets": cfg["small_molecule_target"],
        "boltz_wrapper": None,
    }

    # import + init BoltzWrapper (same pattern as miner.py)
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "boltz"))
        from boltz_wrapper import BoltzWrapper
        state["boltz_wrapper"] = BoltzWrapper()
        logger.info("✅ BoltzWrapper initialized")
    except Exception as e:
        logger.warning(f"⚠️  BoltzWrapper unavailable: {e}")

    # 1. load existing scores, train surrogate
    scored_df = load_scored_molecules(rxn_id, SCORE_RESULTS_DB)
    surrogate = SurrogateModel()
    if not scored_df.empty:
        surrogate.fit(scored_df["smiles"].tolist(), scored_df["score"].tolist())

    init_new_db(NEW_FOUND_DB)

    # 2. run coordinate ascent
    if manager.is_three_component:
        final_results = await run_3component_search(state, manager, cfg, surrogate)
    else:
        final_results = await run_2component_search(state, manager, cfg, surrogate)

    # 3. write results
    if not final_results.empty:
        write_results(final_results.to_dict("records"), NEW_FOUND_DB)
        best = final_results.sort_values("score", ascending=False).iloc[0]
        logger.info(f"🏆 Best found: {best['name']} score={best['score']:.6f}")
    else:
        logger.warning("⚠️  No results produced")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rxn_id", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.rxn_id))
