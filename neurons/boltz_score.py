"""
get_boltz_score.py

Standalone helper for the Nova DPEX-DJA miner codebase.

Given a component ID from pool A and a component ID from pool B (plus the
fixed reaction id), this:
  1. Builds the canonical molecule name "rxn:{rxn_id}:{A}:{B}"
     (same convention used throughout miner.py).
  2. Resolves its product SMILES via MoleculeUtils.get_smiles_from_reaction_cached
     (same resolver miner.py uses).
  3. Checks score_results_{rxn_id}.sqlite for a cached score first
     (same schema/table as miner.py's init_score_results_db).
  4. If not cached, runs it through the real BoltzWrapper
     (same code path as score_molecules_with_boltz_batched in miner.py).
  5. Writes the result back to the cache DB.
  6. Returns a single float score.

Usage:
    from get_boltz_score import get_boltz_score

    score = get_boltz_score(124422, 155326, rxn_id=1)
    print(score)   # e.g. 0.1098565...

CLI:
    python3 get_boltz_score.py 124422 155326 --rxn_id 1
"""

import os
import sys
import time
import logging
import sqlite3
from typing import Optional, Dict, Any

import numpy as np

# ── project root (same convention as miner.py) ─────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

from config.config_loader import load_config
from molecules import MoleculeManager, MoleculeUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Lazy singletons — config / MoleculeManager / BoltzWrapper are expensive
# to build. Build once per process (per rxn_id) and reuse across calls.
# ═══════════════════════════════════════════════════════════════════════
_CONFIG: Optional[Dict[str, Any]] = None
_MANAGERS: Dict[int, MoleculeManager] = {}
_BOLTZ_WRAPPER = None
_BOLTZ_AVAILABLE = False


def _get_config() -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
        logger.info("✅ Config loaded")
    return _CONFIG


def _get_manager(rxn_id: int) -> MoleculeManager:
    global _MANAGERS
    if rxn_id not in _MANAGERS:
        config = _get_config()
        cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
        cfg['allowed_reaction'] = f"rxn:{rxn_id}"
        _MANAGERS[rxn_id] = MoleculeManager(config=cfg, db_path=DB_PATH)
        mgr = _MANAGERS[rxn_id]
        logger.info(
            f"✅ MoleculeManager ready for rxn={rxn_id} | "
            f"A={len(mgr.moles_A_id)} B={len(mgr.moles_B_id)} "
            f"C={len(mgr.moles_C_id)}"
        )
    return _MANAGERS[rxn_id]


def _get_boltz_wrapper():
    """Import + instantiate BoltzWrapper exactly like miner.py's main()."""
    global _BOLTZ_WRAPPER, _BOLTZ_AVAILABLE
    if _BOLTZ_WRAPPER is not None:
        return _BOLTZ_WRAPPER

    boltz_src_dir = os.path.join(BASE_DIR, "boltz")
    if boltz_src_dir not in sys.path:
        sys.path.insert(0, boltz_src_dir)

    try:
        from boltz_wrapper import BoltzWrapper
        _BOLTZ_WRAPPER = BoltzWrapper()
        _BOLTZ_AVAILABLE = True
        logger.info("✅ BoltzWrapper initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize BoltzWrapper: {e}")
        import traceback
        logger.error(traceback.format_exc())
        _BOLTZ_WRAPPER = None
        _BOLTZ_AVAILABLE = False

    return _BOLTZ_WRAPPER


# ═══════════════════════════════════════════════════════════════════════
# Score DB helpers (identical schema to miner.py's
# score_results_{rxn_id}.sqlite / init_score_results_db)
# ═══════════════════════════════════════════════════════════════════════
def _score_db_path(rxn_id: int) -> str:
    return os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")


def _init_score_db(rxn_id: int) -> None:
    db_path = _score_db_path(rxn_id)
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)")
    conn.commit()
    conn.close()


def _get_cached_score(rxn_id: int, molecule_name: str) -> Optional[float]:
    db_path = _score_db_path(rxn_id)
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT score FROM scored_molecules WHERE molecule_name = ?",
            (molecule_name,),
        )
        row = cur.fetchone()
        conn.close()
        return float(row[0]) if row else None
    except Exception as e:
        logger.debug(f"Cache lookup failed for {molecule_name}: {e}")
        return None


def _write_score(rxn_id: int, molecule_name: str, score: float) -> None:
    if score is None or not np.isfinite(score):
        logger.warning(f"⚠️  Not caching non-finite score for {molecule_name}")
        return
    db_path = _score_db_path(rxn_id)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO scored_molecules (molecule_name, score, available) "
        "VALUES (?, ?, ?)",
        (molecule_name, float(score), True),
    )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Name / SMILES resolution (same convention as miner.py)
# ═══════════════════════════════════════════════════════════════════════
def build_molecule_name(rxn_id: int, component_A: int, component_B: int,
                         component_C: Optional[int] = None) -> str:
    """
    Builds the project's canonical molecule name string, e.g.:
        rxn:1:124422:155326            (two-component reaction)
        rxn:1:124422:155326:98765      (three-component reaction)
    """
    if component_C is not None:
        return f"rxn:{rxn_id}:{component_A}:{component_B}:{component_C}"
    return f"rxn:{rxn_id}:{component_A}:{component_B}"


def resolve_smiles(molecule_name: str) -> Optional[str]:
    """Resolve product SMILES via the project's own cached resolver."""
    smiles = MoleculeUtils.get_smiles_from_reaction_cached(molecule_name)
    if not smiles:
        logger.warning(f"⚠️  Could not resolve SMILES for {molecule_name}")
        return None
    return smiles


# ═══════════════════════════════════════════════════════════════════════
# Core: run ONE molecule through BoltzWrapper
# (mirrors score_molecules_with_boltz_batched from miner.py,
#  single-molecule case, minus the async/batch/HF-dedup plumbing)
# ═══════════════════════════════════════════════════════════════════════
def _run_boltz_single(molecule_name: str, smiles: str,
                       config: Dict[str, Any]) -> Optional[float]:
    boltz = _get_boltz_wrapper()
    if boltz is None:
        raise RuntimeError("BoltzWrapper is not available")

    target_proteins = config["small_molecule_target"]
    if not target_proteins:
        raise RuntimeError("No target proteins configured (small_molecule_target)")
    primary_target = target_proteins[0]

    # ── prepare output dirs, mirroring miner.py exactly ─────────────
    output_dir = os.path.join(boltz.output_dir, 'boltz_results_inputs')
    processed_dir = os.path.join(output_dir, 'processed')
    os.makedirs(os.path.join(processed_dir, 'structures'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'records'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'msa'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)

    valid_molecules_by_uid = {
        0: {'smiles': [smiles], 'names': [molecule_name]}
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
        'small_molecule_target': config['small_molecule_target'],
        'small_molecule_target_clip_interval': config['small_molecule_target_clip_interval'],
        'boltz_mode': config.get('boltz_mode', 'max'),
        'boltz_metric': config.get(
            'boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']
        ),
        'combination_strategy': config.get('combination_strategy', 'heavy_atom_normalization'),
    }

    logger.info(f"🔬 Running Boltz for {molecule_name} ...")
    t0 = time.time()
    boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)
    logger.info(f"✅ Boltz finished in {time.time()-t0:.2f}s")

    # ── extract score, same fallback chain used in miner.py ─────────
    uid = 0
    smiles_to_score = {}
    final_scores = getattr(boltz, 'final_boltz_scores', {}).get(uid, {})
    if primary_target and primary_target in final_scores:
        smiles_to_score = final_scores[primary_target]
    elif final_scores:
        smiles_to_score = next(iter(final_scores.values()))
    elif hasattr(boltz, 'per_molecule_metric') and uid in boltz.per_molecule_metric:
        smiles_to_score = boltz.per_molecule_metric[uid]

    if smiles in smiles_to_score:
        return float(smiles_to_score[smiles])

    target_scores = score_dict[uid].get('target_scores', [[]])
    if target_scores and len(target_scores[0]) > 0:
        val = target_scores[0][0] if isinstance(target_scores[0], list) else target_scores[0]
        return float(val)

    logger.warning(f"⚠️  No score produced for {molecule_name}")
    return None


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════
def get_boltz_score(component_A: int,
                     component_B: int,
                     rxn_id: int,
                     component_C: Optional[int] = None,
                     use_cache: bool = True) -> float:
    """
    Given a pool-A component id and a pool-B component id (and the fixed
    reaction id they belong to), resolve the product molecule + SMILES
    via this project's own combinatorial_db, run it through Boltz-2
    (or read the cached score if already scored before), and return a
    single float.

    Example:
        score = get_boltz_score(124422, 155326, rxn_id=1)
        print(score)   # 0.1098565...
    """
    config = _get_config()
    _get_manager(rxn_id)          # validates component ids exist for this rxn
    _init_score_db(rxn_id)

    molecule_name = build_molecule_name(rxn_id, component_A, component_B, component_C)

    if use_cache:
        cached = _get_cached_score(rxn_id, molecule_name)
        if cached is not None:
            logger.info(f"✅ Cache hit for {molecule_name}: {cached:.6f}")
            return cached

    smiles = resolve_smiles(molecule_name)
    if smiles is None:
        raise ValueError(f"Could not resolve SMILES for {molecule_name}")

    score = _run_boltz_single(molecule_name, smiles, config)
    if score is None:
        raise RuntimeError(f"Boltz produced no score for {molecule_name}")

    _write_score(rxn_id, molecule_name, score)
    return score


# ═══════════════════════════════════════════════════════════════════════
# CLI / example usage
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get Boltz score for a component-A + component-B pair"
    )
    parser.add_argument("component_A", type=int, help="Pool A component id, e.g. 124422")
    parser.add_argument("component_B", type=int, help="Pool B component id, e.g. 155326")
    parser.add_argument("--rxn_id", type=int, required=True, help="Reaction id, e.g. 1")
    parser.add_argument("--component_C", type=int, default=None,
                         help="Optional pool C component id (3-component reactions)")
    parser.add_argument("--no-cache", action="store_true",
                         help="Force re-scoring even if already cached")
    args = parser.parse_args()

    score = get_boltz_score(
        args.component_A, args.component_B, rxn_id=args.rxn_id,
        component_C=args.component_C, use_cache=not args.no_cache,
    )
    print(score)