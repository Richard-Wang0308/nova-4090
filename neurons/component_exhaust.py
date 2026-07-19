"""
component_exhaust.py — Exhaustive single-component search with surrogate
pre-filtering + Boltz scoring, for a fixed reaction (2 or 3 reactants).

2-reactant (A:B) usage:
    python component_exhaust.py --rxn_id 2 --fix A --value 73951 --n 1000
    python component_exhaust.py --rxn_id 2 --fix B --value 171540 --n 1000

3-reactant (A:B:C) usage:
    python component_exhaust.py --rxn_id 3 --fix A,B --value 55,120 --n 150
    python component_exhaust.py --rxn_id 3 --fix A,C --value 55,9   --n 150
    python component_exhaust.py --rxn_id 3 --fix B,C --value 120,9 --n 150

Pipeline:
  1. Train surrogate on top-4000 + bottom-4000 scored molecules
     (rxn-specific, from score_results_{rxn_id}.sqlite)
  2. Fix the given component(s); enumerate ALL valid molecules by
     varying the remaining (free) component
  3. Validate (heavy atoms / banned atoms / rotatable bonds / RDKit parse)
  4. Drop molecules already in score_results_{rxn_id}.sqlite or data/rxn{rxn_id}.csv
  5. Drop molecules already in HuggingFace Submission-Archive
     (they do NOT count toward the top-n generated budget)
  6. Surrogate pre-score the full valid set
  7. Keep top-n by predicted score
  8. Boltz-score the top-n survivors IN BATCHES, merging each batch
     into score_results_{rxn_id}.sqlite immediately AND printing the
     batch's results table to the terminal (every --boltz_batch_size
     molecules, default 10) — no waiting until the whole run finishes.
"""
import os
import sys
import time
import asyncio
import logging
import sqlite3
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

from config.config_loader import load_config
from utils import get_heavy_atom_count, contains_atom_type, molecule_unique_for_protein_hf
from molecules import MoleculeManager, MoleculeUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fp_cache: Dict[str, np.ndarray] = {}

# ── surrogate training params ────────────────────────────────────────────
SURROGATE_TOP_N = 25000
SURROGATE_BOTTOM_N = 25000


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint helpers
# ═══════════════════════════════════════════════════════════════════════════
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
    if len(_fp_cache) > 100_000:
        for k in list(_fp_cache.keys())[:25_000]:
            del _fp_cache[k]
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# Score DB helpers
# ═══════════════════════════════════════════════════════════════════════════
def score_db_path(rxn_id: int) -> str:
    return os.path.join(BASE_DIR, f"score_results_{rxn_id}.sqlite")


def rxn_csv_path(rxn_id: int) -> str:
    return os.path.join(BASE_DIR, "data", f"rxn{rxn_id}.csv")


def get_scored_names_from_csv(rxn_id: int) -> set:
    """Return molecule names listed in data/rxn{rxn_id}.csv for this reaction."""
    csv_path = rxn_csv_path(rxn_id)
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        if "molecule_name" not in df.columns:
            logger.warning(
                f"[Exhaust] {os.path.basename(csv_path)}: "
                f"no molecule_name column — skipping CSV dedup"
            )
            return set()
        df["molecule_name"] = (
            df["molecule_name"].astype(str).str.strip().str.lstrip("\ufeff")
        )
        prefix = f"rxn:{rxn_id}:"
        names = df[
            df["molecule_name"].str.startswith(prefix, na=False)
        ]["molecule_name"].tolist()
        return set(names)
    except Exception as e:
        logger.warning(f"[Exhaust] Failed to read {csv_path}: {e}")
        return set()


def init_score_results_db(db_path: str) -> None:
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


def load_all_scored(db_path: str, rxn_id: int) -> pd.DataFrame:
    """Load every scored row for this reaction (used for surrogate training)."""
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["name", "score"])
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT molecule_name, score FROM scored_molecules WHERE molecule_name LIKE ?",
        (f"rxn:{rxn_id}:%",),
    )
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["name", "score"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[np.isfinite(df["score"])]
    return df.reset_index(drop=True)


def get_already_scored_names(
    db_path: str,
    rxn_id: int,
    names: List[str],
) -> set:
    """Return names already present in sqlite and/or data/rxn{rxn_id}.csv."""
    if not names:
        return set()

    name_set = set(names)
    found: set = set()

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        placeholders = ",".join("?" * len(names))
        cur.execute(
            f"SELECT molecule_name FROM scored_molecules "
            f"WHERE molecule_name IN ({placeholders})",
            names,
        )
        found |= {r[0] for r in cur.fetchall()}
        conn.close()

    csv_names = get_scored_names_from_csv(rxn_id)
    found |= name_set & csv_names
    return found


def write_scores_to_db(db_path: str, records: List[Dict]) -> int:
    """Merge a batch of {'name','boltz_score'} records into the DB."""
    if not records:
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    to_insert = []
    for r in records:
        name, score = r.get("name"), r.get("boltz_score")
        if not name or score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score_f):
            continue
        to_insert.append((name, score_f, True))
    if to_insert:
        cur.executemany(
            "INSERT OR REPLACE INTO scored_molecules "
            "(molecule_name, score, available) VALUES (?, ?, ?)",
            to_insert,
        )
        conn.commit()
    conn.close()
    return len(to_insert)


# ═══════════════════════════════════════════════════════════════════════════
# Surrogate model (top-4000 + bottom-4000 training)
# ═══════════════════════════════════════════════════════════════════════════
class SurrogateModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def train_from_db(
        self,
        rxn_id: int,
        db_path: str,
        top_n: int = SURROGATE_TOP_N,
        bottom_n: int = SURROGATE_BOTTOM_N,
    ) -> None:
        df = load_all_scored(db_path, rxn_id)
        if df.empty:
            logger.warning("[SURROGATE] No scored data found — cannot train")
            return

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        top = df.head(top_n)
        bottom = df.tail(bottom_n)
        combined = pd.concat([top, bottom]).drop_duplicates(subset="name")

        logger.info(
            f"[SURROGATE] Training set: top={len(top)} bottom={len(bottom)} "
            f"combined_unique={len(combined)}"
        )

        combined = combined.copy()
        combined["smiles"] = combined["name"].apply(
            MoleculeUtils.get_smiles_from_reaction_cached
        )
        combined = combined[combined["smiles"].notna() & (combined["smiles"] != "")]

        X, y = [], []
        for smi, score in zip(combined["smiles"], combined["score"]):
            fp = get_morgan_fingerprint(smi)
            if fp is not None:
                X.append(fp)
                y.append(float(score))

        if len(X) < 50:
            logger.warning(
                f"[SURROGATE] Only {len(X)} usable training samples — "
                f"skipping training (too few)"
            )
            return

        t0 = time.time()
        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True
        logger.info(
            f"[SURROGATE] Trained on {len(X)} samples in {time.time()-t0:.2f}s"
        )

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(smiles_list))
        fps = []
        for s in smiles_list:
            fp = get_morgan_fingerprint(s)
            fps.append(fp if fp is not None else np.zeros(2048, dtype=np.uint8))
        return self.model.predict(np.array(fps))


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════
def validate_smiles(smiles: str, config: Dict) -> bool:
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


def build_and_validate(names: List[str], config: Dict) -> pd.DataFrame:
    df = pd.DataFrame({"name": names})
    df["smiles"] = df["name"].apply(MoleculeUtils.get_smiles_from_reaction_cached)
    df = df[df["smiles"].notna() & (df["smiles"] != "")]
    mask = df["smiles"].apply(lambda s: validate_smiles(s, config))
    df = df[mask].reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Candidate enumeration (generic — works for 2 or 3 component reactions)
# ═══════════════════════════════════════════════════════════════════════════
def build_candidate_names(
    rxn_id: int,
    manager: MoleculeManager,
    vary_role: str,
    fixed: Dict[str, int],
) -> List[str]:
    """
    vary_role: 'A', 'B', or 'C' — the component to enumerate over ALL ids
    fixed: dict of the OTHER component(s) held constant, e.g. {'B': 171540}
           or {'B': 171540, 'C': 9} for 3-component reactions
    """
    pool_map = {
        "A": manager.moles_A_id,
        "B": manager.moles_B_id,
        "C": manager.moles_C_id,
    }
    vary_pool = pool_map[vary_role]

    is_three = manager.is_three_component
    names = []
    for vid in vary_pool:
        parts = {**fixed, vary_role: vid}
        if is_three:
            names.append(f"rxn:{rxn_id}:{parts['A']}:{parts['B']}:{parts['C']}")
        else:
            names.append(f"rxn:{rxn_id}:{parts['A']}:{parts['B']}")
    return names


# ═══════════════════════════════════════════════════════════════════════════
# BoltzWrapper import + scoring
# ═══════════════════════════════════════════════════════════════════════════
BoltzWrapper = None


def _import_boltz_wrapper():
    global BoltzWrapper
    try:
        boltz_src_dir = os.path.join(BASE_DIR, "boltz")
        if boltz_src_dir not in sys.path:
            sys.path.insert(0, boltz_src_dir)
        from boltz_wrapper import BoltzWrapper as BW
        BoltzWrapper = BW
        logger.info("✅ BoltzWrapper imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import BoltzWrapper: {e}")
        return False


async def score_with_boltz(
    boltz,
    config: Dict,
    target_proteins: List[str],
    molecules: List[Dict],
) -> List[Dict]:
    """Score `molecules` (list of {'name','smiles'}) with Boltz."""
    if not molecules:
        return []

    primary_target = target_proteins[0]
    output_dir = os.path.join(boltz.output_dir, "boltz_results_inputs")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(os.path.join(processed_dir, "structures"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "records"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "msa"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

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
        "small_molecule_target_clip_interval": config["small_molecule_target_clip_interval"],
        "boltz_mode": config.get("boltz_mode", "max"),
        "boltz_metric": config.get(
            "boltz_metric", ["affinity_probability_binary", "affinity_pred_value"]
        ),
        "combination_strategy": config.get(
            "combination_strategy", "heavy_atom_normalization"
        ),
    }

    def run_scoring():
        boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)

    t0 = time.time()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_scoring)
    logger.info(f"[Boltz] scored {len(molecules)} molecules in {time.time()-t0:.1f}s")

    final_scores = getattr(boltz, "final_boltz_scores", {}).get(0, {})
    smiles_to_score = final_scores.get(primary_target, {}) if final_scores else {}

    results = []
    for m in molecules:
        score = smiles_to_score.get(m["smiles"])
        m["boltz_score"] = score
        results.append(m)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Generic exhaustive-search runner (incremental DB merge + terminal print)
# ═══════════════════════════════════════════════════════════════════════════
async def run_component_exhaust(
    rxn_id: int,
    manager: MoleculeManager,
    config: Dict,
    surrogate: SurrogateModel,
    boltz,
    target_proteins: List[str],
    vary_role: str,
    fixed: Dict[str, int],
    n: int,
    skip_already_scored: bool = True,
    boltz_batch_size: int = 10,
) -> pd.DataFrame:
    db_path = score_db_path(rxn_id)
    init_score_results_db(db_path)

    logger.info(
        f"[Exhaust] rxn={rxn_id} | vary={vary_role} | fixed={fixed} | top_n={n}"
    )

    # ── 1. Enumerate ALL candidates for the varying component ───────────
    all_names = build_candidate_names(rxn_id, manager, vary_role, fixed)
    logger.info(f"[Exhaust] Enumerated {len(all_names)} raw candidates")

    # ── 2. Validate ────────────────────────────────────────────────────
    valid_df = build_and_validate(all_names, config)
    logger.info(f"[Exhaust] {len(valid_df)} valid molecules after filtering")

    if valid_df.empty:
        logger.warning("[Exhaust] No valid molecules — aborting")
        return pd.DataFrame()

    # ── 3. Optionally skip already-scored (sqlite + rxn CSV) ───────────
    if skip_already_scored:
        already = get_already_scored_names(
            db_path, rxn_id, valid_df["name"].tolist(),
        )
        if already:
            pre = len(valid_df)
            valid_df = valid_df[~valid_df["name"].isin(already)].reset_index(drop=True)
            csv_path = rxn_csv_path(rxn_id)
            logger.info(
                f"[Exhaust] Skipped {pre - len(valid_df)} already-scored molecules "
                f"(checked sqlite + {os.path.basename(csv_path)})"
            )

    if valid_df.empty:
        logger.warning("[Exhaust] All candidates already scored — nothing to do")
        return pd.DataFrame()

    # ── 4. Skip molecules already in HuggingFace (do not count as generated)
    primary_target = target_proteins[0] if target_proteins else None
    if primary_target:
        pre = len(valid_df)
        unique_mask = valid_df["smiles"].apply(
            lambda s: molecule_unique_for_protein_hf(primary_target, s)
        )
        valid_df = valid_df[unique_mask].reset_index(drop=True)
        n_hf_skipped = pre - len(valid_df)
        if n_hf_skipped:
            logger.info(
                f"[Exhaust] Skipped {n_hf_skipped} molecules already in "
                f"HuggingFace for {primary_target} "
                f"(not counted toward top-{n})"
            )
    else:
        logger.warning(
            "[Exhaust] No target protein — skipping HuggingFace uniqueness check"
        )

    if valid_df.empty:
        logger.warning(
            "[Exhaust] All candidates already in HuggingFace — nothing to do"
        )
        return pd.DataFrame()

    # ── 5. Surrogate pre-score ALL valid (non-HF) candidates ────────────
    if surrogate.is_trained:
        t0 = time.time()
        preds = surrogate.predict(valid_df["smiles"].tolist())
        valid_df = valid_df.copy()
        valid_df["surrogate_score"] = preds
        valid_df = valid_df.sort_values("surrogate_score", ascending=False)
        logger.info(
            f"[Exhaust] Surrogate pre-scored {len(valid_df)} molecules "
            f"in {time.time()-t0:.2f}s"
        )
    else:
        logger.warning(
            "[Exhaust] Surrogate NOT trained — falling back to random order"
        )
        valid_df = valid_df.sample(frac=1.0, random_state=42)

    # ── 6. Keep top n (HF molecules already excluded above) ─────────────
    top_df = valid_df.head(n).reset_index(drop=True)
    logger.info(f"[Exhaust] Selected top {len(top_df)} for Boltz scoring")

    # ── 7. Boltz score in batches — PRINT + MERGE after EVERY batch ────
    all_scored = []
    total_batches = (len(top_df) + boltz_batch_size - 1) // boltz_batch_size
    total_written = 0

    for b in range(total_batches):
        batch = top_df.iloc[b * boltz_batch_size : (b + 1) * boltz_batch_size]
        mols = batch[["name", "smiles"]].to_dict("records")
        logger.info(f"[Exhaust] Boltz batch {b+1}/{total_batches} ({len(mols)} mols)")

        scored = await score_with_boltz(boltz, config, target_proteins, mols)
        all_scored.extend(scored)

        # ── Print this batch's results to the terminal right now ───────
        batch_df = pd.DataFrame(scored)
        if not batch_df.empty:
            batch_df = batch_df.sort_values(
                "boltz_score", ascending=False, na_position="last"
            )
            print(
                f"\n{'='*70}\n"
                f"BATCH {b+1}/{total_batches} — {len(scored)} new molecules scored\n"
                f"{'='*70}"
            )
            print(batch_df[["name", "boltz_score"]].to_string(index=False))
            print(f"{'='*70}\n")

        # ✅ Merge this batch into score_results_{rxn_id}.sqlite immediately
        n_written = write_scores_to_db(db_path, scored)
        total_written += n_written
        logger.info(
            f"[Exhaust] 💾 Merged batch {b+1}/{total_batches} "
            f"({n_written} rows) → {db_path} "
            f"(running total: {total_written})"
        )

    logger.info(
        f"[Exhaust] ✅ Finished. Total {total_written} new scores "
        f"written → {db_path}"
    )

    result_df = pd.DataFrame(all_scored)
    if not result_df.empty:
        result_df = result_df.sort_values("boltz_score", ascending=False, na_position="last")
        print(f"\n{'#'*70}\nFINAL RANKED RESULTS ({len(result_df)} molecules)\n{'#'*70}")
        print(result_df[["name", "boltz_score"]].to_string(index=False))
        print(f"{'#'*70}\n")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# Public API — function_change_A / B / C
# ═══════════════════════════════════════════════════════════════════════════
async def function_change_A(
    rxn_id, manager, config, surrogate, boltz, target_proteins,
    fixed: Dict[str, int], n: int, boltz_batch_size: int = 10,
) -> pd.DataFrame:
    """Fix B (and C if 3-component); vary A over all valid molecules."""
    return await run_component_exhaust(
        rxn_id, manager, config, surrogate, boltz, target_proteins,
        vary_role="A", fixed=fixed, n=n, boltz_batch_size=boltz_batch_size,
    )


async def function_change_B(
    rxn_id, manager, config, surrogate, boltz, target_proteins,
    fixed: Dict[str, int], n: int, boltz_batch_size: int = 10,
) -> pd.DataFrame:
    """Fix A (and C if 3-component); vary B over all valid molecules."""
    return await run_component_exhaust(
        rxn_id, manager, config, surrogate, boltz, target_proteins,
        vary_role="B", fixed=fixed, n=n, boltz_batch_size=boltz_batch_size,
    )


async def function_change_C(
    rxn_id, manager, config, surrogate, boltz, target_proteins,
    fixed: Dict[str, int], n: int, boltz_batch_size: int = 10,
) -> pd.DataFrame:
    """Fix A and B; vary C over all valid molecules (3-component only)."""
    return await run_component_exhaust(
        rxn_id, manager, config, surrogate, boltz, target_proteins,
        vary_role="C", fixed=fixed, n=n, boltz_batch_size=boltz_batch_size,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI + main
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Exhaustive single-component search with surrogate + Boltz"
    )
    parser.add_argument("--rxn_id", type=int, required=True)
    parser.add_argument(
        "--fix", type=str, required=True,
        help="Component(s) to fix, comma-separated. "
             "'B' for 2-reactant (fix B, vary A), "
             "'A' for 2-reactant (fix A, vary B), "
             "'B,C' for 3-reactant (fix B & C, vary A), etc.",
    )
    parser.add_argument(
        "--value", type=str, required=True,
        help="Fixed component id(s), comma-separated, matching --fix order.",
    )
    parser.add_argument("--n", type=int, required=True, help="Top-n to Boltz-score")
    parser.add_argument(
        "--boltz_batch_size", type=int, default=10,
        help="Print + merge into DB every this-many molecules (default 10).",
    )
    parser.add_argument(
        "--rescore_seen", action="store_true",
        help="If set, do NOT skip molecules already in sqlite or rxn CSV.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    rxn_id = args.rxn_id

    fix_roles = [r.strip().upper() for r in args.fix.split(",")]
    fix_values = [int(v.strip()) for v in args.value.split(",")]
    if len(fix_roles) != len(fix_values):
        raise ValueError("--fix and --value must have the same number of entries")

    fixed = dict(zip(fix_roles, fix_values))

    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    # ── Determine component count FIRST, then restrict role universe ────
    is_three = manager.is_three_component
    all_roles = {"A", "B", "C"} if is_three else {"A", "B"}

    expected_fixed_count = 2 if is_three else 1
    if len(fix_roles) != expected_fixed_count:
        raise ValueError(
            f"rxn={rxn_id} is {'3' if is_three else '2'}-component — "
            f"expected {expected_fixed_count} fixed role(s), got {len(fix_roles)}"
        )

    vary_candidates = all_roles - set(fix_roles)
    if len(vary_candidates) != 1:
        raise ValueError(
            f"Could not resolve a single varying role from --fix={args.fix} "
            f"(rxn={rxn_id} is {'3' if is_three else '2'}-component, "
            f"all_roles={all_roles})"
        )
    vary_role = vary_candidates.pop()

    db_path = score_db_path(rxn_id)
    init_score_results_db(db_path)

    # ── Train surrogate on top-4000 + bottom-4000 ────────────────────────
    surrogate = SurrogateModel()
    surrogate.train_from_db(rxn_id, db_path)

    # ── Boltz setup ───────────────────────────────────────────────────────
    target_proteins = cfg["small_molecule_target"]
    boltz = None
    if _import_boltz_wrapper():
        boltz = BoltzWrapper()

    if boltz is None:
        logger.error("❌ BoltzWrapper unavailable — cannot score. Aborting.")
        return

    dispatch = {
        "A": function_change_A,
        "B": function_change_B,
        "C": function_change_C,
    }
    fn = dispatch[vary_role]

    result_df = await fn(
        rxn_id, manager, cfg, surrogate, boltz, target_proteins,
        fixed=fixed, n=args.n, boltz_batch_size=args.boltz_batch_size,
    )

    if result_df is None or result_df.empty:
        logger.warning("No results produced.")
    else:
        logger.info(
            f"✅ Done. {len(result_df)} molecules scored and written to {db_path}"
        )


if __name__ == "__main__":
    asyncio.run(main())