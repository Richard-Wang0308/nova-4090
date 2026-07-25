"""
component_exhaust.py — Iterative component search with surrogate
pre-filtering + Boltz scoring, for a fixed reaction (2 or 3 reactants).

2-reactant (A:B) usage — fix 1, vary 1:
    python component_exhaust.py --rxn_id 2 --fix A --value 73951 --n 100 --iterations 10
    python component_exhaust.py --rxn_id 2 --fix B --value 171540 --n 100 --iterations 10

3-reactant (A:B:C) usage — fix 1, vary 2:
    python component_exhaust.py --rxn_id 3 --fix A --value 55  --n 100 --iterations 20
    python component_exhaust.py --rxn_id 3 --fix B --value 120 --n 100 --iterations 20
    python component_exhaust.py --rxn_id 3 --fix C --value 9   --n 100 --iterations 20

Pipeline (repeated for --iterations rounds):
  1. Train / retrain surrogate on top-4000 + bottom-4000 scored molecules
     (rxn-specific, from score_results_{rxn_id}.sqlite)
  2. Sample up to --sample_size candidates (max 10000) from the free
     component space — never materialize the full cartesian product
  3. Validate (heavy atoms / banned atoms / rotatable bonds / RDKit parse)
  4. Drop molecules already scored (sqlite / rxn CSV) or already sampled
  5. Drop molecules already in HuggingFace Submission-Archive
  6. Surrogate pre-score the sampled valid set; keep top-n
  7. Boltz-score top-n in batches, merging each batch into
     score_results_{rxn_id}.sqlite immediately
  8. Retrain surrogate on updated DB before the next iteration
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
from typing import Dict, List, Optional, Set, Tuple
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

# Max candidates considered per iteration (hard cap)
MAX_SAMPLE_PER_ITER = 10_000

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
SURROGATE_TOP_N = 6000
SURROGATE_BOTTOM_N = 6000


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
        # Chunk to stay under SQLite variable limits
        chunk_size = 900
        for i in range(0, len(names), chunk_size):
            chunk = names[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cur.execute(
                f"SELECT molecule_name FROM scored_molecules "
                f"WHERE molecule_name IN ({placeholders})",
                chunk,
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
# Candidate sampling (never materializes the full cartesian product)
# ═══════════════════════════════════════════════════════════════════════════
def _vary_pools(
    manager: MoleculeManager,
    vary_roles: List[str],
) -> List[List[int]]:
    pool_map = {
        "A": manager.moles_A_id,
        "B": manager.moles_B_id,
        "C": manager.moles_C_id,
    }
    if not vary_roles:
        raise ValueError("vary_roles must be non-empty")
    pools = []
    for role in vary_roles:
        if role not in pool_map:
            raise ValueError(f"Invalid vary_role={role}")
        pools.append([int(x) for x in pool_map[role]])
    return pools


def candidate_space_size(manager: MoleculeManager, vary_roles: List[str]) -> int:
    pools = _vary_pools(manager, vary_roles)
    total = 1
    for p in pools:
        total *= max(len(p), 0)
    return int(total)


def _decode_flat_index(
    idx: int,
    vary_roles: List[str],
    pools: List[List[int]],
    sizes: List[int],
) -> Dict[str, int]:
    """Decode a flat product index into role→id (same order as itertools.product)."""
    parts: Dict[str, int] = {}
    rem = int(idx)
    coords = []
    for size in reversed(sizes):
        coords.append(rem % size)
        rem //= size
    coords.reverse()
    for role, pool, coord in zip(vary_roles, pools, coords):
        parts[role] = pool[coord]
    return parts


def _format_name(
    rxn_id: int,
    parts: Dict[str, int],
    is_three: bool,
) -> str:
    if is_three:
        return f"rxn:{rxn_id}:{parts['A']}:{parts['B']}:{parts['C']}"
    return f"rxn:{rxn_id}:{parts['A']}:{parts['B']}"


def sample_candidate_names(
    rxn_id: int,
    manager: MoleculeManager,
    vary_roles: List[str],
    fixed: Dict[str, int],
    sample_size: int,
    rng: np.random.Generator,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """
    Sample up to `sample_size` unique molecule names from the free-role
    cartesian product without enumerating the full space.

    exclude: names already considered / scored — skipped when drawn.
    """
    pools = _vary_pools(manager, vary_roles)
    sizes = [len(p) for p in pools]
    if any(s == 0 for s in sizes):
        return []

    total = 1
    for s in sizes:
        total *= s

    want = min(int(sample_size), total, MAX_SAMPLE_PER_ITER)
    if want <= 0:
        return []

    is_three = manager.is_three_component
    exclude = exclude or set()
    names: List[str] = []
    seen_idx: Set[int] = set()

    # When the space is small enough, draw without replacement in one shot
    if total <= MAX_SAMPLE_PER_ITER * 5:
        all_idx = rng.permutation(total)
        for idx in all_idx:
            idx = int(idx)
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            parts = {**fixed, **_decode_flat_index(idx, vary_roles, pools, sizes)}
            name = _format_name(rxn_id, parts, is_three)
            if name in exclude:
                continue
            names.append(name)
            if len(names) >= want:
                break
        return names

    # Large space: rejection-sample flat indices
    max_attempts = want * 50
    attempts = 0
    while len(names) < want and attempts < max_attempts:
        attempts += 1
        idx = int(rng.integers(0, total))
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        parts = {**fixed, **_decode_flat_index(idx, vary_roles, pools, sizes)}
        name = _format_name(rxn_id, parts, is_three)
        if name in exclude:
            continue
        names.append(name)
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
# Iterative search runner (sample → filter → Boltz → retrain)
# ═══════════════════════════════════════════════════════════════════════════
async def _boltz_score_batches(
    boltz,
    config: Dict,
    target_proteins: List[str],
    top_df: pd.DataFrame,
    db_path: str,
    boltz_batch_size: int,
    iter_label: str,
) -> Tuple[List[Dict], int]:
    """Boltz-score top_df in batches; merge each batch into DB. Returns (scored, n_written)."""
    all_scored: List[Dict] = []
    total_written = 0
    total_batches = (len(top_df) + boltz_batch_size - 1) // boltz_batch_size

    for b in range(total_batches):
        batch = top_df.iloc[b * boltz_batch_size : (b + 1) * boltz_batch_size]
        mols = batch[["name", "smiles"]].to_dict("records")
        logger.info(
            f"[Exhaust] {iter_label} Boltz batch {b+1}/{total_batches} ({len(mols)} mols)"
        )

        scored = await score_with_boltz(boltz, config, target_proteins, mols)
        all_scored.extend(scored)

        batch_df = pd.DataFrame(scored)
        if not batch_df.empty:
            batch_df = batch_df.sort_values(
                "boltz_score", ascending=False, na_position="last"
            )
            print(
                f"\n{'='*70}\n"
                f"{iter_label} BATCH {b+1}/{total_batches} — "
                f"{len(scored)} new molecules scored\n"
                f"{'='*70}"
            )
            print(batch_df[["name", "boltz_score"]].to_string(index=False))
            print(f"{'='*70}\n")

        n_written = write_scores_to_db(db_path, scored)
        total_written += n_written
        logger.info(
            f"[Exhaust] 💾 {iter_label} merged batch {b+1}/{total_batches} "
            f"({n_written} rows) → {db_path}"
        )

    return all_scored, total_written


async def run_component_exhaust(
    rxn_id: int,
    manager: MoleculeManager,
    config: Dict,
    surrogate: SurrogateModel,
    boltz,
    target_proteins: List[str],
    vary_roles: List[str],
    fixed: Dict[str, int],
    n: int,
    iterations: int = 1,
    sample_size: int = MAX_SAMPLE_PER_ITER,
    skip_already_scored: bool = True,
    boltz_batch_size: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    db_path = score_db_path(rxn_id)
    init_score_results_db(db_path)

    sample_size = max(1, min(int(sample_size), MAX_SAMPLE_PER_ITER))
    space_size = candidate_space_size(manager, vary_roles)
    rng = np.random.default_rng(seed)

    logger.info(
        f"[Exhaust] rxn={rxn_id} | vary={vary_roles} | fixed={fixed} | "
        f"top_n={n} | iterations={iterations} | sample_size={sample_size} | "
        f"search_space={space_size:,}"
    )

    primary_target = target_proteins[0] if target_proteins else None
    if not primary_target:
        logger.warning(
            "[Exhaust] No target protein — skipping HuggingFace uniqueness check"
        )

    # Names already considered this run (avoid re-sampling across iterations)
    seen_names: Set[str] = set()
    all_scored: List[Dict] = []
    total_written = 0

    for it in range(1, iterations + 1):
        iter_label = f"iter {it}/{iterations}"
        logger.info(f"[Exhaust] ── {iter_label} START ──")

        # ── 1. Sample ≤ sample_size candidates (no full enumeration) ───
        raw_names = sample_candidate_names(
            rxn_id, manager, vary_roles, fixed,
            sample_size=sample_size, rng=rng, exclude=seen_names,
        )
        if not raw_names:
            logger.warning(
                f"[Exhaust] {iter_label}: no new candidates left to sample — stopping"
            )
            break

        seen_names.update(raw_names)
        logger.info(
            f"[Exhaust] {iter_label}: sampled {len(raw_names)} candidates "
            f"(seen_total={len(seen_names):,} / space={space_size:,})"
        )

        # ── 2. Validate ────────────────────────────────────────────────
        valid_df = build_and_validate(raw_names, config)
        logger.info(
            f"[Exhaust] {iter_label}: {len(valid_df)} valid after filtering"
        )
        if valid_df.empty:
            logger.warning(f"[Exhaust] {iter_label}: no valid molecules — next iter")
            continue

        # ── 3. Skip already-scored (sqlite + rxn CSV) ───────────────────
        if skip_already_scored:
            already = get_already_scored_names(
                db_path, rxn_id, valid_df["name"].tolist(),
            )
            if already:
                pre = len(valid_df)
                valid_df = valid_df[~valid_df["name"].isin(already)].reset_index(drop=True)
                logger.info(
                    f"[Exhaust] {iter_label}: skipped {pre - len(valid_df)} "
                    f"already-scored"
                )

        if valid_df.empty:
            logger.warning(
                f"[Exhaust] {iter_label}: all sampled candidates already scored"
            )
            continue

        # ── 4. Skip HuggingFace duplicates ─────────────────────────────
        if primary_target:
            pre = len(valid_df)
            unique_mask = valid_df["smiles"].apply(
                lambda s: molecule_unique_for_protein_hf(primary_target, s)
            )
            valid_df = valid_df[unique_mask].reset_index(drop=True)
            n_hf = pre - len(valid_df)
            if n_hf:
                logger.info(
                    f"[Exhaust] {iter_label}: skipped {n_hf} already in "
                    f"HuggingFace for {primary_target}"
                )

        if valid_df.empty:
            logger.warning(
                f"[Exhaust] {iter_label}: all remaining candidates already in HF"
            )
            continue

        # ── 5. Surrogate pre-score → keep top-n ─────────────────────────
        if surrogate.is_trained:
            t0 = time.time()
            preds = surrogate.predict(valid_df["smiles"].tolist())
            valid_df = valid_df.copy()
            valid_df["surrogate_score"] = preds
            valid_df = valid_df.sort_values("surrogate_score", ascending=False)
            logger.info(
                f"[Exhaust] {iter_label}: surrogate scored {len(valid_df)} "
                f"in {time.time()-t0:.2f}s"
            )
        else:
            logger.warning(
                f"[Exhaust] {iter_label}: surrogate NOT trained — random order"
            )
            valid_df = valid_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))

        top_df = valid_df.head(n).reset_index(drop=True)
        logger.info(
            f"[Exhaust] {iter_label}: selected top {len(top_df)} for Boltz"
        )

        # ── 6. Boltz score + merge into DB ─────────────────────────────
        scored, n_written = await _boltz_score_batches(
            boltz, config, target_proteins, top_df, db_path,
            boltz_batch_size, iter_label,
        )
        all_scored.extend(scored)
        total_written += n_written

        # ── 7. Retrain surrogate on updated DB for next iteration ──────
        if it < iterations:
            logger.info(
                f"[Exhaust] {iter_label}: retraining surrogate on updated DB…"
            )
            surrogate.train_from_db(rxn_id, db_path)
            if not surrogate.is_trained:
                logger.warning(
                    f"[Exhaust] {iter_label}: retrain failed / insufficient data"
                )

        logger.info(
            f"[Exhaust] ── {iter_label} DONE "
            f"(+{n_written} written, run_total={total_written}) ──"
        )

    logger.info(
        f"[Exhaust] ✅ Finished {iterations} iteration(s). "
        f"Total {total_written} new scores → {db_path}"
    )

    result_df = pd.DataFrame(all_scored)
    if not result_df.empty:
        result_df = result_df.sort_values(
            "boltz_score", ascending=False, na_position="last"
        )
        print(
            f"\n{'#'*70}\n"
            f"FINAL RANKED RESULTS ({len(result_df)} molecules across all iterations)\n"
            f"{'#'*70}"
        )
        print(result_df[["name", "boltz_score"]].to_string(index=False))
        print(f"{'#'*70}\n")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# Public API — fix one role, vary the rest
# ═══════════════════════════════════════════════════════════════════════════
async def function_fix_A(
    rxn_id, manager, config, surrogate, boltz, target_proteins,
    fixed: Dict[str, int], n: int, boltz_batch_size: int = 10,
    iterations: int = 1, sample_size: int = MAX_SAMPLE_PER_ITER,
) -> pd.DataFrame:
    """Fix A; vary B (2-component) or B×C (3-component)."""
    vary_roles = ["B", "C"] if manager.is_three_component else ["B"]
    return await run_component_exhaust(
        rxn_id, manager, config, surrogate, boltz, target_proteins,
        vary_roles=vary_roles, fixed=fixed, n=n,
        iterations=iterations, sample_size=sample_size,
        boltz_batch_size=boltz_batch_size,
    )


async def function_fix_B(
    rxn_id, manager, config, surrogate, boltz, target_proteins,
    fixed: Dict[str, int], n: int, boltz_batch_size: int = 10,
    iterations: int = 1, sample_size: int = MAX_SAMPLE_PER_ITER,
) -> pd.DataFrame:
    """Fix B; vary A (2-component) or A×C (3-component)."""
    vary_roles = ["A", "C"] if manager.is_three_component else ["A"]
    return await run_component_exhaust(
        rxn_id, manager, config, surrogate, boltz, target_proteins,
        vary_roles=vary_roles, fixed=fixed, n=n,
        iterations=iterations, sample_size=sample_size,
        boltz_batch_size=boltz_batch_size,
    )


async def function_fix_C(
    rxn_id, manager, config, surrogate, boltz, target_proteins,
    fixed: Dict[str, int], n: int, boltz_batch_size: int = 10,
    iterations: int = 1, sample_size: int = MAX_SAMPLE_PER_ITER,
) -> pd.DataFrame:
    """Fix C; vary A×B (3-component only)."""
    if not manager.is_three_component:
        raise ValueError("function_fix_C requires a 3-component reaction")
    return await run_component_exhaust(
        rxn_id, manager, config, surrogate, boltz, target_proteins,
        vary_roles=["A", "B"], fixed=fixed, n=n,
        iterations=iterations, sample_size=sample_size,
        boltz_batch_size=boltz_batch_size,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI + main
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative component search with surrogate + Boltz "
                    "(fix 1 reactant; sample+vary the rest each iteration)"
    )
    parser.add_argument("--rxn_id", type=int, required=True)
    parser.add_argument(
        "--fix", type=str, required=True,
        help="Single component to fix. "
             "'A' / 'B' for 2-reactant (vary the other), "
             "'A' / 'B' / 'C' for 3-reactant (vary the other two).",
    )
    parser.add_argument(
        "--value", type=str, required=True,
        help="Fixed component id matching --fix.",
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Top-n from each iteration's sample to Boltz-score.",
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of sample → score → retrain rounds (default 10).",
    )
    parser.add_argument(
        "--sample_size", type=int, default=MAX_SAMPLE_PER_ITER,
        help=f"Candidates to sample per iteration "
             f"(default {MAX_SAMPLE_PER_ITER}, hard max {MAX_SAMPLE_PER_ITER}).",
    )
    parser.add_argument(
        "--boltz_batch_size", type=int, default=10,
        help="Print + merge into DB every this-many molecules (default 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for candidate sampling (default 42).",
    )
    parser.add_argument(
        "--rescore_seen", action="store_true",
        help="If set, do NOT skip molecules already in sqlite or rxn CSV.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    rxn_id = args.rxn_id

    fix_roles = [r.strip().upper() for r in args.fix.split(",") if r.strip()]
    fix_values = [int(v.strip()) for v in args.value.split(",") if v.strip()]
    if len(fix_roles) != len(fix_values):
        raise ValueError("--fix and --value must have the same number of entries")
    if len(fix_roles) != 1:
        raise ValueError(
            f"Expected exactly 1 fixed role (fix one reactant, vary the rest), "
            f"got {len(fix_roles)}: {fix_roles}"
        )

    fixed = dict(zip(fix_roles, fix_values))
    fix_role = fix_roles[0]

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.n < 1:
        raise ValueError("--n must be >= 1")

    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    # ── Determine component count FIRST, then restrict role universe ────
    is_three = manager.is_three_component
    all_roles = {"A", "B", "C"} if is_three else {"A", "B"}

    if fix_role not in all_roles:
        raise ValueError(
            f"Invalid --fix={fix_role} for "
            f"{'3' if is_three else '2'}-component rxn={rxn_id} "
            f"(valid roles: {sorted(all_roles)})"
        )

    vary_roles = sorted(all_roles - {fix_role})
    expected_vary = 2 if is_three else 1
    if len(vary_roles) != expected_vary:
        raise ValueError(
            f"Could not resolve vary roles from --fix={args.fix} "
            f"(rxn={rxn_id} is {'3' if is_three else '2'}-component, "
            f"expected {expected_vary} free role(s), got {vary_roles})"
        )

    db_path = score_db_path(rxn_id)
    init_score_results_db(db_path)

    # ── Initial surrogate train on top-4000 + bottom-4000 ────────────────
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

    result_df = await run_component_exhaust(
        rxn_id, manager, cfg, surrogate, boltz, target_proteins,
        vary_roles=vary_roles, fixed=fixed, n=args.n,
        iterations=args.iterations,
        sample_size=args.sample_size,
        skip_already_scored=not args.rescore_seen,
        boltz_batch_size=args.boltz_batch_size,
        seed=args.seed,
    )

    if result_df is None or result_df.empty:
        logger.warning("No results produced.")
    else:
        logger.info(
            f"✅ Done. {len(result_df)} molecules scored and written to {db_path}"
        )


if __name__ == "__main__":
    asyncio.run(main())