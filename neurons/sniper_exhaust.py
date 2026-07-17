"""
component_exhaust.py — Exhaustive single-component search with surrogate
pre-filtering + Boltz scoring, for a fixed reaction (2 or 3 reactants).

2-reactant usage:
    python3 neurons/component_exhaust.py --rxn_id 2 --fix A --value 73951 --n 1000
    python3 neurons/component_exhaust.py --rxn_id 2 --fix B --value 171540 --n 1000

3-reactant usage:
    python3 neurons/component_exhaust.py --rxn_id 3 --fix A,B --value 55,120 --n 150
    python3 neurons/component_exhaust.py --rxn_id 3 --fix A,C --value 55,9   --n 150
    python3 neurons/component_exhaust.py --rxn_id 3 --fix B,C --value 120,9 --n 150

Behavior:
  1. Train surrogate using:
       - top 2000 scored molecules
       - bottom 2000 scored molecules
       - all scored molecules containing the fixed input component(s)

     Example:
       --rxn_id 2 --fix A --value 1000

     Adds all scored molecules:
       rxn:2:1000:<B>

  2. Enumerate all valid molecules by varying the free component.
  3. Surrogate-rank all valid candidates.
  4. Keep top --n.
  5. Boltz-score in batches.
  6. Every --boltz_batch_size molecules, default 10:
       - merge batch into score_results_{rxn_id}.sqlite
       - print only that batch's results in terminal.
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

from typing import Dict, List, Optional, Set
from sklearn.ensemble import RandomForestRegressor

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


# ──────────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

from config.config_loader import load_config
from utils import get_heavy_atom_count, contains_atom_type
from molecules import MoleculeManager, MoleculeUtils


# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Fingerprint config
# ──────────────────────────────────────────────────────────────────────────
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fp_cache: Dict[str, np.ndarray] = {}


# ──────────────────────────────────────────────────────────────────────────
# Surrogate training params
# ──────────────────────────────────────────────────────────────────────────
SURROGATE_TOP_N = 2000
SURROGATE_BOTTOM_N = 2000


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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_molecule_name ON scored_molecules(molecule_name)")

    conn.commit()
    conn.close()


def load_all_scored(db_path: str, rxn_id: int) -> pd.DataFrame:
    """
    Load every scored molecule for this reaction.
    Expected names:
      2-component: rxn:2:A:B
      3-component: rxn:3:A:B:C
    """
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["name", "score"])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT molecule_name, score
        FROM scored_molecules
        WHERE molecule_name LIKE ?
        """,
        (f"rxn:{rxn_id}:%",),
    )

    rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["name", "score"])
    if df.empty:
        return pd.DataFrame(columns=["name", "score"])

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[np.isfinite(df["score"])]

    return df.reset_index(drop=True)


def parse_reaction_name(name: str) -> Optional[Dict[str, int]]:
    """
    Parse:
      rxn:2:A:B
      rxn:3:A:B:C

    Return:
      {
        "rxn_id": 2,
        "A": 123,
        "B": 456,
        optionally "C": 789
      }
    """
    try:
        parts = name.split(":")
        if len(parts) not in (4, 5):
            return None
        if parts[0] != "rxn":
            return None

        out = {
            "rxn_id": int(parts[1]),
            "A": int(parts[2]),
            "B": int(parts[3]),
        }

        if len(parts) == 5:
            out["C"] = int(parts[4])

        return out

    except Exception:
        return None


def molecule_matches_fixed_components(name: str, rxn_id: int, fixed: Dict[str, int]) -> bool:
    """
    Return True if this scored molecule contains all fixed input components.

    Example:
      fixed = {"A": 1000}
      name = rxn:2:1000:555  -> True
      name = rxn:2:2000:555  -> False

    Example:
      fixed = {"A": 1000, "C": 77}
      name = rxn:3:1000:555:77 -> True
    """
    parsed = parse_reaction_name(name)
    if parsed is None:
        return False

    if parsed.get("rxn_id") != rxn_id:
        return False

    for role, value in fixed.items():
        if parsed.get(role) != int(value):
            return False

    return True


def get_already_scored_names(db_path: str, names: List[str]) -> Set[str]:
    """
    Return set of names already present in scored_molecules.

    Uses chunking so SQLite does not hit max variable limit for large lists.
    """
    if not names or not os.path.exists(db_path):
        return set()

    found = set()
    chunk_size = 900

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for i in range(0, len(names), chunk_size):
        chunk = names[i:i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(
            f"""
            SELECT molecule_name
            FROM scored_molecules
            WHERE molecule_name IN ({placeholders})
            """,
            chunk,
        )
        found.update(r[0] for r in cur.fetchall())

    conn.close()
    return found


def write_scores_to_db(db_path: str, records: List[Dict]) -> int:
    """
    Merge records into DB.

    Each record should contain:
      {
        "name": "rxn:2:A:B",
        "boltz_score": float
      }
    """
    if not records:
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    to_insert = []

    for r in records:
        name = r.get("name")
        score = r.get("boltz_score")

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
            """
            INSERT OR REPLACE INTO scored_molecules
            (molecule_name, score, available)
            VALUES (?, ?, ?)
            """,
            to_insert,
        )
        conn.commit()

    conn.close()
    return len(to_insert)


# ═══════════════════════════════════════════════════════════════════════════
# Terminal display helper
# ═══════════════════════════════════════════════════════════════════════════
def print_batch_results(
    batch_index: int,
    total_batches: int,
    scored_records: List[Dict],
    total_written: int,
    db_path: str,
) -> None:
    """
    Print every new Boltz batch to terminal immediately.
    """
    if not scored_records:
        print("\n" + "=" * 100, flush=True)
        print(f"BATCH {batch_index}/{total_batches}: no records returned", flush=True)
        print("=" * 100 + "\n", flush=True)
        return

    df = pd.DataFrame(scored_records)

    columns = []
    for col in ["name", "smiles", "surrogate_score", "boltz_score"]:
        if col in df.columns:
            columns.append(col)

    df = df[columns].copy()

    if "boltz_score" in df.columns:
        df = df.sort_values("boltz_score", ascending=False, na_position="last")

    print("\n" + "=" * 120, flush=True)
    print(f"BOLTZ BATCH RESULTS {batch_index}/{total_batches}", flush=True)
    print(f"Merged DB: {db_path}", flush=True)
    print(f"Running valid written total: {total_written}", flush=True)
    print("-" * 120, flush=True)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.max_colwidth", 90,
    ):
        print(df.to_string(index=False), flush=True)

    print("=" * 120 + "\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Surrogate model
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
        fixed: Dict[str, int],
        top_n: int = SURROGATE_TOP_N,
        bottom_n: int = SURROGATE_BOTTOM_N,
    ) -> None:
        """
        Train using:
          1. top 2000 by score
          2. bottom 2000 by score
          3. all scored molecules containing fixed input component(s)

        Example:
          rxn_id=2
          fixed={"A": 1000}

        Adds:
          rxn:2:1000:<B>
        """
        df = load_all_scored(db_path, rxn_id)

        if df.empty:
            logger.warning("[SURROGATE] No scored data found — cannot train")
            return

        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        top = df.head(top_n)
        bottom = df.tail(bottom_n)

        fixed_mask = df["name"].apply(
            lambda name: molecule_matches_fixed_components(name, rxn_id, fixed)
        )
        fixed_df = df[fixed_mask].copy()

        combined = (
            pd.concat([top, bottom, fixed_df], ignore_index=True)
            .drop_duplicates(subset="name")
            .reset_index(drop=True)
        )

        logger.info(
            "[SURROGATE] Training source counts: "
            f"top={len(top)} | bottom={len(bottom)} | "
            f"fixed_component_matches={len(fixed_df)} | "
            f"combined_unique={len(combined)}"
        )

        if len(fixed_df) > 0:
            logger.info(
                f"[SURROGATE] Added all existing scored molecules containing fixed={fixed}"
            )
        else:
            logger.warning(
                f"[SURROGATE] No existing scored molecules found containing fixed={fixed}"
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
                "skipping training because this is too few"
            )
            return

        t0 = time.time()
        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True

        logger.info(
            f"[SURROGATE] Trained on {len(X)} usable molecules in {time.time() - t0:.2f}s"
        )

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        if not self.is_trained:
            return np.zeros(len(smiles_list))

        fps = []

        for s in smiles_list:
            fp = get_morgan_fingerprint(s)
            if fp is None:
                fp = np.zeros(2048, dtype=np.uint8)
            fps.append(fp)

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

    df["smiles"] = df["name"].apply(
        MoleculeUtils.get_smiles_from_reaction_cached
    )

    df = df[df["smiles"].notna() & (df["smiles"] != "")]

    mask = df["smiles"].apply(lambda s: validate_smiles(s, config))
    df = df[mask].reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Candidate enumeration
# ═══════════════════════════════════════════════════════════════════════════
def build_candidate_names(
    rxn_id: int,
    manager: MoleculeManager,
    vary_role: str,
    fixed: Dict[str, int],
) -> List[str]:
    """
    Build names by varying one role over all library IDs.

    2-component:
      rxn:2:A:B

    3-component:
      rxn:3:A:B:C
    """
    pool_map = {
        "A": manager.moles_A_id,
        "B": manager.moles_B_id,
        "C": manager.moles_C_id,
    }

    if vary_role not in pool_map:
        raise ValueError(f"Invalid vary_role={vary_role}")

    vary_pool = pool_map[vary_role]

    is_three = manager.is_three_component
    names = []

    for vid in vary_pool:
        parts = {**fixed, vary_role: int(vid)}

        if is_three:
            names.append(f"rxn:{rxn_id}:{parts['A']}:{parts['B']}:{parts['C']}")
        else:
            names.append(f"rxn:{rxn_id}:{parts['A']}:{parts['B']}")

    return names


# ═══════════════════════════════════════════════════════════════════════════
# BoltzWrapper import + scoring
# ═══════════════════════════════════════════════════════════════════════════
BoltzWrapper = None


def _import_boltz_wrapper() -> bool:
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
    """
    Score molecules using BoltzWrapper.

    Input records should include:
      name
      smiles
      surrogate_score, optional

    Output adds:
      boltz_score
    """
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
            "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"],
        ),
        "combination_strategy": config.get(
            "combination_strategy",
            "heavy_atom_normalization",
        ),
    }

    def run_scoring():
        boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)

    t0 = time.time()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_scoring)

    logger.info(
        f"[Boltz] scored {len(molecules)} molecules in {time.time() - t0:.1f}s"
    )

    final_scores = getattr(boltz, "final_boltz_scores", {}).get(0, {})
    smiles_to_score = final_scores.get(primary_target, {}) if final_scores else {}

    results = []

    for m in molecules:
        out = dict(m)
        out["boltz_score"] = smiles_to_score.get(m["smiles"])
        results.append(out)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main exhaustive runner
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

    # 1. Enumerate candidates
    all_names = build_candidate_names(rxn_id, manager, vary_role, fixed)
    logger.info(f"[Exhaust] Enumerated {len(all_names)} raw candidates")

    # 2. Validate molecules
    valid_df = build_and_validate(all_names, config)
    logger.info(f"[Exhaust] {len(valid_df)} valid molecules after filtering")

    if valid_df.empty:
        logger.warning("[Exhaust] No valid molecules — aborting")
        return pd.DataFrame()

    # 3. Skip already scored molecules unless --rescore_seen
    if skip_already_scored:
        already = get_already_scored_names(db_path, valid_df["name"].tolist())
        if already:
            before = len(valid_df)
            valid_df = valid_df[~valid_df["name"].isin(already)].reset_index(drop=True)
            logger.info(
                f"[Exhaust] Skipped {before - len(valid_df)} already-scored molecules"
            )

    if valid_df.empty:
        logger.warning("[Exhaust] All candidates already scored — nothing to do")
        return pd.DataFrame()

    # 4. Surrogate score
    if surrogate.is_trained:
        t0 = time.time()
        preds = surrogate.predict(valid_df["smiles"].tolist())

        valid_df = valid_df.copy()
        valid_df["surrogate_score"] = preds
        valid_df = valid_df.sort_values("surrogate_score", ascending=False).reset_index(drop=True)

        logger.info(
            f"[Exhaust] Surrogate pre-scored {len(valid_df)} molecules "
            f"in {time.time() - t0:.2f}s"
        )

        logger.info("[Exhaust] Top surrogate candidates:")
        with pd.option_context(
            "display.max_rows", 20,
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 90,
        ):
            logger.info(
                "\n" + valid_df[["name", "smiles", "surrogate_score"]]
                .head(20)
                .to_string(index=False)
            )

    else:
        logger.warning("[Exhaust] Surrogate NOT trained — falling back to random order")
        valid_df = valid_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        valid_df["surrogate_score"] = np.nan

    # 5. Keep top n
    top_df = valid_df.head(n).reset_index(drop=True)
    logger.info(f"[Exhaust] Selected top {len(top_df)} for Boltz scoring")

    # 6. Boltz score in batches and print every batch
    all_scored = []
    total_written = 0
    total_batches = (len(top_df) + boltz_batch_size - 1) // boltz_batch_size

    for b in range(total_batches):
        batch = top_df.iloc[b * boltz_batch_size : (b + 1) * boltz_batch_size]

        mols = batch[["name", "smiles", "surrogate_score"]].to_dict("records")

        logger.info(
            f"[Exhaust] Boltz batch {b + 1}/{total_batches} "
            f"({len(mols)} molecules)"
        )

        scored = await score_with_boltz(
            boltz=boltz,
            config=config,
            target_proteins=target_proteins,
            molecules=mols,
        )

        all_scored.extend(scored)

        # Merge immediately after this batch
        n_written = write_scores_to_db(db_path, scored)
        total_written += n_written

        logger.info(
            f"[Exhaust] 💾 Merged batch {b + 1}/{total_batches} "
            f"({n_written} valid rows) → {db_path} "
            f"(running total: {total_written})"
        )

        # Print only this batch result
        print_batch_results(
            batch_index=b + 1,
            total_batches=total_batches,
            scored_records=scored,
            total_written=total_written,
            db_path=db_path,
        )

    logger.info(
        f"[Exhaust] ✅ Finished. Total valid written rows={total_written} → {db_path}"
    )

    result_df = pd.DataFrame(all_scored)

    if not result_df.empty:
        result_df = result_df.sort_values(
            "boltz_score",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        logger.info("[Exhaust] Final sorted results:")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 90,
        ):
            logger.info("\n" + result_df.to_string(index=False))

    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# Public API wrappers
# ═══════════════════════════════════════════════════════════════════════════
async def function_change_A(
    rxn_id,
    manager,
    config,
    surrogate,
    boltz,
    target_proteins,
    fixed: Dict[str, int],
    n: int,
    skip_already_scored: bool = True,
    boltz_batch_size: int = 10,
) -> pd.DataFrame:
    """
    Vary A.
    For 2-component, fixed should contain B.
    For 3-component, fixed should contain B and C.
    """
    return await run_component_exhaust(
        rxn_id=rxn_id,
        manager=manager,
        config=config,
        surrogate=surrogate,
        boltz=boltz,
        target_proteins=target_proteins,
        vary_role="A",
        fixed=fixed,
        n=n,
        skip_already_scored=skip_already_scored,
        boltz_batch_size=boltz_batch_size,
    )


async def function_change_B(
    rxn_id,
    manager,
    config,
    surrogate,
    boltz,
    target_proteins,
    fixed: Dict[str, int],
    n: int,
    skip_already_scored: bool = True,
    boltz_batch_size: int = 10,
) -> pd.DataFrame:
    """
    Vary B.
    For 2-component, fixed should contain A.
    For 3-component, fixed should contain A and C.
    """
    return await run_component_exhaust(
        rxn_id=rxn_id,
        manager=manager,
        config=config,
        surrogate=surrogate,
        boltz=boltz,
        target_proteins=target_proteins,
        vary_role="B",
        fixed=fixed,
        n=n,
        skip_already_scored=skip_already_scored,
        boltz_batch_size=boltz_batch_size,
    )


async def function_change_C(
    rxn_id,
    manager,
    config,
    surrogate,
    boltz,
    target_proteins,
    fixed: Dict[str, int],
    n: int,
    skip_already_scored: bool = True,
    boltz_batch_size: int = 10,
) -> pd.DataFrame:
    """
    Vary C.
    For 3-component only, fixed should contain A and B.
    """
    return await run_component_exhaust(
        rxn_id=rxn_id,
        manager=manager,
        config=config,
        surrogate=surrogate,
        boltz=boltz,
        target_proteins=target_proteins,
        vary_role="C",
        fixed=fixed,
        n=n,
        skip_already_scored=skip_already_scored,
        boltz_batch_size=boltz_batch_size,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Exhaustive single-component search with surrogate + Boltz"
    )

    parser.add_argument(
        "--rxn_id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--fix",
        type=str,
        required=True,
        help=(
            "Component(s) to fix, comma-separated. "
            "For 2-component: A or B. "
            "For 3-component: A,B or A,C or B,C."
        ),
    )

    parser.add_argument(
        "--value",
        type=str,
        required=True,
        help="Fixed component id(s), comma-separated, matching --fix order.",
    )

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Top-n surrogate-ranked candidates to Boltz-score.",
    )

    parser.add_argument(
        "--boltz_batch_size",
        type=int,
        default=10,
        help="Boltz score and merge this many molecules at a time. Default: 10.",
    )

    parser.add_argument(
        "--rescore_seen",
        action="store_true",
        help="If set, do not skip molecules already in the score DB.",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    args = parse_args()
    rxn_id = args.rxn_id

    fix_roles = [r.strip().upper() for r in args.fix.split(",") if r.strip()]
    fix_values = [int(v.strip()) for v in args.value.split(",") if v.strip()]

    if len(fix_roles) != len(fix_values):
        raise ValueError("--fix and --value must have the same number of entries")

    fixed = dict(zip(fix_roles, fix_values))

    logger.info(f"[CLI] rxn_id={rxn_id}")
    logger.info(f"[CLI] fixed={fixed}")
    logger.info(f"[CLI] n={args.n}")
    logger.info(f"[CLI] boltz_batch_size={args.boltz_batch_size}")
    logger.info(f"[CLI] rescore_seen={args.rescore_seen}")

    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"

    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    # Determine whether reaction has 2 or 3 components
    is_three = manager.is_three_component
    all_roles = {"A", "B", "C"} if is_three else {"A", "B"}

    expected_fixed_count = 2 if is_three else 1

    if len(fix_roles) != expected_fixed_count:
        raise ValueError(
            f"rxn={rxn_id} is {'3' if is_three else '2'}-component — "
            f"expected {expected_fixed_count} fixed role(s), got {len(fix_roles)}"
        )

    invalid_roles = set(fix_roles) - all_roles
    if invalid_roles:
        raise ValueError(
            f"Invalid fixed role(s): {invalid_roles}. "
            f"Allowed roles for this reaction: {all_roles}"
        )

    vary_candidates = all_roles - set(fix_roles)

    if len(vary_candidates) != 1:
        raise ValueError(
            f"Could not resolve a single varying role from --fix={args.fix}. "
            f"all_roles={all_roles}, fixed_roles={fix_roles}"
        )

    vary_role = vary_candidates.pop()

    logger.info(
        f"[CLI] Reaction is {'3-component' if is_three else '2-component'}"
    )
    logger.info(f"[CLI] Varying role will be: {vary_role}")

    db_path = score_db_path(rxn_id)
    init_score_results_db(db_path)

    # Train surrogate:
    # top 2000 + bottom 2000 + all molecules containing fixed component(s)
    surrogate = SurrogateModel()
    surrogate.train_from_db(
        rxn_id=rxn_id,
        db_path=db_path,
        fixed=fixed,
        top_n=SURROGATE_TOP_N,
        bottom_n=SURROGATE_BOTTOM_N,
    )

    # Boltz setup
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
        rxn_id=rxn_id,
        manager=manager,
        config=cfg,
        surrogate=surrogate,
        boltz=boltz,
        target_proteins=target_proteins,
        fixed=fixed,
        n=args.n,
        skip_already_scored=not args.rescore_seen,
        boltz_batch_size=args.boltz_batch_size,
    )

    if result_df is None or result_df.empty:
        logger.warning("No results produced.")
    else:
        logger.info(
            f"✅ Done. {len(result_df)} molecules processed. DB: {db_path}"
        )


if __name__ == "__main__":
    asyncio.run(main())