"""
merge_submitted_scores.py — Score submitted molecules with Boltz and merge
them into score_results_2.sqlite (table: scored_molecules).

Logic:
  1. Parse the list of submitted (name, smiles, inchikey) molecules.
  2. Check which molecule_names already exist in scored_molecules.
  3. For the NEW ones only, run them through Boltz (in batches),
     print each batch's results, and merge into the DB immediately
     with the real boltz_score and scored_at = now().

Usage:
    python merge_submitted_scores.py
"""

import os
import sys
import time
import asyncio
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List

import numpy as np

# ── project root ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

RXN_ID = 2
DB_PATH = os.path.join(BASE_DIR, f"score_results_{RXN_ID}.sqlite")
TABLE_NAME = "scored_molecules"
BOLTZ_BATCH_SIZE = 10

# ═══════════════════════════════════════════════════════════════════════════
# Submitted molecules: (molecule_name, smiles, inchikey)
# ═══════════════════════════════════════════════════════════════════════════
SUBMITTED_RAW = """
rxn:2:147486:168750,COc1cc2ccc(C(C)Nc3cc(-c4ccc(=O)[nH]n4)ccc3C)cc2cc1OC,SVJRUZQULZHRLC-UHFFFAOYSA-N
rxn:2:138879:170701,Cc1nc2c(NC(C)c3ccc(C)c(-c4ccc(-c5cccnc5)cc4)c3)n[nH]c(=O)n2n1,MWGAUVOTISDYBU-UHFFFAOYSA-N
rxn:2:73951:170701,Cc1ccc(C(C)Nc2cn[nH]c(=O)c2)cc1-c1ccc(-c2cccnc2)cc1,XYUISOCHDWCRGQ-UHFFFAOYSA-N
rxn:2:153842:114966,CCC(Nc1ccc(-c2n[nH]c(=O)cc2C)cc1)c1ccc(C)cc1,NEPQLUIXQQMMEQ-UHFFFAOYSA-N
rxn:2:138879:171071,Cc1nc2c(NC(C)c3c(C)cc(-c4cnc5[nH]ccc5c4)cc3C)n[nH]c(=O)n2n1,AZLWRIJXLGESBU-UHFFFAOYSA-N
rxn:2:138879:149517,CCn1nccc1-c1cc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)ccc1OC,PUNDCWUYCZVYLZ-UHFFFAOYSA-N
rxn:2:153842:118661,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCC[C@@H](C(C)(C)C)C2)cc1,HAFVKUUIEJBSRL-PYUWXLGESA-N
rxn:2:138879:156162,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1-c1cnc2c(ccn2C)c1,GJYIGSSEGWDJKG-UHFFFAOYSA-N
rxn:2:138879:164572,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1-c1cc(OC)c2nccn2c1,DCZBNFDQDVRVMX-UHFFFAOYSA-N
rxn:2:73951:171369,CCOc1ccc(-c2cccc(C(C)Nc3cn[nH]c(=O)c3)c2C)cc1C,JISNJGRPZPTYCA-UHFFFAOYSA-N
rxn:2:138879:162253,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1-c1cnn(C(C)(C)C)c1,RXTWFRRYVXYWAU-UHFFFAOYSA-N
rxn:2:138879:172098,Cc1nc2c(NC(C)c3ccc(C)c(-c4cnn(CCC(C)C)c4)c3)n[nH]c(=O)n2n1,IVWABQFTBVLPTE-UHFFFAOYSA-N
rxn:2:138879:70600,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1CN1CCCCCC1,QQNCICDCKIBXQL-UHFFFAOYSA-N
rxn:2:147486:166765,COc1ccc(-c2cnc(C(C)Nc3cc(-c4ccc(=O)[nH]n4)ccc3C)cn2)cc1OC,AKGQZAQWJDBKAF-UHFFFAOYSA-N
rxn:2:138879:79735,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)c(C)c1C,NSRIJCSJNKYOHE-UHFFFAOYSA-N
rxn:2:153842:135047,Cc1ccc2c(c1)C(Nc1ccc(-c3n[nH]c(=O)cc3C)cc1)[C@@H](C)C2,MJNSNHRFGZXKIO-XLEXHMCLSA-N
rxn:2:147486:161340,COc1ccc(-c2cc(C(C)Nc3cc(-c4ccc(=O)[nH]n4)ccc3C)no2)cc1OC,UEOXCYGZBMXEPV-UHFFFAOYSA-N
rxn:2:138879:83809,Cc1nc2c(NC(C)c3ccc([Si](C)(C)C)cc3)n[nH]c(=O)n2n1,KKBRVDYGNSVPNZ-UHFFFAOYSA-N
rxn:2:138879:75017,Cc1nc2c(NC(C)c3ccc(C)c4ccccc34)n[nH]c(=O)n2n1,HQMXHVBQWGHHMC-UHFFFAOYSA-N
rxn:2:73951:171139,CCc1cc(C)cc(CC)c1-c1ccc(C(C)Nc2cn[nH]c(=O)c2)c(C)c1,YAGAVZZWNUSXGW-UHFFFAOYSA-N
rxn:2:153842:134088,Cc1cc(=O)[nH]nc1-c1ccc(NC2CC(C(C)C)CCC2C)cc1,CSTRBEVVJQEBCE-UHFFFAOYSA-N
rxn:2:138879:133601,Cc1nc2c(NC(C)c3c(C)oc4c3CC(C)CC4)n[nH]c(=O)n2n1,SGEHRHWPGUUKFG-UHFFFAOYSA-N
rxn:2:153842:78550,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCCC(c3cnn(C)c3)C2)cc1,VBWJPGNDSONOCE-UHFFFAOYSA-N
rxn:2:138879:172271,Cc1nc2c(NC(C)c3c(C)cc(-c4ccc(OCC(C)C)cc4)cc3C)n[nH]c(=O)n2n1,XVDDMWVVCBNVRV-UHFFFAOYSA-N
rxn:2:138879:154023,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1-c1ccc2c(c1)[nH]c(=O)n2C,IHOIHWVRYBRFSY-UHFFFAOYSA-N
rxn:2:153842:86744,CCC(Nc1ccc(-c2n[nH]c(=O)cc2C)cc1)c1cc(OC)ccn1,ZSBZIPQTQXRBKO-UHFFFAOYSA-N
rxn:2:138879:167198,Cc1nc2c(NC(C)c3ccc(-c4ccc5c(=O)nc[nH]c5c4)c(C)c3)n[nH]c(=O)n2n1,WFMSWMMMGJCGBF-UHFFFAOYSA-N
rxn:2:153842:104907,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCCC(C3CCCC3)C2)cc1,VEBGVETWIRYCCX-UHFFFAOYSA-N

""".strip()


def parse_submitted(raw: str) -> List[Dict]:
    """Parse 'name,smiles,inchikey' lines into [{'name', 'smiles'}, ...]"""
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        name = parts[0].strip()
        smiles = parts[1].strip() if len(parts) > 1 else None
        records.append({"name": name, "smiles": smiles})
    return records


# ═══════════════════════════════════════════════════════════════════════════
# DB helpers
# ═══════════════════════════════════════════════════════════════════════════
def init_score_results_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            molecule_name TEXT PRIMARY KEY,
            score         REAL NOT NULL,
            scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            available     BOOLEAN DEFAULT TRUE
        )
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_score ON {TABLE_NAME}(score)")
    conn.commit()
    conn.close()


def get_already_scored_names(db_path: str, names: List[str]) -> set:
    if not names or not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(names))
    cur.execute(
        f"SELECT molecule_name FROM {TABLE_NAME} WHERE molecule_name IN ({placeholders})",
        names,
    )
    found = {r[0] for r in cur.fetchall()}
    conn.close()
    return found


def write_scores_to_db(db_path: str, records: List[Dict]) -> int:
    """Merge a batch of {'name','boltz_score'} into DB, with scored_at = now()."""
    if not records:
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        to_insert.append((name, score_f, now_str, True))
    if to_insert:
        cur.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} "
            "(molecule_name, score, scored_at, available) VALUES (?, ?, ?, ?)",
            to_insert,
        )
        conn.commit()
    conn.close()
    return len(to_insert)


# ═══════════════════════════════════════════════════════════════════════════
# BoltzWrapper import + scoring (same pattern as component_exhaust.py)
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
# Main
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    submitted = parse_submitted(SUBMITTED_RAW)
    logger.info(f"Total submitted molecules parsed: {len(submitted)}")

    init_score_results_db(DB_PATH)

    all_names = [m["name"] for m in submitted]
    already = get_already_scored_names(DB_PATH, all_names)
    new_molecules = [m for m in submitted if m["name"] not in already]

    logger.info(f"Already in DB: {len(submitted) - len(new_molecules)}")
    logger.info(f"New molecules to score: {len(new_molecules)}")

    if not new_molecules:
        logger.info("Nothing new to score. Exiting.")
        return

    config = load_config()
    cfg = dict(config) if isinstance(config, dict) else vars(config).copy()
    target_proteins = cfg["small_molecule_target"]

    if not _import_boltz_wrapper():
        logger.error("❌ BoltzWrapper unavailable — cannot score. Aborting.")
        return

    boltz = BoltzWrapper()

    total_written = 0
    total_batches = (len(new_molecules) + BOLTZ_BATCH_SIZE - 1) // BOLTZ_BATCH_SIZE

    for b in range(total_batches):
        batch = new_molecules[b * BOLTZ_BATCH_SIZE: (b + 1) * BOLTZ_BATCH_SIZE]
        logger.info(f"[Merge] Boltz batch {b+1}/{total_batches} ({len(batch)} mols)")

        scored = await score_with_boltz(boltz, cfg, target_proteins, batch)

        print(f"\n{'='*70}\nBATCH {b+1}/{total_batches} — {len(scored)} molecules scored\n{'='*70}")
        for m in scored:
            print(f"{m['name']:40s} {m.get('boltz_score')}")
        print(f"{'='*70}\n")

        n_written = write_scores_to_db(DB_PATH, scored)
        total_written += n_written
        logger.info(
            f"[Merge] 💾 Merged batch {b+1}/{total_batches} "
            f"({n_written} rows) → {DB_PATH} (running total: {total_written})"
        )

    logger.info(f"✅ Done. Total {total_written} new scores written → {DB_PATH}")


if __name__ == "__main__":
    asyncio.run(main())