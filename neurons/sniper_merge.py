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
rxn:2:111642:158214,CCc1nc(NC(C)c2ccc(C)c(-c3cccc4[nH]ncc34)c2)cc(=O)[nH]1,RYDNGESLIFPRHH-UHFFFAOYSA-N
rxn:2:68063:110720,CCOc1ccc(C)cc1C(CC)Nc1n[nH]c(=O)n1CC,FBVYUHQPSCUWFS-UHFFFAOYSA-N
rxn:2:138879:150240,Cc1nc2c(NC(C)c3cccc(-c4cnn(C5CCC5)c4)c3)n[nH]c(=O)n2n1,OBRLOOUEAWMYEP-UHFFFAOYSA-N
rxn:2:153842:117845,CCOc1ccccc1C(Nc1ccc(-c2n[nH]c(=O)cc2C)cc1)C1CC1,RLIATPNVARTSPD-UHFFFAOYSA-N
rxn:2:153842:132995,Cc1cc(=O)[nH]nc1-c1ccc(NC(C)c2cccc(C3CC3)c2)cc1,BEBSJYDFKVLWAF-UHFFFAOYSA-N
rxn:2:138879:98477,CCC(Nc1n[nH]c(=O)n2nc(C)nc12)c1cccc(OC(C)C)c1,LTVRIFUXSWPLJN-UHFFFAOYSA-N
rxn:2:138879:170620,COc1ccc2cc(-c3ccc(C(C)Nc4n[nH]c(=O)n5nc(C)nc45)c(OC)c3)[nH]c2c1,SOOUYAUAUJFNJX-UHFFFAOYSA-N
rxn:2:138879:172592,CCOCc1cccc(-c2cc(C(C)Nc3n[nH]c(=O)n4nc(C)nc34)ccc2C)c1,SLHKNEZQLMILQN-UHFFFAOYSA-N
rxn:2:138879:160075,Cc1nc2c(NC(C)c3c(C)cc(-c4cc[nH]c4)cc3C)n[nH]c(=O)n2n1,FCTILPUAACTEHF-UHFFFAOYSA-N
rxn:2:138879:151535,Cc1nc2c(NC(C)c3c(C)cc(-c4c[nH]nc4C)cc3C)n[nH]c(=O)n2n1,SOEUGWMRPWZTAY-UHFFFAOYSA-N
rxn:2:153842:101661,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCCC(C3CCCCC3)C2)cc1,DKQSZHUMJKJGJJ-UHFFFAOYSA-N
rxn:2:147486:162612,CCn1ncc2cc(-c3cnc(C(C)Nc4cc(-c5ccc(=O)[nH]n5)ccc4C)c(OC)c3)ccc21,CGTPCTAXJMYGII-UHFFFAOYSA-N
rxn:2:138879:127420,CCC(Nc1n[nH]c(=O)n2nc(C)nc12)c1ccc(C)c(OC)c1,RAHLMPCNWMDHQQ-UHFFFAOYSA-N
rxn:2:138879:169810,COc1ccc2oc(-c3ccc(C(C)Nc4n[nH]c(=O)n5nc(C)nc45)c(C)c3)cc2c1,HYSINTQGKFJBNW-UHFFFAOYSA-N
rxn:2:153842:127803,Cc1ccc(C(Nc2ccc(-c3n[nH]c(=O)cc3C)cc2)C2CC2)cc1C,NGZDMYIEQZQWFP-UHFFFAOYSA-N
rxn:2:153842:75009,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCCC2CCC(C)C)cc1,OMCBBBWBDDCJCL-UHFFFAOYSA-N
rxn:2:153842:138344,Cc1cc(=O)[nH]nc1-c1ccc(NC(c2ccc3[nH]ncc3c2)C2CC2)cc1,VYRQTGDYKVVURE-UHFFFAOYSA-N
rxn:2:138879:149625,Cc1nc2c(NC(C)c3cccc(-c4cnc(C)c(C)c4)c3C)n[nH]c(=O)n2n1,YZVLWOBALDPLLG-UHFFFAOYSA-N
rxn:2:153842:95767,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCCC23CCCCCCC3)cc1,ZUTKXWBRKSZMDT-UHFFFAOYSA-N
rxn:2:138879:157616,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1-c1cc2[nH]ncc2cc1OC,YUBUEOVBYJZFQL-UHFFFAOYSA-N
rxn:2:138879:163872,Cc1nc2c(NC(C)c3ccc(C)c(-c4ccc(N5CCCC5)nc4)c3)n[nH]c(=O)n2n1,SLLCRAMJODEKTL-UHFFFAOYSA-N
rxn:2:153842:129419,COc1ccc2c(c1)CC(C)(C)C2Nc1ccc(-c2n[nH]c(=O)cc2C)cc1,XHOBRWDASYBDKJ-UHFFFAOYSA-N
rxn:2:138879:132831,Cc1nc2c(NC(c3cnc4c(c3)CCCC4)C3CC3)n[nH]c(=O)n2n1,YVLMAAQVMSVRLM-UHFFFAOYSA-N
rxn:2:153842:131763,Cc1cc(=O)[nH]nc1-c1ccc(NC2CCc3cc(C4CC4)ccc32)cc1,TZRQZGKWPNPSCK-UHFFFAOYSA-N
rxn:2:153842:119230,CCC(Nc1ccc(-c2n[nH]c(=O)cc2C)cc1)c1cccc(C)c1,XQHDUFRNAVKSSR-UHFFFAOYSA-N
rxn:2:138879:169289,CCCOc1ncccc1-c1cc(C)c(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)c(C)c1,QDJKPGWTLMSBIN-UHFFFAOYSA-N
rxn:2:138879:130817,Cc1nc2c(NC(C)c3cccc4c3CCCC4)n[nH]c(=O)n2n1,DMZQNAZPPIAAJR-UHFFFAOYSA-N
rxn:2:138879:106310,COc1cccc(C(Nc2n[nH]c(=O)n3nc(C)nc23)C2(C)CC2)c1,FQLABQIGCFXNRI-UHFFFAOYSA-N
rxn:2:138879:166768,Cc1nc2c(NC(C)c3ccc(-c4cccc5cn[nH]c45)cc3C)n[nH]c(=O)n2n1,PVHXNBAUBHJXMZ-UHFFFAOYSA-N
rxn:2:138879:150745,COc1ccc(C(C)Nc2n[nH]c(=O)n3nc(C)nc23)cc1-c1cnc2cc(C)nn2c1,VSSYDCYUWDDBLK-UHFFFAOYSA-N
rxn:2:138879:67392,Cc1nc2c(NC(c3ccc(C)c(C)c3)C(C)C)n[nH]c(=O)n2n1,MHCQIPBYGJOZSQ-UHFFFAOYSA-N
rxn:2:73951:170908,COc1cc(C)c(-c2cccc(C(C)Nc3cn[nH]c(=O)c3)c2C)cc1C,ZHBSJQAGWJFGPD-UHFFFAOYSA-N
rxn:2:153842:73498,COc1ccc2c(c1)C(Nc1ccc(-c3n[nH]c(=O)cc3C)cc1)C(C)(C)C2,NOWQYRZTRODSAC-UHFFFAOYSA-N
rxn:2:138879:74458,COc1cccc(C(Nc2n[nH]c(=O)n3nc(C)nc23)C(C)C)c1,UBQOQLMZIOHSEZ-UHFFFAOYSA-N
rxn:2:138879:169703,Cc1nc2c(NC(C)c3c(C)cc(-c4ccc(CN(C)C)cc4)cc3C)n[nH]c(=O)n2n1,RRMDRGCJQKMAGX-UHFFFAOYSA-N
rxn:2:125845:169656,Cc1ccc(C(C)Nc2n[nH]c(=O)c(=O)[nH]2)cc1-c1cccc(C2CC2)c1,GTOZKWRQELLHDP-UHFFFAOYSA-N


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