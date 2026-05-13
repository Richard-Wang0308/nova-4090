#!/usr/bin/env python3
"""
NANOBODY MINER
==============
Pipeline:
  1. Load & rank HuggingFace archive per target
  2. [LOOP] Generate candidates (Option A + B)
         → Full validator validation
         → BoltzGen scoring
         → Write to nanobodies.sqlite
         → repeat
"""

import os
import sys
import math
import time
import random
import hashlib
import asyncio
import sqlite3
import argparse
import logging
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, NOVA_DIR)
sys.path.insert(0, os.path.join(NOVA_DIR, "NOVA-nanobody-filter"))

import bittensor as bt

from config.config_loader import load_config
from utils.nanobodies import (
    normalize_seq, seq_hash, max_run_length, max_di_repeat_pairs,
    has_plausible_cys_pair, looks_like_signal_peptide,
    analyze_developability, compute_igblast_nativeness,
    index_top_sequences, is_duplicate,
)
from utils.challenge import entry_unique_for_protein_hf
from utils.constants import ALLOWED_AAS
from boltzgen.boltzgen_wrapper import BoltzgenWrapper

# ══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════

LOG_DIR = os.path.join(NOVA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _build_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Console: INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_log_formatter)
    logger.addHandler(ch)

    # File: DEBUG and above (full detail)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_log_formatter)
    logger.addHandler(fh)

    return logger

_ts = time.strftime("%Y%m%d_%H%M%S")
log       = _build_logger("miner",    os.path.join(LOG_DIR, f"miner_{_ts}.log"))
log_gen   = _build_logger("generate", os.path.join(LOG_DIR, f"miner_{_ts}.log"))
log_val   = _build_logger("validate", os.path.join(LOG_DIR, f"miner_{_ts}.log"))
log_score = _build_logger("score",    os.path.join(LOG_DIR, f"miner_{_ts}.log"))
log_db    = _build_logger("db",       os.path.join(LOG_DIR, f"miner_{_ts}.log"))

def _sep(char: str = "─", width: int = 68) -> str:
    return char * width

def _banner(title: str, char: str = "═", width: int = 68) -> list[str]:
    pad = max(0, width - len(title) - 4)
    l   = pad // 2
    r   = pad - l
    return [
        char * width,
        f"{char*2}  {' '*l}{title}{' '*r}  {char*2}",
        char * width,
    ]

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

HF_BASE = "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main"
DB_PATH = os.path.join(NOVA_DIR, "nanobodies.sqlite")

AA = list("ACDEFGHIKLMNPQRSTVWY")

CDR3_AA_PRIOR = {
    "A": 0.07, "C": 0.01, "D": 0.08, "E": 0.04, "F": 0.04,
    "G": 0.10, "H": 0.03, "I": 0.04, "K": 0.04, "L": 0.06,
    "M": 0.02, "N": 0.05, "P": 0.02, "Q": 0.04, "R": 0.07,
    "S": 0.09, "T": 0.07, "V": 0.05, "W": 0.02, "Y": 0.10,
}
CDR3_AA_PRIOR_LIST    = list(CDR3_AA_PRIOR.keys())
CDR3_AA_PRIOR_WEIGHTS = [CDR3_AA_PRIOR[a] for a in CDR3_AA_PRIOR_LIST]

IMGT_CDR_APPROX = {"cdr1": (26, 38), "cdr2": (55, 65), "cdr3": (104, 118)}

METRIC_DIRECTIONS = {
    "design_iiptm":             "max",
    "design_ptm":               "max",
    "design_to_target_iptm":    "max",
    "min_design_to_target_pae": "min",
    "interaction_pae":          "min",
    "plip_hbonds_refolded":     "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded":      "max",
    "liability_score":          "min",
    "liability_num_violations": "min",
}

METRIC_CATEGORIES = {
    "confidence":           ["design_iiptm", "design_ptm", "design_to_target_iptm",
                             "min_design_to_target_pae", "interaction_pae"],
    "physical_interaction": ["plip_hbonds_refolded", "plip_saltbridge_refolded",
                             "delta_sasa_refolded"],
    "developability":       ["liability_score", "liability_num_violations"],
}

ALL_METRICS = list(METRIC_DIRECTIONS.keys())

# ══════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════

def init_db(db_path: str = DB_PATH) -> None:
    log_db.info(_sep())
    log_db.info(f"  Initializing DB: {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS nanobodies (
            id                            INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence                      TEXT    NOT NULL,
            target                        TEXT    NOT NULL,
            available                     BOOLEAN,
            design_iiptm                  REAL,
            design_ptm                    REAL,
            design_to_target_iptm         REAL,
            min_design_to_target_pae      REAL,
            interaction_pae               REAL,
            plip_hbonds_refolded          REAL,
            plip_saltbridge_refolded      REAL,
            delta_sasa_refolded           REAL,
            liability_score               REAL,
            liability_num_violations      REAL,
            confidence_rank_sum           REAL,
            physical_interaction_rank_sum REAL,
            developability_rank_sum       REAL,
            rank_sum                      REAL,
            final_nanobody_score          REAL,
            scored_by                     TEXT DEFAULT 'boltzgen',
            generation_method             TEXT DEFAULT 'option_a',
            calc_time_sec                 REAL,
            created_at                    TEXT DEFAULT (datetime('now')),
            UNIQUE(sequence, target)
        )
    """)
    # Serving-side validation may set ``available``; submit scripts only read rows
    # with available = TRUE. Add column for existing DBs created before this field.
    try:
        c.execute("ALTER TABLE nanobodies ADD COLUMN available BOOLEAN")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    log_db.info(f"  DB ready (schema with optional ``available`` for submit gating)")
    log_db.info(_sep())


def upsert_results(rows: list[dict], db_path: str = DB_PATH) -> None:
    if not rows:
        log_db.warning("  upsert_results called with empty rows list")
        return
    log_db.info(f"  Upserting {len(rows)} rows → {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    inserted = updated = 0
    for row in rows:
        c.execute("SELECT id FROM nanobodies WHERE sequence=? AND target=?",
                  (row["sequence"], row["target"]))
        exists = c.fetchone() is not None
        c.execute("""
            INSERT INTO nanobodies (
                sequence, target,
                design_iiptm, design_ptm, design_to_target_iptm,
                min_design_to_target_pae, interaction_pae,
                plip_hbonds_refolded, plip_saltbridge_refolded,
                delta_sasa_refolded, liability_score, liability_num_violations,
                confidence_rank_sum, physical_interaction_rank_sum,
                developability_rank_sum, rank_sum, final_nanobody_score,
                scored_by, generation_method, calc_time_sec
            ) VALUES (
                :sequence, :target,
                :design_iiptm, :design_ptm, :design_to_target_iptm,
                :min_design_to_target_pae, :interaction_pae,
                :plip_hbonds_refolded, :plip_saltbridge_refolded,
                :delta_sasa_refolded, :liability_score, :liability_num_violations,
                :confidence_rank_sum, :physical_interaction_rank_sum,
                :developability_rank_sum, :rank_sum, :final_nanobody_score,
                :scored_by, :generation_method, :calc_time_sec
            )
            ON CONFLICT(sequence, target) DO UPDATE SET
                design_iiptm                  = excluded.design_iiptm,
                design_ptm                    = excluded.design_ptm,
                design_to_target_iptm         = excluded.design_to_target_iptm,
                min_design_to_target_pae      = excluded.min_design_to_target_pae,
                interaction_pae               = excluded.interaction_pae,
                plip_hbonds_refolded          = excluded.plip_hbonds_refolded,
                plip_saltbridge_refolded      = excluded.plip_saltbridge_refolded,
                delta_sasa_refolded           = excluded.delta_sasa_refolded,
                liability_score               = excluded.liability_score,
                liability_num_violations      = excluded.liability_num_violations,
                confidence_rank_sum           = excluded.confidence_rank_sum,
                physical_interaction_rank_sum = excluded.physical_interaction_rank_sum,
                developability_rank_sum       = excluded.developability_rank_sum,
                rank_sum                      = excluded.rank_sum,
                final_nanobody_score          = excluded.final_nanobody_score,
                scored_by                     = excluded.scored_by,
                generation_method             = excluded.generation_method,
                calc_time_sec                 = excluded.calc_time_sec,
                created_at                    = datetime('now')
        """, row)
        if exists:
            updated += 1
        else:
            inserted += 1
    conn.commit()
    conn.close()
    log_db.info(f"  DB write complete → inserted={inserted}, updated={updated}, "
                f"total_rows_now={count_db_rows(db_path)}")


def get_already_scored(db_path: str = DB_PATH) -> set[tuple[str, str]]:
    if not os.path.exists(db_path):
        log_db.info("  DB file not found — starting fresh")
        return set()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT sequence, target FROM nanobodies")
    rows = {(r[0], r[1]) for r in c.fetchall()}
    conn.close()
    log_db.info(f"  Loaded {len(rows)} already-scored (sequence, target) pairs from DB")
    return rows


def count_db_rows(db_path: str = DB_PATH) -> int:
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM nanobodies")
    n = c.fetchone()[0]
    conn.close()
    return n


def log_db_top(db_path: str, target: str, n: int = 5) -> None:
    """Log the current top-N sequences in the DB for a target."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT sequence, final_nanobody_score, rank_sum, generation_method, "
        "       design_iiptm, liability_score, liability_num_violations "
        "FROM nanobodies WHERE target=? "
        "ORDER BY final_nanobody_score ASC LIMIT ?",
        conn, params=(target, n))
    conn.close()
    if df.empty:
        return
    log_db.info(f"  ┌─ Top-{n} DB sequences for {target} ─────────────────────────────")
    for rank, row in df.iterrows():
        log_db.info(
            f"  │ #{rank+1:>2}  score={row.final_nanobody_score:>9.3f}  "
            f"iiptm={row.design_iiptm:.3f}  "
            f"liab={row.liability_score:.3f}  "
            f"viol={int(row.liability_num_violations or 0)}  "
            f"[{row.generation_method}]  "
            f"{row.sequence[:30]}..."
        )
    log_db.info(f"  └{'─'*65}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — ARCHIVE LOADING & RANKING
# ══════════════════════════════════════════════════════════════════════════

def load_archive(target: str, cache_dir: str = "archive_cache") -> pd.DataFrame:
    Path(cache_dir).mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"{target}_nanobodies.csv"
    if cache_path.exists():
        log.info(f"[Archive] Loading cached: {cache_path}")
        df = pd.read_csv(cache_path)
        log.debug(f"[Archive] Raw rows loaded: {len(df)}")
    else:
        url = f"{HF_BASE}/{target}_nanobodies.csv"
        log.info(f"[Archive] Downloading from HuggingFace: {url}")
        t0   = time.time()
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
        df   = pd.read_csv(cache_path)
        log.info(f"[Archive] Downloaded {len(df)} rows in {time.time()-t0:.1f}s → {cache_path}")

    before = len(df)
    df = df.dropna(subset=ALL_METRICS)
    dropped = before - len(df)
    log.info(f"[Archive] {len(df)} fully-scored sequences for {target} "
             f"(dropped {dropped} with missing metrics)")

    # Log metric summary
    log.debug(f"[Archive] Metric summary for {target}:")
    for m in ALL_METRICS:
        if m in df.columns:
            log.debug(f"[Archive]   {m:<35} "
                      f"min={df[m].min():.3f}  "
                      f"med={df[m].median():.3f}  "
                      f"max={df[m].max():.3f}")
    return df


def rank_archive(df: pd.DataFrame) -> pd.DataFrame:
    log.debug(f"[Archive] Computing dense ranks for {len(df)} sequences...")
    df = df.copy()
    for metric, mode in METRIC_DIRECTIONS.items():
        if metric not in df.columns:
            log.debug(f"[Archive]   Skipping missing metric: {metric}")
            continue
        df[f"rank_{metric}"] = df[metric].rank(
            method="dense", ascending=(mode == "min"))
    rank_cols = [f"rank_{m}" for m in ALL_METRICS if f"rank_{m}" in df.columns]
    df["rank_sum"] = df[rank_cols].sum(axis=1)
    for cat, metrics in METRIC_CATEGORIES.items():
        cat_cols = [f"rank_{m}" for m in metrics if f"rank_{m}" in df.columns]
        df[f"{cat}_rank_sum"] = df[cat_cols].sum(axis=1)
    df = df.sort_values(
        ["rank_sum", "confidence_rank_sum",
         "physical_interaction_rank_sum", "developability_rank_sum"],
        ascending=True,
    ).reset_index(drop=True)
    df["archive_rank"] = df.index + 1

    log.info(f"[Archive] Ranking complete. "
             f"Best rank_sum={df['rank_sum'].min():.0f}, "
             f"Worst={df['rank_sum'].max():.0f}")
    log.debug(f"[Archive] Top-5 archive sequences:")
    for _, row in df.head(5).iterrows():
        log.debug(f"[Archive]   rank={row.archive_rank}  "
                  f"rank_sum={row.rank_sum:.0f}  "
                  f"seq={row.sequence[:40]}...")
    return df


# ══════════════════════════════════════════════════════════════════════════
# CDR UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def get_cdr_positions(sequence: str) -> dict:
    try:
        from abnumber import Chain
        chain = Chain(sequence, scheme="imgt")
        positions = {"cdr1": [], "cdr2": [], "cdr3": []}
        for i, (pos, _) in enumerate(chain):
            region = str(pos.get_region())
            if "CDR1" in region:   positions["cdr1"].append(i)
            elif "CDR2" in region: positions["cdr2"].append(i)
            elif "CDR3" in region: positions["cdr3"].append(i)
        return positions
    except Exception:
        return {k: list(range(*v)) for k, v in IMGT_CDR_APPROX.items()}


def extract_cdr3(sequence: str,
                  positions: dict | None = None) -> tuple[str, int, int]:
    if positions is None:
        positions = get_cdr_positions(sequence)
    cdr3_pos = positions["cdr3"]
    if not cdr3_pos:
        s, e     = IMGT_CDR_APPROX["cdr3"]
        cdr3_pos = list(range(s, min(e, len(sequence))))
    start = cdr3_pos[0]
    end   = cdr3_pos[-1] + 1
    return sequence[start:end], start, end


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2a — OPTION A: CDR-biased point mutation
# ══════════════════════════════════════════════════════════════════════════

def mutate_cdr(sequence: str, n_mutations: int = 3,
                cdr3_bias: float = 0.7) -> str:
    positions = get_cdr_positions(sequence)
    seq       = list(sequence)
    n_cdr3    = max(1, int(n_mutations * cdr3_bias))
    n_other   = n_mutations - n_cdr3

    cdr3_pos = positions["cdr3"]
    if cdr3_pos:
        for pos in random.sample(cdr3_pos, min(n_cdr3, len(cdr3_pos))):
            seq[pos] = random.choice([a for a in AA if a != seq[pos]])

    other_pos = positions["cdr1"] + positions["cdr2"]
    if other_pos and n_other > 0:
        for pos in random.sample(other_pos, min(n_other, len(other_pos))):
            seq[pos] = random.choice([a for a in AA if a != seq[pos]])

    return "".join(seq)


def generate_option_a(seeds: list[str], n_per_seed: int = 100,
                       mutations_range: tuple = (2, 5)) -> list[str]:
    log_gen.info(_sep())
    log_gen.info(f"  Option A — CDR-biased point mutation")
    log_gen.info(f"  Seeds: {len(seeds)}  |  n_per_seed: {n_per_seed}  "
                 f"|  mutations: {mutations_range[0]}–{mutations_range[1]}")
    log_gen.info(f"  CDR3 bias: 70%  |  CDR1+CDR2: 30%")
    t0 = time.time()

    candidates   = []
    mut_counters = Counter()
    for seed_i, seed in enumerate(seeds):
        seed_cands = []
        for _ in range(n_per_seed):
            n_mut = random.randint(*mutations_range)
            mut_counters[n_mut] += 1
            seed_cands.append(mutate_cdr(seed, n_mut))
        candidates.extend(seed_cands)
        log_gen.debug(f"  [A] Seed {seed_i+1:>3}/{len(seeds)}  "
                      f"{seed[:30]}...  → {len(seed_cands)} mutants")

    elapsed = time.time() - t0
    log_gen.info(f"  Generated {len(candidates)} mutants in {elapsed:.2f}s "
                 f"({elapsed/max(len(candidates),1)*1000:.2f}ms/seq)")
    log_gen.info(f"  Mutation distribution: "
                 + "  ".join(f"{k}mut={v}" for k, v in sorted(mut_counters.items())))
    log_gen.info(_sep())
    return candidates


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2b — OPTION B: CDR3 structural diversity
# ══════════════════════════════════════════════════════════════════════════

def _b1_cdr3_block_swap(seq_a: str, seq_b: str) -> str:
    pos_a = get_cdr_positions(seq_a)
    pos_b = get_cdr_positions(seq_b)
    cdr3_a, start_a, end_a = extract_cdr3(seq_a, pos_a)
    cdr3_b, _,       _     = extract_cdr3(seq_b, pos_b)

    if not cdr3_a or not cdr3_b:
        return mutate_cdr(seq_a, random.randint(2, 5))

    block_len = random.randint(
        max(1, min(3, len(cdr3_b))),
        min(len(cdr3_a), len(cdr3_b))
    )
    b_start  = random.randint(0, max(0, len(cdr3_b) - block_len))
    a_start  = random.randint(0, max(0, len(cdr3_a) - block_len))
    new_cdr3 = (cdr3_a[:a_start]
                + cdr3_b[b_start: b_start + block_len]
                + cdr3_a[a_start + block_len:])

    return seq_a[:start_a] + new_cdr3 + seq_a[end_a:]


def _b2_cdr3_length_variation(sequence: str,
                                min_len: int = 6, max_len: int = 20) -> str:
    pos                    = get_cdr_positions(sequence)
    cdr3_seq, start, end   = extract_cdr3(sequence, pos)

    if not cdr3_seq:
        return mutate_cdr(sequence, random.randint(2, 5))

    action = random.choice(["insert", "delete"])

    if action == "insert" and len(cdr3_seq) < max_len:
        n_insert   = random.randint(1, 2)
        insert_pos = random.randint(0, len(cdr3_seq))
        new_aas    = random.choices(
            CDR3_AA_PRIOR_LIST, weights=CDR3_AA_PRIOR_WEIGHTS, k=n_insert)
        new_cdr3   = (cdr3_seq[:insert_pos]
                      + "".join(new_aas)
                      + cdr3_seq[insert_pos:])
    elif action == "delete" and len(cdr3_seq) > min_len:
        n_delete   = random.randint(1, min(2, len(cdr3_seq) - min_len))
        del_start  = random.randint(0, len(cdr3_seq) - n_delete)
        new_cdr3   = cdr3_seq[:del_start] + cdr3_seq[del_start + n_delete:]
    else:
        new_cdr3      = list(cdr3_seq)
        pos_to_mutate = random.randint(0, len(new_cdr3) - 1)
        new_cdr3[pos_to_mutate] = random.choices(
            CDR3_AA_PRIOR_LIST, weights=CDR3_AA_PRIOR_WEIGHTS)[0]
        new_cdr3 = "".join(new_cdr3)

    return sequence[:start] + new_cdr3 + sequence[end:]


def _b3_cdr3_full_resample(sequence: str,
                             target_len: int | None = None) -> str:
    pos                    = get_cdr_positions(sequence)
    cdr3_seq, start, end   = extract_cdr3(sequence, pos)

    if not cdr3_seq:
        return mutate_cdr(sequence, random.randint(2, 5))

    length   = target_len or len(cdr3_seq)
    new_cdr3 = "".join(random.choices(
        CDR3_AA_PRIOR_LIST, weights=CDR3_AA_PRIOR_WEIGHTS, k=length))

    return sequence[:start] + new_cdr3 + sequence[end:]


def generate_option_b(seeds: list[str], n_per_seed: int = 100) -> list[str]:
    log_gen.info(_sep())
    log_gen.info(f"  Option B — CDR3 structural diversity (no external tools)")
    log_gen.info(f"  Seeds: {len(seeds)}  |  n_per_seed: {n_per_seed}")
    log_gen.info(f"  Strategy split: 40% block-swap | 30% indel | 30% full-resample")

    if not seeds:
        log_gen.warning("  No seeds provided — Option B skipped")
        log_gen.info(_sep())
        return []

    t0         = time.time()
    candidates = []
    n_swap     = int(n_per_seed * 0.40)
    n_indel    = int(n_per_seed * 0.30)
    n_resamp   = n_per_seed - n_swap - n_indel

    strategy_counts = {"swap": 0, "indel": 0, "resample": 0}

    for seed_i, seed in enumerate(seeds):
        pos          = get_cdr_positions(seed)
        cdr3_seq, _, _ = extract_cdr3(seed, pos)
        cdr3_len     = len(cdr3_seq) if cdr3_seq else "?"
        other_seeds  = [s for s in seeds if s != seed] or seeds

        seed_cands = []

        # B1: block-swap
        for _ in range(n_swap):
            partner = random.choice(other_seeds)
            seed_cands.append(_b1_cdr3_block_swap(seed, partner))
            strategy_counts["swap"] += 1

        # B2: indel
        for _ in range(n_indel):
            seed_cands.append(_b2_cdr3_length_variation(seed))
            strategy_counts["indel"] += 1

        # B3: full resample
        for _ in range(n_resamp):
            base_len   = len(cdr3_seq) if cdr3_seq else 12
            target_len = max(6, min(20, base_len + random.randint(-2, 2)))
            seed_cands.append(_b3_cdr3_full_resample(seed, target_len))
            strategy_counts["resample"] += 1

        candidates.extend(seed_cands)
        log_gen.debug(f"  [B] Seed {seed_i+1:>3}/{len(seeds)}  "
                      f"CDR3_len={cdr3_len:>3}  "
                      f"{seed[:30]}...  → {len(seed_cands)} candidates")

    elapsed = time.time() - t0
    log_gen.info(f"  Generated {len(candidates)} candidates in {elapsed:.2f}s")
    log_gen.info(f"  Strategy totals: "
                 f"swap={strategy_counts['swap']}  "
                 f"indel={strategy_counts['indel']}  "
                 f"resample={strategy_counts['resample']}")
    log_gen.info(_sep())
    return candidates


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════

async def validate_candidates(candidates: list[str], config: dict,
                                search_engines: dict) -> list[str]:
    log_val.info(_sep())
    log_val.info(f"  Validation pipeline — {len(candidates)} input candidates")
    log_val.info(_sep())

    candidates = list(dict.fromkeys(candidates))
    log_val.info(f"  After dedup:              {len(candidates)}")

    # Per-filter rejection counters
    reject = defaultdict(int)
    pre_validated: list[dict] = []

    for seq in candidates:
        norm = normalize_seq(seq)
        h    = seq_hash(norm)

        if "~" in norm:
            reject["tilde_in_seq"] += 1
            log_val.debug(f"  REJECT tilde  {h[:8]}  {norm[:40]}")
            continue
        if not (config["min_sequence_length"] <= len(norm)
                <= config["max_sequence_length"]):
            reject["length"] += 1
            log_val.debug(f"  REJECT length={len(norm)}  "
                          f"(allowed {config['min_sequence_length']}–"
                          f"{config['max_sequence_length']})  {h[:8]}")
            continue
        bad_aas = set(norm) - ALLOWED_AAS
        if bad_aas:
            reject["invalid_aa"] += 1
            log_val.debug(f"  REJECT invalid_aa={bad_aas}  {h[:8]}")
            continue
        rl = max_run_length(norm)
        if rl > config["max_homopolymer_run"]:
            reject["homopolymer"] += 1
            log_val.debug(f"  REJECT homopolymer run={rl}  {h[:8]}")
            continue
        dr = max_di_repeat_pairs(norm)
        if dr > config["max_di_repeat_pairs"]:
            reject["di_repeat"] += 1
            log_val.debug(f"  REJECT di_repeat={dr}  {h[:8]}")
            continue
        nc = norm.count("C")
        if nc < config["min_cysteines"]:
            reject["cysteine_count"] += 1
            log_val.debug(f"  REJECT cys_count={nc}  {h[:8]}")
            continue
        if config["min_cysteines"] > 1:
            if not has_plausible_cys_pair(norm,
                                          config["cys_pair_min_separation"],
                                          config["cys_pair_max_separation"]):
                reject["cysteine_pair"] += 1
                log_val.debug(f"  REJECT cys_pair  {h[:8]}")
                continue
        if looks_like_signal_peptide(norm, config["sp_window"],
                                     config["sp_hydro_min_in_window"],
                                     config["sp_scan_prefix"]):
            reject["signal_peptide"] += 1
            log_val.debug(f"  REJECT signal_peptide  {h[:8]}")
            continue

        # Exact duplicate vs HF archive
        is_unique = True
        for target in config["nanobody_target"]:
            if not entry_unique_for_protein_hf(target, h, "nanobodies"):
                is_unique = False
                break
        if not is_unique:
            reject["hf_duplicate"] += 1
            log_val.debug(f"  REJECT hf_duplicate  {h[:8]}")
            continue

        # Similarity search
        uid_invalid        = False
        similarity_results = []
        for target in config["nanobody_target"]:
            engine = search_engines.get(target)
            if engine is None:
                continue
            try:
                result = engine.search(norm, include_alignment=True,
                                       exclude_ids=None,
                                       coarse_min_shared=None,
                                       coarse_jaccard=None)
                similarity_results.append(result)
            except Exception as e:
                log_val.warning(f"  Similarity search error {h[:8]}: {e}")
                uid_invalid = True
                break
            if any(is_duplicate(m)[0] for m in result.matches):
                uid_invalid = True
                break
        if uid_invalid:
            reject["similarity_duplicate"] += 1
            log_val.debug(f"  REJECT similarity_dup  {h[:8]}")
            continue

        pre_validated.append({"seq": norm, "hash": h,
                               "similarity_results": similarity_results})

    log_val.info(f"  Pre-validation passed:    {len(pre_validated)} / {len(candidates)}")
    log_val.info(f"  Rejection breakdown:")
    for reason, count in sorted(reject.items(), key=lambda x: -x[1]):
        log_val.info(f"    {reason:<28} {count:>5}")

    if not pre_validated:
        log_val.warning("  No sequences passed pre-validation — aborting")
        log_val.info(_sep())
        return []

    # ── IgBLAST nativeness ─────────────────────────────────────────────────
    log_val.info(f"  Running IgBLAST nativeness on {len(pre_validated)} sequences...")
    t0 = time.time()
    all_seqs_igblast = {e["hash"]: e["seq"] for e in pre_validated}
    results_by_id: dict = {}
    try:
        all_nat        = compute_igblast_nativeness(all_seqs_igblast)
        results_by_id  = {r.sequence_id: r for r in all_nat}
        log_val.info(f"  IgBLAST batch done in {time.time()-t0:.1f}s "
                     f"({len(results_by_id)} results)")
    except Exception as e:
        log_val.warning(f"  Batch IgBLAST failed ({e}), falling back to per-sequence...")
        for entry in pre_validated:
            try:
                res = compute_igblast_nativeness({entry["hash"]: entry["seq"]})
                for r in res:
                    results_by_id[r.sequence_id] = r
            except Exception as e2:
                log_val.warning(f"  IgBLAST failed {entry['hash'][:8]}: {e2}")

    post_nat: list[dict] = []
    nat_reject_vhh = nat_reject_hfw = nat_missing = 0
    for entry in pre_validated:
        nat = results_by_id.get(entry["hash"])
        if nat is None:
            nat_missing += 1
            log_val.debug(f"  REJECT igblast_missing  {entry['hash'][:8]}")
            continue
        if nat.vhh_nativeness < config["min_nativeness_score"]:
            nat_reject_vhh += 1
            log_val.debug(f"  REJECT vhh_nativeness={nat.vhh_nativeness:.3f} "
                          f"< {config['min_nativeness_score']}  {entry['hash'][:8]}")
            continue
        if nat.human_framework < config["min_human_framework_score"]:
            nat_reject_hfw += 1
            log_val.debug(f"  REJECT human_framework={nat.human_framework:.3f} "
                          f"< {config['min_human_framework_score']}  {entry['hash'][:8]}")
            continue
        log_val.debug(f"  PASS igblast  vhh={nat.vhh_nativeness:.3f}  "
                      f"hfw={nat.human_framework:.3f}  {entry['hash'][:8]}")
        post_nat.append(entry)

    log_val.info(f"  IgBLAST passed:           {len(post_nat)} / {len(pre_validated)}")
    log_val.info(f"    Rejected vhh_nativeness: {nat_reject_vhh}")
    log_val.info(f"    Rejected human_framework:{nat_reject_hfw}")
    log_val.info(f"    Missing IgBLAST result:  {nat_missing}")

    if not post_nat:
        log_val.warning("  No sequences passed IgBLAST — aborting")
        log_val.info(_sep())
        return []

    # ── Developability ─────────────────────────────────────────────────────
    log_val.info(f"  Running developability on {len(post_nat)} sequences...")
    t0 = time.time()
    try:
        dev_results = await analyze_developability([e["seq"] for e in post_nat])
    except Exception as e:
        log_val.warning(f"  Developability check failed: {e}")
        log_val.info(_sep())
        return []

    valid      = []
    dev_passed = dev_failed = 0
    for entry, dev in zip(post_nat, dev_results):
        if dev.get("passed", False):
            valid.append(entry["seq"])
            dev_passed += 1
            log_val.debug(f"  PASS dev  {entry['hash'][:8]}")
        else:
            dev_failed += 1
            reasons = dev.get("fail_reasons", [])
            log_val.debug(f"  REJECT dev  {entry['hash'][:8]}  reasons={reasons}")

    log_val.info(f"  Developability done in {time.time()-t0:.1f}s")
    log_val.info(f"  Developability passed:    {dev_passed} / {len(post_nat)}")
    log_val.info(f"  Developability failed:    {dev_failed}")
    log_val.info(_sep("─"))
    log_val.info(f"  VALIDATION SUMMARY")
    log_val.info(f"    Input:                  {len(candidates)}")
    log_val.info(f"    Pre-validation pass:    {len(pre_validated)}")
    log_val.info(f"    IgBLAST pass:           {len(post_nat)}")
    log_val.info(f"    Developability pass:    {dev_passed}  ← sent to BoltzGen")
    log_val.info(f"    Overall pass rate:      "
                 f"{dev_passed/max(len(candidates),1)*100:.1f}%")
    log_val.info(_sep())
    return valid


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — BOLTZGEN SCORING
# ══════════════════════════════════════════════════════════════════════════

def score_with_boltzgen(sequences: list[str],
                         config: dict) -> tuple[dict, dict]:
    if not sequences:
        log_score.warning("  score_with_boltzgen called with empty list")
        return {}, {}

    log_score.info(_sep())
    log_score.info(f"  BoltzGen scoring — {len(sequences)} sequences")
    log_score.info(f"  Estimated time: ~{len(sequences) * 30:.0f}s "
                   f"(~30s/seq on A100)")
    log_score.info(_sep())

    valid_nanobodies_by_uid: dict[int, dict] = {
        i: {"sequences": [seq], "hashes": [seq_hash(seq)]}
        for i, seq in enumerate(sequences)
    }

    t0      = time.time()
    wrapper = BoltzgenWrapper()

    log_score.info(f"  [1/2] Running nanobody inference...")
    t1 = time.time()
    per_nanobody_components = wrapper.run_nanobody_inference(
        valid_nanobodies_by_uid, config)
    log_score.info(f"  [1/2] Inference done in {time.time()-t1:.1f}s")

    log_score.info(f"  [2/2] Finalizing ranking from components...")
    t2 = time.time()
    final_boltzgen_scores, per_nanobody_components = \
        wrapper.finalize_ranking_from_components(valid_nanobodies_by_uid, config)
    log_score.info(f"  [2/2] Ranking done in {time.time()-t2:.1f}s")

    elapsed = time.time() - t0
    log_score.info(f"  Total BoltzGen time: {elapsed:.1f}s "
                   f"({elapsed/max(len(sequences),1):.1f}s/seq)")
    log_score.info(_sep())
    return per_nanobody_components, final_boltzgen_scores


def collect_boltzgen_results(sequences: list[str],
                              per_nanobody_components: dict,
                              final_boltzgen_scores: dict,
                              config: dict,
                              calc_time_sec: float,
                              generation_methods: dict[str, str] | None = None
                              ) -> list[dict]:
    rows      = []
    no_comp   = []
    seq_to_uid = {seq: i for i, seq in enumerate(sequences)}

    for seq in sequences:
        uid       = seq_to_uid[seq]
        seq_comps = per_nanobody_components.get(uid, {}).get(seq, {})

        for target in config["nanobody_target"]:
            comp = seq_comps.get(target, {})
            if not comp:
                no_comp.append(seq[:30])
                log_score.warning(f"  No components for seq={seq[:30]}... target={target}")
                continue

            final_score = (final_boltzgen_scores
                           .get(uid, {}).get(seq, {}).get(target, math.inf))
            if hasattr(final_score, "item"):
                final_score = final_score.item()

            row = {
                "sequence":                       seq,
                "target":                         target,
                **{m: comp.get(m) for m in ALL_METRICS},
                "confidence_rank_sum":            comp.get("confidence_rank_sum"),
                "physical_interaction_rank_sum":  comp.get("physical_interaction_rank_sum"),
                "developability_rank_sum":        comp.get("developability_rank_sum"),
                "rank_sum":                       comp.get("rank_sum"),
                "final_nanobody_score":           final_score,
                "scored_by":                      "boltzgen",
                "generation_method":              (generation_methods or {}).get(
                                                      normalize_seq(seq), "option_a"),
                "calc_time_sec":                  calc_time_sec / max(len(sequences), 1),
            }
            rows.append(row)

            log_score.debug(
                f"  scored  {seq[:30]}...  "
                f"score={final_score:.4f}  "
                f"iiptm={comp.get('design_iiptm', float('nan')):.3f}  "
                f"ptm={comp.get('design_ptm', float('nan')):.3f}  "
                f"iptm={comp.get('design_to_target_iptm', float('nan')):.3f}  "
                f"pae_min={comp.get('min_design_to_target_pae', float('nan')):.2f}  "
                f"pae_int={comp.get('interaction_pae', float('nan')):.2f}  "
                f"hbonds={comp.get('plip_hbonds_refolded', float('nan')):.0f}  "
                f"salt={comp.get('plip_saltbridge_refolded', float('nan')):.0f}  "
                f"dsasa={comp.get('delta_sasa_refolded', float('nan')):.1f}  "
                f"liab={comp.get('liability_score', float('nan')):.3f}  "
                f"viol={comp.get('liability_num_violations', float('nan')):.0f}  "
                f"[{(generation_methods or {}).get(normalize_seq(seq), 'option_a')}]"
            )

    # ── Scoring summary ────────────────────────────────────────────────────
    if rows:
        scores = [r["final_nanobody_score"] for r in rows
                  if r["final_nanobody_score"] != math.inf]
        log_score.info(_sep("─"))
        log_score.info(f"  SCORING SUMMARY ({len(rows)} rows collected)")
        log_score.info(f"    Missing components:   {len(no_comp)}")
        if scores:
            log_score.info(f"    final_nanobody_score:")
            log_score.info(f"      best  = {min(scores):.4f}")
            log_score.info(f"      worst = {max(scores):.4f}")
            log_score.info(f"      mean  = {sum(scores)/len(scores):.4f}")
        # Per-metric summary
        log_score.info(f"    Per-metric means:")
        for m in ALL_METRICS:
            vals = [r[m] for r in rows if r.get(m) is not None]
            if vals:
                log_score.info(f"      {m:<35} mean={sum(vals)/len(vals):.4f}  "
                               f"best={'min' if METRIC_DIRECTIONS[m]=='min' else 'max'}"
                               f"={min(vals) if METRIC_DIRECTIONS[m]=='min' else max(vals):.4f}")
        # Generation method breakdown
        method_counts = Counter(r["generation_method"] for r in rows)
        log_score.info(f"    Generation method: "
                       + "  ".join(f"{k}={v}" for k, v in method_counts.items()))
        log_score.info(_sep())

    return rows


# ══════════════════════════════════════════════════════════════════════════
# SINGLE ITERATION
# ══════════════════════════════════════════════════════════════════════════

async def run_one_iteration(iteration: int,
                             target: str,
                             config: dict,
                             search_engines: dict,
                             already_scored: set[tuple[str, str]],
                             args: argparse.Namespace) -> tuple[int, set]:

    for line in _banner(f"ITERATION {iteration}  |  TARGET {target}"):
        log.info(line)

    db_rows = count_db_rows(args.db_path)
    log.info(f"  DB rows at start of iteration: {db_rows}")
    log.info(f"  Already-scored pairs in memory: {len(already_scored)}")

    # ── Seeds ──────────────────────────────────────────────────────────────
    df_ranked = args._archive_ranked[target]

    if iteration == 1 or not os.path.exists(args.db_path):
        seeds_source = "archive_only"
        top_seeds_a  = df_ranked.head(args.seeds_a)["sequence"].tolist()
        top_seeds_b  = df_ranked.head(args.seeds_b)["sequence"].tolist()
    else:
        seeds_source  = "archive+db"
        archive_seeds = df_ranked.head(args.seeds_a // 2)["sequence"].tolist()
        conn = sqlite3.connect(args.db_path)
        db_top = pd.read_sql_query(
            "SELECT sequence FROM nanobodies WHERE target=? "
            "ORDER BY final_nanobody_score ASC LIMIT ?",
            conn, params=(target, args.seeds_a // 2))
        conn.close()
        top_seeds_a = list(dict.fromkeys(
            archive_seeds + db_top["sequence"].tolist()))
        top_seeds_b = top_seeds_a[:args.seeds_b]

    log.info(f"  Seed source:  {seeds_source}")
    log.info(f"  Seeds (A):    {len(top_seeds_a)}  (top archive + top DB)")
    log.info(f"  Seeds (B):    {len(top_seeds_b)}")
    for i, s in enumerate(top_seeds_a[:3]):
        log.debug(f"    Seed A #{i+1}: {s[:50]}...")
    for i, s in enumerate(top_seeds_b[:3]):
        log.debug(f"    Seed B #{i+1}: {s[:50]}...")

    # ── Generate ───────────────────────────────────────────────────────────
    t_gen   = time.time()
    cands_a = generate_option_a(
        top_seeds_a, args.n_per_seed_a, (args.mut_min, args.mut_max))
    cands_b = generate_option_b(
        top_seeds_b, args.n_per_seed_b)

    generation_methods: dict[str, str] = {}
    for c in cands_a:
        generation_methods[normalize_seq(c)] = "option_a"
    for c in cands_b:
        generation_methods[normalize_seq(c)] = "option_b"

    all_cands = list(dict.fromkeys(cands_a + cands_b))
    overlap   = len(cands_a) + len(cands_b) - len(all_cands)

    log.info(f"  Generation complete in {time.time()-t_gen:.1f}s")
    log.info(f"    Option A:   {len(cands_a)}")
    log.info(f"    Option B:   {len(cands_b)}")
    log.info(f"    A∩B overlap:{overlap}  (deduped)")
    log.info(f"    Total unique:{len(all_cands)}")

    new_cands = [c for c in all_cands
                 if (normalize_seq(c), target) not in already_scored]
    already_seen = len(all_cands) - len(new_cands)
    log.info(f"    Already scored (skipped): {already_seen}")
    log.info(f"    New candidates:           {len(new_cands)}")

    if not new_cands:
        log.info(f"  Nothing new to score — skipping iteration")
        return 0, already_scored

    # ── Validate ───────────────────────────────────────────────────────────
    t_val      = time.time()
    valid_seqs = await validate_candidates(new_cands, config, search_engines)
    val_time   = time.time() - t_val

    log.info(f"  Validation: {len(valid_seqs)}/{len(new_cands)} passed "
             f"in {val_time:.1f}s "
             f"({val_time/max(len(new_cands),1)*1000:.1f}ms/seq)")

    if not valid_seqs:
        log.warning(f"  No valid sequences — skipping BoltzGen")
        return 0, already_scored

    # ── BoltzGen scoring ───────────────────────────────────────────────────
    t_score = time.time()
    per_nanobody_components, final_boltzgen_scores = score_with_boltzgen(
        valid_seqs, config)
    score_time = time.time() - t_score

    rows = collect_boltzgen_results(
        valid_seqs, per_nanobody_components,
        final_boltzgen_scores, config, score_time,
        generation_methods=generation_methods)

    # ── Write DB ───────────────────────────────────────────────────────────
    if rows:
        upsert_results(rows, db_path=args.db_path)
        for row in rows:
            already_scored.add((row["sequence"], row["target"]))

        # Log top results from this iteration
        if rows:
            sorted_rows = sorted(
                [r for r in rows if r["final_nanobody_score"] != math.inf],
                key=lambda r: r["final_nanobody_score"])
            log.info(f"  Top-5 results this iteration:")
            for rank, r in enumerate(sorted_rows[:5]):
                log.info(
                    f"    #{rank+1}  score={r['final_nanobody_score']:.4f}  "
                    f"iiptm={r.get('design_iiptm') or float('nan'):.3f}  "
                    f"pae_min={r.get('min_design_to_target_pae') or float('nan'):.2f}  "
                    f"hbonds={r.get('plip_hbonds_refolded') or float('nan'):.0f}  "
                    f"liab={r.get('liability_score') or float('nan'):.3f}  "
                    f"[{r['generation_method']}]"
                )

        log_db_top(args.db_path, target, n=5)
    else:
        log.warning(f"  BoltzGen returned no rows")

    # ── Iteration summary ──────────────────────────────────────────────────
    iter_total = time.time()
    log.info(_sep("─"))
    log.info(f"  ITERATION {iteration} COMPLETE")
    log.info(f"    Generated:   {len(all_cands)}")
    log.info(f"    Validated:   {len(valid_seqs)}")
    log.info(f"    Scored:      {len(rows)}")
    log.info(f"    DB total:    {count_db_rows(args.db_path)}")
    log.info(_sep())

    return len(rows), already_scored


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

async def run_pipeline(args: argparse.Namespace, config: dict) -> None:
    for line in _banner("NANOBODY MINER  —  BoltzGen Pipeline", "█"):
        log.info(line)

    log.info(f"  Targets:        {config['nanobody_target']}")
    log.info(f"  Length range:   {config['min_sequence_length']}–"
             f"{config['max_sequence_length']}")
    log.info(f"  Min cysteines:  {config['min_cysteines']}")
    log.info(f"  Option A:       {args.seeds_a} seeds × {args.n_per_seed_a} mutants "
             f"({args.mut_min}–{args.mut_max} mutations, 70% CDR3)")
    log.info(f"  Option B:       {args.seeds_b} seeds × {args.n_per_seed_b} candidates "
             f"(40% swap / 30% indel / 30% resample)")
    log.info(f"  Max iterations: {'∞' if args.max_iterations < 0 else args.max_iterations}")
    log.info(f"  DB path:        {args.db_path}")
    log.info(f"  Log file:       {LOG_DIR}/miner_{_ts}.log")
    log.info(_sep())

    init_db(args.db_path)
    already_scored = get_already_scored(args.db_path)

    # ── Load archives ──────────────────────────────────────────────────────
    for line in _banner("Stage 1 — Load & Rank Archives"):
        log.info(line)
    archive_ranked: dict[str, pd.DataFrame] = {}
    for target in config["nanobody_target"]:
        df = load_archive(target)
        archive_ranked[target] = rank_archive(df)
    args._archive_ranked = archive_ranked

    # ── Build search engines ───────────────────────────────────────────────
    for line in _banner("Stage 2 — Build Similarity Search Engines"):
        log.info(line)
    search_engines: dict = {}
    for target in config["nanobody_target"]:
        log.info(f"  Indexing top sequences for {target}...")
        t0 = time.time()
        search_engines[target] = index_top_sequences(target)
        log.info(f"  Search engine ready for {target} in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════
    for line in _banner("Starting Iterative Generation Loop"):
        log.info(line)

    iteration     = 0
    total_written = 0
    loop_start    = time.time()

    while True:
        iteration += 1
        if args.max_iterations > 0 and iteration > args.max_iterations:
            log.info(f"  Reached max {args.max_iterations} iterations — stopping")
            break

        iter_start   = time.time()
        iter_written = 0

        for target in config["nanobody_target"]:
            n_written, already_scored = await run_one_iteration(
                iteration, target, config,
                search_engines, already_scored, args)
            iter_written += n_written

        total_written += iter_written
        iter_elapsed   = time.time() - iter_start
        total_elapsed  = time.time() - loop_start

        log.info(_sep("═"))
        log.info(f"  Iteration {iteration} wall time:  {iter_elapsed:.1f}s")
        log.info(f"  Rows written this iter:   {iter_written}")
        log.info(f"  Total rows written:       {total_written}")
        log.info(f"  DB total rows:            {count_db_rows(args.db_path)}")
        log.info(f"  Total wall time so far:   {total_elapsed:.1f}s "
               f"({total_elapsed/60:.1f}min)")
        log.info(_sep("═"))

        if args.iter_sleep > 0:
            log.info(f"  Sleeping {args.iter_sleep}s before next iteration...")
            await asyncio.sleep(args.iter_sleep)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - loop_start
    for line in _banner("PIPELINE COMPLETE"):
        log.info(line)
    log.info(f"  Iterations completed:   {iteration}")
    log.info(f"  Total rows written:     {total_written}")
    log.info(f"  Final DB row count:     {count_db_rows(args.db_path)}")
    log.info(f"  Total wall time:        {total_elapsed:.1f}s "
            f"({total_elapsed/60:.1f}min)")
    log.info(f"  Log saved to:           {LOG_DIR}/miner_{_ts}.log")

    # Print final top-N per target
    for target in config["nanobody_target"]:
        log_db_top(args.db_path, target, n=10)

    log.info(_sep("═"))


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nanobody Miner — BoltzGen Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",
                    default="config/config.yaml",
                    help="Path to config YAML")
    p.add_argument("--max-iterations",
                    type=int, default=-1, dest="max_iterations",
                    help="Max generation loops. -1 = run forever")
    p.add_argument("--seeds-a",
                    type=int, default=50, dest="seeds_a",
                    help="Top-N archive seeds for Option A mutation")
    p.add_argument("--seeds-b",
                    type=int, default=20, dest="seeds_b",
                    help="Top-N seeds for Option B CDR3 diversity")
    p.add_argument("--n-per-seed-a",
                    type=int, default=100, dest="n_per_seed_a",
                    help="Mutants per seed for Option A")
    p.add_argument("--n-per-seed-b",
                    type=int, default=100, dest="n_per_seed_b",
                    help="Candidates per seed for Option B")
    p.add_argument("--mut-min",
                    type=int, default=2, dest="mut_min",
                    help="Min CDR point mutations for Option A")
    p.add_argument("--mut-max",
                    type=int, default=5, dest="mut_max",
                    help="Max CDR point mutations for Option A")
    p.add_argument("--db-path",
                    default=DB_PATH, dest="db_path",
                    help="Path to nanobodies.sqlite output DB")
    p.add_argument("--iter-sleep",
                    type=float, default=0.0, dest="iter_sleep",
                    help="Seconds to sleep between iterations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    for line in _banner("NANOBODY MINER — STARTUP", "─"):
        log.info(line)
    log.info(f"  Targets:        {config['nanobody_target']}")
    log.info(f"  Length range:   {config['min_sequence_length']}–"
            f"{config['max_sequence_length']}")
    log.info(f"  Min cysteines:  {config['min_cysteines']}")
    log.info(f"  Max iterations: {'∞' if args.max_iterations < 0 else args.max_iterations}")
    log.info(f"  Seeds A/B:      {args.seeds_a} / {args.seeds_b}")
    log.info(f"  Per-seed A/B:   {args.n_per_seed_a} / {args.n_per_seed_b}")
    log.info(f"  Mutations A:    {args.mut_min}–{args.mut_max}")
    log.info(f"  DB path:        {args.db_path}")
    log.info(f"  Log dir:        {LOG_DIR}")
    log.info(_sep("─"))

    asyncio.run(run_pipeline(args, config))


if __name__ == "__main__":
    main()