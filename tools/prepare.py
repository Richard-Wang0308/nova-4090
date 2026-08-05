#!/usr/bin/env python3
"""Collect competition submissions and analyze reactant stats.

One command does both: fetch leaderboard → data/rxn{1..5}.csv → analysis reports.

  python3 tools/prepare.py --end_epoch 24304
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NUM_REACTIONS = 5
ScoreStats = Dict[str, List[float]]


def resolve_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


# ============================================================================
# Collect: API → per-reaction CSVs
# ============================================================================


def fetch_leaderboard_data(epoch: int) -> Optional[dict]:
    url = (
        "https://compound-api-staging.metanova-labs.ai"
        f"/api/competitions/leaderboard/{epoch}/molecules"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error fetching epoch {epoch}: {e}")
        return None


def extract_training_samples(data: dict) -> List[dict]:
    samples = []
    leaderboard = data.get("data", [])
    if not leaderboard:
        logger.warning("No leaderboard data found")
        return samples

    for entry in leaderboard:
        final_score = entry.get("final_score")
        molecules = entry.get("molecules", [])
        if not molecules:
            continue
        for mol_entry in molecules:
            mol_name = mol_entry.get("name", "")
            if not mol_name:
                continue
            samples.append(
                {
                    "molecule_name": mol_name,
                    "final_score": final_score,
                }
            )
    return samples


def get_reaction_index(molecule_name: str) -> Optional[int]:
    for i in range(1, NUM_REACTIONS + 1):
        if molecule_name.startswith(f"rxn:{i}:"):
            return i
    return None


def get_rxn_output_path(output_dir: str, i: int) -> str:
    return os.path.join(output_dir, f"rxn{i}.csv")


def rxn_csv_paths(output_dir: str) -> List[str]:
    return [get_rxn_output_path(output_dir, i) for i in range(1, NUM_REACTIONS + 1)]


def get_last_collected_epoch(output_dir: str) -> Optional[int]:
    last_epoch = None
    for i in range(1, NUM_REACTIONS + 1):
        path = get_rxn_output_path(output_dir, i)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if "epoch" in df.columns and len(df) > 0:
                max_epoch = int(df["epoch"].max())
                if last_epoch is None or max_epoch > last_epoch:
                    last_epoch = max_epoch
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    if last_epoch is not None:
        logger.info(f"Last collected epoch across rxn files: {last_epoch}")
    else:
        logger.info("No previous rxn CSV files found; starting fresh")
    return last_epoch


def split_and_save_samples(samples: List[dict], output_dir: str) -> None:
    if not samples:
        return

    buckets: Dict[int, List[dict]] = {i: [] for i in range(1, NUM_REACTIONS + 1)}
    unmatched = 0
    for sample in samples:
        idx = get_reaction_index(sample["molecule_name"])
        if idx is None:
            unmatched += 1
            logger.debug(f"Unmatched molecule skipped: {sample['molecule_name']}")
            continue
        buckets[idx].append(sample)

    if unmatched:
        logger.warning(f"{unmatched} unmatched molecules skipped (no rxn:N: prefix)")

    for i in range(1, NUM_REACTIONS + 1):
        rows = buckets[i]
        if not rows:
            continue

        path = get_rxn_output_path(output_dir, i)
        df = pd.DataFrame(rows)
        columns_order = ["molecule_name", "final_score", "epoch"]
        df = df[[col for col in columns_order if col in df.columns]]

        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            df = pd.concat([df_existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["molecule_name", "epoch"], keep="last")

        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} total samples to {path} (+{len(rows)} new)")


def collect_single_epoch_data(epoch: int, output_dir: str) -> int:
    data = fetch_leaderboard_data(epoch)
    if not data:
        logger.debug(f"Failed to fetch data for epoch {epoch}")
        return 0

    samples = extract_training_samples(data)
    if not samples:
        logger.debug(f"No samples extracted from epoch {epoch}")
        return 0

    for sample in samples:
        sample["epoch"] = epoch

    logger.info(f"Epoch {epoch}: Extracted {len(samples)} samples")
    split_and_save_samples(samples, output_dir)
    return len(samples)


def collect_epochs(
    start_epoch: int,
    end_epoch: int,
    output_dir: str,
    last_collected_epoch: Optional[int],
) -> int:
    if last_collected_epoch is None:
        collection_start = start_epoch
        logger.info(f"First collection: epochs {start_epoch} → {end_epoch}")
    else:
        collection_start = last_collected_epoch + 1
        logger.info(f"Resuming collection: epochs {collection_start} → {end_epoch}")

    if collection_start > end_epoch:
        logger.info(
            f"No new epochs to collect (last={last_collected_epoch}, end={end_epoch})"
        )
        return 0

    epochs = list(range(collection_start, end_epoch + 1))
    logger.info(f"Collecting {len(epochs)} epochs ({epochs[0]} … {epochs[-1]})")

    total = 0
    for epoch in epochs:
        total += collect_single_epoch_data(epoch, output_dir)
        time.sleep(0.2)
    return total


# ============================================================================
# Analyze: reactant stats from rxn CSVs
# ============================================================================


def resolve_score_column(fieldnames: List[str]) -> str:
    for name in ("final_score", "score"):
        if name in fieldnames:
            return name
    raise ValueError(
        "CSV must contain a score column named 'final_score' or 'score'. "
        f"Found columns: {fieldnames}"
    )


def parse_reactants(molecule_name: str) -> List[str]:
    parts = molecule_name.strip().split(":")
    if len(parts) < 4 or parts[0] != "rxn":
        raise ValueError(f"Unexpected molecule_name format: {molecule_name!r}")

    reactants = parts[2:]
    if len(reactants) not in (2, 3):
        raise ValueError(
            f"Expected 2 or 3 reactants, got {len(reactants)} in {molecule_name!r}"
        )
    return reactants


def analyze_csv(csv_path: str) -> Tuple[Dict[int, ScoreStats], int, int]:
    position_stats: Dict[int, ScoreStats] = defaultdict(lambda: defaultdict(list))
    row_count = 0
    skipped = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        score_col = resolve_score_column(list(reader.fieldnames))
        if "molecule_name" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'molecule_name' column")

        for row in reader:
            molecule_name = (row.get("molecule_name") or "").strip()
            score_raw = (row.get(score_col) or "").strip()
            if not molecule_name or not score_raw:
                skipped += 1
                continue

            try:
                score = float(score_raw)
                reactants = parse_reactants(molecule_name)
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: skipping row {molecule_name!r}: {e}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            row_count += 1
            for pos, reactant_id in enumerate(reactants, start=1):
                position_stats[pos][reactant_id].append(score)

    return position_stats, row_count, skipped


def default_analysis_output_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return f"{base}_reactant_analysis.csv"


def write_report(
    position_stats: Dict[int, ScoreStats],
    output_path: str,
    source_csv: str,
    row_count: int,
) -> None:
    rows_out: List[dict] = []
    for pos in sorted(position_stats):
        items = []
        for reactant_id, scores in position_stats[pos].items():
            items.append(
                {
                    "reactant_position": pos,
                    "reactant_id": reactant_id,
                    "count": len(scores),
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                }
            )
        items.sort(key=lambda r: (-r["count"], -r["max_score"], r["reactant_id"]))
        rows_out.extend(items)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        f.write("# Reactant analysis\n")
        f.write(f"# source: {os.path.abspath(source_csv)}\n")
        f.write(f"# molecules_analyzed: {row_count}\n")
        f.write(
            "# reactant_position: 1 = first reactant after rxn:N, "
            "2 = second, 3 = third (if present)\n"
        )
        f.write(
            "# sorted within each position by count (desc), then max_score (desc)\n"
        )

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reactant_position",
                "reactant_id",
                "count",
                "avg_score",
                "max_score",
            ],
        )
        writer.writeheader()
        for row in rows_out:
            writer.writerow(
                {
                    "reactant_position": row["reactant_position"],
                    "reactant_id": row["reactant_id"],
                    "count": row["count"],
                    "avg_score": f"{row['avg_score']:.10f}",
                    "max_score": f"{row['max_score']:.10f}",
                }
            )

    summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Reactant analysis summary\n")
        f.write("=" * 72 + "\n")
        f.write(f"Source:    {os.path.abspath(source_csv)}\n")
        f.write(f"Molecules: {row_count}\n")
        f.write(f"Detail:    {os.path.abspath(output_path)}\n\n")

        for pos in sorted(position_stats):
            stats = position_stats[pos]
            items = []
            for reactant_id, scores in stats.items():
                items.append(
                    (
                        reactant_id,
                        len(scores),
                        sum(scores) / len(scores),
                        max(scores),
                    )
                )
            items.sort(key=lambda t: (-t[1], -t[3], t[0]))

            f.write("-" * 72 + "\n")
            f.write(f"Position {pos}  ({len(items)} unique reactants)\n")
            f.write("-" * 72 + "\n")
            f.write(
                f"{'reactant_id':>14}  {'count':>6}  "
                f"{'avg_score':>14}  {'max_score':>14}\n"
            )
            for reactant_id, count, avg_score, max_score in items:
                f.write(
                    f"{reactant_id:>14}  {count:>6}  "
                    f"{avg_score:>14.10f}  {max_score:>14.10f}\n"
                )
            f.write("\n")

    logger.info(f"Wrote detailed CSV:  {output_path}")
    logger.info(f"Wrote text summary:  {summary_path}")


def analyze_one(csv_path: str) -> None:
    if not os.path.isfile(csv_path):
        logger.warning(f"Skipping missing CSV: {csv_path}")
        return

    logger.info(f"=== Analyzing {csv_path} ===")
    position_stats, row_count, skipped = analyze_csv(csv_path)
    if row_count == 0:
        logger.warning(f"No valid molecule rows in {csv_path}; skipping")
        return

    write_report(
        position_stats,
        default_analysis_output_path(csv_path),
        csv_path,
        row_count,
    )
    msg = f"Analyzed {row_count} molecules"
    if skipped:
        msg += f" (skipped {skipped})"
    logger.info(msg)
    for pos in sorted(position_stats):
        logger.info(f"  position {pos}: {len(position_stats[pos])} unique reactants")


def run_analyze(output_dir: str) -> None:
    for csv_path in rxn_csv_paths(output_dir):
        analyze_one(csv_path)


# ============================================================================
# Main: collect then analyze
# ============================================================================


def run(end_epoch: int, start_epoch: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Collect + analyze")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {start_epoch} → {end_epoch}")
    logger.info("=" * 70)

    last = get_last_collected_epoch(output_dir)
    n = collect_epochs(start_epoch, end_epoch, output_dir, last)
    logger.info(f"Collection done — {n} new samples")

    logger.info("Running reactant analysis…")
    run_analyze(output_dir)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect submissions into data/rxn{1..5}.csv, then analyze."
    )
    parser.add_argument(
        "--end_epoch",
        type=int,
        required=True,
        help="Collect through this epoch (inclusive), then analyze",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=23863,
        help="First epoch if no prior rxn CSVs exist (default: 23863)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(BASE_DIR, "data"),
        help="Directory for rxn CSVs and analysis outputs",
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    run(args.end_epoch, args.start_epoch, output_dir)


if __name__ == "__main__":
    main()
