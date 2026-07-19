#!/usr/bin/env python3
"""
Analyze reactant statistics from a scored-molecule CSV.

Molecule names look like:
  rxn:2:A:B      (2 reactants)
  rxn:3:A:B:C    (3 reactants)

For each reactant position, reports how often each exact reactant ID
appears, plus its average and maximum score.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ScoreStats = Dict[str, List[float]]  # reactant_id -> list of scores


def resolve_path(path: str) -> str:
    """Resolve relative paths against the project root (nova-4090)."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze reactant appearance counts, average scores, "
            "and max scores per reactant position."
        )
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to input CSV (columns: molecule_name, score/final_score, ...)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output file path (default: <csv_stem>_reactant_analysis.csv "
            "next to the input file)"
        ),
    )
    return parser.parse_args()


def resolve_score_column(fieldnames: List[str]) -> str:
    for name in ("final_score", "score"):
        if name in fieldnames:
            return name
    raise ValueError(
        "CSV must contain a score column named 'final_score' or 'score'. "
        f"Found columns: {fieldnames}"
    )


def parse_reactants(molecule_name: str) -> List[str]:
    """
    Parse reactant IDs from a molecule name.

    Expected formats:
      rxn:<n>:<r1>:<r2>
      rxn:<n>:<r1>:<r2>:<r3>
    """
    parts = molecule_name.strip().split(":")
    if len(parts) < 4 or parts[0] != "rxn":
        raise ValueError(f"Unexpected molecule_name format: {molecule_name!r}")

    reactants = parts[2:]
    if len(reactants) not in (2, 3):
        raise ValueError(
            f"Expected 2 or 3 reactants, got {len(reactants)} in {molecule_name!r}"
        )
    return reactants


def analyze(csv_path: str) -> Tuple[Dict[int, ScoreStats], int, int]:
    """
    Build per-position reactant -> scores mapping.

    Returns:
      (position_stats, row_count, skipped_count)
      position_stats maps 1-based position -> {reactant_id: [scores...]}
    """
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
                print(f"Warning: skipping row {molecule_name!r}: {e}", file=sys.stderr)
                skipped += 1
                continue

            row_count += 1
            for pos, reactant_id in enumerate(reactants, start=1):
                position_stats[pos][reactant_id].append(score)

    return position_stats, row_count, skipped


def default_output_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return f"{base}_reactant_analysis.csv"


def write_report(
    position_stats: Dict[int, ScoreStats],
    output_path: str,
    source_csv: str,
    row_count: int,
) -> None:
    """Write a readable CSV report, one block of rows per reactant position."""
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
        # Most frequent first; break ties by higher max score
        items.sort(key=lambda r: (-r["count"], -r["max_score"], r["reactant_id"]))
        rows_out.extend(items)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        # Header comment lines so the file is easy to skim
        f.write(f"# Reactant analysis\n")
        f.write(f"# source: {os.path.abspath(source_csv)}\n")
        f.write(f"# molecules_analyzed: {row_count}\n")
        f.write(
            "# reactant_position: 1 = first reactant after rxn:N, "
            "2 = second, 3 = third (if present)\n"
        )
        f.write("# sorted within each position by count (desc), then max_score (desc)\n")

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

    # Also write a plain-text summary next to the CSV for quick reading
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
                f"{'reactant_id':>14}  {'count':>6}  {'avg_score':>14}  {'max_score':>14}\n"
            )
            for reactant_id, count, avg_score, max_score in items:
                f.write(
                    f"{reactant_id:>14}  {count:>6}  {avg_score:>14.10f}  "
                    f"{max_score:>14.10f}\n"
                )
            f.write("\n")

    print(f"Wrote detailed CSV:  {output_path}")
    print(f"Wrote text summary:  {summary_path}")


def main() -> None:
    args = parse_args()
    csv_path = resolve_path(args.csv_path)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"File not found: {csv_path}")

    position_stats, row_count, skipped = analyze(csv_path)
    if row_count == 0:
        raise SystemExit("No valid molecule rows found in CSV")

    output_path = resolve_path(args.output) if args.output else default_output_path(csv_path)
    write_report(position_stats, output_path, csv_path, row_count)

    print(f"Analyzed {row_count} molecules" + (f" (skipped {skipped})" if skipped else ""))
    for pos in sorted(position_stats):
        print(f"  position {pos}: {len(position_stats[pos])} unique reactants")


if __name__ == "__main__":
    main()
