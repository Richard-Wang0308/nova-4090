#!/usr/bin/env python3
"""
Merge two SQLite score databases into one.

Rules when molecule_name appears in both:
- score: average of all scores for that molecule
- scored_at: latest timestamp
- available: logical AND across values
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


TABLE_NAME = "scored_molecules"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two SQLite score databases.")
    parser.add_argument(
        "--db1",
        default="score_results_2.sqlite",
        help="Path to first SQLite database.",
    )
    parser.add_argument(
        "--db2",
        default="score_results_2_2769.sqlite",
        help="Path to second SQLite database.",
    )
    parser.add_argument(
        "--out",
        default="merged.sqlite",
        help="Path to output merged SQLite database.",
    )
    return parser.parse_args()


def to_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "t", "yes", "y"}
    return bool(value)


def parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Accept common SQLite timestamp formats, including trailing Z.
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def is_later(current: Optional[str], candidate: Optional[str]) -> bool:
    current_dt = parse_timestamp(current)
    candidate_dt = parse_timestamp(candidate)

    if candidate_dt and current_dt:
        return candidate_dt > current_dt
    if candidate_dt and not current_dt:
        return True
    if current_dt and not candidate_dt:
        return False

    # Fallback to lexicographic compare when parsing fails or both are None.
    return (candidate or "") > (current or "")


def ensure_output_schema(conn_out: sqlite3.Connection) -> None:
    conn_out.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn_out.execute(
        f"""
        CREATE TABLE {TABLE_NAME} (
            molecule_name TEXT PRIMARY KEY,
            score REAL,
            scored_at TIMESTAMP,
            available BOOLEAN
        )
        """
    )
    conn_out.commit()


def merge_databases(db1: Path, db2: Path, out_db: Path) -> int:
    accumulator: dict[str, dict[str, object]] = {}

    for db_path in (db1, db2):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                f"SELECT molecule_name, score, scored_at, available FROM {TABLE_NAME}"
            )
            for molecule_name, score, scored_at, available in cursor:
                if molecule_name not in accumulator:
                    accumulator[molecule_name] = {
                        "score_sum": float(score),
                        "score_count": 1,
                        "scored_at": scored_at,
                        "available": to_bool(available),
                    }
                    continue

                entry = accumulator[molecule_name]
                entry["score_sum"] = float(entry["score_sum"]) + float(score)
                entry["score_count"] = int(entry["score_count"]) + 1
                entry["available"] = bool(entry["available"]) and to_bool(available)
                if is_later(entry["scored_at"], scored_at):
                    entry["scored_at"] = scored_at

    with sqlite3.connect(out_db) as conn_out:
        ensure_output_schema(conn_out)
        rows = []
        for molecule_name, entry in accumulator.items():
            score_avg = float(entry["score_sum"]) / int(entry["score_count"])
            rows.append(
                (
                    molecule_name,
                    score_avg,
                    entry["scored_at"],
                    int(bool(entry["available"])),
                )
            )
        conn_out.executemany(
            f"INSERT INTO {TABLE_NAME} (molecule_name, score, scored_at, available) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn_out.commit()

    return len(accumulator)


def main() -> None:
    args = parse_args()
    db1 = Path(args.db1).expanduser().resolve()
    db2 = Path(args.db2).expanduser().resolve()
    out_db = Path(args.out).expanduser().resolve()

    merged_count = merge_databases(db1, db2, out_db)
    print(f"Merged {merged_count} unique molecules into: {out_db}")


if __name__ == "__main__":
    main()
