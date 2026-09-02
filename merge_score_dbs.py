#!/usr/bin/env python3
"""
merge_score_dbs.py — fold one score DB into another, in place.

    python3 merge_score_dbs.py score_results.sqlite score_results_1.sqlite
    python3 merge_score_dbs.py score_results.sqlite score_results_1.sqlite --apply

SOURCE is read only. DEST gains every molecule it was missing, and for a
molecule both files already hold, DEST keeps one merged row.

SCORES AVERAGE
--------------
A molecule present in both is the same molecule scored twice, so its merged
score is the mean of the two. Where the two files already agree -- which is
most of them, since both descend from the same run -- the mean is that same
value and nothing moves.

Every other column resolves as COALESCE(dest, source): DEST's value wins when
it has one, SOURCE fills the gaps. `scored_at` takes the later of the two, so
the timestamp still means "when this molecule was last measured".

REPLICATES UNION, THEY DO NOT AVERAGE
-------------------------------------
molecule_replicates rows are individual measurements keyed by
(molecule_name, seed, draw_idx). Rows only one file has are copied over. When
both files hold the SAME key with a different score, DEST's row stands and the
collision is reported -- averaging two draws that each claim to be draw 3 would
invent a measurement nobody took, which is exactly the thing the replicate
table exists to avoid.

    NOTE. Averaging a score does not re-derive it from draws. A molecule whose
    stored score was the mean of its three draws, and whose two files disagree,
    ends up with a score that is no longer the mean of the draws on record. The
    script counts those and names them; re-confirming them is the clean fix.

RUN IT ONCE
-----------
Averaging is not idempotent. Merge the same source twice and the second run
averages the already-averaged score against the source's again, dragging it
half-way toward the source every time -- 0.105144 becomes 0.107268, then
0.108330, and nothing in the file says it happened.

So a successful merge stamps DEST's metadata with a fingerprint of the source
content it absorbed, and a second run against the same content refuses. --force
overrides it, which you want only when the source genuinely holds new draws of
the same molecules.

NOTHING IS WRITTEN WITHOUT --apply
----------------------------------
The default is a report. DEST is a file a miner fills over weeks, and the
destructive path should not be one typo away from the safe one. --apply backs
DEST up before touching it unless --no-backup.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sqlite3
import sys
import time
from typing import Dict, List, Tuple

TABLE = "scored_molecules"
REPLICATES = "molecule_replicates"
KEY = "molecule_name"
REP_KEY = ("molecule_name", "seed", "draw_idx")
BUSY_TIMEOUT_MS = 30_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a score DB into another, averaging the scores of "
                    "molecules both files hold. Reports only unless --apply.")
    p.add_argument("source", help="DB to merge FROM (read only)")
    p.add_argument("dest", help="DB to merge INTO (rewritten with --apply)")
    p.add_argument("--apply", action="store_true",
                   help="actually write. Without this the script only reports")
    p.add_argument("--no-backup", action="store_true",
                   help="skip the pre-merge copy of DEST (not recommended)")
    p.add_argument("--force", action="store_true",
                   help="write even though another process has DEST open")
    return p.parse_args()


def connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro" if readonly else path
    conn = sqlite3.connect(uri, uri=readonly, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def check(conn: sqlite3.Connection, label: str) -> bool:
    try:
        verdict = conn.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.Error as e:
        verdict = f"unreadable: {e}"
    if verdict != "ok":
        print(f"{label} failed its integrity check:\n  {verdict}", file=sys.stderr)
        return False
    return True


def holders(path: str) -> List[Tuple[int, str]]:
    """Processes other than this one holding `path` open, read from /proc."""
    target, found, me = os.path.realpath(path), [], os.getpid()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.path.realpath(fd) != target:
                continue
            pid = int(fd.split("/")[2])
            if pid == me:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            if (pid, cmd[:90]) not in found:
                found.append((pid, cmd[:90]))
        except (OSError, ValueError, IndexError):
            continue
    return found


def fingerprint(rows: Dict[str, dict]) -> str:
    """Content hash of a source table: same molecules, same scores, same hash.

    Keyed on content rather than filename so a renamed or copied source is still
    recognised as one this DB has already absorbed.
    """
    h = hashlib.sha256()
    for name in sorted(rows):
        h.update(f"{name}\x00{rows[name].get('score')!r}\x00".encode())
    return h.hexdigest()[:32]


def already_merged(conn: sqlite3.Connection, fp: str) -> str:
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?",
                           (f"merged_from:{fp}",)).fetchone()
    except sqlite3.Error:
        return ""
    return row[0] if row else ""


def columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')")]


def merge_molecules(dst_rows: Dict[str, dict], src_rows: Dict[str, dict],
                    cols: List[str]) -> Tuple[List[dict], dict]:
    """One merged row per molecule. Returns (rows, stats)."""
    stats = {"only_dest": 0, "only_src": 0, "both": 0, "averaged": 0,
             "filled": 0, "moved": []}
    out = []
    for name in sorted(set(dst_rows) | set(src_rows)):
        a, b = dst_rows.get(name), src_rows.get(name)
        if a is None:
            stats["only_src"] += 1
            out.append({k: b.get(k) for k in cols})
            continue
        if b is None:
            stats["only_dest"] += 1
            out.append({k: a.get(k) for k in cols})
            continue

        stats["both"] += 1
        row = {}
        for k in cols:
            av, bv = a.get(k), b.get(k)
            if k == "score":
                if av is not None and bv is not None:
                    row[k] = (float(av) + float(bv)) / 2.0
                    if abs(float(av) - float(bv)) > 1e-12:
                        stats["averaged"] += 1
                        stats["moved"].append((name, float(av), float(bv), row[k]))
                else:
                    row[k] = av if av is not None else bv
            elif k == "scored_at":
                row[k] = max(x for x in (av, bv) if x is not None) \
                    if (av or bv) else None
            else:
                row[k] = av if av is not None else bv
                if av is None and bv is not None:
                    stats["filled"] += 1
        out.append(row)
    return out, stats


def merge_replicates(dst_rows: Dict[tuple, dict], src_rows: Dict[tuple, dict],
                     cols: List[str]) -> Tuple[List[dict], dict]:
    """Union by (name, seed, draw_idx); DEST wins an exact-key collision."""
    stats = {"only_dest": 0, "only_src": 0, "collisions": 0, "conflicting": 0}
    out = []
    for key in sorted(set(dst_rows) | set(src_rows), key=lambda k: tuple(map(str, k))):
        a, b = dst_rows.get(key), src_rows.get(key)
        if a is None:
            stats["only_src"] += 1
            out.append({k: b.get(k) for k in cols})
        elif b is None:
            stats["only_dest"] += 1
            out.append({k: a.get(k) for k in cols})
        else:
            stats["collisions"] += 1
            av, bv = a.get("score"), b.get("score")
            if av is not None and bv is not None and abs(av - bv) > 1e-12:
                stats["conflicting"] += 1
            out.append({k: a.get(k) for k in cols})
    return out, stats


def main() -> int:
    args = parse_args()
    for p in (args.source, args.dest):
        if not os.path.exists(p):
            print(f"no such DB: {p}", file=sys.stderr)
            return 2
    if os.path.realpath(args.source) == os.path.realpath(args.dest):
        print("source and dest are the same file", file=sys.stderr)
        return 2

    src = connect(args.source, readonly=True)
    dst = connect(args.dest)
    if not check(src, args.source) or not check(dst, args.dest):
        return 1

    mol_cols = columns(dst, TABLE)
    missing = [c for c in columns(src, TABLE) if c not in mol_cols]
    if missing:
        print(f"source has columns dest lacks, they would be dropped: {missing}",
              file=sys.stderr)
        return 1

    dst_mol = {r[KEY]: dict(r) for r in dst.execute(f"SELECT * FROM {TABLE}")}
    src_mol = {r[KEY]: dict(r) for r in src.execute(f"SELECT * FROM {TABLE}")}
    merged, mstats = merge_molecules(dst_mol, src_mol, mol_cols)

    rep_cols = columns(dst, REPLICATES)
    keyer = lambda r: tuple(r[k] for k in REP_KEY)
    dst_rep = {keyer(r): dict(r) for r in dst.execute(f"SELECT * FROM {REPLICATES}")}
    src_rep = {keyer(r): dict(r) for r in src.execute(f"SELECT * FROM {REPLICATES}")}
    merged_rep, rstats = merge_replicates(dst_rep, src_rep, rep_cols)

    print(f"source {args.source}: {len(src_mol):,} molecules, {len(src_rep):,} draws")
    print(f"dest   {args.dest}: {len(dst_mol):,} molecules, {len(dst_rep):,} draws")
    print(f"\n{TABLE}")
    print(f"  only in dest              {mstats['only_dest']:>9,}")
    print(f"  only in source (new)      {mstats['only_src']:>9,}")
    print(f"  in both                   {mstats['both']:>9,}")
    print(f"    scores averaged (differ){mstats['averaged']:>9,}")
    print(f"    identical, no change    {mstats['both'] - mstats['averaged']:>9,}")
    print(f"  null fields filled from source {mstats['filled']:>4,}")
    print(f"  merged total              {len(merged):>9,}")
    if mstats["moved"]:
        print("\n  averaged scores:")
        for name, av, bv, new in sorted(mstats["moved"], key=lambda x: -x[3]):
            print(f"    {name:<26} dest={av:.6f} src={bv:.6f} -> {new:.6f} "
                  f"({new - av:+.6f})")
    print(f"\n{REPLICATES}")
    print(f"  only in dest              {rstats['only_dest']:>9,}")
    print(f"  only in source (new)      {rstats['only_src']:>9,}")
    print(f"  same key, dest kept       {rstats['collisions']:>9,}"
          f"  (of which differ: {rstats['conflicting']})")
    print(f"  merged total              {len(merged_rep):>9,}")

    # A molecule whose score was the mean of its draws, and whose two files
    # disagreed, no longer satisfies that. Name them rather than bury them.
    broken = []
    if mstats["moved"]:
        by_name = {}
        for r in merged_rep:
            by_name.setdefault(r["molecule_name"], []).append(r["score"])
        for name, _, _, new in mstats["moved"]:
            draws = [d for d in by_name.get(name, []) if d is not None]
            if draws and abs(sum(draws) / len(draws) - new) > 1e-9:
                broken.append((name, new, draws))
    if broken:
        print(f"\n  warning: {len(broken)} averaged molecule(s) no longer match "
              f"the mean of their recorded draws:")
        for name, new, draws in broken:
            print(f"    {name:<26} score={new:.6f} vs mean of "
                  f"{len(draws)} draws={sum(draws)/len(draws):.6f}")
        print("    re-confirm them to bring score and draws back into line.")

    if not args.apply:
        print("\nreport only — nothing changed. Re-run with --apply to write.")
        return 0

    fp = fingerprint(src_mol)
    seen = already_merged(dst, fp)
    if seen and not args.force:
        print(f"\nthis exact source content was already merged into "
              f"{args.dest}: {seen}\nMerging it again would average the "
              f"already-averaged scores a second time. Pass --force only if "
              f"the source really holds new measurements. Nothing changed.",
              file=sys.stderr)
        return 1
    if seen:
        print(f"\nwarning: already merged ({seen}) — re-averaging because "
              f"--force was given")

    open_by = holders(args.dest)
    if open_by and not args.force:
        print(f"\n{args.dest} is open by another process:", file=sys.stderr)
        for pid, cmd in open_by:
            print(f"    pid {pid}: {cmd}", file=sys.stderr)
        print("Stop it first, or pass --force. Nothing changed.", file=sys.stderr)
        return 1

    if not args.no_backup:
        backup = f"{args.dest}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        with sqlite3.connect(backup) as b:
            dst.backup(b)
        print(f"\nbackup -> {backup} ({os.path.getsize(backup):,} bytes)")

    t0 = time.time()
    with dst:
        dst.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({','.join(mol_cols)}) "
            f"VALUES ({','.join('?' * len(mol_cols))})",
            [[r[k] for k in mol_cols] for r in merged])
        dst.executemany(
            f"INSERT OR REPLACE INTO {REPLICATES} ({','.join(rep_cols)}) "
            f"VALUES ({','.join('?' * len(rep_cols))})",
            [[r[k] for k in rep_cols] for r in merged_rep])
        dst.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (f"merged_from:{fp}",
                     f"{os.path.basename(args.source)} "
                     f"({len(src_mol)} molecules) at "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(f"wrote {len(merged):,} molecules and {len(merged_rep):,} draws "
          f"in {time.time() - t0:.1f}s")

    n = dst.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    nr = dst.execute(f"SELECT COUNT(*) FROM {REPLICATES}").fetchone()[0]
    orph = dst.execute(
        f"SELECT COUNT(*) FROM {REPLICATES} r LEFT JOIN {TABLE} s "
        f"ON s.{KEY} = r.{KEY} WHERE s.{KEY} IS NULL").fetchone()[0]
    print(f"{args.dest} now holds {n:,} molecules, {nr:,} draws | "
          f"orphaned draws: {orph} | integrity: "
          f"{dst.execute('PRAGMA quick_check').fetchone()[0]}")
    src.close(); dst.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
