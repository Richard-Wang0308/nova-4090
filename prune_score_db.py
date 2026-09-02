#!/usr/bin/env python3
"""
prune_score_db.py — delete dead weight from a score DB, in place.

Two things go:

    score < THRESHOLD     molecules too weak to ever reach a submission
    available = FALSE     molecules the DB has already marked unavailable

    python3 prune_score_db.py score_results_1.sqlite 0.05            # report only
    python3 prune_score_db.py score_results_1.sqlite 0.05 --apply    # delete

`score < THRESHOLD` is strict: a molecule sitting exactly on the threshold is
kept, so the same threshold used elsewhere in the pipeline means the same set
here.

NOTHING IS DELETED WITHOUT --apply
----------------------------------
The default is a report. This is not politeness: the destructive path is one
argument away from the reporting path, the two differ only in a flag, and this
script is most useful on exactly the file a miner has been filling for weeks.
Reporting by default means a mistyped or half-remembered command prints a
table instead of deleting 48,000 rows. --apply is the only way to change the
file, and it still takes a backup first unless --no-backup.

REPLICATES GO WITH THE MOLECULE
-------------------------------
molecule_replicates has no foreign key, so nothing cascades on its own. A
pruned molecule whose draws stayed behind is worse than useless: rescore.py
reads prior draws back by name, and a molecule re-scored later would be
averaged against draws belonging to a row that no longer exists. This deletes
both together, in one transaction.

NOT SAFE ON A DB A SEARCHER IS WRITING
--------------------------------------
SQLite will not corrupt under this -- WAL handles the concurrency -- but a
running hunter/orchestrator holds its own `seen` set and a fitted surrogate in
memory, both built from rows this deletes. It will not notice they are gone and
will re-score and re-insert molecules you just removed.

So --apply refuses on two independent signals, either of which is enough:

  * another process has the file open, read from /proc;
  * the file was written inside the last --idle-minutes.

Neither alone is sufficient, which is why both are here. /proc is exact about
open descriptors but hunter does not hold one for most of a round: it opens the
DB, reads it into a DataFrame, and the connection is closed again as soon as the
reader is collected, so a searcher forty minutes deep in Boltz calls looks
exactly like a stopped one. And mtime alone is the mirror image -- hunter
commits in bursts several minutes apart, so a short window reads the gap between
two bursts as an idle database. Together they cover each other: take the open
descriptor when there is one, fall back on recent writes when there is not, and
size the window to span a commit gap rather than a single burst.

Stop the searcher, prune, start it again. --force overrides both.

WHAT SURVIVES IS WHAT THE SEARCHER LEARNS FROM
----------------------------------------------
hunter re-reads this DB every round: `seen` (hunter.main), `surrogate.fit()` and
FieldPrior's component statistics are all built from whatever rows are still
here. Pruning is therefore not free housekeeping -- deleting the low scorers
deletes the negative examples the surrogate is fitted on, and deleting a
molecule takes it out of `seen`, so the searcher is free to generate and pay to
score it again. Prune to reclaim disk, not to "clean up" a working DB.
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
import time
from typing import List, Tuple

TABLE = "scored_molecules"
REPLICATES = "molecule_replicates"
# Long enough to sit out a searcher's batch commit rather than dying on it.
BUSY_TIMEOUT_MS = 30_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delete molecules below a score threshold, and every "
                    "molecule marked unavailable, from a score DB. Reports "
                    "only unless --apply is given.")
    p.add_argument("db", help="path to the score DB (e.g. score_results_1.sqlite)")
    p.add_argument("threshold", type=float,
                   help="delete molecules with score strictly below this")
    p.add_argument("--apply", action="store_true",
                   help="actually delete. Without this the script only reports")
    p.add_argument("--no-backup", action="store_true",
                   help="skip the pre-delete copy (not recommended)")
    p.add_argument("--vacuum", action="store_true",
                   help="VACUUM afterwards to return the freed pages to the "
                        "filesystem. Needs free space equal to the DB size and "
                        "takes an exclusive lock for the duration")
    p.add_argument("--keep-unavailable", action="store_true",
                   help="prune on score only, leaving available=FALSE rows")
    p.add_argument("--idle-minutes", type=float, default=20.0,
                   help="refuse to delete if the DB was written this recently "
                        "(default 20). Spans a searcher's commit gap; 0 "
                        "disables the check")
    p.add_argument("--force", action="store_true",
                   help="delete even though the DB is open or recently written")
    return p.parse_args()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def check_integrity(conn: sqlite3.Connection) -> str:
    """Never delete from a damaged B-tree -- hunter.main refuses to write to one
    for the same reason, and a partial DB is worse than a full one."""
    try:
        # quick_check reports one row per problem and only says "ok" when that
        # is the whole result; reading just the first row turns a page of
        # corruption into a single line.
        rows = [r[0] for r in conn.execute("PRAGMA quick_check").fetchall()]
        return "\n  ".join(rows) if rows else "empty quick_check result"
    except sqlite3.Error as e:
        return f"unreadable: {e}"


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    """molecule_replicates is created by score_store.init_variance_tables, which
    only runs once rescore/backfill has touched the DB. A search-only DB does
    not have it, and prune must still work there."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def disk_size(path: str) -> int:
    """The DB's real footprint: main file plus its WAL. Freed pages sit in -wal
    until a checkpoint, so the main file alone understates both what the DB
    costs now and what VACUUM gives back."""
    return sum(os.path.getsize(path + s)
               for s in ("", "-wal") if os.path.exists(path + s))


def idle_seconds(path: str) -> float:
    """Seconds since the DB was last written, across the whole WAL set.

    A committed write in WAL mode lands in -wal and may not touch the main file
    for a long time, so the main file's mtime on its own can be hours stale on a
    database being actively written.
    """
    newest = 0.0
    for suffix in ("", "-wal", "-shm"):
        try:
            newest = max(newest, os.stat(path + suffix).st_mtime)
        except OSError:
            continue
    return time.time() - newest if newest else float("inf")


def holders(path: str) -> List[Tuple[int, str]]:
    """Every process other than this one holding the DB open, from /proc.

    Exact about descriptors that are open right now, which is not the same as
    "a searcher is running" -- see the module docstring for why idle_seconds()
    covers the other half. Returns [(pid, command)]; an empty list on a system
    without /proc, which is why --apply also tries for an exclusive lock before
    trusting this.
    """
    target = os.path.realpath(path)
    found, me = [], os.getpid()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.path.realpath(fd) != target:
                continue
            pid = int(fd.split("/")[2])
            if pid == me:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            entry = (pid, cmd[:90])
            if entry not in found:
                found.append(entry)
        except (OSError, ValueError, IndexError):
            continue          # the process exited, or is not ours to read
    return found


def where_clause(threshold: float, keep_unavailable: bool) -> Tuple[str, list]:
    """`available` is stored as 0/1. NULL is left alone deliberately: the column
    defaults to TRUE, so a NULL is an unset field rather than a decision, and
    this script does not delete on a value nobody wrote."""
    if keep_unavailable:
        return "score < ?", [threshold]
    return "score < ? OR available = 0", [threshold]


def main(args: argparse.Namespace,
         conn_box: List[sqlite3.Connection]) -> int:
    db = args.db
    if not os.path.exists(db):
        print(f"no such DB: {db}", file=sys.stderr)
        return 2

    # Before connect(), not at the point of use: opening a WAL database creates
    # and touches its -shm, so a connection of our own resets the very clock we
    # are trying to read and every DB looks like it was written a moment ago.
    idle = idle_seconds(db)

    conn = connect(db)
    conn_box.append(conn)
    verdict = check_integrity(conn)
    if verdict != "ok":
        print(f"{db} failed its integrity check:\n  {verdict}\n"
              f"Refusing to delete from a damaged file.", file=sys.stderr)
        return 1

    where, params = where_clause(args.threshold, args.keep_unavailable)
    one = lambda sql, p=(): conn.execute(sql, p).fetchone()[0]

    total = one(f"SELECT COUNT(*) FROM {TABLE}")
    doomed = one(f"SELECT COUNT(*) FROM {TABLE} WHERE {where}", params)
    below = one(f"SELECT COUNT(*) FROM {TABLE} WHERE score < ?", [args.threshold])
    unavail = one(f"SELECT COUNT(*) FROM {TABLE} WHERE available = 0")
    null_avail = one(f"SELECT COUNT(*) FROM {TABLE} WHERE available IS NULL")
    has_reps = has_table(conn, REPLICATES)
    reps = one(f"SELECT COUNT(*) FROM {REPLICATES} WHERE molecule_name IN "
               f"(SELECT molecule_name FROM {TABLE} WHERE {where})",
               params) if has_reps else 0

    print(f"{db}")
    print(f"  rows                        {total:>9,}")
    print(f"  score < {args.threshold:<19g} {below:>9,}")
    if not args.keep_unavailable:
        print(f"  available = FALSE           {unavail:>9,}")
    print(f"  to delete (union)           {doomed:>9,}")
    if has_reps:
        print(f"  replicate draws with them   {reps:>9,}")
    print(f"  surviving                   {total - doomed:>9,}")
    if null_avail:
        print(f"  note: {null_avail:,} rows have available IS NULL — kept")

    if not doomed:
        print("\nnothing to delete.")
        return 0

    if not args.apply:
        top = conn.execute(
            f"SELECT molecule_name, score, available FROM {TABLE} "
            f"WHERE {where} ORDER BY score DESC LIMIT 5", params).fetchall()
        print("\nhighest-scoring rows that would go:")
        for name, score, avail in top:
            print(f"    {name:<28} {score:.6f}  available={avail}")
        print(f"\nreport only — nothing changed. Re-run with --apply to delete "
              f"{doomed:,} molecules.")
        return 0

    open_by = holders(db)
    if open_by and not args.force:
        print(f"\n{db} is open by another process:", file=sys.stderr)
        for pid, cmd in open_by:
            print(f"    pid {pid}: {cmd}", file=sys.stderr)
        print("Stop it first, or pass --force. Nothing changed.", file=sys.stderr)
        return 1

    # No open descriptor is not the same as no searcher: hunter's connections
    # are short-lived, so between commits it leaves nothing in /proc to find.
    if args.idle_minutes > 0 and idle < args.idle_minutes * 60:
        if not args.force:
            print(f"\n{db} was written {idle / 60:.1f} minutes ago — a searcher "
                  f"that commits in bursts looks idle between them. Stop it, "
                  f"wait out --idle-minutes ({args.idle_minutes:g}), or pass "
                  f"--force. Nothing changed.", file=sys.stderr)
            return 1
        print(f"\nwarning: {db} was written {idle / 60:.1f} minutes ago — "
              f"proceeding because --force was given")

    try:
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError as e:
        if not args.force:
            print(f"\ncannot take an exclusive lock on {db} ({e}) — something "
                  f"is writing to it. Nothing changed.", file=sys.stderr)
            return 1
    if open_by:
        print(f"\nwarning: {len(open_by)} other process(es) have {db} open — "
              f"proceeding because --force was given")

    if not args.no_backup:
        backup = f"{db}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        if os.path.exists(backup):
            print(f"\n{backup} already exists — refusing to overwrite a "
                  f"backup. Nothing changed.", file=sys.stderr)
            return 1
        # The backup API copies a consistent snapshot even with a WAL in play;
        # copying the main file alone would miss uncommitted pages. `with` on a
        # connection is a transaction, not a close, so the destination is closed
        # by hand -- an open handle here would leave the copy's own -wal beside
        # it and the backup looking like two files rather than one.
        dst = connect(backup)
        try:
            conn.backup(dst)
        finally:
            dst.close()
        print(f"\nbackup -> {backup} ({os.path.getsize(backup):,} bytes)")

    t0 = time.time()
    try:
        with conn:
            if has_reps:
                conn.execute(
                    f"DELETE FROM {REPLICATES} WHERE molecule_name IN "
                    f"(SELECT molecule_name FROM {TABLE} WHERE {where})", params)
            deleted = conn.execute(
                f"DELETE FROM {TABLE} WHERE {where}", params).rowcount
    except sqlite3.OperationalError as e:
        # Both deletes are one transaction, so a lock taken mid-way rolls the
        # whole thing back rather than leaving replicates without molecules.
        print(f"\ndelete failed ({e}) — the DB is being written to. Nothing "
              f"changed.", file=sys.stderr)
        return 1
    print(f"deleted {deleted:,} molecules"
          + (f" and {reps:,} replicate draws" if has_reps else "")
          + f" in {time.time() - t0:.1f}s")

    if args.vacuum:
        size_before = disk_size(db)
        t0 = time.time()
        conn.execute("VACUUM")
        # VACUUM rewrites the main file but leaves the old WAL sitting beside
        # it until something checkpoints, so without this the flag that exists
        # to hand space back to the filesystem hands back less than it says
        # and the number printed below is measured against a stale -wal.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(f"vacuumed {size_before:,} -> {disk_size(db):,} bytes on disk "
              f"(db + wal) in {time.time() - t0:.1f}s")

    left = one(f"SELECT COUNT(*) FROM {TABLE}")
    if has_reps:
        orphans = one(f"SELECT COUNT(*) FROM {REPLICATES} r LEFT JOIN {TABLE} s "
                      f"ON s.molecule_name = r.molecule_name "
                      f"WHERE s.molecule_name IS NULL")
        print(f"{left:,} molecules remain | orphaned replicate rows: {orphans}")
    else:
        print(f"{left:,} molecules remain")
    return 0


def _cli() -> int:
    """main() returns from a dozen places; closing here means none of them can
    leave the DB open behind a half-finished run."""
    args = parse_args()
    conn_box: List[sqlite3.Connection] = []
    try:
        return main(args, conn_box)
    finally:
        for c in conn_box:
            c.close()


if __name__ == "__main__":
    raise SystemExit(_cli())
