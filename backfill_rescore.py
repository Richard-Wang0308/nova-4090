#!/usr/bin/env python3
"""
backfill_rescore.py — confirm the score DB's existing high scorers, once.

Run this before starting continuous mining. It takes every molecule already in
the DB that is `available=TRUE` and scored above the threshold, re-scores it two
more times, replaces its score with the mean of the three draws, and flags it
`rescored=TRUE` so neither this script nor the searchers ever pay for it again.

The stored score counts as the first draw, so a molecule ends with 3 draws.

WHY BOTHER
----------
A Boltz score is a draw, not a property. Measured on this box, re-scoring the
same molecule at the same seed 68 in the same batch moved it by up to 0.0102.
Your DB's leaders are therefore partly molecules that drew well once, and that
luck does not repeat when a validator scores them. Averaging three draws strips
most of that premium out before you build a submission on it.

RESUMABLE
---------
Work is committed in chunks: each chunk's molecules are averaged and flagged as
soon as it finishes. Kill this at any time and re-run it — completed molecules
are skipped, and it picks up where it stopped.

SAFE ALONGSIDE MINERS
---------------------
BoltzWrapper hardcodes one shared `boltz_tmp_files/{inputs,outputs}` and
`predict()` scores every YAML in it, so two processes sharing it land in each
other's batches and delete each other's outputs. This script gives itself an
isolated workspace.

USAGE
-----
    python3 backfill_rescore.py --rxn-id 2 --dry-run      # what would run
    python3 backfill_rescore.py --rxn-id 2 --limit 200    # highest 200 first
    python3 backfill_rescore.py --rxn-id 2                # everything > 0.1
    python3 backfill_rescore.py --rxn-id 2 --novel-only   # skip unsubmittable
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
for _p in (BASE_DIR, os.path.join(BASE_DIR, "boltz"), os.path.join(BASE_DIR, "miner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rescore
import score_store
from config.config_loader import load_config

try:
    from boltz_wrapper import BoltzWrapper
except Exception:
    BoltzWrapper = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill")


def load_candidates(
    db_path: str,
    rxn_id: int,
    threshold: float,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """
    available=TRUE, above threshold, not already confirmed, highest first.

    Highest-first matters: this is a long job, and stopping it early should
    leave the molecules that actually matter already confirmed.
    """
    # `rescored` only exists once the variance tables have been created; a dry
    # run must work on an untouched DB, so skip that clause when it is absent.
    with score_store.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(scored_molecules)").fetchall()}
    not_done = ("AND (rescored IS NULL OR rescored = FALSE)"
                if "rescored" in cols else "")

    query = f"""
        SELECT molecule_name, smiles, score
        FROM scored_molecules
        WHERE available = TRUE
          AND score > ?
          AND smiles IS NOT NULL AND smiles != ''
          {not_done}
          AND molecule_name LIKE ?
        ORDER BY score DESC
    """
    params: List[Any] = [float(threshold), f"rxn:{rxn_id}:%"]
    if limit and limit > 0:
        query += " LIMIT ?"
        params.append(int(limit))
    with score_store.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {"name": n, "smiles": s, "boltz_score": float(sc)} for n, s, sc in rows
    ]


def drop_unsubmittable(
    candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
    db_path: Optional[str] = None,
    rxn_id: Optional[int] = None,
    scan: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Keep only molecules the validator would still accept: InChIKey not already
    in the archive AND max Tanimoto below config['max_similarity_to_historical'].

    When nothing survives, say WHY rather than just "nothing left": the usual
    cause is that score and novelty are anticorrelated for a reaction — the
    highest-scoring products are the ones every other miner also found — so the
    submittable frontier sits below --threshold. Report where it actually is.
    """
    import novelty

    target = config["small_molecule_target"][0]
    thr = novelty.config_threshold(config)
    guard = novelty.NoveltyGuard(target, thr)
    sims = guard.similarities(
        [c["smiles"] for c in candidates], progress_every=500
    )
    keep = [
        c for c, sim in zip(candidates, sims)
        if sim < thr and novelty.is_unique(c["smiles"], target)
    ]
    log.info(
        f"novelty filter (max Tanimoto to archive < {thr}, InChIKey unique): "
        f"{len(keep)}/{len(candidates)} still submittable"
    )
    if len(keep) < len(candidates):
        q = np.percentile(sims, [0, 25, 50, 75, 100])
        log.info(
            f"  candidate similarity to archive: min {q[0]:.3f} | p25 {q[1]:.3f}"
            f" | median {q[2]:.3f} | p75 {q[3]:.3f} | max {q[4]:.3f}"
        )
    if not keep and db_path is not None:
        _report_submittable_frontier(db_path, rxn_id, guard, thr, scan, target)
    return keep


def _report_submittable_frontier(
    db_path: str, rxn_id: int, guard, thr: float, scan: int, target: str
) -> None:
    """Look further down the score list for the best molecule that IS novel."""
    import novelty

    log.info(
        f"  looking for the submittable frontier in the top {scan} available "
        f"molecules (ignoring --threshold)..."
    )
    rows = load_candidates(db_path, rxn_id, threshold=-1e9, limit=scan)
    if not rows:
        return
    sims = guard.similarities([r["smiles"] for r in rows], progress_every=1000)
    novel = [
        r for r, sim in zip(rows, sims)
        if sim < thr and novelty.is_unique(r["smiles"], target)
    ]
    if not novel:
        log.warning(
            f"  none of the top {scan} available molecules are submittable. "
            f"This reaction's space looks exhausted — search a different rxn."
        )
        return
    best = novel[0]["boltz_score"]              # rows are score-ordered
    log.info(
        f"  {len(novel)}/{len(rows)} of them are submittable; the best of those "
        f"scores {best:.5f}"
    )
    log.info(
        f"  => nothing above --threshold is submittable. To confirm the "
        f"molecules you can actually submit, re-run with:\n"
        f"       --threshold {round(best - 0.005, 3)}   (or --top-submittable 40, "
        f"which ignores --threshold entirely)"
    )


def _snapshot_top(db_path: str, rxn_id: int, n: int = 20) -> List[Any]:
    """Current top-n available molecules, so the run can show what it changed."""
    with score_store.connect(db_path) as conn:
        return conn.execute(
            "SELECT molecule_name, score FROM scored_molecules "
            "WHERE available=TRUE AND molecule_name LIKE ? "
            "ORDER BY score DESC LIMIT ?",
            (f"rxn:{rxn_id}:%", n),
        ).fetchall()


def _chunk_report(
    db_path: str,
    ci: int,
    n_chunks: int,
    before: Dict[str, float],
    averaged: Dict[str, float],
    secs: float,
) -> List[tuple]:
    """
    Per-molecule breakdown of what the extra draws did, logged as each chunk
    commits. Returns the rows so the final summary can reuse them.

    The draws themselves are the point: a molecule whose three draws are
    0.11607 / 0.10353 / 0.10333 got a lucky first roll, and its single-draw
    score was never real. One whose draws are all within 0.001 is solid.
    """
    if not averaged:
        log.info(
            f"[{ci}/{n_chunks}] nothing confirmed in this chunk "
            f"({secs:.0f}s) — all below --threshold or already done"
        )
        return []

    draws = score_store.replicate_scores(db_path, list(averaged))
    rows = []
    for name in sorted(averaged, key=lambda k: -averaged[k]):
        was = before.get(name, float("nan"))
        d = draws.get(name, [])
        sigma = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
        rows.append((name, was, d, averaged[name], averaged[name] - was, sigma))

    log.info(
        f"[{ci}/{n_chunks}] {len(rows)} molecules confirmed in {secs:.0f}s\n"
        f"  {'molecule':<28}{'was':>9}{'now':>10}{'delta':>10}{'sigma':>9}   draws"
    )
    for name, was, d, now, delta, sigma in rows:
        log.info(
            f"  {name:<28}{was:9.5f}{now:10.5f}{delta:+10.5f}{sigma:9.5f}   "
            + " ".join(f"{x:.5f}" for x in d)
        )

    deltas = np.array([r[4] for r in rows], dtype=float)
    worst = rows[int(np.argmin(deltas))]
    best = rows[int(np.argmax(deltas))]
    log.info(
        f"  chunk: mean {deltas.mean():+.5f} | median {np.median(deltas):+.5f} "
        f"| {int((deltas < 0).sum())}/{len(rows)} came down | "
        f"worst {worst[4]:+.5f} ({worst[0]}) | best {best[4]:+.5f} ({best[0]})"
    )
    return rows


def _final_report(
    db_path: str,
    rxn_id: int,
    rows: List[tuple],
    top_before: List[Any],
    minutes: float,
) -> None:
    """Everything that moved, and what it did to the number that matters."""
    log.info("\n" + "=" * 74)
    log.info(f"RESCORING FINISHED — {len(rows)} molecules in {minutes:.0f}m")
    log.info("=" * 74)

    if not rows:
        log.info("  nothing was confirmed")
        return

    deltas = np.array([r[4] for r in rows], dtype=float)
    sigmas = np.array([r[5] for r in rows], dtype=float)
    sigmas = sigmas[~np.isnan(sigmas)]

    log.info(
        f"  score change   mean {deltas.mean():+.6f} | median "
        f"{np.median(deltas):+.6f} | {int((deltas < 0).sum())} down / "
        f"{int((deltas > 0).sum())} up"
    )
    log.info(
        "  |change|       "
        + " | ".join(
            f"p{q} {np.percentile(np.abs(deltas), q):.5f}" for q in (50, 90, 99)
        )
        + f" | max {np.abs(deltas).max():.5f}"
    )
    if len(sigmas):
        log.info(
            "  within-mol sd  "
            + " | ".join(f"p{q} {np.percentile(sigmas, q):.5f}" for q in (50, 90))
            + f" | max {sigmas.max():.5f}"
        )
        noisy = [r for r in rows if not np.isnan(r[5]) and r[5] >= 0.005]
        if noisy:
            log.info(
                f"  {len(noisy)} molecules have sd >= 0.005 — high variance, the "
                f"kind the validator will score differently than you did:"
            )
            for r in sorted(noisy, key=lambda r: -r[5])[:10]:
                log.info(
                    f"    {r[0]:<28}sd {r[5]:.5f}   "
                    + " ".join(f"{x:.5f}" for x in r[2])
                )

    biggest = sorted(rows, key=lambda r: r[4])[:10]
    log.info("\n  10 largest drops (single-draw scores that were not real):")
    log.info(f"    {'molecule':<28}{'was':>9}{'now':>10}{'delta':>10}")
    for r in biggest:
        log.info(f"    {r[0]:<28}{r[1]:9.5f}{r[3]:10.5f}{r[4]:+10.5f}")

    top_after = _snapshot_top(db_path, rxn_id, 20)
    confirmed = score_store.load_rescored_names(db_path, rxn_id)
    sum_b = sum(x[1] for x in top_before)
    sum_a = sum(x[1] for x in top_after)
    before_names = [x[0] for x in top_before]
    churn = sum(1 for x in top_after if x[0] not in before_names)

    log.info("\n  top 20 available, before -> after:")
    log.info(f"    {'molecule':<28}{'score':>10}  {'confirmed':>9}  note")
    for name, sc in top_after:
        was = dict(top_before).get(name)
        note = "unchanged" if was is None else f"was {was:.5f}"
        if name not in before_names:
            note = "NEW to top 20"
        log.info(
            f"    {name:<28}{sc:10.5f}  "
            f"{'yes' if name in confirmed else 'no':>9}  {note}"
        )
    log.info(
        f"\n    top-20 sum {sum_b:.5f} -> {sum_a:.5f} ({sum_a - sum_b:+.5f})"
    )
    log.info(f"    {churn}/20 molecules are new to the top 20")
    unconfirmed = [n for n, _ in top_after if n not in confirmed]
    if unconfirmed:
        log.info(
            f"    {len(unconfirmed)} of the top 20 are still single-draw — "
            f"their scores are not yet trustworthy:"
        )
        for n in unconfirmed[:10]:
            log.info(f"      {n}")


async def run(args) -> None:
    config = load_config()
    db_path = args.db or score_store.score_db_path(args.rxn_id)
    if not os.path.exists(db_path):
        raise SystemExit(f"score DB not found: {db_path}")

    # A dry run inspects only; the schema is created when work actually starts.
    if not args.dry_run:
        score_store.init_variance_tables(db_path)
    candidates = load_candidates(db_path, args.rxn_id, args.threshold, args.limit)

    already = len(score_store.load_rescored_names(db_path, args.rxn_id))
    log.info(
        f"DB={db_path}\n"
        f"  available & score > {args.threshold} & not yet confirmed : {len(candidates)}\n"
        f"  already confirmed in a previous run                      : {already}"
    )
    if not candidates and not args.top_submittable:
        log.info("nothing to do")
        return

    if args.top_submittable:
        # Rank by what can actually be submitted, not by raw score. For a
        # reaction whose high scorers are all already in the archive, this is
        # the only selection that puts GPU time on molecules that can earn.
        pool = load_candidates(db_path, args.rxn_id, -1e9, args.novel_scan)
        candidates = drop_unsubmittable(
            pool, config, db_path, args.rxn_id, args.novel_scan
        )[: args.top_submittable]
        if not candidates:
            log.info("no submittable molecules found — nothing to do")
            return
        log.info(
            f"  selected the top {len(candidates)} submittable molecules "
            f"(scores {candidates[0]['boltz_score']:.5f} .. "
            f"{candidates[-1]['boltz_score']:.5f})"
        )
    elif args.novel_only:
        candidates = drop_unsubmittable(
            candidates, config, db_path, args.rxn_id, args.novel_scan
        )
        if not candidates:
            log.info("nothing left after the novelty filter")
            return

    n_preds = len(candidates) * args.extra_rounds
    log.info(
        f"  plan: {len(candidates)} molecules x {args.extra_rounds} extra draws "
        f"= {n_preds} predictions (~{n_preds * args.sec_per_pred / 3600:.1f} h "
        f"at {args.sec_per_pred:.0f}s each), in chunks of {args.chunk}"
    )
    log.info(
        f"  score range: {candidates[0]['boltz_score']:.5f} .. "
        f"{candidates[-1]['boltz_score']:.5f}"
    )

    if args.dry_run:
        log.info("dry run — nothing scored. Top 10 that would be confirmed:")
        for c in candidates[:10]:
            log.info(f"    {c['name']:<28}{c['boltz_score']:9.5f}")
        return

    if BoltzWrapper is None:
        raise SystemExit(
            "BoltzWrapper unavailable — run from the nova-4090 root in the same "
            "environment miner/miner.py uses."
        )

    boltz = BoltzWrapper()
    rescore.isolate_boltz_workspace(boltz, tag=f"backfill{os.getpid()}")

    target_key, target_label = score_store.target_identity(config)
    chunks = [
        candidates[i:i + args.chunk] for i in range(0, len(candidates), args.chunk)
    ]

    top_before = _snapshot_top(db_path, args.rxn_id, 20)

    t_start = time.time()
    done = 0
    all_rows: List[tuple] = []

    for ci, chunk in enumerate(chunks, start=1):
        t0 = time.time()
        before = {c["name"]: c["boltz_score"] for c in chunk}
        averaged = await rescore.confirm_high_scorers(
            boltz=boltz,
            config=config,
            scored=chunk,
            threshold=args.threshold,
            extra_rounds=args.extra_rounds,
            batch_size=args.batch_size,
            db_path=db_path,
            rxn_id=args.rxn_id,
            round_no=args.round_no,
            target_key=target_key,
            target_label=target_label,
            source=args.source,
            logger=log,
        )
        done += len(chunk)

        if args.detail:
            all_rows.extend(
                _chunk_report(
                    db_path, ci, len(chunks), before, averaged, time.time() - t0
                )
            )
        else:
            draws = score_store.replicate_scores(db_path, list(averaged))
            all_rows.extend(
                (
                    n,
                    before.get(n, float("nan")),
                    draws.get(n, []),
                    averaged[n],
                    averaged[n] - before.get(n, float("nan")),
                    float(np.std(draws[n], ddof=1))
                    if len(draws.get(n, [])) > 1 else float("nan"),
                )
                for n in averaged
            )

        elapsed = time.time() - t_start
        rate = elapsed / max(done, 1)
        log.info(
            f"  progress {done}/{len(candidates)} molecules | "
            f"elapsed {elapsed / 60:.0f}m | "
            f"eta {(len(candidates) - done) * rate / 60:.0f}m"
        )

    _final_report(
        db_path, args.rxn_id, all_rows, top_before, (time.time() - t_start) / 60
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rxn-id", type=int, required=True)
    p.add_argument("--db", type=str, default=None, help="score DB path override")
    p.add_argument("--threshold", type=float, default=rescore.DEFAULT_SCORE_THRESHOLD,
                   help="confirm molecules scoring above this (default 0.1)")
    p.add_argument("--extra-rounds", type=int, default=rescore.DEFAULT_EXTRA_ROUNDS,
                   help="extra draws per molecule (default 2, giving 3 total)")
    p.add_argument("--batch-size", type=int, default=rescore.DEFAULT_BATCH_SIZE)
    p.add_argument("--chunk", type=int, default=50,
                   help="molecules committed per chunk; smaller = finer resume "
                        "granularity (default 50)")
    p.add_argument("--limit", type=int, default=0,
                   help="only the highest-scoring N (0 = all)")
    p.add_argument("--top-submittable", type=int, default=0, metavar="N",
                   help="ignore --threshold: confirm the N highest-scoring "
                        "molecules that still pass the validator's similarity "
                        "rule. Use this when the high scorers are all already "
                        "archived")
    p.add_argument("--novel-scan", type=int, default=4000, metavar="N",
                   help="how deep down the score list to look for submittable "
                        "molecules (default 4000)")
    p.add_argument("--novel-only", action="store_true",
                   help="skip molecules that already fail the validator's "
                        "similarity rule — they cannot be submitted, so "
                        "confirming them is wasted GPU")
    p.add_argument("--round-no", type=int, default=None,
                   help="value written to the iteration/round columns; "
                        "omitted means preserve each molecule's existing round")
    p.add_argument("--no-detail", dest="detail", action="store_false",
                   help="suppress the per-molecule table after each chunk; the "
                        "final summary is printed either way")
    p.set_defaults(detail=True)
    p.add_argument("--source", type=str, default="backfill_confirmed")
    p.add_argument("--sec-per-pred", type=float, default=16.0,
                   help="only used for the ETA estimate")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
