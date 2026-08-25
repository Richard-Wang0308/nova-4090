#!/usr/bin/env python3
"""
rescore.py — confirm high scorers with extra Boltz draws and keep the average.

WHY
---
A Boltz score is a draw, not a property of the molecule: `boltz.main.predict`
seeds once and then consumes the RNG sequentially as records stream through a
dataloader fed by `glob("*")`, so the noise a molecule gets depends on the seed
and on which molecules share its run. Measured here, the same molecule re-scored
at the SAME seed moved by up to 0.0127 (9%).

That makes a single high draw untrustworthy exactly where it matters: a molecule
enters your top-20 partly because its one draw was lucky, and the validator's
independent draw does not repeat that luck.

So after a round finishes, every molecule that cleared `threshold` is scored
`extra_rounds` more times and its stored score becomes the mean of all draws.
Averaging n draws cuts the noise on the estimate by sqrt(n) and strips most of
the selection premium out of the molecules that would otherwise lead your table.

Only molecules above the threshold are confirmed, because they are the only ones
that can reach a submission; the rest keep their single draw and cost nothing.
Once a molecule has TOTAL_DRAWS draws on record it is flagged `rescored` in the
score DB and skipped by every later round, so confirmation is paid once.

SEED
----
Every pass runs at CONFIRM_SEED (68), the seed the searchers already score with,
so a confirmation draw is taken under the same conditions as the original rather
than a synthetic one. Note what this does and does not vary: the seed is fixed
and the confirmation batches are rebuilt from the >threshold subset, so a
confirmation draw differs from the round's original draw through batch
composition (a different set of molecules shares the run). Two confirmation
passes over the *same* batch differ only by GPU non-determinism, which is much
smaller. If the extra passes ever come back identical, that is why -- switch to
distinct seeds, or reshuffle between passes, to recover independence.

USAGE (from a searcher's round loop, after scoring finished)
-----------------------------------------------------------
    averaged = await rescore.confirm_high_scorers(
        boltz=boltz, config=config, scored=scored_molecules,
        db_path=SCORE_RESULTS_DB, rxn_id=RXN_ID, round_no=iteration,
        target_key=TARGET_KEY, target_label=TARGET_LABEL, source="dpex_dja",
        logger=logger,
    )
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import score_store

# A molecule must clear this to be worth confirming. Anything below cannot
# realistically reach a 20-molecule submission, so spending GPU on it is waste.
DEFAULT_SCORE_THRESHOLD = 0.1
# Extra draws. With the round's own draw that makes TOTAL_DRAWS in total.
DEFAULT_EXTRA_ROUNDS = 2
# Total draws a molecule needs before it is marked `rescored` and never
# re-scored again.
TOTAL_DRAWS = 1 + DEFAULT_EXTRA_ROUNDS
# Boltz batch size, matching what the searchers already use.
DEFAULT_BATCH_SIZE = 10
# Every confirmation pass runs at this seed -- the same one the searchers score
# with -- so a confirmation draw is taken under the same conditions as the
# original rather than a synthetic one.
CONFIRM_SEED = 68

_log = logging.getLogger("rescore")


def isolate_boltz_workspace(boltz, tag: Optional[str] = None) -> str:
    """
    Give this process its own Boltz input/output directories.

    BoltzWrapper hardcodes `boltz/boltz_tmp_files/{inputs,outputs}`, and
    `predict(data=input_dir)` scores EVERY yaml sitting in that directory. Two
    searchers running at once therefore land in each other's batches — which,
    per the module docstring, changes the noise each molecule draws — and
    `_cleanup_files` from one deletes the other's predictions mid-run.

    Isolating costs nothing and makes replicate measurements trustworthy while
    your miners keep running.
    """
    tag = tag or f"pid{os.getpid()}"
    root = os.path.join(boltz.tmp_dir, f"iso_{tag}")
    boltz.input_dir = os.path.join(root, "inputs")
    boltz.output_dir = os.path.join(root, "outputs")
    os.makedirs(boltz.input_dir, exist_ok=True)
    os.makedirs(boltz.output_dir, exist_ok=True)
    print(f"[isolation] Boltz workspace -> {root}")
    return root


ISOLATION_PREFIX = "iso_"


def _is_isolated(path: Optional[str]) -> bool:
    """True when `path` sits inside a directory created by isolate_boltz_workspace."""
    if not path:
        return False
    return any(
        part.startswith(ISOLATION_PREFIX)
        for part in os.path.abspath(path).split(os.sep)
    )


def clear_boltz_workspace(boltz) -> None:
    """
    Remove leftover YAML inputs and predictions before a pass.

    `predict(data=input_dir)` scores EVERY yaml in that directory, and the
    wrapper never deletes the ones it wrote. Normally `override: false` hides
    this, because Boltz skips records whose prediction directory already exists.
    Confirmation forces `override: true` (it must, or a repeat draw would be a
    cache hit), which re-enables them all: batch 2 would re-score batch 1, batch
    3 would re-score both, and the cost would grow quadratically.
    """
    # Refuse to delete anything outside an isolated workspace. Callers are
    # supposed to isolate first; if that ever regresses, the failure mode here
    # would be deleting a live searcher's inputs, so make it structurally
    # impossible rather than relying on call order.
    if not _is_isolated(boltz.input_dir) or not _is_isolated(boltz.output_dir):
        _log.warning(
            "clear_boltz_workspace called on a shared directory "
            f"({boltz.input_dir}); refusing to delete. Isolate first."
        )
        return

    shutil.rmtree(os.path.join(boltz.output_dir, "boltz_results_inputs"), ignore_errors=True)
    try:
        for f in os.listdir(boltz.input_dir):
            if f.endswith(".yaml"):
                os.remove(os.path.join(boltz.input_dir, f))
    except FileNotFoundError:
        pass
    os.makedirs(boltz.input_dir, exist_ok=True)
    os.makedirs(boltz.output_dir, exist_ok=True)


def subnet_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of config BoltzWrapper.score_molecules actually reads."""
    return {
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


def _score_sync(
    boltz,
    config: Dict[str, Any],
    molecules: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """One blocking Boltz pass. Returns ({name: score}, {name: components})."""
    valid_molecules_by_uid = {
        0: {
            "smiles": [m["smiles"] for m in molecules],
            "names": [m["name"] for m in molecules],
        }
    }
    score_dict = {
        0: {
            "target_scores": [[]], "antitarget_scores": [[]],
            "entropy": None, "entropy_boltz": None,
            "block_submitted": None, "push_time": "",
        }
    }
    boltz.score_molecules(valid_molecules_by_uid, score_dict, subnet_config(config))

    target = config["small_molecule_target"][0]
    score_map = getattr(boltz, "final_boltz_scores", {}).get(0, {}).get(target, {})
    comp_map = getattr(boltz, "per_molecule_components", {}).get(0, {})

    scores: Dict[str, float] = {}
    comps: Dict[str, Dict[str, Any]] = {}
    for m in molecules:
        v = score_map.get(m["smiles"])
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        scores[m["name"]] = v
        comps[m["name"]] = comp_map.get(m["smiles"], {}).get(target, {}) or {}
    return scores, comps


async def score_pass(
    boltz,
    config: Dict[str, Any],
    molecules: Sequence[Dict[str, Any]],
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Score `molecules` at `seed`, in batches of `batch_size`.

    `override` is forced on for the duration: with it off Boltz reuses an
    existing prediction directory, so a repeat draw would return the identical
    number and averaging would measure nothing. The wrapper's seed and override
    are restored afterwards so the caller's next round is unaffected.
    """
    log = logger or _log
    if not molecules:
        return {}, {}

    prev_seed = getattr(boltz, "base_seed", None)
    prev_override = boltz.config.get("override") if hasattr(boltz, "config") else None

    scores: Dict[str, float] = {}
    comps: Dict[str, Dict[str, Any]] = {}
    try:
        boltz.base_seed = seed
        if hasattr(boltz, "config"):
            boltz.config["override"] = True

        n_batches = (len(molecules) + batch_size - 1) // batch_size
        loop = asyncio.get_event_loop()
        for b in range(n_batches):
            batch = list(molecules[b * batch_size:(b + 1) * batch_size])
            if not batch:
                continue
            t0 = time.time()
            clear_boltz_workspace(boltz)
            s, c = await loop.run_in_executor(
                None, _score_sync, boltz, config, batch
            )
            scores.update(s)
            comps.update(c)
            log.info(
                f"   [confirm seed={seed}] batch {b + 1}/{n_batches}: "
                f"{len(s)}/{len(batch)} scored | {time.time() - t0:.1f}s"
            )
    finally:
        if prev_seed is not None:
            boltz.base_seed = prev_seed
        if prev_override is not None and hasattr(boltz, "config"):
            boltz.config["override"] = prev_override

    return scores, comps


async def confirm_high_scorers(
    boltz,
    config: Dict[str, Any],
    scored: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Non-raising entry point — see _confirm_high_scorers for the real work.

    Confirmation is an optimisation layered onto a searcher's round loop, and
    every caller invokes it unguarded. A transient Boltz failure (CUDA hiccup,
    OOM, full disk) must therefore degrade to "scores stay as single draws",
    never take down a miner that has been running for days. Any partial work is
    already committed: draws are written to molecule_replicates as each pass
    lands, so the next round resumes from them.
    """
    log = kwargs.get("logger") or _log
    try:
        return await _confirm_high_scorers(boltz, config, scored, **kwargs)
    except asyncio.CancelledError:
        raise                      # never swallow cooperative cancellation
    except Exception as e:
        log.error(
            f"[CONFIRM] confirmation failed ({type(e).__name__}: {e}); "
            f"leaving this round's scores as single draws",
            exc_info=True,
        )
        return {}


async def _confirm_high_scorers(
    boltz,
    config: Dict[str, Any],
    scored: Sequence[Dict[str, Any]],
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    extra_rounds: int = DEFAULT_EXTRA_ROUNDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    db_path: Optional[str] = None,
    rxn_id: Optional[int] = None,
    round_no: Optional[int] = 0,
    target_key: Optional[str] = None,
    target_label: Optional[str] = None,
    source: str = "confirmed",
    seed: int = CONFIRM_SEED,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Re-score every molecule above `threshold` until it has `extra_rounds` + 1
    draws, then replace its score with the mean of those draws and flag it
    `rescored` so no future round pays for it again.

    Molecules already flagged `rescored` are skipped up front. A molecule that
    was partially confirmed (a crash mid-round) is topped up to the full count
    rather than restarted.

    `scored` items need 'name', 'smiles' and a score under 'boltz_score' or
    'score'. Returns {name: averaged_score} for the molecules confirmed in this
    call. Each item in `scored` is updated in place, and the DB row is rewritten
    to match, so downstream pools, surrogates and submit.py all see the average.
    """
    log = logger or _log
    if boltz is None or not scored or extra_rounds < 1:
        return {}

    total_draws = 1 + extra_rounds

    # Index the round's results, keeping the best draw per name if duplicated.
    first: Dict[str, float] = {}
    smiles: Dict[str, str] = {}
    for m in scored:
        name = m.get("name")
        s = m.get("boltz_score", m.get("score"))
        if not name or s is None or not m.get("smiles"):
            continue
        try:
            s = float(s)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(s):
            continue
        if name not in first or s > first[name]:
            first[name] = s
            smiles[name] = m["smiles"]

    above = [n for n, s in first.items() if s > threshold]
    if not above:
        log.info(
            f"[CONFIRM] no molecules above {threshold} in this round "
            f"({len(first)} scored) — nothing to confirm"
        )
        return {}

    # Never re-confirm something already finished.
    done: set = set()
    prior: Dict[str, List[float]] = {}
    if db_path:
        try:
            score_store.init_variance_tables(db_path)
            done = score_store.load_rescored_names(db_path, rxn_id)
            prior = score_store.replicate_scores(db_path, above)
        except Exception as e:
            log.debug(f"[CONFIRM] could not read confirmation state: {e}")

    targets = [n for n in above if n not in done]
    n_skipped = len(above) - len(targets)
    if n_skipped:
        log.info(f"[CONFIRM] skipping {n_skipped} already-rescored molecule(s)")
    if not targets:
        log.info("[CONFIRM] every high scorer is already confirmed")
        return {}

    # Seed each molecule's list with any draws already on record, then append
    # this round's, so an interrupted run resumes instead of restarting and the
    # average covers every draw rather than just this round's.
    #
    # Only append when this round's value is genuinely a NEW measurement. A
    # backfill reads `first[n]` back out of scored_molecules, so on resume it is
    # bit-identical to a draw already in `prior`; appending it again would
    # double-count that draw AND let the molecule reach total_draws without ever
    # being re-scored. A real Boltz draw is never bit-identical to a prior one.
    draws: Dict[str, List[float]] = {}
    is_new_draw: Dict[str, bool] = {}
    for n in targets:
        seen = list(prior.get(n, ()))
        fresh = not any(v == first[n] for v in seen)
        is_new_draw[n] = fresh
        draws[n] = seen + ([first[n]] if fresh else [])

    n_resumed = sum(1 for n in targets if prior.get(n))
    if n_resumed:
        log.info(
            f"[CONFIRM] {n_resumed} molecule(s) already have draws on record; "
            f"topping up to {total_draws} rather than restarting"
        )

    draw_idx: Dict[str, int] = {n: len(prior.get(n, ())) for n in targets}
    if db_path:
        try:
            score_store.record_replicates(
                db_path,
                [
                    {"name": n, "seed": seed, "draw_idx": draw_idx[n],
                     "score": first[n]}
                    for n in targets if is_new_draw[n]
                ],
            )
        except Exception as e:
            log.debug(f"[CONFIRM] could not record the round's draw: {e}")
    for n in targets:
        if is_new_draw[n]:
            draw_idx[n] += 1

    log.info(
        f"[CONFIRM] {len(targets)}/{len(first)} molecules scored > {threshold}; "
        f"topping each up to {total_draws} draws at seed {seed}, "
        f"in batches of {batch_size}"
    )

    # Confirmation runs in its own workspace. BoltzWrapper's directories are
    # shared by every process on this box, and clearing between batches (see
    # clear_boltz_workspace) must never delete a running searcher's inputs.
    #
    # Isolation is therefore mandatory, not best-effort: if it fails we abort
    # rather than fall back to the shared directories, because the fallback
    # would wipe a concurrent miner's YAMLs mid-run. Losing one round of
    # confirmation is cheap; corrupting another process's scoring is not.
    prev_in = getattr(boltz, "input_dir", None)
    prev_out = getattr(boltz, "output_dir", None)
    t_all = time.time()
    try:
        isolate_boltz_workspace(boltz, tag=f"confirm{os.getpid()}")
    except Exception as e:
        log.error(
            f"[CONFIRM] could not isolate a Boltz workspace ({e}); skipping "
            f"confirmation this round rather than risk a shared directory"
        )
        return {}

    try:
        for i in range(1, extra_rounds + 1):
            # Only molecules still short of the target count.
            pending = [n for n in targets if len(draws[n]) < total_draws]
            if not pending:
                break
            mols = [{"name": n, "smiles": smiles[n]} for n in pending]
            t0 = time.time()
            got, comps = await score_pass(
                boltz, config, mols, seed, batch_size=batch_size, logger=log
            )
            rows = []
            for n, v in got.items():
                draws[n].append(v)
                rows.append(dict({"name": n, "seed": seed, "draw_idx": draw_idx[n],
                                  "score": v}, **(comps.get(n) or {})))
                draw_idx[n] += 1
            if db_path and rows:
                try:
                    score_store.record_replicates(db_path, rows)
                except Exception as e:
                    log.debug(f"[CONFIRM] could not record replicates: {e}")
            log.info(
                f"[CONFIRM] pass {i}/{extra_rounds}: {len(got)}/{len(mols)} "
                f"re-scored | {time.time() - t0:.1f}s"
            )
    finally:
        # Hand the searcher back its own directories even if a pass raised;
        # otherwise it would keep scoring into the confirmation workspace.
        if prev_in is not None:
            boltz.input_dir = prev_in
        if prev_out is not None:
            boltz.output_dir = prev_out

    # Average whatever draws each molecule actually produced.
    averaged: Dict[str, float] = {}
    finished: List[str] = []
    moved: List[Tuple[str, float, float, int]] = []
    for n, vals in draws.items():
        if len(vals) < 2:
            continue  # every confirmation pass failed for this one; keep as-is
        mean = float(np.mean(vals))
        averaged[n] = mean
        moved.append((n, first[n], mean, len(vals)))
        if len(vals) >= total_draws:
            finished.append(n)

    if not averaged:
        log.warning("[CONFIRM] no molecule produced a usable extra draw")
        return {}

    # Update the caller's own records in place.
    for m in scored:
        n = m.get("name")
        if n in averaged:
            m["boltz_score"] = averaged[n]
            m["score"] = averaged[n]
            m["boltz_score_source"] = "confirmed_mean"

    # Rewrite the DB rows so submit.py and every surrogate see the average,
    # then flag the finished ones so no future round re-scores them.
    if db_path:
        try:
            score_store.write_scores_to_db(
                db_path,
                [
                    {"name": n, "smiles": smiles[n], "boltz_score": v,
                     "source": source}
                    for n, v in averaged.items()
                ],
                rxn_id=rxn_id,
                round_no=round_no,
                target_key=target_key,
                target_label=target_label,
                source=source,
            )
            if finished:
                score_store.mark_rescored(db_path, finished)
        except Exception as e:
            log.error(f"[CONFIRM] could not write averaged scores: {e}")

    deltas = np.array([m[2] - m[1] for m in moved], dtype=float)
    log.info(
        f"[CONFIRM] averaged {len(averaged)} molecules over "
        f"{time.time() - t_all:.1f}s | mean change {deltas.mean():+.6f} | "
        f"{int((deltas < 0).sum())}/{len(deltas)} came down | "
        f"{len(finished)} marked rescored"
    )
    for n, a, b, k in sorted(moved, key=lambda x: -x[2])[:10]:
        log.info(f"   {n:<28} {a:.6f} -> {b:.6f}  ({b - a:+.6f}, n={k})")

    return averaged
