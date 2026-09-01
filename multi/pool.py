"""
multi/pool.py — a persistent pool of single-GPU Boltz workers.

    pool = BoltzPool()                       # auto-detects GPUs
    got  = pool.score(molecules, subnet_config)
    pool.shutdown()

`molecules` is [(name, smiles), ...]; `got` is {smiles: {target: score}} plus
`pool.last_components` for the raw per-metric dicts.

CHUNK SIZING
------------
Per-call setup measured at 22 s (12 s manifest + structure checkpoint, 10 s
affinity checkpoint) against ~7.8 s of GPU per molecule. Setup is paid once per
chunk, so the overhead of a chunk of size c is 22 / (22 + 7.8c):

    c = 10  ->  22% wasted     (what --batch-size 10 does today)
    c = 20  ->  12%
    c = 30  ->  8.6%
    c = 48  ->  5.6%

Bigger is better for throughput and worse for load balance, since one straggler
chunk delays the whole call. The pool aims for ~2 chunks per worker, clamped to
[MIN_CHUNK, MAX_CHUNK], which keeps overhead near 10% while leaving a second
chunk per worker to absorb the spread in molecule size.

This is also why a pool call should be given the WHOLE round's molecules rather
than being driven in batches of ten: with 150 molecules and 2 workers it forms
4 chunks of ~38, not 15 chunks of 10.
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import topology
from ._log import ensure_visible, get_logger
from .worker import worker_main

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MIN_CHUNK = 8
MAX_CHUNK = 48
CHUNKS_PER_WORKER = 2
# A chunk of MAX_CHUNK at ~8 s/molecule is ~7 min; 45 min is generous enough
# that only a genuinely wedged worker trips it.
CHUNK_TIMEOUT_S = 2700.0
WORKER_START_TIMEOUT_S = 600.0

log = get_logger("multi.pool")


@contextlib.contextmanager
def _no_main_fixup():
    """Stop multiprocessing from re-importing the parent's __main__ in workers.

    multiprocessing.spawn.get_preparation_data() tells the child to import the
    parent's main module -- by name if __main__.__spec__ is set, otherwise by
    path from __main__.__file__. Neither is wanted here: the worker target is
    multi.worker.worker_main, a plain module-level function the child can import
    on its own, and re-importing the parent's main module either costs a full
    rdkit+sklearn import per worker (when it is hunter.py) or kills the worker
    outright (when it is "<stdin>", or a runpy-executed path).

    Swapping in a bare module for the duration of start() leaves both __spec__
    and __file__ unset, so get_preparation_data records neither and the child
    imports nothing.
    """
    real = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if real is not None:
            sys.modules["__main__"] = real
        else:
            sys.modules.pop("__main__", None)


def _start_context(logger):
    """forkserver if we can get it, otherwise spawn. Never plain fork.

    fork is unsafe here: the parent may be hunter.py, which holds a fitted
    sklearn ensemble, an asyncio loop and a ThreadPoolExecutor (its Boltz call
    goes through run_in_executor). Forking a process with live threads inherits
    their locks in whatever state they happened to be in.

    Both spawn AND forkserver re-import the parent's __main__ in the child --
    multiprocessing.spawn.prepare() calls _fixup_main_from_path() either way.
    That is measured, not assumed: with __main__ set to "<stdin>" every worker
    died on

        FileNotFoundError: .../nova-4090/<stdin>

    and under multi/run.py it would re-import the searcher being launched. The
    fix is _no_main_fixup() below, which is applied around every Process.start().

    What forkserver still buys over spawn is the fork source: its server is
    launched fresh and stays single-threaded, so workers fork from a clean
    process rather than inheriting the parent's interpreter state.
    """
    try:
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["multi.worker"])
        return ctx
    except Exception as e:                      # not on Linux, or preload failed
        logger.warning("multi: forkserver unavailable (%s); using spawn", e)
        return mp.get_context("spawn")


def choose_chunk_size(n: int, n_workers: int) -> int:
    """Chunk size for n molecules across n_workers.

    Two regimes. With plenty of work, aim for CHUNKS_PER_WORKER chunks each so a
    slow chunk can be absorbed, but never below MIN_CHUNK -- below that the 22 s
    setup dominates. With little work, MIN_CHUNK would put everything in one
    chunk and leave every other worker idle, so fall back to spreading n across
    the workers even though each chunk then carries its own setup.

        n=150, W=2  -> 38   (4 chunks, 2 per worker)
        n=150, W=8  -> 19   (8 chunks, 1 per worker)
        n=10,  W=2  ->  5   (2 chunks; 61 s wall instead of 100 s on one worker)
        n=10,  W=8  ->  2

    The n=10 row is what an unmodified hunter.py --batch-size 10 produces. It
    still wins, but it wastes setup: see README, "raise --batch-size".
    """
    if n <= 0 or n_workers <= 0:
        return MIN_CHUNK
    ceil_div = lambda a, b: (a + b - 1) // b
    target = max(1, ceil_div(n, n_workers * CHUNKS_PER_WORKER))
    if target < MIN_CHUNK:
        target = max(1, ceil_div(n, n_workers))
    return max(1, min(MAX_CHUNK, target))


class BoltzPool:
    """Persistent workers. Start once, score many times, shut down at exit."""

    def __init__(self, workers_per_gpu: Optional[int] = None,
                 plan: Optional[List[int]] = None, logger=None):
        self.log = logger or log
        self.plan = plan if plan is not None else topology.plan_workers(workers_per_gpu)
        self.ctx = _start_context(self.log)
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.ready_q = self.ctx.Queue()
        self.procs: Dict[int, Any] = {}
        self._started = False
        self.last_components: Dict[str, Dict[str, dict]] = {}
        atexit.register(self.shutdown)

    # -- lifecycle ---------------------------------------------------------

    @property
    def n_workers(self) -> int:
        return len(self.plan)

    def start(self) -> None:
        if self._started:
            return
        # Re-assert here too: bittensor may have been imported (and disabled
        # these loggers) after this module was first imported.
        ensure_visible()
        self.log.info("multi: starting %s", topology.describe(self.plan))
        for wid, gpu in enumerate(self.plan):
            self._spawn(wid, gpu)
        # Wait for each worker to report that its BoltzWrapper constructed.
        # A worker that cannot import boltz should fail here, once, rather than
        # on every chunk.
        deadline = time.time() + WORKER_START_TIMEOUT_S
        alive = 0
        for _ in range(len(self.plan)):
            try:
                msg = self.ready_q.get(timeout=max(1.0, deadline - time.time()))
            except queue.Empty:
                break
            if msg.get("ok"):
                alive += 1
                self.log.info("multi: worker %d ready on GPU %d (pid %s)",
                              msg["worker_id"], msg["gpu"], msg.get("pid"))
            else:
                self.log.error("multi: worker %d failed to start: %s",
                               msg.get("worker_id"), msg.get("error"))
                self.log.debug("%s", msg.get("tb", ""))
        if alive == 0:
            raise RuntimeError(
                "multi: no Boltz worker could start. Run "
                "`python3 -m multi.selftest` for the underlying import error.")
        if alive < len(self.plan):
            self.log.warning("multi: %d of %d workers started; continuing on those",
                             alive, len(self.plan))
        self._started = True

    def _spawn(self, wid: int, gpu: int) -> None:
        p = self.ctx.Process(
            target=worker_main,
            args=(gpu, wid, BASE_DIR, self.task_q, self.result_q, self.ready_q),
            daemon=False,        # boltz's DataLoader needs to fork children
            name=f"boltz-w{wid}-gpu{gpu}",
        )
        with _no_main_fixup():
            p.start()
        self.procs[wid] = p

    def shutdown(self) -> None:
        if not self._started:
            return
        self._started = False
        for _ in self.procs:
            try:
                self.task_q.put(None)
            except Exception:
                pass
        for p in self.procs.values():
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join(timeout=10)
        self.procs.clear()

    # -- scoring -----------------------------------------------------------

    def score(self, molecules: Sequence[Tuple[str, str]],
              subnet_config: Dict[str, Any],
              on_chunk=None) -> Dict[str, Dict[str, float]]:
        """Score every (name, smiles). Returns {smiles: {target: score}}.

        Molecules that no worker managed to score are simply absent from the
        result; the caller applies its own sentinel, exactly as it does today
        when BoltzWrapper reports missing metrics.

        `on_chunk(scores, components)` is called as each chunk lands, so a
        caller can persist partial results rather than losing a whole round to
        a crash near the end.
        """
        self.start()
        # Deduplicate: two uids can submit the same molecule, and scoring it
        # twice would double the cost and give two different answers.
        seen, uniq = set(), []
        for name, smi in molecules:
            if not smi or smi in seen:
                continue
            seen.add(smi)
            uniq.append((name, smi))
        if not uniq:
            self.last_components = {}
            return {}

        n_workers = max(1, len(self.procs))
        chunk = choose_chunk_size(len(uniq), n_workers)
        chunks = [uniq[i:i + chunk] for i in range(0, len(uniq), chunk)]
        self.log.info("multi: %d molecules -> %d chunk(s) of <=%d across %d worker(s)",
                      len(uniq), len(chunks), chunk, n_workers)

        pending: Dict[int, List[Tuple[str, str]]] = {}
        for cid, c in enumerate(chunks):
            pending[cid] = c
            self.task_q.put({"chunk_id": cid, "molecules": c,
                             "subnet_config": subnet_config})

        scores: Dict[str, Dict[str, float]] = {}
        components: Dict[str, Dict[str, dict]] = {}
        retried: set = set()
        t0 = time.time()

        while pending:
            try:
                r = self.result_q.get(timeout=CHUNK_TIMEOUT_S)
            except queue.Empty:
                self.log.error("multi: no chunk completed in %.0fs; abandoning %d "
                               "chunk(s)", CHUNK_TIMEOUT_S, len(pending))
                break
            cid = r["chunk_id"]
            if r.get("ok"):
                scores.update(r["scores"])
                components.update(r["components"])
                pending.pop(cid, None)
                self.log.info("multi:   chunk %d done on GPU %d: %d/%d scored | %.0fs",
                              cid, r["gpu"], r["n_out"], r["n_in"], r["elapsed_s"])
                if on_chunk is not None:
                    try:
                        on_chunk(r["scores"], r["components"])
                    except Exception as e:
                        self.log.error("multi: on_chunk callback failed: %s", e)
            else:
                self.log.error("multi:   chunk %d FAILED on GPU %d: %s",
                               cid, r.get("gpu"), r.get("error"))
                self.log.debug("%s", r.get("tb", ""))
                if cid not in retried:
                    # Once. A second failure is nearly always the same failure,
                    # and the molecules are worth less than the wall clock.
                    retried.add(cid)
                    self.task_q.put({"chunk_id": cid, "molecules": pending[cid],
                                     "subnet_config": subnet_config})
                else:
                    pending.pop(cid, None)
            self._reap_dead()

        self.last_components = components
        self.log.info("multi: %d/%d molecules scored in %.0fs (%.1f s/molecule)",
                      len(scores), len(uniq), time.time() - t0,
                      (time.time() - t0) / max(len(scores), 1))
        return scores

    def _reap_dead(self) -> None:
        """Replace workers that died, so a crash costs one chunk and not the pool."""
        for wid, p in list(self.procs.items()):
            if p.is_alive():
                continue
            code = p.exitcode
            self.procs.pop(wid, None)
            gpu = self.plan[wid] if wid < len(self.plan) else 0
            self.log.warning("multi: worker %d (GPU %d) exited with %s; restarting",
                             wid, gpu, code)
            self._spawn(wid, gpu)
            try:
                self.ready_q.get(timeout=WORKER_START_TIMEOUT_S)
            except queue.Empty:
                self.log.error("multi: replacement worker %d did not report ready", wid)
