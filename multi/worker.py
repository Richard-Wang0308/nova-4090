"""
multi/worker.py — one persistent Boltz process, pinned to one GPU.

Runs in a spawned child. The parent puts (chunk_id, molecules, subnet_config)
on a shared task queue; every worker pulls from that same queue, so a worker
that finishes early takes the next chunk instead of idling. That is the whole
load-balancing strategy, and it is why chunks are not pre-assigned: molecules
differ in size by 3-4x, so any static split leaves a straggler.

WHY spawn AND NOT fork
----------------------
The parent may be hunter.py, which holds a fitted sklearn ensemble, an asyncio
loop and a ThreadPoolExecutor (its Boltz call runs in run_in_executor). Forking
a process with live threads inherits their locks in whatever state they were in,
and a child that then takes the same lock deadlocks. spawn costs ~10 s of
interpreter and CUDA start-up per worker, paid once for the life of the pool
rather than once per batch.

WHY THE WORKER OWNS A BoltzWrapper RATHER THAN CALLING predict DIRECTLY
----------------------------------------------------------------------
boltz.main.predict(devices=N) does exist and drives Lightning's DDPStrategy, but
it re-forms a process group on every call, needs a free rendezvous port, and
gives back no per-molecule handle -- the caller still has to walk the output
tree. Worse for us, it makes the *number of molecules* the unit of parallelism
within one call, so the 22 s of per-call setup is still paid once per call on
every rank. Running N independent single-GPU wrappers pays that setup on N
different clocks and overlaps it with other workers' compute.

It also means the scoring maths is never reimplemented here: the worker calls
the real BoltzWrapper and hands back what it produced.
"""
from __future__ import annotations

import os
import sys
import time
import traceback


def _isolate(boltz, tag: str) -> str:
    """Give this worker its own input/output tree.

    BoltzWrapper hardcodes boltz/boltz_tmp_files/{inputs,outputs}, and
    predict(data=input_dir) scores EVERY yaml in that directory. Without this,
    workers would score each other's molecules and _cleanup_files from one would
    delete another's predictions mid-run. The `iso_` prefix is what
    rescore.clear_boltz_workspace checks before it will delete anything, so the
    name is load-bearing, not cosmetic.
    """
    root = os.path.join(boltz.tmp_dir, f"iso_{tag}")
    boltz.input_dir = os.path.join(root, "inputs")
    boltz.output_dir = os.path.join(root, "outputs")
    os.makedirs(boltz.input_dir, exist_ok=True)
    os.makedirs(boltz.output_dir, exist_ok=True)
    return root


def _clear(boltz) -> None:
    """Drop this chunk's yamls and predictions before the next one.

    predict() scores every yaml present, and the wrapper never removes the ones
    it wrote. With override:false boltz skips records whose prediction directory
    already exists, so the cost of not clearing is silent and quadratic in
    manifest-checking time rather than a wrong answer -- but it is still real.
    """
    import glob
    import shutil
    shutil.rmtree(os.path.join(boltz.output_dir, "boltz_results_inputs"),
                  ignore_errors=True)
    try:
        for f in glob.glob(os.path.join(boltz.input_dir, "*.yaml")):
            os.remove(f)
    except OSError:
        pass
    os.makedirs(boltz.input_dir, exist_ok=True)
    os.makedirs(boltz.output_dir, exist_ok=True)


def _stop_bt_log_listener() -> None:
    """Stop bittensor's logging QueueListener before the process tears down.

    bittensor installs a logging.handlers.QueueListener whose thread is named
    after its target, `_monitor`. If the interpreter starts tearing down while
    that thread is still running, it raises into stderr as

        Exception in thread Thread-2 (_monitor)

    which is noise, but noise that shows up in every worker exit and buries real
    errors. Upstream hit the same thing -- see _stop_bt_log_listener in
    nova/utils/inference.py. stop() is not idempotent, hence the unregister.
    """
    try:
        import atexit as _atexit
        import bittensor as bt
        listener = getattr(bt.logging, "_listener", None)
        if listener is None or getattr(listener, "_thread", None) is None:
            return
        _atexit.unregister(listener.stop)
        listener.stop()
    except Exception:
        pass


def worker_main(gpu_id: int, worker_id: int, base_dir: str,
                task_q, result_q, ready_q) -> None:
    # Must happen before torch is imported anywhere in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("NOVA_BOLTZ_THREADS", "1"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("NOVA_BOLTZ_THREADS", "1"))

    for p in (base_dir, os.path.join(base_dir, "boltz"), os.path.join(base_dir, "miner")):
        if p not in sys.path:
            sys.path.insert(0, p)

    tag = f"multi{os.getpid()}_w{worker_id}"
    try:
        from boltz_wrapper import BoltzWrapper
        boltz = BoltzWrapper()
        _isolate(boltz, tag)
        ready_q.put({"worker_id": worker_id, "gpu": gpu_id, "ok": True,
                     "pid": os.getpid(), "workspace": boltz.input_dir})
    except Exception as e:
        ready_q.put({"worker_id": worker_id, "gpu": gpu_id, "ok": False,
                     "pid": os.getpid(),
                     "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})
        return

    while True:
        try:
            task = task_q.get()
        except (EOFError, OSError):
            _stop_bt_log_listener()
            return
        if task is None:                       # shutdown pill
            _stop_bt_log_listener()
            return

        chunk_id = task["chunk_id"]
        molecules = task["molecules"]          # [(name, smiles), ...]
        subnet_config = task["subnet_config"]
        t0 = time.time()
        try:
            _clear(boltz)
            # One synthetic uid. The parent owns the real uid mapping; giving
            # the wrapper a single uid keeps its internal bookkeeping trivial
            # and makes every chunk independent of every other.
            vm = {0: {"smiles": [s for _, s in molecules],
                      "names": [n for n, _ in molecules]}}
            sd = {0: {"target_scores": [[]], "antitarget_scores": [[]],
                      "entropy": None, "entropy_boltz": None,
                      "block_submitted": None, "push_time": ""}}
            boltz.score_molecules(vm, sd, subnet_config)

            # final_boltz_scores[0] is {target: {smiles: score}}; invert it to
            # {smiles: {target: score}} so the parent can merge chunks that hold
            # disjoint molecules without caring which worker produced them.
            by_target = (getattr(boltz, "final_boltz_scores", {}) or {}).get(0, {}) or {}
            scores = {}
            for target, by_smiles in by_target.items():
                for smi, val in by_smiles.items():
                    scores.setdefault(smi, {})[target] = val
            components = dict((getattr(boltz, "per_molecule_components", {})
                               or {}).get(0, {}) or {})

            result_q.put({
                "chunk_id": chunk_id, "worker_id": worker_id, "gpu": gpu_id,
                "ok": True, "scores": scores, "components": components,
                "n_in": len(molecules), "n_out": len(scores),
                "elapsed_s": time.time() - t0,
            })
        except Exception as e:
            result_q.put({
                "chunk_id": chunk_id, "worker_id": worker_id, "gpu": gpu_id,
                "ok": False, "n_in": len(molecules), "n_out": 0,
                "elapsed_s": time.time() - t0,
                "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc(),
            })
        finally:
            try:
                _clear(boltz)
            except Exception:
                pass
