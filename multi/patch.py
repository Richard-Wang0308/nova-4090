"""
multi/patch.py — adopt multi-GPU inference without editing any mining script.

The five searchers import Boltz in two different ways:

    hunter.py, orchestrator.py      from boltz_wrapper import BoltzWrapper   (module level)
    late_stage_search.py, miner.py,
    genetic.py                      from boltz_wrapper import BoltzWrapper as BW  (inside a
                                    lazy _import_boltz_wrapper() called at runtime)

Patching the scripts' own globals would only cover the first two. Installing a
shim under the name `boltz_wrapper` in sys.modules covers all five, because both
forms resolve through the same import, and it works whether the import already
happened or not.

    import multi.patch; multi.patch.enable()
    # anything importing boltz_wrapper from here on gets MultiGPUBoltz

The real class stays reachable as boltz_wrapper.RealBoltzWrapper, which is what
the pool's workers use -- they are separate processes that never import this
module, so they always get the genuine single-GPU wrapper. That separation is
the point: one multi-GPU facade in the parent, N real wrappers in the children.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_ORIGINAL = None
_ENABLED = False


def _real_module():
    """Import the genuine boltz_wrapper, bypassing any shim already installed."""
    shim = sys.modules.pop("boltz_wrapper", None)
    try:
        for p in (BASE_DIR, os.path.join(BASE_DIR, "boltz")):
            if p not in sys.path:
                sys.path.insert(0, p)
        return importlib.import_module("boltz_wrapper")
    finally:
        if shim is not None:
            sys.modules["boltz_wrapper"] = shim


def enable(workers_per_gpu: int | None = None, plan=None) -> None:
    """Make `from boltz_wrapper import BoltzWrapper` yield MultiGPUBoltz."""
    global _ORIGINAL, _ENABLED
    if _ENABLED:
        return

    from .wrapper import MultiGPUBoltz

    try:
        _ORIGINAL = _real_module()
    except Exception:
        _ORIGINAL = None

    # A single shared pool. Two MultiGPUBoltz instances in one process -- which
    # rescore.confirm_high_scorers creates, alongside the searcher's own -- must
    # not each start their own workers, or a 4-GPU box quietly runs 16.
    from .pool import BoltzPool
    shared = BoltzPool(workers_per_gpu=workers_per_gpu, plan=plan)

    class _PooledMultiGPUBoltz(MultiGPUBoltz):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("pool", shared)
            super().__init__(*args, **kwargs)

        def shutdown(self) -> None:
            # Never tear down the shared pool from one consumer's teardown.
            pass

    shim = types.ModuleType("boltz_wrapper")
    shim.BoltzWrapper = _PooledMultiGPUBoltz
    shim.MultiGPUBoltz = MultiGPUBoltz
    shim.RealBoltzWrapper = getattr(_ORIGINAL, "BoltzWrapper", None)
    shim.__doc__ = "shim installed by multi.patch.enable(); real class at RealBoltzWrapper"
    shim.__file__ = __file__
    shim._multi_shim = True
    shim._shared_pool = shared
    sys.modules["boltz_wrapper"] = shim

    # Scripts that already imported the real class keep their own reference, so
    # rebind those too.
    for name, mod in list(sys.modules.items()):
        if mod is None or name.startswith("multi."):
            continue
        cur = getattr(mod, "BoltzWrapper", None)
        if cur is not None and _ORIGINAL is not None and cur is getattr(_ORIGINAL, "BoltzWrapper", None):
            setattr(mod, "BoltzWrapper", _PooledMultiGPUBoltz)

    _ENABLED = True


def disable() -> None:
    global _ENABLED
    if not _ENABLED:
        return
    pool = getattr(sys.modules.get("boltz_wrapper"), "_shared_pool", None)
    if pool is not None:
        pool.shutdown()
    if _ORIGINAL is not None:
        sys.modules["boltz_wrapper"] = _ORIGINAL
    else:
        sys.modules.pop("boltz_wrapper", None)
    _ENABLED = False
