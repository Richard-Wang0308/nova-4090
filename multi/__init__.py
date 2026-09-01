"""
multi — multi-GPU Boltz-2 inference for the NOVA small-molecule searchers.

Adopt it by changing two lines in whichever searcher you run:

    from boltz_wrapper import BoltzWrapper        from multi import MultiGPUBoltz
    boltz = BoltzWrapper()               ->       boltz = MultiGPUBoltz()

Everything else -- score_molecules, final_boltz_scores, per_molecule_components,
input_dir/output_dir -- keeps the same shape, so no call site changes.

GPU count is detected at start-up. One card or eight, the pool sizes itself; see
multi/topology.py for the placement rules and multi/README.md for adoption notes
per script.
"""
from .topology import detect_gpus, plan_workers, describe          # noqa: F401
from .pool import BoltzPool, choose_chunk_size                     # noqa: F401
from .wrapper import MultiGPUBoltz                                 # noqa: F401

__all__ = ["MultiGPUBoltz", "BoltzPool", "plan_workers", "detect_gpus",
           "describe", "choose_chunk_size"]

import logging as _logging
import sys as _sys


def _install_log_handler() -> None:
    """Give the `multi` logger its own stdout handler.

    Importing the real boltz_wrapper pulls in bittensor, which reconfigures root
    logging. That happens before a searcher's own logging.basicConfig() runs, so
    basicConfig becomes a no-op and anything logged under a non-bittensor logger
    name is swallowed. Observed live: the pool ran correctly but not one
    "multi: chunk N done" line reached the log, which is exactly the output an
    operator needs to see whether workers are healthy and how chunks are timed.

    Own handler plus propagate=False means exactly one emission regardless of
    what else has configured the root logger.
    """
    lg = _logging.getLogger("multi")
    if any(getattr(h, "_multi_handler", False) for h in lg.handlers):
        return
    h = _logging.StreamHandler(_sys.stdout)
    h.setFormatter(_logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    h._multi_handler = True
    lg.addHandler(h)
    lg.setLevel(_logging.INFO)
    lg.propagate = False


_install_log_handler()
