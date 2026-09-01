"""
multi/_log.py — keep the pool's logs visible.

bittensor configures logging on import, and that configuration disables loggers
that already exist. multi.run creates multi.pool / multi.wrapper first and only
then imports the real boltz_wrapper (which pulls in bittensor), so every
`multi.*` logger was arriving already switched off:

    [multi] 1 worker(s): GPU0 x1 ...        <- a plain print, survived
    multi: 60 molecules -> 2 chunk(s) ...   <- logging, silently dropped

hunter.py's own logger was unaffected because it is created after bittensor.
The chunk layout, per-chunk timing and worker failures all go through these
loggers, and multi/README.md tells operators to grep for them, so losing them is
not cosmetic.

ensure_visible() re-enables the `multi` logger and gives it a private stdout
handler with propagate=False, so it no longer depends on whatever owns the root
logger. It is idempotent and called again at each pool start, because bittensor
may be imported after the first call.
"""
from __future__ import annotations

import logging
import sys

_NAMES = ("multi", "multi.pool", "multi.wrapper", "multi.topology", "multi.bench")


def ensure_visible(level: int = logging.INFO) -> None:
    parent = logging.getLogger("multi")
    for name in _NAMES:
        lg = logging.getLogger(name)
        lg.disabled = False
        if name != "multi":
            # bittensor does not merely disable pre-existing loggers, it sets an
            # explicit level on them -- measured: multi.pool came back with
            # .level = 50 (CRITICAL). An explicit level stops the logger
            # inheriting from its parent, so raising "multi" to INFO was not
            # enough on its own. NOTSET restores inheritance.
            lg.setLevel(logging.NOTSET)
    if not any(getattr(h, "_multi_handler", False) for h in parent.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        h._multi_handler = True
        parent.addHandler(h)
    parent.setLevel(level)
    # Own handler, so a root logger owned by bittensor can neither swallow the
    # records nor print them twice.
    parent.propagate = False


def get_logger(name: str) -> logging.Logger:
    ensure_visible()
    return logging.getLogger(name)
