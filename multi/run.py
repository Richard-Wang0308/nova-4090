"""
multi/run.py — launch an unmodified searcher on every GPU.

    python3 -m multi.run hunter.py --rxn-id 4 --boltz-budget 150
    python3 -m multi.run orchestrator.py --rxn-id 2
    python3 -m multi.run neurons/late_stage_search.py ...
    python3 -m multi.run miner/miner.py ...
    python3 -m multi.run neurons/genetic.py ...

The target script is run exactly as `python3 <script> <args>` would run it --
same __main__, same argv, same working directory -- with one difference: the
name `boltz_wrapper` resolves to the multi-GPU facade. No searcher is edited.

Pool options come from the environment so they never collide with the target's
own argument parser:

    NOVA_MULTI_WORKERS_PER_GPU=3    override the auto-detected worker count
    NOVA_MULTI_GPU_IDS=0,2          restrict to some cards
    NOVA_BOLTZ_GPUS=0,0,1,1         explicit worker->GPU map (upstream format)
    NOVA_WORKER_VRAM_MIB=7000       per-worker VRAM budget for placement
"""
from __future__ import annotations

import os
import runpy
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 2

    script = sys.argv[1]
    path = script if os.path.isabs(script) else os.path.join(BASE_DIR, script)
    if not os.path.exists(path):
        print(f"multi.run: no such script: {path}", file=sys.stderr)
        return 2

    for p in (BASE_DIR, os.path.join(BASE_DIR, "miner"), os.path.join(BASE_DIR, "boltz")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from . import patch, topology
    print(f"[multi] {topology.describe()}", flush=True)
    patch.enable()

    # Hand the target the argv it expects: argv[0] is the script itself.
    sys.argv = [path] + sys.argv[2:]
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
