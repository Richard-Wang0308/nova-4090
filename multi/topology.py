"""
multi/topology.py — how many workers to run, and on which GPUs.

The parent process deliberately never imports torch. Creating a CUDA context in
the parent costs ~300 MiB per visible device and, worse, makes fork-based worker
start-up unsafe. Everything here reads nvidia-smi instead.

WHY MORE THAN ONE WORKER PER GPU
--------------------------------
A single Boltz call is not GPU-bound end to end. Measured on this box (one
RTX 5090, hunter.py at --batch-size 10, 2026-08-31):

    19:11:06  batch starts, "Checking input data."
    19:11:18  structure prediction starts        <- 12 s, GPU idle
    19:11:59  structure prediction ends          <- 41 s, GPU busy
    19:11:59  "Checking input data for affinity."
    19:12:09  affinity prediction starts         <- 10 s, GPU idle
    19:12:46  affinity prediction ends           <- 37 s, GPU busy
    19:12:47  batch reported, 101 s total

22 s of every 100 s is manifest checking, checkpoint loading and Trainer
construction, and boltz.main.predict does all of it again on every call. Polling
nvidia-smi at 3 s over a minute of steady-state scoring agreed: mean GPU
utilisation 62.6%, with 7 of 20 samples at 0% and VRAM back down to 804 MiB.

A second worker on the same card fills those gaps with the first worker's
compute. Peak VRAM per worker was 5,936 MiB, so a 32 GiB card has room for
several; the limit that bites first is compute, not memory.

    workers_per_gpu = 1   ->  ~62% utilisation, one setup gap per chunk exposed
    workers_per_gpu = 2   ->  gaps overlap; the expected default
    workers_per_gpu = 3+  ->  only helps if setup is a larger share; measure it
                              with `python3 -m multi.bench`

Nothing here guesses beyond 2 without evidence. `multi/calibration.json`, written
by bench.py, overrides the default when present.

ENVIRONMENT OVERRIDES (checked in this order)
---------------------------------------------
    NOVA_BOLTZ_GPUS            explicit worker->GPU map, upstream-compatible.
                               "0,0,1,1" means four workers, two per card.
    NOVA_MULTI_WORKERS_PER_GPU integer, applied to every detected GPU.
    NOVA_MULTI_GPU_IDS         restrict which GPUs are used, e.g. "0,2".
    NOVA_WORKER_VRAM_MIB       per-worker VRAM budget; 0 disables the check.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

# Peak observed was 5,936 MiB; the margin covers a larger ligand and the
# allocator's caching behaviour.
DEFAULT_WORKER_VRAM_MIB = 7000
# Host RAM per worker. This is NOT a theoretical figure: an idle worker measured
# 1,242 MiB RSS, but the box this was written on has 60 GiB total and a
# long-running hunter.py grows to ~50 GiB RSS over 14 hours. Starting two
# workers next to it took the machine out of memory and the kernel killed the
# searcher mid-round (pm2 restarted it; no traceback, which is the signature).
# VRAM was never the constraint there -- 26 GiB of 32 was free. So placement has
# to respect host RAM too, with a reserve for the parent's own growth.
DEFAULT_WORKER_RSS_MIB = 6000
# Left for the parent process and the OS. A searcher that has been running for
# hours is the largest thing on the box, not the workers.
DEFAULT_HOST_RESERVE_MIB = 12000
# Raising this without measuring is how you turn a throughput win into an OOM.
DEFAULT_WORKERS_PER_GPU = 2
MAX_WORKERS_PER_GPU = 8


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    total_mib: int
    free_mib: int


def _nvidia_smi() -> List[Gpu]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: List[Gpu] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), parts[1], int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    return gpus


def detect_gpus() -> List[Gpu]:
    """Every GPU this process is allowed to see, honouring CUDA_VISIBLE_DEVICES.

    When CUDA_VISIBLE_DEVICES is set, nvidia-smi still reports every physical
    card, but torch inside a worker will renumber them. Returning the physical
    ids that survive the mask keeps the two views consistent, because a worker
    is launched with CUDA_VISIBLE_DEVICES set to exactly one physical id.
    """
    gpus = _nvidia_smi()
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None and mask.strip() != "":
        allowed = []
        for tok in mask.split(","):
            tok = tok.strip()
            if tok.isdigit():
                allowed.append(int(tok))
        if allowed:
            gpus = [g for g in gpus if g.index in allowed]
    only = os.environ.get("NOVA_MULTI_GPU_IDS", "").strip()
    if only:
        keep = {int(t) for t in only.split(",") if t.strip().isdigit()}
        gpus = [g for g in gpus if g.index in keep]
    return gpus


def _mem_available_mib() -> int:
    """MemAvailable from /proc/meminfo, or 0 if it cannot be read."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def worker_rss_mib() -> int:
    """Assumed host RAM per worker. Overridable, and measured by multi.bench."""
    env = _int_env("NOVA_MULTI_WORKER_RSS_MIB", 0)
    if env:
        return env
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            v = int(json.load(f).get("peak_worker_rss_mib") or 0)
        if v > 0:
            return v
    except Exception:
        pass
    return DEFAULT_WORKER_RSS_MIB


def worker_vram_mib() -> int:
    raw = os.environ.get("NOVA_WORKER_VRAM_MIB")
    if not raw:
        return DEFAULT_WORKER_VRAM_MIB
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_WORKER_VRAM_MIB


def _calibrated_workers_per_gpu() -> Optional[int]:
    """Value measured by `python3 -m multi.bench`, if it has been run here."""
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        v = int(blob["workers_per_gpu"])
        return v if 1 <= v <= MAX_WORKERS_PER_GPU else None
    except Exception:
        return None


def plan_workers(workers_per_gpu: Optional[int] = None) -> List[int]:
    """Return one physical GPU id per worker to start.

    [0, 0, 1, 1] means four workers: two on GPU 0 and two on GPU 1. The list is
    ordered so that consecutive workers land on different cards, which matters
    when the pool starts fewer workers than planned.
    """
    explicit = os.environ.get("NOVA_BOLTZ_GPUS", "").strip()
    if explicit:
        ids = [int(t) for t in explicit.split(",") if t.strip().lstrip("-").isdigit()]
        if ids:
            return ids

    gpus = detect_gpus()
    if not gpus:
        # No GPU visible. One worker, letting boltz fall back to whatever
        # accelerator it finds, is better than refusing to run.
        return [0]

    if workers_per_gpu is None:
        env = os.environ.get("NOVA_MULTI_WORKERS_PER_GPU", "").strip()
        if env.isdigit() and int(env) > 0:
            workers_per_gpu = int(env)
        else:
            workers_per_gpu = _calibrated_workers_per_gpu() or DEFAULT_WORKERS_PER_GPU
    workers_per_gpu = max(1, min(int(workers_per_gpu), MAX_WORKERS_PER_GPU))

    budget = worker_vram_mib()
    per_gpu: Dict[int, int] = {}
    for g in gpus:
        n = workers_per_gpu
        if budget > 0:
            # free_mib, not total: something else may already be on this card —
            # on this box a hunter.py was mining while the pool was designed.
            fits = max(1, g.free_mib // budget)
            n = min(n, fits)
        per_gpu[g.index] = max(1, n)

    # Round-robin so worker k and worker k+1 are on different cards.
    plan: List[int] = []
    for slot in range(max(per_gpu.values())):
        for g in gpus:
            if slot < per_gpu[g.index]:
                plan.append(g.index)

    # Host RAM is a global limit, so it is applied to the whole plan rather than
    # per card. Trimming from the end keeps the round-robin property, so the
    # workers that survive are still spread across every GPU.
    rss = worker_rss_mib()
    if rss > 0:
        avail = _mem_available_mib()
        if avail > 0:
            reserve = _int_env("NOVA_MULTI_HOST_RESERVE_MIB", DEFAULT_HOST_RESERVE_MIB)
            room = max(0, avail - reserve) // rss
            if room < len(plan):
                dropped = len(plan) - max(1, room)
                plan = plan[:max(1, room)]
                print(f"[multi] host RAM caps the pool: {avail} MiB available, "
                      f"{reserve} MiB reserved for the parent, {rss} MiB/worker "
                      f"-> dropping {dropped} worker(s)")
    return plan


def describe(plan: Optional[List[int]] = None) -> str:
    plan = plan_workers() if plan is None else plan
    gpus = {g.index: g for g in detect_gpus()}
    counts: Dict[int, int] = {}
    for gid in plan:
        counts[gid] = counts.get(gid, 0) + 1
    bits = []
    for gid in sorted(counts):
        g = gpus.get(gid)
        label = f"{g.name} {g.free_mib}/{g.total_mib} MiB free" if g else "unknown device"
        bits.append(f"GPU{gid} x{counts[gid]} ({label})")
    return f"{len(plan)} worker(s): " + "; ".join(bits) if bits else "no GPUs detected"


if __name__ == "__main__":
    p = plan_workers()
    print(describe(p))
    print("plan:", p)
