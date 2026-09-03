#!/usr/bin/env bash
# =============================================================================
# hunter.py under pm2 — one process per reaction
# =============================================================================
#
#   ./run.sh start 4        start rxn4 only          (recommended on this box)
#   ./run.sh start 3 4      start rxn3 and rxn4
#   ./run.sh start all      start all five
#   ./run.sh stop [all|N]   stop
#   ./run.sh restart 4      restart with fresh env
#   ./run.sh logs 4         follow one reaction
#   ./run.sh status         pm2 list + per-reaction progress
#   ./run.sh plan 2         show what start would launch, without launching
#   ./run.sh save           persist the process list across reboot
#
#   MULTI=0 ./run.sh start 4          single-GPU path (the original BoltzWrapper)
#   WORKERS_PER_GPU=2 ./run.sh start 4
#   GPU_IDS=0,1 ./run.sh start 4      restrict the pool to some cards
#   BOLTZ_BUDGET=2400 ./run.sh start 2   pin the round budget, no scaling
#
# -----------------------------------------------------------------------------
# INTERPRETER: it must be the venv python, NOT the system python3.
#   /usr/bin/python3 -c "import rdkit"  ->  ModuleNotFoundError
# hunter.py needs rdkit, torch, sklearn and boltz, all of which live in
# ./.venv only. pm2 gets it via --interpreter.
#
# -----------------------------------------------------------------------------
# GPUs ON THIS MACHINE: 1  (NVIDIA GeForce RTX 5090, 32 GB)
#
# run_all.sh pins reaction N to GPU N-1, which assumes five cards. On this box
# only GPU 0 exists, so CUDA_VISIBLE_DEVICES=1..4 would leave rxn2..rxn5 with no
# device. Everything below therefore pins to GPU 0 by default and starts ONE
# reaction unless told otherwise.
#
# Do not run five hunters on one card. Each holds a Boltz-2 model plus its own
# prediction workspace; they would contend for 32 GB of VRAM and each would get
# roughly a fifth of the throughput, which is strictly worse than searching one
# reaction well. If this box has more cards than nvidia-smi showed when this was
# written, set GPU_FOR below.
#
# WHICH REACTION TO RUN, if you only have one card. Gap between our submittable
# top-20 sum (at novelty 0.6) and the field's CEILING -- the best 20 molecules
# any miner submitted in an epoch, pooled from ~5.5 miners because the field CSV
# carries no uid. That ceiling is not a winning submission and every gap below
# is therefore overstated; the RANKING is fair because all five reactions pool a
# near-identical number of miners. Re-measured 2026-09-03 against all five DBs
# and ~10k validator-scored molecules per reaction:
#
#     rxn1  -0.0388   AHEAD even of the inflated ceiling   <- run this
#     rxn2  +0.2422
#     rxn3  +0.2476   flat: its whole top 20 spans 0.0009
#     rxn4  +0.3029   weakest generator (recent median 0.0374)
#     rxn5  +0.3655   hardest bar in the field (#20 = 0.1191)
#
# THIS HAS COMPLETELY REVERSED. The previous ordering put rxn4 and rxn3 "at
# parity" and said rxn1 "needs ~44x more hits"; rxn1 is now the only reaction
# whose top 20 would beat the median submission, and rxn4 has become the weakest
# of the five. The old numbers were taken before the archive cull that removed
# every near-archive molecule from these DBs.
#
# One card spent on rxn1 is worth more than one card split five ways.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")"
BASE="$(pwd -P)"
PY="$BASE/.venv/bin/python"
source "$BASE/hunter_opts.sh"

# ---------------------------------------------------------------------------
# MULTI-GPU
# ---------------------------------------------------------------------------
# MULTI=1 routes the searcher through `python -m multi.run`, which swaps the
# name `boltz_wrapper` for a pool of single-GPU worker processes. hunter.py is
# NOT modified; see multi/README.md.
#
# WORKERS_PER_GPU defaults to 1 on this box, not to multi/'s own default of 2,
# and that is deliberate. Two workers next to a long-running hunter took this
# machine out of RAM on 31 Aug and the kernel killed the searcher mid-round:
# 60 GiB total, and hunter.py grows to ~50 GiB RSS over 14 hours (14.6 GiB at
# 28 minutes). VRAM was never the constraint -- 26 of 32 GiB were free.
#
# At WORKERS_PER_GPU=1 the pool still pays for itself, because HUNTER_BATCH=60
# is chunked into 2 calls of 30 instead of 6 calls of 10: per-call setup drops
# from 22% of wall clock to ~8.6%.
#
# Raise it to 2+ once the memory growth is understood, or on a box with more
# RAM. `python3 -m multi.bench` measures the right value on an idle machine.
MULTI="${MULTI:-1}"
# Empty = let multi/topology.py decide from free VRAM and free host RAM. It was
# pinned to 1 while hunter.py leaked ~600 MiB per round; that is fixed (the
# per-molecule caches are LRU-bounded now), so the planner can be trusted.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-}"

# How many workers the planner will actually start on this machine, asked once
# so the batch size can be matched to it.
# Everything the launch sizing decides, in one place, so `./run.sh plan` reports
# exactly what `./run.sh start` will use. Emits shell assignments to be eval'd.
size_launch() {
  local flags="$1" workers budget batch base_pool pool
  if [ "$MULTI" != "1" ]; then
    printf 'workers=1 budget=%s batch=%s pool=; SIZED_FLAGS=%q\n' \
      "$HUNTER_BUDGET" "$HUNTER_BATCH" "$flags"
    return
  fi
  workers=$(plan_count)
  budget=$(autoscale_budget "$workers")
  batch=$(autoscale_batch "$workers" "$budget")
  base_pool=$(printf '%s' "$flags" | grep -oE -- '--candidate-pool [0-9]+' | awk '{print $2}')
  pool=$(autoscale_pool "$workers" "$base_pool")
  if [ -n "$base_pool" ] && [ -n "$pool" ]; then
    flags=$(printf '%s' "$flags" | sed -E "s/--candidate-pool $base_pool/--candidate-pool $pool/")
  fi
  printf 'workers=%s budget=%s batch=%s pool=%s; SIZED_FLAGS=%q\n' \
    "$workers" "$budget" "$batch" "${pool:-$base_pool}" "$flags"
}

plan_count() {
  local n
  n=$(NOVA_MULTI_WORKERS_PER_GPU="$WORKERS_PER_GPU" \
      ${GPU_IDS:+NOVA_MULTI_GPU_IDS="$GPU_IDS"} \
      "$PY" -m multi.topology --count 2>/dev/null | tail -1)
  case "$n" in ''|*[!0-9]*) echo 1 ;; *) echo "$n" ;; esac
}

# THREE nested quantities, and only the middle one is a real "batch":
#
#   --boltz-budget   molecules per ROUND (surrogate refit, top-20 recompute)
#   --batch-size     molecules per score_molecules() call
#   pool chunk       molecules per predict() call, = ceil(batch / (workers*2)),
#                    clamped to [8, 48] by multi/pool.py
#   Boltz DataLoader batch_size=1, hardcoded in boltz/.../inferencev2.py:391 --
#                    Boltz scores one molecule at a time and has no batch arg.
#
# The ~22 s of checkpoint loading is paid per predict() call, i.e. per CHUNK, so
# chunk size is the only one that governs overhead. Keeping it at 30 costs 8.6%.
#
# Budget must scale with workers, not just batch. A fixed 150-molecule round
# split 8 ways is ~19 molecules per worker, which cannot amortise the setup no
# matter how the batch is sized:
#
#   workers  chunk  setup%  speedup       (budget fixed at 150)
#      1       30     8.6    1.00x
#      2       15    15.8    1.84x
#      4       15    15.8    3.07x
#      8       10    22.0    6.40x   <- 20% of an 8-GPU box lost to setup
#
# Scaling the budget with the worker count holds chunk at 30 and restores linear
# scaling (1.00 / 2.00 / 4.00 / 8.00x). It also keeps a round at roughly the same
# WALL CLOCK (~21 min) whatever the hardware, so the search re-fits its surrogate
# and refreshes its frontier just as often in time as it does on one GPU.
autoscale_budget() {
  local workers="$1"
  # Explicit pin wins over everything, on either path.
  if [ -n "${BOLTZ_BUDGET:-}" ]; then echo "$BOLTZ_BUDGET"; return; fi
  if [ "$MULTI" != "1" ]; then echo "$HUNTER_BUDGET"; return; fi
  echo $(( ${HUNTER_BUDGET_PER_WORKER:-$HUNTER_BUDGET} * workers ))
}

# 60 per worker, because the pool makes 2 chunks per worker: chunk lands on 30.
autoscale_batch() {
  local workers="$1" budget="$2" b
  if [ "$MULTI" != "1" ]; then echo "$HUNTER_BATCH"; return; fi
  b=$(( workers * 60 ))
  [ "$b" -lt "$HUNTER_BATCH" ] && b="$HUNTER_BATCH"
  [ "$b" -gt "$budget" ] && b="$budget"
  echo "$b"
}

# Scoring more molecules per round means reaching further down the surrogate's
# ranking, so the candidate pool has to grow too or selectivity falls: 22,000
# candidates for 150 slots is 150:1, but for 1,200 slots it is 19:1.
#
# Capped at 3x because generation and ranking are CPU-bound and serialised ahead
# of the GPU work (~50 s per 28,000 candidates measured on rxn2). Beyond ~3x they
# become the next bottleneck -- worth measuring on the real box before raising.
autoscale_pool() {
  local workers="$1" base="$2" mult="$workers"
  # No --candidate-pool in this reaction's flags: nothing to scale.
  case "$base" in ''|*[!0-9]*) echo ""; return ;; esac
  if [ "$MULTI" != "1" ]; then echo "$base"; return; fi
  [ "$mult" -gt 3 ] && mult=3
  echo $(( base * mult ))
}

# Reaction -> GPU index, used ONLY on the single-GPU path (MULTI=0).
#
# With MULTI=1 this is not used and CUDA_VISIBLE_DEVICES is deliberately NOT
# set: the pool has to see every card in order to place workers across them.
# Pinning here would silently confine a 4-GPU box to one card --
#
#   4 GPUs, unpinned              -> plan [0, 1, 2, 3, ...]
#   4 GPUs, CUDA_VISIBLE_DEVICES=0 -> plan [0, 0]
#
# To restrict the pool to some cards, use NOVA_MULTI_GPU_IDS=0,2 instead, which
# multi/topology.py honours without hiding the others from detection.
GPU_FOR() { echo 0; }

NAME_FOR() { echo "hunter-rxn$1"; }

# -----------------------------------------------------------------------------
# The command, spelled out. `start_one 4` runs exactly this:
#
#   CUDA_VISIBLE_DEVICES=0 pm2 start hunter.py \
#       --name hunter-rxn4 \
#       --interpreter /root/workspace/nova-4090/.venv/bin/python \
#       --cwd /root/workspace/nova-4090 \
#       --time --restart-delay 30000 --max-restarts 10 --kill-timeout 30000 \
#       -- --rxn-id 4 --boltz-budget 150 --batch-size 10 \
#          --candidate-pool 30000 --block-frac 0.17 --block-k 3 \
#          --block-seed-reuse 2 --prior-temperature 5.0 --prior-floor 0.12 \
#          --elite-quantile 0.90 --field-weight 0.5 --strict-pool-mult 6
#
# Everything after the bare `--` goes to hunter.py; everything before it is pm2.
#
# --restart-delay/--max-restarts: hunter.py exits deliberately (SystemExit) if
#   its score DB fails PRAGMA quick_check. Without these pm2 would spin on that.
# --kill-timeout 30000: give an in-flight Boltz batch 30 s to finish on stop.
# --time: pm2 timestamps; hunter's own log lines carry timestamps too.
# No --max-memory-restart: killing a process mid-Boltz costs a whole round.
# -----------------------------------------------------------------------------
start_one() {
  local r="$1" name gpu flags
  name="$(NAME_FOR "$r")"
  gpu="$(GPU_FOR "$r")"
  if ! flags="$(opts_for "$r")"; then
    echo "rxn$r: not a valid reaction (1..5)"; return 1
  fi
  if pm2 describe "$name" >/dev/null 2>&1; then
    echo "$name already in pm2 — use './run.sh restart $r'"; return 0
  fi
  local interp_args=()
  local -a envs=()
  local batch budget workers pool
  eval "$(size_launch "$flags")"
  flags="$SIZED_FLAGS"
  if [ "$MULTI" = "1" ]; then
    interp_args=(--interpreter-args "-m multi.run")
    # No CUDA_VISIBLE_DEVICES: the pool must see every card to spread across
    # them. It sets CUDA_VISIBLE_DEVICES itself, per worker, to one id.
    envs=()
    [ -n "$WORKERS_PER_GPU" ] && envs+=(NOVA_MULTI_WORKERS_PER_GPU="$WORKERS_PER_GPU")
    [ -n "${GPU_IDS:-}" ] && envs+=(NOVA_MULTI_GPU_IDS="$GPU_IDS")
    # hunter's per-molecule caches must cover one round's working set, which is
    # the candidate pool PLUS the surrogate's training subset (--train-cap,
    # 30,000). Below that they thrash between fit() and predict() every round.
    # A 4-GPU box plans a 90,000 pool, so the 100,000 default was already too
    # small. _mol is left alone: it is 17.6 KiB per entry against ~2 KiB for the
    # others, and re-parsing a SMILES costs ~100 us.
    if [ -n "${pool:-}" ]; then
      local cache=$(( pool + 50000 ))
      envs+=(HUNTER_FP_CACHE="$cache" HUNTER_BV_CACHE="$cache" HUNTER_DESC_CACHE="$cache")
    fi
    echo "starting $name via multi.run | ${workers} worker(s) auto-planned | budget ${budget} batch ${batch} pool ${pool:-$base_pool} | GPUs: ${GPU_IDS:-all detected}"
  else
    envs=(CUDA_VISIBLE_DEVICES="$gpu")
    echo "starting $name on GPU $gpu (single-GPU path, batch ${batch})"
  fi
  # shellcheck disable=SC2086
  env "${envs[@]}" \
  pm2 start hunter.py \
      --name "$name" \
      --interpreter "$PY" \
      "${interp_args[@]}" \
      --cwd "$BASE" \
      --time \
      --restart-delay 30000 \
      --max-restarts 10 \
      --kill-timeout 30000 \
      -- --rxn-id "$r" \
         --boltz-budget "$budget" \
         --batch-size "$batch" \
         $flags
}

expand() {
  if [ "${1:-}" = "all" ]; then echo 1 2 3 4 5; else echo "$@"; fi
}

cmd="${1:-status}"; shift || true

case "$cmd" in
  start)
    targets="$(expand "${@:-4}")"
    for r in $targets; do start_one "$r"; sleep 3; done
    echo
    echo "logs:   ./run.sh logs <rxn>      (or ~/.pm2/logs/hunter-rxnN-out.log)"
    echo "persist: ./run.sh save"
    ;;
  stop)
    for r in $(expand "${@:-all}"); do pm2 stop "$(NAME_FOR "$r")" 2>/dev/null; done
    ;;
  delete|rm)
    for r in $(expand "${@:-all}"); do pm2 delete "$(NAME_FOR "$r")" 2>/dev/null; done
    ;;
  restart)
    # --update-env so a changed CUDA_VISIBLE_DEVICES actually takes effect.
    for r in $(expand "${@:-all}"); do
      CUDA_VISIBLE_DEVICES="$(GPU_FOR "$r")" pm2 restart "$(NAME_FOR "$r")" --update-env 2>/dev/null
    done
    ;;
  plan)
    # What `start` would launch, without launching it.
    for r in $(expand "${@:-2}"); do
      flags="$(opts_for "$r")" || continue
      eval "$(size_launch "$flags")"
      echo "rxn$r:"
      echo "  $("$PY" -m multi.topology 2>/dev/null | head -1)"
      echo "  workers=$workers  --boltz-budget $budget  --batch-size $batch  --candidate-pool $pool"
      echo "  chunk per predict() = $("$PY" -c "import sys;sys.path.insert(0,'.');from multi.pool import choose_chunk_size as c;print(c($batch,$workers))") molecules"
    done
    ;;
  logs)
    pm2 logs "$(NAME_FOR "${1:-4}")" --lines 50
    ;;
  save)
    pm2 save
    echo "run 'pm2 startup' once if you also want these back after a reboot"
    ;;
  status|*)
    pm2 list
    echo
    for r in 1 2 3 4 5; do
      f="$HOME/.pm2/logs/hunter-rxn${r}-out.log"
      [ -f "$f" ] || continue
      echo "--- rxn$r ---"
      grep -h "round .* done\]" "$f" 2>/dev/null | tail -2
      grep -h "by strategy" "$f" 2>/dev/null | tail -1
    done
    ;;
esac

# =============================================================================
# LEGACY — the previous contents of this file, kept for reference. These are
# older miners/submitters, not part of the hunter pipeline.
# =============================================================================
# python3 neurons/miner_ban_mini_db.py --wallet.name xova --wallet.hotkey xotb --logging.debug
# python3 neurons/miner_ban_synthon_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
# python3 neurons/miner_ban_neighbour_mutate.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
# python3 neurons/miner_ban_random_mutate_db.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
# python3 neurons/simple_submit.py --logging.debug --wallet.name nova --wallet.hotkey notc
# python3 neurons/top_submit.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
# python3 neurons/synthon_miner.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
# python3 neurons/mini_data.py --wallet.name multisig-jjpes-shib --wallet.hotkey hotd --logging.debug
# CUDA_VISIBLE_DEVICES=1 pm2 start "python3 neurons/synthon_data.py --logging.debug" --name "synthon_data"
# pm2 start "python3 neurons/repeat_submit_one.py --wallet.name nova --wallet.hotkey notc --netuid 68 --network finney" --name one
# pm2 start "python3 neurons/multi_submit.py --wallet.name nova --netuid 68 --network finney --logging.debug" --name multi
