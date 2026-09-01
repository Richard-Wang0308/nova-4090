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
#   ./run.sh save           persist the process list across reboot
#
#   MULTI=0 ./run.sh start 4          single-GPU path (the original BoltzWrapper)
#   WORKERS_PER_GPU=2 ./run.sh start 4
#   GPU_IDS=0,1 ./run.sh start 4      restrict the pool to some cards
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
# WHICH REACTION TO RUN, if you only have one card. Measured deficit between our
# available top-20 sum and the median winning sum (see the P0 analysis):
#
#     rxn4   0.008   at parity   <- run this
#     rxn3   0.009   at parity   <- then this
#     rxn2   0.036   close
#     rxn1   0.116   needs ~44x more hits above the winning threshold
#     rxn5   0.156   needs ~190x
#
# rxn3 and rxn4 are together ~40% of epochs and are the only two where the gap
# is smaller than what these changes plausibly deliver. One card spent on rxn4
# is worth more than one card split five ways.
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
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"

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
  if [ "$MULTI" = "1" ]; then
    interp_args=(--interpreter-args "-m multi.run")
    # No CUDA_VISIBLE_DEVICES: the pool must see every card to spread across
    # them. It sets CUDA_VISIBLE_DEVICES itself, per worker, to one id.
    envs=(NOVA_MULTI_WORKERS_PER_GPU="$WORKERS_PER_GPU")
    [ -n "${GPU_IDS:-}" ] && envs+=(NOVA_MULTI_GPU_IDS="$GPU_IDS")
    echo "starting $name via multi.run (${WORKERS_PER_GPU} worker/GPU, batch ${HUNTER_BATCH}, GPUs: ${GPU_IDS:-all detected})"
  else
    envs=(CUDA_VISIBLE_DEVICES="$gpu")
    echo "starting $name on GPU $gpu (single-GPU path, batch ${HUNTER_BATCH})"
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
         --boltz-budget "$HUNTER_BUDGET" \
         --batch-size "$HUNTER_BATCH" \
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
