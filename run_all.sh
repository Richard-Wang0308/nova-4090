#!/usr/bin/env bash
# Launch one hunter per reaction, one GPU each.
#
#   ./run_all.sh              start all five
#   ./run_all.sh 2 5          start only rxn2 and rxn5
#   ./run_all.sh stop         stop everything
#
# Reaction N is pinned to GPU (N-1). Change GPU_FOR below if your device
# indices differ — check with `nvidia-smi -L`.

set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
LOGDIR=logs
mkdir -p "$LOGDIR"

GPU_FOR() { echo $(( $1 - 1 )); }

# Boltz costs ~15 s/molecule, so a 150-budget round is ~40 min including the
# 3-draw confirmation of anything above 0.1.
BUDGET=150
POOL=30000
BATCH=10

stop_all() {
  for r in 1 2 3 4 5; do
    pkill -f "hunter.py --rxn-id $r" 2>/dev/null && echo "stopped rxn$r"
  done
  exit 0
}
[ "${1:-}" = "stop" ] && stop_all

RXNS=("$@")
[ ${#RXNS[@]} -eq 0 ] && RXNS=(1 2 3 4 5)

for r in "${RXNS[@]}"; do
  if pgrep -f "hunter.py --rxn-id $r" >/dev/null; then
    echo "rxn$r already running — skipping"; continue
  fi
  gpu=$(GPU_FOR "$r")
  log="$LOGDIR/hunter_rxn${r}.log"
  setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" hunter.py \
      --rxn-id "$r" \
      --boltz-budget "$BUDGET" \
      --candidate-pool "$POOL" \
      --batch-size "$BATCH" \
      >> "$log" 2>&1 < /dev/null &
  disown
  echo "rxn$r -> GPU $gpu  (log: $log)"
  sleep 2
done

echo
echo "watch:    tail -f $LOGDIR/hunter_rxn2.log"
echo "progress: grep 'round .* done' $LOGDIR/hunter_rxn*.log | tail"
echo "stop:     ./run_all.sh stop"
