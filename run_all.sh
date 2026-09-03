#!/usr/bin/env bash
# Launch one hunter per reaction, one GPU each.
#
#   ./run_all.sh              start all five
#   ./run_all.sh 2 5          start only rxn2 and rxn5
#   ./run_all.sh stop         stop everything
#
# Reaction N is pinned to GPU (N-1). Change GPU_FOR below if your device
# indices differ -- check with `nvidia-smi -L`.
#
# ---------------------------------------------------------------------------
# PER-REACTION SETTINGS
# ---------------------------------------------------------------------------
# The five reactions are not the same search problem and no longer share a
# parameter set. Component pools are fixed by the combinatorial DB; coverage is
# what WE have tried, re-counted 2026-09-03 from the score DBs:
#
#   rxn  type  A pool   B pool   C pool   coverage A/B/C
#    1    2c    25,728   35,774     --      30.4% / 18.0%
#    2    2c    83,307   24,165     --      21.1% /  7.8%
#    3    3c       817    2,731  18,330  73.1% / 49.2% / 22.7%
#    4    2c    49,108    6,959     --       8.4% / 62.0%
#    5    3c     2,054    2,874   2,874  48.7% / 67.7% / 67.0%
#
# The coverage figures the previous header recorded (rxn3 "99.3%/98.1%",
# rxn5 "95.2%/94.3%/96.1%") are not reproducible from these DBs and should not
# be trusted; rxn2's B position at 7.8% and rxn4's A at 8.4% are the two least
# explored, and drive the flat priors those reactions now carry.
#
# BLOCK-K is chosen as the SMALLEST k whose fresh-cell supply still fills the
# quota, because block enrichment falls monotonically with k (rxn1: 21.1x at
# k=2, 18.5x at k=4, 16.0x at k=6, 12.2x at k=12). Measured fresh cells per
# round at --block-seed-pool 3000:
#
#   rxn1  k=4/reuse2 -> 4,429   k=6/reuse1 -> 5,352
#   rxn2  k=4/reuse2 -> 4,181   k=6/reuse1 -> 4,912
#   rxn3  k=3/reuse2 -> 4,451   k=2/reuse2 -> 1,826
#   rxn4  k=3/reuse2 -> 4,819   k=4/reuse1 -> 4,346
#   rxn5  k=2/reuse2 -> 4,066   k=3/reuse1 -> 5,335
#
# BLOCK-SEED-REUSE is 2 everywhere. The binding constraint is not pool size but
# how concentrated our own top-3,000 scorers are on a handful of reactants, and
# that is severe in every reaction: at reuse 1 the operator starved on rxn3 and
# rxn5 (8 and 6 seeds in a live dry run) and ran at roughly half supply on rxn2
# (2,350 cells vs 4,181) and rxn4 (2,649 vs 4,819). Reuse 3+ was not used: it
# re-concentrates the scan on the same chemistry, which is the failure mode the
# operator exists to escape.
#
# PRIOR-TEMPERATURE and PRIOR-FLOOR track pool coverage: a reaction with 75-80%
# of its building blocks never tried needs a flatter prior and a larger uniform
# floor; rxn3 and rxn5 are ~95% covered, so there is nothing left to discover at
# the component level and the prior should be sharp.
#
# FIELD-WEIGHT tracks how much the field CSV actually knows about the reaction.
# Molecules above 0.125 in data/rxn{N}.csv: rxn1 252, rxn2 476, rxn3 4,
# rxn4 320, rxn5 1,071. rxn3's field prior is built on four data points.
#
# STRICT-POOL-MULT tracks the novelty pass rate at 0.6, re-measured on a random
# sample of each DB: rxn1 50.6%, rxn2 28.1%, rxn3 31.5%, rxn4 63.9%, rxn5 33.6%
# -- so 1.6x to 3.6x is the true requirement, against values of 10-20x that were
# set for a 0.7 gate on archive-saturated pools. BRENK now rejects NOTHING (100%
# pass on all five), so the old "rxn1 loses 58% to BRENK and validity" no longer
# holds and CANDIDATE-POOL is no longer inflated to compensate for it.
# ---------------------------------------------------------------------------

set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
LOGDIR=logs
mkdir -p "$LOGDIR"

GPU_FOR() { echo $(( $1 - 1 )); }

# Boltz costs ~15 s/molecule. Round-leader confirmation is off by default now,
# so a 150-budget round is ~38 min rather than ~43.
# Per-reaction flags and BUDGET/BATCH live in hunter_opts.sh, shared with
# run.sh (the pm2 launcher), so the two cannot drift apart.
source "$(pwd -P)/hunter_opts.sh"
BUDGET="$HUNTER_BUDGET"
BATCH="$HUNTER_BATCH"

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
    echo "rxn$r already running -- skipping"; continue
  fi
  gpu=$(GPU_FOR "$r")
  log="$LOGDIR/hunter_rxn${r}.log"
  # shellcheck disable=SC2046
  setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" hunter.py \
      --rxn-id "$r" \
      --boltz-budget "$BUDGET" \
      --batch-size "$BATCH" \
      $(opts_for "$r") \
      >> "$log" 2>&1 < /dev/null &
  disown
  echo "rxn$r -> GPU $gpu  (log: $log)"
  sleep 2
done

echo
echo "watch:     tail -f $LOGDIR/hunter_rxn2.log"
echo "progress:  grep 'round .* done' $LOGDIR/hunter_rxn*.log | tail"
echo "operators: grep 'by strategy' $LOGDIR/hunter_rxn*.log | tail"
echo "stop:      ./run_all.sh stop"
