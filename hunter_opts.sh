#!/usr/bin/env bash
# Per-reaction hunter.py settings, shared by run.sh (pm2) and run_all.sh (nohup).
# Sourced, never executed. Keep the two launchers in sync by editing ONLY here.
#
# Derivation of every value is documented in the header of run_all.sh.

HUNTER_BUDGET=150      # Boltz molecules per round, single-GPU path (MULTI=0)

# Molecules per round PER WORKER, used by run.sh's autoscale_budget when the
# multi-GPU pool is active: budget = this x workers.
#
#   1 worker   ->  300      2 workers ->  600
#   8 workers  -> 2400     16 workers -> 4800
#
# Because it scales with the worker count, a round takes about the same WALL
# CLOCK on any box -- ~43 min at the measured 22 s setup + 7.8 s/molecule with
# chunks of 30. Raising this lengthens the round everywhere rather than only on
# large machines, which is the point: the round is the unit at which the
# surrogate re-fits and the submittable frontier is recomputed.
#
# A hard override is available and skips the scaling entirely:
#   BOLTZ_BUDGET=2400 ./run.sh start 2
HUNTER_BUDGET_PER_WORKER=300

# Molecules handed to Boltz per call. Raised from 10 to 60.
#
# boltz.main.predict() reloads the structure and affinity checkpoints and
# re-checks the manifest on EVERY call. Measured on the live rxn2 log:
#
#   12 s  manifest check + structure checkpoint   (GPU idle)
#   41 s  structure prediction                    (GPU busy)
#   10 s  affinity checkpoint                     (GPU idle)
#   37 s  affinity prediction                     (GPU busy)
#  ----
#  101 s  for 10 molecules  ->  22 s of fixed setup per call, 22% of wall clock
#
# That 22 s is paid once per call regardless of batch size, so:
#
#   batch 10 -> 22% overhead      batch 30 -> 8.6%
#   batch 20 -> 12%               batch 60 -> 4.6% (split into 2 chunks: 8.6%)
#
# This is worth ~15% on a SINGLE GPU, with or without multi/. The cost is that a
# crash loses the in-flight batch rather than 10 molecules, and that batch
# composition perturbs individual scores (see multi/README.md) -- so do not
# compare scores across a change to this value.
HUNTER_BATCH=60

opts_for() {
  case "$1" in
    1) echo "--candidate-pool 40000 --block-frac 0.16 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.15 --elite-quantile 0.88 \
             --field-weight 0.5 --strict-pool-mult 10" ;;
    2) echo "--candidate-pool 30000 --block-frac 0.15 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.15 --elite-quantile 0.90 \
             --field-weight 0.5 --strict-pool-mult 6" ;;
    3) echo "--candidate-pool 40000 --block-frac 0.14 --block-k 3 --block-seed-reuse 2 \
             --prior-temperature 8.0 --prior-floor 0.05 --elite-quantile 0.92 \
             --field-weight 0.3 --strict-pool-mult 10" ;;
    4) echo "--candidate-pool 30000 --block-frac 0.17 --block-k 3 --block-seed-reuse 2 \
             --prior-temperature 5.0 --prior-floor 0.12 --elite-quantile 0.90 \
             --field-weight 0.5 --strict-pool-mult 6" ;;
    5) echo "--candidate-pool 30000 --block-frac 0.14 --block-k 2 --block-seed-reuse 2 \
             --prior-temperature 8.0 --prior-floor 0.05 --elite-quantile 0.92 \
             --field-weight 0.5 --strict-pool-mult 8" ;;
    *) return 1 ;;
  esac
}
