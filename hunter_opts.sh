#!/usr/bin/env bash
# Per-reaction hunter.py settings, shared by run.sh (pm2) and run_all.sh (nohup).
# Sourced, never executed. Keep the two launchers in sync by editing ONLY here.
#
# Derivation of every value is documented in the header of run_all.sh.

HUNTER_BUDGET=150      # Boltz molecules per round (~38 min/round at ~15 s/molecule)
HUNTER_BATCH=10

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
