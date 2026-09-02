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

# ---------------------------------------------------------------------------
# PER-REACTION SETTINGS, RE-DERIVED AT max_similarity_to_historical = 0.6
#
# The five reactions are NOT in the same situation, and the 0.7 -> 0.6 novelty
# change (epoch 24876) pulled them further apart. Measured over the top 500
# available molecules of each score DB against the 48,382-fingerprint P40261
# archive, together with the last 2,000 molecules each searcher actually
# scored:
#
#   rxn  passes 0.6   lost vs 0.7   median score   > frontier   scan depth to 20
#     1       80.0%         19.8%         0.0436        0.25%                 34
#     2      100.0%          0.0%         0.0697        0.90%                 21
#     3      100.0%          0.0%         0.0665        0.20%                 21
#     4       30.8%         69.2%         0.0500        0.85%                 54
#     5        5.0%         95.0%         0.0642        1.50%                418
#
# Two different failure modes, needing opposite corrections:
#
#   ARCHIVE-SATURATED (rxn4, rxn5). Their chemistry sits inside the 0.6-0.7
#   similarity band that just became unsubmittable -- median similarity to the
#   archive is 0.634 and 0.677 against 0.579 for rxn2. rxn5 must now scan 418
#   molecules deep to field twenty. Yet both carried the most EXPLOITATIVE
#   settings in this file (rxn5: temperature 8.0, floor 0.05, elite 0.92), which
#   is exactly backwards: a sharp prior keeps re-mining the components that
#   produced the molecules everyone else has already submitted. They need to be
#   pushed off the region they have been working.
#
#   UNPRODUCTIVE BUT NOVEL (rxn1, rxn3). Novelty is not their constraint -- they
#   lose little or nothing to 0.6 -- but they generate weak candidates (rxn1's
#   median is the lowest of the five) and almost nothing reaches the frontier.
#   rxn1 already had the most exploratory settings and it is not paying, so the
#   correction is the other direction: concentrate on what its prior says is
#   good instead of sampling broadly and scoring junk.
#
#   rxn2 is the healthy one on every axis and is left alone, as the control.
#
# --strict-pool-mult deserves its own note: it sets how many ranked candidates
# get novelty-screened per round (budget x mult) to end up with `budget` novel
# ones. At a 5% pass rate rxn5 needs roughly 20x to avoid starving the round;
# it was at 8. Screening is CPU work against the cached archive, not Boltz, so
# raising it costs wall clock on the host rather than GPU budget.
#
# These are reasoned from the DBs, not A/B tested. Give them a few rounds and
# re-read the two columns that matter: "> frontier" should rise on rxn1/rxn3,
# and "scan depth to 20" should fall on rxn4/rxn5.
# ---------------------------------------------------------------------------
opts_for() {
  case "$1" in
    # Novel enough (80% passes 0.6) but the weakest generator of the five:
    # median 0.0436, and only 0.25% of what it scores reaches the frontier.
    # It was already the most exploratory config here, so tighten instead:
    # elite 0.88 -> 0.92, temperature 4.0 -> 6.0, floor 0.15 -> 0.10.
    1) echo "--candidate-pool 40000 --block-frac 0.16 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 6.0 --prior-floor 0.10 --elite-quantile 0.92 \
             --field-weight 0.5 --strict-pool-mult 10" ;;
    # The control. Healthiest on every axis: nothing lost to 0.6, best median,
    # frontier reached at scan depth 21. Unchanged deliberately.
    2) echo "--candidate-pool 30000 --block-frac 0.15 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.15 --elite-quantile 0.90 \
             --field-weight 0.5 --strict-pool-mult 6" ;;
    # Loses nothing to 0.6, but the worst frontier yield of all five (0.20%)
    # while running the sharpest prior in the file. Exploitation is not the
    # problem it looked like: loosen to 5.0/0.12/0.88 and widen each block
    # (k 3 -> 4, so 25 cells instead of 16).
    3) echo "--candidate-pool 40000 --block-frac 0.14 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 5.0 --prior-floor 0.12 --elite-quantile 0.88 \
             --field-weight 0.3 --strict-pool-mult 10" ;;
    # 69% of its top 500 became unsubmittable at 0.6 and scan depth tripled to
    # 54. Diversify: temperature 5.0 -> 4.0, floor 0.12 -> 0.15, elite 0.90 ->
    # 0.86, more of the pool from the 2-D block scan, and a wider pool to
    # screen from. Pass rate 31% means mult 6 was only just adequate.
    4) echo "--candidate-pool 40000 --block-frac 0.20 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.15 --elite-quantile 0.86 \
             --field-weight 0.5 --strict-pool-mult 10" ;;
    # The urgent one: 95% of its top 500 is now unsubmittable and it scans 418
    # deep to field twenty, while running the most exploitative settings in the
    # file. Every lever goes the other way -- temperature 8.0 -> 4.0, floor
    # 0.05 -> 0.18, elite 0.92 -> 0.85, block-frac 0.14 -> 0.22, block-k 2 -> 4
    # -- and mult 8 -> 20 so a 5% pass rate still fills the round.
    5) echo "--candidate-pool 45000 --block-frac 0.22 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.18 --elite-quantile 0.85 \
             --field-weight 0.5 --strict-pool-mult 20" ;;
    *) return 1 ;;
  esac
}
