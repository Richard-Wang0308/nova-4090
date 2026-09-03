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
# PER-REACTION SETTINGS — RE-DERIVED 2026-09-03 against all five score DBs and
# the field's own submissions (data/rxn{N}.csv, ~10k validator-scored molecules
# each, 69-95 epochs).
#
# THE PREVIOUS DERIVATION IS VOID. It was built on rxn5 being 95% unsubmittable
# at similarity 0.6 and having to scan 418 molecules deep to field twenty. The
# AVAILABLE pool is now clean on all five -- every molecule that failed 0.6 has
# been marked unavailable, and each frontier is reached at scan depth 21.
#
# But do not read that as "novelty is solved". Both easy measurements of it are
# tautological: the top of the available pool is clean because the dirty rows
# were culled, and everything in the DB passed the novelty screen before it was
# ever scored. The only honest number is hunter's own screening log, and it says
# 70% of RANKED candidates still fail the 0.6 gate (rxn2, 168 rounds: median
# pass 29.9%, worst 22.8%, and trending DOWN -- 29.3, 25.0, 24.8, 22.8, 24.6%
# over the most recent rounds). Novelty is still the binding filter on
# generation; what changed is that it no longer poisons the submittable pool.
#
# WHERE EACH REACTION STANDS, against the field's best-20-of-all-miners ceiling:
#
#   rxn  our top20  field ceiling  gap      recent median  recent p99
#    1      2.1949         2.1561  -0.0388          0.0586      0.1027
#    2      2.1121         2.3543  +0.2422          0.0505      0.0923
#    3      1.8580         2.1056  +0.2476          0.0608      0.0869
#    4      2.0363         2.3392  +0.3029          0.0374      0.0866
#    5      2.0969         2.4624  +0.3655          0.0442      0.1044
#
# The "ceiling" is the best 20 molecules ANY miner submitted in an epoch, taken
# from ~107-117 pooled molecules per epoch. The CSV carries no uid, so single
# submissions cannot be reconstructed and this is NOT a winning submission --
# it is a composite no individual miner achieved, and every gap above is
# therefore overstated. It is still a fair RANKING between reactions, because
# all five pool a near-identical 5.3-5.8 miners per epoch.
#
# Even against that inflated bar rxn1 comes out ahead, which is the finding that
# matters: the old run.sh guidance said rxn1 "needs ~44x more hits" and rxn4 was
# "at parity". Both have reversed.
#
# COMPONENT COVERAGE is what the prior settings key off, and it is nothing like
# the numbers the old header recorded (rxn3 "99.3%/98.1%", rxn5 "95.2%/94.3%"):
#
#   rxn  A pool   A cov   B pool   B cov   C pool   C cov
#    1   25,728   30.4%   35,774   18.0%       --      --
#    2   83,307   21.1%   24,165    7.8%       --      --
#    3      817   73.1%    2,731   49.2%   18,330   22.7%
#    4   49,108    8.4%    6,959   62.0%       --      --
#    5    2,054   48.7%    2,874   67.7%    2,874   67.0%
#
# A reaction with most of its building blocks never tried has more to gain from
# a flat prior than from a sharp one, because a sharp prior can only re-rank
# what it has already seen. rxn2 (7.8% of B) and rxn4 (8.4% of A) are the two
# with the most unexplored ground, and they get the flattest priors. rxn5 is the
# best covered on every position and gets the sharpest.
#
# STRICT-POOL-MULT sets how many ranked candidates get novelty-screened
# (budget x mult) to yield `budget` novel ones. It CANNOT be derived from the
# score DBs: everything in them already passed the screen, so any sample drawn
# from them reports a pass rate near 100% and would justify a mult of 1 that
# starves every round. The measurement has to come from hunter's screening log.
#
# rxn2, 168 rounds: median pass 29.9%, worst 22.8% -> 3.3x needed typically,
# 4.4x in the worst round observed, and the rate is falling as the searcher
# works through the novel chemistry near its favoured region. 8x leaves headroom
# for a round worse than any yet seen.
#
# The other four reactions have no usable log history (rxn4 has one round at
# 44%), so there is no evidence on which to differentiate them, and they get the
# same 8x rather than a number invented per reaction. The previous 10/6/10/10/20
# spread was not measured either; rxn5 at 20x was screening twenty times its
# budget with nothing to justify it.
#
# FIELD-WEIGHT: the field CSV is worth far more than the 41x the module header
# claims. P(above that reaction's own winning bar) in the field CSV against our
# DB: rxn1 261x, rxn3 1,613x, rxn4 2,780x, rxn5 4,262x, and rxn2 is unbounded --
# we have ZERO molecules above rxn2's bar of 0.1140. rxn3 was held at 0.3 only
# because its `field_hi` slice was four molecules; field_prior.py now cuts that
# slice by quantile and rxn3 gets 387, so the reason for holding it back is gone.
# ---------------------------------------------------------------------------
opts_for() {
  case "$1" in
    # Ahead of the median winner and the best generator of the five. Do not
    # disturb what is working: moderate prior, tight elite. Coverage is low
    # (30%/18%) so the floor stays generous enough to keep finding new blocks.
    1) echo "--candidate-pool 35000 --block-frac 0.16 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 5.0 --prior-floor 0.15 --elite-quantile 0.90 \
             --field-weight 0.55 --strict-pool-mult 8" ;;
    # 7.8% of its B pool tried -- the least-explored position in any reaction --
    # and not one molecule above its 0.1140 bar. A sharp prior cannot fix that;
    # only trying more of the 24,165 B blocks can. Flattest prior, biggest pool.
    2) echo "--candidate-pool 45000 --block-frac 0.18 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.18 --elite-quantile 0.88 \
             --field-weight 0.60 --strict-pool-mult 8" ;;
    # The flat one: its top 20 spans 0.0009, so it has no upside, only a
    # plateau. Its C position is 22.7% covered of 18,330 -- that is where the
    # unexplored ground is. Broad elite so the plateau is not mistaken for a
    # ranking, and field-weight up now that its hi-slice is 387 not 4.
    3) echo "--candidate-pool 40000 --block-frac 0.16 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 5.0 --prior-floor 0.15 --elite-quantile 0.86 \
             --field-weight 0.55 --strict-pool-mult 8" ;;
    # The weakest generator: recent median 0.0374 against rxn1's 0.0586, and
    # p99 barely above its own frontier. 8.4% of a 49,108 A pool tried. Same
    # prescription as rxn2 and for the same reason -- explore A.
    4) echo "--candidate-pool 45000 --block-frac 0.20 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 4.0 --prior-floor 0.18 --elite-quantile 0.85 \
             --field-weight 0.60 --strict-pool-mult 8" ;;
    # The hardest reaction to win: the field's bar is the highest of the five
    # (ceiling #20 = 0.1191) and our gap is the largest. It is also the best
    # covered (49/68/67%), so there is least left to discover at the component
    # level, and its p99 of 0.1044 says the upside is real when it lands -- both
    # arguments for a sharper prior. Sharper than the others, then, but not the
    # extreme: with 70% of ranked candidates still failing the novelty gate,
    # concentrating hard on known-good components is how a searcher walks back
    # into the chemistry the archive already covers.
    5) echo "--candidate-pool 30000 --block-frac 0.14 --block-k 4 --block-seed-reuse 2 \
             --prior-temperature 6.0 --prior-floor 0.12 --elite-quantile 0.90 \
             --field-weight 0.60 --strict-pool-mult 8" ;;
    *) return 1 ;;
  esac
}
