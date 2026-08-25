# Why local scores beat validator scores — and what to do about it

Written against `score_results_2.sqlite` (rxn 2, target P40261) with everything
below measured on this machine, not assumed.

## The four leaks, largest first

### 1. Novelty erosion — your top-20 is mostly unsubmittable

The validator only accepts a molecule whose maximum Tanimoto similarity to the
HuggingFace Submission-Archive is **below `config['max_similarity_to_historical']`,
which is 0.7**. Measured against the live 31,045-molecule archive:

| | |
|---|---|
| top-300 scoring molecules already archived at sim ≥ 0.99 | **78%** |
| top-20 *available* molecules that pass the 0.7 rule | **1 of 20** |
| rank of the 20th genuinely submittable molecule | **242** |
| naive available top-20 sum | 2.29677 |
| actually submittable top-20 sum | 2.17346 |

Two causes, both fixed in this repo:

* **Threshold drift.** `neurons/genetic.py`, `neurons/component_exhaust.py` and
  `neurons/late_stage_search.py` hardcoded `0.9`; `miner/miner.py` had no novelty
  check whatsoever. Everything mined in the 0.7–0.9 band was unsubmittable.
* **Archive staleness.** The archive grows every epoch, but a long-running
  searcher loaded it once at startup. Molecules scored on 2026-08-25 are 0%
  already-archived; molecules scored on 2026-08-14 are 97%. Your DB's high-score
  region is a graveyard of molecules that have since been consumed — including by
  your own submissions.

Note `available=FALSE` only ever tracked *your* submissions. Anything submitted by
someone else stayed at the top of your list looking perfectly good.

### 2. Winner's curse — you are selecting noise, not signal

A Boltz score is a **draw, not a property**. `boltz.main.predict` calls
`seed_everything(seed)` once, then consumes the RNG sequentially as records stream
through a dataloader fed by `glob("*")`. A molecule's noise therefore depends on
the seed *and* on which molecules share its run and in what order. Re-scoring the
same 6 molecules at the **same seed 68** moved them by up to 0.0127 (9%) and
completely reshuffled their ranking.

Measured per-molecule spread over 6 independent draws each: **σ ≈ 0.00195 median,
ranging 0.00122 → 0.01384 — an 11.3x spread in stability between molecules.**

The trap is concrete. `rxn:2:69935:124690` was stored at **0.13658**, 4th best in
the DB. Re-measured six times its true mean is **0.12077 — the worst of the six
molecules tested** — with σ = 0.01384 and a range of 0.0371. It got into the top
by drawing well once. Compare `rxn:2:71886:173019`: σ = 0.00122, rock solid.
Across all six, re-measurement came in **0.00194 below** the stored draw on
average, i.e. the stored values were systematically lucky.

Now the arithmetic that matters. Scoring 45,759 molecules once each and keeping
the best 20 does not select the 20 best molecules — it selects the 20 luckiest
draws. Expected luck in a top-20 of 45,759 is **+3.58σ per molecule**:

```
+3.58 x 0.00195 = +0.0070 per molecule  (~6% of a 0.115 score)
x 20             = +0.139 on the top-20 sum
```

That premium does not survive the validator's independent re-draw. Meanwhile the
run-to-run spread of the *sum itself* is only `sqrt(20) x σ ≈ 0.0085` — so the
**bias is ~15x larger than the noise it came from.** Fixing selection matters far
more than shaving variance.

### 3. Repetition penalty — popular molecules get divided

`neurons/validator/ranking.py` divides each molecule's score by the number of
UIDs that submitted it:

```python
mol_score = mol_score / (config['molecule_repetition_weight'] * molecule_name_counts[name])
```

Your submittable top-20 averages ≈ 0.1087/molecule. If validator scores land near
0.05, a divisor of 2 accounts for it exactly. Convergent searches find convergent
molecules — prefer molecules others are unlikely to reach.

### 4. Concurrent searchers corrupt each other's batches

`BoltzWrapper.__init__` hardcodes one shared workspace:

```python
self.tmp_dir    = os.path.join(NOVA_DIR, "boltz", "boltz_tmp_files")
self.input_dir  = os.path.join(self.tmp_dir, "inputs")
self.output_dir = os.path.join(self.tmp_dir, "outputs")
```

and `predict(data=self.input_dir)` scores **every YAML sitting in that
directory**. Running `miner.py`, `crossover.py`, `genetic.py` and
`orchestrator.py` together therefore means:

* each process's molecules end up inside the other's batch, changing the RNG
  each molecule draws — this *is* leak #2, amplified on purpose;
* `_cleanup_files()` in one process `rmtree`s `outputs/boltz_results_inputs`
  while another is mid-run.

Observed directly here: a measurement run died with
`FileNotFoundError: .../boltz_tmp_files/inputs` the moment a live
`neurons/crossover.py --rxn_id 2` reached its cleanup.

`novelty.py`, `rescore.py` and `backfill_rescore.py` isolate themselves
(`boltz_tmp_files/iso_<pid>/`) so they are safe to run next to a live miner.
**Give each of your own searchers its own workspace the same way**, or run them
one at a time — otherwise every score in the DB was drawn under conditions you
cannot reproduce, and the variance in leak #2 is larger than it needs to be.

## The fix, in order

```bash
# 1. Re-check the DB against today's archive; mark the stale rows unavailable.
#    Run this every epoch. Start with --dry-run to see the damage.
python3 novelty.py --rxn-id 2 --audit --dry-run
python3 novelty.py --rxn-id 2 --audit

# 2. See what a rescore would cost before spending the GPU time.
python3 backfill_rescore.py --rxn-id 2 --novel-only --dry-run

# 3. Confirm the shortlist: 2 extra draws each at seed 68, score replaced by
#    the 3-draw mean. ~16 s/prediction, so 100 x 2 = 200 preds is roughly 55 m.
python3 backfill_rescore.py --rxn-id 2 --novel-only --limit 100
```

Confirmation rewrites `score` with the mean of three draws and sets
`rescored = TRUE` so the molecule is never re-drawn. The individual draws stay in
`molecule_replicates`, so the spread is always recoverable. Molecules the run
was interrupted on resume correctly: they top up to three draws rather than
restarting.

Useful flags: `--top-submittable N` (rank by what can actually be submitted
rather than by raw score — use it when the high scorers are all already
archived), `--threshold` (default 0.1), `--chunk` (commit granularity, default
50), `--no-detail` (suppress the per-molecule table), `--db PATH` (work on a
copy first).

`variance_rescore.py` — LCB ranking on `mu - lambda*sigma`, plus `--diagnose`
and `--revert` — was removed as redundant with the above. If you want
risk-adjusted ranking back rather than plain means, that is where it lived.

## One-off: confirm the DB you already have

The searchers confirm their own winners from now on, but everything already in
the DB was scored once. Fix that before starting continuous mining:

```bash
# see the plan without scoring anything
python3 backfill_rescore.py --rxn-id 2 --dry-run

# the highest-scoring 200 first (~1.8 h) — safe to stop and resume
python3 backfill_rescore.py --rxn-id 2 --limit 200

# everything above 0.1 that can still be submitted (recommended)
python3 backfill_rescore.py --rxn-id 2 --novel-only
```

It takes every `available=TRUE` molecule scoring above `--threshold` (0.1), gives
it two more draws at seed 68 in batches of 10, replaces the score with the mean
of the three, and flags it `rescored=TRUE`. Highest scores go first, so stopping
early still leaves the molecules that matter confirmed.

Measured on this DB (rxn 2, 46k rows):

| selection | molecules | predictions | est. time |
|---|---|---|---|
| `available=TRUE`, score > 0.1 | 1,242 | 2,484 | ~11.0 h |
| the same, **`--novel-only`** | **441** | 882 | **~3.9 h** |

`--novel-only` drops the 801 molecules that already fail the validator's 0.7
similarity rule. They cannot be submitted whatever they score, so confirming them
is pure waste — that flag is worth 7 hours.

Work is committed per `--chunk` (default 50), so killing it and re-running picks
up where it stopped; already-confirmed molecules are skipped. It runs in an
isolated Boltz workspace, so it is safe to run while your miners are going.

## Automatic confirmation inside every searcher

`orchestrator.py`, `miner/miner.py`, `neurons/crossover.py`, `neurons/genetic.py`
and `neurons/late_stage_search.py` now confirm their own winners at the end of
each round, via the shared `rescore.py`:

* molecules scoring **> `CONFIRM_SCORE_THRESHOLD` (0.1)** are re-scored
  **`CONFIRM_EXTRA_ROUNDS` (2)** more times, in batches of 10;
* the stored score becomes the **mean of all 3 draws**, in memory and in the DB,
  before it reaches the surrogate, the top pool or the DPEX populations;
* the molecule is then flagged **`rescored = TRUE`** in `scored_molecules`, and
  every later round skips it — confirmation is paid exactly once per molecule;
* each draw is kept in `molecule_replicates` (keyed
  `molecule_name, seed, draw_idx`), so per-molecule sigma accumulates for free.

Molecules at or below the threshold keep their single draw and cost nothing:
they cannot reach a 20-molecule submission anyway.

All passes run at **seed 68** — the seed the searchers already use — so a
confirmation draw is taken under the same conditions as the original.

A fair worry about a fixed seed is that the repeat passes might return identical
numbers, making the average pointless. Measured directly (`varlab/same_seed_probe.py`,
5 molecules scored 3x at seed 68 in the *same* batch):

| molecule | pass 1 | pass 2 | pass 3 | spread |
|---|---|---|---|---|
| rxn:2:69935:124690 | 0.118632 | 0.112284 | 0.108438 | **0.010194** |
| rxn:2:160183:137856 | 0.146793 | 0.141241 | 0.141286 | 0.005552 |
| rxn:2:149894:137856 | 0.129054 | 0.131787 | 0.130814 | 0.002733 |
| rxn:2:150258:137856 | 0.140786 | 0.139811 | 0.142281 | 0.002470 |
| rxn:2:71886:173019 | 0.135139 | 0.137484 | 0.135901 | 0.002345 |

Boltz is **not** deterministic even at a fixed seed with a fixed batch — the run
never calls `torch.use_deterministic_algorithms`, so GPU kernel non-determinism
alone moves a score by ~0.0025 typically and 0.0102 at worst. Averaging at seed
68 is therefore meaningful, and the spread is comparable to what varying the seed
produced (sigma 0.00122-0.01384), so little is lost by holding it fixed.

Note which molecule tops both tables: `rxn:2:69935:124690` is unstable however
you probe it. That is the signal worth selecting against.

Cost: proportional to how many molecules clear 0.1, and paid once each. A round
with 40 qualifying molecules adds roughly 20 minutes at ~16 s/prediction, and
those 40 are never re-scored again.

## How many replicates?

Monte Carlo over the real distributions (3,715 submittable scores as mu,
per-molecule sigma sampled from the measured 0.00122-0.01384 range). "% of gap"
is how much of the distance between one-shot selection and a perfect oracle
gets recovered.

Shortlist fixed at 120:

| K | predictions | true top-20 sum | % of achievable gain |
|---|---|---|---|
| 0 (today) | 0 | 1.9306 | 0% |
| 1 | 120 | 2.0679 | 56.5% |
| 2 | 240 | 2.1027 | 70.9% |
| **3** | **360** | **2.1226** | **79.1%** |
| 5 | 600 | 2.1402 | 86.3% |
| 8 | 960 | 2.1527 | 91.4% |
| oracle | - | 2.1735 | 100% |

**K=3 is not too little.** It captures 79% of everything replication can buy.
K=3 -> K=5 adds 7 points for 67% more GPU, and the curve is flat after that.
Because replicates accumulate in `molecule_replicates`, starting at 3 costs you
nothing: top up later and the estimate just gets sharper.

What is clearly wrong is K=1 or K=2 — at that point the mean is still too noisy
to re-rank reliably.

If you are GPU-constrained, **cut the shortlist, not the replicates.** At a fixed
budget the optimum sits near M=80-100:

| budget | best split | % of gap | vs same budget at K=3 |
|---|---|---|---|
| 240 preds | M=80 x K=3 | 69.9% | — |
| 360 preds | M=90 x K=4 | 76.5% | M=120 x K=3 = 72.7% |
| 600 preds | M=100 x K=6 | 82.4% | M=200 x K=3 = 75.7% |

Widening the shortlist past ~120 mostly adds molecules that will never make the
final 20, while shallow replication leaves too much noise to rank what is there.
Defaults are therefore `--top 100 --replicates 3`.

One detail worth knowing: the tool deliberately ignores the original one-shot
score when averaging. Folding it in looks like a free extra draw but costs
**-0.0225** at K=3 — that draw is *why* the molecule was shortlisted, so
including it re-imports the winner's curse.

## Answering "how do I find low-variance molecules?"

σ is **not** uniform — the measured spread across molecules was 11.3x
(0.00122 to 0.01384). Stability is a real, molecule-specific property you can
select on, which is what the subnet owner meant.

* **Directly:** confirmation measures σ per molecule (kept in
  `molecule_replicates`) and ranks on
  `mu - λσ`. This is the ground truth.
* **For free, on every molecule you already score:** Boltz returns two affinity
  ensemble heads (`affinity_probability_binary1/2`, `affinity_pred_value1/2`) plus
  pose confidence (`ligand_iptm`, `complex_plddt`, `confidence_score`).
  `BoltzWrapper.per_molecule_components` has always computed these and every
  miner threw them away. They are now persisted in `molecule_replicates`.
  Head disagreement and low `ligand_iptm` are cheap predictors of an unstable
  molecule — use them to shortlist before spending replicates.

Physically this is what you would expect: a ligand the model cannot place
confidently gets a different pose on each diffusion draw, and the affinity head
reads a different structure each time.

## Files

| file | role |
|---|---|
| `novelty.py` | archive-backed novelty guard (config threshold, disk cache, `refresh()`) + DB audit |
| `backfill_rescore.py` | one-off confirmation of a DB you already have: replicate scoring, σ estimation, per-molecule reporting |
| `score_store.py` | canonical DB layer + `molecule_replicates` table and consensus columns |
| `varlab/replicate_probe.py` | the reproducibility experiment behind these numbers |
| `varlab/submittable_scan.py` | what fraction of the DB is actually submittable |
| `varlab/replicate_budget_sim.py` | the Monte Carlo behind the replicate-count table |
| `varlab/same_seed_probe.py` | proves Boltz is non-deterministic at a fixed seed |
| `rescore.py` | the round-end confirmation pass used by all five searchers |
| `backfill_rescore.py` | one-off confirmation of the DB's existing high scorers |
