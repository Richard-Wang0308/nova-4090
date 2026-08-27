# Phase 2 — Why the orchestrator searches badly

All numbers measured on the real DBs, not asserted.

### 2.1 The component prior has collapsed to near-uniform

`ComponentStats._aggregate` shrinks each component toward `global_mean` with
`prior_n = 5`, then `_role_probs` weights by `exp(0.7 * z)` and mixes 35%
uniform.

| | rxn1 | rxn2 |
|---|---|---|
| global_std | 0.0364 | 0.0373 |
| A components with n=1 | 47.7% | 60.5% |
| A components with n<5 (prior dominates) | 89.7% | 93.1% |
| **max/uniform sampling ratio after ε** | **1.69×** | **1.56×** |
| top-1% of components hold | 1.54% of mass | 1.48% of mass |

A perfectly uniform prior would put 1.00% of mass in the top 1%. This is random
sampling with a rounding error of signal — over a **2.0 billion** molecule space.
That is the root cause: the surrogate can only rank what generation proposes.

### 2.2 The surrogate ranks well but cannot extrapolate

Held-out test, 14,000 molecules, trained on 35,000:

* `spearman(mu, truth) = 0.827` — the model genuinely knows what is good
* top-150 by `ei`: **54.4×** enrichment at >0.12, **23.7×** at >0.11
* but `mu.max() = 0.0912` while `frontier(#20) = 0.1213`

The model can never predict a value above the frontier, so
`z = (mu − frontier)/σ` is negative for **100%** of candidates. The EI term
`(mu − t)·Φ` is always negative and the acquisition degenerates.

### 2.3 The composite acquisition is the worst of the four rules

| rule (top 150) | enrichment >0.11 | >0.12 |
|---|---|---|
| `ei` | 23.7× | 54.4× |
| `mu` | 21.7× | 46.7× |
| **`acq` (production)** | **16.4×** | 46.7× |
| random | 0.7× | 0× |

### 2.4 40% of every Boltz round is spent on unproductive slices

`choose_boltz_batch` splits 60% acquisition / 25% σ / 15% diversity:

| slice | n | hits >0.11 | mean score |
|---|---|---|---|
| exploit | 90 | 23 | 0.10333 |
| **σ (most-uncertain)** | **38** | **2** | **0.06637** |
| diversity | 22 | — | selected on `(1−maxsim)+0.15·ucb` |

### 2.5 The DB's own natural experiment

Hit rate above 0.11, by generation source, rxn2:

| source | n | >0.11 rate |
|---|---|---|
| `top20_acquisition` (orchestrator) | 41,788 | 0.82% |
| **`crossover` (elite-anchored)** | **2,693** | **6.91%** |

Elite-anchored generation is **8.4× more Boltz-efficient** at producing
submittable-quality molecules. The orchestrator reaches higher maxima (0.1422 vs
0.1215) but wastes the budget getting there.

### 2.6 Size blindness

`max_heavy_atoms = 10**9` (lines 1258, 1612). Correct for validity — wrong for
scoring, because the score is `(prob_binary − pred_value) / heavy_atom_count`.
rxn1 spends 7.3% of its budget above 40 heavy atoms, mean score 0.026, **zero**
hits >0.12 ever. Best bucket is 30–35.

---

# Phase 3 — Design

## Reframing the objective

The validator sums 20 molecules and divides each by how many UIDs submitted it.
Every molecule I submit enters the archive and can never be submitted again.
So the objective is not "find one 0.16 hero" — it is **a renewable supply of
0.115+ molecules that other miners are not also finding.** Throughput and
non-obviousness matter as much as peak score.

## The five changes, in order of measured leverage

**1. Field-data prior (new — 6.1× free enrichment).**
`data/rxn{N}.csv` is 8,084 validator-scored molecules, **41× enriched** for
submittable quality versus my own DB. Restricting generation to molecules whose
A *and* B are field-elite lifts P(>0.11) from 1.23% to 7.52% at zero Boltz cost.
This is public data from the submission archive — the same source the novelty
guard already pulls.

Build per-position and per-pair statistics from it, and **blend** with my own DB
statistics (my DB has the unbiased negatives the field's CSV lacks).

**2. Rank-based component priors (fixes the collapse).**
Replace `exp(0.7·z)` on shrunk means — which collapses when `global_std` swamps
component spread — with a *rank-based* weight over a component-level UCB:
`ucb_c = mean_c + β·√(log N / n_c)`. Rank weighting cannot collapse regardless
of score scale, and the UCB term makes rarely-tried components genuinely
explorable instead of shrunk to the mean.

**3. Rank on EI; delete the composite `acq`.** Measured 54.4× vs 16.4×.
Reference the EI threshold to an *attainable* quantile of the model's own
predictive distribution, not to a frontier the model cannot reach.

**4. Reallocate the budget.** Drop the σ slice (0.066 mean). 75% exploit,
25% structured exploration that is *archive-aware* rather than merely diverse —
novelty from the archive is what earns, structural novelty for its own sake is not.

**5. Size gating.** Prefer 26–36 heavy atoms, hard-drop >40.

## What I am keeping

The RF/ET surrogate (ρ=0.83 is a good screening model), the score DB layer,
`rescore.confirm_high_scorers` for the winner's-curse correction, and the
novelty guard. This is a targeted repair of generation and selection, not a
rewrite.

---

# Phase 5 — Test results, including where I was wrong

## Correction to 2.1

I called the collapsed component prior "the dominant defect". That overstates it.
Only the `global` strategy (20% of the orchestrator's pool) draws from that flat
prior; `crossover`, `single_anchor`, `pair_anchor` and `local_neighbour` (80%)
are elite-anchored and do not depend on it. The collapse is real and worth
fixing, but it degrades one fifth of generation, not all of it.

## Selection A/B (varlab/ab_selection.py, 3 trials, budget 150)

First run showed +290% on hits >0.11. That was **leakage**: 738 molecules are in
both my DB and the field CSV, and hunter's `pair_bonus` matches those exactly.
With `--drop-field-overlap`:

| selector | hits >0.10 | >0.11 | >0.12 | mean score | top-20 sum |
|---|---|---|---|---|---|
| orchestrator | 45.7 | 11.7 | 0.7 | 0.07137 | 2.21643 |
| **hunter** | **59.3** | **12.7** | 0.7 | **0.09531** | 2.22135 |
| random | 5.7 | 1.0 | 0.0 | 0.04932 | 1.93562 |

Honest verdict: **+30% hits >0.10, +9% hits >0.11, +34% mean score, no change
above 0.12.** Real but modest.

Ablation (`--ablate`) runs hunter's selection rule on the orchestrator's own
features: 12.3 hits >0.11, mean 0.09587 — statistically indistinguishable from
full hunter. **The field-prior features contribute nothing to selection.** The
gain comes from the rule: EI against an attainable reference, no sigma slice,
and the size multiplier.

## Generation A/B — a test that did not work

`varlab/ab_generation.py` holds out part of the DB and measures which
generator's proposals land on it. It reported hunter as 3.6x *worse*. That test
is invalid for this question: the holdout consists of molecules the orchestrator
previously *chose to score*, so it is a sample of the orchestrator's own
acquisition picks. Only 1.3% of proposals land at all, and any generator that
mimics the orchestrator's history scores well by construction. The script is
kept, with the confound documented in its docstring, because the negative result
is informative about the method, not about hunter.

## Generation, measured against an independent oracle

The field CSV is independent of both my DB and either generator, and Phase-1
established (leakage-free) that molecules whose A and B are both field-elite
carry **5.65x** enrichment for P(>0.11) on rxn2.

| | orchestrator | hunter |
|---|---|---|
| proposals using a field-elite A | 13.97% | **24.70%** |
| proposals using a field-elite B | 21.61% | **29.35%** |
| both components field-elite | 1.32% | **5.06%** |
| unique-new yield | 62% | **94%** |
| top-10 A concentration | 5.9% | **2.8%** |

Hunter directs 3.8x more of its proposals into the enriched region while being
*less* concentrated on individual components, so this is better targeting rather
than premature exploitation.

## What remains unverified

Whether this converts into a higher submittable top-20 sum can only be settled
on GPU. Component-prior sharpness (1.56x -> 5.5x), proposal targeting (3.8x) and
selection quality (+34% mean) are all measured; the product of the three is not.
Run both searchers on the same reaction with an equal Boltz budget and compare
real hit rates before committing to either.
