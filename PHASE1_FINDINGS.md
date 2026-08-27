# Phase 1 — What is actually wrong

## The gap, measured

| | rxn1 | rxn2 |
|---|---|---|
| my submittable top-20 sum | 1.88064 | 2.12892 |
| field median (last 10 epochs) | 2.06772 | 2.26721 |
| **gap** | **-0.18709** | **-0.13829** |
| my best submittable molecule | 0.10061 | 0.10722 |
| field's typical #20 | ~0.111 | ~0.110 |

My single best submittable molecule scores about what the field's *worst*
submitted molecule scores.

## It is NOT a scoring problem

539 (rxn1) and 738 (rxn2) of the field's submitted molecules are also in my DB.
Comparing my Boltz score against the validator's for those:

* mean(mine - validator) = **+0.0012**, median +0.0007, sd 0.0084
* Pearson 0.83 (rxn1) / 0.59 (rxn2), Spearman 0.88 / 0.68

My Boltz agrees with the validator. The molecules they submit, I score the same.

## It is NOT elite-reactant coverage

Of the field's A-components appearing in >0.125 molecules I have already scored
94% (rxn1) and 75% (rxn2). I am using the right building blocks.

## It IS combination selection

rxn2, A=125845 — the field's most-used reactant:

| | field | me |
|---|---|---|
| molecules scored with it | 799 | 751 |
| distinct B partners | 799 | 751 |
| **overlapping partners** | **6** | |
| p50 | 0.08438 | 0.06201 |
| p90 | 0.11467 | 0.08840 |
| max | **0.15131** | **0.10855** |

Same reactant, same effort, near-zero partner overlap, and their whole
distribution is shifted up — not just the max. Of the 19 B partners the field
used to exceed 0.13 with this A, I tried 1.

## Search efficiency

| | rxn1 | rxn2 |
|---|---|---|
| library size | 920,393,472 | 2,013,113,655 |
| I have scored | 34,443 (0.0037%) | 49,935 (0.0025%) |
| hit rate >0.12 | 0.285% | 0.086% |
| hit rate >0.13 | 0.055% | 0.010% |
| Boltz calls per 20 hits | ~7,000 | ~23,000 |

At ~16 s/prediction that is 31 h (rxn1) and 103 h (rxn2) of GPU per refreshed
top-20 — and the archive consumes hits as fast as they are found.

## Secondary waste

* rxn1 spends **7.3%** of its Boltz budget on >40-heavy-atom molecules, whose
  mean score is 0.026 and which have produced **zero** hits above 0.12.
  `orchestrator.py` sets `max_heavy_atoms = 10**9` deliberately (lines 1258,
  1612). Correct for *validity*, wrong for *scoring*: the score is
  `(prob_binary - pred_value) / heavy_atom_count`, so size is a direct divisor.
  Best bucket is 30-35 heavy atoms.

## CORRECTION (added after leakage testing)

An earlier version of this document claimed a "6.1x free enrichment" from
restricting to field-elite components, and quoted spearman(field-B-max, my
score) = +0.36. Both were measured on a set that still contained the 539/738
molecules present in *both* my DB and the field CSV. Those molecules are
field submissions, so they are disproportionately good, and including them
inflated everything — the base rate itself fell from 1.23% to 0.39% once they
were removed.

Leakage-free numbers (evaluation set excludes every molecule in the field CSV):

| | rxn1 | rxn2 |
|---|---|---|
| base P(>0.11) | 0.502% | 0.394% |
| both components field-max >0.125 | 2.44x | **5.65x** |
| spearman(field A max, my score) | +0.069 | +0.007 |
| spearman(field B max, my score) | +0.099 | **+0.256** |
| blended field+local prior, top 1% | 4.69x | 5.68x |
| field-only prior, top 1% | 0.59x | 0.52x |

Two things follow, and they shape the design:

1. The field prior is real but **~5x, not 56x**, and it is much stronger for
   rxn2 than rxn1. The B position carries nearly all of the transferable signal.
2. A field-only prior is **worse than random** at the top of the ranking. The
   local DB supplies the negatives that make the blend work. Neither source is
   sufficient alone.

## The real question for Phase 2

Random sampling of 2e9 molecules at a 0.086% hit rate is what an *uninformed*
search looks like. The orchestrator has component priors, a surrogate and an
acquisition function that are all supposed to beat that. Do they?
