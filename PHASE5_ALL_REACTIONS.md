# Testing hunter.py across all five reactions

Reaction structure, which drives everything below:

| rxn | 3-component | A | B | C | search space | field CSV |
|---|---|---|---|---|---|---|
| 1 | no | 25,728 | 35,774 | — | 920,393,472 | 6,543 |
| 2 | no | 83,307 | 24,165 | — | 2,013,113,655 | 8,084 |
| 3 | **yes** | 817 | 2,731 | 18,330 | 40,898,390,910 | 5,200 |
| 4 | no | 49,108 | 6,959 | — | 341,742,572 | 7,454 |
| 5 | **yes** | 2,054 | 2,874 | 2,874 | 16,965,785,304 | 7,509 |

## Starting position (submittable top-20, validator rules applied)

| rxn | my top-20 sum | field median | gap | my #1 |
|---|---|---|---|---|
| 1 | 1.88064 | 2.06772 | −0.18709 | 0.10061 |
| 2 | 2.12892 | 2.26721 | −0.13829 | 0.10722 |
| 3 | 1.97547 | 2.07081 | −0.09534 | 0.10366 |
| 4 | 2.06042 | 2.31087 | **−0.25045** | 0.10750 |
| 5 | **2.25127** | 2.37006 | −0.11879 | **0.11934** |

rxn5 is the strongest position and rxn4 the weakest. rxn5 is also the only
reaction where a submittable molecule clears 0.11.

## Three-component bugs found and fixed

rxn3 and rxn5 exercised code paths the two-component reactions never touched:

1. **`anchored()` pinned C to the elite set in both halves**, so C was never
   explored. On rxn3 that is crippling — C holds 18,330 of the reaction's
   building blocks against an elite pool of ~283. Fixed by rotating the pinned
   position across every position present. Distinct C values produced on rxn3
   went from 283 to **3,200**.
2. **`pair_mutants()` drew C one at a time and never mutated it.** Field pairs
   are (A,B) only, so C is now drawn fresh, half elite and half explored:
   2,594 distinct C values.
3. **The surrogate ignored C entirely** — `_features` read only A and B
   component scores. Now carries C score and C hi-component evidence.
4. **`three_component` was detected from the field CSV alone**, so local C
   statistics would silently never load if that file were absent. Now detected
   from whichever source has a C column.

## Two of my own defects, found by testing and corrected

**The explore slice was too large.** I shipped 25%, criticising the
orchestrator for spending 40% on unproductive slices. Measured on a held-out
pool, going 0.25 -> 0.00 recovered **13 hits above 0.11 on rxn5 and 5 on rxn2**.
Reduced to 0.10 — not zero, because the validator requires MACCS entropy >= 0.1
across the submitted 20. A single-round test sees this slice's cost but not its
across-round benefit, so 0.10 is a judgement call, not a measured optimum.

**Ranking on expected improvement was wrong.** This is the same defect I
diagnosed in the orchestrator, reintroduced at a smaller scale. My "attainable"
reference was the 99.5th percentile of the model's own predictions — on rxn5,
`ref=0.09989` against `mu.max=0.10441`, so 99.5% of candidates still had a
negative z and EI collapsed back into a sigma term.

Ranking key, hits above 0.11 in the top 150, mean of 3 trials per reaction:

| rxn | hunter `mu` | hunter `ei` | orch `acq` | mu vs acq |
|---|---|---|---|---|
| 1 | **20.3** | 16.0 | 10.7 | **+91%** |
| 2 | **14.3** | 13.3 | 7.3 | **+96%** |
| 3 | **1.7** | 0.0 | 0.0 | +1.7 |
| 4 | 0.3 | 0.7 | 0.3 | tie |
| 5 | **80.3** | 62.7 | 50.3 | **+60%** |

Default changed to `mu`; `--rank-key ei` remains available.

**Budget under-fill.** On a live rxn1 round hunter returned 21 molecules for a
budget of 40. Its top-N by acquisition is concentrated in the productive region,
which is precisely the region other miners have already mined and archived, so
proportionally more of it fails the novelty gate. The screen now widens until
the budget is filled: rxn1 went from 21/40 to 108 submittable from a 400 screen.

## LIVE GPU A/B — real Boltz, all five reactions

Both searchers ran their true generate -> rank -> select pipeline from an
identical DB state, 40 molecules per arm, same novelty gate, same scoring path.
**Zero overlap between arms on every reaction** — they propose entirely
disjoint molecule sets.

This run used the pre-correction configuration (`ei` ranking, and the
under-filling screen on rxn1/rxn3), so it understates hunter.

| rxn | arm | n | mean | best | >0.09 | >0.10 |
|---|---|---|---|---|---|---|
| 1 | orchestrator | 40 | 0.04416 | 0.07711 | 0 | 0 |
| 1 | **hunter** | 21 | **0.05824** | **0.08984** | 0 | 0 |
| 2 | orchestrator | 40 | 0.05035 | 0.10692 | 7 | 2 |
| 2 | **hunter** | 40 | **0.08053** | **0.10765** | **10** | **8** |
| 3 | orchestrator | 40 | 0.05458 | **0.09595** | 2 | 0 |
| 3 | **hunter** | 13 | **0.06732** | 0.09273 | 1 | 0 |
| 4 | orchestrator | 40 | 0.03620 | 0.09563 | 1 | 0 |
| 4 | **hunter** | 40 | **0.06022** | **0.11007** | **2** | **1** |
| 5 | orchestrator | 40 | 0.02986 | 0.08980 | 0 | 0 |
| 5 | **hunter** | 40 | **0.05638** | **0.10241** | **3** | **2** |

hunter wins the mean on **5 of 5** reactions (+23% to +89%), the best molecule
on 4 of 5, and takes **11 hits above 0.10 against 2** — on fewer molecules.

## FINAL offline selection A/B — corrected configuration

`--rank-key mu`, `--explore-frac 0.10`. 3 trials per reaction, budget 150,
held-out pool, every molecule present in the field CSV removed first.

| rxn | base P(>0.11) | hits>0.10 orch -> hunter | hits>0.11 orch -> hunter | mean score orch -> hunter |
|---|---|---|---|---|
| 1 | 0.499% | 50.7 -> **73.0** | 13.7 -> **18.7** | 0.07936 -> **0.09667** |
| 2 | 0.393% | 46.7 -> **63.7** | 11.7 -> **13.7** | 0.07233 -> **0.09673** |
| 3 | 0.068% | 25.7 -> **47.3** | 0.0 -> **2.7** | 0.07799 -> **0.09499** |
| 4 | 0.016% | 13.7 -> **13.7** | 0.0 -> **1.0** | 0.06274 -> **0.08352** |
| 5 | 3.188% | 85.0 -> **122.0** | 62.3 -> **75.3** | 0.08528 -> **0.10753** |
| **total** | | 222 -> **320** (+44%) | 88 -> **111** (+27%) | |

hunter now wins or ties on **all five** reactions. The rxn5 regression that the
`ei` configuration produced (65.7 -> 59.0, a 10% loss) is gone: 62.3 -> 75.3,
a 21% gain.

## Confirming LIVE GPU A/B with the corrected configuration

| rxn | arm | n | mean | best | >0.09 | >0.10 |
|---|---|---|---|---|---|---|
| 2 | orchestrator | 40 | 0.04416 | 0.09562 | 2 | 0 |
| 2 | **hunter** | 40 | 0.08401 | 0.10694 | 17 | 7 |
| 5 | orchestrator | 40 | 0.04125 | 0.10813 | 3 | 1 |
| 5 | **hunter** | 40 | 0.05855 | 0.10288 | 3 | 2 |

rxn2 reproduces strongly: mean +90%, and 17 molecules above 0.09 against 2.
rxn5 is milder than the offline prediction (+42% mean, 2 hits vs 1) and the
orchestrator found the single best molecule there (0.10813 vs 0.10288). At
n=40 per arm, single-molecule extremes are noise; the mean and the hit counts
are the parts to trust.

## What is proven and what is not

Proven, on real Boltz across all five reactions: hunter selects molecules with a
**higher mean score in 5 of 5** reactions, and takes more hits above 0.10.
Proven offline with leakage controls: **+44% hits above 0.10 and +27% above
0.11** across the five reactions combined.

Not proven: that this raises the submittable top-20 *sum*, which is what the
validator pays on. That is a multi-round property — each round's winners are
consumed by the archive — and cannot be established by single-round tests. It
needs hunter running for several epochs against the live submission cycle.


---

## Adaptive confirmation threshold

A fixed `CONFIRM_SCORE_THRESHOLD = 0.1` was wrong on every reaction, in both
directions. What deserves three draws is anything that could enter the
*submittable* top 20, and that bar is reaction-specific and moves as the search
progresses:

| rxn | submittable #20 | a fixed 0.1 |
|---|---|---|
| 1 | 0.09119 | confirms **nothing** — every real hit is missed |
| 2 | 0.10606 | confirms ~19% of a round, most of it unable to reach top 20 |
| 4 | 0.10199 | same waste |
| 5 | 0.11097 | same waste |

`--confirm-threshold auto` (the default) resolves to `submittable #20 - margin`
each round, so it adapts across reactions *and* across progress as #20 climbs.
The margin (default 0.005) exists because the frontier is itself measured from
single draws — Boltz spread at a fixed seed is ~0.0025 typical, 0.0102 worst —
so molecules just under #20 may really be above it.

Guards: `--confirm-floor` (default 0.08) keeps an early or weak round from
confirming everything; `--confirm-max-frac` (default 0.30) raises the threshold
to the matching quantile if too much of a round would qualify, since each
confirmation costs two extra Boltz calls.

Verified live on rxn1 — the reaction where a fixed 0.1 confirmed nothing:

```
[round 1] confirm threshold 0.08619 (auto: submittable #20=0.09119 - margin 0.005)
          -> 1/12 molecules = 2 extra predictions
[CONFIRM] averaged 1 molecules | mean change +0.000114 | 1 marked rescored
```

The round-completion line now reports the hit rate against the submittable
frontier rather than a fixed number, so "0% hit rate" means what it says.

## SQLite corruption — what happened and how to recover

`score_results_3.sqlite` developed B-tree corruption during the concurrent
testing above (`invalid page number` in Tree 2, the `scored_molecules` table
itself). It kept answering simple `SELECT`s; the failure only surfaced on a
query that touched the damaged pages.

**Cause: not established.** Disk had 765 GB free and dmesg showed no filesystem
errors. The plausible candidates are a process killed mid-checkpoint, or
`tools/check.py`'s `fix_available_column()` — which does a full
CREATE/INSERT-SELECT/DROP/RENAME table rebuild — running while another process
held the database. Do not run `tools/check.py` against a database a searcher is
writing.

**Recovery** (64,835 of 64,840 rows salvaged; max score, `>0.11` count and
metadata all matched the pre-corruption values exactly):

```python
# walk by rowid so one bad page loses only its own rows, committing as you go
src = sqlite3.connect("score_results_N.sqlite")
dst = sqlite3.connect("recovered.sqlite")
# copy DDL from src's sqlite_master, then for each 500-rowid block:
#   try the block; on failure retry row by row and skip what will not read
# finally copy metadata and molecule_replicates, and PRAGMA integrity_check
```

The full script is `scratchpad/recover3b.py` from that session.

**Prevention:** hunter now runs `PRAGMA quick_check` at startup and refuses to
write to a damaged database rather than compounding the damage. Check all five
at any time with:

```bash
for r in 1 2 3 4 5; do
  printf "rxn$r: "
  ./.venv/bin/python -c "import sqlite3;print(sqlite3.connect('score_results_$r.sqlite').execute('PRAGMA integrity_check').fetchone()[0])"
done
```
