#!/usr/bin/env python3
"""
field_prior.py — learn which building blocks actually win, from the validator's
own scores of other miners' submissions.

WHY THIS EXISTS
---------------
`data/rxn{N}.csv` holds `molecule_name, final_score, epoch` for every molecule
the field submitted. Those scores come from the validator, and a submitted
molecule is one some miner already selected as their best. Measured on rxn2:

    my score DB      44,706 molecules   P(score > 0.11) =  1.23%
    data/rxn2.csv     8,000 molecules   P(score > 0.11) = 50.45%

That is a **41x enrichment** for submittable-quality molecules, on exactly the
distribution we are trying to sample from. My own DB is mostly a record of
mediocre molecules; this file is a record of winners.

Restricting candidate generation to molecules whose A *and* B components are
both field-elite lifts P(>0.11) from 1.23% to 7.52% — a **6.1x enrichment before
a single Boltz call**, which multiplies with whatever the surrogate adds on top.

WHAT IT DOES NOT TELL YOU
-------------------------
The CSV is selection-biased: it contains almost no negatives, because nobody
submits a molecule they know is bad. A component's *mean* in this file is
therefore not its true mean. Two consequences, both handled below:

  * Use it to rank components, never to predict a molecule's score.
  * Blend it with statistics from the local DB, which has the negatives.

Every molecule in this file is also, by construction, already in the
Submission-Archive and can never be submitted again. The value is entirely in
the components, not the molecules.
"""
from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# A component needs at least this many observations before it may enter the
# elite pool. This was 3, which permanently disqualified any building block
# whose one trial happened to be unlucky. Epoch 24796's winner drew 18 of its 20
# molecules from A in {45185,45224,45235,45257} x B in {6067,6070,6071,6074};
# this DB had scored 45224 exactly once (0.0190), 45257 once (0.0049), 45185
# twice (best 0.0325) and 6074 never -- so at min_obs=3 not one of them could
# ever reach the elite pool that feeds 55% of generation. One observation is
# information, not disqualification; the shrinkage in _ucb is what stops a
# single lucky draw from dominating instead.
MIN_OBS = 1

# --- UCB calibration ---------------------------------------------------------
# Multiplier on a component's standard error, sd/sqrt(n). Expressing the bonus
# in standard errors rather than as beta*sqrt(log(N)/n) makes it self-calibrating
# per reaction: it lands in the same units as the spread of the shrunk means it
# has to compete with, so it needs no per-reaction tuning.
UCB_BETA = 0.5

# Shrinkage strength toward the position's grand mean, in pseudo-observations.
# n=1 lands ~89% of the way to the grand mean, n=8 halfway, n>=100 essentially
# on its own mean.
PRIOR_M = 8.0

# How much of a component's sample-size-corrected best observation survives into
# its score. See _ucb for why the raw maximum cannot be used directly.
UPSIDE_WEIGHT = 0.15

# Scores at or below this are Boltz failures / degenerate structures, not signal.
# The reactant summaries contain entries like avg_score = -17.79 from these.
MIN_VALID_SCORE = -1.0

# Fallback `hit_threshold` for a reaction with no field CSV. With a CSV the
# threshold is DERIVED per reaction instead (see _derive_hit_threshold): 0.11 is
# a single global number and it is wrong for every reaction, because the score
# distribution of what gets submitted differs by reaction. Derived values, and
# where each sits in its own reaction's field distribution:
#
#     rxn1 0.1035 (p69)   rxn2 0.1140 (p73)   rxn3 0.1025 (p66)
#     rxn4 0.1114 (p80)   rxn5 0.1191 (p77)
#
# At a flat 0.11 the hit rate meant "top 12%" on rxn3 and "top 49%" on rxn5, so
# the same feature carried a different meaning in every reaction's prior. The
# derived bar lands consistently around p70-p80, which is what makes it
# comparable across reactions -- that consistency is the property being bought
# here, NOT any claim to be a winning threshold.
DEFAULT_HIT_THRESHOLD = 0.11

# The `field_hi` signal takes components appearing in the field's best molecules.
# It was a hardcoded 0.125, which is a quantile of wildly different depth per
# reaction -- 1,087 molecules on rxn5 but FOUR on rxn3, so rxn3's strongest
# component signal was built on four data points and was pure noise. A quantile
# of that reaction's own field scores keeps the slice the same size everywhere
# (~500 molecules per reaction at 0.95).
HI_QUANTILE = 0.95


def field_csv_path(rxn_id: int, data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, f"rxn{rxn_id}.csv")


def _derive_hit_threshold(df: pd.DataFrame, fallback: float) -> float:
    """A per-reaction bar separating good field molecules from ordinary ones.

    For each epoch, the 20th best molecule ANY miner submitted; then the median
    across epochs. Note what this is not: the field CSV pools every miner (~5.5
    per epoch, no uid column), so this is a composite, not one miner's #20 and
    not a winning threshold. Its value is that it lands at a consistent p70-p80
    of each reaction's own distribution, so a component's hit rate means the
    same thing in every reaction -- which a flat 0.11 did not.
    """
    if "epoch" not in df.columns:
        return fallback
    twentieths = [
        g.nlargest(20, "final_score")["final_score"].iloc[-1]
        for _, g in df.groupby("epoch") if len(g) >= 20
    ]
    if not twentieths:
        return fallback
    return float(np.median(twentieths))


def parse_components(name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """`rxn:2:125845:127220` -> (125845, 127220, None)."""
    try:
        parts = name.split(":")
        if len(parts) == 4:
            return int(parts[2]), int(parts[3]), None
        if len(parts) >= 5:
            return int(parts[2]), int(parts[3]), int(parts[4])
    except Exception:
        pass
    return None, None, None


@dataclass
class ComponentTable:
    """Per-component statistics for one position (A, B or C)."""

    n: Dict[int, int] = field(default_factory=dict)
    mean: Dict[int, float] = field(default_factory=dict)
    best: Dict[int, float] = field(default_factory=dict)
    # Fraction of this component's molecules that cleared the "useful" bar.
    hit: Dict[int, float] = field(default_factory=dict)

    # Position-wide statistics over this source's scores. _ucb uses grand_mean
    # as its shrinkage target and grand_sd as the scale for both the
    # extreme-value correction and the exploration bonus, which is what makes
    # the prior self-calibrating across reactions.
    grand_mean: float = 0.0
    grand_sd: float = 0.0

    def ids(self) -> Set[int]:
        return set(self.n)

    def observed(self, min_obs: int = MIN_OBS) -> Set[int]:
        return {k for k, v in self.n.items() if v >= min_obs}


def _tabulate(
    df: pd.DataFrame, col: str, score_col: str, hit_threshold: float
) -> ComponentTable:
    t = ComponentTable()
    g = df.groupby(col)[score_col]
    t.n = g.size().to_dict()
    t.mean = g.mean().to_dict()
    t.best = g.max().to_dict()
    t.hit = (df.assign(_h=df[score_col] > hit_threshold)
               .groupby(col)["_h"].mean().to_dict())
    vals = df[score_col].to_numpy(dtype=float)
    t.grand_mean = float(vals.mean()) if len(vals) else 0.0
    # ddof=0: this is the whole population of what this source has seen, not a
    # sample drawn from it. Floored so a degenerate table cannot zero out the
    # exploration bonus and silently restore the old collapse.
    t.grand_sd = max(float(vals.std()) if len(vals) > 1 else 0.0, 1e-4)
    return t


class FieldPrior:
    """
    Component-level priors learned from the field's submissions, optionally
    blended with the local score DB.

    The public surface is deliberately small:

        prior.weights(ids, "A")     -> sampling distribution over component ids
        prior.elite("A", 0.8)       -> the top fraction of components
        prior.pair_bonus(a, b)      -> observed synergy for a specific pair
        prior.score_hint(name)      -> cheap pre-Boltz quality estimate
    """

    def __init__(
        self,
        rxn_id: int,
        *,
        data_dir: str = DATA_DIR,
        db_path: Optional[str] = None,
        hit_threshold: Optional[float] = None,
        field_weight: float = 0.65,
        logger=None,
    ):
        self.rxn_id = rxn_id
        # None means "derive it from this reaction's own field CSV". _load_field
        # fills it in before anything reads it; the fallback stands only when
        # there is no CSV to derive from.
        self.hit_threshold = (DEFAULT_HIT_THRESHOLD if hit_threshold is None
                              else float(hit_threshold))
        self._hit_threshold_given = hit_threshold is not None
        self.field_weight = float(np.clip(field_weight, 0.0, 1.0))
        self.log = logger
        self.three_component = False

        self.field: Dict[str, ComponentTable] = {}
        self.local: Dict[str, ComponentTable] = {}
        self.pairs: Dict[Tuple[int, int], Tuple[int, float, float]] = {}
        self.n_field = 0
        self.n_local = 0
        self.field_hi: Dict[str, Dict[int, float]] = {}

        self._score_cache: Dict[Tuple[str, float], Dict[int, float]] = {}
        self._impute_cache: Dict[str, float] = {}
        self._load_field(data_dir)
        if db_path:
            self._load_local(db_path)
        # Tables are complete now; drop anything cached during ingestion.
        self._score_cache.clear()
        self._impute_cache.clear()

    # -- ingestion ---------------------------------------------------------

    def _say(self, msg: str) -> None:
        if self.log:
            self.log.info(msg)
        else:
            print(msg)

    def _load_field(self, data_dir: str) -> None:
        path = field_csv_path(self.rxn_id, data_dir)
        if not os.path.exists(path):
            self._say(f"[field] no {path}; running without a field prior")
            return
        df = pd.read_csv(path)
        if "final_score" not in df.columns or "molecule_name" not in df.columns:
            self._say(f"[field] unexpected columns in {path}: {list(df.columns)}")
            return

        df = df.drop_duplicates("molecule_name")
        df = df[np.isfinite(df["final_score"]) & (df["final_score"] > MIN_VALID_SCORE)]
        comps = df["molecule_name"].map(parse_components)
        df = df.assign(
            A=[c[0] for c in comps], B=[c[1] for c in comps], C=[c[2] for c in comps]
        )
        df = df[df["A"].notna() & df["B"].notna()]
        self.three_component = bool(df["C"].notna().all()) and len(df) > 0
        self.n_field = len(df)
        if not self.n_field:
            return

        # Derive the bar before tabulating: _tabulate's hit rate depends on it.
        if not self._hit_threshold_given:
            self.hit_threshold = _derive_hit_threshold(df, self.hit_threshold)

        roles = ["A", "B"] + (["C"] if self.three_component else [])
        for role in roles:
            self.field[role] = _tabulate(df, role, "final_score", self.hit_threshold)

        # Pair synergy: only pairs seen enough times to mean anything.
        pg = df.groupby(["A", "B"])["final_score"]
        for (a, b), sub in pg:
            n = len(sub)
            # Every (A,B) appears exactly once in the field CSV — molecule_name
            # is unique — so requiring n>=2 would discard every pair. A single
            # observation of a pair at 0.15 is still strong evidence.
            self.pairs[(int(a), int(b))] = (n, float(sub.mean()), float(sub.max()))

        # Components that appear in genuinely high-scoring molecules. This is
        # the signal that transferred best in testing (spearman +0.36 for B).
        # Cut by quantile, not by a fixed score: see HI_QUANTILE.
        hi_cut = float(df["final_score"].quantile(HI_QUANTILE))
        hi = df[df["final_score"] > hi_cut]
        for role in roles:
            self.field_hi[role] = hi.groupby(role)["final_score"].max().to_dict()

        self._say(
            f"[field] rxn{self.rxn_id}: {self.n_field} validator-scored molecules | "
            + " ".join(
                f"{r}={len(self.field[r].n)}" for r in roles
            )
            + f" | pairs={len(self.pairs)} | hit_bar={self.hit_threshold:.4f}"
            + f" | >{hi_cut:.4f}: {len(hi)}"
        )

    def _load_local(self, db_path: str) -> None:
        if not os.path.exists(db_path):
            return
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT molecule_name, score FROM scored_molecules "
                    "WHERE score IS NOT NULL AND molecule_name LIKE ?",
                    (f"rxn:{self.rxn_id}:%",),
                ).fetchall()
        except Exception as e:
            self._say(f"[field] could not read local DB ({e})")
            return
        if not rows:
            return

        df = pd.DataFrame(rows, columns=["molecule_name", "score"])
        df = df[np.isfinite(df["score"]) & (df["score"] > MIN_VALID_SCORE)]
        comps = df["molecule_name"].map(parse_components)
        df = df.assign(
            A=[c[0] for c in comps], B=[c[1] for c in comps], C=[c[2] for c in comps]
        )
        df = df[df["A"].notna() & df["B"].notna()]
        self.n_local = len(df)
        # Detect from whichever source actually has a C column. Deriving this
        # from the field CSV alone couples the local tables to a file that may
        # be absent, and would silently drop C for rxn3/rxn5.
        if not self.three_component and df["C"].notna().any():
            self.three_component = True
        roles = ["A", "B"] + (["C"] if self.three_component else [])
        for role in roles:
            if df[role].notna().any():
                self.local[role] = _tabulate(df, role, "score", self.hit_threshold)
        self._say(
            f"[field] local DB: {self.n_local} molecules "
            f"(supplies the negatives the field CSV lacks)"
        )

    # -- component scoring -------------------------------------------------

    def _ucb(self, table: ComponentTable, cid: int, total: int,
             beta: float = UCB_BETA) -> Optional[float]:
        """
        Optimistic estimate of a component's quality, corrected for how often it
        has been tried.

        WHY THIS IS NOT `best` ANY MORE
        -------------------------------
        The previous version returned the component's BEST observed score plus
        `0.01 * sqrt(log(N)/n)`. The maximum of n draws is an extreme-value
        statistic whose expectation grows like `sd * sqrt(2 ln n)`, so `best`
        measures sampling effort at least as much as it measures quality.
        Measured on this repo's own score DBs:

            rxn1 slot0   n=1 -> mean(best)=0.033    n>=300 -> 0.128
            rxn1 slot1   n=1 -> mean(best)=0.020    n>=300 -> 0.124
            rxn2 slot0   n=1 -> mean(best)=0.027    n>=300 -> 0.114
            rxn4 slot1   n=1 -> mean(best)=0.025    n>=300 -> 0.120

            spearman(n_evals, best) = +0.65 / +0.59 / +0.72 / +0.81
            spearman(n_evals, mean) = +0.25 / +0.26 / +0.46 / +0.37

        `best` was therefore 50-80% explained by prior sampling effort, while
        the exploration bonus that was meant to offset it was worth at most
        0.033 against a `base` gap of 0.09-0.11 -- an order of magnitude too
        small. The result was a rich-get-richer loop: a heavily-sampled
        component ranked high, so it was sampled more, so it ranked higher. By
        28 Aug the searcher was touching 0.0% new B components per day on rxn2
        and 0.6% new A components on rxn4, against pool coverage of 37% and 26%.

        WHAT IT DOES INSTEAD
        --------------------
        1. Shrink the component's MEAN toward the position's grand mean with
           `PRIOR_M` pseudo-observations. A single unlucky draw no longer
           condemns a component, and a single lucky one no longer crowns it.
        2. Keep a small slice of `best` as an upside signal -- the field-max
           signal really did transfer (spearman +0.36 for B) -- but subtract
           its sample-size-driven expectation first, so it contributes the part
           of the maximum that is NOT explained by having been tried often.
        3. Make the exploration bonus a multiple of the standard error,
           `sd/sqrt(n)`. That is in the same units as the spread of the shrunk
           means it competes against, so it is self-calibrating per reaction
           and needs no hand-tuning. `total` is retained for API compatibility.
        """
        n = table.n.get(cid)
        if not n:
            return None

        sd = table.grand_sd or 1e-4

        # 1. shrunk mean
        mean_c = table.mean.get(cid, table.grand_mean)
        shrunk = ((n * mean_c) + (PRIOR_M * table.grand_mean)) / (n + PRIOR_M)

        # 2. sample-size-corrected upside
        expected_max = sd * math.sqrt(2.0 * math.log(max(n, 2)))
        upside = UPSIDE_WEIGHT * (table.best.get(cid, 0.0) - expected_max)

        # A little weight on the hit rate separates "one lucky draw" from
        # "reliably produces useful molecules".
        upside += 0.25 * self.hit_threshold * table.hit.get(cid, 0.0)

        # 3. exploration bonus, in standard errors
        bonus = beta * sd / math.sqrt(n)

        return shrunk + upside + bonus

    def component_scores(self, role: str, beta: float = UCB_BETA) -> Dict[int, float]:
        """
        Blended field+local quality estimate per component id.

        Cached: score_hint() calls this per molecule, and recomputing a
        15,000-entry dict per candidate turns pool scoring into O(n^2).
        """
        ck = (role, round(float(beta), 6))
        hit = self._score_cache.get(ck)
        if hit is not None:
            return hit
        out: Dict[int, float] = {}
        ft = self.field.get(role)
        lt = self.local.get(role)
        n_f = max(self.n_field, 1)
        n_l = max(self.n_local, 1)
        ids: Set[int] = set()
        if ft:
            ids |= ft.ids()
        if lt:
            ids |= lt.ids()
        for cid in ids:
            fv = self._ucb(ft, cid, n_f, beta) if ft else None
            lv = self._ucb(lt, cid, n_l, beta) if lt else None
            if fv is None and lv is None:
                continue
            if fv is None:
                out[cid] = lv
            elif lv is None:
                # Field-only components are the interesting ones: other miners
                # win with them and I have never tried them.
                out[cid] = fv
            else:
                w = self.field_weight
                out[cid] = w * fv + (1.0 - w) * lv
        self._score_cache[ck] = out
        return out

    def n_obs(self, role: str, cid: Optional[int]) -> int:
        """
        How many scored molecules this component appears in, across both
        sources. The surrogate carries this as a feature so the model can learn
        "few observations" as uncertainty rather than inferring it from a low
        score it was never shown.
        """
        if cid is None:
            return 0
        total = 0
        ft = self.field.get(role)
        lt = self.local.get(role)
        if ft:
            total += int(ft.n.get(cid, 0))
        if lt:
            total += int(lt.n.get(cid, 0))
        return total

    def impute_value(self, role: str) -> float:
        """
        What an UNOBSERVED component is worth.

        Unknown is not the same as bad, so it sits just below the observed
        median rather than at the bottom. This is the single source of truth for
        that choice: rank_weights and hunter's Surrogate._features must agree,
        because a generator that imputes at the 40th percentile feeding a model
        that imputes at 0.0 is a generator whose novel proposals are ranked last
        and never scored.
        """
        cached = self._impute_cache.get(role)
        if cached is not None:
            return cached
        scores = self.component_scores(role)
        val = float(np.quantile(list(scores.values()), 0.40)) if scores else 0.0
        self._impute_cache[role] = val
        return val

    def rank_weights(self, ids: Sequence[int], role: str, *,
                     temperature: float = 6.0,
                     floor: float = 0.10) -> Optional[np.ndarray]:
        """
        Sampling distribution over `ids`, by RANK rather than by z-score.

        Rank weighting is the fix for the collapse measured in Phase 2: the old
        `exp(0.7 * z)` scheme produced a 1.56x max/uniform ratio because
        `global_std` (0.037) dwarfed the spread between component means. A rank
        transform is invariant to that scale, so the prior stays sharp no matter
        how the score distribution is shaped.

        `floor` keeps a uniform component so nothing is permanently unreachable.
        """
        ids = np.asarray(list(ids), dtype=np.int64)
        if len(ids) == 0:
            return None
        scores = self.component_scores(role)
        if not scores:
            return None

        # Unobserved components sit just below the observed median rather than
        # at the bottom: unknown is not the same as bad.
        vals = np.array([scores.get(int(i), np.nan) for i in ids], dtype=float)
        known = np.isfinite(vals)
        if known.sum() < 2:
            return None
        vals[~known] = self.impute_value(role)

        order = np.argsort(np.argsort(vals))          # 0 = worst
        r = order / max(len(order) - 1, 1)            # 0..1
        # exp(T*r) puts roughly T times uniform weight on the best component.
        # The old orchestrator managed only 1.56x, which is why its search was
        # indistinguishable from random over a 2e9 space.
        w = np.exp(temperature * r)
        w /= w.sum()
        u = np.full(len(w), 1.0 / len(w))
        w = (1.0 - floor) * w + floor * u
        return w / w.sum()

    def elite(self, role: str, quantile: float = 0.85,
              min_obs: int = MIN_OBS) -> List[int]:
        """Components in the top `1-quantile` of blended quality."""
        scores = self.component_scores(role)
        ft = self.field.get(role)
        lt = self.local.get(role)
        seen = set()
        if ft:
            seen |= ft.observed(min_obs)
        if lt:
            seen |= lt.observed(min_obs)
        pool = {k: v for k, v in scores.items() if k in seen}
        if len(pool) < 10:
            pool = scores
        if not pool:
            return []
        cut = np.quantile(list(pool.values()), quantile)
        return [k for k, v in sorted(pool.items(), key=lambda kv: -kv[1]) if v >= cut]

    def hi_components(self, role: str) -> Dict[int, float]:
        """Components that appear in field molecules scoring above 0.125."""
        return self.field_hi.get(role, {})

    def pair_bonus(self, a: Optional[int], b: Optional[int]) -> float:
        """Observed max for this exact A-B pair, 0.0 when never seen."""
        if a is None or b is None:
            return 0.0
        rec = self.pairs.get((int(a), int(b)))
        return rec[2] if rec else 0.0

    def score_hint(self, name: str) -> float:
        """
        A cheap pre-Boltz quality estimate for one molecule name.

        Used only to order a large candidate pool before the surrogate sees it;
        it is a prior, not a prediction, and is never written to the score DB.
        """
        a, b, c = parse_components(name)
        ca = self.component_scores("A")
        cb = self.component_scores("B")
        sa = ca.get(a, 0.0) if a is not None else 0.0
        sb = cb.get(b, 0.0) if b is not None else 0.0
        val = 0.5 * (sa + sb)
        if self.three_component and c is not None:
            val = (2.0 * val + self.component_scores("C").get(c, 0.0)) / 3.0
        return val + 0.25 * self.pair_bonus(a, b)

    def score_hints(self, names: Sequence[str]) -> np.ndarray:
        """Vectorised score_hint over many names (the pool-ordering path)."""
        ca = self.component_scores("A")
        cb = self.component_scores("B")
        cc = self.component_scores("C") if self.three_component else {}
        out = np.empty(len(names), dtype=float)
        for i, nm in enumerate(names):
            a, b, c = parse_components(nm)
            sa = ca.get(a, 0.0) if a is not None else 0.0
            sb = cb.get(b, 0.0) if b is not None else 0.0
            v = 0.5 * (sa + sb)
            if cc and c is not None:
                v = (2.0 * v + cc.get(c, 0.0)) / 3.0
            out[i] = v + 0.25 * self.pair_bonus(a, b)
        return out

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"FieldPrior rxn{self.rxn_id}: field={self.n_field} local={self.n_local} "
            f"blend={self.field_weight:.2f}"
        ]
        for role in ("A", "B", "C"):
            if role not in self.field and role not in self.local:
                continue
            el = self.elite(role, 0.85)
            hi = self.hi_components(role)
            lines.append(
                f"  {role}: elite={len(el)} | appear in >0.125 molecules={len(hi)}"
            )
        return "\n".join(lines)


def available(rxn_id: int, data_dir: str = DATA_DIR) -> bool:
    return os.path.exists(field_csv_path(rxn_id, data_dir))
