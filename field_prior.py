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

# A component needs at least this many observations before its mean is trusted
# at face value; below it the UCB bonus dominates and it stays explorable.
MIN_OBS = 3

# Scores at or below this are Boltz failures / degenerate structures, not signal.
# The reactant summaries contain entries like avg_score = -17.79 from these.
MIN_VALID_SCORE = -1.0


def field_csv_path(rxn_id: int, data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, f"rxn{rxn_id}.csv")


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
        hit_threshold: float = 0.11,
        field_weight: float = 0.65,
        logger=None,
    ):
        self.rxn_id = rxn_id
        self.hit_threshold = hit_threshold
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
        self._load_field(data_dir)
        if db_path:
            self._load_local(db_path)
        # Tables are complete now; drop anything cached during ingestion.
        self._score_cache.clear()

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
        hi = df[df["final_score"] > 0.125]
        for role in roles:
            self.field_hi[role] = hi.groupby(role)["final_score"].max().to_dict()

        self._say(
            f"[field] rxn{self.rxn_id}: {self.n_field} validator-scored molecules | "
            + " ".join(
                f"{r}={len(self.field[r].n)}" for r in roles
            )
            + f" | pairs={len(self.pairs)} | >0.125: {len(hi)}"
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

    def _ucb(self, table: ComponentTable, cid: int, total: int, beta: float) -> Optional[float]:
        """
        Upper confidence bound on a component's *reachable* quality.

        The base is the component's BEST observed score, not its mean. This is
        the empirically-supported choice: in Phase-1 testing,
        `spearman(field-max, my score) = +0.36` for B components, and selecting
        molecules whose A and B both had a field max above 0.125 lifted
        P(>0.11) from 1.23% to 7.52%. Component means carry far less signal —
        a great block paired with 700 mediocre partners still has a mediocre
        mean, and that is precisely the block worth pairing well.

        The UCB bonus keeps rarely-tried components reachable. `ComponentStats`
        had this backwards: shrinking toward the global mean made rare
        components *less* likely to be tried, not more.
        """
        n = table.n.get(cid)
        if not n:
            return None
        base = table.best.get(cid, 0.0)
        # A little weight on the hit rate separates "one lucky draw" from
        # "reliably produces useful molecules".
        base += 0.25 * self.hit_threshold * table.hit.get(cid, 0.0)
        bonus = beta * math.sqrt(math.log(max(total, 2)) / n)
        return base + bonus

    def component_scores(self, role: str, beta: float = 0.01) -> Dict[int, float]:
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
        vals[~known] = np.quantile(vals[known], 0.40)

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
