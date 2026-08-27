#!/usr/bin/env python3
"""
Full evidence run for hunter.py across all five reactions.

Five independent measurements per reaction, each stating its own confounds:

  1. BASELINE     my submittable top-20 vs what the field actually submits
  2. PRIOR        component-prior sharpness, orchestrator vs hunter
  3. TRANSFER     leakage-free enrichment of the field prior on local data
  4. SELECTION    A/B on a held-out pool, both surrogates, N trials
  5. TARGETING    where each generator aims, judged by an independent oracle

Leakage discipline, because an earlier version of this analysis was wrong:
molecules present in BOTH the local DB and data/rxn{N}.csv are removed from
every evaluation set. hunter's pair_bonus matches those exactly, and they are
field submissions so they are disproportionately good — including them inflated
one earlier result from ~5x to ~56x.

    python3 varlab/bench_all.py                # all five
    python3 varlab/bench_all.py --rxn 2 5      # subset
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (BASE, os.path.join(BASE, "miner")):
    if p not in sys.path:
        sys.path.insert(0, p)

import field_prior
import hunter
import novelty
from config.config_loader import load_config
from molecules import MoleculeManager

CFG = load_config()
TARGET = CFG["small_molecule_target"][0]
THR = novelty.config_threshold(CFG)


class Args:
    def __init__(self, **kw):
        self.seed = 68
        self.parent_pool = 250
        self.elite_anchors = 40
        self.pair_anchors = 400
        self.neighbour_top_k = 30
        self.neighbour_min_sim = 0.35
        self.min_train = 500
        self.elite_quantile = 0.90
        self.prior_temperature = 6.0
        self.prior_floor = 0.10
        self.max_heavy_atoms = 0
        self.candidate_pool = 40000
        for k, v in kw.items():
            setattr(self, k, v)


def load_orchestrator():
    src = open(os.path.join(BASE, "orchestrator.py")).read()
    mod = types.ModuleType("orch")
    mod.__dict__["__file__"] = os.path.join(BASE, "orchestrator.py")
    exec(compile(src[: src.index("def final_top20(")], "orchestrator.py", "exec"),
         mod.__dict__)
    return mod


def local_df(rxn: int) -> pd.DataFrame:
    db = os.path.join(BASE, f"score_results_{rxn}.sqlite")
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT molecule_name, smiles, score, COALESCE(round,iteration,0), "
            "COALESCE(available,1) FROM scored_molecules "
            "WHERE smiles IS NOT NULL AND smiles!='' AND score IS NOT NULL "
            "AND molecule_name LIKE ?", (f"rxn:{rxn}:%",)).fetchall()
    df = pd.DataFrame(rows, columns=["name", "smiles", "score", "round", "available"])
    return df[np.isfinite(df["score"]) & (df["score"] > -1)].reset_index(drop=True)


def field_df(rxn: int) -> pd.DataFrame:
    p = os.path.join(BASE, "data", f"rxn{rxn}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p).drop_duplicates("molecule_name")
    d = d[d["final_score"] > 0].copy()
    comps = d["molecule_name"].map(field_prior.parse_components)
    d["A"] = [c[0] for c in comps]
    d["B"] = [c[1] for c in comps]
    d["C"] = [c[2] for c in comps]
    return d


# --------------------------------------------------------------------------
# 1. baseline
# --------------------------------------------------------------------------

def baseline(rxn: int, guard: novelty.NoveltyGuard, scan: int = 3000) -> dict:
    db = os.path.join(BASE, f"score_results_{rxn}.sqlite")
    top = hunter.submittable_top20(db, rxn, guard, TARGET, CFG, scan=scan)
    fd = field_df(rxn)
    out = {"my_top20_sum": float(top["score"].sum()) if len(top) else 0.0,
           "my_n": len(top),
           "my_best": float(top["score"].max()) if len(top) else 0.0}
    if not fd.empty and "epoch" in fd.columns:
        recent = sorted(fd["epoch"].unique())[-10:]
        sums = [fd[fd.epoch == e].nlargest(20, "final_score")["final_score"].sum()
                for e in recent]
        out["field_median"] = float(np.median(sums))
        out["field_best"] = float(np.max(sums))
        out["gap"] = out["my_top20_sum"] - out["field_median"]
    return out


# --------------------------------------------------------------------------
# 2. prior sharpness
# --------------------------------------------------------------------------

def sharpness(rxn: int, orch, df: pd.DataFrame, prior) -> dict:
    """max/uniform sampling weight for each searcher's A prior."""
    import math
    st = orch.ComponentStats()
    st.fit(df)
    A = sorted(st.single["A"].keys())
    gm, gs = st.global_mean, max(st.global_std, 1e-6)
    raw = np.array([math.exp(0.7 * np.clip(
        ((0.65 * st.single["A"][a][0] + 0.35 * st.single["A"][a][2]) - gm) / gs,
        -4, 4)) for a in A])
    p = raw / raw.sum()
    p = 0.65 * p + 0.35 * np.full(len(p), 1 / len(p))
    p /= p.sum()
    ids = sorted(prior.component_scores("A").keys())
    w = prior.rank_weights(ids, "A")
    return {"orch_sharpness": float(p.max() * len(p)),
            "hunter_sharpness": float(w.max() * len(w)) if w is not None else 1.0}


# --------------------------------------------------------------------------
# 3. transfer
# --------------------------------------------------------------------------

def transfer(rxn: int, df: pd.DataFrame, fd: pd.DataFrame) -> dict:
    """Leakage-free: does the field's component evidence predict local scores?"""
    if fd.empty:
        return {}
    clean = df[~df["name"].isin(set(fd["molecule_name"]))].copy()
    if len(clean) < 500:
        return {}
    comps = clean["name"].map(field_prior.parse_components)
    clean["A"] = [c[0] for c in comps]
    clean["B"] = [c[1] for c in comps]
    hiA = fd.groupby("A")["final_score"].max()
    hiB = fd.groupby("B")["final_score"].max()
    clean["fA"] = clean["A"].map(hiA)
    clean["fB"] = clean["B"].map(hiB)
    base = float((clean["score"] > 0.11).mean())
    out = {"n_clean": len(clean), "base_hit11": base,
           "dropped_overlap": len(df) - len(clean)}
    if base > 0:
        sub = clean[(clean["fA"] > 0.125) & (clean["fB"] > 0.125)]
        out["both_elite_n"] = len(sub)
        out["both_elite_enr"] = (float((sub["score"] > 0.11).mean()) / base
                                 if len(sub) > 50 else float("nan"))
    sp = clean[["fB", "score"]].dropna()
    out["spearman_fB"] = (float(sp.corr(method="spearman").iloc[0, 1])
                          if len(sp) > 500 else float("nan"))
    return out


# --------------------------------------------------------------------------
# 4. selection A/B
# --------------------------------------------------------------------------

def selection_ab(rxn: int, orch, df: pd.DataFrame, fd: pd.DataFrame,
                 trials: int, budget: int) -> dict:
    pool_src = df[~df["name"].isin(set(fd["molecule_name"]))] if not fd.empty else df
    pool_src = pool_src.reset_index(drop=True)
    if len(pool_src) < 5000:
        return {}
    rows = []
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        idx = rng.permutation(len(pool_src))
        ntr = int(len(pool_src) * 0.70)
        train = pool_src.iloc[idx[:ntr]].reset_index(drop=True)
        pool = pool_src.iloc[idx[ntr:]].reset_index(drop=True)
        base = {k: float((pool["score"] > k).mean()) for k in (0.10, 0.11, 0.12)}

        o = orch.Top20Surrogate(500, 30000, 68)
        o.fit(train)
        op = o.predict(pool)
        osel = orch.choose_boltz_batch(op, budget, np.random.default_rng(68))

        # hunter's prior must see only the training half
        hp = field_prior.FieldPrior(rxn, db_path=None, field_weight=0.5)
        hp.local = {}
        for role in ("A", "B", "C"):
            pos = {"A": 0, "B": 1, "C": 2}[role]
            col = [field_prior.parse_components(n)[pos] for n in train["name"]]
            k2 = train.assign(**{role: col}).dropna(subset=[role])
            if len(k2):
                hp.local[role] = field_prior._tabulate(k2, role, "score",
                                                       hp.hit_threshold)
                if role == "C":
                    hp.three_component = True
        hp.n_local = len(train)
        hp._score_cache.clear()
        h = hunter.Surrogate(hp, 500, 30000, 68)
        h.fit(train)
        hpred = h.predict(pool)
        hsel = hunter.select_batch(hpred, budget, 0.10, np.random.default_rng(68), key="mu")
        rnd = pool.sample(n=budget, random_state=t)

        for who, sel in (("orch", osel), ("hunter", hsel), ("random", rnd)):
            r = {"who": who, "mean": float(sel["score"].mean()),
                 "best": float(sel["score"].max())}
            for k in (0.10, 0.11, 0.12):
                r[f"h{k}"] = int((sel["score"] > k).sum())
                r[f"e{k}"] = ((r[f"h{k}"] / len(sel)) / base[k]
                              if base[k] > 0 else float("nan"))
            rows.append(r)
    res = pd.DataFrame(rows)
    out = {}
    for who in ("orch", "hunter", "random"):
        g = res[res.who == who]
        for k in (0.10, 0.11, 0.12):
            out[f"{who}_h{k}"] = float(g[f"h{k}"].mean())
            out[f"{who}_e{k}"] = float(g[f"e{k}"].mean())
        out[f"{who}_mean"] = float(g["mean"].mean())
        out[f"{who}_best"] = float(g["best"].mean())
    return out


# --------------------------------------------------------------------------
# 5. generation targeting, judged by an independent oracle
# --------------------------------------------------------------------------

def targeting(rxn: int, orch, manager, df: pd.DataFrame, fd: pd.DataFrame,
              pool: int) -> dict:
    if fd.empty:
        return {}
    hunter._CONFIG = CFG
    a = Args(candidate_pool=pool, seed=68)
    orch._GLOBAL_CONFIG = CFG
    orch._GLOBAL_MANAGER = manager
    orch._GLOBAL_ARGS = a

    osur = orch.Top20Surrogate(500, 30000, 68)
    osur.stats.fit(df)
    og = orch.CandidateGenerator(rxn, manager, osur, a)
    on = (og.global_candidates(int(pool * 0.20)) + og.single_anchor(int(pool * 0.18))
          + og.pair_anchor(int(pool * 0.22)) + og.crossover(df, int(pool * 0.20)))

    prior = field_prior.FieldPrior(
        rxn, db_path=os.path.join(BASE, f"score_results_{rxn}.sqlite"),
        field_weight=0.5)
    hg = hunter.Generator(rxn, manager, prior, Args(candidate_pool=pool, seed=68))
    hg.refresh()
    hn = (hg.elite_pairs(int(pool * 0.30)) + hg.anchored(int(pool * 0.25))
          + hg.pair_mutants(int(pool * 0.15)) + hg.broad(int(pool * 0.20)))

    seen = set(df["name"])
    hi = fd[fd["final_score"] > 0.125]
    eA, eB = set(hi["A"].dropna()), set(hi["B"].dropna())
    eC = set(hi["C"].dropna()) if hi["C"].notna().any() else None
    out = {}
    for who, names in (("orch", on), ("hunter", hn)):
        u = [n for n in dict.fromkeys(names) if n not in seen]
        if not u:
            continue
        comps = [field_prior.parse_components(n) for n in u]
        A = np.array([c[0] if c[0] is not None else -1 for c in comps])
        B = np.array([c[1] if c[1] is not None else -1 for c in comps])
        ia, ib = np.isin(A, list(eA)), np.isin(B, list(eB))
        both = ia & ib
        if eC:
            C = np.array([c[2] if c[2] is not None else -1 for c in comps])
            both = both & np.isin(C, list(eC))
        out[f"{who}_proposed"] = len(names)
        out[f"{who}_unique"] = len(u)
        out[f"{who}_yield"] = len(u) / max(len(names), 1)
        out[f"{who}_eliteA"] = float(ia.mean())
        out[f"{who}_eliteB"] = float(ib.mean())
        out[f"{who}_both"] = float(both.mean())
        out[f"{who}_distinctA"] = int(len(set(A)))
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--gen-pool", type=int, default=40000)
    ap.add_argument("--out", default=os.path.join(BASE, "varlab", "bench_all.json"))
    args = ap.parse_args()

    orch = load_orchestrator()
    guard = novelty.NoveltyGuard(TARGET, THR)
    results = {}

    for rxn in args.rxn:
        db = os.path.join(BASE, f"score_results_{rxn}.sqlite")
        if not os.path.exists(db):
            print(f"rxn{rxn}: no DB, skipped")
            continue
        print(f"\n{'='*78}\nrxn{rxn}\n{'='*78}", flush=True)
        df = local_df(rxn)
        fd = field_df(rxn)
        cfg = dict(CFG)
        cfg["allowed_reaction"] = f"rxn:{rxn}"
        cfg["max_heavy_atoms"] = 10 ** 9
        manager = MoleculeManager(config=cfg, db_path=os.path.join(
            BASE, "combinatorial_db", "molecules.sqlite"))
        prior = field_prior.FieldPrior(rxn, db_path=db, field_weight=0.5)

        r = {"rxn": rxn, "local_n": len(df), "field_n": len(fd),
             "three_component": bool(prior.three_component)}
        print("  [1/5] baseline ...", flush=True)
        r.update(baseline(rxn, guard))
        print("  [2/5] prior sharpness ...", flush=True)
        r.update(sharpness(rxn, orch, df, prior))
        print("  [3/5] transfer ...", flush=True)
        r.update(transfer(rxn, df, fd))
        print("  [4/5] selection A/B ...", flush=True)
        r.update(selection_ab(rxn, orch, df, fd, args.trials, args.budget))
        print("  [5/5] generation targeting ...", flush=True)
        r.update(targeting(rxn, orch, manager, df, fd, args.gen_pool))
        results[rxn] = r
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in r.items()}, indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
