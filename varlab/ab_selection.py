#!/usr/bin/env python3
"""
Offline A/B: orchestrator.py's selection vs hunter.py's, on already-scored data.

METHOD
    Split the score DB into train / pool. Fit each searcher's surrogate on
    train only. Let each pick `budget` molecules from the pool. Because every
    molecule in the pool already has a real Boltz score, we can read off exactly
    what each selector would have got.

WHAT THIS DOES AND DOES NOT MEASURE
    It measures SELECTION. It cannot measure GENERATION, because the pool is
    fixed and consists of whatever the orchestrator already chose to generate.
    Generation is the dominant defect (the prior collapsed to 1.56x uniform), so
    this test is biased *against* hunter — its main advantage is invisible here.
    Read the result as a lower bound.

    Repeated over several random splits so the numbers are not one lucky draw.
"""
from __future__ import annotations

import argparse
import os
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

import sqlite3
import field_prior
import hunter


def load_orchestrator():
    """Import orchestrator's surrogate + selector without running main()."""
    src = open(os.path.join(BASE, "orchestrator.py")).read()
    head = src[: src.index("def final_top20(")]
    mod = types.ModuleType("orch")
    mod.__dict__["__file__"] = os.path.join(BASE, "orchestrator.py")
    exec(compile(head, "orchestrator.py", "exec"), mod.__dict__)
    return mod


def load_pool(rxn_id: int) -> pd.DataFrame:
    db = os.path.join(BASE, f"score_results_{rxn_id}.sqlite")
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT molecule_name, smiles, score, COALESCE(round,iteration,0) "
            "FROM scored_molecules WHERE smiles IS NOT NULL AND smiles!='' "
            "AND score IS NOT NULL AND molecule_name LIKE ?",
            (f"rxn:{rxn_id}:%",),
        ).fetchall()
    df = pd.DataFrame(rows, columns=["name", "smiles", "score", "round"])
    df = df[np.isfinite(df["score"]) & (df["score"] > -1)]
    return df.reset_index(drop=True)


def evaluate(sel: pd.DataFrame, thresholds) -> dict:
    out = {"n": len(sel), "best": float(sel["score"].max()),
           "mean": float(sel["score"].mean()),
           "top20sum": float(sel.nlargest(20, "score")["score"].sum())}
    for t in thresholds:
        out[f">{t}"] = int((sel["score"] > t).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn-id", type=int, default=2)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--drop-field-overlap", action="store_true",
                    help="remove molecules present in data/rxn{N}.csv from the "
                         "pool; hunter's pair_bonus matches those exactly, "
                         "which would leak")
    ap.add_argument("--ablate", action="store_true",
                    help="also run hunter's selection rule on the orchestrator's "
                         "features, to separate rule from features")
    args = ap.parse_args()

    thresholds = (0.10, 0.11, 0.12)
    df = load_pool(args.rxn_id)
    print(f"rxn{args.rxn_id}: {len(df)} scored molecules available")
    if args.drop_field_overlap:
        fcsv = os.path.join(BASE, "data", f"rxn{args.rxn_id}.csv")
        if os.path.exists(fcsv):
            overlap = set(pd.read_csv(fcsv)["molecule_name"])
            n0 = len(df)
            df = df[~df["name"].isin(overlap)].reset_index(drop=True)
            print(f"  dropped {n0 - len(df)} molecules that also appear in the "
                  f"field CSV (leakage guard) -> {len(df)}")
    orch = load_orchestrator()
    prior = field_prior.FieldPrior(args.rxn_id, db_path=None, field_weight=1.0)

    rows = []
    for trial in range(args.trials):
        rng = np.random.default_rng(1000 + trial)
        idx = rng.permutation(len(df))
        ntr = int(len(df) * args.train_frac)
        train = df.iloc[idx[:ntr]].reset_index(drop=True)
        pool = df.iloc[idx[ntr:]].reset_index(drop=True)
        base = {f">{t}": float((pool["score"] > t).mean()) for t in thresholds}
        print(f"\n--- trial {trial + 1}: train={len(train)} pool={len(pool)} "
              f"| base rates " +
              " ".join(f">{t}:{100*base[f'>{t}']:.3f}%" for t in thresholds))

        # ---- orchestrator ----
        o = orch.Top20Surrogate(min_train=500, train_cap=30000, seed=68)
        o.fit(train)
        opred = o.predict(pool)
        osel = orch.choose_boltz_batch(opred, args.budget, np.random.default_rng(68))

        # ---- hunter ----
        # Prior must not see the pool's own scores, or the comparison leaks.
        hp = field_prior.FieldPrior(args.rxn_id, db_path=None, field_weight=1.0)
        hp.local = {}
        hp.n_local = 0
        for role in ("A", "B", "C"):
            if role in hp.field:
                hp.local[role] = field_prior._tabulate(
                    train.assign(**{
                        role: [field_prior.parse_components(n)[
                            {"A": 0, "B": 1, "C": 2}[role]] for n in train["name"]]
                    }).dropna(subset=[role]),
                    role, "score", hp.hit_threshold)
        hp.n_local = len(train)
        hp._score_cache.clear()
        h = hunter.Surrogate(hp, min_train=500, train_cap=30000, seed=68)
        h.fit(train)
        hpred = h.predict(pool)
        hsel = hunter.select_batch(hpred, args.budget, 0.10,
                                   np.random.default_rng(68))

        rnd = pool.sample(n=args.budget, random_state=trial)
        runs = [("orchestrator", osel), ("hunter", hsel), ("random", rnd)]

        if args.ablate:
            # hunter's SELECTION RULE applied to the orchestrator's own
            # mu/sigma, isolating rule from features.
            op = opred.copy()
            z = (op["mu"] - float(np.quantile(op["mu"], 0.995))) / op["sigma"]
            from math import erf, sqrt, pi
            Phi = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
            phi = np.exp(-0.5 * z * z) / sqrt(2.0 * pi)
            op["ei"] = (op["mu"] - float(np.quantile(op["mu"], 0.995))) * Phi \
                       + op["sigma"] * phi
            runs.append(("rule-only", hunter.select_batch(
                op, args.budget, 0.10, np.random.default_rng(68))))

        for label, sel in runs:
            r = evaluate(sel, thresholds)
            r["trial"] = trial
            r["who"] = label
            for t in thresholds:
                r[f"enr>{t}"] = ((r[f">{t}"] / len(sel)) / base[f">{t}"]
                                 if base[f">{t}"] > 0 else float("nan"))
            rows.append(r)
            print(f"   {label:<13}" +
                  " ".join(f">{t}:{r[f'>{t}']:>3}({r[f'enr>{t}']:>5.1f}x)"
                           for t in thresholds) +
                  f" | best {r['best']:.5f} | mean {r['mean']:.5f}"
                  f" | top20sum {r['top20sum']:.5f}")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"MEAN OVER {args.trials} TRIALS (budget={args.budget})")
    print("=" * 78)
    hdr = f"{'selector':<14}" + "".join(f"{'>'+str(t):>9}{'enr':>8}" for t in thresholds)
    print(hdr + f"{'best':>10}{'mean':>10}{'top20sum':>11}")
    for who in [w for w in ("orchestrator", "rule-only", "hunter", "random")
                if w in set(res.who)]:
        g = res[res.who == who]
        line = f"{who:<14}"
        for t in thresholds:
            line += f"{g[f'>{t}'].mean():>9.1f}{g[f'enr>{t}'].mean():>7.1f}x"
        line += f"{g['best'].mean():>10.5f}{g['mean'].mean():>10.5f}{g['top20sum'].mean():>11.5f}"
        print(line)

    o = res[res.who == "orchestrator"]
    h = res[res.who == "hunter"]
    print()
    for t in thresholds:
        a, b = o[f">{t}"].mean(), h[f">{t}"].mean()
        print(f"  hits >{t}: orchestrator {a:.1f} -> hunter {b:.1f} "
              f"({'+' if b >= a else ''}{100*(b-a)/max(a,1e-9):.0f}%)")
    print(f"  mean score of selection: {o['mean'].mean():.5f} -> {h['mean'].mean():.5f} "
          f"({'+' if h['mean'].mean() >= o['mean'].mean() else ''}"
          f"{100*(h['mean'].mean()-o['mean'].mean())/abs(o['mean'].mean()):.0f}%)")


if __name__ == "__main__":
    main()
