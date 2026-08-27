#!/usr/bin/env python3
"""
Offline A/B of CANDIDATE GENERATION: orchestrator.py vs hunter.py.

Selection can only rank what generation proposes, and Phase-2 measured the
generator as the dominant defect (component prior collapsed to 1.56x uniform
over a 2e9 space). ab_selection.py cannot see that, because it hands both
searchers the same fixed pool.

METHOD
    Hold out a random slice of the score DB. Let each generator propose
    candidates blind to it. Whatever a generator proposes that happens to land
    in the held-out slice has a real Boltz score we can read off. Compare the
    score distribution of each generator's landed proposals.

BIAS, STATED UP FRONT
    Every molecule in the DB was originally proposed by the orchestrator's own
    generator, so the held-out slice is drawn from the orchestrator's
    distribution. That biases the overlap test *in the orchestrator's favour* —
    it is playing at home. Any hunter advantage here is therefore a lower bound.
"""
from __future__ import annotations

import argparse
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

from config.config_loader import load_config
from molecules import MoleculeManager

import field_prior
import hunter


class Args:
    """Minimal stand-in for argparse namespaces used by both generators."""

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
        for k, v in kw.items():
            setattr(self, k, v)


def load_orchestrator():
    src = open(os.path.join(BASE, "orchestrator.py")).read()
    head = src[: src.index("def final_top20(")]
    mod = types.ModuleType("orch")
    mod.__dict__["__file__"] = os.path.join(BASE, "orchestrator.py")
    exec(compile(head, "orchestrator.py", "exec"), mod.__dict__)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn-id", type=int, default=2)
    ap.add_argument("--pool", type=int, default=40000, help="candidates per generator")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--holdout", type=float, default=0.35)
    args = ap.parse_args()

    config = load_config()
    rxn_id = args.rxn_id
    db = os.path.join(BASE, f"score_results_{rxn_id}.sqlite")
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT molecule_name, smiles, score, COALESCE(round,iteration,0) "
            "FROM scored_molecules WHERE smiles IS NOT NULL AND smiles!='' "
            "AND score IS NOT NULL AND molecule_name LIKE ?",
            (f"rxn:{rxn_id}:%",),
        ).fetchall()
    df = pd.DataFrame(rows, columns=["name", "smiles", "score", "round"])
    df = df[np.isfinite(df["score"]) & (df["score"] > -1)].reset_index(drop=True)
    print(f"rxn{rxn_id}: {len(df)} scored molecules")

    cfg = dict(config)
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    cfg["max_heavy_atoms"] = 10 ** 9
    manager = MoleculeManager(config=cfg, db_path=os.path.join(
        BASE, "combinatorial_db", "molecules.sqlite"))
    orch = load_orchestrator()
    orch._GLOBAL_CONFIG = config
    orch._GLOBAL_MANAGER = manager
    hunter._CONFIG = config

    out = []
    for trial in range(args.trials):
        rng = np.random.default_rng(500 + trial)
        idx = rng.permutation(len(df))
        nho = int(len(df) * args.holdout)
        hold = df.iloc[idx[:nho]].reset_index(drop=True)
        known = df.iloc[idx[nho:]].reset_index(drop=True)
        truth = dict(zip(hold["name"], hold["score"]))
        base_all = float(hold["score"].mean())
        print(f"\n--- trial {trial+1}: known={len(known)} holdout={len(hold)} "
              f"| holdout mean {base_all:.5f} "
              f"P(>0.11)={100*(hold['score']>0.11).mean():.3f}%")

        seen = set(known["name"])

        # ---- orchestrator generator ----
        oargs = Args(candidate_pool=args.pool, seed=68 + trial)
        orch._GLOBAL_ARGS = oargs
        osur = orch.Top20Surrogate(min_train=500, train_cap=30000, seed=68)
        osur.stats.fit(known)
        ogen = orch.CandidateGenerator(rxn_id, manager, osur, oargs)
        onames = []
        for fn, frac in (("global_candidates", 0.20), ("single_anchor", 0.18),
                         ("pair_anchor", 0.22)):
            onames += getattr(ogen, fn)(int(args.pool * frac))
        onames += ogen.crossover(known, int(args.pool * 0.20))

        # ---- hunter generator ----
        hargs = Args(candidate_pool=args.pool, seed=68 + trial)
        prior = field_prior.FieldPrior(rxn_id, db_path=None, field_weight=0.5)
        # give it only the KNOWN half as local evidence
        for role in ("A", "B", "C"):
            if role in prior.field or role in ("A", "B"):
                pos = {"A": 0, "B": 1, "C": 2}[role]
                col = [field_prior.parse_components(n)[pos] for n in known["name"]]
                k2 = known.assign(**{role: col}).dropna(subset=[role])
                if len(k2):
                    prior.local[role] = field_prior._tabulate(
                        k2, role, "score", prior.hit_threshold)
        prior.n_local = len(known)
        prior._score_cache.clear()
        hgen = hunter.Generator(rxn_id, manager, prior, hargs)
        hgen.refresh()
        hnames = (hgen.elite_pairs(int(args.pool * 0.30))
                  + hgen.anchored(int(args.pool * 0.25))
                  + hgen.pair_mutants(int(args.pool * 0.15))
                  + hgen.broad(int(args.pool * 0.20)))

        # ---- random baseline ----
        A = np.asarray(hgen.ids["A"]); B = np.asarray(hgen.ids["B"])
        rr = np.random.default_rng(trial)
        rnames = [f"rxn:{rxn_id}:{a}:{b}" for a, b in
                  zip(rr.choice(A, args.pool), rr.choice(B, args.pool))]

        for label, names in (("orchestrator", onames), ("hunter", hnames),
                             ("random", rnames)):
            uniq = [n for n in dict.fromkeys(names) if n not in seen]
            landed = [(n, truth[n]) for n in uniq if n in truth]
            if not landed:
                print(f"   {label:<13} proposed {len(uniq):>6} | landed 0")
                continue
            sc = np.array([x[1] for x in landed])
            rec = {
                "trial": trial, "who": label, "proposed": len(uniq),
                "landed": len(landed), "mean": float(sc.mean()),
                "p90": float(np.percentile(sc, 90)), "best": float(sc.max()),
                "hit11": float((sc > 0.11).mean()),
                "hit12": float((sc > 0.12).mean()),
            }
            out.append(rec)
            print(f"   {label:<13} proposed {len(uniq):>6} | landed {len(landed):>5} "
                  f"| mean {sc.mean():.5f} | p90 {np.percentile(sc,90):.5f} "
                  f"| P(>0.11) {100*(sc>0.11).mean():5.2f}% "
                  f"| P(>0.12) {100*(sc>0.12).mean():5.3f}% | best {sc.max():.5f}")

    res = pd.DataFrame(out)
    if res.empty:
        print("\nno overlap — increase --pool or --holdout")
        return
    print("\n" + "=" * 78)
    print(f"GENERATION QUALITY, mean over {args.trials} trials")
    print("=" * 78)
    print(f"{'generator':<14}{'landed':>8}{'mean':>10}{'p90':>10}"
          f"{'P(>0.11)':>10}{'P(>0.12)':>10}{'best':>10}")
    for who in ("orchestrator", "hunter", "random"):
        g = res[res.who == who]
        if g.empty:
            continue
        print(f"{who:<14}{g['landed'].mean():>8.0f}{g['mean'].mean():>10.5f}"
              f"{g['p90'].mean():>10.5f}{100*g['hit11'].mean():>9.2f}%"
              f"{100*g['hit12'].mean():>9.3f}%{g['best'].mean():>10.5f}")
    o = res[res.who == "orchestrator"]
    h = res[res.who == "hunter"]
    if not o.empty and not h.empty:
        print()
        print(f"  mean score of proposals : {o['mean'].mean():.5f} -> "
              f"{h['mean'].mean():.5f} "
              f"({100*(h['mean'].mean()-o['mean'].mean())/abs(o['mean'].mean()):+.0f}%)")
        print(f"  P(>0.11) of proposals   : {100*o['hit11'].mean():.2f}% -> "
              f"{100*h['hit11'].mean():.2f}% "
              f"({h['hit11'].mean()/max(o['hit11'].mean(),1e-9):.2f}x)")


if __name__ == "__main__":
    main()
