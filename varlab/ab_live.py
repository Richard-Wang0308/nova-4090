#!/usr/bin/env python3
"""
LIVE head-to-head on real Boltz: orchestrator.py vs hunter.py, per reaction.

Every other test in varlab/ is an offline proxy. This is the real thing — both
searchers run their true generate->rank->select pipeline from the same DB state,
and every selected molecule is actually scored on the GPU. There is no oracle to
confound and no held-out set to leak.

    python3 varlab/ab_live.py --rxn 1 2 3 4 5 --per-arm 30

PROTOCOL
    For each reaction, from an identical starting DB:
      orchestrator: CandidateGenerator -> Top20Surrogate -> choose_boltz_batch
      hunter:       Generator          -> Surrogate      -> select_batch
    Both arms are novelty-filtered the same way, both are scored by the same
    boltz_score(), interleaved batch by batch so any GPU drift hits both arms
    equally. Molecules picked by BOTH arms are scored once and credited to both.

WRITES
    Scores go to the real score_results_{rxn}.sqlite, tagged
    source='ab_orch' / 'ab_hunter' / 'ab_both' so they can be identified later.
    This is ordinary searcher behaviour — the molecules are genuinely scored and
    are worth keeping — but the tag makes the experiment reversible.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (BASE, os.path.join(BASE, "miner"), os.path.join(BASE, "boltz")):
    if p not in sys.path:
        sys.path.insert(0, p)

import field_prior
import hunter
import novelty
import score_store
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
        self.train_cap = 30000
        self.elite_quantile = 0.90
        self.prior_temperature = 6.0
        self.prior_floor = 0.10
        self.max_heavy_atoms = 0
        self.candidate_pool = 30000
        self.boltz_budget = 30
        for k, v in kw.items():
            setattr(self, k, v)


def load_orchestrator():
    src = open(os.path.join(BASE, "orchestrator.py")).read()
    mod = types.ModuleType("orch")
    mod.__dict__["__file__"] = os.path.join(BASE, "orchestrator.py")
    exec(compile(src[: src.index("def final_top20(")], "orchestrator.py", "exec"),
         mod.__dict__)
    return mod


def novelty_filter(df: pd.DataFrame, guard: novelty.NoveltyGuard) -> pd.DataFrame:
    """Identical gate for both arms: archive similarity + InChIKey uniqueness."""
    if df.empty:
        return df
    sims = guard.similarities(df["smiles"].tolist())
    out = df[sims < guard.max_similarity]
    if out.empty:
        return out
    return out[out["smiles"].map(lambda s: novelty.is_unique(s, TARGET))]


def pick_orchestrator(orch, rxn, manager, scored, seen, guard, args):
    a = Args(candidate_pool=args.pool, boltz_budget=args.per_arm, seed=args.seed)
    orch._GLOBAL_CONFIG = CFG
    orch._GLOBAL_MANAGER = manager
    orch._GLOBAL_ARGS = a
    sur = orch.Top20Surrogate(a.min_train, a.train_cap, a.seed)
    sur.fit(scored)
    gen = orch.CandidateGenerator(rxn, manager, sur, a)
    raw = gen.generate(scored, seen, a.candidate_pool)
    if raw.empty:
        return raw
    ranked = sur.predict(raw) if sur.trained else raw.assign(
        mu=0.0, sigma=1.0, p_improve=0.5, ei=0.0, ucb=0.0, acq=0.0)
    short = orch.choose_boltz_batch(
        ranked, min(len(ranked), max(a.boltz_budget * 6, 400)),
        np.random.default_rng(a.seed))
    ok = novelty_filter(short, guard)
    if ok.empty:
        return ok
    return orch.choose_boltz_batch(ok, a.boltz_budget, np.random.default_rng(a.seed))


def pick_hunter(rxn, manager, scored, seen, guard, args, db):
    hunter._CONFIG = CFG
    a = Args(candidate_pool=args.pool, boltz_budget=args.per_arm, seed=args.seed)
    prior = field_prior.FieldPrior(rxn, db_path=db, field_weight=0.5)
    sur = hunter.Surrogate(prior, a.min_train, a.train_cap, a.seed)
    sur.fit(scored)
    gen = hunter.Generator(rxn, manager, prior, a)
    raw = gen.generate(scored, seen, a.candidate_pool)
    if raw.empty:
        return raw
    ranked = sur.predict(raw)
    short = ranked.nlargest(min(len(ranked), max(a.boltz_budget * 6, 400)), "mu")
    ok = novelty_filter(short, guard)
    if ok.empty:
        return ok
    return hunter.select_batch(ok, a.boltz_budget, 0.10,
                               np.random.default_rng(a.seed), key="mu")


async def run_reaction(rxn: int, orch, guard, args) -> dict:
    from boltz_wrapper import BoltzWrapper

    db = score_store.score_db_path(rxn)
    tkey, tlabel = score_store.target_identity(CFG)
    scored = score_store.load_all_scored(db, rxn)
    scored = scored[np.isfinite(scored["score"])] if not scored.empty else scored
    seen = set(scored["name"]) if not scored.empty else set()

    cfg = dict(CFG)
    cfg["allowed_reaction"] = f"rxn:{rxn}"
    cfg["max_heavy_atoms"] = 10 ** 9
    manager = MoleculeManager(config=cfg, db_path=os.path.join(
        BASE, "combinatorial_db", "molecules.sqlite"))

    print(f"\n  selecting (DB has {len(scored)} scored molecules) ...", flush=True)
    t0 = time.time()
    osel = pick_orchestrator(orch, rxn, manager, scored, seen, guard, args)
    print(f"    orchestrator picked {len(osel)} | {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    hsel = pick_hunter(rxn, manager, scored, seen, guard, args, db)
    print(f"    hunter       picked {len(hsel)} | {time.time()-t0:.0f}s", flush=True)
    if osel.empty or hsel.empty:
        return {"rxn": rxn, "error": "an arm produced no candidates"}

    on, hn = set(osel["name"]), set(hsel["name"])
    both = on & hn
    allmol = {}
    for _, r in pd.concat([osel, hsel]).iterrows():
        allmol.setdefault(r["name"], r["smiles"])
    payload = [{"name": k, "smiles": v, "source": "ab"} for k, v in allmol.items()]
    print(f"    union {len(payload)} molecules to score "
          f"({len(both)} chosen by both arms) | ~{len(payload)*16/60:.0f} min",
          flush=True)

    boltz = BoltzWrapper()
    got = await hunter.boltz_score(boltz, CFG, payload, args.batch_size)
    smap = {r["name"]: float(r["boltz_score"]) for r in got}
    if not smap:
        return {"rxn": rxn, "error": "no finite scores returned"}

    recs = []
    for name, sc in smap.items():
        tag = ("ab_both" if name in both else
               "ab_orch" if name in on else "ab_hunter")
        recs.append({"name": name, "smiles": allmol[name],
                     "boltz_score": sc, "source": tag})
    score_store.write_scores_to_db(db, recs, rxn_id=rxn, round_no=None,
                                   target_key=tkey, target_label=tlabel,
                                   source="ab")

    out = {"rxn": rxn, "scored": len(smap), "overlap": len(both)}
    for label, names in (("orch", on), ("hunter", hn)):
        v = np.array([smap[n] for n in names if n in smap], dtype=float)
        if not len(v):
            continue
        out[f"{label}_n"] = int(len(v))
        out[f"{label}_mean"] = float(v.mean())
        out[f"{label}_median"] = float(np.median(v))
        out[f"{label}_best"] = float(v.max())
        for t in (0.09, 0.10, 0.11, 0.12):
            out[f"{label}_hit{t}"] = int((v > t).sum())
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--per-arm", type=int, default=30)
    ap.add_argument("--pool", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=68)
    ap.add_argument("--out", default=os.path.join(BASE, "varlab", "ab_live.json"))
    args = ap.parse_args()

    orch = load_orchestrator()
    guard = novelty.NoveltyGuard(TARGET, THR)
    results = {}
    t_all = time.time()
    for rxn in args.rxn:
        if not os.path.exists(score_store.score_db_path(rxn)):
            print(f"rxn{rxn}: no DB, skipped")
            continue
        print(f"\n{'='*78}\nLIVE A/B rxn{rxn}\n{'='*78}", flush=True)
        try:
            r = await run_reaction(rxn, orch, guard, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"rxn": rxn, "error": f"{type(e).__name__}: {e}"}
        results[rxn] = r
        print("  " + json.dumps(r), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*78}\nLIVE A/B SUMMARY ({(time.time()-t_all)/60:.0f} min)\n{'='*78}")
    print(f"{'rxn':<6}{'arm':<9}{'n':>4}{'mean':>10}{'median':>10}{'best':>10}"
          f"{'>0.09':>7}{'>0.10':>7}{'>0.11':>7}{'>0.12':>7}")
    for rxn, r in results.items():
        if "error" in r:
            print(f"  rxn{rxn}: {r['error']}")
            continue
        for arm in ("orch", "hunter"):
            if f"{arm}_n" not in r:
                continue
            print(f"  rxn{rxn:<3}{arm:<9}{r[f'{arm}_n']:>4}{r[f'{arm}_mean']:>10.5f}"
                  f"{r[f'{arm}_median']:>10.5f}{r[f'{arm}_best']:>10.5f}"
                  + "".join(f"{r[f'{arm}_hit{t}']:>7}" for t in (0.09, 0.10, 0.11, 0.12)))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
