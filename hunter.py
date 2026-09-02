#!/usr/bin/env python3
"""
hunter.py — SN68 NOVA small-molecule searcher, rebuilt around what was actually
measured to be wrong with orchestrator.py.

See PHASE1_FINDINGS.md and PHASE2_3_DESIGN.md for the measurements. In short:

  * The orchestrator's component prior collapsed to 1.56x max/uniform, i.e.
    random sampling over a 2e9 molecule space. That is the dominant defect —
    a surrogate can only rank what generation proposes.
  * Its RF/ET surrogate ranks well (spearman 0.83, 54x enrichment on `ei`) but
    cannot extrapolate: mu.max()=0.091 against a frontier of 0.121, so
    z=(mu-frontier)/sigma was negative for 100% of candidates.
  * Its composite `acq` was the *worst* of the four available rules
    (16.4x vs 23.7x for plain `ei`).
  * 40% of every Boltz round went to a sigma slice (mean score 0.066) and a
    quality-blind diversity slice, versus 0.103 for the exploit slice.
  * `neurons/crossover.py`, doing nothing but elite-anchored generation, gets a
    6.91% hit rate above 0.11 against the orchestrator's 0.82% — 8.4x better
    Boltz efficiency.

WHAT THIS DOES DIFFERENTLY

  1. Generation is anchored on evidence, not sampled near-uniformly. Component
     priors are RANK-based (scale-invariant, cannot collapse) over a UCB built
     from the local DB and the field's validator-scored submissions in
     data/rxn{N}.csv.
  2. Ranking is on the surrogate's posterior mean, never a composite. Measured
     across all five reactions (3 trials each, held-out pool, budget 150): mu
     beats the orchestrator's `acq` on every reaction (rxn1 +91%, rxn2 +96%,
     rxn5 +60%) and beats expected improvement on four of five.
  3. Budget goes 90% exploit / 10% diversity insurance. No sigma slice.
  4. Heavy atoms are treated as what they are — the score's denominator.
  5. Novelty is enforced before Boltz, and the top-20 is tracked in
     *submittable* terms, which is the only number that pays.

WHAT CHANGED AFTER EPOCHS 24793-24796 (UID 14 ranked 15/51, 12/41, 11/42, 22/50)

  The first version of the above raised the MEAN quality of what we score
  (rxn2 0.045 -> 0.074 in two days) and simultaneously shut exploration off.
  From 28 Aug it touched 0.0% new B components per day on rxn2 and 0.6% new A
  components on rxn4, having covered 37% and 26% of those pools; the maximum
  score stopped moving entirely (rxn1: 0-1 molecules above 0.10 per day on
  4,000-6,000 evaluations) and the submittable frontier flattened to a plateau
  0.004 wide against a winner's spread of 0.026. Four fixes, all measured:

  P0.1 field_prior._ucb no longer ranks components by their BEST observed
       score. That statistic was 50-80% explained by sampling effort
       (spearman(n, best) = +0.48..+0.81), so the prior was a ranking of what
       we had already tried, and it fed itself. It is now a shrunk mean plus a
       sample-size-corrected upside plus a standard-error bonus:
       spearman(n, prior) is now -0.12..+0.31. MIN_OBS 3 -> 1.
  P0.2 Surrogate._features no longer encodes an unseen component as 0.0 —
       strictly below every observed one, which made it impossible for any
       candidate containing a novel building block to survive mu-ranking.
       Unknown is imputed at the same 40th percentile the generator uses, and
       missingness is passed to the model as its own feature.
  P0.3 New `block` operator: 2-D reactant-neighbourhood cross products. See
       Generator's docstring — this is what won epoch 24796 for someone else.
  P0.4 Round-leader 3-draw confirmation is off by default. It shrank a 0.0034
       variance component while the sd of (validator - us) is 0.0095, for
       6-11% of the GPU budget. Confirmation now runs once, on the submittable
       top-20 only, where the -0.0028/molecule regression actually costs us.
  P0.5 The novelty archive is refreshed inside the round loop. It never was,
       and one stale molecule voids all 20 (the validator's check is
       all-or-nothing).

Run it exactly like the orchestrator:

    python3 hunter.py --rxn-id 2
    python3 hunter.py --rxn-id 2 --boltz-budget 180 --candidate-pool 40000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
for _p in (BASE_DIR, os.path.join(BASE_DIR, "miner"), os.path.join(BASE_DIR, "boltz")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from config.config_loader import load_config
from molecules import MoleculeManager
from utils.molecules import get_brenk_matches

import field_prior
import novelty
import rescore
import score_store

try:
    from tools import SynthonLibrary
except Exception:
    SynthonLibrary = None

try:
    from utils.molecules import compute_fingerprint_entropy
except Exception:
    compute_fingerprint_entropy = None

try:
    from boltz_wrapper import BoltzWrapper
except Exception:
    BoltzWrapper = None

# logging.basicConfig defaults to stderr, which under pm2 sends every line to
# <name>-error.log and leaves out.log holding nothing but Boltz's own prints.
# Search progress is normal output, not errors — send it to stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("hunter")

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
OUTPUT_DIR = os.path.join(BASE_DIR, "top20_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Morgan fingerprints of the REACTANTS themselves (not the products), used by
# Generator.block_scan to find each building block's structural neighbours.
# A reaction's building-block set never changes, so this is built once and
# reused for the life of the installation.
REACTANT_FP_DIR = os.path.join(BASE_DIR, "data", "reactant_fp_cache")
os.makedirs(REACTANT_FP_DIR, exist_ok=True)

# Confirmation of a round's winners — see rescore.py. One Boltz draw of the same
# molecule at the same seed moves by up to 0.010, and acquisition deliberately
# selects the top of that noise, so a round's leaders are the values least
# likely to reproduce on the validator.
#
# The threshold is NOT a constant. What deserves three draws is anything that
# could enter the submittable top 20, and that bar differs per reaction and
# rises as the search progresses. Measured today:
#
#     rxn1  #20 = 0.09119   a fixed 0.1 confirms NOTHING — every hit is missed
#     rxn2  #20 = 0.10606   a fixed 0.1 confirms ~19% of a round, most of it
#     rxn4  #20 = 0.10199   unable to reach the top 20 — wasted GPU
#     rxn5  #20 = 0.11097
#
# `--confirm-threshold auto` (the default) tracks the reaction's own submittable
# #20 minus a margin for single-draw noise; a literal float overrides it. This
# value is only the fallback when no frontier exists yet.
#
# Re-measured at max_similarity_to_historical = 0.6 (epoch 24876), which moves
# the frontier down by up to 0.0044 and, far more importantly, changes how deep
# the search must scan to find twenty submittable molecules:
#
#     rxn1  #20 = 0.0962   scan 34    80% of its top 500 still passes 0.6
#     rxn2  #20 = 0.1019   scan 21   100%
#     rxn3  #20 = 0.0951   scan 21   100%
#     rxn4  #20 = 0.0995   scan 54    31%
#     rxn5  #20 = 0.1022   scan 418    5%
#
# CONFIRMATION IS NOT OPTIONAL AT THESE MARGINS
# ---------------------------------------------
# The whole submittable band is narrower than the noise on one draw. Measured
# per reaction, #1 minus #20 against the p90 redraw range of that reaction's
# own replicates:
#
#     rxn1  band 0.0097   p90 redraw range 0.0154
#     rxn2  band 0.0056                    0.0093
#     rxn3  band 0.0085                    0.0130
#     rxn4  band 0.0139                    0.0216
#     rxn5  band 0.0104                    0.0282
#
# In every one of them the noise exceeds the entire band, so a single draw
# cannot order the twenty molecules that actually get submitted -- it can only
# tell you roughly which molecules are in the running. That is the argument for
# a wider --confirm-margin, not GPU thrift.
CONFIRM_FALLBACK_THRESHOLD = 0.1
# Extra draws per confirmed molecule, for the ROUND-LEADER confirmation. Two,
# so that every molecule clearing the round's threshold ends up measured three
# times and its stored score is the mean of those three draws — rescore.py's
# TOTAL_DRAWS. It was 0 for a while on the argument recorded under
# --confirm-extra-rounds; that argument is about how much the averaging buys,
# not about whether it works, and three draws is now the policy.
CONFIRM_EXTRA_ROUNDS = 2

# The score is (affinity_probability_binary - affinity_pred_value) / heavy_atoms.
# Heavy atoms are the denominator, so size is a direct penalty, not a neutral
# property. Measured on rxn1: molecules above 40 heavy atoms have a mean score of
# 0.026 and have never once produced a hit above 0.12, yet consumed 7.3% of the
# Boltz budget because orchestrator.py sets max_heavy_atoms = 10**9.
HEAVY_HARD_MAX = 42
HEAVY_SWEET_LO, HEAVY_SWEET_HI = 26, 36

# =============================================================================
# Per-molecule caches — BOUNDED
# =============================================================================
# These were plain dicts and never evicted anything. Every round generates
# ~22,000 molecules that have never been seen before (they are novel by
# construction — `seen` filtering guarantees it), and each one is parsed,
# fingerprinted and described, so each round permanently added ~22,000 entries
# to five caches.
#
# Measured cost per molecule, on 10,000 real SMILES from score_results_2:
#
#     _mol   (RDKit Mol)          17.56 KiB     65%
#     _fp    (float32 x 2048)      8.17 KiB     30%
#     _desc                        0.77 KiB
#     _bv    (ExplicitBitVect)     0.49 KiB
#     _heavy                       0.03 KiB
#     ----------------------------------------
#                                 27.01 KiB  ->  ~604 MiB per round
#
# That is ~18 GiB over a 14-hour run, and it matched the observed growth of the
# rxn2 searcher from 5 GiB to ~50 GiB RSS. It is also what makes multi-GPU
# scaling impossible: eight Boltz workers need ~48 GiB on a 60 GiB box, leaving
# nothing for a parent that grows without bound.
#
# Two changes. The caches are now LRU-bounded, and the fingerprint is stored as
# uint8 rather than float32 — a 4x saving for a bit vector that only ever holds
# 0 and 1. `Surrogate._features` concatenates it with float32 arrays and casts
# the result, so the promotion happens exactly where it did before.
# neurons/genetic.py and neurons/miner.py already store their fingerprints as
# uint8; this brings hunter into line with them.
#
# Caps are sized so a round's working set still hits. The passes in generate()
# are column-wise (`map(brenk_ok)` over every row, then `map(heavy_atoms)`, then
# `map(inchikey)`), so a cache smaller than one round's pool thrashes between
# passes rather than degrading gracefully. `_mol` is the exception: it is only
# ever a stepping stone to _fp/_desc/_heavy, and re-parsing 20,000 SMILES costs
# ~2 s against a ~1,600 s round, so it gets the smallest cap and saves the most.
def _cap(name: str, default: int) -> int:
    try:
        return max(1024, int(os.environ.get(name, default)))
    except ValueError:
        return default


class _Lru:
    """Insertion-ordered cache with a hard entry cap.

    Stores a sentinel for "computed and the answer was None", so an unparseable
    SMILES is not re-parsed on every pass.
    """

    __slots__ = ("_d", "cap")

    MISS = object()
    NONE = object()

    def __init__(self, cap: int):
        from collections import OrderedDict
        self._d = OrderedDict()
        self.cap = cap

    def get(self, key):
        v = self._d.get(key, self.MISS)
        if v is self.MISS:
            return self.MISS
        self._d.move_to_end(key)
        return None if v is self.NONE else v

    def put(self, key, value):
        self._d[key] = self.NONE if value is None else value
        self._d.move_to_end(key)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)
        return value

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)


_mol = _Lru(_cap("HUNTER_MOL_CACHE", 20_000))        # ~350 MiB at 17.6 KiB each
_fp = _Lru(_cap("HUNTER_FP_CACHE", 100_000))         # ~200 MiB at 2.05 KiB each
_bv = _Lru(_cap("HUNTER_BV_CACHE", 100_000))         # ~50 MiB
_desc = _Lru(_cap("HUNTER_DESC_CACHE", 100_000))     # ~77 MiB
_heavy = _Lru(_cap("HUNTER_HEAVY_CACHE", 500_000))   # ~15 MiB


def _rss_mib() -> int:
    """Resident set size of this process, for the per-round growth log."""
    try:
        with open(f"/proc/{os.getpid()}/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    except Exception:
        return 0


def _malloc_trim() -> None:
    """Return freed arena memory to the OS.

    glibc keeps freed blocks in per-thread arenas rather than handing them back,
    and a round allocates and frees several hundred MiB (the feature matrix, the
    candidate pool, two tree ensembles). Without this the caches can be bounded
    and RSS still only ratchets upward.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def mol_of(smiles: str):
    v = _mol.get(smiles)
    if v is not _Lru.MISS:
        return v
    try:
        m = Chem.MolFromSmiles(smiles)
    except Exception:
        m = None
    return _mol.put(smiles, m)


def heavy_atoms(smiles: str) -> int:
    v = _heavy.get(smiles)
    if v is not _Lru.MISS and v is not None:
        return v
    m = mol_of(smiles)
    return _heavy.put(smiles, m.GetNumHeavyAtoms() if m is not None else 0)


def fp_array(smiles: str) -> Optional[np.ndarray]:
    c = _fp.get(smiles)
    if c is not _Lru.MISS:
        return c
    m = mol_of(smiles)
    if m is None:
        return None
    # uint8, not float32: 2,048 bytes instead of 8,192 for a vector of 0s and 1s.
    arr = np.zeros(2048, dtype=np.uint8)
    arr[list(MORGAN.GetFingerprint(m).GetOnBits())] = 1
    return _fp.put(smiles, arr)


def fp_bv(smiles: str):
    c = _bv.get(smiles)
    if c is not _Lru.MISS:
        return c
    m = mol_of(smiles)
    if m is None:
        return None
    return _bv.put(smiles, MORGAN.GetFingerprint(m))


def descriptors(smiles: str) -> Optional[np.ndarray]:
    c = _desc.get(smiles)
    if c is not _Lru.MISS:
        return c
    m = mol_of(smiles)
    if m is None:
        return None
    ha = float(m.GetNumHeavyAtoms())
    na = max(1.0, float(m.GetNumAtoms()))
    arom = float(sum(1 for a in m.GetAtoms() if a.GetIsAromatic()))
    v = np.array(
        [
            Descriptors.MolWt(m) / 1000.0,
            Descriptors.MolLogP(m) / 10.0,
            Descriptors.TPSA(m) / 250.0,
            float(Lipinski.NumHDonors(m)) / 10.0,
            float(Lipinski.NumHAcceptors(m)) / 20.0,
            float(Descriptors.NumRotatableBonds(m)) / 15.0,
            ha / 100.0,
            float(Lipinski.RingCount(m)) / 10.0,
            arom / na,
            float(Lipinski.FractionCSP3(m)),
            # 1/heavy_atoms enters the score directly; give the model the term.
            1.0 / max(ha, 1.0) * 10.0,
        ],
        dtype=np.float32,
    )
    return _desc.put(smiles, v)


def brenk_ok(smiles: str) -> bool:
    m = mol_of(smiles)
    return m is not None and not get_brenk_matches(m)


def inchikey(smiles: str) -> str:
    m = mol_of(smiles)
    try:
        return Chem.MolToInchiKey(m) if m is not None else ""
    except Exception:
        return ""


def make_name(rxn: int, a: int, b: int, c: Optional[int]) -> str:
    return f"rxn:{rxn}:{a}:{b}" if c is None else f"rxn:{rxn}:{a}:{b}:{c}"


parse_components = field_prior.parse_components


# =============================================================================
# Surrogate
# =============================================================================

class Surrogate:
    """
    RandomForest + ExtraTrees over Morgan bits, physchem descriptors and
    component priors.

    Kept from the orchestrator because it measurably works — spearman 0.83 on
    held-out data, 54x enrichment at the top 150. What changed is how its output
    is used. Trees average leaf values and cannot predict above their training
    mean, so comparing `mu` to the current #20 produced a negative z for 100% of
    candidates and turned expected improvement into noise. The reference here is
    a high quantile of the model's *own* predictions, which is attainable by
    construction, so EI keeps its intended meaning.
    """

    def __init__(self, prior: field_prior.FieldPrior, min_train: int,
                 train_cap: int, seed: int):
        self.prior = prior
        self.min_train = min_train
        self.train_cap = train_cap
        self.seed = seed
        self.trained = False
        self.reference = 0.0
        self.true_frontier = -math.inf
        self.rf = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=seed,
            bootstrap=True, max_samples=0.85,
        )
        self.et = ExtraTreesRegressor(
            n_estimators=200, max_depth=22, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=seed + 1,
            bootstrap=True, max_samples=0.85,
        )

    def _features(self, name: str, smiles: str) -> Optional[np.ndarray]:
        """
        Morgan bits + physchem descriptors + component evidence.

        WHY THE COMPONENT BLOCK LOOKS LIKE THIS
        ---------------------------------------
        The previous version wrote `ca.get(a, 0.0)` — an unobserved component
        entered the model as 0.0, which is strictly below every observed
        component's score. The trees, trained on data where a low component
        prior really does mean a low score, learned exactly that, and gave the
        bottom `mu` to every candidate containing a building block we had never
        tried. Since `select_batch` picks by `mu` and even its exploration slice
        is gated at `mu >= quantile(0.70)`, there was NO path through this file
        by which a molecule containing a novel reactant could reach the GPU.
        Measured consequence: on 28 and 29 Aug the rxn2 searcher touched 0.0%
        new B components, having covered 37% of that pool; rxn4 touched 0.6%
        new A components against 26% coverage.

        This was also inconsistent with the generator, which imputes unknown
        components at the 40th percentile on the explicit ground that "unknown
        is not the same as bad" (FieldPrior.rank_weights). Both now call
        FieldPrior.impute_value(), so proposal and ranking agree.

        Missingness is instead handed to the model as what it is: three flags
        saying "this value was imputed" and three log-counts saying how much
        evidence stands behind it. A tree can then represent "few observations"
        as uncertainty rather than having to infer it from a score it was never
        shown.
        """
        fp = fp_array(smiles)
        d = descriptors(smiles)
        if fp is None or d is None:
            return None
        a, b, c = parse_components(name)
        # C is carried explicitly: rxn3 and rxn5 are three-component, and on
        # rxn3 the C position holds 18,330 of the reaction's building blocks —
        # more than A and B combined. Dropping it would blind the model to most
        # of that reaction's variation.
        vals, miss, obs = [], [], []
        for role, cid in (("A", a), ("B", b), ("C", c)):
            if cid is None:
                vals.append(0.0)
                miss.append(0.0)
                obs.append(0.0)
                continue
            table = self.prior.component_scores(role)
            v = table.get(cid)
            n = self.prior.n_obs(role, cid)
            if v is None:
                vals.append(self.prior.impute_value(role))
                miss.append(1.0)
            else:
                vals.append(v)
                miss.append(0.0)
            obs.append(math.log1p(n))
        comp = np.array(
            vals + miss + obs + [
                self.prior.pair_bonus(a, b),
                self.prior.hi_components("A").get(a, 0.0) if a is not None else 0.0,
                self.prior.hi_components("B").get(b, 0.0) if b is not None else 0.0,
                self.prior.hi_components("C").get(c, 0.0) if c is not None else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([fp, d, comp]).astype(np.float32)

    def _subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.train_cap:
            return df.copy()
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        n_elite = int(self.train_cap * 0.45)
        n_recent = int(self.train_cap * 0.25)
        n_rand = self.train_cap - n_elite - n_recent
        elite = df.head(n_elite)
        recent = df.sort_values("round", ascending=False).head(n_recent)
        rest = df.drop(index=set(elite.index) | set(recent.index), errors="ignore")
        rand = rest.sample(n=min(n_rand, len(rest)), random_state=self.seed)
        return pd.concat([elite, recent, rand], ignore_index=True) \
                 .drop_duplicates("name").head(self.train_cap)

    def fit(self, df: pd.DataFrame) -> None:
        self.trained = False
        if df.empty or len(df) < self.min_train:
            return
        s = df["score"].astype(float).sort_values(ascending=False)
        self.true_frontier = float(s.iloc[19]) if len(s) >= 20 else float(s.iloc[-1])

        train = self._subset(df)
        p80 = float(np.quantile(train["score"], 0.80))
        p95 = float(np.quantile(train["score"], 0.95))
        X, y, w = [], [], []
        for _, row in train.iterrows():
            f = self._features(row["name"], row["smiles"])
            if f is None:
                continue
            sc = float(row["score"])
            X.append(f)
            y.append(sc)
            # Upweight the tail the model has to get right, but not so hard that
            # it distorts the ranking that actually drives selection.
            w.append(4.0 if sc >= self.true_frontier else
                     2.5 if sc >= p95 else
                     1.5 if sc >= p80 else 1.0)
        if len(X) < self.min_train:
            return
        t0 = time.time()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        self.rf.fit(X, y, sample_weight=w)
        self.et.fit(X, y, sample_weight=w)
        self.trained = True
        log.info(
            "surrogate: trained on %d | true #20=%.6f | %.1fs",
            len(X), self.true_frontier, time.time() - t0,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not self.trained:
            out = df.copy()
            out["mu"] = 0.0
            out["sigma"] = 1.0
            out["ei"] = 0.0
            return out

        feats, keep = [], []
        for idx, row in df.iterrows():
            f = self._features(row["name"], row["smiles"])
            if f is not None:
                feats.append(f)
                keep.append(idx)
        if not feats:
            return df.head(0)

        X = np.asarray(feats, dtype=np.float32)
        trees = np.vstack(
            [np.vstack([t.predict(X) for t in self.rf.estimators_]),
             np.vstack([t.predict(X) for t in self.et.estimators_])]
        )
        mu = trees.mean(axis=0)
        sigma = trees.std(axis=0) + 1e-9

        # An ATTAINABLE reference. Using the real #20 here is what broke the
        # orchestrator: the model's ceiling sits below it, so every z was
        # negative and EI degenerated into an uncertainty term.
        self.reference = float(np.quantile(mu, 0.995))

        z = (mu - self.reference) / sigma
        Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

        out = df.loc[keep].copy()
        out["mu"] = mu
        out["sigma"] = sigma
        out["ei"] = (mu - self.reference) * Phi + sigma * phi
        return out


# =============================================================================
# Generation
# =============================================================================

class Generator:
    """
    Evidence-anchored candidate generation.

    The orchestrator drew A and B almost independently from a near-uniform
    prior. Over 83,307 x 24,165 that is a lottery. Here every strategy commits
    to something already known to work and varies the rest:

      elite    — both positions drawn from the prior's elite set
      anchor   — one position pinned to an elite component, the other explored
      pair     — a field-observed high-scoring (A,B) pair, one side mutated
      neighbour— synthon-similar variations of my own best molecules
      broad    — prior-weighted sampling, the only strategy that can leave the
                 region entirely; kept deliberately as insurance against the
                 prior being wrong
      block    — 2-D SAR matrix scan: expand BOTH reactants of a good molecule
                 to their structural neighbours and emit the full cross product

    `block` is the operator this file was missing. Every other strategy varies
    one position at a time against a globally-drawn partner, so none of them can
    express an A-family x B-family interaction — quality that exists only in the
    intersection of two homologous series and is invisible in either marginal.

    Epoch 24796 was lost to exactly that. The winner (UID 18, 2.1675 against our
    1.7854) drew 18 of its 20 molecules from one crossed block:

        A in {45185, 45191, 45202, 45213, 45224, 45235, 45257}   N-alkyl
             homologues of 3-chloro-4-hydroxy-pyrrolinone
        B in {6038, 6062, 6067, 6070, 6071, 6074, 6075}          N-alkyl /
             O-CH2-cyclopropyl homologues of a pyrazolyl-triazole

    Individually those blocks look mediocre in this DB — 45224 scored once at
    0.0190, 45257 once at 0.0049, 6074 never. This DB held 112 molecules with an
    A from that range and 67 with a B from it, and ZERO with both. The winning
    cell had never been sampled once.
    """

    def __init__(self, rxn_id: int, manager: MoleculeManager,
                 prior: field_prior.FieldPrior, args):
        self.rxn_id = rxn_id
        self.manager = manager
        self.sub = manager.for_rxn(rxn_id)
        self.prior = prior
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.py = random.Random(args.seed)
        self.three = bool(self.sub.is_three_component)
        self.ids = {
            "A": np.asarray(self.sub.moles_A_id, dtype=np.int64),
            "B": np.asarray(self.sub.moles_B_id, dtype=np.int64),
            "C": np.asarray(self.sub.moles_C_id, dtype=np.int64)
                 if self.three else np.asarray([], dtype=np.int64),
        }
        self._w: Dict[str, Optional[np.ndarray]] = {}
        self._elite: Dict[str, np.ndarray] = {}
        # Reactant-space fingerprint index for block_scan, built lazily per role
        # and memoised on disk: {role: (ids ndarray, [fingerprints])}.
        self._rfp: Dict[str, Optional[Tuple[np.ndarray, List[Any]]]] = {}
        self._knn: Dict[Tuple[str, int, int], List[int]] = {}
        self.synthon = None
        if SynthonLibrary is not None:
            try:
                self.synthon = SynthonLibrary(self.sub)
            except Exception as e:
                log.warning("synthon library unavailable: %s", e)

    def refresh(self) -> None:
        self._w.clear()
        self._elite.clear()
        for role in ("A", "B", "C"):
            if len(self.ids[role]) == 0:
                continue
            self._w[role] = self.prior.rank_weights(
                self.ids[role], role,
                temperature=self.args.prior_temperature,
                floor=self.args.prior_floor,
            )
            el = [c for c in self.prior.elite(role, self.args.elite_quantile)
                  if c in set(self.ids[role].tolist())]
            self._elite[role] = np.asarray(el, dtype=np.int64)
        log.info(
            "generator: elite pools A=%d B=%d%s | prior sharpness A=%.1fx B=%.1fx%s",
            len(self._elite.get("A", [])), len(self._elite.get("B", [])),
            f" C={len(self._elite.get('C', []))}" if self.three else "",
            self._sharpness("A"), self._sharpness("B"),
            f" C={self._sharpness('C'):.1f}x" if self.three else "",
        )

    def _sharpness(self, role: str) -> float:
        w = self._w.get(role)
        return float(w.max() * len(w)) if w is not None else 1.0

    def _draw(self, role: str, n: int) -> np.ndarray:
        arr = self.ids[role]
        if len(arr) == 0 or n <= 0:
            return np.asarray([], dtype=np.int64)
        return self.rng.choice(arr, size=n, replace=True, p=self._w.get(role))

    def _draw_elite(self, role: str, n: int) -> np.ndarray:
        el = self._elite.get(role)
        if el is None or len(el) == 0:
            return self._draw(role, n)
        return self.rng.choice(el, size=n, replace=True)

    def _emit(self, A, B, C) -> List[str]:
        if self.three and C is not None:
            return [make_name(self.rxn_id, int(a), int(b), int(c))
                    for a, b, c in zip(A, B, C)]
        return [make_name(self.rxn_id, int(a), int(b), None) for a, b in zip(A, B)]

    def elite_pairs(self, n: int) -> List[str]:
        if n <= 0:
            return []
        return self._emit(self._draw_elite("A", n), self._draw_elite("B", n),
                          self._draw_elite("C", n) if self.three else None)

    def anchored(self, n: int) -> List[str]:
        """
        Pin one position to an elite component and explore the others.

        The anchored position rotates across every position present, including
        C. Pinning C to its elite set on every draw would be crippling for a
        three-component reaction — rxn3 has 18,330 C components against an
        elite pool of ~283, so C carries most of that reaction's degrees of
        freedom and has to be explorable.
        """
        if n <= 0:
            return []
        roles = ["A", "B"] + (["C"] if self.three else [])
        per = n // len(roles)
        out: List[str] = []
        for i, pin in enumerate(roles):
            k = per if i < len(roles) - 1 else n - per * (len(roles) - 1)
            if k <= 0:
                continue
            draw = {r: (self._draw_elite(r, k) if r == pin else self._draw(r, k))
                    for r in roles}
            out += self._emit(draw["A"], draw["B"],
                              draw.get("C") if self.three else None)
        return out

    def pair_mutants(self, n: int) -> List[str]:
        """Take (A,B) pairs the field scored highly and mutate one side."""
        if n <= 0 or not self.prior.pairs:
            return []
        best = sorted(self.prior.pairs.items(), key=lambda kv: -kv[1][2])
        best = best[: max(200, self.args.pair_anchors)]
        if not best:
            return []
        okA = set(self.ids["A"].tolist())
        okB = set(self.ids["B"].tolist())
        best = [(p, v) for p, v in best if p[0] in okA and p[1] in okB]
        if not best:
            return []
        mutA = self._draw("A", n)
        mutB = self._draw("B", n)
        # Field pairs are (A,B) only, so C is drawn fresh — half elite, half
        # explored, rather than pinned. On rxn3 C is the largest position and
        # pinning it would collapse the strategy to a few hundred molecules.
        eliC = self._draw_elite("C", n) if self.three else None
        expC = self._draw("C", n) if self.three else None
        out = []
        for i in range(n):
            (a, b), _ = best[self.py.randrange(len(best))]
            if self.py.random() < 0.5:
                a = int(mutA[i])
            else:
                b = int(mutB[i])
            c = None
            if self.three:
                c = int(eliC[i]) if self.py.random() < 0.5 else int(expC[i])
            out.append(make_name(self.rxn_id, int(a), int(b), c))
        return out

    def neighbours(self, scored: pd.DataFrame, n: int) -> List[str]:
        if n <= 0 or self.synthon is None or scored.empty:
            return []
        seeds = scored.nlargest(min(self.args.elite_anchors, len(scored)),
                                "score")["name"].tolist()
        if not seeds:
            return []
        try:
            out = self.synthon.generate_similar_molecules(
                seeds,
                n_per_base=max(2, int(math.ceil(n / len(seeds)))),
                min_similarity=self.args.neighbour_min_sim,
            )
        except Exception as e:
            log.warning("synthon neighbours failed: %s", e)
            return []
        if len(out) > n:
            self.py.shuffle(out)
            out = out[:n]
        return out

    # -- block scan --------------------------------------------------------

    def _reactant_index(self, role: str) -> Optional[Tuple[np.ndarray, List[Any]]]:
        """
        (ids, fingerprints) for every building block valid at `role`.

        Built once and pickled: a reaction's building-block set is fixed, and
        rebuilding 83,307 Morgan fingerprints on every process start would cost
        more than the operator saves.
        """
        if role in self._rfp:
            return self._rfp[role]

        import pickle
        path = os.path.join(REACTANT_FP_DIR, f"rxn{self.rxn_id}_{role}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("n_bits") == 2048 and blob.get("ids") is not None:
                    self._rfp[role] = (np.asarray(blob["ids"], dtype=np.int64),
                                       blob["fps"])
                    log.info("block: reactant index %s = %d fingerprints (cache)",
                             role, len(blob["fps"]))
                    return self._rfp[role]
            except Exception as e:
                log.warning("block: reactant cache unreadable (%s); rebuilding", e)

        mols = {"A": getattr(self.sub, "molecules_A", None),
                "B": getattr(self.sub, "molecules_B", None),
                "C": getattr(self.sub, "molecules_C", None)}.get(role)
        if not mols:
            self._rfp[role] = None
            return None

        t0 = time.time()
        ids, fps = [], []
        for row in mols:
            cid, smi = int(row[0]), row[1]
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                continue          # attachment-point dummies parse fine; junk does not
            ids.append(cid)
            fps.append(MORGAN.GetFingerprint(m))
        if not fps:
            self._rfp[role] = None
            return None

        self._rfp[role] = (np.asarray(ids, dtype=np.int64), fps)
        try:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"n_bits": 2048, "ids": ids, "fps": fps}, f)
            os.replace(tmp, path)   # atomic: five hunters share this directory
        except Exception as e:
            log.warning("block: could not cache reactant index: %s", e)
        log.info("block: reactant index %s = %d fingerprints | %.1fs",
                 role, len(fps), time.time() - t0)
        return self._rfp[role]

    def _reactant_knn(self, role: str, cid: Optional[int], k: int) -> List[int]:
        """`cid` plus its k nearest building blocks by Tanimoto, same role."""
        if cid is None:
            return [None]
        if k <= 0:
            return [int(cid)]
        ck = (role, int(cid), int(k))
        hit = self._knn.get(ck)
        if hit is not None:
            return hit
        idx = self._reactant_index(role)
        if idx is None:
            self._knn[ck] = [int(cid)]
            return self._knn[ck]
        ids, fps = idx
        pos = np.flatnonzero(ids == int(cid))
        if not len(pos):
            self._knn[ck] = [int(cid)]
            return self._knn[ck]
        p = int(pos[0])
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[p], fps),
                          dtype=np.float32)
        sims[p] = -1.0                                  # exclude self, re-added first
        take = min(k, len(sims) - 1)
        near = np.argpartition(-sims, take)[:take]
        near = near[np.argsort(-sims[near])]
        out = [int(cid)] + [int(ids[j]) for j in near]
        self._knn[ck] = out
        return out

    def block_scan(self, scored: pd.DataFrame, seen: Set[str], n: int) -> List[str]:
        """
        Enumerate the CROSS PRODUCT of each seed molecule's reactant
        neighbourhoods — the operator that finds A-family x B-family effects.

        Cells already in `seen` are skipped, so a block that has been mined out
        costs nothing and the walk moves on to the next seed. That doubles as a
        cheap dead-block memory: re-seeding on an exhausted region yields no
        candidates instead of re-proposing molecules that get filtered later.
        """
        if n <= 0 or scored.empty:
            return []
        k = max(0, int(self.args.block_k))

        # Seeds are walked in score order but capped per reactant. Taking the
        # plain top-N would seed every block from the same over-mined building
        # block: rxn1's highest scorers are almost all `59225:*`, a component
        # this DB has evaluated 2,017 times, and its neighbourhood is mined out.
        # A first pass with these caps returned 236 candidates against a quota
        # of 1,800 for exactly that reason. Capping seed reuse spreads the scan
        # over distinct chemistry and fills the quota from live blocks.
        reuse = max(1, int(self.args.block_seed_reuse))
        pool = scored.nlargest(min(self.args.block_seed_pool, len(scored)),
                               "score")["name"].tolist()
        used_a: Dict[int, int] = {}
        used_b: Dict[int, int] = {}

        out: List[str] = []
        local: Set[str] = set()
        n_seeds = 0
        for nm in pool:
            if len(out) >= n or n_seeds >= self.args.block_seeds:
                break
            a, b, c = parse_components(nm)
            if a is None or b is None:
                continue
            if used_a.get(a, 0) >= reuse or used_b.get(b, 0) >= reuse:
                continue
            used_a[a] = used_a.get(a, 0) + 1
            used_b[b] = used_b.get(b, 0) + 1
            n_seeds += 1

            An = self._reactant_knn("A", a, k)
            Bn = self._reactant_knn("B", b, k)
            Cn = self._reactant_knn("C", c, k) if self.three else [None]
            for aa in An:
                for bb in Bn:
                    for cc in Cn:
                        if aa is None or bb is None:
                            continue
                        m = make_name(self.rxn_id, aa, bb, cc)
                        if m in seen or m in local:
                            continue
                        local.add(m)
                        out.append(m)
        log.info("  block: %d seeds -> %d unscored cells", n_seeds, len(out))
        if len(out) > n:
            self.py.shuffle(out)
            out = out[:n]
        return out

    def broad(self, n: int) -> List[str]:
        if n <= 0:
            return []
        return self._emit(self._draw("A", n), self._draw("B", n),
                          self._draw("C", n) if self.three else None)

    def generate(self, scored: pd.DataFrame, seen: Set[str],
                 n_total: int) -> pd.DataFrame:
        t0 = time.time()
        self.refresh()
        warm = len(scored) >= self.args.min_train
        # Weights measured, not guessed. For each reaction, 250 top-1% molecules
        # were expanded by each move and the resulting cells' hit rate at the
        # reaction's own p99.5 frontier compared against the same expansion of
        # 250 RANDOM scored molecules (a paired design, so the bias from both
        # arms being already-scored largely cancels):
        #
        #     move                      rxn1   rxn2   rxn3   rxn4   rxn5
        #     one-sided, same A         1.5x   1.3x   0.7x   0.5x   0.6x
        #     one-sided, same B         2.9x   0.6x   0.8x   0.6x   0.3x
        #     block (both sides, k=2)  21.1x   4.8x   5.2x   4.1x   2.8x
        #
        # `elite`, `anchored` and `pair` are all one-sided moves — they pin or
        # vary one position and draw the partner from the prior — and they are
        # at or BELOW the random-block baseline in four of five reactions.
        # `pair` is the weakest of the three and takes the largest cut. What
        # they are still worth is reaching regions `block` cannot see at all:
        # block only proposes cells adjacent to something already scored, so it
        # harvests, and `broad` + `anchored` are what discover. Hence the share
        # moved out of `pair`/`elite` goes mostly to `broad`, which with the
        # de-biased prior (field_prior._ucb) is no longer near-random sampling.
        base = ({"elite": 0.12, "anchored": 0.22, "pair": 0.06,
                 "neighbour": 0.06, "broad": 0.34}
                if warm else
                {"elite": 0.12, "anchored": 0.22, "pair": 0.06,
                 "neighbour": 0.06, "broad": 0.54})
        bf = float(np.clip(self.args.block_frac if warm
                           else self.args.block_frac * 0.5, 0.0, 0.9))
        scale = (1.0 - bf) / sum(base.values())
        mix = {"block": bf}
        mix.update({k: v * scale for k, v in base.items()})

        names: List[str] = []
        origin: Dict[str, str] = {}
        for label, frac in mix.items():
            k = int(n_total * frac)
            t = time.time()
            if label == "block":
                got = self.block_scan(scored, seen, k)
            elif label == "elite":
                got = self.elite_pairs(k)
            elif label == "anchored":
                got = self.anchored(k)
            elif label == "pair":
                got = self.pair_mutants(k)
            elif label == "neighbour":
                got = self.neighbours(scored, k)
            else:
                got = self.broad(k)
            log.info("  gen/%-10s %6d candidates | %.2fs", label, len(got),
                     time.time() - t)
            for nm in got:
                origin.setdefault(nm, label)
            names += got

        # `block` and `neighbour` can both come up short — a block whose cells
        # are all already scored yields nothing, which is the correct behaviour
        # but must not silently shrink the round's pool.
        shortfall = n_total - len(names)
        if shortfall > n_total * 0.05:
            t = time.time()
            got = self.broad(shortfall)
            log.info("  gen/%-10s %6d candidates | %.2fs (top-up)", "broad+",
                     len(got), time.time() - t)
            for nm in got:
                origin.setdefault(nm, "broad")
            names += got

        uniq, local = [], set()
        for nm in names:
            if nm in seen or nm in local:
                continue
            local.add(nm)
            uniq.append(nm)
        if not uniq:
            return pd.DataFrame(columns=["name", "smiles"])

        cfg = dict(self.manager.config) if hasattr(self.manager, "config") else {}
        cfg = dict(_CONFIG)
        cfg["allowed_reaction"] = f"rxn:{self.rxn_id}"
        cfg["max_heavy_atoms"] = HEAVY_HARD_MAX
        valid = self.manager.validate_molecules(cfg, pd.DataFrame({"name": uniq}))
        if valid.empty:
            return valid

        n0 = len(valid)
        valid = valid[valid["smiles"].map(brenk_ok)]
        n1 = len(valid)
        if valid.empty:
            return valid
        valid = valid[valid["smiles"].map(lambda s: heavy_atoms(s) <= HEAVY_HARD_MAX)]
        valid["inchikey"] = valid["smiles"].map(inchikey)
        valid = valid[valid["inchikey"] != ""].drop_duplicates("inchikey")
        valid["strategy"] = valid["name"].map(origin).fillna("unknown")
        log.info(
            "  gen/total   %6d proposed -> %d valid -> %d post-BRENK -> %d unique | %.1fs",
            len(uniq), n0, n1, len(valid), time.time() - t0,
        )
        return valid.reset_index(drop=True)


# =============================================================================
# Selection
# =============================================================================

def size_multiplier(smiles: str) -> float:
    """
    Mild preference for the measured sweet spot. Deliberately gentle — the
    surrogate already sees 1/heavy_atoms as a feature, and a hard preference
    would forbid the large-but-excellent molecules that do exist.
    """
    h = heavy_atoms(smiles)
    if HEAVY_SWEET_LO <= h <= HEAVY_SWEET_HI:
        return 1.0
    if h < HEAVY_SWEET_LO:
        return 0.97
    return max(0.80, 1.0 - 0.02 * (h - HEAVY_SWEET_HI))


def select_batch(ranked: pd.DataFrame, budget: int,
                 explore_frac: float, rng: np.random.Generator,
                 key: str = "mu") -> pd.DataFrame:
    """
    90% by expected improvement, 10% diversity insurance.

    The orchestrator spent 25% of every round on its most-uncertain candidates,
    whose mean score was 0.066 against 0.103 for its exploit slice, plus 15% on
    a diversity term that barely weighted quality. Uncertainty sampling is the
    right move when the model is the bottleneck; here generation was, and the
    model already ranks at 20-50x enrichment. The exploration that remains is
    aimed at *archive* distance — being far from what other miners submit is
    what earns, because the validator divides a score by the number of UIDs
    submitting the same molecule. Structural novelty for its own sake does not.

    The first version of this used 25%, and a held-out measurement showed that
    was the same mistake in smaller form: going 0.25 -> 0.00 recovered 13 hits
    above 0.11 on rxn5 and 5 on rxn2. It is not zero because the validator
    requires atom-pair fingerprint entropy >= 0.25 across the submitted 20,
    and a pure-exploit portfolio can fail that. Note a single-round offline test
    can see this slice's cost but not its across-round benefit, so 0.10 is a
    deliberate compromise rather than the measured optimum.

    That constraint tightened at epoch 24876 and this figure predates it: the
    same twenty molecules score about 0.06 lower over 2048 atom-pair bits than
    over 167 MACCS keys, while the floor went from 0.1 to 0.25. 0.10 was chosen
    against the old, slacker constraint and is worth re-measuring.
    """
    if ranked.empty:
        return ranked
    if len(ranked) <= budget:
        return ranked.copy()

    ranked = ranked.copy()
    base = ranked[key] if key in ranked.columns else ranked["mu"]
    # The size multiplier scales a positive quality estimate. Applying it to a
    # value that can go negative would invert the preference for large
    # molecules, so shift onto a non-negative footing first.
    shifted = base - min(0.0, float(base.min()))
    ranked["rank_adj"] = shifted * ranked["smiles"].map(size_multiplier)

    n_exploit = int(round(budget * (1.0 - explore_frac)))
    top = ranked.nlargest(n_exploit, "rank_adj")
    picked = list(top.index)

    rest = ranked.drop(index=picked, errors="ignore")
    n_explore = budget - len(picked)
    if n_explore > 0 and not rest.empty:
        # Among candidates that are still plausibly good, prefer those least
        # like what has already been picked this round.
        floor = float(rest["mu"].quantile(0.70))
        pool = rest[rest["mu"] >= floor]
        if pool.empty:
            pool = rest
        pool = pool.sample(n=min(len(pool), max(1500, n_explore * 20)),
                           random_state=int(rng.integers(1 << 31)))
        chosen_fps = [fp_bv(s) for s in top["smiles"].head(200)]
        chosen_fps = [f for f in chosen_fps if f is not None]
        rows = []
        for idx, row in pool.iterrows():
            f = fp_bv(row["smiles"])
            if f is None:
                continue
            sim = (max(DataStructs.BulkTanimotoSimilarity(f, chosen_fps))
                   if chosen_fps else 0.0)
            rows.append(((1.0 - sim) + 0.5 * float(row["mu"]) /
                         max(abs(float(ranked["mu"].max())), 1e-9), idx))
        rows.sort(reverse=True)
        for _, idx in rows[:n_explore]:
            picked.append(idx)

    if len(picked) < budget:
        for idx in ranked.nlargest(budget * 2, "rank_adj").index:
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= budget:
                break
    return ranked.loc[picked[:budget]].reset_index(drop=True)


# =============================================================================
# Confirmation threshold
# =============================================================================

def resolve_confirm_threshold(args, top20: pd.DataFrame) -> Tuple[float, str]:
    """
    Decide what score earns three draws this round.

    `--confirm-threshold auto` returns (submittable #20) - margin. That adapts on
    both axes the operator asked about: across reactions, because each has its
    own frontier (0.091 on rxn1 vs 0.111 on rxn5), and across search progress,
    because #20 climbs as the top 20 improves and the bar climbs with it.

    The margin exists because the frontier is itself measured from single draws.
    A molecule sitting just under #20 may really be above it — measured Boltz
    spread at a fixed seed is ~0.0025 typically and up to 0.0102 worst case — so
    a catch radius of about two typical noise widths is the point.

    Returns (threshold, human-readable reason) so the round log can explain
    itself rather than printing a bare number.
    """
    raw = str(args.confirm_threshold).strip().lower()
    if raw not in ("auto", "adaptive"):
        return float(raw), "fixed by --confirm-threshold"

    n_needed = int(_CONFIG.get("num_molecules", 20))
    if top20 is None or len(top20) < n_needed:
        return (args.confirm_floor,
                f"floor — only {0 if top20 is None else len(top20)}/{n_needed} "
                f"submittable so far, no frontier yet")

    frontier = float(top20["score"].iloc[-1])
    thr = max(args.confirm_floor, frontier - args.confirm_margin)
    why = f"auto: submittable #{n_needed}={frontier:.5f} - margin {args.confirm_margin}"
    if thr <= args.confirm_floor + 1e-12 and frontier - args.confirm_margin < args.confirm_floor:
        why += f" (clamped up to floor {args.confirm_floor})"
    return thr, why


def cap_confirm_threshold(threshold: float, scores: Sequence[float],
                          max_frac: float) -> Tuple[float, int]:
    """
    Stop a weak round from spending its whole GPU slot on confirmation.

    Each confirmed molecule costs two extra Boltz calls, so confirming 50% of a
    150-molecule round adds 150 predictions — as long as the round itself. If
    more than `max_frac` of the round clears the threshold, raise it to the
    matching quantile and confirm only the best of them.
    """
    vals = np.asarray([v for v in scores if np.isfinite(v)], dtype=float)
    if not len(vals) or max_frac >= 1.0:
        return threshold, int((vals > threshold).sum()) if len(vals) else 0
    n_over = int((vals > threshold).sum())
    allowed = max(1, int(math.floor(len(vals) * max_frac)))
    if n_over <= allowed:
        return threshold, n_over
    raised = float(np.quantile(vals, 1.0 - max_frac))
    return max(threshold, raised), int((vals > max(threshold, raised)).sum())


# =============================================================================
# Boltz
# =============================================================================

async def boltz_score(boltz, config: Dict[str, Any],
                      molecules: List[Dict[str, Any]],
                      batch_size: int,
                      mark_at: float = float("inf"),
                      on_batch=None) -> List[Dict[str, Any]]:
    """
    Score `molecules` in batches.

    `on_batch(records)` is called with results as soon as they are read, so a
    crash never costs more than the work still in flight: writing once at the
    end of a round means a kill at molecule 149 of 150 throws away all 149.

    A batch is not one unit of work under multi.run. MultiGPUBoltz fans it out
    across every worker as ~30-molecule chunks that land minutes apart, so when
    the wrapper exposes the chunk hook, `on_batch` is driven per chunk instead
    of per batch. At batch_size=480 on 8 workers that cuts the window in which
    a kill discards finished scores from ~19 minutes to ~7, and each chunk's
    molecules reach the log as it lands rather than 16 chunks' worth at once.
    Against a plain BoltzWrapper there is no hook and the batch stays the unit.
    """
    if not molecules:
        return []
    targets = config["small_molecule_target"]
    subnet = {
        "small_molecule_target": targets,
        "small_molecule_target_clip_interval":
            config["small_molecule_target_clip_interval"],
        "boltz_mode": config.get("boltz_mode", "max"),
        "boltz_metric": config.get(
            "boltz_metric",
            ["affinity_probability_binary", "affinity_pred_value"]),
        "combination_strategy": config.get(
            "combination_strategy", "heavy_atom_normalization"),
    }

    # Chunks come back in completion order, not input order, so the [i/N] index
    # has to be looked up rather than counted.
    pos: Dict[str, int] = {}
    for i, m in enumerate(molecules, start=1):
        pos.setdefault(m["smiles"], i)

    def log_molecules(records):
        """Every molecule, in the order Boltz was handed it.

        Without this the log only shows aggregates, so a single strong hit is
        invisible until the round ends.
        """
        for r in sorted(records, key=lambda x: pos.get(x.get("smiles", ""), 0)):
            sc, smi = r["boltz_score"], r.get("smiles", "")
            log.info("    [%4d/%d] %.6f  ha=%2d  %s%s",
                     pos.get(smi, 0), len(molecules), sc,
                     heavy_atoms(smi) if smi else 0, r["name"],
                     " <<<" if sc >= mark_at else "")

    out: List[Dict[str, Any]] = []
    for start in range(0, len(molecules), batch_size):
        batch = molecules[start:start + batch_size]
        # One smiles can carry several records; the pool scores it once and the
        # score belongs to every one of them.
        by_smiles: Dict[str, List[Dict[str, Any]]] = {}
        for x in batch:
            by_smiles.setdefault(x["smiles"], []).append(x)
        vm = {0: {"smiles": [x["smiles"] for x in batch],
                  "names": [x["name"] for x in batch]}}
        sd = {0: {"target_scores": [[]], "antitarget_scores": [[]],
                  "entropy": None, "entropy_boltz": None,
                  "block_submitted": None, "push_time": ""}}
        t0 = time.time()
        done: Dict[str, float] = {}   # smiles already logged and persisted

        def on_chunk(chunk_scores, _components, _by=by_smiles, _done=done):
            """Persist and print one worker's chunk the moment it lands."""
            recs = []
            for smi, per_target in (chunk_scores or {}).items():
                if smi in _done or smi not in _by:
                    continue
                try:
                    sc = float(per_target.get(targets[0]))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(sc):
                    continue
                for rec in _by[smi]:
                    r = dict(rec)
                    r["boltz_score"] = sc
                    recs.append(r)
            if not recs:
                return
            vals = [r["boltz_score"] for r in recs]
            log.info("  boltz chunk: %d/%d scored | best %.6f mean %.6f | +%.0fs",
                     len(recs), len(chunk_scores or {}), max(vals),
                     float(np.mean(vals)), time.time() - t0)
            log_molecules(recs)
            if on_batch is not None:
                try:
                    on_batch(recs)
                except Exception as e:
                    # Leave these out of `done` so the end-of-batch write retries.
                    log.error("could not persist chunk: %s", e, exc_info=True)
                    return
            _done.update({r["smiles"]: r["boltz_score"] for r in recs})

        hook = hasattr(boltz, "on_chunk_scores")
        if hook:
            boltz.on_chunk_scores = on_chunk
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: boltz.score_molecules(vm, sd, subnet))
        finally:
            # The pool is shared with rescore's own wrapper; never leave a
            # callback bound to this batch armed for someone else's call.
            if hook:
                boltz.on_chunk_scores = None
        got = sd.get(0, {}).get("molecule_scores", [])
        scores = got[0] if got else []
        if len(scores) != len(batch):
            fmap = getattr(boltz, "final_boltz_scores", {}).get(0, {}).get(targets[0], {})
            scores = [fmap.get(x["smiles"], -math.inf) for x in batch]
        vals, fresh = [], []
        for rec, sc in zip(batch, scores):
            try:
                sc = float(sc)
            except Exception:
                continue
            if not np.isfinite(sc):
                continue
            r = dict(rec)
            r["boltz_score"] = sc
            out.append(r)
            vals.append(sc)
            if rec["smiles"] not in done:
                fresh.append(r)
        if vals:
            log.info(
                "  boltz %d-%d: %d/%d | best %.6f mean %.6f | %.0fs",
                start + 1, min(start + len(batch), len(molecules)),
                len(vals), len(batch), max(vals), float(np.mean(vals)),
                time.time() - t0,
            )
            # Anything the chunk hook already printed is not repeated here.
            if fresh:
                log_molecules(fresh)

        if on_batch is not None and fresh:
            try:
                on_batch(fresh)
            except Exception as e:
                log.error("could not persist batch: %s", e, exc_info=True)

        # BoltzWrapper defines _cleanup_files() but never calls it, so inputs and
        # prediction outputs accumulate for the life of the process — 65 MB in
        # ten minutes on a live run, and Boltz rescans every stale input on each
        # batch. Scores are already extracted from score_dict by this point, so
        # the workspace can be dropped. Safe only because the directory is
        # isolated per process; clear_boltz_workspace refuses a shared one.
        try:
            rescore.clear_boltz_workspace(boltz)
        except Exception as e:
            log.debug("could not clear boltz workspace: %s", e)
    return out


# =============================================================================
# Submittable top-20
# =============================================================================

def submittable_top20(db_path: str, rxn_id: int, guard: novelty.NoveltyGuard,
                      target: str, config: Dict[str, Any],
                      scan: int = 4000) -> pd.DataFrame:
    """
    The only number that pays.

    orchestrator.py's `final_top20` walks the score-ordered DB but stops after
    `max(100, 20*5)` rows. When most high scorers are already archived — which
    is the normal state, since every molecule anyone submits enters the archive
    — it runs out of rows before finding 20 submittable ones and reports a
    top-20 that cannot actually be submitted. Scanning deeper is cheap and makes
    the reported number honest.
    """
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT molecule_name, smiles, score FROM scored_molecules "
            "WHERE available=TRUE AND smiles IS NOT NULL AND smiles!='' "
            "AND molecule_name LIKE ? ORDER BY score DESC LIMIT ?",
            (f"rxn:{rxn_id}:%", scan),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["name", "smiles", "score"])

    n = int(config.get("num_molecules", 20))
    keep, seen_ik = [], set()
    for name, smiles, score in rows:
        if len(keep) >= n:
            break
        ik = inchikey(smiles)
        if not ik or ik in seen_ik:
            continue
        if not brenk_ok(smiles):
            continue
        if not novelty.is_unique(smiles, target):
            continue
        if guard.max_similarity_to_archive(smiles) >= guard.max_similarity:
            continue
        seen_ik.add(ik)
        keep.append({"name": name, "smiles": smiles, "score": float(score)})
    return pd.DataFrame(keep)


def export_top20(df: pd.DataFrame, rxn_id: int, target: str) -> None:
    if df.empty:
        return
    csv = os.path.join(OUTPUT_DIR, f"TOP20_rxn{rxn_id}_{target}.csv")
    txt = os.path.join(OUTPUT_DIR, f"TOP20_rxn{rxn_id}_{target}.txt")
    df.to_csv(csv, index=False)
    with open(txt, "w", encoding="utf-8") as f:
        f.write(",".join(df["name"].tolist()))
    log.info("=" * 74)
    log.info("SUBMITTABLE TOP %d | sum=%.6f | #%d=%.6f",
             len(df), float(df["score"].sum()), len(df),
             float(df["score"].iloc[-1]))
    for i, (_, r) in enumerate(df.iterrows(), 1):
        log.info("  %02d  %.6f  %s", i, float(r["score"]), r["name"])
    log.info("csv=%s", csv)
    log.info("=" * 74)


# =============================================================================
# Main
# =============================================================================

_CONFIG: Dict[str, Any] = {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rxn-id", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--boltz-budget", type=int, default=150)
    p.add_argument("--candidate-pool", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--min-train", type=int, default=500)
    p.add_argument("--train-cap", type=int, default=30000)
    p.add_argument("--block-frac", type=float, default=0.30,
                   help="fraction of the candidate pool built by the 2-D "
                        "reactant-block scan. This is the only operator that "
                        "can express an A-family x B-family interaction; epoch "
                        "24796's winner took 18 of 20 molecules from one such "
                        "block and this DB had never sampled a single cell of "
                        "it (112 molecules with that A range, 67 with that B "
                        "range, 0 with both)")
    p.add_argument("--block-seeds", type=int, default=400,
                   help="cap on blocks expanded per round. block_scan is meant "
                        "to be SUPPLY-limited, not seed-limited: it stops as "
                        "soon as --block-frac is met, and the cap only prevents "
                        "a runaway on a reaction with tiny blocks. At 60 it was "
                        "the binding constraint and starved the operator")
    p.add_argument("--block-seed-pool", type=int, default=3000,
                   help="how deep into the score-ordered DB block_scan may look "
                        "for seeds that still pass the per-reactant cap")
    p.add_argument("--block-seed-reuse", type=int, default=1,
                   help="how many times one reactant may anchor a block in a "
                        "single round. 1 forces every block onto distinct "
                        "chemistry; without it rxn1 would seed nearly every "
                        "block from component 59225, which has already taken "
                        "2,017 evaluations and whose neighbourhood is exhausted")
    p.add_argument("--block-k", type=int, default=6,
                   help="structural neighbours per reactant position, so a "
                        "two-component block is (k+1)^2 = 49 cells")
    p.add_argument("--elite-anchors", type=int, default=40)
    p.add_argument("--pair-anchors", type=int, default=400)
    p.add_argument("--neighbour-min-sim", type=float, default=0.35)
    p.add_argument("--elite-quantile", type=float, default=0.90,
                   help="component quality quantile that counts as elite")
    p.add_argument("--prior-temperature", type=float, default=6.0,
                   help="prior sharpness; ~T times uniform weight on the best "
                        "component (orchestrator.py achieved 1.56)")
    p.add_argument("--prior-floor", type=float, default=0.10,
                   help="uniform mass kept so nothing is unreachable")
    p.add_argument("--field-weight", type=float, default=0.5,
                   help="blend between field CSV and local DB component stats; "
                        "a field-only prior measured WORSE than random at the "
                        "top of the ranking, so this stays at or below 0.5")
    p.add_argument("--rank-key", choices=["mu", "ei"], default="mu",
                   help="acquisition key. Measured over all 5 reactions x 3 "
                        "trials: mu beats ei on 4 of 5 and beats the "
                        "orchestrator's composite acq on all 5. EI degenerates "
                        "here because the tree ensemble cannot predict above "
                        "its training mean, so almost every candidate sits "
                        "below any useful reference and the sigma term takes "
                        "over")
    p.add_argument("--explore-frac", type=float, default=0.10,
                   help="fraction of the Boltz budget spent away from the top of "
                        "the ranking. Measured cost of 0.25 on a held-out pool: "
                        "13 hits >0.11 on rxn5, 5 on rxn2. Kept non-zero only "
                        "because the validator requires atom-pair entropy "
                        ">= 0.25 across the submitted 20, so an all-exploit "
                        "portfolio risks failing that check. Measured at 0.1 "
                        "on MACCS keys; the floor tightened at epoch 24876")
    p.add_argument("--strict-pool-mult", type=int, default=6)
    p.add_argument("--max-rounds", type=int, default=10 ** 9)
    p.add_argument("--seed", type=int, default=68)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--confirm-threshold", default="auto", metavar="FLOAT|auto",
                   help="score above which a molecule is re-scored to 3 draws "
                        "and its score replaced by the mean. 'auto' (default) "
                        "uses this reaction's own submittable #20 minus "
                        "--confirm-margin, so it adapts both per reaction and as "
                        "the search progresses. Measured #20 at similarity "
                        "0.6: rxn1 0.0962, rxn2 0.1019, rxn3 0.0951, rxn4 "
                        "0.0995, rxn5 0.1022 — a fixed 0.1 confirms nothing on "
                        "rxn1/rxn3 and wastes GPU on rxn2/5")
    p.add_argument("--confirm-margin", type=float, default=0.010,
                   help="how far below the frontier still earns confirmation. "
                        "Raised from 0.005 after measuring the redraw spread "
                        "on all 2,537 molecules that carry >=2 replicate draws "
                        "in the five score DBs, rather than the handful the "
                        "old 0.0025/0.0102 figures came from: median sd "
                        "0.0029, but p90 RANGE 0.0163 and max 0.1417. A 0.005 "
                        "margin caught under half of real redraw variation. "
                        "0.010 catches 78% and is the largest value that "
                        "still leaves every reaction under --confirm-max-frac "
                        "(worst is rxn5 at 7.5%% of a round). See the "
                        "confirmation note above for why this matters more "
                        "than it looks (default 0.010)")
    p.add_argument("--confirm-floor", type=float, default=0.08,
                   help="threshold never drops below this, so an early or weak "
                        "round does not confirm everything it scored")
    p.add_argument("--confirm-max-frac", type=float, default=0.10,
                   help="hard cap on the fraction of a round that gets "
                        "confirmed; above it the threshold is raised to the "
                        "matching quantile. Each confirmation costs 2 extra "
                        "Boltz calls. Raised from 0.08 to leave the wider "
                        "--confirm-margin room: at 0.010 the busiest reaction "
                        "confirms 7.5%% of a round, close enough to 0.08 that "
                        "a good round would have been silently clipped back "
                        "to the 92nd percentile (default 0.10)")
    p.add_argument("--confirm-extra-rounds", type=int, default=CONFIRM_EXTRA_ROUNDS,
                   help="extra draws per ROUND-LEADER molecule, on top of the "
                        "round's own draw: 2 gives three draws in total and a "
                        "stored score that is their mean. Known trade-off, "
                        "kept here because it is the reason this was once 0: "
                        "our own same-seed replicate "
                        "spread is sd=0.0034 (rxn1 .0033 rxn2 .0026 rxn3 .0033 "
                        "rxn4 .0046), but the sd of (validator score - our "
                        "score) is 0.0095 on n=519 molecules other miners "
                        "submitted that we had also scored. Averaging three "
                        "local draws therefore moves selection error from "
                        "~0.0100 to ~0.0097 — about 9%% of the variance that "
                        "decides our score — for 6-11%% of the GPU budget "
                        "(measured from molecule_replicates: 28 Aug rxn2 wrote "
                        "834 extra predictions against 7,310 fresh ones). It is "
                        "also one-sided: only molecules above the threshold are "
                        "re-drawn, so it can demote a lucky molecule but never "
                        "promote an unlucky one — it corrects the reported "
                        "number without repairing the ranking underneath. Set "
                        "0 to disable and leave confirmation to "
                        "--top20-confirm-draws alone")
    p.add_argument("--no-confirm", action="store_true",
                   help="skip the confirmation of this round's leaders")
    p.add_argument("--top20-confirm-draws", type=int, default=2,
                   help="extra draws given to any molecule that has entered the "
                        "SUBMITTABLE TOP 20 and has not been confirmed before. "
                        "Matches --confirm-extra-rounds so both gates settle a "
                        "molecule at the same three draws; a lower value here "
                        "would flag it `rescored` after two and lock it out of "
                        "the third for good. "
                        "This is the only set that is ever submitted, so it is "
                        "the only set whose noise costs anything: our top-20 "
                        "regressed by -0.0028 per molecule against the "
                        "validator across epochs 24793-24796 (-0.079, -0.024, "
                        "-0.067, -0.091 on the sum), and in 24793 and 24795 "
                        "that loss alone exceeded the gap to the winner. Cost "
                        "is bounded at 20 predictions per round and falls to "
                        "~0 once the frontier stops moving, because "
                        "rescore.confirm_high_scorers never pays twice for the "
                        "same molecule. 0 disables")
    p.add_argument("--dry-run", action="store_true",
                   help="generate, rank and report — no Boltz, no DB writes")
    return p.parse_args()


async def main() -> None:
    global _CONFIG
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config()
    _CONFIG = config
    rxn_id = args.rxn_id
    target = config["small_molecule_target"][0]
    target_key, target_label = score_store.target_identity(config)
    db_path = score_store.score_db_path(rxn_id)

    cfg = dict(config)
    cfg["allowed_reaction"] = f"rxn:{rxn_id}"
    cfg["max_heavy_atoms"] = HEAVY_HARD_MAX
    manager = MoleculeManager(config=cfg, db_path=DB_PATH)

    if not args.dry_run:
        # Refuse to write into a damaged file. score_results_3.sqlite developed
        # B-tree corruption during concurrent testing and still answered simple
        # SELECTs for a while — the failure only surfaced on a query that
        # touched the bad pages. Writing more rows into that state risks
        # compounding the damage, and a silent partial DB is worse than a hard
        # stop.
        import sqlite3 as _sq
        if os.path.exists(db_path):
            try:
                with _sq.connect(db_path) as _c:
                    verdict = _c.execute("PRAGMA quick_check").fetchone()[0]
            except Exception as e:
                verdict = f"unreadable: {e}"
            if verdict != "ok":
                raise SystemExit(
                    f"{db_path} failed its integrity check:\n  {verdict}\n"
                    f"Refusing to write. Recover it first — salvage readable "
                    f"rows into a fresh file (see PHASE5_ALL_REACTIONS.md) "
                    f"rather than continuing on a damaged B-tree.")
        score_store.init_score_results_db(db_path, rxn_id, target_key, target_label)
        score_store.init_variance_tables(db_path)

    prior = field_prior.FieldPrior(
        rxn_id, db_path=db_path, field_weight=args.field_weight, logger=log)
    log.info(prior.summary())

    guard = novelty.NoveltyGuard(target, novelty.config_threshold(config))
    surrogate = Surrogate(prior, args.min_train, args.train_cap, args.seed)
    generator = Generator(rxn_id, manager, prior, args)
    rng = np.random.default_rng(args.seed)

    boltz = None
    if not args.dry_run:
        if BoltzWrapper is None:
            raise SystemExit("BoltzWrapper unavailable — run from the nova-4090 root "
                             "in the same environment miner/miner.py uses.")
        boltz = BoltzWrapper()
        # BoltzWrapper hardcodes boltz/boltz_tmp_files/{inputs,outputs} for every
        # instance, and its _cleanup_files() deletes every *.yaml in that input
        # directory plus the whole results tree. Two searchers sharing it will
        # delete each other's inputs mid-prediction. Give this process its own
        # workspace before anything is written.
        try:
            rescore.isolate_boltz_workspace(
                boltz, tag=f"hunter{rxn_id}_{os.getpid()}")
            log.info("boltz workspace: %s", boltz.input_dir)
        except Exception as e:
            raise SystemExit(
                f"could not isolate a Boltz workspace ({e}). Refusing to run on "
                f"the shared directory — concurrent searchers would corrupt each "
                f"other's predictions.")

    log.info("hunter | rxn=%d target=%s | A=%d B=%d%s",
             rxn_id, target_label,
             len(generator.ids["A"]), len(generator.ids["B"]),
             f" C={len(generator.ids['C'])}" if generator.three else "")
    log.info("similarity gate=%.2f (from config) | heavy atoms <= %d | db=%s",
             guard.max_similarity, HEAVY_HARD_MAX, db_path)

    for round_no in range(1, args.max_rounds + 1):
        t_round = time.time()
        scored = score_store.load_all_scored(db_path, rxn_id)
        if not scored.empty:
            scored = scored[np.isfinite(scored["score"])]
        seen = set(scored["name"].tolist()) if not scored.empty else set()

        # The archive grows every epoch — roughly 900 new molecules from ~45
        # miners — and a molecule that was novel when it was scored stops being
        # novel the moment anyone submits it. novelty.py documents refresh() as
        # "call this periodically inside long runs" and this loop never did, so
        # a searcher running for days screened against a days-old archive.
        #
        # That is not a cosmetic error: the validator's molecule check is
        # all-or-nothing (molecule_validity.py — one molecule failing any test
        # `break`s the loop and the whole 20-molecule submission is discarded),
        # which is the shape of the 14 consecutive -999.99 epochs 24760-24773.
        if round_no == 1 or round_no % 2 == 0:
            try:
                n_arch = guard.refresh()
                # The InChIKey half of the gate has its own cache, and it goes
                # stale in exactly the same way.
                novelty._INCHIKEY_CACHE.pop(target, None)
                log.info("novelty: archive refreshed — %d fingerprints", n_arch)
            except Exception as e:
                log.warning("novelty: archive refresh failed (%s); "
                            "continuing on the cached archive", e)

        # Refreshed every other round rather than every fifth: a round is ~43
        # minutes, so the old cadence left component statistics up to 3.5 hours
        # behind the DB they are computed from, and block_scan seeds off exactly
        # those statistics.
        #
        # This has to happen BEFORE fit(). The surrogate's component features are
        # read from this object, so refreshing between fit() and predict() would
        # train on one prior and rank with another — harmless when the features
        # were raw maxima, but not now that they carry an imputed value for
        # unobserved components and a log-count of the evidence behind each one.
        if round_no == 1 or round_no % 2 == 0:
            prior.__init__(rxn_id, db_path=db_path,
                           field_weight=args.field_weight, logger=None)

        surrogate.fit(scored)

        before = submittable_top20(db_path, rxn_id, guard, target, config)
        sum_before = float(before["score"].sum()) if len(before) else 0.0

        raw = generator.generate(scored, seen, args.candidate_pool)
        if raw.empty:
            log.info("[round %d] nothing generated", round_no)
            await asyncio.sleep(args.sleep)
            continue

        ranked = surrogate.predict(raw)
        if ranked.empty:
            await asyncio.sleep(args.sleep)
            continue

        # Novelty is expensive, so apply it to an acquisition-ordered shortlist
        # rather than the whole pool — but apply it BEFORE Boltz, never after.
        #
        # Widen the shortlist until the budget is actually filled. A fixed
        # shortlist silently under-fills whenever the archive is dense: on a
        # live rxn1 round a budget of 40 came back with 21 molecules, wasting
        # half the GPU slot for that round.
        key = args.rank_key if surrogate.trained else "mu"
        order = ranked.sort_values(key, ascending=False)
        novel = ranked.head(0)
        taken = 0
        step = max(args.boltz_budget * args.strict_pool_mult, 400)
        while taken < len(order) and len(novel) < args.boltz_budget:
            chunk = order.iloc[taken:taken + step]
            taken += step
            if chunk.empty:
                break
            sims = guard.similarities(chunk["smiles"].tolist())
            ok = chunk[sims < guard.max_similarity]
            if not ok.empty:
                ok = ok[ok["smiles"].map(lambda s: novelty.is_unique(s, target))]
            if not ok.empty:
                novel = pd.concat([novel, ok]) if len(novel) else ok
        log.info("[round %d] pool=%d -> screened=%d -> submittable=%d (budget %d)",
                 round_no, len(ranked), min(taken, len(order)), len(novel),
                 args.boltz_budget)
        if novel.empty:
            await asyncio.sleep(args.sleep)
            continue

        chosen = select_batch(novel, args.boltz_budget, args.explore_frac, rng,
                              key=key)
        chosen["source"] = "hunter"
        if "strategy" in chosen.columns:
            share = chosen["strategy"].value_counts()
            log.info("[round %d] selected by strategy: %s", round_no,
                     " ".join(f"{k}={v}" for k, v in share.items()))

        if args.dry_run:
            log.info("[round %d] DRY RUN — would score %d molecules (key=%s):",
                     round_no, len(chosen), key)
            for _, r in chosen.head(15).iterrows():
                log.info("    mu=%.6f ei=%.6f ha=%2d  %s",
                         r["mu"], r["ei"], heavy_atoms(r["smiles"]), r["name"])
            return

        payload = chosen[["name", "smiles", "source"]].to_dict("records")
        # Mark anything that clears the current submittable frontier as it is
        # scored, so a hit is visible in the log the moment it appears.
        frontier = (float(before["score"].iloc[-1]) if len(before) >= 20
                    else args.confirm_floor)
        # Persist as each batch lands rather than at round end. A round is ~43
        # minutes of GPU at budget 150; writing only at the end means a crash or
        # a pm2 restart at molecule 149 discards all 149 scores. The DB also
        # then reflects progress live instead of jumping once per round.
        written = 0

        def persist(records):
            nonlocal written
            n = score_store.write_scores_to_db(
                db_path, records, rxn_id=rxn_id, round_no=round_no,
                target_key=target_key, target_label=target_label,
                source="hunter")
            written += len(records)
            log.info("    -> wrote %d molecules to DB (%d this round)",
                     len(records), written)

        new = await boltz_score(boltz, config, payload, args.batch_size,
                                mark_at=frontier, on_batch=persist)

        if new and not args.no_confirm and args.confirm_extra_rounds > 0:
            # `before` is this round's pre-scoring submittable top-20, so the
            # threshold reflects the bar the round actually had to beat.
            conf_thr, why = resolve_confirm_threshold(args, before)
            conf_thr, n_conf = cap_confirm_threshold(
                conf_thr, [float(r["boltz_score"]) for r in new],
                args.confirm_max_frac)
            log.info("[round %d] confirm threshold %.5f (%s) -> %d/%d molecules "
                     "= %d extra predictions",
                     round_no, conf_thr, why, n_conf, len(new),
                     n_conf * args.confirm_extra_rounds)
            if n_conf:
                pre = {r["name"]: float(r["boltz_score"]) for r in new}
                averaged = await rescore.confirm_high_scorers(
                    boltz=boltz, config=config, scored=new,
                    threshold=conf_thr,
                    extra_rounds=args.confirm_extra_rounds,
                    batch_size=args.batch_size, db_path=db_path, rxn_id=rxn_id,
                    round_no=round_no, target_key=target_key,
                    target_label=target_label, source="hunter", logger=log)
                if averaged:
                    draws = score_store.replicate_scores(db_path, list(averaged))
                    log.info("[round %d] confirmed %d molecules "
                             "(score is now the mean of %d draws at seed 68):",
                             round_no, len(averaged),
                             args.confirm_extra_rounds + 1)
                    log.info("    %-30s%10s%11s%10s   draws",
                             "molecule", "single", "confirmed", "delta")
                    for nm in sorted(averaged, key=lambda k: -averaged[k]):
                        d = draws.get(nm, [])
                        was = pre.get(nm, float("nan"))
                        log.info("    %-30s%10.6f%11.6f%+10.6f   %s",
                                 nm, was, averaged[nm], averaged[nm] - was,
                                 " ".join(f"{x:.6f}" for x in d))
                    moves = np.array([averaged[n] - pre.get(n, averaged[n])
                                      for n in averaged], dtype=float)
                    log.info("    net %+.6f | %d of %d came down | "
                             "still above frontier %.5f: %d",
                             float(moves.sum()), int((moves < 0).sum()),
                             len(moves), frontier,
                             sum(1 for v in averaged.values() if v > frontier))
                    # Reflect the confirmed values in this round's own summary,
                    # otherwise `best=` below reports a single draw that has
                    # already been superseded.
                    for r in new:
                        if r["name"] in averaged:
                            r["boltz_score"] = averaged[r["name"]]

        if new:
            # Batch writes already stored the single-draw values; re-write so any
            # score replaced by a confirmation mean lands in the DB too.
            score_store.write_scores_to_db(
                db_path, new, rxn_id=rxn_id, round_no=round_no,
                target_key=target_key, target_label=target_label,
                source="hunter")

        after = submittable_top20(db_path, rxn_id, guard, target, config)

        # Confirm the only set that is ever submitted. Round-leader confirmation
        # re-measured hundreds of molecules that were never going to be
        # submitted; this pays for at most `num_molecules` predictions and only
        # for molecules that have actually reached the submittable frontier.
        # confirm_high_scorers skips anything already flagged `rescored`, so the
        # steady-state cost is one draw per NEW entrant to the top 20.
        if (not args.dry_run and args.top20_confirm_draws > 0
                and boltz is not None and len(after)):
            records = after.assign(source="hunter").to_dict("records")
            averaged = await rescore.confirm_high_scorers(
                boltz=boltz, config=config, scored=records,
                threshold=-math.inf,
                extra_rounds=args.top20_confirm_draws,
                batch_size=args.batch_size, db_path=db_path, rxn_id=rxn_id,
                round_no=round_no, target_key=target_key,
                target_label=target_label, source="hunter", logger=log)
            if averaged:
                moved = np.array(
                    [averaged[n] - float(after.loc[after["name"] == n, "score"].iloc[0])
                     for n in averaged if (after["name"] == n).any()], dtype=float)
                log.info("[round %d] top-20 gate: confirmed %d new entrant(s) | "
                         "net %+.6f | %d came down",
                         round_no, len(averaged),
                         float(moved.sum()) if len(moved) else 0.0,
                         int((moved < 0).sum()) if len(moved) else 0)
                # Confirmed values can reorder the frontier, so re-read it.
                after = submittable_top20(db_path, rxn_id, guard, target, config)

        export_top20(after, rxn_id, target)
        sum_after = float(after["score"].sum()) if len(after) else 0.0

        vals = [float(r["boltz_score"]) for r in new] if new else []
        # Report the hit rate against the bar that actually matters — the
        # submittable frontier — not a fixed number that may sit far from it.
        bar = (float(before["score"].iloc[-1]) if len(before) >= 20
               else args.confirm_floor)
        hits = sum(1 for v in vals if v > bar)
        # Hand freed arena memory back before reporting, so the number logged is
        # the steady-state footprint rather than this round's high-water mark.
        _malloc_trim()
        log.info(
            "[round %d done] scored=%d | >%.5f (submittable #20): %d "
            "(%.2f%% hit rate) | best=%.6f | submittable top20 %.6f -> %.6f "
            "(%+.6f) | %.0fs | rss=%dMiB cache mol/fp=%d/%d",
            round_no, len(vals), bar, hits,
            100.0 * hits / max(len(vals), 1), max(vals) if vals else 0.0,
            sum_before, sum_after, sum_after - sum_before, time.time() - t_round,
            _rss_mib(), len(_mol), len(_fp),
        )
        await asyncio.sleep(args.sleep)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("stopped by user")
