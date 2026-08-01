"""
DPEX_DJA – Dual-Population EXchange with Discrete Jaya Algorithm
=================================================================
Population A  – global exploration via DJA update rule (discrete Jaya)
Population B  – local refinement via tabu-enhanced neighbourhood search
Exchange      – periodically injects best-of-A into B, evicts worst-of-B

Reference pseudocode: DPEX_DJA_algorithm.md

Algorithm structure
-------------------
FOR each iteration t:
    Part A  : apply DJA update to every member of pop_A
                  ai' = ai + r1*(best_A - |ai|) - r2*(worst_A - |ai|)
              (discrete: probabilistic component-slot attraction/repulsion)
    Part B  : tabu-enhanced local search on pop_B elites
              generate k neighbours per elite via synthon similarity,
              block tabu moves unless aspiration holds
    Part C  : every T_ex iters, exchange m best-of-A into B
              (pop_B is trimmed to N_B after merge)
    Global  : accumulate all scored candidates into top_pool
"""
from __future__ import annotations

import random
import bittensor as bt
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from molecules import MoleculeManager
from tools import SynthonLibrary

# Global ranker weights (set by miner before each DJA call) — zero-overhead injection.
# Per-rxn keyed: {rxn_id: np.ndarray}. Legacy single-rxn callers use rxn_id=None.
_rw_A: dict = {}
_rw_B: dict = {}
_rw_C: dict = {}

def set_ranker_weights(w_A, w_B, w_C=None, rxn_id=None):
    """Store ranker weight arrays. rxn_id=None preserves legacy single-rxn semantics."""
    global _rw_A, _rw_B, _rw_C
    _rw_A[rxn_id] = w_A
    _rw_B[rxn_id] = w_B
    _rw_C[rxn_id] = w_C

def _smart_choice(pool, role='A', rxn_id=None):
    """Ranker-weighted random — same speed as random.choice when no weights.
    Falls back from per-rxn → legacy None-keyed weights → uniform."""
    table = {'A': _rw_A, 'B': _rw_B, 'C': _rw_C}.get(role, {})
    w = table.get(rxn_id) if rxn_id is not None else None
    if w is None:
        w = table.get(None)
    if w is not None and len(w) == len(pool):
        return pool[np.random.choice(len(pool), p=w)]
    return random.choice(pool)

# ── tunables ──────────────────────────────────────────────────────────────────
N_A_DEFAULT  = 500   # population A capacity  (moving-window of scored mols)
N_B_DEFAULT  = 100   # population B capacity  (elite pool for tabu search)
T_EX_DEFAULT = 5     # exchange every T_ex iterations
M_EX_DEFAULT = 10    # molecules exchanged per cycle
TABU_MAXLEN  = 40    # maximum tabu entries per component role
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DPEXDJAState:
    """Persistent DPEX_DJA state carried across iterations.

    v3 NOTE: in multi-rxn mode (call ``init_multi_rxn(rxn_ids)``), the search
    runs as N independent arms — each rxn has its own pop_A slot, pop_B slot,
    and tabu memory. The flat ``pop_A`` / ``pop_B`` lists below are kept as
    aggregated views for legacy callers (e.g. miner.py logging that does
    ``len(dpex.pop_A)``); they are rebuilt as the union of all per-rxn arms
    after every population update.

    Single-rxn mode (rxn_ids empty) preserves bit-identical behavior to v2.
    """
    pop_A:    List[Dict]        = field(default_factory=list)
    pop_B:    List[Dict]        = field(default_factory=list)
    tabu:     Dict[str, deque]  = field(default_factory=lambda: {
        'A': deque(maxlen=TABU_MAXLEN),
        'B': deque(maxlen=TABU_MAXLEN),
        'C': deque(maxlen=TABU_MAXLEN),
    })
    # ── v3 per-rxn structures ─────────────────────────────────────────────
    rxn_ids:   List[int]                              = field(default_factory=list)
    pop_A_rxn: Dict[int, List[Dict]]                  = field(default_factory=dict)
    pop_B_rxn: Dict[int, List[Dict]]                  = field(default_factory=dict)
    tabu_rxn:  Dict[int, Dict[str, deque]]            = field(default_factory=dict)
    # ──────────────────────────────────────────────────────────────────────
    N_A:      int = N_A_DEFAULT
    N_B:      int = N_B_DEFAULT
    T_ex:     int = T_EX_DEFAULT
    m_ex:     int = M_EX_DEFAULT
    iteration: int = 0
    global_seen: Set[str] = field(default_factory=set)  # 5️⃣ Duplicate prevention
    global_best_molecule: Optional[Dict] = None  # 🔟 Global best tracking
    global_best_score: float = float('-inf')  # 🔟 Global best tracking

    def init_multi_rxn(self, rxn_ids: List[int]) -> None:
        """v3 — initialize per-rxn arms. Idempotent. Call from miner.py once
        when MoleculeManager has been built and is_multi=True."""
        self.rxn_ids = list(rxn_ids)
        for r in self.rxn_ids:
            if r not in self.pop_A_rxn:
                self.pop_A_rxn[r] = []
            if r not in self.pop_B_rxn:
                self.pop_B_rxn[r] = []
            if r not in self.tabu_rxn:
                self.tabu_rxn[r] = {
                    'A': deque(maxlen=TABU_MAXLEN),
                    'B': deque(maxlen=TABU_MAXLEN),
                    'C': deque(maxlen=TABU_MAXLEN),
                }

    @property
    def is_multi_rxn(self) -> bool:
        return len(self.rxn_ids) > 0

    @property
    def N_A_per_rxn(self) -> int:
        """Per-arm pop_A capacity in multi-rxn mode (so total ~ N_A)."""
        if not self.is_multi_rxn:
            return self.N_A
        return max(40, self.N_A // max(1, len(self.rxn_ids)))

    @property
    def N_B_per_rxn(self) -> int:
        if not self.is_multi_rxn:
            return self.N_B
        return max(10, self.N_B // max(1, len(self.rxn_ids)))

    def _rebuild_flat_views(self) -> None:
        """Reconstruct the flat pop_A and pop_B lists from per-rxn arms.
        Sorted by score desc, capped at N_A / N_B respectively. Keeps legacy
        callers (logging, len(state.pop_A) etc.) working unchanged."""
        if not self.is_multi_rxn:
            return
        all_A: List[Dict] = []
        for r in self.rxn_ids:
            all_A.extend(self.pop_A_rxn.get(r, []))
        self.pop_A = sorted(all_A, key=lambda x: x.get('score', float('-inf')), reverse=True)[: self.N_A]
        all_B: List[Dict] = []
        for r in self.rxn_ids:
            all_B.extend(self.pop_B_rxn.get(r, []))
        self.pop_B = sorted(all_B, key=lambda x: x.get('score', float('-inf')), reverse=True)[: self.N_B]

    def augment_pop_B(self, records) -> None:
        """Merge a list of mol records into pop_B (per-rxn arm in multi-rxn mode,
        flat in single-rxn mode). Use this from miner.py instead of writing
        directly to ``state.pop_B`` so that per-rxn arms stay in sync.

        Records may be a list of dicts with keys 'name', 'smiles', 'score', etc.
        """
        if not records:
            return
        if not self.is_multi_rxn:
            by_name = {m['name']: m for m in self.pop_B}
            for mol in records:
                if 'name' in mol:
                    by_name[mol['name']] = mol
            self.pop_B = sorted(by_name.values(), key=lambda x: x.get('score', float('-inf')), reverse=True)[: self.N_B]
            return
        # v3 multi-rxn — partition by rxn parsed from name
        for mol in records:
            r = _rxn_of(mol.get('name', ''))
            if r is None or r not in self.pop_B_rxn:
                continue
            arm = {m['name']: m for m in self.pop_B_rxn[r]}
            arm[mol['name']] = mol
            self.pop_B_rxn[r] = sorted(
                arm.values(), key=lambda x: x.get('score', float('-inf')), reverse=True
            )[: self.N_B_per_rxn]
        self._rebuild_flat_views()

def _parse(name: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None, None
    try:
        return (
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]) if len(parts) > 4 else None,
        )
    except (ValueError, IndexError):
        return None, None, None, None


def _build(rxn: int, A: int, B: int, C: Optional[int]) -> str:
    return f"rxn:{rxn}:{A}:{B}" if C is None else f"rxn:{rxn}:{A}:{B}:{C}"

def _dja_move(
    name:      str,
    best_name: str,
    worst_name: str,
    manager,                              # MoleculeManager OR SubManager
    avoid:     Set[str],
    exploration_rate: float = 0.25,  # 9️⃣ Iteration-dependent exploration
) -> Optional[str]:

    rxn, A,  B,  C  = _parse(name)
    rxn_b, bA, bB, bC = _parse(best_name)
    rxn_w, wA, wB, wC = _parse(worst_name)

    if rxn is None or bA is None or wA is None:
        return None

    # Route to the input mol's rxn sub. Single-rxn (manager has no for_rxn) → use as-is.
    sub = manager.for_rxn(rxn) if hasattr(manager, "for_rxn") else manager
    rxn_key = rxn  # used to key per-rxn ranker weights

    # If best/worst are from a DIFFERENT rxn (population-wide best across rxns),
    # they're not directionally meaningful — fall back to the current mol's components,
    # so the DJA step degenerates into mutation-only for this mol. This keeps moves
    # within-rxn even when populations are mixed.
    if rxn_b != rxn:
        bA, bB, bC = A, B, C
    if rxn_w != rxn:
        wA, wB, wC = A, B, C

    # 2️⃣ Better DJA discrete move rule
    def _step(cur: int, best: int, worst: int, pool: List[int], role: str = 'A') -> int:
        r = random.random()

        explore_prob = exploration_rate * 0.25
        exploit_prob = 0.6 + (0.25 - explore_prob)

        if r < exploit_prob:
            return best
        elif r < exploit_prob + explore_prob:
            return _smart_choice(pool, role, rxn_id=rxn_key)  # ranker-weighted exploration
        else:
            return cur

    nA = _step(A, bA, wA, sub.moles_A_id, 'A')
    nB = _step(B, bB, wB, sub.moles_B_id, 'B')

    nC: Optional[int] = None
    if sub.is_three_component and C is not None:
        nC = _step(
            C,
            bC if bC is not None else C,
            wC if wC is not None else C,
            sub.moles_C_id,
            'C',
        )

    new_name = _build(rxn, nA, nB, nC)

    # 3️⃣ Add mutation operator (15% probability)
    if random.random() < 0.15:
        components = ['A', 'B']
        if sub.is_three_component and nC is not None:
            components.append('C')

        mutate_component = random.choice(components)
        if mutate_component == 'A':
            nA = _smart_choice(sub.moles_A_id, 'A', rxn_id=rxn_key)
        elif mutate_component == 'B':
            nB = _smart_choice(sub.moles_B_id, 'B', rxn_id=rxn_key)
        elif mutate_component == 'C' and nC is not None:
            nC = _smart_choice(sub.moles_C_id, 'C', rxn_id=rxn_key)

        new_name = _build(rxn, nA, nB, nC)

    return None if (new_name == name or new_name in avoid) else new_name


def dja_generate(
    state:    DPEXDJAState,
    manager:  MoleculeManager,
    n_samples: int,
    avoid:    Set[str],
    rxn_weights: Optional[Dict[int, float]] = None,   # NEW v3 — per-rxn arm budget
) -> pd.DataFrame:
    """Discrete-Jaya generation step.

    v3: in multi-rxn mode, runs per-rxn arms — each with its own pop_A and its
    own best/worst (no cross-rxn degeneracy). Per-arm sample budget is split by
    ``rxn_weights`` if provided, else even.
    Single-rxn mode preserves bit-identical behavior.
    """
    if not state.pop_A:
        return pd.DataFrame(columns=["name"])

    # 9️⃣ Iteration-dependent exploration
    exploration_rate = max(0.15, 1.5 - state.iteration / 80)

    new_names: Set[str] = set()

    if not state.is_multi_rxn:
        # ===== legacy single-rxn path — bit-identical to v2 =====
        by_score  = sorted(state.pop_A, key=lambda x: x.get('score', float('-inf')), reverse=True)
        best_mol  = by_score[0]
        worst_mol = by_score[-1]

        for mol in state.pop_A:
            if len(new_names) >= n_samples:
                break
            n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], manager, avoid, exploration_rate)
            if n and n not in state.global_seen:
                new_names.add(n)

        attempts = 0
        while len(new_names) < n_samples and attempts < n_samples * 4:
            attempts += 1
            mol = random.choice(state.pop_A)
            n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], manager, avoid, exploration_rate)
            if n and n not in new_names and n not in state.global_seen:
                new_names.add(n)

        return pd.DataFrame({"name": list(new_names)}) if new_names else pd.DataFrame(columns=["name"])

    # ===== v3 multi-rxn path =====
    # Budget per arm — uniform if no weights provided
    rxn_ids_active = [r for r in state.rxn_ids if state.pop_A_rxn.get(r)]
    if not rxn_ids_active:
        return pd.DataFrame(columns=["name"])

    if rxn_weights:
        per_arm_budget = {r: max(1, int(round(n_samples * rxn_weights.get(r, 1.0 / len(rxn_ids_active)))))
                          for r in rxn_ids_active}
    else:
        share = max(1, n_samples // len(rxn_ids_active))
        per_arm_budget = {r: share for r in rxn_ids_active}

    for r in rxn_ids_active:
        arm_pop = state.pop_A_rxn.get(r, [])
        if not arm_pop:
            continue
        arm_target = per_arm_budget[r]
        # Best/worst WITHIN this arm — meaningful Jaya step
        by_score = sorted(arm_pop, key=lambda x: x.get('score', float('-inf')), reverse=True)
        best_mol = by_score[0]
        worst_mol = by_score[-1]
        # Route to this rxn's SubManager so synthon pools come from the right rxn
        sub = manager.for_rxn(r) if hasattr(manager, "for_rxn") else manager

        arm_added: int = 0
        for mol in arm_pop:
            if arm_added >= arm_target:
                break
            n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], sub, avoid, exploration_rate)
            if n and n not in state.global_seen and n not in new_names:
                new_names.add(n)
                arm_added += 1

        # Top-up via random picks within this arm
        attempts = 0
        max_attempts = arm_target * 4
        while arm_added < arm_target and attempts < max_attempts:
            attempts += 1
            mol = random.choice(arm_pop)
            n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], sub, avoid, exploration_rate)
            if n and n not in state.global_seen and n not in new_names:
                new_names.add(n)
                arm_added += 1

    return pd.DataFrame({"name": list(new_names)}) if new_names else pd.DataFrame(columns=["name"])

def _tabu_hit(tabu_set: Set[Tuple[int, int]], old_id: int, new_id: int) -> bool:
    return (old_id, new_id) in tabu_set

def tabu_generate(
    state:             DPEXDJAState,
    synthon_lib:       SynthonLibrary,
    manager:           MoleculeManager,
    avoid:             Set[str],
    k_per_elite:       int   = 15,
    k_elites:          int   = 10,
    global_best_score: float = float('-inf'),
    tabued_molecules:  Set[str] = set(),
    rxn_weights: Optional[Dict[int, float]] = None,   # NEW v3 — per-rxn elite budget
) -> Tuple[pd.DataFrame, List[Tuple[str, int, int]]]:
    """Tabu-enhanced neighbourhood search.

    v3 multi-rxn: each rxn arm has its own pop_B + own tabu memory + own elite
    budget (proportional to ``rxn_weights``). Tabu lookup keyed by (rxn, role)
    so a tabu entry in rxn=3 doesn't falsely block the same numeric pair in rxn=1.
    Single-rxn mode preserves bit-identical behavior.
    """

    if not state.pop_B or synthon_lib is None:
        return pd.DataFrame(columns=["name"]), []

    # 6️⃣ Adaptive tabu list — applied to BOTH legacy and per-rxn deques
    adaptive_tabu_len = min(200, 20 + state.iteration * 2)
    for role in ('A', 'B', 'C'):
        if role in state.tabu:
            state.tabu[role] = deque(state.tabu[role], maxlen=adaptive_tabu_len)
    if state.is_multi_rxn:
        for r in state.rxn_ids:
            if r not in state.tabu_rxn:
                state.tabu_rxn[r] = {}
            for role in ('A', 'B', 'C'):
                if role not in state.tabu_rxn[r]:
                    state.tabu_rxn[r][role] = deque(maxlen=adaptive_tabu_len)
                else:
                    state.tabu_rxn[r][role] = deque(state.tabu_rxn[r][role], maxlen=adaptive_tabu_len)

    new_names:    List[str]                            = []
    applied_moves: List[Tuple[str, int, int, int]]     = []  # NEW: (rxn, role, old, new) — rxn always present (or -1 for single-rxn legacy)

    is_registry = hasattr(synthon_lib, "lib_for_rxn")

    # ===== Build the per-arm elite pools and per-arm budgets =====
    if state.is_multi_rxn:
        # Per-arm elites + per-arm tabu sets
        rxn_arms = [r for r in state.rxn_ids if state.pop_B_rxn.get(r)]
        if not rxn_arms:
            return pd.DataFrame(columns=["name"]), []
        if rxn_weights:
            arm_budget_elites = {r: max(1, int(round(k_elites * rxn_weights.get(r, 1.0 / len(rxn_arms)))))
                                 for r in rxn_arms}
        else:
            share = max(1, k_elites // len(rxn_arms))
            arm_budget_elites = {r: share for r in rxn_arms}

        for r in rxn_arms:
            arm_pop = state.pop_B_rxn.get(r, [])
            if not arm_pop:
                continue
            arm_n_elites = min(arm_budget_elites[r], len(arm_pop))
            scores = [max(0.0, m.get('score', 0)) for m in arm_pop]
            total = sum(scores)
            if total > 0:
                w = [s / total for s in scores]
                elite_indices = random.choices(range(len(arm_pop)), weights=w, k=arm_n_elites)
                elites = [arm_pop[i] for i in elite_indices]
            else:
                elites = random.choices(arm_pop, k=arm_n_elites) if arm_pop else []

            tabu_sets_arm: Dict[str, Set] = {role: set(state.tabu_rxn[r].get(role, deque())) for role in ('A', 'B', 'C')}
            sub = manager.for_rxn(r) if hasattr(manager, "for_rxn") else manager
            elite_is_three = sub.is_three_component
            elite_lib = synthon_lib.lib_for_rxn(r) if is_registry else synthon_lib
            if elite_lib is None:
                continue

            for mol in elites:
                if mol["name"] in tabued_molecules:
                    continue
                rxn_p, A, B, C = _parse(mol['name'])
                if rxn_p is None or rxn_p != r:
                    continue
                mol_score = mol.get('score', float('-inf'))
                aspiration = (state.global_best_score > float('-inf')
                              and mol_score >= state.global_best_score * 0.9)

                similar = elite_lib.find_similar_to_molecule_name(
                    mol['name'],
                    vary_component='both' if not elite_is_three else 'all',
                    top_k_per_component=k_per_elite,
                    min_similarity=0.50,
                )

                for new_A in similar.get('A', [])[:k_per_elite]:
                    nn = _build(r, new_A, B, C)
                    if _tabu_hit(tabu_sets_arm['A'], A, new_A) and not aspiration:
                        continue
                    if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                        new_names.append(nn)
                        applied_moves.append((r, 'A', A, new_A))

                for new_B in similar.get('B', [])[:k_per_elite]:
                    nn = _build(r, A, new_B, C)
                    if _tabu_hit(tabu_sets_arm['B'], B, new_B) and not aspiration:
                        continue
                    if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                        new_names.append(nn)
                        applied_moves.append((r, 'B', B, new_B))

                if elite_is_three and C is not None:
                    for new_C in similar.get('C', [])[:k_per_elite]:
                        nn = _build(r, A, B, new_C)
                        if _tabu_hit(tabu_sets_arm['C'], C, new_C) and not aspiration:
                            continue
                        if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                            new_names.append(nn)
                            applied_moves.append((r, 'C', C, new_C))
        # Fall through to common return below
    else:
        # ===== legacy single-rxn path — bit-identical to v2 =====
        tabu_sets: Dict[str, Set] = {role: set(state.tabu[role]) for role in ('A', 'B', 'C')}

        n_elites = min(k_elites, len(state.pop_B))
        scores = [max(0, mol.get('score', 0)) for mol in state.pop_B]
        total_score = sum(scores)
        if total_score > 0:
            weights = [s / total_score for s in scores]
            elite_indices = random.choices(range(len(state.pop_B)), weights=weights, k=n_elites)
            elites = [state.pop_B[i] for i in elite_indices]
        else:
            elites = random.choices(state.pop_B, k=n_elites) if state.pop_B else []

        for mol in elites:
            if mol["name"] in tabued_molecules:
                continue
            rxn, A, B, C = _parse(mol['name'])
            mol_score    = mol.get('score', float('-inf'))
            if rxn is None:
                continue
            aspiration = (state.global_best_score > float('-inf')
                          and mol_score >= state.global_best_score * 0.9)
            if is_registry:
                elite_lib = synthon_lib.lib_for_rxn(rxn)
                if elite_lib is None:
                    continue
                sub = manager.for_rxn(rxn) if hasattr(manager, "for_rxn") else manager
                elite_is_three = sub.is_three_component
            else:
                elite_lib = synthon_lib
                elite_is_three = manager.is_three_component

            similar = elite_lib.find_similar_to_molecule_name(
                mol['name'],
                vary_component='both' if not elite_is_three else 'all',
                top_k_per_component=k_per_elite,
                min_similarity=0.50,
            )

            for new_A in similar.get('A', [])[:k_per_elite]:
                nn = _build(rxn, new_A, B, C)
                if _tabu_hit(tabu_sets['A'], A, new_A) and not aspiration:
                    continue
                if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                    new_names.append(nn)
                    applied_moves.append((-1, 'A', A, new_A))   # rxn=-1 sentinel for legacy

            for new_B in similar.get('B', [])[:k_per_elite]:
                nn = _build(rxn, A, new_B, C)
                if _tabu_hit(tabu_sets['B'], B, new_B) and not aspiration:
                    continue
                if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                    new_names.append(nn)
                    applied_moves.append((-1, 'B', B, new_B))

            if elite_is_three and C is not None:
                for new_C in similar.get('C', [])[:k_per_elite]:
                    nn = _build(rxn, A, B, new_C)
                    if _tabu_hit(tabu_sets['C'], C, new_C) and not aspiration:
                        continue
                    if nn not in avoid and nn not in new_names and nn not in state.global_seen:
                        new_names.append(nn)
                        applied_moves.append((-1, 'C', C, new_C))

    return (
        pd.DataFrame({"name": new_names}) if new_names else pd.DataFrame(columns=["name"]),
        applied_moves,
    )


def update_tabu(state: DPEXDJAState, moves) -> None:
    """Append moves to the appropriate tabu deque(s).

    Accepts both shapes:
      - v3 4-tuple: (rxn_id, role, old, new) — routes to state.tabu_rxn[rxn_id][role]
        (rxn_id == -1 sentinel routes to legacy state.tabu)
      - legacy 3-tuple: (role, old, new) — routes to state.tabu[role]
    """
    for entry in moves:
        if len(entry) == 4:
            rxn_id, role, old_id, new_id = entry
            if rxn_id == -1 or not state.is_multi_rxn:
                if role in state.tabu:
                    state.tabu[role].append((old_id, new_id))
            else:
                if rxn_id in state.tabu_rxn and role in state.tabu_rxn[rxn_id]:
                    state.tabu_rxn[rxn_id][role].append((old_id, new_id))
        else:
            role, old_id, new_id = entry
            if role in state.tabu:
                state.tabu[role].append((old_id, new_id))

def _rxn_of(name: str) -> Optional[int]:
    try:
        return int(name.split(':')[1])
    except (IndexError, ValueError):
        return None


def dpex_exchange(state: DPEXDJAState) -> None:
    """Exchange best-of-A→B and best-of-B→A. v3: per-rxn arms in multi-rxn mode
    so a winning rxn doesn't crowd losing rxns out of pop_B."""
    if not state.pop_A or state.m_ex <= 0:
        return

    if not state.is_multi_rxn:
        # ===== legacy single-rxn path =====
        best_of_A = sorted(state.pop_A, key=lambda x: x.get('score', float('-inf')), reverse=True)[:state.m_ex]
        best_of_B = sorted(state.pop_B, key=lambda x: x.get('score', float('-inf')), reverse=True)[:state.m_ex]
        seen_B: Set[str] = set()
        merged_B: List[Dict] = []
        for mol in list(state.pop_B) + best_of_A:
            if mol['name'] not in seen_B:
                seen_B.add(mol['name']); merged_B.append(mol)
        merged_B.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)
        state.pop_B = merged_B[:state.N_B]
        combined_A = {mol['name']: mol for mol in state.pop_A}
        for mol in best_of_B:
            if mol['name'] not in combined_A or mol.get('score', float('-inf')) > combined_A[mol['name']].get('score', float('-inf')):
                combined_A[mol['name']] = mol
        state.pop_A = sorted(combined_A.values(), key=lambda x: x.get('score', float('-inf')), reverse=True)[:state.N_A]
        bt.logging.info(
            f"[DPEX] Bidirectional Exchange: {state.m_ex} best A→B, {state.m_ex} best B→A  |  pop_A={len(state.pop_A)}, pop_B={len(state.pop_B)}"
        )
        return

    # ===== v3 multi-rxn path: per-arm exchange =====
    per_arm_m = max(1, state.m_ex // max(1, len(state.rxn_ids)))
    for r in state.rxn_ids:
        arm_A = state.pop_A_rxn.get(r, [])
        arm_B = state.pop_B_rxn.get(r, [])
        if not arm_A and not arm_B:
            continue
        best_of_A = sorted(arm_A, key=lambda x: x.get('score', float('-inf')), reverse=True)[:per_arm_m]
        best_of_B = sorted(arm_B, key=lambda x: x.get('score', float('-inf')), reverse=True)[:per_arm_m]

        # best_of_A → arm_B
        by_name_B = {mol['name']: mol for mol in arm_B}
        for mol in best_of_A:
            by_name_B[mol['name']] = mol
        state.pop_B_rxn[r] = sorted(
            by_name_B.values(), key=lambda x: x.get('score', float('-inf')), reverse=True
        )[: state.N_B_per_rxn]

        # best_of_B → arm_A
        by_name_A = {mol['name']: mol for mol in arm_A}
        for mol in best_of_B:
            if mol['name'] not in by_name_A or mol.get('score', float('-inf')) > by_name_A[mol['name']].get('score', float('-inf')):
                by_name_A[mol['name']] = mol
        state.pop_A_rxn[r] = sorted(
            by_name_A.values(), key=lambda x: x.get('score', float('-inf')), reverse=True
        )[: state.N_A_per_rxn]

    state._rebuild_flat_views()
    bt.logging.info(
        f"[DPEX v3] Per-rxn exchange: {per_arm_m} per-arm  |  pop_A={len(state.pop_A)} pop_B={len(state.pop_B)}"
    )


def update_populations(
    state:    DPEXDJAState,
    scored_A: pd.DataFrame,
    scored_B: pd.DataFrame,
) -> None:
    """Merge new scored mols into pop_A / pop_B. v3: in multi-rxn mode, partition
    new mols by rxn parsed from name, merge into per-rxn arms with per-arm caps,
    then rebuild the flat aggregated views."""

    _required = ('name', 'smiles', 'score')

    def _to_records(df: pd.DataFrame) -> List[Dict]:
        if df.empty or not all(c in df.columns for c in _required):
            return []
        cols = [c for c in ('name', 'smiles', 'score', 'target', 'anti') if c in df.columns]
        return df[cols].dropna(subset=['score']).to_dict('records')

    new_A_records = _to_records(scored_A)
    new_B_records = _to_records(scored_B)

    if not state.is_multi_rxn:
        # ===== legacy single-rxn path — bit-identical to v2 =====
        if new_A_records:
            combined_A = {mol['name']: mol for mol in state.pop_A}
            for mol in new_A_records:
                if mol['name'] not in combined_A or mol.get('score', float('-inf')) > combined_A[mol['name']].get('score', float('-inf')):
                    combined_A[mol['name']] = mol
            state.pop_A = sorted(combined_A.values(), key=lambda x: x.get('score', float('-inf')), reverse=True)[:state.N_A]

        if new_B_records:
            by_name = {mol['name']: mol for mol in state.pop_B}
            for mol in new_B_records:
                by_name[mol['name']] = mol
            state.pop_B = sorted(by_name.values(), key=lambda x: x.get('score', float('-inf')), reverse=True)[:state.N_B]
    else:
        # ===== v3 multi-rxn path: per-rxn arm updates =====
        if new_A_records:
            # Partition new records by rxn
            new_per_rxn: Dict[int, List[Dict]] = {}
            for mol in new_A_records:
                r = _rxn_of(mol['name'])
                if r is None or r not in state.pop_A_rxn:
                    continue
                new_per_rxn.setdefault(r, []).append(mol)
            for r, new_arm in new_per_rxn.items():
                combined = {mol['name']: mol for mol in state.pop_A_rxn[r]}
                for mol in new_arm:
                    if mol['name'] not in combined or mol.get('score', float('-inf')) > combined[mol['name']].get('score', float('-inf')):
                        combined[mol['name']] = mol
                state.pop_A_rxn[r] = sorted(
                    combined.values(), key=lambda x: x.get('score', float('-inf')), reverse=True
                )[: state.N_A_per_rxn]

        if new_B_records:
            new_per_rxn_B: Dict[int, List[Dict]] = {}
            for mol in new_B_records:
                r = _rxn_of(mol['name'])
                if r is None or r not in state.pop_B_rxn:
                    continue
                new_per_rxn_B.setdefault(r, []).append(mol)
            for r, new_arm in new_per_rxn_B.items():
                by_name = {mol['name']: mol for mol in state.pop_B_rxn[r]}
                for mol in new_arm:
                    by_name[mol['name']] = mol
                state.pop_B_rxn[r] = sorted(
                    by_name.values(), key=lambda x: x.get('score', float('-inf')), reverse=True
                )[: state.N_B_per_rxn]

        state._rebuild_flat_views()

    # 5️⃣ Update global seen set (rxn-agnostic)
    for mol in new_A_records + new_B_records:
        state.global_seen.add(mol['name'])

    # 🔟 Update global best
    all_mols = state.pop_A + state.pop_B
    if all_mols:
        best_mol = max(all_mols, key=lambda x: x.get('score', float('-inf')))
        best_score = best_mol.get('score', float('-inf'))
        if best_score > state.global_best_score:
            state.global_best_score = best_score
            state.global_best_molecule = best_mol
            bt.logging.info(f"[DPEX] New global best: {best_mol['name']} with score {best_score:.4f}")