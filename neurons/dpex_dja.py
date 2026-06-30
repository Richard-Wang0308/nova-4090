"""
dpex_dja.py — adapted from reference code
Removed: time-constraint checks, nova_ph2, PSICHIC references
Kept:    bt.logging, ALL core DPEX-DJA logic (bit-identical to reference)
Added:   exploit_intensity property, is_stagnating property
         for unlimited-run stagnation management
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

# ── ranker weight globals ─────────────────────────────────────────────────
_rw_A: dict = {}
_rw_B: dict = {}
_rw_C: dict = {}


def set_ranker_weights(w_A, w_B, w_C=None, rxn_id=None):
    """Store ranker weight arrays. rxn_id=None preserves legacy single-rxn."""
    global _rw_A, _rw_B, _rw_C
    _rw_A[rxn_id] = w_A
    _rw_B[rxn_id] = w_B
    _rw_C[rxn_id] = w_C


def _smart_choice(pool, role="A", rxn_id=None):
    """Ranker-weighted random. Falls back: per-rxn → None-keyed → uniform."""
    table = {"A": _rw_A, "B": _rw_B, "C": _rw_C}.get(role, {})
    w = table.get(rxn_id) if rxn_id is not None else None
    if w is None:
        w = table.get(None)
    if w is not None and len(w) == len(pool):
        return pool[np.random.choice(len(pool), p=w)]
    return random.choice(pool)


# ── tunables ──────────────────────────────────────────────────────────────
N_A_DEFAULT  = 500
N_B_DEFAULT  = 100
T_EX_DEFAULT = 5
M_EX_DEFAULT = 10
TABU_MAXLEN  = 40
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class DPEXDJAState:
    """
    Persistent DPEX-DJA state across ALL rounds (no time budget).

    Unlimited-run additions vs reference:
      - exploit_intensity: continuous stagnation signal (replaces binary switch)
      - is_stagnating: convenience property
      - global_seen: prevents re-scoring known molecules indefinitely
    """

    pop_A:    List[Dict]       = field(default_factory=list)
    pop_B:    List[Dict]       = field(default_factory=list)
    tabu:     Dict[str, deque] = field(default_factory=lambda: {
        "A": deque(maxlen=TABU_MAXLEN),
        "B": deque(maxlen=TABU_MAXLEN),
        "C": deque(maxlen=TABU_MAXLEN),
    })

    # per-rxn structures (v3)
    rxn_ids:   List[int]                   = field(default_factory=list)
    pop_A_rxn: Dict[int, List[Dict]]       = field(default_factory=dict)
    pop_B_rxn: Dict[int, List[Dict]]       = field(default_factory=dict)
    tabu_rxn:  Dict[int, Dict[str, deque]] = field(default_factory=dict)

    N_A:       int   = N_A_DEFAULT
    N_B:       int   = N_B_DEFAULT
    T_ex:      int   = T_EX_DEFAULT
    m_ex:      int   = M_EX_DEFAULT
    iteration: int   = 0

    # unlimited-run tracking
    global_seen:            Set[str]       = field(default_factory=set)
    global_best_molecule:   Optional[Dict] = None
    global_best_score:      float          = float("-inf")
    no_improvement_counter: int            = 0
    score_improvement_rate: float          = 0.0

    # ── multi-rxn ─────────────────────────────────────────────────────────

    def init_multi_rxn(self, rxn_ids: List[int]) -> None:
        """Initialize per-rxn arms. Idempotent."""
        self.rxn_ids = list(rxn_ids)
        for r in self.rxn_ids:
            if r not in self.pop_A_rxn:
                self.pop_A_rxn[r] = []
            if r not in self.pop_B_rxn:
                self.pop_B_rxn[r] = []
            if r not in self.tabu_rxn:
                self.tabu_rxn[r] = {
                    "A": deque(maxlen=TABU_MAXLEN),
                    "B": deque(maxlen=TABU_MAXLEN),
                    "C": deque(maxlen=TABU_MAXLEN),
                }

    @property
    def is_multi_rxn(self) -> bool:
        return len(self.rxn_ids) > 0

    @property
    def N_A_per_rxn(self) -> int:
        if not self.is_multi_rxn:
            return self.N_A
        return max(40, self.N_A // max(1, len(self.rxn_ids)))

    @property
    def N_B_per_rxn(self) -> int:
        if not self.is_multi_rxn:
            return self.N_B
        return max(10, self.N_B // max(1, len(self.rxn_ids)))

    # ── unlimited-run properties ──────────────────────────────────────────

    @property
    def is_stagnating(self) -> bool:
        """True when no improvement for 3+ consecutive rounds."""
        return self.no_improvement_counter >= 3

    @property
    def exploit_intensity(self) -> float:
        """
        Continuous exploit fraction driven by stagnation depth.
        0.10 (no stagnation) → 0.50 (10+ rounds stagnation).
        Replaces binary stagnation_counter >= 3 switch from
        enhanced_generation_dpex.py.
        """
        return min(0.50, 0.10 + self.no_improvement_counter * 0.04)

    # ── flat view rebuild ─────────────────────────────────────────────────

    def _rebuild_flat_views(self) -> None:
        if not self.is_multi_rxn:
            return
        all_A: List[Dict] = []
        for r in self.rxn_ids:
            all_A.extend(self.pop_A_rxn.get(r, []))
        self.pop_A = sorted(
            all_A,
            key=lambda x: x.get("score", float("-inf")),
            reverse=True,
        )[: self.N_A]
        all_B: List[Dict] = []
        for r in self.rxn_ids:
            all_B.extend(self.pop_B_rxn.get(r, []))
        self.pop_B = sorted(
            all_B,
            key=lambda x: x.get("score", float("-inf")),
            reverse=True,
        )[: self.N_B]

    def augment_pop_B(self, records) -> None:
        """Merge records into pop_B, routing to per-rxn arm in multi-rxn."""
        if not records:
            return
        if not self.is_multi_rxn:
            by_name = {m["name"]: m for m in self.pop_B}
            for mol in records:
                by_name[mol["name"]] = mol
            self.pop_B = sorted(
                by_name.values(),
                key=lambda x: x.get("score", float("-inf")),
                reverse=True,
            )[: self.N_B]
            return
        for mol in records:
            parts = str(mol.get("name", "")).split(":")
            if len(parts) < 2:
                continue
            try:
                r = int(parts[1])
            except ValueError:
                continue
            if r in self.pop_B_rxn:
                by_name = {m["name"]: m for m in self.pop_B_rxn[r]}
                by_name[mol["name"]] = mol
                self.pop_B_rxn[r] = sorted(
                    by_name.values(),
                    key=lambda x: x.get("score", float("-inf")),
                    reverse=True,
                )[: self.N_B_per_rxn]
        self._rebuild_flat_views()


# ═══════════════════════════════════════════════════════════════════════════
# Core algorithm functions (bit-identical to reference except bt.logging)
# ═══════════════════════════════════════════════════════════════════════════

def dja_generate(
    state: DPEXDJAState,
    molecule_manager: MoleculeManager,
    n_samples: int,
    mutation_prob: float = 0.40,
    rxn_id: int = None,
) -> List[str]:
    """Population A: DJA update rule (discrete Jaya)."""
    if state.is_multi_rxn and rxn_id is not None:
        pop = state.pop_A_rxn.get(rxn_id, [])
        sub = molecule_manager.for_rxn(rxn_id)
    else:
        pop = state.pop_A
        sub = molecule_manager

    if len(pop) < 2:
        from tools import generate_valid_random_molecules
        return generate_valid_random_molecules(molecule_manager, n_samples)

    sorted_pop  = sorted(pop, key=lambda x: x.get("score", float("-inf")), reverse=True)
    best        = sorted_pop[0]
    worst       = sorted_pop[-1]
    pool_A      = sub.moles_A_id
    pool_B      = sub.moles_B_id
    pool_C      = sub.moles_C_id if sub.is_three_component else []
    eff_rxn_id  = rxn_id if rxn_id is not None else sub.rxn_id
    new_names   = []

    for _ in range(n_samples):
        current = random.choice(pop)
        parts   = str(current.get("name", "")).split(":")
        if len(parts) < 4:
            continue
        try:
            A = int(parts[2]); B = int(parts[3])
            C = int(parts[4]) if len(parts) > 4 else None
        except (ValueError, IndexError):
            continue

        best_parts  = str(best.get("name", "")).split(":")
        worst_parts = str(worst.get("name", "")).split(":")

        # Attract to best A
        if random.random() < mutation_prob and len(best_parts) >= 4:
            try: A = int(best_parts[2])
            except ValueError: pass
        # Repel from worst A
        if random.random() < mutation_prob and len(worst_parts) >= 4:
            try:
                worst_A = int(worst_parts[2])
                diff_A  = [x for x in pool_A if x != worst_A]
                if diff_A: A = _smart_choice(diff_A, "A", eff_rxn_id)
            except ValueError: pass
        # Attract to best B
        if random.random() < mutation_prob and len(best_parts) >= 5:
            try: B = int(best_parts[3])
            except ValueError: pass
        # Repel from worst B
        if random.random() < mutation_prob and len(worst_parts) >= 5:
            try:
                worst_B = int(worst_parts[3])
                diff_B  = [x for x in pool_B if x != worst_B]
                if diff_B: B = _smart_choice(diff_B, "B", eff_rxn_id)
            except ValueError: pass

        name = (
            f"rxn:{eff_rxn_id}:{A}:{B}:{C}"
            if (sub.is_three_component and pool_C and C is not None)
            else f"rxn:{eff_rxn_id}:{A}:{B}"
        )
        if name not in state.global_seen:
            new_names.append(name)

    return new_names


def tabu_generate(
    state: DPEXDJAState,
    synthon_lib: SynthonLibrary,
    n_samples: int,
    neighborhood_size: int = 10,
    rxn_id: int = None,
) -> List[str]:
    """Population B: tabu-enhanced neighbourhood search."""
    if state.is_multi_rxn and rxn_id is not None:
        pop_b = state.pop_B_rxn.get(rxn_id, [])
        tabu  = state.tabu_rxn.get(rxn_id, state.tabu)
    else:
        pop_b = state.pop_B
        tabu  = state.tabu

    if not pop_b:
        return []

    eff_rxn_id = rxn_id if rxn_id is not None else synthon_lib.rxn_id
    new_names  = []

    for _ in range(n_samples):
        elite        = random.choice(pop_b)
        elite_name   = elite.get("name", "")
        elite_smiles = elite.get("smiles", "")
        if not elite_smiles:
            continue

        parts = elite_name.split(":")
        if len(parts) < 4:
            continue
        try:
            A = int(parts[2]); B = int(parts[3])
            C = int(parts[4]) if len(parts) > 4 else None
        except (ValueError, IndexError):
            continue

        roles = ["A", "B"] + (["C"] if C is not None else [])
        role  = random.choice(roles)

        similar = synthon_lib.find_similar_components(
            elite_smiles, role=role,
            top_k=neighborhood_size, min_similarity=0.55,
        )

        for comp_id, sim in similar:
            aspiration = elite.get("score", float("-inf")) > state.global_best_score
            is_tabu    = any(
                cid == comp_id and rem > 0
                for cid, rem in tabu.get(role, deque())
            )
            if is_tabu and not aspiration:
                continue

            if role == "A":
                name = f"rxn:{eff_rxn_id}:{comp_id}:{B}:{C}" if C is not None else f"rxn:{eff_rxn_id}:{comp_id}:{B}"
            elif role == "B":
                name = f"rxn:{eff_rxn_id}:{A}:{comp_id}:{C}" if C is not None else f"rxn:{eff_rxn_id}:{A}:{comp_id}"
            else:
                name = f"rxn:{eff_rxn_id}:{A}:{B}:{comp_id}"

            if name not in state.global_seen:
                new_names.append(name)
                break

    return new_names


def update_tabu(
    state: DPEXDJAState,
    molecule_name: str,
    tenure: int = 8,
    rxn_id: int = None,
) -> None:
    """Add molecule's components to tabu memory."""
    parts = molecule_name.split(":")
    if len(parts) < 4:
        return
    try:
        A = int(parts[2]); B = int(parts[3])
        C = int(parts[4]) if len(parts) > 4 else None
    except (ValueError, IndexError):
        return
    tabu = (
        state.tabu_rxn.get(rxn_id, state.tabu)
        if (state.is_multi_rxn and rxn_id is not None)
        else state.tabu
    )
    tabu["A"].append((A, tenure))
    tabu["B"].append((B, tenure))
    if C is not None:
        tabu["C"].append((C, tenure))


def dpex_exchange(state: DPEXDJAState, rxn_id: int = None) -> None:
    """Inject m_ex best-of-A into B every T_ex iterations."""
    if state.iteration % state.T_ex != 0:
        return

    if state.is_multi_rxn and rxn_id is not None:
        pop_a = state.pop_A_rxn.get(rxn_id, [])
        pop_b = state.pop_B_rxn.get(rxn_id, [])
        cap   = state.N_B_per_rxn
    else:
        pop_a = state.pop_A
        pop_b = state.pop_B
        cap   = state.N_B

    if not pop_a:
        return

    sorted_a  = sorted(pop_a, key=lambda x: x.get("score", float("-inf")), reverse=True)
    to_inject = sorted_a[: state.m_ex]
    pop_b.extend(to_inject)
    merged = sorted(
        {m["name"]: m for m in pop_b}.values(),
        key=lambda x: x.get("score", float("-inf")),
        reverse=True,
    )[:cap]

    if state.is_multi_rxn and rxn_id is not None:
        state.pop_B_rxn[rxn_id] = merged
        state._rebuild_flat_views()
    else:
        state.pop_B = merged

    bt.logging.info(
        f"[DPEX Exchange] iter={state.iteration}: "
        f"injected {len(to_inject)} A→B "
        f"(|A|={len(pop_a)}, |B|={len(merged)})"
    )


def update_populations(
    state: DPEXDJAState,
    scored_df: pd.DataFrame,
    rxn_id: int = None,
) -> None:
    """Merge scored molecules into pop_A/pop_B, update global best."""
    if scored_df.empty:
        return

    improved    = False
    new_records = scored_df.to_dict("records")

    for rec in new_records:
        name  = rec.get("name", "")
        score = rec.get("score")
        if score is None or (isinstance(score, float) and pd.isna(score)):
            continue
        state.global_seen.add(name)
        if float(score) > state.global_best_score:
            state.global_best_score    = float(score)
            state.global_best_molecule = rec
            improved = True

    if improved:
        state.no_improvement_counter = 0
        bt.logging.info(
            f"[DPEX] New global best: {state.global_best_score:.6f} "
            f"({state.global_best_molecule.get('name', '')})"
        )
    else:
        state.no_improvement_counter += 1
        if state.no_improvement_counter % 3 == 0:
            bt.logging.info(
                f"[DPEX] Stagnation counter: {state.no_improvement_counter} "
                f"(exploit_intensity={state.exploit_intensity:.2f})"
            )

    scored_records = [
        r for r in new_records
        if r.get("score") is not None
        and not (isinstance(r.get("score"), float) and pd.isna(r["score"]))
    ]

    if state.is_multi_rxn and rxn_id is not None:
        arm_records = [
            r for r in scored_records
            if str(r.get("name", "")).split(":")[1:2] == [str(rxn_id)]
        ]
        state.pop_A_rxn.setdefault(rxn_id, []).extend(arm_records)
        if len(state.pop_A_rxn[rxn_id]) > state.N_A_per_rxn:
            state.pop_A_rxn[rxn_id] = sorted(
                state.pop_A_rxn[rxn_id],
                key=lambda x: x.get("score", float("-inf")),
                reverse=True,
            )[: state.N_A_per_rxn]
        # Pop B: tabu/dja/exploit only
        elite = [
            r for r in arm_records
            if r.get("generation_method", "") in ("tabu", "dja", "exploit")
        ]
        if elite:
            state.pop_B_rxn.setdefault(rxn_id, []).extend(elite)
            if len(state.pop_B_rxn[rxn_id]) > state.N_B_per_rxn:
                state.pop_B_rxn[rxn_id] = sorted(
                    state.pop_B_rxn[rxn_id],
                    key=lambda x: x.get("score", float("-inf")),
                    reverse=True,
                )[: state.N_B_per_rxn]
        state._rebuild_flat_views()
    else:
        state.pop_A.extend(scored_records)
        if len(state.pop_A) > state.N_A:
            state.pop_A = sorted(
                state.pop_A,
                key=lambda x: x.get("score", float("-inf")),
                reverse=True,
            )[: state.N_A]
        elite = [
            r for r in scored_records
            if r.get("generation_method", "") in ("tabu", "dja", "exploit")
        ]
        if elite:
            state.pop_B.extend(elite)
            if len(state.pop_B) > state.N_B:
                state.pop_B = sorted(
                    state.pop_B,
                    key=lambda x: x.get("score", float("-inf")),
                    reverse=True,
                )[: state.N_B]

    state.iteration += 1