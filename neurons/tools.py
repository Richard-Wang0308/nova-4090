"""
tools.py — adapted from reference code
Removed: time-constraint sampling (get_nsamples_from_time), nova_ph2
Kept:    bt.logging, all generation utilities, SynthonLibrary,
         SynthonLibraryRegistry, build_component_weights, entropy_phase_fix
Changed: IterationParams uses iteration-budget logic instead of time-budget
"""
import os
import random
import math
import bittensor as bt
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from itertools import chain

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

from molecules import MoleculeManager, MoleculeUtils, SubManager
from combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from utils.molecules import get_heavy_atom_count

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ═══════════════════════════════════════════════════════════════════════════
# IterationParams — iteration-budget (replaces time-budget)
# ═══════════════════════════════════════════════════════════════════════════

class IterationParams:
    """
    Controls sampling parameters across iterations.

    Key change from reference code:
      get_nsamples_from_time() → get_nsamples_from_iteration()
      Sampling decays mildly as populations mature.
      Stagnation boosts sampling for diversity recovery.
    """

    def __init__(self, config: dict):
        self.seen_molecules         = set()
        self.use_synthon_search     = False
        self.use_exploit_mode       = False
        self.base_samples           = 350
        self.no_improvement_counter = 0
        self.score_improvement_rate = 0.0
        self.mutation_prob          = 0.40
        self.elite_prob             = 0.60
        self.exploited_reactants    = set()
        self.synthon_lib            = None
        self.synthon_lib_registry   = None

        allowed = config.get("allowed_reaction")
        if allowed is None:
            self.n_samples_start = self.base_samples * 3
        else:
            self.n_samples_start = (
                self.base_samples * 3
                if allowed == "rxn:5"
                else self.base_samples * 4
            )

    def get_nsamples_from_iteration(self, iteration: int) -> int:
        """
        Iteration-budget adaptive sampling.
        Replaces get_nsamples_from_time() from reference code.

        Stagnation boost: if stuck, increase samples for diversity.
        Mild decay: reduce slightly as populations mature.
        """
        # Stagnation: boost samples to escape local optima
        if self.no_improvement_counter >= 5:
            return int(self.base_samples * 1.20)
        if self.no_improvement_counter >= 3:
            return int(self.base_samples * 1.10)

        # Mild decay by iteration
        if iteration < 10:
            return self.base_samples
        elif iteration < 30:
            return int(self.base_samples * 0.95)
        elif iteration < 60:
            return int(self.base_samples * 0.90)
        else:
            return int(self.base_samples * 0.85)


# ═══════════════════════════════════════════════════════════════════════════
# SynthonLibrary
# ═══════════════════════════════════════════════════════════════════════════

class SynthonLibrary:
    """
    Per-rxn FP index over A/B/C synthon pools.
    Constructible from SubManager (preferred) or MoleculeManager (legacy).
    """

    def __init__(self, molecule_manager):
        if isinstance(molecule_manager, SubManager):
            sub = molecule_manager
            self.molecule_manager = sub
            self.rxn_id = sub.rxn_id
        else:
            self.molecule_manager = molecule_manager
            self.rxn_id = getattr(molecule_manager, "rxn_id", None)
            sub = molecule_manager

        self.fps_A = SynthonLibrary._build_fingerprint_index(sub.molecules_A)
        self.fps_B = SynthonLibrary._build_fingerprint_index(sub.molecules_B)
        self.fps_C = (
            SynthonLibrary._build_fingerprint_index(sub.molecules_C)
            if sub.is_three_component
            else {}
        )

        bt.logging.info(
            f"[SynthonLibrary] rxn={self.rxn_id}: "
            f"{len(self.fps_A)} A, {len(self.fps_B)} B"
            + (f", {len(self.fps_C)} C" if sub.is_three_component else "")
        )

    @staticmethod
    def _build_fingerprint_index(
        molecules: List[Tuple[int, str, int]]
    ) -> Dict[int, object]:
        fps = {}
        for mol_id, smiles, _ in molecules:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol:
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                fps[mol_id] = fp
        return fps

    def find_similar_components(
        self,
        target_smiles: str,
        role: str = "A",
        top_k: int = 80,
        min_similarity: float = 0.5,
    ) -> List[Tuple[int, float]]:
        target_mol = MoleculeUtils.mol_from_smiles_cached(target_smiles)
        if not target_mol:
            return []
        target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)

        if role == "A":
            fps_dict = self.fps_A
        elif role == "B":
            fps_dict = self.fps_B
        elif role == "C" and self.molecule_manager.is_three_component:
            fps_dict = self.fps_C
        else:
            return []

        similarities = []
        for mol_id, fp in fps_dict.items():
            try:
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                if sim >= min_similarity:
                    similarities.append((mol_id, sim))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_similar_to_molecule_name(
        self,
        molecule_name: str,
        vary_component: str = "both",
        top_k_per_component: int = 10,
        min_similarity: float = 0.6,
    ) -> Dict[str, List[int]]:
        parts = molecule_name.split(":")
        if len(parts) < 4:
            return {}
        try:
            if len(parts) == 4:
                _, rxn, A_id, B_id = parts
                A_id, B_id = int(A_id), int(B_id)
                C_id = None
            else:
                _, rxn, A_id, B_id, C_id = parts
                A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
        except (ValueError, IndexError):
            return {}

        result = {}
        if vary_component in ["A", "both", "all"]:
            A_smiles = self._get_component_smiles(A_id, "A")
            if A_smiles:
                similar_As = self.find_similar_components(
                    A_smiles, "A", top_k_per_component, min_similarity
                )
                result["A"] = [mid for mid, _ in similar_As if mid != A_id]

        if vary_component in ["B", "both", "all"]:
            B_smiles = self._get_component_smiles(B_id, "B")
            if B_smiles:
                similar_Bs = self.find_similar_components(
                    B_smiles, "B", top_k_per_component, min_similarity
                )
                result["B"] = [mid for mid, _ in similar_Bs if mid != B_id]

        if (
            self.molecule_manager.is_three_component
            and C_id
            and vary_component in ["C", "all"]
        ):
            C_smiles = self._get_component_smiles(C_id, "C")
            if C_smiles:
                similar_Cs = self.find_similar_components(
                    C_smiles, "C", top_k_per_component, min_similarity
                )
                result["C"] = [mid for mid, _ in similar_Cs if mid != C_id]

        return result

    def _get_component_smiles(self, mol_id: int, role: str) -> Optional[str]:
        if role == "A":
            molecules = self.molecule_manager.molecules_A
        elif role == "B":
            molecules = self.molecule_manager.molecules_B
        elif role == "C":
            molecules = self.molecule_manager.molecules_C
        else:
            return None
        for mid, smiles, _ in molecules:
            if mid == mol_id:
                return smiles
        return None

    def generate_similar_molecules(
        self,
        base_molecule_names: List[str],
        n_per_base: int = 5,
        min_similarity: float = 0.6,
    ) -> List[str]:
        new_molecules = []
        is_single = len(base_molecule_names) == 1
        effective_n = (
            n_per_base if (is_single and n_per_base >= 80)
            else n_per_base * 3 if is_single
            else n_per_base
        )

        for base_name in base_molecule_names:
            parts = base_name.split(":")
            if len(parts) < 4:
                continue
            try:
                if len(parts) == 4:
                    _, rxn, A_id, B_id = parts
                    A_id, B_id = int(A_id), int(B_id)
                    similar = self.find_similar_to_molecule_name(
                        base_name, "both", effective_n, min_similarity
                    )
                    for new_A in similar.get("A", [])[:effective_n]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}")
                    for new_B in similar.get("B", [])[:effective_n]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}")
                else:
                    _, rxn, A_id, B_id, C_id = parts
                    A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
                    similar = self.find_similar_to_molecule_name(
                        base_name, "all", effective_n, min_similarity
                    )
                    for new_A in similar.get("A", [])[:effective_n]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}:{C_id}")
                    for new_B in similar.get("B", [])[:effective_n]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}:{C_id}")
                    for new_C in similar.get("C", [])[:effective_n]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{B_id}:{new_C}")
            except (ValueError, IndexError) as e:
                bt.logging.warning(f"Could not parse {base_name}: {e}")
                continue

        return list(dict.fromkeys(new_molecules))


# ═══════════════════════════════════════════════════════════════════════════
# SynthonLibraryRegistry
# ═══════════════════════════════════════════════════════════════════════════

class SynthonLibraryRegistry:
    """Registry of SynthonLibrary instances keyed by rxn_id."""

    def __init__(self, molecule_manager: MoleculeManager):
        self._registry: Dict[int, SynthonLibrary] = {}
        for r in molecule_manager.rxn_ids:
            try:
                sub = molecule_manager.for_rxn(r)
                self._registry[r] = SynthonLibrary(sub)
            except Exception as e:
                bt.logging.warning(
                    f"Could not build SynthonLibrary for rxn {r}: {e}"
                )

    def for_rxn(self, rxn_id: int) -> Optional[SynthonLibrary]:
        return self._registry.get(rxn_id)

    @property
    def rxn_ids(self) -> List[int]:
        return list(self._registry.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Candidate generation utilities
# ═══════════════════════════════════════════════════════════════════════════

def generate_valid_random_molecules(
    molecule_manager,
    n_samples: int,
    seen_molecules: set = None,
    component_weights: dict = None,
) -> List[str]:
    """Generate random valid molecule names from component pools."""
    seen   = seen_molecules or set()
    sub    = molecule_manager
    pool_A = sub.moles_A_id
    pool_B = sub.moles_B_id
    pool_C = sub.moles_C_id if sub.is_three_component else []
    rxn_id = sub.rxn_id

    if not pool_A or not pool_B:
        return []

    def _weighted_choice(pool, role):
        if component_weights and role in component_weights:
            w_dict  = component_weights[role]
            weights = [w_dict.get(i, 0.05) for i in pool]
            total   = sum(weights)
            weights = [w / total for w in weights] if total > 0 else None
            return random.choices(pool, weights=weights, k=1)[0] if weights else random.choice(pool)
        return random.choice(pool)

    names    = []
    attempts = 0
    max_att  = n_samples * 5

    while len(names) < n_samples and attempts < max_att:
        attempts += 1
        A = _weighted_choice(pool_A, "A")
        B = _weighted_choice(pool_B, "B")
        if sub.is_three_component and pool_C:
            C    = _weighted_choice(pool_C, "C")
            name = f"rxn:{rxn_id}:{A}:{B}:{C}"
        else:
            name = f"rxn:{rxn_id}:{A}:{B}"
        if name not in seen:
            names.append(name)
            seen.add(name)

    return names


def cpu_random_candidates_with_similarity(
    molecule_manager,
    synthon_lib: SynthonLibrary,
    elite_names: List[str],
    n_samples: int,
    similarity_fraction: float = 0.5,
    min_similarity: float = 0.55,
    component_weights: dict = None,
    seen_molecules: set = None,
) -> List[str]:
    """Mix of similarity-guided and random candidates."""
    seen      = seen_molecules or set()
    n_similar = int(n_samples * similarity_fraction)
    n_random  = n_samples - n_similar

    similar_names = []
    if elite_names and n_similar > 0:
        n_per = max(1, n_similar // len(elite_names))
        for name in elite_names:
            new = synthon_lib.generate_similar_molecules(
                [name], n_per_base=n_per, min_similarity=min_similarity
            )
            similar_names.extend(new)
        similar_names = [n for n in similar_names if n not in seen][:n_similar]

    random_names = generate_valid_random_molecules(
        molecule_manager, n_random, seen, component_weights
    )
    return list(dict.fromkeys(similar_names + random_names))[:n_samples]


def build_component_weights(
    top_pool: pd.DataFrame, rxn_id: int
) -> Dict[str, Dict[int, float]]:
    """Rank-based exponential component weights from scored pool."""
    from collections import defaultdict
    weights = {"A": defaultdict(float), "B": defaultdict(float), "C": defaultdict(float)}
    counts  = {"A": defaultdict(int),   "B": defaultdict(int),   "C": defaultdict(int)}

    if top_pool.empty:
        return weights

    for rank, (_, row) in enumerate(top_pool.iterrows()):
        score = row.get("score", 0.0)
        if pd.isna(score):
            continue
        rank_weight    = 2.5 * math.exp(-rank / 18.0)
        weighted_score = max(0, score) * rank_weight
        parts = str(row["name"]).split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2]); B_id = int(parts[3])
                weights["A"][A_id] += weighted_score
                weights["B"][B_id] += weighted_score
                counts["A"][A_id]  += 1
                counts["B"][B_id]  += 1
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights["C"][C_id] += weighted_score
                    counts["C"][C_id]  += 1
            except (ValueError, IndexError):
                continue

    for role in ["A", "B", "C"]:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = (
                    weights[role][comp_id] / counts[role][comp_id] + 0.15
                )
    return weights


def build_component_weights_multi(
    top_pool: pd.DataFrame, rxn_ids: List[int]
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """Multi-rxn component weights: {rxn_id: component_weights}."""
    return {
        r: build_component_weights(
            top_pool[top_pool["name"].str.startswith(f"rxn:{r}:")].copy(), r
        )
        for r in rxn_ids
    }


def compute_rxn_weights(
    top_pool: pd.DataFrame, rxn_ids: List[int]
) -> Dict[int, float]:
    """Normalized per-rxn sampling weights from top pool scores."""
    if top_pool.empty or not rxn_ids:
        return {r: 1.0 / len(rxn_ids) for r in rxn_ids}
    rxn_scores: Dict[int, List[float]] = {r: [] for r in rxn_ids}
    for _, row in top_pool.iterrows():
        parts = str(row.get("name", "")).split(":")
        if len(parts) >= 2:
            try:
                r = int(parts[1])
                if r in rxn_scores and not pd.isna(row.get("score")):
                    rxn_scores[r].append(float(row["score"]))
            except ValueError:
                continue
    raw   = {r: (max(s) if s else 0.0) for r, s in rxn_scores.items()}
    total = sum(raw.values())
    return (
        {r: v / total for r, v in raw.items()}
        if total > 0
        else {r: 1.0 / len(rxn_ids) for r in rxn_ids}
    )


def entropy_phase_fix(
    candidates: List[str],
    seen_molecules: set,
    molecule_manager,
    n_inject: int = 50,
) -> List[str]:
    """Inject random molecules when diversity drops (stagnation recovery)."""
    random_names = generate_valid_random_molecules(
        molecule_manager, n_inject, seen_molecules
    )
    combined = list(dict.fromkeys(candidates + random_names))
    bt.logging.info(
        f"[EntropyFix] Injected {len(random_names)} random molecules "
        f"(total: {len(combined)})"
    )
    return combined