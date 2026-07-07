"""
Enhanced Molecule Generation with DPEX-DJA Integration
========================================================

Integrates:
- Synthon Library (similarity-based generation)
- Tabu Memory (component tracking)
- DJA Population Dynamics (dual-population exploration/exploitation)
- Exploit Mode (reactive intensification on stagnation)

Reference: DJAYA algorithm for discrete optimization
"""

import random
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs

logger = logging.getLogger(__name__)

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)


@dataclass
class MoleculeCandidate:
    """Represents a candidate molecule with metadata."""
    name: str
    smiles: str
    components: Tuple[int, ...] = None  # (A_id, B_id) or (A_id, B_id, C_id)
    score: Optional[float] = None
    generation_method: str = "unknown"  # "crossover", "dja", "tabu", "exploit"
    fingerprint: object = None
    
    def __post_init__(self):
        if self.smiles and self.fingerprint is None:
            try:
                mol = Chem.MolFromSmiles(self.smiles)
                if mol:
                    self.fingerprint = MORGAN_FP_GENERATOR.GetFingerprint(mol)
            except Exception as e:
                logger.debug(f"Error computing fingerprint: {e}")


@dataclass
class TabuMemory:
    """Tabu memory for component tracking."""
    max_length: int = 100
    component_tabu: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=100))
    )
    
    def add_tabu_move(self, role: str, component_id: int, iterations: int = 10):
        """Add component to tabu list for specific role."""
        if not isinstance(component_id, int):
            logger.warning(f"Expected int component_id, got {type(component_id)}")
            return
        self.component_tabu[role].append((component_id, iterations))
    
    def is_tabu(self, role: str, component_id: int) -> bool:
        """Check if move is tabu."""
        if role not in self.component_tabu:
            return False
        
        for comp_id, remaining_iters in list(self.component_tabu[role]):
            if comp_id == component_id and remaining_iters > 0:
                return True
        return False
    
    def decay_tabu(self):
        """Decrease tabu tenure for all moves."""
        for role in self.component_tabu:
            updated = deque(maxlen=self.max_length)
            for comp_id, remaining_iters in self.component_tabu[role]:
                if remaining_iters > 1:
                    updated.append((comp_id, remaining_iters - 1))
            self.component_tabu[role] = updated


class SynthonLibraryEnhanced:
    """Enhanced Synthon Library with similarity indexing."""
    
    def __init__(
        self,
        molecules_A: List[Tuple[int, str, int]],
        molecules_B: List[Tuple[int, str, int]],
        molecules_C: List[Tuple[int, str, int]] = None
    ):
        """
        Initialize synthon library from component lists.
        
        Args:
            molecules_A: List of (mol_id, smiles, role_mask)
            molecules_B: List of (mol_id, smiles, role_mask)
            molecules_C: List of (mol_id, smiles, role_mask) or None
        """
        self.molecules_A = molecules_A
        self.molecules_B = molecules_B
        self.molecules_C = molecules_C or []
        
        self.fps_A = self._build_fingerprint_index(molecules_A)
        self.fps_B = self._build_fingerprint_index(molecules_B)
        self.fps_C = self._build_fingerprint_index(molecules_C) if molecules_C else {}
        
        logger.info(
            f"[SynthonLibrary] Initialized: {len(self.fps_A)} A, "
            f"{len(self.fps_B)} B, {len(self.fps_C)} C components"
        )
    
    @staticmethod
    def _build_fingerprint_index(
        molecules: List[Tuple[int, str, int]]
    ) -> Dict[int, object]:
        """Build fingerprint index for molecules."""
        fps = {}
        for mol_id, smiles, _ in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                    fps[mol_id] = fp
            except Exception as e:
                logger.debug(f"Error building FP for {mol_id}: {e}")
        return fps
    
    def find_similar_components(
        self,
        target_smiles: str,
        role: str = 'A',
        top_k: int = 50,
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find similar components by fingerprint similarity.
        
        Args:
            target_smiles: SMILES string to compare against
            role: 'A', 'B', or 'C'
            top_k: Number of top results to return
            min_similarity: Minimum Tanimoto similarity threshold
        
        Returns:
            List of (component_id, similarity_score) tuples
        """
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            if not target_mol:
                return []
            
            target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)
            
            fps_dict = {
                'A': self.fps_A,
                'B': self.fps_B,
                'C': self.fps_C
            }.get(role, {})
            
            if not fps_dict:
                return []
            
            similarities = []
            for mol_id, fp in fps_dict.items():
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                if sim >= min_similarity:
                    similarities.append((mol_id, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.debug(f"Error finding similar components: {e}")
            return []


class DJAPopulationManager:
    """
    Manages dual populations for DPEX-DJA algorithm.
    
    Population A: Global exploration via DJA update rule
    Population B: Local refinement via tabu-enhanced search
    """
    
    def __init__(
        self,
        rxn_id: int,
        pop_a_size: int = 150,
        pop_b_size: int = 100,
        exchange_frequency: int = 2,
        exchange_count: int = 20
    ):
        self.rxn_id = rxn_id
        self.pop_a: List[MoleculeCandidate] = []
        self.pop_b: List[MoleculeCandidate] = []
        self.pop_a_size = pop_a_size
        self.pop_b_size = pop_b_size
        self.exchange_frequency = exchange_frequency
        self.exchange_count = exchange_count
        self.iteration = 0
        self.tabu_memory = TabuMemory(max_length=80)
    
    def dja_update(
        self,
        current: MoleculeCandidate,
        best: MoleculeCandidate,
        worst: MoleculeCandidate,
        component_ids_A: List[int],
        component_ids_B: List[int],
        component_ids_C: List[int] = None,
        mutation_prob: float = 0.3
    ) -> MoleculeCandidate:
        """
        Apply Discrete Jaya Algorithm update to molecule.
        
        DJA rule (discrete version):
        - Attract to best solution's components with probability
        - Repel from worst solution's components with probability
        
        Args:
            current: Current molecule candidate
            best: Best molecule in population
            worst: Worst molecule in population
            component_ids_A: Available A component IDs
            component_ids_B: Available B component IDs
            component_ids_C: Available C component IDs (optional)
            mutation_prob: Probability of mutation per component
        
        Returns:
            New MoleculeCandidate with updated components
        """
        if not current.components or not best.components:
            return current
        
        num_components = len(current.components)
        new_components = list(current.components)
        
        component_pools = [component_ids_A, component_ids_B]
        if component_ids_C:
            component_pools.append(component_ids_C)
        
        for role_idx in range(num_components):
            if role_idx >= len(component_pools):
                break
            
            available_ids = component_pools[role_idx]
            if not available_ids:
                continue
            
            # Attraction to best
            if random.random() < mutation_prob and best.components:
                new_components[role_idx] = best.components[role_idx]
            
            # Repulsion from worst
            elif random.random() < mutation_prob and worst.components:
                # Find different component from worst
                worst_id = worst.components[role_idx]
                different_ids = [c for c in available_ids if c != worst_id]
                if different_ids:
                    new_components[role_idx] = random.choice(different_ids)
        
        new_candidate = MoleculeCandidate(
            name=self._make_reaction_name(new_components),
            smiles=current.smiles,
            components=tuple(new_components),
            generation_method="dja"
        )
        return new_candidate
    
    def tabu_neighborhood_search(
        self,
        elite: MoleculeCandidate,
        synthon_library: SynthonLibraryEnhanced,
        component_ids_A: List[int],
        component_ids_B: List[int],
        component_ids_C: List[int] = None,
        neighborhood_size: int = 10
    ) -> List[MoleculeCandidate]:
        """
        Generate neighborhood via tabu-enhanced local search.
        
        - Finds similar components using synthon library
        - Blocks tabu moves unless aspiration criteria met
        - Generates diverse neighbors
        """
        neighbors = []
        
        if not elite.smiles or not elite.components:
            return neighbors
        
        roles = ['A', 'B', 'C']
        component_pools = [component_ids_A, component_ids_B]
        if component_ids_C:
            component_pools.append(component_ids_C)
        
        for role_idx in range(len(elite.components)):
            if role_idx >= len(roles) or role_idx >= len(component_pools):
                break
            
            role = roles[role_idx]
            
            # Find similar components
            similar = synthon_library.find_similar_components(
                elite.smiles,
                role=role,
                top_k=neighborhood_size,
                min_similarity=0.55
            )
            
            for comp_id, similarity in similar[:neighborhood_size]:
                # Check tabu status
                if self.tabu_memory.is_tabu(role, comp_id):
                    continue
                
                # Create neighbor by swapping component
                new_components = list(elite.components)
                new_components[role_idx] = comp_id
                
                neighbor = MoleculeCandidate(
                    name=self._make_reaction_name(new_components),
                    smiles=elite.smiles,
                    components=tuple(new_components),
                    generation_method="tabu"
                )
                neighbors.append(neighbor)
            
            # Add to tabu memory
            self.tabu_memory.add_tabu_move(
                role, elite.components[role_idx], iterations=8
            )
        
        self.tabu_memory.decay_tabu()
        return neighbors
    
    def _make_reaction_name(self, components: List[int]) -> str:
        """
        Create reaction name from component IDs.
        Format: rxn:rxn_id:A_id:B_id[:C_id]
        """
        parts = [f"rxn:{self.rxn_id}"]
        parts.extend(str(c) for c in components)
        return ':'.join(parts)
    
    def exchange_populations(self):
        """
        Exchange best of A into B (every T_ex iterations).
        
        - Injects best-of-A into population B
        - Removes worst-of-B to maintain size
        """
        if self.iteration % self.exchange_frequency != 0:
            return
        
        if len(self.pop_a) == 0:
            return
        
        # Sort both populations by score
        pop_a_sorted = sorted(
            self.pop_a,
            key=lambda m: m.score if m.score is not None else float('-inf'),
            reverse=True
        )
        
        # Exchange top molecules
        exchange_count = min(self.exchange_count, len(pop_a_sorted))
        to_inject = pop_a_sorted[:exchange_count]
        
        self.pop_b.extend(to_inject)
        
        # Trim to size
        if len(self.pop_b) > self.pop_b_size:
            self.pop_b = sorted(
                self.pop_b,
                key=lambda m: m.score if m.score is not None else float('-inf'),
                reverse=True
            )[:self.pop_b_size]
        
        logger.info(
            f"[Exchange] Iteration {self.iteration}: "
            f"Injected {exchange_count} from A into B "
            f"(pop_A size: {len(self.pop_a)}, pop_B size: {len(self.pop_b)})"
        )
    
    def update_populations(
        self,
        new_candidates: List[MoleculeCandidate],
        scored_results: Dict[str, float]
    ):
        """Update populations with newly scored candidates."""
        # Score new candidates
        for candidate in new_candidates:
            if candidate.name in scored_results:
                candidate.score = scored_results[candidate.name]
        
        # Add to population A (moving window)
        self.pop_a.extend(new_candidates)
        if len(self.pop_a) > self.pop_a_size:
            self.pop_a = sorted(
                self.pop_a,
                key=lambda m: m.score if m.score is not None else float('-inf'),
                reverse=True
            )[:self.pop_a_size]
        
        # Optionally add to population B
        scored_candidates = [
            c for c in new_candidates
            if c.score is not None and c.generation_method in ["tabu", "dja"]
        ]
        if scored_candidates:
            self.pop_b.extend(scored_candidates)
            if len(self.pop_b) > self.pop_b_size:
                self.pop_b = sorted(
                    self.pop_b,
                    key=lambda m: m.score if m.score is not None else float('-inf'),
                    reverse=True
                )[:self.pop_b_size]
        
        self.iteration += 1


class EnhancedMoleculeGenerator:
    """
    Enhanced molecule generator combining multiple strategies.
    
    Strategies:
    1. DJA-based generation (global exploration)
    2. Tabu-enhanced local search (local refinement)
    3. Exploit mode (reactive intensification)
    4. Crossover (baseline)
    """
    
    def __init__(
        self,
        rxn_id: int,
        synthon_library: SynthonLibraryEnhanced,
        component_ids_A: List[int],
        component_ids_B: List[int],
        component_ids_C: List[int] = None
    ):
        """
        Initialize enhanced generator.
        
        Args:
            rxn_id: Reaction ID
            synthon_library: SynthonLibraryEnhanced instance
            component_ids_A: List of A component IDs
            component_ids_B: List of B component IDs
            component_ids_C: List of C component IDs (optional)
        """
        self.rxn_id = rxn_id
        self.synthon_library = synthon_library
        self.component_ids_A = component_ids_A
        self.component_ids_B = component_ids_B
        self.component_ids_C = component_ids_C or []
        self.dja_manager = DJAPopulationManager(rxn_id)
        self.stagnation_counter = 0
        self.last_best_score = float('-inf')
        self.generation_counter = 0

    def seed_populations_from_molecules(self, molecules: List[Dict]) -> int:
        """Warm-start DJA populations from scored seed molecules."""
        if len(self.dja_manager.pop_a) >= 2:
            return 0

        seeded = 0
        for mol in molecules:
            name = mol.get('name', '')
            if not name.startswith(f'rxn:{self.rxn_id}:'):
                continue

            parts = name.split(':')
            if len(parts) not in (4, 5):
                continue

            try:
                components = tuple(int(p) for p in parts[2:])
            except ValueError:
                continue

            candidate = MoleculeCandidate(
                name=name,
                smiles=mol.get('smiles', ''),
                components=components,
                score=mol.get('score'),
                generation_method='seed',
            )
            self.dja_manager.pop_a.append(candidate)
            self.dja_manager.pop_b.append(candidate)
            seeded += 1

        if len(self.dja_manager.pop_a) > self.dja_manager.pop_a_size:
            self.dja_manager.pop_a = sorted(
                self.dja_manager.pop_a,
                key=lambda m: m.score if m.score is not None else float('-inf'),
                reverse=True,
            )[:self.dja_manager.pop_a_size]

        if len(self.dja_manager.pop_b) > self.dja_manager.pop_b_size:
            self.dja_manager.pop_b = sorted(
                self.dja_manager.pop_b,
                key=lambda m: m.score if m.score is not None else float('-inf'),
                reverse=True,
            )[:self.dja_manager.pop_b_size]

        if seeded:
            logger.info(
                f"[Seed] Warm-started populations with {seeded} molecules "
                f"(pop_A={len(self.dja_manager.pop_a)}, "
                f"pop_B={len(self.dja_manager.pop_b)})"
            )
        return seeded

    def _generate_random_batch(self, count: int) -> List[MoleculeCandidate]:
        """Sample random component combinations from the full building-block pools."""
        candidates = []
        pools = [self.component_ids_A, self.component_ids_B]
        if self.component_ids_C:
            pools.append(self.component_ids_C)

        if not all(pools):
            return candidates

        for _ in range(count):
            components = tuple(random.choice(pool) for pool in pools)
            candidates.append(
                MoleculeCandidate(
                    name=self.dja_manager._make_reaction_name(list(components)),
                    smiles='',
                    components=components,
                    generation_method='random',
                )
            )
            self.generation_counter += 1

        return candidates

    def _generate_exploration_fallback(
        self,
        top_molecules: List[Dict],
        count: int,
    ) -> List[MoleculeCandidate]:
        """Use global random sampling when populations are not yet seeded."""
        if count <= 0:
            return []
        if len(self.dja_manager.pop_a) < 2:
            return self._generate_random_batch(count)
        return self._generate_crossover_batch(top_molecules, count)
    
    def generate_batch(
        self,
        top_molecules: List[Dict],
        strategy: str = "hybrid",
        batch_size: int = 100
    ) -> List[MoleculeCandidate]:
        """
        Generate batch of molecules using specified strategy.
        
        Strategies:
        - "hybrid": Mix of DJA, tabu, and crossover (40/30/20/10)
        - "dja": Pure DJA exploration
        - "tabu": Pure tabu local search
        - "crossover": Pure crossover (baseline)
        - "exploit": Reactive exploit mode
        
        Args:
            top_molecules: List of top molecules (dicts with 'name', 'smiles')
            strategy: Generation strategy
            batch_size: Number of molecules to generate
        
        Returns:
            List of MoleculeCandidate objects
        """
        candidates = []
        
        if strategy == "hybrid":
            # 40% DJA, 30% Tabu, 20% Crossover, 10% Exploit
            dja_count = int(batch_size * 0.40)
            tabu_count = int(batch_size * 0.30)
            crossover_count = int(batch_size * 0.20)
            exploit_count = batch_size - dja_count - tabu_count - crossover_count
            
            candidates.extend(
                self._generate_dja_batch(top_molecules, dja_count)
            )
            candidates.extend(
                self._generate_tabu_batch(top_molecules, tabu_count)
            )
            candidates.extend(
                self._generate_crossover_batch(top_molecules, crossover_count)
            )
            if exploit_count > 0 and self.stagnation_counter > 2:
                candidates.extend(
                    self._generate_exploit_batch(top_molecules, exploit_count)
                )
            else:
                candidates.extend(
                    self._generate_exploration_fallback(top_molecules, exploit_count)
                )

            if len(candidates) < batch_size:
                candidates.extend(
                    self._generate_exploration_fallback(
                        top_molecules,
                        batch_size - len(candidates),
                    )
                )
        
        elif strategy == "dja":
            candidates = self._generate_dja_batch(top_molecules, batch_size)
            if len(candidates) < batch_size // 2:
                candidates.extend(
                    self._generate_exploration_fallback(
                        top_molecules,
                        batch_size - len(candidates),
                    )
                )
        
        elif strategy == "tabu":
            candidates = self._generate_tabu_batch(top_molecules, batch_size)
            if len(candidates) < batch_size // 2:
                candidates.extend(
                    self._generate_exploration_fallback(
                        top_molecules,
                        batch_size - len(candidates),
                    )
                )
        
        elif strategy == "exploit":
            candidates = self._generate_exploit_batch(top_molecules, batch_size)
            if len(candidates) < batch_size // 2:
                candidates.extend(
                    self._generate_exploration_fallback(
                        top_molecules,
                        batch_size - len(candidates),
                    )
                )
        
        else:  # crossover (default)
            candidates = self._generate_crossover_batch(top_molecules, batch_size)
            if len(candidates) < batch_size // 2:
                candidates.extend(
                    self._generate_exploration_fallback(
                        top_molecules,
                        batch_size - len(candidates),
                    )
                )
        
        return candidates
    
    def _generate_dja_batch(
        self,
        top_molecules: List[Dict],
        count: int
    ) -> List[MoleculeCandidate]:
        """Generate molecules using DJA update rule."""
        candidates = []
        
        # If population not seeded, return empty
        if len(self.dja_manager.pop_a) < 2:
            return candidates
        
        pop_a_sorted = sorted(
            self.dja_manager.pop_a,
            key=lambda m: m.score if m.score is not None else float('-inf'),
            reverse=True
        )
        
        best = pop_a_sorted[0]
        worst = pop_a_sorted[-1]
        
        for _ in range(count):
            current = random.choice(self.dja_manager.pop_a)
            updated = self.dja_manager.dja_update(
                current, best, worst,
                self.component_ids_A,
                self.component_ids_B,
                self.component_ids_C
            )
            candidates.append(updated)
            self.generation_counter += 1
        
        return candidates
    
    def _generate_tabu_batch(
        self,
        top_molecules: List[Dict],
        count: int
    ) -> List[MoleculeCandidate]:
        """Generate molecules using tabu-enhanced local search."""
        candidates = []
        
        # If population not seeded, return empty
        if len(self.dja_manager.pop_b) == 0:
            return candidates
        
        for _ in range(count):
            elite = random.choice(self.dja_manager.pop_b)
            neighbors = self.dja_manager.tabu_neighborhood_search(
                elite,
                self.synthon_library,
                self.component_ids_A,
                self.component_ids_B,
                self.component_ids_C,
                neighborhood_size=10
            )
            if neighbors:
                candidates.append(random.choice(neighbors))
                self.generation_counter += 1
        
        return candidates
    
    def _generate_crossover_batch(
        self,
        top_molecules: List[Dict],
        count: int
    ) -> List[MoleculeCandidate]:
        """Generate molecules using crossover (baseline)."""
        candidates = []
        
        if len(top_molecules) < 2:
            return candidates
        
        for _ in range(count):
            parent1 = random.choice(top_molecules)
            parent2 = random.choice(top_molecules)
            
            name1 = parent1.get('name', '')
            name2 = parent2.get('name', '')
            
            parts1 = name1.split(':')
            parts2 = name2.split(':')
            
            # Validate format: rxn:rxn_id:A_id:B_id[:C_id]
            if len(parts1) == len(parts2) and len(parts1) in [4, 5]:
                try:
                    # Verify both are same rxn_id
                    if parts1[1] != parts2[1]:
                        continue
                    
                    offspring_parts = parts1.copy()
                    swap_idx = random.randint(2, len(parts1) - 1)
                    offspring_parts[swap_idx] = parts2[swap_idx]
                    
                    offspring_name = ':'.join(offspring_parts)
                    candidate = MoleculeCandidate(
                        name=offspring_name,
                        smiles=parent1.get('smiles', ''),
                        generation_method="crossover"
                    )
                    candidates.append(candidate)
                    self.generation_counter += 1
                except Exception as e:
                    logger.debug(f"Error in crossover: {e}")
        
        return candidates
    
    def _generate_exploit_batch(
        self,
        top_molecules: List[Dict],
        count: int
    ) -> List[MoleculeCandidate]:
        """Generate molecules using reactive exploit mode."""
        candidates = []
        
        if len(top_molecules) == 0:
            return candidates
        
        # Focus on top molecule's neighborhood
        best = top_molecules[0]
        best_smiles = best.get('smiles', '')
        best_name = best.get('name', '')
        
        if not best_smiles or not best_name:
            return candidates
        
        # Parse best molecule's components
        parts = best_name.split(':')
        if len(parts) not in [4, 5]:
            return candidates
        
        try:
            base_components = [int(p) for p in parts[2:]]
        except ValueError:
            return candidates
        
        roles = ['A', 'B', 'C']
        component_pools = [
            self.component_ids_A,
            self.component_ids_B,
            self.component_ids_C
        ]
        
        generated_per_role = count // len(roles)
        
        for role_idx, role in enumerate(roles):
            if role_idx >= len(base_components):
                break
            
            component_pool = component_pools[role_idx]
            if not component_pool:
                continue
            
            # Find similar components
            similar = self.synthon_library.find_similar_components(
                best_smiles,
                role=role,
                top_k=generated_per_role,
                min_similarity=0.60
            )
            
            for comp_id, _ in similar[:generated_per_role]:
                new_components = base_components.copy()
                new_components[role_idx] = comp_id
                
                candidate = MoleculeCandidate(
                    name=self.dja_manager._make_reaction_name(new_components),
                    smiles=best_smiles,
                    components=tuple(new_components),
                    generation_method="exploit"
                )
                candidates.append(candidate)
                self.generation_counter += 1
        
        return candidates
    
    def update_stagnation(self, best_score: float):
        """Track stagnation for exploit mode activation."""
        if best_score > self.last_best_score:
            self.stagnation_counter = 0
            self.last_best_score = best_score
            logger.info(
                f"[Progress] New best score: {best_score:.6f} "
                f"(stagnation: 0)"
            )
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter % 2 == 0:
                logger.info(
                    f"[Stagnation] Counter: {self.stagnation_counter} "
                    f"(best: {self.last_best_score:.6f})"
                )
        
        if self.stagnation_counter >= 3:
            logger.info(
                f"[Exploit] Activating exploit mode "
                f"(stagnation: {self.stagnation_counter})"
            )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get generator statistics."""
        return {
            'total_generated': self.generation_counter,
            'pop_a_size': len(self.dja_manager.pop_a),
            'pop_b_size': len(self.dja_manager.pop_b),
            'stagnation_counter': self.stagnation_counter,
            'last_best_score': self.last_best_score,
            'dja_iterations': self.dja_manager.iteration,
        }