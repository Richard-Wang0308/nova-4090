"""
DPEX_DJA + Boltz-2 Core Implementation (Config-Driven)
======================================================

Complete hybrid metaheuristic for expensive molecular scoring.

Pool sizes and parameters are loaded from dpex_pool_config.yaml
No code changes needed - just update the YAML file!

KEY FIXES:
- Removed state.generated_molecules.add() from DJAGenerator
- Removed state.generated_molecules.add() from TabuGenerator
- Removed state.generated_molecules.add() from ExploitModeGenerator
- Reordered QualityFilter checks (tabu check moved to LAST)
- Molecules marked as generated ONLY after successful scoring
"""

import os
import sqlite3
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from collections import deque
from dataclasses import dataclass, field
import time

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from combinatorial_db.reactions import get_smiles_from_reaction
from molecules_base import generate_inchikey

# Import config loader
from dpex_config_loader import load_dpex_config, DPEXConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# GLOBAL CONFIG (loaded at startup)
# ════════════════════════════════════════════════════════════════

DPEX_CONFIG: Optional[DPEXConfig] = None


def initialize_dpex_config(config_path: str = None) -> DPEXConfig:
    """Initialize DPEX configuration from YAML file."""
    global DPEX_CONFIG
    
    DPEX_CONFIG = load_dpex_config(config_path)
    
    # Setup logging level
    log_level = getattr(logging, DPEX_CONFIG.logging.level, logging.INFO)
    logger.setLevel(log_level)
    
    logger.info("✅ DPEX configuration initialized")
    return DPEX_CONFIG


def get_dpex_config() -> DPEXConfig:
    """Get current DPEX configuration."""
    global DPEX_CONFIG
    if DPEX_CONFIG is None:
        DPEX_CONFIG = initialize_dpex_config()
    return DPEX_CONFIG


# Fingerprint generators (cached)
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)
FCFP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)


@dataclass
class DPEXDJABoltzState:
    """Persistent state for DPEX_DJA with Boltz-2 scoring."""
    
    # Populations (sizes loaded from config)
    pop_A: List[Dict] = field(default_factory=list)
    pop_B: List[Dict] = field(default_factory=list)
    top_pool: pd.DataFrame = None
    
    # Tabu lists (sizes loaded from config)
    tabu_A: deque = field(default_factory=deque)
    tabu_B: deque = field(default_factory=deque)
    tabu_C: deque = field(default_factory=deque)
    
    # Tracking
    iteration: int = 0
    best_score: float = float('-inf')
    best_molecule: Optional[Dict] = None
    score_history: deque = field(default_factory=lambda: deque(maxlen=20))
    no_improvement_counter: int = 0
    use_exploit_mode: bool = False
    
    # Diversity tracking
    seen_inchikeys: Set[str] = field(default_factory=set)
    generated_molecules: Set[str] = field(default_factory=set)
    candidate_quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Exploit mode tracking
    exploited_reactants: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize tabu lists with config-driven sizes."""
        config = get_dpex_config()
        
        self.tabu_A = deque(maxlen=config.tabu.tabu_A_maxlen)
        self.tabu_B = deque(maxlen=config.tabu.tabu_B_maxlen)
        self.tabu_C = deque(maxlen=config.tabu.tabu_C_maxlen)
    
    def detect_stagnation(self) -> bool:
        """Detect if search is stagnating."""
        config = get_dpex_config()
        
        if len(self.score_history) < config.stagnation.window_size:
            return False
        
        scores = list(self.score_history)
        improvement = scores[-1] - scores[0]
        
        if improvement < config.stagnation.improvement_threshold:
            self.no_improvement_counter += 1
            return self.no_improvement_counter >= config.stagnation.exploit_mode_threshold
        else:
            self.no_improvement_counter = 0
            return False
    
    def update_best(self, molecule: Dict, score: float) -> bool:
        """Update best molecule if score improves."""
        if score > self.best_score:
            self.best_score = score
            self.best_molecule = molecule.copy()
            self.score_history.append(score)
            return True
        else:
            self.score_history.append(score)
            return False


class CandidateQualityFilter:
    """Pre-scoring candidate filter for expensive Boltz-2."""
    
    def __init__(self, elite_molecules: pd.DataFrame):
        """Initialize with elite molecules."""
        self.elite_molecules = elite_molecules
        self.elite_fps = self._build_fingerprints(elite_molecules)
        logger.info(f"[QualityFilter] Built fingerprints for {len(self.elite_fps)} elite molecules")
    
    @staticmethod
    def _build_fingerprints(molecules_df: pd.DataFrame) -> Dict[str, object]:
        """Build Morgan fingerprints for elite molecules."""
        fps = {}
        for idx, row in molecules_df.iterrows():
            try:
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                    fps[row['name']] = fp
            except Exception as e:
                logger.debug(f"Could not compute FP for {row['name']}: {e}")
        return fps
    
    def filter_candidates(
        self,
        candidates: List[Dict],
        state: DPEXDJABoltzState,
        max_candidates: int = 100
    ) -> Tuple[List[Dict], Dict]:
        """Filter candidates for scoring.
        
        FIXED ORDER:
        1. Check InChIKey duplicates
        2. Check similarity to elites
        3. Check diversity with filtered candidates
        4. Check if already generated (LAST)
        """
        
        config = get_dpex_config()
        
        stats = {
            'input_count': len(candidates),
            'removed_duplicates': 0,
            'removed_low_diversity': 0,
            'removed_low_similarity': 0,
            'removed_tabu': 0,
            'output_count': 0,
        }
        
        filtered = []
        seen_inchikeys = state.seen_inchikeys.copy()
        
        for candidate in candidates:
            name = candidate['name']
            smiles = candidate.get('smiles')
            inchikey = candidate.get('InChIKey')
            
            # ── STEP 1: Skip InChIKey duplicates ──
            if inchikey and inchikey in seen_inchikeys:
                stats['removed_duplicates'] += 1
                continue
            
            # ── STEP 2: Check similarity to elites ──
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    stats['removed_low_similarity'] += 1
                    continue
                
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                
                # Compute max similarity to any elite
                max_sim = 0.0
                for elite_fp in self.elite_fps.values():
                    sim = DataStructs.TanimotoSimilarity(fp, elite_fp)
                    max_sim = max(max_sim, sim)
                
                if max_sim < config.quality_filter.min_similarity_to_elite:
                    stats['removed_low_similarity'] += 1
                    continue
                
                # ── STEP 3: Check diversity with already-filtered candidates ──
                is_diverse = True
                for filtered_mol in filtered:
                    try:
                        filtered_fp = MORGAN_FP_GENERATOR.GetFingerprint(
                            Chem.MolFromSmiles(filtered_mol['smiles'])
                        )
                        sim = DataStructs.TanimotoSimilarity(fp, filtered_fp)
                        if sim > config.quality_filter.diversity_threshold:
                            is_diverse = False
                            break
                    except:
                        pass
                
                if not is_diverse:
                    stats['removed_low_diversity'] += 1
                    continue
                
                # ── STEP 4: Check if already generated (TABU) - LAST ──
                if name in state.generated_molecules:
                    stats['removed_tabu'] += 1
                    continue
                
                # ✅ Passed all filters
                filtered.append(candidate)
                if inchikey:
                    seen_inchikeys.add(inchikey)
                
                if len(filtered) >= max_candidates:
                    break
            
            except Exception as e:
                logger.debug(f"Error filtering {name}: {e}")
                stats['removed_low_similarity'] += 1
                continue
        
        stats['output_count'] = len(filtered)
        state.seen_inchikeys = seen_inchikeys
        
        if config.logging.verbose_filtering:
            logger.info(
                f"[QualityFilter] Stats: {stats['output_count']}/{stats['input_count']} passed "
                f"(removed: {stats['removed_duplicates']} dup, "
                f"{stats['removed_low_similarity']} low_sim, "
                f"{stats['removed_low_diversity']} low_div, "
                f"{stats['removed_tabu']} tabu)"
            )
        
        return filtered, stats


class AdaptiveBudgetController:
    """Manages candidate generation and scoring budget."""
    
    def __init__(self, dpex_config=None):
        """Initialize budget controller with optional config."""
        
        # Get config from global if not provided
        if dpex_config is None:
            try:
                dpex_config = get_dpex_config()
            except:
                dpex_config = None
        
        # Set scoring cost from config or use default
        if dpex_config and hasattr(dpex_config, 'scoring'):
            self.scoring_cost = dpex_config.scoring.cost_per_molecule
        else:
            self.scoring_cost = 30.0  # Default for Boltz-2
        
        self.iteration = 0
        self.improvement_history: List[float] = []
        self.candidate_budget_history: List[int] = []
        self.wall_clock_start = time.time()
        self.total_scoring_time = 0.0
        
        logger.info(f"[BudgetController] Initialized with scoring_cost={self.scoring_cost}s/molecule")
    
    def get_candidate_budget(
        self,
        iteration: int,
        improvement_rate: Optional[float] = None,
        wall_clock_remaining: Optional[float] = None
    ) -> Tuple[int, Dict]:
        """Compute candidate budget for this iteration."""
        
        config = get_dpex_config()
        
        budget_info = {
            'iteration': iteration,
            'improvement_rate': improvement_rate,
            'wall_clock_remaining': wall_clock_remaining,
            'reasoning': '',
        }
        
        # Phase 1: Seed (iteration 1)
        if iteration == 1:
            num_candidates = config.candidates.seed_budget
            budget_info['reasoning'] = 'Seed phase: build initial pool'
            return num_candidates, budget_info
        
        # Phase 2: Early exploration (iterations 2-5)
        if iteration <= 5:
            num_candidates = config.candidates.early_budget
            budget_info['reasoning'] = 'Early exploration phase'
            return num_candidates, budget_info
        
        # Phase 3: Adaptive (iteration 6+)
        if improvement_rate is None:
            num_candidates = config.candidates.normal_budget
            budget_info['reasoning'] = 'No improvement data, default'
            return num_candidates, budget_info
        
        # Adaptive logic
        if improvement_rate > 0.01:  # >1% improvement
            num_candidates = config.candidates.early_budget
            budget_info['reasoning'] = 'High improvement: maintain exploration'
        
        elif improvement_rate > 0.001:  # 0.1-1% improvement
            num_candidates = config.candidates.normal_budget
            budget_info['reasoning'] = 'Medium improvement: balanced search'
        
        else:  # <0.1% improvement
            num_candidates = config.candidates.stagnation_budget
            budget_info['reasoning'] = 'Low improvement: focus on refinement + exploit'
        
        # Adjust for time pressure
        if wall_clock_remaining and wall_clock_remaining < 300:
            num_candidates = max(20, int(num_candidates * 0.5))
            budget_info['reasoning'] += ' [TIME PRESSURE: reduced]'
        
        self.candidate_budget_history.append(num_candidates)
        return num_candidates, budget_info
    
    def get_dja_tabu_split(
        self,
        iteration: int,
        improvement_rate: Optional[float] = None
    ) -> Tuple[int, int, Dict]:
        """Compute DJA/Tabu split."""
        
        config = get_dpex_config()
        
        split_info = {
            'iteration': iteration,
            'improvement_rate': improvement_rate,
        }
        
        # Early iterations: favor exploration
        if iteration <= 5:
            dja_ratio = config.candidates.dja_ratio_early
            tabu_ratio = config.candidates.tabu_ratio_early
            split_info['reasoning'] = 'Early: favor exploration'
        
        # Late iterations: adapt based on improvement
        elif improvement_rate is None or improvement_rate > 0.001:
            dja_ratio = config.candidates.dja_ratio_early
            tabu_ratio = config.candidates.tabu_ratio_early
            split_info['reasoning'] = 'Good improvement: maintain balance'
        
        else:  # Stagnating
            dja_ratio = config.candidates.dja_ratio_stagnation
            tabu_ratio = config.candidates.tabu_ratio_stagnation
            split_info['reasoning'] = 'Stagnation: shift to refinement'
        
        num_candidates = self.candidate_budget_history[-1] if self.candidate_budget_history else 100
        num_dja = int(num_candidates * dja_ratio)
        num_tabu = num_candidates - num_dja
        
        return num_dja, num_tabu, split_info
    
    def should_activate_exploit_mode(
        self,
        improvement_rate: float,
        no_improvement_iterations: int
    ) -> Tuple[bool, str]:
        """Decide whether to activate exploit mode."""
        
        config = get_dpex_config()
        
        if no_improvement_iterations >= config.stagnation.exploit_mode_threshold:
            return True, f"Stagnation for {no_improvement_iterations} iterations"
        
        if improvement_rate < 0.0001:
            return True, f"Improvement rate critically low: {improvement_rate:.6f}"
        
        return False, ""
    
    def log_budget_summary(self) -> None:
        """Log budget allocation summary."""
        if not self.candidate_budget_history:
            return
        
        total_candidates = sum(self.candidate_budget_history)
        avg_candidates = np.mean(self.candidate_budget_history)
        
        logger.info(
            f"📊 Budget Summary:"
            f"\n   Total candidates generated: {total_candidates}"
            f"\n   Average per iteration: {avg_candidates:.1f}"
            f"\n   Total scoring time: {self.total_scoring_time:.1f}s"
            f"\n   Iterations: {len(self.candidate_budget_history)}"
        )


class DJAGenerator:
    """Discrete Jaya Algorithm for global exploration."""
    
    def __init__(self, molecule_manager):
        """Initialize DJA generator."""
        self.molecule_manager = molecule_manager
    
    def generate_candidates(
        self,
        pop_A: List[Dict],
        num_candidates: int,
        state: DPEXDJABoltzState
    ) -> List[Dict]:
        """Generate candidates via DJA update rule.
        
        FIXED: Do NOT add to state.generated_molecules here.
        This will be done AFTER scoring in the main loop.
        """
        
        config = get_dpex_config()
        
        if not pop_A:
            logger.warning("[DJA] Population A is empty, returning empty candidates")
            return []
        
        # Use elite diversity if configured
        if config.dja.use_elite_diversity and len(pop_A) > 100:
            sorted_pop = sorted(pop_A, key=lambda x: x.get('score', float('-inf')), reverse=True)
            elite_size = max(10, int(len(pop_A) * config.dja.elite_percentage))
            elite_best = sorted_pop[:elite_size]
            elite_worst = sorted_pop[-elite_size:]
        else:
            best_A = max(pop_A, key=lambda x: x.get('score', float('-inf')))
            worst_A = min(pop_A, key=lambda x: x.get('score', float('inf')))
            elite_best = [best_A]
            elite_worst = [worst_A]
        
        candidates = []
        attempts = 0
        max_attempts = num_candidates * 5
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            
            # Select random member from pop_A
            parent = random.choice(pop_A)
            parent_name = parent['name']
            
            # Parse components
            try:
                parts = parent_name.split(':')
                if len(parts) not in [4, 5]:
                    continue
                
                num_components = len(parts) - 2
                component_indices = list(range(2, 2 + num_components))
                
                # DJA update: probabilistic attraction/repulsion
                offspring_parts = parts.copy()
                
                for comp_idx in component_indices:
                    # Adopt best_A's component with configured probability
                    if random.random() < config.dja.best_adoption_probability:
                        best_sol = random.choice(elite_best)
                        offspring_parts[comp_idx] = best_sol['name'].split(':')[comp_idx]
                    
                    # If matches worst_A, escape with configured probability
                    if offspring_parts[comp_idx] in [w['name'].split(':')[comp_idx] for w in elite_worst]:
                        if random.random() < config.dja.escape_probability:
                            # Random component from pool
                            if comp_idx == 2:
                                pool = self.molecule_manager.molecules_A
                            elif comp_idx == 3:
                                pool = self.molecule_manager.molecules_B
                            else:
                                pool = self.molecule_manager.molecules_C if hasattr(self.molecule_manager, 'molecules_C') else []
                            
                            if pool:
                                random_id = random.choice(pool)[0]
                                offspring_parts[comp_idx] = str(random_id)
                
                offspring_name = ':'.join(offspring_parts)
                
                # Validate
                if offspring_name in state.generated_molecules:
                    continue
                
                try:
                    offspring_smiles = get_smiles_from_reaction(offspring_name)
                    if offspring_smiles:
                        mol = Chem.MolFromSmiles(offspring_smiles)
                        if mol is not None:
                            inchikey = generate_inchikey(offspring_smiles)
                            if inchikey:
                                candidates.append({
                                    'name': offspring_name,
                                    'smiles': offspring_smiles,
                                    'InChIKey': inchikey,
                                    'type': 'dja',
                                    'parent': parent_name,
                                })
                                # FIXED: Do NOT add to generated_molecules here
                except:
                    pass
            
            except Exception as e:
                logger.debug(f"[DJA] Error generating candidate: {e}")
                continue
        
        if config.logging.verbose_generation:
            logger.info(f"[DJA] Generated {len(candidates)} valid candidates (attempts: {attempts})")
        
        return candidates


class TabuGenerator:
    """Tabu-enhanced local search for refinement."""
    
    def __init__(self, molecule_manager):
        """Initialize Tabu generator."""
        self.molecule_manager = molecule_manager
    
    def generate_candidates(
        self,
        pop_B: List[Dict],
        num_candidates: int,
        state: DPEXDJABoltzState
    ) -> List[Dict]:
        """Generate candidates via tabu-enhanced local search.
        
        FIXED: Do NOT add to state.generated_molecules here.
        This will be done AFTER scoring in the main loop.
        """
        
        config = get_dpex_config()
        
        if not pop_B:
            logger.warning("[Tabu] Population B is empty, returning empty candidates")
            return []
        
        candidates = []
        
        # Select top elites from pop_B
        elites = sorted(pop_B, key=lambda x: x.get('score', float('-inf')), reverse=True)[:10]
        
        for elite in elites:
            elite_name = elite['name']
            
            try:
                parts = elite_name.split(':')
                if len(parts) not in [4, 5]:
                    continue
                
                num_components = len(parts) - 2
                component_indices = list(range(2, 2 + num_components))
                
                # Generate neighbors by swapping components
                for comp_idx in component_indices:
                    # Get similar components from pool
                    current_id = int(parts[comp_idx])
                    
                    # Find similar components
                    similar_ids = self._find_similar_components(
                        current_id, comp_idx,
                        min_similarity=config.tabu_algo.min_similarity_for_neighbors
                    )
                    
                    for new_id in similar_ids[:config.tabu_algo.num_similar_components]:
                        # Check tabu list
                        move = (current_id, new_id)
                        
                        if move in state.tabu_A:
                            # Check aspiration
                            if elite.get('score', 0) < state.best_score * config.tabu_algo.aspiration_threshold:
                                continue
                        
                        # Create neighbor
                        neighbor_parts = parts.copy()
                        neighbor_parts[comp_idx] = str(new_id)
                        neighbor_name = ':'.join(neighbor_parts)
                        
                        if neighbor_name in state.generated_molecules:
                            continue
                        
                        try:
                            neighbor_smiles = get_smiles_from_reaction(neighbor_name)
                            if neighbor_smiles:
                                mol = Chem.MolFromSmiles(neighbor_smiles)
                                if mol is not None:
                                    inchikey = generate_inchikey(neighbor_smiles)
                                    if inchikey:
                                        candidates.append({
                                            'name': neighbor_name,
                                            'smiles': neighbor_smiles,
                                            'InChIKey': inchikey,
                                            'type': 'tabu',
                                            'parent': elite_name,
                                            'move': move,
                                        })
                                        # FIXED: Do NOT add to generated_molecules here
                                        
                                        if len(candidates) >= num_candidates:
                                            break
                        except:
                            pass
                    
                    if len(candidates) >= num_candidates:
                        break
            
            except Exception as e:
                logger.debug(f"[Tabu] Error generating neighbor for {elite_name}: {e}")
                continue
            
            if len(candidates) >= num_candidates:
                break
        
        if config.logging.verbose_generation:
            logger.info(f"[Tabu] Generated {len(candidates)} valid candidates")
        
        return candidates
    
    def _find_similar_components(
        self,
        component_id: int,
        component_type: int,
        min_similarity: float = 0.50
    ) -> List[int]:
        """Find similar components using fingerprint similarity."""
        
        try:
            # Get the component pool
            if component_type == 2:
                pool = self.molecule_manager.molecules_A
            elif component_type == 3:
                pool = self.molecule_manager.molecules_B
            else:
                pool = self.molecule_manager.molecules_C if hasattr(self.molecule_manager, 'molecules_C') else []
            
            if not pool:
                return []
            
            # Find component SMILES
            target_smiles = None
            for mol_id, smiles, _ in pool:
                if mol_id == component_id:
                    target_smiles = smiles
                    break
            
            if not target_smiles:
                return []
            
            # Compute similarity to all components
            target_mol = Chem.MolFromSmiles(target_smiles)
            if not target_mol:
                return []
            
            target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)
            
            similarities = []
            for mol_id, smiles, _ in pool:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                        sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                        if sim >= min_similarity and mol_id != component_id:
                            similarities.append((mol_id, sim))
                except:
                    pass
            
            # Sort by similarity and return IDs
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [mol_id for mol_id, _ in similarities[:20]]
        
        except Exception as e:
            logger.debug(f"[Tabu] Error finding similar components: {e}")
            return []


class ExploitModeGenerator:
    """Deep search on stagnation using similarity-based filtering."""
    
    def __init__(self, molecule_manager):
        """Initialize exploit mode generator."""
        self.molecule_manager = molecule_manager
    
    def generate_candidates(
        self,
        best_molecule: Dict,
        num_candidates: int,
        state: DPEXDJABoltzState
    ) -> List[Dict]:
        """Generate candidates via deep similarity-based search.
        
        FIXED: Do NOT add to state.generated_molecules here.
        This will be done AFTER scoring in the main loop.
        """
        
        config = get_dpex_config()
        
        if not best_molecule:
            logger.warning("[Exploit] No best molecule available")
            return []
        
        best_name = best_molecule['name']
        best_smiles = best_molecule.get('smiles')
        
        if not best_smiles:
            return []
        
        try:
            best_mol = Chem.MolFromSmiles(best_smiles)
            if not best_mol:
                return []
            
            best_fp = MORGAN_FP_GENERATOR.GetFingerprint(best_mol)
        except:
            return []
        
        candidates = []
        
        # Parse best molecule
        try:
            parts = best_name.split(':')
            if len(parts) not in [4, 5]:
                return []
            
            num_components = len(parts) - 2
            component_indices = list(range(2, 2 + num_components))
            
            # For each component, find highly similar components
            for comp_idx in component_indices:
                current_id = int(parts[comp_idx])
                
                # Get component pool
                if comp_idx == 2:
                    pool = self.molecule_manager.molecules_A
                elif comp_idx == 3:
                    pool = self.molecule_manager.molecules_B
                else:
                    pool = self.molecule_manager.molecules_C if hasattr(self.molecule_manager, 'molecules_C') else []
                
                if not pool:
                    continue
                
                # Find highly similar components
                similar_components = []
                for mol_id, smiles, _ in pool:
                    if mol_id == current_id:
                        continue
                    
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                            sim = DataStructs.TanimotoSimilarity(best_fp, fp)
                            if sim > config.exploit.min_similarity_threshold:
                                similar_components.append((mol_id, sim))
                    except:
                        pass
                
                # Sort by similarity
                similar_components.sort(key=lambda x: x[1], reverse=True)
                
                # Generate candidates by swapping with similar components
                for new_id, sim in similar_components[:config.exploit.num_similar_to_explore]:
                    neighbor_parts = parts.copy()
                    neighbor_parts[comp_idx] = str(new_id)
                    neighbor_name = ':'.join(neighbor_parts)
                    
                    if neighbor_name in state.generated_molecules:
                        continue
                    
                    try:
                        neighbor_smiles = get_smiles_from_reaction(neighbor_name)
                        if neighbor_smiles:
                            mol = Chem.MolFromSmiles(neighbor_smiles)
                            if mol is not None:
                                inchikey = generate_inchikey(neighbor_smiles)
                                if inchikey:
                                    candidates.append({
                                        'name': neighbor_name,
                                        'smiles': neighbor_smiles,
                                        'InChIKey': inchikey,
                                        'type': 'exploit',
                                        'parent': best_name,
                                        'similarity_to_best': sim,
                                    })
                                    # FIXED: Do NOT add to generated_molecules here
                                    
                                    if len(candidates) >= num_candidates:
                                        break
                    except:
                        pass
                
                if len(candidates) >= num_candidates:
                    break
        
        except Exception as e:
            logger.debug(f"[Exploit] Error generating candidates: {e}")
        
        if config.logging.verbose_generation:
            logger.info(f"[Exploit] Generated {len(candidates)} valid candidates via deep search")
        
        return candidates