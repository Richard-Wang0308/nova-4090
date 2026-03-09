"""
Optimized pool size configuration for DPEX_DJA + Boltz-2

Key insight: Larger pools enable better elite-based generation
with fewer total scoring calls for expensive Boltz-2 model.
"""

# ════════════════════════════════════════════════════════════════
# POOL SIZE CONFIGURATION
# ════════════════════════════════════════════════════════════════

class PoolSizeConfig:
    """Adaptive pool sizing based on scoring cost."""
    
    # Original (Light Psychic - ~5-10s per molecule)
    LIGHT_PSYCHIC_CONFIG = {
        'pop_A_size': 200,      # Global exploration pool
        'pop_B_size': 100,      # Local refinement pool
        'elite_pool_size': 100, # Top molecules to keep
        'candidates_per_round': 100,
        'scoring_cost_per_molecule': 5.0,  # seconds
    }
    
    # Proposed (Heavy Boltz-2 - ~30-60s per molecule)
    HEAVY_BOLTZ2_CONFIG = {
        'pop_A_size': 1200,     # Global exploration pool (6x larger)
        'pop_B_size': 1000,     # Local refinement pool (10x larger)
        'elite_pool_size': 700, # Top molecules to keep
        'candidates_per_round': 80,  # Fewer but higher quality
        'scoring_cost_per_molecule': 30.0,  # seconds
    }
    
    # Adaptive config (recommended)
    ADAPTIVE_CONFIG = {
        'pop_A_size': 1200,
        'pop_B_size': 1000,
        'elite_pool_size': 700,
        'candidates_per_round': 80,
        'scoring_cost_per_molecule': 30.0,
        
        # Additional parameters for Boltz-2
        'quality_filter_threshold': 0.55,  # Min similarity to elite
        'diversity_threshold': 0.65,        # Min diversity between candidates
        'exploit_mode_threshold': 3,        # Iterations before exploit mode
        'batch_size_for_scoring': 10,      # Boltz-2 batch size
    }


# ════════════════════════════════════════════════════════════════
# UPDATED DPEX STATE WITH LARGER POOLS
# ════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import List, Dict, Set
from collections import deque
import pandas as pd

@dataclass
class DPEXDJABoltzStateLargePool:
    """Enhanced DPEX_DJA state with larger pools for Boltz-2."""
    
    # ── Larger populations for better elite-based generation ──
    pop_A: List[Dict] = field(default_factory=list)      # 1200 molecules
    pop_B: List[Dict] = field(default_factory=list)      # 1000 molecules
    top_pool: pd.DataFrame = None                         # All scored molecules
    
    # ── Tabu lists (larger to avoid cycling) ──
    tabu_A: deque = field(default_factory=lambda: deque(maxlen=150))  # 2.5x larger
    tabu_B: deque = field(default_factory=lambda: deque(maxlen=150))
    tabu_C: deque = field(default_factory=lambda: deque(maxlen=150))
    
    # ── Tracking ──
    iteration: int = 0
    best_score: float = float('-inf')
    best_molecule: Dict = None
    score_history: deque = field(default_factory=lambda: deque(maxlen=20))  # Longer history
    no_improvement_counter: int = 0
    use_exploit_mode: bool = False
    
    # ── Diversity tracking ──
    seen_inchikeys: Set[str] = field(default_factory=set)
    generated_molecules: Set[str] = field(default_factory=set)
    candidate_quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # ── Elite component tracking (new) ──
    elite_components_A: Dict[int, List] = field(default_factory=dict)  # Component ID → [similar IDs]
    elite_components_B: Dict[int, List] = field(default_factory=dict)
    elite_components_C: Dict[int, List] = field(default_factory=dict)
    
    def get_pool_stats(self) -> Dict:
        """Get statistics about current pools."""
        return {
            'pop_A_size': len(self.pop_A),
            'pop_B_size': len(self.pop_B),
            'pop_A_avg_score': sum(m.get('score', 0) for m in self.pop_A) / max(len(self.pop_A), 1),
            'pop_B_avg_score': sum(m.get('score', 0) for m in self.pop_B) / max(len(self.pop_B), 1),
            'total_generated': len(self.generated_molecules),
            'unique_inchikeys': len(self.seen_inchikeys),
        }


# ════════════════════════════════════════════════════════════════
# UPDATED INITIALIZATION LOGIC
# ════════════════════════════════════════════════════════════════

def initialize_pools_large(
    molecules_df: pd.DataFrame,
    pop_A_size: int = 1200,
    pop_B_size: int = 1000
) -> tuple:
    """Initialize large pools from elite molecules."""
    
    if len(molecules_df) < pop_A_size + pop_B_size:
        # If not enough molecules, use what we have
        available = len(molecules_df)
        pop_A_size = min(pop_A_size, available // 2)
        pop_B_size = min(pop_B_size, available - pop_A_size)
    
    # Population A: Top molecules for exploration
    pop_A = molecules_df.head(pop_A_size).to_dict('records')
    for mol in pop_A:
        mol['score'] = mol.get('score', 0.0)
    
    # Population B: Slightly different selection for refinement
    # Use molecules ranked 100-1100 to avoid complete overlap with A
    pop_B_start = min(100, len(molecules_df) // 10)
    pop_B_end = min(pop_B_start + pop_B_size, len(molecules_df))
    pop_B = molecules_df.iloc[pop_B_start:pop_B_end].to_dict('records')
    for mol in pop_B:
        mol['score'] = mol.get('score', 0.0)
    
    return pop_A, pop_B


# ════════════════════════════════════════════════════════════════
# UPDATED DJA GENERATOR FOR LARGER POOLS
# ════════════════════════════════════════════════════════════════

class DJAGeneratorLargePool:
    """Enhanced DJA generator with larger pool support."""
    
    def __init__(self, molecule_manager):
        self.molecule_manager = molecule_manager
        self.component_cache = {}  # Cache for component lookups
    
    def generate_candidates(
        self,
        pop_A: List[Dict],
        num_candidates: int,
        state,
        use_elite_diversity: bool = True
    ) -> List[Dict]:
        """
        Generate candidates with elite diversity awareness.
        
        With larger pools, we can be more selective about which
        best/worst solutions to use for DJA updates.
        """
        
        if not pop_A:
            return []
        
        candidates = []
        attempts = 0
        max_attempts = num_candidates * 5
        
        # With large pools, use top 10% and bottom 10% for diversity
        if use_elite_diversity and len(pop_A) > 100:
            sorted_pop = sorted(pop_A, key=lambda x: x.get('score', float('-inf')), reverse=True)
            elite_best = sorted_pop[:max(10, len(sorted_pop) // 10)]
            elite_worst = sorted_pop[-max(10, len(sorted_pop) // 10):]
        else:
            elite_best = [max(pop_A, key=lambda x: x.get('score', float('-inf')))]
            elite_worst = [min(pop_A, key=lambda x: x.get('score', float('-inf')))]
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            
            # Select random parent from pop_A
            parent = random.choice(pop_A)
            parent_name = parent['name']
            
            try:
                parts = parent_name.split(':')
                if len(parts) not in [4, 5]:
                    continue
                
                num_components = len(parts) - 2
                component_indices = list(range(2, 2 + num_components))
                
                # DJA update with elite diversity
                offspring_parts = parts.copy()
                
                for comp_idx in component_indices:
                    # 60% chance: adopt from elite_best (increased from 50%)
                    if random.random() < 0.6:
                        best_sol = random.choice(elite_best)
                        offspring_parts[comp_idx] = best_sol['name'].split(':')[comp_idx]
                    
                    # If matches elite_worst, escape
                    if offspring_parts[comp_idx] in [w['name'].split(':')[comp_idx] for w in elite_worst]:
                        if random.random() < 0.5:
                            # Random component
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
                
                if offspring_name in state.generated_molecules:
                    continue
                
                try:
                    from combinatorial_db.reactions import get_smiles_from_reaction
                    from molecules_base import generate_inchikey
                    from rdkit import Chem
                    
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
                                state.generated_molecules.add(offspring_name)
                except:
                    pass
            
            except Exception as e:
                continue
        
        return candidates


# ════════════════════════════════════════════════════════════════
# UPDATED POPULATION REFRESH LOGIC
# ════════════════════════════════════════════════════════════════

def refresh_populations_large(
    molecules_df: pd.DataFrame,
    dpex_state,
    pop_A_size: int = 1200,
    pop_B_size: int = 1000
) -> None:
    """
    Refresh large pools with intelligent selection.
    
    Strategy:
    1. Keep top pop_A_size in pop_A
    2. Keep molecules ranked 100-1100 in pop_B (to avoid overlap)
    3. Preserve tabu lists to avoid cycling
    """
    
    if molecules_df.empty:
        return
    
    # Refresh pop_A with top molecules
    new_pop_A = molecules_df.head(pop_A_size).to_dict('records')
    for mol in new_pop_A:
        mol['score'] = mol.get('score', 0.0)
    
    # Merge with existing (keep good solutions)
    for mol in dpex_state.pop_A:
        if mol['name'] not in [m['name'] for m in new_pop_A]:
            # Check if it's still in top molecules
            if mol['name'] in molecules_df['name'].values:
                new_pop_A.append(mol)
    
    # Keep only top pop_A_size
    new_pop_A = sorted(
        new_pop_A,
        key=lambda x: x.get('score', float('-inf')),
        reverse=True
    )[:pop_A_size]
    
    dpex_state.pop_A = new_pop_A
    
    # Refresh pop_B with staggered selection
    pop_B_start = min(100, len(molecules_df) // 10)
    pop_B_end = min(pop_B_start + pop_B_size, len(molecules_df))
    new_pop_B = molecules_df.iloc[pop_B_start:pop_B_end].to_dict('records')
    for mol in new_pop_B:
        mol['score'] = mol.get('score', 0.0)
    
    dpex_state.pop_B = new_pop_B


# ════════════════════════════════════════════════════════════════
# UPDATED BUDGET CONTROLLER FOR LARGER POOLS
# ════════════════════════════════════════════════════════════════

class AdaptiveBudgetControllerLargePool:
    """Budget controller optimized for large pools + expensive scoring."""
    
    def __init__(self, scoring_cost_per_molecule: float = 30.0):
        self.scoring_cost = scoring_cost_per_molecule
        self.iteration = 0
        self.improvement_history = []
        self.candidate_budget_history = []
    
    def get_candidate_budget(
        self,
        iteration: int,
        improvement_rate: float = None,
        wall_clock_remaining: float = None
    ) -> tuple:
        """
        Compute budget with large pools in mind.
        
        Key insight: With larger pools providing better elite solutions,
        we can afford to be more selective (fewer candidates) while
        maintaining quality.
        """
        
        budget_info = {
            'iteration': iteration,
            'improvement_rate': improvement_rate,
            'wall_clock_remaining': wall_clock_remaining,
            'reasoning': '',
        }
        
        # Phase 1: Seed (iteration 1-2)
        if iteration <= 2:
            num_candidates = 80  # Start conservative
            budget_info['reasoning'] = 'Seed phase with large pools'
            return num_candidates, budget_info
        
        # Phase 2: Early exploration (iterations 3-10)
        if iteration <= 10:
            num_candidates = 100
            budget_info['reasoning'] = 'Early exploration with large pools'
            return num_candidates, budget_info
        
        # Phase 3: Adaptive (iteration 11+)
        if improvement_rate is None:
            num_candidates = 80
            budget_info['reasoning'] = 'Default with large pools'
            return num_candidates, budget_info
        
        # Adaptive logic
        if improvement_rate > 0.01:  # >1% improvement
            num_candidates = 100
            budget_info['reasoning'] = 'High improvement: maintain exploration'
        elif improvement_rate > 0.001:  # 0.1-1% improvement
            num_candidates = 80
            budget_info['reasoning'] = 'Medium improvement: balanced search'
        else:  # <0.1% improvement
            num_candidates = 60
            budget_info['reasoning'] = 'Low improvement: focus on refinement'
        
        # Time pressure adjustment
        if wall_clock_remaining and wall_clock_remaining < 300:
            num_candidates = max(30, int(num_candidates * 0.5))
            budget_info['reasoning'] += ' [TIME PRESSURE]'
        
        self.candidate_budget_history.append(num_candidates)
        return num_candidates, budget_info
    
    def estimate_total_time_to_top_1(
        self,
        current_iteration: int,
        improvement_rate: float,
        current_best_rank: int
    ) -> float:
        """
        Estimate time to find top 1 molecule.
        
        With large pools:
        - Better elite diversity → faster convergence
        - Fewer total iterations needed
        """
        
        if improvement_rate <= 0:
            return float('inf')
        
        # Estimate iterations remaining
        # With large pools, convergence is faster
        iterations_remaining = max(10, int(np.log(current_best_rank) / np.log(1 + improvement_rate * 10)))
        
        # Average candidates per iteration
        avg_candidates = np.mean(self.candidate_budget_history) if self.candidate_budget_history else 80
        
        # Total scoring calls
        total_scoring_calls = iterations_remaining * avg_candidates
        
        # Time estimate
        estimated_time_seconds = total_scoring_calls * self.scoring_cost
        
        return estimated_time_seconds