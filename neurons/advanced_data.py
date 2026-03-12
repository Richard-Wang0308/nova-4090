"""
Advanced Molecule Generation System with Intelligent Genetic Algorithms
Integrated with existing codebase
"""

import os
import sys
import random
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import time
import sqlite3
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from dataclasses import dataclass
from collections import defaultdict

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 5
STARTING_EPOCH = 21353
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'mols.csv')
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")

from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    compute_maccs_entropy,
    molecule_unique_for_protein_hf,
    find_chemically_identical,
    is_reaction_allowed,
    contains_atom_type
)
from utils.molecules import molecule_unique_for_protein_hf
from molecules_base import generate_inchikey, get_molecules_by_role
from combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info

BOLTZ_AVAILABLE = False
BoltzWrapper = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ComponentScore:
    """Statistics for a component."""
    mean: float
    std: float
    count: int
    max: float
    min: float


@dataclass
class GenerationStrategy:
    """Strategy parameters for molecule generation."""
    crossover_ratio: float
    mutation_ratio: float
    local_search_ratio: float
    pool_size: int
    use_guided_ops: bool
    phase: str  # 'exploration', 'exploitation', 'refinement'


# ============================================================================
# Validation Functions
# ============================================================================

def validate_molecule_smiles(molecule_name: str, smiles: str) -> Tuple[bool, str]:
    """Validate SMILES string with RDKit."""
    if not smiles:
        return False, "No SMILES provided"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"RDKit cannot parse SMILES: {smiles}"
        return True, ""
    except Exception as e:
        return False, f"RDKit parsing error: {str(e)}"


def validate_molecule_heavy_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate heavy atom count."""
    try:
        heavy_atom_count = get_heavy_atom_count(smiles)
        min_atoms = 20
        max_atoms = 25
        
        if heavy_atom_count < min_atoms:
            return False, f"Insufficient heavy atoms: {heavy_atom_count} < {min_atoms}"
        if heavy_atom_count > max_atoms:
            return False, f"Too many heavy atoms: {heavy_atom_count} > {max_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Heavy atom count error: {str(e)}"


def validate_molecule_banned_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate molecule doesn't contain banned atom types."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for banned atom check"
        
        banned_atoms = config.get('banned_atom_types', [])
        if not banned_atoms:
            return True, ""
        
        if contains_atom_type(mol, banned_atoms):
            return False, f"Contains banned atom types: {banned_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Banned atom check error: {str(e)}"


def validate_molecule_rotatable_bonds(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate rotatable bonds are within acceptable range."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for rotatable bonds check"
        
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        min_bonds = config.get('min_rotatable_bonds', 1)
        max_bonds = config.get('max_rotatable_bonds', 10)
        
        if num_rotatable_bonds < min_bonds or num_rotatable_bonds > max_bonds:
            return False, f"Rotatable bonds out of range: {num_rotatable_bonds} (expected {min_bonds}-{max_bonds})"
        return True, ""
    except Exception as e:
        return False, f"Rotatable bonds check error: {str(e)}"


async def validate_molecule_huggingface_unique(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str
) -> Tuple[bool, str]:
    """Validate molecule is unique (NOT in HuggingFace dataset)."""
    if not state.get('current_challenge_targets'):
        return False, "No target proteins available"
    
    primary_target = state['current_challenge_targets'][0]
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            return False, f"Molecule already in HuggingFace dataset for {primary_target}"
        return True, ""
    except Exception as e:
        return False, f"HuggingFace uniqueness check error: {str(e)}"


async def check_molecule_unique(state: Dict[str, Any], molecule_name: str, smiles: str) -> bool:
    """
    Check if molecule is unique for target protein (NOT in HuggingFace dataset).
    
    Returns:
        True if molecule is NOT in HuggingFace (i.e., it's unique/new)
        False if molecule IS in HuggingFace (i.e., it's already known)
    """
    if not state.get('current_challenge_targets'):
        print("WARNING: No target proteins available")
        return False
    
    primary_target = state['current_challenge_targets'][0]
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            return False
        
        return True
    except Exception as e:
        print(f"Error checking uniqueness: {e}")
        return False


async def validate_molecule_complete(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any] = None
) -> Tuple[bool, List[str]]:
    """Perform complete validation on a molecule."""
    if config is None:
        config = state.get('config', {})
    
    errors = []
    
    # 1. SMILES validity
    is_valid, error_msg = validate_molecule_smiles(molecule_name, smiles)
    if not is_valid:
        errors.append(f"[SMILES] {error_msg}")
        return False, errors
    
    # 2. Heavy atom count
    is_valid, error_msg = validate_molecule_heavy_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[HEAVY_ATOMS] {error_msg}")
    
    # 3. Banned atoms
    is_valid, error_msg = validate_molecule_banned_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[BANNED_ATOMS] {error_msg}")
    
    # 4. Rotatable bonds
    is_valid, error_msg = validate_molecule_rotatable_bonds(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[ROTATABLE_BONDS] {error_msg}")
    
    # 5. HuggingFace uniqueness
    is_valid, error_msg = await validate_molecule_huggingface_unique(state, molecule_name, smiles)
    if not is_valid:
        errors.append(f"[HF_UNIQUE] {error_msg}")
    
    return len(errors) == 0, errors


# ============================================================================
# Smart Parent Selection
# ============================================================================

class SmartParentSelector:
    """
    Intelligent parent selection based on:
    - Score quality
    - Component performance analysis
    - Chemical diversity
    - Tournament selection
    """
    
    def __init__(self, top_molecules_df: pd.DataFrame):
        self.molecules = top_molecules_df
        self.component_scores = self._analyze_component_scores()
        self.fingerprints = self._compute_fingerprints()
        self.selected_parents = []
        
        print(
            f"📊 Component Analysis Complete:\n"
            f"   - Role A: {len(self.component_scores['A'])} unique components\n"
            f"   - Role B: {len(self.component_scores['B'])} unique components\n"
            f"   - Role C: {len(self.component_scores['C'])} unique components"
        )
    
    def _analyze_component_scores(self) -> Dict[str, Dict[int, ComponentScore]]:
        """
        Analyze which components lead to high scores.
        
        Returns:
            Dict mapping role -> component_id -> ComponentScore
        """
        component_scores = {'A': defaultdict(list), 'B': defaultdict(list), 'C': defaultdict(list)}
        
        for _, row in self.molecules.iterrows():
            parts = row['name'].split(':')
            score = row.get('score')
            
            if score is None or pd.isna(score):
                continue
            
            try:
                # Component A (position 2)
                comp_a = int(parts[2])
                component_scores['A'][comp_a].append(score)
                
                # Component B (position 3)
                comp_b = int(parts[3])
                component_scores['B'][comp_b].append(score)
                
                # Component C (position 4, if exists)
                if len(parts) > 4:
                    comp_c = int(parts[4])
                    component_scores['C'][comp_c].append(score)
            except (IndexError, ValueError) as e:
                continue
        
        # Calculate statistics for each component
        result = {'A': {}, 'B': {}, 'C': {}}
        for role in component_scores:
            for comp_id, scores in component_scores[role].items():
                if scores:
                    result[role][comp_id] = ComponentScore(
                        mean=float(np.mean(scores)),
                        std=float(np.std(scores)),
                        count=len(scores),
                        max=float(max(scores)),
                        min=float(min(scores))
                    )
        
        return result
    
    def _compute_fingerprints(self) -> Dict[str, np.ndarray]:
        """
        Compute Morgan fingerprints for diversity calculation.
        
        Returns:
            Dict mapping molecule_name -> fingerprint array
        """
        fingerprints = {}
        
        for _, row in self.molecules.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fingerprints[row['name']] = np.array(fp)
            except Exception as e:
                continue
        
        print(f"   - Computed {len(fingerprints)} molecular fingerprints")
        return fingerprints
    
    def _tanimoto_distance(self, mol1_name: str, mol2_name: str) -> float:
        """
        Calculate Tanimoto distance between two molecules.
        
        Returns:
            Distance (0 = identical, 1 = completely different)
        """
        if mol1_name not in self.fingerprints or mol2_name not in self.fingerprints:
            return 0.5  # Unknown similarity
        
        fp1 = self.fingerprints[mol1_name]
        fp2 = self.fingerprints[mol2_name]
        
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
        
        tanimoto_similarity = intersection / union
        return 1.0 - tanimoto_similarity  # Convert to distance
    
    def select_parents_tournament(
        self, 
        tournament_size: int = 5,
        diversity_weight: float = 0.3
    ) -> Tuple[str, str]:
        """
        Tournament selection with diversity bonus.
        
        Args:
            tournament_size: Number of candidates in tournament
            diversity_weight: Weight for diversity vs score (0-1)
            
        Returns:
            Tuple of (parent1_name, parent2_name)
        """
        if len(self.molecules) < tournament_size:
            tournament_size = len(self.molecules)
        
        # Select tournament candidates
        candidates = self.molecules.sample(n=tournament_size)
        
        # Score each candidate (base score + diversity bonus)
        scored_candidates = []
        for _, cand in candidates.iterrows():
            base_score = cand.get('score', 0)
            if base_score is None or pd.isna(base_score):
                base_score = 0
            
            # Diversity bonus: average distance to already selected parents
            diversity_bonus = 0
            if self.selected_parents:
                distances = [
                    self._tanimoto_distance(cand['name'], p)
                    for p in self.selected_parents[-5:]  # Only consider recent 5
                ]
                diversity_bonus = np.mean(distances) if distances else 0
            
            total_score = base_score + diversity_weight * diversity_bonus
            scored_candidates.append((cand['name'], total_score, base_score))
        
        # Select best two
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        parent1 = scored_candidates[0][0]
        parent2 = scored_candidates[1][0] if len(scored_candidates) > 1 else parent1
        
        # Track selected parents
        self.selected_parents.extend([parent1, parent2])
        
        return parent1, parent2
    
    def select_parents_component_aware(self) -> Tuple[str, str]:
        """
        Select parents with complementary high-scoring components.
        
        Strategy:
        - Find molecules with high-scoring component A
        - Find molecules with high-scoring component B
        - Select one from each group to combine good components
        
        Returns:
            Tuple of (parent1_name, parent2_name)
        """
        # Calculate score thresholds (75th percentile)
        score_threshold = self.molecules['score'].quantile(0.75)
        
        high_a_mols = []
        high_b_mols = []
        
        for _, row in self.molecules.iterrows():
            parts = row['name'].split(':')
            score = row.get('score')
            
            if score is None or pd.isna(score):
                continue
            
            try:
                # Check component A
                comp_a = int(parts[2])
                if comp_a in self.component_scores['A']:
                    comp_a_score = self.component_scores['A'][comp_a].mean
                    if comp_a_score >= score_threshold:
                        high_a_mols.append(row['name'])
                
                # Check component B
                comp_b = int(parts[3])
                if comp_b in self.component_scores['B']:
                    comp_b_score = self.component_scores['B'][comp_b].mean
                    if comp_b_score >= score_threshold:
                        high_b_mols.append(row['name'])
            except (IndexError, ValueError):
                continue
        
        # Select one from each group
        if high_a_mols and high_b_mols:
            parent1 = random.choice(high_a_mols)
            parent2 = random.choice(high_b_mols)
            return parent1, parent2
        
        # Fallback to tournament if not enough high-scoring components
        return self.select_parents_tournament()
    
    def get_top_components(self, role: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get top N components by average score for a given role.
        
        Args:
            role: 'A', 'B', or 'C'
            top_n: Number of top components to return
            
        Returns:
            List of (component_id, mean_score) tuples
        """
        if role not in self.component_scores:
            return []
        
        components = [
            (comp_id, stats.mean)
            for comp_id, stats in self.component_scores[role].items()
        ]
        components.sort(key=lambda x: x[1], reverse=True)
        
        return components[:top_n]


# ============================================================================
# Advanced Genetic Algorithm Operator
# ============================================================================

class AdvancedGeneticOperator:
    """
    Enhanced genetic algorithm with:
    - Multi-component crossover
    - Guided mutation
    - Local search
    - Component-preserving operations
    """
    
    def __init__(self, rxn_id: int, db_path: str):
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()
        self.component_scores: Dict[str, Dict[int, ComponentScore]] = {}
        
        # Load reaction info and component pools
        self.reaction_info = get_reaction_info(rxn_id, db_path)
        if self.reaction_info:
            self.smarts, self.roleA, self.roleB, self.roleC = self.reaction_info
            self.is_three_component = self.roleC is not None and self.roleC != 0
            
            # Load component pools
            self.molecules_A = get_molecules_by_role(self.roleA, db_path)
            self.molecules_B = get_molecules_by_role(self.roleB, db_path)
            self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
            
            # Extract component IDs
            self.component_ids_A = [mol[0] for mol in self.molecules_A] if self.molecules_A else []
            self.component_ids_B = [mol[0] for mol in self.molecules_B] if self.molecules_B else []
            self.component_ids_C = [mol[0] for mol in self.molecules_C] if self.molecules_C else []
            
            print(
                f"🧬 GA Operator initialized: "
                f"{len(self.component_ids_A)} A, {len(self.component_ids_B)} B"
                f"{f', {len(self.component_ids_C)} C' if self.is_three_component else ''} components"
            )
        else:
            print(f"Could not load reaction info for rxn_id {rxn_id}")
            self.reaction_info = None
            self.component_ids_A = []
            self.component_ids_B = []
            self.component_ids_C = []
            self.is_three_component = False
    
    def set_component_scores(self, component_scores: Dict[str, Dict[int, ComponentScore]]):
        """Set component scores from parent selector."""
        self.component_scores = component_scores
    
    # ------------------------------------------------------------------------
    # Crossover Operations
    # ------------------------------------------------------------------------
    
    def crossover_single_component(
        self, 
        mol_name_1: str, 
        mol_name_2: str
    ) -> Optional[str]:
        """
        Original single-component crossover (swap one random component).
        """
        try:
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            if (parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                return None
            
            if len(parts1) != len(parts2):
                return None
            
            if len(parts1) not in [4, 5]:
                return None
            
            try:
                rxn_id_1 = int(parts1[1])
                rxn_id_2 = int(parts2[1])
                if rxn_id_1 != self.rxn_id or rxn_id_2 != self.rxn_id:
                    return None
            except (ValueError, IndexError):
                return None
            
            # Randomly select which component to swap
            num_components = len(parts1) - 2
            component_indices = list(range(2, 2 + num_components))
            swap_idx = random.choice(component_indices)
            
            # Create offspring
            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)
            
            # Check if already generated
            if offspring_name in self.generated_molecule_names:
                return None
            
            # Validate
            offspring_smiles = get_smiles_from_reaction(offspring_name)
            if offspring_smiles:
                mol = Chem.MolFromSmiles(offspring_smiles)
                if mol is not None:
                    self.generated_molecule_names.add(offspring_name)
                    return offspring_name
            
            return None
        
        except Exception as e:
            return None
    
    def crossover_multi_component(
        self, 
        mol_name_1: str, 
        mol_name_2: str,
        swap_probability: float = 0.5
    ) -> Optional[str]:
        """
        Multi-component crossover: each component has independent swap probability.
        """
        try:
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            if (parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                return None
            
            if len(parts1) != len(parts2):
                return None
            
            if len(parts1) not in [4, 5]:
                return None
            
            try:
                rxn_id_1 = int(parts1[1])
                rxn_id_2 = int(parts2[1])
                if rxn_id_1 != self.rxn_id or rxn_id_2 != self.rxn_id:
                    return None
            except (ValueError, IndexError):
                return None
            
            offspring_parts = parts1.copy()
            
            # For each component position, randomly decide to swap
            num_components = len(parts1) - 2
            swapped_any = False
            
            for i in range(2, 2 + num_components):
                if random.random() < swap_probability:
                    offspring_parts[i] = parts2[i]
                    swapped_any = True
            
            # Ensure at least one component was swapped
            if not swapped_any:
                swap_idx = random.choice(range(2, 2 + num_components))
                offspring_parts[swap_idx] = parts2[swap_idx]
            
            offspring_name = ':'.join(offspring_parts)
            
            # Check if identical to parents
            if offspring_name in [mol_name_1, mol_name_2]:
                return None
            
            # Check if already generated
            if offspring_name in self.generated_molecule_names:
                return None
            
            # Validate
            offspring_smiles = get_smiles_from_reaction(offspring_name)
            if offspring_smiles:
                mol = Chem.MolFromSmiles(offspring_smiles)
                if mol is not None:
                    self.generated_molecule_names.add(offspring_name)
                    return offspring_name
            
            return None
        
        except Exception as e:
            return None
    
    def crossover_component_preserving(
        self,
        mol_name_1: str,
        mol_name_2: str,
        preserve_probability: float = 0.7
    ) -> Optional[str]:
        """
        Crossover that preferentially preserves high-scoring components.
        """
        try:
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            if (parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                return None
            
            if len(parts1) != len(parts2):
                return None
            
            if len(parts1) not in [4, 5]:
                return None
            
            # If no component scores available, fall back to simple crossover
            if not self.component_scores:
                return self.crossover_single_component(mol_name_1, mol_name_2)
            
            offspring_parts = ['rxn', parts1[1]]
            
            # For each component position, choose the better one (or random)
            num_components = len(parts1) - 2
            for i in range(num_components):
                pos = i + 2
                comp1 = int(parts1[pos])
                comp2 = int(parts2[pos])
                
                role = ['A', 'B', 'C'][i]
                
                # Get component scores
                score1 = 0
                score2 = 0
                if role in self.component_scores:
                    if comp1 in self.component_scores[role]:
                        score1 = self.component_scores[role][comp1].mean
                    if comp2 in self.component_scores[role]:
                        score2 = self.component_scores[role][comp2].mean
                
                # Choose component
                if random.random() < preserve_probability:
                    # Choose better component
                    chosen = comp1 if score1 >= score2 else comp2
                else:
                    # Choose randomly for diversity
                    chosen = random.choice([comp1, comp2])
                
                offspring_parts.append(str(chosen))
            
            offspring_name = ':'.join(offspring_parts)
            
            # Check if identical to parents
            if offspring_name in [mol_name_1, mol_name_2]:
                return None
            
            # Check if already generated
            if offspring_name in self.generated_molecule_names:
                return None
            
            # Validate
            offspring_smiles = get_smiles_from_reaction(offspring_name)
            if offspring_smiles:
                mol = Chem.MolFromSmiles(offspring_smiles)
                if mol is not None:
                    self.generated_molecule_names.add(offspring_name)
                    return offspring_name
            
            return None
        
        except Exception as e:
            return None
    
    # ------------------------------------------------------------------------
    # Mutation Operations
    # ------------------------------------------------------------------------
    
    def mutate_random(self, mol_name: str) -> Optional[str]:
        """
        Original random mutation (replace one component with random alternative).
        """
        try:
            if not self.reaction_info:
                return None
            
            parts = mol_name.split(':')
            
            if parts[0] != 'rxn':
                return None
            
            if len(parts) not in [4, 5]:
                return None
            
            try:
                rxn_id = int(parts[1])
                if rxn_id != self.rxn_id:
                    return None
            except (ValueError, IndexError):
                return None
            
            # Select component to mutate
            num_components = len(parts) - 2
            component_indices = list(range(2, 2 + num_components))
            mutate_idx = random.choice(component_indices)
            component_position = mutate_idx - 2
            
            # Get component pool
            if component_position == 0:
                component_pool = self.component_ids_A
            elif component_position == 1:
                component_pool = self.component_ids_B
            elif component_position == 2 and self.is_three_component:
                component_pool = self.component_ids_C
            else:
                return None
            
            if not component_pool:
                return None
            
            current_component_id = int(parts[mutate_idx])
            available_components = [cid for cid in component_pool if cid != current_component_id]
            
            if not available_components:
                return None
            
            new_component_id = random.choice(available_components)
            
            # Create mutated molecule
            mutated_parts = parts.copy()
            mutated_parts[mutate_idx] = str(new_component_id)
            mutated_name = ':'.join(mutated_parts)
            
            # Check if already generated
            if mutated_name in self.generated_molecule_names:
                return None
            
            # Validate
            mutated_smiles = get_smiles_from_reaction(mutated_name)
            if mutated_smiles:
                mol = Chem.MolFromSmiles(mutated_smiles)
                if mol is not None:
                    self.generated_molecule_names.add(mutated_name)
                    return mutated_name
            
            return None
        
        except Exception as e:
            return None
    
    def mutate_guided(
        self, 
        mol_name: str,
        exploration_bonus: float = 0.1
    ) -> Optional[str]:
        """
        Guided mutation that preferentially selects high-scoring components.
        """
        try:
            if not self.reaction_info:
                return None
            
            # If no component scores, fall back to random mutation
            if not self.component_scores:
                return self.mutate_random(mol_name)
            
            parts = mol_name.split(':')
            
            if parts[0] != 'rxn':
                return None
            
            if len(parts) not in [4, 5]:
                return None
            
            try:
                rxn_id = int(parts[1])
                if rxn_id != self.rxn_id:
                    return None
            except (ValueError, IndexError):
                return None
            
            # Select component to mutate
            num_components = len(parts) - 2
            mutate_idx = random.choice(range(2, 2 + num_components))
            component_position = mutate_idx - 2
            
            # Get role and component pool
            role = ['A', 'B', 'C'][component_position]
            
            if component_position == 0:
                component_pool = self.component_ids_A
            elif component_position == 1:
                component_pool = self.component_ids_B
            else:
                component_pool = self.component_ids_C
            
            if not component_pool:
                return None
            
            current_component_id = int(parts[mutate_idx])
            available = [c for c in component_pool if c != current_component_id]
            
            if not available:
                return None
            
            # Calculate weights for each available component
            component_weights = []
            for comp_id in available:
                if comp_id in self.component_scores.get(role, {}):
                    score_info = self.component_scores[role][comp_id]
                    # Weight = mean score + exploration bonus / (count + 1)
                    weight = score_info.mean + exploration_bonus / (score_info.count + 1)
                else:
                    # Unknown component: give moderate exploration bonus
                    weight = 0.5
                
                component_weights.append(max(weight, 0.01))  # Ensure positive weights
            
            # Normalize weights to probabilities
            total_weight = sum(component_weights)
            if total_weight == 0:
                probabilities = [1/len(available)] * len(available)
            else:
                probabilities = [w/total_weight for w in component_weights]
            
            # Select component using weighted probabilities
            new_component_id = np.random.choice(available, p=probabilities)
            
            # Create mutated molecule
            mutated_parts = parts.copy()
            mutated_parts[mutate_idx] = str(new_component_id)
            mutated_name = ':'.join(mutated_parts)
            
            # Check if already generated
            if mutated_name in self.generated_molecule_names:
                return None
            
            # Validate
            mutated_smiles = get_smiles_from_reaction(mutated_name)
            if mutated_smiles:
                mol = Chem.MolFromSmiles(mutated_smiles)
                if mol is not None:
                    self.generated_molecule_names.add(mutated_name)
                    return mutated_name
            
            return None
        
        except Exception as e:
            return None
    
    def mutate_local_search(
        self,
        mol_name: str,
        num_neighbors: int = 5
    ) -> List[str]:
        """
        Generate multiple mutations of a single molecule (local search).
        """
        neighbors = []
        
        try:
            parts = mol_name.split(':')
            
            if parts[0] != 'rxn':
                return neighbors
            
            if len(parts) not in [4, 5]:
                return neighbors
            
            num_components = len(parts) - 2
            
            for _ in range(num_neighbors):
                # Randomly select component to mutate
                mutate_idx = random.choice(range(2, 2 + num_components))
                component_position = mutate_idx - 2
                
                # Get component pool
                if component_position == 0:
                    component_pool = self.component_ids_A
                elif component_position == 1:
                    component_pool = self.component_ids_B
                else:
                    component_pool = self.component_ids_C
                
                if not component_pool:
                    continue
                
                current_id = int(parts[mutate_idx])
                available = [c for c in component_pool if c != current_id]
                
                if not available:
                    continue
                
                new_id = random.choice(available)
                
                neighbor_parts = parts.copy()
                neighbor_parts[mutate_idx] = str(new_id)
                neighbor_name = ':'.join(neighbor_parts)
                
                # Check if already generated
                if neighbor_name in self.generated_molecule_names:
                    continue
                
                # Validate
                try:
                    neighbor_smiles = get_smiles_from_reaction(neighbor_name)
                    if neighbor_smiles:
                        mol = Chem.MolFromSmiles(neighbor_smiles)
                        if mol is not None:
                            neighbors.append(neighbor_name)
                            self.generated_molecule_names.add(neighbor_name)
                except Exception as e:
                    continue
        
        except Exception as e:
            pass
        
        return neighbors


# ============================================================================
# Adaptive Generation Strategy Manager
# ============================================================================

class AdaptiveStrategyManager:
    """
    Dynamically adjust generation strategy based on progress.
    
    Phases:
    - Exploration: Broad search, high mutation rate
    - Exploitation: Focus on promising regions, balanced operations
    - Refinement: Local search around best molecules
    """
    
    def __init__(self):
        self.generation_history: List[Dict[str, float]] = []
        self.current_phase = 'exploration'
    
    def update_strategy(
        self,
        new_molecules: List[Dict],
        best_score_so_far: float
    ) -> GenerationStrategy:
        """
        Determine optimal strategy based on generation quality.
        """
        # Analyze new molecules
        scores = [
            m.get('boltz_score', 0) 
            for m in new_molecules 
            if m.get('boltz_score') is not None
        ]
        
        if not scores:
            # No valid scores: increase exploration
            print("📊 No valid scores, increasing exploration")
            return GenerationStrategy(
                crossover_ratio=0.2,
                mutation_ratio=0.8,
                local_search_ratio=0.0,
                pool_size=200,
                use_guided_ops=False,
                phase='exploration'
            )
        
        avg_score = np.mean(scores)
        max_score = max(scores)
        improvement = max_score - best_score_so_far
        
        self.generation_history.append({
            'avg_score': avg_score,
            'max_score': max_score,
            'improvement': improvement
        })
        
        # Determine phase based on recent improvements
        if len(self.generation_history) < 3:
            phase = 'exploration'
        else:
            recent_improvements = [
                h['improvement'] for h in self.generation_history[-3:]
            ]
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement > 0.01:  # Significant improvement
                phase = 'exploitation'
            elif avg_improvement > 0:  # Small improvement
                phase = 'refinement'
            else:  # No improvement
                phase = 'exploration'
        
        self.current_phase = phase
        
        # Strategy based on phase
        if phase == 'exploration':
            strategy = GenerationStrategy(
                crossover_ratio=0.4,
                mutation_ratio=0.6,
                local_search_ratio=0.0,
                pool_size=200,
                use_guided_ops=False,
                phase='exploration'
            )
            print(
                f"📊 Phase: EXPLORATION (avg_score: {avg_score:.6f}, "
                f"improvement: {improvement:.6f})"
            )
        
        elif phase == 'exploitation':
            strategy = GenerationStrategy(
                crossover_ratio=0.6,
                mutation_ratio=0.3,
                local_search_ratio=0.1,
                pool_size=100,
                use_guided_ops=True,
                phase='exploitation'
            )
            print(
                f"📊 Phase: EXPLOITATION (avg_score: {avg_score:.6f}, "
                f"improvement: {improvement:.6f})"
            )
        
        else:  # refinement
            strategy = GenerationStrategy(
                crossover_ratio=0.5,
                mutation_ratio=0.2,
                local_search_ratio=0.3,
                pool_size=50,
                use_guided_ops=True,
                phase='refinement'
            )
            print(
                f"📊 Phase: REFINEMENT (avg_score: {avg_score:.6f}, "
                f"improvement: {improvement:.6f})"
            )
        
        return strategy


# ============================================================================
# Main Advanced Generation Function
# ============================================================================

async def generate_molecules_adaptive(
    state: Dict[str, Any],
    top_200_df: pd.DataFrame,
    desired_count: int = 100,
    max_rounds: int = 10
) -> List[Dict[str, Any]]:
    """
    Adaptive molecule generation with intelligent strategies.
    """
    if top_200_df.empty:
        print("Top 200 DataFrame is empty")
        return []
    
    print(
        f"\n{'='*70}\n"
        f"🧬 ADAPTIVE MOLECULE GENERATION\n"
        f"{'='*70}\n"
        f"Target: {desired_count} unique molecules\n"
        f"Pool size: {len(top_200_df)} molecules\n"
        f"Max rounds: {max_rounds}\n"
        f"{'='*70}"
    )
    
    # Initialize components
    parent_selector = SmartParentSelector(top_200_df)
    ga_operator = AdvancedGeneticOperator(HARDCODED_RXN_ID, DB_PATH)
    strategy_manager = AdaptiveStrategyManager()
    
    # Share component scores with GA operator
    ga_operator.set_component_scores(parent_selector.component_scores)
    
    # Track results
    unique_molecules = []
    best_score_so_far = float('-inf')
    
    # Get or initialize generated molecules tracking
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    # Generation statistics
    total_stats = {
        'crossovers_attempted': 0,
        'crossovers_successful': 0,
        'mutations_attempted': 0,
        'mutations_successful': 0,
        'local_searches_attempted': 0,
        'local_searches_successful': 0,
    }
    
    # Main generation loop
    for round_num in range(max_rounds):
        if len(unique_molecules) >= desired_count:
            print(f"✅ Target reached: {len(unique_molecules)}/{desired_count} molecules")
            break
        
        print(
            f"\n{'─'*70}\n"
            f"🔄 Round {round_num + 1}/{max_rounds}\n"
            f"{'─'*70}"
        )
        
        # Get adaptive strategy
        strategy = strategy_manager.update_strategy(
            unique_molecules,
            best_score_so_far
        )
        
        print(
            f"Strategy: Phase={strategy.phase.upper()}, "
            f"Pool={strategy.pool_size}, "
            f"Crossover={strategy.crossover_ratio:.1%}, "
            f"Mutation={strategy.mutation_ratio:.1%}, "
            f"LocalSearch={strategy.local_search_ratio:.1%}"
        )
        
        # Select pool based on strategy
        pool_molecules = top_200_df.head(strategy.pool_size)
        pool_names = pool_molecules['name'].tolist()
        
        # Calculate operation counts
        remaining = desired_count - len(unique_molecules)
        total_ops = min(remaining * 2, 100)  # Generate 2x needed
        
        num_crossovers = int(total_ops * strategy.crossover_ratio)
        num_mutations = int(total_ops * strategy.mutation_ratio)
        num_local_search = int(total_ops * strategy.local_search_ratio)
        
        print(
            f"Operations: {num_crossovers} crossovers, "
            f"{num_mutations} mutations, {num_local_search} local searches"
        )
        
        new_molecules = []
        round_stats = {
            'crossovers': 0,
            'mutations': 0,
            'local_searches': 0
        }
        
        # ====================================================================
        # CROSSOVER OPERATIONS
        # ====================================================================
        
        for i in range(num_crossovers):
            total_stats['crossovers_attempted'] += 1
            
            offspring = None
            
            if strategy.use_guided_ops:
                # Use intelligent parent selection and component-preserving crossover
                parent1, parent2 = parent_selector.select_parents_component_aware()
                offspring = ga_operator.crossover_component_preserving(parent1, parent2)
                
                # Fallback to multi-component if component-preserving fails
                if not offspring:
                    offspring = ga_operator.crossover_multi_component(parent1, parent2)
            else:
                # Use tournament selection and multi-component crossover
                parent1, parent2 = parent_selector.select_parents_tournament()
                offspring = ga_operator.crossover_multi_component(parent1, parent2)
            
            if offspring:
                try:
                    smiles = get_smiles_from_reaction(offspring)
                    inchikey = generate_inchikey(smiles)
                    
                    if smiles and inchikey:
                        # Check if already generated
                        if offspring not in generated_molecules and inchikey not in generated_inchikeys:
                            new_molecules.append({
                                'name': offspring,
                                'smiles': smiles,
                                'InChIKey': inchikey,
                                'type': 'crossover',
                                'round': round_num + 1
                            })
                            generated_molecules.add(offspring)
                            generated_inchikeys.add(inchikey)
                            round_stats['crossovers'] += 1
                            total_stats['crossovers_successful'] += 1
                
                except Exception as e:
                    pass
        
        # ====================================================================
        # MUTATION OPERATIONS
        # ====================================================================
        
        for i in range(num_mutations):
            total_stats['mutations_attempted'] += 1
            
            parent = random.choice(pool_names)
            mutated = None
            
            if strategy.use_guided_ops:
                mutated = ga_operator.mutate_guided(parent)
            else:
                mutated = ga_operator.mutate_random(parent)
            
            if mutated:
                try:
                    smiles = get_smiles_from_reaction(mutated)
                    inchikey = generate_inchikey(smiles)
                    
                    if smiles and inchikey:
                        # Check if already generated
                        if mutated not in generated_molecules and inchikey not in generated_inchikeys:
                            new_molecules.append({
                                'name': mutated,
                                'smiles': smiles,
                                'InChIKey': inchikey,
                                'type': 'mutation',
                                'round': round_num + 1
                            })
                            generated_molecules.add(mutated)
                            generated_inchikeys.add(inchikey)
                            round_stats['mutations'] += 1
                            total_stats['mutations_successful'] += 1
                
                except Exception as e:
                    pass
        
        # ====================================================================
        # LOCAL SEARCH OPERATIONS
        # ====================================================================
        
        if num_local_search > 0:
            # Select top molecules for local search
            top_for_search = min(5, len(pool_molecules))
            top_mols = pool_molecules.head(top_for_search)['name'].tolist()
            
            for parent in top_mols:
                total_stats['local_searches_attempted'] += 1
                
                neighbors = ga_operator.mutate_local_search(parent, num_neighbors=3)
                
                for neighbor in neighbors:
                    try:
                        smiles = get_smiles_from_reaction(neighbor)
                        inchikey = generate_inchikey(smiles)
                        
                        if smiles and inchikey:
                            # Check if already generated
                            if neighbor not in generated_molecules and inchikey not in generated_inchikeys:
                                new_molecules.append({
                                    'name': neighbor,
                                    'smiles': smiles,
                                    'InChIKey': inchikey,
                                    'type': 'local_search',
                                    'round': round_num + 1
                                })
                                generated_molecules.add(neighbor)
                                generated_inchikeys.add(inchikey)
                                round_stats['local_searches'] += 1
                                total_stats['local_searches_successful'] += 1
                    
                    except Exception as e:
                        pass
        
        # ====================================================================
        # VALIDATION
        # ====================================================================
        
        print(f"Validating {len(new_molecules)} generated molecules...")
        
        validated_molecules = []
        validation_failures = {
            'smiles': 0,
            'heavy_atoms': 0,
            'banned_atoms': 0,
            'rotatable_bonds': 0,
            'hf_unique': 0,
            'other': 0
        }
        
        for mol in new_molecules:
            # Validate with config settings
            is_valid, errors = await validate_molecule_complete(
                state, mol['name'], mol['smiles'], state['config']
            )
            
            if is_valid:
                validated_molecules.append(mol)
                
                # Update best score if this molecule has a score
                score = mol.get('boltz_score')
                if score and score > best_score_so_far:
                    best_score_so_far = score
            else:
                # Track validation failure reasons
                for error in errors:
                    if "[SMILES]" in error:
                        validation_failures['smiles'] += 1
                    elif "[HEAVY_ATOMS]" in error:
                        validation_failures['heavy_atoms'] += 1
                    elif "[BANNED_ATOMS]" in error:
                        validation_failures['banned_atoms'] += 1
                    elif "[ROTATABLE_BONDS]" in error:
                        validation_failures['rotatable_bonds'] += 1
                    elif "[HF_UNIQUE]" in error:
                        validation_failures['hf_unique'] += 1
                    else:
                        validation_failures['other'] += 1
        
        unique_molecules.extend(validated_molecules)
        
        # ====================================================================
        # ROUND SUMMARY
        # ====================================================================
        
        print(
            f"\n📊 Round {round_num + 1} Summary:\n"
            f"   Generated: {len(new_molecules)} molecules\n"
            f"   - Crossovers: {round_stats['crossovers']}\n"
            f"   - Mutations: {round_stats['mutations']}\n"
            f"   - Local searches: {round_stats['local_searches']}\n"
            f"   Validated: {len(validated_molecules)}/{len(new_molecules)}\n"
            f"   Validation failures:\n"
            f"   - SMILES: {validation_failures['smiles']}\n"
            f"   - Heavy atoms: {validation_failures['heavy_atoms']}\n"
            f"   - Banned atoms: {validation_failures['banned_atoms']}\n"
            f"   - Rotatable bonds: {validation_failures['rotatable_bonds']}\n"
            f"   - HF uniqueness: {validation_failures['hf_unique']}\n"
            f"   - Other: {validation_failures['other']}\n"
            f"   Total unique: {len(unique_molecules)}/{desired_count}"
        )
        
        # Early stopping if no progress
        if len(validated_molecules) == 0 and round_num > 2:
            print(
                f"⚠️  No valid molecules generated in round {round_num + 1}, stopping early"
            )
            break
        
        # Small delay
        await asyncio.sleep(0.1)
    
    # Update state with tracking sets
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(
        f"\n{'='*70}\n"
        f"✅ GENERATION COMPLETE\n"
        f"{'='*70}\n"
        f"Total unique molecules: {len(unique_molecules)}/{desired_count}\n"
        f"Total tracked: {len(generated_molecules)}\n"
        f"\n"
        f"Operation Statistics:\n"
        f"  Crossovers: {total_stats['crossovers_successful']}/{total_stats['crossovers_attempted']} "
        f"({total_stats['crossovers_successful']/max(total_stats['crossovers_attempted'],1)*100:.1f}%)\n"
        f"  Mutations: {total_stats['mutations_successful']}/{total_stats['mutations_attempted']} "
        f"({total_stats['mutations_successful']/max(total_stats['mutations_attempted'],1)*100:.1f}%)\n"
        f"  Local searches: {total_stats['local_searches_successful']}/{total_stats['local_searches_attempted']} "
        f"({total_stats['local_searches_successful']/max(total_stats['local_searches_attempted'],1)*100:.1f}%)\n"
        f"{'='*70}"
    )
    
    return unique_molecules[:desired_count]


# ============================================================================
# Database Functions (from original code)
# ============================================================================

def init_score_results_db(db_path: str = None) -> None:
    """Initialize/create the score_results.sqlite database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scored_molecules (
                molecule_name TEXT PRIMARY KEY,
                score REAL NOT NULL,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available BOOLEAN DEFAULT TRUE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing score_results database: {e}")


def get_score_from_db(molecule_name: str, db_path: str = None) -> Optional[float]:
    """Get score for a molecule from the database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT score FROM scored_molecules WHERE molecule_name = ?", (molecule_name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return float(result[0])
        return None
    except Exception as e:
        return None


def write_scores_to_db(molecules: List[Dict[str, Any]], db_path: str = None) -> None:
    """Write scored molecules to the database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    if not molecules:
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        to_insert = []
        for mol in molecules:
            molecule_name = mol.get('name')
            score = mol.get('boltz_score')
            
            if molecule_name and score is not None:
                to_insert.append((molecule_name, float(score), True))
        
        if to_insert:
            cursor.executemany(
                "INSERT OR REPLACE INTO scored_molecules (molecule_name, score, available) VALUES (?, ?, ?)",
                to_insert
            )
            conn.commit()
            print(f"✅ Wrote {len(to_insert)} scored molecules to database")
        
        conn.close()
    except Exception as e:
        print(f"Error writing scores to database: {e}")


def batch_get_scores_from_db(molecule_names: List[str], db_path: str = None) -> Dict[str, float]:
    """Get scores for multiple molecules from the database in batch."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    if not molecule_names:
        return {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(molecule_names))
        cursor.execute(
            f"SELECT molecule_name, score FROM scored_molecules WHERE molecule_name IN ({placeholders})",
            molecule_names
        )
        results = cursor.fetchall()
        conn.close()
        
        return {name: float(score) for name, score in results}
    except Exception as e:
        return {}


def load_molecules_from_db_with_validation(
    db_path: str,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from SQLite database with validation."""
    if config is None:
        config = {}
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        print(f"Loading molecules from database {db_path} for rxn_id={rxn_id}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT molecule_name, score FROM scored_molecules")
        db_results = cursor.fetchall()
        conn.close()
        
        if not db_results:
            print("No molecules found in database")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        banned_atom_count = 0
        heavy_atom_count = 0
        wrong_rxn_id_count = 0
        
        for molecule_name, score in db_results:
            try:
                if not molecule_name.startswith(f"rxn:{rxn_id}:"):
                    wrong_rxn_id_count += 1
                    continue
                
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_count += 1
                    continue
                
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    banned_atom_count += 1
                    continue
                
                min_heavy_atoms = 20
                max_heavy_atoms = 25
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms or heavy_atom_count_val > max_heavy_atoms:
                    heavy_atom_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    failed_count += 1
                    continue
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': float(score) if score is not None else None,
                })
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                print(
                    f"✅ Loaded {len(result_df)} molecules from database "
                    f"(successful: {successful_count}, failed: {failed_count})"
                )
        
        return result_df
        
    except Exception as e:
        print(f"Error loading molecules from database: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_csv_with_validation(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from CSV file with validation."""
    if config is None:
        config = {}
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        print(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            print("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            print("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if df.empty:
            print("No matching molecules found in CSV")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        
        for _, row in df.iterrows():
            molecule_name = row['molecule_name']
            
            try:
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_count += 1
                    continue
                
                # Check banned atoms
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    failed_count += 1
                    continue
                
                # Check heavy atom count
                min_heavy_atoms = 20
                max_heavy_atoms = 25
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms or heavy_atom_count_val > max_heavy_atoms:
                    failed_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    failed_count += 1
                    continue
                
                final_score = row.get('final_score', None)
                if pd.isna(final_score):
                    final_score = None
                else:
                    final_score = float(final_score)
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': final_score,
                })
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                print(
                    f"✅ Loaded {len(result_df)} molecules from CSV "
                    f"(successful: {successful_count}, failed: {failed_count})"
                )
        
        return result_df
        
    except Exception as e:
        print(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(
    csv_path: str,
    db_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Load molecules from both CSV and database, merge them, and deduplicate.
    When duplicates exist (by InChIKey), prefer the one with the higher score.
    """
    if config is None:
        config = {}
    
    print(f"🔄 Loading molecules from CSV and database...")
    
    # Load from CSV
    csv_df = load_molecules_from_csv_with_validation(
        csv_path, target_proteins, starting_epoch, rxn_id, config
    )
    
    # Load from database
    db_df = load_molecules_from_db_with_validation(
        db_path, rxn_id, config
    )
    
    # Combine both dataframes
    if csv_df.empty and db_df.empty:
        print("No molecules loaded from either CSV or database")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if csv_df.empty:
        print("No molecules from CSV, using database only")
        return db_df
    
    if db_df.empty:
        print("No molecules from database, using CSV only")
        return csv_df
    
    # Merge dataframes
    csv_df['source'] = 'csv'
    db_df['source'] = 'database'
    
    combined_df = csv_df
    # combined_df = pd.concat([csv_df, db_df], ignore_index=True)
    
    # Deduplicate by InChIKey, keeping the one with the highest score
    combined_df = combined_df.sort_values(
        by=['score', 'source'],
        ascending=[False, True],
        na_position='last'
    )
    
    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')
    combined_df = combined_df.drop(columns=['source'])
    combined_df = combined_df.sort_values(by='score', ascending=False, na_position='last')
    
    csv_count = len(csv_df)
    db_count = len(db_df)
    combined_count = len(combined_df)
    duplicates_removed = csv_count + db_count - combined_count
    
    print(
        f"✅ Combined loading complete: "
        f"{csv_count} from CSV, {db_count} from database, "
        f"{combined_count} unique molecules after deduplication "
        f"({duplicates_removed} duplicates removed)"
    )
    
    if combined_count > 0:
        scores = combined_df['score'].dropna()
        if len(scores) > 0:
            print(
                f"   Combined score range: {scores.min():.6f} to {scores.max():.6f} "
                f"(top 3: {scores.head(3).tolist()})"
            )
    
    return combined_df


# ============================================================================
# Boltz Scoring Functions
# ============================================================================

async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Score molecules using BoltzWrapper in batches.
    """
    if state.get('boltz_wrapper') is None:
        print("BoltzWrapper not available, skipping scoring")
        return molecules
    
    if not molecules:
        return molecules
    
    print(f"🔬 Processing {len(molecules)} molecules for scoring in batches of {batch_size}...")
    
    init_score_results_db()
    
    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    
    # Process molecules in batches
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]
        
        print(
            f"📦 Batch {batch_idx + 1}/{total_batches}: "
            f"Scoring {len(batch)} molecules"
        )
        
        # Score this batch
        molecules_to_score = []
        molecules_with_db_scores = []
        molecules_in_hf = []
        
        target_proteins = state.get('current_challenge_targets', [])
        primary_target = target_proteins[0] if target_proteins else None
        
        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)
        
        print(f"   Found {len(db_scores)} molecules already in database")
        
        # Separate molecules by source
        for mol in batch:
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
            if molecule_name in db_scores:
                mol['boltz_score'] = db_scores[molecule_name]
                mol['boltz_score_source'] = 'database'
                molecules_with_db_scores.append(mol)
                continue
            
            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                    if not is_unique_hf:
                        molecules_in_hf.append(mol)
                        continue
                except Exception as e:
                    pass
            
            molecules_to_score.append(mol)
        
        print(
            f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
            f"{len(molecules_in_hf)} in HuggingFace (skipped), "
            f"{len(molecules_to_score)} need scoring"
        )
        
        newly_scored_molecules = []
        if molecules_to_score:
            print(f"   Scoring {len(molecules_to_score)} new molecules with Boltz...")
            
            boltz = state['boltz_wrapper']
            config = state['config']
            target_proteins = state.get('current_challenge_targets', [])
            antitarget_proteins = state.get('current_challenge_antitargets', [])
            
            if not target_proteins:
                print("No target proteins available for scoring")
                all_scored_molecules.extend(molecules_with_db_scores)
                continue
            
            primary_target = target_proteins[0]
            
            try:
                output_dir = os.path.join(boltz.output_dir, 'boltz_results_inputs')
                if os.path.exists(output_dir):
                    try:
                        lightning_logs_dir = os.path.join(output_dir, 'lightning_logs')
                        if os.path.exists(lightning_logs_dir):
                            import shutil
                            shutil.rmtree(lightning_logs_dir, ignore_errors=True)
                    except Exception as cleanup_err:
                        pass
                
                processed_dir = os.path.join(output_dir, 'processed')
                structures_dir = os.path.join(processed_dir, 'structures')
                records_dir = os.path.join(processed_dir, 'records')
                msa_dir = os.path.join(processed_dir, 'msa')
                predictions_dir = os.path.join(output_dir, 'predictions')
                
                os.makedirs(structures_dir, exist_ok=True)
                os.makedirs(records_dir, exist_ok=True)
                os.makedirs(msa_dir, exist_ok=True)
                os.makedirs(predictions_dir, exist_ok=True)
                
                valid_molecules_by_uid = {
                    0: {
                        'smiles': [mol['smiles'] for mol in molecules_to_score],
                        'names': [mol['name'] for mol in molecules_to_score]
                    }
                }
                
                score_dict = {
                    0: {
                        "target_scores": [[]],
                        "antitarget_scores": [[]],
                        "entropy": None,
                        "entropy_boltz": None,
                        "block_submitted": None,
                        "push_time": ""
                    }
                }
                
                num_molecules_to_score = len(molecules_to_score)
                subnet_config = {
                    'weekly_target': primary_target,
                    'num_antitargets': len(antitarget_proteins),
                    'binding_pocket': config.get('binding_pocket'),
                    'max_distance': config.get('max_distance'),
                    'force': config.get('force', False),
                    'num_molecules_boltz': num_molecules_to_score,
                    'boltz_metric': config.get('boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']),
                    'combination_strategy': config.get('combination_strategy', 'heavy_atom_normalization'),
                    'sample_selection': config.get('sample_selection', 'first'),
                }
                
                final_block_hash = "0x" + "0" * 64
                
                start_time = time.time()
                
                def run_scoring():
                    boltz.score_molecules_target(
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config,
                        final_block_hash
                    )
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, run_scoring)
                
                elapsed = time.time() - start_time
                print(f"   ✅ Boltz scoring completed in {elapsed:.2f} seconds")
                
                uid = 0
                smiles_to_score = {}
                if uid in boltz.per_molecule_metric:
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()
                
                target_scores_list = None
                target_scores = score_dict[uid].get('target_scores', [[]])
                if target_scores and len(target_scores[0]) > 0:
                    target_scores_list = target_scores[0] if isinstance(target_scores[0], list) else [target_scores[0]]
                
                avg_score = None
                if not smiles_to_score and not target_scores_list:
                    avg_score = score_dict[uid].get('boltz_score')
                
                for mol_idx, mol in enumerate(molecules_to_score):
                    smiles = mol['smiles']
                    score = None
                    
                    if smiles in smiles_to_score:
                        score = smiles_to_score[smiles]
                    elif target_scores_list and mol_idx < len(target_scores_list):
                        score = target_scores_list[mol_idx]
                    elif target_scores_list:
                        try:
                            valid_idx = valid_molecules_by_uid[uid]['smiles'].index(smiles)
                            if valid_idx < len(target_scores_list):
                                score = target_scores_list[valid_idx]
                        except (ValueError, IndexError):
                            pass
                    
                    if score is None and avg_score is not None:
                        score = avg_score
                    
                    mol['boltz_score'] = score
                    if score is not None:
                        newly_scored_molecules.append(mol)
                
                if newly_scored_molecules:
                    write_scores_to_db(newly_scored_molecules)
            
            except Exception as e:
                print(f"❌ Error scoring batch with Boltz: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Combine results from this batch
        batch_results = molecules_with_db_scores + newly_scored_molecules
        
        for mol in molecules_in_hf:
            mol['boltz_score'] = None
            mol['boltz_score_source'] = 'huggingface_skipped'
            batch_results.append(mol)
        
        all_scored_molecules.extend(batch_results)
        
        print(
            f"   ✅ Batch {batch_idx + 1} complete: "
            f"{len(molecules_with_db_scores)} from DB, "
            f"{len(newly_scored_molecules)} newly scored, "
            f"{len(molecules_in_hf)} skipped"
        )
    
    # Sort all results by score
    scored_molecules = sorted(
        all_scored_molecules,
        key=lambda m: m.get('boltz_score') if m.get('boltz_score') is not None else float('-inf'),
        reverse=True
    )
    
    print(f"✅ Batch scoring complete: {len(scored_molecules)} total molecules scored")
    
    return scored_molecules


def _import_boltz_wrapper():
    """Import BoltzWrapper."""
    global BOLTZ_AVAILABLE, BoltzWrapper
    
    try:
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        
        if not os.path.exists(BOLTZ_SCORING_DIR):
            print(f"⚠️  Boltz-scoring directory not found at {BOLTZ_SCORING_DIR}")
            return False
        
        if BOLTZ_SCORING_DIR not in sys.path:
            sys.path.append(BOLTZ_SCORING_DIR)
        
        if BOLTZ_SRC_DIR not in sys.path:
            sys.path.insert(0, BOLTZ_SRC_DIR)
        
        boltz_utils_path = os.path.join(BOLTZ_SCORING_DIR, 'utils')
        if os.path.exists(boltz_utils_path) and boltz_utils_path not in sys.path:
            sys.path.insert(0, boltz_utils_path)
        
        from boltz.wrapper import BoltzWrapper as BW
        BoltzWrapper = BW
        BOLTZ_AVAILABLE = True
        print(f"✅ BoltzWrapper imported successfully")
        return True
        
    except ImportError as e:
        print(f"⚠️  Failed to import BoltzWrapper: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error setting up BoltzWrapper: {e}")
        return False


# ============================================================================
# Startup and Main Loop
# ============================================================================

async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Initialize score_results database
    2. Import and initialize BoltzWrapper
    3. Load molecules from CSV and database with validation
    4. Prepare top_pool
    """
    print("🚀 Starting STARTUP phase: Initialize DB, Boltz & Load CSV/DB...")
    
    try:
        # Initialize score_results database
        print("💾 Initializing score_results database...")
        init_score_results_db()
        print(f"✅ Score results database initialized")
        
        # Log validation config
        config = state['config']
        print(
            f"✅ Loaded validation config from config.yaml:"
            f"\n   - min_heavy_atoms: 18"
            f"\n   - max_heavy_atoms: 30"
            f"\n   - min_rotatable_bonds: {config.get('min_rotatable_bonds', 1)}"
            f"\n   - max_rotatable_bonds: {config.get('max_rotatable_bonds', 10)}"
            f"\n   - banned_atom_types: {config.get('banned_atom_types', [])}"
        )
        
        # Import BoltzWrapper
        print("🔬 Importing BoltzWrapper...")
        boltz_imported = _import_boltz_wrapper()
        
        # Initialize BoltzWrapper
        if boltz_imported and BoltzWrapper is not None:
            print("🔬 Initializing BoltzWrapper...")
            try:
                state['boltz_wrapper'] = BoltzWrapper()
                print("✅ BoltzWrapper initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize BoltzWrapper: {e}")
                import traceback
                print(traceback.format_exc())
                state['boltz_wrapper'] = None
        else:
            print("⚠️  BoltzWrapper not available, scoring will be skipped")
            state['boltz_wrapper'] = None
        
        # Load molecules from both CSV and database
        print("📂 Loading molecules from CSV and database with validation...")
        molecules_df = load_molecules_combined(
            REACTION_TRAIN_CSV,
            SCORE_RESULTS_DB,
            state['current_challenge_targets'],
            STARTING_EPOCH,
            HARDCODED_RXN_ID,
            config
        )
        
        if molecules_df.empty:
            print("⚠️  No valid molecules loaded from CSV or database")
            return
        
        # Get top 500 molecules
        top_500_df = molecules_df.head(500)
        
        # Store in state
        state['top_pool'] = molecules_df.copy()
        state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
        state['top_200_df'] = top_500_df
        
        print(f"✅ Loaded {len(molecules_df)} molecules from CSV and database (top 500: {len(top_500_df)})")
        
        print(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Top 500 molecules: {len(state['top_200_df'])}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        print(f"Error in startup phase: {e}")
        import traceback
        print(traceback.format_exc())


async def run_continuous_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Continuous genetic algorithm loop with ADVANCED generation:
    1. Generate 100 unique molecules using adaptive strategy
    2. Score them in batches of 10
    3. Save best molecule
    4. Update top pool
    5. Repeat infinitely
    """
    print("🚀 Starting CONTINUOUS ADVANCED genetic algorithm loop...")
    
    desired_unique_count = 1000
    batch_size = 10
    round_number = 0
    
    while not state['shutdown_event'].is_set():
        try:
            round_number += 1
            print(f"\n{'='*70}")
            print(f"🔄 Generation Round {round_number}")
            print(f"{'='*70}")
            
            # Use existing top_200_df from state
            top_200_df = state.get('top_200_df', pd.DataFrame())
            
            if top_200_df.empty:
                print("No top molecules found, waiting...")
                await asyncio.sleep(10)
                continue
            
            # ================================================================
            # ADVANCED GENERATION with adaptive strategy
            # ================================================================
            print(f"🧬 Generating {desired_unique_count} unique molecules with ADAPTIVE strategy...")
            unique_molecules = await generate_molecules_adaptive(
                state=state,
                top_200_df=top_200_df,
                desired_count=desired_unique_count,
                max_rounds=10
            )
            
            if not unique_molecules:
                print("Failed to generate unique molecules, retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue
            
            print(f"✅ Generated {len(unique_molecules)} valid unique molecules")
            
            # ================================================================
            # SCORE MOLECULES
            # ================================================================
            print(f"🔬 Round {round_number}: Scoring {len(unique_molecules)} molecules in batches of {batch_size}...")
            
            scored_molecules = await score_molecules_with_boltz_batched(
                state, unique_molecules, batch_size=batch_size
            )
            
            # Filter molecules with valid scores
            molecules_with_scores = [m for m in scored_molecules if m.get('boltz_score') is not None]
            
            if molecules_with_scores:
                # Sort by score
                molecules_with_scores.sort(key=lambda m: m.get('boltz_score', float('-inf')), reverse=True)
                
                # Get best molecule
                best_molecule = molecules_with_scores[0]
                best_score = best_molecule.get('boltz_score')
                
                print(
                    f"\n{'='*70}"
                    f"\n🏆 Round {round_number} Best Molecule:"
                    f"\n   Name: {best_molecule['name']}"
                    f"\n   Score: {best_score:.6f}"
                    f"\n   SMILES: {best_molecule.get('smiles', 'N/A')}"
                    f"\n   Type: {best_molecule.get('type', 'N/A')}"
                    f"\n{'='*70}\n"
                )
                
                # Update state with best molecule
                state['best_molecule'] = best_molecule
                state['best_score'] = best_score
                
                # ============================================================
                # UPDATE TOP POOL
                # ============================================================
                new_rows = []
                for mol in molecules_with_scores:
                    new_rows.append({
                        'name': mol['name'],
                        'smiles': mol['smiles'],
                        'InChIKey': mol['InChIKey'],
                        'score': mol['boltz_score']
                    })
                
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    combined_df = pd.concat([state['top_200_df'], new_df], ignore_index=True)
                    
                    # Deduplicate by InChIKey, keeping highest score
                    combined_df = combined_df.sort_values(by='score', ascending=False, na_position='last')
                    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')
                    
                    # Keep top 500
                    state['top_200_df'] = combined_df.head(500)
                    
                    print(f"✅ Updated top pool: {len(state['top_200_df'])} molecules")
            else:
                print("⚠️  No molecules with valid scores in this round")
            
            print(f"✅ Round {round_number} complete. Starting next round...")
            
            # Small delay before next round
            await asyncio.sleep(1)
        
        except Exception as e:
            print(f"Error in continuous GA loop: {e}")
            import traceback
            print(traceback.format_exc())
            await asyncio.sleep(10)


async def run_generator() -> None:
    """Main generation loop."""
    print("🚀 Starting ADVANCED molecule generator...")
    
    # Load config from config.yaml
    config_dict = load_config()
    
    # Get target protein from config
    target_protein = config_dict.get('weekly_target', 'KRAS')
    
    print(f"Configuration loaded from config.yaml:")
    print(f"  Target protein: {target_protein}")
    print(f"  Reaction ID: {HARDCODED_RXN_ID}")
    print(f"  Database: {DB_PATH}")
    print(f"  Score DB: {SCORE_RESULTS_DB}")
    
    state: Dict[str, Any] = {
        'config': config_dict,
        'startup_complete': False,
        'shutdown_event': asyncio.Event(),
        'current_challenge_targets': [target_protein],
        'current_challenge_antitargets': [],
        'rxn_id': HARDCODED_RXN_ID,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),
        'generated_inchikeys': set(),
        'boltz_wrapper': None,
        'top_200_df': pd.DataFrame(),
        'best_molecule': None,
        'best_score': float('-inf'),
    }
    
    print("🚀 Entering main generator loop...")
    
    # Run startup phase
    await startup_phase(state)
    
    # Run continuous genetic loop with ADVANCED generation
    await run_continuous_genetic_loop(state)


def main():
    """Main entry point."""
    try:
        asyncio.run(run_generator())
    except KeyboardInterrupt:
        print("\n🛑 Generator interrupted by user")
    except Exception as e:
        print(f"Fatal error in generator: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
