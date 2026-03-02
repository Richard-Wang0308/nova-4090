"""
Bayesian Optimization System for Molecule Generation
Uses Gaussian Process to intelligently select promising molecules
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
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from dataclasses import dataclass
from collections import defaultdict

# Bayesian Optimization imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 21075
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
# Bayesian Optimization Core
# ============================================================================

class BayesianMoleculeOptimizer:
    """
    Bayesian Optimization for molecular component selection.
    
    Strategy:
    1. Encode molecules as vectors (component IDs normalized to [0,1])
    2. Train Gaussian Process on scored molecules
    3. Use Expected Improvement to select next best candidates
    4. Score and repeat
    """
    
    def __init__(self, rxn_id: int, db_path: str):
        self.rxn_id = rxn_id
        self.db_path = db_path
        
        # Load component pools
        self.reaction_info = get_reaction_info(rxn_id, db_path)
        if self.reaction_info:
            self.smarts, self.roleA, self.roleB, self.roleC = self.reaction_info
            self.is_three_component = self.roleC is not None and self.roleC != 0
            
            self.molecules_A = get_molecules_by_role(self.roleA, db_path)
            self.molecules_B = get_molecules_by_role(self.roleB, db_path)
            self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
            
            self.component_ids_A = [mol[0] for mol in self.molecules_A]
            self.component_ids_B = [mol[0] for mol in self.molecules_B]
            self.component_ids_C = [mol[0] for mol in self.molecules_C] if self.is_three_component else []
            
            # Create component ID to index mapping
            self.comp_a_to_idx = {cid: idx for idx, cid in enumerate(self.component_ids_A)}
            self.comp_b_to_idx = {cid: idx for idx, cid in enumerate(self.component_ids_B)}
            self.comp_c_to_idx = {cid: idx for idx, cid in enumerate(self.component_ids_C)} if self.is_three_component else {}
            
            print(f"🔬 Bayesian Optimizer initialized:")
            print(f"   Component A: {len(self.component_ids_A)} options")
            print(f"   Component B: {len(self.component_ids_B)} options")
            if self.is_three_component:
                print(f"   Component C: {len(self.component_ids_C)} options")
        else:
            raise ValueError(f"Could not load reaction info for rxn_id {rxn_id}")
        
        # Gaussian Process model
        # Use Matern kernel with constant kernel for better scaling
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * (3 if self.is_three_component else 2),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        
        # Training data
        self.X_train = []  # Encoded molecules (normalized indices)
        self.y_train = []  # Scores
        self.molecule_names = []  # Keep track of molecule names
        self.trained = False
        
        # Component statistics for fallback
        self.comp_a_scores = defaultdict(list)
        self.comp_b_scores = defaultdict(list)
        self.comp_c_scores = defaultdict(list)
    
    def encode_molecule(self, molecule_name: str) -> np.ndarray:
        """
        Encode molecule as normalized vector.
        
        Format: rxn:2:compA:compB or rxn:2:compA:compB:compC
        Returns: [comp_a_idx_norm, comp_b_idx_norm] or [comp_a_idx_norm, comp_b_idx_norm, comp_c_idx_norm]
        
        Normalization: idx / (max_idx) to get values in [0, 1]
        """
        parts = molecule_name.split(':')
        
        comp_a = int(parts[2])
        comp_b = int(parts[3])
        
        comp_a_idx = self.comp_a_to_idx.get(comp_a, 0)
        comp_b_idx = self.comp_b_to_idx.get(comp_b, 0)
        
        # Normalize to [0, 1]
        comp_a_norm = comp_a_idx / max(len(self.component_ids_A) - 1, 1)
        comp_b_norm = comp_b_idx / max(len(self.component_ids_B) - 1, 1)
        
        if self.is_three_component and len(parts) > 4:
            comp_c = int(parts[4])
            comp_c_idx = self.comp_c_to_idx.get(comp_c, 0)
            comp_c_norm = comp_c_idx / max(len(self.component_ids_C) - 1, 1)
            return np.array([comp_a_norm, comp_b_norm, comp_c_norm])
        else:
            return np.array([comp_a_norm, comp_b_norm])
    
    def decode_molecule(self, encoded: np.ndarray) -> str:
        """
        Decode normalized vector back to molecule name.
        """
        # Denormalize
        comp_a_idx = int(encoded[0] * (len(self.component_ids_A) - 1))
        comp_b_idx = int(encoded[1] * (len(self.component_ids_B) - 1))
        
        # Clip to valid range
        comp_a_idx = np.clip(comp_a_idx, 0, len(self.component_ids_A) - 1)
        comp_b_idx = np.clip(comp_b_idx, 0, len(self.component_ids_B) - 1)
        
        comp_a = self.component_ids_A[comp_a_idx]
        comp_b = self.component_ids_B[comp_b_idx]
        
        if self.is_three_component and len(encoded) > 2:
            comp_c_idx = int(encoded[2] * (len(self.component_ids_C) - 1))
            comp_c_idx = np.clip(comp_c_idx, 0, len(self.component_ids_C) - 1)
            comp_c = self.component_ids_C[comp_c_idx]
            return f"rxn:{self.rxn_id}:{comp_a}:{comp_b}:{comp_c}"
        else:
            return f"rxn:{self.rxn_id}:{comp_a}:{comp_b}"
    
    def add_training_data(self, molecules_df: pd.DataFrame):
        """
        Add scored molecules to training set.
        """
        added = 0
        for _, row in molecules_df.iterrows():
            if row.get('score') is not None and not pd.isna(row['score']):
                encoded = self.encode_molecule(row['name'])
                self.X_train.append(encoded)
                self.y_train.append(row['score'])
                self.molecule_names.append(row['name'])
                added += 1
                
                # Update component statistics
                parts = row['name'].split(':')
                comp_a = int(parts[2])
                comp_b = int(parts[3])
                self.comp_a_scores[comp_a].append(row['score'])
                self.comp_b_scores[comp_b].append(row['score'])
                if self.is_three_component and len(parts) > 4:
                    comp_c = int(parts[4])
                    self.comp_c_scores[comp_c].append(row['score'])
        
        print(f"   Added {added} molecules to training set (total: {len(self.X_train)})")
    
    def train(self) -> bool:
        """
        Train Gaussian Process on current data.
        """
        if len(self.X_train) < 5:
            print(f"   ⚠️  Not enough training data ({len(self.X_train)} < 5), using component statistics")
            return False
        
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        print(f"   Training GP on {len(X)} molecules...")
        print(f"   Score range: {y.min():.6f} to {y.max():.6f} (mean: {y.mean():.6f})")
        
        try:
            self.gp.fit(X, y)
            self.trained = True
            
            # Calculate R² score
            score = self.gp.score(X, y)
            print(f"   ✅ GP trained successfully (R² score: {score:.3f})")
            
            return True
        except Exception as e:
            print(f"   ❌ GP training failed: {e}")
            self.trained = False
            return False
    
    def expected_improvement(
        self, 
        X_candidates: np.ndarray,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.
        
        EI(x) = E[max(f(x) - f(x_best), 0)]
        
        Args:
            X_candidates: Array of encoded molecules
            xi: Exploration parameter (larger = more exploration)
        
        Returns:
            Expected improvement values (higher = more promising)
        """
        if not self.trained:
            # Fallback: use component statistics
            return self._component_based_score(X_candidates)
        
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        
        # Best observed value
        y_best = np.max(self.y_train)
        
        # Calculate EI
        with np.errstate(divide='warn', invalid='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _component_based_score(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Fallback scoring based on component statistics.
        Used when GP is not trained yet.
        """
        scores = []
        
        for x in X_candidates:
            # Denormalize to get component indices
            comp_a_idx = int(x[0] * (len(self.component_ids_A) - 1))
            comp_b_idx = int(x[1] * (len(self.component_ids_B) - 1))
            
            comp_a_idx = np.clip(comp_a_idx, 0, len(self.component_ids_A) - 1)
            comp_b_idx = np.clip(comp_b_idx, 0, len(self.component_ids_B) - 1)
            
            comp_a = self.component_ids_A[comp_a_idx]
            comp_b = self.component_ids_B[comp_b_idx]
            
            # Get average scores for these components
            score_a = np.mean(self.comp_a_scores[comp_a]) if comp_a in self.comp_a_scores else 0.5
            score_b = np.mean(self.comp_b_scores[comp_b]) if comp_b in self.comp_b_scores else 0.5
            
            if self.is_three_component and len(x) > 2:
                comp_c_idx = int(x[2] * (len(self.component_ids_C) - 1))
                comp_c_idx = np.clip(comp_c_idx, 0, len(self.component_ids_C) - 1)
                comp_c = self.component_ids_C[comp_c_idx]
                score_c = np.mean(self.comp_c_scores[comp_c]) if comp_c in self.comp_c_scores else 0.5
                score = (score_a + score_b + score_c) / 3
            else:
                score = (score_a + score_b) / 2
            
            # Add exploration bonus for unseen components
            exploration_bonus = 0.1
            if comp_a not in self.comp_a_scores:
                score += exploration_bonus
            if comp_b not in self.comp_b_scores:
                score += exploration_bonus
            
            scores.append(score)
        
        return np.array(scores)
    
    def suggest_molecules(
        self,
        n_suggestions: int = 100,
        n_candidates: int = 10000,
        xi: float = 0.01,
        use_top_components: bool = True,
        top_component_ratio: float = 0.3
    ) -> List[str]:
        """
        Suggest next molecules to evaluate using Bayesian Optimization.
        
        Strategy:
        1. Generate candidate molecules:
           - Mix of random sampling
           - Biased sampling from top-performing components
        2. Calculate Expected Improvement for all candidates
        3. Return top N by EI
        
        Args:
            n_suggestions: Number of molecules to suggest
            n_candidates: Number of random candidates to evaluate
            xi: Exploration parameter (0.01 = balanced, 0.1 = more exploration)
            use_top_components: Whether to bias sampling toward good components
            top_component_ratio: Fraction of candidates from top components
        
        Returns:
            List of molecule names
        """
        print(f"   Generating {n_candidates} candidate molecules...")
        
        candidates_encoded = []
        seen = set()
        
        # Add existing training data to seen set
        for x in self.X_train:
            seen.add(tuple(x))
        
        # Calculate how many candidates from each strategy
        n_top_component = int(n_candidates * top_component_ratio) if use_top_components else 0
        n_random = n_candidates - n_top_component
        
        # Strategy 1: Random sampling
        print(f"      Random sampling: {n_random} candidates")
        attempts = 0
        max_attempts = n_random * 10
        
        while len(candidates_encoded) < n_random and attempts < max_attempts:
            attempts += 1
            
            # Random component selection (already normalized)
            comp_a_norm = np.random.rand()
            comp_b_norm = np.random.rand()
            
            if self.is_three_component:
                comp_c_norm = np.random.rand()
                encoded = np.array([comp_a_norm, comp_b_norm, comp_c_norm])
            else:
                encoded = np.array([comp_a_norm, comp_b_norm])
            
            # Check if already evaluated (with some tolerance for floating point)
            encoded_rounded = tuple(np.round(encoded, 4))
            if encoded_rounded in seen:
                continue
            
            seen.add(encoded_rounded)
            candidates_encoded.append(encoded)
        
        # Strategy 2: Biased sampling from top components
        if use_top_components and len(self.comp_a_scores) > 0:
            print(f"      Top-component sampling: {n_top_component} candidates")
            
            # Get top components by average score
            top_a = sorted(
                self.comp_a_scores.items(),
                key=lambda x: np.mean(x[1]),
                reverse=True
            )[:max(10, len(self.comp_a_scores) // 10)]
            
            top_b = sorted(
                self.comp_b_scores.items(),
                key=lambda x: np.mean(x[1]),
                reverse=True
            )[:max(10, len(self.comp_b_scores) // 10)]
            
            if self.is_three_component:
                top_c = sorted(
                    self.comp_c_scores.items(),
                    key=lambda x: np.mean(x[1]),
                    reverse=True
                )[:max(10, len(self.comp_c_scores) // 10)]
            
            attempts = 0
            max_attempts = n_top_component * 10
            
            while len(candidates_encoded) < n_candidates and attempts < max_attempts:
                attempts += 1
                
                # Sample from top components
                comp_a = random.choice([c[0] for c in top_a])
                comp_b = random.choice([c[0] for c in top_b])
                
                comp_a_idx = self.comp_a_to_idx[comp_a]
                comp_b_idx = self.comp_b_to_idx[comp_b]
                
                comp_a_norm = comp_a_idx / max(len(self.component_ids_A) - 1, 1)
                comp_b_norm = comp_b_idx / max(len(self.component_ids_B) - 1, 1)
                
                if self.is_three_component:
                    comp_c = random.choice([c[0] for c in top_c])
                    comp_c_idx = self.comp_c_to_idx[comp_c]
                    comp_c_norm = comp_c_idx / max(len(self.component_ids_C) - 1, 1)
                    encoded = np.array([comp_a_norm, comp_b_norm, comp_c_norm])
                else:
                    encoded = np.array([comp_a_norm, comp_b_norm])
                
                encoded_rounded = tuple(np.round(encoded, 4))
                if encoded_rounded in seen:
                    continue
                
                seen.add(encoded_rounded)
                candidates_encoded.append(encoded)
        
        print(f"   Generated {len(candidates_encoded)} unique candidates")
        
        if not candidates_encoded:
            return []
        
        # Calculate Expected Improvement
        X_candidates = np.array(candidates_encoded)
        ei_values = self.expected_improvement(X_candidates, xi=xi)
        
        # Sort by EI (descending)
        sorted_indices = np.argsort(ei_values)[::-1]
        
        # Decode top N candidates
        suggestions = []
        for i in sorted_indices[:n_suggestions]:
            mol_name = self.decode_molecule(X_candidates[i])
            suggestions.append(mol_name)
        
        print(f"   Selected top {len(suggestions)} by Expected Improvement")
        print(f"   EI range: {ei_values[sorted_indices[0]]:.6f} to {ei_values[sorted_indices[min(n_suggestions-1, len(sorted_indices)-1)]]:.6f}")
        
        return suggestions
    
    def get_top_components(self, role: str, top_n: int = 10) -> List[Tuple[int, float, int]]:
        """
        Get top components by average score.
        
        Returns:
            List of (component_id, mean_score, count)
        """
        if role == 'A':
            scores = self.comp_a_scores
        elif role == 'B':
            scores = self.comp_b_scores
        elif role == 'C':
            scores = self.comp_c_scores
        else:
            return []
        
        components = [
            (cid, np.mean(score_list), len(score_list))
            for cid, score_list in scores.items()
        ]
        components.sort(key=lambda x: x[1], reverse=True)
        
        return components[:top_n]


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
        min_atoms = 18
        max_atoms = 30
        
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
# Main Generation Function with Bayesian Optimization
# ============================================================================

async def generate_molecules_bayesian(
    state: Dict[str, Any],
    top_molecules_df: pd.DataFrame,
    desired_count: int = 1000,
    exploration_factor: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Generate molecules using Bayesian Optimization.
    
    Strategy:
    1. Train GP on existing scored molecules
    2. Use Expected Improvement to select promising candidates
    3. Return candidates for scoring
    
    Args:
        state: Global state dictionary
        top_molecules_df: DataFrame of top-scoring molecules
        desired_count: Number of molecules to generate
        exploration_factor: xi parameter for EI (0.01=balanced, 0.1=explore more)
    
    Returns:
        List of molecule dictionaries
    """
    print(f"\n{'='*70}")
    print(f"🔬 BAYESIAN OPTIMIZATION - Molecule Generation")
    print(f"{'='*70}")
    print(f"Target: {desired_count} molecules")
    print(f"Training data: {len(top_molecules_df)} molecules")
    print(f"Exploration factor (xi): {exploration_factor}")
    print(f"{'='*70}\n")
    
    # Initialize optimizer
    optimizer = BayesianMoleculeOptimizer(HARDCODED_RXN_ID, DB_PATH)
    
    # Add training data
    optimizer.add_training_data(top_molecules_df)
    
    # Train GP
    trained = optimizer.train()
    
    # Adjust parameters based on training status
    if trained:
        print(f"   Using GP-guided selection (Expected Improvement)")
        n_candidates = min(desired_count * 20, 100000)
        use_top_components = True
    else:
        print(f"   Using component-based selection (GP not ready)")
        n_candidates = min(desired_count * 10, 50000)
        use_top_components = True
    
    # Suggest molecules
    suggested_names = optimizer.suggest_molecules(
        n_suggestions=desired_count * 2,  # Generate 2x to account for validation failures
        n_candidates=n_candidates,
        xi=exploration_factor,
        use_top_components=use_top_components,
        top_component_ratio=0.3
    )
    
    print(f"\n   Converting {len(suggested_names)} suggestions to molecules...")
    
    # Convert to molecule dictionaries
    molecules = []
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    for mol_name in suggested_names:
        # Skip if already generated
        if mol_name in generated_molecules:
            continue
        
        try:
            smiles = get_smiles_from_reaction(mol_name)
            if not smiles:
                continue
            
            inchikey = generate_inchikey(smiles)
            if not inchikey:
                continue
            
            # Skip if already seen
            if inchikey in generated_inchikeys:
                continue
            
            molecules.append({
                'name': mol_name,
                'smiles': smiles,
                'InChIKey': inchikey,
                'type': 'bayesian'
            })
            
            generated_molecules.add(mol_name)
            generated_inchikeys.add(inchikey)
            
            # Stop if we have enough
            if len(molecules) >= desired_count:
                break
        
        except Exception as e:
            continue
    
    # Show top components
    print(f"\n   📊 Top Components by Average Score:")
    top_a = optimizer.get_top_components('A', top_n=5)
    top_b = optimizer.get_top_components('B', top_n=5)
    
    if top_a:
        print(f"      Role A: {[(cid, f'{score:.4f}', count) for cid, score, count in top_a]}")
    if top_b:
        print(f"      Role B: {[(cid, f'{score:.4f}', count) for cid, score, count in top_b]}")
    
    print(f"\n✅ Generated {len(molecules)} unique molecules for evaluation")
    
    return molecules


# ============================================================================
# Database Functions
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
        
        for molecule_name, score in db_results:
            try:
                if not molecule_name.startswith(f"rxn:{rxn_id}:"):
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
                    failed_count += 1
                    continue
                
                min_heavy_atoms = 10
                max_heavy_atoms = 30
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms or heavy_atom_count_val > max_heavy_atoms:
                    failed_count += 1
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
                
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    failed_count += 1
                    continue
                
                min_heavy_atoms = 10
                max_heavy_atoms = 30
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
    """
    if config is None:
        config = {}
    
    print(f"🔄 Loading molecules from CSV and database...")
    
    csv_df = load_molecules_from_csv_with_validation(
        csv_path, target_proteins, starting_epoch, rxn_id, config
    )
    
    db_df = load_molecules_from_db_with_validation(
        db_path, rxn_id, config
    )
    
    if csv_df.empty and db_df.empty:
        print("No molecules loaded from either CSV or database")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if csv_df.empty:
        print("No molecules from CSV, using database only")
        return db_df
    
    if db_df.empty:
        print("No molecules from database, using CSV only")
        return csv_df
    
    csv_df['source'] = 'csv'
    db_df['source'] = 'database'
    
    combined_df = pd.concat([csv_df, db_df], ignore_index=True)
    
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
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]
        
        print(
            f"📦 Batch {batch_idx + 1}/{total_batches}: "
            f"Scoring {len(batch)} molecules"
        )
        
        molecules_to_score = []
        molecules_with_db_scores = []
        molecules_in_hf = []
        
        target_proteins = state.get('current_challenge_targets', [])
        primary_target = target_proteins[0] if target_proteins else None
        
        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)
        
        print(f"   Found {len(db_scores)} molecules already in database")
        
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
    3. Load molecules from CSV and database
    """
    print("🚀 Starting STARTUP phase: Initialize DB, Boltz & Load CSV/DB...")
    
    try:
        print("💾 Initializing score_results database...")
        init_score_results_db()
        print(f"✅ Score results database initialized")
        
        config = state['config']
        print(
            f"✅ Loaded validation config from config.yaml:"
            f"\n   - min_heavy_atoms: 18"
            f"\n   - max_heavy_atoms: 30"
            f"\n   - min_rotatable_bonds: {config.get('min_rotatable_bonds', 1)}"
            f"\n   - max_rotatable_bonds: {config.get('max_rotatable_bonds', 10)}"
            f"\n   - banned_atom_types: {config.get('banned_atom_types', [])}"
        )
        
        print("🔬 Importing BoltzWrapper...")
        boltz_imported = _import_boltz_wrapper()
        
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
        
        print("📂 Loading molecules from CSV and database...")
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
        
        top_1000_df = molecules_df.head(1000)
        
        state['top_pool'] = molecules_df.copy()
        state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
        state['top_1000_df'] = top_1000_df
        
        print(f"✅ Loaded {len(molecules_df)} molecules from CSV and database (top 1000: {len(top_1000_df)})")
        
        print(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Top 1000 molecules: {len(state['top_1000_df'])}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        print(f"Error in startup phase: {e}")
        import traceback
        print(traceback.format_exc())


async def run_continuous_bayesian_loop(state: Dict[str, Any]) -> None:
    """
    Continuous Bayesian Optimization loop:
    1. Generate 1000 molecules using Bayesian Optimization
    2. Validate them
    3. Score in batches of 10
    4. Update top pool
    5. Repeat infinitely
    """
    print("🚀 Starting CONTINUOUS BAYESIAN OPTIMIZATION loop...")
    
    desired_unique_count = 1000
    batch_size = 10
    round_number = 0
    
    # Exploration factor: starts high, decreases over time
    exploration_factor_start = 0.1  # More exploration initially
    exploration_factor_min = 0.01   # Less exploration later
    
    while not state['shutdown_event'].is_set():
        try:
            round_number += 1
            print(f"\n{'='*70}")
            print(f"🔄 Bayesian Optimization Round {round_number}")
            print(f"{'='*70}")
            
            top_1000_df = state.get('top_1000_df', pd.DataFrame())
            
            if top_1000_df.empty:
                print("No top molecules found, waiting...")
                await asyncio.sleep(10)
                continue
            
            # Adaptive exploration: decrease over rounds
            exploration_factor = max(
                exploration_factor_min,
                exploration_factor_start * (0.9 ** (round_number - 1))
            )
            
            print(f"Exploration factor (xi): {exploration_factor:.4f}")
            
            # ================================================================
            # BAYESIAN OPTIMIZATION
            # ================================================================
            print(f"🔬 Generating {desired_unique_count} molecules with Bayesian Optimization...")
            
            generated_molecules = await generate_molecules_bayesian(
                state=state,
                top_molecules_df=top_1000_df,
                desired_count=desired_unique_count,
                exploration_factor=exploration_factor
            )
            
            if not generated_molecules:
                print("Failed to generate molecules, retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue
            
            print(f"✅ Generated {len(generated_molecules)} molecules")
            
            # ================================================================
            # VALIDATION
            # ================================================================
            print(f"Validating {len(generated_molecules)} generated molecules...")
            
            validated_molecules = []
            validation_failures = {
                'smiles': 0,
                'heavy_atoms': 0,
                'banned_atoms': 0,
                'rotatable_bonds': 0,
                'hf_unique': 0,
                'other': 0
            }
            
            for mol in generated_molecules:
                is_valid, errors = await validate_molecule_complete(
                    state, mol['name'], mol['smiles'], state['config']
                )
                
                if is_valid:
                    validated_molecules.append(mol)
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
            
            print(
                f"Validation complete: {len(validated_molecules)}/{len(generated_molecules)} passed\n"
                f"   Failures: SMILES={validation_failures['smiles']}, "
                f"Heavy_atoms={validation_failures['heavy_atoms']}, "
                f"Banned_atoms={validation_failures['banned_atoms']}, "
                f"Rotatable_bonds={validation_failures['rotatable_bonds']}, "
                f"HF_unique={validation_failures['hf_unique']}, "
                f"Other={validation_failures['other']}"
            )
            
            if not validated_molecules:
                print("⚠️  No valid molecules after validation, retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue
            
            # ================================================================
            # SCORE MOLECULES
            # ================================================================
            print(f"🔬 Round {round_number}: Scoring {len(validated_molecules)} molecules in batches of {batch_size}...")
            
            scored_molecules = await score_molecules_with_boltz_batched(
                state, validated_molecules, batch_size=batch_size
            )
            
            # Filter molecules with valid scores
            molecules_with_scores = [m for m in scored_molecules if m.get('boltz_score') is not None]
            
            if molecules_with_scores:
                # Sort by score
                molecules_with_scores.sort(key=lambda m: m.get('boltz_score', float('-inf')), reverse=True)
                
                # Get best molecule
                best_molecule = molecules_with_scores[0]
                best_score = best_molecule.get('boltz_score')
                
                # Get statistics
                scores = [m['boltz_score'] for m in molecules_with_scores]
                avg_score = np.mean(scores)
                median_score = np.median(scores)
                std_score = np.std(scores)
                
                print(
                    f"\n{'='*70}"
                    f"\n🏆 Round {round_number} Results:"
                    f"\n   Best molecule: {best_molecule['name']}"
                    f"\n   Best score: {best_score:.6f}"
                    f"\n   SMILES: {best_molecule.get('smiles', 'N/A')}"
                    f"\n   "
                    f"\n   Score statistics:"
                    f"\n   - Mean: {avg_score:.6f}"
                    f"\n   - Median: {median_score:.6f}"
                    f"\n   - Std: {std_score:.6f}"
                    f"\n   - Min: {min(scores):.6f}"
                    f"\n   - Max: {max(scores):.6f}"
                    f"\n{'='*70}\n"
                )
                
                # Update state with best molecule
                if state.get('best_score', float('-inf')) < best_score:
                    state['best_molecule'] = best_molecule
                    state['best_score'] = best_score
                    print(f"🎉 NEW BEST SCORE: {best_score:.6f} (previous: {state.get('best_score', 0):.6f})")
                
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
                    combined_df = pd.concat([state['top_1000_df'], new_df], ignore_index=True)
                    
                    # Deduplicate by InChIKey, keeping highest score
                    combined_df = combined_df.sort_values(by='score', ascending=False, na_position='last')
                    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')
                    
                    # Keep top 1000
                    state['top_1000_df'] = combined_df.head(1000)
                    
                    # Also update top_pool
                    state['top_pool'] = combined_df.copy()
                    
                    print(f"✅ Updated top pool: {len(state['top_1000_df'])} molecules")
                    
                    # Show score distribution in top pool
                    top_scores = state['top_1000_df']['score'].dropna()
                    if len(top_scores) > 0:
                        print(
                            f"   Top pool score range: {top_scores.min():.6f} to {top_scores.max():.6f} "
                            f"(mean: {top_scores.mean():.6f})"
                        )
            else:
                print("⚠️  No molecules with valid scores in this round")
            
            print(f"✅ Round {round_number} complete. Starting next round...")
            
            # Small delay before next round
            await asyncio.sleep(1)
        
        except Exception as e:
            print(f"Error in continuous Bayesian loop: {e}")
            import traceback
            print(traceback.format_exc())
            await asyncio.sleep(10)


async def run_generator() -> None:
    """Main generation loop."""
    print("🚀 Starting BAYESIAN OPTIMIZATION molecule generator...")
    
    # Load config from config.yaml
    config_dict = load_config()
    
    # Get target protein from config
    target_protein = config_dict.get('weekly_target', 'KRAS')
    
    print(f"Configuration loaded from config.yaml:")
    print(f"  Target protein: {target_protein}")
    print(f"  Reaction ID: {HARDCODED_RXN_ID}")
    print(f"  Database: {DB_PATH}")
    print(f"  Score DB: {SCORE_RESULTS_DB}")
    print(f"  Generation per round: 1000 molecules")
    print(f"  Batch size: 10 molecules")
    print(f"  Method: Bayesian Optimization with Gaussian Process")
    
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
        'top_1000_df': pd.DataFrame(),
        'best_molecule': None,
        'best_score': float('-inf'),
    }
    
    print("🚀 Entering main generator loop...")
    
    # Run startup phase
    await startup_phase(state)
    
    # Run continuous Bayesian optimization loop
    await run_continuous_bayesian_loop(state)


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

