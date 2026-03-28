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
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 21632
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
from molecules_base import generate_inchikey, SynthonLibrary, generate_molecules_from_synthon_library
from combinatorial_db.reactions import get_smiles_from_reaction

BOLTZ_AVAILABLE = False
BoltzWrapper = None

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
        # min_atoms = config.get('min_heavy_atoms', 10)
        min_atoms = 14
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
            print(f"❌ Molecule {molecule_name} already in HuggingFace dataset")
            return False
        
        print(f"✅ Molecule {molecule_name} is NOT in HuggingFace (unique!)")
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


def safe_torch_load(path, map_location='cpu'):
    """Safely load PyTorch checkpoint with numpy scalar support."""
    import torch
    import numpy as np
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    print(f"Loading checkpoint from {path}...")
    
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )
        print(f"✅ Checkpoint loaded successfully")
        return checkpoint
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


class HybridMoleculeGenerator:
    """Generates molecules using hierarchical synthon search + fallback crossover."""
    
    def __init__(self, rxn_id: int, db_path: str):
        """Initialize hybrid generator."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()
        self.synthon_lib: Optional[SynthonLibrary] = None
        self.synthon_lib_ready = False
    
    def initialize_synthon_library(self) -> bool:
        """
        Initialize SynthonLibrary for synthon search.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔬 Initializing SynthonLibrary for rxn_id={self.rxn_id}...")
            start_time = time.time()
            
            self.synthon_lib = SynthonLibrary(self.db_path, self.rxn_id)
            
            # Check if library has components
            has_components = (
                len(self.synthon_lib.fps_A) > 0 and 
                len(self.synthon_lib.fps_B) > 0
            )
            self.synthon_lib_ready = has_components
            
            elapsed = time.time() - start_time
            if self.synthon_lib_ready:
                print(
                    f"✅ SynthonLibrary initialized successfully in {elapsed:.2f}s "
                    f"({len(self.synthon_lib.fps_A)} A, {len(self.synthon_lib.fps_B)} B components)"
                )
            else:
                print(f"⚠️  SynthonLibrary loaded but has no components")
            
            return self.synthon_lib_ready
        
        except Exception as e:
            print(f"❌ Failed to initialize SynthonLibrary: {e}")
            import traceback
            print(traceback.format_exc())
            self.synthon_lib_ready = False
            return False

    def generate_synthon_molecules(
        self,
        top_pool_df: pd.DataFrame,
        desired_count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate molecules using hierarchical synthon search with NEW API.
        Uses generate_molecules_from_synthon_library() from molecules_base.py
        """
        if not self.synthon_lib_ready or self.synthon_lib is None:
            print("⚠️  SynthonLibrary not ready, skipping synthon generation")
            return []
        
        if top_pool_df.empty:
            print("⚠️  No top molecules available for synthon search")
            return []
        
        new_molecules = []
        
        try:
            print(
                f"🧬 Generating {desired_count} molecules using HIERARCHICAL SYNTHON SEARCH..."
            )
            
            # Get current max score for adaptive strategy
            current_max_score = top_pool_df['score'].max() if 'score' in top_pool_df.columns else None
            
            # Determine search strategy based on score
            has_very_high_score = current_max_score is not None and current_max_score > 0.015
            has_high_score = current_max_score is not None and current_max_score > 0.01
            
            # Define hierarchical search strategy
            if has_very_high_score:
                print(
                    f"🎯 Very high score detected ({current_max_score:.6f}), "
                    f"using TIGHT hierarchical strategy"
                )
                search_configs = [
                    {'tier': 'Tier 1 (Ultra-tight)', 'n_mols': 1, 'budget': 0.40, 'min_sim': 0.90, 'n_per': 50},
                    {'tier': 'Tier 2 (Tight)', 'n_mols': 5, 'budget': 0.30, 'min_sim': 0.80, 'n_per': 30},
                    {'tier': 'Tier 3 (Medium)', 'n_mols': 20, 'budget': 0.30, 'min_sim': 0.55, 'n_per': 15},
                ]
            elif has_high_score:
                print(
                    f"🎯 High score detected ({current_max_score:.6f}), "
                    f"using BALANCED hierarchical strategy"
                )
                search_configs = [
                    {'tier': 'Tier 1 (Tight)', 'n_mols': 5, 'budget': 0.40, 'min_sim': 0.80, 'n_per': 30},
                    {'tier': 'Tier 2 (Medium)', 'n_mols': 30, 'budget': 0.35, 'min_sim': 0.65, 'n_per': 20},
                    {'tier': 'Tier 3 (Broad)', 'n_mols': 50, 'budget': 0.25, 'min_sim': 0.40, 'n_per': 10},
                ]
            else:
                print(
                    f"🎯 Standard score, using BROAD hierarchical strategy"
                )
                search_configs = [
                    {'tier': 'Tier 1 (Medium)', 'n_mols': 20, 'budget': 0.50, 'min_sim': 0.65, 'n_per': 20},
                    {'tier': 'Tier 2 (Broad)', 'n_mols': 50, 'budget': 0.50, 'min_sim': 0.40, 'n_per': 10},
                ]
            
            # Execute hierarchical search using NEW API
            for config in search_configs:
                tier_name = config['tier']
                n_mols = config['n_mols']
                tier_budget = int(desired_count * config['budget'])
                min_sim = config['min_sim']
                n_per = config['n_per']
                
                print(
                    f"   {tier_name}: budget={tier_budget}, "
                    f"min_similarity={min_sim}, n_per_base={n_per}"
                )
                
                # Get top N molecules for this tier
                tier_molecules = top_pool_df.head(n_mols)
                
                # Use NEW API from molecules_base.py
                tier_df = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    tier_molecules,
                    n_samples=tier_budget,
                    min_similarity=min_sim,
                    n_per_base=n_per
                )
                
                tier_count = 0
                for _, row in tier_df.iterrows():
                    mol_name = row['name']
                    
                    if mol_name in self.generated_molecule_names:
                        continue
                    
                    try:
                        smiles = get_smiles_from_reaction(mol_name)
                        if not smiles:
                            continue
                        
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue
                        
                        inchikey = generate_inchikey(smiles)
                        if not inchikey:
                            continue
                        
                        new_molecules.append({
                            'name': mol_name,
                            'smiles': smiles,
                            'InChIKey': inchikey,
                            'type': 'synthon',
                            'similarity': None
                        })
                        
                        self.generated_molecule_names.add(mol_name)
                        tier_count += 1
                        
                        if tier_count >= tier_budget:
                            break
                    
                    except Exception as e:
                        print(f"Error processing synthon molecule: {e}")
                        continue
                
                print(f"   {tier_name}: Generated {tier_count}/{tier_budget} molecules")
            
            print(
                f"✅ Synthon search complete: {len(new_molecules)} molecules generated"
            )
            return new_molecules
        
        except Exception as e:
            print(f"❌ Error in synthon generation: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """
        Crossover two molecules by swapping random components.
        
        Supports formats:
        - 2 components: rxn:5:comp1:comp2 (4 parts)
        - 3 components: rxn:5:comp1:comp2:comp3 (5 parts)
        
        Args:
            mol_name_1: First parent molecule name
            mol_name_2: Second parent molecule name
            
        Returns:
            New molecule name from crossover, or None if failed
        """
        try:
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            # Validate format
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
            
            # Randomly select component to swap
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
            
            # Validate offspring
            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        self.generated_molecule_names.add(offspring_name)
                        print(
                            f"✅ Crossover: {mol_name_1} × {mol_name_2} → {offspring_name}"
                        )
                        return offspring_name
            except Exception as e:
                print(f"Error validating crossover: {e}")
            
            return None
        
        except Exception as e:
            print(f"Error in crossover_molecules: {e}")
            return None
    
    def apply_hybrid_generation(
        self,
        top_molecules: List[str],
        top_pool_df: pd.DataFrame,
        desired_count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid generation: Try synthon search first, fallback to crossover.
        
        Strategy:
        1. Generate synthon molecules (primary method)
        2. If synthon count < desired_count, fill gap with crossover
        3. Continue until reaching desired_count or max attempts
        
        Args:
            top_molecules: List of top molecule names
            top_pool_df: DataFrame with top molecules
            desired_count: Target number of molecules
            
        Returns:
            List of new molecules (synthon + crossover combined)
        """
        new_molecules = []
        self.generated_molecule_names.clear()
        
        print(
            f"🧬 Applying HYBRID generation (synthon-first + fallback crossover)..."
        )
        print(f"   Target: {desired_count} total molecules")
        
        # PART 1: SYNTHON SEARCH (Primary)
        synthon_molecules = []
        if self.synthon_lib_ready:
            synthon_molecules = self.generate_synthon_molecules(
                top_pool_df,
                desired_count
            )
            new_molecules.extend(synthon_molecules)
            print(
                f"   ✅ Synthon: {len(synthon_molecules)}/{desired_count} molecules generated"
            )
        else:
            print(
                f"   ⚠️  SynthonLibrary not ready, using crossover for all molecules"
            )
        
        # PART 2: CROSSOVER FALLBACK (Fill gap)
        remaining_needed = desired_count - len(new_molecules)
        
        if remaining_needed > 0:
            print(
                f"   📊 Synthon generated {len(synthon_molecules)}, "
                f"need {remaining_needed} more molecules"
            )
            print(f"   🔄 Fallback: Generating {remaining_needed} molecules via CROSSOVER...")
            
            crossover_attempts = 0
            crossovers_created = 0
            max_crossover_attempts = remaining_needed * 5
            
            while crossovers_created < remaining_needed and crossover_attempts < max_crossover_attempts:
                parent1 = random.choice(top_molecules)
                parent2 = random.choice(top_molecules)
                crossover_attempts += 1
                
                if parent1 != parent2:
                    offspring = self.crossover_molecules(parent1, parent2)
                    
                    if offspring:
                        try:
                            smiles = get_smiles_from_reaction(offspring)
                            inchikey = generate_inchikey(smiles)
                            
                            if smiles and inchikey:
                                new_molecules.append({
                                    'name': offspring,
                                    'smiles': smiles,
                                    'InChIKey': inchikey,
                                    'type': 'crossover'
                                })
                                crossovers_created += 1
                        
                        except Exception as e:
                            print(f"Error processing crossover: {e}")
            
            print(
                f"   ✅ Crossover: {crossovers_created}/{remaining_needed} molecules generated "
                f"(attempts: {crossover_attempts})"
            )
        else:
            print(f"   ✅ Synthon search met target, no crossover needed")
        
        print(
            f"🧬 HYBRID generation complete: {len(new_molecules)} total molecules "
            f"({len(synthon_molecules)} synthon + {len(new_molecules) - len(synthon_molecules)} crossover)"
        )
        
        return new_molecules


def init_score_results_db(db_path: str = None) -> None:
    """
    Initialize/create the score_results.sqlite database.
    
    Creates a table with molecule_name and score fields.
    """
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
        
        # Create index on score for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)
        
        conn.commit()
        conn.close()
        print(f"Initialized score_results database at {db_path}")
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
        print(f"Error getting score from DB for {molecule_name}: {e}")
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
                "INSERT INTO scored_molecules (molecule_name, score, available) VALUES (?, ?, ?)",
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
        print(f"Error batch getting scores from DB: {e}")
        return {}


def load_molecules_from_db_with_validation(
    db_path: str,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from SQLite database with validation from config.yaml."""
    if config is None:
        config = {}
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        print(
            f"Loading molecules from database {db_path} for rxn_id={rxn_id}"
        )
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query all scored molecules
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
                # Filter by rxn_id
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
                
                # Check banned atoms
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    banned_atom_count += 1
                    continue
                
                # Check heavy atom count
                # min_heavy_atoms = config.get('min_heavy_atoms', 10)
                min_heavy_atoms = 14    
                max_heavy_atoms = 30
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    heavy_atom_count += 1
                    continue
                if heavy_atom_count_val > max_heavy_atoms:
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
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                    f"wrong rxn_id: {wrong_rxn_id_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        print(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            print(
                f"No valid molecules loaded from database "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                f"wrong rxn_id: {wrong_rxn_id_count})"
            )
        
        return result_df
        
    except Exception as e:
        print(f"Error loading molecules from database: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_csv_with_validation(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from CSV file with validation from config.yaml."""
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
        banned_atom_count = 0
        heavy_atom_count = 0
        
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
                    banned_atom_count += 1
                    continue
                
                # Check heavy atom count
                # min_heavy_atoms = config.get('min_heavy_atoms', 10)
                min_heavy_atoms = 14
                max_heavy_atoms = 30
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    heavy_atom_count += 1
                    continue
                if heavy_atom_count_val > max_heavy_atoms:
                    heavy_atom_count += 1
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
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        print(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            print(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
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


async def load_submissions_from_csv(
    state: Dict[str, Any],
    csv_path: str,
    start_epoch: int,
    rxn_id: int,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Load submissions from both CSV and database, merge them, and return top 200.
    
    Args:
        state: State dictionary
        csv_path: Path to CSV file
        start_epoch: Minimum epoch to include
        rxn_id: Reaction ID to filter
        config: Configuration dictionary
        
    Returns:
        DataFrame with top 200 molecules
    """
    try:
        target_proteins = state.get('current_challenge_targets', [])
        molecules_df = load_molecules_combined(
            csv_path,
            SCORE_RESULTS_DB,
            target_proteins,
            start_epoch,
            rxn_id,
            config
        )
        
        if molecules_df.empty:
            print("⚠️  No valid molecules loaded from CSV or database")
            return pd.DataFrame()
        
        # top_200 = molecules_df(500)
        top_200 = molecules_df[50:250]
        
        print(f"✅ Selected top 200 molecules from combined CSV and database")
        
        if len(top_200) > 0 and 'score' in top_200.columns:
            scores = top_200['score'].dropna()
            if len(scores) > 0:
                print(
                    f"   Top 200 score range: {scores.min():.6f} to {scores.max():.6f} "
                    f"(top 3: {scores.head(3).tolist()})"
                )
        
        return top_200
        
    except Exception as e:
        print(f"Error loading molecules from CSV and database: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()


async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Score molecules using BoltzWrapper in batches.
    
    Args:
        state: State dictionary
        molecules: List of molecules to score
        batch_size: Number of molecules per batch (default 10)
        
    Returns:
        List of scored molecules
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
                print(f"   ✓ {molecule_name}: score from DB = {db_scores[molecule_name]:.6f}")
                continue
            
            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                    if not is_unique_hf:
                        print(f"   ⏭️  {molecule_name}: already in HuggingFace, skipping")
                        molecules_in_hf.append(mol)
                        continue
                except Exception as e:
                    print(f"   Error checking HuggingFace for {molecule_name}: {e}")
            
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
                            print(f"Cleaned up old lightning_logs directory")
                    except Exception as cleanup_err:
                        print(f"Could not clean up old logs: {cleanup_err}")
                
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
                
                print(f"   Running Boltz scoring for {len(molecules_to_score)} molecules...")
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
                    print(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores")
                
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
    """Import BoltzWrapper following DataGenerator pattern."""
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


async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Initialize score_results database
    2. Import and initialize BoltzWrapper
    3. Load molecules from CSV and database with validation
    4. Prepare top_pool
    """
    print("🚀 Starting STARTUP phase: Initialize DB, Boltz, & Load CSV/DB...")
    
    try:
        # Initialize score_results database
        print("💾 Initializing score_results database...")
        init_score_results_db()
        print(f"✅ Score results database initialized")
        
        # Initialize hybrid generator
        print("🧬 Initializing HybridMoleculeGenerator...")
        hybrid_generator = HybridMoleculeGenerator(HARDCODED_RXN_ID, DB_PATH)
        
        # Try to initialize synthon library
        synthon_ready = hybrid_generator.initialize_synthon_library()
        
        if synthon_ready:
            print("✅ Synthon library ready for hybrid generation")
        else:
            print("⚠️  Synthon library not available, will use crossover fallback")
        
        # Store generator in state
        state['hybrid_generator'] = hybrid_generator

        # Log validation config
        config = state['config']
        print(
            f"✅ Loaded validation config from config.yaml:"
            f"\n   - min_heavy_atoms: {config.get('min_heavy_atoms', 10)}"
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
        
        # Load submissions from both CSV and database
        print("📂 Loading submissions from CSV and database...")
        top_200_df = await load_submissions_from_csv(
            state,
            REACTION_TRAIN_CSV,
            STARTING_EPOCH,
            HARDCODED_RXN_ID,
            config
        )
        
        # Store top_200_df in state
        state['top_200_df'] = top_200_df
        
        if top_200_df.empty:
            print("⚠️  No molecules loaded from CSV or database")
            return
        
        print(f"✅ Loaded {len(top_200_df)} top molecules from CSV and database")
        
        # Update state with top molecules
        state['top_pool'] = top_200_df.copy()
        state['seen_inchikeys'].update(top_200_df['InChIKey'].tolist())
        
        print(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Top 200 molecules: {len(state['top_200_df'])}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        print(f"Error in startup phase: {e}")
        import traceback
        print(traceback.format_exc())


async def generate_unique_molecules_from_top200(
    state: Dict[str, Any],
    top_200_df: pd.DataFrame,
    hybrid_generator: HybridMoleculeGenerator,
    desired_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules using hybrid generator (synthon-first + crossover fallback).
    
    Args:
        state: State dictionary
        top_200_df: DataFrame with top 200 molecules
        hybrid_generator: Initialized HybridMoleculeGenerator instance
        desired_count: Target number of molecules to generate
        
    Returns:
        List of validated unique molecules
    """
    if top_200_df.empty:
        print("Top 200 DataFrame is empty")
        return []
    
    all_names = top_200_df['name'].tolist()
    
    print(
        f"🧬 Generating {desired_count} unique molecules with validation "
        f"(hybrid: synthon-first + crossover fallback)..."
    )
    
    unique_molecules = []
    attempts = 0
    max_attempts = 500
    last_successful_attempt = 0
    
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    validation_stats = {
        'total_generated': 0,
        'passed_validation': 0,
        'failed_smiles': 0,
        'failed_heavy_atoms': 0,
        'failed_banned_atoms': 0,
        'failed_rotatable_bonds': 0,
        'failed_hf_unique': 0,
        'failed_other': 0,
        'synthon_generated': 0,
        'crossover_generated': 0,
    }
    
    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1
        
        # Apply hybrid generation
        new_molecules = hybrid_generator.apply_hybrid_generation(
            all_names,
            top_200_df,
            desired_count
        )
        
        for mol in new_molecules:
            if len(unique_molecules) >= desired_count:
                break
            
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            mol_type = mol.get('type', 'unknown')
            
            validation_stats['total_generated'] += 1
            if mol_type == 'synthon':
                validation_stats['synthon_generated'] += 1
            elif mol_type == 'crossover':
                validation_stats['crossover_generated'] += 1
            
            # Check if already generated
            if molecule_name in [m['name'] for m in unique_molecules]:
                continue
            
            if molecule_name in generated_molecules:
                print(f"   ⏭️  Molecule {molecule_name} already generated")
                continue
            
            # Generate InChIKey
            inchikey = None
            try:
                inchikey = generate_inchikey(smiles) if smiles else None
                if inchikey and inchikey in generated_inchikeys:
                    print(
                        f"   ⏭️  Molecule {molecule_name} (InChIKey: {inchikey}) already generated"
                    )
                    continue
            except Exception as e:
                print(f"   Could not generate InChIKey for {molecule_name}: {e}")
                validation_stats['failed_other'] += 1
                continue
            
            # Validate molecule
            is_valid, errors = await validate_molecule_complete(
                state,
                molecule_name,
                smiles,
                state['config']
            )
            
            if not is_valid:
                for error in errors:
                    print(f"   ❌ {molecule_name}: {error}")
                    if "[SMILES]" in error:
                        validation_stats['failed_smiles'] += 1
                    elif "[HEAVY_ATOMS]" in error:
                        validation_stats['failed_heavy_atoms'] += 1
                    elif "[BANNED_ATOMS]" in error:
                        validation_stats['failed_banned_atoms'] += 1
                    elif "[ROTATABLE_BONDS]" in error:
                        validation_stats['failed_rotatable_bonds'] += 1
                    elif "[HF_UNIQUE]" in error:
                        validation_stats['failed_hf_unique'] += 1
                    else:
                        validation_stats['failed_other'] += 1
                continue
            
            # Add to unique molecules
            unique_molecules.append(mol)
            generated_molecules.add(molecule_name)
            if inchikey:
                generated_inchikeys.add(inchikey)
            last_successful_attempt = attempts
            validation_stats['passed_validation'] += 1
            
            print(
                f"   ✅ Added valid molecule {molecule_name} "
                f"({mol_type}, {len(unique_molecules)}/{desired_count})"
            )
        
        if len(unique_molecules) >= desired_count:
            break
        
        await asyncio.sleep(0.1)
    
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    print(
        f"✅ Generated {len(unique_molecules)} valid molecules (attempts: {attempts})"
        f"\n   Validation stats:"
        f"\n   - Total generated: {validation_stats['total_generated']}"
        f"\n   - Synthon: {validation_stats['synthon_generated']}"
        f"\n   - Crossover: {validation_stats['crossover_generated']}"
        f"\n   - Passed validation: {validation_stats['passed_validation']}"
        f"\n   - Failed SMILES: {validation_stats['failed_smiles']}"
        f"\n   - Failed heavy atoms: {validation_stats['failed_heavy_atoms']}"
        f"\n   - Failed banned atoms: {validation_stats['failed_banned_atoms']}"
        f"\n   - Failed rotatable bonds: {validation_stats['failed_rotatable_bonds']}"
        f"\n   - Failed HF uniqueness: {validation_stats['failed_hf_unique']}"
        f"\n   - Failed other: {validation_stats['failed_other']}"
    )
    
    return unique_molecules


async def run_continuous_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Continuous genetic algorithm loop:
    1. Generate 100 unique molecules
    2. Score them in batches of 10
    3. Save best molecule
    4. Repeat infinitely until user stops
    """
    print("🚀 Starting CONTINUOUS genetic algorithm loop with batch scoring...")
    
    desired_unique_count = 100
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
                print("No top 200 molecules found, waiting...")
                await asyncio.sleep(10)
                continue
            
            # Generate unique molecules
            print(f"🧬 Generating {desired_unique_count} unique molecules with validation...")
            unique_molecules = await generate_unique_molecules_from_top200(
                state, top_200_df, state['hybrid_generator'], desired_unique_count
            )
            
            if not unique_molecules:
                print("Failed to generate unique molecules, retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue
            
            print(f"✅ Generated {len(unique_molecules)} valid unique molecules")
            
            # Score molecules in batches
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
                    f"\n{'='*70}\n"
                )
                
                # Update state with best molecule
                state['best_molecule'] = best_molecule
                state['best_score'] = best_score
                
                # Update top_200_df with new scored molecules
                # Add new molecules to top_200_df and keep top 200
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
                    
                    # Keep top 200
                    # state['top_200_df'] = combined_df.head(200)
                    state['top_200_df'] = combined_df[50:250]
                    
                    print(f"✅ Updated top_200_df: {len(state['top_200_df'])} molecules")
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
    print("🚀 Starting molecule generator...")
    
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
        'hybrid_generator': None,
    }
    
    print("🚀 Entering main generator loop...")
    
    # Run startup phase
    await startup_phase(state)
    
    # Run continuous genetic loop
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

