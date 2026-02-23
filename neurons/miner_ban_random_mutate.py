import os
import sys
import random
import argparse
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
import bittensor as bt
from bittensor.core.errors import MetadataError
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 21074
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'mols.csv')
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "..", "nova-4090", "score_results.sqlite")

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
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
from btdr import QuicknetBittensorDrandTimelock

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
        min_atoms = config.get('min_heavy_atoms', 10)
        
        if heavy_atom_count < min_atoms:
            return False, f"Insufficient heavy atoms: {heavy_atom_count} < {min_atoms}"
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
        bt.logging.warning("No target proteins available")
        return False
    
    primary_target = state['current_challenge_targets'][0]
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            bt.logging.debug(f"❌ Molecule {molecule_name} already in HuggingFace dataset")
            return False
        
        bt.logging.info(f"✅ Molecule {molecule_name} is NOT in HuggingFace (unique!)")
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
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
    
    bt.logging.info(f"Loading checkpoint from {path}...")
    
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )
        bt.logging.info(f"✅ Checkpoint loaded successfully")
        return checkpoint
    except Exception as e:
        bt.logging.error(f"❌ Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e

class GeneticAlgorithmOperator:
    """Performs genetic algorithm operations on molecules (CROSSOVER + MUTATION)."""
    
    def __init__(self, rxn_id: int, db_path: str):
        """Initialize GA operator."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()
        
        # Load reaction info and component pools for mutation
        self.reaction_info = get_reaction_info(rxn_id, db_path)
        if self.reaction_info:
            self.smarts, self.roleA, self.roleB, self.roleC = self.reaction_info
            self.is_three_component = self.roleC is not None and self.roleC != 0
            
            # Load component pools for mutation
            self.molecules_A = get_molecules_by_role(self.roleA, db_path)
            self.molecules_B = get_molecules_by_role(self.roleB, db_path)
            self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
            
            # Extract component IDs for each role
            self.component_ids_A = [mol[0] for mol in self.molecules_A] if self.molecules_A else []
            self.component_ids_B = [mol[0] for mol in self.molecules_B] if self.molecules_B else []
            self.component_ids_C = [mol[0] for mol in self.molecules_C] if self.molecules_C else []
            
            bt.logging.debug(
                f"GA Operator initialized: {len(self.component_ids_A)} A, {len(self.component_ids_B)} B"
                f"{f', {len(self.component_ids_C)} C' if self.is_three_component else ''} components"
            )
        else:
            bt.logging.warning(f"Could not load reaction info for rxn_id {rxn_id}, mutation will be disabled")
            self.reaction_info = None
            self.component_ids_A = []
            self.component_ids_B = []
            self.component_ids_C = []
            self.is_three_component = False
    
    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """
        Crossover two molecules by swapping random components.
        
        Supports two formats:
        - 2 components: rxn:5:comp1:comp2 (4 parts)
        - 3 components: rxn:5:comp1:comp2:comp3 (5 parts)
        
        Both parents must have the same number of components.
        We randomly swap one component between the two molecules.
        """
        try:
            from rdkit import Chem
            
            # Parse molecule names
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            bt.logging.debug(f"Crossover: {mol_name_1} x {mol_name_2}")
            
            # Validate format: must start with 'rxn' and have same length
            if (parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                bt.logging.debug(f"Invalid format: must start with 'rxn'")
                return None
            
            if len(parts1) != len(parts2):
                bt.logging.debug(f"Invalid format: different number of components ({len(parts1)} vs {len(parts2)})")
                return None
            
            if len(parts1) not in [4, 5]:
                bt.logging.debug(f"Invalid format: expected 4 or 5 parts, got {len(parts1)}")
                return None
            
            try:
                rxn_id_1 = int(parts1[1])
                rxn_id_2 = int(parts2[1])
                if rxn_id_1 != self.rxn_id or rxn_id_2 != self.rxn_id:
                    bt.logging.debug(f"Wrong rxn_ids: {rxn_id_1}, {rxn_id_2}")
                    return None
            except (ValueError, IndexError) as e:
                bt.logging.debug(f"Error parsing rxn_ids: {e}")
                return None
            
            # Determine which component indices can be swapped
            num_components = len(parts1) - 2
            component_indices = list(range(2, 2 + num_components))
            
            # Randomly select which component to swap
            swap_idx = random.choice(component_indices)
            
            # Create offspring by swapping one component
            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)
            
            # Check if already generated
            if offspring_name in self.generated_molecule_names:
                bt.logging.debug(f"⚠️  Offspring {offspring_name} already generated in this batch")
                return None
            
            # Validate offspring
            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        self.generated_molecule_names.add(offspring_name)
                        bt.logging.info(f"✅ Crossover successful: {mol_name_1} × {mol_name_2} → {offspring_name}")
                        return offspring_name
                    else:
                        bt.logging.debug(f"Invalid SMILES from RDKit: {offspring_smiles}")
                else:
                    bt.logging.debug(f"No SMILES generated for offspring")
            except Exception as e:
                bt.logging.debug(f"Error validating crossover: {e}")
            
            return None
        
        except Exception as e:
            bt.logging.debug(f"Error in crossover_molecules: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return None
    
    def mutate_molecule(self, mol_name: str) -> Optional[str]:
        """
        Mutate a molecule by replacing a random component with a random component from the same role.
        
        Supports two formats:
        - 2 components: rxn:5:comp1:comp2 (4 parts) - components at indices 2, 3
        - 3 components: rxn:5:comp1:comp2:comp3 (5 parts) - components at indices 2, 3, 4
        """
        try:
            from rdkit import Chem
            
            # Check if mutation is available
            if not self.reaction_info or not self.component_ids_A or not self.component_ids_B:
                bt.logging.debug("Mutation not available: reaction info or component pools not loaded")
                return None
            
            # Parse molecule name
            parts = mol_name.split(':')
            
            bt.logging.debug(f"Mutation: {mol_name}")
            
            # Validate format: must start with 'rxn'
            if parts[0] != 'rxn':
                bt.logging.debug(f"Invalid format: must start with 'rxn'")
                return None
            
            if len(parts) not in [4, 5]:
                bt.logging.debug(f"Invalid format: expected 4 or 5 parts, got {len(parts)}")
                return None
            
            try:
                rxn_id = int(parts[1])
                if rxn_id != self.rxn_id:
                    bt.logging.debug(f"Wrong rxn_id: {rxn_id}")
                    return None
            except (ValueError, IndexError) as e:
                bt.logging.debug(f"Error parsing rxn_id: {e}")
                return None
            
            # Determine which component indices can be mutated
            num_components = len(parts) - 2
            component_indices = list(range(2, 2 + num_components))
            
            # Randomly select which component to mutate
            mutate_idx = random.choice(component_indices)
            component_position = mutate_idx - 2  # 0 for A, 1 for B, 2 for C
            
            bt.logging.debug(f"Mutating component at index {mutate_idx} (position {component_position})")
            
            # Get the appropriate component pool based on position
            if component_position == 0:  # Component A
                component_pool = self.component_ids_A
                current_component_id = int(parts[mutate_idx])
            elif component_position == 1:  # Component B
                component_pool = self.component_ids_B
                current_component_id = int(parts[mutate_idx])
            elif component_position == 2 and self.is_three_component:  # Component C
                component_pool = self.component_ids_C
                current_component_id = int(parts[mutate_idx])
            else:
                bt.logging.debug(f"Invalid component position: {component_position}")
                return None
            
            if not component_pool:
                bt.logging.debug(f"No components available for position {component_position}")
                return None
            
            # Select a random component from the pool (excluding current one to ensure mutation)
            available_components = [cid for cid in component_pool if cid != current_component_id]
            if not available_components:
                bt.logging.debug(f"No alternative components available for position {component_position}")
                return None
            
            new_component_id = random.choice(available_components)
            
            # Create mutated molecule by replacing the component
            mutated_parts = parts.copy()
            mutated_parts[mutate_idx] = str(new_component_id)
            mutated_name = ':'.join(mutated_parts)
            
            bt.logging.debug(f"Mutated: {mutated_name}")
            
            # Check if already generated
            if mutated_name in self.generated_molecule_names:
                bt.logging.debug(f"⚠️  Mutated molecule {mutated_name} already generated in this batch")
                return None
            
            # Validate mutated molecule
            try:
                mutated_smiles = get_smiles_from_reaction(mutated_name)
                if mutated_smiles:
                    mol = Chem.MolFromSmiles(mutated_smiles)
                    if mol is not None:
                        self.generated_molecule_names.add(mutated_name)
                        bt.logging.info(f"✅ Mutation successful: {mol_name} → {mutated_name} (replaced component {component_position})")
                        return mutated_name
                    else:
                        bt.logging.debug(f"Invalid SMILES from RDKit: {mutated_smiles}")
                else:
                    bt.logging.debug(f"No SMILES generated for mutated molecule")
            except Exception as e:
                bt.logging.debug(f"Error validating mutation: {e}")
            
            return None
        
        except Exception as e:
            bt.logging.debug(f"Error in mutate_molecule: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return None
    
    def apply_genetic_operations(
        self,
        top_molecules: List[str],
        num_crossovers: int = 5,
        num_mutations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Apply genetic operations (CROSSOVER + MUTATION) to top molecules.
        
        Args:
            top_molecules: List of top molecule names
            num_crossovers: Number of crossovers to attempt
            num_mutations: Number of mutations to attempt
            
        Returns:
            List of new molecules with their SMILES and names (in order generated)
        """
        new_molecules = []
        
        # Reset tracking for this batch
        self.generated_molecule_names.clear()
        
        bt.logging.info(f"🧬 Applying genetic operations (CROSSOVER + MUTATION) to top {len(top_molecules)} molecules...")
        bt.logging.info(f"   Sample molecules: {top_molecules[:3]}")
        bt.logging.info(f"   Operations: {num_crossovers} crossovers, {num_mutations} mutations")
        
        # Apply crossovers
        crossovers_created = 0
        
        for i in range(num_crossovers):
            parent1 = random.choice(top_molecules)
            parent2 = random.choice(top_molecules)
            
            bt.logging.info(f"   Attempting crossover {i+1}/{num_crossovers}: {parent1} x {parent2}")
            
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
                            bt.logging.info(f"   ✅ Crossover #{crossovers_created}: {parent1} × {parent2} → {offspring}")
                    
                    except Exception as e:
                        bt.logging.debug(f"Error processing offspring: {e}")
                else:
                    bt.logging.debug(f"   ❌ Crossover {i+1} failed (duplicate or invalid)")
            else:
                bt.logging.debug(f"   ⏭️  Crossover {i+1}: parents are identical, skipping")
        
        bt.logging.info(f"   Crossovers: {crossovers_created}/{num_crossovers} successful")
        
        # Apply mutations
        mutations_created = 0
        
        for i in range(num_mutations):
            parent = random.choice(top_molecules)
            
            bt.logging.info(f"   Attempting mutation {i+1}/{num_mutations}: {parent}")
            
            mutated = self.mutate_molecule(parent)
            
            if mutated:
                try:
                    smiles = get_smiles_from_reaction(mutated)
                    inchikey = generate_inchikey(smiles)
                    
                    if smiles and inchikey:
                        new_molecules.append({
                            'name': mutated,
                            'smiles': smiles,
                            'InChIKey': inchikey,
                            'type': 'mutation'
                        })
                        mutations_created += 1
                        bt.logging.info(f"   ✅ Mutation #{mutations_created}: {parent} → {mutated}")
                
                except Exception as e:
                    bt.logging.debug(f"Error processing mutated molecule: {e}")
            else:
                bt.logging.debug(f"   ❌ Mutation {i+1} failed (duplicate or invalid)")
        
        bt.logging.info(f"   Mutations: {mutations_created}/{num_mutations} successful")
        bt.logging.info(f"🧬 Generated {len(new_molecules)} new molecules ({crossovers_created} crossovers, {mutations_created} mutations)")
        
        return new_molecules

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    
    config = bt.config(parser)
    config.update(load_config())
    
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )
    
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """Constructs the path for GitHub operations."""
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')
    
    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing GitHub environment variables")
    
    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"
    
    if len(github_path) > 100:
        raise ValueError("GitHub path too long (max 100 chars)")
    
    return github_path


def setup_logging(config: argparse.Namespace) -> None:
    """Sets up Bittensor logging."""
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid}")

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """Initializes wallet, subtensor, and metagraph with retry logic."""
    bt.logging.info("Setting up Bittensor objects.")
    
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")
    
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            bt.logging.info(f"Attempting to connect to Bittensor network (attempt {attempt + 1}/{max_retries})...")
            
            subtensor = bt.async_subtensor(network=config.network)
            
            async with subtensor:
                metagraph = await subtensor.metagraph(config.netuid)
                await metagraph.sync()
                bt.logging.info(f"Metagraph synced successfully.")
                
                miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
                bt.logging.info(f"Miner UID: {miner_uid}")
                
                epoch_length = 361
                bt.logging.info(f"Epoch length: {epoch_length} blocks")
            
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            
            return wallet, subtensor, metagraph, miner_uid, epoch_length
            
        except (ConnectionError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                bt.logging.warning(
                    f"Connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
            else:
                bt.logging.error(f"Failed to connect after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            bt.logging.error(f"Unexpected error during connection: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                raise

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
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)
        
        conn.commit()
        conn.close()
        bt.logging.debug(f"Initialized score_results database at {db_path}")
    except Exception as e:
        bt.logging.error(f"Error initializing score_results database: {e}")


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
        bt.logging.debug(f"Error getting score from DB for {molecule_name}: {e}")
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
                # Add True for available column
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
        bt.logging.debug(f"Error batch getting scores from DB: {e}")
        return {}

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
        bt.logging.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        bt.logging.info(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            bt.logging.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            bt.logging.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if df.empty:
            bt.logging.info("No matching molecules found in CSV")
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
                    bt.logging.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    bt.logging.debug(f"Cannot parse SMILES for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Check banned atoms
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    bt.logging.debug(f"Molecule {molecule_name} contains banned atoms {banned_atoms}, skipping")
                    banned_atom_count += 1
                    continue
                
                # Check heavy atom count
                min_heavy_atoms = config.get('min_heavy_atoms', 10)
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    bt.logging.debug(f"Molecule {molecule_name} has insufficient heavy atoms ({heavy_atom_count_val} < {min_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    bt.logging.debug(f"Could not generate InChIKey for {molecule_name}")
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
                bt.logging.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                bt.logging.info(
                    f"✅ Loaded {len(result_df)} molecules from CSV "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        bt.logging.info(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            bt.logging.warning(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
            )
        
        return result_df
        
    except Exception as e:
        bt.logging.error(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Score molecules using BoltzWrapper in batches.
    
    Args:
        state: Miner state dictionary
        molecules: List of molecules to score
        batch_size: Number of molecules per batch (default 10)
        
    Returns:
        Tuple of (scored_molecules, should_submit_flag)
    """
    if state.get('boltz_wrapper') is None:
        bt.logging.warning("BoltzWrapper not available, skipping scoring")
        return molecules, False
    
    if not molecules:
        return molecules, False
    
    bt.logging.info(f"🔬 Processing {len(molecules)} molecules for scoring in batches of {batch_size}...")
    
    init_score_results_db()
    
    # Calculate epoch info
    current_block = await state['subtensor'].get_current_block()
    current_epoch = current_block // state['epoch_length']
    next_epoch_block = (current_epoch + 1) * state['epoch_length']
    
    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    
    # Process molecules in batches
    for batch_idx in range(total_batches):
        # Check blocks remaining before each batch
        current_block = await state['subtensor'].get_current_block()
        blocks_remaining = next_epoch_block - current_block
        
        if blocks_remaining < 50:
            bt.logging.warning(
                f"⏰ Only {blocks_remaining} blocks until next epoch, "
                f"stopping batch scoring and returning for submission"
            )
            return all_scored_molecules, True
        
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]
        
        bt.logging.info(
            f"📦 Batch {batch_idx + 1}/{total_batches}: "
            f"Scoring {len(batch)} molecules "
            f"(blocks remaining: {blocks_remaining})"
        )
        
        # Score this batch
        molecules_to_score = []
        molecules_with_db_scores = []
        molecules_in_hf = []
        
        target_proteins = state.get('current_challenge_targets', [])
        primary_target = target_proteins[0] if target_proteins else None
        
        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)
        
        bt.logging.info(f"   Found {len(db_scores)} molecules already in database")
        
        # Separate molecules by source
        for mol in batch:
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
            if molecule_name in db_scores:
                mol['boltz_score'] = db_scores[molecule_name]
                mol['boltz_score_source'] = 'database'
                molecules_with_db_scores.append(mol)
                bt.logging.debug(f"   ✓ {molecule_name}: score from DB = {db_scores[molecule_name]:.6f}")
                continue
            
            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                    if not is_unique_hf:
                        bt.logging.debug(f"   ⏭️  {molecule_name}: already in HuggingFace, skipping")
                        molecules_in_hf.append(mol)
                        continue
                except Exception as e:
                    bt.logging.debug(f"   Error checking HuggingFace for {molecule_name}: {e}")
            
            molecules_to_score.append(mol)
        
        bt.logging.info(
            f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
            f"{len(molecules_in_hf)} in HuggingFace (skipped), "
            f"{len(molecules_to_score)} need scoring"
        )
        
        newly_scored_molecules = []
        if molecules_to_score:
            bt.logging.info(f"   Scoring {len(molecules_to_score)} new molecules with Boltz...")
            
            boltz = state['boltz_wrapper']
            config = state['config']
            target_proteins = state.get('current_challenge_targets', [])
            antitarget_proteins = state.get('current_challenge_antitargets', [])
            
            if not target_proteins:
                bt.logging.warning("No target proteins available for scoring")
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
                            bt.logging.debug(f"Cleaned up old lightning_logs directory")
                    except Exception as cleanup_err:
                        bt.logging.debug(f"Could not clean up old logs: {cleanup_err}")
                
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
                    'binding_pocket': getattr(config, 'binding_pocket', None),
                    'max_distance': getattr(config, 'max_distance', None),
                    'force': getattr(config, 'force', False),
                    'num_molecules_boltz': num_molecules_to_score,
                    'boltz_metric': getattr(config, 'boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']),
                    'combination_strategy': getattr(config, 'combination_strategy', 'heavy_atom_normalization'),
                    'sample_selection': getattr(config, 'sample_selection', 'first'),
                }
                
                final_block_hash = "0x" + "0" * 64
                
                bt.logging.info(f"   Running Boltz scoring for {len(molecules_to_score)} molecules...")
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
                bt.logging.info(f"   ✅ Boltz scoring completed in {elapsed:.2f} seconds")
                
                uid = 0
                smiles_to_score = {}
                if uid in boltz.per_molecule_metric:
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()
                    bt.logging.info(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores")
                
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
                bt.logging.error(f"❌ Error scoring batch with Boltz: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
        
        # Combine results from this batch
        batch_results = molecules_with_db_scores + newly_scored_molecules
        
        for mol in molecules_in_hf:
            mol['boltz_score'] = None
            mol['boltz_score_source'] = 'huggingface_skipped'
            batch_results.append(mol)
        
        all_scored_molecules.extend(batch_results)
        
        bt.logging.info(
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
    
    bt.logging.info(f"✅ Batch scoring complete: {len(scored_molecules)} total molecules scored")
    
    return scored_molecules, False

async def submit_response(state: Dict[str, Any]) -> None:
    """Encrypts and submits the current candidate product."""
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return
    
    bt.logging.info(f"📤 Starting submission process for product: {candidate_product}")
    
    try:
        current_block = await state['subtensor'].get_current_block()
        encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
        bt.logging.info(f"🔐 Encrypted response generated successfully")
        
        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()
            
            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()
            
            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"📝 Prepared commit content: {commit_content}")
            
            bt.logging.info(f"⛓️  Attempting chain commitment...")
            try:
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=state['wallet'],
                    netuid=state['config'].netuid,
                    data=commit_content
                )
                bt.logging.info(f"⛓️  Chain commitment status: {commitment_status}")
            except MetadataError:
                bt.logging.info("⏳ Too soon to commit again. Will try next epoch.")
                return
            
            if commitment_status:
                try:
                    bt.logging.info(f"✅ Commitment set successfully for {commit_content}")
                    bt.logging.info("📤 Attempting GitHub upload...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(f"✅ File uploaded successfully to {commit_content}")
                        state['last_submitted_product'] = candidate_product
                        state['last_submission_time'] = datetime.datetime.now()
                        current_epoch = current_block // state['epoch_length']
                        state['last_submission_epoch'] = current_epoch
                        bt.logging.info(f"✅ Submission recorded for epoch {current_epoch}")
                    else:
                        bt.logging.error(f"❌ Failed to upload file to GitHub for {commit_content}")
                except Exception as e:
                    bt.logging.error(f"❌ Failed to upload file for {commit_content}: {e}")
    
    except Exception as e:
        bt.logging.error(f"❌ Error in submit_response: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


def _import_boltz_wrapper():
    """Import BoltzWrapper following DataGenerator pattern."""
    global BOLTZ_AVAILABLE, BoltzWrapper
    
    try:
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        
        if not os.path.exists(BOLTZ_SCORING_DIR):
            bt.logging.warning(f"⚠️  Boltz-scoring directory not found at {BOLTZ_SCORING_DIR}")
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
        bt.logging.info(f"✅ BoltzWrapper imported successfully")
        return True
        
    except ImportError as e:
        bt.logging.warning(f"⚠️  Failed to import BoltzWrapper: {e}")
        return False
    except Exception as e:
        bt.logging.warning(f"⚠️  Error setting up BoltzWrapper: {e}")
        return False

async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Initialize score_results database
    2. Import and initialize BoltzWrapper
    3. Load molecules from CSV with validation (CSV is automatically updated every epoch)
    4. Prepare top_pool
    """
    bt.logging.info("🚀 Starting STARTUP phase: Initialize DB, Boltz & Load CSV...")
    
    try:
        # Initialize score_results database
        bt.logging.info("💾 Initializing score_results database...")
        init_score_results_db()
        bt.logging.info(f"✅ Score results database initialized")
        
        # Log validation config from config.yaml
        config = state['config']
        bt.logging.info(
            f"✅ Loaded validation config from config.yaml:"
            f"\n   - min_heavy_atoms: {config.get('min_heavy_atoms', 10)}"
            f"\n   - min_rotatable_bonds: {config.get('min_rotatable_bonds', 1)}"
            f"\n   - max_rotatable_bonds: {config.get('max_rotatable_bonds', 10)}"
            f"\n   - banned_atom_types: {config.get('banned_atom_types', [])}"
        )
        
        # Import BoltzWrapper
        bt.logging.info("🔬 Importing BoltzWrapper...")
        boltz_imported = _import_boltz_wrapper()
        
        # Initialize BoltzWrapper
        if boltz_imported and BoltzWrapper is not None:
            bt.logging.info("🔬 Initializing BoltzWrapper...")
            try:
                state['boltz_wrapper'] = BoltzWrapper()
                bt.logging.info("✅ BoltzWrapper initialized successfully")
            except Exception as e:
                bt.logging.error(f"❌ Failed to initialize BoltzWrapper: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
                state['boltz_wrapper'] = None
        else:
            bt.logging.warning("⚠️  BoltzWrapper not available, scoring will be skipped")
            state['boltz_wrapper'] = None
        
        # Load molecules directly from CSV (CSV is automatically updated every epoch by prepare_training_data.py)
        bt.logging.info("📂 Loading molecules from CSV with validation...")
        molecules_df = load_molecules_from_csv_with_validation(
            REACTION_TRAIN_CSV,
            state['current_challenge_targets'],
            STARTING_EPOCH,
            HARDCODED_RXN_ID,
            config
        )
        
        if molecules_df.empty:
            bt.logging.warning("⚠️  No valid molecules loaded from CSV")
            return
        
        # Get top 200 molecules (already sorted by score in load_molecules_from_csv_with_validation)
        top_200_df = molecules_df.head(200)
        
        # Store in state for use in adaptive GA loop
        state['top_pool'] = molecules_df.copy()
        state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
        state['top_200_df'] = top_200_df
        
        bt.logging.info(f"✅ Loaded {len(molecules_df)} molecules from CSV (top 200: {len(top_200_df)})")
        
        bt.logging.info(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Top 200 molecules: {len(state['top_200_df'])}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        bt.logging.error(f"Error in startup phase: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


async def generate_unique_molecules_from_top200(
    state: Dict[str, Any],
    top_200_df: pd.DataFrame,
    desired_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules (NOT in HuggingFace) using genetic algorithm from top 200 molecules.
    Uses both CROSSOVER and MUTATION operations.
    Uses adaptive pool sizing: starts with top 30, increases to 50, 100, 150, 200 if generation fails.
    Keeps generating until desired_count unique molecules are found.
    """
    if top_200_df.empty:
        bt.logging.warning("Top 200 DataFrame is empty")
        return []
    
    ga_operator = GeneticAlgorithmOperator(HARDCODED_RXN_ID, DB_PATH)
    
    # Get all molecule names from top 200
    all_names = top_200_df['name'].tolist()
    
    # Adaptive pool sizes: start with 30, increase if generation fails
    pool_sizes = [30, 50, 100, 150, 200]
    current_pool_size_idx = 0
    current_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
    
    bt.logging.info(
        f"🧬 Generating {desired_count} unique molecules using CROSSOVER + MUTATION "
        f"(starting with top {current_pool_size})..."
    )
    
    unique_molecules = []
    attempts = 0
    max_attempts = 500  # Maximum generation attempts
    last_successful_attempt = 0  # Track when we last found a unique molecule
    
    # Get or initialize generated molecules tracking set
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    # Validation statistics
    validation_stats = {
        'total_generated': 0,
        'passed_validation': 0,
        'failed_smiles': 0,
        'failed_heavy_atoms': 0,
        'failed_banned_atoms': 0,
        'failed_rotatable_bonds': 0,
        'failed_hf_unique': 0,
        'failed_other': 0,
        'crossovers_total': 0,
        'mutations_total': 0,
    }
    
    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1
        
        # Check if we should increase pool size (if 100 attempts failed to find new unique molecules)
        if attempts - last_successful_attempt >= 100 and current_pool_size_idx < len(pool_sizes) - 1:
            current_pool_size_idx += 1
            new_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
            if new_pool_size > current_pool_size:
                current_pool_size = new_pool_size
                bt.logging.info(
                    f"📈 Increasing pool size to top {current_pool_size} "
                    f"(after {attempts - last_successful_attempt} failed attempts)"
                )
                last_successful_attempt = attempts  # Reset counter
        
        # Use current pool size
        current_pool_names = all_names[:current_pool_size]
        
        # Apply genetic operations (crossover + mutation)
        # Balance: 10 crossovers + 10 mutations per batch
        new_molecules = ga_operator.apply_genetic_operations(
            current_pool_names,
            num_crossovers=10,  # 10 crossovers per batch
            num_mutations=10    # 10 mutations per batch
        )
        
        # Track operation types
        for mol in new_molecules:
            if mol.get('type') == 'crossover':
                validation_stats['crossovers_total'] += 1
            elif mol.get('type') == 'mutation':
                validation_stats['mutations_total'] += 1
        
        # Check each new molecule for uniqueness
        for mol in new_molecules:
            if len(unique_molecules) >= desired_count:
                break
            
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            mol_type = mol.get('type', 'unknown')
            
            validation_stats['total_generated'] += 1
            
            # Skip if already in our unique list for this generation
            if molecule_name in [m['name'] for m in unique_molecules]:
                continue
            
            # Skip if already generated in previous generations
            if molecule_name in generated_molecules:
                bt.logging.debug(f"   ⏭️  Molecule {molecule_name} already generated, skipping")
                continue
            
            # Generate InChIKey to check for duplicates
            inchikey = None
            try:
                inchikey = generate_inchikey(smiles) if smiles else None
                if inchikey and inchikey in generated_inchikeys:
                    bt.logging.debug(
                        f"   ⏭️  Molecule {molecule_name} (InChIKey: {inchikey}) already generated, skipping"
                    )
                    continue
            except Exception as e:
                bt.logging.debug(f"   Could not generate InChIKey for {molecule_name}: {e}")
                validation_stats['failed_other'] += 1
                continue
            
            # Validate with config.yaml settings
            is_valid, errors = await validate_molecule_complete(state, molecule_name, smiles, state['config'])
            
            if not is_valid:
                for error in errors:
                    bt.logging.debug(f"   ❌ {molecule_name}: {error}")
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
            
            # Check if unique (NOT on HuggingFace)
            is_unique = await check_molecule_unique(state, molecule_name, smiles)
            
            if is_unique:
                unique_molecules.append(mol)
                # Track this molecule as generated
                generated_molecules.add(molecule_name)
                if inchikey:
                    generated_inchikeys.add(inchikey)
                last_successful_attempt = attempts  # Update last successful attempt
                validation_stats['passed_validation'] += 1
                
                bt.logging.info(
                    f"   ✅ Added unique {mol_type} molecule {molecule_name} "
                    f"(pool size: {current_pool_size}, {len(unique_molecules)}/{desired_count})"
                )
            else:
                bt.logging.debug(f"   ❌ Molecule {molecule_name} is already on HuggingFace")
                validation_stats['failed_hf_unique'] += 1
        
        if len(unique_molecules) >= desired_count:
            break
        
        # Small delay to avoid overwhelming
        await asyncio.sleep(0.1)
    
    # Update state with generated molecules tracking
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    # Log detailed statistics
    bt.logging.info(
        f"✅ Generated {len(unique_molecules)} unique molecules using pool size {current_pool_size} "
        f"(attempts: {attempts}, total tracked: {len(generated_molecules)})"
        f"\n   Operation breakdown:"
        f"\n   - Crossovers generated: {validation_stats['crossovers_total']}"
        f"\n   - Mutations generated: {validation_stats['mutations_total']}"
        f"\n   Validation stats:"
        f"\n   - Total generated: {validation_stats['total_generated']}"
        f"\n   - Passed validation: {validation_stats['passed_validation']}"
        f"\n   - Failed SMILES: {validation_stats['failed_smiles']}"
        f"\n   - Failed heavy atoms: {validation_stats['failed_heavy_atoms']}"
        f"\n   - Failed banned atoms: {validation_stats['failed_banned_atoms']}"
        f"\n   - Failed rotatable bonds: {validation_stats['failed_rotatable_bonds']}"
        f"\n   - Failed HF uniqueness: {validation_stats['failed_hf_unique']}"
        f"\n   - Failed other: {validation_stats['failed_other']}"
    )
    
    return unique_molecules

async def run_adaptive_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Updated genetic algorithm loop with continuous generation:
    1. When epoch changes, generate 100 unique molecules
    2. Score them in batches of 10, checking blocks remaining after each batch
    3. If blocks remaining < 50, submit best molecule and exit
    4. If blocks remaining >= 50 after all batches, generate another round of 100 molecules and repeat
    5. Continue until blocks remaining < 50, then submit
    """
    bt.logging.info("🚀 Starting ADAPTIVE genetic algorithm loop with continuous generation and batch scoring...")
    
    last_processed_epoch = state.get('last_processed_epoch', -1)
    desired_unique_count = 100
    
    while not state['shutdown_event'].is_set():
        try:
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            last_submission_epoch = state.get('last_submission_epoch', -1)
            
            # Check if epoch changed - if so, reload CSV and start generation process
            if current_epoch != last_processed_epoch:
                bt.logging.info(f"\n{'='*70}")
                bt.logging.info(f"🔄 Epoch changed: {last_processed_epoch} → {current_epoch}")
                bt.logging.info(f"{'='*70}")
                
                # Reload molecules from CSV (CSV is automatically updated every epoch by prepare_training_data.py)
                bt.logging.info("📂 Reloading molecules from CSV for new epoch...")
                config = state['config']
                molecules_df = load_molecules_from_csv_with_validation(
                    REACTION_TRAIN_CSV,
                    state['current_challenge_targets'],
                    STARTING_EPOCH,
                    HARDCODED_RXN_ID,
                    config
                )
                
                if molecules_df.empty:
                    bt.logging.warning("⚠️  No valid molecules loaded from CSV, skipping this epoch")
                    last_processed_epoch = current_epoch
                    state['last_processed_epoch'] = current_epoch
                    await asyncio.sleep(10)
                    continue
                
                # Get top 200 molecules (already sorted by score)
                top_200_df = molecules_df.head(200)
                
                # Update state with new molecules
                state['top_pool'] = molecules_df.copy()
                state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
                state['top_200_df'] = top_200_df
                
                bt.logging.info(f"✅ Reloaded {len(molecules_df)} molecules from CSV (top 200: {len(top_200_df)})")
                
                # Calculate blocks until next epoch
                next_epoch_block = (current_epoch + 1) * state['epoch_length']
                
                # ✅ CONTINUOUS GENERATION AND SCORING LOOP
                # Keep generating and scoring batches of 100 until we're within 50 blocks of next epoch
                batch_size = 10
                all_scored_molecules = []
                best_molecule_so_far = None
                best_score_so_far = float('-inf')
                generation_round = 0
                submitted = False
                
                while not submitted:
                    generation_round += 1
                    
                    # Check blocks remaining before starting this generation round
                    current_block_before_round = await state['subtensor'].get_current_block()
                    blocks_remaining = next_epoch_block - current_block_before_round
                    
                    # If we're already within 50 blocks, submit and exit
                    if blocks_remaining < 50:
                        if best_molecule_so_far:
                            # Check if we already submitted in this epoch
                            if last_submission_epoch == current_epoch:
                                bt.logging.info(f"⏭️  Already submitted in epoch {current_epoch}")
                            else:
                                # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                                molecule_name = best_molecule_so_far['name']
                                smiles = best_molecule_so_far.get('smiles')
                                
                                if not smiles:
                                    bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
                                    # Try to find next best unique molecule
                                    best_molecule_so_far = None
                                    best_score_so_far = float('-inf')
                                    for mol in sorted(all_scored_molecules, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
                                        if mol.get('smiles'):
                                            is_unique = await check_molecule_unique(state, mol['name'], mol['smiles'])
                                            if is_unique:
                                                best_molecule_so_far = mol
                                                best_score_so_far = mol.get('boltz_score', float('-inf'))
                                                bt.logging.info(f"✅ Found next best unique molecule: {mol['name']} (score: {best_score_so_far:.6f})")
                                                break
                                    
                                    if not best_molecule_so_far:
                                        bt.logging.warning("⚠️  No unique molecules found to submit")
                                        last_processed_epoch = current_epoch
                                        state['last_processed_epoch'] = current_epoch
                                        submitted = True
                                        break
                                    molecule_name = best_molecule_so_far['name']
                                    smiles = best_molecule_so_far.get('smiles')
                                
                                # Check uniqueness
                                is_unique = await check_molecule_unique(state, molecule_name, smiles)
                                
                                if not is_unique:
                                    bt.logging.warning(
                                        f"❌ Best molecule {molecule_name} is NOT unique (already in HuggingFace), "
                                        f"finding next best unique molecule..."
                                    )
                                    # Find next best unique molecule
                                    best_molecule_so_far = None
                                    best_score_so_far = float('-inf')
                                    for mol in sorted(all_scored_molecules, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
                                        mol_name = mol['name']
                                        mol_smiles = mol.get('smiles')
                                        if not mol_smiles:
                                            continue
                                        
                                        is_unique_check = await check_molecule_unique(state, mol_name, mol_smiles)
                                        if is_unique_check:
                                            best_molecule_so_far = mol
                                            best_score_so_far = mol.get('boltz_score', float('-inf'))
                                            bt.logging.info(
                                                f"✅ Found next best unique molecule: {mol_name} "
                                                f"(score: {best_score_so_far:.6f})"
                                            )
                                            break
                                    
                                    if not best_molecule_so_far:
                                        bt.logging.warning(
                                            "⚠️  No unique molecules found in scored molecules. "
                                            "Cannot submit non-unique molecule."
                                        )
                                        last_processed_epoch = current_epoch
                                        state['last_processed_epoch'] = current_epoch
                                        submitted = True
                                        break
                                
                                bt.logging.info(
                                    f"✅ Best molecule {best_molecule_so_far['name']} is unique, proceeding with submission"
                                )
                                state['candidate_product'] = best_molecule_so_far['name']
                                
                                try:
                                    await submit_response(state)
                                    bt.logging.info(
                                        f"✅ Submission successful for {best_molecule_so_far['name']} "
                                        f"(score: {best_score_so_far:.6f})!"
                                    )
                                    state['last_submission_epoch'] = current_epoch
                                    submitted = True
                                    
                                    # After successful submission, mark epoch as processed and wait for next epoch
                                    last_processed_epoch = current_epoch
                                    state['last_processed_epoch'] = current_epoch
                                    bt.logging.info(f"⏳ Waiting for next epoch to start...")
                                    break  # Exit generation loop
                                except Exception as e:
                                    bt.logging.error(f"❌ Error submitting response: {e}")
                                    import traceback
                                    bt.logging.error(traceback.format_exc())
                                    # Mark as submitted to exit loop even if submission failed
                                    submitted = True
                                    last_processed_epoch = current_epoch
                                    state['last_processed_epoch'] = current_epoch
                                    break
                        else:
                            bt.logging.warning(
                                f"⚠️  Only {blocks_remaining} blocks until next epoch, "
                                f"but no valid molecules scored yet"
                            )
                            # Mark epoch as processed and exit
                            last_processed_epoch = current_epoch
                            state['last_processed_epoch'] = current_epoch
                            submitted = True
                            break
                    
                    # If this is not the first round, generate another batch of 100 unique molecules
                    if generation_round > 1:
                        bt.logging.info(
                            f"\n{'='*70}"
                            f"\n🔄 Generation Round {generation_round}: "
                            f"Blocks remaining ({blocks_remaining}) >= 50, generating another {desired_unique_count} molecules..."
                            f"\n{'='*70}"
                        )
                        # Get top_200_df from state (stored when epoch changed)
                        top_200_df = state.get('top_200_df')
                        if top_200_df is None or top_200_df.empty:
                            bt.logging.warning("No top_200_df in state, skipping generation")
                            break
                        
                        unique_molecules = await generate_unique_molecules_from_top200(
                            state, top_200_df, desired_unique_count
                        )
                        
                        if not unique_molecules:
                            bt.logging.warning("Failed to generate unique molecules, stopping generation loop")
                            break
                        
                        bt.logging.info(f"✅ Generated {len(unique_molecules)} unique molecules for round {generation_round}")
                    else:
                        # First round: generate initial batch
                        bt.logging.info(f"🧬 Generating {desired_unique_count} unique molecules with validation...")
                        unique_molecules = await generate_unique_molecules_from_top200(
                            state, top_200_df, desired_unique_count
                        )
                        
                        if not unique_molecules:
                            bt.logging.warning("Failed to generate unique molecules, skipping this epoch")
                            last_processed_epoch = current_epoch
                            state['last_processed_epoch'] = current_epoch
                            await asyncio.sleep(10)
                            break
                        
                        bt.logging.info(f"✅ Generated {len(unique_molecules)} valid unique molecules")
                    
                    # Score this batch of molecules in batches of 10
                    total_batches = (len(unique_molecules) + batch_size - 1) // batch_size
                    bt.logging.info(f"🔬 Round {generation_round}: Scoring {len(unique_molecules)} molecules in {total_batches} batches of {batch_size}...")
                    
                    for batch_idx in range(total_batches):
                        # Check blocks remaining before starting batch
                        current_block_before_batch = await state['subtensor'].get_current_block()
                        blocks_remaining = next_epoch_block - current_block_before_batch
                        
                        # Only submit when < 50 blocks remain
                        if blocks_remaining < 50:
                            if best_molecule_so_far:
                                # Check if we already submitted in this epoch
                                if last_submission_epoch == current_epoch:
                                    bt.logging.info(f"⏭️  Already submitted in epoch {current_epoch}")
                                else:
                                    # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                                    molecule_name = best_molecule_so_far['name']
                                    smiles = best_molecule_so_far.get('smiles')
                                    
                                    if not smiles:
                                        bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
                                        # Try to find next best unique molecule
                                        best_molecule_so_far = None
                                        best_score_so_far = float('-inf')
                                        for mol in sorted(all_scored_molecules, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
                                            if mol.get('smiles'):
                                                is_unique = await check_molecule_unique(state, mol['name'], mol['smiles'])
                                                if is_unique:
                                                    best_molecule_so_far = mol
                                                    best_score_so_far = mol.get('boltz_score', float('-inf'))
                                                    bt.logging.info(f"✅ Found next best unique molecule: {mol['name']} (score: {best_score_so_far:.6f})")
                                                    break
                                        
                                        if not best_molecule_so_far:
                                            bt.logging.warning("⚠️  No unique molecules found to submit")
                                            last_processed_epoch = current_epoch
                                            state['last_processed_epoch'] = current_epoch
                                            submitted = True
                                            break
                                        molecule_name = best_molecule_so_far['name']
                                        smiles = best_molecule_so_far.get('smiles')
                                    
                                    # Check uniqueness
                                    is_unique = await check_molecule_unique(state, molecule_name, smiles)
                                    
                                    if not is_unique:
                                        bt.logging.warning(
                                            f"❌ Best molecule {molecule_name} is NOT unique (already in HuggingFace), "
                                            f"finding next best unique molecule..."
                                        )
                                        # Find next best unique molecule
                                        best_molecule_so_far = None
                                        best_score_so_far = float('-inf')
                                        for mol in sorted(all_scored_molecules, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
                                            mol_name = mol['name']
                                            mol_smiles = mol.get('smiles')
                                            if not mol_smiles:
                                                continue
                                            
                                            is_unique_check = await check_molecule_unique(state, mol_name, mol_smiles)
                                            if is_unique_check:
                                                best_molecule_so_far = mol
                                                best_score_so_far = mol.get('boltz_score', float('-inf'))
                                                bt.logging.info(
                                                    f"✅ Found next best unique molecule: {mol_name} "
                                                    f"(score: {best_score_so_far:.6f})"
                                                )
                                                break
                                        
                                        if not best_molecule_so_far:
                                            bt.logging.warning(
                                                "⚠️  No unique molecules found in scored molecules. "
                                                "Cannot submit non-unique molecule."
                                            )
                                            last_processed_epoch = current_epoch
                                            state['last_processed_epoch'] = current_epoch
                                            submitted = True
                                            break
                                    
                                    bt.logging.info(
                                        f"✅ Best molecule {best_molecule_so_far['name']} is unique, proceeding with submission"
                                    )
                                    state['candidate_product'] = best_molecule_so_far['name']
                                    
                                    try:
                                        await submit_response(state)
                                        bt.logging.info(
                                            f"✅ Submission successful for {best_molecule_so_far['name']} "
                                            f"(score: {best_score_so_far:.6f})!"
                                        )
                                        state['last_submission_epoch'] = current_epoch
                                        submitted = True
                                        
                                        # After successful submission, mark epoch as processed and wait for next epoch
                                        last_processed_epoch = current_epoch
                                        state['last_processed_epoch'] = current_epoch
                                        bt.logging.info(f"⏳ Waiting for next epoch to start...")
                                        break  # Exit batch loop
                                    except Exception as e:
                                        bt.logging.error(f"❌ Error submitting response: {e}")
                                        import traceback
                                        bt.logging.error(traceback.format_exc())
                                        # Mark as submitted to exit loop even if submission failed
                                        submitted = True
                                        last_processed_epoch = current_epoch
                                        state['last_processed_epoch'] = current_epoch
                                        break
                            else:
                                bt.logging.warning(
                                    f"⚠️  Only {blocks_remaining} blocks until next epoch, "
                                    f"but no valid molecules scored yet"
                                )
                                # Mark epoch as processed and exit
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                                submitted = True
                                break  # Exit batch loop
                        
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(unique_molecules))
                        batch = unique_molecules[start_idx:end_idx]
                        
                        # Score this batch using the batched scoring function
                        bt.logging.info(
                            f"   📦 Round {generation_round}, Batch {batch_idx + 1}/{total_batches}: "
                            f"Scoring {len(batch)} molecules "
                            f"(blocks remaining: {blocks_remaining})"
                        )
                        
                        scored_batch, should_submit_batch = await score_molecules_with_boltz_batched(
                            state, batch, batch_size=len(batch)
                        )
                        
                        # Filter molecules with valid scores
                        batch_with_scores = [m for m in scored_batch if m.get('boltz_score') is not None]
                        all_scored_molecules.extend(batch_with_scores)
                        
                        # Update best molecule so far
                        for mol in batch_with_scores:
                            score = mol.get('boltz_score')
                            if score is not None and score > best_score_so_far:
                                best_score_so_far = score
                                best_molecule_so_far = mol
                                source = mol.get('boltz_score_source', 'unknown')
                                bt.logging.info(
                                    f"   🏆 New best in round {generation_round}, batch {batch_idx + 1}: "
                                    f"{mol['name']} (score: {score:.6f}, source: {source})"
                                )
                        
                        # If should_submit_batch is True, we need to exit
                        if should_submit_batch:
                            submitted = True
                            break
                        
                        # Check epoch boundary after each batch
                        current_block_after_batch = await state['subtensor'].get_current_block()
                        blocks_remaining_after = next_epoch_block - current_block_after_batch
                        
                        bt.logging.info(f"   ⏱️  Blocks remaining after round {generation_round}, batch {batch_idx + 1}: {blocks_remaining_after}")
                        
                        # If we hit the 50 block threshold during batch scoring, exit batch loop
                        if blocks_remaining_after < 50:
                            break
                    
                    # After completing all batches for this round, check if we should continue
                    if submitted:
                        break
                    
                    # Check blocks remaining after all batches in this round
                    current_block_after_round = await state['subtensor'].get_current_block()
                    blocks_remaining_after_round = next_epoch_block - current_block_after_round
                    
                    bt.logging.info(f"   ⏱️  Blocks remaining after round {generation_round}: {blocks_remaining_after_round}")
                    
                    # If blocks remaining < 50, submit and exit
                    if blocks_remaining_after_round < 50:
                        if best_molecule_so_far and last_submission_epoch != current_epoch:
                            # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                            molecule_name = best_molecule_so_far['name']
                            smiles = best_molecule_so_far.get('smiles')
                            
                            if not smiles:
                                bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
                                # Try to find next best unique molecule
                                best_molecule_so_far = None
                                best_score_so_far = float('-inf')
                                for mol in sorted(all_scored_molecules, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
                                    if mol.get('smiles'):
                                        is_unique = await check_molecule_unique(state, mol['name'], mol['smiles'])
                                        if is_unique:
                                            best_molecule_so_far = mol
                                            best_score_so_far = mol.get('boltz_score', float('-inf'))
                                            bt.logging.info(f"✅ Found next best unique molecule: {mol['name']} (score: {best_score_so_far:.6f})")
                                            break
                                
                                if not best_molecule_so_far:
                                    bt.logging.warning("⚠️  No unique molecules found to submit")
                                    submitted = True
                                    last_processed_epoch = current_epoch
                                    state['last_processed_epoch'] = current_epoch
                                    break
                                molecule_name = best_molecule_so_far['name']
                                smiles = best_molecule_so_far.get('smiles')
                            
                            # Check uniqueness
                            is_unique = await check_molecule_unique(state, molecule_name, smiles)
                            
                            if not is_unique:
                                bt.logging.warning(
                                    f"❌ Best molecule {molecule_name} is NOT unique (already in HuggingFace), "
                                    f"finding next best unique molecule..."
                                )
                                # Find next best unique molecule
                                best_molecule_so_far = None
                                best_score_so_far = float('-inf')
                                for mol in sorted(all_scored_molecules, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
                                    mol_name = mol['name']
                                    mol_smiles = mol.get('smiles')
                                    if not mol_smiles:
                                        continue
                                    
                                    is_unique_check = await check_molecule_unique(state, mol_name, mol_smiles)
                                    if is_unique_check:
                                        best_molecule_so_far = mol
                                        best_score_so_far = mol.get('boltz_score', float('-inf'))
                                        bt.logging.info(
                                            f"✅ Found next best unique molecule: {mol_name} "
                                            f"(score: {best_score_so_far:.6f})"
                                        )
                                        break
                                
                                if not best_molecule_so_far:
                                    bt.logging.warning(
                                        "⚠️  No unique molecules found in scored molecules. "
                                        "Cannot submit non-unique molecule."
                                    )
                                    submitted = True
                                    last_processed_epoch = current_epoch
                                    state['last_processed_epoch'] = current_epoch
                                    break
                            
                            bt.logging.info(
                                f"🏆 Best molecule from all rounds: {best_molecule_so_far['name']} "
                                f"with Boltz score: {best_score_so_far:.6f} (unique: ✅)"
                            )
                            
                            state['candidate_product'] = best_molecule_so_far['name']
                            
                            try:
                                await submit_response(state)
                                bt.logging.info(
                                    f"✅ Submission successful for {best_molecule_so_far['name']} "
                                    f"(score: {best_score_so_far:.6f})!"
                                )
                                state['last_submission_epoch'] = current_epoch
                                submitted = True
                                
                                # After successful submission, mark epoch as processed and wait for next epoch
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                                bt.logging.info(f"⏳ Waiting for next epoch to start...")
                            except Exception as e:
                                bt.logging.error(f"❌ Error submitting response: {e}")
                                import traceback
                                bt.logging.error(traceback.format_exc())
                                # Mark as submitted to exit loop even if submission failed
                                submitted = True
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                        else:
                            bt.logging.warning("No valid molecules to submit")
                            submitted = True
                            last_processed_epoch = current_epoch
                            state['last_processed_epoch'] = current_epoch
                        break  # Exit generation loop
                    else:
                        # Still have >= 50 blocks remaining, continue to next generation round
                        score_str = f"{best_score_so_far:.6f}" if best_molecule_so_far else "N/A"
                        bt.logging.info(
                            f"⏭️  Blocks remaining ({blocks_remaining_after_round}) >= 50, "
                            f"continuing to next generation round. "
                            f"Best molecule so far: {best_molecule_so_far['name'] if best_molecule_so_far else 'None'} "
                            f"(score: {score_str})"
                        )
                        # Continue to next iteration of generation loop
            
            # Check if we already submitted in this epoch
            if last_submission_epoch == current_epoch:
                bt.logging.info(
                    f"⏭️  Already submitted in epoch {current_epoch}, waiting for next epoch..."
                )
                await asyncio.sleep(10)
                continue
            
            # If we haven't processed this epoch yet, wait
            if last_processed_epoch != current_epoch:
                await asyncio.sleep(10)
                continue
            
            # Wait a bit before checking again
            await asyncio.sleep(10)
        
        except Exception as e:
            bt.logging.error(f"Error in adaptive GA loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)


async def run_miner(config: argparse.Namespace) -> None:
    """Main mining loop."""
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)
    
    state: Dict[str, Any] = {
        'config': config,
        'github_path': load_github_path(),
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,
        'bdt': QuicknetBittensorDrandTimelock(),
        'candidate_product': None,
        'last_submitted_product': None,
        'last_submission_time': None,
        'last_submission_epoch': -1,
        'last_processed_epoch': -1,
        'startup_complete': False,
        'shutdown_event': asyncio.Event(),
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
        'rxn_id': None,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),
        'generated_inchikeys': set(),
        'boltz_wrapper': None,
        'top_200_df': pd.DataFrame(),
    }
    
    bt.logging.info("🚀 Entering main miner loop...")
    
    state['rxn_id'] = HARDCODED_RXN_ID
    
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    block_hash = await subtensor.determine_block_hash(last_boundary)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )
    
    if startup_proteins:
        state['current_challenge_targets'] = startup_proteins["targets"]
        state['last_challenge_targets'] = startup_proteins["targets"]
        state['current_challenge_antitargets'] = startup_proteins["antitargets"]
        state['last_challenge_antitargets'] = startup_proteins["antitargets"]
    
    # Run startup phase
    await startup_phase(state)
    
    # Run adaptive genetic loop
    await run_adaptive_genetic_loop(state)


def main():
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    
    try:
        asyncio.run(run_miner(config))
    except KeyboardInterrupt:
        bt.logging.info("Miner interrupted by user")
    except Exception as e:
        bt.logging.error(f"Fatal error in miner: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
