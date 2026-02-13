#!/usr/bin/env python3
"""
BITTENSOR MINER - Hybrid Generation (70% Synthon + 30% Crossover) with Validation

Workflow:
1. Load molecules from CSV (filter: rxn_id, no banned atoms, min heavy atoms)
2. Initialize BoltzWrapper for scoring
3. Generate molecules:
   - 70% using SYNTHON SEARCH (intelligent fragment recombination)
   - 30% using CROSSOVER genetic algorithm
4. Validate each molecule with config.yaml settings
5. Collect unique molecules (NOT on HuggingFace)
6. Score all molecules using BoltzWrapper
7. Submit top-scoring molecule
"""

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

# ============================================================================
# CONFIGURATION & IMPORTS
# ============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 5
STARTING_EPOCH = 20795
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'mols.csv')
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "..", "nova-4090", "score_results5.sqlite")

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_smiles,
    get_heavy_atom_count,
    contains_atom_type,
)
from utils.molecules import molecule_unique_for_protein_hf
from molecules_base import (
    generate_inchikey,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock

BOLTZ_AVAILABLE = False
BoltzWrapper = None

# ============================================================================
# VALIDATION FUNCTIONS (from config.yaml)
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
    """Validate heavy atom count from config.yaml."""
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
    """Validate molecule doesn't contain banned atom types from config.yaml."""
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
    """Validate rotatable bonds are within acceptable range from config.yaml."""
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
    """
    Perform complete validation on a molecule using config.yaml settings.
    
    Checks:
    1. SMILES validity
    2. Heavy atom count (min_heavy_atoms)
    3. Banned atoms (banned_atom_types)
    4. Rotatable bonds (min_rotatable_bonds, max_rotatable_bonds)
    5. HuggingFace uniqueness
    """
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
# PYTORCH 2.6+ COMPATIBILITY
# ============================================================================

def safe_torch_load(path, map_location='cpu'):
    """Safely load PyTorch checkpoint with numpy scalar support (PyTorch 2.6+)."""
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


# ============================================================================
# HYBRID MOLECULE GENERATOR (70% SYNTHON + 30% CROSSOVER)
# ============================================================================

class HybridMoleculeGenerator:
    """Generates molecules using 70% synthon search + 30% crossover with validation."""
    
    def __init__(self, rxn_id: int, db_path: str, config: Dict[str, Any]):
        """Initialize hybrid generator with config validation settings."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.config = config
        self.generated_molecule_names: Set[str] = set()
        self.synthon_lib: Optional[SynthonLibrary] = None
        self.synthon_lib_ready = False
    
    def initialize_synthon_library(self) -> bool:
        """Initialize SynthonLibrary for synthon search."""
        try:
            bt.logging.info(f"🔬 Initializing SynthonLibrary for rxn_id={self.rxn_id}...")
            start_time = time.time()
            
            self.synthon_lib = SynthonLibrary(self.db_path, self.rxn_id)
            self.synthon_lib_ready = True
            
            elapsed = time.time() - start_time
            bt.logging.info(f"✅ SynthonLibrary initialized successfully in {elapsed:.2f}s")
            return True
        except Exception as e:
            bt.logging.error(f"❌ Failed to initialize SynthonLibrary: {e}")
            bt.logging.error(traceback.format_exc())
            self.synthon_lib_ready = False
            return False
    
    def generate_synthon_molecules(
        self,
        top_molecules: List[str],
        top_pool_df: pd.DataFrame,
        num_synthon: int = 70
    ) -> List[Dict[str, Any]]:
        """Generate molecules using synthon search with validation."""
        if not self.synthon_lib_ready or self.synthon_lib is None:
            bt.logging.warning("⚠️  SynthonLibrary not ready, skipping synthon generation")
            return []
        
        if top_pool_df.empty:
            bt.logging.warning("⚠️  No top molecules available for synthon search")
            return []
        
        new_molecules = []
        
        try:
            bt.logging.info(f"🧬 Generating {num_synthon} molecules using SYNTHON SEARCH...")
            
            current_max_score = top_pool_df['score'].max() if 'score' in top_pool_df.columns else None
            has_high_score = current_max_score is not None and current_max_score > 0.01
            has_very_high_score = current_max_score is not None and current_max_score > 0.015
            
            if has_very_high_score:
                bt.logging.info(f"🎯 Very high score detected ({current_max_score:.6f}), using TIGHT synthon strategy")
                
                n_part1 = int(num_synthon * 0.40)
                synthon_part1 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(1),
                    n_part1,
                    min_similarity=0.90,
                    n_per_base=50
                )
                bt.logging.info(f"   Part 1: Generated {len(synthon_part1)} ultra-tight synthon")
                
                n_part2 = int(num_synthon * 0.30)
                synthon_part2 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(5),
                    n_part2,
                    min_similarity=0.80,
                    n_per_base=30
                )
                bt.logging.info(f"   Part 2: Generated {len(synthon_part2)} tight synthon")
                
                n_part3 = int(num_synthon * 0.30)
                synthon_part3 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(20),
                    n_part3,
                    min_similarity=0.55,
                    n_per_base=15
                )
                bt.logging.info(f"   Part 3: Generated {len(synthon_part3)} medium synthon")
                
                synthon_df = pd.concat([synthon_part1, synthon_part2, synthon_part3], ignore_index=True)
            
            elif has_high_score:
                bt.logging.info(f"🎯 High score detected ({current_max_score:.6f}), using BALANCED synthon strategy")
                
                n_part1 = int(num_synthon * 0.40)
                synthon_part1 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(5),
                    n_part1,
                    min_similarity=0.80,
                    n_per_base=30
                )
                bt.logging.info(f"   Part 1: Generated {len(synthon_part1)} tight synthon")
                
                n_part2 = int(num_synthon * 0.30)
                seed_medium = top_pool_df.iloc[10:40] if len(top_pool_df) > 40 else top_pool_df.iloc[5:]
                synthon_part2 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    seed_medium,
                    n_part2,
                    min_similarity=0.55,
                    n_per_base=15
                )
                bt.logging.info(f"   Part 2: Generated {len(synthon_part2)} medium synthon")
                
                n_part3 = int(num_synthon * 0.30)
                synthon_part3 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(50),
                    n_part3,
                    min_similarity=0.40,
                    n_per_base=20
                )
                bt.logging.info(f"   Part 3: Generated {len(synthon_part3)} broad synthon")
                
                synthon_df = pd.concat([synthon_part1, synthon_part2, synthon_part3], ignore_index=True)
            
            else:
                bt.logging.info(f"🎯 Standard score, using BROAD synthon strategy")
                
                synthon_df = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(30),
                    num_synthon,
                    min_similarity=0.65,
                    n_per_base=20
                )
                bt.logging.info(f"   Generated {len(synthon_df)} broad synthon")
            
            # ✅ VALIDATE SYNTHON MOLECULES
            if not synthon_df.empty:
                bt.logging.info(f"   Processing {len(synthon_df)} synthon molecules...")
                
                synthon_df = validate_molecules(synthon_df, self.config)
                bt.logging.info(f"   After validation: {len(synthon_df)} valid synthon molecules")
                
                if synthon_df.empty:
                    bt.logging.warning("   ⚠️  No molecules passed validation")
                    return []
                
                synthon_df = synthon_df.drop_duplicates(subset=['InChIKey'], keep='first')
                bt.logging.info(f"   After deduplication: {len(synthon_df)} unique synthon molecules")
                
                for _, row in synthon_df.iterrows():
                    mol_name = row.get('name')
                    smiles = row.get('smiles')
                    inchikey = row.get('InChIKey')
                    
                    if mol_name and smiles and inchikey:
                        if mol_name not in self.generated_molecule_names:
                            new_molecules.append({
                                'name': mol_name,
                                'smiles': smiles,
                                'InChIKey': inchikey,
                                'type': 'synthon'
                            })
                            self.generated_molecule_names.add(mol_name)
            
            bt.logging.info(f"✅ Generated {len(new_molecules)} valid synthon molecules")
            return new_molecules
        
        except Exception as e:
            bt.logging.error(f"❌ Error in synthon generation: {e}")
            bt.logging.error(traceback.format_exc())
            return []
    
    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """Crossover two molecules by swapping random components."""
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
            
            num_components = len(parts1) - 2
            component_indices = list(range(2, 2 + num_components))
            swap_idx = random.choice(component_indices)
            
            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)
            
            if offspring_name in self.generated_molecule_names:
                return None
            
            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        self.generated_molecule_names.add(offspring_name)
                        bt.logging.debug(f"✅ Crossover: {mol_name_1} × {mol_name_2} → {offspring_name}")
                        return offspring_name
            except Exception as e:
                bt.logging.debug(f"Error validating crossover: {e}")
            
            return None
        
        except Exception as e:
            bt.logging.debug(f"Error in crossover_molecules: {e}")
            return None
    
    def apply_hybrid_generation(
        self,
        top_molecules: List[str],
        top_pool_df: pd.DataFrame,
        num_synthon: int = 70,
        num_crossover: int = 30
    ) -> List[Dict[str, Any]]:
        """Apply hybrid generation: 70% synthon search + 30% crossover."""
        new_molecules = []
        self.generated_molecule_names.clear()
        
        bt.logging.info(f"🧬 Applying HYBRID generation (70% synthon + 30% crossover)...")
        bt.logging.info(f"   Target: {num_synthon} synthon + {num_crossover} crossover = {num_synthon + num_crossover} total")
        
        # PART 1: SYNTHON SEARCH (70%)
        if self.synthon_lib_ready:
            synthon_molecules = self.generate_synthon_molecules(
                top_molecules,
                top_pool_df,
                num_synthon
            )
            new_molecules.extend(synthon_molecules)
            bt.logging.info(f"   ✅ Synthon: {len(synthon_molecules)}/{num_synthon} molecules generated")
        else:
            bt.logging.warning(f"   ⚠️  SynthonLibrary not ready, using crossover for all molecules")
            num_crossover = num_synthon + num_crossover
        
        # PART 2: CROSSOVER (30%)
        crossover_attempts = 0
        crossovers_created = 0
        
        for i in range(num_crossover):
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
                        bt.logging.debug(f"Error processing crossover offspring: {e}")
        
        bt.logging.info(f"   ✅ Crossover: {crossovers_created}/{num_crossover} molecules generated")
        bt.logging.info(f"🧬 HYBRID generation complete: {len(new_molecules)} total molecules")
        
        return new_molecules


# ============================================================================
# DATABASE FUNCTIONS
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
                to_insert.append((molecule_name, float(score)))
        
        if to_insert:
            cursor.executemany(
                "INSERT OR REPLACE INTO scored_molecules (molecule_name, score) VALUES (?, ?)",
                to_insert
            )
            conn.commit()
            bt.logging.info(f"✅ Wrote {len(to_insert)} scored molecules to database")
        
        conn.close()
    except Exception as e:
        bt.logging.error(f"Error writing scores to database: {e}")
        bt.logging.error(traceback.format_exc())


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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
            bt.logging.error(traceback.format_exc())
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                raise


def load_molecules_from_csv(
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
        
        if 'molecule_name' not in df.columns:
            bt.logging.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        
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
        bt.logging.error(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


async def run_adaptive_genetic_loop(state: Dict[str, Any]) -> None:
    """Adaptive genetic algorithm loop with hybrid generation and validation."""
    bt.logging.info("🚀 Starting HYBRID genetic algorithm loop (70% synthon + 30% crossover)...")
    
    csv_path = os.path.join(BASE_DIR, 'data', 'mols.csv')
    last_processed_epoch = state.get('last_processed_epoch', -1)
    desired_unique_count = 100
    
    hybrid_generator = HybridMoleculeGenerator(HARDCODED_RXN_ID, DB_PATH, state['config'])
    
    while not state['shutdown_event'].is_set():
        try:
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            last_submission_epoch = state.get('last_submission_epoch', -1)
            
            if current_epoch != last_processed_epoch:
                bt.logging.info(f"\n{'='*70}")
                bt.logging.info(f"🔄 Epoch changed: {last_processed_epoch} → {current_epoch}")
                bt.logging.info(f"{'='*70}")
                
                top_200_df = load_molecules_from_csv(
                    csv_path,
                    state['current_challenge_targets'],
                    STARTING_EPOCH,
                    HARDCODED_RXN_ID,
                    state['config']
                )
                
                if top_200_df.empty:
                    bt.logging.warning("No top 200 molecules found, skipping epoch")
                    last_processed_epoch = current_epoch
                    state['last_processed_epoch'] = current_epoch
                    await asyncio.sleep(10)
                    continue
                
                top_200_df = top_200_df.head(200)
                state['top_200_df'] = top_200_df
                
                if not hybrid_generator.synthon_lib_ready:
                    bt.logging.info("🔬 Initializing SynthonLibrary...")
                    hybrid_generator.initialize_synthon_library()
                
                next_epoch_block = (current_epoch + 1) * state['epoch_length']
                generation_round = 0
                all_scored_molecules = []
                best_molecule_so_far = None
                best_score_so_far = float('-inf')
                current_seed_pool = top_200_df
                submitted = False
                
                while not submitted:
                    generation_round += 1
                    current_block_before_round = await state['subtensor'].get_current_block()
                    blocks_remaining_before_round = next_epoch_block - current_block_before_round
                    
                    bt.logging.info(f"\n{'='*70}")
                    bt.logging.info(f"🧬 Generation Round {generation_round}")
                    bt.logging.info(f"   Blocks remaining: {blocks_remaining_before_round}")
                    bt.logging.info(f"{'='*70}")
                    
                    # Check if time to submit
                    if blocks_remaining_before_round < 50:
                        await _submit_best_molecule(state, best_molecule_so_far, best_score_so_far, all_scored_molecules, current_epoch, last_submission_epoch)
                        submitted = True
                        break
                    
                    # Generate unique molecules
                    bt.logging.info(f"🧬 Generating {desired_unique_count} unique molecules...")
                    unique_molecules = await generate_unique_molecules_from_top200(
                        state, current_seed_pool, hybrid_generator, desired_unique_count
                    )
                    
                    if not unique_molecules:
                        bt.logging.warning(f"Failed to generate molecules in round {generation_round}")
                        last_processed_epoch = current_epoch
                        state['last_processed_epoch'] = current_epoch
                        submitted = True
                        break
                    
                    bt.logging.info(f"✅ Generated {len(unique_molecules)} unique molecules")
                    
                    # Score molecules in batches
                    batch_size = 10
                    total_batches = (len(unique_molecules) + batch_size - 1) // batch_size
                    bt.logging.info(f"🔬 Scoring {len(unique_molecules)} molecules in {total_batches} batches...")
                    
                    for batch_idx in range(total_batches):
                        current_block_before_batch = await state['subtensor'].get_current_block()
                        blocks_remaining = next_epoch_block - current_block_before_batch
                        
                        if blocks_remaining < 50:
                            await _submit_best_molecule(state, best_molecule_so_far, best_score_so_far, all_scored_molecules, current_epoch, last_submission_epoch)
                            submitted = True
                            break
                        
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(unique_molecules))
                        batch = unique_molecules[start_idx:end_idx]
                        
                        bt.logging.info(f"📦 Round {generation_round}, Batch {batch_idx + 1}/{total_batches}: Scoring {len(batch)} molecules")
                        
                        scored_batch = await score_molecules_with_boltz(state, batch)
                        
                        if scored_batch:
                            batch_with_scores = [m for m in scored_batch if m.get('boltz_score') is not None]
                            all_scored_molecules.extend(batch_with_scores)
                            
                            for mol in batch_with_scores:
                                score = mol.get('boltz_score')
                                if score is not None and score > best_score_so_far:
                                    best_score_so_far = score
                                    best_molecule_so_far = mol
                                    bt.logging.info(f"🏆 New best: {mol['name']} (score: {score:.6f})")
                    
                    if submitted:
                        break
                    
                    # Check after round
                    current_block_after_round = await state['subtensor'].get_current_block()
                    blocks_remaining_after_round = next_epoch_block - current_block_after_round
                    
                    bt.logging.info(f"✅ Round {generation_round} complete")
                    bt.logging.info(f"   Blocks remaining: {blocks_remaining_after_round}")
                    
                    if blocks_remaining_after_round < 50:
                        await _submit_best_molecule(state, best_molecule_so_far, best_score_so_far, all_scored_molecules, current_epoch, last_submission_epoch)
                        submitted = True
                        break
                    else:
                        # Prepare for next round
                        top_scored = sorted(
                            all_scored_molecules,
                            key=lambda m: m.get('boltz_score', float('-inf')),
                            reverse=True
                        )[:30]
                        
                        if top_scored:
                            current_seed_pool = pd.DataFrame(top_scored)
                            bt.logging.info(f"📈 Next round: Using top 30 molecules as seed pool")
                        else:
                            current_seed_pool = state['top_200_df']
                            bt.logging.warning(f"📈 Next round: Using original top 200 as seed")
                
                if last_processed_epoch != current_epoch:
                    last_processed_epoch = current_epoch
                    state['last_processed_epoch'] = current_epoch
            
            if last_submission_epoch == current_epoch:
                bt.logging.info(f"⏭️  Already submitted in epoch {current_epoch}")
                await asyncio.sleep(10)
                continue
            
            if last_processed_epoch != current_epoch:
                await asyncio.sleep(10)
                continue
            
            await asyncio.sleep(10)
        
        except Exception as e:
            bt.logging.error(f"Error in adaptive GA loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)


# ============================================================================
# HELPER FUNCTION: SUBMIT BEST MOLECULE
# ============================================================================

async def _submit_best_molecule(state, best_molecule, best_score, all_scored, current_epoch, last_submission_epoch):
    """Helper to submit best molecule with validation."""
    if best_molecule and last_submission_epoch != current_epoch:
        molecule_name = best_molecule['name']
        smiles = best_molecule.get('smiles')
        
        # Validate SMILES exists
        if not smiles:
            bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
            best_molecule = await _find_next_best_unique(all_scored, state)
            if not best_molecule:
                bt.logging.warning("⚠️  No unique molecules found to submit")
                return
            smiles = best_molecule.get('smiles')
        
        # Check uniqueness
        is_unique = await check_molecule_unique(state, molecule_name, smiles)
        
        if not is_unique:
            bt.logging.warning(f"❌ Best molecule {molecule_name} is NOT unique, finding next best...")
            best_molecule = await _find_next_best_unique(all_scored, state)
            if not best_molecule:
                bt.logging.warning("⚠️  No unique molecules found in scored molecules")
                return
        
        # Log if already in DB
        db_score = get_score_from_db(best_molecule['name'])
        if db_score is not None:
            bt.logging.info(f"⚠️  Molecule {best_molecule['name']} already in DB (score: {db_score:.6f})")
        
        # Submit
        bt.logging.info(f"🏆 Submitting: {best_molecule['name']} (score: {best_score:.6f})")
        state['candidate_product'] = best_molecule['name']
        
        try:
            await submit_response(state)
            bt.logging.info(f"✅ Submission successful!")
            state['last_submission_epoch'] = current_epoch
            state['last_processed_epoch'] = current_epoch
        except Exception as e:
            bt.logging.error(f"❌ Error submitting: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
    else:
        bt.logging.warning("No valid molecules to submit")


async def _find_next_best_unique(all_scored, state):
    """Find next best unique molecule from scored list."""
    for mol in sorted(all_scored, key=lambda m: m.get('boltz_score', float('-inf')), reverse=True):
        mol_name = mol['name']
        mol_smiles = mol.get('smiles')
        
        if not mol_smiles:
            continue
        
        is_unique = await check_molecule_unique(state, mol_name, mol_smiles)
        if is_unique:
            bt.logging.info(f"✅ Found next best: {mol_name} (score: {mol.get('boltz_score', 'N/A'):.6f})")
            return mol
    
    return None


# ============================================================================
# STARTUP PHASE
# ============================================================================

async def startup_phase(state: Dict[str, Any]) -> None:
    """Startup phase with validation from config.yaml."""
    bt.logging.info("🚀 Starting STARTUP phase...")
    
    try:
        bt.logging.info("💾 Initializing score_results database...")
        init_score_results_db()
        bt.logging.info(f"✅ Database initialized at {SCORE_RESULTS_DB}")
        
        # Log validation config
        config = state['config']
        bt.logging.info(
            f"✅ Validation config:"
            f"\n   - min_heavy_atoms: {config.get('min_heavy_atoms', 10)}"
            f"\n   - min_rotatable_bonds: {config.get('min_rotatable_bonds', 1)}"
            f"\n   - max_rotatable_bonds: {config.get('max_rotatable_bonds', 10)}"
            f"\n   - banned_atom_types: {config.get('banned_atom_types', [])}"
        )
        
        # Import and initialize BoltzWrapper
        bt.logging.info("🔬 Importing BoltzWrapper...")
        boltz_imported = _import_boltz_wrapper()
        
        if boltz_imported and BoltzWrapper is not None:
            bt.logging.info("🔬 Initializing BoltzWrapper...")
            try:
                state['boltz_wrapper'] = BoltzWrapper()
                bt.logging.info("✅ BoltzWrapper initialized")
            except Exception as e:
                bt.logging.error(f"❌ Failed to initialize BoltzWrapper: {e}")
                state['boltz_wrapper'] = None
        else:
            bt.logging.warning("⚠️  BoltzWrapper not available")
            state['boltz_wrapper'] = None
        
        # Load molecules from CSV
        bt.logging.info("📂 Loading molecules from CSV...")
        molecules_df = load_molecules_from_csv(
            REACTION_TRAIN_CSV,
            state['current_challenge_targets'],
            STARTING_EPOCH,
            HARDCODED_RXN_ID,
            config
        )
        
        if molecules_df.empty:
            bt.logging.warning("⚠️  No molecules loaded from CSV")
            state['top_pool'] = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
            state['seen_inchikeys'] = set()
        else:
            bt.logging.info(f"✅ Loaded {len(molecules_df)} molecules")
            state['top_pool'] = molecules_df.copy()
            state['seen_inchikeys'] = set(molecules_df['InChIKey'].tolist())
        
        bt.logging.info(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules: {len(state['top_pool'])}"
            f"\n   Sample: {state['top_pool']['name'].head(3).tolist() if not state['top_pool'].empty else 'None'}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        bt.logging.error(f"Error in startup phase: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


# ============================================================================
# BOLTZ WRAPPER IMPORT
# ============================================================================

def _import_boltz_wrapper():
    """Import BoltzWrapper from boltz-scoring directory."""
    global BOLTZ_AVAILABLE, BoltzWrapper
    
    try:
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        
        if not os.path.exists(BOLTZ_SCORING_DIR):
            bt.logging.warning(f"⚠️  Boltz-scoring directory not found")
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


# ============================================================================
# MAIN MINER LOOP
# ============================================================================

async def run_miner(config: argparse.Namespace) -> None:
    """Main mining loop with hybrid generation and validation."""
    
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
        'hybrid_generator': None,
    }
    
    bt.logging.info("🚀 Entering main miner loop...")
    
    state['rxn_id'] = HARDCODED_RXN_ID
    
    # Get startup parameters
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
        
        bt.logging.info(f"Using reaction ID: {state['rxn_id']}")
        bt.logging.info(
            f"Startup targets: {startup_proteins['targets']}, "
            f"antitargets: {startup_proteins['antitargets']}"
        )
        
        try:
            await startup_phase(state)
            bt.logging.info("✅ Startup phase completed!")
        except Exception as e:
            bt.logging.error(f"Error in startup: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
        
        try:
            state['ga_task'] = asyncio.create_task(run_adaptive_genetic_loop(state))
            bt.logging.info("✅ Adaptive hybrid GA loop started!")
        except Exception as e:
            bt.logging.error(f"Error starting GA loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
    
    # Main loop
    while True:
        try:
            current_block = await subtensor.get_current_block()
            
            if current_block % epoch_length == 0:
                current_epoch = current_block // epoch_length
                bt.logging.info(f"⏰ Epoch boundary at block {current_block} (epoch {current_epoch})")
            
            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Nodes: {metagraph.n} | "
                    f"Epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)
            
            await asyncio.sleep(1)
        
        except RuntimeError as e:
            bt.logging.error(e)
            import traceback
            traceback.print_exc()
        
        except KeyboardInterrupt:
            bt.logging.success("⛔ Keyboard interrupt detected. Exiting.")
            state['shutdown_event'].set()
            break


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())