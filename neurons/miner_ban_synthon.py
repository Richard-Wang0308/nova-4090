#!/usr/bin/env python3
"""
UPDATED BITTENSOR MINER - Genetic Algorithm + Synthon Search

Updated workflow:
1. Load molecules from CSV at startup
2. Initialize BoltzWrapper for scoring
3. Generate molecules:
   - 70% using SYNTHON SEARCH (intelligent fragment recombination)
   - 30% using CROSSOVER-ONLY genetic algorithm
4. Collect 10 unique molecules (NOT on HuggingFace)
5. Score all 10 molecules using BoltzWrapper
6. Submit the top-scoring molecule
7. Do not submit again in the same epoch
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Database path for combinatorial DB
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 5
STARTING_EPOCH = 20795

# ✅ CSV for loading initial molecules (now directly under nova-4090/data/)
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'mols.csv')

# ✅ Database for storing scored molecules
SCORE_RESULTS_DB = os.path.join(BASE_DIR,"..","nova-4090", "score_results5.sqlite")

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
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

# BoltzWrapper import - following the same pattern as DataGenerator/main.py
BOLTZ_AVAILABLE = False
BoltzWrapper = None


# ============================================================================
# PyTorch 2.6+ Compatibility
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
# ✅ HYBRID MOLECULE GENERATION (70% SYNTHON + 30% CROSSOVER)
# ============================================================================

class HybridMoleculeGenerator:
    """Generates molecules using 70% synthon search + 30% crossover."""
    
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
        Should be called after we have good molecules (iteration 2+).
        
        Returns:
            True if successful, False otherwise
        """
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
            import traceback
            bt.logging.error(traceback.format_exc())
            self.synthon_lib_ready = False
            return False
    
    def generate_synthon_molecules(
        self,
        top_molecules: List[str],
        top_pool_df: pd.DataFrame,
        num_synthon: int = 70
    ) -> List[Dict[str, Any]]:
        """
        Generate molecules using synthon search (70% of generation).
        
        Uses multi-range strategy:
        - Tight on top 5 molecules
        - Medium on molecules 10-40
        - Broad on top 50 molecules
        
        Args:
            top_molecules: List of top molecule names
            top_pool_df: DataFrame with top molecules (name, smiles, InChIKey, score)
            num_synthon: Number of synthon molecules to generate
            
        Returns:
            List of new molecules with SMILES and names
        """
        if not self.synthon_lib_ready or self.synthon_lib is None:
            bt.logging.warning("⚠️  SynthonLibrary not ready, skipping synthon generation")
            return []
        
        if top_pool_df.empty:
            bt.logging.warning("⚠️  No top molecules available for synthon search")
            return []
        
        new_molecules = []
        
        try:
            bt.logging.info(f"🧬 Generating {num_synthon} molecules using SYNTHON SEARCH...")
            
            # Get current max score for adaptive strategy
            current_max_score = top_pool_df['score'].max() if 'score' in top_pool_df.columns else None
            
            # Determine strategy based on score
            has_high_score = current_max_score is not None and current_max_score > 0.01
            has_very_high_score = current_max_score is not None and current_max_score > 0.015
            
            if has_very_high_score:
                bt.logging.info(f"🎯 Very high score detected ({current_max_score:.6f}), using TIGHT synthon strategy")
                
                # Part 1: Ultra-tight on TOP 1 (40% of budget)
                n_part1 = int(num_synthon * 0.40)
                synthon_part1 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(1),
                    n_part1,
                    min_similarity=0.90,
                    n_per_base=50
                )
                bt.logging.info(f"   Part 1: Generated {len(synthon_part1)} ultra-tight synthon (top 1, sim=0.90)")
                
                # Part 2: Tight on TOP 5 (30% of budget)
                n_part2 = int(num_synthon * 0.30)
                synthon_part2 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(5),
                    n_part2,
                    min_similarity=0.80,
                    n_per_base=30
                )
                bt.logging.info(f"   Part 2: Generated {len(synthon_part2)} tight synthon (top 5, sim=0.80)")
                
                # Part 3: Medium on TOP 20 (30% of budget)
                n_part3 = int(num_synthon * 0.30)
                synthon_part3 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(20),
                    n_part3,
                    min_similarity=0.55,
                    n_per_base=15
                )
                bt.logging.info(f"   Part 3: Generated {len(synthon_part3)} medium synthon (top 20, sim=0.55)")
                
                # Combine all parts
                synthon_df = pd.concat([synthon_part1, synthon_part2, synthon_part3], ignore_index=True)
            
            elif has_high_score:
                bt.logging.info(f"🎯 High score detected ({current_max_score:.6f}), using BALANCED synthon strategy")
                
                # Part 1: Tight on TOP 5 (40% of budget)
                n_part1 = int(num_synthon * 0.40)
                synthon_part1 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(5),
                    n_part1,
                    min_similarity=0.80,
                    n_per_base=30
                )
                bt.logging.info(f"   Part 1: Generated {len(synthon_part1)} tight synthon (top 5, sim=0.80)")
                
                # Part 2: Medium on molecules 10-40 (30% of budget)
                n_part2 = int(num_synthon * 0.30)
                seed_medium = top_pool_df.iloc[10:40] if len(top_pool_df) > 40 else top_pool_df.iloc[5:]
                synthon_part2 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    seed_medium,
                    n_part2,
                    min_similarity=0.55,
                    n_per_base=15
                )
                bt.logging.info(f"   Part 2: Generated {len(synthon_part2)} medium synthon (molecules 10-40, sim=0.55)")
                
                # Part 3: Broad on TOP 50 (30% of budget)
                n_part3 = int(num_synthon * 0.30)
                synthon_part3 = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(50),
                    n_part3,
                    min_similarity=0.40,
                    n_per_base=20
                )
                bt.logging.info(f"   Part 3: Generated {len(synthon_part3)} broad synthon (top 50, sim=0.40)")
                
                # Combine all parts
                synthon_df = pd.concat([synthon_part1, synthon_part2, synthon_part3], ignore_index=True)
            
            else:
                bt.logging.info(f"🎯 Standard score, using BROAD synthon strategy")
                
                # Simple approach: Medium on top 30
                synthon_df = generate_molecules_from_synthon_library(
                    self.synthon_lib,
                    top_pool_df.head(30),
                    num_synthon,
                    min_similarity=0.65,
                    n_per_base=20
                )
                bt.logging.info(f"   Generated {len(synthon_df)} broad synthon (top 30, sim=0.65)")
            
            # ✅ FIX: VALIDATE FIRST to add smiles and InChIKey columns
            if not synthon_df.empty:
                bt.logging.info(f"   Processing {len(synthon_df)} synthon molecules...")
                
                # ✅ VALIDATE MOLECULES FIRST - this adds 'smiles' and 'InChIKey' columns
                # Create a minimal config for validation
                validation_config = {
                    'min_heavy_atoms': 5,
                    'min_rotatable_bonds': 0,
                    'max_rotatable_bonds': 20
                }
                
                synthon_df = validate_molecules(synthon_df, validation_config)
                bt.logging.info(f"   After validation: {len(synthon_df)} valid synthon molecules")
                
                if synthon_df.empty:
                    bt.logging.warning("   ⚠️  No molecules passed validation")
                    return []
                
                # Deduplicate by InChIKey (now that we have it)
                synthon_df = synthon_df.drop_duplicates(subset=['InChIKey'], keep='first')
                bt.logging.info(f"   After deduplication: {len(synthon_df)} unique synthon molecules")
                
                # ✅ NOW we can safely access smiles and InChIKey columns
                for _, row in synthon_df.iterrows():
                    mol_name = row.get('name')
                    smiles = row.get('smiles')
                    inchikey = row.get('InChIKey')
                    
                    # ✅ All columns now exist after validate_molecules
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
            import traceback
            bt.logging.error(traceback.format_exc())
            return []

    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """
        Crossover two molecules by swapping random components (30% of generation).
        
        Supports two formats:
        - 2 components: rxn:5:comp1:comp2 (4 parts)
        - 3 components: rxn:5:comp1:comp2:comp3 (5 parts)
        
        Args:
            mol_name_1: First parent molecule name
            mol_name_2: Second parent molecule name
            
        Returns:
            New molecule name from crossover, or None if crossover failed
        """
        try:
            from rdkit import Chem
            
            # Parse molecule names
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
            
            # Validate offspring
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
        """
        Apply hybrid generation: 70% synthon search + 30% crossover.
        
        Args:
            top_molecules: List of top molecule names
            top_pool_df: DataFrame with top molecules
            num_synthon: Number of synthon molecules (70%)
            num_crossover: Number of crossover molecules (30%)
            
        Returns:
            List of new molecules (synthon + crossover combined)
        """
        new_molecules = []
        
        # Reset tracking for this batch
        self.generated_molecule_names.clear()
        
        bt.logging.info(f"🧬 Applying HYBRID generation (70% synthon + 30% crossover)...")
        bt.logging.info(f"   Target: {num_synthon} synthon + {num_crossover} crossover = {num_synthon + num_crossover} total")
        
        # ✅ PART 1: SYNTHON SEARCH (70%)
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
            num_crossover = num_synthon + num_crossover  # Use all budget for crossover
        
        # ✅ PART 2: CROSSOVER (30%)
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
# HELPER FUNCTIONS (from original code)
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
            import traceback
            bt.logging.error(traceback.format_exc())
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
        import traceback
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


async def check_molecule_unique(state: Dict[str, Any], molecule_name: str, smiles: str) -> bool:
    """Check if molecule is unique for target protein (NOT in HuggingFace dataset)."""
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


def load_molecules_from_csv(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int
) -> pd.DataFrame:
    """Load molecules from CSV file, sorted by final_score descending."""
    if not os.path.exists(csv_path):
        bt.logging.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        bt.logging.info(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        # Filter by epoch
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            bt.logging.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        # Filter by rxn_id using molecule_name column
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
        
        for _, row in df.iterrows():
            molecule_name = row['molecule_name']
            
            try:
                # Generate SMILES from reaction ID
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    bt.logging.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Generate InChIKey from SMILES
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    bt.logging.debug(f"Could not generate InChIKey for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Get score from CSV
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
            # Remove duplicates by InChIKey
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            # Sort by score descending
            result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
            
            bt.logging.info(
                f"✅ Loaded {len(result_df)} molecules from CSV "
                f"(successful: {successful_count}, failed: {failed_count})"
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
                f"(successful: {successful_count}, failed: {failed_count})"
            )
        
        return result_df
        
    except Exception as e:
        bt.logging.error(f"Error loading molecules from CSV: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


# ============================================================================
# SCORING AND SUBMISSION (same as original)
# ============================================================================

async def score_molecules_with_boltz(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Score molecules using BoltzWrapper (same as original code)."""
    if state.get('boltz_wrapper') is None:
        bt.logging.warning("BoltzWrapper not available, skipping scoring")
        return molecules
    
    if not molecules:
        return molecules
    
    bt.logging.info(f"🔬 Processing {len(molecules)} molecules for scoring...")
    
    init_score_results_db()
    
    molecules_to_score = []
    molecules_with_db_scores = []
    molecules_in_hf = []
    
    target_proteins = state.get('current_challenge_targets', [])
    primary_target = target_proteins[0] if target_proteins else None
    
    molecule_names = [mol['name'] for mol in molecules]
    db_scores = batch_get_scores_from_db(molecule_names)
    
    bt.logging.info(f"   Found {len(db_scores)} molecules already scored in database")
    
    for mol in molecules:
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
        bt.logging.info(f"🔬 Scoring {len(molecules_to_score)} new molecules with Boltz...")
        
        boltz = state['boltz_wrapper']
        config = state['config']
        target_proteins = state.get('current_challenge_targets', [])
        antitarget_proteins = state.get('current_challenge_antitargets', [])
        
        if not target_proteins:
            bt.logging.warning("No target proteins available for scoring")
            all_results = molecules_with_db_scores + [mol for mol in molecules if mol.get('boltz_score') is None]
            return all_results
        
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
                bt.logging.info(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores from per_molecule_metric")
            
            target_scores_list = None
            target_scores = score_dict[uid].get('target_scores', [[]])
            if target_scores and len(target_scores[0]) > 0:
                target_scores_list = target_scores[0] if isinstance(target_scores[0], list) else [target_scores[0]]
                bt.logging.debug(f"   Found {len(target_scores_list)} scores in target_scores")
            
            avg_score = None
            if not smiles_to_score and not target_scores_list:
                avg_score = score_dict[uid].get('boltz_score')
                if avg_score is not None and isinstance(avg_score, (int, float)):
                    bt.logging.warning(f"   ⚠️  Only average boltz_score available: {avg_score} (using as fallback)")
            
            molecules_with_individual_scores = 0
            molecules_with_avg_scores = 0
            molecules_without_scores = 0
            
            for mol_idx, mol in enumerate(molecules_to_score):
                smiles = mol['smiles']
                score = None
                score_source = None
                
                if smiles in smiles_to_score:
                    score = smiles_to_score[smiles]
                    score_source = "boltz_scoring"
                    molecules_with_individual_scores += 1
                elif target_scores_list and mol_idx < len(target_scores_list):
                    score = target_scores_list[mol_idx]
                    score_source = "boltz_scoring"
                    molecules_with_individual_scores += 1
                elif target_scores_list:
                    try:
                        valid_idx = valid_molecules_by_uid[uid]['smiles'].index(smiles)
                        if valid_idx < len(target_scores_list):
                            score = target_scores_list[valid_idx]
                            score_source = "boltz_scoring"
                            molecules_with_individual_scores += 1
                    except (ValueError, IndexError):
                        pass
                
                if score is None and avg_score is not None:
                    score = avg_score
                    score_source = "boltz_scoring_avg"
                    molecules_with_avg_scores += 1
                    bt.logging.warning(f"   ⚠️  Using average score for {mol['name']} (SMILES: {smiles})")
                
                mol['boltz_score'] = score
                mol['boltz_score_source'] = score_source
                if score is None:
                    molecules_without_scores += 1
                    bt.logging.error(f"   ❌ No score found for {mol['name']} (SMILES: {smiles})")
                else:
                    newly_scored_molecules.append(mol)
            
            bt.logging.info(
                f"   Score extraction complete: "
                f"{molecules_with_individual_scores} individual scores, "
                f"{molecules_with_avg_scores} average scores, "
                f"{molecules_without_scores} without scores"
            )
            
            if newly_scored_molecules:
                write_scores_to_db(newly_scored_molecules)
        
        except FileNotFoundError as e:
            if 'structures' in str(e) or '.npz' in str(e):
                bt.logging.warning(f"Structure file missing: {e}")
                bt.logging.warning("This may indicate a Boltz2 preprocessing issue.")
            else:
                bt.logging.error(f"File not found error: {e}")
        except RuntimeError as e:
            if 'Missing folder' in str(e) or 'lightning_logs' in str(e):
                bt.logging.warning(f"Directory creation issue: {e}")
                try:
                    if hasattr(boltz, 'output_dir'):
                        lightning_logs_path = os.path.join(boltz.output_dir, 'boltz_results_inputs', 'lightning_logs')
                        os.makedirs(lightning_logs_path, exist_ok=True)
                        bt.logging.info("Created missing lightning_logs directory")
                except Exception as dir_err:
                    bt.logging.error(f"Could not create directory: {dir_err}")
            else:
                bt.logging.error(f"Runtime error: {e}")
        except Exception as e:
            bt.logging.error(f"❌ Error scoring molecules with Boltz: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
    
    all_results = molecules_with_db_scores + newly_scored_molecules
    
    for mol in molecules_in_hf:
        mol['boltz_score'] = None
        mol['boltz_score_source'] = 'huggingface_skipped'
        all_results.append(mol)
    
    scored_molecules = sorted(
        all_results,
        key=lambda m: m.get('boltz_score') if m.get('boltz_score') is not None else float('-inf'),
        reverse=True
    )
    
    bt.logging.info(f"   📊 Final results: {len(molecules_with_db_scores)} from DB, "
                   f"{len(newly_scored_molecules)} newly scored, "
                   f"{len(molecules_in_hf)} skipped (HuggingFace)")
    bt.logging.info(f"   📊 Top scores: {[(m['name'], m.get('boltz_score')) for m in scored_molecules[:5] if m.get('boltz_score') is not None]}")
    
    return scored_molecules


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


# ============================================================================
# MAIN MINING LOOP
# ============================================================================

async def generate_unique_molecules_from_top200(
    state: Dict[str, Any], 
    top_200_df: pd.DataFrame,
    hybrid_generator: HybridMoleculeGenerator,
    desired_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules (NOT in HuggingFace) using HYBRID approach.
    70% synthon search + 30% crossover.
    """
    if top_200_df.empty:
        bt.logging.warning("Top 200 DataFrame is empty")
        return []
    
    # ✅ FIX: Use 'name' column instead of 'molecule_name'
    all_names = top_200_df['name'].tolist()
    
    bt.logging.info(f"🧬 Generating {desired_count} unique molecules using HYBRID approach (70% synthon + 30% crossover)...")
    
    unique_molecules = []
    attempts = 0
    max_attempts = 500
    
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1
        
        # Apply hybrid generation (70% synthon + 30% crossover)
        num_synthon = int(desired_count * 0.70)
        num_crossover = int(desired_count * 0.30)
        
        new_molecules = hybrid_generator.apply_hybrid_generation(
            all_names,
            top_200_df,
            num_synthon=num_synthon,
            num_crossover=num_crossover
        )
        
        # Check each new molecule for uniqueness
        for mol in new_molecules:
            if len(unique_molecules) >= desired_count:
                break
            
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
            if molecule_name in [m['name'] for m in unique_molecules]:
                continue
            
            if molecule_name in generated_molecules:
                bt.logging.debug(f"   ⏭️  Molecule {molecule_name} already generated, skipping")
                continue
            
            inchikey = None
            try:
                inchikey = generate_inchikey(smiles) if smiles else None
                if inchikey and inchikey in generated_inchikeys:
                    bt.logging.debug(f"   ⏭️  Molecule {molecule_name} (InChIKey: {inchikey}) already generated, skipping")
                    continue
            except Exception as e:
                bt.logging.debug(f"   Could not generate InChIKey for {molecule_name}: {e}")
            
            # Check if unique (NOT on HuggingFace)
            is_unique = await check_molecule_unique(state, molecule_name, smiles)
            
            if is_unique:
                unique_molecules.append(mol)
                generated_molecules.add(molecule_name)
                if inchikey:
                    generated_inchikeys.add(inchikey)
                bt.logging.info(
                    f"   ✅ Added unique molecule {molecule_name} "
                    f"({len(unique_molecules)}/{desired_count})"
                )
            else:
                bt.logging.debug(f"   ❌ Molecule {molecule_name} is already on HuggingFace")
        
        if len(unique_molecules) >= desired_count:
            break
        
        await asyncio.sleep(0.1)
    
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    bt.logging.info(f"✅ Generated {len(unique_molecules)} unique molecules (attempts: {attempts}, total tracked: {len(generated_molecules)})")
    
    return unique_molecules

async def run_adaptive_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Updated genetic algorithm loop with HYBRID generation (70% synthon + 30% crossover).
    
    Workflow:
    1. Epoch changes
    2. Load top 200 molecules from CSV
    3. Generate 100 unique molecules (HYBRID: 70% synthon + 30% crossover)
    4. Score molecules in batches of 10
    5. Track best molecule
    6. After each batch, check if < 50 blocks remain
       - If YES: Submit best unique molecule and wait for next epoch
       - If NO: Continue to next batch
    7. After scoring all 100 molecules in current round:
       - If < 50 blocks remain: Submit and wait for next epoch
       - If >= 50 blocks remain: Generate another 100 molecules (Round 2) using top molecules as seed
    8. Repeat steps 4-7 for each generation round until submission
    """
    bt.logging.info("🚀 Starting HYBRID genetic algorithm loop (70% synthon + 30% crossover)...")

    csv_path = os.path.join(BASE_DIR, 'data', 'mols.csv')
    last_processed_epoch = state.get('last_processed_epoch', -1)
    desired_unique_count = 100  # Desired number of unique molecules to generate per round

    # Initialize hybrid generator
    hybrid_generator = HybridMoleculeGenerator(HARDCODED_RXN_ID, DB_PATH)

    while not state['shutdown_event'].is_set():
        try:
            # Get current epoch
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            last_submission_epoch = state.get('last_submission_epoch', -1)
            
            # Check if epoch changed - if so, load molecules and start generation
            if current_epoch != last_processed_epoch:
                bt.logging.info(f"\n{'='*70}")
                bt.logging.info(f"🔄 Epoch changed: {last_processed_epoch} → {current_epoch}")
                bt.logging.info(f"{'='*70}")
                
                # Load top 200 molecules from CSV
                top_200_df = load_molecules_from_csv(
                    csv_path,
                    state['current_challenge_targets'],
                    STARTING_EPOCH,
                    HARDCODED_RXN_ID
                )
                
                if top_200_df.empty:
                    bt.logging.warning("No top 200 molecules found, skipping this epoch")
                    last_processed_epoch = current_epoch
                    state['last_processed_epoch'] = current_epoch
                    await asyncio.sleep(10)
                    continue
                
                # Limit to top 200
                top_200_df = top_200_df.head(200)
                state['top_200_df'] = top_200_df
                
                # Initialize synthon library (after we have molecules)
                if not hybrid_generator.synthon_lib_ready:
                    bt.logging.info("🔬 Initializing SynthonLibrary for hybrid generation...")
                    hybrid_generator.initialize_synthon_library()
                
                # Calculate blocks until next epoch
                next_epoch_block = (current_epoch + 1) * state['epoch_length']
                
                # ✅ CONTINUOUS GENERATION AND SCORING LOOP
                # Keep generating and scoring batches until we're within 50 blocks of next epoch
                generation_round = 0
                all_scored_molecules = []
                best_molecule_so_far = None
                best_score_so_far = float('-inf')
                current_seed_pool = top_200_df  # Start with top 200 as seed
                submitted = False
                
                while not submitted:
                    generation_round += 1
                    
                    # Check blocks remaining before starting this generation round
                    current_block_before_round = await state['subtensor'].get_current_block()
                    blocks_remaining_before_round = next_epoch_block - current_block_before_round
                    
                    bt.logging.info(f"\n{'='*70}")
                    bt.logging.info(f"🧬 Generation Round {generation_round}")
                    bt.logging.info(f"   Blocks remaining: {blocks_remaining_before_round}")
                    bt.logging.info(f"{'='*70}")
                    
                    # If we're already within 50 blocks, submit and exit
                    if blocks_remaining_before_round < 50:
                        bt.logging.info(f"⏰ Only {blocks_remaining_before_round} blocks until next epoch, submitting best molecule...")
                        
                        if best_molecule_so_far and last_submission_epoch != current_epoch:
                            # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                            molecule_name = best_molecule_so_far['name']
                            smiles = best_molecule_so_far.get('smiles')
                            
                            if not smiles:
                                bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
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
                            
                            # Check if molecule is already in DB
                            db_score = get_score_from_db(best_molecule_so_far['name'])
                            if db_score is not None:
                                bt.logging.info(
                                    f"⚠️  Best molecule {best_molecule_so_far['name']} already in DB "
                                    f"(score: {db_score:.6f}), but submitting as top molecule"
                                )
                            
                            bt.logging.info(
                                f"🏆 Best molecule: {best_molecule_so_far['name']} "
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
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                                bt.logging.info(f"⏳ Waiting for next epoch to start...")
                            except Exception as e:
                                bt.logging.error(f"❌ Error submitting response: {e}")
                                import traceback
                                bt.logging.error(traceback.format_exc())
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                            submitted = True
                            break
                        else:
                            bt.logging.warning("No valid molecules to submit")
                            last_processed_epoch = current_epoch
                            state['last_processed_epoch'] = current_epoch
                            submitted = True
                            break
                    
                    # ✅ GENERATE 100 UNIQUE MOLECULES FOR THIS ROUND
                    bt.logging.info(f"🧬 Generating {desired_unique_count} unique molecules for round {generation_round}...")
                    unique_molecules = await generate_unique_molecules_from_top200(
                        state, current_seed_pool, hybrid_generator, desired_unique_count
                    )
                    
                    if not unique_molecules:
                        bt.logging.warning(f"Failed to generate unique molecules in round {generation_round}, stopping generation")
                        last_processed_epoch = current_epoch
                        state['last_processed_epoch'] = current_epoch
                        submitted = True
                        break
                    
                    bt.logging.info(f"✅ Generated {len(unique_molecules)} unique molecules for round {generation_round}")
                    
                    # ✅ SCORE MOLECULES IN BATCHES OF 10
                    batch_size = 10
                    total_batches = (len(unique_molecules) + batch_size - 1) // batch_size
                    
                    bt.logging.info(f"🔬 Scoring {len(unique_molecules)} molecules in {total_batches} batches of {batch_size}...")
                    
                    for batch_idx in range(total_batches):
                        current_block_before_batch = await state['subtensor'].get_current_block()
                        blocks_remaining = next_epoch_block - current_block_before_batch
                        
                        # ✅ CHECK BLOCKS REMAINING AFTER EACH BATCH
                        if blocks_remaining < 50:
                            bt.logging.info(f"⏰ Only {blocks_remaining} blocks until next epoch after batch {batch_idx}, submitting best molecule...")
                            
                            if best_molecule_so_far and last_submission_epoch != current_epoch:
                                # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                                molecule_name = best_molecule_so_far['name']
                                smiles = best_molecule_so_far.get('smiles')
                                
                                if not smiles:
                                    bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
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
                                
                                # Check if molecule is already in DB
                                db_score = get_score_from_db(best_molecule_so_far['name'])
                                if db_score is not None:
                                    bt.logging.info(
                                        f"⚠️  Best molecule {best_molecule_so_far['name']} already in DB "
                                        f"(score: {db_score:.6f}), but submitting as top molecule"
                                    )
                                
                                bt.logging.info(
                                    f"🏆 Best molecule: {best_molecule_so_far['name']} "
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
                                    last_processed_epoch = current_epoch
                                    state['last_processed_epoch'] = current_epoch
                                    bt.logging.info(f"⏳ Waiting for next epoch to start...")
                                except Exception as e:
                                    bt.logging.error(f"❌ Error submitting response: {e}")
                                    import traceback
                                    bt.logging.error(traceback.format_exc())
                                    last_processed_epoch = current_epoch
                                    state['last_processed_epoch'] = current_epoch
                            else:
                                bt.logging.warning("No valid molecules to submit")
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                            submitted = True
                            break  # Exit batch loop
                        
                        # Get batch of molecules to score
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(unique_molecules))
                        batch = unique_molecules[start_idx:end_idx]
                        
                        bt.logging.info(
                            f"📦 Round {generation_round}, Batch {batch_idx + 1}/{total_batches}: "
                            f"Scoring {len(batch)} molecules "
                            f"(blocks remaining: {blocks_remaining})"
                        )
                        
                        # Score this batch
                        scored_batch = await score_molecules_with_boltz(state, batch)
                        
                        if scored_batch:
                            # Filter molecules with valid scores
                            batch_with_scores = [m for m in scored_batch if m.get('boltz_score') is not None]
                            all_scored_molecules.extend(batch_with_scores)
                            
                            # Update best molecule so far
                            for mol in batch_with_scores:
                                score = mol.get('boltz_score')
                                if score is not None and score > best_score_so_far:
                                    best_score_so_far = score
                                    best_molecule_so_far = mol
                                    bt.logging.info(
                                        f"🏆 New best in round {generation_round}, batch {batch_idx + 1}: "
                                        f"{mol['name']} (score: {score:.6f})"
                                    )
                    
                    # After completing all batches for this round, check if we should continue
                    if submitted:
                        break
                    
                    # Check blocks remaining after all batches in this round
                    current_block_after_round = await state['subtensor'].get_current_block()
                    blocks_remaining_after_round = next_epoch_block - current_block_after_round
                    
                    bt.logging.info(f"✅ Completed round {generation_round}: Scored {len(unique_molecules)} molecules")
                    bt.logging.info(f"   ⏱️  Blocks remaining after round {generation_round}: {blocks_remaining_after_round}")
                    bt.logging.info(f"   🏆 Best molecule so far: {best_molecule_so_far['name'] if best_molecule_so_far else 'None'} (score: {best_score_so_far:.6f if best_molecule_so_far else 'N/A'})")
                    
                    # ✅ DECISION: Continue to next round or submit
                    if blocks_remaining_after_round < 50:
                        bt.logging.info(f"⏰ Only {blocks_remaining_after_round} blocks until next epoch, submitting best molecule...")
                        
                        if best_molecule_so_far and last_submission_epoch != current_epoch:
                            # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                            molecule_name = best_molecule_so_far['name']
                            smiles = best_molecule_so_far.get('smiles')
                            
                            if not smiles:
                                bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, finding next best...")
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
                            
                            # Check if molecule is already in DB
                            db_score = get_score_from_db(best_molecule_so_far['name'])
                            if db_score is not None:
                                bt.logging.info(
                                    f"⚠️  Best molecule {best_molecule_so_far['name']} already in DB "
                                    f"(score: {db_score:.6f}), but submitting as top molecule"
                                )
                            
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
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                                bt.logging.info(f"⏳ Waiting for next epoch to start...")
                            except Exception as e:
                                bt.logging.error(f"❌ Error submitting response: {e}")
                                import traceback
                                bt.logging.error(traceback.format_exc())
                                last_processed_epoch = current_epoch
                                state['last_processed_epoch'] = current_epoch
                        else:
                            bt.logging.warning("No valid molecules to submit")
                            last_processed_epoch = current_epoch
                            state['last_processed_epoch'] = current_epoch
                        submitted = True
                        break
                    else:
                        # ✅ CONTINUE TO NEXT GENERATION ROUND
                        # Use top 30 scored molecules as seed for next round
                        top_scored = sorted(
                            all_scored_molecules,
                            key=lambda m: m.get('boltz_score', float('-inf')),
                            reverse=True
                        )[:30]
                        
                        if top_scored:
                            # Create DataFrame with top scored molecules for next round seed
                            current_seed_pool = pd.DataFrame(top_scored)
                            bt.logging.info(
                                f"📈 Continuing to round {generation_round + 1}: "
                                f"Using top 30 scored molecules as seed pool "
                                f"(top score: {top_scored[0].get('boltz_score', 'N/A'):.6f})"
                            )
                        else:
                            # Fallback to original top 200
                            current_seed_pool = state['top_200_df']
                            bt.logging.warning(
                                f"📈 Continuing to round {generation_round + 1}: "
                                f"No scored molecules available, using original top 200 as seed"
                            )
                
                # Mark epoch as processed
                if last_processed_epoch != current_epoch:
                    last_processed_epoch = current_epoch
                    state['last_processed_epoch'] = current_epoch
            
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

async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Initialize score_results database
    2. Import and initialize BoltzWrapper
    3. Load molecules from CSV
    4. Initialize hybrid generator with synthon library
    5. Prepare for main loop
    """
    bt.logging.info("🚀 Starting STARTUP phase: Initialize DB, Boltz, Load CSV & Synthon Library...")

    try:
        # Initialize score_results database
        bt.logging.info("💾 Initializing score_results database...")
        init_score_results_db()
        bt.logging.info(f"✅ Score results database initialized at {SCORE_RESULTS_DB}")
        
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
        
        # Load molecules from CSV
        bt.logging.info("📂 Loading molecules from CSV...")
        molecules_df = load_molecules_from_csv(
            REACTION_TRAIN_CSV,
            state['current_challenge_targets'],
            STARTING_EPOCH,
            HARDCODED_RXN_ID
        )
        
        if molecules_df.empty:
            bt.logging.warning("⚠️  No molecules loaded from CSV")
            state['top_pool'] = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
            state['seen_inchikeys'] = set()
        else:
            bt.logging.info(f"✅ Loaded {len(molecules_df)} molecules from CSV")
            state['top_pool'] = molecules_df.copy()
            state['seen_inchikeys'] = set(molecules_df['InChIKey'].tolist())
        
        bt.logging.info(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Sample molecules: {state['top_pool']['name'].head(3).tolist() if not state['top_pool'].empty else 'None'}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True

    except Exception as e:
        bt.logging.error(f"Error in startup phase: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

def _import_boltz_wrapper():
    """
    Import BoltzWrapper following the same pattern as DataGenerator/main.py.
    This is done as a function so it can be called after logging is initialized.
    """
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
        bt.logging.info(f"✅ BoltzWrapper imported successfully from {BOLTZ_SCORING_DIR}")
        return True
        
    except ImportError as e:
        bt.logging.warning(f"⚠️  Failed to import BoltzWrapper: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return False
    except Exception as e:
        bt.logging.warning(f"⚠️  Error setting up BoltzWrapper: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return False

async def run_miner(config: argparse.Namespace) -> None:
    """Main mining loop with hybrid generation (70% synthon + 30% crossover)."""

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
        
        bt.logging.info(f"Using hardcoded reaction ID: {state['rxn_id']}")
        bt.logging.info(
            f"Startup targets: {startup_proteins['targets']}, "
            f"antitargets: {startup_proteins['antitargets']}"
        )

        # Run startup phase
        try:
            await startup_phase(state)
            bt.logging.info("✅ Startup phase completed!")
        except Exception as e:
            bt.logging.error(f"Error in startup: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

        # Launch adaptive GA loop with hybrid generation
        try:
            state['ga_task'] = asyncio.create_task(run_adaptive_genetic_loop(state))
            bt.logging.info("✅ Adaptive hybrid GA loop started!")
        except Exception as e:
            bt.logging.error(f"Error starting GA loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # Main monitoring loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            if current_block % epoch_length == 0:
                current_epoch = current_block // epoch_length
                bt.logging.info(
                    f"⏰ Epoch boundary at block {current_block} (epoch {current_epoch})"
                )

            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            import traceback
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("⛔ Keyboard interrupt detected. Exiting miner.")
            state['shutdown_event'].set()
            break

async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())