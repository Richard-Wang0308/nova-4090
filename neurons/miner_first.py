#!/usr/bin/env python3
"""
SIMPLIFIED BITTENSOR MINER - Genetic Algorithm Based Molecule Generation

Simple workflow:
1. Load molecules from CSV at startup
2. Initialize BoltzWrapper for scoring
3. Apply genetic operations (CROSSOVER ONLY)
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
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 21074

# ✅ CSV for loading initial molecules
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'BoltzPredictor', 'data', 'mols.csv')

# ✅ Database for storing scored molecules
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
)
from utils.molecules import molecule_unique_for_protein_hf
from molecules_base import (
    generate_inchikey,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock

# BoltzWrapper import - following the same pattern as DataGenerator/main.py
# We'll import it lazily in startup_phase after logging is initialized
BOLTZ_AVAILABLE = False
BoltzWrapper = None


# ============================================================================
# PyTorch 2.6+ Compatibility
# ============================================================================

def safe_torch_load(path, map_location='cpu'):
    """
    Safely load PyTorch checkpoint with numpy scalar support (PyTorch 2.6+).
    
    This function handles PyTorch 2.6+ compatibility by:
    - Adding numpy.core.multiarray.scalar to safe globals
    - Using weights_only=False for backward compatibility
    
    Args:
        path: Path to checkpoint file (str or Path)
        map_location: Device to load to (default: 'cpu')
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    import torch
    import numpy as np
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    bt.logging.info(f"Loading checkpoint from {path}...")
    
    try:
        # Add safe globals for numpy scalars (PyTorch 2.6+)
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        
        # Load checkpoint with weights_only=False for compatibility
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
# ✅ GENETIC ALGORITHM OPERATIONS (CROSSOVER ONLY)
# ============================================================================

class GeneticAlgorithmOperator:
    """Performs genetic algorithm operations on molecules (CROSSOVER ONLY)."""
    
    def __init__(self, rxn_id: int, db_path: str):
        """Initialize GA operator."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()  # Track generated molecule names
    
    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """
        Crossover two molecules by swapping random components.
        
        Format: rxn:3:comp1:comp2:comp3 (3-component reaction)
        We can swap comp1, comp2, or comp3 (indices 2, 3, 4) between two molecules
        rxn and 3 (indices 0, 1) cannot be replaced
        
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
            
            bt.logging.debug(f"Crossover: {mol_name_1} x {mol_name_2}")
            bt.logging.debug(f"  Parts1: {parts1}, Parts2: {parts2}")
            
            # Expected format: rxn:3:comp1:comp2:comp3 (5 parts each for 3-component reaction)
            if (len(parts1) != 5 or len(parts2) != 5 or 
                parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                bt.logging.debug(f"Invalid format for crossover: expected 5 parts (rxn:3:comp1:comp2:comp3), got {len(parts1)} and {len(parts2)}")
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
            
            # Randomly select which component to swap: comp1 (index 2), comp2 (index 3), or comp3 (index 4)
            # rxn (index 0) and 3 (index 1) cannot be replaced
            swap_idx = random.choice([2, 3, 4])
            bt.logging.debug(f"Swapping component at index {swap_idx} (comp{swap_idx-1})")
            
            # Create offspring by swapping one component
            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)
            
            bt.logging.debug(f"Offspring: {offspring_name}")
            
            # ✅ CHECK IF WE ALREADY GENERATED THIS MOLECULE
            if offspring_name in self.generated_molecule_names:
                bt.logging.debug(f"⚠️  Offspring {offspring_name} already generated in this batch")
                return None
            
            # Validate offspring
            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        # ✅ TRACK THIS MOLECULE NAME
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
    
    def apply_genetic_operations(
        self,
        top_molecules: List[str],
        num_crossovers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Apply genetic operations (CROSSOVER ONLY) to top molecules.
        
        Args:
            top_molecules: List of top molecule names
            num_crossovers: Number of crossovers to attempt
            
        Returns:
            List of new molecules with their SMILES and names (in order generated)
        """
        new_molecules = []
        
        # ✅ RESET TRACKING FOR THIS BATCH
        self.generated_molecule_names.clear()
        
        bt.logging.info(f"🧬 Applying CROSSOVER-ONLY genetic operations to top {len(top_molecules)} molecules...")
        bt.logging.info(f"   Sample molecules: {top_molecules[:3]}")
        
        # Apply crossovers only
        crossover_attempts = 0
        crossovers_created = 0
        
        for i in range(num_crossovers):
            parent1 = random.choice(top_molecules)
            parent2 = random.choice(top_molecules)
            crossover_attempts += 1
            
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
        bt.logging.info(f"🧬 Generated {len(new_molecules)} new molecules from crossover operations")
        
        # ✅ VERIFY ORDER IS PRESERVED
        bt.logging.info(f"   Generated molecules in order: {[m['name'] for m in new_molecules]}")
        
        return new_molecules


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
    """Initializes wallet, subtensor, and metagraph."""
    bt.logging.info("Setting up Bittensor objects.")

    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            epoch_length = 361
            bt.logging.info(f"Epoch length: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        raise


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
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on score for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)
        
        conn.commit()
        conn.close()
        bt.logging.debug(f"Initialized score_results database at {db_path}")
    except Exception as e:
        bt.logging.error(f"Error initializing score_results database: {e}")


def get_score_from_db(molecule_name: str, db_path: str = None) -> Optional[float]:
    """
    Get score for a molecule from the database.
    
    Returns:
        Score if found, None otherwise
    """
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
    """
    Write scored molecules to the database.
    
    Args:
        molecules: List of molecule dicts with 'name' and 'boltz_score'
        db_path: Path to database (default: SCORE_RESULTS_DB)
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    if not molecules:
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prepare data for insertion (only molecules with valid scores)
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
    """
    Get scores for multiple molecules from the database in batch.
    
    Returns:
        Dictionary mapping molecule_name -> score
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    if not molecule_names:
        return {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Use IN clause for batch query
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


def generate_random_initial_molecules(
    rxn_id: int,
    num_molecules: int = 200,
    db_path: str = None,
    subnet_config: dict = None
) -> pd.DataFrame:
    """
    Generate random initial molecules from the combinatorial database when CSV is not available.
    Uses the same approach as generate_valid_random_molecules_batch from molecules_base.py.
    
    Args:
        rxn_id: Reaction ID (e.g., 3)
        num_molecules: Number of random molecules to generate
        db_path: Path to combinatorial database (default: DB_PATH)
        subnet_config: Subnet configuration dict with validation parameters (min_heavy_atoms, etc.)
        
    Returns:
        DataFrame with columns: name, smiles, InChIKey, score
        Score will be None for randomly generated molecules
    """
    from molecules_base import (
        get_molecules_by_role,
        generate_molecules_from_pools,
        validate_molecules
    )
    from combinatorial_db.reactions import get_reaction_info
    
    if db_path is None:
        db_path = DB_PATH
    
    if not os.path.exists(db_path):
        bt.logging.warning(f"Database not found at {db_path}, cannot generate random molecules")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    # Default subnet_config if not provided
    if subnet_config is None:
        subnet_config = {
            'min_heavy_atoms': 20,
            'max_heavy_atoms': 45,
            'min_rotatable_bonds': 1,
            'max_rotatable_bonds': 10
        }
    else:
        # Ensure required parameters are set
        if 'max_heavy_atoms' not in subnet_config:
            subnet_config['max_heavy_atoms'] = 45
        if 'min_heavy_atoms' not in subnet_config:
            subnet_config['min_heavy_atoms'] = 20
        if 'max_rotatable_bonds' not in subnet_config:
            subnet_config['max_rotatable_bonds'] = 10
        if 'min_rotatable_bonds' not in subnet_config:
            subnet_config['min_rotatable_bonds'] = 1
    
    bt.logging.info(f"🎲 Generating {num_molecules} random initial molecules for rxn:{rxn_id}...")
    
    try:
        # Get reaction info to determine roles and component count
        reaction_info = get_reaction_info(rxn_id, db_path)
        if not reaction_info:
            bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        smarts, roleA, roleB, roleC = reaction_info
        is_three_component = roleC is not None and roleC != 0
        
        bt.logging.info(f"   Reaction {rxn_id}: {'3-component' if is_three_component else '2-component'}")
        
        # Get molecules filtered by role (this is critical!)
        molecules_A = get_molecules_by_role(roleA, db_path)
        molecules_B = get_molecules_by_role(roleB, db_path)
        molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
        
        if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
            bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        bt.logging.info(
            f"   Found {len(molecules_A)} A components, {len(molecules_B)} B components"
            f"{f', {len(molecules_C)} C components' if is_three_component else ''}"
        )
        
        # Generate molecules using the proper function from molecules_base
        # This ensures we use role-filtered molecules and proper validation
        batch_size = min(500, num_molecules * 3)  # Generate more than needed to account for validation failures
        valid_dfs = []
        seen_keys = set()
        total_valid = 0
        max_batches = 20  # Limit number of batches to avoid infinite loops
        
        batch_count = 0
        while total_valid < num_molecules and batch_count < max_batches:
            batch_count += 1
            needed = num_molecules - total_valid
            batch_size_actual = min(batch_size, needed * 3)
            
            # Generate molecule names using role-filtered pools
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, 
                is_three_component, seed=None, component_weights=None
            )
            
            if not batch_molecules:
                bt.logging.warning(f"   Batch {batch_count}: No molecules generated")
                continue
            
            # Convert to DataFrame and validate
            batch_df = pd.DataFrame({"name": batch_molecules})
            batch_df = batch_df[batch_df["name"].notna()]
            
            if batch_df.empty:
                continue
            
            # Validate molecules (checks heavy atoms, rotatable bonds, generates SMILES and InChIKey)
            batch_df = validate_molecules(batch_df, subnet_config)
            
            if batch_df.empty:
                bt.logging.debug(f"   Batch {batch_count}: No valid molecules after validation")
                continue
            
            # Filter by max_heavy_atoms if specified (validate_molecules doesn't check this)
            if 'max_heavy_atoms' in subnet_config and subnet_config['max_heavy_atoms'] is not None:
                if 'heavy_atoms' in batch_df.columns:
                    max_heavy = subnet_config['max_heavy_atoms']
                    batch_df = batch_df[batch_df['heavy_atoms'] <= max_heavy]
                    if batch_df.empty:
                        bt.logging.debug(f"   Batch {batch_count}: No molecules with heavy_atoms <= {max_heavy}")
                        continue
                else:
                    bt.logging.warning(f"   Batch {batch_count}: heavy_atoms column not found in batch_df")
            
            # Remove duplicates by InChIKey
            batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
            
            # Filter out already seen InChIKeys
            mask = ~batch_df["InChIKey"].isin(seen_keys)
            batch_df = batch_df[mask]
            
            if batch_df.empty:
                continue
            
            # Add score column (None for random molecules)
            batch_df['score'] = None
            
            seen_keys.update(batch_df["InChIKey"].values)
            valid_dfs.append(batch_df[["name", "smiles", "InChIKey", "score"]].copy())
            total_valid += len(batch_df)
            
            bt.logging.info(f"   Batch {batch_count}: Generated {len(batch_df)} valid molecules (total: {total_valid}/{num_molecules})")
            
            if total_valid >= num_molecules:
                break
        
        if not valid_dfs:
            bt.logging.warning(f"Failed to generate any random molecules after {batch_count} batches")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        # Concatenate all DataFrames
        result_df = pd.concat(valid_dfs, ignore_index=True)
        result_df = result_df.head(num_molecules)  # Take exactly num_molecules
        
        bt.logging.info(
            f"✅ Generated {len(result_df)} random initial molecules "
            f"(batches: {batch_count}, total attempted: {sum(len(df) for df in valid_dfs)})"
        )
        
        return result_df
        
    except Exception as e:
        bt.logging.error(f"Error generating random molecules: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_csv(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int
) -> pd.DataFrame:
    """
    Load molecules from CSV file, sorted by final_score descending.
    
    Returns DataFrame with columns: name, smiles, InChIKey, score
    Sorted by score (final_score from CSV) descending.
    """
    if not os.path.exists(csv_path):
        bt.logging.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        bt.logging.info(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        # Filter by target protein
        if 'target_protein' in df.columns:
            df = df[df['target_protein'].isin(target_proteins)]
        else:
            bt.logging.warning("CSV file does not have 'target_protein' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        # Filter by epoch
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            bt.logging.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        # Filter by reaction ID
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
        
        for _, row in df.iterrows():
            molecule_name = row['molecule_name']
            
            try:
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    bt.logging.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    bt.logging.debug(f"Could not generate InChIKey for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Get final_score from CSV if available
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
            
            # ✅ SORT BY final_score DESCENDING (highest scores first)
            if 'score' in result_df.columns:
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
                bt.logging.warning("No 'score' column found, molecules not sorted by score")
        else:
            bt.logging.warning(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count})"
            )
        
        return result_df
        
    except Exception as e:
        bt.logging.error(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


# ============================================================================
# ✅ ADAPTIVE GENETIC OPERATIONS WITH SUBMISSION LOGIC (CROSSOVER ONLY)
# ============================================================================

async def score_molecules_with_boltz(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Score molecules using BoltzWrapper.
    Checks database and HuggingFace Hub first to avoid redundant scoring.
    
    Args:
        state: Miner state dictionary
        molecules: List of molecule dicts with 'name', 'smiles', 'InChIKey'
        
    Returns:
        List of molecules with added 'boltz_score' field, sorted by score descending
    """
    if state.get('boltz_wrapper') is None:
        bt.logging.warning("BoltzWrapper not available, skipping scoring")
        return molecules
    
    if not molecules:
        return molecules
    
    bt.logging.info(f"🔬 Processing {len(molecules)} molecules for scoring...")
    
    # Initialize score database
    init_score_results_db()
    
    # Separate molecules into: already scored, in HuggingFace, need scoring
    molecules_to_score = []
    molecules_with_db_scores = []
    molecules_in_hf = []
    
    # Get target protein for HuggingFace check
    target_proteins = state.get('current_challenge_targets', [])
    primary_target = target_proteins[0] if target_proteins else None
    
    # Batch check database for all molecules
    molecule_names = [mol['name'] for mol in molecules]
    db_scores = batch_get_scores_from_db(molecule_names)
    
    bt.logging.info(f"   Found {len(db_scores)} molecules already scored in database")
    
    # Categorize molecules
    for mol in molecules:
        molecule_name = mol['name']
        smiles = mol.get('smiles')
        
        # Check database first
        if molecule_name in db_scores:
            mol['boltz_score'] = db_scores[molecule_name]
            mol['boltz_score_source'] = 'database'
            molecules_with_db_scores.append(mol)
            bt.logging.debug(f"   ✓ {molecule_name}: score from DB = {db_scores[molecule_name]:.6f}")
            continue
        
        # Check HuggingFace Hub (skip if already submitted)
        if primary_target and smiles:
            try:
                is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                if not is_unique_hf:
                    bt.logging.debug(f"   ⏭️  {molecule_name}: already in HuggingFace, skipping")
                    molecules_in_hf.append(mol)
                    continue
            except Exception as e:
                bt.logging.debug(f"   Error checking HuggingFace for {molecule_name}: {e}")
                # Continue to scoring if check fails
        
        # Need to score this molecule
        molecules_to_score.append(mol)
    
    bt.logging.info(
        f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
        f"{len(molecules_in_hf)} in HuggingFace (skipped), "
        f"{len(molecules_to_score)} need scoring"
    )
    
    # Score molecules that need scoring
    newly_scored_molecules = []
    if molecules_to_score:
        bt.logging.info(f"🔬 Scoring {len(molecules_to_score)} new molecules with Boltz...")
        
        boltz = state['boltz_wrapper']
        config = state['config']
        target_proteins = state.get('current_challenge_targets', [])
        antitarget_proteins = state.get('current_challenge_antitargets', [])
        
        if not target_proteins:
            bt.logging.warning("No target proteins available for scoring")
            # Return what we have from DB
            all_results = molecules_with_db_scores + [mol for mol in molecules if mol.get('boltz_score') is None]
            return all_results
        
        primary_target = target_proteins[0]
        
        try:
            # Clean up any previous failed runs to avoid conflicts (same as DataGenerator)
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
            
            # Ensure output directories exist before scoring (same as DataGenerator)
            processed_dir = os.path.join(output_dir, 'processed')
            structures_dir = os.path.join(processed_dir, 'structures')
            records_dir = os.path.join(processed_dir, 'records')
            msa_dir = os.path.join(processed_dir, 'msa')
            predictions_dir = os.path.join(output_dir, 'predictions')
            
            # Create all necessary directories
            os.makedirs(structures_dir, exist_ok=True)
            os.makedirs(records_dir, exist_ok=True)
            os.makedirs(msa_dir, exist_ok=True)
            os.makedirs(predictions_dir, exist_ok=True)
        
            # Prepare data structures for BoltzWrapper (same as DataGenerator)
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
            
            # Build subnet_config from config (same pattern as DataGenerator)
            # ✅ IMPORTANT: Set num_molecules_boltz to score ALL molecules, not just 1
            num_molecules_to_score = len(molecules_to_score)
            subnet_config = {
                'weekly_target': primary_target,
                'num_antitargets': len(antitarget_proteins),
                'binding_pocket': getattr(config, 'binding_pocket', None),
                'max_distance': getattr(config, 'max_distance', None),
                'force': getattr(config, 'force', False),
                'num_molecules_boltz': num_molecules_to_score,  # ✅ Score ALL molecules, not just 1
                'boltz_metric': getattr(config, 'boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']),
                'combination_strategy': getattr(config, 'combination_strategy', 'heavy_atom_normalization'),
                'sample_selection': getattr(config, 'sample_selection', 'first'),
            }
            
            # Use dummy block hash for scoring
            final_block_hash = "0x" + "0" * 64
            
            bt.logging.info(f"   Running Boltz scoring for {len(molecules_to_score)} molecules...")
            start_time = time.time()
            
            # Run scoring (this is synchronous, so we run it in executor to avoid blocking)
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
            
            # Extract scores following DataGenerator pattern
            # Important: When multiple molecules have the same SMILES, Boltz only processes unique SMILES
            # We need to map the score back to all molecules with that SMILES
            uid = 0
            
            # First, build a SMILES -> score mapping from per_molecule_metric (most reliable)
            smiles_to_score = {}
            if uid in boltz.per_molecule_metric:
                smiles_to_score = boltz.per_molecule_metric[uid].copy()
                bt.logging.info(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores from per_molecule_metric")
                bt.logging.debug(f"   Unique SMILES scored: {list(smiles_to_score.keys())[:5]}...")
            
            # Also check target_scores as a fallback (by index in valid_molecules_by_uid)
            target_scores_list = None
            target_scores = score_dict[uid].get('target_scores', [[]])
            if target_scores and len(target_scores[0]) > 0:
                target_scores_list = target_scores[0] if isinstance(target_scores[0], list) else [target_scores[0]]
                bt.logging.debug(f"   Found {len(target_scores_list)} scores in target_scores")
            
            # Fallback: try to get from score_dict's boltz_score (average of all) - only if no individual scores
            avg_score = None
            if not smiles_to_score and not target_scores_list:
                avg_score = score_dict[uid].get('boltz_score')
                if avg_score is not None and isinstance(avg_score, (int, float)):
                    bt.logging.warning(f"   ⚠️  Only average boltz_score available: {avg_score} (using as fallback)")
            
            # Assign scores to molecules_to_score
            molecules_with_individual_scores = 0
            molecules_with_avg_scores = 0
            molecules_without_scores = 0
            
            for mol_idx, mol in enumerate(molecules_to_score):
                smiles = mol['smiles']
                score = None
                score_source = None
                
                # Priority 1: Get score from per_molecule_metric by SMILES (works for duplicate SMILES)
                if smiles in smiles_to_score:
                    score = smiles_to_score[smiles]
                    score_source = "boltz_scoring"
                    molecules_with_individual_scores += 1
                # Priority 2: Try target_scores by index (if available and index is valid)
                elif target_scores_list and mol_idx < len(target_scores_list):
                    score = target_scores_list[mol_idx]
                    score_source = "boltz_scoring"
                    molecules_with_individual_scores += 1
                # Priority 3: Try to find SMILES in valid_molecules_by_uid and get score by index
                elif target_scores_list:
                    try:
                        valid_idx = valid_molecules_by_uid[uid]['smiles'].index(smiles)
                        if valid_idx < len(target_scores_list):
                            score = target_scores_list[valid_idx]
                            score_source = "boltz_scoring"
                            molecules_with_individual_scores += 1
                    except (ValueError, IndexError):
                        pass
                # Priority 4: Use average as last resort (but log warning)
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
            
            # Write newly scored molecules to database
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
    
    # Combine all results: DB scores, newly scored, HuggingFace skipped (with None score)
    all_results = molecules_with_db_scores + newly_scored_molecules
    
    # Add HuggingFace molecules with None score (for tracking)
    for mol in molecules_in_hf:
        mol['boltz_score'] = None
        mol['boltz_score_source'] = 'huggingface_skipped'
        all_results.append(mol)
    
    # Sort by boltz_score descending (None scores go to end)
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


async def collect_and_process_submissions(state: Dict[str, Any], start_epoch: int, csv_path: str) -> pd.DataFrame:
    """
    Collect submissions using prepare_training_data.py and process CSV.
    
    Since prepare_training_data.py appends the whole result, we need to:
    1. Run prepare_training_data.py
    2. Load CSV and deduplicate (remove duplicates)
    3. Filter for rxn:3 molecules
    4. Sort by score and return top 30
    """
    import subprocess
    
    bt.logging.info(f"📥 Collecting submissions from epoch {start_epoch}...")
    
    # Run prepare_training_data.py
    script_path = os.path.join(BASE_DIR, 'BoltzPredictor', 'scripts', 'prepare_training_data.py')
    if not os.path.exists(script_path):
        bt.logging.error(f"prepare_training_data.py not found at {script_path}")
        return pd.DataFrame()
    
    try:
        # Run the script
        result = subprocess.run(
            ['python3', script_path, '--start_epoch', str(start_epoch), '--output', csv_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            bt.logging.error(f"prepare_training_data.py failed: {result.stderr}")
            return pd.DataFrame()
        
        bt.logging.info(f"✅ prepare_training_data.py completed successfully")
        
    except subprocess.TimeoutExpired:
        bt.logging.error("prepare_training_data.py timed out")
        return pd.DataFrame()
    except Exception as e:
        bt.logging.error(f"Error running prepare_training_data.py: {e}")
        return pd.DataFrame()
    
    # Load and process CSV
    if not os.path.exists(csv_path):
        bt.logging.warning(f"CSV file not created at {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        bt.logging.info(f"📊 Loaded {len(df)} rows from CSV")
        
        # Since prepare_training_data appends the whole result, remove duplicates and rewrite
        original_len = len(df)
        if 'molecule_name' in df.columns:
            df = df.drop_duplicates(subset=['molecule_name'], keep='last')
            if len(df) < original_len:
                bt.logging.info(f"   Removed {original_len - len(df)} duplicates, rewriting CSV...")
                # Rewrite CSV without duplicates
                df.to_csv(csv_path, index=False)
                bt.logging.info(f"   ✅ Rewrote CSV with {len(df)} unique rows")
            else:
                bt.logging.info(f"   No duplicates found ({len(df)} rows)")
        
        # Filter for rxn:3 molecules only
        df = df[df['molecule_name'].str.startswith('rxn:3:', na=False)]
        bt.logging.info(f"   After filtering rxn:3: {len(df)} rows")
        
        if df.empty:
            bt.logging.warning("No rxn:3 molecules found in CSV")
            return pd.DataFrame()
        
        # Sort by final_score descending
        if 'final_score' in df.columns:
            df = df.sort_values(by='final_score', ascending=False, na_position='last')
            bt.logging.info(f"   Sorted by final_score")
        
        # Take top 200
        top_200 = df.head(200)
        bt.logging.info(f"✅ Selected top 200 molecules (scores: {top_200['final_score'].head(5).tolist() if 'final_score' in top_200.columns else 'N/A'})")
        
        return top_200
        
    except Exception as e:
        bt.logging.error(f"Error processing CSV: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return pd.DataFrame()


async def generate_unique_molecules_from_top200(
    state: Dict[str, Any], 
    top_200_df: pd.DataFrame,
    desired_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules (NOT in HuggingFace) using genetic algorithm from top 200 molecules.
    Uses adaptive pool sizing: starts with top 30, increases to 50, 100, 150, 200 if generation fails.
    Keeps generating until desired_count unique molecules are found.
    """
    if top_200_df.empty:
        bt.logging.warning("Top 200 DataFrame is empty")
        return []
    
    ga_operator = GeneticAlgorithmOperator(HARDCODED_RXN_ID, DB_PATH)
    
    # Get all molecule names from top 200
    all_names = top_200_df['molecule_name'].tolist()
    
    # Adaptive pool sizes: start with 30, increase if generation fails
    pool_sizes = [30, 50, 100, 150, 200]
    current_pool_size_idx = 0
    current_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
    
    bt.logging.info(f"🧬 Generating {desired_count} unique molecules using adaptive pool sizing (starting with top {current_pool_size})...")
    
    unique_molecules = []
    attempts = 0
    max_attempts = 500  # Maximum generation attempts
    last_successful_attempt = 0  # Track when we last found a unique molecule
    
    # Get or initialize generated molecules tracking set
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1
        
        # Check if we should increase pool size (if first 100 attempts failed to find new unique molecules)
        if attempts - last_successful_attempt >= 100 and current_pool_size_idx < len(pool_sizes) - 1:
            current_pool_size_idx += 1
            new_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
            if new_pool_size > current_pool_size:
                current_pool_size = new_pool_size
                bt.logging.info(f"📈 Increasing pool size to top {current_pool_size} (after {attempts - last_successful_attempt} failed attempts)")
                last_successful_attempt = attempts  # Reset counter
        
        # Use current pool size
        current_pool_names = all_names[:current_pool_size]
        
        # Apply genetic operations (crossover)
        new_molecules = ga_operator.apply_genetic_operations(
            current_pool_names,
            num_crossovers=10  # 10 crossovers per batch
        )
        
        # Check each new molecule for uniqueness
        for mol in new_molecules:
            if len(unique_molecules) >= desired_count:
                break
            
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
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
                    bt.logging.debug(f"   ⏭️  Molecule {molecule_name} (InChIKey: {inchikey}) already generated, skipping")
                    continue
            except Exception as e:
                bt.logging.debug(f"   Could not generate InChIKey for {molecule_name}: {e}")
            
            # Check if unique (NOT on HuggingFace)
            is_unique = await check_molecule_unique(state, molecule_name, smiles)
            
            if is_unique:
                unique_molecules.append(mol)
                # Track this molecule as generated
                generated_molecules.add(molecule_name)
                if inchikey:
                    generated_inchikeys.add(inchikey)
                last_successful_attempt = attempts  # Update last successful attempt
                bt.logging.info(
                    f"   ✅ Added unique molecule {molecule_name} "
                    f"(pool size: {current_pool_size}, {len(unique_molecules)}/{desired_count})"
                )
            else:
                bt.logging.debug(
                    f"   ❌ Molecule {molecule_name} is already on HuggingFace"
                )
        
        if len(unique_molecules) >= desired_count:
            break
        
        # Small delay to avoid overwhelming
        await asyncio.sleep(0.1)
    
    # Update state with generated molecules tracking
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    bt.logging.info(f"✅ Generated {len(unique_molecules)} unique molecules using pool size {current_pool_size} (attempts: {attempts}, total tracked: {len(generated_molecules)})")
    
    return unique_molecules




async def run_adaptive_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Updated genetic algorithm loop:
    1. When epoch changes, collect submissions using prepare_training_data.py
    2. Load CSV, filter rxn:3, sort, take top 30
    3. Generate unique molecules (NOT in HuggingFace) - keep generating until desired count
    4. Score in batches of 10, checking blocks remaining after each batch
    5. Only submit when < 80 blocks remain until next epoch
    6. Skip molecules already in DB UNLESS they're the top-scoring molecule
    7. When epoch changes, restart process
    """
    bt.logging.info("🚀 Starting ADAPTIVE genetic algorithm loop with CSV-based generation...")
    
    csv_path = os.path.join(BASE_DIR, 'BoltzPredictor', 'data', 'mols.csv')
    last_processed_epoch = state.get('last_processed_epoch', -1)
    desired_unique_count = 100  # Desired number of unique molecules to generate
    
    while not state['shutdown_event'].is_set():
        try:
            # Get current epoch
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            last_submission_epoch = state.get('last_submission_epoch', -1)
            
            # Check if epoch changed - if so, collect new submissions and start generation
            if current_epoch != last_processed_epoch:
                bt.logging.info(f"\n{'='*70}")
                bt.logging.info(f"🔄 Epoch changed: {last_processed_epoch} → {current_epoch}")
                bt.logging.info(f"{'='*70}")
                
                # Collect submissions from STARTING_EPOCH
                top_200_df = await collect_and_process_submissions(state, STARTING_EPOCH, csv_path)
                
                # If CSV collection failed, fall back to using molecules from top_pool (randomly generated)
                if top_200_df.empty:
                    bt.logging.warning("No top 200 molecules found from CSV, checking top_pool...")
                    top_pool = state.get('top_pool')
                    if top_pool is not None and not top_pool.empty:
                        bt.logging.info(f"✅ Found {len(top_pool)} molecules in top_pool, using them as seed")
                        # Convert top_pool to the format needed for top_200_df
                        # top_200_df needs: molecule_name, final_score columns
                        top_200_df = pd.DataFrame({
                            'molecule_name': top_pool['name'].tolist(),
                            'final_score': top_pool.get('score', [None] * len(top_pool)).tolist()
                        })
                        # Take top 200 (or all if less than 200)
                        top_200_df = top_200_df.head(200)
                        bt.logging.info(f"✅ Using {len(top_200_df)} molecules from top_pool as seed for this epoch")
                    else:
                        bt.logging.warning("No top 200 molecules found and top_pool is empty, skipping this epoch")
                        last_processed_epoch = current_epoch
                        state['last_processed_epoch'] = current_epoch
                        await asyncio.sleep(10)
                        continue
                
                # Store top_200_df in state for use in generation loop
                state['top_200_df'] = top_200_df
                
                # If molecules don't have scores (e.g., randomly generated), we'll score them along with new ones
                # For now, generate unique molecules from the seed molecules
                bt.logging.info(f"🧬 Generating {desired_unique_count} unique molecules from top 200 (adaptive pool sizing)...")
                unique_molecules = await generate_unique_molecules_from_top200(
                    state, top_200_df, desired_unique_count
                )
                
                # If generation failed, try using the seed molecules directly (for randomly generated case)
                if not unique_molecules:
                    bt.logging.warning("Failed to generate unique molecules from top 200, using seed molecules directly...")
                    # Convert top_200_df to the format needed for scoring
                    seed_molecules = []
                    for _, row in top_200_df.head(100).iterrows():  # Use top 100 as seed
                        molecule_name = row['molecule_name']
                        try:
                            smiles = get_smiles_from_reaction(molecule_name)
                            if smiles:
                                inchikey = generate_inchikey(smiles)
                                if inchikey:
                                    seed_molecules.append({
                                        'name': molecule_name,
                                        'smiles': smiles,
                                        'InChIKey': inchikey,
                                        'score': row.get('final_score')  # May be None for random molecules
                                    })
                        except Exception as e:
                            bt.logging.debug(f"Error processing seed molecule {molecule_name}: {e}")
                            continue
                    
                    if seed_molecules:
                        unique_molecules = seed_molecules[:desired_unique_count]
                        bt.logging.info(f"✅ Using {len(unique_molecules)} seed molecules directly for scoring")
                    else:
                        bt.logging.warning("No valid seed molecules found, skipping this epoch")
                        last_processed_epoch = current_epoch
                        state['last_processed_epoch'] = current_epoch
                        await asyncio.sleep(10)
                        continue
                
                bt.logging.info(f"✅ Generated/selected {len(unique_molecules)} molecules for scoring")
                
                # Calculate blocks until next epoch
                next_epoch_block = (current_epoch + 1) * state['epoch_length']
                
                # ✅ CONTINUOUS GENERATION AND SCORING LOOP
                # Keep generating and scoring batches of 100 until we're within 80 blocks of next epoch
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
                    
                    # If we're already within 80 blocks, submit and exit
                    if blocks_remaining < 80:
                        if best_molecule_so_far:
                            # Check if we already submitted in this epoch
                            if last_submission_epoch == current_epoch:
                                bt.logging.info(f"⏭️  Already submitted in epoch {current_epoch}")
                            else:
                                # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                                molecule_name = best_molecule_so_far['name']
                                smiles = best_molecule_so_far.get('smiles')
                                
                                if not smiles:
                                    bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, skipping submission")
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
                                
                                # Check if molecule is already in DB
                                db_score = get_score_from_db(best_molecule_so_far['name'])
                                if db_score is not None:
                                    bt.logging.info(
                                        f"⚠️  Best molecule {best_molecule_so_far['name']} already in DB "
                                        f"(score: {db_score:.6f}), but submitting as top molecule"
                                    )
                                
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
                            f"Blocks remaining ({blocks_remaining}) >= 80, generating another {desired_unique_count} molecules..."
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
                    
                    # Score this batch of molecules in batches of 10
                    total_batches = (len(unique_molecules) + batch_size - 1) // batch_size
                    bt.logging.info(f"🔬 Round {generation_round}: Scoring {len(unique_molecules)} molecules in {total_batches} batches of {batch_size}...")
                    
                    for batch_idx in range(total_batches):
                        # Check blocks remaining before starting batch
                        current_block_before_batch = await state['subtensor'].get_current_block()
                        blocks_remaining = next_epoch_block - current_block_before_batch
                        
                        # Only submit when < 80 blocks remain
                        if blocks_remaining < 80:
                            if best_molecule_so_far:
                                # Check if we already submitted in this epoch
                                if last_submission_epoch == current_epoch:
                                    bt.logging.info(f"⏭️  Already submitted in epoch {current_epoch}")
                                else:
                                    # ✅ CHECK UNIQUENESS BEFORE SUBMISSION
                                    molecule_name = best_molecule_so_far['name']
                                    smiles = best_molecule_so_far.get('smiles')
                                    
                                    if not smiles:
                                        bt.logging.warning(f"⚠️  Best molecule {molecule_name} has no SMILES, skipping submission")
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
                                    
                                    # Check if molecule is already in DB
                                    db_score = get_score_from_db(best_molecule_so_far['name'])
                                    if db_score is not None:
                                        bt.logging.info(
                                            f"⚠️  Best molecule {best_molecule_so_far['name']} already in DB "
                                            f"(score: {db_score:.6f}), but submitting as top molecule"
                                        )
                                    
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
                        
                        # Check which molecules in this batch are already in DB
                        batch_molecule_names = [m['name'] for m in batch]
                        db_scores = batch_get_scores_from_db(batch_molecule_names)
                        
                        # Separate molecules: those in DB (skip scoring) vs those needing scoring
                        batch_to_score = []
                        batch_from_db = []
                        
                        for mol in batch:
                            mol_name = mol['name']
                            if mol_name in db_scores:
                                # Already in DB - skip scoring but track for potential submission
                                mol['boltz_score'] = db_scores[mol_name]
                                mol['boltz_score_source'] = 'database'
                                batch_from_db.append(mol)
                                bt.logging.debug(
                                    f"   ⏭️  Molecule {mol_name} already in DB "
                                    f"(score: {db_scores[mol_name]:.6f}), skipping scoring"
                                )
                            else:
                                batch_to_score.append(mol)
                        
                        bt.logging.info(
                            f"   📦 Round {generation_round}, Batch {batch_idx + 1}/{total_batches}: "
                            f"Scoring {len(batch_to_score)} new molecules, "
                            f"{len(batch_from_db)} already in DB "
                            f"(blocks remaining: {blocks_remaining})"
                        )
                        
                        # Score only molecules not in DB
                        scored_batch = []
                        if batch_to_score:
                            scored_batch = await score_molecules_with_boltz(state, batch_to_score)
                        
                        # Combine scored molecules with DB molecules
                        all_batch_molecules = batch_from_db + (scored_batch if scored_batch else [])
                        
                        if all_batch_molecules:
                            # Filter molecules with valid scores
                            batch_with_scores = [m for m in all_batch_molecules if m.get('boltz_score') is not None]
                            all_scored_molecules.extend(batch_with_scores)
                            
                            # Update best molecule so far (consider all scored molecules, including from DB)
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
                        
                        # Check epoch boundary after each batch
                        current_block_after_batch = await state['subtensor'].get_current_block()
                        blocks_remaining_after = next_epoch_block - current_block_after_batch
                        
                        bt.logging.info(f"   ⏱️  Blocks remaining after round {generation_round}, batch {batch_idx + 1}: {blocks_remaining_after}")
                        
                        # If we hit the 80 block threshold during batch scoring, exit batch loop
                        if blocks_remaining_after < 80:
                            break
                    
                    # After completing all batches for this round, check if we should continue
                    if submitted:
                        break
                    
                    # Check blocks remaining after all batches in this round
                    current_block_after_round = await state['subtensor'].get_current_block()
                    blocks_remaining_after_round = next_epoch_block - current_block_after_round
                    
                    bt.logging.info(f"   ⏱️  Blocks remaining after round {generation_round}: {blocks_remaining_after_round}")
                    
                    # If blocks remaining < 80, submit and exit
                    if blocks_remaining_after_round < 80:
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
                        # Still have >= 80 blocks remaining, continue to next generation round
                        score_str = f"{best_score_so_far:.6f}" if best_molecule_so_far else "N/A"
                        bt.logging.info(
                            f"⏭️  Blocks remaining ({blocks_remaining_after_round}) >= 80, "
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


# ============================================================================
# ✅ STARTUP: LOAD CSV
# ============================================================================

def _import_boltz_wrapper():
    """
    Import BoltzWrapper following the same pattern as DataGenerator/main.py.
    This is done as a function so it can be called after logging is initialized.
    """
    global BOLTZ_AVAILABLE, BoltzWrapper
    
    try:
        # BASE_DIR is nova/ (parent of neurons/)
        # So boltz-scoring should be at nova/boltz-scoring
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        
        if not os.path.exists(BOLTZ_SCORING_DIR):
            bt.logging.warning(f"⚠️  Boltz-scoring directory not found at {BOLTZ_SCORING_DIR}")
            return False
        
        # Add boltz-scoring to path (same as DataGenerator/main.py does)
        # This allows: from boltz.wrapper import BoltzWrapper
        if BOLTZ_SCORING_DIR not in sys.path:
            sys.path.append(BOLTZ_SCORING_DIR)
        
        # Add boltz-scoring/boltz/src to path BEFORE importing
        # This is critical: the boltz package is at boltz/src/boltz/, so we need
        # boltz/src/ in the path so that "from boltz.data import const" works
        if BOLTZ_SRC_DIR not in sys.path:
            sys.path.insert(0, BOLTZ_SRC_DIR)  # Insert at beginning for priority
        
        # Try to import from boltz-scoring's utils (local copy) - same as DataGenerator
        boltz_utils_path = os.path.join(BOLTZ_SCORING_DIR, 'utils')
        if os.path.exists(boltz_utils_path) and boltz_utils_path not in sys.path:
            sys.path.insert(0, boltz_utils_path)
        
        # Now import BoltzWrapper (same as DataGenerator/main.py)
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


async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Initialize score_results database
    2. Import and initialize BoltzWrapper
    3. Collect submissions (creates/updates CSV)
    4. Load molecules from CSV
    5. Prepare top_pool
    """
    bt.logging.info("🚀 Starting STARTUP phase: Initialize DB, Boltz, Collect Submissions & Load CSV...")
    
    try:
        # Initialize score_results database
        bt.logging.info("💾 Initializing score_results database...")
        init_score_results_db()
        bt.logging.info(f"✅ Score results database initialized at {SCORE_RESULTS_DB}")
        
        # Import BoltzWrapper (following DataGenerator/main.py pattern)
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
        
        # Collect submissions FIRST (this creates/updates the CSV)
        bt.logging.info("📥 Collecting submissions from epoch...")
        top_200_df = await collect_and_process_submissions(state, STARTING_EPOCH, REACTION_TRAIN_CSV)
        
        if top_200_df.empty:
            bt.logging.warning("⚠️  No submissions collected, CSV may be empty or outdated")
            # Fallback: try loading from CSV anyway
            bt.logging.info("📂 Fallback: Loading molecules from CSV...")
            molecules_df = load_molecules_from_csv(
                REACTION_TRAIN_CSV,
                state['current_challenge_targets'],
                STARTING_EPOCH,
                HARDCODED_RXN_ID
            )
            if molecules_df.empty:
                bt.logging.warning("No molecules loaded from CSV, generating random initial molecules...")
                # Generate random molecules as fallback
                # Build subnet_config from config
                subnet_config = {
                    'min_heavy_atoms': getattr(state['config'], 'min_heavy_atoms', 20),
                    'max_heavy_atoms': getattr(state['config'], 'max_heavy_atoms', 45),
                    'min_rotatable_bonds': getattr(state['config'], 'min_rotatable_bonds', 1),
                    'max_rotatable_bonds': getattr(state['config'], 'max_rotatable_bonds', 10)
                }
                molecules_df = generate_random_initial_molecules(
                    HARDCODED_RXN_ID,
                    num_molecules=200,
                    subnet_config=subnet_config
                )
                if molecules_df.empty:
                    bt.logging.error("Failed to generate random molecules, cannot proceed")
                    return
                bt.logging.info(f"✅ Generated {len(molecules_df)} random initial molecules")
            state['top_pool'] = molecules_df.copy()
            state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
        else:
            bt.logging.info(f"✅ Collected {len(top_200_df)} top submissions")
            
            # Convert top_200_df to the format needed for top_pool (name, smiles, InChIKey, score)
            bt.logging.info("🔄 Processing top 200 molecules for top_pool...")
            result_rows = []
            successful_count = 0
            failed_count = 0
            
            for _, row in top_200_df.iterrows():
                molecule_name = row['molecule_name']
                final_score = row.get('final_score', None)
                if pd.isna(final_score):
                    final_score = None
                else:
                    final_score = float(final_score)
                
                try:
                    smiles = get_smiles_from_reaction(molecule_name)
                    if not smiles:
                        bt.logging.debug(f"No SMILES found for {molecule_name}")
                        failed_count += 1
                        continue
                    
                    inchikey = generate_inchikey(smiles)
                    if not inchikey:
                        bt.logging.debug(f"Could not generate InChIKey for {molecule_name}")
                        failed_count += 1
                        continue
                    
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
            
            if result_rows:
                molecules_df = pd.DataFrame(result_rows)
                molecules_df = molecules_df.drop_duplicates(subset=['InChIKey'], keep='first')
                # Sort by score descending (should already be sorted, but ensure it)
                if 'score' in molecules_df.columns:
                    molecules_df = molecules_df.sort_values(by='score', ascending=False, na_position='last')
                
                bt.logging.info(f"✅ Processed {len(molecules_df)} molecules from top 200 submissions (successful: {successful_count}, failed: {failed_count})")
                
                # Update state with top 200 from collected submissions
                state['top_pool'] = molecules_df.copy()
                state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
            else:
                bt.logging.warning("⚠️  Could not process any molecules from top 200, falling back to loading from CSV...")
                molecules_df = load_molecules_from_csv(
                    REACTION_TRAIN_CSV,
                    state['current_challenge_targets'],
                    STARTING_EPOCH,
                    HARDCODED_RXN_ID
                )
                if molecules_df.empty:
                    bt.logging.warning("No molecules loaded from CSV, generating random initial molecules...")
                    # Generate random molecules as fallback
                    # Build subnet_config from config
                    subnet_config = {
                        'min_heavy_atoms': getattr(state['config'], 'min_heavy_atoms', 20),
                        'max_heavy_atoms': getattr(state['config'], 'max_heavy_atoms', 45),
                        'min_rotatable_bonds': getattr(state['config'], 'min_rotatable_bonds', 1),
                        'max_rotatable_bonds': getattr(state['config'], 'max_rotatable_bonds', 10)
                    }
                    molecules_df = generate_random_initial_molecules(
                        HARDCODED_RXN_ID,
                        num_molecules=200,
                        subnet_config=subnet_config
                    )
                    if molecules_df.empty:
                        bt.logging.error("Failed to generate random molecules, cannot proceed")
                        return
                    bt.logging.info(f"✅ Generated {len(molecules_df)} random initial molecules")
                state['top_pool'] = molecules_df.copy()
                state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
        
        bt.logging.info(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Sample molecules: {state['top_pool']['name'].head(3).tolist()}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        bt.logging.error(f"Error in startup phase: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


# ============================================================================
# SUBMISSION LOGIC
# ============================================================================

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
        'last_processed_epoch': -1,  # Track last epoch we processed CSV for
        'startup_complete': False,
        'shutdown_event': asyncio.Event(),
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
        'rxn_id': None,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),  # Track generated molecule names to avoid duplicates
        'generated_inchikeys': set(),  # Track generated InChIKeys to avoid duplicates
        'boltz_wrapper': None,  # Will be initialized in startup_phase
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

        # Launch adaptive GA loop
        try:
            state['ga_task'] = asyncio.create_task(run_adaptive_genetic_loop(state))
            bt.logging.info("✅ Adaptive GA loop started!")
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
