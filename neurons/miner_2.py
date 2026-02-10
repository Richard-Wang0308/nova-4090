#!/usr/bin/env python3
"""
BoltzPredictor Miner for Bittensor Subnet 68.

Continuously generates and scores molecules using BoltzPredictor model,
submitting the best candidates to the network.

Target-only version with combinatorial database generation.

✅ FIXED: Heavy Atom Count Filtering (18-24)
- Only score molecules with 18 <= heavy_atom_count <= 24
- Skip others without scoring
"""

import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from types import SimpleNamespace

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from rdkit import Chem

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Database path for combinatorial DB
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

# Hardcoded reaction ID
HARDCODED_RXN_ID = 2  # Change this to the desired reaction ID

# Starting epoch for loading existing molecules from CSV
STARTING_EPOCH = 20516  # Change this to the starting epoch for the target protein

# Path to reaction2 training data CSV
REACTION2_TRAIN_CSV = os.path.join(BASE_DIR, 'BoltzPredictor', 'data', 'train.csv')

# BoltzPredictor checkpoint path
BOLTZ_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'BoltzPredictor', 'checkpoints', 'single_target', 'final_model.pt')

# ✅ Heavy atom count constraints
MIN_HEAVY_ATOMS = 18
MAX_HEAVY_ATOMS = 24

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,
)
from utils.molecules import (
    molecule_unique_for_protein_hf,
)
from molecules_base import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
    generate_inchikey,
)
from combinatorial_db.reactions import get_smiles_from_reaction

# BoltzPredictor imports
sys.path.insert(0, os.path.join(BASE_DIR, 'BoltzPredictor'))
print(f"BoltzPredictor path: {os.path.join(BASE_DIR, 'BoltzPredictor')}")

try:
    from boltzpredictor.models import create_model
    from boltzpredictor.data import MoleculePreprocessor
    BOLTZ_AVAILABLE = True
except ImportError as e:
    bt.logging.warning(f"BoltzPredictor not available: {e}")
    BOLTZ_AVAILABLE = False
    create_model = None
    MoleculePreprocessor = None

from btdr import QuicknetBittensorDrandTimelock

# ============================================================================
# PyTorch 2.6+ Compatibility
# ============================================================================

def safe_torch_load(path, map_location='cpu'):
    """
    Safely load PyTorch checkpoint with numpy scalar support (PyTorch 2.6+).
    
    Handles:
    - numpy.core.multiarray.scalar globals
    - weights_only=False for backward compatibility
    - Proper error handling and logging
    
    Args:
        path: Path to checkpoint file
        map_location: Device to load to (default: 'cpu')
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
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

def denormalize_predictions(predictions, normalizer_dict):
    """
    Denormalize predictions using saved normalizer statistics.
    
    Args:
        predictions: Single value, list, or numpy array of normalized predictions
        normalizer_dict: Dictionary with 'method', 'mean', 'std' keys
        
    Returns:
        Denormalized predictions in original scale
    """
    if normalizer_dict is None:
        bt.logging.warning("No normalizer provided. Returning raw predictions (normalized space).")
        return predictions
    
    method = normalizer_dict.get('method', 'standardization')
    mean = normalizer_dict.get('mean')
    std = normalizer_dict.get('std')
    
    if mean is None or std is None:
        bt.logging.warning("Normalizer missing mean/std. Returning raw predictions.")
        return predictions
    
    # Convert to numpy array for processing
    predictions_array = np.array(predictions)
    is_scalar = predictions_array.ndim == 0
    
    if method == 'standardization':
        # Z-score denormalization: x = z * std + mean
        denormalized = predictions_array * std + mean
        
    elif method == 'minmax':
        # Min-max denormalization: x = z * (max - min) + min
        min_val = normalizer_dict.get('min')
        max_val = normalizer_dict.get('max')
        if min_val is None or max_val is None:
            bt.logging.warning("Min-max normalizer missing min/max values.")
            return predictions
        denormalized = predictions_array * (max_val - min_val) + min_val
        
    else:
        bt.logging.warning(f"Unknown normalization method: {method}")
        return predictions
    
    # Return as scalar if input was scalar
    if is_scalar:
        return denormalized.item()
    
    return denormalized.tolist() if isinstance(denormalized, np.ndarray) else denormalized

# ============================================================================
# Heavy Atom Count Filtering
# ============================================================================

def get_heavy_atom_count_from_smiles(smiles: str) -> Optional[int]:
    """
    Get heavy atom count from SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Heavy atom count or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol.GetNumHeavyAtoms()
    except Exception as e:
        bt.logging.debug(f"Error getting heavy atom count: {e}")
        return None

def filter_by_heavy_atom_count(df: pd.DataFrame, min_atoms: int = MIN_HEAVY_ATOMS, max_atoms: int = MAX_HEAVY_ATOMS) -> pd.DataFrame:
    """
    Filter dataframe to keep only molecules with heavy atom count in range.
    
    Args:
        df: DataFrame with 'smiles' column
        min_atoms: Minimum heavy atom count (default: 18)
        max_atoms: Maximum heavy atom count (default: 24)
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    bt.logging.info(f"Filtering molecules by heavy atom count: {min_atoms} <= count <= {max_atoms}")
    
    # Calculate heavy atom counts
    heavy_atom_counts = []
    for smiles in df['smiles']:
        count = get_heavy_atom_count_from_smiles(smiles)
        heavy_atom_counts.append(count)
    
    df = df.copy()
    df['heavy_atom_count'] = heavy_atom_counts
    
    # Filter by atom count
    df_filtered = df[(df['heavy_atom_count'] >= min_atoms) & (df['heavy_atom_count'] <= max_atoms)]
    
    bt.logging.info(
        f"Heavy atom count filter: {len(df)} → {len(df_filtered)} molecules "
        f"({len(df) - len(df_filtered)} filtered out)"
    )
    
    return df_filtered.drop(columns=['heavy_atom_count'])

# ============================================================================
# 1. CONFIG & ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    # Bittensor standard argument additions.
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse combined config
    config = bt.config(parser)

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')  # e.g., "nova"
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')  # e.g., "main"
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')  # e.g., "metanova-labs"
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')  # e.g., "data/results" or ""

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


# ============================================================================
# 2. LOGGING SETUP
# ============================================================================

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ============================================================================
# 3. BITTENSOR & NETWORK SETUP
# ============================================================================

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")
            
            # Sync metagraph
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            bt.logging.info(f"Subtensor: {subtensor}")
            bt.logging.info(f"Metagraph synced: {metagraph}")

            # Get miner UID
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            # Query epoch length
            epoch_length = 361
            bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise


# ============================================================================
# 4. DATA LOADING FUNCTIONS
# ============================================================================

def load_existing_molecules_from_csv(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int
) -> pd.DataFrame:
    """
    Load existing molecules from CSV file that match the target proteins and epoch.
    ✅ Filter by heavy atom count (18-24)
    
    Args:
        csv_path: Path to the reaction2_train.csv file
        target_proteins: List of target protein codes to match
        starting_epoch: Starting epoch number (load molecules with epoch >= starting_epoch)
        rxn_id: Reaction ID to filter molecules (molecule names should start with f"rxn:{rxn_id}:")
        
    Returns:
        DataFrame with columns: name, smiles, InChIKey, score, target_affinity, antitarget_affinity
    """
    if not os.path.exists(csv_path):
        bt.logging.warning(f"CSV file not found at {csv_path}, skipping loading existing molecules")
        return pd.DataFrame(
            columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
        )
    
    try:
        bt.logging.info(
            f"Loading existing molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        # Filter by target protein
        if 'target_protein' in df.columns:
            df = df[df['target_protein'].isin(target_proteins)]
        else:
            bt.logging.warning("CSV file does not have 'target_protein' column")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        # Filter by epoch
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            bt.logging.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        # Filter by reaction ID
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            bt.logging.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        if df.empty:
            bt.logging.info("No matching molecules found in CSV")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        # Extract molecule names and scores
        result_rows = []
        successful_count = 0
        failed_count = 0
        
        for _, row in df.iterrows():
            molecule_name = row['molecule_name']
            
            try:
                # ✅ FIXED: Only pass molecule_name (NO DB_PATH)
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    bt.logging.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Generate InChIKey
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    bt.logging.debug(f"Could not generate InChIKey for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Get final score
                final_score = row.get('final_score', 0.0)
                if pd.isna(final_score):
                    final_score = 0.0
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': float(final_score),
                    'target_affinity': float('nan'),
                    'antitarget_affinity': float('nan'),
                })
                successful_count += 1
                
            except Exception as e:
                bt.logging.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            # Remove duplicates by InChIKey, keep highest score
            result_df = result_df.sort_values('score', ascending=False)
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            # ✅ Filter by heavy atom count
            result_df = filter_by_heavy_atom_count(result_df, MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS)
            
            bt.logging.info(
                f"Loaded {len(result_df)} existing molecules from CSV "
                f"(successful: {successful_count}, failed: {failed_count})"
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
        return pd.DataFrame(
            columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
        )


# ============================================================================
# 5. INFERENCE AND SCORING LOGIC
# ============================================================================

async def check_molecule_unique(state: Dict[str, Any], molecule_name: str, smiles: str) -> bool:
    """
    Check if a molecule is unique for the target protein using HF check.
    
    Args:
        state: Shared state dictionary
        molecule_name: Molecule name (e.g., "rxn:2:...")
        smiles: SMILES string of the molecule
        
    Returns:
        True if molecule is unique (not seen), False otherwise
    """
    if not state.get('current_challenge_targets'):
        bt.logging.warning("No target proteins available for uniqueness check")
        return False
    
    # Use primary target protein
    primary_target = state['current_challenge_targets'][0]
    
    try:
        # Check HF uniqueness (uses SMILES)
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            bt.logging.info(f"Molecule {molecule_name} already seen in HF dataset for {primary_target}")
            return False
        
        # Check passed - molecule is unique
        bt.logging.info(f"Molecule {molecule_name} is unique for {primary_target}")
        return True
        
    except Exception as e:
        bt.logging.error(f"Error checking molecule uniqueness: {e}")
        # On error, assume not unique to be safe
        return False


async def recalculate_top_pool_scores(state: Dict[str, Any], top_pool: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate scores for molecules in top_pool using the loaded model.
    
    Args:
        state: Shared state dictionary with model, preprocessor, etc.
        top_pool: DataFrame with molecules to rescore
        
    Returns:
        DataFrame with recalculated scores, sorted by new scores
    """
    if top_pool.empty:
        return top_pool
    
    model = state['boltz_model']
    preprocessor = state['boltz_preprocessor']
    device = state['boltz_device']
    model_type = state.get('boltz_model_type', 'single_target')
    normalizer_dict = state.get('normalizer_dict')
    
    bt.logging.info(f"Recalculating scores for {len(top_pool)} molecules from CSV...")
    
    smiles_list = top_pool['smiles'].tolist()
    batch_size = state.get('batch_size', 128)
    all_final_scores = []
    
    # Process in batches
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        actual_batch_size = len(batch_smiles)
        
        try:
            # Process molecules
            mol_data_batch = preprocessor.batch_process(batch_smiles)
            mol_data_batch = {k: v.to(device) for k, v in mol_data_batch.items()}
            
            # Run inference based on model type
            with torch.no_grad():
                if model_type == 'single_target':
                    # ✅ SINGLE-TARGET: No protein sequences needed
                    outputs = model(mol_data=mol_data_batch)
                else:
                    # Multi-target: requires target sequences
                    if not state.get('current_challenge_targets'):
                        bt.logging.warning("No target proteins available for multi-target model")
                        all_final_scores.extend([float('nan')] * len(batch_smiles))
                        continue
                    
                    # Get primary target sequence
                    primary_target = state['current_challenge_targets'][0]
                    seq = get_sequence_from_protein_code(primary_target)
                    if not seq:
                        bt.logging.warning(f"Could not get sequence for {primary_target}")
                        all_final_scores.extend([float('nan')] * len(batch_smiles))
                        continue
                    
                    target_seq_list = [seq] * actual_batch_size
                    outputs = model(
                        mol_data=mol_data_batch,
                        target_seqs=target_seq_list,
                    )
            
            # Extract scores
            batch_final_scores = outputs['final_score']
            if batch_final_scores.dim() > 1:
                batch_final_scores = batch_final_scores.squeeze(-1)
            
            batch_final_scores_np = batch_final_scores.cpu().numpy()
            batch_final_scores_list = batch_final_scores_np.tolist()
            
            # Denormalize if normalizer available
            if normalizer_dict:
                batch_final_scores_denorm = denormalize_predictions(
                    batch_final_scores_list,
                    normalizer_dict
                )
                if isinstance(batch_final_scores_denorm, list):
                    all_final_scores.extend(batch_final_scores_denorm)
                else:
                    all_final_scores.append(batch_final_scores_denorm)
            else:
                all_final_scores.extend(batch_final_scores_list)
                
        except Exception as e:
            bt.logging.error(f"Error recalculating scores for batch {i//batch_size + 1}: {e}")
            all_final_scores.extend([float('nan')] * len(batch_smiles))
    
    # Update scores in top_pool
    top_pool = top_pool.copy()
    top_pool['score'] = all_final_scores
    
    # Sort by new scores
    top_pool = top_pool.sort_values(by="score", ascending=False)
    top_pool = top_pool.reset_index(drop=True)
    
    bt.logging.info(
        f"✅ Recalculated scores. Top score: {top_pool['score'].iloc[0]:.6f}, "
        f"Bottom score: {top_pool['score'].iloc[-1]:.6f}"
    )
    
    return top_pool


async def find_unique_candidate(state: Dict[str, Any], top_pool: pd.DataFrame, max_candidates: int = 10) -> Optional[str]:
    """
    Find a unique candidate molecule from top_pool for submission.
    
    Args:
        state: Shared state dictionary
        top_pool: DataFrame with top molecules sorted by score
        max_candidates: Maximum number of candidates to check
        
    Returns:
        Molecule name if unique candidate found, None otherwise
    """
    if top_pool.empty:
        bt.logging.warning("Top pool is empty, no candidates to check")
        return None
    
    if not state.get('current_challenge_targets'):
        bt.logging.warning("No target proteins available")
        return None
    
    primary_target = state['current_challenge_targets'][0]
    bt.logging.info(f"Checking uniqueness for top candidates against target {primary_target}")
    
    # Check up to max_candidates from top_pool
    candidates_to_check = min(max_candidates, len(top_pool))
    
    for idx in range(candidates_to_check):
        candidate_row = top_pool.iloc[idx]
        molecule_name = candidate_row['name']
        smiles = candidate_row['smiles']
        score = candidate_row['score']
        
        bt.logging.info(
            f"Checking candidate #{idx + 1}: {molecule_name} "
            f"(score: {score:.6f})"
        )
        
        # Check if unique using HF check
        is_unique = await check_molecule_unique(state, molecule_name, smiles)
        
        if is_unique:
            bt.logging.info(
                f"✅ Found unique candidate #{idx + 1}: {molecule_name} "
                f"(score: {score:.6f})"
            )
            return molecule_name
        else:
            bt.logging.info(
                f"❌ Candidate #{idx + 1} ({molecule_name}) is not unique, trying next..."
            )
    
    bt.logging.warning(f"No unique candidates found in top {candidates_to_check} molecules")
    return None

async def run_model_loop(state: Dict[str, Any]) -> None:
    """
    Continuously runs the BoltzPredictor model on batches of molecules generated 
    from combinatorial database. Updates the best candidate whenever a higher score 
    is found, but only submits when close to epoch end.

    ✅ FIXED: Heavy Atom Count Filtering (18-24)
    - Only score molecules with 18 <= heavy_atom_count <= 24
    - Skip others without scoring

    Args:
        state (dict): A shared state dict containing references to:
            'chunk_size', 'current_challenge_targets',
            'current_challenge_antitargets', 'best_score',
            'candidate_product', 'submission_interval', 'last_submission_time',
            'last_submitted_product', 'shutdown_event', 'rxn_id', 'top_pool', etc.
    """
    bt.logging.info("Starting BoltzPredictor model inference loop with combinatorial DB generation.")
    bt.logging.info(f"✅ Heavy Atom Count Filter: {MIN_HEAVY_ATOMS} <= count <= {MAX_HEAVY_ATOMS}")
    
    # Initialize generation state
    iteration = 0
    base_n_samples = state.get('chunk_size', 128)
    top_pool = state.get('top_pool', pd.DataFrame(
        columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
    ))
    seen_inchikeys = state.get('seen_inchikeys', set())
    synthon_lib = state.get('synthon_lib', None)
    use_synthon_search = state.get('use_synthon_search', False)
    rxn_id = HARDCODED_RXN_ID  # Use hardcoded reaction ID
    
    # Initialize BoltzPredictor model if not already done
    if 'boltz_model' not in state or state['boltz_model'] is None:
        if not BOLTZ_AVAILABLE:
            bt.logging.error("BoltzPredictor not available. Cannot proceed.")
            return
        
        if not os.path.exists(BOLTZ_CHECKPOINT_PATH):
            bt.logging.error(f"BoltzPredictor checkpoint not found at {BOLTZ_CHECKPOINT_PATH}")
            return
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bt.logging.info(f"Loading BoltzPredictor on {device}...")
            
            # ✅ FIXED: Use safe_torch_load for PyTorch 2.6+ compatibility
            checkpoint = safe_torch_load(BOLTZ_CHECKPOINT_PATH, map_location=device)
            config = checkpoint.get('config', {})
            
            # Detect model type from checkpoint (same logic as inference.py)
            model_type = config.get('model_type', 'single_target')
            if 'target_embedding_dim' in config and 'protein_model' not in config:
                model_type = 'single_target'
            
            bt.logging.info(f"Detected model type: {model_type.upper()}")
            
            # ✅ FIXED: Use create_model to support both single-target and multi-target
            model = create_model(
                model_type=model_type,
                mol_hidden_dim=int(config.get('mol_hidden_dim', 256)),
                protein_embedding_dim=int(config.get('protein_embedding_dim', 1280)),
                interaction_dim=int(config.get('interaction_dim', 512)),
                num_layers=int(config.get('num_interaction_layers', 3)),
                target_embedding_dim=int(config.get('target_embedding_dim', 512)),
                dropout=float(config.get('dropout', 0.1)),
                protein_model_name=config.get('protein_model', 'facebook/esm2_t33_650M_UR50D'),
                protein_cache_dir=config.get('cache_dir', 'cache/protein_embeddings'),
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            state['boltz_model'] = model
            state['boltz_model_type'] = model_type
            state['boltz_preprocessor'] = MoleculePreprocessor()
            state['boltz_device'] = device
            
            total_params = sum(p.numel() for p in model.parameters())
            bt.logging.info(f"✅ BoltzPredictor ({model_type}) loaded - Total parameters: {total_params:,}")
            
        except Exception as e:
            bt.logging.error(f"Failed to load BoltzPredictor: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return
    
    # ✅ LOCATION 2: Load normalizer if not already loaded
    if 'normalizer_dict' not in state or state['normalizer_dict'] is None:
        try:
            checkpoint_dir = os.path.dirname(BOLTZ_CHECKPOINT_PATH)
            normalizer_path = os.path.join(checkpoint_dir, 'normalizer.json')
            
            if os.path.exists(normalizer_path):
                with open(normalizer_path, 'r') as f:
                    normalizer_dict = json.load(f)
                state['normalizer_dict'] = normalizer_dict
                bt.logging.info(
                    f"✅ Loaded normalizer from {normalizer_path}"
                )
                bt.logging.info(
                    f"   Method: {normalizer_dict.get('method')}"
                )
                bt.logging.info(
                    f"   Mean: {normalizer_dict.get('mean'):.6f}"
                )
                bt.logging.info(
                    f"   Std: {normalizer_dict.get('std'):.6f}"
                )
            else:
                bt.logging.warning(f"Normalizer not found at {normalizer_path}")
                bt.logging.warning("⚠️ Predictions will be in normalized (Z-score) space")
                state['normalizer_dict'] = None
        except Exception as e:
            bt.logging.error(f"Failed to load normalizer: {e}")
            state['normalizer_dict'] = None
    
    model = state['boltz_model']
    preprocessor = state['boltz_preprocessor']
    device = state['boltz_device']
    
    # ✅ Recalculate scores for top_pool if CSV data was loaded at startup
    if state.get('csv_data_loaded', False) and not top_pool.empty:
        bt.logging.info("Recalculating scores for molecules loaded from CSV...")
        top_pool = await recalculate_top_pool_scores(state, top_pool)
        state['top_pool'] = top_pool
        bt.logging.info(f"Updated top_pool with recalculated scores (top 100 molecules)")
    
    mutation_prob = 0.15
    elite_frac = 0.85
    
    # Convert config to dict format for molecule generation functions
    config_dict = {
        'min_heavy_atoms': state['config'].min_heavy_atoms,
        'min_rotatable_bonds': state['config'].min_rotatable_bonds,
        'max_rotatable_bonds': state['config'].max_rotatable_bonds,
    }

    while not state['shutdown_event'].is_set():
        try:
            iteration += 1
            
            # Build component weights if we have a top pool
            component_weights = None
            if not top_pool.empty:
                component_weights = build_component_weights(top_pool, rxn_id)
            
            # Select diverse elites
            elite_df = select_diverse_elites(
                top_pool, min(200, len(top_pool))
            ) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            # Generate molecules
            if iteration == 1:
                # First iteration: check if CSV data was already loaded at startup
                if state.get('csv_data_loaded', False):
                    bt.logging.info("CSV data already loaded at startup, using existing top_pool")
                else:
                    # Load existing molecules from CSV (only if not already loaded)
                    bt.logging.info("First iteration: Loading existing molecules from CSV...")
                    existing_df = load_existing_molecules_from_csv(
                        REACTION2_TRAIN_CSV,
                        state['current_challenge_targets'],
                        STARTING_EPOCH,
                        rxn_id
                    )
                    
                    if not existing_df.empty:
                        # Add to top_pool
                        top_pool = pd.concat([top_pool, existing_df], ignore_index=True)
                        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                        top_pool = top_pool.sort_values(by="score", ascending=False)
                        
                        # Keep only top 100
                        top_pool = top_pool.head(100)
                        
                        # Add to seen_inchikeys
                        seen_inchikeys.update(top_pool["InChIKey"].tolist())
                        bt.logging.info(
                            f"Loaded {len(existing_df)} existing molecules, kept top 100 in top_pool"
                        )
                        
                        # Update state
                        state['top_pool'] = top_pool
                        state['seen_inchikeys'] = seen_inchikeys
                        state['csv_data_loaded'] = True
                
                # Build component weights from loaded molecules
                component_weights = None
                if not top_pool.empty:
                    component_weights = build_component_weights(top_pool, rxn_id)
                
                # Select diverse elites
                elite_df = select_diverse_elites(
                    top_pool, min(200, len(top_pool))
                ) if not top_pool.empty else pd.DataFrame()
                elite_names = elite_df["name"].tolist() if not elite_df.empty else None
                
                # Generate new molecules based on loaded ones
                n_samples = base_n_samples
                df = generate_valid_random_molecules_batch(
                    rxn_id, n_samples=n_samples, db_path=DB_PATH,
                    subnet_config=config_dict,
                    batch_size=400,
                    elite_names=elite_names, elite_frac=elite_frac,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys, component_weights=component_weights,
                )
            elif use_synthon_search and iteration > 1 and not top_pool.empty:
                # Use synthon library for focused generation
                if synthon_lib is None:
                    try:
                        bt.logging.info("Building synthon library...")
                        synthon_lib = SynthonLibrary(DB_PATH, rxn_id)
                        state['synthon_lib'] = synthon_lib
                        use_synthon_search = True
                        state['use_synthon_search'] = True
                    except Exception as e:
                        bt.logging.warning(f"Could not build synthon library: {e}")
                        use_synthon_search = False
                        state['use_synthon_search'] = False
                
                if use_synthon_search:
                    # Generate from synthon library
                    synthon_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.head(3), int(base_n_samples * 0.7),
                        min_similarity=0.85, n_per_base=55
                    )
                    synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")
                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config_dict)
                    
                    # Fill remaining with traditional generation
                    n_traditional = base_n_samples - len(synthon_df) if not synthon_df.empty else base_n_samples
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id, n_samples=n_traditional, db_path=DB_PATH,
                            subnet_config=config_dict,
                            batch_size=400,
                            elite_names=elite_names, elite_frac=elite_frac,
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                    
                    df = pd.concat([synthon_df, traditional_df], ignore_index=True) if not synthon_df.empty else traditional_df
                    df = df.drop_duplicates(subset=["name"], keep="first")
                else:
                    # Fallback to traditional generation
                    df = generate_valid_random_molecules_batch(
                        rxn_id, n_samples=base_n_samples, db_path=DB_PATH,
                        subnet_config=config_dict,
                        batch_size=400,
                        elite_names=elite_names, elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )
            else:
                # Traditional generation
                df = generate_valid_random_molecules_batch(
                    rxn_id, n_samples=base_n_samples, db_path=DB_PATH,
                    subnet_config=config_dict,
                    batch_size=400,
                    elite_names=elite_names, elite_frac=elite_frac,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=component_weights,
                )
            
            if df.empty:
                await asyncio.sleep(2)
                continue
            
            # Filter out already seen molecules
            df = df[~df["InChIKey"].isin(seen_inchikeys)]
            if df.empty:
                await asyncio.sleep(2)
                continue
            
            # Get model type
            model_type = state.get('boltz_model_type', 'single_target')
            
            # For single-target model, no protein sequences needed
            # For multi-target model, we would need protein sequences
            if model_type == 'multi_target':
                # Get protein sequences for current challenge
                if not state['current_challenge_targets']:
                    bt.logging.warning("No target proteins available. Skipping scoring.")
                    await asyncio.sleep(2)
                    continue
                
                # Get sequences for all targets
                target_sequences = []
                for target_protein in state['current_challenge_targets']:
                    try:
                        seq = get_sequence_from_protein_code(target_protein)
                        if seq:
                            target_sequences.append(seq)
                    except Exception as e:
                        bt.logging.error(f"Error getting sequence for target {target_protein}: {e}")
                
                if not target_sequences:
                    bt.logging.warning("Could not get protein sequences. Skipping scoring.")
                    await asyncio.sleep(2)
                    continue
                
                primary_target_seq = target_sequences[0]
            else:
                # Single-target model: no protein sequences needed
                primary_target_seq = None
            
            # ✅ LOCATION 3: Batch process molecules with BoltzPredictor
            # ✅ FILTER BY HEAVY ATOM COUNT BEFORE SCORING
            smiles_list = df['smiles'].tolist()
            batch_size = state.get('batch_size', 128)

            all_final_scores = []
            normalizer_dict = state.get('normalizer_dict')
            
            # ✅ Filter by heavy atom count BEFORE scoring
            filtered_smiles = []
            filtered_indices = []
            skipped_count = 0
            
            for idx, smiles in enumerate(smiles_list):
                atom_count = get_heavy_atom_count_from_smiles(smiles)
                
                if atom_count is None:
                    # Invalid SMILES
                    all_final_scores.append(float('nan'))
                    skipped_count += 1
                    continue
                
                if MIN_HEAVY_ATOMS <= atom_count <= MAX_HEAVY_ATOMS:
                    # Valid atom count - will score
                    filtered_smiles.append(smiles)
                    filtered_indices.append(idx)
                else:
                    # Outside range - skip without scoring
                    all_final_scores.append(float('nan'))
                    skipped_count += 1
            
            if skipped_count > 0:
                bt.logging.info(
                    f"Skipped {skipped_count} molecules outside heavy atom count range "
                    f"({MIN_HEAVY_ATOMS}-{MAX_HEAVY_ATOMS})"
                )
            
            if not filtered_smiles:
                # All molecules were filtered out
                bt.logging.warning("All molecules filtered out by heavy atom count, retrying...")
                await asyncio.sleep(2)
                continue
            
            bt.logging.info(f"Scoring {len(filtered_smiles)} molecules (within atom count range)")

            # Process in batches (only filtered molecules)
            scored_values = []
            for i in range(0, len(filtered_smiles), batch_size):
                batch_smiles = filtered_smiles[i:i + batch_size]
                actual_batch_size = len(batch_smiles)
                
                try:
                    # Process molecules
                    mol_data_batch = preprocessor.batch_process(batch_smiles)
                    mol_data_batch = {k: v.to(device) for k, v in mol_data_batch.items()}
                    
                    # Run inference based on model type
                    with torch.no_grad():
                        if model_type == 'single_target':
                            # ✅ SINGLE-TARGET: No protein sequences needed
                            outputs = model(mol_data=mol_data_batch)
                        else:
                            # Multi-target: requires target sequences
                            target_seq_list = [primary_target_seq] * actual_batch_size
                            outputs = model(
                                mol_data=mol_data_batch,
                                target_seqs=target_seq_list,
                            )
                    
                    # ✅ FIXED: Extract scores and ensure correct shape
                    batch_final_scores = outputs['final_score']  # Shape: [batch_size]
                    
                    # Convert to numpy
                    if batch_final_scores.dim() > 1:
                        batch_final_scores = batch_final_scores.squeeze(-1)
                    
                    batch_final_scores_np = batch_final_scores.cpu().numpy()
                    batch_final_scores_list = batch_final_scores_np.tolist()
                    
                    # ✅ DENORMALIZE if normalizer available
                    if normalizer_dict:
                        batch_final_scores_denorm = denormalize_predictions(
                            batch_final_scores_list,
                            normalizer_dict
                        )
                        
                        # Handle both list and scalar returns
                        if isinstance(batch_final_scores_denorm, list):
                            scored_values.extend(batch_final_scores_denorm)
                            bt.logging.debug(
                                f"Batch {i//batch_size + 1}: "
                                f"Normalized: {batch_final_scores_list[:2]}... "
                                f"→ Denormalized: {batch_final_scores_denorm[:2]}..."
                            )
                        else:
                            scored_values.append(batch_final_scores_denorm)
                    else:
                        bt.logging.warning("⚠️ No normalizer available. Using normalized scores (0.7-0.8 range).")
                        scored_values.extend(batch_final_scores_list)
                    
                except Exception as e:
                    bt.logging.error(f"Error in batch inference: {e}")
                    import traceback
                    bt.logging.error(traceback.format_exc())
                    # Add NaN for failed batch
                    scored_values.extend([float('nan')] * len(batch_smiles))

            # ✅ Reconstruct full results array with NaN for skipped molecules
            result_scores = [float('nan')] * len(smiles_list)
            for idx, score in zip(filtered_indices, scored_values):
                result_scores[idx] = score
            
            # Assign scores to dataframe
            df['score'] = result_scores

            # Sort by score
            df.sort_values(by=['score'], ascending=[False], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Update top pool
            seen_inchikeys.update(df["InChIKey"].tolist())
            total_data = df[["name", "smiles", "InChIKey", "score"]].copy()
            total_data['target_affinity'] = float('nan')
            total_data['antitarget_affinity'] = float('nan')
            
            if not total_data.empty:
                top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                top_pool = top_pool.sort_values(by="score", ascending=False)
                top_pool = top_pool.head(100)  # Keep top 100 molecules
            
            # Update state
            state['top_pool'] = top_pool
            state['seen_inchikeys'] = seen_inchikeys

            # Update best score from top molecule
            top_molecule = top_pool.head(1)
            if not top_molecule.empty:
                final_score = top_molecule['score'].iloc[0]
                
                # Skip NaN scores
                if not np.isnan(final_score):
                    if final_score > state['best_score']:
                        state['best_score'] = final_score
                        bt.logging.info(
                            f"New best score: {state['best_score']:.6f}"
                        )

                    # Only submit if we're close to epoch end and haven't submitted yet this epoch
                    try:
                        current_block = await state['subtensor'].get_current_block()
                        current_epoch = current_block // state['epoch_length']
                        next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                        blocks_until_epoch = next_epoch_block - current_block
                        
                        bt.logging.debug(
                            f"Current block: {current_block}, Epoch: {current_epoch}, "
                            f"Next epoch block: {next_epoch_block}, Blocks until epoch: {blocks_until_epoch}"
                        )
                        
                        # Check if we've already submitted in this epoch
                        last_submission_epoch = state.get('last_submission_epoch', -1)
                        can_submit = (last_submission_epoch < current_epoch)
                        
                        if blocks_until_epoch <= 20 and can_submit:
                            bt.logging.info(
                                f"Close to epoch end ({blocks_until_epoch} blocks remaining), "
                                f"searching for unique candidate to submit..."
                            )
                            
                            # Find a unique candidate from top_pool
                            unique_candidate = await find_unique_candidate(state, top_pool, max_candidates=10)
                            
                            if unique_candidate:
                                # Only submit if it's different from last submission
                                if unique_candidate != state.get('last_submitted_product'):
                                    state['candidate_product'] = unique_candidate
                                    bt.logging.info(f"Attempting to submit unique candidate: {unique_candidate}")
                                    try:
                                        await submit_response(state)
                                        # Mark that we've submitted in this epoch
                                        state['last_submission_epoch'] = current_epoch
                                        bt.logging.info(f"Submission successful. Will not submit again until next epoch.")
                                    except Exception as e:
                                        bt.logging.error(f"Error submitting response: {e}")
                                else:
                                    bt.logging.info("Skipping submission - same product as last submission")
                            else:
                                bt.logging.warning(
                                    f"No unique candidates found in top pool. "
                                    f"Skipping submission for this epoch."
                                )
                        elif not can_submit:
                            bt.logging.debug(f"Already submitted in epoch {current_epoch}. Waiting for next epoch.")
                    except Exception as e:
                        bt.logging.error(f"Error checking epoch end: {e}")

            await asyncio.sleep(2)

        except Exception as e:
            bt.logging.error(f"Error in BoltzPredictor model loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(2)

# ============================================================================
# 6. SUBMISSION LOGIC
# ============================================================================

async def submit_response(state: Dict[str, Any]) -> None:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")
    
    try:
        # 1) Encrypt the response
        current_block = await state['subtensor'].get_current_block()
        encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
        bt.logging.info(f"Encrypted response generated successfully")

        # 2) Create temp file, write content
        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()

            # Read, base64-encode
            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            # Generate short hash-based filename
            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"Prepared commit content: {commit_content}")

            # 3) Attempt chain commitment
            bt.logging.info(f"Attempting chain commitment...")
            try: 
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=state['wallet'],
                    netuid=state['config'].netuid,
                    data=commit_content
                )
                bt.logging.info(f"Chain commitment status: {commitment_status}")
            except MetadataError:
                bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
                return

            # 4) If chain commitment success, upload to GitHub
            if commitment_status:
                try:
                    bt.logging.info(f"Commitment set successfully for {commit_content}")
                    bt.logging.info("Attempting GitHub upload...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(f"File uploaded successfully to {commit_content}")
                        state['last_submitted_product'] = candidate_product
                        state['last_submission_time'] = datetime.datetime.now()
                        # Mark submission epoch (will be set by caller, but also set here for safety)
                        current_epoch = current_block // state['epoch_length']
                        state['last_submission_epoch'] = current_epoch
                    else:
                        bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
                except Exception as e:
                    bt.logging.error(f"Failed to upload file for {commit_content}: {e}")
    
    except Exception as e:
        bt.logging.error(f"Error in submit_response: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


# ============================================================================
# 7. MAIN MINING LOOP
# ============================================================================

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        'chunk_size': 512,
        'submission_interval': 1200,

        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Models - BoltzPredictor
        'boltz_model': None,
        'boltz_preprocessor': None,
        'boltz_device': None,
        'normalizer_dict': None,  # ✅ LOCATION 4: Add this line
        'batch_size': 128,  # Batch size for BoltzPredictor inference
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_product': None,
        'best_score': float('-inf'),
        'last_submitted_product': None,
        'last_submission_time': None,
        'last_submission_epoch': -1,  # Track which epoch we last submitted in
        'csv_data_loaded': False,  # Track if CSV data was loaded
        'shutdown_event': asyncio.Event(),

        # Challenges
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
        
        # Combinatorial DB generation state
        'rxn_id': None,
        'top_pool': pd.DataFrame(
            columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
        ),
        'seen_inchikeys': set(),
        'synthon_lib': None,
        'use_synthon_search': False,
    }

    bt.logging.info("Entering main miner loop...")

    # 3) Load CSV data once at startup (before inference loop)
    bt.logging.info("Loading existing molecules from CSV at startup...")
    state['rxn_id'] = HARDCODED_RXN_ID
    
    # Get initial challenge targets (we'll use these throughout, not changing per epoch)
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
        
        # Load CSV data once at startup
        existing_df = load_existing_molecules_from_csv(
            REACTION2_TRAIN_CSV,
            state['current_challenge_targets'],
            STARTING_EPOCH,
            state['rxn_id']
        )
        
        if not existing_df.empty:
            # Fill top_pool with top 100 from CSV
            existing_df = existing_df.sort_values(by="score", ascending=False)
            top_pool = existing_df.head(100).copy()
            
            # Add to seen_inchikeys
            seen_inchikeys = set(top_pool["InChIKey"].tolist())
            
            state['top_pool'] = top_pool
            state['seen_inchikeys'] = seen_inchikeys
            state['csv_data_loaded'] = True
            
            bt.logging.info(
                f"Loaded {len(existing_df)} existing molecules from CSV, "
                f"filled top_pool with top 100 molecules"
            )
        else:
            bt.logging.info("No existing molecules found in CSV, starting with empty pool")
            state['csv_data_loaded'] = True

        # 4) Launch the inference loop (runs continuously across epochs)
        try:
            state['inference_task'] = asyncio.create_task(run_model_loop(state))
            bt.logging.info("Inference loop started. Will continue across epochs without resetting.")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # 5) Main epoch-based loop (only for monitoring, no resets)
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # Just log epoch boundaries, don't reset anything
            if current_block % epoch_length == 0:
                current_epoch = current_block // epoch_length
                bt.logging.info(
                    f"Epoch boundary at block {current_block} (epoch {current_epoch}). "
                    f"Continuing inference without resetting state."
                )

            # Periodically update our knowledge of the network
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
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            state['shutdown_event'].set()  # ✅ Signal shutdown to inference loop
            break

# ============================================================================
# 8. ENTRY POINT
# ============================================================================

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())