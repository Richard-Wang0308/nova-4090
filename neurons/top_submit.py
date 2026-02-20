#!/usr/bin/env python3
"""
TOP MOLECULE SUBMISSION SCRIPT

Workflow:
1. Load molecules from score_results.sqlite ordered by score (descending)
2. For each molecule:
   - Check if it is already in HuggingFace
   - Validate molecule (banned atoms, etc.)
   - Try to submit it
   - If submission fails, try next molecule
   - Continue until one succeeds
"""

import os
import sys
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import sqlite3
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError
from rdkit import Chem

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    contains_atom_type,
    get_heavy_atom_count,
)
from utils.molecules import molecule_unique_for_protein_hf
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock

# Path to score_results database
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    parser.add_argument('--max-attempts', type=int, default=200, help="Maximum number of molecules to try")
    parser.add_argument('--rxn-id', type=int, default=None, help="Filter by reaction ID (optional)")
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
            'top_submit',
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
    bt.logging.info(f"Running top_submit for subnet: {config.netuid}")


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


def load_top_molecules_from_db(
    db_path: str,
    max_molecules: int = 100,
    rxn_id: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Load top molecules from score_results database ordered by score (descending).
    
    Args:
        db_path: Path to SQLite database
        max_molecules: Maximum number of molecules to load
        rxn_id: Optional reaction ID filter (e.g., 1 or 2)
    
    Returns:
        List of tuples (molecule_name, score) ordered by score descending
    """
    if not os.path.exists(db_path):
        bt.logging.warning(f"Database file not found at {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if rxn_id is not None:
            # Filter by rxn_id
            query = """
                SELECT molecule_name, score 
                FROM scored_molecules 
                WHERE molecule_name LIKE ?
                ORDER BY score DESC
                LIMIT ?
            """
            pattern = f"rxn:{rxn_id}:%"
            cursor.execute(query, (pattern, max_molecules))
        else:
            # Load all molecules
            query = """
                SELECT molecule_name, score 
                FROM scored_molecules 
                ORDER BY score DESC
                LIMIT ?
            """
            cursor.execute(query, (max_molecules,))
        
        results = cursor.fetchall()
        conn.close()
        
        bt.logging.info(f"✅ Loaded {len(results)} molecules from database")
        if len(results) > 0:
            bt.logging.info(f"   Top score: {results[0][1]:.6f}, Bottom score: {results[-1][1]:.6f}")
        
        return results
    except Exception as e:
        bt.logging.error(f"Error loading molecules from database: {e}")
        bt.logging.error(traceback.format_exc())
        return []


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
        
        bt.logging.debug(f"✅ Molecule {molecule_name} is NOT in HuggingFace (unique!)")
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
        return False


def validate_molecule(molecule_name: str, smiles: str, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate molecule meets requirements.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES"
        
        # Check banned atoms
        banned_atoms = config.get('banned_atom_types', [])
        if banned_atoms and contains_atom_type(mol, banned_atoms):
            return False, f"Contains banned atom types: {banned_atoms}"
        
        # Check heavy atom count
        min_heavy_atoms = config.get('min_heavy_atoms', 10)
        heavy_atom_count = get_heavy_atom_count(smiles)
        if heavy_atom_count < min_heavy_atoms:
            return False, f"Insufficient heavy atoms: {heavy_atom_count} < {min_heavy_atoms}"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def submit_response(state: Dict[str, Any]) -> bool:
    """
    Encrypts and submits the current candidate product.
    
    Returns:
        True if submission was successful, False otherwise
    """
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return False

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
            except MetadataError as e:
                bt.logging.warning(f"⏳ Too soon to commit again: {e}")
                return False

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
                        return True
                    else:
                        bt.logging.error(f"❌ Failed to upload file to GitHub for {commit_content}")
                        return False
                except Exception as e:
                    bt.logging.error(f"❌ Failed to upload file for {commit_content}: {e}")
                    return False
            else:
                bt.logging.warning("❌ Commitment status was False")
                return False
    
    except Exception as e:
        bt.logging.error(f"❌ Error in submit_response: {e}")
        bt.logging.error(traceback.format_exc())
        return False


# ============================================================================
# MAIN SUBMISSION LOGIC
# ============================================================================

async def run_top_submit(config: argparse.Namespace) -> None:
    """Main submission logic: load top molecules from DB, try submitting each until one succeeds."""
    
    # Setup Bittensor objects
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)
    
    # Initialize state
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
        'current_challenge_targets': [],
    }
    
    bt.logging.info("🚀 Starting top molecule submission process...")
    
    # Get challenge targets from current block
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    block_hash = await subtensor.determine_block_hash(last_boundary)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )
    
    if not startup_proteins:
        bt.logging.error("❌ Failed to get challenge targets from blockhash")
        return
    
    state['current_challenge_targets'] = startup_proteins["targets"]
    bt.logging.info(
        f"Targets: {startup_proteins['targets']}, "
        f"Antitargets: {startup_proteins['antitargets']}"
    )
    
    # Load top molecules from database
    rxn_id = getattr(config, 'rxn_id', None)
    max_attempts = getattr(config, 'max_attempts', 200)
    
    molecules = load_top_molecules_from_db(
        SCORE_RESULTS_DB,
        max_molecules=max_attempts,
        rxn_id=rxn_id
    )
    
    if not molecules:
        bt.logging.error("❌ No molecules found in database")
        return
    
    bt.logging.info(f"📊 Will try submitting up to {len(molecules)} molecules (ordered by score)")
    
    # Try submitting each molecule until one succeeds
    successful_submission = False
    attempted_count = 0
    skipped_not_unique = 0
    skipped_invalid = 0
    failed_submission = 0
    
    for idx, (molecule_name, score) in enumerate(molecules, 1):
        attempted_count += 1
        bt.logging.info(
            f"\n{'='*70}\n"
            f"Attempt {idx}/{len(molecules)}: {molecule_name} (score: {score:.6f})\n"
            f"{'='*70}"
        )
        
        try:
            # Get SMILES from molecule name
            smiles = get_smiles_from_reaction(molecule_name)
            if not smiles:
                bt.logging.warning(f"⚠️  Could not get SMILES for {molecule_name}, skipping...")
                skipped_invalid += 1
                continue
            
            # Validate molecule
            is_valid, error_msg = validate_molecule(molecule_name, smiles, config)
            if not is_valid:
                bt.logging.warning(f"⚠️  Molecule {molecule_name} failed validation: {error_msg}")
                skipped_invalid += 1
                continue
            
            # Check if molecule is unique (not in HuggingFace)
            bt.logging.info(f"🔍 Checking if molecule {molecule_name} is unique...")
            is_unique = await check_molecule_unique(state, molecule_name, smiles)
            
            if not is_unique:
                bt.logging.warning(f"⚠️  Molecule {molecule_name} is already in HuggingFace, skipping...")
                skipped_not_unique += 1
                continue
            
            # Molecule is valid and unique, try to submit it
            bt.logging.info(f"✅ Molecule {molecule_name} is valid and unique! Attempting submission...")
            state['candidate_product'] = molecule_name
            
            submission_success = await submit_response(state)
            
            if submission_success:
                bt.logging.info(
                    f"\n{'='*70}\n"
                    f"✅ SUCCESS! Successfully submitted molecule {molecule_name} (score: {score:.6f})\n"
                    f"{'='*70}"
                )
                successful_submission = True
                break
            else:
                bt.logging.warning(f"⚠️  Submission failed for {molecule_name}, trying next molecule...")
                failed_submission += 1
                # Continue to next molecule
        
        except Exception as e:
            bt.logging.error(f"❌ Error processing molecule {molecule_name}: {e}")
            bt.logging.error(traceback.format_exc())
            failed_submission += 1
            # Continue to next molecule
    
    # Summary
    bt.logging.info(
        f"\n{'='*70}\n"
        f"SUBMISSION SUMMARY\n"
        f"{'='*70}\n"
        f"Total molecules attempted: {attempted_count}\n"
        f"Successful submissions: {1 if successful_submission else 0}\n"
        f"Skipped (not unique): {skipped_not_unique}\n"
        f"Skipped (invalid): {skipped_invalid}\n"
        f"Failed submissions: {failed_submission}\n"
        f"{'='*70}"
    )
    
    if not successful_submission:
        bt.logging.warning("❌ No molecules were successfully submitted")


async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    await run_top_submit(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
