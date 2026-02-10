#!/usr/bin/env python3
"""
SIMPLE MOLECULE SUBMISSION SCRIPT

Simple workflow:
1. Load molecule from molecule.json
2. Check if it is already in HuggingFace
3. If it isn't there, submit it immediately
"""

import os
import sys
import json
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
from typing import Any, Dict, Tuple
from pathlib import Path

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
)
from utils.molecules import molecule_unique_for_protein_hf
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock

# Path to molecule.json file
MOLECULE_JSON_PATH = os.path.join(BASE_DIR, "molecule.json")


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
            'simple_submit',
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
    bt.logging.info(f"Running simple_submit for subnet: {config.netuid}")


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


def load_molecule_from_json(json_path: str) -> Dict[str, Any]:
    """
    Load molecule from JSON file.
    
    Expected JSON format:
    {
        "name": "rxn:1:12345:67890"
    }
    
    SMILES will be derived from the molecule name using get_smiles_from_reaction.
    
    Returns:
        Dictionary with 'name' and 'smiles' keys
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Molecule JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            molecule_data = json.load(f)
        
        if 'name' not in molecule_data:
            raise ValueError("Molecule JSON missing 'name' field")
        
        molecule_name = molecule_data['name']
        
        # Get SMILES from molecule name
        smiles = get_smiles_from_reaction(molecule_name)
        if not smiles:
            raise ValueError(f"Could not get SMILES from molecule name: {molecule_name}")
        
        molecule_data['smiles'] = smiles
        
        bt.logging.info(f"✅ Loaded molecule from {json_path}")
        bt.logging.info(f"   Name: {molecule_name}")
        bt.logging.info(f"   SMILES: {smiles}")
        
        return molecule_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading molecule from {json_path}: {e}")


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
            bt.logging.warning(f"❌ Molecule {molecule_name} already in HuggingFace dataset")
            return False
        
        bt.logging.info(f"✅ Molecule {molecule_name} is NOT in HuggingFace (unique!)")
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
        return False


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
                bt.logging.warning("⏳ Too soon to commit again. Will try next epoch.")
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
        bt.logging.error(traceback.format_exc())


# ============================================================================
# MAIN SUBMISSION LOGIC
# ============================================================================

async def run_simple_submit(config: argparse.Namespace) -> None:
    """Main submission logic: load molecule, check uniqueness, submit if unique."""
    
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
    
    bt.logging.info("🚀 Starting simple submission process...")
    
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
    
    # Load molecule from JSON
    try:
        molecule = load_molecule_from_json(MOLECULE_JSON_PATH)
        molecule_name = molecule['name']
        molecule_smiles = molecule['smiles']
    except Exception as e:
        bt.logging.error(f"❌ Failed to load molecule from JSON: {e}")
        bt.logging.error(traceback.format_exc())
        return
    
    # Check if molecule is unique (not in HuggingFace)
    bt.logging.info(f"🔍 Checking if molecule {molecule_name} is unique...")
    is_unique = await check_molecule_unique(state, molecule_name, molecule_smiles)
    
    if not is_unique:
        bt.logging.warning(f"❌ Molecule {molecule_name} is already in HuggingFace. Not submitting.")
        return
    
    # Molecule is unique, submit it
    bt.logging.info(f"✅ Molecule {molecule_name} is unique! Submitting...")
    state['candidate_product'] = molecule_name
    
    try:
        await submit_response(state)
        bt.logging.info(f"✅ Successfully submitted molecule {molecule_name}!")
    except Exception as e:
        bt.logging.error(f"❌ Error submitting molecule: {e}")
        bt.logging.error(traceback.format_exc())


async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    await run_simple_submit(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
