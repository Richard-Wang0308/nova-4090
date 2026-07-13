#!/usr/bin/env python3
"""
SIMPLE MOLECULE SUBMISSION SCRIPT

Simple workflow:
1. Load molecules from molecule.json (array of molecule entries)
2. Check each molecule for uniqueness in HuggingFace
3. Submit one molecule per wallet/hotkey pair
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
from typing import Any, Dict, List, Tuple

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
)
from utils.molecules import molecule_unique_for_protein_hf
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock

# Path to molecule.json file
MOLECULE_JSON_PATH = os.path.join(BASE_DIR, "molecule.json")

# ============================================================================
# WALLET + HOTKEY CONFIGURATION
# Each entry is a (wallet_name, hotkey_name) pair.
# One molecule is submitted per pair, in order.
# ============================================================================
WALLET_HOTKEY_PAIRS: List[Tuple[str, str]] = [
    ("nova", "nota"),
    ("nova", "notb"),
]
# ============================================================================

SUBMISSION_DELAY = 0.5  # Seconds between each hotkey submission


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

    primary_wallet_name = WALLET_HOTKEY_PAIRS[0][0] if WALLET_HOTKEY_PAIRS else config.wallet.name

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            primary_wallet_name,
            "multi_wallet_hotkey",
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
    bt.logging.info(f"Wallet/hotkey pairs: {len(WALLET_HOTKEY_PAIRS)}")
    for idx, (wallet_name, hotkey_name) in enumerate(WALLET_HOTKEY_PAIRS, 1):
        bt.logging.info(f"   {idx}. wallet={wallet_name}  hotkey={hotkey_name}")


async def setup_bittensor_objects(
    config: argparse.Namespace,
) -> Tuple[List[Any], Any, Any, List[int], int]:
    """Initializes multiple wallets, subtensor, and metagraph."""
    bt.logging.info("Setting up Bittensor objects.")

    epoch_length = 361
    subtensor = bt.async_subtensor(network=config.network)

    async with subtensor:
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()
        bt.logging.info("Metagraph synced successfully.")

        wallets: List[Any] = []
        miner_uids: List[int] = []

        for idx, (wallet_name, hotkey_name) in enumerate(WALLET_HOTKEY_PAIRS, 1):
            label = f"{wallet_name}/{hotkey_name}"
            try:
                wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
                _ = wallet.hotkey

                miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
                wallets.append(wallet)
                miner_uids.append(miner_uid)

                bt.logging.info(
                    f"   {idx}. {label} -> UID {miner_uid} "
                    f"({wallet.hotkey.ss58_address[:10]}...)"
                )
            except ValueError:
                bt.logging.warning(f"   {idx}. {label} -> NOT FOUND in metagraph (skipping)")
            except FileNotFoundError:
                bt.logging.warning(f"   {idx}. {label} -> Hotkey file not found (skipping)")
            except Exception as e:
                bt.logging.error(f"   {idx}. {label} -> ERROR: {e}")

        if not wallets:
            raise ValueError(
                "No valid wallet/hotkey pairs found in metagraph. "
                "Check WALLET_HOTKEY_PAIRS configuration."
            )

    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    bt.logging.info(f"Epoch length: {epoch_length} blocks")
    return wallets, subtensor, metagraph, miner_uids, epoch_length


def load_molecules_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load molecules from JSON file.

    Expected JSON format:
    [
        {"name": "rxn:1:12345:67890"},
        {"name": "rxn:2:11111:22222"}
    ]

    SMILES will be derived from each molecule name using get_smiles_from_reaction.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Molecule JSON file not found: {json_path}")

    try:
        with open(json_path, 'r') as f:
            molecule_data = json.load(f)

        if isinstance(molecule_data, dict):
            if 'name' in molecule_data:
                molecule_entries = [molecule_data]
            elif 'molecules' in molecule_data:
                molecule_entries = molecule_data['molecules']
            else:
                raise ValueError("Molecule JSON object must contain 'name' or 'molecules'")
        elif isinstance(molecule_data, list):
            molecule_entries = molecule_data
        else:
            raise ValueError("Molecule JSON must be an array or object with molecule entries")

        if not molecule_entries:
            raise ValueError("Molecule JSON contains no entries")

        molecules: List[Dict[str, Any]] = []
        for idx, entry in enumerate(molecule_entries, 1):
            if not isinstance(entry, dict) or 'name' not in entry:
                raise ValueError(f"Molecule entry {idx} is missing 'name' field")

            molecule_name = entry['name']
            smiles = get_smiles_from_reaction(molecule_name)
            if not smiles:
                raise ValueError(f"Could not get SMILES from molecule name: {molecule_name}")

            molecules.append({
                'name': molecule_name,
                'smiles': smiles,
            })

            bt.logging.info(f"Loaded molecule {idx} from {json_path}")
            bt.logging.info(f"   Name: {molecule_name}")
            bt.logging.info(f"   SMILES: {smiles}")

        return molecules
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading molecules from {json_path}: {e}")


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
            bt.logging.warning(f"Molecule {molecule_name} already in HuggingFace dataset")
            return False

        bt.logging.info(f"Molecule {molecule_name} is NOT in HuggingFace (unique!)")
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
        return False


def validate_molecule(
    config: argparse.Namespace,
    molecule_name: str,
    molecule_smiles: str,
) -> bool:
    """Validate SMILES and banned atom types."""
    mol = Chem.MolFromSmiles(molecule_smiles)
    if mol is None:
        bt.logging.warning(f"Molecule {molecule_name} is not a valid SMILES. Skipping.")
        return False
    if contains_atom_type(mol, config.banned_atom_types):
        bt.logging.warning(f"Molecule {molecule_name} contains banned atom types. Skipping.")
        return False
    return True


async def submit_response(
    wallet: Any,
    miner_uid: int,
    candidate_product: str,
    state: Dict[str, Any],
    submission_number: int,
    total_submissions: int,
) -> bool:
    """Encrypts and submits a molecule using the specified wallet/hotkey."""
    if not candidate_product:
        bt.logging.warning(f"No candidate product to submit for UID {miner_uid}")
        return False

    wallet_name = wallet.name if hasattr(wallet, 'name') else 'unknown'
    hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
    label = f"{wallet_name}/{hotkey_name}"

    bt.logging.info(
        f"[{submission_number}/{total_submissions}] Submitting UID {miner_uid} ({label})"
    )
    bt.logging.info(f"Molecule: {candidate_product}")

    try:
        current_block = await state['subtensor'].get_current_block()
        encrypted_response = state['bdt'].encrypt(miner_uid, candidate_product, current_block)
        bt.logging.info("Encrypted response generated successfully")

        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()

            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"Prepared commit content: {commit_content}")

            bt.logging.info("Attempting chain commitment...")
            try:
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=wallet,
                    netuid=state['config'].netuid,
                    data=commit_content
                )
                bt.logging.info(f"Chain commitment status: {commitment_status}")
            except MetadataError:
                bt.logging.warning(
                    f"Too soon to commit again for UID {miner_uid} ({label}). Skipping."
                )
                return False

            if commitment_status:
                try:
                    bt.logging.info(f"Commitment set successfully for {commit_content}")
                    bt.logging.info("Attempting GitHub upload...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(f"File uploaded successfully to {commit_content}")
                        current_epoch = current_block // state['epoch_length']
                        bt.logging.info(f"Submission recorded for epoch {current_epoch}")
                        return True
                    else:
                        bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
                except Exception as e:
                    bt.logging.error(f"Failed to upload file for {commit_content}: {e}")

        return False

    except Exception as e:
        bt.logging.error(f"Error in submit_response for UID {miner_uid} ({label}): {e}")
        bt.logging.error(traceback.format_exc())
        return False


# ============================================================================
# MAIN SUBMISSION LOGIC
# ============================================================================

async def run_simple_submit(config: argparse.Namespace) -> None:
    """Load molecules, validate uniqueness, and submit one per wallet/hotkey."""
    wallets, subtensor, metagraph, miner_uids, epoch_length = await setup_bittensor_objects(config)

    state: Dict[str, Any] = {
        'config': config,
        'github_path': load_github_path(),
        'wallets': wallets,
        'miner_uids': miner_uids,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'epoch_length': epoch_length,
        'bdt': QuicknetBittensorDrandTimelock(),
        'current_challenge_targets': [],
    }

    bt.logging.info("Starting simple submission process...")

    try:
        current_block = await subtensor.get_current_block()
        last_boundary = (current_block // epoch_length) * epoch_length
        block_hash = await subtensor.determine_block_hash(last_boundary)
        startup_proteins = get_challenge_params_from_blockhash(
            block_hash=block_hash,
            weekly_target=config.weekly_target,
            num_antitargets=config.num_antitargets
        )

        if not startup_proteins:
            bt.logging.error("Failed to get challenge targets from blockhash")
            return

        state['current_challenge_targets'] = startup_proteins["targets"]
        bt.logging.info(
            f"Targets: {startup_proteins['targets']}, "
            f"Antitargets: {startup_proteins['antitargets']}"
        )

        molecules = load_molecules_from_json(MOLECULE_JSON_PATH)
    except Exception as e:
        bt.logging.error(f"Failed during startup: {e}")
        bt.logging.error(traceback.format_exc())
        return

    num_pairs = len(wallets)
    if len(molecules) > num_pairs:
        bt.logging.warning(
            f"More molecules ({len(molecules)}) than wallet/hotkey pairs ({num_pairs}). "
            "Extra molecules will be skipped."
        )
    elif len(molecules) < num_pairs:
        bt.logging.warning(
            f"Fewer molecules ({len(molecules)}) than wallet/hotkey pairs ({num_pairs}). "
            "Only the first molecules will be submitted."
        )

    submissions = list(zip(molecules, wallets, miner_uids))
    results = []

    for idx, (molecule, wallet, miner_uid) in enumerate(submissions, 1):
        molecule_name = molecule['name']
        molecule_smiles = molecule['smiles']

        bt.logging.info(f"Checking molecule {idx}/{len(submissions)}: {molecule_name}")

        if not validate_molecule(config, molecule_name, molecule_smiles):
            results.append(False)
            continue

        is_unique = await check_molecule_unique(state, molecule_name, molecule_smiles)
        if not is_unique:
            bt.logging.warning(
                f"Molecule {molecule_name} is already in HuggingFace. Not submitting."
            )
            results.append(False)
            continue

        bt.logging.info(f"Molecule {molecule_name} is unique. Submitting...")
        success = await submit_response(
            wallet,
            miner_uid,
            molecule_name,
            state,
            idx,
            len(submissions),
        )
        results.append(success)

        if idx < len(submissions):
            bt.logging.info(f"Waiting {SUBMISSION_DELAY}s before next submission...")
            await asyncio.sleep(SUBMISSION_DELAY)

    success_count = sum(1 for result in results if result)
    bt.logging.info(
        f"Submission complete: {success_count}/{len(results)} successful"
    )


async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    await run_simple_submit(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
