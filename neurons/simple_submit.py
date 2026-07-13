#!/usr/bin/env python3
"""
SIMPLE MOLECULE SUBMISSION SCRIPT

Simple workflow:
1. Monitor blockchain for epoch boundaries
2. On startup and whenever the epoch changes, determine allowed reaction
3. Submit only when the allowed reaction is rxn:2
4. Load molecules from molecule.json (array of molecule entries)
5. Check each molecule for uniqueness in HuggingFace
6. Submit one molecule per wallet/hotkey pair
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
import signal
from typing import Any, Dict, List, Optional, Tuple

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
    ("nova", "notc"),
    ("nova", "notd"),
]
# ============================================================================

SUBMISSION_DELAY = 0.5       # Seconds between each hotkey submission
REQUIRED_ALLOWED_REACTION = "rxn:2"
EPOCH_LENGTH = 361           # Blocks per epoch
STATUS_LOG_INTERVAL = 60     # Log status every N seconds
POLL_INTERVAL = 6            # Seconds between block polls


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    bt.logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# CHALLENGE PARAM COMPAT
# ============================================================================

def resolve_challenge_params(config: argparse.Namespace, block_hash: str) -> Optional[Dict[str, Any]]:
    """Resolve epoch challenge params, including allowed_reaction."""
    def _cfg(key: str, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    small_molecule_target = _cfg("small_molecule_target", "")
    if isinstance(small_molecule_target, list):
        small_molecule_target = small_molecule_target[0] if small_molecule_target else ""

    nanobody_target = _cfg("nanobody_target", "")
    if isinstance(nanobody_target, list):
        nanobody_target = nanobody_target[0] if nanobody_target else ""

    num_antitargets = _cfg("num_antitargets", 0) or 0

    return get_challenge_params_from_blockhash(
        block_hash=block_hash,
        small_molecule_target=small_molecule_target,
        nanobody_target=nanobody_target,
        num_antitargets=num_antitargets,
        include_reaction=_cfg("random_valid_reaction", True),
    )


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
    bt.logging.info(f"Required allowed reaction: {REQUIRED_ALLOWED_REACTION}")
    bt.logging.info(f"Epoch length: {EPOCH_LENGTH} blocks")
    bt.logging.info(f"Trigger: immediately on startup, then on every epoch change")
    bt.logging.info(f"Poll interval: {POLL_INTERVAL}s")
    bt.logging.info(f"Wallet/hotkey pairs: {len(WALLET_HOTKEY_PAIRS)}")
    for idx, (wallet_name, hotkey_name) in enumerate(WALLET_HOTKEY_PAIRS, 1):
        bt.logging.info(f"   {idx}. wallet={wallet_name}  hotkey={hotkey_name}")


async def setup_bittensor_objects(
    config: argparse.Namespace,
) -> Tuple[List[Any], Any, Any, List[int], int]:
    """Initializes multiple wallets, subtensor, and metagraph."""
    bt.logging.info("Setting up Bittensor objects.")

    epoch_length = EPOCH_LENGTH
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
# EPOCH SUBMISSION
# ============================================================================

async def do_epoch_submission(state: Dict[str, Any], current_epoch: int) -> None:
    """Submit molecules from molecule.json when the epoch allows rxn:2."""
    config = state['config']
    subtensor = state['subtensor']
    epoch_length = state['epoch_length']

    bt.logging.info("=" * 70)
    bt.logging.info(f"SUBMITTING FOR EPOCH {current_epoch}")
    bt.logging.info("=" * 70)

    try:
        current_block = await subtensor.get_current_block()
        start_block = current_epoch * epoch_length
        start_block_hash = await subtensor.determine_block_hash(start_block)
        challenge_params = resolve_challenge_params(config, start_block_hash)

        if not challenge_params:
            bt.logging.error("Failed to get challenge params from blockhash")
            return

        allowed_reaction = challenge_params.get("allowed_reaction")
        bt.logging.info(f"Current block: {current_block}")
        bt.logging.info(f"Current epoch: {current_epoch}")
        bt.logging.info(f"Allowed reaction this epoch: {allowed_reaction}")

        if allowed_reaction != REQUIRED_ALLOWED_REACTION:
            bt.logging.warning(
                f"Allowed reaction is {allowed_reaction}, not {REQUIRED_ALLOWED_REACTION}. "
                "Skipping submission for this epoch."
            )
            return

        small_molecule_target = challenge_params.get("small_molecule_target")
        if not small_molecule_target:
            bt.logging.error("No small_molecule_target in challenge params")
            return

        state['current_challenge_targets'] = [small_molecule_target]
        bt.logging.info(
            f"Target: {small_molecule_target}, "
            f"Antitargets: {challenge_params.get('antitargets', [])}"
        )

        molecules = load_molecules_from_json(MOLECULE_JSON_PATH)
    except Exception as e:
        bt.logging.error(f"Failed during epoch {current_epoch} setup: {e}")
        bt.logging.error(traceback.format_exc())
        return

    wallets = state['wallets']
    miner_uids = state['miner_uids']
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

        if not molecule_name.startswith(f"{REQUIRED_ALLOWED_REACTION}:"):
            bt.logging.warning(
                f"Molecule {molecule_name} does not use {REQUIRED_ALLOWED_REACTION}. Skipping."
            )
            results.append(False)
            continue

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
        f"Epoch {current_epoch} submission complete: {success_count}/{len(results)} successful"
    )


async def run_epoch_loop(state: Dict[str, Any]) -> None:
    """
    Monitor blockchain and submit immediately on startup and on epoch change.
    """
    bt.logging.info("Starting epoch monitoring loop...")

    last_acted_epoch = None
    last_status_log = datetime.datetime.now()

    while not shutdown_event.is_set():
        try:
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            epoch_start_block = current_epoch * state['epoch_length']
            blocks_into_epoch = current_block - epoch_start_block

            now = datetime.datetime.now()
            if (now - last_status_log).total_seconds() >= STATUS_LOG_INTERVAL:
                bt.logging.info(
                    f"Status | Block: {current_block} | Epoch: {current_epoch} | "
                    f"Blocks into epoch: {blocks_into_epoch} | "
                    f"Last acted epoch: {last_acted_epoch}"
                )
                last_status_log = now

            if current_epoch != last_acted_epoch:
                if last_acted_epoch is None:
                    bt.logging.info(
                        f"First run detected — submitting immediately "
                        f"for current epoch {current_epoch}"
                    )
                else:
                    bt.logging.info(
                        f"Epoch changed ({last_acted_epoch} -> {current_epoch}) "
                        f"— submitting immediately"
                    )

                await do_epoch_submission(state, current_epoch)
                last_acted_epoch = current_epoch
                await asyncio.sleep(POLL_INTERVAL)
                continue

            await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            bt.logging.error(f"Error in epoch loop: {e}")
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)

    bt.logging.info("Epoch loop terminated by shutdown signal")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_simple_submit(config: argparse.Namespace) -> None:
    """Set up Bittensor objects and run the epoch monitoring loop."""
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

    try:
        await run_epoch_loop(state)
    finally:
        try:
            await subtensor.close()
            bt.logging.info("Subtensor connection closed")
        except Exception:
            pass


async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    try:
        await run_simple_submit(config)
    except KeyboardInterrupt:
        bt.logging.info("Interrupted by user")


if __name__ == "__main__":
    load_dotenv()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
