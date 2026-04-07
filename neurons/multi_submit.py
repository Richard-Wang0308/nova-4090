import os
import sys
import asyncio
import argparse
import datetime
import tempfile
import traceback
import base64
import hashlib
import subprocess
import sqlite3
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 5
STARTING_EPOCH = 21492
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")
ADD_COLUMN_SCRIPT = os.path.join(BASE_DIR, "add_column.py")

# ============================================================================
# HARDCODED HOTKEYS - EDIT THIS LIST
# ============================================================================
HOTKEY_NAMES = [
    "notc",
    "notd",
    "note",
    # Add more hotkey names as needed
]
# ============================================================================

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock


# ---------------------------------------------------------------------------
# Argument parsing & logging
# ---------------------------------------------------------------------------

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
            "multi_hotkey",  # Generic folder name since we use multiple hotkeys
            config.netuid,
            'miner',
        )
    )
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """Constructs the path for GitHub operations."""
    github_repo_name   = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_repo_owner  = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path   = os.environ.get('GITHUB_REPO_PATH')

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
    bt.logging.info(f"Using {len(HOTKEY_NAMES)} hotkeys: {HOTKEY_NAMES}")


# ---------------------------------------------------------------------------
# Bittensor setup
# ---------------------------------------------------------------------------

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[List[Any], Any, Any, List[int], int]:
    """
    Initializes multiple wallets (same wallet name, different hotkeys), subtensor, and metagraph.
    Returns: (wallets_list, subtensor, metagraph, miner_uids_list, epoch_length)
    """
    bt.logging.info("Setting up Bittensor objects with multiple hotkeys.")

    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            bt.logging.info(
                f"Attempting to connect to Bittensor network "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            subtensor = bt.async_subtensor(network=config.network)

            async with subtensor:
                metagraph = await subtensor.metagraph(config.netuid)
                await metagraph.sync()
                bt.logging.info("Metagraph synced successfully.")

                # Create wallet objects for each hotkey
                wallets = []
                miner_uids = []
                
                for hotkey_name in HOTKEY_NAMES:
                    # Create wallet with same wallet name but different hotkey
                    wallet = bt.wallet(name=config.wallet.name, hotkey=hotkey_name)
                    bt.logging.info(f"Wallet: {config.wallet.name}, Hotkey: {hotkey_name}")
                    
                    try:
                        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
                        bt.logging.info(f"  → UID: {miner_uid} for hotkey {hotkey_name}")
                        wallets.append(wallet)
                        miner_uids.append(miner_uid)
                    except ValueError:
                        bt.logging.warning(
                            f"  → Hotkey {hotkey_name} not found in metagraph. Skipping."
                        )
                        continue

                if not wallets:
                    raise ValueError("No valid hotkeys found in metagraph!")

                bt.logging.info(f"Successfully initialized {len(wallets)} wallets/hotkeys.")

                epoch_length = 361
                bt.logging.info(f"Epoch length: {epoch_length} blocks")

            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()

            return wallets, subtensor, metagraph, miner_uids, epoch_length

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


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_top_n_molecules_from_db(n: int, db_path: str = None) -> List[Tuple[str, float]]:
    """
    Return the top N molecules (molecule_name, score) that are still available
    (available = TRUE) from score_results.sqlite, ordered by score DESC.
    Returns empty list if no molecules are available.
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not os.path.exists(db_path):
        bt.logging.warning(f"score_results database not found at {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT molecule_name, score
            FROM   scored_molecules
            WHERE  available = TRUE
            ORDER  BY score DESC
            LIMIT  ?
            """,
            (n,)
        )
        rows = cursor.fetchall()
        conn.close()

        if rows:
            bt.logging.info(f"Retrieved top {len(rows)} available molecules from DB:")
            for idx, (mol_name, score) in enumerate(rows, 1):
                bt.logging.info(f"  {idx}. {mol_name} (score: {score})")
            return rows
        else:
            bt.logging.warning("No available molecules found in score_results database.")
            return []

    except Exception as e:
        bt.logging.error(f"Error querying top N molecules from DB: {e}")
        return []


def run_add_column_script() -> bool:
    """
    Run  `python3 add_column.py --skip-fix`  to mark submitted molecules
    as unavailable in the database.
    Returns True on success, False on failure.
    """
    try:
        bt.logging.info(f"Running: python3 {ADD_COLUMN_SCRIPT} --skip-fix")
        result = subprocess.run(
            ["python3", ADD_COLUMN_SCRIPT, "--skip-fix"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            bt.logging.info(
                f"✅ add_column.py completed successfully.\\n"
                f"   stdout: {result.stdout.strip()}"
            )
            return True
        else:
            bt.logging.error(
                f"❌ add_column.py exited with code {result.returncode}.\\n"
                f"   stderr: {result.stderr.strip()}\\n"
                f"   stdout: {result.stdout.strip()}"
            )
            return False

    except subprocess.TimeoutExpired:
        bt.logging.error("❌ add_column.py timed out after 60 seconds.")
        return False
    except Exception as e:
        bt.logging.error(f"❌ Error running add_column.py: {e}")
        return False


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

async def submit_response(
    wallet: Any,
    miner_uid: int,
    candidate_product: str,
    state: Dict[str, Any]
) -> bool:
    """
    Encrypts and submits the candidate product using the specified wallet/hotkey.
    Returns True on successful upload, False otherwise.
    """
    if not candidate_product:
        bt.logging.warning(f"No candidate product to submit for UID {miner_uid}.")
        return False

    bt.logging.info(
        f"📤 Starting submission for UID {miner_uid} (hotkey: {wallet.hotkey_str}): {candidate_product}"
    )

    try:
        current_block = await state['subtensor'].get_current_block()
        encrypted_response = state['bdt'].encrypt(
            miner_uid, candidate_product, current_block
        )
        bt.logging.info(f"🔐 Encrypted response generated for UID {miner_uid}.")

        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()

            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"📝 Prepared commit content for UID {miner_uid}: {commit_content}")

            bt.logging.info(f"⛓️  Attempting chain commitment for UID {miner_uid}...")
            try:
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=wallet,
                    netuid=state['config'].netuid,
                    data=commit_content,
                )
                bt.logging.info(f"⛓️  Chain commitment status for UID {miner_uid}: {commitment_status}")
            except MetadataError:
                bt.logging.info(f"⏳ Too soon to commit again for UID {miner_uid}. Will try next epoch.")
                return False

            if commitment_status:
                try:
                    bt.logging.info(f"📤 Attempting GitHub upload for UID {miner_uid}...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(
                            f"✅ File uploaded successfully for UID {miner_uid} to {commit_content}"
                        )
                        return True
                    else:
                        bt.logging.error(
                            f"❌ Failed to upload file to GitHub for UID {miner_uid}: {commit_content}"
                        )
                        return False
                except Exception as e:
                    bt.logging.error(f"❌ Failed to upload file for UID {miner_uid}: {e}")
                    return False

    except Exception as e:
        bt.logging.error(f"❌ Error in submit_response for UID {miner_uid}: {e}")
        bt.logging.error(traceback.format_exc())
        return False

    return False


# ---------------------------------------------------------------------------
# Main epoch loop
# ---------------------------------------------------------------------------

BLOCKS_BEFORE_BOUNDARY = 25   # trigger point: N blocks left until epoch end


async def run_epoch_loop(state: Dict[str, Any]) -> None:
    """
    Main loop:
      - Poll the current block every ~6 seconds.
      - When blocks_remaining_in_epoch <= BLOCKS_BEFORE_BOUNDARY AND we have
        not yet acted in this epoch:
          1. Run `python3 add_column.py --skip-fix` to refresh DB availability.
          2. Fetch the top N available molecules from score_results.sqlite.
          3. Submit them concurrently using N different hotkeys.
      - Repeat for every subsequent epoch.
    """
    bt.logging.info("🚀 Entering epoch-boundary submission loop...")

    last_acted_epoch: int = -1
    num_hotkeys = len(state['wallets'])

    while not state['shutdown_event'].is_set():
        try:
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            next_epoch_block = (current_epoch + 1) * state['epoch_length']
            blocks_remaining = next_epoch_block - current_block

            bt.logging.debug(
                f"Block: {current_block} | Epoch: {current_epoch} | "
                f"Next epoch at: {next_epoch_block} | Blocks remaining: {blocks_remaining}"
            )

            # ----------------------------------------------------------------
            # Act when we are within BLOCKS_BEFORE_BOUNDARY of the boundary
            # and have not yet acted in this epoch.
            # ----------------------------------------------------------------
            if blocks_remaining <= BLOCKS_BEFORE_BOUNDARY and current_epoch != last_acted_epoch:

                bt.logging.info(
                    f"\\n{'='*70}\\n"
                    f"⏰ {blocks_remaining} blocks left in epoch {current_epoch} — "
                    f"triggering submission sequence.\\n"
                    f"{'='*70}"
                )

                # Step 1: Update DB (mark submitted molecules as unavailable)
                bt.logging.info("💾 Step 1: Updating database via add_column.py --skip-fix ...")
                db_updated = run_add_column_script()
                if not db_updated:
                    bt.logging.warning(
                        "⚠️  add_column.py did not complete cleanly; "
                        "proceeding with submission anyway."
                    )

                # Step 2: Fetch top N available molecules
                bt.logging.info(f"🔍 Step 2: Fetching top {num_hotkeys} available molecules from DB...")
                top_molecules = get_top_n_molecules_from_db(num_hotkeys)

                if not top_molecules:
                    bt.logging.warning(
                        "⚠️  No available molecules found in DB. "
                        "Skipping submission for this epoch."
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                # Step 3: Submit concurrently using all hotkeys
                bt.logging.info(
                    f"📤 Step 3: Submitting {len(top_molecules)} molecules using {num_hotkeys} hotkeys..."
                )

                # Create submission tasks
                submission_tasks = []
                for idx, (molecule_name, score) in enumerate(top_molecules):
                    if idx >= len(state['wallets']):
                        bt.logging.warning(
                            f"⚠️  More molecules ({len(top_molecules)}) than hotkeys ({num_hotkeys}). "
                            f"Skipping molecule: {molecule_name}"
                        )
                        break

                    wallet = state['wallets'][idx]
                    miner_uid = state['miner_uids'][idx]
                    
                    bt.logging.info(
                        f"  → Hotkey {idx+1}/{num_hotkeys} (UID {miner_uid}): "
                        f"submitting '{molecule_name}' (score: {score})"
                    )
                    
                    task = submit_response(wallet, miner_uid, molecule_name, state)
                    submission_tasks.append((task, wallet.hotkey_str, molecule_name, miner_uid))

                # Execute all submissions concurrently
                bt.logging.info(f"⚡ Executing {len(submission_tasks)} submissions concurrently...")
                results = await asyncio.gather(
                    *[task for task, _, _, _ in submission_tasks],
                    return_exceptions=True
                )

                # Log results
                success_count = 0
                failure_count = 0
                for idx, result in enumerate(results):
                    _, hotkey_str, molecule_name, miner_uid = submission_tasks[idx]
                    if isinstance(result, Exception):
                        bt.logging.error(
                            f"❌ Submission {idx+1} FAILED (UID {miner_uid}, {hotkey_str}): "
                            f"{molecule_name} - Exception: {result}"
                        )
                        failure_count += 1
                    elif result:
                        bt.logging.info(
                            f"✅ Submission {idx+1} SUCCESS (UID {miner_uid}, {hotkey_str}): "
                            f"{molecule_name}"
                        )
                        success_count += 1
                    else:
                        bt.logging.error(
                            f"❌ Submission {idx+1} FAILED (UID {miner_uid}, {hotkey_str}): "
                            f"{molecule_name}"
                        )
                        failure_count += 1

                bt.logging.info(
                    f"\\n{'='*70}\\n"
                    f"📊 Epoch {current_epoch} Summary:\\n"
                    f"   ✅ Successful: {success_count}/{len(submission_tasks)}\\n"
                    f"   ❌ Failed: {failure_count}/{len(submission_tasks)}\\n"
                    f"{'='*70}\\n"
                )

                # Mark this epoch as handled
                last_acted_epoch = current_epoch

                # Wait a bit to avoid re-triggering on the same block
                await asyncio.sleep(12)
                continue

            # Not yet at the trigger point — sleep and poll again
            await asyncio.sleep(6)

        except Exception as e:
            bt.logging.error(f"Error in epoch loop: {e}")
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """Main mining coroutine."""
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
        'shutdown_event': asyncio.Event(),
    }

    # Resolve challenge targets
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    block_hash = await subtensor.determine_block_hash(last_boundary)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets,
    )
    if startup_proteins:
        state['current_challenge_targets']    = startup_proteins["targets"]
        state['current_challenge_antitargets'] = startup_proteins["antitargets"]

    bt.logging.info(
        f"✅ Miner initialised with {len(wallets)} hotkeys. Starting epoch-boundary loop."
    )
    await run_epoch_loop(state)


def main():
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)

    try:
        asyncio.run(run_miner(config))
    except KeyboardInterrupt:
        bt.logging.info("Miner interrupted by user.")
    except Exception as e:
        bt.logging.error(f"Fatal error in miner: {e}")
        bt.logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()