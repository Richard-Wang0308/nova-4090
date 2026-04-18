#!/usr/bin/env python3
"""
MULTI-HOTKEY EPOCH-BASED MOLECULE SUBMISSION SCRIPT

Workflow:
1. Monitor blockchain for epoch boundaries
2. At 28 blocks before boundary, update database
3. Fetch top N molecules from database
4. Submit all N molecules SEQUENTIALLY (one at a time with delays)
5. Wait for next epoch's submission window
"""

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
import signal
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Database paths
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")
ADD_COLUMN_SCRIPT = os.path.join(BASE_DIR, "add_column.py")

# Wallet configuration
WALLET_NAME = "nova"  # Hardcoded wallet name

# Hotkey configuration - EDIT THIS LIST
HOTKEY_NAMES = [
    'nota',
    'notb',
    'note',
    'notd',
    'notf'
]

# Timing configuration
BLOCKS_BEFORE_BOUNDARY = 30  # Trigger point: 28 blocks before epoch end
EPOCH_LENGTH = 361           # Blocks per epoch
STATUS_LOG_INTERVAL = 60     # Log status every N seconds
SUBMISSION_DELAY = 2        # Seconds between each hotkey submission

# ============================================================================

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock


# ============================================================================
# SIGNAL HANDLING FOR PM2
# ============================================================================

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    bt.logging.info(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# ARGUMENT PARSING & LOGGING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-hotkey epoch-based molecule submission miner"
    )
    parser.add_argument(
        '--network',
        default=os.getenv('SUBTENSOR_NETWORK', 'finney'),
        help='Bittensor network to use'
    )
    parser.add_argument(
        '--netuid',
        type=int,
        default=68,
        help="The chain subnet uid"
    )
    
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    config = bt.config(parser)
    config.update(load_config())
    
    # IMPORTANT: Override wallet name with hardcoded value
    config.wallet.name = WALLET_NAME

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            WALLET_NAME,  # Use hardcoded wallet name
            "multi_hotkey",
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
    github_repo_path   = os.environ.get('GITHUB_REPO_PATH', '')

    if not all([github_repo_name, github_repo_branch, github_repo_owner]):
        raise ValueError(
            "Missing required GitHub environment variables: "
            "GITHUB_REPO_NAME, GITHUB_REPO_BRANCH, GITHUB_REPO_OWNER"
        )

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError(f"GitHub path too long (max 100 chars): {len(github_path)} chars")

    return github_path


def setup_logging(config: argparse.Namespace) -> None:
    """Sets up Bittensor logging."""
    bt.logging(config=config, logging_dir=config.full_path)
    
    bt.logging.info("\n" + "="*70)
    bt.logging.info("🚀 MULTI-HOTKEY EPOCH MINER STARTING")
    bt.logging.info("="*70)
    bt.logging.info(f"📡 Network: {config.network}")
    bt.logging.info(f"🔗 Netuid: {config.netuid}")
    bt.logging.info(f"💼 Wallet: {WALLET_NAME}")
    bt.logging.info(f"👥 Hotkeys configured: {len(HOTKEY_NAMES)}")
    bt.logging.info(f"   {HOTKEY_NAMES}")
    bt.logging.info(f"⏰ Trigger point: {BLOCKS_BEFORE_BOUNDARY} blocks before epoch boundary")
    bt.logging.info(f"📊 Epoch length: {EPOCH_LENGTH} blocks")
    bt.logging.info(f"⏱️  Submission delay: {SUBMISSION_DELAY}s between hotkeys")
    bt.logging.info("="*70 + "\n")


# ============================================================================
# BITTENSOR SETUP
# ============================================================================

async def setup_bittensor_objects(
    config: argparse.Namespace
) -> Tuple[List[Any], Any, Any, List[int], int]:
    """
    Initializes multiple wallets (same wallet name, different hotkeys),
    subtensor, and metagraph.
    
    Returns:
        (wallets_list, subtensor, metagraph, miner_uids_list, epoch_length)
    """
    bt.logging.info("🔧 Setting up Bittensor objects with multiple hotkeys...")

    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            bt.logging.info(
                f"   Attempting connection (attempt {attempt + 1}/{max_retries})..."
            )

            subtensor = bt.async_subtensor(network=config.network)

            async with subtensor:
                metagraph = await subtensor.metagraph(config.netuid)
                await metagraph.sync()
                bt.logging.info("   ✅ Metagraph synced successfully\n")

                # Create wallet objects for each hotkey
                bt.logging.info(f"   📋 Initializing {len(HOTKEY_NAMES)} hotkeys:")
                wallets = []
                miner_uids = []
                
                for idx, hotkey_name in enumerate(HOTKEY_NAMES, 1):
                    try:
                        # Create wallet with hardcoded wallet name and specific hotkey
                        wallet = bt.wallet(name=WALLET_NAME, hotkey=hotkey_name)
                        
                        # Get UID from metagraph
                        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
                        
                        wallets.append(wallet)
                        miner_uids.append(miner_uid)
                        
                        bt.logging.info(
                            f"      {idx}. ✅ {hotkey_name:<12} → UID {miner_uid:>3} "
                            f"({wallet.hotkey.ss58_address[:10]}...)"
                        )
                        
                    except ValueError:
                        bt.logging.warning(
                            f"      {idx}. ⚠️  {hotkey_name:<12} → NOT FOUND in metagraph (skipping)"
                        )
                        continue
                    except Exception as e:
                        bt.logging.error(
                            f"      {idx}. ❌ {hotkey_name:<12} → ERROR: {e}"
                        )
                        continue

                if not wallets:
                    raise ValueError(
                        "❌ No valid hotkeys found in metagraph! "
                        "Please check your hotkey configuration."
                    )

                bt.logging.info(f"\n   ✅ Successfully initialized {len(wallets)}/{len(HOTKEY_NAMES)} hotkeys\n")

            # Reinitialize subtensor for main loop
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()

            return wallets, subtensor, metagraph, miner_uids, EPOCH_LENGTH

        except (ConnectionError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                bt.logging.warning(
                    f"   ⚠️  Connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)
            else:
                bt.logging.error(
                    f"   ❌ Failed to connect after {max_retries} attempts: {e}"
                )
                raise
                
        except Exception as e:
            bt.logging.error(f"   ❌ Unexpected error during setup: {e}")
            bt.logging.error(traceback.format_exc())
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                raise


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def get_top_n_molecules_from_db(n: int, db_path: str = None) -> List[Tuple[str, float]]:
    """
    Fetch top N available molecules from score_results database.
    
    Args:
        n: Number of molecules to fetch
        db_path: Path to database (defaults to SCORE_RESULTS_DB)
        
    Returns:
        List of (molecule_name, score) tuples, ordered by score DESC
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not os.path.exists(db_path):
        bt.logging.error(f"   ❌ Database not found: {db_path}")
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
            bt.logging.info(f"   ✅ Retrieved {len(rows)} available molecules:")
            for idx, (mol_name, score) in enumerate(rows, 1):
                bt.logging.info(f"      {idx}. {mol_name:<30} | Score: {score:.6f}")
            return rows
        else:
            bt.logging.warning("   ⚠️  No available molecules found in database")
            return []

    except sqlite3.Error as e:
        bt.logging.error(f"   ❌ Database error: {e}")
        return []
    except Exception as e:
        bt.logging.error(f"   ❌ Error querying database: {e}")
        bt.logging.error(traceback.format_exc())
        return []


def run_add_column_script() -> bool:
    """
    Execute add_column.py --skip-fix to update database availability.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        bt.logging.info(f"   💾 Running: python3 {ADD_COLUMN_SCRIPT} --skip-fix")
        
        start_time = datetime.datetime.now()
        
        result = subprocess.run(
            ["python3", ADD_COLUMN_SCRIPT, "--skip-fix"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=BASE_DIR,
        )
        
        elapsed = (datetime.datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            bt.logging.info(f"   ✅ Database update completed in {elapsed:.2f}s")
            if result.stdout.strip():
                bt.logging.debug(f"   Output: {result.stdout.strip()}")
            return True
        else:
            bt.logging.error(
                f"   ❌ Database update failed (exit code {result.returncode})\n"
                f"   stderr: {result.stderr.strip()}"
            )
            return False

    except subprocess.TimeoutExpired:
        bt.logging.error("   ❌ Database update timed out after 120 seconds")
        return False
    except FileNotFoundError:
        bt.logging.error(f"   ❌ Script not found: {ADD_COLUMN_SCRIPT}")
        return False
    except Exception as e:
        bt.logging.error(f"   ❌ Error running database update: {e}")
        bt.logging.error(traceback.format_exc())
        return False


# ============================================================================
# SUBMISSION
# ============================================================================

async def submit_response(
    wallet: Any,
    miner_uid: int,
    candidate_product: str,
    state: Dict[str, Any],
    submission_number: int,
    total_submissions: int
) -> bool:
    """
    Encrypt and submit a molecule using the specified wallet/hotkey.
    
    Args:
        wallet: Bittensor wallet object
        miner_uid: Miner UID
        candidate_product: Molecule name to submit
        state: Global state dictionary
        submission_number: Current submission number (for logging)
        total_submissions: Total number of submissions (for logging)
        
    Returns:
        True if submission successful, False otherwise
    """
    if not candidate_product:
        bt.logging.warning(f"      ⚠️  UID {miner_uid}: No candidate product")
        return False

    hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
    
    bt.logging.info(f"\n   [{submission_number}/{total_submissions}] 📤 SUBMITTING: UID {miner_uid} ({hotkey_name})")
    bt.logging.info(f"      Molecule: {candidate_product}")
    
    try:
        # Get current block
        current_block = await state['subtensor'].get_current_block()
        bt.logging.info(f"      Current block: {current_block}")
        
        # Encrypt response
        bt.logging.info(f"      🔐 Encrypting response...")
        message = f"{candidate_product}|~"
        encrypted_response = state['bdt'].encrypt(
            miner_uid, message, current_block
        )
        bt.logging.info(f"      ✅ Encryption successful")
        
        # Create temporary file with encrypted content
        tmp_file = tempfile.NamedTemporaryFile(delete=True, mode='w+')
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()
            f.seek(0)
            
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()
            
            # Generate filename hash
            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"      📝 Commit path: {commit_content}")
            
            # Commit to blockchain
            bt.logging.info(f"      ⛓️  Attempting blockchain commitment...")
            try:
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=wallet,
                    netuid=state['config'].netuid,
                    data=commit_content,
                )
                
                bt.logging.info(f"      ✅ Commitment status: {commitment_status}")
                
                if not commitment_status:
                    bt.logging.error(
                        f"      ❌ SUBMISSION FAILED for UID {miner_uid}: "
                        f"Blockchain commitment returned False"
                    )
                    return False
                    
            except MetadataError as e:
                bt.logging.warning(
                    f"      ⏳ MetadataError for UID {miner_uid}: {e}"
                )
                bt.logging.warning(
                    f"      ⏳ Too soon to commit again (rate limited)"
                )
                bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid}")
                return False
            
            # Upload to GitHub
            bt.logging.info(f"      📤 Uploading to GitHub...")
            try:
                github_status = upload_file_to_github(filename, encoded_content)
                
                if github_status:
                    bt.logging.info(
                        f"      ✅ SUBMISSION SUCCESSFUL for UID {miner_uid}"
                    )
                    return True
                else:
                    bt.logging.error(
                        f"      ❌ GitHub upload failed for UID {miner_uid}"
                    )
                    bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid}")
                    return False
                    
            except Exception as e:
                bt.logging.error(
                    f"      ❌ GitHub upload error for UID {miner_uid}: {e}"
                )
                bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid}")
                return False

    except Exception as e:
        bt.logging.error(
            f"      ❌ Submission error for UID {miner_uid}: {e}"
        )
        bt.logging.error(traceback.format_exc())
        bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid}")
        return False


# ============================================================================
# MAIN EPOCH LOOP
# ============================================================================

async def run_epoch_loop(state: Dict[str, Any]) -> None:
    """
    Main monitoring and submission loop.
    
    Workflow:
    1. Poll blockchain every 6 seconds
    2. When blocks_remaining <= BLOCKS_BEFORE_BOUNDARY (and not yet acted):
       a. Update database (run add_column.py)
       b. Fetch top N molecules
       c. Submit all molecules SEQUENTIALLY with delays
    3. Wait for next epoch's submission window
    4. Repeat
    """
    bt.logging.info("🔄 Starting epoch monitoring loop...\n")

    last_acted_epoch = -1
    last_status_log = datetime.datetime.now()
    num_hotkeys = len(state['wallets'])

    while not shutdown_event.is_set():
        try:
            # Get current blockchain state
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            next_epoch_block = (current_epoch + 1) * state['epoch_length']
            blocks_remaining = next_epoch_block - current_block

            # Periodic status logging
            now = datetime.datetime.now()
            if (now - last_status_log).total_seconds() >= STATUS_LOG_INTERVAL:
                bt.logging.info(
                    f"📊 Status | Block: {current_block} | Epoch: {current_epoch} | "
                    f"Next boundary: {next_epoch_block} | Blocks remaining: {blocks_remaining}"
                )
                last_status_log = now

            # ================================================================
            # SUBMISSION WINDOW: <= 28 blocks before epoch boundary
            # Changed from == to <= to avoid missing the window
            # ================================================================
            if blocks_remaining <= BLOCKS_BEFORE_BOUNDARY and current_epoch != last_acted_epoch:
                
                bt.logging.info("\n" + "="*70)
                bt.logging.info("⏰ SUBMISSION WINDOW REACHED")
                bt.logging.info("="*70)
                bt.logging.info(f"📍 Current block: {current_block}")
                bt.logging.info(f"📍 Current epoch: {current_epoch}")
                bt.logging.info(f"📍 Blocks until boundary: {blocks_remaining}")
                bt.logging.info("="*70 + "\n")

                submission_start_time = datetime.datetime.now()

                # ============================================================
                # STEP 1: Update Database
                # ============================================================
                bt.logging.info("🔹 STEP 1/3: Database Update")
                bt.logging.info("💾 DATABASE UPDATE STARTING")
                
                db_update_success = run_add_column_script()
                
                if not db_update_success:
                    bt.logging.error(
                        "   ⚠️  Database update failed, but continuing with submission...\n"
                    )
                
                bt.logging.info("")

                # ============================================================
                # STEP 2: Fetch Top Molecules
                # ============================================================
                bt.logging.info("🔹 STEP 2/3: Fetching Top Molecules")
                
                top_molecules = get_top_n_molecules_from_db(num_hotkeys)

                if not top_molecules:
                    bt.logging.warning(
                        "   ⚠️  No available molecules found. "
                        "Skipping submission for this epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                bt.logging.info("")

                # ============================================================
                # STEP 3: Submit SEQUENTIALLY (one at a time with delays)
                # ============================================================
                bt.logging.info("🔹 STEP 3/3: Sequential Submission with Delays")
                bt.logging.info(f"   ⚡ Submitting {len(top_molecules)} molecules using {num_hotkeys} hotkeys")
                bt.logging.info(f"   ⏱️  Delay between submissions: {SUBMISSION_DELAY}s")

                # Submit one at a time
                results = []
                submission_details = []
                
                for idx, (molecule_name, score) in enumerate(top_molecules):
                    if idx >= len(state['wallets']):
                        bt.logging.warning(
                            f"   ⚠️  More molecules ({len(top_molecules)}) than hotkeys "
                            f"({num_hotkeys}). Skipping: {molecule_name}"
                        )
                        break

                    wallet = state['wallets'][idx]
                    miner_uid = state['miner_uids'][idx]
                    
                    # Submit this hotkey
                    success = await submit_response(
                        wallet, miner_uid, molecule_name, state,
                        idx + 1, len(top_molecules)
                    )
                    
                    results.append(success)
                    submission_details.append((wallet, molecule_name, miner_uid, score))
                    
                    # Wait before next submission (except for last one)
                    if idx < len(top_molecules) - 1:
                        bt.logging.info(f"\n   ⏳ Waiting {SUBMISSION_DELAY}s before next submission...\n")
                        await asyncio.sleep(SUBMISSION_DELAY)

                # ============================================================
                # Process Results
                # ============================================================
                bt.logging.info("")
                bt.logging.info("="*70)
                bt.logging.info(f"📊 EPOCH {current_epoch} SUBMISSION RESULTS")
                bt.logging.info("="*70)

                success_count = sum(1 for r in results if r)
                failure_count = len(results) - success_count

                submission_elapsed = (datetime.datetime.now() - submission_start_time).total_seconds()

                bt.logging.info(f"✅ Successful: {success_count}/{len(results)}")
                bt.logging.info(f"❌ Failed: {failure_count}/{len(results)}")
                bt.logging.info(f"⏱️  Total time: {submission_elapsed:.2f}s")
                bt.logging.info("="*70)
                
                # Detailed results
                bt.logging.info("\n📋 Detailed Results:")
                for idx, (success, (wallet, molecule_name, miner_uid, score)) in enumerate(zip(results, submission_details), 1):
                    status = "✅" if success else "❌"
                    hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
                    bt.logging.info(
                        f"   {idx}. {status} UID {miner_uid:>3} ({hotkey_name:<12}) | "
                        f"{molecule_name:<30} | Score: {score:.6f}"
                    )

                # Calculate next submission window
                next_submission_epoch = current_epoch + 1
                next_submission_block = (next_submission_epoch + 1) * state['epoch_length'] - BLOCKS_BEFORE_BOUNDARY
                blocks_until_next = next_submission_block - current_block
                time_until_next = blocks_until_next * 12  # ~12 seconds per block

                bt.logging.info("")
                bt.logging.info("="*70)
                bt.logging.info(f"⏭️  Next submission window:")
                bt.logging.info(f"   Epoch: {next_submission_epoch}")
                bt.logging.info(f"   Block: ~{next_submission_block}")
                bt.logging.info(f"   ETA: ~{time_until_next // 60} minutes ({time_until_next} seconds)")
                bt.logging.info("="*70 + "\n")

                # Mark this epoch as handled
                last_acted_epoch = current_epoch

                # Sleep briefly to avoid re-triggering
                await asyncio.sleep(12)
                continue

            # Not at submission window - continue monitoring
            await asyncio.sleep(6)

        except Exception as e:
            bt.logging.error(f"❌ Error in epoch loop: {e}")
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)

    bt.logging.info("\n🛑 Epoch loop terminated by shutdown signal")


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

async def run_miner(config: argparse.Namespace) -> None:
    """Main miner coroutine."""
    try:
        # Setup Bittensor objects
        wallets, subtensor, metagraph, miner_uids, epoch_length = \
            await setup_bittensor_objects(config)

        # Initialize state
        state: Dict[str, Any] = {
            'config': config,
            'github_path': load_github_path(),
            'wallets': wallets,
            'miner_uids': miner_uids,
            'subtensor': subtensor,
            'metagraph': metagraph,
            'epoch_length': epoch_length,
            'bdt': QuicknetBittensorDrandTimelock(),
        }

        # Get challenge targets
        current_block = await subtensor.get_current_block()
        last_boundary = (current_block // epoch_length) * epoch_length
        block_hash = await subtensor.determine_block_hash(last_boundary)
        
        startup_proteins = get_challenge_params_from_blockhash(
            block_hash=block_hash,
            weekly_target=config.weekly_target,
            num_antitargets=config.num_antitargets,
        )
        
        if startup_proteins:
            state['current_challenge_targets'] = startup_proteins["targets"]
            state['current_challenge_antitargets'] = startup_proteins["antitargets"]
            bt.logging.info(f"🎯 Challenge targets: {startup_proteins['targets']}")
            bt.logging.info(f"🚫 Anti-targets: {startup_proteins['antitargets']}\n")

        # Start main loop
        await run_epoch_loop(state)

    except Exception as e:
        bt.logging.error(f"❌ Fatal error in miner: {e}")
        bt.logging.error(traceback.format_exc())
        raise
    finally:
        if 'subtensor' in locals():
            try:
                await subtensor.close()
                bt.logging.info("✅ Subtensor connection closed")
            except:
                pass


def main():
    """Main entry point."""
    load_dotenv()
    
    config = parse_arguments()
    setup_logging(config)

    try:
        asyncio.run(run_miner(config))
    except KeyboardInterrupt:
        bt.logging.info("\n🛑 Miner interrupted by user")
    except Exception as e:
        bt.logging.error(f"❌ Fatal error: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
