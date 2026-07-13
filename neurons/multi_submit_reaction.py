#!/usr/bin/env python3
"""
MULTI-WALLET MULTI-HOTKEY EPOCH-BASED MOLECULE SUBMISSION SCRIPT

Workflow:
1. On startup, immediately determine current epoch's allowed reaction and submit.
2. Monitor blockchain for epoch boundaries.
3. As soon as the epoch counter changes, determine that epoch's allowed reaction,
   update the matching database, fetch top-N molecules, and submit immediately.
4. Repeat.
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
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results_1.sqlite")
ADD_COLUMN_SCRIPT = os.path.join(BASE_DIR, "add_column.py")

# ============================================================================
# WALLET + HOTKEY CONFIGURATION
# Each entry is a (wallet_name, hotkey_name) pair.
# Add as many wallets/hotkeys as needed.
# ============================================================================
WALLET_HOTKEY_PAIRS: List[Tuple[str, str]] = [
    ("nova",   "notc"),
    ("nova",   "notd")
    # ("alpha",  "hotkey1"),
    # ("beta",   "hotkey1"),
]
# ============================================================================

# Timing configuration
# NOTE: Submission triggers:
#   (a) IMMEDIATELY on script startup (first loop iteration), and
#   (b) IMMEDIATELY whenever the epoch counter changes thereafter.
EPOCH_LENGTH = 361           # Blocks per epoch
STATUS_LOG_INTERVAL = 60     # Log status every N seconds
SUBMISSION_DELAY = 0.5       # Seconds between each hotkey submission
POLL_INTERVAL = 6            # Seconds between block polls

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
# CHALLENGE PARAM COMPAT
# ============================================================================

def resolve_challenge_params(config: argparse.Namespace, block_hash: str) -> Optional[Dict[str, Any]]:
    """
    Resolve epoch challenge params using utils.challenge.get_challenge_params_from_blockhash.
    """
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
# ARGUMENT PARSING & LOGGING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-wallet multi-hotkey epoch-based molecule submission miner"
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

    # Use first wallet name for logging path (representative)
    primary_wallet_name = WALLET_HOTKEY_PAIRS[0][0] if WALLET_HOTKEY_PAIRS else "multi_wallet"

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            primary_wallet_name,
            "multi_wallet_hotkey",
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

    # Build a summary of unique wallets and total pairs
    unique_wallets = sorted(set(w for w, _ in WALLET_HOTKEY_PAIRS))

    bt.logging.info("\n" + "="*70)
    bt.logging.info("🚀 MULTI-WALLET MULTI-HOTKEY EPOCH MINER STARTING")
    bt.logging.info("="*70)
    bt.logging.info(f"📡 Network: {config.network}")
    bt.logging.info(f"🔗 Netuid: {config.netuid}")
    bt.logging.info(f"💼 Unique wallets: {len(unique_wallets)}  →  {unique_wallets}")
    bt.logging.info(f"👥 Total wallet/hotkey pairs: {len(WALLET_HOTKEY_PAIRS)}")
    for idx, (wname, hname) in enumerate(WALLET_HOTKEY_PAIRS, 1):
        bt.logging.info(f"   {idx:>2}. wallet={wname:<12}  hotkey={hname}")
    bt.logging.info(f"⏰ Trigger: IMMEDIATELY on startup, then on every epoch change")
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
    Initializes multiple wallets (potentially different wallet names and hotkeys),
    subtensor, and metagraph.

    Returns:
        (wallets_list, subtensor, metagraph, miner_uids_list, epoch_length)
    """
    bt.logging.info("🔧 Setting up Bittensor objects with multiple wallets/hotkeys...")

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

                # Create wallet objects for each (wallet_name, hotkey_name) pair
                bt.logging.info(
                    f"   📋 Initializing {len(WALLET_HOTKEY_PAIRS)} wallet/hotkey pairs:"
                )
                wallets: List[Any] = []
                miner_uids: List[int] = []

                for idx, (wallet_name, hotkey_name) in enumerate(WALLET_HOTKEY_PAIRS, 1):
                    label = f"{wallet_name}/{hotkey_name}"
                    try:
                        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)

                        # Verify the hotkey exists on disk before querying metagraph
                        _ = wallet.hotkey  # raises if key file missing

                        miner_uid = metagraph.hotkeys.index(
                            wallet.hotkey.ss58_address
                        )

                        wallets.append(wallet)
                        miner_uids.append(miner_uid)

                        bt.logging.info(
                            f"      {idx:>2}. ✅ {label:<22} → UID {miner_uid:>3} "
                            f"({wallet.hotkey.ss58_address[:10]}...)"
                        )

                    except ValueError:
                        bt.logging.warning(
                            f"      {idx:>2}. ⚠️  {label:<22} → NOT FOUND in metagraph (skipping)"
                        )
                        continue
                    except FileNotFoundError:
                        bt.logging.warning(
                            f"      {idx:>2}. ⚠️  {label:<22} → Hotkey file not found on disk (skipping)"
                        )
                        continue
                    except Exception as e:
                        bt.logging.error(
                            f"      {idx:>2}. ❌ {label:<22} → ERROR: {e}"
                        )
                        continue

                if not wallets:
                    raise ValueError(
                        "❌ No valid wallet/hotkey pairs found in metagraph! "
                        "Please check your WALLET_HOTKEY_PAIRS configuration."
                    )

                bt.logging.info(
                    f"\n   ✅ Successfully initialized "
                    f"{len(wallets)}/{len(WALLET_HOTKEY_PAIRS)} wallet/hotkey pairs\n"
                )

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

def get_reaction_score_db_path(allowed_reaction: str) -> Optional[str]:
    """
    Resolve per-reaction score DB path for rxn:1 .. rxn:5 only
    (e.g. rxn:2 -> score_results_2.sqlite).
    """
    if not allowed_reaction:
        return None

    if str(allowed_reaction).lower() == "savi":
        return None

    if not allowed_reaction.startswith("rxn:"):
        bt.logging.warning(
            f"   ⚠️  Unsupported allowed reaction format for per-reaction DB: {allowed_reaction}"
        )
        return None

    try:
        reaction_id = int(allowed_reaction.split(":", 1)[1])
    except (ValueError, IndexError):
        bt.logging.warning(
            f"   ⚠️  Could not parse reaction id from allowed reaction: {allowed_reaction}"
        )
        return None

    if reaction_id not in (1, 2, 3, 4, 5):
        bt.logging.warning(
            f"   ⚠️  Allowed reaction must be rxn:1..rxn:5, got: {allowed_reaction}"
        )
        return None

    return os.path.join(BASE_DIR, f"score_results_{reaction_id}.sqlite")


def get_top_n_molecules_from_db(
    n: int,
    db_path: str = None,
) -> List[Tuple[str, float]]:
    """
    Fetch top N available molecules from a score database.

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
            (n,),
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


def run_add_column_script(db_path: str) -> bool:
    """
    Execute add_column.py --skip-fix for a specific score database.

    Returns:
        True if successful, False otherwise
    """
    try:
        bt.logging.info(
            f"   💾 Running: python3 {ADD_COLUMN_SCRIPT} --skip-fix --db-path {db_path}"
        )

        start_time = datetime.datetime.now()

        result = subprocess.run(
            ["python3", ADD_COLUMN_SCRIPT, "--skip-fix", "--db-path", db_path],
            capture_output=True,
            text=True,
            timeout=120,
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

    wallet_name = wallet.name if hasattr(wallet, 'name') else 'unknown'
    hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
    label = f"{wallet_name}/{hotkey_name}"

    bt.logging.info(
        f"\n   [{submission_number}/{total_submissions}] 📤 SUBMITTING: "
        f"UID {miner_uid} ({label})"
    )
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
                        f"      ❌ SUBMISSION FAILED for UID {miner_uid} ({label}): "
                        f"Blockchain commitment returned False"
                    )
                    return False

            except MetadataError as e:
                bt.logging.warning(
                    f"      ⏳ MetadataError for UID {miner_uid} ({label}): {e}"
                )
                bt.logging.warning(
                    f"      ⏳ Too soon to commit again (rate limited)"
                )
                bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid} ({label})")
                return False

            # Upload to GitHub
            bt.logging.info(f"      📤 Uploading to GitHub...")
            try:
                github_status = upload_file_to_github(filename, encoded_content)

                if github_status:
                    bt.logging.info(
                        f"      ✅ SUBMISSION SUCCESSFUL for UID {miner_uid} ({label})"
                    )
                    return True
                else:
                    bt.logging.error(
                        f"      ❌ GitHub upload failed for UID {miner_uid} ({label})"
                    )
                    bt.logging.error(
                        f"      ❌ SUBMISSION FAILED for UID {miner_uid} ({label})"
                    )
                    return False

            except Exception as e:
                bt.logging.error(
                    f"      ❌ GitHub upload error for UID {miner_uid} ({label}): {e}"
                )
                bt.logging.error(
                    f"      ❌ SUBMISSION FAILED for UID {miner_uid} ({label})"
                )
                return False

    except Exception as e:
        bt.logging.error(
            f"      ❌ Submission error for UID {miner_uid} ({label}): {e}"
        )
        bt.logging.error(traceback.format_exc())
        bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid} ({label})")
        return False


# ============================================================================
# SUBMISSION FOR ONE EPOCH (used both on startup and on epoch change)
# ============================================================================

async def do_epoch_submission(state: Dict[str, Any], current_epoch: int) -> None:
    """
    Full pipeline for a given epoch:
      1. Determine that epoch's allowed reaction
      2. Update matching database
      3. Fetch top-N molecules
      4. Submit sequentially with delays
    """
    num_pairs = len(state['wallets'])
    submission_start_time = datetime.datetime.now()

    current_block = await state["subtensor"].get_current_block()
    bt.logging.info("\n" + "="*70)
    bt.logging.info(f"⏰ SUBMITTING FOR EPOCH {current_epoch}")
    bt.logging.info("="*70)
    bt.logging.info(f"📍 Current block: {current_block}")
    bt.logging.info(f"📍 Current epoch: {current_epoch}")
    bt.logging.info("="*70 + "\n")

    # ==========================================================
    # STEP 1: Determine allowed reaction
    # ==========================================================
    bt.logging.info("🔹 STEP 1/4: Determine Allowed Reaction")
    cfg = state["config"]
    start_block = current_epoch * state["epoch_length"]
    try:
        start_block_hash = await state["subtensor"].determine_block_hash(start_block)
        challenge_params = resolve_challenge_params(cfg, start_block_hash)
        allowed_reaction = (
            challenge_params.get("allowed_reaction") if challenge_params else None
        )
    except Exception as e:
        bt.logging.error(
            f"   ❌ Failed to get challenge params for epoch {current_epoch}: {e}"
        )
        return

    if not allowed_reaction:
        bt.logging.warning(
            f"   ⚠️  No allowed reaction for epoch {current_epoch} "
            f"(random_valid_reaction={cfg.random_valid_reaction}). "
            "Skipping submission for this epoch.\n"
        )
        return

    reaction_db_path = get_reaction_score_db_path(allowed_reaction)
    if not reaction_db_path:
        bt.logging.warning(
            f"   ⚠️  Invalid allowed reaction for this miner "
            f"(need rxn:1..rxn:5): {allowed_reaction}. Skipping epoch.\n"
        )
        return

    bt.logging.info(
        f"   🎯 Allowed reaction for epoch {current_epoch}: {allowed_reaction}"
    )

    # ==========================================================
    # STEP 2: Update matching database(s)
    # ==========================================================
    bt.logging.info("🔹 STEP 2/4: Database Update")
    bt.logging.info("💾 DATABASE UPDATE STARTING")

    if not os.path.exists(reaction_db_path):
        bt.logging.warning(
            f"   ⚠️  Missing DB for {allowed_reaction}: {reaction_db_path}. "
            "Skipping submission for this epoch.\n"
        )
        return

    db_update_success = run_add_column_script(reaction_db_path)
    if not db_update_success:
        bt.logging.error(
            f"   ⚠️  Database update failed for {reaction_db_path}. "
            "Skipping submission for this epoch.\n"
        )
        return

    bt.logging.info("")

    # ==========================================================
    # STEP 3: Fetch top molecules for allowed reaction
    # ==========================================================
    bt.logging.info("🔹 STEP 3/4: Fetching Top Molecules for Allowed Reaction")
    bt.logging.info(f"   🗄️  Using score DB: {reaction_db_path}")

    top_molecules = get_top_n_molecules_from_db(
        n=num_pairs,
        db_path=reaction_db_path,
    )

    if not top_molecules:
        bt.logging.warning(
            f"   ⚠️  No available molecules found for {allowed_reaction}. "
            "Skipping submission for this epoch.\n"
        )
        return

    bt.logging.info("")

    # ==========================================================
    # STEP 4: Submit SEQUENTIALLY (one at a time with delays)
    # ==========================================================
    bt.logging.info("🔹 STEP 4/4: Sequential Submission with Delays")
    bt.logging.info(
        f"   ⚡ Submitting {len(top_molecules)} molecules "
        f"using {num_pairs} wallet/hotkey pairs"
    )
    bt.logging.info(f"   ⏱️  Delay between submissions: {SUBMISSION_DELAY}s")

    results = []
    submission_details = []

    for idx, (molecule_name, score) in enumerate(top_molecules):
        if idx >= len(state['wallets']):
            bt.logging.warning(
                f"   ⚠️  More molecules ({len(top_molecules)}) than "
                f"wallet/hotkey pairs ({num_pairs}). Skipping: {molecule_name}"
            )
            break

        wallet   = state['wallets'][idx]
        miner_uid = state['miner_uids'][idx]

        success = await submit_response(
            wallet, miner_uid, molecule_name, state,
            idx + 1, len(top_molecules)
        )

        results.append(success)
        submission_details.append((wallet, molecule_name, miner_uid, score))

        # Wait before next submission (except for last one)
        if idx < len(top_molecules) - 1:
            bt.logging.info(
                f"\n   ⏳ Waiting {SUBMISSION_DELAY}s before next submission...\n"
            )
            await asyncio.sleep(SUBMISSION_DELAY)

    # ==========================================================
    # Process Results
    # ==========================================================
    bt.logging.info("")
    bt.logging.info("="*70)
    bt.logging.info(f"📊 EPOCH {current_epoch} SUBMISSION RESULTS")
    bt.logging.info("="*70)

    success_count = sum(1 for r in results if r)
    failure_count = len(results) - success_count
    submission_elapsed = (
        datetime.datetime.now() - submission_start_time
    ).total_seconds()

    bt.logging.info(f"✅ Successful: {success_count}/{len(results)}")
    bt.logging.info(f"❌ Failed:     {failure_count}/{len(results)}")
    bt.logging.info(f"⏱️  Total time: {submission_elapsed:.2f}s")
    bt.logging.info("="*70)

    # Detailed results
    bt.logging.info("\n📋 Detailed Results:")
    for idx, (success, (wallet, molecule_name, miner_uid, score)) in enumerate(
        zip(results, submission_details), 1
    ):
        status      = "✅" if success else "❌"
        wallet_name = wallet.name if hasattr(wallet, 'name') else 'unknown'
        hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
        label       = f"{wallet_name}/{hotkey_name}"
        bt.logging.info(
            f"   {idx:>2}. {status} UID {miner_uid:>3} ({label:<22}) | "
            f"{molecule_name:<30} | Score: {score:.6f}"
        )
    bt.logging.info("")


# ============================================================================
# MAIN EPOCH LOOP
# ============================================================================

async def run_epoch_loop(state: Dict[str, Any]) -> None:
    """
    Main monitoring and submission loop.

    Workflow:
    1. On the very first iteration, submit immediately for whatever the
       current epoch is (last_acted_epoch starts at None, so it never
       matches the current epoch on the first check).
    2. Poll blockchain every POLL_INTERVAL seconds.
    3. As soon as current_epoch != last_acted_epoch, submit immediately
       for the new epoch.
    4. Repeat.
    """
    bt.logging.info("🔄 Starting epoch monitoring loop...\n")

    last_acted_epoch = None   # guarantees first check always triggers
    last_status_log = datetime.datetime.now()

    while not shutdown_event.is_set():
        try:
            # Get current blockchain state
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            epoch_start_block = current_epoch * state['epoch_length']
            blocks_into_epoch = current_block - epoch_start_block

            # Periodic status logging
            now = datetime.datetime.now()
            if (now - last_status_log).total_seconds() >= STATUS_LOG_INTERVAL:
                bt.logging.info(
                    f"📊 Status | Block: {current_block} | Epoch: {current_epoch} | "
                    f"Blocks into epoch: {blocks_into_epoch} | Last acted epoch: {last_acted_epoch}"
                )
                last_status_log = now

            # ==============================================================
            # TRIGGER: first run (last_acted_epoch is None) OR epoch changed
            # ==============================================================
            if current_epoch != last_acted_epoch:
                if last_acted_epoch is None:
                    bt.logging.info(
                        f"🟢 First run detected — submitting immediately "
                        f"for current epoch {current_epoch}\n"
                    )
                else:
                    bt.logging.info(
                        f"🟢 Epoch changed ({last_acted_epoch} → {current_epoch}) "
                        f"— submitting immediately\n"
                    )

                await do_epoch_submission(state, current_epoch)

                last_acted_epoch = current_epoch
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # No epoch change - just keep polling
            await asyncio.sleep(POLL_INTERVAL)

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
        wallets, subtensor, metagraph, miner_uids, epoch_length = \
            await setup_bittensor_objects(config)

        state: Dict[str, Any] = {
            'config':      config,
            'github_path': load_github_path(),
            'wallets':     wallets,
            'miner_uids':  miner_uids,
            'subtensor':   subtensor,
            'metagraph':   metagraph,
            'epoch_length': epoch_length,
            'bdt':         QuicknetBittensorDrandTimelock(),
        }

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
            except Exception:
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