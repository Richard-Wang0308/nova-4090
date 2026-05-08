#!/usr/bin/env python3
"""
MULTI-WALLET MULTI-HOTKEY EPOCH-BASED MOLECULE SUBMISSION SCRIPT

Workflow:
1. Monitor blockchain for epoch boundaries
2. At 30 blocks before boundary, update database
3. Fetch top N molecules from score DB + top N nanobodies from nanobodies.sqlite
4. Submit all N molecules SEQUENTIALLY (one at a time with delays)
   Format: {candidate_product}|{candidate_nanobody}
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
DB_PATH            = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
SCORE_RESULTS_DB   = os.path.join(BASE_DIR, "score_results_2.sqlite")
ADD_COLUMN_SCRIPT  = os.path.join(BASE_DIR, "add_column.py")
NANOBODY_DB_PATH   = os.path.join(BASE_DIR, "nanobodies.sqlite")   # ← NEW

# ============================================================================
# WALLET + HOTKEY CONFIGURATION
# Each entry is a (wallet_name, hotkey_name) pair.
# Add as many wallets/hotkeys as needed.
# ============================================================================
WALLET_HOTKEY_PAIRS: List[Tuple[str, str]] = [
    ("nova",   "nota"),
    ("nova",   "notb"),
    ("nova",   "notc"),
    # ("alpha",  "hotkey1"),
    # ("alpha",  "hotkey2"),
    # ("beta",   "hotkey1"),
]
# ============================================================================

# Timing configuration
BLOCKS_BEFORE_BOUNDARY = 30   # Trigger point: 30 blocks before epoch end
EPOCH_LENGTH           = 361  # Blocks per epoch
STATUS_LOG_INTERVAL    = 60   # Log status every N seconds
SUBMISSION_DELAY       = 0.5  # Seconds between each hotkey submission

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

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# CHALLENGE PARAM COMPAT
# ============================================================================

def resolve_challenge_params(config: argparse.Namespace, block_hash: str) -> Optional[Dict[str, Any]]:
    """
    Handle both challenge helper signatures:
    - weekly_target/num_antitargets (validator-style)
    - small_molecule_target/nanobody_target/num_antitargets (challenge.py)
    """
    def cfg_get(key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    small_molecule_targets = cfg_get("small_molecule_target", []) or []
    nanobody_targets       = cfg_get("nanobody_target", []) or []
    target                 = small_molecule_targets[0] if small_molecule_targets else cfg_get("weekly_target")
    nanobody_target        = nanobody_targets[0] if nanobody_targets else target

    try:
        return get_challenge_params_from_blockhash(
            block_hash=block_hash,
            weekly_target=target,
            num_antitargets=cfg_get("num_antitargets", 0),
            include_reaction=cfg_get("random_valid_reaction", True),
        )
    except TypeError:
        return get_challenge_params_from_blockhash(
            block_hash=block_hash,
            small_molecule_target=target,
            nanobody_target=nanobody_target,
            num_antitargets=cfg_get("num_antitargets", 0),
            include_reaction=cfg_get("random_valid_reaction", True),
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

    unique_wallets = sorted(set(w for w, _ in WALLET_HOTKEY_PAIRS))

    bt.logging.info("\n" + "="*70)
    bt.logging.info("🚀 MULTI-WALLET MULTI-HOTKEY EPOCH MINER STARTING")
    bt.logging.info("="*70)
    bt.logging.info(f"📡 Network:  {config.network}")
    bt.logging.info(f"🔗 Netuid:   {config.netuid}")
    bt.logging.info(f"💼 Unique wallets: {len(unique_wallets)}  →  {unique_wallets}")
    bt.logging.info(f"👥 Total wallet/hotkey pairs: {len(WALLET_HOTKEY_PAIRS)}")
    for idx, (wname, hname) in enumerate(WALLET_HOTKEY_PAIRS, 1):
        bt.logging.info(f"   {idx:>2}. wallet={wname:<12}  hotkey={hname}")
    bt.logging.info(f"⏰ Trigger point: {BLOCKS_BEFORE_BOUNDARY} blocks before epoch boundary")
    bt.logging.info(f"📊 Epoch length: {EPOCH_LENGTH} blocks")
    bt.logging.info(f"⏱️  Submission delay: {SUBMISSION_DELAY}s between hotkeys")
    bt.logging.info(f"🧬 Nanobody DB: {NANOBODY_DB_PATH}")
    bt.logging.info("="*70 + "\n")


# ============================================================================
# BITTENSOR SETUP
# ============================================================================

async def setup_bittensor_objects(
    config: argparse.Namespace
) -> Tuple[List[Any], Any, Any, List[int], int]:
    """
    Initializes multiple wallets, subtensor, and metagraph.

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

                bt.logging.info(
                    f"   📋 Initializing {len(WALLET_HOTKEY_PAIRS)} wallet/hotkey pairs:"
                )
                wallets: List[Any] = []
                miner_uids: List[int] = []

                for idx, (wallet_name, hotkey_name) in enumerate(WALLET_HOTKEY_PAIRS, 1):
                    label = f"{wallet_name}/{hotkey_name}"
                    try:
                        wallet    = bt.wallet(name=wallet_name, hotkey=hotkey_name)
                        _         = wallet.hotkey  # raises if key file missing
                        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

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
    Resolve per-reaction score DB path for rxn:1 .. rxn:5 only.
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

    Returns:
        List of (molecule_name, score) tuples, ordered by score DESC
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB

    if not os.path.exists(db_path):
        bt.logging.error(f"   ❌ Database not found: {db_path}")
        return []

    try:
        conn   = sqlite3.connect(db_path)
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


def get_top_n_nanobodies_from_db(
    n: int,
    db_path: str = None,
) -> List[Tuple[str, float]]:
    """
    Fetch top N nanobodies from nanobodies.sqlite ordered by score DESC.

    Expects a table named `nanobodies` with at least:
        - nanobody_name  TEXT
        - score          REAL

    Returns:
        List of (nanobody_name, score) tuples, ordered by score DESC.
        Falls back to an empty list on any error.
    """
    if db_path is None:
        db_path = NANOBODY_DB_PATH

    if not os.path.exists(db_path):
        bt.logging.error(f"   ❌ Nanobody database not found: {db_path}")
        return []

    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ---------------------------------------------------------------------------
        # Auto-detect the score column name (score / docking_score / binding_score …)
        # ---------------------------------------------------------------------------
        cursor.execute("PRAGMA table_info(nanobodies)")
        columns = [row[1] for row in cursor.fetchall()]

        score_col = None
        for candidate in ("score", "docking_score", "binding_score", "value"):
            if candidate in columns:
                score_col = candidate
                break

        if score_col is None:
            bt.logging.error(
                f"   ❌ Could not find a score column in nanobodies table. "
                f"Available columns: {columns}"
            )
            conn.close()
            return []

        # ---------------------------------------------------------------------------
        # Auto-detect the name column
        # ---------------------------------------------------------------------------
        name_col = None
        for candidate in ("nanobody_name", "name", "id", "nanobody_id"):
            if candidate in columns:
                name_col = candidate
                break

        if name_col is None:
            bt.logging.error(
                f"   ❌ Could not find a name column in nanobodies table. "
                f"Available columns: {columns}"
            )
            conn.close()
            return []

        cursor.execute(
            f"""
            SELECT {name_col}, {score_col}
            FROM   nanobodies
            ORDER  BY {score_col} DESC
            LIMIT  ?
            """,
            (n,),
        )

        rows = cursor.fetchall()
        conn.close()

        if rows:
            bt.logging.info(f"   ✅ Retrieved {len(rows)} nanobodies:")
            for idx, (nb_name, nb_score) in enumerate(rows, 1):
                bt.logging.info(f"      {idx}. {nb_name:<30} | Score: {nb_score:.6f}")
            return rows
        else:
            bt.logging.warning("   ⚠️  No nanobodies found in database")
            return []

    except sqlite3.Error as e:
        bt.logging.error(f"   ❌ Nanobody database error: {e}")
        return []
    except Exception as e:
        bt.logging.error(f"   ❌ Error querying nanobody database: {e}")
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
    candidate_nanobody: str,
    state: Dict[str, Any],
    submission_number: int,
    total_submissions: int,
) -> bool:
    """
    Encrypt and submit a molecule+nanobody pair using the specified wallet/hotkey.

    Submission format: {candidate_product}|{candidate_nanobody}

    Args:
        wallet:             Bittensor wallet object
        miner_uid:          Miner UID
        candidate_product:  Molecule name to submit
        candidate_nanobody: Nanobody name to submit
        state:              Global state dictionary
        submission_number:  Current submission number (for logging)
        total_submissions:  Total number of submissions (for logging)

    Returns:
        True if submission successful, False otherwise
    """
    if not candidate_product:
        bt.logging.warning(f"      ⚠️  UID {miner_uid}: No candidate product")
        return False

    if not candidate_nanobody:
        bt.logging.warning(f"      ⚠️  UID {miner_uid}: No candidate nanobody")
        return False

    wallet_name = wallet.name        if hasattr(wallet, 'name')       else 'unknown'
    hotkey_name = wallet.hotkey_str  if hasattr(wallet, 'hotkey_str') else 'unknown'
    label       = f"{wallet_name}/{hotkey_name}"

    bt.logging.info(
        f"\n   [{submission_number}/{total_submissions}] 📤 SUBMITTING: "
        f"UID {miner_uid} ({label})"
    )
    bt.logging.info(f"      Molecule:  {candidate_product}")
    bt.logging.info(f"      Nanobody:  {candidate_nanobody}")

    try:
        # Get current block
        current_block = await state['subtensor'].get_current_block()
        bt.logging.info(f"      Current block: {current_block}")

        # Build message in new format
        message = f"{candidate_product}|{candidate_nanobody}"
        bt.logging.info(f"      📝 Message: {message}")

        # Encrypt response
        bt.logging.info(f"      🔐 Encrypting response...")
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

            content_str     = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            # Generate filename hash
            filename       = hashlib.sha256(content_str.encode()).hexdigest()[:20]
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
# MAIN EPOCH LOOP
# ============================================================================

async def run_epoch_loop(state: Dict[str, Any]) -> None:
    """
    Main monitoring and submission loop.

    Workflow:
    1. Poll blockchain every 6 seconds
    2. When blocks_remaining <= BLOCKS_BEFORE_BOUNDARY (and not yet acted):
       a. Determine allowed reaction
       b. Update matching reaction database(s)
       c. Fetch top N molecules  (from score DB)
       d. Fetch top N nanobodies (from nanobodies.sqlite)
       e. Pair molecule[i] with nanobody[i] and submit SEQUENTIALLY
          Format: {candidate_product}|{candidate_nanobody}
    3. Wait for next epoch's submission window
    4. Repeat
    """
    bt.logging.info("🔄 Starting epoch monitoring loop...\n")

    last_acted_epoch = -1
    last_status_log  = datetime.datetime.now()
    num_pairs        = len(state['wallets'])

    while not shutdown_event.is_set():
        try:
            # Get current blockchain state
            current_block    = await state['subtensor'].get_current_block()
            current_epoch    = current_block // state['epoch_length']
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

            # ==============================================================
            # SUBMISSION WINDOW
            # ==============================================================
            if blocks_remaining <= BLOCKS_BEFORE_BOUNDARY and current_epoch != last_acted_epoch:

                bt.logging.info("\n" + "="*70)
                bt.logging.info("⏰ SUBMISSION WINDOW REACHED")
                bt.logging.info("="*70)
                bt.logging.info(f"📍 Current block:          {current_block}")
                bt.logging.info(f"📍 Current epoch:          {current_epoch}")
                bt.logging.info(f"📍 Blocks until boundary:  {blocks_remaining}")
                bt.logging.info("="*70 + "\n")

                submission_start_time = datetime.datetime.now()

                # ============================================================
                # STEP 1: Determine allowed reaction
                # ============================================================
                bt.logging.info("🔹 STEP 1/5: Determine Allowed Reaction")
                cfg         = state["config"]
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
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                if not allowed_reaction:
                    bt.logging.warning(
                        f"   ⚠️  No allowed reaction for epoch {current_epoch}. "
                        "Skipping submission for this epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                reaction_db_path = get_reaction_score_db_path(allowed_reaction)
                if not reaction_db_path:
                    bt.logging.warning(
                        f"   ⚠️  Invalid allowed reaction (need rxn:1..rxn:5): "
                        f"{allowed_reaction}. Skipping epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                bt.logging.info(
                    f"   🎯 Allowed reaction for epoch {current_epoch}: {allowed_reaction}"
                )

                # ============================================================
                # STEP 2: Update matching database
                # ============================================================
                bt.logging.info("🔹 STEP 2/5: Database Update")

                if not os.path.exists(reaction_db_path):
                    bt.logging.warning(
                        f"   ⚠️  Missing DB for {allowed_reaction}: {reaction_db_path}. "
                        "Skipping submission for this epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                db_update_success = run_add_column_script(reaction_db_path)
                if not db_update_success:
                    bt.logging.error(
                        f"   ⚠️  Database update failed for {reaction_db_path}. "
                        "Skipping submission for this epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                bt.logging.info("")

                # ============================================================
                # STEP 3: Fetch top N molecules
                # ============================================================
                bt.logging.info("🔹 STEP 3/5: Fetching Top Molecules")
                bt.logging.info(f"   🗄️  Score DB: {reaction_db_path}")

                top_molecules = get_top_n_molecules_from_db(
                    n=num_pairs,
                    db_path=reaction_db_path,
                )

                if not top_molecules:
                    bt.logging.warning(
                        f"   ⚠️  No available molecules found for {allowed_reaction}. "
                        "Skipping submission for this epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                bt.logging.info("")

                # ============================================================
                # STEP 4: Fetch top N nanobodies
                # ============================================================
                bt.logging.info("🔹 STEP 4/5: Fetching Top Nanobodies")
                bt.logging.info(f"   🗄️  Nanobody DB: {NANOBODY_DB_PATH}")

                top_nanobodies = get_top_n_nanobodies_from_db(n=num_pairs)

                if not top_nanobodies:
                    bt.logging.warning(
                        "   ⚠️  No nanobodies found in nanobodies.sqlite. "
                        "Skipping submission for this epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                bt.logging.info("")

                # ============================================================
                # STEP 5: Submit SEQUENTIALLY — molecule[i] + nanobody[i]
                # ============================================================
                bt.logging.info("🔹 STEP 5/5: Sequential Submission with Delays")

                # How many submissions can we actually make?
                n_submissions = min(len(top_molecules), len(top_nanobodies), num_pairs)

                bt.logging.info(
                    f"   ⚡ Submitting {n_submissions} pairs "
                    f"(molecules={len(top_molecules)}, "
                    f"nanobodies={len(top_nanobodies)}, "
                    f"wallets={num_pairs})"
                )
                bt.logging.info(f"   ⏱️  Delay between submissions: {SUBMISSION_DELAY}s")

                results            = []
                submission_details = []

                for idx in range(n_submissions):
                    molecule_name, mol_score = top_molecules[idx]
                    nanobody_name, nb_score  = top_nanobodies[idx]
                    wallet                   = state['wallets'][idx]
                    miner_uid                = state['miner_uids'][idx]

                    success = await submit_response(
                        wallet=wallet,
                        miner_uid=miner_uid,
                        candidate_product=molecule_name,
                        candidate_nanobody=nanobody_name,
                        state=state,
                        submission_number=idx + 1,
                        total_submissions=n_submissions,
                    )

                    results.append(success)
                    submission_details.append(
                        (wallet, molecule_name, nanobody_name, miner_uid, mol_score, nb_score)
                    )

                    # Wait before next submission (except for last one)
                    if idx < n_submissions - 1:
                        bt.logging.info(
                            f"\n   ⏳ Waiting {SUBMISSION_DELAY}s before next submission...\n"
                        )
                        await asyncio.sleep(SUBMISSION_DELAY)

                # ============================================================
                # Results summary
                # ============================================================
                bt.logging.info("")
                bt.logging.info("="*70)
                bt.logging.info(f"📊 EPOCH {current_epoch} SUBMISSION RESULTS")
                bt.logging.info("="*70)

                success_count      = sum(1 for r in results if r)
                failure_count      = len(results) - success_count
                submission_elapsed = (
                    datetime.datetime.now() - submission_start_time
                ).total_seconds()

                bt.logging.info(f"✅ Successful: {success_count}/{len(results)}")
                bt.logging.info(f"❌ Failed:     {failure_count}/{len(results)}")
                bt.logging.info(f"⏱️  Total time: {submission_elapsed:.2f}s")
                bt.logging.info("="*70)

                bt.logging.info("\n📋 Detailed Results:")
                for i, (success, (wallet, mol_name, nb_name, miner_uid, mol_score, nb_score)) in enumerate(
                    zip(results, submission_details), 1
                ):
                    status      = "✅" if success else "❌"
                    wallet_name = wallet.name       if hasattr(wallet, 'name')       else 'unknown'
                    hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
                    label       = f"{wallet_name}/{hotkey_name}"
                    bt.logging.info(
                        f"   {i:>2}. {status} UID {miner_uid:>3} ({label:<22}) | "
                        f"{mol_name:<28} (score={mol_score:.4f}) | "
                        f"{nb_name:<28} (score={nb_score:.4f})"
                    )

                # Next submission window ETA
                next_submission_epoch = current_epoch + 1
                next_submission_block = (
                    (next_submission_epoch + 1) * state['epoch_length'] - BLOCKS_BEFORE_BOUNDARY
                )
                blocks_until_next = next_submission_block - current_block
                time_until_next   = blocks_until_next * 12  # ~12 seconds per block

                bt.logging.info("")
                bt.logging.info("="*70)
                bt.logging.info(f"⏭️  Next submission window:")
                bt.logging.info(f"   Epoch: {next_submission_epoch}")
                bt.logging.info(f"   Block: ~{next_submission_block}")
                bt.logging.info(
                    f"   ETA:   ~{time_until_next // 60} minutes ({time_until_next} seconds)"
                )
                bt.logging.info("="*70 + "\n")

                last_acted_epoch = current_epoch
                await asyncio.sleep(12)
                continue

            # Not at submission window — keep polling
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
        wallets, subtensor, metagraph, miner_uids, epoch_length = \
            await setup_bittensor_objects(config)

        state: Dict[str, Any] = {
            'config':       config,
            'github_path':  load_github_path(),
            'wallets':      wallets,
            'miner_uids':   miner_uids,
            'subtensor':    subtensor,
            'metagraph':    metagraph,
            'epoch_length': epoch_length,
            'bdt':          QuicknetBittensorDrandTimelock(),
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