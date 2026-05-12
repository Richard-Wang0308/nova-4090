#!/usr/bin/env python3
"""
MULTI-WALLET MULTI-HOTKEY EPOCH-BASED MOLECULE + NANOBODY SUBMISSION SCRIPT

Workflow:
1. Monitor blockchain for epoch boundaries
2. At 28 blocks before boundary, update database
3. Fetch top N molecules from reaction DB
4. Fetch top N nanobodies DIRECTLY from local nanobody SQLite DB (no Flask)
5. Submit all N pairs SEQUENTIALLY as "molecule|nanobody"
   - If nanobody DB is unavailable → fallback to "molecule|~"
6. Wait for next epoch's submission window
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
DB_PATH           = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
SCORE_RESULTS_DB  = os.path.join(BASE_DIR, "score_results_2.sqlite")
ADD_COLUMN_SCRIPT = os.path.join(BASE_DIR, "add_column.py")

# ============================================================================
# NANOBODY DB CONFIGURATION
# Direct SQLite access — no Flask server needed
# ============================================================================
NANOBODY_DB_PATH  = os.getenv(
    "NANOBODY_DB_PATH",
    "/root/workspace/nova-4090/nanobodies.sqlite"
)
NANOBODY_FALLBACK = "~"

# ============================================================================
# WALLET + HOTKEY CONFIGURATION
# ============================================================================
WALLET_HOTKEY_PAIRS: List[Tuple[str, str]] = [
    ("nova",   "notd"),
    ("gentle",   "hota"),
    ("gentle",   "hotb")
]

# Timing configuration
BLOCKS_BEFORE_BOUNDARY = 30
EPOCH_LENGTH           = 361
STATUS_LOG_INTERVAL    = 60
SUBMISSION_DELAY       = 0.5

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
    bt.logging.info(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# CHALLENGE PARAM COMPAT
# ============================================================================

def resolve_challenge_params(
    config: argparse.Namespace, block_hash: str
) -> Optional[Dict[str, Any]]:
    try:
        return get_challenge_params_from_blockhash(
            block_hash=block_hash,
            weekly_target=config.weekly_target,
            num_antitargets=config.num_antitargets,
            include_reaction=config.random_valid_reaction,
        )
    except TypeError:
        return get_challenge_params_from_blockhash(
            block_hash=block_hash,
            small_molecule_target=config.weekly_target,
            nanobody_target=config.weekly_target,
            num_antitargets=config.num_antitargets,
            include_reaction=config.random_valid_reaction,
        )


# ============================================================================
# ARGUMENT PARSING & LOGGING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-wallet multi-hotkey epoch-based molecule + nanobody submission miner"
    )
    parser.add_argument(
        "--network",
        default=os.getenv("SUBTENSOR_NETWORK", "finney"),
        help="Bittensor network to use",
    )
    parser.add_argument("--netuid", type=int, default=68, help="The chain subnet uid")
    parser.add_argument(
        "--nanobody-db-path",
        default=NANOBODY_DB_PATH,
        dest="nanobody_db_path",
        help="Path to nanobody SQLite database (e.g. /root/workspace/nanobodies.sqlite)",
    )
    parser.add_argument(
        "--nanobody-target",
        default=os.getenv("NANOBODY_TARGET", ""),
        dest="nanobody_target",
        help="Nanobody target protein ID (e.g. Q9NZQ7). "
             "If empty, derived from challenge params.",
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
            "miner",
        )
    )
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    github_repo_name   = os.environ.get("GITHUB_REPO_NAME")
    github_repo_branch = os.environ.get("GITHUB_REPO_BRANCH")
    github_repo_owner  = os.environ.get("GITHUB_REPO_OWNER")
    github_repo_path   = os.environ.get("GITHUB_REPO_PATH", "")

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
        raise ValueError(
            f"GitHub path too long (max 100 chars): {len(github_path)} chars"
        )
    return github_path


def setup_logging(config: argparse.Namespace) -> None:
    bt.logging(config=config, logging_dir=config.full_path)
    unique_wallets = sorted(set(w for w, _ in WALLET_HOTKEY_PAIRS))

    bt.logging.info("\n" + "=" * 70)
    bt.logging.info("🚀 MULTI-WALLET MULTI-HOTKEY EPOCH MINER STARTING")
    bt.logging.info("=" * 70)
    bt.logging.info(f"📡 Network:              {config.network}")
    bt.logging.info(f"🔗 Netuid:               {config.netuid}")
    bt.logging.info(f"💼 Unique wallets:       {len(unique_wallets)}  →  {unique_wallets}")
    bt.logging.info(f"👥 Total wallet/hotkey pairs: {len(WALLET_HOTKEY_PAIRS)}")
    for idx, (wname, hname) in enumerate(WALLET_HOTKEY_PAIRS, 1):
        bt.logging.info(f"   {idx:>2}. wallet={wname:<12}  hotkey={hname}")
    bt.logging.info(f"🧬 Nanobody DB path:     {config.nanobody_db_path}")
    bt.logging.info(f"🧬 Nanobody target:      {config.nanobody_target or '(from challenge params)'}")
    bt.logging.info(f"⏰ Trigger point:        {BLOCKS_BEFORE_BOUNDARY} blocks before epoch boundary")
    bt.logging.info(f"📊 Epoch length:         {EPOCH_LENGTH} blocks")
    bt.logging.info(f"⏱️  Submission delay:     {SUBMISSION_DELAY}s between hotkeys")
    bt.logging.info("=" * 70 + "\n")


# ============================================================================
# BITTENSOR SETUP
# ============================================================================

async def setup_bittensor_objects(
    config: argparse.Namespace,
) -> Tuple[List[Any], Any, Any, List[int], int]:
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
                        _         = wallet.hotkey
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
                    except FileNotFoundError:
                        bt.logging.warning(
                            f"      {idx:>2}. ⚠️  {label:<22} → Hotkey file not found (skipping)"
                        )
                    except Exception as e:
                        bt.logging.error(
                            f"      {idx:>2}. ❌ {label:<22} → ERROR: {e}"
                        )

                if not wallets:
                    raise ValueError(
                        "❌ No valid wallet/hotkey pairs found in metagraph!"
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
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                bt.logging.error(f"   ❌ Failed to connect after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            bt.logging.error(f"   ❌ Unexpected error during setup: {e}")
            bt.logging.error(traceback.format_exc())
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                raise


# ============================================================================
# NANOBODY DIRECT SQLite ACCESS
# Replaces the Flask API client entirely — no network calls needed
# ============================================================================

def check_nanobody_db_health(db_path: str) -> bool:
    """
    Check that the nanobody SQLite DB exists and is readable.
    Mirrors the old check_nanobody_api_health() behaviour.
    """
    if not os.path.exists(db_path):
        bt.logging.warning(
            f"   ⚠️  [Nanobody DB] File not found: {db_path}"
        )
        return False
    try:
        conn   = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nanobodies")
        total_rows = cursor.fetchone()[0]

        try:
            cursor.execute(
                "SELECT created_at FROM nanobodies ORDER BY rowid DESC LIMIT 1"
            )
            row        = cursor.fetchone()
            last_write = row[0] if row else "N/A"
        except sqlite3.OperationalError:
            last_write = "N/A"

        conn.close()
        bt.logging.info(
            f"   ✅ [Nanobody DB] Health OK  "
            f"total_rows={total_rows}  last_write={last_write}"
        )
        return True
    except sqlite3.Error as e:
        bt.logging.warning(f"   ⚠️  [Nanobody DB] Health check failed: {e}")
        return False


def fetch_top_nanobodies(
    target: str,
    n: int,
    db_path: str,
    offset: int = 0,
) -> List[str]:
    """
    Fetch nanobody sequences directly from the local SQLite DB,
    ordered by final_nanobody_score ASC (lowest scores first).

    Args:
        target: Protein target id (matches nanobodies.target column).
        n: Maximum rows to return in this page.
        db_path: Path to nanobodies.sqlite.
        offset: SQL OFFSET for pagination (used when scanning past HF duplicates).

    Returns:
        List of sequence strings (length up to n), lowest score first.
        Returns empty list on any error — caller uses NANOBODY_FALLBACK.
    """
    if not target:
        bt.logging.warning(
            "   ⚠️  [Nanobody DB] No target specified — skipping nanobody fetch"
        )
        return []

    if not os.path.exists(db_path):
        bt.logging.warning(
            f"   ⚠️  [Nanobody DB] File not found: {db_path}  "
            f"→ falling back to '{NANOBODY_FALLBACK}'"
        )
        return []

    try:
        bt.logging.info(
            f"   🧬 [Nanobody DB] Querying lowest-scoring {n} for target={target}  "
            f"db={db_path}"
        )
        # Open read-only to avoid locking conflicts with nano.py writer
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT sequence, final_nanobody_score
            FROM   nanobodies
            WHERE  target = ?
              AND  sequence IS NOT NULL
              AND  sequence != ''
            ORDER  BY final_nanobody_score ASC
            LIMIT  ? OFFSET ?
            """,
            (target, n, offset),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            bt.logging.warning(
                f"   ⚠️  [Nanobody DB] No sequences found for target={target}  "
                f"→ falling back to '{NANOBODY_FALLBACK}'"
            )
            return []

        seqs = [row["sequence"] for row in rows]
        bt.logging.info(
            f"   ✅ [Nanobody DB] Got {len(seqs)} nanobodies for target={target}"
        )
        for i, row in enumerate(rows, 1):
            bt.logging.info(
                f"      {i:>2}. score={row['final_nanobody_score']}  "
                f"{row['sequence'][:40]}..."
            )
        return seqs

    except sqlite3.OperationalError as e:
        bt.logging.warning(
            f"   ⚠️  [Nanobody DB] DB not ready yet ({e})  "
            f"→ falling back to '{NANOBODY_FALLBACK}'"
        )
        return []
    except sqlite3.Error as e:
        bt.logging.warning(
            f"   ⚠️  [Nanobody DB] SQLite error: {e}  "
            f"→ falling back to '{NANOBODY_FALLBACK}'"
        )
        return []
    except Exception as e:
        bt.logging.error(
            f"   ❌ [Nanobody DB] Unexpected error: {e}  "
            f"→ falling back to '{NANOBODY_FALLBACK}'"
        )
        return []


def fetch_hf_unique_top_nanobodies_from_db(
    target: str,
    n: int,
    db_path: str,
) -> List[str]:
    """
    Build up to ``n`` nanobody sequences for submission that are not already
    present on the Hugging Face Submission-Archive for this nanobody target,
    scanning the local DB in score order (same hash contract as nano.py).
    """
    from utils.nanobodies import nanobody_unique_for_target_hf

    picked: List[str] = []
    seen_sequences: set[str] = set()
    offset = 0
    chunk = max(64, n * 4)
    max_rounds = 200

    for _ in range(max_rounds):
        if len(picked) >= n:
            break
        batch = fetch_top_nanobodies(
            target=target,
            n=chunk,
            db_path=db_path,
            offset=offset,
        )
        if not batch:
            break
        offset += len(batch)
        for seq in batch:
            if seq in seen_sequences:
                continue
            if not nanobody_unique_for_target_hf(target, seq):
                bt.logging.debug(
                    "   [Nanobody DB] Skipping sequence already in HF archive for "
                    f"target={target}: {seq[:40]}..."
                )
                continue
            picked.append(seq)
            seen_sequences.add(seq)
            if len(picked) >= n:
                break

    if len(picked) < n:
        bt.logging.warning(
            f"   ⚠️  Only {len(picked)} HF-unique nanobodies for target={target} "
            f"(wanted {n})"
        )
    else:
        bt.logging.info(
            f"   ✅ Selected {len(picked)} HF-unique nanobodies for target={target}"
        )
    return picked


# ============================================================================
# DATABASE OPERATIONS (molecules)
# ============================================================================

def get_reaction_score_db_path(allowed_reaction: str) -> Optional[str]:
    if not allowed_reaction:
        return None
    if str(allowed_reaction).lower() == "savi":
        return None
    if not allowed_reaction.startswith("rxn:"):
        bt.logging.warning(
            f"   ⚠️  Unsupported allowed reaction format: {allowed_reaction}"
        )
        return None
    try:
        reaction_id = int(allowed_reaction.split(":", 1)[1])
    except (ValueError, IndexError):
        bt.logging.warning(
            f"   ⚠️  Could not parse reaction id from: {allowed_reaction}"
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
        else:
            bt.logging.warning("   ⚠️  No available molecules found in database")
        return rows
    except sqlite3.Error as e:
        bt.logging.error(f"   ❌ Database error: {e}")
        return []
    except Exception as e:
        bt.logging.error(f"   ❌ Error querying database: {e}")
        bt.logging.error(traceback.format_exc())
        return []


def run_add_column_script(db_path: str) -> bool:
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
    nanobody_sequence: str,
    state: Dict[str, Any],
    submission_number: int,
    total_submissions: int,
) -> bool:
    """
    Encrypt and submit a molecule + nanobody pair using the specified wallet/hotkey.

    Submission format:
      "molecule|nanobody_sequence"   — when nanobody is available
      "molecule|~"                   — fallback when no nanobody
    """
    if not candidate_product:
        bt.logging.warning(f"      ⚠️  UID {miner_uid}: No candidate product")
        return False

    wallet_name = wallet.name       if hasattr(wallet, "name")       else "unknown"
    hotkey_name = wallet.hotkey_str if hasattr(wallet, "hotkey_str") else "unknown"
    label       = f"{wallet_name}/{hotkey_name}"

    nanobody_part = nanobody_sequence if nanobody_sequence else NANOBODY_FALLBACK
    message       = f"{candidate_product}|{nanobody_part}"

    bt.logging.info(
        f"\n   [{submission_number}/{total_submissions}] 📤 SUBMITTING: "
        f"UID {miner_uid} ({label})"
    )
    bt.logging.info(f"      Molecule:  {candidate_product}")
    bt.logging.info(
        f"      Nanobody:  "
        f"{'(fallback ~)' if nanobody_part == NANOBODY_FALLBACK else nanobody_part[:40] + '...'}"
    )
    bt.logging.info(f"      Message:   {message[:80]}{'...' if len(message) > 80 else ''}")

    try:
        current_block = await state["subtensor"].get_current_block()
        bt.logging.info(f"      Current block: {current_block}")

        bt.logging.info(f"      🔐 Encrypting response...")
        encrypted_response = state["bdt"].encrypt(miner_uid, message, current_block)
        bt.logging.info(f"      ✅ Encryption successful")

        tmp_file = tempfile.NamedTemporaryFile(delete=True, mode="w+")
        with open(tmp_file.name, "w+") as f:
            f.write(str(encrypted_response))
            f.flush()
            f.seek(0)

            content_str     = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()
            filename        = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content  = f"{state['github_path']}/{filename}.txt"

            bt.logging.info(f"      📝 Commit path: {commit_content}")
            bt.logging.info(f"      ⛓️  Attempting blockchain commitment...")

            try:
                commitment_status = await state["subtensor"].set_commitment(
                    wallet=wallet,
                    netuid=state["config"].netuid,
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
                bt.logging.warning(f"      ⏳ Too soon to commit again (rate limited)")
                bt.logging.error(f"      ❌ SUBMISSION FAILED for UID {miner_uid} ({label})")
                return False

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
                    return False
            except Exception as e:
                bt.logging.error(
                    f"      ❌ GitHub upload error for UID {miner_uid} ({label}): {e}"
                )
                return False

    except Exception as e:
        bt.logging.error(
            f"      ❌ Submission error for UID {miner_uid} ({label}): {e}"
        )
        bt.logging.error(traceback.format_exc())
        return False


# ============================================================================
# MAIN EPOCH LOOP
# ============================================================================

async def run_epoch_loop(state: Dict[str, Any]) -> None:
    bt.logging.info("🔄 Starting epoch monitoring loop...\n")

    bt.logging.info("🧬 Checking nanobody DB health at startup...")
    check_nanobody_db_health(state["config"].nanobody_db_path)

    last_acted_epoch = -1
    last_status_log  = datetime.datetime.now()
    num_pairs        = len(state["wallets"])

    while not shutdown_event.is_set():
        try:
            current_block    = await state["subtensor"].get_current_block()
            current_epoch    = current_block // state["epoch_length"]
            next_epoch_block = (current_epoch + 1) * state["epoch_length"]
            blocks_remaining = next_epoch_block - current_block

            now = datetime.datetime.now()
            if (now - last_status_log).total_seconds() >= STATUS_LOG_INTERVAL:
                bt.logging.info(
                    f"📊 Status | Block: {current_block} | Epoch: {current_epoch} | "
                    f"Next boundary: {next_epoch_block} | "
                    f"Blocks remaining: {blocks_remaining}"
                )
                last_status_log = now

            if (blocks_remaining <= BLOCKS_BEFORE_BOUNDARY
                    and current_epoch != last_acted_epoch):

                bt.logging.info("\n" + "=" * 70)
                bt.logging.info("⏰ SUBMISSION WINDOW REACHED")
                bt.logging.info("=" * 70)
                bt.logging.info(f"📍 Current block:          {current_block}")
                bt.logging.info(f"📍 Current epoch:          {current_epoch}")
                bt.logging.info(f"📍 Blocks until boundary:  {blocks_remaining}")
                bt.logging.info("=" * 70 + "\n")

                submission_start_time = datetime.datetime.now()

                # STEP 1: Determine allowed reaction
                bt.logging.info("🔹 STEP 1/5: Determine Allowed Reaction")
                cfg         = state["config"]
                start_block = current_epoch * state["epoch_length"]
                try:
                    start_block_hash = await state["subtensor"].determine_block_hash(
                        start_block
                    )
                    challenge_params = resolve_challenge_params(cfg, start_block_hash)
                    allowed_reaction = (
                        challenge_params.get("allowed_reaction")
                        if challenge_params else None
                    )
                except Exception as e:
                    bt.logging.error(
                        f"   ❌ Failed to get challenge params for epoch "
                        f"{current_epoch}: {e}"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                if not allowed_reaction:
                    bt.logging.warning(
                        f"   ⚠️  No allowed reaction for epoch {current_epoch}. "
                        "Skipping submission.\n"
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
                    f"   🎯 Allowed reaction for epoch {current_epoch}: "
                    f"{allowed_reaction}"
                )

                # STEP 2: Verify & update reaction DB
                bt.logging.info("🔹 STEP 2/5: Database Update")

                if not os.path.exists(reaction_db_path):
                    bt.logging.warning(
                        f"   ⚠️  Missing DB for {allowed_reaction}: "
                        f"{reaction_db_path}. Skipping epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                db_update_success = run_add_column_script(reaction_db_path)
                if not db_update_success:
                    bt.logging.error(
                        f"   ⚠️  Database update failed for {reaction_db_path}. "
                        "Skipping epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                # STEP 3: Fetch top N molecules
                bt.logging.info("🔹 STEP 3/5: Fetching Top Molecules")
                bt.logging.info(f"   🗄️  Using score DB: {reaction_db_path}")

                top_molecules = get_top_n_molecules_from_db(
                    n=num_pairs,
                    db_path=reaction_db_path,
                )

                if not top_molecules:
                    bt.logging.warning(
                        f"   ⚠️  No available molecules for {allowed_reaction}. "
                        "Skipping epoch.\n"
                    )
                    last_acted_epoch = current_epoch
                    await asyncio.sleep(12)
                    continue

                # STEP 4: Fetch top N nanobodies — DIRECT SQLite
                bt.logging.info("🔹 STEP 4/5: Fetching Top Nanobodies from local DB")

                nanobody_target = cfg.nanobody_target or ""
                if not nanobody_target and challenge_params:
                    nanobody_target = (
                        challenge_params.get("nanobody_target")
                        or challenge_params.get("weekly_target")
                        or ""
                    )

                if nanobody_target:
                    bt.logging.info(f"   🧬 Nanobody target: {nanobody_target}")
                    nanobody_seqs = fetch_hf_unique_top_nanobodies_from_db(
                        target=nanobody_target,
                        n=num_pairs,
                        db_path=cfg.nanobody_db_path,
                    )
                else:
                    bt.logging.warning(
                        "   ⚠️  Could not determine nanobody target — "
                        f"falling back to '{NANOBODY_FALLBACK}' for all submissions"
                    )
                    nanobody_seqs = []

                n_mols = len(top_molecules)
                if nanobody_seqs:
                    if len(nanobody_seqs) < n_mols:
                        bt.logging.warning(
                            f"   ⚠️  Only {len(nanobody_seqs)} nanobodies for "
                            f"{n_mols} molecules — reusing best nanobody for remainder"
                        )
                        best_nanobody = nanobody_seqs[0]
                        nanobody_seqs = nanobody_seqs + [best_nanobody] * (
                            n_mols - len(nanobody_seqs)
                        )
                    else:
                        nanobody_seqs = nanobody_seqs[:n_mols]
                else:
                    bt.logging.warning(
                        f"   ⚠️  No nanobodies available — "
                        f"all submissions will use '{NANOBODY_FALLBACK}'"
                    )
                    nanobody_seqs = [NANOBODY_FALLBACK] * n_mols

                bt.logging.info(f"   📋 Submission pairs ({n_mols} total):")
                for i, ((mol, score), nb) in enumerate(
                    zip(top_molecules, nanobody_seqs), 1
                ):
                    nb_display = (
                        f"(fallback ~)" if nb == NANOBODY_FALLBACK
                        else f"{nb[:30]}..."
                    )
                    bt.logging.info(
                        f"      {i:>2}. mol={mol:<28} score={score:.4f}  "
                        f"nb={nb_display}"
                    )

                # STEP 5: Submit SEQUENTIALLY
                bt.logging.info("🔹 STEP 5/5: Sequential Submission")
                bt.logging.info(
                    f"   ⚡ Submitting {n_mols} pairs using "
                    f"{num_pairs} wallet/hotkey pairs"
                )
                bt.logging.info(
                    f"   ⏱️  Delay between submissions: {SUBMISSION_DELAY}s"
                )

                results            = []
                submission_details = []

                for idx, ((molecule_name, score), nanobody_seq) in enumerate(
                    zip(top_molecules, nanobody_seqs)
                ):
                    if idx >= len(state["wallets"]):
                        bt.logging.warning(
                            f"   ⚠️  More molecules ({n_mols}) than "
                            f"wallet/hotkey pairs ({num_pairs}). "
                            f"Skipping: {molecule_name}"
                        )
                        break

                    wallet    = state["wallets"][idx]
                    miner_uid = state["miner_uids"][idx]

                    success = await submit_response(
                        wallet=wallet,
                        miner_uid=miner_uid,
                        candidate_product=molecule_name,
                        nanobody_sequence=nanobody_seq,
                        state=state,
                        submission_number=idx + 1,
                        total_submissions=n_mols,
                    )

                    results.append(success)
                    submission_details.append(
                        (wallet, molecule_name, nanobody_seq, miner_uid, score)
                    )

                    if idx < n_mols - 1:
                        bt.logging.info(
                            f"\n   ⏳ Waiting {SUBMISSION_DELAY}s before "
                            f"next submission...\n"
                        )
                        await asyncio.sleep(SUBMISSION_DELAY)

                # Results summary
                bt.logging.info("")
                bt.logging.info("=" * 70)
                bt.logging.info(f"📊 EPOCH {current_epoch} SUBMISSION RESULTS")
                bt.logging.info("=" * 70)

                success_count      = sum(1 for r in results if r)
                failure_count      = len(results) - success_count
                submission_elapsed = (
                    datetime.datetime.now() - submission_start_time
                ).total_seconds()

                bt.logging.info(f"✅ Successful: {success_count}/{len(results)}")
                bt.logging.info(f"❌ Failed:     {failure_count}/{len(results)}")
                bt.logging.info(f"⏱️  Total time: {submission_elapsed:.2f}s")
                bt.logging.info("=" * 70)

                bt.logging.info("\n📋 Detailed Results:")
                for i, (success, (wallet, mol, nb, uid, score)) in enumerate(
                    zip(results, submission_details), 1
                ):
                    status      = "✅" if success else "❌"
                    wname       = wallet.name       if hasattr(wallet, "name")       else "?"
                    hname       = wallet.hotkey_str if hasattr(wallet, "hotkey_str") else "?"
                    label       = f"{wname}/{hname}"
                    nb_display  = (
                        "(fallback ~)" if nb == NANOBODY_FALLBACK
                        else f"{nb[:25]}..."
                    )
                    bt.logging.info(
                        f"   {i:>2}. {status} UID {uid:>3} ({label:<22}) | "
                        f"mol={mol:<28} | score={score:.4f} | nb={nb_display}"
                    )

                next_submission_epoch = current_epoch + 1
                next_submission_block = (
                    (next_submission_epoch + 1) * state["epoch_length"]
                    - BLOCKS_BEFORE_BOUNDARY
                )
                blocks_until_next = next_submission_block - current_block
                time_until_next   = blocks_until_next * 12

                bt.logging.info("")
                bt.logging.info("=" * 70)
                bt.logging.info(f"⏭️  Next submission window:")
                bt.logging.info(f"   Epoch: {next_submission_epoch}")
                bt.logging.info(f"   Block: ~{next_submission_block}")
                bt.logging.info(
                    f"   ETA:   ~{time_until_next // 60} min "
                    f"({time_until_next}s)"
                )
                bt.logging.info("=" * 70 + "\n")

                last_acted_epoch = current_epoch
                await asyncio.sleep(12)
                continue

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
    try:
        wallets, subtensor, metagraph, miner_uids, epoch_length = \
            await setup_bittensor_objects(config)

        state: Dict[str, Any] = {
            "config":       config,
            "github_path":  load_github_path(),
            "wallets":      wallets,
            "miner_uids":   miner_uids,
            "subtensor":    subtensor,
            "metagraph":    metagraph,
            "epoch_length": epoch_length,
            "bdt":          QuicknetBittensorDrandTimelock(),
        }

        await run_epoch_loop(state)

    except Exception as e:
        bt.logging.error(f"❌ Fatal error in miner: {e}")
        bt.logging.error(traceback.format_exc())
        raise
    finally:
        if "subtensor" in locals():
            try:
                await subtensor.close()
                bt.logging.info("✅ Subtensor connection closed")
            except Exception:
                pass


def main():
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
