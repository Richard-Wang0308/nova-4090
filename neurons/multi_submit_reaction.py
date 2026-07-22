
#!/usr/bin/env python3
"""
MULTI-WALLET MULTI-HOTKEY EPOCH-BASED MOLECULE SUBMISSION SCRIPT

Workflow:
1. On startup, immediately determine current epoch's allowed reaction and submit.
2. Monitor blockchain for epoch boundaries.
3. As soon as the epoch counter changes, determine that epoch's allowed reaction,
    verify availability for only the top candidate pool (not the whole DB),
    PREPARE all payloads, fire ALL chain commits in parallel (this is the
    time-critical race against other miners), then BATCH-UPLOAD all
    successfully-committed files to GitHub in a SINGLE commit (via the Git
    Data API: blobs -> tree -> commit -> one ref update) — this avoids the
    read-modify-write race that the per-file Contents API had when multiple
    wallets uploaded concurrently. Then mark submitted molecules unavailable.
4. Repeat.
"""

import os
import sys
import asyncio
import argparse
import datetime
import traceback
import base64
import hashlib
import sqlite3
import signal
import requests
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

# ============================================================================
# WALLET + HOTKEY CONFIGURATION
# Each entry is a (wallet_name, hotkey_name) pair.
# Add as many wallets/hotkeys as needed.
# ============================================================================
WALLET_HOTKEY_PAIRS: List[Tuple[str, str]] = [
    ("nova",   "notd"),
    ("nova",   "notc"),
    # ("nova",   "notd")
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
POLL_INTERVAL = 6            # Seconds between block polls

# Only re-check HuggingFace uniqueness for this many top-scoring candidates
# (instead of scanning the entire available=TRUE set).
AVAILABILITY_CANDIDATE_POOL = 100

# Retries for the GitHub batch-commit flow (re-fetches branch head + retries
# the whole blob->tree->commit->ref-update sequence on conflict).
GITHUB_BATCH_MAX_RETRIES = 3

# ============================================================================

from config.config_loader import load_config
from utils import (
    get_challenge_params_from_blockhash,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from utils.molecules import molecule_unique_for_protein_hf
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

    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Wallet.add_args(parser)

    # bt.Config on bittensor>=10 drops custom args and nested store_true
    # flags (e.g. --logging.debug). Capture from argparse and write back.
    args = parser.parse_args()
    config = bt.Config(parser)
    config.update(load_config())
    config.netuid = args.netuid
    config.network = args.network
    if getattr(config, "subtensor", None) is not None:
        config.subtensor.network = args.network

    args_dict = vars(args)
    for key in (
        "logging.debug",
        "logging.trace",
        "logging.info",
        "logging.record_log",
        "logging.enable_third_party_loggers",
    ):
        if key in args_dict and getattr(config, "logging", None) is not None:
            setattr(config.logging, key.split(".", 1)[1], args_dict[key])
    if "logging.logging_dir" in args_dict and getattr(config, "logging", None) is not None:
        config.logging.logging_dir = args_dict["logging.logging_dir"]

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
    """Constructs the path for GitHub operations (used in on-chain commit metadata)."""
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
    bt.logging.info(
        f"⚡ Parallel CHAIN COMMIT (per-wallet connections) | GitHub upload "
        f"batched into ONE commit after chain commit | availability pool: "
        f"top {AVAILABILITY_CANDIDATE_POOL} candidates"
    )
    bt.logging.info("="*70 + "\n")


# ============================================================================
# BITTENSOR SETUP
# ============================================================================

async def setup_bittensor_objects(
    config: argparse.Namespace
) -> Tuple[List[Any], List[Any], Any, Any, List[int], int]:
    """
    Initializes multiple wallets (potentially different wallet names and hotkeys),
    ONE DEDICATED AsyncSubtensor connection PER WALLET (used only for the
    time-critical set_commitment call so concurrent commits are never
    serialized on a shared websocket), a shared subtensor for general reads
    (get_current_block, metagraph, etc.), and metagraph.

    Returns:
        (wallets_list, wallet_subtensors_list, shared_subtensor, metagraph,
        miner_uids_list, epoch_length)
    """
    bt.logging.info("🔧 Setting up Bittensor objects with multiple wallets/hotkeys...")

    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            bt.logging.info(
                f"   Attempting connection (attempt {attempt + 1}/{max_retries})..."
            )

            subtensor = bt.AsyncSubtensor(network=config.network)

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
                        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

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

            # Reinitialize shared subtensor for main loop (used for reads:
            # get_current_block, determine_block_hash, etc.)
            subtensor = bt.AsyncSubtensor(network=config.network)
            await subtensor.initialize()

            # Create ONE DEDICATED connection per wallet, used ONLY for
            # set_commitment, so that concurrent commits from different
            # wallets are never queued behind each other on a shared
            # websocket/transport.
            bt.logging.info(
                f"   🔌 Opening {len(wallets)} dedicated connection(s) "
                f"for parallel chain commits..."
            )
            wallet_subtensors: List[Any] = []
            for idx, (wallet_name, hotkey_name) in enumerate(
                WALLET_HOTKEY_PAIRS[:len(wallets)], 1
            ):
                ws = bt.AsyncSubtensor(network=config.network)
                await ws.initialize()
                wallet_subtensors.append(ws)
                bt.logging.info(
                    f"      {idx:>2}. ✅ Dedicated commit-connection ready "
                    f"for {wallet_name}/{hotkey_name}"
                )
            bt.logging.info("")

            return wallets, wallet_subtensors, subtensor, metagraph, miner_uids, EPOCH_LENGTH

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


def ensure_available_column(db_path: str) -> bool:
    """Ensure scored_molecules.available exists as a BOOLEAN column."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='scored_molecules' LIMIT 1"
        )
        if cursor.fetchone() is None:
            bt.logging.error(f"   ❌ Table 'scored_molecules' not found in {db_path}")
            conn.close()
            return False

        cursor.execute("PRAGMA table_info(scored_molecules)")
        columns = {row[1] for row in cursor.fetchall()}
        if "available" not in columns:
            cursor.execute(
                "ALTER TABLE scored_molecules ADD COLUMN available BOOLEAN"
            )
            conn.commit()
            bt.logging.info("   ✅ Added missing 'available' column")

        conn.close()
        return True
    except Exception as e:
        bt.logging.error(f"   ❌ Failed to ensure available column: {e}")
        bt.logging.error(traceback.format_exc())
        return False


def _set_molecule_available(
    db_path: str, molecule_name: str, available: bool
) -> None:
    """Update a single molecule's available flag."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE scored_molecules SET available = ? WHERE molecule_name = ?",
            (available, molecule_name),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        bt.logging.error(
            f"   ❌ Failed to set available={available} for {molecule_name}: {e}"
        )


def mark_molecules_unavailable(
    db_path: str, molecule_names: List[str]
) -> None:
    """Mark molecules as unavailable after successful submission."""
    if not molecule_names:
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE scored_molecules SET available = FALSE WHERE molecule_name = ?",
            [(name,) for name in molecule_names],
        )
        conn.commit()
        conn.close()
        bt.logging.info(
            f"   🔒 Marked {len(molecule_names)} submitted molecule(s) "
            f"as available=FALSE"
        )
    except Exception as e:
        bt.logging.error(f"   ❌ Failed to mark molecules unavailable: {e}")
        bt.logging.error(traceback.format_exc())


def check_molecule_unique(
    target_protein: str, molecule_name: str, smiles: str
) -> bool:
    """
    Return True if molecule is NOT already in the HuggingFace dataset
    for the target protein.
    """
    if not target_protein:
        bt.logging.warning(
            "   ⚠️  No target protein provided for uniqueness check"
        )
        return False

    try:
        return bool(molecule_unique_for_protein_hf(target_protein, smiles))
    except Exception as e:
        bt.logging.error(
            f"   ❌ Uniqueness check failed for {molecule_name}: {e}"
        )
        return False


async def get_verified_top_n_molecules(
    n: int,
    db_path: str,
    target_protein: str,
    candidate_pool: int = AVAILABILITY_CANDIDATE_POOL,
) -> List[Tuple[str, float]]:
    """
    Fast path for epoch submission:
    1. Load only the top `candidate_pool` molecules by score that are still
        marked available (available = TRUE).
    2. Re-check HuggingFace uniqueness for those candidates only.
    3. Stop as soon as `n` verified-available molecules are found.
    4. Persist updated available flags for every candidate checked.
    """
    if n <= 0:
        return []

    if not os.path.exists(db_path):
        bt.logging.error(f"   ❌ Database not found: {db_path}")
        return []

    if not ensure_available_column(db_path):
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
            (candidate_pool,),
        )
        candidates = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        bt.logging.error(
            f"   ❌ Database error while loading candidates: {e}"
        )
        return []

    if not candidates:
        bt.logging.warning("   ⚠️  No candidate molecules found in database")
        return []

    bt.logging.info(
        f"   🔍 Checking availability for top {len(candidates)} candidates "
        f"(need {n} available)"
    )

    verified: List[Tuple[str, float]] = []
    checked = 0
    marked_false = 0

    for molecule_name, score in candidates:
        if len(verified) >= n:
            break

        checked += 1
        smiles = get_smiles_from_reaction(molecule_name)
        if smiles is None:
            bt.logging.warning(
                f"      ❌ {molecule_name}: could not derive SMILES "
                f"→ available=FALSE"
            )
            _set_molecule_available(db_path, molecule_name, False)
            marked_false += 1
            continue

        # Offload sync HF lookup so the event loop stays responsive.
        is_unique = await asyncio.to_thread(
            check_molecule_unique, target_protein, molecule_name, smiles
        )

        if is_unique:
            _set_molecule_available(db_path, molecule_name, True)
            verified.append((molecule_name, score))
            bt.logging.info(
                f"      ✅ [{len(verified)}/{n}] {molecule_name:<30} "
                f"| Score: {score:.6f}"
            )
        else:
            _set_molecule_available(db_path, molecule_name, False)
            marked_false += 1
            bt.logging.info(
                f"      ❌ {molecule_name}: already known → available=FALSE"
            )

    bt.logging.info(
        f"   📊 Availability check done: checked={checked}, "
        f"verified={len(verified)}, marked_false={marked_false}"
    )

    if not verified:
        bt.logging.warning(
            "   ⚠️  No available molecules found in the top candidate pool"
        )

    return verified


# ============================================================================
# GITHUB BATCH UPLOAD — Git Data API (blobs -> tree -> commit -> ref update)
#
# Why: The old approach used the Contents API (PUT /repos/.../contents/{path})
# once PER FILE. That endpoint does an internal read-modify-write on the
# branch ref: it reads the current branch head, builds a new commit on top
# of it, then fast-forwards the ref. When 2+ uploads run concurrently, both
# can read the SAME branch head before either has pushed, so the second
# push is rejected with a 409 ("is at X but expected Y").
#
# Fix: combine ALL files for the epoch into ONE tree and ONE commit, so the
# branch ref is only read-and-updated ONCE per epoch, regardless of how
# many wallets submitted. This eliminates the race between our own wallets
# entirely (only external, unrelated pushes to the same branch could still
# theoretically conflict — handled via retry-with-refetch below).
# ============================================================================

def _github_api_base() -> str:
    owner = os.environ.get('GITHUB_REPO_OWNER')
    repo = os.environ.get('GITHUB_REPO_NAME')
    if not owner or not repo:
        raise ValueError(
            "Missing required GitHub environment variables: "
            "GITHUB_REPO_OWNER, GITHUB_REPO_NAME"
        )
    return f"https://api.github.com/repos/{owner}/{repo}"


def _github_headers() -> Dict[str, str]:
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError(
            "Missing required GITHUB_TOKEN environment variable "
            "(needs 'repo' write scope for Git Data API calls)"
        )
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


def upload_files_to_github_batch(
    files: List[Dict[str, str]],
    commit_message: str,
    max_retries: int = GITHUB_BATCH_MAX_RETRIES,
) -> bool:
    """
    Upload multiple files to GitHub in a SINGLE commit.

    files: list of {"filename": str, "encoded_content": str (base64 text)}
    Returns True if the entire batch commit + ref update succeeded.

    Retries the WHOLE flow (re-fetching the branch head each time) if the
    final ref update is rejected due to a conflicting push from elsewhere
    (422 "not a fast forward" / 409). This is now a rare edge case since
    our own wallets no longer race each other — only a truly external
    concurrent push to the same branch could trigger it.
    """
    if not files:
        return True

    try:
        api_base = _github_api_base()
        headers = _github_headers()
    except ValueError as e:
        bt.logging.error(f"   ❌ GitHub batch upload config error: {e}")
        return False

    branch = os.environ.get('GITHUB_REPO_BRANCH')
    repo_path = os.environ.get('GITHUB_REPO_PATH', '')

    if not branch:
        bt.logging.error("   ❌ Missing GITHUB_REPO_BRANCH environment variable")
        return False

    for attempt in range(1, max_retries + 1):
        try:
            # ---- 1. Get latest commit SHA on the branch ----
            ref_resp = requests.get(
                f"{api_base}/git/ref/heads/{branch}", headers=headers, timeout=15
            )
            ref_resp.raise_for_status()
            latest_commit_sha = ref_resp.json()["object"]["sha"]

            # ---- 2. Get the tree SHA of that commit ----
            commit_resp = requests.get(
                f"{api_base}/git/commits/{latest_commit_sha}", headers=headers, timeout=15
            )
            commit_resp.raise_for_status()
            base_tree_sha = commit_resp.json()["tree"]["sha"]

            # ---- 3. Build tree entries with inline content (auto-creates blobs) ----
            tree_entries = []
            for f in files:
                full_path = (
                    f"{repo_path}/{f['filename']}.txt" if repo_path
                    else f"{f['filename']}.txt"
                )
                raw_content = base64.b64decode(f["encoded_content"]).decode()
                tree_entries.append({
                    "path": full_path,
                    "mode": "100644",
                    "type": "blob",
                    "content": raw_content,
                })

            tree_resp = requests.post(
                f"{api_base}/git/trees",
                headers=headers,
                json={"base_tree": base_tree_sha, "tree": tree_entries},
                timeout=20,
            )
            tree_resp.raise_for_status()
            new_tree_sha = tree_resp.json()["sha"]

            # ---- 4. Create a new commit on top of the current head ----
            commit_create_resp = requests.post(
                f"{api_base}/git/commits",
                headers=headers,
                json={
                    "message": commit_message,
                    "tree": new_tree_sha,
                    "parents": [latest_commit_sha],
                },
                timeout=15,
            )
            commit_create_resp.raise_for_status()
            new_commit_sha = commit_create_resp.json()["sha"]

            # ---- 5. Update the branch ref to point to the new commit ----
            update_ref_resp = requests.patch(
                f"{api_base}/git/refs/heads/{branch}",
                headers=headers,
                json={"sha": new_commit_sha, "force": False},
                timeout=15,
            )
            update_ref_resp.raise_for_status()

            bt.logging.info(
                f"      ✅ Batch commit created: {new_commit_sha[:10]} "
                f"({len(files)} file(s))"
                + (f" [attempt {attempt}]" if attempt > 1 else "")
            )
            return True

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = e.response.text if e.response is not None else str(e)
            is_conflict = status in (409, 422)

            if is_conflict and attempt < max_retries:
                bt.logging.warning(
                    f"      ⏳ GitHub ref conflict (status {status}), "
                    f"refetching branch head and retrying "
                    f"(attempt {attempt}/{max_retries})..."
                )
                continue

            bt.logging.error(
                f"      ❌ GitHub batch upload failed (status {status}): {body}"
            )
            if attempt >= max_retries:
                bt.logging.error(traceback.format_exc())
            return False

        except Exception as e:
            bt.logging.error(f"      ❌ GitHub batch upload error: {e}")
            if attempt >= max_retries:
                bt.logging.error(traceback.format_exc())
                return False
            continue

    bt.logging.error(
        f"      ❌ GitHub batch upload FAILED after {max_retries} attempts"
    )
    return False


# ============================================================================
# SUBMISSION — 3-PHASE PIPELINE
#   Phase 0: prepare  (encrypt + build payload)   -> parallel, off-chain
#   Phase 1: commit   (set_commitment ONLY)        -> parallel, TIME-CRITICAL
#   Phase 2: upload   (ONE batched GitHub commit)  -> sequential, NOT critical
# ============================================================================

async def prepare_submission(
    wallet: Any,
    miner_uid: int,
    candidate_product: str,
    state: Dict[str, Any],
    current_block: int,
) -> Optional[Dict[str, Any]]:
    """
    Build the encrypted payload, filename, and commit-path string for a
    single wallet. Pure CPU/memory work — no chain call, no GitHub call.
    """
    wallet_name = wallet.name if hasattr(wallet, 'name') else 'unknown'
    hotkey_name = wallet.hotkey_str if hasattr(wallet, 'hotkey_str') else 'unknown'
    label = f"{wallet_name}/{hotkey_name}"

    if not candidate_product:
        bt.logging.warning(f"      ⚠️  UID {miner_uid}: No candidate product")
        return None

    try:
        message = f"{candidate_product}|~"

        # Offload in case encrypt() is CPU-bound (timelock crypto usually is)
        encrypted_response = await asyncio.to_thread(
            state['bdt'].encrypt, miner_uid, message, current_block
        )

        content_str = str(encrypted_response)
        encoded_content = base64.b64encode(content_str.encode()).decode()
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"

        bt.logging.info(
            f"      🧪 Prepared {label} → {candidate_product} "
            f"(commit path: {commit_content})"
        )

        return {
            "wallet": wallet,
            "miner_uid": miner_uid,
            "molecule_name": candidate_product,
            "commit_content": commit_content,
            "encoded_content": encoded_content,
            "filename": filename,
            "label": label,
        }
    except Exception as e:
        bt.logging.error(f"      ❌ Prepare failed for {label}: {e}")
        bt.logging.error(traceback.format_exc())
        return None


async def commit_only(
    payload: Dict[str, Any],
    subtensor_conn: Any,
    state: Dict[str, Any],
) -> bool:
    """
    THE TIME-CRITICAL CALL. Nothing but the chain extrinsic happens here.
    Uses a connection dedicated to this wallet so it is never queued
    behind another wallet's commit on a shared websocket.
    """
    label = payload["label"]
    miner_uid = payload["miner_uid"]

    bt.logging.info(f"      ⛓️  Committing on-chain for {label} (UID {miner_uid})...")

    try:
        commitment_status = await subtensor_conn.set_commitment(
            wallet=payload["wallet"],
            netuid=state['config'].netuid,
            data=payload["commit_content"],
            wait_for_inclusion=True,      # need confirmed ordering
            wait_for_finalization=False,  # don't wait extra blocks
        )

        if commitment_status:
            bt.logging.info(f"      ✅ Commit OK for {label} (UID {miner_uid})")
        else:
            bt.logging.error(
                f"      ❌ Commit returned False for {label} (UID {miner_uid})"
            )
        return bool(commitment_status)

    except MetadataError as e:
        bt.logging.warning(
            f"      ⏳ MetadataError (rate limited) for {label} "
            f"(UID {miner_uid}): {e}"
        )
        return False
    except Exception as e:
        bt.logging.error(
            f"      ❌ Commit error for {label} (UID {miner_uid}): {e}"
        )
        bt.logging.error(traceback.format_exc())
        return False


# ============================================================================
# SUBMISSION FOR ONE EPOCH (used both on startup and on epoch change)
# ============================================================================

async def do_epoch_submission(state: Dict[str, Any], current_epoch: int) -> None:
    """
    Full pipeline for a given epoch:
    1. Determine that epoch's allowed reaction + target protein
    2. Verify availability for only the top candidate pool
    3. Prepare payloads -> fire ALL chain commits in parallel -> batch-upload
        ALL successfully-committed files to GitHub in ONE commit
    4. Mark successfully submitted molecules as unavailable
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
    # STEP 1: Determine allowed reaction + target
    # ==========================================================
    bt.logging.info("🔹 STEP 1/3: Determine Allowed Reaction")
    cfg = state["config"]
    start_block = current_epoch * state["epoch_length"]
    try:
        start_block_hash = await state["subtensor"].determine_block_hash(start_block)
        challenge_params = resolve_challenge_params(cfg, start_block_hash)
        allowed_reaction = (
            challenge_params.get("allowed_reaction") if challenge_params else None
        )
        target_protein = (
            challenge_params.get("small_molecule_target") if challenge_params else None
        )
        if isinstance(target_protein, list):
            target_protein = target_protein[0] if target_protein else None
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

    if not target_protein:
        bt.logging.error(
            "   ❌ No small_molecule_target in challenge params. "
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
    bt.logging.info(f"   🎯 Target protein: {target_protein}")

    if not os.path.exists(reaction_db_path):
        bt.logging.warning(
            f"   ⚠️  Missing DB for {allowed_reaction}: {reaction_db_path}. "
            "Skipping submission for this epoch.\n"
        )
        return

    # ==========================================================
    # STEP 2: Fast availability check on top candidate pool only
    # ==========================================================
    bt.logging.info("🔹 STEP 2/3: Fast Availability Check (Top Candidates Only)")
    bt.logging.info(f"   🗄️  Using score DB: {reaction_db_path}")
    bt.logging.info(
        f"   ⚡ Candidate pool size: {AVAILABILITY_CANDIDATE_POOL} "
        f"(need {num_pairs} available)"
    )

    top_molecules = await get_verified_top_n_molecules(
        n=num_pairs,
        db_path=reaction_db_path,
        target_protein=target_protein,
        candidate_pool=AVAILABILITY_CANDIDATE_POOL,
    )

    if not top_molecules:
        bt.logging.warning(
            f"   ⚠️  No available molecules found for {allowed_reaction}. "
            "Skipping submission for this epoch.\n"
        )
        return

    bt.logging.info("")

    # ==========================================================
    # STEP 3a: PREPARE all payloads in parallel (no chain/GitHub calls)
    # ==========================================================
    bt.logging.info("🔹 STEP 3/3: Prepare → Parallel Commit → Batched Upload")
    bt.logging.info(f"   🧪 Preparing {len(top_molecules)} payload(s)...")

    # Re-fetch block right before preparing, so the encrypted payload uses
    # the freshest block available before we race to commit.
    current_block = await state["subtensor"].get_current_block()

    prep_pairs: List[Tuple[Any, int, str, float]] = []
    for idx, (molecule_name, score) in enumerate(top_molecules):
        if idx >= len(state['wallets']):
            bt.logging.warning(
                f"   ⚠️  More molecules ({len(top_molecules)}) than "
                f"wallet/hotkey pairs ({num_pairs}). Skipping: {molecule_name}"
            )
            break
        wallet = state['wallets'][idx]
        miner_uid = state['miner_uids'][idx]
        prep_pairs.append((wallet, miner_uid, molecule_name, score))

    prep_tasks = [
        prepare_submission(wallet, miner_uid, molecule_name, state, current_block)
        for (wallet, miner_uid, molecule_name, _score) in prep_pairs
    ]
    prep_results = await asyncio.gather(*prep_tasks, return_exceptions=True)

    payloads: List[Dict[str, Any]] = []
    scores: List[float] = []
    for (wallet, miner_uid, molecule_name, score), result in zip(prep_pairs, prep_results):
        if isinstance(result, Exception) or result is None:
            bt.logging.error(
                f"   ❌ Skipping {molecule_name} (UID {miner_uid}): prepare failed"
            )
            continue
        payloads.append(result)
        scores.append(score)

    if not payloads:
        bt.logging.warning("   ⚠️  No payloads prepared successfully. Skipping epoch.\n")
        return

    # ==========================================================
    # STEP 3b: FIRE ALL CHAIN COMMITS IN PARALLEL — TIME CRITICAL
    # Each wallet uses its OWN dedicated AsyncSubtensor connection so
    # concurrent commits are not serialized on a shared websocket.
    # ==========================================================
    bt.logging.info(
        f"   ⚡ Firing {len(payloads)} chain commit(s) in PARALLEL "
        f"(dedicated connection per wallet)..."
    )

    commit_tasks = [
        commit_only(payload, state['wallet_subtensors'][idx], state)
        for idx, payload in enumerate(payloads)
    ]
    commit_start = datetime.datetime.now()
    commit_results = await asyncio.gather(*commit_tasks, return_exceptions=True)
    commit_elapsed = (datetime.datetime.now() - commit_start).total_seconds()
    bt.logging.info(f"   ⏱️  All commits resolved in {commit_elapsed:.2f}s")

    commit_ok: List[bool] = []
    for idx, outcome in enumerate(commit_results):
        if isinstance(outcome, Exception):
            bt.logging.error(
                f"   ❌ Commit exception for {payloads[idx]['label']} "
                f"({payloads[idx]['molecule_name']}): {outcome}"
            )
            commit_ok.append(False)
        else:
            commit_ok.append(bool(outcome))

    bt.logging.info("")

    # ==========================================================
    # STEP 3c: BATCH UPLOAD to GitHub — ONE commit for ALL successfully
    # committed files. NOT time-critical (doesn't affect ranking), but
    # since it's a single atomic operation now, there's no benefit to
    # doing it in parallel with anything else — just run it once.
    # ==========================================================
    upload_indices = [i for i, ok in enumerate(commit_ok) if ok]
    bt.logging.info(
        f"   📤 Batch-uploading {len(upload_indices)}/{len(payloads)} "
        f"successfully-committed molecule(s) to GitHub in ONE commit..."
    )

    batch_upload_ok = False
    if upload_indices:
        files_for_batch = [
            {
                "filename": payloads[i]["filename"],
                "encoded_content": payloads[i]["encoded_content"],
            }
            for i in upload_indices
        ]
        commit_message = (
            f"Epoch {current_epoch} submissions "
            f"({len(files_for_batch)} molecule(s))"
        )
        batch_upload_ok = await asyncio.to_thread(
            upload_files_to_github_batch, files_for_batch, commit_message
        )
    else:
        bt.logging.warning("   ⚠️  No successful commits to upload.")

    # Every file in the batch succeeds or fails together, since it's a
    # single atomic commit.
    upload_ok_map: Dict[int, bool] = {i: batch_upload_ok for i in upload_indices}

    # Final success = commit succeeded AND batch upload succeeded.
    results: List[bool] = []
    for idx in range(len(payloads)):
        if not commit_ok[idx]:
            results.append(False)
        else:
            results.append(upload_ok_map.get(idx, False))

    # Mark successfully submitted molecules as unavailable so they are not
    # reused on later epochs / restarts. We mark based on COMMIT success
    # (the part that matters for ranking) — even if the GitHub batch
    # upload needs to be retried out-of-band, the molecule name is now
    # "spent" on-chain.
    submitted_ok = [
        payloads[idx]["molecule_name"]
        for idx in range(len(payloads))
        if commit_ok[idx]
    ]
    mark_molecules_unavailable(reaction_db_path, submitted_ok)

    # ==========================================================
    # Process Results
    # ==========================================================
    bt.logging.info("")
    bt.logging.info("="*70)
    bt.logging.info(f"📊 EPOCH {current_epoch} SUBMISSION RESULTS")
    bt.logging.info("="*70)

    success_count = sum(1 for r in results if r)
    commit_success_count = sum(1 for r in commit_ok if r)
    failure_count = len(results) - success_count
    submission_elapsed = (
        datetime.datetime.now() - submission_start_time
    ).total_seconds()

    bt.logging.info(f"✅ Fully successful (commit+upload): {success_count}/{len(results)}")
    bt.logging.info(f"⛓️  Chain commit successful:          {commit_success_count}/{len(results)}")
    bt.logging.info(f"❌ Failed:                            {failure_count}/{len(results)}")
    bt.logging.info(f"⏱️  Commit phase time:  {commit_elapsed:.2f}s")
    bt.logging.info(f"⏱️  Total epoch time:   {submission_elapsed:.2f}s")
    bt.logging.info("="*70)

    # Detailed results
    bt.logging.info("\n📋 Detailed Results:")
    for idx, payload in enumerate(payloads, 0):
        status = "✅" if results[idx] else ("⛓️❌" if not commit_ok[idx] else "📤❌")
        label = payload["label"]
        miner_uid = payload["miner_uid"]
        molecule_name = payload["molecule_name"]
        score = scores[idx]
        bt.logging.info(
            f"   {idx+1:>2}. {status} UID {miner_uid:>3} ({label:<22}) | "
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
            blocks_remaining = state['epoch_length'] - blocks_into_epoch

            # Periodic status logging
            now = datetime.datetime.now()
            if (now - last_status_log).total_seconds() >= STATUS_LOG_INTERVAL:
                bt.logging.info(
                    f"📊 Status | Block: {current_block} | Epoch: {current_epoch} | "
                    f"Blocks into epoch: {blocks_into_epoch} | "
                    f"Blocks remaining: {blocks_remaining} | "
                    f"Last acted epoch: {last_acted_epoch}"
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
    subtensor = None
    wallet_subtensors: List[Any] = []
    try:
        wallets, wallet_subtensors, subtensor, metagraph, miner_uids, epoch_length = \
            await setup_bittensor_objects(config)

        state: Dict[str, Any] = {
            'config':           config,
            'github_path':      load_github_path(),
            'wallets':           wallets,
            'wallet_subtensors': wallet_subtensors,  # dedicated per-wallet connections for parallel commits
            'miner_uids':        miner_uids,
            'subtensor':         subtensor,           # shared connection for reads (get_current_block, etc.)
            'metagraph':         metagraph,
            'epoch_length':      epoch_length,
            'bdt':               QuicknetBittensorDrandTimelock(),
        }

        await run_epoch_loop(state)

    except Exception as e:
        bt.logging.error(f"❌ Fatal error in miner: {e}")
        bt.logging.error(traceback.format_exc())
        raise
    finally:
        # Close the shared subtensor connection
        if subtensor is not None:
            try:
                await subtensor.close()
                bt.logging.info("✅ Shared subtensor connection closed")
            except Exception:
                pass

        # Close all dedicated per-wallet commit connections
        for idx, ws in enumerate(wallet_subtensors, 1):
            try:
                await ws.close()
                bt.logging.info(f"✅ Wallet commit-connection {idx} closed")
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
