
#!/usr/bin/env python3
"""
MULTI-WALLET MULTI-HOTKEY EPOCH-BASED MOLECULE SUBMISSION SCRIPT

Workflow:
1. On startup, immediately determine current epoch's allowed reaction and submit.
2. Monitor blockchain for epoch boundaries.
3. As soon as the epoch counter changes, determine that epoch's allowed reaction,
    build a diverse set of num_molecules (default 20) per wallet from the top
    candidate pool — each set must pass HF uniqueness, BRENK structural alerts,
    historical diversity,
    no InChIKey duplicates within the set, and atom-pair fingerprint entropy
    >= min_entropy over the completed set (same checks the validator applies).
    PREPARE all payloads (comma-separated molecule names, timelock-encrypted),
    fire ALL chain commits concurrently over ONE persistent AsyncSubtensor
    connection, then BATCH-UPLOAD all successfully-committed files to GitHub
    in a SINGLE commit. Mark submitted molecules
    unavailable.
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
import time
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
    ("nova",   "nota")
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

# Only re-check HuggingFace uniqueness + historical diversity for this many
# top-scoring candidates (instead of scanning the entire available=TRUE set).
# Keep large enough for num_molecules × wallet count greedy selection.
AVAILABILITY_CANDIDATE_POOL = 1000

# Fallback only — overwritten from config/config.yaml right after the imports
# below (load_config is not importable this early in the file). Never edit this
# by hand: hardcoding 0.9 here while config said 0.7 is what let molecules the
# validator rejects reach the submission set.
MAX_SIMILARITY_TO_HISTORICAL = 0.6

# Atom-pair fingerprint width for the entropy check. Must equal the validator's
# (utils.molecules.ENTROPY_FP_SIZE) -- folding to any other width silently
# changes every entropy value the threshold was calibrated against.
ENTROPY_FP_SIZE = 2048

# How many extra qualified candidates to gather when the score-ordered set
# misses the entropy floor and has to be repaired. 0 means the entire remaining
# candidate pool. Only paid for when a repair is actually needed.
#
# WAS 400, AND A CAPPED RESERVE IS WHAT MADE rxn5 SKIP EPOCH 24907. A reserve
# truncated by score is chemically narrow -- the next 400 molecules by score are
# near-neighbours of the twenty already picked -- so the swap search runs out of
# distinct chemistry and stalls at a local maximum BELOW the floor while the
# swap cap is nowhere near reached. Measured on that epoch's pool (999 qualified
# candidates, floor 0.25):
#
#   reserve=400  ->  entropy 0.2479 after 21 swaps, STALLED, epoch thrown away
#   reserve=999  ->  entropy 0.2502 after  7 swaps, and a HIGHER mean score
#
# A bigger reserve is not just insurance: it reaches the floor in fewer swaps
# and gives up less score doing it, because each swap chooses from more
# chemistry. The gathering cost is one availability check per candidate
# (~7-15 ms against the cached HF table), paid only on a repair, against an
# epoch that is over an hour long.
ENTROPY_REPAIR_RESERVE = 0

# Cap on repair swaps.
#
# WAS 8, AND 8 WAS TOO FEW -- this is what made rxn5 skip epoch 24891. The
# score-ordered top 20 of a small confirmed pool can start FAR below the floor
# (0.1679 measured on real rxn5 chemistry against a 0.25 requirement), and the
# repair climbs at a steady ~0.0107 per swap with no plateau:
#
#   0.1679 0.1854 0.1992 0.2112 0.2224 0.2317 0.2404 0.2478 | 0.2537
#                                                       8th ^   ^ 9th
#
# It ran out of swaps one short of clearing the bar and threw the epoch away.
# The loop exits the moment entropy reaches the floor, so a high cap costs
# nothing in the normal case -- it is only ever reached when the alternative is
# submitting nothing at all. At ~0.08 s per swap even 60 is under five seconds.
ENTROPY_REPAIR_MAX_SWAPS = 60

# Cap on the score-recovery swaps of the diversity-first fallback
# (_entropy_rebuild). Each round buys back the most score it can without
# dropping the set below the floor and stops on its own once no swap helps;
# 16 rounds were used on epoch 24907, so this is a runaway guard, not a budget.
ENTROPY_RECOVERY_MAX_SWAPS = 200

# --- Score-maximising search at the entropy floor -------------------------
#
# Clearing the floor is necessary but NOT the goal. The validator scores what
# is submitted, so among all sets that clear the floor the one to submit is the
# highest-scoring one -- the entropy repair is a constrained optimisation
# (maximise total score subject to entropy >= floor), not a feasibility search.
#
# Both constructive heuristics above are single-swap hill climbs, and both stop
# at the first feasible local optimum they reach. Measured on the epoch-24907
# pool (999 candidates, floor 0.25, unconstrained top-20 total 2.64543):
#
#   repair + rebuild, best of the two   total 2.42422   (gives up 0.22120)
#   annealed search, 8/8 restarts beat it  up to 2.46966   (gives up 0.17577)
#
# Every independent restart beat the hill climbs, from every starting point:
# they were leaving 0.045 of total score -- a fifth of everything they gave up
# -- on the table. The hill climbs pay for the floor by dropping to very deep,
# very exotic molecules (pool ranks 477-778); the annealed set instead spreads
# the cost over more mid-ranked ones (ranks up to 186) and reaches the same
# entropy for less score.
#
# So the constructive results are treated as SEEDS, and this search then
# maximises score along the floor. A swap costs one entropy evaluation (17.7 us
# measured; an incremental bit-delta version benchmarked SLOWER at 34.1 us
# because of the index union, so the straightforward vector form is used).
# Restart count and iteration budget from a sweep on the epoch-24907 pool,
# against a 468 s / 14-restart reference run that converged on 2.47067:
#
#   restarts  iters   total     time    headroom recovered over the seeds
#      2      250k   2.46827   10.5 s        94.8 %
#      4      250k   2.46827   21.1 s        94.8 %
#      4      400k   2.46645   33.7 s        90.9 %   <- longer runs are WORSE
#      6      250k   2.46900   32.1 s        96.4 %   <- chosen
#      8      250k   2.46900   42.9 s        96.4 %   <- no further gain
#
# More restarts beat longer restarts: the search converges well before 250k
# iterations and what it converges to depends on where it started, so spending
# the budget on more starting points is what finds the better sets.
ENTROPY_OPT_RESTARTS = 6
ENTROPY_OPT_ITERS = 250_000
ENTROPY_OPT_PAIR_TRIES = 60_000

# Wall-clock ceiling for the whole search, checked between restarts, so a slow
# box degrades to fewer restarts instead of overrunning the epoch. Only ever
# spent when the score-ordered set missed the floor, against an epoch of
# EPOCH_LENGTH * 12 s (~72 min) that has already spent ~10 s qualifying the
# candidate pool. NOTE: paid once PER WALLET.
ENTROPY_OPT_TIME_BUDGET = 60.0

# Fixed so a given pool always yields the same submission; the search is
# stochastic but the run is not.
ENTROPY_OPT_SEED = 0

# --- Last-resort donor tier: unconfirmed molecules for diversity only -------
#
# Every submitted set takes the pool's most distinctive chemistry with it, so a
# pool that is not being replenished goes CHEMICALLY flat long before it goes
# empty. Simulated on the epoch-24907 pool with hunter stalled (no new
# confirmations, 20 spent per epoch):
#
#   epoch 14: last submittable set
#   epoch 15: 719 molecules still available, but the best 20-subset of them
#             reaches only 0.2437 -- BELOW the 0.25 floor. Genuinely
#             infeasible: no algorithm submits from that pool.
#
# The molecules that break the deadlock are needed for their chemistry, not
# their score, and MIN_CONFIRMED_DRAWS exists to protect the SCORE. So when the
# confirmed pool cannot reach the floor, unconfirmed (single-draw) molecules are
# admitted as diversity donors only, under a hard cap. Measured on that dead
# epoch-15 pool:
#
#   0 donors -> ceiling 0.2434 (infeasible)    6 donors -> 0.2943
#   2 donors -> ceiling 0.2646 (FEASIBLE)     12 donors -> 0.3222
#
# Two donors are already enough to clear the floor, so the cap stays small: the
# submission keeps a confirmed, re-score-stable core and rents just enough
# chemistry to be valid at all. The alternative is submitting nothing.
ENTROPY_DONOR_ENABLED = True
ENTROPY_DONOR_POOL = 4000
MAX_UNCONFIRMED_PER_SET = 6

# Single-draw scores are optimistic: a molecule picked on one draw is picked
# partly for a lucky draw. Measured over the 1967 molecules in this DB that
# have >=3 draws, comparing first draw against the consensus of all draws:
#
#   all molecules            median +0.00167   mean +0.00585   p90 +0.01695
#   top 10% by first draw    median +0.00204   mean +0.01082   p90 +0.04879
#   top 2%  by first draw    median +0.00216   mean +0.01163   p90 +0.04992
#
# The submit band is the top 2%, so a donor's score is discounted by that
# band's mean optimism before the optimiser compares it with a confirmed
# molecule. Without this, unconfirmed molecules simply outbid the confirmed
# pool on raw score (their max was 0.1436 against 0.1404) and take over the
# submission -- measured 18 of 20 slots -- which is precisely the winner's
# curse MIN_CONFIRMED_DRAWS exists to prevent.
UNCONFIRMED_SCORE_HAIRCUT = 0.0116

# Minimum independent Boltz draws a molecule needs before it may be submitted.
#
# A Boltz score is a draw, not a property: the same molecule re-scored moves by
# a median 0.0053 and a p90 of 0.0163 across the replicate tables, which is
# WIDER than the whole #1-to-#20 span of the submittable band. Picking the top
# 20 on single draws therefore selects the luckiest draws, and they regress the
# moment the validator re-scores them -- the classic winner's curse.
#
# Measured on this DB: of the 20 molecules the score-ordered query returned,
# 18 had 3 draws and the 2 that did not landed at positions 16 and 20 with
# scores that were single-draw outliers. Requiring 3 draws removes exactly that
# failure.
#
# hunter's confirmation pass (--confirm-extra-rounds) is what produces the
# draws; a molecule that never cleared the confirm threshold never earns them
# and is not submittable under this rule.
MIN_CONFIRMED_DRAWS = 3

# Retries for the GitHub batch-commit flow (re-fetches branch head + retries
# the whole blob->tree->commit->ref-update sequence on conflict).
GITHUB_BATCH_MAX_RETRIES = 3

# ============================================================================

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from config.config_loader import load_config

try:
    MAX_SIMILARITY_TO_HISTORICAL = float(
        load_config()["max_similarity_to_historical"]
    )
except Exception as _e:
    print(f"[submit] could not read max_similarity_to_historical from config "
          f"({_e}); using {MAX_SIMILARITY_TO_HISTORICAL}")
from utils import (
    get_challenge_params_from_blockhash,
    get_historical_submissions,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from utils.molecules import (
    compute_fingerprint_entropy,
    molecule_unique_for_protein_hf,
    get_brenk_matches,
)
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

def _cfg_val(config: Any, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


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
        f"⚡ Concurrent CHAIN COMMIT (one persistent connection) | GitHub upload "
        f"batched into ONE commit after chain commit | "
        f"{_cfg_val(config, 'num_molecules', 20)} molecules/wallet | "
        f"HF unique + BRENK + similarity<{MAX_SIMILARITY_TO_HISTORICAL} + "
        f"set entropy≥{_cfg_val(config, 'min_entropy', 0.25)} on top "
        f"{AVAILABILITY_CANDIDATE_POOL} candidates"
    )
    bt.logging.info("="*70 + "\n")


# ============================================================================
# BITTENSOR SETUP
# ============================================================================

async def setup_bittensor_objects(
    config: argparse.Namespace
) -> Tuple[List[Any], List[Any], Any, Any, List[int], int]:
    """
    Initialize wallets/hotkeys and ONE persistent AsyncSubtensor connection.

    Why a single connection:
      * The public Finney RPC can reject rapid/repeated WebSocket handshakes
        with HTTP 429.
      * AsyncSubtensor/async-substrate-interface supports concurrent async RPC
        calls over one connection, so separate WebSockets are unnecessary here.
      * websocket_shutdown_timer=None keeps the connection alive between the
        6-second block polls instead of allowing an idle-close/reconnect cycle.

    `wallet_subtensors` is retained as a compatibility alias for the existing
    submission pipeline. Every entry intentionally points to the SAME shared
    AsyncSubtensor instance.

    Returns:
        (wallets_list, wallet_subtensors_list, shared_subtensor, metagraph,
         miner_uids_list, epoch_length)
    """
    bt.logging.info(
        "🔧 Setting up Bittensor objects with one persistent WebSocket connection..."
    )

    max_retries = 10
    retry_delay = 5
    max_retry_delay = 120

    for attempt in range(max_retries):
        subtensor = None
        try:
            bt.logging.info(
                f"   Attempting connection (attempt {attempt + 1}/{max_retries})..."
            )

            # IMPORTANT: keep this single AsyncSubtensor alive for the lifetime
            # of the miner.  Passing websocket_shutdown_timer=None prevents the
            # SDK from closing it during short idle periods between block polls.
            subtensor = bt.AsyncSubtensor(
                network=config.network,
                config=config,
                websocket_shutdown_timer=None,
            )
            await subtensor.initialize()

            # AsyncSubtensor.metagraph() already returns a synced metagraph.
            # Do NOT call metagraph.sync() again here; it is redundant traffic.
            metagraph = await subtensor.metagraph(config.netuid)
            bt.logging.info("   ✅ Persistent Subtensor connected")
            bt.logging.info("   ✅ Metagraph synced successfully\n")

            bt.logging.info(
                f"   📋 Initializing {len(WALLET_HOTKEY_PAIRS)} wallet/hotkey pairs:"
            )

            wallets: List[Any] = []
            miner_uids: List[int] = []

            for idx, (wallet_name, hotkey_name) in enumerate(
                WALLET_HOTKEY_PAIRS, 1
            ):
                label = f"{wallet_name}/{hotkey_name}"
                try:
                    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

                    # Verify the hotkey exists on disk before querying metagraph.
                    _ = wallet.hotkey

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
                f"{len(wallets)}/{len(WALLET_HOTKEY_PAIRS)} wallet/hotkey pairs"
            )

            # Keep the old list-based interface so the rest of the submission
            # code does not need invasive changes. All entries reference the
            # SAME persistent AsyncSubtensor. asyncio.gather() can still issue
            # the set_commitment calls concurrently.
            wallet_subtensors: List[Any] = [subtensor for _ in wallets]

            bt.logging.info(
                f"   🔌 Using ONE persistent WebSocket for reads and "
                f"{len(wallets)} concurrent wallet commit task(s)\n"
            )

            return (
                wallets,
                wallet_subtensors,
                subtensor,
                metagraph,
                miner_uids,
                EPOCH_LENGTH,
            )

        except Exception as e:
            # Always close a partially-created connection before retrying.
            if subtensor is not None:
                try:
                    await subtensor.close()
                except Exception:
                    pass

            error_text = str(e)
            is_rate_limited = (
                "HTTP 429" in error_text
                or "Too Many Requests" in error_text
                or "429" in error_text
            )

            if attempt >= max_retries - 1:
                bt.logging.error(
                    f"   ❌ Failed to initialize Bittensor after "
                    f"{max_retries} attempts: {e}"
                )
                bt.logging.error(traceback.format_exc())
                raise

            wait_time = min(
                max_retry_delay,
                retry_delay * (2 ** attempt),
            )

            # A 429 means the gateway explicitly asked us to slow connection
            # attempts. Give it a larger minimum cooldown than ordinary errors.
            if is_rate_limited:
                wait_time = max(30, wait_time)
                bt.logging.warning(
                    f"   ⚠️  Finney RPC WebSocket rate-limited this process "
                    f"(HTTP 429). Retrying in {wait_time}s using the same "
                    f"single-connection architecture..."
                )
            else:
                bt.logging.warning(
                    f"   ⚠️  Bittensor setup attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )

            await asyncio.sleep(wait_time)


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


def load_historical_fingerprints(target_protein: str):
    """
    Load historical submissions for the target protein once and precompute
    Morgan fingerprints for fast Tanimoto similarity comparisons.

    Returns:
        A DataFrame with a 'fingerprint' column, or None if no historical
        submissions exist for this target.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    historical_df = get_historical_submissions(target_protein, "molecules")

    if historical_df is None or historical_df.empty:
        bt.logging.warning(
            f"   ⚠️  No historical submissions found for target '{target_protein}'"
        )
        return None

    mols = [Chem.MolFromSmiles(smi) for smi in historical_df["SMILES"]]

    # Filter out any SMILES that failed to parse, keeping df/mols in sync
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    if len(valid_idx) != len(mols):
        bt.logging.warning(
            f"   ⚠️  Dropped {len(mols) - len(valid_idx)} unparsable historical "
            f"SMILES for target '{target_protein}'"
        )
        historical_df = historical_df.iloc[valid_idx].reset_index(drop=True)
        mols = [mols[i] for i in valid_idx]

    if not mols:
        return None

    fps = morgan_gen.GetFingerprints(mols, numThreads=8)
    historical_df = historical_df.copy()
    historical_df["fingerprint"] = list(fps)

    return historical_df


def is_diverse_enough(
    mol_fp,
    historical_df,
    max_similarity: float,
) -> bool:
    """
    Return True if no historical submission has Tanimoto similarity
    >= max_similarity to mol_fp. No historical data => treat as diverse.
    """
    if historical_df is None or historical_df.empty:
        return True

    similarities = DataStructs.BulkTanimotoSimilarity(
        mol_fp,
        list(historical_df["fingerprint"]),
    )
    return not any(sim >= max_similarity for sim in similarities)


def check_molecule_available(
    target_protein: str,
    molecule_name: str,
    smiles: str,
    historical_df,
    morgan_gen,
    max_similarity: float = MAX_SIMILARITY_TO_HISTORICAL,
) -> Tuple[bool, str]:
    """
    A molecule is available only if ALL of:
      1. NOT already in the HuggingFace dataset for the target protein, AND
      2. NOT disallowed by the BRENK structural-alert filter, AND
      3. NOT too similar (Tanimoto >= max_similarity) to any historical
         submission for that protein.

    Returns:
        (available, reason) where reason is '' on success, else a short cause.
    """
    if not target_protein:
        bt.logging.warning(
            "   ⚠️  No target protein provided for availability check"
        )
        return False, "no_target"

    try:
        # Step 1: exact-match uniqueness against HuggingFace
        if not bool(molecule_unique_for_protein_hf(target_protein, smiles)):
            return False, "hf_duplicate"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bt.logging.warning(
                f"   ⚠️  Could not parse SMILES for {molecule_name}"
            )
            return False, "bad_smiles"

        # Step 2: BRENK structural alerts — a single match makes the validator
        # discard the whole submission, so such a molecule is never submittable.
        brenk_reasons = get_brenk_matches(mol)
        if brenk_reasons:
            bt.logging.debug(
                f"   ⚠️  {molecule_name} disallowed by BRENK: "
                f"{'; '.join(brenk_reasons)}"
            )
            return False, "brenk"

        # Step 3: diversity vs historical submissions
        mol_fp = morgan_gen.GetFingerprint(mol)
        if not is_diverse_enough(mol_fp, historical_df, max_similarity):
            return False, "too_similar"

        return True, ""
    except Exception as e:
        bt.logging.error(
            f"   ❌ Availability check failed for {molecule_name}: {e}"
        )
        return False, "error"


def _smiles_inchikey(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


_ENTROPY_FP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=ENTROPY_FP_SIZE)
_ENTROPY_FP_CACHE: Dict[str, Any] = {}


def _entropy_fp(smiles: str):
    """Atom-pair fingerprint as a float vector, memoised per SMILES.

    The repair loop below evaluates tens of thousands of candidate sets. Calling
    compute_fingerprint_entropy() on SMILES for each one re-parses and
    re-fingerprints all twenty molecules every time: 11.6 ms a call against
    0.02 ms here, which is 5.7 minutes for a single swap inside a submission
    path that is racing the epoch boundary.
    """
    fp = _ENTROPY_FP_CACHE.get(smiles)
    if fp is None:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        fp = np.array(_ENTROPY_FP_GEN.GetFingerprint(mol), dtype=float)
        _ENTROPY_FP_CACHE[smiles] = fp
    return fp


def _entropy_from_counts(bit_counts, n_mols: int) -> float:
    """Mean per-bit binary entropy from summed fingerprint bits.

    Same arithmetic as compute_fingerprint_entropy(), expressed over a running
    bit-count vector so that swapping one molecule for another costs a single
    subtract and add instead of re-fingerprinting the set.
    """
    p = bit_counts / n_mols
    per_bit = np.where(
        (p > 0) & (p < 1),
        -p * np.log2(np.clip(p, 1e-12, 1.0))
        - (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1.0)),
        0.0,
    )
    return float(per_bit.mean())


def _set_entropy(entries: List[Tuple[str, float, str]]) -> Optional[float]:
    """Fast entropy of a candidate set of (name, score, smiles) triples.

    Divides by the number of parseable molecules, matching the validator's
    valid_mols count. Returns None when nothing in the set parses.
    """
    fps = [fp for fp in (_entropy_fp(e[2]) for e in entries) if fp is not None]
    if not fps:
        return None
    return _entropy_from_counts(np.sum(fps, axis=0), len(fps))


def _repair_entropy(
    selected: List[Tuple[str, float, str]],
    reserve: List[Tuple[str, float, str]],
    min_entropy: float,
) -> Tuple[List[Tuple[str, float, str]], Optional[float], int]:
    """Swap the cheapest molecules out of a set that misses the entropy floor.

    Entropy is pass/fail at the validator, never scored, so the goal is the
    highest-scoring set that clears the floor rather than the most diverse one.
    Each round therefore takes the swap that crosses the threshold for the
    smallest loss of score, and only falls back to the largest entropy gain
    while nothing crosses it yet.

    Returns (selected, entropy, swaps). `selected` keeps its score ordering
    loosely; the caller re-sorts nothing because the validator does not care.
    """
    fps_sel = [_entropy_fp(e[2]) for e in selected]
    if any(fp is None for fp in fps_sel):
        return selected, _set_entropy(selected), 0

    selected = list(selected)
    reserve = [r for r in reserve if _entropy_fp(r[2]) is not None]
    n = len(selected)
    counts = np.sum(fps_sel, axis=0)
    swaps = 0

    while swaps < ENTROPY_REPAIR_MAX_SWAPS and reserve:
        current = _entropy_from_counts(counts, n)
        if current >= min_entropy:
            break

        best_cross = None   # (score_loss, i, j) — clears the floor
        best_gain = None    # (entropy, i, j)    — best progress toward it
        for i in range(n):
            base = counts - fps_sel[i]
            for j, cand in enumerate(reserve):
                e = _entropy_from_counts(base + _entropy_fp(cand[2]), n)
                if e >= min_entropy:
                    loss = selected[i][1] - cand[1]
                    if best_cross is None or loss < best_cross[0]:
                        best_cross = (loss, i, j)
                if best_gain is None or e > best_gain[0]:
                    best_gain = (e, i, j)

        if best_cross is not None:
            _loss, i, j = best_cross
        elif best_gain is not None and best_gain[0] > current + 1e-12:
            _e, i, j = best_gain
        else:
            break

        new_fp = _entropy_fp(reserve[j][2])
        counts = counts - fps_sel[i] + new_fp
        selected[i], reserve[j] = reserve[j], selected[i]
        fps_sel[i] = new_fp
        swaps += 1

    return selected, _entropy_from_counts(counts, n), swaps


def _entropy_rebuild(
    pool: List[Tuple[str, float, str]],
    num_molecules: int,
    min_entropy: float,
) -> Tuple[Optional[List[Tuple[str, float, str]]], Optional[float]]:
    """Diversity-first fallback for when the swap repair stalls below the floor.

    _repair_entropy() is a steepest-ascent hill climb starting from the
    score-ordered top twenty, and a hill climb can stop at a local maximum: on
    epoch 24907 it ran out of improving single swaps at 0.2479 against a 0.25
    floor, with 39 of its 60 swaps unused. No swap budget rescues that, because
    the problem is the starting point, not the number of steps.

    So build from the other end. Greedily add whichever candidate maximises the
    entropy of the growing set -- ignoring score entirely -- which lands far
    ABOVE the floor (0.3124 on that epoch's pool), then trade diversity back for
    score: repeatedly swap in the highest-scoring candidate that keeps the set
    at or above the floor. Approaching the floor from above means every
    intermediate set is already valid, so this returns a submittable set
    whenever one exists in the pool.

    On epoch 24907 it finished at entropy 0.2502 with mean score 0.1211 --
    higher than the 0.1193 the uncapped swap repair reached -- in ~3 s.

    Returns (selected, entropy), or (None, None) when the pool cannot reach the
    floor at all.
    """
    pool = [c for c in pool if _entropy_fp(c[2]) is not None]
    if len(pool) < num_molecules:
        return None, None

    # Seed: greedy maximum entropy, started from the highest scorer so the
    # cheapest tie-breaks favour score.
    selected = [pool[0]]
    counts = _entropy_fp(pool[0][2]).copy()
    rest = pool[1:]
    while len(selected) < num_molecules:
        n_next = len(selected) + 1
        best = None
        for k, cand in enumerate(rest):
            e = _entropy_from_counts(counts + _entropy_fp(cand[2]), n_next)
            if best is None or e > best[0]:
                best = (e, k)
        counts = counts + _entropy_fp(rest[best[1]][2])
        selected.append(rest.pop(best[1]))

    n = num_molecules
    entropy = _entropy_from_counts(counts, n)
    if entropy < min_entropy:
        # The most diverse set the pool can build still misses the floor;
        # nothing else in here will do better.
        return None, None

    # Recovery: buy score back, one swap at a time, never dropping below the
    # floor. Candidates stay score-sorted so the inner scan can stop as soon as
    # it reaches molecules no better than the one it would replace.
    names = {e[0] for e in selected}
    cands = sorted(
        (c for c in pool if c[0] not in names), key=lambda c: -c[1]
    )
    for _round in range(ENTROPY_RECOVERY_MAX_SWAPS):
        best = None    # (score_gain, i, j)
        for i in range(n):
            base = counts - _entropy_fp(selected[i][2])
            for j, cand in enumerate(cands):
                gain = cand[1] - selected[i][1]
                if gain <= 0:
                    break
                if best is not None and gain <= best[0]:
                    break
                if _entropy_from_counts(base + _entropy_fp(cand[2]), n) >= min_entropy:
                    best = (gain, i, j)
                    break
        if best is None:
            break
        _gain, i, j = best
        counts = counts - _entropy_fp(selected[i][2]) + _entropy_fp(cands[j][2])
        selected[i], cands[j] = cands[j], selected[i]
        cands.sort(key=lambda c: -c[1])

    return selected, _entropy_from_counts(counts, n)


def _polish_single(idx, fps, scores, n, floor, donors=None, max_donors=None):
    """Best-improvement single swaps that never leave the feasible region.

    `fps`/`scores` are ordered by score descending, so the inner scan can stop
    as soon as it reaches a candidate no better than the member it would
    replace, or no better than the best swap already found.

    `donors`, when given, is a 0/1 array marking unconfirmed candidates; no swap
    may push their count in the set above `max_donors`.
    """
    cur = list(idx)
    counts = fps[cur].sum(axis=0)
    total = float(scores[cur].sum())
    m = len(scores)
    n_don = int(donors[cur].sum()) if donors is not None else 0

    while True:
        best = None    # (score_gain, i, j)
        inset = set(cur)
        for i in range(n):
            base = counts - fps[cur[i]]
            held = scores[cur[i]]
            for j in range(m):
                gain = scores[j] - held
                if gain <= 1e-12:
                    break
                if best is not None and gain <= best[0]:
                    break
                if j in inset:
                    continue
                if donors is not None and (
                    n_don - donors[cur[i]] + donors[j] > max_donors
                ):
                    continue
                if _entropy_from_counts(base + fps[j], n) >= floor:
                    best = (gain, i, j)
                    break
        if best is None:
            return cur, total
        gain, i, j = best
        counts = counts - fps[cur[i]] + fps[j]
        total += gain
        if donors is not None:
            n_don += int(donors[j]) - int(donors[cur[i]])
        cur[i] = j


def _polish_pair(idx, fps, scores, n, floor, rng, tries, donors=None, max_donors=None):
    """Randomised two-at-a-time swaps.

    Some improvements are unreachable one swap at a time: trading two mid-range
    molecules for one high scorer plus one diversity donor can raise the total
    while a set sitting exactly on the floor blocks either half on its own.
    """
    cur = list(idx)
    counts = fps[cur].sum(axis=0)
    total = float(scores[cur].sum())
    inset = set(cur)
    m = len(scores)
    n_don = int(donors[cur].sum()) if donors is not None else 0

    i_draw = rng.integers(0, n, size=(tries, 2))
    j_draw = rng.integers(0, m, size=(tries, 2))
    for t in range(tries):
        i1, i2 = int(i_draw[t, 0]), int(i_draw[t, 1])
        j1, j2 = int(j_draw[t, 0]), int(j_draw[t, 1])
        if i1 == i2 or j1 == j2 or j1 in inset or j2 in inset:
            continue
        gain = scores[j1] + scores[j2] - scores[cur[i1]] - scores[cur[i2]]
        if gain <= 1e-12:
            continue
        if donors is not None:
            new_don = (
                n_don - donors[cur[i1]] - donors[cur[i2]]
                + donors[j1] + donors[j2]
            )
            if new_don > max_donors:
                continue
        new_counts = (
            counts - fps[cur[i1]] - fps[cur[i2]] + fps[j1] + fps[j2]
        )
        if _entropy_from_counts(new_counts, n) < floor:
            continue
        inset.discard(cur[i1]); inset.discard(cur[i2])
        inset.add(j1); inset.add(j2)
        counts = new_counts
        total += float(gain)
        if donors is not None:
            n_don = int(new_don)
        cur[i1], cur[i2] = j1, j2
    return cur, total


def _anneal_at_floor(start, fps, scores, n, floor, iters, rng,
                     donors=None, max_donors=None):
    """Simulated annealing on score, with the floor as a ramped penalty.

    Feasible-only local search cannot cross the ridge: from a set sitting on the
    floor, every single swap that gains score drops entropy below it, so the
    climb stops. Letting the walk go briefly infeasible -- under a penalty that
    ramps up until only feasible sets survive -- is what reaches the sets the
    hill climbs cannot see. Only feasible states are ever recorded as best, so
    an infeasible walk can never be returned.

    The donor cap, unlike the floor, is a HARD constraint here: a move that
    would exceed it is refused outright rather than penalised. `start` must
    already satisfy it -- from a state that already exceeds the cap, every
    single swap still exceeds it and the walk would be frozen.
    """
    m = len(scores)
    cur = list(start)
    counts = fps[cur].sum(axis=0)
    total = float(scores[cur].sum())
    entropy = _entropy_from_counts(counts, n)
    inset = np.zeros(m, dtype=bool)
    inset[cur] = True
    n_don = int(donors[cur].sum()) if donors is not None else 0

    best_idx, best_total = (list(cur), total) if entropy >= floor else (None, -1e18)

    t_hi, t_lo = 0.02, 1e-5
    ratio = t_lo / t_hi
    chunk = 8192
    pos = chunk
    for it in range(iters):
        if pos >= chunk:
            r_u = rng.random(chunk)         # candidate draw
            r_k = rng.random(chunk)         # uniform vs score-biased
            r_i = rng.integers(0, n, chunk) # member to evict
            r_a = rng.random(chunk)         # acceptance
            pos = 0

        frac = it / iters
        temp = t_hi * ratio ** frac
        # Penalty ramps: early on the walk may dip below the floor to cross a
        # ridge, by the end nothing infeasible can survive.
        pen = 2.0 + 200.0 * frac

        u = r_u[pos]
        # fps/scores are score-sorted, so u*u concentrates draws on high
        # scorers while the uniform half keeps the exotic tail reachable.
        j = int(m * (u * u if r_k[pos] < 0.5 else u))
        i = int(r_i[pos])
        acc = r_a[pos]
        pos += 1

        if j >= m or inset[j]:
            continue

        out = cur[i]
        if donors is not None:
            new_don = n_don - int(donors[out]) + int(donors[j])
            if new_don > max_donors:
                continue
        new_counts = counts - fps[out] + fps[j]
        new_entropy = _entropy_from_counts(new_counts, n)
        new_total = total - scores[out] + scores[j]

        delta = (
            (new_total - pen * max(0.0, floor - new_entropy))
            - (total - pen * max(0.0, floor - entropy))
        )
        if delta <= 0 and acc >= np.exp(delta / temp):
            continue

        inset[out] = False
        inset[j] = True
        cur[i] = j
        counts, total, entropy = new_counts, float(new_total), new_entropy
        if donors is not None:
            n_don = new_don

        if entropy >= floor and total > best_total:
            best_idx, best_total = list(cur), total

    return best_idx, best_total


def _optimize_score_at_floor(
    pool: List[Tuple[str, float, str]],
    num_molecules: int,
    min_entropy: float,
    starts: List[List[Tuple[str, float, str]]],
    donor_names: Optional[set] = None,
    max_donors: Optional[int] = None,
) -> Tuple[Optional[List[Tuple[str, float, str]]], Optional[float], Dict[str, Any]]:
    """Highest-scoring set in `pool` that clears `min_entropy`.

    `starts` are the sets the constructive heuristics produced. They seed the
    search and, when one of them is already feasible, they also floor the
    result: the best feasible start is returned if nothing beats it, so this
    can only improve on _repair_entropy/_entropy_rebuild, never regress.

    An INFEASIBLE start is still useful. The annealer treats the floor as a
    penalty rather than a wall, so it can walk out of a region the hill climbs
    cannot escape and reach a feasible set from a start that misses the floor.
    That is the difference between skipping an epoch and submitting one, so
    starts are never rejected for being infeasible -- only the returned set has
    to clear the floor.

    `donor_names`/`max_donors` cap how many unconfirmed molecules the set may
    contain; every start must already satisfy that cap.

    Returns (selected, entropy, stats); (None, None, stats) if nothing feasible
    was found.
    """
    pool = [c for c in pool if _entropy_fp(c[2]) is not None]
    pool = sorted(pool, key=lambda c: -c[1])
    index_of = {c[0]: k for k, c in enumerate(pool)}

    fps = np.asarray([_entropy_fp(c[2]) for c in pool])
    scores = np.asarray([c[1] for c in pool], dtype=float)
    n = num_molecules
    m = len(pool)
    stats: Dict[str, Any] = {
        "opt_restarts": 0, "opt_gain": 0.0, "donors_used": 0,
        "opt_from_scratch": False,
    }

    donors = None
    if donor_names and max_donors is not None:
        donors = np.asarray(
            [1 if c[0] in donor_names else 0 for c in pool], dtype=int
        )
        plain = np.flatnonzero(donors == 0)
    else:
        plain = np.arange(m)

    start_idx: List[List[int]] = []
    for st in starts:
        try:
            idx = [index_of[c[0]] for c in st]
        except KeyError:
            continue
        if len(set(idx)) != n:
            continue
        if donors is not None and int(donors[idx].sum()) > max_donors:
            continue
        start_idx.append(idx)
    if not start_idx or m < n:
        return None, None, stats

    def feasible_total(idx):
        if _entropy_from_counts(fps[idx].sum(axis=0), n) < min_entropy:
            return None
        return float(scores[idx].sum())

    best, best_total = None, -1e18
    for idx in start_idx:
        t = feasible_total(idx)
        if t is not None and t > best_total:
            best, best_total = idx, t
    seed_best = best_total if best is not None else None

    rng = np.random.default_rng(ENTROPY_OPT_SEED)
    deadline = time.monotonic() + ENTROPY_OPT_TIME_BUDGET

    for r in range(ENTROPY_OPT_RESTARTS):
        if time.monotonic() >= deadline:
            break
        if r < len(start_idx) * 2:
            start = start_idx[r % len(start_idx)]
        else:
            # Random restarts draw from the confirmed candidates only, so the
            # start always satisfies the donor cap.
            source = plain if len(plain) >= n else np.arange(m)
            start = list(rng.choice(source, n, replace=False))
        idx, _t = _anneal_at_floor(
            start, fps, scores, n, min_entropy, ENTROPY_OPT_ITERS, rng,
            donors, max_donors,
        )
        if idx is None:
            continue
        idx, total = _polish_single(
            idx, fps, scores, n, min_entropy, donors, max_donors)
        idx, total = _polish_pair(
            idx, fps, scores, n, min_entropy, rng, ENTROPY_OPT_PAIR_TRIES,
            donors, max_donors)
        idx, total = _polish_single(
            idx, fps, scores, n, min_entropy, donors, max_donors)
        stats["opt_restarts"] = r + 1
        if total > best_total:
            best, best_total = idx, total

    if best is None:
        return None, None, stats

    entropy = _entropy_from_counts(fps[best].sum(axis=0), n)
    if entropy < min_entropy:
        # Cannot happen -- only feasible states are ever recorded -- but never
        # hand back a set the validator would discard.
        return None, None, stats
    if donors is not None:
        stats["donors_used"] = int(donors[best].sum())
    stats["opt_from_scratch"] = seed_best is None
    stats["opt_gain"] = 0.0 if seed_best is None else best_total - seed_best
    return [pool[k] for k in best], entropy, stats


def select_diverse_molecule_set(
    candidates: List[Tuple[str, float, str]],
    num_molecules: int,
    min_entropy: float,
    target_protein: str,
    historical_df,
    morgan_gen,
    max_similarity: float,
    donor_names: Optional[set] = None,
    max_donors: Optional[int] = None,
) -> Tuple[List[Tuple[str, float, str]], Dict[str, Any], List[Tuple[str, str]]]:
    """
    Greedily build a set of num_molecules from score-sorted candidates.

    Each pick must pass HF uniqueness + BRENK + historical diversity and must
    not be chemically identical to an already-picked molecule. Those are
    per-molecule tests, so they are applied as the set is built.

    Entropy is not, because it is not a per-molecule property. The validator
    computes it once over the complete submission and discards the whole
    submission when it falls below min_entropy (molecule_validity.py). Applying
    it to every growing prefix, as this function used to, asks a pair of
    molecules to clear a bar calibrated for twenty. Under MACCS keys that was
    slack enough not to matter; under atom-pair fingerprints no PAIR in the
    pool reaches 0.25 (best measured on rxn2: 0.2178), so the prefix rule would
    reject every candidate after the first and the miner would submit nothing.

    The set is therefore assembled on score, checked once when complete, and
    repaired by swapping when it misses the floor.

    Returns:
        (selected, stats, failed_availability) where selected is a list of
        (molecule_name, score, smiles) and failed_availability is a list of
        (molecule_name, reason) for HF/historical/smiles failures that should
        be marked unavailable in the DB.
    """
    selected: List[Tuple[str, float, str]] = []
    reserve: List[Tuple[str, float, str]] = []
    qualified_inchikeys: set = set()
    failed_availability: List[Tuple[str, str]] = []
    stats: Dict[str, Any] = {
        "checked": 0,
        "rejected_hf": 0,
        "rejected_brenk": 0,
        "rejected_similar": 0,
        "rejected_dup": 0,
        "rejected_bad_smiles": 0,
        "reserve": 0,
        "entropy_swaps": 0,
        "entropy": None,
        "rebuilt": False,
        "opt_restarts": 0,
        "opt_gain": 0.0,
        "donors_used": 0,
        "donor_cap_relaxed": False,
        "opt_from_scratch": False,
    }

    # Stays 0 unless the completed set fails entropy, so the common case scans
    # no further than it did before and pays for no reserve at all. Once a
    # repair IS needed, the reserve is the whole remaining pool by default
    # (ENTROPY_REPAIR_RESERVE = 0); a reserve truncated by score is what stalled
    # the repair below the floor on epoch 24907.
    reserve_target = 0
    full_reserve = ENTROPY_REPAIR_RESERVE or len(candidates)

    for molecule_name, score, smiles in candidates:
        if len(selected) >= num_molecules and len(reserve) >= reserve_target:
            break

        stats["checked"] += 1

        inchikey = _smiles_inchikey(smiles)
        if inchikey is None:
            stats["rejected_bad_smiles"] += 1
            continue
        if inchikey in qualified_inchikeys:
            stats["rejected_dup"] += 1
            continue

        is_available, reason = check_molecule_available(
            target_protein=target_protein,
            molecule_name=molecule_name,
            smiles=smiles,
            historical_df=historical_df,
            morgan_gen=morgan_gen,
            max_similarity=max_similarity,
        )

        if not is_available:
            if reason in (
                "hf_duplicate", "brenk", "too_similar",
                "bad_smiles", "no_target", "error",
            ):
                failed_availability.append((molecule_name, reason))
            if reason == "hf_duplicate":
                stats["rejected_hf"] += 1
            elif reason == "brenk":
                stats["rejected_brenk"] += 1
            elif reason == "too_similar":
                stats["rejected_similar"] += 1
            else:
                stats["rejected_bad_smiles"] += 1
            continue

        qualified_inchikeys.add(inchikey)
        entry = (molecule_name, score, smiles)

        if len(selected) < num_molecules:
            selected.append(entry)
            if len(selected) == num_molecules:
                # First moment the set is complete. Only a set that misses the
                # floor is worth gathering a reserve for.
                ent = _set_entropy(selected)
                if ent is not None and ent < min_entropy:
                    reserve_target = full_reserve
        else:
            reserve.append(entry)

    stats["reserve"] = len(reserve)

    if len(selected) == num_molecules:
        entropy = _set_entropy(selected)
        # In the donor tier the score-ordered top set can be all donors -- their
        # raw scores outbid the confirmed pool -- so it has to go through the
        # capped search even when its entropy already clears the floor.
        over_cap = (
            donor_names is not None and max_donors is not None
            and sum(1 for c in selected if c[0] in donor_names) > max_donors
        )
        if entropy is not None and (entropy < min_entropy or over_cap):
            # The score-ordered top set misses the floor, so some score has to
            # be given up. How much is the whole game: everything below exists
            # to give up as little as possible.
            pool = selected + reserve

            # The constructive seeds must respect the donor cap, so they are
            # built from the confirmed candidates alone; the search below is
            # what spends the donor allowance, and only as far as it must.
            if donor_names:
                seed_pool = [c for c in pool if c[0] not in donor_names]
            else:
                seed_pool = pool
            seeds: List[List[Tuple[str, float, str]]] = []

            if len(seed_pool) >= num_molecules:
                # Seed 1 -- cheapest swaps out of the score-ordered set. Can stop
                # at a local maximum below the floor (epoch 24907 did, at
                # 0.2479); it is kept as a starting point either way.
                repaired, _rep_entropy, swaps = _repair_entropy(
                    seed_pool[:num_molecules], seed_pool[num_molecules:],
                    min_entropy,
                )
                stats["entropy_swaps"] = swaps
                if repaired:
                    seeds.append(repaired)

                # Seed 2 -- built from maximum diversity and walked back down to
                # the floor. Returns None when even maximum diversity misses the
                # floor, which is exactly when the donor tier is needed.
                rebuilt, _reb_entropy = _entropy_rebuild(
                    seed_pool, num_molecules, min_entropy
                )
                if rebuilt is not None:
                    seeds.append(rebuilt)
                    stats["rebuilt"] = True

            # Search from every seed, feasible or not. Both seeds are single-swap
            # hill climbs that stop at the first local optimum they reach; this
            # searches along the floor for the highest-scoring feasible set, and
            # returns the best feasible seed if it finds nothing better -- so it
            # never regresses.
            best, best_entropy, opt_stats = _optimize_score_at_floor(
                pool, num_molecules, min_entropy, seeds,
                donor_names, max_donors,
            )
            stats.update(opt_stats)

            if (
                best is None and seeds and donor_names
                and max_donors is not None and max_donors < num_molecules
            ):
                # Last resort before losing the epoch: the donor cap protects
                # score quality, not validity, and a submission built on
                # single-draw scores still beats no submission at all. Only
                # reached when nothing under the cap clears the floor.
                best, best_entropy, opt_stats = _optimize_score_at_floor(
                    pool, num_molecules, min_entropy, seeds,
                    donor_names, num_molecules,
                )
                stats.update(opt_stats)
                if best is not None:
                    stats["donor_cap_relaxed"] = True

            if best is not None:
                selected, entropy = best, best_entropy
            elif seeds:
                # Nothing feasible exists here. Hand back the best effort; the
                # caller's authoritative gate refuses to submit it and, if a
                # donor tier is still available, retries with it.
                selected = max(seeds, key=lambda st: sum(e[1] for e in st))
                entropy = _set_entropy(selected)
        stats["entropy"] = entropy

    return selected, stats, failed_availability


def _load_donor_candidates(
    db_path: str, exclude: set, limit: int = ENTROPY_DONOR_POOL
) -> List[Tuple[str, float, str]]:
    """Unconfirmed (single-draw) molecules, score-discounted, for diversity only.

    Loaded lazily -- only when the confirmed pool cannot reach the entropy floor
    on its own -- because these molecules are exactly what MIN_CONFIRMED_DRAWS
    keeps out of a submission. Their scores are haircut by the measured
    single-draw optimism so that the optimiser only ever takes one when it buys
    entropy, never because it outbids a confirmed molecule on score.
    """
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT molecule_name, score
            FROM   scored_molecules
            WHERE  available = TRUE
            ORDER  BY score DESC
            LIMIT  ?
            """,
            (limit + len(exclude),),
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        bt.logging.error(f"   ❌ Could not load donor candidates: {e}")
        return []

    donors: List[Tuple[str, float, str]] = []
    for name, score in rows:
        if name in exclude:
            continue
        smiles = get_smiles_from_reaction(name)
        if smiles is None:
            continue
        donors.append((name, score - UNCONFIRMED_SCORE_HAIRCUT, smiles))
        if len(donors) >= limit:
            break
    return donors


async def get_verified_molecule_sets(
    n_wallets: int,
    molecules_per_wallet: int,
    min_entropy: float,
    db_path: str,
    target_protein: str,
    candidate_pool: int = AVAILABILITY_CANDIDATE_POOL,
    max_similarity: float = MAX_SIMILARITY_TO_HISTORICAL,
) -> List[List[Tuple[str, float]]]:
    """
    Build one diverse molecule set per wallet for epoch submission.

    1. Load top `candidate_pool` available molecules by score.
    2. For each wallet, greedily select `molecules_per_wallet` molecules that
       pass HF/historical checks and have no InChIKey duplicates within the
       set, then verify the completed set clears min_entropy.
    3. Molecules assigned to an earlier wallet are excluded from later wallets.
    4. Persist updated available flags for every candidate checked.
    """
    if n_wallets <= 0 or molecules_per_wallet <= 0:
        return []

    if not os.path.exists(db_path):
        bt.logging.error(f"   ❌ Database not found: {db_path}")
        return []

    if not ensure_available_column(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # molecule_replicates only exists once rescore/backfill has run. On a
        # search-only DB there is nothing to confirm against, and filtering on
        # a missing table would mean never submitting at all, so fall back to
        # the unfiltered query and say so loudly.
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='molecule_replicates' LIMIT 1"
        )
        have_replicates = cursor.fetchone() is not None

        if have_replicates and MIN_CONFIRMED_DRAWS > 1:
            # Count real draws rather than trusting scored_molecules.rescored:
            # that flag is also set on molecules that stopped at 2 draws, and
            # n_reps/mu are only populated by rescore.py's consensus commit,
            # which hunter's own confirmation pass does not call. The replicate
            # rows are the one place the draw count is always true.
            cursor.execute(
                """
                SELECT s.molecule_name, s.score
                FROM   scored_molecules s
                JOIN  (SELECT molecule_name
                       FROM   molecule_replicates
                       GROUP  BY molecule_name
                       HAVING COUNT(*) >= ?) r
                  ON   r.molecule_name = s.molecule_name
                WHERE  s.available = TRUE
                ORDER  BY s.score DESC
                LIMIT  ?
                """,
                (MIN_CONFIRMED_DRAWS, candidate_pool),
            )
        else:
            if not have_replicates:
                bt.logging.warning(
                    f"   ⚠️  {db_path} has no molecule_replicates table — "
                    f"cannot tell confirmed molecules from single draws. "
                    f"Submitting on unconfirmed scores; run rescore/backfill "
                    f"to populate it."
                )
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
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        bt.logging.error(
            f"   ❌ Database error while loading candidates: {e}"
        )
        return []

    if not rows:
        bt.logging.warning("   ⚠️  No candidate molecules found in database")
        return []

    bt.logging.info(
        f"   🔍 Building {n_wallets} set(s) of {molecules_per_wallet} molecules "
        f"from top {len(rows)} candidates "
        f"(>={MIN_CONFIRMED_DRAWS} Boltz draws + HF unique + historical "
        f"similarity<{max_similarity}, set entropy≥{min_entropy})"
    )
    needed = n_wallets * molecules_per_wallet
    if len(rows) < needed:
        bt.logging.warning(
            f"   ⚠️  only {len(rows)} molecule(s) have {MIN_CONFIRMED_DRAWS}+ "
            f"draws and are still available; {needed} are needed before the "
            f"HF/BRENK/novelty filters even run. hunter confirms molecules that "
            f"clear its --confirm-threshold, so a thin pool here means the "
            f"search is not reaching the frontier often enough."
        )

    historical_df = await asyncio.to_thread(
        load_historical_fingerprints, target_protein
    )
    hist_count = 0 if historical_df is None else len(historical_df)
    bt.logging.info(
        f"   📚 Historical submissions for diversity check: {hist_count}"
    )
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Pre-resolve SMILES for all candidates once.
    candidate_triples: List[Tuple[str, float, str]] = []
    for molecule_name, score in rows:
        smiles = get_smiles_from_reaction(molecule_name)
        if smiles is None:
            bt.logging.warning(
                f"      ❌ {molecule_name}: could not derive SMILES "
                f"→ available=FALSE"
            )
            _set_molecule_available(db_path, molecule_name, False)
            continue
        candidate_triples.append((molecule_name, score, smiles))

    wallet_sets: List[List[Tuple[str, float]]] = []
    used_names: set = set()
    donor_triples: Optional[List[Tuple[str, float, str]]] = None
    confirmed_names = {name for name, _s, _sm in candidate_triples}

    for wallet_idx in range(n_wallets):
        remaining = [
            (name, score, smiles)
            for name, score, smiles in candidate_triples
            if name not in used_names
        ]

        selected, stats, failed_availability = await asyncio.to_thread(
            select_diverse_molecule_set,
            remaining,
            molecules_per_wallet,
            min_entropy,
            target_protein,
            historical_df,
            morgan_gen,
            max_similarity,
        )

        # Tier 2 -- the confirmed pool cannot reach the floor on its own. Every
        # submitted set strips the pool of its most distinctive chemistry, so a
        # pool that is not being replenished goes flat while it is still large:
        # measured infeasible at epoch 15 with 719 molecules left. Rather than
        # skip the epoch, rent a capped number of unconfirmed molecules as
        # diversity donors. Only reached when the alternative is submitting
        # nothing at all.
        needs_donors = (
            len(selected) < molecules_per_wallet
            or stats.get("entropy") is None
            or stats["entropy"] < min_entropy
        )
        if needs_donors and ENTROPY_DONOR_ENABLED:
            if donor_triples is None:
                bt.logging.warning(
                    f"   ⚠️  Confirmed pool cannot reach entropy "
                    f"{min_entropy} — loading up to {ENTROPY_DONOR_POOL} "
                    f"unconfirmed molecules as diversity donors "
                    f"(max {MAX_UNCONFIRMED_PER_SET} per set, scores "
                    f"discounted by {UNCONFIRMED_SCORE_HAIRCUT})"
                )
                donor_triples = await asyncio.to_thread(
                    _load_donor_candidates, db_path, confirmed_names
                )
                bt.logging.info(
                    f"   🧬 Donor candidates available: {len(donor_triples)}"
                )
            donor_names = {
                name for name, _s, _sm in donor_triples
                if name not in used_names
            }
            if donor_names:
                combined = remaining + [
                    t for t in donor_triples if t[0] in donor_names
                ]
                combined.sort(key=lambda t: -t[1])
                selected, stats, failed_availability = await asyncio.to_thread(
                    select_diverse_molecule_set,
                    combined,
                    molecules_per_wallet,
                    min_entropy,
                    target_protein,
                    historical_df,
                    morgan_gen,
                    max_similarity,
                    donor_names,
                    MAX_UNCONFIRMED_PER_SET,
                )

        for name, reason in failed_availability:
            _set_molecule_available(db_path, name, False)
            if reason == "hf_duplicate":
                detail = "already known (HF)"
            elif reason == "brenk":
                detail = "disallowed by BRENK filter"
            elif reason == "too_similar":
                detail = f"too similar to historical (≥{max_similarity})"
            else:
                detail = reason or "failed availability check"
            bt.logging.info(
                f"      ❌ {name}: {detail} → available=FALSE"
            )

        if len(selected) < molecules_per_wallet:
            bt.logging.warning(
                f"   ⚠️  Wallet set {wallet_idx + 1}/{n_wallets}: only found "
                f"{len(selected)}/{molecules_per_wallet} molecules "
                f"(checked={stats['checked']}, HF dup={stats['rejected_hf']}, "
                f"BRENK={stats['rejected_brenk']}, "
                f"historical={stats['rejected_similar']}, "
                f"dup InChIKey={stats['rejected_dup']}, "
                f"bad SMILES={stats['rejected_bad_smiles']})"
            )
            continue

        # The SMILES the selector actually used. Re-deriving them here would
        # repeat work and would hand compute_fingerprint_entropy() a None for
        # any name that stopped resolving in between.
        final_smiles = [smiles for _name, _score, smiles in selected]

        # Authoritative gate: the same function, over the same SMILES, that the
        # validator will run. The cached-fingerprint path drives the search;
        # this is what decides whether we submit at all.
        #
        # Submitting a set that fails is strictly worse than submitting
        # nothing. The validator discards it whole, and mark_molecules_unavailable()
        # still burns all twenty on chain-commit success, so a known-bad
        # submission costs the molecules as well as the epoch.
        try:
            entropy = compute_fingerprint_entropy(final_smiles)
        except Exception as e:
            bt.logging.error(
                f"   ❌ Wallet set {wallet_idx + 1}/{n_wallets}: entropy check "
                f"failed ({e}). Not submitting this set."
            )
            continue

        if entropy < min_entropy:
            bt.logging.warning(
                f"   ⚠️  Wallet set {wallet_idx + 1}/{n_wallets}: entropy "
                f"{entropy:.4f} < {min_entropy}. No 20-molecule subset of this "
                f"pool clears the floor — not a search failure but a depleted "
                f"pool: the chemistry left is too alike, even after the donor "
                f"tier. The validator would discard this submission and the "
                f"molecules would be spent, so skipping. hunter must confirm "
                f"new molecules ({MIN_CONFIRMED_DRAWS}+ draws) to restore it."
            )
            continue

        molecule_names = [name for name, _score, _smiles in selected]
        used_names.update(molecule_names)

        for name, score, _smiles in selected:
            _set_molecule_available(db_path, name, True)
            bt.logging.info(
                f"      ✅ wallet {wallet_idx + 1} | {name:<30} "
                f"| Score: {score:.6f}"
            )

        if stats["entropy_swaps"] or stats.get("rebuilt"):
            total_score = sum(score for _n, score, _s in selected)
            donors_used = stats.get("donors_used", 0)
            if donors_used and stats.get("donor_cap_relaxed"):
                donor_note = (
                    f", ⚠️ DONOR CAP RELAXED: {donors_used} unconfirmed "
                    f"molecule(s) — nothing under the cap of "
                    f"{MAX_UNCONFIRMED_PER_SET} cleared the floor, so this set "
                    f"rests on single-draw scores"
                )
            elif donors_used:
                donor_note = (
                    f", {donors_used} unconfirmed donor(s) of "
                    f"{MAX_UNCONFIRMED_PER_SET} allowed"
                )
            else:
                donor_note = ""
            if stats.get("opt_from_scratch"):
                origin = (
                    f"no constructive seed reached the floor; found by "
                    f"{stats.get('opt_restarts', 0)} search restart(s)"
                )
            else:
                origin = (
                    f"+{stats.get('opt_gain', 0.0):.5f} from "
                    f"{stats.get('opt_restarts', 0)} search restart(s) over the "
                    f"best constructive seed"
                )
            repaired = (
                f", floor-constrained over a pool of "
                f"{stats['reserve'] + len(selected)} "
                f"(total score {total_score:.5f}, {origin}{donor_note})"
            )
        else:
            repaired = ""
        bt.logging.info(
            f"   📦 Wallet set {wallet_idx + 1}/{n_wallets}: "
            f"{len(selected)} molecules, entropy={entropy:.4f} "
            f"(min {min_entropy}){repaired}"
        )
        wallet_sets.append([(name, score) for name, score, _smiles in selected])

    if not wallet_sets:
        bt.logging.warning(
            "   ⚠️  No valid molecule sets found in the top candidate pool"
        )

    return wallet_sets


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
    candidate_products: List[str],
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
    num_molecules = _cfg_val(state['config'], 'num_molecules', 1)

    if not candidate_products:
        bt.logging.warning(f"      ⚠️  UID {miner_uid}: No candidate products")
        return None

    if len(candidate_products) != num_molecules:
        bt.logging.warning(
            f"      ⚠️  UID {miner_uid}: expected {num_molecules} molecules, "
            f"got {len(candidate_products)}"
        )
        return None

    if len(set(candidate_products)) != len(candidate_products):
        bt.logging.warning(
            f"      ⚠️  UID {miner_uid}: submission contains duplicate molecules"
        )
        return None

    try:
        # Decrypted payload format (validator + read_local_input_file):
        #   mol_name1,mol_name2,...|protein_seq1,protein_seq2,...
        # Small-molecule-only submissions use "~" as the sequences placeholder
        # (same convention as the original single-molecule "|~" format).
        message = f"{','.join(candidate_products)}|~"

        # Offload in case encrypt() is CPU-bound (timelock crypto usually is)
        encrypted_response = await asyncio.to_thread(
            state['bdt'].encrypt, miner_uid, message, current_block
        )

        content_str = str(encrypted_response)
        encoded_content = base64.b64encode(content_str.encode()).decode()
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"

        bt.logging.info(
            f"      🧪 Prepared {label} → {len(candidate_products)} molecules "
            f"(commit path: {commit_content})"
        )

        return {
            "wallet": wallet,
            "miner_uid": miner_uid,
            "molecule_names": candidate_products,
            "commit_content": commit_content,
            "encoded_content": encoded_content,
            "filename": filename,
            "label": label,
        }
    except Exception as e:
        bt.logging.error(f"      ❌ Prepare failed for {label}: {e}")
        bt.logging.error(traceback.format_exc())
        return None


# Substrate rejects a re-used nonce with InvalidTransaction::Stale, surfaced as
# error 1010 with this text. Matched on the message because the SDK flattens the
# chain error into a string by the time it reaches us.
_STALE_NONCE_MARKERS = (
    "transaction is outdated",
    "priority is too low",
    "stale",
)


def _extrinsic_ok(response: Any) -> Tuple[bool, Optional[str]]:
    """Did the extrinsic actually succeed?

    bittensor>=10 returns an ExtrinsicResponse dataclass, and it defines no
    __bool__ -- so a FAILED response is still truthy and `if response:` reports
    every rejected commit as a success. That is not cosmetic here: commit
    success is what mark_molecules_unavailable() keys on, so a rejected
    transaction used to burn all twenty molecules while nothing reached the
    chain. Read .success explicitly, and keep the bool path for older SDKs that
    really did return one.
    """
    if isinstance(response, bool):
        return response, None
    success = getattr(response, "success", None)
    message = getattr(response, "message", None)
    if success is None:
        # Unknown shape: fall back to truthiness rather than silently failing.
        return bool(response), message
    return bool(success), message


def _is_stale_nonce(message: Optional[str]) -> bool:
    if not message:
        return False
    text = str(message).lower()
    return any(marker in text for marker in _STALE_NONCE_MARKERS)


def _clear_nonce_cache(subtensor_conn: Any, wallet: Any) -> bool:
    """Drop the SDK's cached nonce for this hotkey.

    async-substrate-interface caches the account nonce per connection and then
    only ever does `self._nonces[addr] += 1`; it never re-reads the chain. This
    miner deliberately holds ONE AsyncSubtensor for its whole life to avoid RPC
    handshake throttling, so that counter has to stay correct across every
    epoch. Once it drifts below the chain's value -- one commit that never
    landed is enough -- every later epoch is rejected as outdated, which is
    exactly the failure this recovers from.
    """
    try:
        substrate = getattr(subtensor_conn, "substrate", None)
        clear = getattr(substrate, "clear_nonce_cache_for_account", None)
        if clear is None:
            return False
        clear(wallet.hotkey.ss58_address)
        return True
    except Exception as e:
        bt.logging.warning(f"      ⚠️  could not clear nonce cache: {e}")
        return False


async def commit_only(
    payload: Dict[str, Any],
    subtensor_conn: Any,
    state: Dict[str, Any],
) -> bool:
    """
    THE TIME-CRITICAL CALL. Nothing but the chain extrinsic happens here.
    All wallet commit coroutines may share the same persistent AsyncSubtensor;
    async-substrate-interface multiplexes concurrent requests on that transport.
    """
    label = payload["label"]
    miner_uid = payload["miner_uid"]

    bt.logging.info(f"      ⛓️  Committing on-chain for {label} (UID {miner_uid})...")

    for attempt in (1, 2):
        try:
            response = await subtensor_conn.set_commitment(
                wallet=payload["wallet"],
                netuid=state['config'].netuid,
                data=payload["commit_content"],
                wait_for_inclusion=True,      # need confirmed ordering
                wait_for_finalization=False,  # don't wait extra blocks
            )

            ok, message = _extrinsic_ok(response)
            if ok:
                bt.logging.info(
                    f"      ✅ Commit OK for {label} (UID {miner_uid})"
                    + (f" [attempt {attempt}]" if attempt > 1 else "")
                )
                return True

            bt.logging.error(
                f"      ❌ Commit REJECTED for {label} (UID {miner_uid}): "
                f"{message or 'no message'}"
            )
            # A stale nonce means the extrinsic never entered a block, so
            # re-signing it is safe -- and is the only way to recover without
            # dropping the epoch. Anything else is not retried blind.
            if attempt == 1 and _is_stale_nonce(message):
                if _clear_nonce_cache(subtensor_conn, payload["wallet"]):
                    bt.logging.warning(
                        f"      🔄 stale nonce for {label}; cache cleared, "
                        f"re-signing once"
                    )
                    continue
            return False

        except MetadataError as e:
            bt.logging.warning(
                f"      ⏳ MetadataError (rate limited) for {label} "
                f"(UID {miner_uid}): {e}"
            )
            return False
        except Exception as e:
            if attempt == 1 and _is_stale_nonce(str(e)):
                if _clear_nonce_cache(subtensor_conn, payload["wallet"]):
                    bt.logging.warning(
                        f"      🔄 stale nonce for {label} ({e}); cache "
                        f"cleared, re-signing once"
                    )
                    continue
            bt.logging.error(
                f"      ❌ Commit error for {label} (UID {miner_uid}): {e}"
            )
            bt.logging.error(traceback.format_exc())
            return False

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

    num_molecules = _cfg_val(cfg, "num_molecules", 1)
    min_entropy = _cfg_val(cfg, "min_entropy", 0.25)
    max_similarity = _cfg_val(
        cfg, "max_similarity_to_historical", MAX_SIMILARITY_TO_HISTORICAL
    )

    # ==========================================================
    # STEP 2: Build diverse molecule sets (HF + historical + set entropy)
    # ==========================================================
    bt.logging.info(
        "🔹 STEP 2/3: Molecule Selection "
        f"({num_molecules} per wallet, HF unique, BRENK-clean, "
        f"historical similarity<{max_similarity}, set entropy≥{min_entropy})"
    )
    bt.logging.info(f"   🗄️  Using score DB: {reaction_db_path}")
    bt.logging.info(
        f"   ⚡ Candidate pool size: {AVAILABILITY_CANDIDATE_POOL} "
        f"(need {num_pairs} wallet set(s) × {num_molecules} molecules)"
    )

    molecule_sets = await get_verified_molecule_sets(
        n_wallets=num_pairs,
        molecules_per_wallet=num_molecules,
        min_entropy=min_entropy,
        db_path=reaction_db_path,
        target_protein=target_protein,
        candidate_pool=AVAILABILITY_CANDIDATE_POOL,
        max_similarity=max_similarity,
    )

    if not molecule_sets:
        bt.logging.warning(
            f"   ⚠️  Built no complete molecule set for {allowed_reaction}. "
            "Skipping submission for this epoch.\n"
        )
        return

    if len(molecule_sets) < num_pairs:
        # Submit what was built rather than throwing the epoch away. Sets are
        # interchangeable across wallets -- they are paired positionally below
        # and every set is disjoint -- so a wallet that could not be filled
        # costs only its own submission, not everyone else's.
        bt.logging.warning(
            f"   ⚠️  Only built {len(molecule_sets)}/{num_pairs} complete "
            f"molecule set(s) for {allowed_reaction}. Submitting the "
            f"{len(molecule_sets)} that were built; "
            f"{num_pairs - len(molecule_sets)} wallet(s) sit this epoch out.\n"
        )

    bt.logging.info("")

    # ==========================================================
    # STEP 3a: PREPARE all payloads in parallel (no chain/GitHub calls)
    # ==========================================================
    bt.logging.info("🔹 STEP 3/3: Prepare → Parallel Commit → Batched Upload")
    bt.logging.info(
        f"   🧪 Preparing {len(molecule_sets)} payload(s) "
        f"({num_molecules} molecules each)..."
    )

    # Re-fetch block right before preparing, so the encrypted payload uses
    # the freshest block available before we race to commit.
    current_block = await state["subtensor"].get_current_block()

    prep_pairs: List[Tuple[Any, int, List[str], List[float]]] = []
    for idx, molecule_set in enumerate(molecule_sets):
        if idx >= len(state['wallets']):
            break
        wallet = state['wallets'][idx]
        miner_uid = state['miner_uids'][idx]
        molecule_names = [name for name, _score in molecule_set]
        molecule_scores = [score for _name, score in molecule_set]
        prep_pairs.append((wallet, miner_uid, molecule_names, molecule_scores))

    prep_tasks = [
        prepare_submission(wallet, miner_uid, molecule_names, state, current_block)
        for (wallet, miner_uid, molecule_names, _scores) in prep_pairs
    ]
    prep_results = await asyncio.gather(*prep_tasks, return_exceptions=True)

    payloads: List[Dict[str, Any]] = []
    set_scores: List[List[float]] = []
    for (wallet, miner_uid, molecule_names, scores), result in zip(prep_pairs, prep_results):
        if isinstance(result, Exception) or result is None:
            bt.logging.error(
                f"   ❌ Skipping UID {miner_uid} ({len(molecule_names)} molecules): "
                f"prepare failed"
            )
            continue
        payloads.append(result)
        set_scores.append(scores)

    if not payloads:
        bt.logging.warning("   ⚠️  No payloads prepared successfully. Skipping epoch.\n")
        return

    # ==========================================================
    # STEP 3b: FIRE ALL CHAIN COMMITS CONCURRENTLY — TIME CRITICAL
    # All wallets intentionally reuse ONE persistent AsyncSubtensor/WebSocket.
    # This avoids public-RPC handshake bursts / HTTP 429 while asyncio.gather()
    # still dispatches the independent set_commitment coroutines concurrently.
    # ==========================================================
    bt.logging.info(
        f"   ⚡ Firing {len(payloads)} chain commit(s) CONCURRENTLY "
        f"over one persistent WebSocket..."
    )

    # Re-read every hotkey's nonce from the chain for this epoch. The cache is
    # per-account, so clearing here cannot make two concurrent wallets collide
    # -- and it must happen BEFORE the gather, never inside it, because the
    # cache's increment is what keeps concurrent commits on one account in
    # order. One extra account_nextIndex RPC per wallet per epoch.
    for payload in payloads:
        _clear_nonce_cache(state['wallet_subtensors'][0], payload["wallet"])

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
                f"({len(payloads[idx]['molecule_names'])} molecules): {outcome}"
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
            f"({len(files_for_batch)} wallet file(s))"
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
        name
        for idx in range(len(payloads))
        if commit_ok[idx]
        for name in payloads[idx]["molecule_names"]
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
        molecule_names = payload["molecule_names"]
        scores = set_scores[idx]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        bt.logging.info(
            f"   {idx+1:>2}. {status} UID {miner_uid:>3} ({label:<22}) | "
            f"{len(molecule_names)} molecules | avg score: {avg_score:.6f}"
        )
        for mol_idx, (name, score) in enumerate(zip(molecule_names, scores), 1):
            bt.logging.info(
                f"        {mol_idx:>2}. {name:<30} | Score: {score:.6f}"
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
            'wallet_subtensors': wallet_subtensors,  # aliases to one persistent connection for concurrent commits
            'miner_uids':        miner_uids,
            'subtensor':         subtensor,           # one persistent connection for reads + commits
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
        # wallet_subtensors intentionally contains aliases to the same shared
        # object. Close each distinct AsyncSubtensor object exactly once.
        connections_to_close: List[Any] = []
        seen_connection_ids = set()

        for conn in [subtensor, *wallet_subtensors]:
            if conn is None:
                continue
            conn_id = id(conn)
            if conn_id in seen_connection_ids:
                continue
            seen_connection_ids.add(conn_id)
            connections_to_close.append(conn)

        for idx, conn in enumerate(connections_to_close, 1):
            try:
                await conn.close()
                bt.logging.info(
                    f"✅ Bittensor connection {idx}/{len(connections_to_close)} closed"
                )
            except Exception as e:
                bt.logging.debug(f"Connection close ignored: {e}")


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
