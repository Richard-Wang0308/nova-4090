
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
# misses the entropy floor and has to be repaired. Only paid for when a repair
# is actually needed: measured on rxn2, 1 score-ordered top-20 window in 50
# falls below 0.25, and those are fixed by a single swap costing 0.000017 of
# average score.
ENTROPY_REPAIR_RESERVE = 80

# A set that cannot be brought over the floor in this many swaps is not going
# to be, and an unbounded search here would run into the epoch boundary.
ENTROPY_REPAIR_MAX_SWAPS = 8

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


def select_diverse_molecule_set(
    candidates: List[Tuple[str, float, str]],
    num_molecules: int,
    min_entropy: float,
    target_protein: str,
    historical_df,
    morgan_gen,
    max_similarity: float,
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
    }

    # Stays 0 unless the completed set fails entropy, so the common case scans
    # no further than it did before and pays for no reserve at all.
    reserve_target = 0

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
                    reserve_target = ENTROPY_REPAIR_RESERVE
        else:
            reserve.append(entry)

    stats["reserve"] = len(reserve)

    if len(selected) == num_molecules:
        entropy = _set_entropy(selected)
        if entropy is not None and entropy < min_entropy:
            selected, entropy, swaps = _repair_entropy(
                selected, reserve, min_entropy
            )
            stats["entropy_swaps"] = swaps
        stats["entropy"] = entropy

    return selected, stats, failed_availability


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
            break

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
            break

        if entropy < min_entropy:
            bt.logging.warning(
                f"   ⚠️  Wallet set {wallet_idx + 1}/{n_wallets}: entropy "
                f"{entropy:.4f} < {min_entropy} after "
                f"{stats['entropy_swaps']} repair swap(s) over a reserve of "
                f"{stats['reserve']}. The validator would discard this "
                f"submission and the molecules would be spent, so skipping."
            )
            break

        molecule_names = [name for name, _score, _smiles in selected]
        used_names.update(molecule_names)

        for name, score, _smiles in selected:
            _set_molecule_available(db_path, name, True)
            bt.logging.info(
                f"      ✅ wallet {wallet_idx + 1} | {name:<30} "
                f"| Score: {score:.6f}"
            )

        repaired = (
            f", repaired with {stats['entropy_swaps']} swap(s) from a reserve "
            f"of {stats['reserve']}" if stats["entropy_swaps"] else ""
        )
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

    if len(molecule_sets) < num_pairs:
        bt.logging.warning(
            f"   ⚠️  Only built {len(molecule_sets)}/{num_pairs} complete "
            f"molecule set(s) for {allowed_reaction}. "
            "Skipping submission for this epoch.\n"
        )
        return

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
