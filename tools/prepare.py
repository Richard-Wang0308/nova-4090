"""Script to continuously collect training data from competition API and
split it directly into per-reaction CSV files (data/rxn1.csv ~ data/rxn5.csv).

No intermediate data/mols.csv is created anymore.
"""

import argparse
import requests
import asyncio
import pandas as pd
import time
from datetime import datetime
import os
import sys
import logging
from typing import Optional, Dict, List

# Add BASE_DIR (nova-4090) to path
base_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, base_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EPOCH_DURATION = 12 * 361  # 12 seconds per block * 361 blocks = 4332 seconds
BLOCK_TIME = 12  # seconds per block
NUM_REACTIONS = 5


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Continuously collect training data from competition API and split by reaction type"
    )
    parser.add_argument('--start_epoch', type=int, default=23863, help='Starting epoch number to collect from')
    parser.add_argument('--end_epoch', type=int, default=23865, help='Ending epoch number to collect to')
    parser.add_argument('--metric', type=str, default='boltz', help='Metric type')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for rxn{1..5}.csv files')
    parser.add_argument('--time', type=int, default=1,
                       help='Remaining time in seconds until first collection. If None, collect immediately.')

    args = parser.parse_args()
    return args


def fetch_leaderboard_data(epoch: int, metric: str = 'boltz') -> Optional[dict]:
    """
    Fetch leaderboard data for a specific epoch.

    Args:
        epoch: Epoch number
        metric: Metric type (default: 'boltz')

    Returns:
        JSON response as dict, or None if failed
    """
    url = f"https://compound-api-staging.metanova-labs.ai/api/competitions/leaderboard/{epoch}/molecules"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error fetching epoch {epoch}: {e}")
        return None


def extract_training_samples(data: dict) -> List[dict]:
    """
    Extract training samples from API response.

    Args:
        data: JSON response from API

    Returns:
        List of training samples as dicts
    """
    samples = []

    leaderboard = data.get('data', [])
    if not leaderboard:
        logger.warning("No leaderboard data found")
        return samples

    for entry in leaderboard:
        final_score = entry.get('final_score')

        molecules = entry.get('molecules', [])
        if not molecules:
            continue

        for mol_entry in molecules:
            mol_name = mol_entry.get('name', '')
            if not mol_name:
                continue

            samples.append({
                'molecule_name': mol_name,
                'final_score': final_score,
            })

    return samples


def get_reaction_index(molecule_name: str) -> Optional[int]:
    """Determine which reaction (1-5) a molecule belongs to, based on its name prefix 'rxn:N:'."""
    for i in range(1, NUM_REACTIONS + 1):
        if molecule_name.startswith(f"rxn:{i}:"):
            return i
    return None


def get_rxn_output_path(output_dir: str, i: int) -> str:
    return os.path.join(output_dir, f"rxn{i}.csv")


def get_last_collected_epoch(output_dir: str) -> Optional[int]:
    """
    Get the last collected epoch by checking the max 'epoch' value across
    all existing rxn{1..5}.csv files (since we no longer keep a single mols.csv).

    Args:
        output_dir: Directory containing rxn{1..5}.csv files

    Returns:
        Last epoch number, or None if no files/data exist yet
    """
    last_epoch = None

    for i in range(1, NUM_REACTIONS + 1):
        path = get_rxn_output_path(output_dir, i)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if 'epoch' in df.columns and len(df) > 0:
                max_epoch = int(df['epoch'].max())
                if last_epoch is None or max_epoch > last_epoch:
                    last_epoch = max_epoch
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    if last_epoch is not None:
        logger.info(f"Last collected epoch across rxn files: {last_epoch}")
    else:
        logger.info("No previous rxn CSV files found; starting fresh")

    return last_epoch


def split_and_save_samples(samples: List[dict], output_dir: str):
    """
    Split samples by reaction type (rxn:N: prefix) and append each to its
    corresponding rxn{N}.csv file, deduplicating on (molecule_name, epoch).

    Args:
        samples: List of sample dicts (must include 'epoch')
        output_dir: Directory to write rxn{1..5}.csv files into
    """
    if not samples:
        return

    buckets: Dict[int, List[dict]] = {i: [] for i in range(1, NUM_REACTIONS + 1)}
    unmatched = 0

    for sample in samples:
        idx = get_reaction_index(sample['molecule_name'])
        if idx is None:
            unmatched += 1
            logger.debug(f"Unmatched molecule skipped: {sample['molecule_name']}")
            continue
        buckets[idx].append(sample)

    if unmatched:
        logger.warning(f"{unmatched} unmatched molecules skipped (no rxn:N: prefix)")

    for i in range(1, NUM_REACTIONS + 1):
        rows = buckets[i]
        if not rows:
            continue

        path = get_rxn_output_path(output_dir, i)
        df = pd.DataFrame(rows)

        columns_order = ['molecule_name', 'final_score', 'epoch']
        df = df[[col for col in columns_order if col in df.columns]]

        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            df = pd.concat([df_existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['molecule_name', 'epoch'], keep='last')

        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} total samples to {path} (+{len(rows)} new)")


def collect_single_epoch_data(epoch: int, metric: str, output_dir: str) -> int:
    """
    Collect training data for a single epoch and split/save it directly
    into the per-reaction CSV files.

    Args:
        epoch: Epoch number to collect
        metric: Metric type
        output_dir: Output directory for rxn{1..5}.csv

    Returns:
        Number of samples collected
    """
    data = fetch_leaderboard_data(epoch, metric)
    if not data:
        logger.debug(f"Failed to fetch data for epoch {epoch}")
        return 0

    samples = extract_training_samples(data)
    if not samples:
        logger.debug(f"No samples extracted from epoch {epoch}")
        return 0

    for sample in samples:
        sample['epoch'] = epoch

    logger.info(f"Epoch {epoch}: Extracted {len(samples)} samples")

    split_and_save_samples(samples, output_dir)

    return len(samples)


async def wait_for_remaining_time(remaining_seconds: int):
    """
    Wait for the specified remaining time in seconds.

    Args:
        remaining_seconds: Number of seconds to wait
    """
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        remaining = remaining_seconds - elapsed

        if remaining <= 0:
            logger.info("✓ Remaining time reached! Starting collection...")
            return

        if remaining > 60:
            logger.info(f"Waiting for collection... {int(remaining)} seconds remaining ({int(remaining/60)} minutes)")
            await asyncio.sleep(60)
        else:
            logger.info(f"Waiting for collection... {int(remaining)} seconds remaining")
            await asyncio.sleep(min(5, remaining))


async def collect_new_epochs(start_epoch: int, last_collected_epoch: Optional[int], latest_epoch: int,
                            metric: str, output_dir: str) -> int:
    """
    Collect data from all new epochs.

    Args:
        start_epoch: Starting epoch (used if no previous data exists)
        last_collected_epoch: Last epoch that was collected (or None)
        latest_epoch: Latest available epoch
        metric: Metric type
        output_dir: Output directory for rxn{1..5}.csv

    Returns:
        Total samples collected
    """
    total_samples = 0

    if last_collected_epoch is None:
        collection_start = start_epoch
        logger.info(f"First collection run: will collect from epoch {start_epoch} to {latest_epoch}")
    else:
        collection_start = last_collected_epoch + 1
        logger.info(f"Subsequent collection run: will collect from epoch {collection_start} to {latest_epoch}")

    if collection_start > latest_epoch:
        logger.info(f"No new epochs to collect (last collected: {last_collected_epoch}, latest: {latest_epoch})")
        return 0

    epochs_to_collect = list(range(collection_start, latest_epoch + 1))
    logger.info(f"Found {len(epochs_to_collect)} epochs to collect (from {epochs_to_collect[0]} to {epochs_to_collect[-1]})")

    for epoch in epochs_to_collect:
        samples = collect_single_epoch_data(epoch, metric, output_dir)
        total_samples += samples
        time.sleep(0.2)  # Rate limiting

    return total_samples


async def continuous_collection_loop(config: argparse.Namespace):
    """
    Main continuous collection loop that triggers at specified intervals.

    Args:
        config: Configuration arguments
    """
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Starting Continuous Training Data Collection & Splitting")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Metric: {config.metric}")
    logger.info(f"Start epoch: {config.start_epoch}")
    logger.info(f"Epoch duration: {EPOCH_DURATION} seconds ({EPOCH_DURATION/60:.1f} minutes)")
    logger.info("=" * 70)

    collection_count = 0

    if config.time is not None and config.time > 0:
        logger.info(f"First collection in {config.time} seconds ({config.time/60:.1f} minutes)")
        await wait_for_remaining_time(config.time)
    else:
        logger.info("Collecting immediately, then every epoch...")

    while True:
        try:
            collection_count += 1
            logger.info(f"\n[Collection #{collection_count}] Starting collection cycle")
            logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            last_collected_epoch = get_last_collected_epoch(config.output_dir)

            latest_epoch = config.end_epoch
            # latest_epoch = find_latest_epoch(search_start, config.metric)

            samples_collected = await collect_new_epochs(
                config.start_epoch,
                last_collected_epoch,
                latest_epoch,
                config.metric,
                config.output_dir
            )

            logger.info(f"[Collection #{collection_count}] Completed - {samples_collected} new samples collected\n")

            logger.info(f"Next collection in {EPOCH_DURATION} seconds ({EPOCH_DURATION/60:.1f} minutes)")
            await wait_for_remaining_time(EPOCH_DURATION)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 70)
            logger.info("Continuous collection interrupted by user")
            logger.info(f"Total collection cycles completed: {collection_count}")
            logger.info("=" * 70)
            break
        except Exception as e:
            logger.error(f"Error in continuous collection loop: {e}")
            logger.info("Retrying in 30 seconds...")
            await asyncio.sleep(30)


async def main_async(config: argparse.Namespace):
    """Main async function with continuous epoch collection."""
    try:
        await continuous_collection_loop(config)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise


def main():
    args = parse_arguments()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()