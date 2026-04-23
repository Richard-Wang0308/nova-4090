"""Script to prepare training data from competition API - continuously collect every epoch."""

import argparse
import requests
import asyncio
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
import sys
import logging
from collections import defaultdict
from typing import Tuple, Any, Set

# Add BASE_DIR (nova-4090) to path
base_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, base_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EPOCH_DURATION = 12 * 361  # 12 seconds per block * 361 blocks = 4332 seconds
BLOCK_TIME = 12  # seconds per block

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Continuously collect training data from competition API")
    parser.add_argument('--start_epoch', type=int, default=22237, help='Starting epoch number to collect from')
    parser.add_argument('--end_epoch', type=int, default=22240, help='Ending epoch number to collect to')
    parser.add_argument('--metric', type=str, default='boltz', help='Metric type')
    parser.add_argument('--output', type=str, default='data/data.csv', help='Output CSV file')
    parser.add_argument('--time', type=int, default=1, 
                       help='Remaining time in seconds until first collection. If None, collect immediately.')
    
    args = parser.parse_args()
    return args


def fetch_leaderboard_data(epoch: int, metric: str = 'boltz') -> dict:
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


def extract_training_samples(data: dict) -> list:
    """
    Extract training samples from API response.
    
    Args:
        data: JSON response from API
    
    Returns:
        List of training samples as dicts
    """
    samples = []
    
    # Extract competition info
    competition = "Q63380"
    
    # Process leaderboard entries
    leaderboard = data.get('data', [])
    if not leaderboard:
        logger.warning("No leaderboard data found")
        return samples
    
    for entry in leaderboard:
        # Get final score
        final_score = entry.get('final_score')
        
        # Get molecules
        molecules = entry.get('molecules', [])
        if not molecules:
            continue
        
        # Process each molecule
        for mol_entry in molecules:
            mol_name = mol_entry.get('name', '')
            if not mol_name:
                continue
            
            # Create training sample
            sample = {
                'molecule_name': mol_name,
                'final_score': final_score,
            }
            samples.append(sample)
    
    return samples


def get_last_collected_epoch(output_file: str) -> int:
    """
    Get the last epoch number from the CSV file.
    
    Args:
        output_file: Output CSV file path
    
    Returns:
        Last epoch number, or None if file doesn't exist
    """
    if not os.path.exists(output_file):
        logger.info(f"Output file {output_file} does not exist yet")
        return None
    
    try:
        df = pd.read_csv(output_file)
        if 'epoch' in df.columns and len(df) > 0:
            last_epoch = int(df['epoch'].max())
            logger.info(f"Last collected epoch from CSV: {last_epoch}")
            return last_epoch
        else:
            logger.warning("No 'epoch' column found in CSV")
            return None
    except Exception as e:
        logger.warning(f"Could not read existing file: {e}")
        return None


def find_latest_epoch(start_epoch: int, metric: str = 'boltz', max_search: int = 5000) -> int:
    """
    Find the latest available epoch by binary search.
    
    Args:
        start_epoch: Starting epoch to search from
        metric: Metric type
        max_search: Maximum epochs to search ahead
    
    Returns:
        Latest epoch number found
    """
    logger.info(f"Finding latest epoch (searching from {start_epoch})...")
    
    # Binary search for the latest epoch
    low = start_epoch
    high = start_epoch + max_search
    
    # First, find an upper bound
    while high - low > 1:
        mid = (low + high) // 2
        data = fetch_leaderboard_data(mid, metric)
        if data:
            low = mid
        else:
            high = mid
        time.sleep(0.1)
    
    # Refine by checking a few epochs ahead
    latest = low
    for offset in [1, 2, 5, 10, 20, 50, 100]:
        test_epoch = low + offset
        data = fetch_leaderboard_data(test_epoch, metric)
        if data:
            latest = test_epoch
        else:
            break
        time.sleep(0.1)
    
    logger.info(f"Latest epoch found: {latest}")
    return latest


def collect_single_epoch_data(epoch: int, metric: str = 'boltz', 
                             output_file: str = 'data/mols.csv') -> int:
    """
    Collect training data for a single epoch.
    
    Args:
        epoch: Epoch number to collect
        metric: Metric type
        output_file: Output CSV file path
    
    Returns:
        Number of samples collected
    """
    # Fetch data
    data = fetch_leaderboard_data(epoch, metric)
    if not data:
        logger.debug(f"Failed to fetch data for epoch {epoch}")
        return 0
    
    # Extract samples
    samples = extract_training_samples(data)
    
    if not samples:
        logger.debug(f"No samples extracted from epoch {epoch}")
        return 0
    
    # Add epoch number to samples
    for sample in samples:
        sample['epoch'] = epoch
    
    logger.info(f"Epoch {epoch}: Extracted {len(samples)} samples")
    
    # Save samples
    save_samples(samples, output_file, append=True)
    
    return len(samples)


def save_samples(samples: list, output_file: str, append: bool = True):
    """Save samples to CSV file."""
    if not samples:
        return
    
    df = pd.DataFrame(samples)
    
    # Reorder columns - save: molecule_name, final_score, epoch
    columns_order = ['molecule_name', 'final_score', 'epoch']
    df = df[[col for col in columns_order if col in df.columns]]
    
    if append and os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df = pd.concat([df_existing, df], ignore_index=True)
        # Remove duplicates based on molecule_name and epoch
        df = df.drop_duplicates(subset=['molecule_name', 'epoch'], keep='last')
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} total samples to {output_file}")


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
            logger.info(f"✓ Remaining time reached! Starting collection...")
            return
        
        # Log every minute
        if remaining > 60:
            logger.info(f"Waiting for collection... {int(remaining)} seconds remaining ({int(remaining/60)} minutes)")
            await asyncio.sleep(60)
        else:
            logger.info(f"Waiting for collection... {int(remaining)} seconds remaining")
            await asyncio.sleep(min(5, remaining))


async def collect_new_epochs(start_epoch: int, last_collected_epoch: int, latest_epoch: int, 
                            metric: str, output_file: str) -> int:
    """
    Collect data from all new epochs.
    
    Args:
        start_epoch: Starting epoch (used if no CSV exists)
        last_collected_epoch: Last epoch that was collected (or None)
        latest_epoch: Latest available epoch
        metric: Metric type
        output_file: Output CSV file path
    
    Returns:
        Total samples collected
    """
    total_samples = 0
    
    # Determine starting epoch for this collection
    if last_collected_epoch is None:
        # First run: collect from start_epoch to latest_epoch
        collection_start = start_epoch
        logger.info(f"First collection run: will collect from epoch {start_epoch} to {latest_epoch}")
    else:
        # Subsequent runs: collect only new epochs after last_collected_epoch
        collection_start = last_collected_epoch + 1
        logger.info(f"Subsequent collection run: will collect from epoch {collection_start} to {latest_epoch}")
    
    # Check if there are new epochs to collect
    if collection_start > latest_epoch:
        logger.info(f"No new epochs to collect (last collected: {last_collected_epoch}, latest: {latest_epoch})")
        return 0
    
    epochs_to_collect = list(range(collection_start, latest_epoch + 1))
    logger.info(f"Found {len(epochs_to_collect)} epochs to collect (from {epochs_to_collect[0]} to {epochs_to_collect[-1]})")
    
    # Collect data from each epoch
    for epoch in epochs_to_collect:
        samples = collect_single_epoch_data(epoch, metric, output_file)
        total_samples += samples
        time.sleep(0.2)  # Rate limiting
    
    return total_samples


async def continuous_collection_loop(config: argparse.Namespace):
    """
    Main continuous collection loop that triggers at specified intervals.
    
    Args:
        config: Configuration arguments
    """
    os.makedirs(os.path.dirname(config.output) if os.path.dirname(config.output) else '.', exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Starting Continuous Training Data Collection")
    logger.info("=" * 70)
    logger.info(f"Output file: {config.output}")
    logger.info(f"Metric: {config.metric}")
    logger.info(f"Start epoch: {config.start_epoch}")
    logger.info(f"Epoch duration: {EPOCH_DURATION} seconds ({EPOCH_DURATION/60:.1f} minutes)")
    logger.info("=" * 70)
    
    collection_count = 0
    
    # Wait for first collection if time is specified
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
            
            # Get last collected epoch from CSV
            last_collected_epoch = get_last_collected_epoch(config.output)
            
            # Determine start epoch for binary search
            if last_collected_epoch is None:
                search_start = config.start_epoch
            else:
                search_start = last_collected_epoch
            
            # Find latest epoch
            latest_epoch = config.end_epoch
            # latest_epoch = find_latest_epoch(search_start, config.metric)
            
            # Collect all new epochs
            samples_collected = await collect_new_epochs(
                config.start_epoch,
                last_collected_epoch,
                latest_epoch,
                config.metric,
                config.output
            )
            
            logger.info(f"[Collection #{collection_count}] Completed - {samples_collected} new samples collected\n")
            
            # Schedule next collection
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
        # Start continuous collection loop
        await continuous_collection_loop(config)
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise


def main():
    args = parse_arguments()
    
    # Run async main
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()