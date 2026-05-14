"""Script to prepare training data from competition API."""

import argparse
import requests
import pandas as pd
import json
import time
from tqdm import tqdm
import os
import sys
import logging
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProteinSequenceCache:
    """Cache for protein sequences to avoid repeated API calls."""
    
    def __init__(self, cache_file='cache/protein_sequences.json'):
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_sequence(self, protein_code):
        """Get sequence, using cache if available."""
        if protein_code in self.cache:
            return self.cache[protein_code]
        
        sequence = get_sequence_from_protein_code(protein_code)
        if sequence:
            self.cache[protein_code] = sequence
            self._save_cache()
        return sequence




def fetch_leaderboard_data(epoch: int, metric: str = 'boltz') -> dict:
    """
    Fetch leaderboard data for a specific epoch.
    
    Args:
        epoch: Epoch number
        metric: Metric type (default: 'boltz')
    
    Returns:
        JSON response as dict, or None if failed
    """
    url = f"https://dashboard-backend-multitarget.up.railway.app/api/leaderboard/{epoch}?metric={metric}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching epoch {epoch}: {e}")
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
    competition = data.get('competition', {})
    
    # Try multiple ways to get target/antitarget proteins
    target_proteins = competition.get('target_proteins', [])
    if not target_proteins:
        target_proteins = data.get('target_proteins', [])
    
    # antitarget_proteins = competition.get('antitarget_proteins', [])
    # if not antitarget_proteins:
    #     # Check if there's a single antitarget in the data
    #     if 'antitarget_proteins' in data:
    #         antitarget_proteins = data.get('antitarget_proteins', [])
    
    # if not target_proteins or not antitarget_proteins:

    # Process leaderboard entries
    leaderboard = data.get('leaderboard', [])
    if not leaderboard:
        logger.warning("No leaderboard data found")
        return samples
    
    for entry in leaderboard:
        # Get final score (try multiple field names for future compatibility)
        final_score = entry.get('boltz_score')
        
        # Get molecules
        molecules = entry.get('molecules', [])
        if not molecules:
            continue
        
        # Process each molecule
        for mol_entry in molecules:
            mol_name = mol_entry.get('name', '')
            if not mol_name:
                continue
            
            # # Get antitarget (may vary per epoch)
            # # For now, use the first antitarget, but we should handle multiple
            # if isinstance(antitarget_proteins, list):
            #     antitarget_code = antitarget_proteins[0] if antitarget_proteins else None
            # else:
            #     antitarget_code = antitarget_proteins
            
            # if not antitarget_code:
            #     continue
            
            # # Fetch antitarget sequence
            # antitarget_seq = protein_cache.get_sequence(antitarget_code)
            # if not antitarget_seq:
            #     logger.warning(f"Could not get sequence for antitarget {antitarget_code}, skipping")
            #     continue
            
            # Create training sample (save both codes and sequences)
            sample = {
                'molecule_name': mol_name,
                # 'antitarget_protein': antitarget_code,
                # 'antitarget_seq': antitarget_seq,
                'final_score': final_score,
            }
            samples.append(sample)
    
    return samples


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
    logger.info("Finding latest epoch...")
    
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
        time.sleep(0.2)
    
    # Refine by checking a few epochs ahead
    latest = low
    for offset in [1, 2, 5, 10, 20, 50, 100]:
        test_epoch = low + offset
        data = fetch_leaderboard_data(test_epoch, metric)
        if data:
            latest = test_epoch
        else:
            break
        time.sleep(0.2)
    
    logger.info(f"Latest epoch found: {latest}")
    return latest


def collect_training_data(start_epoch: int, end_epoch: int = None, metric: str = 'boltz', 
                         output_file: str = 'data/train.csv', resume: bool = True):
    """
    Collect training data from multiple epochs.
    
    Args:
        start_epoch: Starting epoch number (e.g., 19959)
        end_epoch: Ending epoch number (None for auto-detect latest)
        metric: Metric type
        output_file: Output CSV file path
        resume: If True, skip epochs already in output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing data if resuming
    existing_epochs = set()
    if resume and os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            if 'epoch' in df_existing.columns:
                existing_epochs = set(df_existing['epoch'].unique())
            logger.info(f"Found {len(existing_epochs)} existing epochs in {output_file}")
        except:
            logger.warning("Could not read existing file, starting fresh")
    
    # Determine end epoch if not specified
    if end_epoch is None:
        end_epoch = find_latest_epoch(start_epoch, metric)
    
    logger.info(f"Collecting data from epoch {start_epoch} to {end_epoch}")
    
    
    # Collect data
    all_samples = []
    failed_epochs = []
    total_samples_collected = 0
    
    # Process epochs
    for epoch in tqdm(range(start_epoch, end_epoch + 1), desc="Fetching epochs"):
        # Skip if already processed
        if epoch in existing_epochs:
            continue
        
        # Fetch data
        data = fetch_leaderboard_data(epoch, metric)
        if not data:
            failed_epochs.append(epoch)
            time.sleep(0.5)  # Rate limiting
            continue
        
        # Extract samples
        samples = extract_training_samples(data)
        
        if not samples:
            logger.warning(f"No samples extracted from epoch {epoch}")
            continue
        
        # Add epoch number to samples
        for sample in samples:
            sample['epoch'] = epoch
        
        all_samples.extend(samples)
        total_samples_collected += len(samples)
        logger.info(f"Epoch {epoch}: Extracted {len(samples)} samples")
        
        # Save periodically
        if len(all_samples) >= 1000:
            save_samples(all_samples, output_file, append=resume)
            all_samples = []  # Clear after saving
        
        time.sleep(0.3)  # Rate limiting
    
    # Save remaining samples
    if all_samples:
        save_samples(all_samples, output_file, append=resume)
        total_samples_collected += len(all_samples)
    
    logger.info(f"Collection complete. Total samples collected: {total_samples_collected}")
    if failed_epochs:
        logger.warning(f"Failed epochs: {failed_epochs[:10]}..." if len(failed_epochs) > 10 else f"Failed epochs: {failed_epochs}")


def save_samples(samples: list, output_file: str, append: bool = True):
    """Save samples to CSV file."""
    if not samples:
        return
    
    df = pd.DataFrame(samples)

    print(df.columns)
    # Reorder columns - save: molecule_name, target_protein, target_seq, antitarget_protein, antitarget_seq, final_score, epoch
    # columns_order = ['molecule_name', 'target_protein', 'target_seq', 'antitarget_protein', 'antitarget_seq', 'final_score', 'epoch']
    # Reorder columns - save: molecule_name, target_protein, target_seq, final_score, epoch
    columns_order = ['molecule_name', 'target_protein', 'target_seq', 'final_score', 'epoch']
    df = df[[col for col in columns_order if col in df.columns]]
    
    if append and os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df = pd.concat([df_existing, df], ignore_index=True)
        # Remove duplicates based on molecule_name, target_protein, antitarget_protein, epoch
        # df = df.drop_duplicates(subset=['molecule_name', 'target_protein', 'antitarget_protein', 'epoch'], keep='last')
        # Remove duplicates based on molecule_name, target_protein, epoch
        df = df.drop_duplicates(subset=['molecule_name', 'epoch'], keep='last')
        # df = df.drop_duplicates(subset=['molecule_name', 'target_protein', 'epoch'], keep='last')
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from competition API")
    parser.add_argument('--start_epoch', type=int, default=20377, help='Starting epoch number')
    parser.add_argument('--end_epoch', type=int, default=20515, help='Ending epoch number (None for auto-detect latest)')
    parser.add_argument('--metric', type=str, default='boltz', help='Metric type')
    parser.add_argument('--output', type=str, default='data/train1.csv', help='Output CSV file')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from existing file')
    args = parser.parse_args()
    
    collect_training_data(
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        metric=args.metric,
        output_file=args.output,
        resume=not args.no_resume,
    )


if __name__ == '__main__':
    main()