"""
Script to precompute protein embeddings using ESM-2.

Target-only version:
- Only precomputes target protein sequences
- No antitarget sequences
- Efficient batch processing
- Caching support
"""

import argparse
import pandas as pd
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from boltzpredictor.models.protein_encoder import ProteinEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProteinEmbeddingPrecomputer:
    """Precompute and cache protein embeddings using ESM-2."""
    
    def __init__(
        self,
        cache_dir='cache/protein_embeddings',
        model_name='facebook/esm2_t33_650M_UR50D',
        device='cuda',
        batch_size=32
    ):
        """
        Initialize precomputer.
        
        Args:
            cache_dir: Directory to save embeddings
            model_name: ESM-2 model name
            device: Device to use (cuda or cpu)
            batch_size: Batch size for processing
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        logger.info(f"Initializing ProteinEmbeddingPrecomputer")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.batch_size}\n")
        
        # Initialize encoder
        logger.info("Loading ESM-2 model...")
        self.encoder = ProteinEncoder(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir)
        )
        logger.info("✅ ESM-2 model loaded\n")
    
    
    def _get_embedding_path(self, sequence):
        """
        Get cache file path for a sequence.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Path to embedding file
        """
        # Create hash of sequence for filename
        seq_hash = hash(sequence) % (10 ** 8)
        return self.cache_dir / f"embedding_{seq_hash}.npy"
    
    
    def _embedding_exists(self, sequence):
        """Check if embedding is already cached."""
        return self._get_embedding_path(sequence).exists()
    
    
    def _load_embedding(self, sequence):
        """Load embedding from cache."""
        path = self._get_embedding_path(sequence)
        if path.exists():
            return np.load(path)
        return None
    
    
    def _save_embedding(self, sequence, embedding):
        """Save embedding to cache."""
        path = self._get_embedding_path(sequence)
        np.save(path, embedding)
    
    
    def precompute_embeddings(self, sequences, skip_existing=True):
        """
        Precompute embeddings for a list of sequences.
        
        Args:
            sequences: List of protein sequences
            skip_existing: Skip sequences that already have cached embeddings
        """
        logger.info("=" * 80)
        logger.info("Precomputing Protein Embeddings")
        logger.info("=" * 80)
        logger.info(f"Total sequences to process: {len(sequences):,}\n")
        
        # Filter sequences
        if skip_existing:
            new_sequences = [seq for seq in sequences if not self._embedding_exists(seq)]
            cached_sequences = len(sequences) - len(new_sequences)
            
            if cached_sequences > 0:
                logger.info(f"Found {cached_sequences:,} cached embeddings")
                logger.info(f"Need to compute: {len(new_sequences):,}\n")
            
            sequences = new_sequences
        
        if len(sequences) == 0:
            logger.info("✅ All embeddings already cached!")
            return
        
        # Process in batches
        total_processed = 0
        failed_sequences = []
        
        for batch_start in tqdm(
            range(0, len(sequences), self.batch_size),
            desc="Processing batches",
            total=(len(sequences) + self.batch_size - 1) // self.batch_size
        ):
            batch_end = min(batch_start + self.batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]
            
            try:
                # Compute embeddings for batch
                embeddings = self.encoder(batch_seqs)  # [batch_size, embedding_dim]
                
                # Save each embedding
                for seq, emb in zip(batch_seqs, embeddings):
                    emb_np = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb
                    self._save_embedding(seq, emb_np)
                    total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                failed_sequences.extend(batch_seqs)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Precomputation Summary")
        logger.info("=" * 80)
        logger.info(f"Successfully processed: {total_processed:,}")
        
        if failed_sequences:
            logger.warning(f"Failed sequences: {len(failed_sequences):,}")
            logger.warning(f"Failed sequences: {failed_sequences[:10]}")  # Show first 10
        else:
            logger.info(f"✅ All sequences processed successfully!")
        
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Total cached files: {len(list(self.cache_dir.glob('embedding_*.npy'))):,}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Precompute protein embeddings using ESM-2 (Target-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Precompute embeddings from CSV file
  python precompute_proteins.py \\
    --data_file data/molecules.csv \\
    --target_col target_seq \\
    --cache_dir cache/protein_embeddings \\
    --batch_size 32

  # Precompute with GPU
  python precompute_proteins.py \\
    --data_file data/molecules.csv \\
    --target_col target_seq \\
    --device cuda \\
    --batch_size 64

  # Precompute on CPU
  python precompute_proteins.py \\
    --data_file data/molecules.csv \\
    --target_col target_seq \\
    --device cpu \\
    --batch_size 16

  # Skip already cached embeddings
  python precompute_proteins.py \\
    --data_file data/molecules.csv \\
    --target_col target_seq \\
    --skip_existing
        """
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='CSV file with protein sequences'
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='target_seq',
        help='Column name for target sequences (default: target_seq)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/protein_embeddings',
        help='Cache directory for embeddings (default: cache/protein_embeddings)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/esm2_t33_650M_UR50D',
        help='ESM-2 model name (default: facebook/esm2_t33_650M_UR50D)'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip sequences that already have cached embeddings'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Protein Embedding Precomputation (Target-only)")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Verify input file
    if not os.path.exists(args.data_file):
        logger.error(f"❌ Data file not found: {args.data_file}")
        return 1
    
    logger.info(f"Loading data from: {args.data_file}")
    
    try:
        # Load data
        df = pd.read_csv(args.data_file)
        logger.info(f"✅ Loaded {len(df):,} rows\n")
        
        # Check column exists
        if args.target_col not in df.columns:
            logger.error(f"❌ Column '{args.target_col}' not found in CSV")
            logger.error(f"Available columns: {list(df.columns)}")
            return 1
        
        # Get unique sequences
        logger.info(f"Extracting unique sequences from column '{args.target_col}'...")
        target_seqs = df[args.target_col].dropna().unique().tolist()
        logger.info(f"✅ Found {len(target_seqs):,} unique target sequences\n")
        
        # Validate sequences
        invalid_seqs = [seq for seq in target_seqs if not isinstance(seq, str) or len(seq) == 0]
        if invalid_seqs:
            logger.warning(f"⚠️ Found {len(invalid_seqs)} invalid sequences (empty or non-string)")
            target_seqs = [seq for seq in target_seqs if isinstance(seq, str) and len(seq) > 0]
            logger.info(f"Valid sequences: {len(target_seqs):,}\n")
        
        # Precompute embeddings
        precomputer = ProteinEmbeddingPrecomputer(
            cache_dir=args.cache_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=int(args.batch_size)
        )
        
        precomputer.precompute_embeddings(
            target_seqs,
            skip_existing=args.skip_existing
        )
        
        logger.info("=" * 80)
        logger.info("🎉 Precomputation Complete!")
        logger.info("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during precomputation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
