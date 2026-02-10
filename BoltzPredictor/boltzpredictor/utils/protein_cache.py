"""Utilities for precomputing and caching protein embeddings."""

import os
from tqdm import tqdm
from ..models.protein_encoder import ProteinEncoder


class ProteinEmbeddingCache:
    """Helper class to precompute and cache protein embeddings."""
    
    def __init__(self, cache_dir, model_name="facebook/esm2_t33_650M_UR50D"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.encoder = ProteinEncoder(model_name=model_name, cache_dir=cache_dir)
    
    def precompute_embeddings(self, sequences, batch_size=32):
        """
        Precompute embeddings for a list of sequences.
        
        Args:
            sequences: List of unique protein sequences
            batch_size: Batch size for processing
        """
        device = next(self.encoder.model.parameters()).device
        
        # Filter sequences that are already cached
        uncached = []
        for seq in sequences:
            cache_key = self.encoder._get_cache_key(seq)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if not os.path.exists(cache_path):
                uncached.append(seq)
        
        if len(uncached) == 0:
            print("All sequences already cached.")
            return
        
        print(f"Precomputing embeddings for {len(uncached)} sequences...")
        
        # Process in batches
        for i in tqdm(range(0, len(uncached), batch_size)):
            batch = uncached[i:i + batch_size]
            _ = self.encoder(batch, use_cache=True)
        
        print("Precomputation complete.")
