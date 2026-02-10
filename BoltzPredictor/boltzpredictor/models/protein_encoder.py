"""ESM-2 protein encoder with caching support."""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
import hashlib
import os
import pickle
import warnings
import logging

# Suppress the pooler weights warning (harmless - we don't use pooler)
warnings.filterwarnings("ignore", message=".*pooler.dense.*")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class ProteinEncoder(nn.Module):
    """Frozen ESM-2 protein encoder with embedding caching."""
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", cache_dir=None, max_length=1000):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load ESM-2 model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        # Create cache directory if specified
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, sequence):
        """Generate cache key from protein sequence."""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key):
        """Load embedding from cache if exists."""
        if self.cache_dir is None:
            return None
        
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key, embedding):
        """Save embedding to cache."""
        if self.cache_dir is None:
            return
        
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    
    def _truncate_sequence(self, sequence):
        """Truncate sequence if too long."""
        if len(sequence) > self.max_length:
            # Take first max_length residues
            sequence = sequence[:self.max_length]
        return sequence
    
    def forward(self, sequences, use_cache=True):
        """
        Encode protein sequences.
        
        Args:
            sequences: List of protein sequences (strings) or single string
            use_cache: Whether to use embedding cache
        
        Returns:
            Protein embeddings [batch_size, embedding_dim]
        """
        # Handle single sequence
        if isinstance(sequences, str):
            sequences = [sequences]
        
        embeddings = []
        device = next(self.model.parameters()).device
        
        for seq in sequences:
            # Check cache
            if use_cache:
                cache_key = self._get_cache_key(seq)
                cached_emb = self._load_from_cache(cache_key)
                if cached_emb is not None:
                    # Move cached embedding to device
                    if isinstance(cached_emb, torch.Tensor):
                        cached_emb = cached_emb.to(device)
                    else:
                        # If it's a numpy array or list, convert to tensor first
                        cached_emb = torch.tensor(cached_emb, dtype=torch.float32).to(device)
                    embeddings.append(cached_emb)
                    continue
            
            # Truncate if needed
            seq = self._truncate_sequence(seq)
            
            # Tokenize
            encoded = self.tokenizer(
                seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length + 2,  # +2 for special tokens
            )
            
            # Move to same device as model
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get embeddings (no gradient)
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Mean pool over sequence length (excluding special tokens)
                # First and last tokens are special tokens
                hidden_states = outputs.last_hidden_state
                # Remove special tokens: [CLS] and [SEP]
                if hidden_states.size(1) > 2:
                    pooled = hidden_states[:, 1:-1, :].mean(dim=1)
                else:
                    pooled = hidden_states.mean(dim=1)
            
            emb = pooled.squeeze(0)  # Keep on device, don't move to CPU
            embeddings.append(emb)
            
            # Save to cache (move to CPU for storage)
            if use_cache:
                emb_cpu = emb.cpu().detach()
                self._save_to_cache(cache_key, emb_cpu)
        
        # Stack embeddings (all on same device now)
        embeddings = torch.stack(embeddings)
        
        return embeddings
