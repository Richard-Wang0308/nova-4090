"""Main BoltzPredictor model combining molecule and protein encoders - Target-only version."""

import torch
import torch.nn as nn
import logging
from .molecule_encoder import MoleculeEncoder
from .protein_encoder import ProteinEncoder

logger = logging.getLogger(__name__)


class BoltzPredictor(nn.Module):
    """
    Main model for predicting binding affinity scores - Target-only version.
    
    Architecture:
    - Shared affinity model: S(m, p) = f_θ(molecule, protein)
    - Final score: y = S(m, target)
    
    ✅ TARGET-ONLY: Simple regression, no antitarget, no competition term
    """
    
    def __init__(
        self,
        mol_hidden_dim=256,
        protein_embedding_dim=1280,  # ESM-2 650M
        interaction_dim=512,
        num_interaction_layers=3,
        dropout=0.1,
        protein_model_name="facebook/esm2_t33_650M_UR50D",
        protein_cache_dir=None,
    ):
        """
        Initialize BoltzPredictor model.
        
        Args:
            mol_hidden_dim: Molecule encoder hidden dimension
            protein_embedding_dim: Protein embedding dimension (ESM-2)
            interaction_dim: Interaction network hidden dimension
            num_interaction_layers: Number of interaction layers
            dropout: Dropout rate
            protein_model_name: ESM-2 model name
            protein_cache_dir: Directory to cache protein embeddings
        """
        super().__init__()
        
        # Encoders
        self.molecule_encoder = MoleculeEncoder(hidden_dim=mol_hidden_dim)
        self.protein_encoder = ProteinEncoder(
            model_name=protein_model_name,
            cache_dir=protein_cache_dir,
        )
        
        # Interaction head: molecule + protein -> binding affinity score
        input_dim = mol_hidden_dim + protein_embedding_dim
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, interaction_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Middle layers
        for _ in range(num_interaction_layers - 2):
            layers.append(nn.Linear(interaction_dim, interaction_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Final layer: output single score
        layers.append(nn.Linear(interaction_dim, 128))
        layers.append(nn.GELU())
        layers.append(nn.Linear(128, 1))
        
        self.interaction_head = nn.Sequential(*layers)
        
        logger.info(
            f"Initialized BoltzPredictor (Target-only) - "
            f"mol_hidden_dim={mol_hidden_dim}, "
            f"protein_embedding_dim={protein_embedding_dim}, "
            f"interaction_dim={interaction_dim}, "
            f"num_interaction_layers={num_interaction_layers}, "
            f"dropout={dropout}"
        )
    
    def forward(self, mol_data, target_seqs):
        """
        Forward pass - Target-only version.
        
        Args:
            mol_data: Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
            target_seqs: Target protein sequences (list of strings or single string)
        
        Returns:
            Dict with:
                - 'final_score': Predicted binding affinity [batch_size]
        """
        # Encode molecule
        mol_emb = self.molecule_encoder(
            mol_data['x'],
            mol_data['edge_index'],
            mol_data['edge_attr'],
            mol_data.get('batch', None),
        )
        
        # Encode target protein(s)
        target_emb = self.protein_encoder(target_seqs)
        
        # Align batch dimensions
        mol_emb, target_emb = self._align_batch_dims(mol_emb, target_emb)
        
        # Predict binding affinity: concatenate molecule and protein embeddings
        interaction_input = torch.cat([mol_emb, target_emb], dim=1)
        final_score = self.interaction_head(interaction_input)
        
        # ✅ FIXED: Squeeze to [batch_size]
        final_score = final_score.squeeze(-1)

        # ✅ TARGET-ONLY: final_score = target binding affinity
        return {
            'final_score': final_score,
        }
    
    def _align_batch_dims(self, mol_emb, protein_emb):
        """
        Align batch dimensions between molecule and protein embeddings.
        
        Args:
            mol_emb: Molecule embedding [batch_size, mol_hidden_dim]
            protein_emb: Protein embedding [batch_size, protein_embedding_dim]
        
        Returns:
            Aligned (mol_emb, protein_emb) tensors
        """
        # Ensure 2D tensors
        if mol_emb.dim() == 1:
            mol_emb = mol_emb.unsqueeze(0)
        if protein_emb.dim() == 1:
            protein_emb = protein_emb.unsqueeze(0)
        
        # Get batch sizes
        mol_batch_size = mol_emb.size(0)
        protein_batch_size = protein_emb.size(0)
        
        # Align if needed
        if mol_batch_size == 1 and protein_batch_size > 1:
            # Single molecule, multiple proteins: expand molecule
            mol_emb = mol_emb.expand(protein_batch_size, -1)
        elif protein_batch_size == 1 and mol_batch_size > 1:
            # Multiple molecules, single protein: expand protein
            protein_emb = protein_emb.expand(mol_batch_size, -1)
        elif mol_batch_size != protein_batch_size:
            raise ValueError(
                f"Batch size mismatch: molecule {mol_batch_size} vs protein {protein_batch_size}. "
                f"Cannot align batch dimensions."
            )
        
        return mol_emb, protein_emb
    
    def predict_affinity(self, mol_data, protein_seq):
        """
        Predict binding affinity for molecule-protein pair(s).
        
        Args:
            mol_data: Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
            protein_seq: Protein sequence (string or list of strings)
        
        Returns:
            Affinity score [batch_size]
        """
        mol_emb = self.molecule_encoder(
            mol_data['x'],
            mol_data['edge_index'],
            mol_data['edge_attr'],
            mol_data.get('batch', None),
        )
        
        protein_emb = self.protein_encoder(protein_seq)
        
        # Align batch dimensions
        mol_emb, protein_emb = self._align_batch_dims(mol_emb, protein_emb)
        
        # Predict affinity
        interaction_input = torch.cat([mol_emb, protein_emb], dim=1)
        affinity = self.interaction_head(interaction_input).squeeze(-1)
        
        return affinity
    
    def get_molecule_embedding(self, mol_data):
        """
        Get molecule embedding without protein interaction.
        
        Args:
            mol_data: Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
        
        Returns:
            Molecule embedding [batch_size, mol_hidden_dim]
        """
        mol_emb = self.molecule_encoder(
            mol_data['x'],
            mol_data['edge_index'],
            mol_data['edge_attr'],
            mol_data.get('batch', None),
        )
        return mol_emb
    
    def get_protein_embedding(self, protein_seq):
        """
        Get protein embedding without molecule interaction.
        
        Args:
            protein_seq: Protein sequence (string or list of strings)
        
        Returns:
            Protein embedding [batch_size, protein_embedding_dim]
        """
        protein_emb = self.protein_encoder(protein_seq)
        return protein_emb


# ============================================================================
# SINGLE-TARGET PREDICTOR: Optimized for single protein + single reaction
# ============================================================================

class SingleTargetAffinityPredictor(nn.Module):
    """
    Optimized model for single-target binding affinity prediction.
    
    Architecture:
    - Since protein is FIXED, we only learn: f_θ(molecule) → binding affinity
    - Much simpler and more effective than multi-target model
    - Better for competition scenarios with fixed target + reaction
    
    ✅ SINGLE-TARGET: Learns fixed target representation + molecule encoder
    ✅ FASTER: No protein encoding needed at inference time
    ✅ BETTER: Specializes to one protein's binding pocket
    """
    
    def __init__(
        self,
        mol_hidden_dim=256,
        target_embedding_dim=512,
        interaction_dim=512,
        num_layers=4,
        dropout=0.1,
    ):
        """
        Initialize single-target predictor.
        
        Args:
            mol_hidden_dim: Molecule encoder hidden dimension
            target_embedding_dim: Learnable target embedding dimension
            interaction_dim: Interaction network hidden dimension
            num_layers: Number of interaction layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Molecule encoder (same as BoltzPredictor)
        self.molecule_encoder = MoleculeEncoder(hidden_dim=mol_hidden_dim)
        
        # ✅ FIXED TARGET REPRESENTATION: Learnable parameter
        # This represents the fixed target protein in the competition
        self.target_representation = nn.Parameter(
            torch.randn(1, target_embedding_dim) * 0.01  # Small initialization
        )
        
        # Interaction head: molecule + fixed target -> affinity
        input_dim = mol_hidden_dim + target_embedding_dim
        layers = []
        
        # Build interaction network
        for i in range(num_layers):
            if i == 0:
                # First layer
                layers.append(nn.Linear(input_dim, interaction_dim))
            else:
                # Middle layers
                layers.append(nn.Linear(interaction_dim, interaction_dim))
            
            layers.append(nn.GELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layers
        layers.append(nn.Linear(interaction_dim, 128))
        layers.append(nn.GELU())
        layers.append(nn.Linear(128, 1))
        
        self.interaction_head = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._init_weights()
        
        logger.info(
            f"Initialized SingleTargetAffinityPredictor - "
            f"mol_hidden_dim={mol_hidden_dim}, "
            f"target_embedding_dim={target_embedding_dim}, "
            f"interaction_dim={interaction_dim}, "
            f"num_layers={num_layers}, "
            f"dropout={dropout}"
        )
    
    def _init_weights(self):
        """Initialize network weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for better convergence
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, mol_data):
        """
        Forward pass - Single-target version.
        
        Args:
            mol_data: Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
        
        Returns:
            Dict with:
                - 'final_score': Predicted binding affinity [batch_size]
        """
        # Encode molecule
        mol_emb = self.molecule_encoder(
            mol_data['x'],
            mol_data['edge_index'],
            mol_data['edge_attr'],
            mol_data.get('batch', None),
        )
        
        # ✅ FIXED TARGET: Use learnable target representation
        batch_size = mol_emb.size(0)
        target_emb = self.target_representation.expand(batch_size, -1)
        
        # Predict binding affinity
        interaction_input = torch.cat([mol_emb, target_emb], dim=1)
        final_score = self.interaction_head(interaction_input)
        
        # ✅ FIXED: Squeeze to [batch_size]
        final_score = final_score.squeeze(-1)
        
        return {
            'final_score': final_score,
        }
    
    def predict_affinity(self, mol_data):
        """
        Predict binding affinity for molecule(s).
        
        Args:
            mol_data: Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
        
        Returns:
            Affinity score [batch_size]
        """
        mol_emb = self.molecule_encoder(
            mol_data['x'],
            mol_data['edge_index'],
            mol_data['edge_attr'],
            mol_data.get('batch', None),
        )
        
        batch_size = mol_emb.size(0)
        target_emb = self.target_representation.expand(batch_size, -1)
        
        interaction_input = torch.cat([mol_emb, target_emb], dim=1)
        affinity = self.interaction_head(interaction_input).squeeze(-1)
        
        return affinity
    
    def get_molecule_embedding(self, mol_data):
        """
        Get molecule embedding without target interaction.
        
        Args:
            mol_data: Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
        
        Returns:
            Molecule embedding [batch_size, mol_hidden_dim]
        """
        mol_emb = self.molecule_encoder(
            mol_data['x'],
            mol_data['edge_index'],
            mol_data['edge_attr'],
            mol_data.get('batch', None),
        )
        return mol_emb
    
    def get_target_embedding(self):
        """
        Get learned target representation.
        
        Returns:
            Target embedding [1, target_embedding_dim]
        """
        return self.target_representation
    
    def set_target_embedding(self, embedding):
        """
        Set target embedding from external source (e.g., pre-computed protein embedding).
        
        Args:
            embedding: Target embedding tensor [1, target_embedding_dim]
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        with torch.no_grad():
            self.target_representation.copy_(embedding)
        
        logger.info(f"Target embedding updated: {embedding.shape}")


# ============================================================================
# FACTORY FUNCTION: Create model based on type
# ============================================================================

def create_model(
    model_type='multi_target',
    mol_hidden_dim=256,
    protein_embedding_dim=1280,
    interaction_dim=512,
    num_layers=3,
    dropout=0.1,
    protein_model_name="facebook/esm2_t33_650M_UR50D",
    protein_cache_dir=None,
    **kwargs
):
    """
    Factory function to create model.
    
    Args:
        model_type: 'multi_target' or 'single_target'
        mol_hidden_dim: Molecule encoder hidden dimension
        protein_embedding_dim: Protein embedding dimension (for multi_target)
        interaction_dim: Interaction network hidden dimension
        num_layers: Number of interaction layers
        dropout: Dropout rate
        protein_model_name: ESM-2 model name (for multi_target)
        protein_cache_dir: Directory to cache protein embeddings (for multi_target)
        **kwargs: Additional arguments
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'multi_target':
        return BoltzPredictor(
            mol_hidden_dim=mol_hidden_dim,
            protein_embedding_dim=protein_embedding_dim,
            interaction_dim=interaction_dim,
            num_interaction_layers=num_layers,
            dropout=dropout,
            protein_model_name=protein_model_name,
            protein_cache_dir=protein_cache_dir,
        )
    
    elif model_type == 'single_target':
        target_embedding_dim = kwargs.get('target_embedding_dim', 512)
        return SingleTargetAffinityPredictor(
            mol_hidden_dim=mol_hidden_dim,
            target_embedding_dim=target_embedding_dim,
            interaction_dim=interaction_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: 'multi_target', 'single_target'"
        )
