"""Data preprocessing utilities."""

from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from ..models.molecule_encoder import mol_to_graph_data
import torch


class MoleculePreprocessor:
    """Preprocessor for converting SMILES to graph data."""
    
    def __init__(self, sanitize=True):
        self.sanitize = sanitize
    
    def process_smiles(self, smiles):
        """
        Convert SMILES string to graph data.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Dict with keys 'x', 'edge_index', 'edge_attr'
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            if self.sanitize:
                Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
            
            x, edge_index, edge_attr = mol_to_graph_data(mol)
            
            return {
                'x': x,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
            }
        except Exception as e:
            raise ValueError(f"Error processing SMILES {smiles}: {str(e)}")
    
    def batch_process(self, smiles_list):
        """
        Process multiple SMILES and create a batch.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            Dict with keys 'x', 'edge_index', 'edge_attr', 'batch'
        """
        from torch_geometric.data import Data, Batch
        
        graph_data_list = []
        for smiles in smiles_list:
            graph_data = self.process_smiles(smiles)
            # Create PyTorch Geometric Data object
            data = Data(
                x=graph_data['x'],
                edge_index=graph_data['edge_index'],
                edge_attr=graph_data['edge_attr'],
            )
            graph_data_list.append(data)
        
        # Combine into batch
        batch = Batch.from_data_list(graph_data_list)
        
        return {
            'x': batch.x,
            'edge_index': batch.edge_index,
            'edge_attr': batch.edge_attr,
            'batch': batch.batch,
        }
