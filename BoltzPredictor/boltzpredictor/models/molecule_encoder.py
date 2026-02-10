"""GINE-based molecule encoder with RDKit features."""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np


def get_atom_features(atom):
    """Extract atom features from RDKit atom object."""
    features = []
    
    # Atomic number (one-hot for common elements: C, N, O, F, P, S, Cl, Br, I, others)
    atomic_num = atom.GetAtomicNum()
    common_elements = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
    atomic_onehot = [1 if atomic_num == e else 0 for e in common_elements]
    atomic_onehot.append(1 if atomic_num not in common_elements else 0)
    features.extend(atomic_onehot)
    
    # Degree (0-5, one-hot)
    degree = min(atom.GetDegree(), 5)
    degree_onehot = [0] * 6
    degree_onehot[degree] = 1
    features.extend(degree_onehot)
    
    # Formal charge (normalized)
    features.append(atom.GetFormalCharge() / 4.0)
    
    # Aromatic flag
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    
    # Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2, other)
    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.HybridizationType.SP: 0,
        Chem.HybridizationType.SP2: 1,
        Chem.HybridizationType.SP3: 2,
        Chem.HybridizationType.SP3D: 3,
        Chem.HybridizationType.SP3D2: 4,
    }
    hyb_onehot = [0] * 6
    hyb_onehot[hyb_map.get(hyb, 5)] = 1
    features.extend(hyb_onehot)
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    """Extract bond features from RDKit bond object."""
    features = []
    
    # Bond type (one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC)
    bt = bond.GetBondType()
    bond_type_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    bt_onehot = [0] * 4
    bt_onehot[bond_type_map.get(bt, 0)] = 1
    features.extend(bt_onehot)
    
    # Conjugation
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    
    # Ring membership
    features.append(1.0 if bond.IsInRing() else 0.0)
    
    return np.array(features, dtype=np.float32)


def mol_to_graph_data(mol):
    """Convert RDKit molecule to PyTorch Geometric data format."""
    if mol is None:
        raise ValueError("Invalid molecule")
    
    # Get atom features
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    # Convert list of numpy arrays to single numpy array first (much faster)
    atom_features_array = np.array(atom_features, dtype=np.float32)
    x = torch.from_numpy(atom_features_array)
    
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Undirected graph
        bond_features = get_bond_features(bond)
        edge_features.append(bond_features)
        edge_features.append(bond_features)
    
    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        # Convert list of numpy arrays to single numpy array first (much faster)
        edge_features_array = np.array(edge_features, dtype=np.float32)
        edge_attr = torch.from_numpy(edge_features_array)
    
    return x, edge_index, edge_attr


class MoleculeEncoder(nn.Module):
    """GINE-based molecule encoder."""
    
    def __init__(self, atom_dim=24, bond_dim=6, hidden_dim=256, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial atom embedding
        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)
        
        # GINE layers
        self.gine_layers = nn.ModuleList()
        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gine_layers.append(
                GINEConv(nn_layer, train_eps=True, edge_dim=bond_dim)
            )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, atom_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, bond_dim]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Molecular embedding [batch_size, hidden_dim]
        """
        # Initial embedding
        h = self.atom_embedding(x)
        
        # GINE layers
        for gine, bn in zip(self.gine_layers, self.batch_norms):
            h = gine(h, edge_index, edge_attr)
            h = bn(h)
            h = torch.relu(h)
        
        # Global pooling
        if batch is None:
            # Single molecule
            h = h.mean(dim=0, keepdim=True)
        else:
            h = global_mean_pool(h, batch)
        
        # Output projection
        h = self.output_proj(h)
        
        return h
