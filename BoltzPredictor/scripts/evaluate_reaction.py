"""
Evaluate BoltzPredictor model on validation/test data (Single-Target Only).

Single-target version:
- Simple regression evaluation (NO protein sequences needed)
- MSE, RMSE, MAE, Correlation metrics
- Support for reaction SMILES conversion
- Comprehensive logging and error handling
- ✅ FIXED: Correct normalizer loading and denormalization
"""

import torch
import torch.nn as nn
import torch.serialization
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
import logging
import time
import os
import sqlite3
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from boltzpredictor.models import SingleTargetAffinityPredictor, create_model
from boltzpredictor.data import MoleculePreprocessor
from boltzpredictor.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ============================================================================
# Normalizer Functions (FIXED)
# ============================================================================

def load_normalizer(normalizer_path):
    """
    Load normalizer from JSON file.
    
    Args:
        normalizer_path: Path to normalizer.json
        
    Returns:
        Dictionary with normalizer parameters
        
    Raises:
        FileNotFoundError: If normalizer file not found
        ValueError: If normalizer format is invalid
    """
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Normalizer not found at {normalizer_path}")
    
    try:
        with open(normalizer_path, 'r') as f:
            normalizer = json.load(f)
        
        logger.info(f"✅ Normalizer loaded from {normalizer_path}")
        logger.info(f"   Method: {normalizer.get('method', 'unknown')}")
        
        # Validate normalizer format
        if normalizer.get('method') == 'standardization':
            required_keys = ['method', 'mean', 'std']
            if not all(k in normalizer for k in required_keys):
                raise ValueError(f"Missing keys in standardization normalizer: {required_keys}")
            
            logger.info(f"   Mean: {normalizer['mean']:.6f}")
            logger.info(f"   Std: {normalizer['std']:.6f}\n")
            
        elif normalizer.get('method') == 'minmax':
            required_keys = ['method', 'min', 'max']
            if not all(k in normalizer for k in required_keys):
                raise ValueError(f"Missing keys in minmax normalizer: {required_keys}")
            
            logger.info(f"   Min: {normalizer['min']:.6f}")
            logger.info(f"   Max: {normalizer['max']:.6f}\n")
        
        else:
            raise ValueError(f"Unknown normalization method: {normalizer.get('method')}")
        
        return normalizer
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in normalizer file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading normalizer: {e}")


def denormalize_predictions(predictions, normalizer):
    """
    Denormalize predictions using normalizer parameters.
    
    ✅ FIXED: Correct denormalization formula
    
    Args:
        predictions: numpy array of normalized predictions (Z-scores)
        normalizer: Dictionary with normalizer parameters
        
    Returns:
        Denormalized predictions in original scale
        
    Raises:
        ValueError: If normalizer format is invalid
    """
    if normalizer is None:
        logger.warning("⚠️ No normalizer provided, returning raw predictions")
        return predictions
    
    method = normalizer.get('method')
    
    if method == 'standardization':
        # ✅ CORRECT Z-score denormalization:
        # normalized = (original - mean) / std
        # original = normalized * std + mean
        
        mean = normalizer['mean']
        std = normalizer['std']
        
        denormalized = predictions * std + mean
        
        logger.debug(
            f"Denormalization (Standardization):\n"
            f"  Formula: denormalized = normalized * std + mean\n"
            f"  Mean: {mean:.6f}, Std: {std:.6f}\n"
            f"  Sample: {predictions[0]:.6f} → {denormalized[0]:.6f}"
        )
        
        return denormalized
    
    elif method == 'minmax':
        # ✅ CORRECT Min-Max denormalization:
        # normalized = (original - min) / (max - min)
        # original = normalized * (max - min) + min
        
        min_val = normalizer['min']
        max_val = normalizer['max']
        
        denormalized = predictions * (max_val - min_val) + min_val
        
        logger.debug(
            f"Denormalization (Min-Max):\n"
            f"  Formula: denormalized = normalized * (max - min) + min\n"
            f"  Min: {min_val:.6f}, Max: {max_val:.6f}\n"
            f"  Sample: {predictions[0]:.6f} → {denormalized[0]:.6f}"
        )
        
        return denormalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def verify_denormalization(predictions, targets, normalizer):
    """
    Verify denormalization is correct by checking value ranges.
    
    Args:
        predictions: Denormalized predictions
        targets: Target values
        normalizer: Normalizer parameters
    """
    logger.info("Verifying denormalization...")
    
    pred_min, pred_max = predictions.min(), predictions.max()
    target_min, target_max = targets.min(), targets.max()
    
    logger.info(f"  Predictions range: [{pred_min:.6f}, {pred_max:.6f}]")
    logger.info(f"  Targets range:     [{target_min:.6f}, {target_max:.6f}]")
    
    if normalizer.get('method') == 'standardization':
        expected_min = normalizer['mean'] - 3 * normalizer['std']
        expected_max = normalizer['mean'] + 3 * normalizer['std']
    elif normalizer.get('method') == 'minmax':
        expected_min = normalizer['min']
        expected_max = normalizer['max']
    else:
        return
    
    logger.info(f"  Expected range:    [{expected_min:.6f}, {expected_max:.6f}]")
    
    # Check if ranges are reasonable
    if pred_min < expected_min * 0.5 or pred_max > expected_max * 1.5:
        logger.warning(
            f"⚠️ Predictions outside expected range!\n"
            f"   This may indicate incorrect denormalization."
        )
    else:
        logger.info("✅ Denormalization verified - ranges look correct\n")


def safe_torch_load(path, map_location='cpu'):
    """Safely load PyTorch checkpoint with numpy scalar support (PyTorch 2.6+)."""
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    return torch.load(path, map_location=map_location, weights_only=False)


# ============================================================================
# Reaction Chemistry Functions
# ============================================================================

def get_reaction_info(rxn_id: int, db_path: str) -> tuple:
    """
    Get reaction SMARTS and role information from database.
    
    Args:
        rxn_id: Reaction ID
        db_path: Path to molecules.sqlite database
        
    Returns:
        Tuple of (smarts, roleA, roleB, roleC) or None
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT smarts, roleA, roleB, roleC FROM reactions WHERE rxn_id = ?",
            (rxn_id,)
        )
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error getting reaction info for rxn_id {rxn_id}: {e}")
        return None


def get_molecules(mol_ids: list, db_path: str) -> list:
    """
    Get molecules from database.
    
    Args:
        mol_ids: List of molecule IDs
        db_path: Path to molecules.sqlite database
        
    Returns:
        List of (smiles, role_mask) tuples
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        molecules = []
        
        for mol_id in mol_ids:
            cursor.execute(
                "SELECT smiles, role_mask FROM molecules WHERE mol_id = ?",
                (mol_id,)
            )
            result = cursor.fetchone()
            molecules.append(result)
        
        conn.close()
        return molecules
    except Exception as e:
        logger.error(f"Error getting molecules: {e}")
        return [None] * len(mol_ids)


def combine_triazole_synthons(azide_smiles: str, alkyne_smiles: str) -> str:
    """
    Combine azide and alkyne synthons to form triazole.
    
    Args:
        azide_smiles: SMILES with [1*] attachment point
        alkyne_smiles: SMILES with [2*] attachment point
        
    Returns:
        Product SMILES or None if reaction fails
    """
    try:
        m1 = Chem.RWMol(Chem.MolFromSmiles(azide_smiles))
        m2 = Chem.RWMol(Chem.MolFromSmiles(alkyne_smiles))
        
        if not m1 or not m2:
            logger.warning(f"Invalid SMILES in triazole synthesis")
            return None
        
        # Find attachment points
        a1 = next(
            (i for i, atom in enumerate(m1.GetAtoms())
             if atom.GetSymbol() == '*' and atom.GetIsotope() == 1),
            None
        )
        a2 = next(
            (i for i, atom in enumerate(m2.GetAtoms())
             if atom.GetSymbol() == '*' and atom.GetIsotope() == 2),
            None
        )
        
        if a1 is None or a2 is None:
            logger.warning("Missing attachment points in triazole synthesis")
            return None
        
        # Get neighbors of attachment points
        n1 = m1.GetAtomWithIdx(a1).GetNeighbors()[0].GetIdx()
        n2 = m2.GetAtomWithIdx(a2).GetNeighbors()[0].GetIdx()
        
        # Create combined molecule
        combined = Chem.RWMol(m1)
        atom_mapping = {}
        
        # Add alkyne atoms (except attachment point)
        for i, atom in enumerate(m2.GetAtoms()):
            if i != a2:
                atom_mapping[i] = combined.AddAtom(atom)
        
        # Add alkyne bonds (except those involving attachment point)
        for bond in m2.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a2 not in (begin_idx, end_idx):
                combined.AddBond(
                    atom_mapping[begin_idx],
                    atom_mapping[end_idx],
                    bond.GetBondType()
                )
        
        # Remove azide attachment point and connect
        combined.RemoveAtom(a1)
        n1_adj = n1 - (1 if n1 > a1 else 0)
        n2_adj = atom_mapping[n2] - (1 if atom_mapping[n2] > a1 else 0)
        combined.AddBond(n1_adj, n2_adj, Chem.BondType.SINGLE)
        
        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
        
    except Exception as e:
        logger.error(f"Error in triazole synthesis: {e}")
        return None


def perform_smarts_reaction(smiles1: str, smiles2: str, smarts: str) -> str:
    """
    Perform SMARTS-based reaction.
    
    Args:
        smiles1: First reactant SMILES
        smiles2: Second reactant SMILES
        smarts: Reaction SMARTS
        
    Returns:
        Product SMILES or None if reaction fails
    """
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if not mol1 or not mol2:
            logger.warning(f"Invalid SMILES in SMARTS reaction")
            return None
        
        products = rxn.RunReactants((mol1, mol2))
        return Chem.MolToSmiles(products[0][0]) if products else None
        
    except Exception as e:
        logger.error(f"Error in SMARTS reaction: {e}")
        return None


def validate_and_order_reactants(
    smiles1: str,
    smiles2: str,
    role_mask1: int,
    role_mask2: int,
    roleA: int,
    roleB: int,
    smiles3: str = None,
    role_mask3: int = None,
    roleC: int = None
) -> tuple:
    """
    Validate reactants can react and return in correct order.
    
    Args:
        smiles1, smiles2: Reactant SMILES
        role_mask1, role_mask2: Role masks for reactants
        roleA, roleB: Expected roles
        smiles3, role_mask3, roleC: Optional third reactant
        
    Returns:
        Ordered reactants or None if validation fails
    """
    try:
        if smiles3 is None:
            # Check if reactants match roles
            can_react = (
                (((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)) or
                (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
            )
            
            if not can_react:
                return None, None
            
            # Order reactants based on roles
            if ((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB):
                return smiles1, smiles2
            else:
                return smiles2, smiles1
        
        else:
            # Check if first two molecules are valid for their roles
            can_react_12 = (
                (((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)) or
                (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
            )
            can_react_3 = (role_mask3 & roleC) == roleC
            
            if not can_react_12 or not can_react_3:
                return None, None, None
            
            # Order first two reactants based on roles
            if (role_mask1 & roleA) and (role_mask2 & roleB):
                return smiles1, smiles2, smiles3
            else:
                return smiles2, smiles1, smiles3
            
    except Exception as e:
        logger.error(f"Error validating reactants: {e}")
        return (None, None) if smiles3 is None else (None, None, None)


def react_molecules(rxn_id: int, mol1_id: int, mol2_id: int, db_path: str) -> str:
    """
    Perform 2-component reaction.
    
    Args:
        rxn_id: Reaction ID
        mol1_id, mol2_id: Molecule IDs
        db_path: Path to database
        
    Returns:
        Product SMILES or None
    """
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id], db_path)
        
        if not reaction_info or not all(molecules):
            return None
            
        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2) = molecules
        
        reactant1, reactant2 = validate_and_order_reactants(
            smiles1, smiles2, role_mask1, role_mask2, roleA, roleB
        )
        
        if not reactant1 or not reactant2:
            return None
            
        if rxn_id == 1:  # Triazole synthesis
            return combine_triazole_synthons(reactant1, reactant2)
        else:  # SMARTS-based reactions
            return perform_smarts_reaction(reactant1, reactant2, smarts)
        
    except Exception as e:
        logger.error(f"Error reacting molecules {mol1_id}, {mol2_id}: {e}")
        return None


def react_three_components(
    rxn_id: int,
    mol1_id: int,
    mol2_id: int,
    mol3_id: int,
    db_path: str
) -> str:
    """
    Perform 3-component reaction.
    
    Args:
        rxn_id: Reaction ID
        mol1_id, mol2_id, mol3_id: Molecule IDs
        db_path: Path to database
        
    Returns:
        Product SMILES or None
    """
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id, mol3_id], db_path)
        
        if not reaction_info or not all(molecules):
            return None
            
        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2), (smiles3, role_mask3) = molecules
        
        validation_result = validate_and_order_reactants(
            smiles1, smiles2, role_mask1, role_mask2, roleA, roleB,
            smiles3, role_mask3, roleC
        )
        
        if not all(validation_result):
            return None
        
        reactant1, reactant2, reactant3 = validation_result
        
        if rxn_id == 3:  # click_amide_cascade
            # Triazole formation
            triazole_cooh = combine_triazole_synthons(reactant1, reactant2)
            if not triazole_cooh:
                return None
            
            # Amide coupling
            amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
            return perform_smarts_reaction(triazole_cooh, reactant3, amide_smarts)
        
        if rxn_id == 5:  # suzuki_bromide_then_chloride (two-step cascade)
            suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"

            # First couple at bromide
            intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
            if not intermediate:
                return None

            # Then couple at chloride
            final_product = perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts)
            return final_product
        
        return None
        
    except Exception as e:
        logger.error(f"Error in 3-component reaction {mol1_id}, {mol2_id}, {mol3_id}: {e}")
        return None


def get_smiles_from_reaction(product_name: str, db_path: str) -> str:
    """
    Handle reaction format: rxn:reaction_id:mol1_id:mol2_id or rxn:reaction_id:mol1_id:mol2_id:mol3_id
    
    Args:
        product_name: Reaction identifier string
        db_path: Path to molecules.sqlite database
    
    Returns:
        SMILES string or None if reaction fails
    """
    try:
        parts = product_name.split(":")
        
        if len(parts) == 4:
            _, rxn_id, mol1_id, mol2_id = parts
            rxn_id, mol1_id, mol2_id = int(rxn_id), int(mol1_id), int(mol2_id)
            return react_molecules(rxn_id, mol1_id, mol2_id, db_path)
            
        elif len(parts) == 5:
            _, rxn_id, mol1_id, mol2_id, mol3_id = parts
            rxn_id, mol1_id, mol2_id, mol3_id = int(rxn_id), int(mol1_id), int(mol2_id), int(mol3_id)
            return react_three_components(rxn_id, mol1_id, mol2_id, mol3_id, db_path)
            
        else:
            logger.error(f"Invalid reaction format: {product_name}")
            return None
        
    except Exception as e:
        logger.error(f"Error in combinatorial reaction {product_name}: {e}")
        return None


# ============================================================================
# Evaluator Class (Single-Target Only)
# ============================================================================

class ReactionEvaluator:
    """Evaluator for single-target reaction prediction model."""
    
    def __init__(self, checkpoint_path, device='cuda', db_path=None, normalizer_path=None):
        """
        Initialize evaluator by loading checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (cuda or cpu)
            db_path: Path to molecules.sqlite database
            normalizer_path: Path to normalizer.json (optional, will search if not provided)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup database path
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), '..', '..',
                'combinatorial_db', 'molecules.sqlite'
            )
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        self.db_path = db_path
        logger.info(f"Using database: {self.db_path}\n")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        start_time = time.time()
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        load_time = time.time() - start_time
        logger.info(f"Checkpoint loaded in {load_time:.2f} seconds\n")
        
        # Extract config
        config = checkpoint.get('config', {})
        logger.debug(f"Checkpoint config keys: {list(config.keys())}")
        
        # Create SINGLE-TARGET model only
        logger.info(f"Model type: SINGLE-TARGET")
        
        # Create model with config parameters
        logger.info("Creating model...")
        self.model = create_model(
            model_type='single_target',
            mol_hidden_dim=int(config.get('mol_hidden_dim', 256)),
            interaction_dim=int(config.get('interaction_dim', 512)),
            num_layers=int(config.get('num_interaction_layers', 3)),
            target_embedding_dim=int(config.get('target_embedding_dim', 512)),
            dropout=float(config.get('dropout', 0.1)),
        ).to(self.device)
        
        logger.info("Loading model weights...")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded - Total parameters: {total_params:,}")
        logger.info("Model set to evaluation mode\n")
        
        # Create preprocessor
        logger.info("Initializing molecule preprocessor...")
        self.preprocessor = MoleculePreprocessor()
        logger.info("✅ Preprocessor initialized\n")
        
        # ✅ LOAD NORMALIZER
        self.normalizer = None
        if normalizer_path is None:
            # Try to find normalizer near checkpoint
            checkpoint_dir = os.path.dirname(checkpoint_path)
            possible_paths = [
                os.path.join(checkpoint_dir, 'normalizer.json'),
                os.path.join(checkpoint_dir, '..', 'normalizer.json'),
                os.path.join(checkpoint_dir, '..', '..', 'normalizer.json'),
                'checkpoints/normalizer.json',
                'normalizer.json',
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    normalizer_path = path
                    break
        
        if normalizer_path and os.path.exists(normalizer_path):
            try:
                self.normalizer = load_normalizer(normalizer_path)
                logger.info(f"✅ Normalizer loaded successfully\n")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load normalizer: {e}")
                logger.warning("Predictions will NOT be denormalized\n")
        else:
            logger.warning(f"⚠️ Normalizer not found at {normalizer_path}")
            logger.warning("Predictions will NOT be denormalized\n")
    
    
    def _map_column_names(self, df):
        """
        Map CSV column names to expected names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping expected names to actual column names
        """
        column_mapping = {}
        
        # Map score column
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['final_score', 'score', 'affinity', 'binding_affinity', 'y', 'label']:
                column_mapping['final_score'] = col
                break
        
        # Map molecule name column (optional)
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['molecule_name', 'mol_name', 'id', 'name', 'compound_id', 'rxn', 'smiles']:
                column_mapping['molecule_name'] = col
                break
        
        return column_mapping
    
    
    def load_data(self, data_file, batch_size=16):
        """
        Load data from CSV file and convert reaction names to SMILES.
        
        Args:
            data_file: Path to CSV file
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (DataFrame, column_mapping)
        """
        logger.info(f"Loading data from {data_file}...")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        start_time = time.time()
        df = pd.read_csv(data_file)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(df):,} samples in {load_time:.2f} seconds")
        
        # Debug: Print column info
        logger.info(f"CSV shape: {df.shape}")
        logger.info(f"Column names: {df.columns.tolist()}")
        logger.debug(f"First row:\n{df.iloc[0]}")
        
        # Map column names
        col_mapping = self._map_column_names(df)
        logger.info(f"Column mapping: {col_mapping}\n")
        
        # Validate required columns
        required_cols = ['final_score', 'molecule_name']
        
        missing_cols = [col for col in required_cols if col not in col_mapping]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available: {df.columns.tolist()}"
            )
        
        # Rename columns for consistency
        df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})
        
        # Convert reaction names to SMILES
        logger.info("Converting reaction names to SMILES...")
        smiles_list = []
        failed_conversions = []
        
        for idx, mol_name in enumerate(tqdm(
            df_renamed['molecule_name'],
            desc="Converting reactions to SMILES"
        )):
            try:
                smiles = get_smiles_from_reaction(str(mol_name), self.db_path)
                if smiles is None:
                    logger.debug(f"Failed to convert {mol_name} to SMILES")
                    failed_conversions.append((idx, mol_name))
                smiles_list.append(smiles)
            except Exception as e:
                logger.error(f"Error converting {mol_name}: {e}")
                failed_conversions.append((idx, mol_name))
                smiles_list.append(None)
        
        df_renamed['smiles'] = smiles_list
        
        # Remove rows with failed conversions
        if failed_conversions:
            logger.warning(f"Failed to convert {len(failed_conversions)} molecules")
            logger.info(f"Sample failed: {failed_conversions[:5]}")
            df_renamed = df_renamed.dropna(subset=['smiles'])
            logger.info(f"Remaining samples: {len(df_renamed)}\n")
        
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of batches: {(len(df_renamed) + batch_size - 1) // batch_size}\n")
        
        return df_renamed, col_mapping
    
    
    def evaluate(self, df, batch_size=16, dataset_name="Validation"):
        """
        Evaluate model on dataframe - SINGLE-TARGET ONLY.
        
        ✅ FIXED: Correct normalizer usage and denormalization
        
        Args:
            df: DataFrame with columns: smiles, final_score, molecule_name
            batch_size: Batch size for processing
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 80)
        logger.info(f"{dataset_name} Evaluation (SINGLE-TARGET)")
        logger.info("=" * 80 + "\n")
        
        all_predictions = []
        all_targets = []
        all_mol_names = []
        failed_samples = []
        
        total_start_time = time.time()
        
        for i in tqdm(
            range(0, len(df), batch_size),
            desc=f"Evaluating {dataset_name}",
            total=(len(df) + batch_size - 1) // batch_size
        ):
            batch_df = df.iloc[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug(f"Processing batch {batch_num} ({len(batch_df)} samples)")
            
            smiles_list = batch_df['smiles'].tolist()
            final_scores = batch_df['final_score'].tolist()
            mol_names = batch_df['molecule_name'].tolist()
            
            try:
                # Process molecules
                batch_start = time.time()
                mol_data_batch = self.preprocessor.batch_process(smiles_list)
                mol_data_batch = {k: v.to(self.device) for k, v in mol_data_batch.items()}
                process_time = time.time() - batch_start
                logger.debug(f"  Batch {batch_num}: Molecule processing: {process_time*1000:.2f} ms")
                
                # Predict - SINGLE-TARGET: NO protein sequences needed!
                inference_start = time.time()
                with torch.no_grad():
                    outputs = self.model(mol_data=mol_data_batch)
                
                inference_time = time.time() - inference_start
                logger.debug(f"  Batch {batch_num}: Inference: {inference_time*1000:.2f} ms")
                
                predictions = outputs['final_score'].cpu().numpy().flatten().tolist()
                
                all_predictions.extend(predictions)
                all_targets.extend(final_scores)
                all_mol_names.extend(mol_names)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                failed_samples.extend(range(i, min(i + batch_size, len(df))))
                all_predictions.extend([None] * len(batch_df))
                all_targets.extend(final_scores)
                all_mol_names.extend(mol_names)
        
        total_time = time.time() - total_start_time
        logger.info(f"\nProcessed {len(df):,} samples in {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time/len(df)*1000:.2f} ms\n")
        
        if failed_samples:
            logger.warning(f"Failed to process {len(failed_samples)} samples")
        
        # Calculate metrics (excluding None predictions)
        valid_predictions = [p for p in all_predictions if p is not None]
        valid_targets = [t for p, t in zip(all_predictions, all_targets) if p is not None]
        valid_mol_names = [m for p, m in zip(all_predictions, all_mol_names) if p is not None]
        
        if len(valid_predictions) == 0:
            logger.error("❌ No valid predictions!")
            return None
        
        predictions = np.array(valid_predictions)
        targets = np.array(valid_targets)
        
        # ✅ DENORMALIZE PREDICTIONS (FIXED)
        if self.normalizer is not None:
            logger.info("Denormalizing predictions...")
            predictions = denormalize_predictions(predictions, self.normalizer)
            verify_denormalization(predictions, targets, self.normalizer)
        else:
            logger.warning("⚠️ No normalizer available - predictions are in normalized space")
        
        # Calculate metrics
        mse = np.mean((predictions - targets)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Pearson correlation
        if len(predictions) > 1:
            correlation, p_value = stats.pearsonr(predictions, targets)
        else:
            correlation, p_value = 0.0, 1.0
        
        # Spearman correlation
        if len(predictions) > 1:
            spearman_corr, spearman_p = stats.spearmanr(predictions, targets)
        else:
            spearman_corr, spearman_p = 0.0, 1.0
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'predictions': predictions,
            'targets': targets,
            'mol_names': valid_mol_names,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_correlation': float(correlation),
            'pearson_p_value': float(p_value),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'r_squared': float(r_squared),
            'num_samples': len(predictions),
            'num_failed': len(failed_samples),
            'normalizer': self.normalizer
        }
    
    
    def print_results(self, results, dataset_name="Validation"):
        """
        Print evaluation results.
        
        Args:
            results: Dictionary with evaluation results
            dataset_name: Name of dataset for logging
        """
        if results is None:
            logger.error("❌ No results to print")
            return
        
        logger.info("=" * 80)
        logger.info(f"{dataset_name} Results (SINGLE-TARGET)")
        logger.info("=" * 80)
        logger.info(f"Total Samples:         {results['num_samples']:,}")
        logger.info(f"Failed Samples:        {results['num_failed']:,}")
        logger.info(f"Successful Samples:    {results['num_samples'] - results['num_failed']:,}")
        logger.info("")
        
        if results['normalizer']:
            logger.info(f"Normalizer Method:     {results['normalizer'].get('method')}")
            if results['normalizer'].get('method') == 'standardization':
                logger.info(f"  Mean: {results['normalizer']['mean']:.6f}")
                logger.info(f"  Std:  {results['normalizer']['std']:.6f}")
            elif results['normalizer'].get('method') == 'minmax':
                logger.info(f"  Min: {results['normalizer']['min']:.6f}")
                logger.info(f"  Max: {results['normalizer']['max']:.6f}")
            logger.info("")
        else:
            logger.warning("⚠️ Predictions NOT denormalized (no normalizer)")
            logger.info("")
        
        logger.info("Regression Metrics:")
        logger.info(f"  MSE:                 {results['mse']:.6f}")
        logger.info(f"  RMSE:                {results['rmse']:.6f}")
        logger.info(f"  MAE:                 {results['mae']:.6f}")
        logger.info(f"  R²:                  {results['r_squared']:.6f}")
        logger.info("")
        logger.info("Correlation Metrics:")
        logger.info(f"  Pearson r:           {results['pearson_correlation']:.6f} (p={results['pearson_p_value']:.2e})")
        logger.info(f"  Spearman ρ:          {results['spearman_correlation']:.6f} (p={results['spearman_p_value']:.2e})")
        logger.info("=" * 80 + "\n")
    
    
    def save_predictions(self, results, output_file):
        """
        Save predictions to CSV.
        
        Args:
            results: Dictionary with evaluation results
            output_file: Path to output CSV file
        """
        if results is None:
            logger.error("❌ No results to save")
            return
        
        results_df = pd.DataFrame({
            'molecule_name': results['mol_names'],
            'predicted_score': results['predictions'],
            'target_score': results['targets'],
            'error': results['predictions'] - results['targets'],
            'abs_error': np.abs(results['predictions'] - results['targets']),
            'percent_error': np.abs(
                (results['predictions'] - results['targets']) / (results['targets'] + 1e-10)
            ) * 100
        })
        
        # Sort by absolute error (worst predictions first)
        results_df = results_df.sort_values('abs_error', ascending=False)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        logger.info(f"✅ Predictions saved to {output_file}\n")
        
        # Print summary statistics
        logger.info("Prediction Statistics:")
        logger.info(f"  Mean Error:          {results_df['error'].mean():.6f}")
        logger.info(f"  Std Error:           {results_df['error'].std():.6f}")
        logger.info(f"  Max Abs Error:       {results_df['abs_error'].max():.6f}")
        logger.info(f"  Min Abs Error:       {results_df['abs_error'].min():.6f}")
        logger.info(f"  Median Abs Error:    {results_df['abs_error'].median():.6f}")
        logger.info(f"  Mean % Error:        {results_df['percent_error'].mean():.2f}%\n")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate BoltzPredictor on reaction data (Single-Target Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single-target model on test set
  python evaluate_reactions.py \\
    --checkpoint checkpoints/single_target/final_model.pt \\
    --data_file data/test_reactions.csv \\
    --output results/test_predictions.csv

  # Evaluate with custom normalizer path
  python evaluate_reactions.py \\
    --checkpoint checkpoints/final_model.pt \\
    --data_file data/test_reactions.csv \\
    --normalizer checkpoints/normalizer.json \\
    --output results/test_predictions.csv

  # Evaluate on CPU
  python evaluate_reactions.py \\
    --checkpoint checkpoints/final_model.pt \\
    --data_file data/test_reactions.csv \\
    --device cpu

  # Custom database path
  python evaluate_reactions.py \\
    --checkpoint checkpoints/final_model.pt \\
    --data_file data/test_reactions.csv \\
    --db_path /path/to/molecules.sqlite

  # Debug mode
  python evaluate_reactions.py \\
    --checkpoint checkpoints/final_model.pt \\
    --data_file data/test_reactions.csv \\
    --log_level DEBUG
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to data file (CSV)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output predictions file (CSV)'
    )
    parser.add_argument(
        '--normalizer',
        type=str,
        default=None,
        help='Path to normalizer.json (optional, will auto-search if not provided)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default=None,
        help='Path to molecules.sqlite database'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level=log_level, log_dir='logs')
    
    logger.info("=" * 80)
    logger.info("BoltzPredictor Evaluation - Single-Target")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    try:
        # Initialize evaluator
        evaluator = ReactionEvaluator(
            args.checkpoint,
            args.device,
            args.db_path,
            args.normalizer
        )
        
        # Load data
        df, col_mapping = evaluator.load_data(args.data_file, args.batch_size)
        
        # Evaluate
        logger.info(f"{'='*80}")
        logger.info(f"Evaluation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}\n")
        
        results = evaluator.evaluate(df, args.batch_size, dataset_name="Test")
        evaluator.print_results(results, dataset_name="Test")
        
        # Save predictions if requested
        if args.output and results is not None:
            evaluator.save_predictions(results, args.output)
        
        logger.info(f"{'='*80}")
        logger.info(f"Evaluation Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
