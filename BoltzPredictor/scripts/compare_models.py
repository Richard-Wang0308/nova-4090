"""
Compare two BoltzPredictor models on the same test data (Single-Target Only).

✅ FIXED: Each model uses its own normalizer for correct denormalization!

Single-target version:
- Simple regression comparison (NO protein sequences needed)
- NO antitarget logic
- Comprehensive metrics and visualization
- Statistical significance testing
"""

import torch
import torch.serialization
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
import time
import os
import sys
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from boltzpredictor.models import create_model
from boltzpredictor.data import MoleculePreprocessor
from boltzpredictor.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def safe_torch_load(path, map_location='cpu'):
    """Safely load PyTorch checkpoint with numpy scalar support (PyTorch 2.6+)."""
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    return torch.load(path, map_location=map_location, weights_only=False)


# ============================================================================
# Reaction Chemistry Functions (Simplified for single-target)
# ============================================================================

def get_smiles_from_reaction(product_name: str, db_path: str) -> str:
    """
    Handle reaction format: rxn:reaction_id:mol1_id:mol2_id or rxn:reaction_id:mol1_id:mol2_id:mol3_id
    
    Args:
        product_name: Reaction identifier string
        db_path: Path to molecules.sqlite database
    
    Returns:
        SMILES string or None if reaction fails
    """
    import sqlite3
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    def get_reaction_info(rxn_id: int, db_path: str) -> tuple:
        """Get reaction SMARTS and role information from database."""
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
            logger.error(f"Error getting reaction info: {e}")
            return None

    def get_molecules(mol_ids: list, db_path: str) -> list:
        """Get molecules from database."""
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
        """Combine azide and alkyne synthons to form triazole."""
        try:
            m1 = Chem.RWMol(Chem.MolFromSmiles(azide_smiles))
            m2 = Chem.RWMol(Chem.MolFromSmiles(alkyne_smiles))
            
            if not m1 or not m2:
                return None
            
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
                return None
            
            n1 = m1.GetAtomWithIdx(a1).GetNeighbors()[0].GetIdx()
            n2 = m2.GetAtomWithIdx(a2).GetNeighbors()[0].GetIdx()
            
            combined = Chem.RWMol(m1)
            atom_mapping = {}
            
            for i, atom in enumerate(m2.GetAtoms()):
                if i != a2:
                    atom_mapping[i] = combined.AddAtom(atom)
            
            for bond in m2.GetBonds():
                begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a2 not in (begin_idx, end_idx):
                    combined.AddBond(
                        atom_mapping[begin_idx],
                        atom_mapping[end_idx],
                        bond.GetBondType()
                    )
            
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
        """Perform SMARTS-based reaction."""
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if not mol1 or not mol2:
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
        """Validate reactants can react and return in correct order."""
        try:
            if smiles3 is None:
                can_react = (
                    (((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)) or
                    (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
                )
                if not can_react:
                    return None, None
                
                if ((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB):
                    return smiles1, smiles2
                else:
                    return smiles2, smiles1
            
            else:
                can_react_12 = (
                    (((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)) or
                    (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
                )
                can_react_3 = (role_mask3 & roleC) == roleC
                
                if not can_react_12 or not can_react_3:
                    return None, None, None
                
                if (role_mask1 & roleA) and (role_mask2 & roleB):
                    return smiles1, smiles2, smiles3
                else:
                    return smiles2, smiles1, smiles3
                
        except Exception as e:
            logger.error(f"Error validating reactants: {e}")
            return (None, None) if smiles3 is None else (None, None, None)

    def react_molecules(rxn_id: int, mol1_id: int, mol2_id: int, db_path: str) -> str:
        """Perform 2-component reaction."""
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
                
            if rxn_id == 1:
                return combine_triazole_synthons(reactant1, reactant2)
            else:
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
        """Perform 3-component reaction."""
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
            
            if rxn_id == 3:
                triazole_cooh = combine_triazole_synthons(reactant1, reactant2)
                if not triazole_cooh:
                    return None
                
                amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
                return perform_smarts_reaction(triazole_cooh, reactant3, amide_smarts)
            
            if rxn_id == 5:
                suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
                suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"

                intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
                if not intermediate:
                    return None

                final_product = perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts)
                return final_product
            
            return None
            
        except Exception as e:
            logger.error(f"Error in 3-component reaction {mol1_id}, {mol2_id}, {mol3_id}: {e}")
            return None

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
# Model Evaluator Class (Single-Target Only)
# ============================================================================

class ModelEvaluator:
    """Evaluator for comparing models (Single-Target Only)."""
    
    def __init__(self, checkpoint_path, device='cuda', db_path=None, normalizer_path=None):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (cuda or cpu)
            db_path: Path to molecules.sqlite database
            normalizer_path: Path to normalizer.json for THIS model
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # ✅ FIXED: Load normalizer for THIS specific model
        self.normalizer = None
        if normalizer_path:
            if os.path.exists(normalizer_path):
                with open(normalizer_path, 'r') as f:
                    self.normalizer = json.load(f)
                logger.info(f"✅ Loaded normalizer from {normalizer_path}")
                logger.info(f"   Method: {self.normalizer.get('method')}")
                logger.info(f"   Mean: {self.normalizer.get('mean'):.6f}")
                logger.info(f"   Std: {self.normalizer.get('std'):.6f}")
            else:
                logger.warning(f"⚠️  Normalizer not found at {normalizer_path}")
                logger.warning("   Using raw predictions (normalized space)")
        else:
            # Try to find normalizer in checkpoint directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            default_normalizer_path = os.path.join(checkpoint_dir, 'normalizer.json')
            
            if os.path.exists(default_normalizer_path):
                logger.info(f"Found normalizer in checkpoint directory: {default_normalizer_path}")
                with open(default_normalizer_path, 'r') as f:
                    self.normalizer = json.load(f)
                logger.info(f"✅ Loaded normalizer")
                logger.info(f"   Method: {self.normalizer.get('method')}")
                logger.info(f"   Mean: {self.normalizer.get('mean'):.6f}")
                logger.info(f"   Std: {self.normalizer.get('std'):.6f}")
            else:
                logger.warning(f"⚠️  No normalizer found. Using raw predictions (normalized space)")

        # Setup database path
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), '..', '..',
                'combinatorial_db', 'molecules.sqlite'
            )
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        self.db_path = db_path
        logger.info(f"Using database: {self.db_path}")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        start_time = time.time()
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        load_time = time.time() - start_time
        logger.info(f"Checkpoint loaded in {load_time:.2f} seconds")
        
        # Extract config
        config = checkpoint.get('config', {})
        logger.debug(f"Checkpoint config keys: {list(config.keys())}")
        
        # Create SINGLE-TARGET model only
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
        logger.info(f"Model loaded - Total parameters: {total_params:,}\n")
        
        # Create preprocessor
        logger.info("Initializing molecule preprocessor...")
        self.preprocessor = MoleculePreprocessor()
        logger.info("✅ Preprocessor initialized\n")
    
    
    def _map_column_names(self, df):
        """Map CSV column names to expected names."""
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
        """Load data from CSV file."""
        logger.info(f"Loading data from {data_file}...")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        start_time = time.time()
        df = pd.read_csv(data_file)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(df):,} samples in {load_time:.2f} seconds")
        
        logger.info(f"CSV shape: {df.shape}")
        logger.info(f"Column names: {df.columns.tolist()}")
        
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
        
        # Rename columns
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
            df_renamed = df_renamed.dropna(subset=['smiles'])
            logger.info(f"Remaining samples: {len(df_renamed)}\n")
        
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of batches: {(len(df_renamed) + batch_size - 1) // batch_size}\n")
        
        return df_renamed, col_mapping
    
    
    def evaluate(self, df, batch_size=16, model_name="Model"):
        """
        Evaluate model on dataframe - SINGLE-TARGET ONLY.
        
        Args:
            df: DataFrame with columns: smiles, final_score, molecule_name
            batch_size: Batch size for processing
            model_name: Name of model for logging
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 80)
        logger.info(f"Evaluating {model_name} (SINGLE-TARGET)")
        logger.info("=" * 80 + "\n")
        
        all_predictions = []
        all_targets = []
        failed_samples = []
        
        total_start_time = time.time()
        
        for i in tqdm(
            range(0, len(df), batch_size),
            desc=f"Evaluating {model_name}",
            total=(len(df) + batch_size - 1) // batch_size
        ):
            batch_df = df.iloc[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug(f"Processing batch {batch_num} ({len(batch_df)} samples)")
            
            smiles_list = batch_df['smiles'].tolist()
            final_scores = batch_df['final_score'].tolist()
            
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
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                failed_samples.extend(range(i, min(i + batch_size, len(df))))
                all_predictions.extend([None] * len(batch_df))
                all_targets.extend(final_scores)
        
        total_time = time.time() - total_start_time
        logger.info(f"\nProcessed {len(df):,} samples in {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time/len(df)*1000:.2f} ms\n")
        
        if failed_samples:
            logger.warning(f"Failed to process {len(failed_samples)} samples")
        
        # Calculate metrics (excluding None predictions)
        valid_predictions = [p for p in all_predictions if p is not None]
        valid_targets = [t for p, t in zip(all_predictions, all_targets) if p is not None]
        
        if len(valid_predictions) == 0:
            logger.error("❌ No valid predictions!")
            return None
        
        predictions = np.array(valid_predictions)
        targets = np.array(valid_targets)
        
        # ✅ FIXED: Denormalize using THIS model's normalizer
        if self.normalizer and self.normalizer.get('method') == 'standardization':
            mean = self.normalizer.get('mean')
            std = self.normalizer.get('std')
            predictions_denorm = predictions * std + mean
            logger.info(f"✅ Denormalized predictions using {model_name}'s normalizer")
            logger.info(f"   Method: {self.normalizer.get('method')}")
            logger.info(f"   Mean: {mean:.6f}, Std: {std:.6f}")
            logger.info(f"   Prediction range: [{predictions_denorm.min():.6f}, {predictions_denorm.max():.6f}]")
            logger.info(f"   Target range: [{targets.min():.6f}, {targets.max():.6f}]")
            predictions = predictions_denorm
        else:
            logger.warning(f"⚠️  No normalizer for {model_name}. Using raw predictions (normalized space)")
            logger.info(f"   Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            logger.info(f"   Target range: [{targets.min():.6f}, {targets.max():.6f}]")

        # Calculate metrics
        mse = np.mean((predictions - targets)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Pearson correlation
        if len(predictions) > 1:
            pearson_corr, pearson_p = stats.pearsonr(predictions, targets)
        else:
            pearson_corr, pearson_p = 0.0, 1.0
        
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
            'model': model_name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'r_squared': float(r_squared),
            'num_samples': len(predictions),
            'num_failed': len(failed_samples),
            'predictions': predictions,
            'targets': targets,
            'has_normalizer': self.normalizer is not None  # ✅ Track if denormalized
        }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_comparison(model1_results, model2_results, output_dir='results'):
    """
    Create comparison plots.
    
    Args:
        model1_results: Results from first model
        model2_results: Results from second model
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison: Single-Target Predictions', fontsize=16, fontweight='bold')
    
    # Plot 1: Predictions vs Targets (Model 1)
    ax = axes[0, 0]
    ax.scatter(model1_results['targets'], model1_results['predictions'], alpha=0.5, s=20)
    ax.plot([model1_results['targets'].min(), model1_results['targets'].max()],
            [model1_results['targets'].min(), model1_results['targets'].max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Target Score')
    ax.set_ylabel('Predicted Score')
    title1 = f"{model1_results['model']}\nR² = {model1_results['r_squared']:.4f}"
    if model1_results['has_normalizer']:
        title1 += " (Denormalized)"
    else:
        title1 += " (Normalized)"
    ax.set_title(title1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Targets (Model 2)
    ax = axes[0, 1]
    ax.scatter(model2_results['targets'], model2_results['predictions'], alpha=0.5, s=20, color='orange')
    ax.plot([model2_results['targets'].min(), model2_results['targets'].max()],
            [model2_results['targets'].min(), model2_results['targets'].max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Target Score')
    ax.set_ylabel('Predicted Score')
    title2 = f"{model2_results['model']}\nR² = {model2_results['r_squared']:.4f}"
    if model2_results['has_normalizer']:
        title2 += " (Denormalized)"
    else:
        title2 += " (Normalized)"
    ax.set_title(title2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residuals Comparison
    ax = axes[0, 2]
    residuals1 = model1_results['predictions'] - model1_results['targets']
    residuals2 = model2_results['predictions'] - model2_results['targets']
    ax.hist(residuals1, bins=30, alpha=0.5, label=model1_results['model'])
    ax.hist(residuals2, bins=30, alpha=0.5, label=model2_results['model'])
    ax.set_xlabel('Residual (Predicted - Target)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Metrics Comparison
    ax = axes[1, 0]
    metrics = ['RMSE', 'MAE', 'MSE']
    model1_vals = [model1_results['rmse'], model1_results['mae'], model1_results['mse']]
    model2_vals = [model2_results['rmse'], model2_results['mae'], model2_results['mse']]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, model1_vals, width, label=model1_results['model'])
    ax.bar(x + width/2, model2_vals, width, label=model2_results['model'])
    ax.set_ylabel('Value')
    ax.set_title('Error Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Correlation Comparison
    ax = axes[1, 1]
    corr_metrics = ['Pearson r', 'Spearman ρ', 'R²']
    model1_corr = [
        model1_results['pearson_correlation'],
        model1_results['spearman_correlation'],
        model1_results['r_squared']
    ]
    model2_corr = [
        model2_results['pearson_correlation'],
        model2_results['spearman_correlation'],
        model2_results['r_squared']
    ]
    x = np.arange(len(corr_metrics))
    ax.bar(x - width/2, model1_corr, width, label=model1_results['model'])
    ax.bar(x + width/2, model2_corr, width, label=model2_results['model'])
    ax.set_ylabel('Value')
    ax.set_title('Correlation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(corr_metrics)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Error Distribution
    ax = axes[1, 2]
    errors1 = np.abs(model1_results['predictions'] - model1_results['targets'])
    errors2 = np.abs(model2_results['predictions'] - model2_results['targets'])
    ax.boxplot([errors1, errors2], labels=[model1_results['model'], model2_results['model']])
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Comparison plot saved to {plot_path}")
    plt.close()


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare two BoltzPredictor models (Single-Target Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare Phase 2 and Reaction 2 models (each with their own normalizer)
  python compare_models.py \\
    --model1_checkpoint checkpoints/phase2_final.pt \\
    --model1_normalizer checkpoints/phase2_normalizer.json \\
    --model2_checkpoint checkpoints/reaction2/best.pt \\
    --model2_normalizer checkpoints/reaction2/normalizer.json \\
    --test_file data/test_reactions.csv

  # Auto-detect normalizers from checkpoint directories
  python compare_models.py \\
    --model1_checkpoint checkpoints/phase2_final.pt \\
    --model2_checkpoint checkpoints/reaction2/best.pt \\
    --test_file data/test_reactions.csv

  # With custom batch size
  python compare_models.py \\
    --model1_checkpoint checkpoints/phase2_final.pt \\
    --model1_normalizer checkpoints/phase2_normalizer.json \\
    --model2_checkpoint checkpoints/reaction2/best.pt \\
    --model2_normalizer checkpoints/reaction2/normalizer.json \\
    --test_file data/test_reactions.csv \\
    --batch_size 64

  # On CPU
  python compare_models.py \\
    --model1_checkpoint checkpoints/phase2_final.pt \\
    --model1_normalizer checkpoints/phase2_normalizer.json \\
    --model2_checkpoint checkpoints/reaction2/best.pt \\
    --model2_normalizer checkpoints/reaction2/normalizer.json \\
    --test_file data/test_reactions.csv \\
    --device cpu

  # Debug mode
  python compare_models.py \\
    --model1_checkpoint checkpoints/phase2_final.pt \\
    --model1_normalizer checkpoints/phase2_normalizer.json \\
    --model2_checkpoint checkpoints/reaction2/best.pt \\
    --model2_normalizer checkpoints/reaction2/normalizer.json \\
    --test_file data/test_reactions.csv \\
    --log_level DEBUG
        """
    )
    
    parser.add_argument(
        '--model1_checkpoint',
        type=str,
        default='checkpoints/phase2_final.pt',
        help='Path to first model checkpoint (Phase 2)'
    )
    parser.add_argument(
        '--model1_normalizer',
        type=str,
        default=None,
        help='Path to first model normalizer.json (auto-detect if not provided)'
    )
    parser.add_argument(
        '--model2_checkpoint',
        type=str,
        default='checkpoints/reaction2/best.pt',
        help='Path to second model checkpoint (Reaction 2)'
    )
    parser.add_argument(
        '--model2_normalizer',
        type=str,
        default=None,
        help='Path to second model normalizer.json (auto-detect if not provided)'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help='Test data file (CSV)'
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
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
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
    logger.info("Model Comparison: Single-Target Version")
    logger.info("✅ FIXED: Each model uses its own normalizer")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    try:
        # Evaluate Model 1
        logger.info("=" * 80)
        logger.info("MODEL 1 (Phase 2 / Original)")
        logger.info("=" * 80 + "\n")
        
        try:
            evaluator1 = ModelEvaluator(
                args.model1_checkpoint,
                args.device,
                args.db_path,
                args.model1_normalizer  # ✅ FIXED: Pass model1's normalizer
            )
            df1, _ = evaluator1.load_data(args.test_file, args.batch_size)
            model1_results = evaluator1.evaluate(df1, args.batch_size, "Model 1 (Phase 2)")
        except Exception as e:
            logger.error(f"Failed to evaluate Model 1: {e}")
            import traceback
            traceback.print_exc()
            model1_results = None
        
        # Evaluate Model 2
        logger.info("\n" + "=" * 80)
        logger.info("MODEL 2 (Reaction 2 / Fine-tuned)")
        logger.info("=" * 80 + "\n")
        
        try:
            evaluator2 = ModelEvaluator(
                args.model2_checkpoint,
                args.device,
                args.db_path,
                args.model2_normalizer  # ✅ FIXED: Pass model2's normalizer
            )
            df2, _ = evaluator2.load_data(args.test_file, args.batch_size)
            model2_results = evaluator2.evaluate(df2, args.batch_size, "Model 2 (Reaction 2)")
        except Exception as e:
            logger.error(f"Failed to evaluate Model 2: {e}")
            import traceback
            traceback.print_exc()
            model2_results = None
        
        # Compare results
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 80 + "\n")
        
        if model1_results and model2_results:
            # Create comparison dataframe
            comparison_data = {
                'Metric': [
                    'RMSE', 'MAE', 'MSE', 'R²',
                    'Pearson r', 'Spearman ρ',
                    'Samples', 'Failed', 'Normalizer'
                ],
                'Model 1': [
                    f"{model1_results['rmse']:.6f}",
                    f"{model1_results['mae']:.6f}",
                    f"{model1_results['mse']:.6f}",
                    f"{model1_results['r_squared']:.6f}",
                    f"{model1_results['pearson_correlation']:.6f}",
                    f"{model1_results['spearman_correlation']:.6f}",
                    f"{model1_results['num_samples']:,}",
                    f"{model1_results['num_failed']:,}",
                    "✅ Yes" if model1_results['has_normalizer'] else "❌ No"
                ],
                'Model 2': [
                    f"{model2_results['rmse']:.6f}",
                    f"{model2_results['mae']:.6f}",
                    f"{model2_results['mse']:.6f}",
                    f"{model2_results['r_squared']:.6f}",
                    f"{model2_results['pearson_correlation']:.6f}",
                    f"{model2_results['spearman_correlation']:.6f}",
                    f"{model2_results['num_samples']:,}",
                    f"{model2_results['num_failed']:,}",
                    "✅ Yes" if model2_results['has_normalizer'] else "❌ No"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            logger.info("\n" + comparison_df.to_string(index=False))
            
            # Calculate improvements
            logger.info("\n" + "=" * 80)
            logger.info("IMPROVEMENTS (Model 2 vs Model 1)")
            logger.info("=" * 80 + "\n")
            
            rmse_improvement = (model1_results['rmse'] - model2_results['rmse']) / model1_results['rmse'] * 100
            mae_improvement = (model1_results['mae'] - model2_results['mae']) / model1_results['mae'] * 100
            mse_improvement = (model1_results['mse'] - model2_results['mse']) / model1_results['mse'] * 100
            r2_improvement = (model2_results['r_squared'] - model1_results['r_squared']) / abs(model1_results['r_squared'] + 1e-8) * 100
            pearson_improvement = (model2_results['pearson_correlation'] - model1_results['pearson_correlation']) / abs(model1_results['pearson_correlation'] + 1e-8) * 100
            spearman_improvement = (model2_results['spearman_correlation'] - model1_results['spearman_correlation']) / abs(model1_results['spearman_correlation'] + 1e-8) * 100
            
            logger.info(f"RMSE:           {rmse_improvement:+.2f}%")
            logger.info(f"MAE:            {mae_improvement:+.2f}%")
            logger.info(f"MSE:            {mse_improvement:+.2f}%")
            logger.info(f"R²:             {r2_improvement:+.2f}%")
            logger.info(f"Pearson r:      {pearson_improvement:+.2f}%")
            logger.info(f"Spearman ρ:     {spearman_improvement:+.2f}%\n")
            
            # Determine winner
            logger.info("=" * 80)
            if rmse_improvement > 0:
                logger.info("🏆 WINNER: Model 2 (Better RMSE)")
            elif rmse_improvement < 0:
                logger.info("🏆 WINNER: Model 1 (Better RMSE)")
            else:
                logger.info("⚖️  TIE: Both models have similar RMSE")
            logger.info("=" * 80 + "\n")
            
            # Save comparison results
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            comparison_df.to_csv(
                os.path.join(args.output_dir, 'model_comparison_metrics.csv'),
                index=False
            )
            logger.info(f"✅ Metrics saved to {args.output_dir}/model_comparison_metrics.csv")
            
            # Save improvements
            improvements_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'MSE', 'R²', 'Pearson r', 'Spearman ρ'],
                'Improvement (%)': [
                    rmse_improvement, mae_improvement, mse_improvement,
                    r2_improvement, pearson_improvement, spearman_improvement
                ]
            })
            improvements_df.to_csv(
                os.path.join(args.output_dir, 'model_improvements.csv'),
                index=False
            )
            logger.info(f"✅ Improvements saved to {args.output_dir}/model_improvements.csv")
            
            # Create visualizations
            logger.info("\nCreating comparison plots...")
            plot_comparison(model1_results, model2_results, args.output_dir)
            
        else:
            logger.error("❌ Could not complete comparison due to evaluation errors")
            return 1
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Comparison Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
