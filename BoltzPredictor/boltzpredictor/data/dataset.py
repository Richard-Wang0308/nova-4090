"""Dataset class for training - Target-only and Single-Target versions."""

import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocessing import MoleculePreprocessor
from ..combinatorial_db import get_smiles_from_reaction
from ..utils.proteins import get_sequence_from_protein_code
import logging

logger = logging.getLogger(__name__)


class ProteinSequenceCache:
    """Cache for protein sequences during dataset loading."""
    
    def __init__(self):
        self.cache = {}
    
    def get_sequence(self, protein_code):
        """Get sequence, using cache if available."""
        if protein_code in self.cache:
            return self.cache[protein_code]
        
        sequence = get_sequence_from_protein_code(protein_code)
        if sequence:
            self.cache[protein_code] = sequence
        return sequence


class BoltzDataset(Dataset):
    """
    Dataset for BoltzPredictor training - Target-only version.
    
    For multi-target models that learn: f(molecule, protein) → binding_affinity
    """
    
    def __init__(
        self,
        data_path=None,
        dataframe=None,
        molecule_name_col='molecule_name',
        target_protein_col='target_protein',
        target_seq_col='target_seq',
        score_col='final_score',
        epoch_col='epoch',
        preprocessor=None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to CSV file
            dataframe: Pandas DataFrame (alternative to data_path)
            molecule_name_col: Column name for molecule name (rxn format or SMILES)
            target_protein_col: Column name for target protein code (optional if target_seq_col provided)
            target_seq_col: Column name for target protein sequence
            score_col: Column name for final score
            epoch_col: Column name for epoch ID (optional, for tracking)
            preprocessor: MoleculePreprocessor instance
        
        Expected CSV format:
            molecule_name, target_protein, target_seq, final_score, epoch
            rxn:1:43634:31358, P31652, METTPLNS..., 0.137, 20262
        """
        if data_path is not None:
            logger.info(f"Loading dataset from {data_path}")
            self.df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.df)} rows")
        elif dataframe is not None:
            self.df = dataframe.copy()
            logger.info(f"Using provided dataframe with {len(self.df)} rows")
        else:
            raise ValueError("Either data_path or dataframe must be provided")
        
        logger.info(f"Available columns: {self.df.columns.tolist()}")
        
        self.molecule_name_col = molecule_name_col
        self.target_protein_col = target_protein_col
        self.target_seq_col = target_seq_col
        self.score_col = score_col
        self.epoch_col = epoch_col
        
        self.preprocessor = preprocessor or MoleculePreprocessor()
        
        # ========== VALIDATE COLUMNS ==========
        self._validate_and_map_columns()
        
        # ========== CHECK IF SEQUENCES ARE IN CSV ==========
        self.has_target_sequences = self.target_seq_col in self.df.columns
        
        logger.info(f"Target sequences in CSV: {self.has_target_sequences}")
        
        # Initialize protein cache only if sequences are not in CSV
        self.protein_cache = None
        if not self.has_target_sequences:
            logger.info("Target sequences not found in CSV, will fetch from protein codes")
            self.protein_cache = ProteinSequenceCache()
            
            # Pre-fetch and cache all unique target protein sequences
            if self.target_protein_col in self.df.columns:
                unique_targets = self.df[self.target_protein_col].dropna().unique().tolist()
                logger.info(f"Pre-fetching {len(unique_targets)} unique target protein sequences...")
                for protein_code in unique_targets:
                    seq = self.protein_cache.get_sequence(protein_code)
                    if seq is None:
                        logger.warning(f"Failed to fetch sequence for {protein_code}")
                logger.info("Target protein sequences cached")
            else:
                raise ValueError(f"Column '{self.target_protein_col}' not found and no sequences in CSV")
        else:
            logger.info("Target sequences found in CSV, using directly (no fetching needed)")
        
        # ========== FILTER VALID SAMPLES ==========
        logger.info("Validating molecules...")
        self.valid_indices = []
        invalid_count = 0
        
        for idx, row in self.df.iterrows():
            try:
                mol_name = str(row[self.molecule_name_col]).strip()
                
                # Skip empty molecule names
                if not mol_name or mol_name == 'nan':
                    logger.debug(f"Skipping empty molecule name at index {idx}")
                    invalid_count += 1
                    continue
                
                # Convert molecule name to SMILES
                smiles = get_smiles_from_reaction(mol_name)
                if not smiles:
                    logger.debug(f"Could not convert {mol_name} to SMILES at index {idx}")
                    invalid_count += 1
                    continue
                
                # Validate SMILES
                mol_data = self.preprocessor.process_smiles(smiles)
                if mol_data is None or mol_data['x'].shape[0] == 0:
                    logger.debug(f"Invalid SMILES {smiles} at index {idx}")
                    invalid_count += 1
                    continue
                
                # Validate target sequence
                if self.has_target_sequences:
                    target_seq = str(row[self.target_seq_col]).strip()
                    if not target_seq or target_seq == 'nan' or len(target_seq) < 10:
                        logger.debug(f"Invalid target sequence at index {idx}")
                        invalid_count += 1
                        continue
                else:
                    # Check if protein code exists
                    if self.target_protein_col not in self.df.columns:
                        logger.debug(f"No target protein code at index {idx}")
                        invalid_count += 1
                        continue
                    
                    target_code = str(row[self.target_protein_col]).strip()
                    if not target_code or target_code == 'nan':
                        logger.debug(f"Empty target protein code at index {idx}")
                        invalid_count += 1
                        continue
                    
                    # Check if sequence can be fetched
                    target_seq = self.protein_cache.get_sequence(target_code)
                    if not target_seq:
                        logger.debug(f"Could not fetch sequence for {target_code} at index {idx}")
                        invalid_count += 1
                        continue
                
                # Validate score
                try:
                    score = float(row[self.score_col])
                except (ValueError, TypeError):
                    logger.debug(f"Invalid score at index {idx}")
                    invalid_count += 1
                    continue
                
                self.valid_indices.append(idx)
            
            except Exception as e:
                logger.debug(f"Error validating index {idx}: {e}")
                invalid_count += 1
                continue
        
        # Keep only valid samples
        self.df = self.df.loc[self.valid_indices].reset_index(drop=True)
        logger.info(f"✅ Dataset validated: {len(self.df)} valid samples (filtered {invalid_count} invalid)")
    
    def _validate_and_map_columns(self):
        """Validate and map column names."""
        logger.info("Validating columns...")
        
        # Check if required columns exist
        required_cols = [self.molecule_name_col, self.score_col]
        missing_required = [col for col in required_cols if col not in self.df.columns]
        
        if missing_required:
            logger.warning(f"Missing required columns: {missing_required}")
            logger.info(f"Available columns: {self.df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Check for target sequence or protein code
        has_target_seq = self.target_seq_col in self.df.columns
        has_target_code = self.target_protein_col in self.df.columns
        
        if not has_target_seq and not has_target_code:
            logger.error(f"Neither '{self.target_seq_col}' nor '{self.target_protein_col}' found in CSV")
            logger.info(f"Available columns: {self.df.columns.tolist()}")
            raise ValueError(
                f"CSV must contain either '{self.target_seq_col}' (sequences) "
                f"or '{self.target_protein_col}' (protein codes)"
            )
        
        logger.info(f"✅ Column validation passed")
        logger.info(f"  - Molecule names: {self.molecule_name_col}")
        logger.info(f"  - Target protein: {self.target_protein_col if has_target_code else 'N/A (using sequences)'}")
        logger.info(f"  - Target sequences: {self.target_seq_col if has_target_seq else 'N/A (will fetch from codes)'}")
        logger.info(f"  - Scores: {self.score_col}")
        if self.epoch_col in self.df.columns:
            logger.info(f"  - Epoch IDs: {self.epoch_col}")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            Dict with:
                - 'mol_data': Molecule graph data
                - 'target_seq': Target protein sequence
                - 'final_score': Binding affinity score
                - 'epoch_id': Epoch ID (optional)
        """
        row = self.df.iloc[idx]
        
        # ========== GET MOLECULE DATA ==========
        mol_name = str(row[self.molecule_name_col]).strip()
        
        # Convert molecule name to SMILES
        smiles = get_smiles_from_reaction(mol_name)
        if not smiles:
            raise ValueError(f"Could not convert {mol_name} to SMILES")
        
        # Process molecule
        mol_data = self.preprocessor.process_smiles(smiles)
        if mol_data is None:
            raise ValueError(f"Could not process SMILES: {smiles}")
        
        # ========== GET TARGET SEQUENCE ==========
        if self.has_target_sequences:
            # Use sequence directly from CSV
            target_seq = str(row[self.target_seq_col]).strip()
            
            if not target_seq or target_seq == 'nan' or len(target_seq) < 10:
                # Fallback to fetching if sequence is invalid
                if self.target_protein_col in self.df.columns:
                    target_code = str(row[self.target_protein_col]).strip()
                    if self.protein_cache is None:
                        self.protein_cache = ProteinSequenceCache()
                    target_seq = self.protein_cache.get_sequence(target_code)
                    if not target_seq:
                        raise ValueError(f"Could not get sequence for protein: {target_code}")
                else:
                    raise ValueError(f"Invalid target sequence at index {idx}")
        else:
            # Fetch sequence from protein code
            target_code = str(row[self.target_protein_col]).strip()
            
            if not target_code or target_code == 'nan':
                raise ValueError(f"Empty protein code at index {idx}")
            
            target_seq = self.protein_cache.get_sequence(target_code)
            
            if not target_seq:
                raise ValueError(f"Could not get sequence for protein: {target_code}")
        
        # ========== GET SCORE ==========
        final_score = float(row[self.score_col])
        
        # ========== GET EPOCH ID (OPTIONAL) ==========
        epoch_id = None
        if self.epoch_col in self.df.columns:
            try:
                epoch_val = row[self.epoch_col]
                if pd.notna(epoch_val):
                    epoch_id = int(epoch_val)
            except (ValueError, TypeError):
                epoch_id = None
        
        return {
            'mol_data': mol_data,
            'target_seq': target_seq,
            'final_score': final_score,
            'epoch_id': epoch_id,
        }
    
    def get_unique_proteins(self):
        """Get unique target protein codes for precomputation."""
        if self.target_protein_col in self.df.columns:
            proteins = self.df[self.target_protein_col].dropna().unique().tolist()
            logger.info(f"Found {len(proteins)} unique target proteins")
            return proteins
        else:
            logger.warning("No target protein codes in dataset")
            return []


# ============================================================================
# SINGLE-TARGET DATASET: Optimized for single protein + single reaction
# ============================================================================

class SingleTargetDataset(Dataset):
    """
    Dataset for single-target binding affinity prediction.
    
    Optimized for competition scenarios with:
    - Fixed target protein
    - Fixed reaction type
    - Only learns: f(molecule) → binding_affinity
    
    ✅ SIMPLER: No protein sequences needed
    ✅ FASTER: No protein encoding at inference
    ✅ BETTER: Specializes to one protein's binding pocket
    """
    
    def __init__(
        self,
        data_path=None,
        dataframe=None,
        molecule_name_col='molecule_name',
        score_col='final_score',
        epoch_col='epoch',
        preprocessor=None,
    ):
        """
        Initialize single-target dataset.
        
        Args:
            data_path: Path to CSV file
            dataframe: Pandas DataFrame (alternative to data_path)
            molecule_name_col: Column name for molecule name (rxn format or SMILES)
            score_col: Column name for final score
            epoch_col: Column name for epoch ID (optional)
            preprocessor: MoleculePreprocessor instance
        
        Expected CSV format (minimal):
            molecule_name, final_score, epoch
            rxn:1:43634:31358, 0.137, 20262
        
        Note:
            - No protein information needed
            - Target protein is FIXED and learned as parameter
            - Much simpler than multi-target dataset
        """
        if data_path is not None:
            logger.info(f"Loading single-target dataset from {data_path}")
            self.df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.df)} rows")
        elif dataframe is not None:
            self.df = dataframe.copy()
            logger.info(f"Using provided dataframe with {len(self.df)} rows")
        else:
            raise ValueError("Either data_path or dataframe must be provided")
        
        logger.info(f"Available columns: {self.df.columns.tolist()}")
        
        self.molecule_name_col = molecule_name_col
        self.score_col = score_col
        self.epoch_col = epoch_col
        
        self.preprocessor = preprocessor or MoleculePreprocessor()
        
        # ========== VALIDATE COLUMNS ==========
        self._validate_columns()
        
        # ========== FILTER VALID SAMPLES ==========
        logger.info("Validating molecules...")
        self.valid_indices = []
        invalid_count = 0
        
        for idx, row in self.df.iterrows():
            try:
                mol_name = str(row[self.molecule_name_col]).strip()
                
                # Skip empty molecule names
                if not mol_name or mol_name == 'nan':
                    logger.debug(f"Skipping empty molecule name at index {idx}")
                    invalid_count += 1
                    continue
                
                # Convert molecule name to SMILES
                smiles = get_smiles_from_reaction(mol_name)
                if not smiles:
                    logger.debug(f"Could not convert {mol_name} to SMILES at index {idx}")
                    invalid_count += 1
                    continue
                
                # Validate SMILES
                mol_data = self.preprocessor.process_smiles(smiles)
                if mol_data is None or mol_data['x'].shape[0] == 0:
                    logger.debug(f"Invalid SMILES {smiles} at index {idx}")
                    invalid_count += 1
                    continue
                
                # Validate score
                try:
                    score = float(row[self.score_col])
                except (ValueError, TypeError):
                    logger.debug(f"Invalid score at index {idx}")
                    invalid_count += 1
                    continue
                
                self.valid_indices.append(idx)
            
            except Exception as e:
                logger.debug(f"Error validating index {idx}: {e}")
                invalid_count += 1
                continue
        
        # Keep only valid samples
        self.df = self.df.loc[self.valid_indices].reset_index(drop=True)
        logger.info(
            f"✅ Single-target dataset validated: {len(self.df)} valid samples "
            f"(filtered {invalid_count} invalid)"
        )
    
    def _validate_columns(self):
        """Validate required columns."""
        logger.info("Validating columns...")
        
        # Check required columns
        required_cols = [self.molecule_name_col, self.score_col]
        missing_required = [col for col in required_cols if col not in self.df.columns]
        
        if missing_required:
            logger.error(f"Missing required columns: {missing_required}")
            logger.info(f"Available columns: {self.df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_required}")
        
        logger.info(f"✅ Column validation passed")
        logger.info(f"  - Molecule names: {self.molecule_name_col}")
        logger.info(f"  - Scores: {self.score_col}")
        if self.epoch_col in self.df.columns:
            logger.info(f"  - Epoch IDs: {self.epoch_col}")
        logger.info(f"  - Target protein: FIXED (learned as parameter)")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            Dict with:
                - 'mol_data': Molecule graph data
                - 'final_score': Binding affinity score
                - 'epoch_id': Epoch ID (optional)
        """
        row = self.df.iloc[idx]
        
        # ========== GET MOLECULE DATA ==========
        mol_name = str(row[self.molecule_name_col]).strip()
        
        # Convert molecule name to SMILES
        smiles = get_smiles_from_reaction(mol_name)
        if not smiles:
            raise ValueError(f"Could not convert {mol_name} to SMILES")
        
        # Process molecule
        mol_data = self.preprocessor.process_smiles(smiles)
        if mol_data is None:
            raise ValueError(f"Could not process SMILES: {smiles}")
        
        # ========== GET SCORE ==========
        final_score = float(row[self.score_col])
        
        # ========== GET EPOCH ID (OPTIONAL) ==========
        epoch_id = None
        if self.epoch_col in self.df.columns:
            try:
                epoch_val = row[self.epoch_col]
                if pd.notna(epoch_val):
                    epoch_id = int(epoch_val)
            except (ValueError, TypeError):
                epoch_id = None
        
        return {
            'mol_data': mol_data,
            'final_score': final_score,
            'epoch_id': epoch_id,
        }
    
    def get_statistics(self):
        """Get dataset statistics."""
        scores = self.df[self.score_col].values
        
        stats = {
            'num_samples': len(self.df),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'score_median': float(pd.Series(scores).median()),
        }
        
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Samples: {stats['num_samples']}")
        logger.info(f"  Score range: [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")
        logger.info(f"  Score mean: {stats['score_mean']:.4f} ± {stats['score_std']:.4f}")
        logger.info(f"  Score median: {stats['score_median']:.4f}")
        
        return stats


# ============================================================================
# FACTORY FUNCTION: Create dataset based on type
# ============================================================================

def create_dataset(
    dataset_type='multi_target',
    data_path=None,
    dataframe=None,
    molecule_name_col='molecule_name',
    target_protein_col='target_protein',
    target_seq_col='target_seq',
    score_col='final_score',
    epoch_col='epoch',
    preprocessor=None,
    **kwargs
):
    """
    Factory function to create dataset.
    
    Args:
        dataset_type: 'multi_target' or 'single_target'
        data_path: Path to CSV file
        dataframe: Pandas DataFrame
        molecule_name_col: Column name for molecule names
        target_protein_col: Column name for target protein codes
        target_seq_col: Column name for target sequences
        score_col: Column name for scores
        epoch_col: Column name for epoch IDs
        preprocessor: MoleculePreprocessor instance
        **kwargs: Additional arguments
    
    Returns:
        Dataset instance
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type == 'multi_target':
        return BoltzDataset(
            data_path=data_path,
            dataframe=dataframe,
            molecule_name_col=molecule_name_col,
            target_protein_col=target_protein_col,
            target_seq_col=target_seq_col,
            score_col=score_col,
            epoch_col=epoch_col,
            preprocessor=preprocessor,
        )
    
    elif dataset_type == 'single_target':
        return SingleTargetDataset(
            data_path=data_path,
            dataframe=dataframe,
            molecule_name_col=molecule_name_col,
            score_col=score_col,
            epoch_col=epoch_col,
            preprocessor=preprocessor,
        )
    
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Choose from: 'multi_target', 'single_target'"
        )
