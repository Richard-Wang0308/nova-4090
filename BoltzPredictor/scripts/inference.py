"""
Inference script for BoltzPredictor - Single-Target Only.

Single-target: f(molecule) → binding_affinity (FIXED protein)
NO protein sequences needed!

FIXED: Proper denormalization of predictions
"""

import argparse
import torch
import torch.serialization
import pandas as pd
from tqdm import tqdm
import os
import sys
import logging
import time
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from boltzpredictor.models import create_model
from boltzpredictor.data import MoleculePreprocessor
from boltzpredictor.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def safe_torch_load(path, map_location='cpu'):
    """Safely load PyTorch checkpoint with numpy scalar support (PyTorch 2.6+)."""
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    return torch.load(path, map_location=map_location, weights_only=False)


def load_normalizer(checkpoint_dir):
    """
    Load target normalizer from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing normalizer.json
        
    Returns:
        Dictionary with normalization parameters or None
    """
    normalizer_path = os.path.join(checkpoint_dir, 'normalizer.json')
    if os.path.exists(normalizer_path):
        logger.info(f"Loading normalizer from {normalizer_path}")
        with open(normalizer_path, 'r') as f:
            normalizer_dict = json.load(f)
        logger.info(f"Normalizer method: {normalizer_dict.get('method')}")
        logger.info(f"  Mean: {normalizer_dict.get('mean'):.6f}")
        logger.info(f"  Std:  {normalizer_dict.get('std'):.6f}")
        return normalizer_dict
    else:
        logger.warning(f"Normalizer not found at {normalizer_path}")
        logger.warning("Predictions will be in normalized (Z-score) space")
        return None


def denormalize_predictions(predictions, normalizer_dict):
    """
    Denormalize predictions using saved normalizer statistics.
    
    Handles both single values and arrays.
    
    Args:
        predictions: Single value, list, or numpy array of normalized predictions
        normalizer_dict: Dictionary with 'method', 'mean', 'std' keys
        
    Returns:
        Denormalized predictions in original scale
    """
    if normalizer_dict is None:
        logger.warning("No normalizer provided. Returning raw predictions (normalized space).")
        return predictions
    
    method = normalizer_dict.get('method', 'standardization')
    mean = normalizer_dict.get('mean')
    std = normalizer_dict.get('std')
    
    if mean is None or std is None:
        logger.warning("Normalizer missing mean/std. Returning raw predictions.")
        return predictions
    
    # Convert to numpy array for processing
    predictions_array = np.array(predictions)
    is_scalar = predictions_array.ndim == 0
    
    if method == 'standardization':
        # Z-score denormalization: x = z * std + mean
        denormalized = predictions_array * std + mean
        logger.debug(f"Applied Z-score denormalization: x = z * {std:.6f} + {mean:.6f}")
        
    elif method == 'minmax':
        # Min-max denormalization: x = z * (max - min) + min
        min_val = normalizer_dict.get('min')
        max_val = normalizer_dict.get('max')
        if min_val is None or max_val is None:
            logger.warning("Min-max normalizer missing min/max values. Returning raw predictions.")
            return predictions
        denormalized = predictions_array * (max_val - min_val) + min_val
        logger.debug(f"Applied Min-max denormalization: x = z * ({max_val:.6f} - {min_val:.6f}) + {min_val:.6f}")
        
    else:
        logger.warning(f"Unknown normalization method: {method}. Returning raw predictions.")
        return predictions
    
    # Return as scalar if input was scalar
    if is_scalar:
        return denormalized.item()
    
    return denormalized.tolist() if isinstance(denormalized, np.ndarray) else denormalized


def create_model_from_checkpoint(checkpoint, device):
    """
    Create and load model from checkpoint - SINGLE TARGET ONLY.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        device: torch device
        
    Returns:
        Loaded model in evaluation mode
    """
    config = checkpoint.get('config', {})
    
    logger.info(f"Model type: SINGLE-TARGET")
    logger.info(f"Config keys: {list(config.keys())}")
    
    # Create single-target model
    model = create_model(
        model_type='single_target',
        mol_hidden_dim=int(config.get('mol_hidden_dim', 256)),
        interaction_dim=int(config.get('interaction_dim', 512)),
        num_layers=int(config.get('num_interaction_layers', 3)),
        target_embedding_dim=int(config.get('target_embedding_dim', 512)),
        dropout=float(config.get('dropout', 0.1)),
    ).to(device)
    
    # Load weights
    logger.info("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded - Total parameters: {total_params:,}")
    logger.info("Model set to evaluation mode ✅\n")
    
    return model


def infer_single(model, preprocessor, smiles, device, normalizer_dict=None):
    """
    Inference for single molecule - Single-target version.
    
    NO protein sequence needed!
    
    Args:
        model: PyTorch model
        preprocessor: Molecule preprocessor
        smiles: SMILES string
        device: torch device
        normalizer_dict: Normalizer dictionary (optional)
        
    Returns:
        Tuple of (normalized_score, denormalized_score or None)
    """
    logger.debug("Processing molecule...")
    start_time = time.time()
    
    try:
        mol_data = preprocessor.process_smiles(smiles)
        mol_data = {k: v.to(device) for k, v in mol_data.items()}
    except Exception as e:
        logger.error(f"Failed to process SMILES: {e}")
        return None, None
    
    process_time = time.time() - start_time
    logger.debug(f"Molecule processed in {process_time*1000:.2f} ms")
    
    logger.debug("Running inference...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Single-target: ONLY molecule data needed!
            outputs = model(mol_data=mol_data)
        final_score_normalized = outputs['final_score'].item()
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None, None
    
    inference_time = time.time() - start_time
    logger.debug(f"Inference completed in {inference_time*1000:.2f} ms")
    
    # Denormalize if normalizer provided
    final_score_denormalized = None
    if normalizer_dict:
        final_score_denormalized = denormalize_predictions(
            final_score_normalized,
            normalizer_dict
        )
    
    return final_score_normalized, final_score_denormalized


def infer_batch(model, preprocessor, smiles_list, device, batch_size=32, normalizer_dict=None):
    """
    Batch inference - Single-target version.
    
    NO protein sequences needed!
    
    Args:
        model: PyTorch model
        preprocessor: Molecule preprocessor
        smiles_list: List of SMILES strings
        device: torch device
        batch_size: Batch size for processing
        normalizer_dict: Normalizer dictionary (optional)
        
    Returns:
        Tuple of (predictions_normalized, predictions_denormalized, failed_indices)
    """
    predictions_normalized = []
    predictions_denormalized = []
    failed_indices = []
    
    total_start_time = time.time()
    
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing batches"):
        batch_smiles = smiles_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug(f"Processing batch {batch_num} ({len(batch_smiles)} samples)")
        
        try:
            # Process molecules
            batch_start = time.time()
            mol_data_batch = preprocessor.batch_process(batch_smiles)
            mol_data_batch = {k: v.to(device) for k, v in mol_data_batch.items()}
            process_time = time.time() - batch_start
            logger.debug(f"  Batch {batch_num}: Molecule processing: {process_time*1000:.2f} ms")
            
            # Predict (NO protein sequences needed!)
            inference_start = time.time()
            with torch.no_grad():
                # Single-target: ONLY molecule data!
                outputs = model(mol_data=mol_data_batch)
            inference_time = time.time() - inference_start
            logger.debug(f"  Batch {batch_num}: Inference: {inference_time*1000:.2f} ms")
            
            # Extract predictions (normalized)
            batch_predictions_norm = outputs['final_score'].cpu().numpy().flatten().tolist()
            predictions_normalized.extend(batch_predictions_norm)
            
            # Denormalize if normalizer provided
            if normalizer_dict:
                batch_predictions_denorm = denormalize_predictions(
                    batch_predictions_norm,
                    normalizer_dict
                )
                if isinstance(batch_predictions_denorm, (list, np.ndarray)):
                    predictions_denormalized.extend(batch_predictions_denorm)
                else:
                    predictions_denormalized.extend([batch_predictions_denorm])
            else:
                predictions_denormalized.extend([None] * len(batch_predictions_norm))
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            logger.debug(f"Exception details:", exc_info=True)
            failed_indices.extend(range(i, min(i + batch_size, len(smiles_list))))
            # Add None values for failed samples
            predictions_normalized.extend([None] * len(batch_smiles))
            predictions_denormalized.extend([None] * len(batch_smiles))
    
    total_time = time.time() - total_start_time
    logger.info(f"Processed {len(smiles_list)} samples in {total_time:.2f} seconds")
    logger.info(f"Average time per sample: {total_time/len(smiles_list)*1000:.2f} ms")
    
    if failed_indices:
        logger.warning(f"Failed to process {len(failed_indices)} samples")
    
    return predictions_normalized, predictions_denormalized, failed_indices


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with BoltzPredictor - Single-Target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single molecule prediction (normalized space)
  python inference.py \\
    --checkpoint checkpoints/model.pt \\
    --smiles "CCO"

  # Single molecule prediction (denormalized to original scale)
  python inference.py \\
    --checkpoint checkpoints/model.pt \\
    --smiles "CCO" \\
    --denormalize

  # Batch prediction from file (denormalized)
  python inference.py \\
    --checkpoint checkpoints/model.pt \\
    --input_file molecules.csv \\
    --output_file predictions.csv \\
    --denormalize \\
    --batch_size 64

  # Debug mode
  python inference.py \\
    --checkpoint checkpoints/model.pt \\
    --input_file molecules.csv \\
    --denormalize \\
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
        '--smiles',
        type=str,
        default=None,
        help='Single SMILES string for prediction'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='CSV file with column: smiles'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    parser.add_argument(
        '--denormalize',
        action='store_true',
        help='Denormalize predictions to original scale using training statistics'
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
    logger.info("BoltzPredictor Inference - Single-Target")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    start_time = time.time()
    checkpoint = safe_torch_load(args.checkpoint, map_location=device)
    load_time = time.time() - start_time
    logger.info(f"Checkpoint loaded in {load_time:.2f} seconds\n")
    
    # Create model from checkpoint
    model = create_model_from_checkpoint(checkpoint, device)
    
    # Load normalizer if denormalization requested
    normalizer_dict = None
    if args.denormalize:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        normalizer_dict = load_normalizer(checkpoint_dir)
        if not normalizer_dict:
            logger.warning("⚠️ Denormalization requested but normalizer not found")
            logger.warning("   Predictions will be in normalized space")
    
    # Create preprocessor
    logger.info("Initializing molecule preprocessor...")
    preprocessor = MoleculePreprocessor()
    logger.info("✅ Preprocessor initialized\n")
    
    # ========== SINGLE PREDICTION ==========
    if args.smiles is not None:
        logger.info("=" * 80)
        logger.info("Single Molecule Prediction")
        logger.info("=" * 80)
        logger.info(f"SMILES: {args.smiles}")
        logger.info(f"Target: FIXED (learned during training)")
        logger.info(f"Denormalize: {args.denormalize}\n")
        
        final_score_norm, final_score_denorm = infer_single(
            model,
            preprocessor,
            args.smiles,
            device,
            normalizer_dict
        )
        
        if final_score_norm is None:
            logger.error("❌ Inference failed!")
            return 1
        
        logger.info("-" * 80)
        logger.info("Results:")
        logger.info("-" * 80)
        logger.info(f"Final Score (normalized):    {final_score_norm:.6f}")
        
        if args.denormalize and final_score_denorm is not None:
            logger.info(f"Final Score (original scale): {final_score_denorm:.6f} ✅")
        
        logger.info("-" * 80 + "\n")
        
        return 0
    
    # ========== BATCH PREDICTION FROM FILE ==========
    if args.input_file is None:
        raise ValueError("Either --smiles or --input_file must be provided")
    
    logger.info("=" * 80)
    logger.info("Batch Prediction from File")
    logger.info("=" * 80)
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Denormalize: {args.denormalize}\n")
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    start_time = time.time()
    df = pd.read_csv(args.input_file)
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(df):,} samples in {load_time:.2f} seconds")
    
    # Check required columns
    if 'smiles' not in df.columns:
        raise ValueError("Missing required column: 'smiles'")
    
    logger.info(f"Target: FIXED (learned during training)")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of batches: {(len(df) + args.batch_size - 1) // args.batch_size}\n")
    
    # Process in batches
    smiles_list = df['smiles'].tolist()
    predictions_norm, predictions_denorm, failed_samples = infer_batch(
        model,
        preprocessor,
        smiles_list,
        device,
        args.batch_size,
        normalizer_dict
    )
    
    logger.info("")
    
    # Add predictions to dataframe
    df['predicted_score_normalized'] = predictions_norm
    
    if args.denormalize:
        df['predicted_score'] = predictions_denorm
        logger.info("✅ Predictions denormalized to original scale")
    else:
        df['predicted_score'] = predictions_norm
        logger.info("⚠️ Predictions in normalized (Z-score) space")
    
    # Save results
    output_file = args.output_file or args.input_file.replace('.csv', '_predictions.csv')
    logger.info(f"Saving predictions to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Summary statistics
    valid_predictions_norm = [p for p in predictions_norm if p is not None]
    valid_predictions_denorm = [p for p in predictions_denorm if p is not None]
    
    if valid_predictions_norm:
        logger.info("\n" + "=" * 80)
        logger.info("Prediction Statistics (Normalized):")
        logger.info("=" * 80)
        logger.info(f"  Total samples:     {len(df):,}")
        logger.info(f"  Successful:        {len(valid_predictions_norm):,}")
        logger.info(f"  Failed:            {len(failed_samples):,}")
        logger.info(f"  Mean:              {np.mean(valid_predictions_norm):.6f}")
        logger.info(f"  Std:               {np.std(valid_predictions_norm):.6f}")
        logger.info(f"  Min:               {np.min(valid_predictions_norm):.6f}")
        logger.info(f"  Max:               {np.max(valid_predictions_norm):.6f}")
        
        if args.denormalize and valid_predictions_denorm:
            logger.info("\n" + "=" * 80)
            logger.info("Prediction Statistics (Original Scale):")
            logger.info("=" * 80)
            logger.info(f"  Mean:              {np.mean(valid_predictions_denorm):.6f}")
            logger.info(f"  Std:               {np.std(valid_predictions_denorm):.6f}")
            logger.info(f"  Min:               {np.min(valid_predictions_denorm):.6f}")
            logger.info(f"  Max:               {np.max(valid_predictions_denorm):.6f}")
        
        logger.info("=" * 80 + "\n")
    
    logger.info("✅ Inference complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
