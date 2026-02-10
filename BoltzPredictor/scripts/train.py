"""Training script for BoltzPredictor - Single-Target Only.

Single-target: f(molecule) → binding_affinity (FIXED protein)

UPDATED: Optimized for small value range (-0.2 to 0.2) with Combined Loss
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.serialization
from tqdm import tqdm
import numpy as np
import logging
import json
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from boltzpredictor.models import SingleTargetAffinityPredictor, create_model
from boltzpredictor.data import SingleTargetDataset, create_dataset
from boltzpredictor.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# PyTorch 2.6+ compatibility: Allow numpy scalars in checkpoint loading
def safe_torch_load(path, map_location='cpu'):
    """Safely load PyTorch checkpoint with numpy scalar support."""
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    return torch.load(path, map_location=map_location, weights_only=False)


# ============================================================================
# LOSS FUNCTIONS - OPTIMIZED FOR SMALL VALUE RANGE (-0.2 to 0.2)
# ============================================================================

class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error Loss
    Perfect for small value ranges like (-0.2 to 0.2)
    
    Formula: MAPE = (1/n) * Σ|y_pred - y_true| / |y_true|
    
    Why MAPE for small ranges:
    - Focuses on relative error, not absolute
    - 0.01 error on 0.12 = 8.3% (penalized)
    - 0.01 error on 1.0 = 1% (less penalized)
    - Better gradient signals for optimization
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        logger.info(f"Initialized MAPELoss (epsilon={epsilon})")
    
    def forward(self, y_pred, y_true):
        # Squeeze to 1D if needed
        y_pred = y_pred.squeeze(-1) if y_pred.dim() > 1 else y_pred
        y_true = y_true.squeeze(-1) if y_true.dim() > 1 else y_true
        
        # Avoid division by zero
        denominator = torch.abs(y_true) + self.epsilon
        
        # Calculate percentage error
        percentage_error = torch.abs(y_pred - y_true) / denominator
        
        # Return mean percentage error (multiply by 100 for percentage)
        loss = torch.mean(percentage_error) * 100
        
        return loss


class RankingLoss(nn.Module):
    """
    Ranking Loss - Focuses on relative ordering
    Perfect for competition scoring where ranking matters more than absolute values
    
    Why Ranking Loss:
    - Penalizes incorrect pairwise rankings
    - If y_true_i > y_true_j, then y_pred_i should > y_pred_j
    - Doesn't care about absolute values
    - Perfect for leaderboard optimization
    """
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin
        logger.info(f"Initialized RankingLoss (margin={margin})")
    
    def forward(self, y_pred, y_true):
        # Squeeze to 1D if needed
        y_pred = y_pred.squeeze(-1) if y_pred.dim() > 1 else y_pred
        y_true = y_true.squeeze(-1) if y_true.dim() > 1 else y_true
        
        # Create pairwise differences
        # Shape: (batch_size, batch_size)
        y_pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        y_true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
        
        # Get signs of true differences
        true_signs = torch.sign(y_true_diff)
        
        # Ranking loss: penalize if prediction sign doesn't match true sign
        # If true_diff > 0, we want pred_diff > 0
        # If true_diff < 0, we want pred_diff < 0
        ranking_loss = torch.nn.functional.relu(
            -y_pred_diff * true_signs + self.margin
        )
        
        # Mask out diagonal (same sample) and zero differences
        mask = ~torch.eye(y_pred.size(0), dtype=torch.bool, device=y_pred.device)
        mask = mask & (true_signs != 0)  # Only consider pairs with different targets
        
        ranking_loss = ranking_loss[mask]
        
        if ranking_loss.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        
        return torch.mean(ranking_loss)


class CombinedLoss(nn.Module):
    """
    Combined Loss = α * MAPE + β * Ranking Loss
    
    Best of both worlds:
    - MAPE ensures accuracy in the (-0.2 to 0.2) range
    - Ranking ensures correct ordering for leaderboard
    
    For score range (-0.2 to 0.2):
    - alpha=0.7, beta=0.3: Focus more on accuracy
    - alpha=0.5, beta=0.5: Balanced (RECOMMENDED)
    - alpha=0.3, beta=0.7: Focus more on ranking
    """
    def __init__(self, alpha=0.5, beta=0.5, margin=0.01, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.epsilon = epsilon
        
        self.mape_loss = MAPELoss(epsilon=epsilon)
        self.ranking_loss = RankingLoss(margin=margin)
        logger.info(
            f"Initialized CombinedLoss (alpha={alpha}, beta={beta}, margin={margin})"
        )
    
    def forward(self, y_pred, y_true):
        # Squeeze to 1D if needed
        y_pred = y_pred.squeeze(-1) if y_pred.dim() > 1 else y_pred
        y_true = y_true.squeeze(-1) if y_true.dim() > 1 else y_true
        
        # Calculate both losses
        mape = self.mape_loss(y_pred, y_true)
        ranking = self.ranking_loss(y_pred, y_true)
        
        # Combine with weights
        total_loss = self.alpha * mape + self.beta * ranking
        
        return total_loss, mape, ranking


class HuberLossSmallRange(nn.Module):
    """
    Huber Loss adapted for small value ranges
    Smooth transition between L2 (small errors) and L1 (large errors)
    
    Why Huber for small ranges:
    - Smooth gradients for small errors (L2 behavior)
    - Robust to outliers (L1 behavior)
    - delta=0.02 is 10% of your range (-0.2 to 0.2)
    """
    def __init__(self, delta=0.02):
        super().__init__()
        self.delta = delta
        logger.info(f"Initialized HuberLossSmallRange (delta={delta})")
    
    def forward(self, y_pred, y_true):
        # Squeeze to 1D if needed
        y_pred = y_pred.squeeze(-1) if y_pred.dim() > 1 else y_pred
        y_true = y_true.squeeze(-1) if y_true.dim() > 1 else y_true
        
        error = y_pred - y_true
        
        # Huber loss formula
        loss = torch.where(
            torch.abs(error) <= self.delta,
            0.5 * error ** 2,  # L2 for small errors
            self.delta * (torch.abs(error) - 0.5 * self.delta)  # L1 for large errors
        )
        
        return torch.mean(loss)


class TargetNormalizer:
    """Handles target normalization and de-normalization."""
    
    def __init__(self, method='standardization'):
        """
        Args:
            method: 'standardization' or 'robust'
        """
        self.method = method
        self.mean = None
        self.std = None
        self.q25 = None
        self.q75 = None
        
    def fit(self, targets):
        """Compute normalization statistics from training data."""
        targets = np.array(targets).flatten()
        
        if self.method == 'standardization':
            self.mean = float(np.mean(targets))
            self.std = float(np.std(targets))
            if self.std < 1e-8:
                self.std = 1.0
                logger.warning("Target std is near zero, setting to 1.0")
        elif self.method == 'robust':
            self.q25 = float(np.percentile(targets, 25))
            self.q75 = float(np.percentile(targets, 75))
            iqr = self.q75 - self.q25
            if iqr < 1e-8:
                iqr = 1.0
            self.mean = float(np.median(targets))
            self.std = iqr / 1.35
        
        logger.info(f"Target normalizer fitted ({self.method})")
        logger.info(f"  Mean: {self.mean:.6f}, Std: {self.std:.6f}")
        
    def normalize(self, targets):
        """Normalize targets."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return (np.array(targets) - self.mean) / self.std
    
    def denormalize(self, targets_norm):
        """De-normalize predictions."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return np.array(targets_norm) * self.std + self.mean
    
    def to_dict(self):
        """Serialize for saving."""
        return {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'q25': self.q25,
            'q75': self.q75,
        }
    
    @staticmethod
    def from_dict(data):
        """Deserialize from saved dict."""
        normalizer = TargetNormalizer(method=data['method'])
        normalizer.mean = data['mean']
        normalizer.std = data['std']
        normalizer.q25 = data.get('q25')
        normalizer.q75 = data.get('q75')
        return normalizer


class EarlyStoppingRegression:
    """Early stopping for regression tasks."""
    
    def __init__(self, patience=3, min_delta=0.01, metric='rmse', mode='min'):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum relative improvement (0.01 = 1%)
            metric: 'rmse' or 'mae'
            mode: 'min' for minimization
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.best_epoch = None
    
    def __call__(self, current_value, epoch):
        """
        Args:
            current_value: Current metric value (RMSE or MAE)
            epoch: Current epoch number (0-indexed)
        
        Returns:
            should_stop: bool
        """
        if self.best_value is None:
            # First epoch - initialize best value
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            logger.info(
                f"Initial {self.metric.upper()}: {self.best_value:.6f} at epoch {self.best_epoch + 1}"
            )
            return False
        
        if self.mode == 'min':
            # Calculate relative improvement
            improvement = (self.best_value - current_value) / (abs(self.best_value) + 1e-8)
            
            if improvement > self.min_delta:
                # ✅ IMPROVEMENT FOUND - Update best and reset counter
                self.best_value = current_value
                self.best_epoch = epoch
                self.counter = 0
                logger.info(
                    f"✓ {self.metric.upper()} improved: {self.best_value:.6f} at epoch {self.best_epoch + 1} "
                    f"(improvement: {improvement*100:.2f}%)"
                )
                return False
            else:
                # ❌ NO IMPROVEMENT - Increment counter
                self.counter += 1
                logger.info(
                    f"No improvement in {self.metric.upper()}. "
                    f"Patience: {self.counter}/{self.patience}"
                )
                
                # Check if patience exceeded
                if self.counter > self.patience:
                    logger.info(
                        f"Early stopping triggered! Best {self.metric.upper()}: "
                        f"{self.best_value:.6f} at epoch {self.best_epoch + 1}"
                    )
                    return True
                
                return False
        
        return False


# ============================================================================
# METRICS CALCULATION - OPTIMIZED FOR SMALL RANGES
# ============================================================================

def calculate_metrics_small_range(y_pred, y_true):
    """
    Calculate metrics suitable for small value ranges (-0.2 to 0.2)
    
    Returns:
        dict with: MAPE, MAE, RMSE, Ranking_Accuracy
    """
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    # Mean Absolute Percentage Error
    epsilon = 1e-6
    mape = np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + epsilon))) * 100
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Ranking Accuracy (% of correct pairwise rankings)
    n = len(y_pred)
    if n > 1:
        correct_pairs = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
                    correct_pairs += 1
                total_pairs += 1
        ranking_accuracy = correct_pairs / total_pairs * 100 if total_pairs > 0 else 0.0
    else:
        ranking_accuracy = 0.0
    
    return {
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse,
        'Ranking_Accuracy': ranking_accuracy,
    }


# ============================================================================
# COLLATE FUNCTION - SINGLE TARGET ONLY
# ============================================================================

def collate_fn_single_target(batch):
    """Collate function for single-target batching (NO protein sequences)."""
    mol_x_list = []
    mol_edge_index_list = []
    mol_edge_attr_list = []
    mol_batch_list = []
    
    final_scores = []
    
    node_offset = 0
    
    for item in batch:
        mol_data = item['mol_data']
        mol_x_list.append(mol_data['x'])
        mol_edge_index_list.append(mol_data['edge_index'] + node_offset)
        mol_edge_attr_list.append(mol_data['edge_attr'])
        
        num_nodes = mol_data['x'].size(0)
        mol_batch_list.append(torch.full((num_nodes,), len(mol_batch_list), dtype=torch.long))
        node_offset += num_nodes
        
        final_scores.append(item['final_score'])
    
    mol_data_batch = {
        'x': torch.cat(mol_x_list, dim=0),
        'edge_index': torch.cat(mol_edge_index_list, dim=1),
        'edge_attr': torch.cat(mol_edge_attr_list, dim=0),
        'batch': torch.cat(mol_batch_list, dim=0),
    }
    
    final_scores = torch.tensor(final_scores, dtype=torch.float32).unsqueeze(-1)
    
    return {
        'mol_data': mol_data_batch,
        'final_score': final_scores,
    }


# ============================================================================
# TRAINING FUNCTIONS - SINGLE TARGET ONLY
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None, 
                normalizer=None, add_noise=False, noise_std=0.01, phase_name=""):
    """Train for one epoch - Single-target version (NO protein encoding)."""
    model.train()
    total_loss = 0.0
    predictions_list = []
    targets_list = []
    
    logger.info(f"Starting training epoch {epoch + 1} {phase_name}")
    logger.debug(f"Number of batches: {len(dataloader)}, Batch size: {dataloader.batch_size}")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} {phase_name}")
    for batch_idx, batch in enumerate(pbar):
        mol_data = {k: v.to(device) for k, v in batch['mol_data'].items()}
        final_scores = batch['final_score'].to(device)
        
        # Normalize targets
        if normalizer is not None:
            final_scores_np = final_scores.cpu().numpy()
            final_scores_norm = normalizer.normalize(final_scores_np)
            final_scores = torch.tensor(final_scores_norm, dtype=torch.float32, device=device)
            
            if add_noise:
                noise = torch.randn_like(final_scores) * noise_std
                final_scores = final_scores + noise
        
        # Forward pass - Single-target: ONLY molecule (fixed target)
        outputs = model(mol_data=mol_data)
        
        predictions = outputs['final_score']
        
        # Compute loss - Check if criterion returns tuple (combined loss)
        loss_output = criterion(predictions, final_scores)
        
        if isinstance(loss_output, tuple):
            # Combined loss returns (total_loss, mape, ranking)
            loss, mape, ranking = loss_output
        else:
            # Standard loss returns scalar
            loss = loss_output
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        
        # Store for metrics
        predictions_list.append(predictions.detach().cpu().numpy())
        targets_list.append(final_scores.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar(f'Train{phase_name}/Loss', loss.item(), global_step)
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            logger.debug(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute regression metrics
    predictions_all = np.concatenate(predictions_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    
    metrics = calculate_metrics_small_range(predictions_all, targets_all)
    mape = metrics['MAPE']
    mae = metrics['MAE']
    rmse = metrics['RMSE']
    ranking_acc = metrics['Ranking_Accuracy']
    
    logger.info(
        f"Epoch {epoch + 1} training complete {phase_name} - "
        f"Loss: {avg_loss:.4f} | MAPE: {mape:.2f}%, MAE: {mae:.6f}, RMSE: {rmse:.6f}, Ranking Acc: {ranking_acc:.2f}%"
    )
    
    return avg_loss, mape, mae, rmse, ranking_acc


def validate(model, dataloader, criterion, device, normalizer=None, phase_name=""):
    """Validate model - Single-target version."""
    model.eval()
    total_loss = 0.0
    predictions_list = []
    targets_list = []
    
    logger.info(f"Starting validation {phase_name} - {len(dataloader)} batches")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating {phase_name}"):
            mol_data = {k: v.to(device) for k, v in batch['mol_data'].items()}
            final_scores = batch['final_score'].to(device)
            
            # Normalize targets
            if normalizer is not None:
                final_scores_np = final_scores.cpu().numpy()
                final_scores_norm = normalizer.normalize(final_scores_np)
                final_scores = torch.tensor(final_scores_norm, dtype=torch.float32, device=device)
            
            # Forward pass (NO protein sequences)
            outputs = model(mol_data=mol_data)
            
            predictions = outputs['final_score']
            
            # Compute loss
            loss_output = criterion(predictions, final_scores)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]
            else:
                loss = loss_output
            
            total_loss += loss.item()
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(final_scores.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # Compute metrics
    metrics = calculate_metrics_small_range(predictions, targets)
    mape = metrics['MAPE']
    mae = metrics['MAE']
    rmse = metrics['RMSE']
    ranking_acc = metrics['Ranking_Accuracy']
    
    logger.info(
        f"Validation complete {phase_name} - Loss: {avg_loss:.4f} | "
        f"MAPE: {mape:.2f}%, MAE: {mae:.6f}, RMSE: {rmse:.6f}, Ranking Acc: {ranking_acc:.2f}%"
    )
    
    return avg_loss, mape, mae, rmse, ranking_acc


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_config(config):
    """Validate and set defaults for configuration."""
    logger.info("Validating configuration...")
    
    # Required keys
    required_keys = [
        'train_data',
        'checkpoint_dir',
        'cache_dir',
        'log_dir',
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    logger.info(f"Model type: SINGLE-TARGET")
    
    # Set defaults for optional keys
    defaults = {
        # Data
        'molecule_name_col': 'molecule_name',
        'score_col': 'final_score',
        'epoch_col': 'epoch',
        
        # Model
        'mol_hidden_dim': 256,
        'interaction_dim': 512,
        'num_interaction_layers': 3,
        'target_embedding_dim': 512,  # For single-target
        'dropout': 0.2,
        
        # Optimization - UPDATED FOR SMALL RANGES
        'learning_rate': 0.01,  # INCREASED from 5e-5
        'weight_decay': 0.0001,  # UPDATED
        'optimizer': 'adamw',
        'batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'num_epochs': 20,
        
        # Early stopping
        'early_stopping_patience': 10,  # INCREASED from 3
        'early_stopping_min_delta': 0.01,
        'early_stopping_metric': 'rmse',
        
        # Scheduler
        'scheduler': 'cosine_annealing',
        'warmup_ratio': 0.08,
        
        # Loss - UPDATED FOR SMALL RANGES
        'loss_function': 'combined',  # CHANGED from 'mse' to 'combined'
        'loss_alpha': 0.5,  # MAPE weight
        'loss_beta': 0.5,   # Ranking weight
        'loss_margin': 0.01,  # Ranking margin
        'huber_delta': 0.02,  # UPDATED for small range
        'smooth_l1_beta': 1.0,
        
        # Normalization
        'normalization_method': 'standardization',
        
        # Label noise
        'add_label_noise': False,
        'label_noise_std': 0.01,
        
        # Reproducibility
        'random_seed': 42,
        
        # Phase 2
        'phase2_epoch_adjustment': 3,  # Number of epochs beyond best phase 1 epoch
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            logger.debug(f"Using default for '{key}': {default_value}")
    
    # Validate specific values
    valid_loss_functions = ['mse', 'huber', 'smooth_l1', 'mape', 'ranking', 'combined']
    if config['loss_function'] not in valid_loss_functions:
        raise ValueError(
            f"Invalid loss_function: {config['loss_function']}. "
            f"Must be one of {valid_loss_functions}"
        )
    
    valid_normalization_methods = ['standardization', 'robust']
    if config['normalization_method'] not in valid_normalization_methods:
        raise ValueError(
            f"Invalid normalization_method: {config['normalization_method']}. "
            f"Must be one of {valid_normalization_methods}"
        )
    
    valid_schedulers = ['cosine_annealing', 'linear', 'exponential']
    if config['scheduler'] not in valid_schedulers:
        raise ValueError(
            f"Invalid scheduler: {config['scheduler']}. "
            f"Must be one of {valid_schedulers}"
        )
    
    logger.info("✅ Configuration validated successfully")
    return config


def create_loss_criterion(config):
    """
    Create loss criterion based on config
    
    UPDATED: Supports new loss functions optimized for small ranges
    """
    loss_type = config['loss_function'].lower()
    
    if loss_type == 'mse':
        logger.info("Using MSE Loss (standard)")
        return nn.MSELoss()
    
    elif loss_type == 'mape':
        logger.info("Using MAPE Loss (optimized for small ranges)")
        return MAPELoss(epsilon=1e-6)
    
    elif loss_type == 'ranking':
        margin = float(config.get('loss_margin', 0.01))
        logger.info(f"Using Ranking Loss (margin={margin})")
        return RankingLoss(margin=margin)
    
    elif loss_type == 'combined':
        alpha = float(config.get('loss_alpha', 0.5))
        beta = float(config.get('loss_beta', 0.5))
        margin = float(config.get('loss_margin', 0.01))
        logger.info(f"Using Combined Loss (MAPE + Ranking)")
        logger.info(f"  α (MAPE weight): {alpha}")
        logger.info(f"  β (Ranking weight): {beta}")
        logger.info(f"  Ranking margin: {margin}")
        return CombinedLoss(alpha=alpha, beta=beta, margin=margin, epsilon=1e-6)
    
    elif loss_type == 'huber':
        delta = float(config.get('huber_delta', 0.02))
        logger.info(f"Using Huber Loss (delta={delta})")
        return HuberLossSmallRange(delta=delta)
    
    elif loss_type == 'smooth_l1':
        logger.info("Using Smooth L1 Loss")
        return nn.SmoothL1Loss()
    
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train BoltzPredictor Single-Target (Two-Phase: Selection + Full Retrain)"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                       help='Override training phase (1=selection 80/20, 2=retrain 100%)')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config file is empty or invalid YAML")
    
    # Validate config
    config = validate_config(config)
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging(
        log_level=log_level,
        log_dir=config['log_dir']
    )
    
    logger.info("=" * 80)
    logger.info("Starting BoltzPredictor Training - Single-Target")
    logger.info("(Two-Phase: Selection + Full Retrain)")
    logger.info(f"Loss Function: {config['loss_function'].upper()}")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Log file: {log_file}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    logger.info("Loading datasets...")
    logger.info(f"Train data: {config['train_data']}")
    
    # ✅ Create dataset - SINGLE TARGET ONLY
    full_dataset = create_dataset(
        dataset_type='single_target',
        data_path=config['train_data'],
        molecule_name_col=config['molecule_name_col'],
        score_col=config['score_col'],
        epoch_col=config['epoch_col'],
    )
    logger.info(f"Full dataset loaded: {len(full_dataset)} samples")
    
    # ========== TARGET NORMALIZATION (fit on full data) ==========
    normalizer_path = os.path.join(config['checkpoint_dir'], 'normalizer.json')
    
    if os.path.exists(normalizer_path):
        logger.info(f"Loading existing normalizer from {normalizer_path}")
        with open(normalizer_path, 'r') as f:
            normalizer_dict = json.load(f)
        normalizer = TargetNormalizer.from_dict(normalizer_dict)
        logger.info(f"Normalizer loaded: method={normalizer.method}, mean={normalizer.mean:.6f}, std={normalizer.std:.6f}")
    else:
        logger.info("Fitting target normalizer on full dataset...")
        normalizer = TargetNormalizer(method=config['normalization_method'])
        full_targets = []
        for i in range(len(full_dataset)):
            try:
                item = full_dataset[i]
                full_targets.append(item['final_score'])
            except Exception as e:
                logger.debug(f"Error loading item {i}: {e}")
                continue
        
        if full_targets:
            normalizer.fit(full_targets)
            with open(normalizer_path, 'w') as f:
                json.dump(normalizer.to_dict(), f, indent=2)
            logger.info(f"Normalizer saved to {normalizer_path}")
        else:
            logger.error("No valid targets found for normalization!")
            raise ValueError("Cannot fit normalizer - no valid data")
    
    # Determine which phase to run
    current_phase = 1
    phase1_best_rmse_path = os.path.join(config['checkpoint_dir'], 'phase1_best_rmse.json')
    phase1_best_epoch_path = os.path.join(config['checkpoint_dir'], 'phase1_best_epoch.json')
    
    if os.path.exists(phase1_best_rmse_path) and os.path.exists(phase1_best_epoch_path):
        logger.info("Phase 1 already completed. Proceeding to Phase 2.")
        current_phase = 2
    
    if args.phase is not None:
        current_phase = args.phase
        logger.info(f"Phase override: starting with phase {current_phase}")
    
    # ========== PHASE 1: MODEL SELECTION (80/20 SPLIT) ==========
    if current_phase == 1:
        logger.info("=" * 80)
        logger.info("PHASE 1: Model Selection (80/20 Train/Val Split)")
        logger.info("=" * 80)
        
        # Split dataset: 80/20
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Use fixed seed for reproducibility
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config['random_seed'])
        )
        
        logger.info(f"Train split: {len(train_dataset)} samples (80%)")
        logger.info(f"Val split: {len(val_dataset)} samples (20%)")
        
        # Create data loaders
        batch_size = int(config['batch_size'])
        num_workers = int(config['num_workers'])
        pin_memory = bool(config['pin_memory'])
        logger.info(f"Creating data loaders - Batch size: {batch_size}, Workers: {num_workers}, Pin memory: {pin_memory}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_single_target,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.info(f"Train loader: {len(train_loader)} batches")
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_single_target,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.info(f"Val loader: {len(val_loader)} batches")
        
        # Create model - SINGLE TARGET ONLY
        logger.info("Creating model...")
        model = create_model(
            model_type='single_target',
            mol_hidden_dim=int(config['mol_hidden_dim']),
            interaction_dim=int(config['interaction_dim']),
            num_layers=int(config['num_interaction_layers']),
            target_embedding_dim=int(config.get('target_embedding_dim', 512)),
            dropout=float(config['dropout']),
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created - Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Create optimizer
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params_list)
        lr = float(config['learning_rate'])
        weight_decay = float(config['weight_decay'])
        logger.info(f"Optimizer: AdamW, LR: {lr}, Weight decay: {weight_decay}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
        
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        num_epochs = int(config['num_epochs'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_loader),
            T_mult=1,
            eta_min=1e-6,
        )
        logger.info(f"Scheduler: Cosine annealing with warm restarts")
        
        # Early stopping
        early_stopping = EarlyStoppingRegression(
            patience=int(config['early_stopping_patience']),
            min_delta=float(config['early_stopping_min_delta']),
            metric=config['early_stopping_metric'],
            mode='min',
        )
        
        # Loss function
        criterion = create_loss_criterion(config)
        
        # TensorBoard
        tensorboard_dir = os.path.join(config['log_dir'], 'phase1')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")
        
        writer.add_text('Config/Phase', 'Phase 1: Model Selection', 0)
        writer.add_text('Config/ModelType', 'SINGLE-TARGET', 0)
        writer.add_text('Config/LossFunction', config['loss_function'].upper(), 0)
        writer.add_text('Config/Normalizer', 
                       f'Method: {normalizer.method}, Mean: {normalizer.mean:.6f}, Std: {normalizer.std:.6f}', 0)
        writer.flush()
        
        add_noise = bool(config['add_label_noise'])
        noise_std = float(config['label_noise_std'])
        
        # Training loop for Phase 1
        logger.info("=" * 80)
        logger.info(f"PHASE 1: Training for up to {num_epochs} epochs")
        logger.info("=" * 80)
        
        best_val_rmse = float('inf')
        best_epoch = -1
        
        for epoch in range(num_epochs):
            logger.info("-" * 80)
            logger.info(f"PHASE 1 - Epoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 80)
            
            # Train
            train_loss, train_mape, train_mae, train_rmse, train_ranking_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                normalizer=normalizer, add_noise=add_noise, noise_std=noise_std,
                phase_name="[Phase1]"
            )
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Train MAPE: {train_mape:.2f}%, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, Ranking Acc: {train_ranking_acc:.2f}%")
            
            writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            writer.add_scalar('Train/MAPE', train_mape, epoch)
            writer.add_scalar('Train/MAE', train_mae, epoch)
            writer.add_scalar('Train/RMSE', train_rmse, epoch)
            writer.add_scalar('Train/RankingAccuracy', train_ranking_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # Validate
            val_loss, val_mape, val_mae, val_rmse, val_ranking_acc = validate(
                model, val_loader, criterion, device, normalizer=normalizer,
                phase_name="[Phase1]"
            )
            logger.info(f"Val Loss: {val_loss:.4f}, MAPE: {val_mape:.2f}%, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, Ranking Acc: {val_ranking_acc:.2f}%")
            
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/MAPE', val_mape, epoch)
            writer.add_scalar('Val/MAE', val_mae, epoch)
            writer.add_scalar('Val/RMSE', val_rmse, epoch)
            writer.add_scalar('Val/RankingAccuracy', val_ranking_acc, epoch)
            
            # Save best model
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_phase1.pt')
                torch.save({
                    'epoch': epoch,
                    'phase': 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_rmse': float(best_val_rmse),
                    'config': config,
                }, checkpoint_path)
                logger.info(f"✓ New best model (Phase 1)! Val RMSE: {best_val_rmse:.6f}")
                logger.info(f"  Saved to: {checkpoint_path}")
            
            # Early stopping
            if early_stopping(val_rmse, epoch):
                logger.info("Early stopping triggered in Phase 1!")
                break
            
            # Save checkpoint
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_phase1_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_rmse': float(best_val_rmse),
                'config': config,
            }, checkpoint_path)
            
            scheduler.step()
        
        writer.close()
        
        logger.info("=" * 80)
        logger.info(f"PHASE 1 COMPLETE")
        logger.info(f"Best Val RMSE: {best_val_rmse:.6f}")
        logger.info(f"Best Epoch: {best_epoch + 1}")
        logger.info("=" * 80)
        
        # Save phase 1 results
        with open(phase1_best_rmse_path, 'w') as f:
            json.dump({'best_val_rmse': float(best_val_rmse)}, f)
        with open(phase1_best_epoch_path, 'w') as f:
            json.dump({'best_epoch': int(best_epoch)}, f)
        
        logger.info("Phase 1 complete. Run Phase 2 to retrain on 100% data.")
        logger.info(f"Command: python train.py --config {args.config} --phase 2")
        
        return
    
    # ========== PHASE 2: FINAL TRAINING (100% DATA) ==========
    if current_phase == 2:
        logger.info("=" * 80)
        logger.info("PHASE 2: Final Training on 100% Data")
        logger.info("=" * 80)
        
        # Load phase 1 results
        if os.path.exists(phase1_best_rmse_path) and os.path.exists(phase1_best_epoch_path):
            with open(phase1_best_rmse_path, 'r') as f:
                phase1_results = json.load(f)
            with open(phase1_best_epoch_path, 'r') as f:
                phase1_epoch_results = json.load(f)
            
            best_phase1_rmse = float(phase1_results['best_val_rmse'])
            best_phase1_epoch = int(phase1_epoch_results['best_epoch'])
            
            logger.info(f"Phase 1 Results:")
            logger.info(f"  Best Val RMSE: {best_phase1_rmse:.6f}")
            logger.info(f"  Best Epoch: {best_phase1_epoch + 1}")
        else:
            logger.warning("Phase 1 results not found. Using defaults.")
            best_phase1_epoch = int(config['num_epochs']) // 2
        
        # Use full dataset (no validation split)
        batch_size = int(config['batch_size'])
        num_workers = int(config['num_workers'])
        pin_memory = bool(config['pin_memory'])
        
        full_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_single_target,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.info(f"Full dataset loader: {len(full_loader)} batches ({len(full_dataset)} samples)")
        
        # Create model - SINGLE TARGET ONLY
        logger.info("Creating model...")
        model = create_model(
            model_type='single_target',
            mol_hidden_dim=int(config['mol_hidden_dim']),
            interaction_dim=int(config['interaction_dim']),
            num_layers=int(config['num_interaction_layers']),
            target_embedding_dim=int(config.get('target_embedding_dim', 512)),
            dropout=float(config['dropout']),
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created - Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Load best phase 1 model weights
        best_phase1_path = os.path.join(config['checkpoint_dir'], 'best_phase1.pt')
        if os.path.exists(best_phase1_path):
            logger.info(f"Loading best Phase 1 model from {best_phase1_path}")
            checkpoint = safe_torch_load(best_phase1_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Phase 1 model weights loaded")
        else:
            logger.warning("Best Phase 1 model not found. Starting with random initialization.")
        
        # Create optimizer with FIXED hyperparameters from Phase 1
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params_list)
        lr = float(config['learning_rate'])
        weight_decay = float(config['weight_decay'])
        logger.info(f"Phase 2 - Using FIXED hyperparameters from Phase 1")
        logger.info(f"Optimizer: AdamW, LR: {lr}, Weight decay: {weight_decay}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
        
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(full_loader),
            T_mult=1,
            eta_min=1e-6,
        )
        logger.info(f"Scheduler: Cosine annealing with warm restarts")
        
        # Loss function
        criterion = create_loss_criterion(config)
        
        # TensorBoard
        tensorboard_dir = os.path.join(config['log_dir'], 'phase2')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")
        
        writer.add_text('Config/Phase', 'Phase 2: Final Training on 100% Data', 0)
        writer.add_text('Config/ModelType', 'SINGLE-TARGET', 0)
        writer.add_text('Config/LossFunction', config['loss_function'].upper(), 0)
        writer.add_text('Config/Normalizer', 
                       f'Method: {normalizer.method}, Mean: {normalizer.mean:.6f}, Std: {normalizer.std:.6f}', 0)
        writer.flush()
        
        add_noise = bool(config['add_label_noise'])
        noise_std = float(config['label_noise_std'])
        
        # ========== FIXED NUMBER OF EPOCHS (best_epoch + adjustment) ==========
        phase2_epoch_adjustment = int(config.get('phase2_epoch_adjustment', 3))
        phase2_epochs = best_phase1_epoch + 1 + phase2_epoch_adjustment
        
        logger.info("=" * 80)
        logger.info(f"PHASE 2: Training for FIXED {phase2_epochs} epochs")
        logger.info(f"(Based on best epoch from Phase 1: {best_phase1_epoch + 1} + adjustment: {phase2_epoch_adjustment})")
        logger.info("=" * 80)
        
        for epoch in range(phase2_epochs):
            logger.info("-" * 80)
            logger.info(f"PHASE 2 - Epoch {epoch + 1}/{phase2_epochs}")
            logger.info("-" * 80)
            
            # Train (NO validation)
            train_loss, train_mape, train_mae, train_rmse, train_ranking_acc = train_epoch(
                model, full_loader, criterion, optimizer, device, epoch, writer,
                normalizer=normalizer, add_noise=add_noise, noise_std=noise_std,
                phase_name="[Phase2]"
            )
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Train MAPE: {train_mape:.2f}%, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, Ranking Acc: {train_ranking_acc:.2f}%")
            
            writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            writer.add_scalar('Train/MAPE', train_mape, epoch)
            writer.add_scalar('Train/MAE', train_mae, epoch)
            writer.add_scalar('Train/RMSE', train_rmse, epoch)
            writer.add_scalar('Train/RankingAccuracy', train_ranking_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_phase2_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'phase': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            scheduler.step()
        
        # Save final model
        final_model_path = os.path.join(config['checkpoint_dir'], 'final_model.pt')
        torch.save({
            'phase': 2,
            'model_state_dict': model.state_dict(),
            'config': config,
        }, final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
        
        # Save training summary
        summary_path = os.path.join(config['checkpoint_dir'], 'training_summary.json')
        summary = {
            'phase': 2,
            'total_epochs': phase2_epochs,
            'final_model': final_model_path,
            'normalizer': normalizer_path,
            'config': {
                'model_type': 'single_target',
                'loss_function': config['loss_function'],
                'learning_rate': lr,
                'batch_size': batch_size,
                'num_workers': num_workers,
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Training summary saved: {summary_path}")
        
        writer.close()
        
        logger.info("=" * 80)
        logger.info("PHASE 2 COMPLETE - Final Model Ready for Inference")
        logger.info("=" * 80)
        logger.info(f"Final model: {final_model_path}")
        logger.info(f"Normalizer: {normalizer_path}")
        logger.info(f"Training summary: {summary_path}")


if __name__ == '__main__':
    main()
