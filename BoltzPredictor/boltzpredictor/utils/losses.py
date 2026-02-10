"""Loss functions for training - Optimized for small value range (-0.2 to 0.2)."""

import torch
import torch.nn as nn
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# NEW LOSS FUNCTIONS - OPTIMIZED FOR SMALL VALUE RANGES
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
    
    Example:
        >>> loss_fn = MAPELoss()
        >>> pred = torch.tensor([[0.11], [0.15]])
        >>> target = torch.tensor([[0.12], [0.14]])
        >>> loss = loss_fn(pred, target)
        >>> print(loss)  # ~8.3% error
    """
    
    def __init__(self, epsilon=1e-6):
        """
        Initialize MAPE Loss.
        
        Args:
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
        logger.info(f"Initialized MAPELoss (epsilon={epsilon})")
    
    def forward(self, predictions, targets):
        """
        Compute MAPE loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar MAPE loss value (percentage)
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Ensure shapes are compatible
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        # Avoid division by zero
        denominator = torch.abs(targets) + self.epsilon
        
        # Calculate percentage error
        percentage_error = torch.abs(predictions - targets) / denominator
        
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
    
    Formula:
        For each pair (i, j):
        L = max(0, -sign(y_true_i - y_true_j) * (y_pred_i - y_pred_j) + margin)
    
    Example:
        >>> loss_fn = RankingLoss(margin=0.01)
        >>> pred = torch.tensor([[0.11], [0.15]])
        >>> target = torch.tensor([[0.12], [0.14]])
        >>> loss = loss_fn(pred, target)
        >>> print(loss)  # Ranking loss
    """
    
    def __init__(self, margin=0.01):
        """
        Initialize Ranking Loss.
        
        Args:
            margin: Margin for ranking pairs
        """
        super().__init__()
        self.margin = margin
        logger.info(f"Initialized RankingLoss (margin={margin})")
    
    def forward(self, predictions, targets):
        """
        Compute Ranking loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar ranking loss value
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Ensure shapes are compatible
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        # Create pairwise differences
        # Shape: (batch_size, batch_size)
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        
        # Get signs of true differences
        true_signs = torch.sign(target_diff)
        
        # Ranking loss: penalize if prediction sign doesn't match true sign
        # If true_diff > 0, we want pred_diff > 0
        # If true_diff < 0, we want pred_diff < 0
        ranking_loss = torch.nn.functional.relu(
            -pred_diff * true_signs + self.margin
        )
        
        # Mask out diagonal (same sample) and zero target differences
        mask = ~torch.eye(predictions.size(0), dtype=torch.bool, device=predictions.device)
        mask = mask & (true_signs != 0)  # Only consider pairs with different targets
        
        ranking_loss = ranking_loss[mask]
        
        if ranking_loss.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        
        return torch.mean(ranking_loss)


class CombinedLossSmallRange(nn.Module):
    """
    Combined Loss = α * MAPE + β * Ranking Loss
    
    Best of both worlds:
    - MAPE ensures accuracy in the (-0.2 to 0.2) range
    - Ranking ensures correct ordering for leaderboard
    
    For score range (-0.2 to 0.2):
    - alpha=0.7, beta=0.3: Focus more on accuracy
    - alpha=0.5, beta=0.5: Balanced (RECOMMENDED)
    - alpha=0.3, beta=0.7: Focus more on ranking
    
    Example:
        >>> loss_fn = CombinedLossSmallRange(alpha=0.5, beta=0.5)
        >>> pred = torch.tensor([[0.11], [0.15]])
        >>> target = torch.tensor([[0.12], [0.14]])
        >>> loss, mape, ranking = loss_fn(pred, target)
        >>> print(f"Total: {loss:.4f}, MAPE: {mape:.2f}%, Ranking: {ranking:.6f}")
    """
    
    def __init__(self, alpha=0.5, beta=0.5, margin=0.01, epsilon=1e-6):
        """
        Initialize Combined Loss.
        
        Args:
            alpha: Weight for MAPE loss (0.0-1.0)
            beta: Weight for Ranking loss (0.0-1.0)
            margin: Margin for ranking loss
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.epsilon = epsilon
        
        self.mape_loss = MAPELoss(epsilon=epsilon)
        self.ranking_loss = RankingLoss(margin=margin)
        
        logger.info(
            f"Initialized CombinedLossSmallRange "
            f"(alpha={alpha}, beta={beta}, margin={margin})"
        )
    
    def forward(self, predictions, targets):
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            Tuple of (total_loss, mape_loss, ranking_loss)
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Ensure shapes are compatible
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        # Calculate both losses
        mape = self.mape_loss(predictions, targets)
        ranking = self.ranking_loss(predictions, targets)
        
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
    
    Formula:
        L(y, ŷ) = {
            0.5 * (y - ŷ)^2              if |y - ŷ| <= delta
            delta * (|y - ŷ| - 0.5*delta) if |y - ŷ| > delta
        }
    
    Example:
        >>> loss_fn = HuberLossSmallRange(delta=0.02)
        >>> pred = torch.tensor([[0.11], [0.15]])
        >>> target = torch.tensor([[0.12], [0.14]])
        >>> loss = loss_fn(pred, target)
        >>> print(loss)
    """
    
    def __init__(self, delta=0.02):
        """
        Initialize Huber Loss for small ranges.
        
        Args:
            delta: Threshold for switching between L2 and L1 loss
                   Recommendation: 0.01-0.05 for small ranges
        """
        super().__init__()
        self.delta = delta
        logger.info(f"Initialized HuberLossSmallRange (delta={delta})")
    
    def forward(self, predictions, targets):
        """
        Compute Huber loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar Huber loss value
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Ensure shapes are compatible
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        error = predictions - targets
        
        # Huber loss formula
        loss = torch.where(
            torch.abs(error) <= self.delta,
            0.5 * error ** 2,  # L2 for small errors
            self.delta * (torch.abs(error) - 0.5 * self.delta)  # L1 for large errors
        )
        
        return torch.mean(loss)


# ============================================================================
# STANDARD LOSS FUNCTIONS (KEPT FOR COMPATIBILITY)
# ============================================================================

class RegressionLoss(nn.Module):
    """
    Simple MSE loss for regression tasks.
    
    Used in target-only training (no ranking loss, no antitarget).
    
    Formula: MSE = (1/n) * Σ(y_pred - y_true)^2
    
    Note: For small value ranges (-0.2 to 0.2), consider using
    MAPELoss or CombinedLossSmallRange instead.
    """
    
    def __init__(self):
        """Initialize MSE loss."""
        super().__init__()
        self.mse_loss = nn.MSELoss()
        logger.info("Initialized RegressionLoss (MSE)")
    
    def forward(self, predictions, targets):
        """
        Compute MSE loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar MSE loss value
        """
        # ✅ FIXED: Squeeze both to ensure consistent shapes
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Ensure shapes are compatible
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        loss = self.mse_loss(predictions, targets)
        return loss


class HuberLoss(nn.Module):
    """
    Huber loss for robust regression (less sensitive to outliers).
    
    L(y, ŷ) = {
        0.5 * (y - ŷ)^2              if |y - ŷ| <= delta
        delta * (|y - ŷ| - 0.5*delta) if |y - ŷ| > delta
    }
    
    Note: For small value ranges, use HuberLossSmallRange instead.
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between L2 and L1 loss
        """
        super().__init__()
        self.huber_loss = nn.HuberLoss(delta=delta, reduction='mean')
        self.delta = delta
        logger.info(f"Initialized HuberLoss (delta={delta})")
    
    def forward(self, predictions, targets):
        """
        Compute Huber loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar Huber loss value
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        loss = self.huber_loss(predictions, targets)
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss (also known as Huber loss in some frameworks).
    
    Similar to Huber loss but with a different formulation.
    """
    
    def __init__(self, beta=1.0):
        """
        Initialize Smooth L1 loss.
        
        Args:
            beta: Scaling factor for the loss
        """
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta, reduction='mean')
        self.beta = beta
        logger.info(f"Initialized SmoothL1Loss (beta={beta})")
    
    def forward(self, predictions, targets):
        """
        Compute Smooth L1 loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar Smooth L1 loss value
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        loss = self.smooth_l1_loss(predictions, targets)
        return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for handling imbalanced or weighted samples.
    
    Useful when different samples have different importance.
    """
    
    def __init__(self):
        """Initialize weighted MSE loss."""
        super().__init__()
        logger.info("Initialized WeightedMSELoss")
    
    def forward(self, predictions, targets, weights=None):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: Predicted scores [batch_size, 1] or [batch_size]
            targets: Target scores [batch_size, 1] or [batch_size]
            weights: Optional sample weights [batch_size, 1] or [batch_size]
        
        Returns:
            loss: scalar weighted MSE loss value
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        # Compute MSE for each sample
        mse = (predictions - targets) ** 2
        
        # Apply weights if provided
        if weights is not None:
            if weights.dim() > 1:
                weights = weights.squeeze(-1)
            
            if weights.shape[0] != predictions.shape[0]:
                raise ValueError(
                    f"Weight shape mismatch: {weights.shape[0]} vs {predictions.shape[0]}"
                )
            
            # Reshape weights for broadcasting
            weights = weights.unsqueeze(-1)
            mse = mse * weights
        
        loss = mse.mean()
        return loss


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_loss_function(loss_type='mse', **kwargs):
    """
    Factory function to create loss functions.
    
    UPDATED: Supports new loss functions optimized for small ranges
    
    Args:
        loss_type: Type of loss
                   Options: 'mse', 'mape', 'ranking', 'combined', 'huber', 'huber_small', 'smooth_l1', 'weighted_mse'
        **kwargs: Additional arguments for the loss function
                  - For 'combined': alpha, beta, margin
                  - For 'huber' or 'huber_small': delta
                  - For 'smooth_l1': beta
                  - For 'mape': epsilon
                  - For 'ranking': margin
    
    Returns:
        Loss function instance
    
    Examples:
        >>> # MSE loss (standard)
        >>> loss_fn = create_loss_function('mse')
        
        >>> # MAPE loss (for small ranges)
        >>> loss_fn = create_loss_function('mape')
        
        >>> # Ranking loss (for ordering)
        >>> loss_fn = create_loss_function('ranking', margin=0.01)
        
        >>> # Combined loss (RECOMMENDED for small ranges)
        >>> loss_fn = create_loss_function('combined', alpha=0.5, beta=0.5, margin=0.01)
        
        >>> # Huber loss (robust)
        >>> loss_fn = create_loss_function('huber_small', delta=0.02)
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'mse':
        logger.info("Creating MSE Loss")
        return RegressionLoss()
    
    elif loss_type == 'mape':
        epsilon = kwargs.get('epsilon', 1e-6)
        logger.info(f"Creating MAPE Loss (epsilon={epsilon})")
        return MAPELoss(epsilon=epsilon)
    
    elif loss_type == 'ranking':
        margin = kwargs.get('margin', 0.01)
        logger.info(f"Creating Ranking Loss (margin={margin})")
        return RankingLoss(margin=margin)
    
    elif loss_type == 'combined':
        alpha = kwargs.get('alpha', 0.5)
        beta = kwargs.get('beta', 0.5)
        margin = kwargs.get('margin', 0.01)
        epsilon = kwargs.get('epsilon', 1e-6)
        logger.info(
            f"Creating Combined Loss (alpha={alpha}, beta={beta}, margin={margin})"
        )
        return CombinedLossSmallRange(alpha=alpha, beta=beta, margin=margin, epsilon=epsilon)
    
    elif loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        logger.info(f"Creating Huber Loss (delta={delta})")
        return HuberLoss(delta=delta)
    
    elif loss_type == 'huber_small':
        delta = kwargs.get('delta', 0.02)
        logger.info(f"Creating Huber Loss for Small Ranges (delta={delta})")
        return HuberLossSmallRange(delta=delta)
    
    elif loss_type == 'smooth_l1':
        beta = kwargs.get('beta', 1.0)
        logger.info(f"Creating Smooth L1 Loss (beta={beta})")
        return SmoothL1Loss(beta=beta)
    
    elif loss_type == 'weighted_mse':
        logger.info("Creating Weighted MSE Loss")
        return WeightedMSELoss()
    
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Valid options: 'mse', 'mape', 'ranking', 'combined', 'huber', 'huber_small', 'smooth_l1', 'weighted_mse'"
        )


# ============================================================================
# DEPRECATED: The following classes are NO LONGER USED
# (Kept for reference only - can be removed in future versions)
# ============================================================================

class CombinedLoss(nn.Module):
    """
    DEPRECATED: Combined regression and ranking loss (OLD VERSION).
    
    This class is no longer used in the target-only version.
    Kept for backward compatibility only.
    
    Use CombinedLossSmallRange instead for small value ranges.
    
    L = α * L_reg + (1 - α) * L_rank
    """
    
    def __init__(self, regression_weight=0.7, ranking_weight=0.3, margin=0.1):
        super().__init__()
        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.mse_loss = nn.MSELoss()
        logger.warning(
            "CombinedLoss is DEPRECATED. Use CombinedLossSmallRange instead."
        )
    
    def forward(self, predictions, targets, epoch_mask=None):
        """
        Compute combined loss.
        
        DEPRECATED: Use CombinedLossSmallRange instead.
        
        Args:
            predictions: Predicted scores [batch_size, 1]
            targets: Target scores [batch_size, 1]
            epoch_mask: Optional mask to group samples by epoch for ranking loss
                        [batch_size] with epoch IDs
        
        Returns:
            Tuple of (total_loss, reg_loss, rank_loss)
        """
        # Squeeze to 1D if needed
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets
        
        # Regression loss
        reg_loss = self.mse_loss(predictions, targets)
        
        # Ranking loss (if epoch_mask provided)
        rank_loss = torch.tensor(0.0, device=predictions.device)
        if epoch_mask is not None and self.ranking_weight > 0:
            rank_loss = self._compute_ranking_loss(
                predictions, targets, epoch_mask
            )
        
        # Combined loss
        total_loss = (
            self.regression_weight * reg_loss +
            self.ranking_weight * rank_loss
        )
        
        return total_loss, reg_loss, rank_loss
    
    def _compute_ranking_loss(self, predictions, targets, epoch_mask):
        """
        Compute ranking loss within each epoch.
        
        For pairs (i, j) in the same epoch:
        L_rank = max(0, margin - (pred_i - pred_j)) if target_i > target_j
        """
        predictions = predictions.squeeze(-1)  # [batch_size]
        targets = targets.squeeze(-1)  # [batch_size]
        
        # Get unique epochs
        unique_epochs = torch.unique(epoch_mask)
        rank_losses = []
        
        for epoch_id in unique_epochs:
            # Get indices for this epoch
            epoch_indices = (epoch_mask == epoch_id).nonzero(as_tuple=True)[0]
            
            if len(epoch_indices) < 2:
                continue
            
            # Get predictions and targets for this epoch
            epoch_preds = predictions[epoch_indices]
            epoch_targets = targets[epoch_indices]
            
            # Create pairs where target_i > target_j
            n = len(epoch_indices)
            for i in range(n):
                for j in range(i + 1, n):
                    if epoch_targets[i] > epoch_targets[j]:
                        # pred_i should be > pred_j
                        diff = epoch_preds[i] - epoch_preds[j]
                        rank_loss = torch.clamp(self.margin - diff, min=0.0)
                        rank_losses.append(rank_loss)
                    elif epoch_targets[j] > epoch_targets[i]:
                        # pred_j should be > pred_i
                        diff = epoch_preds[j] - epoch_preds[i]
                        rank_loss = torch.clamp(self.margin - diff, min=0.0)
                        rank_losses.append(rank_loss)
        
        if len(rank_losses) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return torch.stack(rank_losses).mean()
