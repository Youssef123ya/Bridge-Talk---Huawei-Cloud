"""
Custom Loss Functions Module for Arabic Sign Language Recognition
Provides specialized loss functions optimized for sign language classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in sign language recognition
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean', ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.ones(32) * alpha  # Assuming 32 classes
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute alpha_t if alpha is provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss for better generalization
    Prevents the model from becoming overconfident on training data
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss
        
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smooth labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CenterLoss(nn.Module):
    """
    Center Loss for learning discriminative features
    Encourages features of the same class to be close to their center
    """
    
    def __init__(self, num_classes: int, feature_dim: int, lambda_c: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c
        
        # Initialize class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss
        
        Args:
            features: Feature representations [batch_size, feature_dim]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Center loss value
        """
        batch_size = features.size(0)
        
        # Compute distances to centers
        centers_batch = self.centers[targets]  # [batch_size, feature_dim]
        
        # Compute center loss
        center_loss = F.mse_loss(features, centers_batch, reduction='sum') / (2 * batch_size)
        
        return self.lambda_c * center_loss
    
    def update_centers(self, features: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5):
        """
        Update class centers using exponential moving average
        
        Args:
            features: Feature representations
            targets: Ground truth labels
            alpha: Update rate
        """
        with torch.no_grad():
            for class_id in range(self.num_classes):
                mask = (targets == class_id)
                if mask.sum() > 0:
                    class_features = features[mask]
                    center_update = class_features.mean(dim=0)
                    self.centers[class_id] = (1 - alpha) * self.centers[class_id] + alpha * center_update


class TripletLoss(nn.Module):
    """
    Triplet Loss for learning embeddings where same class samples are closer
    than different class samples by a margin
    """
    
    def __init__(self, margin: float = 1.0, p: int = 2, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor samples [batch_size, feature_dim]
            positive: Positive samples (same class as anchor) [batch_size, feature_dim]
            negative: Negative samples (different class) [batch_size, feature_dim]
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        
        # Compute triplet loss
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return triplet_loss.mean()
        elif self.reduction == 'sum':
            return triplet_loss.sum()
        else:
            return triplet_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for learning better representations
    Reference: Khosla et al. "Supervised Contrastive Learning"
    """
    
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all', base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss
        
        Args:
            features: Feature representations [batch_size, feature_dim] or [batch_size, n_views, feature_dim]
            labels: Ground truth labels [batch_size]
            mask: Contrastive mask [batch_size, batch_size]
            
        Returns:
            Supervised contrastive loss value
        """
        device = features.device
        
        if len(features.shape) != 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for transferring knowledge from teacher to student model
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_logits: Logits from student model [batch_size, num_classes]
            teacher_logits: Logits from teacher model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Distillation loss value
        """
        # Distillation loss (soft targets)
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


class DiceLoss(nn.Module):
    """
    Dice Loss adapted for classification tasks
    Useful when dealing with class imbalance
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute dice loss
        
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Dice loss value
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Compute dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross entropy loss
        
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Weighted cross entropy loss value
        """
        if self.class_weights is not None and self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights
    """
    
    def __init__(self, losses: List[Tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute combined loss
        
        Returns:
            Combined weighted loss value
        """
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(*args, **kwargs)
        
        return total_loss


# Loss factory functions
def get_focal_loss(alpha: Optional[List[float]] = None, gamma: float = 2.0) -> FocalLoss:
    """Create focal loss with given parameters"""
    alpha_tensor = torch.tensor(alpha) if alpha is not None else None
    return FocalLoss(alpha=alpha_tensor, gamma=gamma)


def get_label_smoothing_loss(num_classes: int = 32, smoothing: float = 0.1) -> LabelSmoothingLoss:
    """Create label smoothing loss"""
    return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)


def get_weighted_loss(class_counts: List[int]) -> WeightedCrossEntropyLoss:
    """Create weighted cross entropy loss from class counts"""
    # Compute inverse frequency weights
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    
    return WeightedCrossEntropyLoss(class_weights=weight_tensor)


def get_class_balanced_loss(beta: float = 0.9999, num_classes: int = 32) -> Callable:
    """
    Create class-balanced loss function
    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples"
    """
    def create_cb_loss(class_counts: List[int]):
        # Compute effective numbers
        effective_nums = [(1 - beta ** n) / (1 - beta) for n in class_counts]
        
        # Compute weights
        weights = [(1 - beta) / en for en in effective_nums]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        
        return WeightedCrossEntropyLoss(class_weights=weight_tensor)
    
    return create_cb_loss


# Loss registry
LOSS_FUNCTIONS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'center': CenterLoss,
    'triplet': TripletLoss,
    'supcon': SupConLoss,
    'distillation': DistillationLoss,
    'dice': DiceLoss,
    'weighted_ce': WeightedCrossEntropyLoss,
}


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name
    
    Args:
        name: Loss function name
        **kwargs: Loss function parameters
        
    Returns:
        Loss function instance
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_FUNCTIONS.keys())}")
    
    return LOSS_FUNCTIONS[name](**kwargs)


def create_adaptive_loss_weights(train_dataset, validation_split: float = 0.1) -> torch.Tensor:
    """
    Create adaptive loss weights based on class distribution
    
    Args:
        train_dataset: Training dataset
        validation_split: Fraction for validation
        
    Returns:
        Class weights tensor
    """
    # Get class counts
    if hasattr(train_dataset, 'df'):
        # CSV dataset
        class_counts = train_dataset.df['label'].value_counts().sort_index().values
    else:
        # Regular dataset
        class_counts = [0] * train_dataset.num_classes
        for _, label in train_dataset:
            class_counts[label] += 1
    
    # Compute inverse frequency weights
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_counts) * max(count, 1)) for count in class_counts]
    
    return torch.tensor(weights, dtype=torch.float32)