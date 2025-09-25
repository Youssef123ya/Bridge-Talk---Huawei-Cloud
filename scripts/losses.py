import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tensor

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, 
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""

    def __init__(self, 
                 num_classes: int,
                 smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = F.log_softmax(pred, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.sum(-true_dist * pred, dim=1).mean()

class ArcFaceLoss(nn.Module):
    """ArcFace loss for improved feature learning"""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 s: float = 30.0,
                 m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        # Normalize features and weights
        input = F.normalize(input, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(input, weight)

        # Add margin to target classes
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m))

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, label)

class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CenterLoss(nn.Module):
    """Center loss for intra-class compactness"""

    def __init__(self, 
                 num_classes: int,
                 feat_dim: int,
                 lambda_c: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        batch_size = features.size(0)

        # Compute distances to centers
        centers_batch = self.centers.index_select(0, labels.long())
        criterion = (features - centers_batch).pow(2).sum() / 2.0 / batch_size

        return self.lambda_c * criterion

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""

    def __init__(self, 
                 temperature: float = 0.07,
                 contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: Tensor, labels: Tensor = None, mask: Tensor = None) -> Tensor:
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

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
            self.temperature)

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

def get_loss_function(loss_name: str, 
                     num_classes: int = 32,
                     **kwargs) -> nn.Module:
    """Factory function to get loss function by name"""

    loss_name = loss_name.lower()

    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss(**kwargs)

    elif loss_name == 'focal':
        return FocalLoss(**kwargs)

    elif loss_name == 'label_smoothing':
        return LabelSmoothingLoss(num_classes=num_classes, **kwargs)

    elif loss_name == 'arcface':
        in_features = kwargs.get('in_features', 512)
        return ArcFaceLoss(in_features=in_features, out_features=num_classes, **kwargs)

    elif loss_name == 'center':
        feat_dim = kwargs.get('feat_dim', 512)
        return CenterLoss(num_classes=num_classes, feat_dim=feat_dim, **kwargs)

    elif loss_name == 'supcon':
        return SupConLoss(**kwargs)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
