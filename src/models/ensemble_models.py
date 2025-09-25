"""
Ensemble Models Module for Arabic Sign Language Recognition
Provides ensemble learning techniques to combine multiple models for improved accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path

from .base_model import BaseSignLanguageModel
from .cnn_architectures import CNN_ARCHITECTURES


class VotingEnsemble(BaseSignLanguageModel):
    """
    Voting ensemble that combines predictions from multiple models
    Supports both hard voting (majority) and soft voting (average probabilities)
    """
    
    def __init__(self, models: List[BaseSignLanguageModel], voting: str = 'soft', 
                 weights: Optional[List[float]] = None, num_classes: int = 32, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.models = nn.ModuleList(models)
        self.voting = voting  # 'hard' or 'soft'
        self.weights = weights
        self.num_models = len(models)
        
        if self.weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            # Normalize weights
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        
        # Validate models
        for model in self.models:
            if model.num_classes != num_classes:
                raise ValueError(f"All models must have the same number of classes ({num_classes})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.voting == 'soft':
            # Soft voting: average the probabilities
            predictions = []
            for i, model in enumerate(self.models):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                weighted_probs = probs * self.weights[i]
                predictions.append(weighted_probs)
            
            # Average weighted probabilities
            ensemble_probs = torch.stack(predictions).sum(dim=0)
            # Convert back to logits for loss computation
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            return ensemble_logits
            
        else:
            # Hard voting: majority vote
            predictions = []
            for model in self.models:
                logits = model(x)
                pred_classes = torch.argmax(logits, dim=1)
                predictions.append(pred_classes)
            
            # Stack predictions and find mode
            pred_stack = torch.stack(predictions, dim=0)  # [num_models, batch_size]
            
            # Convert to one-hot and sum
            batch_size = x.size(0)
            vote_counts = torch.zeros(batch_size, self.num_classes, device=x.device)
            
            for i, preds in enumerate(predictions):
                weight = self.weights[i]
                for j, pred in enumerate(preds):
                    vote_counts[j, pred] += weight
            
            return vote_counts
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = []
        for i, model in enumerate(self.models):
            info = model.get_model_info()
            model_info.append({
                'architecture': info['architecture'],
                'weight': self.weights[i],
                'parameters': info['total_parameters']
            })
        
        return {
            'architecture': 'VotingEnsemble',
            'voting_type': self.voting,
            'num_models': self.num_models,
            'models': model_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'description': f'{self.voting.capitalize()} voting ensemble of {self.num_models} models'
        }


class StackingEnsemble(BaseSignLanguageModel):
    """
    Stacking ensemble that uses a meta-learner to combine base model predictions
    Base models make predictions, then a meta-model learns how to combine them
    """
    
    def __init__(self, base_models: List[BaseSignLanguageModel], meta_learner: Optional[nn.Module] = None,
                 num_classes: int = 32, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.base_models = nn.ModuleList(base_models)
        self.num_base_models = len(base_models)
        
        # Validate base models
        for model in self.base_models:
            if model.num_classes != num_classes:
                raise ValueError(f"All base models must have the same number of classes ({num_classes})")
        
        # Create meta-learner if not provided
        if meta_learner is None:
            meta_input_size = self.num_base_models * num_classes
            self.meta_learner = nn.Sequential(
                nn.Linear(meta_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        else:
            self.meta_learner = meta_learner
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            base_predictions.append(probs)
        
        # Concatenate all predictions
        meta_input = torch.cat(base_predictions, dim=1)  # [batch_size, num_models * num_classes]
        
        # Meta-learner makes final prediction
        final_logits = self.meta_learner(meta_input)
        
        return final_logits
    
    def freeze_base_models(self):
        """Freeze base models for meta-learner training"""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_base_models(self):
        """Unfreeze base models for fine-tuning"""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get stacking ensemble information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        base_model_info = []
        for model in self.base_models:
            info = model.get_model_info()
            base_model_info.append({
                'architecture': info['architecture'],
                'parameters': info['total_parameters']
            })
        
        meta_params = sum(p.numel() for p in self.meta_learner.parameters())
        
        return {
            'architecture': 'StackingEnsemble',
            'num_base_models': self.num_base_models,
            'base_models': base_model_info,
            'meta_learner_parameters': meta_params,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'description': f'Stacking ensemble with {self.num_base_models} base models and meta-learner'
        }


class BaggingEnsemble(BaseSignLanguageModel):
    """
    Bagging ensemble that trains multiple models on different subsets of data
    Each model sees a bootstrap sample of the training data
    """
    
    def __init__(self, base_architecture: str, num_models: int = 5, num_classes: int = 32, 
                 bootstrap_ratio: float = 0.8, **model_kwargs):
        super().__init__(num_classes=num_classes)
        
        self.base_architecture = base_architecture
        self.num_models = num_models
        self.bootstrap_ratio = bootstrap_ratio
        
        # Create multiple instances of the same architecture
        if base_architecture not in CNN_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {base_architecture}")
        
        arch_class = CNN_ARCHITECTURES[base_architecture]
        self.models = nn.ModuleList([
            arch_class(num_classes=num_classes, **model_kwargs) 
            for _ in range(num_models)
        ])
        
        # Store bootstrap indices for each model (will be set during training)
        self.bootstrap_indices = [None] * num_models
    
    def set_bootstrap_indices(self, dataset_size: int, seed: int = 42):
        """Set bootstrap indices for each model"""
        np.random.seed(seed)
        sample_size = int(dataset_size * self.bootstrap_ratio)
        
        self.bootstrap_indices = []
        for i in range(self.num_models):
            # Bootstrap sampling with replacement
            indices = np.random.choice(dataset_size, size=sample_size, replace=True)
            self.bootstrap_indices.append(indices)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average predictions from all models
        predictions = []
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
        
        # Average probabilities
        ensemble_probs = torch.stack(predictions).mean(dim=0)
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        return ensemble_logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get bagging ensemble information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get info from first model (all are the same architecture)
        base_info = self.models[0].get_model_info()
        
        return {
            'architecture': 'BaggingEnsemble',
            'base_architecture': self.base_architecture,
            'num_models': self.num_models,
            'bootstrap_ratio': self.bootstrap_ratio,
            'base_model_info': base_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'description': f'Bagging ensemble of {self.num_models} {self.base_architecture} models'
        }


class AdaptiveEnsemble(BaseSignLanguageModel):
    """
    Adaptive ensemble that learns to weight models based on their confidence
    Models with higher confidence on specific samples get higher weights
    """
    
    def __init__(self, models: List[BaseSignLanguageModel], num_classes: int = 32, 
                 temperature: float = 1.0, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.temperature = temperature
        
        # Validate models
        for model in self.models:
            if model.num_classes != num_classes:
                raise ValueError(f"All models must have the same number of classes ({num_classes})")
        
        # Attention mechanism to compute adaptive weights
        self.attention = nn.Sequential(
            nn.Linear(num_classes * self.num_models, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_models),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from all models
        model_predictions = []
        model_probs = []
        
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits / self.temperature, dim=1)
            model_predictions.append(logits)
            model_probs.append(probs)
        
        # Concatenate all probabilities for attention
        all_probs = torch.cat(model_probs, dim=1)  # [batch_size, num_models * num_classes]
        
        # Compute adaptive weights
        attention_weights = self.attention(all_probs)  # [batch_size, num_models]
        
        # Weight and combine predictions
        weighted_probs = torch.zeros_like(model_probs[0])
        for i, probs in enumerate(model_probs):
            weight = attention_weights[:, i:i+1]  # [batch_size, 1]
            weighted_probs += weight * probs
        
        # Convert back to logits
        ensemble_logits = torch.log(weighted_probs + 1e-8)
        
        return ensemble_logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get adaptive ensemble information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = []
        for model in self.models:
            info = model.get_model_info()
            model_info.append({
                'architecture': info['architecture'],
                'parameters': info['total_parameters']
            })
        
        attention_params = sum(p.numel() for p in self.attention.parameters())
        
        return {
            'architecture': 'AdaptiveEnsemble',
            'num_models': self.num_models,
            'temperature': self.temperature,
            'models': model_info,
            'attention_parameters': attention_params,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'description': f'Adaptive ensemble with attention mechanism over {self.num_models} models'
        }


# Ensemble factory functions
def create_voting_ensemble(architectures: List[str], num_classes: int = 32, 
                          voting: str = 'soft', weights: Optional[List[float]] = None,
                          **model_kwargs) -> VotingEnsemble:
    """
    Create a voting ensemble from architecture names
    
    Args:
        architectures: List of architecture names
        num_classes: Number of output classes
        voting: 'soft' or 'hard' voting
        weights: Optional weights for each model
        **model_kwargs: Additional arguments for model creation
        
    Returns:
        VotingEnsemble instance
    """
    models = []
    for arch_name in architectures:
        if arch_name not in CNN_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch_name}")
        
        arch_class = CNN_ARCHITECTURES[arch_name]
        model = arch_class(num_classes=num_classes, **model_kwargs)
        models.append(model)
    
    return VotingEnsemble(models=models, voting=voting, weights=weights, num_classes=num_classes)


def create_stacking_ensemble(base_architectures: List[str], num_classes: int = 32,
                           meta_learner: Optional[nn.Module] = None, **model_kwargs) -> StackingEnsemble:
    """
    Create a stacking ensemble from architecture names
    
    Args:
        base_architectures: List of base architecture names
        num_classes: Number of output classes
        meta_learner: Optional custom meta-learner
        **model_kwargs: Additional arguments for model creation
        
    Returns:
        StackingEnsemble instance
    """
    base_models = []
    for arch_name in base_architectures:
        if arch_name not in CNN_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch_name}")
        
        arch_class = CNN_ARCHITECTURES[arch_name]
        model = arch_class(num_classes=num_classes, **model_kwargs)
        base_models.append(model)
    
    return StackingEnsemble(base_models=base_models, meta_learner=meta_learner, num_classes=num_classes)


def create_bagging_ensemble(architecture: str, num_models: int = 5, num_classes: int = 32,
                          bootstrap_ratio: float = 0.8, **model_kwargs) -> BaggingEnsemble:
    """
    Create a bagging ensemble
    
    Args:
        architecture: Base architecture name
        num_models: Number of models in ensemble
        num_classes: Number of output classes
        bootstrap_ratio: Fraction of data to use for each model
        **model_kwargs: Additional arguments for model creation
        
    Returns:
        BaggingEnsemble instance
    """
    return BaggingEnsemble(
        base_architecture=architecture,
        num_models=num_models,
        num_classes=num_classes,
        bootstrap_ratio=bootstrap_ratio,
        **model_kwargs
    )


# Ensemble registry
ENSEMBLE_TYPES = {
    'voting': create_voting_ensemble,
    'stacking': create_stacking_ensemble,
    'bagging': create_bagging_ensemble,
}


def create_ensemble(ensemble_type: str, **kwargs):
    """
    Create an ensemble by type name
    
    Args:
        ensemble_type: Type of ensemble ('voting', 'stacking', 'bagging')
        **kwargs: Arguments specific to the ensemble type
        
    Returns:
        Ensemble model instance
    """
    if ensemble_type not in ENSEMBLE_TYPES:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}. Available: {list(ENSEMBLE_TYPES.keys())}")
    
    return ENSEMBLE_TYPES[ensemble_type](**kwargs)