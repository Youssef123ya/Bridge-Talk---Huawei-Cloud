import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
from .base_model import BaseModel, ModelRegistry

@ModelRegistry.register('simple_ensemble')
class SimpleEnsemble(BaseModel):
    """Simple ensemble that averages predictions from multiple models"""

    def __init__(self, 
                 models: List[BaseModel],
                 weights: Optional[List[float]] = None,
                 **kwargs):
        super().__init__(models[0].num_classes, **kwargs)

        self.models = nn.ModuleList(models)

        # Set ensemble weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self.num_models = len(models)

        # Ensure all models have the same number of classes
        for i, model in enumerate(models):
            assert model.num_classes == self.num_classes, f"Model {i} has {model.num_classes} classes, expected {self.num_classes}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        predictions = []

        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                pred = model(x)
                predictions.append(pred)

        # Weighted average of predictions
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred

        return ensemble_pred

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the first model"""
        return self.models[0].get_features(x)

    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each individual model"""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        return predictions

    def get_prediction_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get ensemble prediction with confidence metrics"""
        individual_preds = self.get_individual_predictions(x)

        # Convert to probabilities
        probs = [F.softmax(pred, dim=1) for pred in individual_preds]

        # Ensemble probability
        ensemble_prob = torch.zeros_like(probs[0])
        for prob, weight in zip(probs, self.weights):
            ensemble_prob += weight * prob

        # Confidence metrics
        mean_prob = torch.mean(torch.stack(probs), dim=0)
        std_prob = torch.std(torch.stack(probs), dim=0)

        # Agreement score (1 - variance across models)
        agreement = 1.0 - torch.mean(std_prob, dim=1)

        return {
            'ensemble_prediction': torch.log(ensemble_prob + 1e-8),  # Convert back to logits
            'ensemble_probability': ensemble_prob,
            'individual_probabilities': probs,
            'prediction_std': std_prob,
            'agreement_score': agreement
        }

@ModelRegistry.register('stacked_ensemble')
class StackedEnsemble(BaseModel):
    """Stacked ensemble with a meta-learner that learns to combine predictions"""

    def __init__(self,
                 base_models: List[BaseModel],
                 meta_learner_hidden_dim: int = 128,
                 **kwargs):
        super().__init__(base_models[0].num_classes, **kwargs)

        self.base_models = nn.ModuleList(base_models)
        self.num_base_models = len(base_models)

        # Meta-learner that takes concatenated predictions
        input_dim = self.num_classes * self.num_base_models
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, meta_learner_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_learner_hidden_dim, meta_learner_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_learner_hidden_dim // 2, self.num_classes)
        )

        # Freeze base models during meta-learner training
        self.freeze_base_models()

    def freeze_base_models(self):
        """Freeze all base model parameters"""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
        print("🔒 Base models frozen for meta-learner training")

    def unfreeze_base_models(self):
        """Unfreeze base model parameters for fine-tuning"""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = True
        print("🔓 Base models unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked ensemble"""
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            pred = model(x)
            base_predictions.append(pred)

        # Concatenate predictions
        stacked_predictions = torch.cat(base_predictions, dim=1)

        # Pass through meta-learner
        final_prediction = self.meta_learner(stacked_predictions)

        return final_prediction

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the first base model"""
        return self.base_models[0].get_features(x)

    def get_base_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from all base models"""
        predictions = []
        for model in self.base_models:
            pred = model(x)
            predictions.append(pred)
        return predictions

@ModelRegistry.register('adaptive_ensemble')
class AdaptiveEnsemble(BaseModel):
    """Adaptive ensemble that learns instance-specific weights for combining models"""

    def __init__(self,
                 base_models: List[BaseModel],
                 gating_network_dim: int = 256,
                 **kwargs):
        super().__init__(base_models[0].num_classes, **kwargs)

        self.base_models = nn.ModuleList(base_models)
        self.num_base_models = len(base_models)

        # Extract feature dimensions from base models
        self.feature_dims = []
        for model in base_models:
            # Assume all models have the same feature dimension
            # This is a simplification - in practice, you'd need to handle different dims
            if hasattr(model, 'feature_dim'):
                self.feature_dims.append(model.feature_dim)
            else:
                # Default feature dimension for ResNet-50
                self.feature_dims.append(2048)

        # Use the first model's feature dimension for gating network
        feature_dim = self.feature_dims[0]

        # Gating network that decides weights for each model
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim, gating_network_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gating_network_dim, gating_network_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gating_network_dim // 2, self.num_base_models),
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive ensemble"""
        # Get features from the first model for gating
        features = self.base_models[0].get_features(x)

        # Flatten features for gating network
        if features.dim() > 2:
            features = torch.flatten(features, 1)

        # Compute instance-specific weights
        weights = self.gating_network(features)  # Shape: (batch_size, num_models)

        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            pred = model(x)
            base_predictions.append(pred)

        # Stack predictions
        stacked_predictions = torch.stack(base_predictions, dim=2)  # Shape: (batch_size, num_classes, num_models)

        # Apply instance-specific weights
        weights = weights.unsqueeze(1)  # Shape: (batch_size, 1, num_models)
        weighted_predictions = torch.sum(stacked_predictions * weights, dim=2)

        return weighted_predictions

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the first base model"""
        return self.base_models[0].get_features(x)

    def get_model_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get the instance-specific weights for each model"""
        features = self.base_models[0].get_features(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return self.gating_network(features)

class EnsembleBuilder:
    """Builder class for creating different types of ensembles"""

    @staticmethod
    def create_diverse_ensemble(num_classes: int = 32,
                              architectures: List[str] = None,
                              ensemble_type: str = 'simple') -> BaseModel:
        """Create an ensemble with diverse architectures"""

        if architectures is None:
            architectures = ['resnet50', 'efficientnet_b0', 'mobilenet_v2']

        # Create base models
        base_models = []
        for arch in architectures:
            try:
                model = ModelRegistry.get_model(arch, num_classes=num_classes, pretrained=True)
                base_models.append(model)
                print(f"✅ Added {arch} to ensemble")
            except Exception as e:
                print(f"⚠️  Could not add {arch} to ensemble: {e}")

        if not base_models:
            raise ValueError("No valid models could be created for ensemble")

        # Create ensemble based on type
        if ensemble_type == 'simple':
            return SimpleEnsemble(base_models)
        elif ensemble_type == 'stacked':
            return StackedEnsemble(base_models)
        elif ensemble_type == 'adaptive':
            return AdaptiveEnsemble(base_models)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    @staticmethod
    def create_same_architecture_ensemble(architecture: str,
                                        num_models: int = 3,
                                        num_classes: int = 32,
                                        ensemble_type: str = 'simple') -> BaseModel:
        """Create an ensemble with the same architecture but different initializations"""

        base_models = []
        for i in range(num_models):
            model = ModelRegistry.get_model(architecture, num_classes=num_classes, pretrained=True)
            base_models.append(model)
            print(f"✅ Added {architecture} model {i+1} to ensemble")

        if ensemble_type == 'simple':
            return SimpleEnsemble(base_models)
        elif ensemble_type == 'stacked':
            return StackedEnsemble(base_models)
        elif ensemble_type == 'adaptive':
            return AdaptiveEnsemble(base_models)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

def create_ensemble_from_checkpoints(checkpoint_paths: List[str],
                                   model_configs: List[Dict[str, Any]],
                                   ensemble_type: str = 'simple') -> BaseModel:
    """Create ensemble from pre-trained model checkpoints"""

    base_models = []

    for checkpoint_path, config in zip(checkpoint_paths, model_configs):
        # Create model
        model = ModelRegistry.get_model(config['architecture'], **config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        base_models.append(model)
        print(f"✅ Loaded model from {checkpoint_path}")

    # Create ensemble
    if ensemble_type == 'simple':
        return SimpleEnsemble(base_models)
    elif ensemble_type == 'stacked':
        return StackedEnsemble(base_models)
    elif ensemble_type == 'adaptive':
        return AdaptiveEnsemble(base_models)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
