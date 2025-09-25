import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import os
from collections import OrderedDict

class BaseModel(nn.Module, ABC):
    """Base class for all Arabic Sign Language recognition models"""

    def __init__(self, num_classes: int = 32, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_config = kwargs
        self.feature_extractor = None
        self.classifier = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        pass

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification"""
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_feature_extractor(self) -> None:
        """Freeze feature extractor parameters"""
        if self.feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("🔒 Feature extractor frozen")

    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze feature extractor parameters"""
        if self.feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            print("🔓 Feature extractor unfrozen")

    def freeze_classifier(self) -> None:
        """Freeze classifier parameters"""
        if self.classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
            print("🔒 Classifier frozen")

    def unfreeze_classifier(self) -> None:
        """Unfreeze classifier parameters"""
        if self.classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True
            print("🔓 Classifier unfrozen")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'model_name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_trainable_parameters(),
            'model_size_mb': self.get_num_parameters() * 4 / (1024 * 1024),  # Assuming float32
            'config': self.model_config
        }

        # Add layer information
        info['layers'] = []
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                info['layers'].append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': params,
                    'trainable': any(p.requires_grad for p in module.parameters())
                })

        return info

    def save_model_info(self, filepath: str) -> None:
        """Save model information to JSON file"""
        info = self.get_model_info()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        print(f"💾 Model info saved to {filepath}")

    def print_model_summary(self) -> None:
        """Print a detailed model summary"""
        info = self.get_model_info()

        print(f"\n📋 Model Summary: {info['model_name']}")
        print("=" * 50)
        print(f"Classes: {info['num_classes']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")

        # Print layer information
        print(f"\n📊 Layer Details:")
        print("-" * 50)
        total_params = 0
        for layer in info['layers']:
            if layer['parameters'] > 0:
                status = "✅" if layer['trainable'] else "🔒"
                print(f"{status} {layer['name']:<30} {layer['type']:<15} {layer['parameters']:>10,}")
                total_params += layer['parameters']

        print("-" * 50)
        print(f"{'Total':<30} {'':<15} {total_params:>10,}")

    def get_flops(self, input_size: tuple = (1, 3, 224, 224)) -> int:
        """Estimate FLOPs for the model (requires ptflops)"""
        try:
            from ptflops import get_model_complexity_info
            macs, params = get_model_complexity_info(
                self, input_size[1:], as_strings=False,
                print_per_layer_stat=False, verbose=False
            )
            return macs * 2  # MACs to FLOPs
        except ImportError:
            print("⚠️  ptflops not installed. Cannot compute FLOPs.")
            return 0

    def profile_model(self, input_size: tuple = (1, 3, 224, 224), device: str = 'cpu') -> Dict[str, Any]:
        """Profile model performance"""
        import time

        self.eval()
        self.to(device)

        # Create dummy input
        dummy_input = torch.randn(input_size).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self(dummy_input)

        # Time inference
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(100):
                _ = self(dummy_input)

        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / 100

        profile = {
            'device': device,
            'input_size': input_size,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': 1.0 / avg_inference_time,
            'parameters': self.get_num_parameters(),
            'model_size_mb': self.get_num_parameters() * 4 / (1024 * 1024)
        }

        # Add FLOPs if available
        flops = self.get_flops(input_size)
        if flops > 0:
            profile['flops'] = flops
            profile['flops_per_second'] = flops / avg_inference_time

        return profile

class ModelRegistry:
    """Registry for managing different model architectures"""

    _models = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model"""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str, **kwargs) -> BaseModel:
        """Get a model by name"""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(cls._models.keys())}")

        return cls._models[name](**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models"""
        return list(cls._models.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found")

        model_class = cls._models[name]

        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'doc': model_class.__doc__ or "No documentation available"
        }

def create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """Factory function to create model from configuration"""

    model_name = config.get('architecture', 'resnet50')
    model_kwargs = {
        'num_classes': config.get('num_classes', 32),
        'pretrained': config.get('pretrained', True),
        'dropout_rate': config.get('dropout_rate', 0.3)
    }

    # Add architecture-specific parameters
    if 'backbone_config' in config:
        model_kwargs.update(config['backbone_config'])

    try:
        model = ModelRegistry.get_model(model_name, **model_kwargs)
        print(f"✅ Created {model_name} model with {model.get_num_parameters():,} parameters")
        return model
    except ValueError as e:
        print(f"❌ Error creating model: {e}")
        print(f"Available models: {ModelRegistry.list_models()}")
        raise
