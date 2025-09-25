import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import json
import os
from .base_model import BaseModel, ModelRegistry, create_model_from_config
from .cnn_architectures import *
from .ensemble_models import *

class ModelFactory:
    """Factory class for creating models with advanced configuration options"""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """Create a model from configuration"""
        return create_model_from_config(config)

    @staticmethod
    def create_ensemble(config: Dict[str, Any]) -> BaseModel:
        """Create an ensemble model from configuration"""

        ensemble_type = config.get('ensemble_type', 'simple')
        base_models_config = config.get('base_models', [])

        if not base_models_config:
            raise ValueError("No base models specified for ensemble")

        # Create base models
        base_models = []
        for model_config in base_models_config:
            model = create_model_from_config(model_config)
            base_models.append(model)

        # Create ensemble
        if ensemble_type == 'simple':
            weights = config.get('weights', None)
            return SimpleEnsemble(base_models, weights)
        elif ensemble_type == 'stacked':
            hidden_dim = config.get('meta_learner_hidden_dim', 128)
            return StackedEnsemble(base_models, hidden_dim)
        elif ensemble_type == 'adaptive':
            gating_dim = config.get('gating_network_dim', 256)
            return AdaptiveEnsemble(base_models, gating_dim)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    @staticmethod
    def get_model_recommendations(dataset_size: int,
                                computational_budget: str = 'medium',
                                target_accuracy: float = 0.9) -> List[Dict[str, Any]]:
        """Get model recommendations based on requirements"""

        recommendations = []

        if computational_budget == 'low':
            # Lightweight models
            recommendations.extend([
                {
                    'name': 'MobileNet V2',
                    'config': {'architecture': 'mobilenet_v2', 'num_classes': 32},
                    'pros': ['Fast inference', 'Low memory', 'Mobile-friendly'],
                    'cons': ['Lower accuracy potential'],
                    'expected_accuracy': 0.85
                },
                {
                    'name': 'EfficientNet B0',
                    'config': {'architecture': 'efficientnet_b0', 'num_classes': 32},
                    'pros': ['Efficient', 'Good accuracy/speed tradeoff'],
                    'cons': ['Moderate training time'],
                    'expected_accuracy': 0.88
                }
            ])

        elif computational_budget == 'medium':
            # Balanced models
            recommendations.extend([
                {
                    'name': 'ResNet-50',
                    'config': {'architecture': 'resnet50', 'num_classes': 32},
                    'pros': ['Proven architecture', 'Good balance', 'Transfer learning'],
                    'cons': ['Moderate size'],
                    'expected_accuracy': 0.90
                },
                {
                    'name': 'EfficientNet B2',
                    'config': {'architecture': 'efficientnet_b2', 'num_classes': 32},
                    'pros': ['High efficiency', 'Good accuracy'],
                    'cons': ['Longer training time'],
                    'expected_accuracy': 0.91
                }
            ])

        elif computational_budget == 'high':
            # High-performance models
            recommendations.extend([
                {
                    'name': 'ResNet-101',
                    'config': {'architecture': 'resnet101', 'num_classes': 32},
                    'pros': ['High capacity', 'Excellent accuracy'],
                    'cons': ['Large model', 'Slow inference'],
                    'expected_accuracy': 0.92
                },
                {
                    'name': 'Vision Transformer',
                    'config': {'architecture': 'vision_transformer', 'num_classes': 32},
                    'pros': ['State-of-the-art', 'Attention mechanism'],
                    'cons': ['Requires large dataset', 'Computational intensive'],
                    'expected_accuracy': 0.93
                },
                {
                    'name': 'Ensemble (ResNet + EfficientNet)',
                    'config': {
                        'ensemble_type': 'simple',
                        'base_models': [
                            {'architecture': 'resnet50', 'num_classes': 32},
                            {'architecture': 'efficientnet_b2', 'num_classes': 32}
                        ]
                    },
                    'pros': ['Highest accuracy', 'Robust predictions'],
                    'cons': ['Slowest inference', 'Most memory'],
                    'expected_accuracy': 0.94
                }
            ])

        # Filter by expected accuracy
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec['expected_accuracy'] >= target_accuracy
        ]

        return filtered_recommendations if filtered_recommendations else recommendations

    @staticmethod
    def benchmark_models(model_configs: List[Dict[str, Any]],
                        input_size: tuple = (1, 3, 224, 224),
                        device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
        """Benchmark multiple model configurations"""

        results = {}

        for i, config in enumerate(model_configs):
            model_name = config.get('architecture', f'model_{i}')

            try:
                # Create model
                model = create_model_from_config(config)

                # Get model info
                model_info = model.get_model_info()

                # Profile model
                profile = model.profile_model(input_size, device)

                results[model_name] = {
                    'parameters': model_info['total_parameters'],
                    'size_mb': model_info['model_size_mb'],
                    'inference_time_ms': profile['avg_inference_time_ms'],
                    'fps': profile['fps'],
                    'flops': profile.get('flops', 0),
                    'config': config
                }

                print(f"✅ Benchmarked {model_name}")

            except Exception as e:
                results[model_name] = {'error': str(e)}
                print(f"❌ Failed to benchmark {model_name}: {e}")

        return results

    @staticmethod
    def create_progressive_training_models(base_config: Dict[str, Any]) -> List[BaseModel]:
        """Create models for progressive training (start small, grow larger)"""

        models = []

        # Stage 1: Lightweight model
        stage1_config = base_config.copy()
        stage1_config['architecture'] = 'mobilenet_v2'
        models.append(create_model_from_config(stage1_config))

        # Stage 2: Medium model
        stage2_config = base_config.copy()
        stage2_config['architecture'] = 'resnet50'
        models.append(create_model_from_config(stage2_config))

        # Stage 3: Large model
        stage3_config = base_config.copy()
        stage3_config['architecture'] = 'resnet101'
        models.append(create_model_from_config(stage3_config))

        return models

    @staticmethod
    def save_model_config(model: BaseModel, filepath: str) -> None:
        """Save model configuration for reproducibility"""

        config = {
            'model_class': model.__class__.__name__,
            'model_config': model.model_config,
            'num_classes': model.num_classes,
            'architecture': getattr(model, 'architecture', 'unknown'),
            'total_parameters': model.get_num_parameters(),
            'model_info': model.get_model_info()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"💾 Model configuration saved to {filepath}")

    @staticmethod
    def load_model_from_config_file(filepath: str) -> BaseModel:
        """Load model from saved configuration file"""

        with open(filepath, 'r') as f:
            config = json.load(f)

        model_config = config['model_config']
        model_config['num_classes'] = config['num_classes']

        return create_model_from_config(model_config)

class ModelSelector:
    """Intelligent model selector based on dataset and requirements"""

    def __init__(self, dataset_size: int, num_classes: int = 32):
        self.dataset_size = dataset_size
        self.num_classes = num_classes

    def select_optimal_model(self,
                           accuracy_priority: float = 0.5,
                           speed_priority: float = 0.3,
                           memory_priority: float = 0.2,
                           available_gpus: int = 1) -> Dict[str, Any]:
        """Select optimal model based on priorities"""

        # Get all available models
        model_scores = {}

        for model_name in ModelRegistry.list_models():
            try:
                # Create temporary model for analysis
                temp_model = ModelRegistry.get_model(model_name, num_classes=self.num_classes)

                # Calculate scores
                accuracy_score = self._estimate_accuracy_score(model_name, temp_model)
                speed_score = self._estimate_speed_score(temp_model)
                memory_score = self._estimate_memory_score(temp_model)

                # Weighted combination
                total_score = (
                    accuracy_priority * accuracy_score +
                    speed_priority * speed_score +
                    memory_priority * memory_score
                )

                model_scores[model_name] = {
                    'total_score': total_score,
                    'accuracy_score': accuracy_score,
                    'speed_score': speed_score,
                    'memory_score': memory_score,
                    'parameters': temp_model.get_num_parameters()
                }

                del temp_model  # Clean up

            except Exception as e:
                print(f"⚠️  Could not analyze {model_name}: {e}")

        # Select best model
        if model_scores:
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k]['total_score'])

            return {
                'recommended_model': best_model,
                'scores': model_scores[best_model],
                'all_scores': model_scores
            }
        else:
            return {'error': 'No models could be analyzed'}

    def _estimate_accuracy_score(self, model_name: str, model: BaseModel) -> float:
        """Estimate accuracy score based on model characteristics"""

        # Heuristic scoring based on model type and size
        params = model.get_num_parameters()

        if 'resnet' in model_name.lower():
            base_score = 0.8
            if '101' in model_name or '152' in model_name:
                base_score = 0.9
        elif 'efficientnet' in model_name.lower():
            base_score = 0.85
        elif 'vision_transformer' in model_name.lower():
            base_score = 0.9
        elif 'mobilenet' in model_name.lower():
            base_score = 0.75
        elif 'custom' in model_name.lower():
            base_score = 0.8
        else:
            base_score = 0.7

        # Adjust based on parameters (more params often = higher capacity)
        if params > 50_000_000:
            base_score += 0.05
        elif params < 5_000_000:
            base_score -= 0.05

        # Adjust based on dataset size
        if self.dataset_size < 10_000:
            base_score -= 0.1  # Smaller models better for small datasets
        elif self.dataset_size > 100_000:
            base_score += 0.05  # Larger models can utilize more data

        return min(1.0, max(0.0, base_score))

    def _estimate_speed_score(self, model: BaseModel) -> float:
        """Estimate speed score based on model size"""

        params = model.get_num_parameters()

        if params < 5_000_000:
            return 0.9
        elif params < 15_000_000:
            return 0.8
        elif params < 30_000_000:
            return 0.6
        elif params < 50_000_000:
            return 0.4
        else:
            return 0.2

    def _estimate_memory_score(self, model: BaseModel) -> float:
        """Estimate memory score based on model size"""

        size_mb = model.get_num_parameters() * 4 / (1024 * 1024)

        if size_mb < 20:
            return 0.9
        elif size_mb < 50:
            return 0.8
        elif size_mb < 100:
            return 0.6
        elif size_mb < 200:
            return 0.4
        else:
            return 0.2
