"""
Model Factory for Sign Language Recognition
Centralized model creation and configuration management
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from pathlib import Path

from .base_model import BaseSignLanguageModel, ModelRegistry


class ModelFactory:
    """
    Factory class for creating and managing sign language recognition models
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize model factory
        
        Args:
            device: Target device for models ('cpu', 'cuda', 'auto')
        """
        self.device = device or self._get_default_device()
        self.logger = logging.getLogger(__name__)
        self._model_configs = {}
        
    def _get_default_device(self) -> str:
        """Get default device based on availability"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def create_model(self, 
                     model_name: Optional[str] = None,
                     model_config: Optional[Dict[str, Any]] = None,
                     num_classes: int = 32,
                     input_channels: int = 3,
                     pretrained: bool = False,
                     checkpoint_path: Optional[str] = None,
                     **kwargs) -> BaseSignLanguageModel:
        """
        Create a model instance
        
        Args:
            model_name: Name of the model architecture
            model_config: Model configuration dictionary (alternative to model_name)
            num_classes: Number of output classes
            input_channels: Number of input channels
            pretrained: Whether to load pretrained weights
            checkpoint_path: Path to model checkpoint
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model instance
        """
        try:
            # Handle model config dictionary
            if model_config is not None:
                model_name = model_config.get('architecture', model_name)
                num_classes = model_config.get('num_classes', num_classes)
                input_channels = model_config.get('input_channels', input_channels)
                pretrained = model_config.get('pretrained', pretrained)
                kwargs.update(model_config.get('kwargs', {}))
            
            if model_name is None:
                raise ValueError("Either model_name or model_config with 'architecture' key must be provided")
            
            # Create model using registry
            model = ModelRegistry.create_model(
                model_name, 
                num_classes=num_classes,
                input_channels=input_channels,
                **kwargs
            )
            
            # Move to device
            model = model.to(self.device)
            
            # Load checkpoint if provided
            if checkpoint_path:
                self.load_checkpoint(model, checkpoint_path)
            elif pretrained:
                self.logger.warning(f"Pretrained weights not available for {model_name}")
            
            # Store configuration
            self._model_configs[id(model)] = {
                'model_name': model_name,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'device': self.device,
                'kwargs': kwargs
            }
            
            self.logger.info(f"Created {model_name} model with {model.get_model_info()['total_parameters']:,} parameters")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {str(e)}")
            raise
    
    @staticmethod
    def create_ensemble(config: Dict[str, Any]) -> 'ModelEnsemble':
        """
        Create an ensemble of models from configuration
        
        Args:
            config: Ensemble configuration dictionary
            
        Returns:
            ModelEnsemble instance
        """
        factory = ModelFactory()
        
        ensemble_type = config.get('ensemble_type', 'simple')
        base_models_config = config.get('base_models', [])
        
        models = []
        for model_config in base_models_config:
            # Create model from config
            model_name = model_config.get('architecture', 'cnn_basic')
            num_classes = model_config.get('num_classes', 32)
            
            # Map architecture names to our models and ensure 3 input channels
            if model_name in ['mobilenet_v2', 'efficientnet_b0', 'resnet50']:
                model_name = 'cnn_advanced'
            
            model = factory.create_model(model_name, num_classes=num_classes, input_channels=3)
            models.append(model)
        
        method = 'average' if ensemble_type == 'simple' else 'weighted'
        return ModelEnsemble(models, method=method, device=factory.device)

    def create_ensemble_from_list(self,
                       model_configs: List[Dict[str, Any]],
                       ensemble_method: str = 'average') -> 'ModelEnsemble':
        """
        Create an ensemble of models
        
        Args:
            model_configs: List of model configuration dictionaries
            ensemble_method: Method for combining predictions ('average', 'weighted', 'voting')
            
        Returns:
            ModelEnsemble instance
        """
        models = []
        for config in model_configs:
            model = self.create_model(**config)
            models.append(model)
        
        return ModelEnsemble(models, method=ensemble_method, device=self.device)
    
    def load_checkpoint(self, model: BaseSignLanguageModel, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            model: Model instance
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
            # Return metadata
            metadata = {
                'checkpoint_path': str(checkpoint_path),
                'loaded_keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['state_dict']
            }
            
            # Add additional metadata if available
            for key in ['epoch', 'best_accuracy', 'loss', 'optimizer_state_dict']:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
            raise
    
    def save_checkpoint(self,
                       model: BaseSignLanguageModel,
                       checkpoint_path: str,
                       epoch: Optional[int] = None,
                       accuracy: Optional[float] = None,
                       loss: Optional[float] = None,
                       optimizer_state: Optional[Dict] = None,
                       additional_info: Optional[Dict] = None) -> None:
        """
        Save model checkpoint
        
        Args:
            model: Model instance
            checkpoint_path: Path to save checkpoint
            epoch: Training epoch
            accuracy: Model accuracy
            loss: Model loss
            optimizer_state: Optimizer state dict
            additional_info: Additional information to save
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint data
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_info': model.get_model_info(),
                'model_config': self._model_configs.get(id(model), {}),
                'device': self.device
            }
            
            # Add optional information
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if accuracy is not None:
                checkpoint['best_accuracy'] = accuracy
            if loss is not None:
                checkpoint['loss'] = loss
            if optimizer_state is not None:
                checkpoint['optimizer_state_dict'] = optimizer_state
            if additional_info is not None:
                checkpoint.update(additional_info)
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {str(e)}")
            raise
    
    def get_model_summary(self, model: BaseSignLanguageModel, 
                         input_shape: tuple = (1, 1, 64, 64)) -> Dict[str, Any]:
        """
        Get detailed model summary
        
        Args:
            model: Model instance
            input_shape: Input tensor shape for analysis
            
        Returns:
            Model summary dictionary
        """
        from .base_model import test_model_architecture
        
        # Run architecture test
        test_results = test_model_architecture(model, input_shape, self.device)
        
        # Get model configuration
        model_config = self._model_configs.get(id(model), {})
        
        return {
            'architecture_test': test_results,
            'configuration': model_config,
            'device': self.device
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available model architectures
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for model_name in ModelRegistry.list_models():
            try:
                model_info = ModelRegistry.get_model_info(model_name)
                models.append(model_info)
            except Exception as e:
                self.logger.warning(f"Error getting info for {model_name}: {str(e)}")
        
        return models
    
    def benchmark_models(self,
                        model_names: Optional[List[str]] = None,
                        num_classes: int = 32,
                        input_shape: tuple = (1, 1, 64, 64),
                        num_runs: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple model architectures
        
        Args:
            model_names: List of model names to benchmark (None for all)
            num_classes: Number of output classes
            input_shape: Input tensor shape
            num_runs: Number of runs for timing
            
        Returns:
            Benchmark results dictionary
        """
        if model_names is None:
            model_names = ModelRegistry.list_models()
        
        results = {}
        
        for model_name in model_names:
            try:
                self.logger.info(f"Benchmarking {model_name}...")
                
                # Create model
                model = self.create_model(model_name, num_classes=num_classes)
                
                # Run benchmark
                benchmark_results = self._benchmark_single_model(model, input_shape, num_runs)
                results[model_name] = benchmark_results
                
                # Cleanup
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def get_model_recommendations(dataset_size: int, computational_budget: str = 'medium',
                                target_accuracy: float = 0.9) -> List[Dict[str, Any]]:
        """
        Get model recommendations based on requirements
        
        Args:
            dataset_size: Size of the dataset
            computational_budget: 'low', 'medium', 'high'
            target_accuracy: Target accuracy (0-1)
            
        Returns:
            List of recommended model configurations
        """
        recommendations = []
        
        # Simple heuristic-based recommendations
        if computational_budget == 'low':
            recommendations.append({
                'name': 'cnn_basic',
                'expected_accuracy': 0.85,
                'reasoning': 'Lightweight CNN suitable for limited resources',
                'parameters': 'Default configuration'
            })
        elif computational_budget == 'medium':
            recommendations.extend([
                {
                    'name': 'cnn_basic',
                    'expected_accuracy': 0.87,
                    'reasoning': 'Good balance of speed and accuracy',
                    'parameters': 'Default configuration'
                },
                {
                    'name': 'cnn_advanced',
                    'expected_accuracy': 0.92,
                    'reasoning': 'Advanced CNN with attention mechanism',
                    'parameters': 'use_attention=True'
                }
            ])
        else:  # high
            recommendations.extend([
                {
                    'name': 'cnn_advanced',
                    'expected_accuracy': 0.94,
                    'reasoning': 'High-capacity model with attention',
                    'parameters': 'use_attention=True, dropout_rate=0.3'
                }
            ])
        
        # Filter by target accuracy
        recommendations = [r for r in recommendations if r['expected_accuracy'] >= target_accuracy]
        
        return recommendations
    
    @staticmethod
    def benchmark_models(model_configs: List[Dict[str, Any]], 
                        input_size: Tuple[int, ...] = (1, 3, 224, 224),
                        device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple models
        
        Args:
            model_configs: List of model configurations
            input_size: Input tensor shape
            device: Device to run benchmark on
            
        Returns:
            Benchmark results dictionary
        """
        factory = ModelFactory(device=device)
        results = {}
        
        for config in model_configs:
            model_name = config.get('architecture', 'cnn_basic')
            num_classes = config.get('num_classes', 32)
            
            # Map architecture names
            if model_name in ['mobilenet_v2', 'efficientnet_b0', 'resnet50']:
                model_name = 'cnn_advanced'
            
            try:
                model = factory.create_model(model_name, num_classes=num_classes, input_channels=3)
                
                # Run benchmark
                profile_results = model.profile_model(input_size, device, num_runs=10)
                model_info = model.get_model_info()
                
                results[config.get('architecture', model_name)] = {
                    'fps': profile_results['fps'],
                    'parameters': model_info['total_parameters'],
                    'model_size_mb': model_info['model_size_mb'],
                    'inference_time_ms': profile_results['avg_inference_time_ms']
                }
                
                # Cleanup
                del model
                
            except Exception as e:
                results[config.get('architecture', model_name)] = {
                    'error': str(e)
                }
        
        return results
    
    def _benchmark_single_model(self,
                               model: BaseSignLanguageModel,
                               input_shape: tuple,
                               num_runs: int) -> Dict[str, Any]:
        """Benchmark a single model"""
        from .base_model import test_model_architecture
        import time
        
        model.eval()
        
        # Warmup runs
        for _ in range(3):
            test_model_architecture(model, input_shape, self.device)
        
        # Timing runs
        times = []
        for _ in range(num_runs):
            result = test_model_architecture(model, input_shape, self.device)
            if result['success']:
                times.append(result['inference_time_ms'])
        
        if not times:
            return {'error': 'No successful runs'}
        
        # Calculate statistics
        import statistics
        
        return {
            'model_info': model.get_model_info(),
            'avg_inference_time_ms': round(statistics.mean(times), 2),
            'min_inference_time_ms': round(min(times), 2),
            'max_inference_time_ms': round(max(times), 2),
            'std_inference_time_ms': round(statistics.stdev(times) if len(times) > 1 else 0, 2),
            'num_runs': len(times),
            'device': self.device
        }


class ModelSelector:
    """
    Intelligent model selection based on requirements and constraints
    """
    
    def __init__(self, dataset_size: int, num_classes: int = 32, device: str = 'auto'):
        """
        Initialize model selector
        
        Args:
            dataset_size: Size of the training dataset
            num_classes: Number of output classes
            device: Target device for inference
        """
        self.dataset_size = dataset_size
        self.num_classes = num_classes
        self.device = device if device != 'auto' else self._get_default_device()
        self.factory = ModelFactory(device=self.device)
        
        # Cache model performance data
        self._model_cache = {}
        self._benchmark_cache = {}
        
    def _get_default_device(self) -> str:
        """Get default device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _get_model_performance_estimate(self, model_name: str) -> Dict[str, float]:
        """Get performance estimates for a model"""
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        try:
            # Create model and get basic info
            model = self.factory.create_model(model_name, num_classes=self.num_classes)
            model_info = model.get_model_info()
            
            # Run quick performance test
            profile_results = model.profile_model(
                input_size=(1, 3, 224, 224), 
                device=self.device, 
                num_runs=5
            )
            
            # Estimate accuracy based on model complexity and dataset size
            param_count = model_info['total_parameters']
            complexity_score = min(1.0, param_count / 10_000_000)  # Normalize to 10M params
            dataset_score = min(1.0, self.dataset_size / 100_000)  # Normalize to 100K samples
            
            # Heuristic accuracy estimation
            base_accuracy = 0.75  # Base accuracy for simple models
            complexity_bonus = complexity_score * 0.15  # Up to 15% bonus for complexity
            dataset_bonus = dataset_score * 0.10  # Up to 10% bonus for large datasets
            estimated_accuracy = base_accuracy + complexity_bonus + dataset_bonus
            
            performance = {
                'estimated_accuracy': min(0.95, estimated_accuracy),  # Cap at 95%
                'inference_time_ms': profile_results['avg_inference_time_ms'],
                'fps': profile_results['fps'],
                'parameters': param_count,
                'model_size_mb': model_info['model_size_mb'],
                'memory_usage_mb': model_info['model_size_mb'] * 2,  # Estimate runtime memory
            }
            
            self._model_cache[model_name] = performance
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return performance
            
        except Exception as e:
            return {
                'estimated_accuracy': 0.7,
                'inference_time_ms': 100.0,
                'fps': 10.0,
                'parameters': 1_000_000,
                'model_size_mb': 10.0,
                'memory_usage_mb': 20.0,
                'error': str(e)
            }
    
    def _calculate_model_score(self, model_name: str, 
                              accuracy_priority: float = 0.5,
                              speed_priority: float = 0.3,
                              memory_priority: float = 0.2) -> Dict[str, float]:
        """Calculate weighted score for a model"""
        performance = self._get_model_performance_estimate(model_name)
        
        # Normalize scores (higher is better)
        accuracy_score = performance['estimated_accuracy']
        speed_score = min(1.0, performance['fps'] / 100.0)  # Normalize to 100 FPS
        memory_score = max(0.1, 1.0 - (performance['model_size_mb'] / 100.0))  # Penalize large models
        
        # Calculate weighted total score
        total_score = (
            accuracy_score * accuracy_priority +
            speed_score * speed_priority +
            memory_score * memory_priority
        )
        
        return {
            'total_score': total_score,
            'accuracy_score': accuracy_score,
            'speed_score': speed_score,
            'memory_score': memory_score,
            'parameters': performance['parameters'],
            'model_size_mb': performance['model_size_mb'],
            'fps': performance['fps']
        }
    
    def select_optimal_model(self, 
                           accuracy_priority: float = 0.5,
                           speed_priority: float = 0.3,
                           memory_priority: float = 0.2) -> Dict[str, Any]:
        """
        Select optimal model based on priorities
        
        Args:
            accuracy_priority: Weight for accuracy (0-1)
            speed_priority: Weight for inference speed (0-1)
            memory_priority: Weight for memory efficiency (0-1)
            
        Returns:
            Dictionary with selection results
        """
        # Validate priorities
        total_priority = accuracy_priority + speed_priority + memory_priority
        if abs(total_priority - 1.0) > 0.01:
            return {'error': f'Priorities must sum to 1.0, got {total_priority}'}
        
        try:
            available_models = ModelRegistry.list_models()
            model_scores = {}
            
            print(f"🔍 Evaluating {len(available_models)} models...")
            
            for model_name in available_models:
                print(f"   Analyzing {model_name}...")
                scores = self._calculate_model_score(
                    model_name, accuracy_priority, speed_priority, memory_priority
                )
                model_scores[model_name] = scores
            
            # Find best model
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k]['total_score'])
            best_score = model_scores[best_model]
            
            # Sort models by score
            ranked_models = sorted(
                model_scores.items(), 
                key=lambda x: x[1]['total_score'], 
                reverse=True
            )
            
            return {
                'recommended_model': best_model,
                'scores': best_score,
                'all_models': dict(ranked_models),
                'selection_criteria': {
                    'accuracy_priority': accuracy_priority,
                    'speed_priority': speed_priority,
                    'memory_priority': memory_priority
                },
                'dataset_info': {
                    'size': self.dataset_size,
                    'num_classes': self.num_classes,
                    'device': self.device
                }
            }
            
        except Exception as e:
            return {'error': f'Model selection failed: {str(e)}'}
    
    def get_model_recommendations(self, scenarios: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Get recommendations for different scenarios"""
        if scenarios is None:
            scenarios = ['accuracy_focused', 'balanced', 'speed_focused', 'memory_efficient']
        
        scenario_configs = {
            'accuracy_focused': {'accuracy_priority': 0.7, 'speed_priority': 0.2, 'memory_priority': 0.1},
            'balanced': {'accuracy_priority': 0.5, 'speed_priority': 0.3, 'memory_priority': 0.2},
            'speed_focused': {'accuracy_priority': 0.3, 'speed_priority': 0.5, 'memory_priority': 0.2},
            'memory_efficient': {'accuracy_priority': 0.4, 'speed_priority': 0.2, 'memory_priority': 0.4}
        }
        
        recommendations = {}
        for scenario in scenarios:
            if scenario in scenario_configs:
                config = scenario_configs[scenario]
                recommendations[scenario] = self.select_optimal_model(**config)
        
        return recommendations


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models for improved performance
    """
    
    def __init__(self, models: List[BaseSignLanguageModel], 
                 method: str = 'average', device: str = 'cpu'):
        """
        Initialize model ensemble
        
        Args:
            models: List of model instances
            method: Ensemble method ('average', 'weighted', 'voting')
            device: Target device
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.method = method
        self.device = device
        self.num_models = len(models)
        
        # Initialize weights for weighted ensemble
        if method == 'weighted':
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        # Move to device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # Shape: (num_models, batch_size, num_classes)
        
        # Combine predictions based on method
        if self.method == 'average':
            return torch.mean(predictions, dim=0)
        elif self.method == 'weighted':
            weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
            return torch.sum(predictions * weights, dim=0)
        elif self.method == 'voting':
            # Hard voting - return most common prediction
            pred_classes = torch.argmax(predictions, dim=2)  # Shape: (num_models, batch_size)
            ensemble_pred = torch.mode(pred_classes, dim=0)[0]  # Most common class
            
            # Convert back to logits (simplified)
            batch_size, num_classes = predictions.shape[1], predictions.shape[2]
            result = torch.zeros(batch_size, num_classes).to(self.device)
            result.scatter_(1, ensemble_pred.unsqueeze(1), 1.0)
            return result
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble information"""
        model_infos = [model.get_model_info() for model in self.models]
        total_params = sum(info['total_parameters'] for info in model_infos)
        
        return {
            'ensemble_method': self.method,
            'num_models': self.num_models,
            'total_parameters': total_params,
            'individual_models': model_infos,
            'device': self.device
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters in ensemble"""
        return sum(sum(p.numel() for p in model.parameters()) for model in self.models)
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from individual models"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        return predictions
    
    def get_prediction_confidence(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get confidence metrics for predictions"""
        individual_preds = self.get_individual_predictions(x)
        
        # Calculate agreement between models
        pred_classes = [torch.argmax(pred, dim=1) for pred in individual_preds]
        
        # Stack predictions
        stacked_classes = torch.stack(pred_classes, dim=0)  # (num_models, batch_size)
        
        # Calculate agreement rate
        mode_classes = torch.mode(stacked_classes, dim=0)[0]
        agreement_rates = []
        
        for i in range(stacked_classes.shape[1]):  # For each sample
            sample_preds = stacked_classes[:, i]
            agreement = (sample_preds == mode_classes[i]).float().mean().item()
            agreement_rates.append(agreement)
        
        return {
            'agreement_rates': agreement_rates,
            'avg_agreement': sum(agreement_rates) / len(agreement_rates),
            'individual_predictions': individual_preds,
            'consensus_prediction': mode_classes
        }


# Convenience functions
def create_model(model_name: str, **kwargs) -> BaseSignLanguageModel:
    """Convenience function to create a model"""
    factory = ModelFactory()
    return factory.create_model(model_name, **kwargs)


def load_model_from_checkpoint(checkpoint_path: str, 
                              model_name: Optional[str] = None,
                              **kwargs) -> BaseSignLanguageModel:
    """Load model from checkpoint with automatic architecture detection"""
    factory = ModelFactory()
    
    # Load checkpoint to inspect
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get model name from checkpoint
    if model_name is None:
        if 'model_config' in checkpoint and 'model_name' in checkpoint['model_config']:
            model_name = checkpoint['model_config']['model_name']
        else:
            raise ValueError("Model name not found in checkpoint. Please specify model_name parameter.")
    
    # Get configuration from checkpoint if available
    if 'model_config' in checkpoint:
        checkpoint_config = checkpoint['model_config'].get('kwargs', {})
        kwargs = {**checkpoint_config, **kwargs}
    
    # Create and load model
    model = factory.create_model(model_name, checkpoint_path=checkpoint_path, **kwargs)
    
    return model


if __name__ == "__main__":
    # Test model factory
    print("Testing Model Factory...")
    
    factory = ModelFactory()
    
    # List available models
    print(f"Available models: {[info['name'] for info in factory.list_available_models()]}")
    
    # Create and test models
    for model_name in ['cnn_basic', 'cnn_advanced']:
        try:
            model = factory.create_model(model_name, num_classes=32)
            summary = factory.get_model_summary(model)
            print(f"{model_name}: {summary['architecture_test']['total_parameters']:,} parameters")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    print("✓ Model factory testing completed!")