"""
Configuration Management System for Sign Language Recognition Project
Centralized configuration handling with validation and environment support
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import torch


class ModelType(Enum):
    """Supported model architectures"""
    CNN_BASIC = "cnn_basic"
    CNN_ADVANCED = "cnn_advanced"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNET_B0 = "efficientnet_b0"
    VISION_TRANSFORMER = "vision_transformer"


class AugmentationStrength(Enum):
    """Augmentation strength levels"""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class OptimizerType(Enum):
    """Supported optimizers"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Learning rate schedulers"""
    STEP_LR = "step_lr"
    COSINE_ANNEALING = "cosine_annealing"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    EXPONENTIAL = "exponential"


@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = "data/processed"
    raw_data_dir: str = "data/raw"
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    input_size: tuple = (64, 64)
    num_classes: int = 32
    image_channels: int = 1  # Grayscale
    
    # Cross-validation settings
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Data quality settings
    min_samples_per_class: int = 100
    max_class_imbalance_ratio: float = 10.0
    
    def __post_init__(self):
        """Validate data configuration after initialization"""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")


@dataclass
class AugmentationConfig:
    """Augmentation configuration"""
    use_augmentation: bool = True
    strength: AugmentationStrength = AugmentationStrength.MEDIUM
    preserve_hands: bool = True
    
    # Geometric augmentations
    rotation_range: float = 20.0
    zoom_range: float = 0.2
    shift_range: float = 0.1
    shear_range: float = 5.0
    
    # Photometric augmentations
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    noise_probability: float = 0.2
    blur_probability: float = 0.2
    
    # Advanced augmentations
    elastic_transform: bool = True
    grid_distortion: bool = False
    
    # Test-time augmentation
    use_tta: bool = False
    tta_samples: int = 5


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: ModelType = ModelType.CNN_ADVANCED
    pretrained: bool = False
    dropout_rate: float = 0.5
    
    # CNN-specific parameters
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    
    # Dense layer parameters
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    
    # Batch normalization and activation
    use_batch_norm: bool = True
    activation: str = "relu"
    
    # Advanced features
    use_attention: bool = False
    use_residual_connections: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: OptimizerType = OptimizerType.ADAM
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9     # For Adam
    beta2: float = 0.999   # For Adam
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    step_size: int = 30    # For StepLR
    gamma: float = 0.1     # For StepLR and ExponentialLR
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Gradient clipping
    use_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Loss function
    loss_function: str = "cross_entropy"
    label_smoothing: float = 0.1
    
    # Metrics to track
    track_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])


@dataclass
class ValidationConfig:
    """Validation and evaluation configuration"""
    validation_frequency: int = 1  # Validate every N epochs
    save_best_model: bool = True
    save_checkpoint_frequency: int = 10
    
    # Evaluation metrics
    primary_metric: str = "accuracy"
    compute_confusion_matrix: bool = True
    compute_per_class_metrics: bool = True
    
    # Model saving
    save_top_k_models: int = 3
    monitor_metric: str = "val_accuracy"
    monitor_mode: str = "max"  # "max" or "min"


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    random_seed: int = 42
    deterministic: bool = True
    benchmark: bool = False  # cudnn.benchmark
    
    # Logging and output
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Experiment tracking
    experiment_name: str = "sign_language_recognition"
    project_name: str = "arabic_sign_language"
    use_wandb: bool = False
    wandb_project: str = "sign-language-recognition"
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class ProjectConfig:
    """Complete project configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_value(value):
            if hasattr(value, 'value'):  # Handle enums
                return value.value
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            else:
                return value
        
        config_dict = {}
        for field_name, field_value in asdict(self).items():
            config_dict[field_name] = convert_value(field_value)
        return config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'data.batch_size')"""
        try:
            keys = key.split('.')
            value = self
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration as dictionary (for backward compatibility)"""
        return self.to_dict()
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file (JSON or YAML)"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ProjectConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectConfig':
        """Create configuration from dictionary"""
        
        def convert_enum_values(data_dict, config_class):
            """Convert string enum values back to enum objects"""
            converted = {}
            for key, value in data_dict.items():
                # Get the field type annotation if available
                if hasattr(config_class, '__dataclass_fields__'):
                    field_info = config_class.__dataclass_fields__.get(key)
                    if field_info and hasattr(field_info.type, '__name__'):
                        field_type = field_info.type
                        # Check if it's an enum
                        if hasattr(field_type, '__bases__') and any(base.__name__ == 'Enum' for base in field_type.__bases__):
                            try:
                                value = field_type(value)
                            except (ValueError, TypeError):
                                pass  # Keep original value if conversion fails
                converted[key] = value
            return converted
        
        # Create individual config objects with enum conversion
        data_dict = config_dict.get('data', {})
        data_config = DataConfig(**data_dict)
        
        aug_dict = convert_enum_values(config_dict.get('augmentation', {}), AugmentationConfig)
        augmentation_config = AugmentationConfig(**aug_dict)
        
        model_dict = convert_enum_values(config_dict.get('model', {}), ModelConfig)
        model_config = ModelConfig(**model_dict)
        
        train_dict = convert_enum_values(config_dict.get('training', {}), TrainingConfig)
        training_config = TrainingConfig(**train_dict)
        
        validation_config = ValidationConfig(**config_dict.get('validation', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        return cls(
            data=data_config,
            augmentation=augmentation_config,
            model=model_config,
            training=training_config,
            validation=validation_config,
            system=system_config
        )
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for section, values in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def validate(self) -> List[str]:
        """Validate the complete configuration"""
        errors = []
        
        # Validate data splits
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            errors.append(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate positive values
        if self.data.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.training.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        # Validate scheduler configuration
        if self.training.use_scheduler:
            if self.training.scheduler_patience <= 0:
                errors.append("Scheduler patience must be positive")
        
        # Validate early stopping configuration
        if self.training.use_early_stopping:
            if self.training.early_stopping_patience <= 0:
                errors.append("Early stopping patience must be positive")
        
        return errors


class ConfigManager:
    """Configuration manager with environment support"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config = None
    
    def create_default_config(self) -> ProjectConfig:
        """Create default configuration"""
        return ProjectConfig()
    
    def load_config(self, 
                   config_name: str = "default",
                   environment: str = "development") -> ProjectConfig:
        """
        Load configuration with environment support
        
        Args:
            config_name: Name of the configuration file
            environment: Environment (development, production, testing)
        """
        # Try to load environment-specific config first
        env_config_path = self.config_dir / f"{config_name}_{environment}.json"
        if env_config_path.exists():
            config = ProjectConfig.load_from_file(env_config_path)
        else:
            # Fall back to base config
            base_config_path = self.config_dir / f"{config_name}.json"
            if base_config_path.exists():
                config = ProjectConfig.load_from_file(base_config_path)
            else:
                # Create default config
                config = self.create_default_config()
                # Save as JSON to avoid YAML enum issues
                json_path = self.config_dir / f"{config_name}.json"
                config.save_to_file(json_path)
        
        # Apply environment variables overrides
        config = self._apply_env_overrides(config)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        self._config = config
        return config
    
    def save_config(self, config: ProjectConfig, config_name: str = "default") -> None:
        """Save configuration to file"""
        config_path = self.config_dir / f"{config_name}.json"
        config.save_to_file(config_path)
    
    def _apply_env_overrides(self, config: ProjectConfig) -> ProjectConfig:
        """Apply environment variable overrides"""
        # System configuration overrides
        if "DEVICE" in os.environ:
            config.system.device = os.environ["DEVICE"]
        
        if "RANDOM_SEED" in os.environ:
            config.system.random_seed = int(os.environ["RANDOM_SEED"])
        
        # Training configuration overrides
        if "LEARNING_RATE" in os.environ:
            config.training.learning_rate = float(os.environ["LEARNING_RATE"])
        
        if "BATCH_SIZE" in os.environ:
            config.data.batch_size = int(os.environ["BATCH_SIZE"])
        
        if "NUM_EPOCHS" in os.environ:
            config.training.num_epochs = int(os.environ["NUM_EPOCHS"])
        
        # Data configuration overrides
        if "DATA_DIR" in os.environ:
            config.data.data_dir = os.environ["DATA_DIR"]
        
        return config
    
    def get_current_config(self) -> Optional[ProjectConfig]:
        """Get current loaded configuration"""
        return self._config
    
    def create_experiment_configs(self) -> Dict[str, ProjectConfig]:
        """Create configurations for different experiments"""
        configs = {}
        
        # Baseline configuration
        baseline = self.create_default_config()
        configs["baseline"] = baseline
        
        # Light augmentation experiment
        light_aug = ProjectConfig()
        light_aug.augmentation.strength = AugmentationStrength.LIGHT
        light_aug.system.experiment_name = "light_augmentation"
        configs["light_augmentation"] = light_aug
        
        # Heavy augmentation experiment
        heavy_aug = ProjectConfig()
        heavy_aug.augmentation.strength = AugmentationStrength.HEAVY
        heavy_aug.system.experiment_name = "heavy_augmentation"
        configs["heavy_augmentation"] = heavy_aug
        
        # Different model architectures
        for model_type in [ModelType.CNN_BASIC, ModelType.RESNET18, ModelType.MOBILENET_V2]:
            model_config = ProjectConfig()
            model_config.model.model_type = model_type
            model_config.system.experiment_name = f"model_{model_type.value}"
            configs[f"model_{model_type.value}"] = model_config
        
        return configs
    
    def save_experiment_configs(self):
        """Save all experiment configurations"""
        configs = self.create_experiment_configs()
        
        for name, config in configs.items():
            self.save_config(config, name)
            print(f"Saved configuration: {name}")


def get_device_config() -> str:
    """Automatically detect the best device configuration"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


def create_quick_config(
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    model_type: str = "cnn_advanced",
    data_dir: str = "data/processed"
) -> ProjectConfig:
    """Create a quick configuration with common parameters"""
    config = ProjectConfig()
    
    # Update with provided parameters
    config.data.batch_size = batch_size
    config.data.data_dir = data_dir
    config.training.learning_rate = learning_rate
    config.training.num_epochs = num_epochs
    config.model.model_type = ModelType(model_type)
    config.system.device = get_device_config()
    
    return config


def get_config(config_name: str = "default", 
               config_dir: str = "configs",
               environment: str = "development") -> ProjectConfig:
    """
    Get project configuration (convenience function)
    
    Args:
        config_name: Name of the configuration
        config_dir: Configuration directory
        environment: Environment name
        
    Returns:
        Project configuration
    """
    config_manager = ConfigManager(config_dir)
    
    try:
        config = config_manager.load_config(config_name, environment)
    except (FileNotFoundError, ValueError):
        # Create default config if none exists
        print(f"Creating default configuration...")
        config = config_manager.create_default_config()
        config_manager.save_config(config, config_name)
    
    return config


if __name__ == "__main__":
    # Test configuration system
    print("Testing Configuration Management System...")
    
    # Create config manager
    config_manager = ConfigManager("test_configs")
    
    # Create and test default configuration
    default_config = config_manager.create_default_config()
    print(f"✓ Default configuration created")
    
    # Validate configuration
    errors = default_config.validate()
    if errors:
        print(f"✗ Configuration validation failed: {errors}")
    else:
        print(f"✓ Configuration validation passed")
    
    # Test saving and loading
    config_manager.save_config(default_config, "test")
    loaded_config = config_manager.load_config("test")
    print(f"✓ Configuration save/load test passed")
    
    # Test quick configuration
    quick_config = create_quick_config(
        batch_size=64,
        learning_rate=0.01,
        model_type="resnet18"
    )
    print(f"✓ Quick configuration created: {quick_config.model.model_type}")
    
    # Create experiment configurations
    config_manager.save_experiment_configs()
    print(f"✓ Experiment configurations saved")
    
    # Test environment variable overrides
    os.environ["BATCH_SIZE"] = "16"
    os.environ["LEARNING_RATE"] = "0.005"
    
    config_with_overrides = config_manager.load_config("test")
    print(f"✓ Environment overrides applied:")
    print(f"  - Batch size: {config_with_overrides.data.batch_size}")
    print(f"  - Learning rate: {config_with_overrides.training.learning_rate}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_configs", ignore_errors=True)
    
    print("\n✓ Configuration management system tested successfully!")