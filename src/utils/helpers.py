"""
Utility Helper Functions for Sign Language Recognition Project
Provides common utilities for logging, device management, file operations, etc.
"""

import os
import sys
import json
import pickle
import logging
import shutil
import hashlib
import datetime
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


class ProjectLogger:
    """Enhanced logging system for the project"""
    
    def __init__(self, 
                 name: str = "SignLanguageProject",
                 log_dir: str = "logs",
                 level: str = "INFO",
                 file_logging: bool = True,
                 console_logging: bool = True):
        """
        Initialize project logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            file_logging: Enable file logging
            console_logging: Enable console logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper())
        
        # Create log directory
        ensure_directory_exists(self.log_dir)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        if file_logging:
            log_file = self.log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger
    
    def log_system_info(self):
        """Log system information"""
        self.logger.info("=== System Information ===")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Python Version: {sys.version}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        self.logger.info("=" * 30)
    
    def log_data_info(self, data_info: Dict):
        """Log dataset information"""
        self.logger.info("=== Dataset Information ===")
        for key, value in data_info.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 30)
    
    def log_model_info(self, model, model_name: str = "Model"):
        """Log model architecture information"""
        self.logger.info(f"=== {model_name} Information ===")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        # Log model structure
        self.logger.info("Model Architecture:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                self.logger.info(f"  {name}: {module}")
        
        self.logger.info("=" * 30)


class DeviceManager:
    """Manage device selection and optimization"""
    
    def __init__(self, prefer_gpu: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize device manager
        
        Args:
            prefer_gpu: Whether to prefer GPU if available
            logger: Optional logger instance
        """
        self.prefer_gpu = prefer_gpu
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._select_device()
        
    def _select_device(self) -> torch.device:
        """Select the best available device"""
        if self.prefer_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        return device
    
    def get_device(self) -> torch.device:
        """Get the selected device"""
        return self.device
    
    def move_to_device(self, obj: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or model to the selected device"""
        return obj.to(self.device)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get device memory information"""
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "cached": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            return {"device": "cpu", "memory_info": "unavailable"}
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_operation(func):
    """Decorator for safe file operations with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except PermissionError as e:
            logging.error(f"Permission error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in file operation: {e}")
            raise
    return wrapper


@safe_file_operation
def save_json(data: Dict, file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file safely
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    file_path = Path(file_path)
    ensure_directory_exists(file_path.parent)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


@safe_file_operation
def load_json(file_path: Union[str, Path]) -> Dict:
    """
    Load data from JSON file safely
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


@safe_file_operation
def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Save object to pickle file safely
    
    Args:
        obj: Object to save
        file_path: Path to save the file
    """
    file_path = Path(file_path)
    ensure_directory_exists(file_path.parent)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


@safe_file_operation
def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load object from pickle file safely
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of the file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "File does not exist"}
    
    stat = file_path.stat()
    
    return {
        "name": file_path.name,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "created": datetime.datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.datetime.fromtimestamp(stat.st_mtime),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "suffix": file_path.suffix,
        "parent": str(file_path.parent),
        "absolute_path": str(file_path.absolute())
    }


def copy_with_progress(src: Union[str, Path], dst: Union[str, Path], 
                      logger: Optional[logging.Logger] = None) -> None:
    """
    Copy file or directory with progress logging
    
    Args:
        src: Source path
        dst: Destination path
        logger: Optional logger for progress updates
    """
    src, dst = Path(src), Path(dst)
    logger = logger or logging.getLogger(__name__)
    
    if src.is_file():
        ensure_directory_exists(dst.parent)
        shutil.copy2(src, dst)
        logger.info(f"Copied file: {src} -> {dst}")
    elif src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info(f"Copied directory: {src} -> {dst}")
    else:
        raise ValueError(f"Source path does not exist: {src}")


class ProgressTracker:
    """Track and display progress for long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing", 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items to process
            description: Description of the operation
            logger: Optional logger instance
        """
        self.total = total
        self.description = description
        self.logger = logger or logging.getLogger(__name__)
        self.current = 0
        self.start_time = datetime.datetime.now()
        
    def update(self, increment: int = 1, message: str = "") -> None:
        """
        Update progress
        
        Args:
            increment: Number of items processed
            message: Optional message to log
        """
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        elapsed = datetime.datetime.now() - self.start_time
        if self.current > 0:
            estimated_total = elapsed * (self.total / self.current)
            remaining = estimated_total - elapsed
        else:
            remaining = datetime.timedelta(0)
        
        log_message = f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) "
        log_message += f"[Elapsed: {elapsed}, Remaining: {remaining}]"
        
        if message:
            log_message += f" - {message}"
        
        self.logger.info(log_message)
    
    def finish(self, message: str = "Completed"):
        """Mark progress as finished"""
        elapsed = datetime.datetime.now() - self.start_time
        self.logger.info(f"{self.description} {message} in {elapsed}")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def create_experiment_directory(base_dir: Union[str, Path], 
                              experiment_name: str,
                              timestamp: bool = True) -> Path:
    """
    Create organized experiment directory structure
    
    Args:
        base_dir: Base experiments directory
        experiment_name: Name of the experiment
        timestamp: Whether to add timestamp to directory name
        
    Returns:
        Path to the experiment directory
    """
    base_dir = Path(base_dir)
    
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"{experiment_name}_{timestamp_str}"
    else:
        exp_dir = base_dir / experiment_name
    
    # Create standard subdirectories
    subdirs = ['models', 'logs', 'plots', 'data', 'configs', 'results']
    for subdir in subdirs:
        ensure_directory_exists(exp_dir / subdir)
    
    return exp_dir


class ConfigValidator:
    """Validate configuration dictionaries"""
    
    @staticmethod
    def validate_training_config(config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate training configuration
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_keys = [
            'batch_size', 'learning_rate', 'num_epochs', 
            'input_size', 'num_classes'
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        # Validate value ranges
        if 'batch_size' in config and config['batch_size'] <= 0:
            errors.append("batch_size must be positive")
        
        if 'learning_rate' in config and (config['learning_rate'] <= 0 or config['learning_rate'] > 1):
            errors.append("learning_rate must be between 0 and 1")
        
        if 'num_epochs' in config and config['num_epochs'] <= 0:
            errors.append("num_epochs must be positive")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_data_config(config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data configuration
        
        Args:
            config: Data configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_keys = ['data_dir', 'train_split', 'val_split', 'test_split']
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        # Validate splits sum to 1.0
        if all(key in config for key in ['train_split', 'val_split', 'test_split']):
            total_split = config['train_split'] + config['val_split'] + config['test_split']
            if abs(total_split - 1.0) > 1e-6:
                errors.append(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate data directory exists
        if 'data_dir' in config and not Path(config['data_dir']).exists():
            errors.append(f"Data directory does not exist: {config['data_dir']}")
        
        return len(errors) == 0, errors


def visualize_training_metrics(metrics_dict: Dict[str, List[float]], 
                             save_path: Optional[Union[str, Path]] = None,
                             title: str = "Training Metrics") -> None:
    """
    Visualize training metrics
    
    Args:
        metrics_dict: Dictionary with metric names and values
        save_path: Optional path to save the plot
        title: Plot title
    """
    num_metrics = len(metrics_dict)
    
    if num_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, min(num_metrics, 3), figsize=(15, 5))
    if num_metrics == 1:
        axes = [axes]
    elif num_metrics == 2:
        axes = axes
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 3:  # Limit to 3 subplots
            break
        
        ax = axes[i] if num_metrics > 1 else axes[0]
        ax.plot(values, linewidth=2)
        ax.set_title(f"{metric_name.title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.title())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Setup logging for the project (convenience function)
    
    Args:
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    if log_file:
        log_dir = Path(log_file).parent
        ensure_directory_exists(log_dir)
        
        logger_instance = ProjectLogger(
            name="SignLanguageProject",
            log_dir=str(log_dir),
            level=level,
            file_logging=True,
            console_logging=True
        )
    else:
        logger_instance = ProjectLogger(
            name="SignLanguageProject",
            level=level,
            file_logging=False,
            console_logging=True
        )
    
    return logger_instance.get_logger()


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device (convenience function)
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Selected device
    """
    device_manager = DeviceManager(prefer_gpu=prefer_gpu)
    return device_manager.get_device()


def create_directories(dir_list: List[Union[str, Path]]) -> None:
    """
    Create multiple directories (convenience function)
    
    Args:
        dir_list: List of directory paths to create
    """
    for directory in dir_list:
        ensure_directory_exists(directory)


if __name__ == "__main__":
    # Test utility functions
    print("Testing Sign Language Project Utilities...")
    
    # Test logger
    logger_instance = ProjectLogger("TestLogger", "test_logs")
    logger = logger_instance.get_logger()
    logger.info("Logger test successful")
    
    # Test device manager
    device_manager = DeviceManager(logger=logger)
    device = device_manager.get_device()
    logger.info(f"Selected device: {device}")
    
    # Test directory creation
    test_dir = ensure_directory_exists("test_utils")
    logger.info(f"Test directory created: {test_dir}")
    
    # Test file operations
    test_data = {"test": "data", "number": 42}
    save_json(test_data, test_dir / "test.json")
    loaded_data = load_json(test_dir / "test.json")
    logger.info(f"JSON save/load test: {loaded_data}")
    
    # Test progress tracker
    progress = ProgressTracker(100, "Test Operation", logger)
    for i in range(10):
        progress.update(10, f"Step {i+1}")
    progress.finish("Successfully")
    
    # Test random seeds
    set_random_seeds(42)
    logger.info("Random seeds set for reproducibility")
    
    # Test config validation
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'input_size': (64, 64),
        'num_classes': 32
    }
    is_valid, errors = ConfigValidator.validate_training_config(config)
    logger.info(f"Config validation: {is_valid}, Errors: {errors}")
    
    # Cleanup
    shutil.rmtree("test_utils", ignore_errors=True)
    shutil.rmtree("test_logs", ignore_errors=True)
    
    print("✓ All utility functions tested successfully!")