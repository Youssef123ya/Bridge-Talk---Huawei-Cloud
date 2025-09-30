"""
ModelArts Training Job Configuration
Arabic Sign Language Recognition - Phase 3
"""

import json
import os
from pathlib import Path

# Training configuration for ModelArts
TRAINING_CONFIG = {
    "job_name": "arsl-recognition-training-v1",
    "description": "Arabic Sign Language Recognition CNN Training",
    
    # Algorithm configuration
    "algorithm_config": {
        "code_dir": "obs://arsl-youssef-af-cairo-2025/code/",
        "boot_file": "train_arsl.py",
        "engine_id": 122,  # PyTorch 1.8.0-python3.7
        "python_version": "python3.7"
    },
    
    # Data configuration  
    "data_config": {
        "train_url": "obs://arsl-youssef-af-cairo-2025/datasets/raw/",
        "data_url": "obs://arsl-youssef-af-cairo-2025/datasets/processed/",
        "output_url": "obs://arsl-youssef-af-cairo-2025/output/",
        "log_url": "obs://arsl-youssef-af-cairo-2025/logs/"
    },
    
    # Resource configuration
    "resource_config": {
        "flavor": "modelarts.vm.gpu.v100",  # V100 GPU instance
        "node_count": 1,
        "parameter_server_count": 0,  # No parameter server needed
        "worker_count": 1
    },
    
    # Hyperparameters
    "hyperparameters": {
        "batch_size": "64",
        "learning_rate": "0.001", 
        "epochs": "100",
        "optimizer": "adam",
        "patience": "15",
        "min_delta": "0.001",
        "validation_split": "0.2",
        "random_seed": "42",
        "num_classes": "32",
        "image_size": "64",
        "channels": "1"  # Grayscale
    },
    
    # Training limits
    "max_running_time": 28800,  # 8 hours in seconds
    "auto_terminate_time": 86400  # 24 hours auto-terminate
}

def create_training_job_config():
    """Create ModelArts training job configuration file"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "modelarts_training_config.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(TRAINING_CONFIG, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training configuration saved to: {config_file}")
    return config_file

def print_training_summary():
    """Print training configuration summary"""
    print("\n🏋️ ModelArts Training Configuration")
    print("=" * 50)
    print(f"📝 Job Name: {TRAINING_CONFIG['job_name']}")
    print(f"🚀 Engine: PyTorch 1.8.0 Python 3.7")
    print(f"💾 GPU: {TRAINING_CONFIG['resource_config']['flavor']}")
    print(f"⏱️  Max Duration: {TRAINING_CONFIG['max_running_time']//3600} hours")
    print(f"📊 Batch Size: {TRAINING_CONFIG['hyperparameters']['batch_size']}")
    print(f"🔄 Epochs: {TRAINING_CONFIG['hyperparameters']['epochs']}")
    print(f"📈 Learning Rate: {TRAINING_CONFIG['hyperparameters']['learning_rate']}")
    print(f"🎯 Classes: {TRAINING_CONFIG['hyperparameters']['num_classes']}")
    
    print("\n📁 Data Paths:")
    print(f"   📤 Input: {TRAINING_CONFIG['data_config']['train_url']}")
    print(f"   📥 Output: {TRAINING_CONFIG['data_config']['output_url']}")
    print(f"   📋 Logs: {TRAINING_CONFIG['data_config']['log_url']}")

if __name__ == "__main__":
    print("🚀 Preparing ModelArts Training Configuration")
    
    # Create configuration
    config_file = create_training_job_config()
    
    # Print summary
    print_training_summary()
    
    print("\n✅ Phase 3 Configuration Complete!")
    print("📋 Ready to deploy training job to ModelArts")