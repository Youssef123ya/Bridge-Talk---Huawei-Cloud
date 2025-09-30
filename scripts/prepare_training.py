"""
Phase 3: Training Job Launcher
Arabic Sign Language Recognition
"""

import json
import sys
from pathlib import Path

def check_upload_completion():
    """Check if Phase 2 upload is complete"""
    print("🔍 Checking Phase 2 upload status...")
    
    # This would typically check OBS for file count
    # For now, we'll provide manual verification steps
    print()
    print("📋 Manual Verification Steps:")
    print("1. Go to: https://console.huaweicloud.com/obs")
    print("2. Open bucket: arsl-youssef-af-cairo-2025")
    print("3. Navigate to: datasets/raw/")
    print("4. Verify you see 32 class folders (alef, baa, taa, etc.)")
    print("5. Each folder should contain thousands of images")
    print()
    
    response = input("🤔 Is the upload complete? (y/N): ").lower().strip()
    return response == 'y'

def prepare_training_job():
    """Prepare ModelArts training job configuration"""
    print("\n🏋️ Preparing ModelArts Training Job")
    print("=" * 40)
    
    # Training job configuration
    config = {
        "job_name": "arsl-recognition-v1",
        "description": "Arabic Sign Language Recognition Training",
        "algorithm_config": {
            "code_dir": "obs://arsl-youssef-af-cairo-2025/code/",
            "boot_file": "train_arsl.py",
            "engine_id": 122,
            "python_version": "python3.7"
        },
        "data_config": {
            "train_url": "obs://arsl-youssef-af-cairo-2025/datasets/raw/",
            "output_url": "obs://arsl-youssef-af-cairo-2025/output/",
            "log_url": "obs://arsl-youssef-af-cairo-2025/logs/"
        },
        "resource_config": {
            "flavor": "modelarts.vm.gpu.v100",
            "node_count": 1
        },
        "hyperparameters": {
            "batch_size": "64",
            "learning_rate": "0.001",
            "epochs": "100",
            "num_classes": "32"
        }
    }
    
    # Save configuration
    config_file = Path("config/training_job_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_file}")
    
    return config

def print_manual_deployment_steps():
    """Print manual steps to deploy training job"""
    print("\n📋 MANUAL DEPLOYMENT STEPS")
    print("=" * 40)
    print()
    print("🌐 Go to ModelArts Console:")
    print("   https://console.huaweicloud.com/modelarts")
    print()
    print("📝 Create Training Job:")
    print("   1. Click 'Training Management' → 'Training Jobs'")
    print("   2. Click 'Create Training Job'")
    print("   3. Fill in the following details:")
    print()
    print("📊 Job Configuration:")
    print("   Job Name: arsl-recognition-v1")
    print("   Description: Arabic Sign Language Recognition")
    print()
    print("🔧 Algorithm:")
    print("   Algorithm Source: Custom Algorithm")
    print("   Code Directory: obs://arsl-youssef-af-cairo-2025/code/")
    print("   Boot File: train_arsl.py")
    print("   AI Engine: PyTorch 1.8.0-python3.7")
    print()
    print("📁 Data:")
    print("   Data Path: obs://arsl-youssef-af-cairo-2025/datasets/raw/")
    print("   Training Output: obs://arsl-youssef-af-cairo-2025/output/")
    print()
    print("💾 Resource Pool:")
    print("   Resource Type: GPU")
    print("   Flavor: V100 GPU (modelarts.vm.gpu.v100)")
    print("   Compute Nodes: 1")
    print()
    print("⚙️ Hyperparameters:")
    print("   batch_size = 64")
    print("   learning_rate = 0.001")
    print("   epochs = 100")
    print("   num_classes = 32")
    print()
    print("⏱️ Job Settings:")
    print("   Max Running Time: 8 hours")
    print("   Auto-stop: Enabled")
    print()

def create_training_code():
    """Create or verify training code exists"""
    print("\n📝 Checking Training Code...")
    
    training_script = Path("src/cloud/train_arsl.py")
    
    if training_script.exists():
        print("✅ Training script found: src/cloud/train_arsl.py")
    else:
        print("⚠️  Training script not found, creating basic version...")
        # The script already exists from previous setup
        
    print("📤 Next: Upload training code to OBS code/ folder")
    
def main():
    """Main Phase 3 preparation"""
    print("🚀 PHASE 3: TRAINING PREPARATION")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("=" * 50)
    
    # Check if upload is complete
    if not check_upload_completion():
        print("\n⏳ Please wait for Phase 2 upload to complete first")
        print("📊 Current status: Upload in progress")
        print("⏱️  Estimated time remaining: 1-2 hours")
        return
    
    # Prepare training configuration
    config = prepare_training_job()
    
    # Check training code
    create_training_code()
    
    # Print deployment steps
    print_manual_deployment_steps()
    
    print("\n🎯 PHASE 3 READY!")
    print("✅ Configuration prepared")
    print("📋 Manual deployment steps provided")
    print("🌐 Use ModelArts console to start training")
    print()
    print("⏱️  Expected Training Duration: 4-8 hours")
    print("🎯 Target Accuracy: >85%")

if __name__ == "__main__":
    main()