"""
Create ModelArts Training Job
Step 2: Deploy training job with uploaded code
"""

import json
import sys
import os
from pathlib import Path

def print_modelarts_job_creation():
    """Print detailed instructions for creating ModelArts job"""
    
    print("🏗️ STEP 2: CREATE MODELARTS TRAINING JOB")
    print("=" * 50)
    print()
    
    print("🌐 1. GO TO MODELARTS CONSOLE:")
    print("   URL: https://console.huaweicloud.com/modelarts")
    print("   Login with: yyacoup account")
    print("   Region: AF-Cairo")
    print()
    
    print("📋 2. NAVIGATE TO TRAINING:")
    print("   • Click 'Training Management'")
    print("   • Click 'Training Jobs'") 
    print("   • Click 'Create Training Job'")
    print()
    
    print("📝 3. BASIC INFORMATION:")
    print("   Job Name: arsl-recognition-training-v1")
    print("   Description: Arabic Sign Language Recognition CNN Training")
    print("   Resource Pool: Public resource pools")
    print()
    
    print("🔧 4. ALGORITHM CONFIGURATION:")
    print("   Algorithm Source: Custom Algorithm")
    print("   Code Directory: obs://arsl-youssef-af-cairo-2025/code/")
    print("   Boot File: train_arsl.py")
    print("   AI Engine: PyTorch 1.8.0-python3.7")
    print()
    
    print("📁 5. DATA CONFIGURATION:")
    print("   Data Path: obs://arsl-youssef-af-cairo-2025/datasets/raw/")
    print("   Training Output: obs://arsl-youssef-af-cairo-2025/output/")
    print("   Job Log Path: obs://arsl-youssef-af-cairo-2025/logs/")
    print()
    
    print("💾 6. RESOURCE CONFIGURATION:")
    print("   Resource Type: GPU")
    print("   Resource Pool: Public resource pools")
    print("   Resource Flavor: GPU V100 (32GB)")
    print("   Compute Nodes: 1")
    print()
    
    print("⚙️ 7. HYPERPARAMETERS:")
    hyperparams = {
        "batch_size": "64",
        "learning_rate": "0.001", 
        "epochs": "100",
        "num_classes": "32",
        "image_size": "64",
        "validation_split": "0.2",
        "patience": "15",
        "optimizer": "adam"
    }
    
    for key, value in hyperparams.items():
        print(f"   {key} = {value}")
    print()
    
    print("⏱️ 8. TRAINING SETTINGS:")
    print("   Max Running Time: 8 hours")
    print("   Auto Stop: Enabled")
    print("   Automatic Restart: Disabled")
    print()
    
    print("🚀 9. CREATE AND START:")
    print("   • Review all settings")
    print("   • Click 'Create Now'")
    print("   • Job will start automatically")
    print()

def create_job_config_file():
    """Create a configuration file for reference"""
    
    config = {
        "job_name": "arsl-recognition-training-v1",
        "description": "Arabic Sign Language Recognition CNN Training",
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
            "num_classes": "32",
            "image_size": "64",
            "validation_split": "0.2",
            "patience": "15",
            "optimizer": "adam"
        },
        "training_config": {
            "max_running_time": 28800,  # 8 hours
            "auto_stop": True
        }
    }
    
    # Save configuration
    config_file = Path("config/modelarts_job_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Job configuration saved: {config_file}")
    return config_file

def print_monitoring_info():
    """Print training monitoring information"""
    
    print("📊 STEP 3: MONITOR TRAINING PROGRESS")
    print("=" * 50)
    print()
    
    print("🔍 MONITORING LOCATIONS:")
    print("   📈 Training Jobs: https://console.huaweicloud.com/modelarts")
    print("   📊 Job Details: Click on job name for detailed view")
    print("   📋 Real-time Logs: Available in job details")
    print("   📉 Metrics: Loss, accuracy, validation metrics")
    print()
    
    print("⏱️ TRAINING TIMELINE:")
    print("   🚀 Job Creation: ~2-5 minutes")
    print("   💾 Resource Allocation: ~5-10 minutes") 
    print("   📚 Data Loading: ~10-15 minutes")
    print("   🏋️ Training: 4-6 hours (100 epochs)")
    print("   📊 Total Duration: ~5-7 hours")
    print()
    
    print("🎯 SUCCESS CRITERIA:")
    print("   📈 Training Accuracy: >90%")
    print("   📊 Validation Accuracy: >85%")
    print("   📉 Training Loss: <0.5")
    print("   ⚡ No overfitting: Val accuracy close to train accuracy")
    print()
    
    print("🚨 MONITORING ALERTS:")
    print("   ❌ Job Failed: Check logs for errors")
    print("   ⏰ Long Runtime: Normal for 54K images")
    print("   📉 Poor Accuracy: May need hyperparameter tuning")
    print("   💾 Resource Issues: Check GPU utilization")
    print()

def print_next_steps():
    """Print what happens after training"""
    
    print("🎯 STEP 4: AFTER TRAINING COMPLETES")
    print("=" * 50)
    print()
    
    print("✅ TRAINING OUTPUTS:")
    print("   📁 Model File: obs://arsl-youssef-af-cairo-2025/output/best_model.pth")
    print("   📊 Training Metrics: obs://arsl-youssef-af-cairo-2025/output/metrics.json")
    print("   📈 Training History: obs://arsl-youssef-af-cairo-2025/output/history.csv")
    print("   📋 Logs: obs://arsl-youssef-af-cairo-2025/logs/")
    print()
    
    print("🚀 PHASE 4: API DEPLOYMENT")
    print("   1. Import trained model to ModelArts Model Management")
    print("   2. Create real-time inference service")
    print("   3. Deploy API Gateway endpoints")
    print("   4. Configure auto-scaling and monitoring")
    print("   5. Test API with sample images")
    print()
    
    print("⏱️ PHASE 4 DURATION: 1-2 hours")
    print("🎉 FINAL RESULT: Production-ready Arabic Sign Language API")

def main():
    """Main function"""
    print("🏗️ MODELARTS TRAINING JOB CREATION GUIDE")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("Bucket: arsl-youssef-af-cairo-2025")
    print()
    
    # Create config file
    config_file = create_job_config_file()
    print()
    
    # Print creation steps
    print_modelarts_job_creation()
    
    # Print monitoring info
    print_monitoring_info()
    
    # Print next steps
    print_next_steps()
    
    print("🎯 SUMMARY:")
    print("✅ Training code uploaded to OBS")
    print("📋 Job configuration ready")
    print("🌐 Manual job creation required in ModelArts console")
    print("📊 Monitoring setup complete")
    print("🚀 Ready for 4-8 hour training process")

if __name__ == "__main__":
    main()