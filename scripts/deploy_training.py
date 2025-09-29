#!/usr/bin/env python3
"""
Deploy ModelArts training job for Arabic Sign Language Recognition
"""

import sys
import os
import json
import time
sys.path.append('src')

from cloud.huawei_modelarts import HuaweiModelArts
import logging

def main():
    """Deploy training job to ModelArts"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🏋️ Deploying ModelArts Training Job...")
    print("=" * 60)
    
    modelarts = HuaweiModelArts()
    
    try:
        # Training job configuration
        training_config = {
            "algorithm": {
                "algorithm_name": "arsl-recognition",
                "algorithm_version": "1.0",
                "code_dir": "obs://arsl-youssef-af-cairo-2025/code/",
                "boot_file": "train_arsl.py",
                "engine": {
                    "engine_name": "PyTorch",
                    "engine_version": "1.8.0-cuda10.2-py3.7-ubuntu18.04"
                }
            },
            "training_job_name": f"arsl-training-job-{int(time.time())}",
            "description": "Arabic Sign Language Recognition CNN Training",
            "config": {
                "worker_server_num": 1,
                "app_url": "obs://arsl-youssef-af-cairo-2025/code/train_arsl.py",
                "data_url": "obs://arsl-youssef-af-cairo-2025/data/processed/",
                "train_url": "obs://arsl-youssef-af-cairo-2025/output/",
                "log_url": "obs://arsl-youssef-af-cairo-2025/logs/",
                "spec_id": 1,  # modelarts.vm.gpu.v100
                "engine_id": 1,  # PyTorch 1.8.0
                "hyperparameters": [
                    {"name": "learning_rate", "value": "0.001"},
                    {"name": "batch_size", "value": "64"},
                    {"name": "epochs", "value": "100"},
                    {"name": "num_classes", "value": "32"},
                    {"name": "patience", "value": "10"}
                ]
            }
        }
        
        print("📋 Training Configuration:")
        print(f"   🏷️  Job Name: {training_config['training_job_name']}")
        print(f"   🔧 Instance: GPU V100")
        print(f"   📊 Data Path: {training_config['config']['data_url']}")
        print(f"   💾 Output Path: {training_config['config']['train_url']}")
        print(f"   📈 Epochs: 100")
        print(f"   🎯 Batch Size: 64")
        
        # Create training job
        print("\n🚀 Creating training job...")
        job_id = modelarts.create_training_job(training_config)
        
        if job_id:
            print(f"✅ Training job created successfully!")
            print(f"   🆔 Job ID: {job_id}")
            print(f"   🔗 Monitor at: https://console.huaweicloud.com/modelarts")
            
            # Monitor training progress
            print("\n📊 Monitoring training progress...")
            status = modelarts.get_training_status(job_id)
            print(f"   Status: {status}")
            
            if status in ["Running", "Creating"]:
                print("\n⏱️  Training job is starting/running!")
                print("   📈 You can monitor progress in ModelArts console")
                print("   ⏳ Estimated training time: 4-8 hours")
                print("   🎯 Target accuracy: >85%")
                
                # Save job info for later reference
                job_info = {
                    "job_id": job_id,
                    "job_name": training_config['training_job_name'],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": training_config
                }
                
                with open("logs/training_job_info.json", "w") as f:
                    json.dump(job_info, f, indent=2)
                    
                print(f"   💾 Job info saved to: logs/training_job_info.json")
                
            return True
        else:
            print("❌ Failed to create training job")
            return False
            
    except Exception as e:
        print(f"\n❌ Training job deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)