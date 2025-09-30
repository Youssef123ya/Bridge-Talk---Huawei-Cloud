"""
Deploy Training Job to ModelArts
Automatically uploads code and starts distributed training
"""

import sys
import os
import json
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from cloud.huawei_storage import HuaweiCloudStorage
from cloud.huawei_modelarts import HuaweiModelArts

def upload_training_code():
    """Upload training code to OBS for ModelArts"""
    print("📤 Uploading training code to OBS...")
    
    storage = HuaweiCloudStorage()
    
    # Files to upload
    code_files = [
        ("src/cloud/train_arsl.py", "code/train_arsl.py"),
        ("scripts/data_splitter.py", "code/data_splitter.py"),
        ("scripts/cnn_architectures.py", "code/cnn_architectures.py"),
        ("scripts/augmentation.py", "code/augmentation.py"),
        ("requirements.txt", "code/requirements.txt")
    ]
    
    upload_success = True
    for local_file, obs_path in code_files:
        if Path(local_file).exists():
            success = storage.upload_file(local_file, obs_path)
            if success:
                print(f"   ✅ {local_file} -> obs://{storage.bucket_name}/{obs_path}")
            else:
                print(f"   ❌ Failed: {local_file}")
                upload_success = False
        else:
            print(f"   ⚠️  File not found: {local_file}")
    
    storage.close()
    return upload_success

def create_training_job():
    """Create and start ModelArts training job"""
    print("\n🚀 Creating ModelArts training job...")
    
    # Load configuration
    config_file = "config/modelarts_training_config.json"
    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    try:
        # Initialize ModelArts client
        modelarts = HuaweiModelArts()
        
        # Create training job
        job_result = modelarts.create_training_job(config)
        
        if job_result and 'job_id' in job_result:
            job_id = job_result['job_id']
            print(f"✅ Training job created successfully!")
            print(f"   📋 Job ID: {job_id}")
            print(f"   📝 Job Name: {config['job_name']}")
            print(f"   💾 Instance: {config['resource_config']['flavor']}")
            
            # Monitor job status
            print(f"\n📊 Monitoring training job status...")
            monitor_training_job(modelarts, job_id)
            
            return True
        else:
            print("❌ Failed to create training job")
            return False
            
    except Exception as e:
        print(f"❌ Error creating training job: {e}")
        return False

def monitor_training_job(modelarts, job_id, max_wait_minutes=10):
    """Monitor training job initial status"""
    print(f"🔍 Checking job status (monitoring for {max_wait_minutes} minutes)...")
    
    for i in range(max_wait_minutes):
        try:
            status = modelarts.get_training_job_status(job_id)
            print(f"   [{i+1}/{max_wait_minutes}] Status: {status}")
            
            if status in ['COMPLETED', 'FAILED', 'STOPPED']:
                break
            elif status == 'RUNNING':
                print("   🏃 Job is running - monitoring can continue in ModelArts console")
                break
                
            time.sleep(60)  # Wait 1 minute
            
        except Exception as e:
            print(f"   ⚠️  Error checking status: {e}")
            break
    
    print(f"\n🌐 Continue monitoring at: https://console.huaweicloud.com/modelarts")

def print_next_steps():
    """Print what to do next"""
    print("\n" + "="*60)
    print("🎯 PHASE 3 DEPLOYMENT COMPLETE")
    print("="*60)
    print()
    print("📋 What's Happening:")
    print("   🏋️  Training job submitted to ModelArts")
    print("   💾 GPU instance provisioning (~5-10 minutes)")
    print("   🚀 Training will start automatically")
    print("   ⏱️  Expected duration: 4-8 hours")
    print()
    print("📊 Monitor Progress:")
    print("   🌐 ModelArts Console: https://console.huaweicloud.com/modelarts")
    print("   📈 Training Metrics: Real-time loss and accuracy")
    print("   📋 Logs: Detailed training progress")
    print()
    print("📱 Next Steps:")
    print("   1. Monitor training progress in console")
    print("   2. Wait for training completion (~4-8 hours)")
    print("   3. Proceed to Phase 4: API Deployment")
    print()
    print("🔔 You will be notified when training completes!")

def main():
    """Main deployment function"""
    print("🚀 PHASE 3: MODELARTS TRAINING DEPLOYMENT")
    print("=" * 50)
    
    # Step 1: Upload training code
    if not upload_training_code():
        print("❌ Failed to upload training code")
        return False
    
    # Step 2: Create training job
    if not create_training_job():
        print("❌ Failed to create training job")
        return False
    
    # Step 3: Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Phase 3 deployment successful!")
        else:
            print("\n❌ Phase 3 deployment failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)