#!/usr/bin/env python3
"""
Master script to execute all phases of Huawei Cloud deployment
Arabic Sign Language Recognition Project
"""

import sys
import os
import subprocess
import time
import json

def print_phase_header(phase_num, title):
    """Print formatted phase header"""
    print("\n" + "="*80)
    print(f"🚀 PHASE {phase_num}: {title}")
    print("="*80)

def run_script(script_path, description):
    """Run a Python script and return success status"""
    print(f"\n📋 {description}")
    print("-" * 60)
    
    python_exe = "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe"
    
    try:
        result = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run {script_path}: {e}")
        return False

def wait_for_user_confirmation(message):
    """Wait for user to confirm manual step"""
    print(f"\n⏸️  {message}")
    input("Press Enter when ready to continue...")

def main():
    """Execute all deployment phases"""
    
    print("🎯 Arabic Sign Language Recognition - Huawei Cloud Deployment")
    print("🌍 Target Region: AF-Cairo (af-north-1)")
    print("📊 Dataset: 108,098 images, 32 classes")
    print("⏱️  Estimated Total Time: 7-14 hours")
    
    # Phase 1: Bucket Setup
    print_phase_header(1, "BUCKET CREATION & VERIFICATION")
    
    print("📋 Phase 1 requires manual bucket creation in Huawei Cloud console:")
    print("   1. Go to: https://console.huaweicloud.com")
    print("   2. Navigate to: Object Storage Service (OBS)")
    print("   3. Create bucket with:")
    print("      - Name: arsl-youssef-af-cairo-2025")
    print("      - Region: AF-Cairo")
    print("      - Storage Class: Standard")
    print("      - Access Control: Private")
    
    wait_for_user_confirmation("Complete bucket creation in console")
    
    # Verify bucket
    if not run_script("scripts/verify_bucket.py", "Verifying bucket setup"):
        print("❌ Bucket verification failed. Please check bucket creation.")
        return False
    
    print("✅ Phase 1 completed successfully!")
    
    # Phase 2: Dataset Upload
    print_phase_header(2, "DATASET UPLOAD")
    
    print("📤 Starting parallel upload of 108,098 images...")
    print("⏱️  Estimated time: 2-4 hours")
    print("💡 You can monitor progress in the upload output")
    
    if not run_script("scripts/upload_dataset.py", "Uploading dataset"):
        print("❌ Dataset upload failed. Check network and try again.")
        return False
        
    print("✅ Phase 2 completed successfully!")
    
    # Phase 3: Training Setup
    print_phase_header(3, "TRAINING CONFIGURATION")
    
    print("🏋️ Setting up ModelArts training job...")
    print("⏱️  Training duration: 4-8 hours")
    print("🎯 Target accuracy: >85%")
    
    if not run_script("scripts/deploy_training.py", "Deploying training job"):
        print("❌ Training deployment failed. Check ModelArts console.")
        return False
        
    print("✅ Phase 3 completed successfully!")
    print("📈 Training job is running. Monitor progress in ModelArts console.")
    
    # Wait for training completion
    print("\n⏳ Waiting for training to complete...")
    print("💡 This will take 4-8 hours. You can:")
    print("   - Monitor progress in ModelArts console")
    print("   - Check logs and metrics")
    print("   - Come back when training is done")
    
    wait_for_user_confirmation("Training completed and model accuracy is >85%")
    
    # Phase 4: API Deployment
    print_phase_header(4, "API DEPLOYMENT")
    
    print("🌐 Deploying inference API service...")
    print("⚡ Target response time: <500ms")
    print("🔧 Auto-scaling: 1-3 instances")
    
    if not run_script("scripts/deploy_api.py", "Deploying inference API"):
        print("❌ API deployment failed. Check ModelArts console.")
        return False
        
    print("✅ Phase 4 completed successfully!")
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📊 Deployment Summary:")
    print("   ✅ OBS Bucket: arsl-youssef-af-cairo-2025")
    print("   ✅ Dataset: 108,098 images uploaded")
    print("   ✅ Model: Trained with >85% accuracy")
    print("   ✅ API: Real-time inference service deployed")
    
    print("\n🌐 Your Arabic Sign Language Recognition system is now live!")
    
    # Show service information
    if os.path.exists("logs/inference_service_info.json"):
        with open("logs/inference_service_info.json", "r") as f:
            service_info = json.load(f)
        
        print(f"\n📋 API Service Details:")
        print(f"   🏷️  Service: {service_info.get('service_name', 'N/A')}")
        print(f"   🆔 ID: {service_info.get('service_id', 'N/A')}")
        print(f"   🕐 Deployed: {service_info.get('deployed_at', 'N/A')}")
        
        if service_info.get('endpoints'):
            print(f"   🌐 Endpoints:")
            for endpoint in service_info['endpoints']:
                print(f"      📍 {endpoint}")
    
    print("\n📚 Next Steps:")
    print("   1. Test API with sample images")
    print("   2. Integrate with mobile/web applications")
    print("   3. Monitor performance and scale as needed")
    print("   4. Set up automated retraining pipeline")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Deployment completed successfully! 🌟")
    else:
        print("\n❌ Deployment failed. Check logs and try again.")
    
    sys.exit(0 if success else 1)