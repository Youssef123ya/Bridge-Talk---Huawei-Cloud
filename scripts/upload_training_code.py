"""
Upload Training Code to OBS
Prepares all necessary code files for ModelArts training
"""

import sys
import os
from pathlib import Path

# Set environment variables and imports
os.environ['HUAWEI_ACCESS_KEY_ID'] = 'HPUABD9ZCJI1SJDKKTB9'
os.environ['HUAWEI_SECRET_ACCESS_KEY'] = 'jJ8X6NYj7xsJejHcZ2NBUwDsNwieeWvSs15yzv6V'
os.environ['HUAWEI_PROJECT_ID'] = '15634f45a08445fab1a473d2c2e6f6cb'
os.environ['HUAWEI_REGION'] = 'af-north-1'

sys.path.append('src')
from cloud.huawei_storage import HuaweiCloudStorage

def upload_training_files():
    """Upload all training code files to OBS"""
    print("📤 UPLOADING TRAINING CODE TO OBS")
    print("=" * 50)
    
    storage = HuaweiCloudStorage()
    
    # Training files to upload
    code_files = [
        {
            "local": "src/cloud/train_arsl.py",
            "obs": "code/train_arsl.py",
            "description": "Main training script"
        },
        {
            "local": "scripts/cnn_architectures.py", 
            "obs": "code/cnn_architectures.py",
            "description": "CNN model architectures"
        },
        {
            "local": "scripts/augmentation.py",
            "obs": "code/augmentation.py", 
            "description": "Data augmentation"
        },
        {
            "local": "scripts/base_model.py",
            "obs": "code/base_model.py",
            "description": "Base model classes"
        },
        {
            "local": "requirements.txt",
            "obs": "code/requirements.txt",
            "description": "Python dependencies"
        }
    ]
    
    # Upload each file
    upload_success = True
    uploaded_files = []
    
    for file_info in code_files:
        local_path = Path(file_info["local"])
        obs_path = file_info["obs"]
        description = file_info["description"]
        
        print(f"\n📄 Uploading: {description}")
        print(f"   Local: {local_path}")
        print(f"   OBS: obs://arsl-youssef-af-cairo-2025/{obs_path}")
        
        if local_path.exists():
            try:
                success = storage.upload_file(str(local_path), obs_path)
                if success:
                    print(f"   ✅ SUCCESS")
                    uploaded_files.append(obs_path)
                else:
                    print(f"   ❌ FAILED")
                    upload_success = False
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                upload_success = False
        else:
            print(f"   ⚠️  FILE NOT FOUND - skipping")
    
    # Create a simple training requirements file
    print(f"\n📄 Creating training requirements...")
    training_reqs = """torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
pillow>=8.0.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
tqdm>=4.60.0
"""
    
    try:
        # Save locally first
        req_path = Path("temp_requirements.txt")
        with open(req_path, 'w') as f:
            f.write(training_reqs)
        
        # Upload to OBS
        success = storage.upload_file(str(req_path), "code/requirements.txt")
        if success:
            print(f"   ✅ Training requirements uploaded")
            uploaded_files.append("code/requirements.txt")
        
        # Clean up temp file
        req_path.unlink()
        
    except Exception as e:
        print(f"   ❌ Error uploading requirements: {e}")
    
    storage.close()
    
    # Summary
    print(f"\n📊 UPLOAD SUMMARY")
    print("=" * 30)
    print(f"✅ Files uploaded: {len(uploaded_files)}")
    print(f"📁 Code directory: obs://arsl-youssef-af-cairo-2025/code/")
    
    if uploaded_files:
        print(f"\n📋 Uploaded files:")
        for file in uploaded_files:
            print(f"   • {file}")
    
    if upload_success:
        print(f"\n🎉 CODE UPLOAD COMPLETE!")
        print(f"🚀 Ready for ModelArts training job creation")
        return True
    else:
        print(f"\n❌ Some uploads failed")
        return False

def verify_upload():
    """Verify training code is in OBS"""
    print(f"\n🔍 VERIFYING CODE UPLOAD")
    print("=" * 30)
    
    storage = HuaweiCloudStorage()
    
    try:
        # List objects in code directory
        code_objects = storage.list_objects(prefix="code/")
        
        print(f"📁 Files in code directory: {len(code_objects)}")
        for obj in code_objects:
            print(f"   ✅ {obj}")
        
        if len(code_objects) >= 3:  # At least main files
            print(f"\n✅ CODE VERIFICATION PASSED")
            print(f"🚀 Ready to create ModelArts training job")
            return True
        else:
            print(f"\n⚠️  Insufficient files uploaded")
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False
    
    finally:
        storage.close()

def main():
    """Main function"""
    print("🚀 STEP 1: UPLOAD TRAINING CODE")
    print("Account: yyacoup")
    print("Bucket: arsl-youssef-af-cairo-2025")
    print("Target: code/ directory")
    print()
    
    # Upload files
    upload_ok = upload_training_files()
    
    if upload_ok:
        # Verify upload
        verify_ok = verify_upload()
        
        if verify_ok:
            print(f"\n🎯 NEXT STEP: CREATE MODELARTS JOB")
            print(f"📋 Use: scripts/create_modelarts_job.py")
            print(f"🌐 Or manual: https://console.huaweicloud.com/modelarts")
        else:
            print(f"\n❌ Upload verification failed")
    else:
        print(f"\n❌ Upload failed - check errors above")

if __name__ == "__main__":
    main()