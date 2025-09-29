#!/usr/bin/env python3
"""
Verify bucket creation and prepare for Phase 2
Account: yyacoup (d39414080c594b3296c5490459fde0e0)
"""

import sys
import os
sys.path.append('src')

from cloud.huawei_storage import HuaweiCloudStorage

def verify_bucket_creation():
    """Verify that the bucket was created successfully for yyacoup account"""
    print("🔍 Verifying bucket creation...")
    print("👤 Account: yyacoup")
    print("🌍 Region: AF-Cairo")
    print("� Bucket: arsl-youssef-af-cairo-2025")
    print("=" * 50)
    
    try:
        # Initialize storage connection
        storage = HuaweiCloudStorage()
        
        # Test bucket existence
        print("� Testing bucket connection...")
        bucket_exists = storage.bucket_exists()
        
        if bucket_exists:
            print("✅ SUCCESS: Bucket 'arsl-youssef-af-cairo-2025' found!")
            
            # Get bucket info
            bucket_info = storage.get_bucket_info()
            if bucket_info:
                print(f"📊 Bucket Region: {bucket_info.get('location', 'Unknown')}")
                print(f"📊 Storage Class: {bucket_info.get('storage_class', 'Unknown')}")
                print(f"📊 Creation Date: {bucket_info.get('creation_date', 'Unknown')}")
            
            # Test upload capability
            print("\n🧪 Testing upload capability...")
            test_upload = storage.test_upload()
            
            if test_upload:
                print("✅ SUCCESS: Upload capability verified!")
                
                # Test folder structure
                print("\n📁 Checking folder structure...")
                folders = storage.list_folders()
                expected_folders = ['data/', 'models/', 'output/', 'logs/']
                
                for folder in expected_folders:
                    if folder in folders:
                        print(f"✅ {folder} - Found")
                    else:
                        print(f"⚠️  {folder} - Missing (will be created during upload)")
                
                print("\n🎉 PHASE 1 COMPLETED SUCCESSFULLY!")
                print("📋 Ready to proceed to Phase 2: Dataset Upload")
                print("\nNext steps:")
                print("1. Run dataset upload script")
                print("2. Monitor upload progress")
                print("3. Verify upload completion")
                
                return True
            else:
                print("❌ FAILED: Upload test failed")
                print("🔧 Check bucket permissions and IAM access")
                return False
        else:
            print("❌ FAILED: Bucket not found or not accessible")
            print("🔧 Please complete manual bucket creation first")
            print("� See: scripts/manual_bucket_creation_guide.md")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("🔧 Check your credentials and network connection")
        return False
    
    finally:
        try:
            storage.close()
        except:
            pass

def print_phase2_instructions():
    """Print instructions for Phase 2"""
    print("\n" + "=" * 60)
    print("🚀 READY FOR PHASE 2: DATASET UPLOAD")
    print("=" * 60)
    print()
    print("📤 Upload Command:")
    print('& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/upload_dataset.py')
    print()
    print("⏱️  Expected Duration: 2-4 hours")
    print("📊 Progress: Real-time progress bar")
    print("🔄 Resumable: Can resume if interrupted")
    print("📈 Parallel: Multi-threaded for speed")
    print()
    print("💡 Tips:")
    print("- Keep VS Code open to monitor progress")
    print("- Ensure stable internet connection")
    print("- Upload will continue in background")
    print()

if __name__ == "__main__":
    print("🚀 PHASE 1 VERIFICATION")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("Bucket: arsl-youssef-af-cairo-2025")
    print()
    
    success = verify_bucket_creation()
    
    if success:
        print_phase2_instructions()
        
        # Ask user if they want to proceed
        print("🤔 Would you like to start Phase 2 (Dataset Upload) now?")
        print("Type 'yes' to continue or 'no' to wait:")
        
        # In automation, we'll assume they want to continue
        # but this provides a natural checkpoint
    else:
        print("\n❌ Phase 1 not complete. Please fix issues above first.")
    
    sys.exit(0 if success else 1)