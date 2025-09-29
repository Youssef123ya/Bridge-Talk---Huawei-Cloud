#!/usr/bin/env python3
"""
Simple bucket verification for yyacoup account
Test if bucket exists and can be accessed
"""

import sys
import os
sys.path.append('src')

from cloud.huawei_storage import HuaweiCloudStorage

def test_bucket_access():
    """Test if bucket is accessible and working"""
    print("🚀 PHASE 1 VERIFICATION")
    print("👤 Account: yyacoup")
    print("🌍 Region: AF-Cairo")
    print("📁 Bucket: arsl-youssef-af-cairo-2025")
    print("=" * 50)
    
    try:
        # Initialize storage
        print("📡 Connecting to OBS...")
        storage = HuaweiCloudStorage()
        
        # Test bucket info
        print("🔍 Testing bucket access...")
        bucket_info = storage.get_bucket_info()
        
        if bucket_info and bucket_info.get('exists', False):
            print("✅ SUCCESS: Bucket found and accessible!")
            print(f"📊 Bucket: {bucket_info.get('name')}")
            print(f"📊 Region: {bucket_info.get('region')}")
            
            # Test upload
            print("\n🧪 Testing upload capability...")
            test_content = "Verification test for yyacoup account"
            test_success = storage.upload_file_content(test_content, "test/verify.txt")
            
            if test_success:
                print("✅ SUCCESS: Upload test passed!")
                
                # Test download
                downloaded = storage.download_file_content("test/verify.txt")
                if downloaded and "yyacoup" in downloaded:
                    print("✅ SUCCESS: Download test passed!")
                    
                    print("\n🎉 PHASE 1 COMPLETED!")
                    print("✅ Bucket is ready for dataset upload")
                    print("\n📋 Next: Run Phase 2 (Dataset Upload)")
                    print('& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/upload_dataset.py')
                    
                    return True
                else:
                    print("❌ Download test failed")
            else:
                print("❌ Upload test failed - check permissions")
        else:
            print("❌ BUCKET NOT FOUND!")
            print("\n🔧 MANUAL CREATION REQUIRED:")
            print("1. Go to https://console.huaweicloud.com")
            print("2. Login with yyacoup account")
            print("3. Navigate to Object Storage Service (OBS)")
            print("4. Create bucket with these settings:")
            print("   - Name: arsl-youssef-af-cairo-2025")
            print("   - Region: AF-Cairo")
            print("   - Storage Class: Standard")
            print("   - Access Control: Private")
            print("\n📖 Detailed guide: scripts/manual_bucket_creation_guide.md")
            
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\n🔧 Troubleshooting:")
        print("- Check internet connection")
        print("- Verify Huawei Cloud credentials")
        print("- Ensure yyacoup account has OBS access")
        return False
    
    finally:
        try:
            storage.close()
        except:
            pass

if __name__ == "__main__":
    success = test_bucket_access()
    if not success:
        print("\n❌ Phase 1 incomplete - fix issues above first")
    sys.exit(0 if success else 1)