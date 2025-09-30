#!/usr/bin/env python3
"""
Monitor upload progress by checking OBS bucket contents
"""
import sys
import os
import time
sys.path.append('src')

from cloud.huawei_storage import HuaweiCloudStorage

def monitor_upload_progress():
    """Monitor the upload progress"""
    print("📊 Monitoring Upload Progress...")
    print("=" * 50)
    
    try:
        storage = HuaweiCloudStorage()
        
        while True:
            # Count uploaded files
            objects = storage.list_objects(prefix="datasets/raw/")
            
            # Count by class
            classes = {}
            for obj in objects:
                parts = obj.split('/')
                if len(parts) >= 3:
                    class_name = parts[2]
                    classes[class_name] = classes.get(class_name, 0) + 1
            
            total_uploaded = len(objects)
            total_expected = 54072  # From upload script
            
            print(f"\n📊 Upload Progress: {total_uploaded:,} / {total_expected:,} files")
            print(f"📈 Progress: {(total_uploaded/total_expected)*100:.1f}%")
            print(f"📁 Classes found: {len(classes)}")
            
            if classes:
                print("\n📂 Top uploading classes:")
                sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
                for class_name, count in sorted_classes[:10]:
                    print(f"   {class_name}: {count:,} files")
            
            if total_uploaded >= total_expected:
                print("\n🎉 Upload Complete!")
                break
                
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            storage.close()
        except:
            pass

if __name__ == "__main__":
    monitor_upload_progress()