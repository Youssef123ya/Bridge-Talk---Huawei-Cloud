"""
Monitor Training Progress
Step 3: Track training metrics and progress
"""

import sys
import os
import time
from datetime import datetime

# Set environment variables
os.environ['HUAWEI_ACCESS_KEY_ID'] = 'HPUABD9ZCJI1SJDKKTB9'
os.environ['HUAWEI_SECRET_ACCESS_KEY'] = 'jJ8X6NYj7xsJejHcZ2NBUwDsNwieeWvSs15yzv6V'
os.environ['HUAWEI_PROJECT_ID'] = '15634f45a08445fab1a473d2c2e6f6cb'
os.environ['HUAWEI_REGION'] = 'af-north-1'

sys.path.append('src')

def check_training_outputs():
    """Check if training outputs are available"""
    print("🔍 CHECKING TRAINING OUTPUTS")
    print("=" * 40)
    
    try:
        from cloud.huawei_storage import HuaweiCloudStorage
        storage = HuaweiCloudStorage()
        
        # Check output directory
        output_objects = storage.list_objects(prefix="output/")
        log_objects = storage.list_objects(prefix="logs/")
        
        print(f"📁 Output files: {len(output_objects)}")
        print(f"📋 Log files: {len(log_objects)}")
        
        if output_objects:
            print("\n📊 Training Outputs Found:")
            for obj in output_objects:
                print(f"   ✅ {obj}")
        
        if log_objects:
            print("\n📋 Log Files Found:")
            for obj in log_objects[:5]:  # Show first 5
                print(f"   📄 {obj}")
        
        # Check for model file
        model_found = any("model" in obj.lower() for obj in output_objects)
        metrics_found = any("metric" in obj.lower() for obj in output_objects)
        
        if model_found:
            print("\n🎯 MODEL FILE DETECTED!")
            print("✅ Training likely completed successfully")
            return True
        else:
            print("\n⏳ No model file found yet")
            print("🔄 Training may still be in progress")
            return False
            
        storage.close()
        
    except Exception as e:
        print(f"❌ Error checking outputs: {e}")
        return False

def print_training_monitoring_guide():
    """Print comprehensive monitoring guide"""
    
    print("📊 TRAINING MONITORING GUIDE")
    print("=" * 50)
    print()
    
    print("🌐 MONITORING LOCATIONS:")
    print("   📈 Primary: https://console.huaweicloud.com/modelarts")
    print("   📊 Navigation: Training Management → Training Jobs")
    print("   🔍 Job Details: Click on 'arsl-recognition-training-v1'")
    print()
    
    print("📋 KEY METRICS TO WATCH:")
    print("   📈 Training Loss: Should decrease steadily")
    print("   📊 Training Accuracy: Should increase to >90%")
    print("   📉 Validation Loss: Should decrease without diverging")
    print("   🎯 Validation Accuracy: Target >85%")
    print("   ⏱️ Epoch Time: ~2-4 minutes per epoch")
    print()
    
    print("🚨 WARNING SIGNS:")
    print("   📈 Loss not decreasing: Learning rate too high/low")
    print("   📊 Accuracy plateau: May need more epochs")
    print("   📉 Validation diverging: Overfitting detected")
    print("   ⏰ Very slow epochs: Data loading issues")
    print("   ❌ Job failed: Check logs for errors")
    print()
    
    print("⏰ EXPECTED TIMELINE:")
    print("   🚀 Job Start: 0-10 minutes (resource allocation)")
    print("   📚 Data Loading: 10-20 minutes (54K images)")
    print("   🏋️ Epoch 1-20: 40-80 minutes (rapid learning)")
    print("   📈 Epoch 21-60: 2-4 hours (steady improvement)")
    print("   🎯 Epoch 61-100: 2.5-4 hours (fine-tuning)")
    print("   ✅ Total: 5-7 hours expected")
    print()

def print_success_criteria():
    """Print what constitutes successful training"""
    
    print("🎯 SUCCESS CRITERIA")
    print("=" * 30)
    print()
    
    print("✅ TRAINING SUCCESS INDICATORS:")
    print("   📈 Final Training Accuracy: >90%")
    print("   📊 Final Validation Accuracy: >85%")
    print("   📉 Training Loss: <0.3")
    print("   🔄 Validation Loss: <0.5")
    print("   📊 Gap (Train-Val): <5%")
    print()
    
    print("📁 EXPECTED OUTPUT FILES:")
    print("   🤖 best_model.pth: Trained model weights")
    print("   📊 metrics.json: Final performance metrics")
    print("   📈 training_history.csv: Epoch-by-epoch results")
    print("   📋 training.log: Detailed training logs")
    print()
    
    print("🚀 READY FOR PHASE 4 WHEN:")
    print("   ✅ Job status: COMPLETED")
    print("   ✅ Model file exists in output/")
    print("   ✅ Validation accuracy >85%")
    print("   ✅ No critical errors in logs")

def monitor_training_status():
    """Monitor training status continuously"""
    print("🔄 CONTINUOUS MONITORING MODE")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        check_count = 0
        while True:
            check_count += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n[{check_count}] {current_time}")
            print("-" * 30)
            
            # Check for training outputs
            training_complete = check_training_outputs()
            
            if training_complete:
                print("\n🎉 TRAINING APPEARS COMPLETE!")
                print("🚀 Ready to proceed to Phase 4: API Deployment")
                break
            
            print(f"\n⏰ Next check in 15 minutes...")
            print(f"💡 Monitor real-time: https://console.huaweicloud.com/modelarts")
            
            time.sleep(900)  # Wait 15 minutes
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
        print("💡 Continue monitoring in ModelArts console")

def main():
    """Main monitoring function"""
    print("📊 STEP 3: TRAINING PROGRESS MONITORING")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("Job: arsl-recognition-training-v1")
    print("=" * 50)
    
    # Print monitoring guide
    print_training_monitoring_guide()
    
    # Print success criteria
    print_success_criteria()
    
    # Quick status check
    print("\n🔍 CURRENT STATUS CHECK:")
    training_complete = check_training_outputs()
    
    if not training_complete:
        print(f"\n📋 MONITORING OPTIONS:")
        print(f"1. Real-time: ModelArts Console")
        print(f"2. Periodic: This script with continuous mode")
        print(f"3. Manual: Check outputs periodically")
        
        response = input(f"\n🔄 Start continuous monitoring? (y/N): ").lower().strip()
        if response == 'y':
            monitor_training_status()
    else:
        print(f"\n🎉 Training appears complete!")
        print(f"🚀 Ready for Phase 4: API Deployment")

if __name__ == "__main__":
    main()