#!/usr/bin/env python3
"""
Simple Monitoring Test Execution Script
Basic tests for ARSL monitoring system
"""

import json
import time
import requests
import psutil
from datetime import datetime
from pathlib import Path

def test_system_metrics():
    """Test basic system metrics collection"""
    print("🖥️ Testing System Metrics...")
    
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        print(f"   CPU Usage: {cpu_usage:.1f}%")
        print(f"   Memory Usage: {memory_usage:.1f}%")
        print(f"   Disk Usage: {disk_usage:.1f}%")
        
        # Check if metrics are reasonable
        if all([0 <= cpu_usage <= 100, 0 <= memory_usage <= 100, 0 <= disk_usage <= 100]):
            print("   ✅ System metrics test passed")
            return True
        else:
            print("   ❌ System metrics test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing system metrics: {e}")
        return False

def test_api_health():
    """Test API health endpoints"""
    print("\n🌐 Testing API Health...")
    
    api_base = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
    endpoints = ["/v1/health", "/v1/model/info"]
    
    all_passed = True
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(f"{api_base}{endpoint}", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                print(f"   ✅ {endpoint}: {response.status_code} ({response_time:.1f}ms)")
            else:
                print(f"   ❌ {endpoint}: {response.status_code} ({response_time:.1f}ms)")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ {endpoint}: Error - {e}")
            all_passed = False
    
    return all_passed

def test_log_file_access():
    """Test log file access and creation"""
    print("\n📝 Testing Log File Access...")
    
    try:
        log_dir = Path("/tmp/arsl_test_logs")
        log_dir.mkdir(exist_ok=True)
        
        test_log = log_dir / "test_monitoring.log"
        
        # Write test log entry
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "test": "monitoring_system",
            "status": "testing"
        }
        
        with open(test_log, "w") as f:
            json.dump(test_data, f)
        
        # Verify file was created
        if test_log.exists():
            print("   ✅ Log file creation test passed")
            return True
        else:
            print("   ❌ Log file creation test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing log access: {e}")
        return False

def run_basic_tests():
    """Run basic monitoring tests"""
    print("🧪 RUNNING BASIC MONITORING TESTS")
    print("=" * 40)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("System Metrics", test_system_metrics),
        ("API Health", test_api_health),
        ("Log File Access", test_log_file_access)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Success Rate: {(passed/total*100):.1f}%")
    
    if passed == total:
        print("\n🎉 All basic tests passed!")
        return True
    else:
        print("\n⚠️ Some tests failed. Check system configuration.")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    exit(0 if success else 1)
