#!/usr/bin/env python3
"""
ARSL System Monitoring Script
Collects and reports metrics for the ARSL project
"""

import psutil
import requests
import time
import json
from datetime import datetime

class ARSLMonitor:
    """System monitoring for ARSL project"""
    
    def __init__(self):
        self.api_base = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
        
    def get_system_metrics(self):
        """Collect system performance metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_api_health(self):
        """Test API endpoint health"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base}/v1/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": response_time,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": 0
            }
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print(f"🔍 Starting ARSL monitoring...")
        print(f"API Base: {self.api_base}")
        print(f"Monitoring interval: 30 seconds")
        print("=" * 50)
        
        while True:
            try:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                api_health = self.test_api_health()
                
                # Display metrics
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] System Status:")
                print(f"  CPU: {system_metrics['cpu_percent']:.1f}%")
                print(f"  Memory: {system_metrics['memory_percent']:.1f}%")
                print(f"  Disk: {system_metrics['disk_percent']:.1f}%")
                print(f"  API Status: {api_health['status']}")
                print(f"  API Response: {api_health['response_time_ms']:.1f}ms")
                print("-" * 30)
                
                # Save metrics to file
                metrics_data = {
                    "timestamp": timestamp,
                    "system": system_metrics,
                    "api": api_health
                }
                
                with open("/opt/arsl/metrics.json", "a") as f:
                    f.write(json.dumps(metrics_data) + "\n")
                
                # Sleep
                time.sleep(30)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    monitor = ARSLMonitor()
    monitor.run_monitoring()
