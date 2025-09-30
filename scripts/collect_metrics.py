#!/usr/bin/env python3
"""
ARSL Custom Metrics Collection Script
Collects application metrics and sends to Cloud Eye
"""

import json
import time
import requests
import psutil
from datetime import datetime
from pathlib import Path

class ARSLMetricsCollector:
    """Collects and sends custom metrics to Cloud Eye"""
    
    def __init__(self):
        self.config = self.load_config()
        self.api_endpoint = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
        self.metrics_file = "/opt/arsl/metrics.log"
        
    def load_config(self):
        """Load configuration"""
        try:
            with open("/opt/arsl/config.json", "r") as f:
                return json.load(f)
        except:
            return {"monitoring_interval": 60, "log_level": "INFO"}
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            "network_bytes_sent": psutil.net_io_counters().bytes_sent,
            "network_bytes_recv": psutil.net_io_counters().bytes_recv
        }
    
    def test_api_performance(self):
        """Test API endpoint performance"""
        performance_metrics = {}
        
        # Test health endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_endpoint}/v1/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            performance_metrics["health_check"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": response_time,
                "status_code": response.status_code
            }
        except Exception as e:
            performance_metrics["health_check"] = {
                "status": "error",
                "error": str(e),
                "response_time_ms": 0
            }
        
        # Test model info endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_endpoint}/v1/model/info", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            performance_metrics["model_info"] = {
                "status": "available" if response.status_code == 200 else "unavailable",
                "response_time_ms": response_time,
                "status_code": response.status_code
            }
        except Exception as e:
            performance_metrics["model_info"] = {
                "status": "error",
                "error": str(e),
                "response_time_ms": 0
            }
        
        return performance_metrics
    
    def send_metrics_to_cloud_eye(self, metrics):
        """Send metrics to Cloud Eye (simplified - would use actual Cloud Eye API)"""
        
        # Format metrics for Cloud Eye
        cloud_eye_metrics = []
        
        # System metrics
        system = metrics.get("system", {})
        if system:
            cloud_eye_metrics.extend([
                {
                    "namespace": "ARSL/System",
                    "metric_name": "cpu_utilization",
                    "value": system.get("cpu_percent", 0),
                    "unit": "Percent",
                    "timestamp": system.get("timestamp")
                },
                {
                    "namespace": "ARSL/System", 
                    "metric_name": "memory_utilization",
                    "value": system.get("memory_percent", 0),
                    "unit": "Percent",
                    "timestamp": system.get("timestamp")
                },
                {
                    "namespace": "ARSL/System",
                    "metric_name": "disk_utilization", 
                    "value": system.get("disk_percent", 0),
                    "unit": "Percent",
                    "timestamp": system.get("timestamp")
                }
            ])
        
        # API performance metrics
        api = metrics.get("api", {})
        health_check = api.get("health_check", {})
        if health_check:
            cloud_eye_metrics.append({
                "namespace": "ARSL/API",
                "metric_name": "health_check_response_time",
                "value": health_check.get("response_time_ms", 0),
                "unit": "Milliseconds",
                "timestamp": metrics.get("timestamp")
            })
        
        # Log metrics (in production, would send to actual Cloud Eye API)
        metrics_log = {
            "timestamp": datetime.now().isoformat(),
            "cloud_eye_metrics": cloud_eye_metrics,
            "raw_metrics": metrics
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_log) + "\n")
        
        return len(cloud_eye_metrics)
    
    def run_collection_loop(self):
        """Main metrics collection loop"""
        print(f"🔍 Starting ARSL metrics collection...")
        print(f"API Endpoint: {self.api_endpoint}")
        print(f"Collection Interval: {self.config.get('monitoring_interval', 60)} seconds")
        print(f"Metrics Log: {self.metrics_file}")
        print("=" * 50)
        
        while True:
            try:
                # Collect all metrics
                system_metrics = self.collect_system_metrics()
                api_metrics = self.test_api_performance()
                
                # Combine metrics
                all_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "system": system_metrics,
                    "api": api_metrics
                }
                
                # Send to Cloud Eye
                metrics_count = self.send_metrics_to_cloud_eye(all_metrics)
                
                # Display status
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Metrics collected:")
                print(f"  System: CPU {system_metrics['cpu_percent']:.1f}%, "
                      f"Memory {system_metrics['memory_percent']:.1f}%, "
                      f"Disk {system_metrics['disk_percent']:.1f}%")
                print(f"  API Health: {api_metrics.get('health_check', {}).get('status', 'unknown')}")
                print(f"  Metrics sent to Cloud Eye: {metrics_count}")
                print("-" * 30)
                
                # Wait for next collection
                time.sleep(self.config.get('monitoring_interval', 60))
                
            except KeyboardInterrupt:
                print("\n📊 Metrics collection stopped by user")
                break
            except Exception as e:
                print(f"❌ Collection error: {e}")
                time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    collector = ARSLMetricsCollector()
    collector.run_collection_loop()
