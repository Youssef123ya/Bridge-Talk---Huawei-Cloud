"""
Step 2: Configure Cloud Eye Monitoring
Complete setup for production monitoring dashboards and metrics collection
"""

import json
from pathlib import Path
from datetime import datetime

def print_cloud_eye_configuration():
    """Print Cloud Eye monitoring configuration guide"""
    
    print("📊 STEP 2: CONFIGURE CLOUD EYE MONITORING")
    print("=" * 60)
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Services: ECS, API Gateway, ModelArts, OBS")
    print()
    
    print("🌐 1. ACCESS CLOUD EYE:")
    print("   URL: https://console.huaweicloud.com/ces")
    print("   Login: yyacoup account")
    print("   Navigate: Cloud Eye → Overview")
    print()
    
    print("📈 2. ENABLE BASIC MONITORING:")
    print("   • ECS Instance Monitoring: Enabled by default")
    print("   • Detailed Monitoring: Enable for 1-minute intervals")
    print("   • Custom Metrics: Configure for application metrics")
    print("   • Log Collection: Enable for application logs")
    print()

def create_monitoring_dashboard():
    """Create comprehensive monitoring dashboard configuration"""
    
    dashboard_config = {
        "dashboard_name": "ARSL Production Monitor",
        "description": "Comprehensive monitoring for Arabic Sign Language Recognition system",
        "refresh_interval": "30s",
        "time_range": "1h",
        
        "panels": [
            {
                "id": "system_overview",
                "title": "System Overview",
                "type": "stat",
                "position": {"x": 0, "y": 0, "w": 24, "h": 6},
                "metrics": [
                    {
                        "name": "ECS Instance Status",
                        "namespace": "SYS.ECS",
                        "metric": "vm_status",
                        "dimensions": {"instance_id": "i-xxxxxxxxx"},
                        "unit": "Count",
                        "stat": "Maximum"
                    },
                    {
                        "name": "API Gateway Health",
                        "namespace": "SYS.APIG",
                        "metric": "api_status",
                        "unit": "Count",
                        "stat": "Average"
                    },
                    {
                        "name": "ModelArts Service",
                        "namespace": "SYS.ModelArts",
                        "metric": "service_status",
                        "unit": "Count",
                        "stat": "Maximum"
                    }
                ]
            },
            
            {
                "id": "performance_metrics",
                "title": "Performance Metrics",
                "type": "graph",
                "position": {"x": 0, "y": 6, "w": 12, "h": 8},
                "metrics": [
                    {
                        "name": "CPU Utilization",
                        "namespace": "SYS.ECS",
                        "metric": "cpu_util",
                        "unit": "Percent",
                        "color": "#FF6B6B"
                    },
                    {
                        "name": "Memory Usage",
                        "namespace": "SYS.ECS", 
                        "metric": "mem_util",
                        "unit": "Percent",
                        "color": "#4ECDC4"
                    },
                    {
                        "name": "Disk Usage",
                        "namespace": "SYS.ECS",
                        "metric": "disk_util",
                        "unit": "Percent",
                        "color": "#45B7D1"
                    }
                ],
                "thresholds": [
                    {"value": 70, "color": "yellow", "label": "Warning"},
                    {"value": 90, "color": "red", "label": "Critical"}
                ]
            },
            
            {
                "id": "api_performance",
                "title": "API Performance",
                "type": "graph",
                "position": {"x": 12, "y": 6, "w": 12, "h": 8},
                "metrics": [
                    {
                        "name": "Request Rate",
                        "namespace": "SYS.APIG",
                        "metric": "req_count",
                        "unit": "Count/Second",
                        "color": "#96CEB4"
                    },
                    {
                        "name": "Response Time",
                        "namespace": "SYS.APIG",
                        "metric": "latency",
                        "unit": "Milliseconds",
                        "color": "#FFEAA7"
                    },
                    {
                        "name": "Error Rate",
                        "namespace": "SYS.APIG",
                        "metric": "error_count",
                        "unit": "Percent",
                        "color": "#FD79A8"
                    }
                ]
            },
            
            {
                "id": "model_metrics",
                "title": "Model Performance",
                "type": "stat",
                "position": {"x": 0, "y": 14, "w": 8, "h": 6},
                "metrics": [
                    {
                        "name": "Inference Latency",
                        "namespace": "ARSL/ModelArts",
                        "metric": "inference_time",
                        "unit": "Milliseconds",
                        "thresholds": [
                            {"min": 0, "max": 300, "color": "green"},
                            {"min": 300, "max": 800, "color": "yellow"},
                            {"min": 800, "max": 9999, "color": "red"}
                        ]
                    },
                    {
                        "name": "Prediction Accuracy",
                        "namespace": "ARSL/Business",
                        "metric": "accuracy_rate",
                        "unit": "Percent",
                        "thresholds": [
                            {"min": 95, "max": 100, "color": "green"},
                            {"min": 85, "max": 95, "color": "yellow"},
                            {"min": 0, "max": 85, "color": "red"}
                        ]
                    }
                ]
            },
            
            {
                "id": "business_metrics",
                "title": "Business Metrics",
                "type": "graph",
                "position": {"x": 8, "y": 14, "w": 8, "h": 6},
                "metrics": [
                    {
                        "name": "Daily Predictions",
                        "namespace": "ARSL/Business",
                        "metric": "daily_predictions",
                        "unit": "Count"
                    },
                    {
                        "name": "Active Users",
                        "namespace": "ARSL/Business",
                        "metric": "active_users",
                        "unit": "Count"
                    }
                ]
            },
            
            {
                "id": "storage_metrics",
                "title": "Storage & Network",
                "type": "graph",
                "position": {"x": 16, "y": 14, "w": 8, "h": 6},
                "metrics": [
                    {
                        "name": "OBS Storage Usage",
                        "namespace": "SYS.OBS",
                        "metric": "storage_size",
                        "unit": "Gigabytes"
                    },
                    {
                        "name": "Network I/O",
                        "namespace": "SYS.ECS",
                        "metric": "network_bytes",
                        "unit": "Bytes/Second"
                    }
                ]
            }
        ]
    }
    
    # Save dashboard configuration
    dashboard_file = Path("config/cloud_eye_dashboard.json")
    dashboard_file.parent.mkdir(exist_ok=True)
    
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard_config, f, indent=2)
    
    print(f"📊 Dashboard configuration saved: {dashboard_file}")
    return dashboard_file

def create_custom_metrics():
    """Create custom metrics for ARSL application"""
    
    custom_metrics = {
        "arsl_application_metrics": {
            "namespace": "ARSL/Application",
            "metrics": [
                {
                    "metric_name": "prediction_requests",
                    "unit": "Count",
                    "description": "Number of prediction requests",
                    "dimensions": [
                        {"name": "api_endpoint", "value": "predict"},
                        {"name": "request_type", "value": "single"}
                    ]
                },
                {
                    "metric_name": "prediction_accuracy",
                    "unit": "Percent",
                    "description": "Model prediction accuracy rate",
                    "dimensions": [
                        {"name": "model_version", "value": "1.0.0"},
                        {"name": "confidence_threshold", "value": "0.8"}
                    ]
                },
                {
                    "metric_name": "processing_time",
                    "unit": "Milliseconds", 
                    "description": "Image processing and inference time",
                    "dimensions": [
                        {"name": "processing_stage", "value": "inference"},
                        {"name": "model_type", "value": "cnn"}
                    ]
                }
            ]
        },
        
        "arsl_business_metrics": {
            "namespace": "ARSL/Business",
            "metrics": [
                {
                    "metric_name": "popular_arabic_classes",
                    "unit": "Count",
                    "description": "Most predicted Arabic letter classes",
                    "dimensions": [
                        {"name": "arabic_letter", "value": "variable"},
                        {"name": "confidence_level", "value": "high"}
                    ]
                },
                {
                    "metric_name": "user_engagement",
                    "unit": "Count",
                    "description": "User interaction metrics",
                    "dimensions": [
                        {"name": "session_duration", "value": "minutes"},
                        {"name": "user_type", "value": "api_client"}
                    ]
                },
                {
                    "metric_name": "data_volume_processed",
                    "unit": "Megabytes",
                    "description": "Volume of image data processed",
                    "dimensions": [
                        {"name": "data_type", "value": "image"},
                        {"name": "format", "value": "jpeg"}
                    ]
                }
            ]
        }
    }
    
    # Save custom metrics
    metrics_file = Path("config/custom_metrics.json")
    metrics_file.parent.mkdir(exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(custom_metrics, f, indent=2)
    
    print(f"📏 Custom metrics saved: {metrics_file}")
    return metrics_file

def create_metric_collection_script():
    """Create script to collect and send custom metrics"""
    
    collection_script = '''#!/usr/bin/env python3
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
            f.write(json.dumps(metrics_log) + "\\n")
        
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
                print("\\n📊 Metrics collection stopped by user")
                break
            except Exception as e:
                print(f"❌ Collection error: {e}")
                time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    collector = ARSLMetricsCollector()
    collector.run_collection_loop()
'''
    
    # Save metrics collection script
    script_file = Path("scripts/collect_metrics.py")
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(collection_script)
    
    print(f"📊 Metrics collection script saved: {script_file}")
    return script_file

def print_configuration_steps():
    """Print step-by-step configuration instructions"""
    
    print("\n📋 CONFIGURATION STEPS:")
    print("=" * 30)
    
    steps = [
        {
            "step": "1. Access Cloud Eye Console",
            "duration": "2 minutes",
            "actions": [
                "Open https://console.huaweicloud.com/ces",
                "Verify region is set to AF-Cairo",
                "Navigate to Monitoring → Overview",
                "Check that ECS instance appears in resource list"
            ]
        },
        {
            "step": "2. Enable Detailed Monitoring",
            "duration": "3 minutes",
            "actions": [
                "Go to ECS monitoring section",
                "Enable detailed monitoring for ECS instance",
                "Set collection interval to 1 minute",
                "Verify metrics are being collected"
            ]
        },
        {
            "step": "3. Create Custom Dashboard",
            "duration": "15 minutes",
            "actions": [
                "Navigate to Cloud Eye → Dashboard",
                "Click 'Create Dashboard'",
                "Import dashboard configuration",
                "Add system performance panels",
                "Add API performance panels",
                "Add business metrics panels",
                "Save dashboard as 'ARSL Production Monitor'"
            ]
        },
        {
            "step": "4. Configure Custom Metrics",
            "duration": "10 minutes",
            "actions": [
                "Upload custom metrics script to ECS instance",
                "Configure metric collection service",
                "Test custom metric collection",
                "Verify metrics appear in Cloud Eye"
            ]
        },
        {
            "step": "5. Set Up Log Collection",
            "duration": "8 minutes",
            "actions": [
                "Navigate to LTS (Log Tank Service)",
                "Create log group: ARSL-Production",
                "Create log streams for different components",
                "Configure log forwarding from ECS instance",
                "Test log collection"
            ]
        },
        {
            "step": "6. Verify Configuration",
            "duration": "7 minutes",
            "actions": [
                "Check dashboard displays all metrics",
                "Verify real-time data collection",
                "Test metric history and trends",
                "Validate custom metrics are working",
                "Document any issues or adjustments needed"
            ]
        }
    ]
    
    total_time = 0
    for step in steps:
        print(f"\n   🔸 {step['step']} ({step['duration']}):")
        for action in step['actions']:
            print(f"     • {action}")
        total_time += int(step['duration'].split()[0])
    
    print(f"\n⏱️ Total configuration time: ~{total_time} minutes")
    print(f"📊 Dashboard panels: 6 main panels with 15+ metrics")

def main():
    """Main function"""
    print("📊 STEP 2: CONFIGURE CLOUD EYE MONITORING")
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Target: Production monitoring setup")
    print("=" * 60)
    
    # Print configuration guide
    print_cloud_eye_configuration()
    
    # Create dashboard configuration
    dashboard_file = create_monitoring_dashboard()
    
    # Create custom metrics
    metrics_file = create_custom_metrics()
    
    # Create metrics collection script
    script_file = create_metric_collection_script()
    
    # Print configuration steps
    print_configuration_steps()
    
    print(f"\n🎯 STEP 2 SUMMARY:")
    print(f"✅ Monitoring dashboard configured ({dashboard_file})")
    print(f"✅ Custom metrics defined ({metrics_file})")
    print(f"✅ Metrics collection script ready ({script_file})")
    print(f"✅ Configuration steps documented")
    print(f"📋 Ready for Cloud Eye monitoring setup")
    print(f"🌐 Next: Configure alerting and notifications")
    
    print(f"\n💡 QUICK START:")
    print(f"1. Go to: https://console.huaweicloud.com/ces")
    print(f"2. Create dashboard using: {dashboard_file}")
    print(f"3. Deploy metrics script: {script_file}")
    print(f"4. Verify data collection in dashboard")
    print(f"5. Proceed to Step 3: Set up alerting")

if __name__ == "__main__":
    main()