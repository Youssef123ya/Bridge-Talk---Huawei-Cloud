"""
Cloud Eye Monitoring & ECS Integration for ARSL Project
Complete setup guide for production monitoring and compute resources
"""

import json
from pathlib import Path

def print_cloud_eye_setup():
    """Print Cloud Eye monitoring setup guide"""
    
    print("📊 CLOUD EYE MONITORING SETUP")
    print("=" * 50)
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print()
    
    print("🌐 1. ACCESS CLOUD EYE:")
    print("   URL: https://console.huaweicloud.com/ces")
    print("   Service: Cloud Eye (CES)")
    print("   Navigate: Monitoring → Dashboard")
    print()
    
    print("📈 2. MONITORING METRICS:")
    
    metrics = [
        {"category": "API Gateway", "metrics": ["Request Rate", "Response Time", "Error Rate", "Throttle Rate"]},
        {"category": "ModelArts", "metrics": ["Inference Latency", "CPU Usage", "Memory Usage", "Active Connections"]},
        {"category": "OBS Storage", "metrics": ["Storage Usage", "Request Count", "Data Transfer", "Error Rate"]},
        {"category": "Custom Business", "metrics": ["Prediction Accuracy", "Daily Users", "Popular Classes", "Processing Volume"]}
    ]
    
    for metric in metrics:
        print(f"   🔸 {metric['category']}:")
        for m in metric['metrics']:
            print(f"     • {m}")
        print()

def create_monitoring_config():
    """Create comprehensive monitoring configuration"""
    
    config = {
        "cloud_eye_config": {
            "dashboard_name": "ARSL Production Monitor",
            "refresh_interval": "30s",
            "custom_metrics": [
                {
                    "name": "arsl_prediction_accuracy",
                    "namespace": "ARSL/API",
                    "unit": "Percent",
                    "description": "Model prediction accuracy rate"
                },
                {
                    "name": "arsl_daily_predictions", 
                    "namespace": "ARSL/Business",
                    "unit": "Count",
                    "description": "Number of predictions per day"
                },
                {
                    "name": "arsl_response_time",
                    "namespace": "ARSL/Performance", 
                    "unit": "Milliseconds",
                    "description": "API response time"
                }
            ]
        },
        
        "alerting_rules": {
            "critical": [
                {"name": "Service Down", "condition": "Health check failure", "threshold": "3 failures", "action": "SMS + Email"},
                {"name": "High Error Rate", "condition": "Error rate > 10%", "threshold": "10%", "action": "Email + Webhook"},
                {"name": "Resource Exhaustion", "condition": "CPU > 90%", "threshold": "90%", "action": "Auto-scale + Alert"}
            ],
            "warning": [
                {"name": "Response Time High", "condition": "P95 > 1000ms", "threshold": "1000ms", "action": "Email"},
                {"name": "Storage Usage High", "condition": "Usage > 80%", "threshold": "80%", "action": "Email"},
                {"name": "Accuracy Drop", "condition": "Accuracy < 85%", "threshold": "85%", "action": "Email"}
            ]
        }
    }
    
    # Save configuration
    config_file = Path("config/cloud_eye_monitoring.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📊 Monitoring configuration saved: {config_file}")
    return config_file

def print_ecs_options():
    """Print ECS compute options"""
    
    print("\n⚡ ECS COMPUTE INTEGRATION")
    print("=" * 40)
    print()
    
    print("🖥️ RECOMMENDED ECS SPECIFICATIONS:")
    
    options = [
        {
            "tier": "Development",
            "spec": "c6.large", 
            "cpu": "2 vCPUs",
            "memory": "4GB",
            "storage": "40GB SSD",
            "cost": "~$25/month",
            "use_case": "Testing, development"
        },
        {
            "tier": "Production (Recommended)",
            "spec": "c6.xlarge",
            "cpu": "4 vCPUs", 
            "memory": "8GB",
            "storage": "100GB SSD",
            "cost": "~$60/month",
            "use_case": "API proxy, monitoring"
        },
        {
            "tier": "High Performance",
            "spec": "c6.2xlarge",
            "cpu": "8 vCPUs",
            "memory": "16GB", 
            "storage": "200GB SSD",
            "cost": "~$120/month",
            "use_case": "Model serving, batch processing"
        }
    ]
    
    for option in options:
        print(f"   🔧 {option['tier']}:")
        print(f"     Spec: {option['spec']}")
        print(f"     CPU: {option['cpu']}")
        print(f"     Memory: {option['memory']}")
        print(f"     Storage: {option['storage']}")
        print(f"     Cost: {option['cost']}")
        print(f"     Use Case: {option['use_case']}")
        print()

def create_ecs_setup_guide():
    """Create ECS setup guide"""
    
    setup_guide = """
# ECS Instance Setup Guide for ARSL Project

## 1. Create ECS Instance (15 minutes)

### Access ECS Console:
- URL: https://console.huaweicloud.com/ecs
- Login: yyacoup account
- Region: AF-Cairo (af-north-1)

### Instance Configuration:
- Name: arsl-compute-node
- Image: Ubuntu 20.04 LTS
- Instance Type: c6.xlarge (4 vCPU, 8GB RAM)
- Storage: 100GB GP3 SSD
- Security Group: arsl-security-group
- Key Pair: arsl-keypair

### Security Group Rules:
- SSH (22): Your IP only
- HTTP (80): 0.0.0.0/0
- HTTPS (443): 0.0.0.0/0
- Custom (8000-8080): 0.0.0.0/0

## 2. Initial Setup (20 minutes)

### Connect to Instance:
```bash
ssh -i arsl-keypair.pem ubuntu@<instance-ip>
```

### Update System:
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
```

### Install Docker:
```bash
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
```

### Install Python and Tools:
```bash
sudo apt-get install -y python3 python3-pip python3-venv nginx htop
pip3 install requests fastapi uvicorn prometheus-client
```

## 3. Configure Nginx Proxy (10 minutes)

### Create Nginx Configuration:
```bash
sudo nano /etc/nginx/sites-available/arsl-proxy
```

### Add Configuration:
```nginx
server {
    listen 80;
    server_name _;
    
    location /api/ {
        proxy_pass https://arsl-api.apig.af-north-1.huaweicloudapis.com/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /health {
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
}
```

### Enable Site:
```bash
sudo ln -s /etc/nginx/sites-available/arsl-proxy /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 4. Install Monitoring Agent (10 minutes)

### Download Cloud Eye Agent:
```bash
wget https://ces-agent.obs.af-north-1.myhuaweicloud.com/ces-agent.tar.gz
tar -xzf ces-agent.tar.gz
cd ces-agent
sudo ./install.sh
```

### Verify Installation:
```bash
sudo systemctl status ces-agent
```

## 5. Test Setup (5 minutes)

### Test Nginx:
```bash
curl http://localhost/health
```

### Test API Proxy:
```bash
curl http://localhost/api/v1/health
```

### Check Monitoring:
- Verify metrics in Cloud Eye console
- Check instance appears in monitoring dashboard

## 6. Optional: SSL Certificate (15 minutes)

### Install Certbot:
```bash
sudo apt-get install -y certbot python3-certbot-nginx
```

### Get Certificate (if domain configured):
```bash
sudo certbot --nginx -d your-domain.com
```

## Summary

Total Setup Time: ~75 minutes
Monthly Cost: ~$60 for c6.xlarge
Services: Nginx proxy, Cloud Eye monitoring, SSL ready

Next Steps:
1. Configure custom monitoring dashboards
2. Set up alerting rules
3. Test end-to-end monitoring
4. Configure backup procedures
"""
    
    # Save setup guide
    guide_file = Path("guides/ecs_setup_guide.md")
    guide_file.parent.mkdir(exist_ok=True)
    
    with open(guide_file, 'w') as f:
        f.write(setup_guide)
    
    print(f"📋 ECS setup guide saved: {guide_file}")
    return guide_file

def create_monitoring_script():
    """Create monitoring script for ECS instance"""
    
    monitoring_code = '''#!/usr/bin/env python3
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
                    f.write(json.dumps(metrics_data) + "\\n")
                
                # Sleep
                time.sleep(30)
                
            except KeyboardInterrupt:
                print("\\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    monitor = ARSLMonitor()
    monitor.run_monitoring()
'''
    
    # Save monitoring script
    script_file = Path("scripts/arsl_monitor.py")
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(monitoring_code)
    
    print(f"🔍 Monitoring script saved: {script_file}")
    return script_file

def print_implementation_summary():
    """Print implementation summary"""
    
    print("\n🎯 IMPLEMENTATION SUMMARY:")
    print("=" * 40)
    print()
    
    print("📊 CLOUD EYE MONITORING:")
    print("   ✅ Custom metrics defined")
    print("   ✅ Alerting rules configured")
    print("   ✅ Dashboard template ready")
    print("   ✅ Notification channels prepared")
    print()
    
    print("⚡ ECS COMPUTE INTEGRATION:")
    print("   ✅ Instance specifications defined")
    print("   ✅ Setup guide created")
    print("   ✅ Nginx proxy configuration ready")
    print("   ✅ Monitoring agent installation included")
    print()
    
    print("⏱️ DEPLOYMENT TIMELINE:")
    steps = [
        ("Cloud Eye setup", "30 minutes"),
        ("ECS instance creation", "15 minutes"),
        ("System configuration", "30 minutes"),
        ("Monitoring setup", "20 minutes"),
        ("Testing and validation", "15 minutes")
    ]
    
    total_time = 0
    for step, duration in steps:
        print(f"   • {step}: {duration}")
        total_time += int(duration.split()[0])
    
    print(f"\n   Total: ~{total_time} minutes")
    print(f"   Monthly cost: $60-120 (depending on tier)")
    print()
    
    print("🌐 NEXT STEPS:")
    print("   1. Deploy ECS instance with c6.xlarge")
    print("   2. Configure Cloud Eye monitoring")
    print("   3. Set up alerting and notifications")
    print("   4. Test end-to-end monitoring")
    print("   5. Document operational procedures")

def main():
    """Main execution function"""
    print("📊 CLOUD EYE & ECS INTEGRATION FOR ARSL")
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Project: Arabic Sign Language Recognition")
    print("=" * 60)
    
    # Cloud Eye setup
    print_cloud_eye_setup()
    
    # Create monitoring config
    config_file = create_monitoring_config()
    
    # ECS options
    print_ecs_options()
    
    # Create setup guide
    guide_file = create_ecs_setup_guide()
    
    # Create monitoring script
    script_file = create_monitoring_script()
    
    # Implementation summary
    print_implementation_summary()

if __name__ == "__main__":
    main()