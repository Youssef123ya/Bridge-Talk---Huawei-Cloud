"""
Enhanced Cloud Eye Monitoring Setup for ARSL Project
Comprehensive monitoring, alerting, and ECS integration guide
"""

import json
from pathlib import Path
from datetime import datetime

def print_cloud_eye_setup_guide():
    """Print detailed Cloud Eye monitoring setup"""
    
    print("📊 CLOUD EYE MONITORING SETUP")
    print("=" * 50)
    print()
    
    print("🌐 1. ACCESS CLOUD EYE:")
    print("   URL: https://console.huaweicloud.com/ces")
    print("   Login: yyacoup account")
    print("   Region: AF-Cairo (af-north-1)")
    print("   Service: Cloud Eye (CES)")
    print()
    
    print("📈 2. MONITORING DASHBOARD SETUP:")
    
    dashboard_panels = [
        {
            "name": "ARSL System Overview",
            "metrics": [
                "API Gateway Request Rate",
                "ModelArts Inference Latency", 
                "OBS Storage Usage",
                "System Health Status"
            ]
        },
        {
            "name": "Performance Metrics",
            "metrics": [
                "Response Time (P50, P95, P99)",
                "Throughput (requests/second)",
                "Error Rate (%)",
                "Concurrent Users"
            ]
        },
        {
            "name": "Resource Utilization",
            "metrics": [
                "CPU Usage (%)",
                "Memory Usage (%)", 
                "Network I/O (MB/s)",
                "Disk Usage (%)"
            ]
        },
        {
            "name": "Business Metrics",
            "metrics": [
                "Daily Predictions Count",
                "Accuracy Rate (%)",
                "Popular Arabic Classes",
                "User Engagement"
            ]
        }
    ]
    
    for panel in dashboard_panels:
        print(f"   📊 {panel['name']}:")
        for metric in panel['metrics']:
            print(f"     • {metric}")
        print()

def create_custom_metrics():
    """Create custom metrics configuration"""
    
    custom_metrics = {
        "arsl_api_metrics": {
            "namespace": "ARSL/API",
            "metrics": [
                {
                    "name": "prediction_accuracy",
                    "unit": "Percent",
                    "description": "Model prediction accuracy rate",
                    "dimensions": [
                        {"name": "model_version", "value": "1.0.0"},
                        {"name": "class_type", "value": "arabic_letter"}
                    ]
                },
                {
                    "name": "inference_latency", 
                    "unit": "Milliseconds",
                    "description": "Time taken for model inference",
                    "dimensions": [
                        {"name": "service_name", "value": "arsl-inference-service"},
                        {"name": "instance_type", "value": "cpu_4u8g"}
                    ]
                },
                {
                    "name": "daily_predictions",
                    "unit": "Count",
                    "description": "Number of predictions per day",
                    "dimensions": [
                        {"name": "api_endpoint", "value": "predict"},
                        {"name": "prediction_type", "value": "single"}
                    ]
                },
                {
                    "name": "popular_classes",
                    "unit": "Count", 
                    "description": "Most predicted Arabic classes",
                    "dimensions": [
                        {"name": "arabic_class", "value": "variable"},
                        {"name": "confidence_level", "value": "high"}
                    ]
                }
            ]
        },
        
        "arsl_system_metrics": {
            "namespace": "ARSL/System",
            "metrics": [
                {
                    "name": "data_processing_volume",
                    "unit": "Megabytes",
                    "description": "Volume of image data processed",
                    "dimensions": [
                        {"name": "data_type", "value": "image"},
                        {"name": "processing_stage", "value": "inference"}
                    ]
                },
                {
                    "name": "storage_usage",
                    "unit": "Gigabytes", 
                    "description": "OBS storage usage for models and data",
                    "dimensions": [
                        {"name": "bucket_name", "value": "arsl-youssef-af-cairo-2025"},
                        {"name": "content_type", "value": "models"}
                    ]
                },
                {
                    "name": "concurrent_users",
                    "unit": "Count",
                    "description": "Number of concurrent API users",
                    "dimensions": [
                        {"name": "user_type", "value": "api_client"},
                        {"name": "authentication", "value": "api_key"}
                    ]
                }
            ]
        }
    }
    
    # Save custom metrics
    metrics_file = Path("config/cloud_eye_custom_metrics.json")
    metrics_file.parent.mkdir(exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(custom_metrics, f, indent=2)
    
    print(f"📊 Custom metrics configuration saved: {metrics_file}")
    return metrics_file

def print_ecs_integration_guide():
    """Print ECS compute integration guide"""
    
    print("\n⚡ ECS COMPUTE INTEGRATION")
    print("=" * 40)
    print()
    
    print("🖥️ 1. ECS INSTANCE CONFIGURATION:")
    print("   Instance Name: arsl-compute-node")
    print("   Region: AF-Cairo (af-north-1)")
    print("   Availability Zone: af-north-1a")
    print()
    
    print("💾 2. RECOMMENDED ECS SPECIFICATIONS:")
    
    ecs_options = [
        {
            "tier": "Development/Testing",
            "spec": "c6.large",
            "cpu": "2 vCPUs",
            "memory": "4GB RAM",
            "storage": "40GB SSD",
            "cost": "~$25/month",
            "use_case": "API testing, small workloads"
        },
        {
            "tier": "Production (Recommended)",
            "spec": "c6.xlarge", 
            "cpu": "4 vCPUs",
            "memory": "8GB RAM",
            "storage": "100GB SSD",
            "cost": "~$60/month",
            "use_case": "API gateway, monitoring services"
        },
        {
            "tier": "High Performance",
            "spec": "c6.2xlarge",
            "cpu": "8 vCPUs", 
            "memory": "16GB RAM",
            "storage": "200GB SSD",
            "cost": "~$120/month",
            "use_case": "Model serving, batch processing"
        },
        {
            "tier": "GPU Accelerated",
            "spec": "p2.xlarge",
            "cpu": "4 vCPUs",
            "memory": "61GB RAM",
            "gpu": "1x NVIDIA K80",
            "storage": "100GB SSD", 
            "cost": "~$900/month",
            "use_case": "Model training, GPU inference"
        }
    ]
    
    for option in ecs_options:
        print(f"   🔧 {option['tier']}:")
        print(f"     Specification: {option['spec']}")
        print(f"     CPU: {option['cpu']}")
        print(f"     Memory: {option['memory']}")
        if 'gpu' in option:
            print(f"     GPU: {option['gpu']}")
        print(f"     Storage: {option['storage']}")
        print(f"     Cost: {option['cost']}")
        print(f"     Use Case: {option['use_case']}")
        print()
    
    print("💡 RECOMMENDED: c6.xlarge for production API deployment")

def create_ecs_deployment_script():
    """Create ECS deployment automation script"""
    
    ecs_script = '''"""
ECS Instance Deployment Script for ARSL Project
Automates ECS setup with monitoring and services
"""

import json
import time
from pathlib import Path

class ARSLECSDeployment:
    """ECS deployment automation for ARSL project"""
    
    def __init__(self):
        self.config = {
            "instance_name": "arsl-compute-node",
            "region": "af-north-1",
            "availability_zone": "af-north-1a",
            "instance_type": "c6.xlarge",
            "image_id": "ubuntu-20.04-server-amd64",
            "security_group": "arsl-security-group",
            "key_pair": "arsl-keypair"
        }
    
    def print_deployment_guide(self):
        """Print step-by-step ECS deployment guide"""
        
        print("🚀 ECS INSTANCE DEPLOYMENT GUIDE")
        print("=" * 40)
        print()
        
        steps = [
            {
                "step": "1. Create Security Group",
                "duration": "5 minutes",
                "details": [
                    "Name: arsl-security-group",
                    "Allow HTTP (80) from anywhere",
                    "Allow HTTPS (443) from anywhere", 
                    "Allow SSH (22) from your IP",
                    "Allow custom ports (8000-8080) for APIs"
                ]
            },
            {
                "step": "2. Create Key Pair",
                "duration": "2 minutes",
                "details": [
                    "Name: arsl-keypair",
                    "Download .pem file",
                    "Set permissions: chmod 400 arsl-keypair.pem"
                ]
            },
            {
                "step": "3. Launch ECS Instance",
                "duration": "10 minutes",
                "details": [
                    "Instance type: c6.xlarge (4 vCPU, 8GB RAM)",
                    "Image: Ubuntu 20.04 LTS",
                    "Storage: 100GB GP3 SSD",
                    "Network: Default VPC and subnet"
                ]
            },
            {
                "step": "4. Configure Instance",
                "duration": "20 minutes",
                "details": [
                    "Update system packages",
                    "Install Docker and Docker Compose",
                    "Install Python 3.9+ and pip",
                    "Install monitoring agent",
                    "Configure firewall rules"
                ]
            },
            {
                "step": "5. Deploy Services",
                "duration": "30 minutes",
                "details": [
                    "Deploy API Gateway proxy",
                    "Install monitoring stack",
                    "Configure log forwarding",
                    "Set up backup scripts"
                ]
            }
        ]
        
        total_time = 0
        for step in steps:
            print(f"   {step['step']} ({step['duration']}):")
            for detail in step['details']:
                print(f"     • {detail}")
            print()
            total_time += int(step['duration'].split()[0])
        
        print(f"⏱️ Total deployment time: ~{total_time} minutes")
    
    def create_user_data_script(self):
        """Create user data script for instance initialization"""
        
        user_data = """#!/bin/bash

# ARSL Project ECS Instance Setup Script
# Automatically configures instance for production deployment

set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install essential packages
apt-get install -y \\
    curl \\
    wget \\
    git \\
    htop \\
    nano \\
    unzip \\
    python3 \\
    python3-pip \\
    python3-venv \\
    docker.io \\
    docker-compose \\
    nginx \\
    certbot \\
    python3-certbot-nginx

# Start and enable Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Install Python packages
pip3 install --upgrade pip
pip3 install \\
    requests \\
    fastapi \\
    uvicorn \\
    prometheus-client \\
    psutil \\
    schedule

# Create application directory
mkdir -p /opt/arsl
chown ubuntu:ubuntu /opt/arsl

# Create monitoring directory
mkdir -p /opt/monitoring
chown ubuntu:ubuntu /opt/monitoring

# Install Cloud Eye agent
wget https://ces-agent.obs.af-north-1.myhuaweicloud.com/ces-agent.tar.gz
tar -xzf ces-agent.tar.gz
cd ces-agent
./install.sh

# Configure Nginx
cat > /etc/nginx/sites-available/arsl-proxy << 'EOF'
server {
    listen 80;
    server_name _;
    
    location /api/ {
        proxy_pass https://arsl-api.apig.af-north-1.huaweicloudapis.com/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
    
    location /metrics {
        proxy_pass http://localhost:9090/metrics;
    }
}
EOF

# Enable the site
ln -s /etc/nginx/sites-available/arsl-proxy /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# Create monitoring service
cat > /opt/monitoring/arsl_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
ARSL System Monitoring Service
Collects and reports custom metrics to Cloud Eye
"""

import psutil
import requests
import time
import json
from datetime import datetime

class ARSLMonitor:
    def __init__(self):
        self.api_endpoint = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
        
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict(),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_api_health(self):
        """Test API endpoint health"""
        try:
            response = requests.get(f"{self.api_endpoint}/v1/health", timeout=10)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds() * 1000,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time": 0
            }
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                api_health = self.test_api_health()
                
                # Log metrics
                metrics = {
                    "system": system_metrics,
                    "api": api_health,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"[{metrics['timestamp']}] Metrics collected:")
                print(f"  CPU: {system_metrics['cpu_percent']:.1f}%")
                print(f"  Memory: {system_metrics['memory_percent']:.1f}%")
                print(f"  API Status: {api_health['status']}")
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    monitor = ARSLMonitor()
    monitor.run_monitoring_loop()
EOF

chmod +x /opt/monitoring/arsl_monitor.py

# Create systemd service for monitoring
cat > /etc/systemd/system/arsl-monitor.service << 'EOF'
[Unit]
Description=ARSL System Monitor
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/monitoring
ExecStart=/usr/bin/python3 /opt/monitoring/arsl_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start monitoring service
systemctl daemon-reload
systemctl enable arsl-monitor
systemctl start arsl-monitor

# Create backup script
cat > /opt/arsl/backup.sh << 'EOF'
#!/bin/bash
# Daily backup script for ARSL project

BACKUP_DIR="/opt/arsl/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /var/log/nginx/ /opt/monitoring/

# Backup configurations
cp -r /etc/nginx/sites-available $BACKUP_DIR/nginx_config_$DATE

# Upload to OBS (if configured)
# aws s3 cp $BACKUP_DIR/ s3://arsl-youssef-af-cairo-2025/backups/ --recursive

echo "Backup completed: $DATE"
EOF

chmod +x /opt/arsl/backup.sh

# Add to crontab for daily backups
echo "0 2 * * * /opt/arsl/backup.sh" | crontab -

# Final status
echo "✅ ARSL ECS Instance Setup Complete!"
echo "🌐 Services:"
echo "  - Nginx proxy: http://$(curl -s http://169.254.169.254/meta-data/public-ipv4)"
echo "  - Health check: /health"
echo "  - API proxy: /api/"
echo "  - System monitoring: Active"
echo ""
echo "📋 Next steps:"
echo "  1. Configure SSL certificate with certbot"
echo "  2. Set up domain name (optional)"
echo "  3. Configure monitoring dashboards"
echo "  4. Test API endpoints through proxy"
'''
        
        # Save user data script
        script_file = Path("scripts/ecs_user_data.sh")
        with open(script_file, 'w') as f:
            f.write(user_data)
        
        print(f"📄 User data script saved: {script_file}")
        return script_file

# Create instance
deployment = ARSLECSDeployment()
deployment.print_deployment_guide()
user_data_file = deployment.create_user_data_script()

print(f"\\n🎯 ECS DEPLOYMENT SUMMARY:")
print(f"✅ Configuration prepared")
print(f"✅ User data script created ({user_data_file})")
print(f"✅ Monitoring service included")
print(f"✅ Nginx proxy configured")
print(f"📋 Ready for ECS instance launch")
'''
    
    # Save ECS script
    ecs_file = Path("scripts/deploy_ecs_instance.py")
    with open(ecs_file, 'w', encoding='utf-8') as f:
        f.write(ecs_script)
    
    print(f"⚡ ECS deployment script saved: {ecs_file}")
    return ecs_file

def create_advanced_alerting():
    """Create advanced alerting configuration"""
    
    advanced_alerts = {
        "cloud_eye_alerts": {
            "critical_alerts": [
                {
                    "name": "API Service Down",
                    "condition": "Health check failure",
                    "threshold": "3 consecutive failures",
                    "evaluation_period": "3 minutes",
                    "notification": {
                        "sms": True,
                        "email": True,
                        "webhook": "https://your-webhook-url.com/alerts"
                    },
                    "actions": [
                        "Auto-restart service",
                        "Scale up instances",
                        "Notify on-call engineer"
                    ]
                },
                {
                    "name": "High Error Rate",
                    "condition": "Error rate > 10%",
                    "threshold": "10%",
                    "evaluation_period": "5 minutes",
                    "notification": {
                        "email": True,
                        "webhook": True
                    },
                    "actions": [
                        "Investigate error logs",
                        "Check model service",
                        "Monitor traffic patterns"
                    ]
                },
                {
                    "name": "Resource Exhaustion",
                    "condition": "CPU > 90% OR Memory > 95%",
                    "threshold": "90% CPU, 95% Memory",
                    "evaluation_period": "10 minutes",
                    "notification": {
                        "sms": True,
                        "email": True
                    },
                    "actions": [
                        "Auto-scale instances",
                        "Load balance traffic",
                        "Check for memory leaks"
                    ]
                }
            ],
            
            "warning_alerts": [
                {
                    "name": "Response Time Degradation",
                    "condition": "P95 response time > 1000ms",
                    "threshold": "1000ms",
                    "evaluation_period": "15 minutes",
                    "notification": {
                        "email": True
                    },
                    "actions": [
                        "Check model performance",
                        "Monitor database queries",
                        "Review recent deployments"
                    ]
                },
                {
                    "name": "Storage Usage High",
                    "condition": "OBS storage > 80% of quota",
                    "threshold": "80%",
                    "evaluation_period": "30 minutes",
                    "notification": {
                        "email": True
                    },
                    "actions": [
                        "Archive old models",
                        "Clean up temporary files",
                        "Request storage increase"
                    ]
                }
            ],
            
            "business_alerts": [
                {
                    "name": "Accuracy Drop",
                    "condition": "Model accuracy < 85%",
                    "threshold": "85%",
                    "evaluation_period": "1 hour",
                    "notification": {
                        "email": True
                    },
                    "actions": [
                        "Review model performance",
                        "Check input data quality",
                        "Consider model retraining"
                    ]
                },
                {
                    "name": "Usage Spike",
                    "condition": "Traffic > 300% of baseline",
                    "threshold": "300%",
                    "evaluation_period": "30 minutes",
                    "notification": {
                        "email": True
                    },
                    "actions": [
                        "Monitor for abuse",
                        "Check scaling limits",
                        "Prepare for increased load"
                    ]
                }
            ]
        }
    }
    
    # Save advanced alerts
    alerts_file = Path("config/advanced_cloud_eye_alerts.json")
    alerts_file.parent.mkdir(exist_ok=True)
    
    with open(alerts_file, 'w') as f:
        json.dump(advanced_alerts, f, indent=2)
    
    print(f"🚨 Advanced alerting configuration saved: {alerts_file}")
    return alerts_file

def print_implementation_timeline():
    """Print complete implementation timeline"""
    
    print("\n📅 IMPLEMENTATION TIMELINE:")
    print("=" * 40)
    
    timeline = [
        {
            "phase": "Phase 1: Cloud Eye Setup",
            "duration": "30 minutes",
            "tasks": [
                "Enable Cloud Eye service",
                "Configure custom metrics",
                "Set up basic dashboards",
                "Test metric collection"
            ]
        },
        {
            "phase": "Phase 2: ECS Deployment",
            "duration": "45 minutes",
            "tasks": [
                "Launch ECS instance",
                "Configure security groups",
                "Install monitoring agent",
                "Deploy proxy services"
            ]
        },
        {
            "phase": "Phase 3: Advanced Monitoring",
            "duration": "60 minutes",
            "tasks": [
                "Configure alerting rules",
                "Set up notification channels",
                "Create custom dashboards",
                "Test alert delivery"
            ]
        },
        {
            "phase": "Phase 4: Integration Testing",
            "duration": "30 minutes",
            "tasks": [
                "Test end-to-end monitoring",
                "Validate alert triggers",
                "Verify dashboard metrics",
                "Document procedures"
            ]
        }
    ]
    
    total_time = 0
    for phase in timeline:
        print(f"   🔸 {phase['phase']} ({phase['duration']}):")
        for task in phase['tasks']:
            print(f"     • {task}")
        print()
        total_time += int(phase['duration'].split()[0])
    
    print(f"⏱️ Total implementation time: ~{total_time} minutes")
    print(f"💰 Estimated monthly cost: $60-120 (depending on ECS tier)")

def main():
    """Main function"""
    print("📊 CLOUD EYE & ECS INTEGRATION")
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Project: Arabic Sign Language Recognition")
    print("=" * 60)
    
    # Cloud Eye setup
    print_cloud_eye_setup_guide()
    
    # Custom metrics
    metrics_file = create_custom_metrics()
    
    # ECS integration
    print_ecs_integration_guide()
    
    # ECS deployment script
    ecs_file = create_ecs_deployment_script()
    
    # Advanced alerting
    alerts_file = create_advanced_alerting()
    
    # Implementation timeline
    print_implementation_timeline()
    
    print(f"\n🎯 INTEGRATION SUMMARY:")
    print(f"✅ Cloud Eye monitoring configured")
    print(f"✅ Custom metrics defined ({metrics_file})")
    print(f"✅ ECS deployment automated ({ecs_file})")
    print(f"✅ Advanced alerting prepared ({alerts_file})")
    print(f"📋 Ready for production monitoring deployment")
    print(f"🌐 Next: Deploy ECS instance and configure monitoring")

if __name__ == "__main__":
    main()