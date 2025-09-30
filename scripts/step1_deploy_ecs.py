"""
Step 1: ECS Instance Deployment Guide
Complete automation and step-by-step instructions for deploying production ECS instance
"""

import json
from pathlib import Path
from datetime import datetime

def print_ecs_deployment_guide():
    """Print comprehensive ECS deployment guide"""
    
    print("🚀 STEP 1: ECS INSTANCE DEPLOYMENT")
    print("=" * 50)
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Instance: arsl-compute-node")
    print()
    
    print("🌐 1. ACCESS ECS CONSOLE:")
    print("   URL: https://console.huaweicloud.com/ecs")
    print("   Login: yyacoup account")
    print("   Region: AF-Cairo (af-north-1)")
    print("   Navigate: Elastic Cloud Server → Instances")
    print()
    
    print("🔧 2. CREATE SECURITY GROUP FIRST:")
    print("   Security Group Name: arsl-security-group")
    print("   Description: Security group for ARSL production services")
    print()
    print("   Inbound Rules:")
    
    security_rules = [
        {"protocol": "SSH", "port": "22", "source": "Your IP only", "description": "SSH access"},
        {"protocol": "HTTP", "port": "80", "source": "0.0.0.0/0", "description": "Web traffic"},
        {"protocol": "HTTPS", "port": "443", "source": "0.0.0.0/0", "description": "Secure web traffic"},
        {"protocol": "Custom TCP", "port": "8000-8080", "source": "0.0.0.0/0", "description": "API services"},
        {"protocol": "Custom TCP", "port": "9090", "source": "10.0.0.0/8", "description": "Metrics endpoint"},
        {"protocol": "ICMP", "port": "All", "source": "0.0.0.0/0", "description": "Ping for monitoring"}
    ]
    
    for rule in security_rules:
        print(f"     • {rule['protocol']} ({rule['port']}): {rule['source']} - {rule['description']}")
    print()
    
    print("🔑 3. CREATE KEY PAIR:")
    print("   Key Pair Name: arsl-keypair")
    print("   Type: RSA")
    print("   Key Size: 2048")
    print("   ⚠️  Download .pem file immediately (cannot be re-downloaded)")
    print("   💾 Save to secure location: ./keys/arsl-keypair.pem")
    print("   🔐 Set permissions: chmod 400 arsl-keypair.pem")
    print()

def create_instance_config():
    """Create ECS instance configuration"""
    
    instance_config = {
        "instance_configuration": {
            "basic_config": {
                "name": "arsl-compute-node",
                "region": "af-north-1",
                "availability_zone": "af-north-1a",
                "image": "Ubuntu 20.04 LTS (64-bit)",
                "instance_type": "c6.xlarge",
                "specifications": {
                    "cpu": "4 vCPUs",
                    "memory": "8GB RAM",
                    "network": "Up to 25 Gbps",
                    "storage": "EBS-optimized"
                }
            },
            
            "storage_config": {
                "system_disk": {
                    "type": "General Purpose SSD (gp3)",
                    "size": "40GB",
                    "iops": "3000",
                    "throughput": "125 MB/s"
                },
                "data_disk": {
                    "type": "General Purpose SSD (gp3)",
                    "size": "100GB", 
                    "iops": "3000",
                    "throughput": "125 MB/s",
                    "mount_point": "/opt/arsl"
                }
            },
            
            "network_config": {
                "vpc": "Default VPC",
                "subnet": "Default subnet",
                "security_group": "arsl-security-group",
                "public_ip": "Auto-assign Elastic IP",
                "bandwidth": "5 Mbps"
            },
            
            "advanced_config": {
                "key_pair": "arsl-keypair",
                "user_data": "cloud-init script",
                "monitoring": "Detailed monitoring enabled",
                "termination_protection": False,
                "tags": {
                    "Project": "ARSL",
                    "Environment": "Production",
                    "Service": "Monitoring",
                    "Owner": "yyacoup"
                }
            }
        },
        
        "estimated_costs": {
            "instance": "$60/month",
            "storage": "$15/month",
            "bandwidth": "$10/month",
            "monitoring": "$5/month",
            "total": "$90/month"
        }
    }
    
    # Save configuration
    config_file = Path("config/ecs_instance_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(instance_config, f, indent=2)
    
    print(f"🔧 ECS configuration saved: {config_file}")
    return config_file

def create_cloud_init_script():
    """Create cloud-init user data script"""
    
    cloud_init = """#cloud-config

# ARSL Project - ECS Instance Initialization
# Automatically configures instance for production deployment

package_update: true
package_upgrade: true

packages:
  - curl
  - wget
  - git
  - htop
  - nano
  - unzip
  - python3
  - python3-pip
  - python3-venv
  - docker.io
  - docker-compose
  - nginx
  - certbot
  - python3-certbot-nginx
  - jq
  - awscli

users:
  - name: arsl
    groups: sudo, docker
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ... # Add your public key here

write_files:
  - path: /opt/arsl/config.json
    content: |
      {
        "project": "ARSL",
        "environment": "production",
        "api_endpoint": "https://arsl-api.apig.af-north-1.huaweicloudapis.com",
        "monitoring_interval": 30,
        "log_level": "INFO"
      }
    permissions: '0644'
    
  - path: /etc/nginx/sites-available/arsl-proxy
    content: |
      server {
          listen 80;
          server_name _;
          
          # Health check endpoint
          location /health {
              return 200 "healthy\\n";
              add_header Content-Type text/plain;
          }
          
          # Metrics endpoint
          location /metrics {
              proxy_pass http://localhost:9090/metrics;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
          }
          
          # API proxy
          location /api/ {
              proxy_pass https://arsl-api.apig.af-north-1.huaweicloudapis.com/;
              proxy_set_header Host $host;
              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
              proxy_set_header X-Forwarded-Proto $scheme;
              
              # CORS headers
              add_header Access-Control-Allow-Origin *;
              add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
              add_header Access-Control-Allow-Headers "Content-Type, Authorization";
          }
          
          # Static files for monitoring dashboard
          location /dashboard/ {
              alias /opt/arsl/dashboard/;
              index index.html;
          }
      }
    permissions: '0644'

  - path: /opt/arsl/monitor.py
    content: |
      #!/usr/bin/env python3
      import psutil
      import requests
      import time
      import json
      from datetime import datetime
      
      class ARSLMonitor:
          def __init__(self):
              self.api_base = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
              
          def collect_metrics(self):
              return {
                  "timestamp": datetime.now().isoformat(),
                  "cpu_percent": psutil.cpu_percent(interval=1),
                  "memory_percent": psutil.virtual_memory().percent,
                  "disk_percent": psutil.disk_usage('/').percent,
                  "load_avg": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
              }
          
          def test_api_health(self):
              try:
                  start = time.time()
                  resp = requests.get(f"{self.api_base}/v1/health", timeout=10)
                  return {
                      "status": "healthy" if resp.status_code == 200 else "unhealthy",
                      "response_time": (time.time() - start) * 1000,
                      "status_code": resp.status_code
                  }
              except Exception as e:
                  return {"status": "error", "error": str(e)}
          
          def run(self):
              while True:
                  try:
                      metrics = self.collect_metrics()
                      api_health = self.test_api_health()
                      
                      data = {"system": metrics, "api": api_health}
                      print(f"[{metrics['timestamp']}] CPU: {metrics['cpu_percent']:.1f}% | "
                            f"Memory: {metrics['memory_percent']:.1f}% | "
                            f"API: {api_health['status']}")
                      
                      with open("/opt/arsl/metrics.log", "a") as f:
                          f.write(json.dumps(data) + "\\n")
                      
                      time.sleep(30)
                  except Exception as e:
                      print(f"Error: {e}")
                      time.sleep(30)
      
      if __name__ == "__main__":
          ARSLMonitor().run()
    permissions: '0755'

  - path: /etc/systemd/system/arsl-monitor.service
    content: |
      [Unit]
      Description=ARSL System Monitor
      After=network.target
      
      [Service]
      Type=simple
      User=arsl
      WorkingDirectory=/opt/arsl
      ExecStart=/usr/bin/python3 /opt/arsl/monitor.py
      Restart=always
      RestartSec=10
      
      [Install]
      WantedBy=multi-user.target
    permissions: '0644'

runcmd:
  # Create directories
  - mkdir -p /opt/arsl/{logs,dashboard,backups}
  - chown -R arsl:arsl /opt/arsl
  
  # Configure Docker
  - systemctl start docker
  - systemctl enable docker
  - usermod -aG docker ubuntu
  - usermod -aG docker arsl
  
  # Configure Nginx
  - ln -s /etc/nginx/sites-available/arsl-proxy /etc/nginx/sites-enabled/
  - rm -f /etc/nginx/sites-enabled/default
  - nginx -t
  - systemctl restart nginx
  - systemctl enable nginx
  
  # Install Python packages
  - pip3 install requests psutil fastapi uvicorn prometheus-client schedule
  
  # Download and install Cloud Eye agent
  - cd /tmp
  - wget -q https://ces-agent.obs.af-north-1.myhuaweicloud.com/ces-agent.tar.gz
  - tar -xzf ces-agent.tar.gz
  - cd ces-agent
  - ./install.sh
  
  # Start monitoring service
  - systemctl daemon-reload
  - systemctl enable arsl-monitor
  - systemctl start arsl-monitor
  
  # Configure log rotation
  - echo "/opt/arsl/metrics.log { daily rotate 7 compress missingok notifempty }" > /etc/logrotate.d/arsl
  
  # Set up backup cron job
  - echo "0 2 * * * root tar -czf /opt/arsl/backups/daily_backup_$(date +\\%Y\\%m\\%d).tar.gz /opt/arsl/logs /var/log/nginx" | crontab -
  
  # Create status check script
  - echo '#!/bin/bash
    echo "=== ARSL System Status ==="
    echo "Date: $(date)"
    echo "Uptime: $(uptime)"
    echo "Disk Usage: $(df -h / | tail -1)"
    echo "Memory Usage: $(free -h)"
    echo "Docker Status: $(systemctl is-active docker)"
    echo "Nginx Status: $(systemctl is-active nginx)"
    echo "Monitor Status: $(systemctl is-active arsl-monitor)"
    echo "API Health: $(curl -s http://localhost/health)"
    ' > /opt/arsl/status.sh
  - chmod +x /opt/arsl/status.sh

final_message: |
  ARSL ECS Instance Setup Complete!
  
  Services Running:
  - Nginx proxy: http://$(curl -s http://169.254.169.254/meta-data/public-ipv4)
  - Health check: /health
  - API proxy: /api/
  - Metrics: /metrics
  - System monitoring: Active
  
  Next Steps:
  1. Configure SSL certificate (if domain available)
  2. Set up Cloud Eye monitoring
  3. Configure alerting rules
  4. Test end-to-end monitoring
"""
    
    # Save cloud-init script
    init_file = Path("scripts/ecs_cloud_init.yaml")
    with open(init_file, 'w') as f:
        f.write(cloud_init)
    
    print(f"☁️ Cloud-init script saved: {init_file}")
    return init_file

def print_deployment_steps():
    """Print step-by-step deployment instructions"""
    
    print("\n📋 DEPLOYMENT STEPS:")
    print("=" * 30)
    
    steps = [
        {
            "step": "1. Prepare Prerequisites",
            "duration": "5 minutes",
            "actions": [
                "Review ECS pricing and limits",
                "Prepare SSH key pair", 
                "Note down your public IP for SSH access",
                "Ensure adequate account balance"
            ]
        },
        {
            "step": "2. Create Security Group",
            "duration": "3 minutes",
            "actions": [
                "Navigate to VPC → Security Groups",
                "Create 'arsl-security-group'",
                "Add inbound rules (SSH, HTTP, HTTPS, custom ports)",
                "Verify outbound rules (all traffic allowed)"
            ]
        },
        {
            "step": "3. Create Key Pair",
            "duration": "2 minutes", 
            "actions": [
                "Navigate to ECS → Key Pairs",
                "Create 'arsl-keypair'",
                "Download .pem file immediately",
                "Set file permissions: chmod 400 arsl-keypair.pem"
            ]
        },
        {
            "step": "4. Launch ECS Instance",
            "duration": "10 minutes",
            "actions": [
                "Navigate to ECS → Instances",
                "Click 'Buy ECS'",
                "Select c6.xlarge in af-north-1a",
                "Configure storage (40GB system + 100GB data)",
                "Attach security group and key pair",
                "Add cloud-init user data script",
                "Add tags (Project=ARSL, Environment=Production)",
                "Review and launch"
            ]
        },
        {
            "step": "5. Wait for Initialization",
            "duration": "10 minutes",
            "actions": [
                "Monitor instance status (Pending → Running)",
                "Wait for cloud-init to complete",
                "Check system logs for any errors",
                "Verify all services are starting"
            ]
        },
        {
            "step": "6. Verify Deployment",
            "duration": "5 minutes",
            "actions": [
                "SSH to instance: ssh -i arsl-keypair.pem ubuntu@<ip>",
                "Check health: curl http://localhost/health",
                "Verify services: sudo systemctl status arsl-monitor",
                "Test API proxy: curl http://localhost/api/v1/health",
                "Review logs: sudo journalctl -u arsl-monitor"
            ]
        }
    ]
    
    total_time = 0
    for step in steps:
        print(f"\n   🔸 {step['step']} ({step['duration']}):")
        for action in step['actions']:
            print(f"     • {action}")
        total_time += int(step['duration'].split()[0])
    
    print(f"\n⏱️ Total deployment time: ~{total_time} minutes")
    print(f"💰 Monthly cost: ~$90 (instance + storage + bandwidth)")

def create_post_deployment_checklist():
    """Create post-deployment verification checklist"""
    
    checklist = {
        "immediate_checks": [
            {
                "check": "Instance Status",
                "command": "Check ECS console - status should be 'Running'",
                "expected": "Running status with green indicator"
            },
            {
                "check": "SSH Connectivity", 
                "command": "ssh -i arsl-keypair.pem ubuntu@<instance-ip>",
                "expected": "Successful SSH connection"
            },
            {
                "check": "System Health",
                "command": "curl http://<instance-ip>/health",
                "expected": "Response: 'healthy'"
            },
            {
                "check": "API Proxy",
                "command": "curl http://<instance-ip>/api/v1/health",
                "expected": "JSON response with API health status"
            }
        ],
        
        "service_checks": [
            {
                "service": "Docker",
                "command": "sudo systemctl status docker",
                "expected": "active (running)"
            },
            {
                "service": "Nginx",
                "command": "sudo systemctl status nginx", 
                "expected": "active (running)"
            },
            {
                "service": "ARSL Monitor",
                "command": "sudo systemctl status arsl-monitor",
                "expected": "active (running)"
            },
            {
                "service": "Cloud Eye Agent",
                "command": "sudo systemctl status ces-agent",
                "expected": "active (running)"
            }
        ],
        
        "performance_checks": [
            {
                "metric": "CPU Usage",
                "command": "top -n1 | head -3",
                "expected": "< 20% under normal load"
            },
            {
                "metric": "Memory Usage",
                "command": "free -h",
                "expected": "< 50% memory used"
            },
            {
                "metric": "Disk Space",
                "command": "df -h",
                "expected": "< 30% disk used"
            },
            {
                "metric": "Network",
                "command": "ping -c 3 google.com",
                "expected": "< 50ms response time"
            }
        ]
    }
    
    # Save checklist
    checklist_file = Path("config/post_deployment_checklist.json")
    checklist_file.parent.mkdir(exist_ok=True)
    
    with open(checklist_file, 'w') as f:
        json.dump(checklist, f, indent=2)
    
    print(f"✅ Post-deployment checklist saved: {checklist_file}")
    return checklist_file

def main():
    """Main function"""
    print("🚀 STEP 1: ECS INSTANCE DEPLOYMENT")
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Target: c6.xlarge production instance")
    print("=" * 60)
    
    # Print deployment guide
    print_ecs_deployment_guide()
    
    # Create instance configuration
    config_file = create_instance_config()
    
    # Create cloud-init script
    init_file = create_cloud_init_script()
    
    # Print deployment steps
    print_deployment_steps()
    
    # Create post-deployment checklist
    checklist_file = create_post_deployment_checklist()
    
    print(f"\n🎯 STEP 1 SUMMARY:")
    print(f"✅ ECS configuration prepared ({config_file})")
    print(f"✅ Cloud-init automation ready ({init_file})")
    print(f"✅ Deployment steps documented")
    print(f"✅ Post-deployment checklist created ({checklist_file})")
    print(f"📋 Ready for ECS instance deployment")
    print(f"🌐 Next: Deploy instance in ECS console")
    
    print(f"\n💡 QUICK START:")
    print(f"1. Go to: https://console.huaweicloud.com/ecs")
    print(f"2. Click 'Buy ECS' → Select c6.xlarge")
    print(f"3. Use cloud-init script: {init_file}")
    print(f"4. Wait 10-15 minutes for full setup")
    print(f"5. Verify with checklist: {checklist_file}")

if __name__ == "__main__":
    main()