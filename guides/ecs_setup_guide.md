
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
        return 200 "healthy\n";
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
