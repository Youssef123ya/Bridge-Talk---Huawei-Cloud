# 🌟 Huawei Cloud Integration Guide
## Arabic Sign Language Recognition Project

This guide provides comprehensive instructions for deploying your Arabic Sign Language Recognition project to Huawei Cloud services.

## 🚀 Overview

Your ARSL project is now integrated with the following Huawei Cloud services:

- **🗄️ Object Storage Service (OBS)** - Store datasets and model artifacts
- **🤖 ModelArts** - Train and deploy machine learning models  
- **🌐 API Gateway** - Expose REST API endpoints
- **📊 Cloud Eye** - Monitor performance and set up alerts
- **⚡ Elastic Cloud Server (ECS)** - Scalable compute resources

## 📋 Prerequisites

### 1. Huawei Cloud Account Setup
1. Create a Huawei Cloud account at [https://www.huaweicloud.com](https://www.huaweicloud.com)
2. Complete identity verification
3. Enable the following services:
   - Object Storage Service (OBS)
   - ModelArts
   - API Gateway (APIG)
   - Cloud Eye (CES)

### 2. Access Credentials
Create access credentials and set environment variables:

```powershell
# Set these environment variables in PowerShell
$env:HUAWEI_ACCESS_KEY_ID = "your_access_key_id"
$env:HUAWEI_SECRET_ACCESS_KEY = "your_secret_access_key"  
$env:HUAWEI_PROJECT_ID = "your_project_id"
```

Or create a `.env` file in your project root:
```
HUAWEI_ACCESS_KEY_ID=your_access_key_id
HUAWEI_SECRET_ACCESS_KEY=your_secret_access_key
HUAWEI_PROJECT_ID=your_project_id
```

### 3. Install Dependencies
```powershell
# Install Huawei Cloud SDKs
pip install huaweicloudsdkcore huaweicloudsdkobs huaweicloudsdkmodelarts huaweicloudsdkapig huaweicloudsdkces esdk-obs-python
```

## 🛠️ Configuration

### Main Configuration File
The project uses `config/huawei_cloud_config.yaml` for all cloud settings:

```yaml
# Authentication Configuration
auth:
  access_key_id: "${HUAWEI_ACCESS_KEY_ID}"
  secret_access_key: "${HUAWEI_SECRET_ACCESS_KEY}"
  region: "ap-southeast-1"  # Change to your preferred region
  project_id: "${HUAWEI_PROJECT_ID}"

# Object Storage Configuration
obs:
  bucket_name: "arsl-recognition-data"
  
# ModelArts Configuration  
modelarts:
  training:
    compute_type: "modelarts.vm.gpu.p4"
    framework: "PyTorch-1.8.0-python3.7-cuda10.2-ubuntu18.04"
```

### Supported Regions
- `ap-southeast-1` (Singapore)
- `cn-north-1` (Beijing)
- `cn-east-2` (Shanghai)
- `eu-west-0` (Paris)

## 🚀 Quick Start

### Option 1: Complete Automated Setup
```powershell
# Run complete setup with dataset upload and training
python scripts\setup_huawei_cloud.py --upload-data --start-training
```

### Option 2: Step-by-Step Setup
```powershell
# 1. Setup storage only
python scripts\setup_huawei_cloud.py --component storage --upload-data

# 2. Setup training
python scripts\setup_huawei_cloud.py --component training --start-training

# 3. Setup deployment (after model is trained)
python scripts\setup_huawei_cloud.py --component deployment --service-url "your_modelarts_url"

# 4. Setup monitoring
python scripts\setup_huawei_cloud.py --component monitoring
```

## 📁 Project Structure

New cloud-related files added:
```
├── config/
│   └── huawei_cloud_config.yaml    # Cloud configuration
├── src/
│   └── cloud/
│       ├── huawei_storage.py       # OBS integration
│       ├── huawei_modelarts.py     # ModelArts integration
│       ├── train_arsl.py           # Cloud-optimized training
│       ├── inference_service.py    # Inference service
│       └── api_deployment.py       # API Gateway setup
├── scripts/
│   └── setup_huawei_cloud.py       # Complete setup script
└── requirements.txt                # Updated with cloud SDKs
```

## 🔧 Detailed Usage

### 1. Object Storage Service (OBS)

#### Upload Dataset
```python
from src.cloud.huawei_storage import HuaweiCloudStorage

# Initialize storage client
storage = HuaweiCloudStorage()

# Create bucket and upload data
storage.create_bucket()
upload_arsl_dataset(storage, "data/")
```

#### Download Model Artifacts
```python
# Download trained model
storage.download_file(
    obs_path="models/best_model.pth",
    local_path="models/downloaded_model.pth"
)
```

### 2. ModelArts Training

#### Start Training Job
```python
from src.cloud.huawei_modelarts import ModelArtsManager, create_arsl_training_job

# Initialize ModelArts
modelarts = ModelArtsManager()

# Create training job
job_id = create_arsl_training_job(modelarts)
print(f"Training job started: {job_id}")

# Monitor progress
status = modelarts.get_training_job_status(job_id)
print(f"Status: {status}")
```

#### Training Parameters
The cloud training script accepts these hyperparameters:
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)
- `num_classes`: Number of classes (default: 32)
- `image_size`: Input image size (default: 64)
- `dropout`: Dropout rate (default: 0.5)

### 3. Model Deployment

#### Deploy Inference Service
```python
from src.cloud.huawei_modelarts import ModelArtsManager

modelarts = ModelArtsManager()

# Create model from training output
model_id = modelarts.create_model(
    model_name="arsl-recognition-v1",
    model_source="obs://bucket/output/model/"
)

# Deploy inference service
service_id = modelarts.deploy_model(
    model_id=model_id,
    service_name="arsl-inference-service"
)
```

### 4. API Gateway Setup

#### Create API Endpoints
```python
from src.cloud.api_deployment import HuaweiCloudDeployment

deployment = HuaweiCloudDeployment()

# Create API group and endpoints
group_id = deployment.create_api_group()
endpoints = deployment.create_api_endpoints(group_id, "your_modelarts_url")
```

#### API Endpoints Created
- `POST /predict` - Single image prediction
- `POST /batch_predict` - Batch image prediction  
- `GET /health` - Service health check

#### Sample API Usage
```python
import requests
import base64

# Prepare image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction request
response = requests.post(
    "https://your-api-gateway-url/predict",
    json={
        "image": image_data,
        "top_k": 3
    }
)

result = response.json()
print(f"Prediction: {result['result']['predictions'][0]}")
```

### 5. Monitoring and Alerts

#### View Training Metrics
- Navigate to ModelArts console
- Select your training job
- View real-time metrics and logs

#### API Performance Monitoring
- Open Cloud Eye console
- View API Gateway metrics
- Set up custom alarms

#### Pre-configured Alerts
- High inference latency (>1000ms)
- High error rate (>5%)
- Low model accuracy (<85%)

## 🔍 Troubleshooting

### Common Issues

#### 1. Authentication Errors
```
Error: Failed to create client: Credentials not found
```
**Solution**: Verify environment variables are set correctly

#### 2. Bucket Access Denied
```
Error: Access denied to bucket
```
**Solution**: Check bucket permissions and region settings

#### 3. Training Job Failed
```
Error: Training job failed to start
```
**Solution**: 
- Verify OBS paths are correct
- Check compute resource availability
- Review training script logs

#### 4. API Gateway Timeout
```
Error: Gateway timeout
```
**Solution**: 
- Increase timeout settings
- Check backend service health
- Verify network connectivity

### Debug Commands

```powershell
# Check bucket contents
python -c "from src.cloud.huawei_storage import HuaweiCloudStorage; s=HuaweiCloudStorage(); print(s.list_objects())"

# Check training jobs
python -c "from src.cloud.huawei_modelarts import ModelArtsManager; m=ModelArtsManager(); print(m.list_training_jobs())"

# Test inference service
python src\cloud\inference_service.py --model_path "path/to/model.pth" --test_image "test.jpg"
```

## 💰 Cost Optimization

### Storage Costs
- Use OBS Standard for frequently accessed data
- Use OBS Cold for archival data
- Delete unnecessary training logs periodically

### Compute Costs  
- Use CPU instances for inference if latency allows
- Stop training jobs if not converging
- Use spot instances for non-critical workloads

### Monitoring Costs
- Set up billing alerts
- Review usage reports monthly
- Optimize resource allocation based on metrics

## 🔒 Security Best Practices

### Access Control
- Use IAM roles instead of access keys when possible
- Implement least privilege access
- Rotate credentials regularly

### Data Protection
- Enable OBS encryption at rest
- Use HTTPS for all API communications
- Implement request signing for sensitive operations

### Network Security
- Configure VPC security groups
- Use private networks for internal communication
- Enable API Gateway authentication for production

## 📈 Scaling Strategies

### Horizontal Scaling
- Increase ModelArts instance count
- Use load balancing for API Gateway
- Implement auto-scaling policies

### Performance Optimization
- Use GPU instances for training
- Optimize batch sizes
- Implement model caching

### Regional Deployment
- Deploy in multiple regions for global access
- Use CDN for static content
- Implement geographic load balancing

## 📚 Additional Resources

### Documentation
- [Huawei Cloud ModelArts](https://support.huaweicloud.com/en-us/modelarts/)
- [Object Storage Service](https://support.huaweicloud.com/en-us/obs/)
- [API Gateway](https://support.huaweicloud.com/en-us/apig/)

### Support
- [Huawei Cloud Support](https://support.huaweicloud.com/en-us/)
- [Community Forum](https://developer.huaweicloud.com/en-us/forum/)
- [Technical Documentation](https://support.huaweicloud.com/en-us/)

### SDKs and Tools
- [Huawei Cloud CLI](https://support.huaweicloud.com/en-us/cli/)
- [Python SDK](https://github.com/huaweicloud/huaweicloud-sdk-python-v3)
- [ModelArts Toolkit](https://github.com/huaweicloud/ModelArts-Lab)

---

## 🎯 Next Steps

1. **Complete Environment Setup**
   - Set environment variables
   - Install dependencies
   - Configure regions

2. **Upload Your Dataset**
   - Run storage setup
   - Verify data upload
   - Test data access

3. **Start Model Training**
   - Launch training job
   - Monitor progress
   - Evaluate results

4. **Deploy Inference Service**
   - Create model from training output
   - Deploy to ModelArts
   - Test inference endpoint

5. **Setup API Gateway**
   - Create API endpoints
   - Test API functionality
   - Configure monitoring

6. **Production Deployment**
   - Enable authentication
   - Configure auto-scaling
   - Set up monitoring alerts

---

**🚀 Ready to deploy your Arabic Sign Language Recognition project to Huawei Cloud!**

For questions or issues, refer to the troubleshooting section or contact Huawei Cloud support.