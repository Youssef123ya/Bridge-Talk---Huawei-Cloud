# 🌐 Step 4: Configure Inference API

## 📋 **Prerequisites**
- ✅ Model trained and saved in OBS
- ✅ Training completed successfully
- ✅ Model performance satisfactory (>85% accuracy)

## 🎯 **API Deployment Overview**

Your inference API will provide:
- ⚡ **Real-time predictions** (< 200ms response time)
- 🔗 **REST API endpoints** for easy integration
- 📊 **Auto-scaling** based on traffic
- 🔒 **Secure authentication** options
- 📈 **Performance monitoring** and alerts

## 🚀 **Deployment Options**

### **Option A: Automated API Deployment (Recommended)**
```powershell
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/deploy_api.py
```

### **Option B: Manual ModelArts Deployment**

#### **1. Create Model in ModelArts**
1. Go to **ModelArts Console** → **Model Management**
2. Click **"Import Model"**
3. Configure:
```yaml
Model Name: arsl-recognition-v1
Model Version: 1.0
Model Source: obs://arsl-youssef-af-cairo-2025/output/best_model.pth
AI Engine: PyTorch 1.8.0
Runtime: Python 3.7
Model Description: Arabic Sign Language Recognition CNN
```

#### **2. Deploy Inference Service**
1. Go to **Real-time Services** → **Deploy**
2. Configure deployment:
```yaml
Service Name: arsl-inference-service
Model: arsl-recognition-v1
Model Version: 1.0
Compute Resource: CPU (2 cores, 4GB) or GPU for faster inference
Instance Count: 1 (auto-scaling available)
```

## 🔧 **API Endpoints Created**

### **1. Single Image Prediction**
```http
POST /predict
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "top_k": 3
}
```

**Response:**
```json
{
  "result": {
    "predictions": [
      {
        "class": "alef",
        "class_index": 0,
        "confidence": 0.94
      },
      {
        "class": "baa", 
        "class_index": 1,
        "confidence": 0.04
      }
    ],
    "inference_time_ms": 156,
    "model_version": "1.0"
  },
  "status": "success"
}
```

### **2. Batch Prediction**
```http
POST /batch_predict
Content-Type: application/json

{
  "images": ["base64_image1", "base64_image2"],
  "top_k": 3
}
```

### **3. Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_classes": 32,
  "timestamp": 1695389876.123
}
```

## 🌐 **API Gateway Configuration**

### **1. Create API Gateway Group**
```yaml
Group Name: arsl-recognition-apis
Description: Arabic Sign Language Recognition APIs
Region: AF-Cairo
```

### **2. Configure API Endpoints**
```yaml
APIs Created:
  - /predict (POST) → ModelArts inference service
  - /batch_predict (POST) → ModelArts batch inference
  - /health (GET) → Health check endpoint

Security:
  - API Key authentication (optional)
  - CORS enabled for web apps
  - Rate limiting (1000 requests/hour per key)
```

## 📊 **Performance Optimization**

### **Instance Types for Inference:**
| Type | CPU | Memory | Throughput | Cost |
|------|-----|--------|------------|------|
| **CPU-2U4G** | 2 cores | 4GB | 50 req/min | Low |
| **CPU-4U8G** | 4 cores | 8GB | 100 req/min | Medium |
| **GPU-T4** | T4 GPU | 16GB | 500 req/min | High |

### **Auto-scaling Configuration:**
```yaml
auto_scaling:
  min_instances: 1
  max_instances: 5
  target_cpu_utilization: 70%
  scale_up_threshold: 80%
  scale_down_threshold: 30%
```

## 🔍 **Testing Your API**

### **Test Script (Python):**
```python
import requests
import base64

# Test image
with open("test_sign.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# API request
response = requests.post(
    "https://your-api-gateway-url/predict",
    json={"image": image_b64, "top_k": 3}
)

result = response.json()
print(f"Prediction: {result['result']['predictions'][0]['class']}")
print(f"Confidence: {result['result']['predictions'][0]['confidence']:.2%}")
```

### **Test with cURL:**
```bash
curl -X POST https://your-api-gateway-url/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data", "top_k": 3}'
```

## 📈 **Monitoring & Analytics**

### **Cloud Eye Metrics:**
- 📊 **Request Rate**: Requests per second/minute
- ⏱️ **Response Time**: Average/P95/P99 latency  
- 🚨 **Error Rate**: Failed requests percentage
- 💾 **Resource Usage**: CPU, memory, GPU utilization

### **Alerts Configuration:**
```yaml
alerts:
  high_latency:
    threshold: 1000ms
    duration: 5 minutes
  high_error_rate:
    threshold: 5%
    duration: 2 minutes
  resource_exhaustion:
    cpu_threshold: 90%
    memory_threshold: 85%
```

## 🔒 **Security & Authentication**

### **API Key Authentication:**
1. Generate API keys in API Gateway console
2. Distribute keys to authorized users
3. Monitor usage per key

### **IP Whitelisting:**
```yaml
allowed_ips:
  - "203.0.113.0/24"  # Office network
  - "198.51.100.50"   # Production server
```

### **Rate Limiting:**
```yaml
rate_limits:
  per_api_key: 1000/hour
  per_ip: 100/hour
  burst_limit: 10/minute
```

## 🚨 **Troubleshooting API Issues**

### **Issue**: Slow response times
**Solutions**:
- Switch to GPU instance for inference
- Enable model caching
- Optimize image preprocessing
- Use batch prediction for multiple images

### **Issue**: High error rates
**Solutions**:
- Check model loading in logs
- Verify input image format (base64, correct size)
- Monitor resource utilization
- Check ModelArts service health

### **Issue**: API Gateway timeouts
**Solutions**:
- Increase timeout settings (default 30s)
- Optimize model inference time
- Use async processing for large requests

## 📱 **Integration Examples**

### **Web Application (JavaScript):**
```javascript
async function predictSign(imageFile) {
    const base64 = await fileToBase64(imageFile);
    
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image: base64,
            top_k: 3
        })
    });
    
    const result = await response.json();
    return result.result.predictions[0];
}
```

### **Mobile App (Flutter/React Native):**
```dart
Future<SignPrediction> predictSign(File imageFile) async {
  String base64Image = base64Encode(await imageFile.readAsBytes());
  
  final response = await http.post(
    Uri.parse('$apiUrl/predict'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'image': base64Image,
      'top_k': 3
    })
  );
  
  return SignPrediction.fromJson(jsonDecode(response.body));
}
```

## 🎯 **Production Checklist**

- [ ] **Model Performance**: >85% accuracy validated
- [ ] **API Response Time**: <500ms average
- [ ] **Health Checks**: All endpoints responding
- [ ] **Authentication**: API keys configured
- [ ] **Monitoring**: Alerts and dashboards set up
- [ ] **Auto-scaling**: Configured for expected load
- [ ] **Error Handling**: Graceful degradation implemented
- [ ] **Documentation**: API docs published
- [ ] **Testing**: Load testing completed

---

## 🚀 **Quick Deployment Commands**

```powershell
# 1. Deploy API service
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/deploy_api.py

# 2. Test API endpoints  
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/test_api.py

# 3. Monitor API performance
& "D:/Youtube/co/HU/mon/SIGN project/pex2/venv/Scripts/python.exe" scripts/monitor_api.py
```

**🎉 Congratulations! Your Arabic Sign Language Recognition API is now live and ready to serve users globally!** 🌍