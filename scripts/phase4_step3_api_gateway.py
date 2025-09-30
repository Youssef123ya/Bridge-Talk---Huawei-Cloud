"""
Phase 4: Step 3 - Deploy API Gateway Endpoints
Create public REST API endpoints for the inference service
"""

import json
import requests
from pathlib import Path
from datetime import datetime

def print_api_gateway_guide():
    """Print detailed guide for API Gateway deployment"""
    
    print("🌐 STEP 3: DEPLOY API GATEWAY ENDPOINTS")
    print("=" * 60)
    print()
    
    print("🔗 1. ACCESS API GATEWAY:")
    print("   URL: https://console.huaweicloud.com/apig")
    print("   Login: yyacoup account")
    print("   Region: AF-Cairo") 
    print("   Navigate: API Gateway → Dedicated Gateways")
    print()
    
    print("🏗️ 2. CREATE DEDICATED GATEWAY:")
    print("   Gateway Name: arsl-api-gateway")
    print("   Specification: Basic (500 requests/second)")
    print("   Version: 1.0")
    print("   VPC: Default VPC")
    print("   Subnet: Default subnet")
    print("   Security Group: Default")
    print("   EIP: Auto-assign (for public access)")
    print()
    
    print("📚 3. CREATE API GROUP:")
    print("   Group Name: arsl-recognition-api")
    print("   Description: Arabic Sign Language Recognition APIs")
    print("   Subdomain: arsl-api")
    print("   Full Domain: arsl-api.apig.af-north-1.huaweicloudapis.com")
    print()
    
    print("🔧 4. CREATE API DEFINITIONS:")
    
    apis = [
        {
            "name": "predict-single",
            "path": "/v1/predict",
            "method": "POST",
            "description": "Predict single sign language image",
            "auth": "None"
        },
        {
            "name": "predict-batch", 
            "path": "/v1/predict/batch",
            "method": "POST",
            "description": "Predict multiple images in batch",
            "auth": "None"
        },
        {
            "name": "health-check",
            "path": "/v1/health",
            "method": "GET", 
            "description": "Check API service health",
            "auth": "None"
        },
        {
            "name": "model-info",
            "path": "/v1/model/info",
            "method": "GET",
            "description": "Get model information and metadata",
            "auth": "None"
        }
    ]
    
    for api in apis:
        print(f"   API: {api['name']}")
        print(f"   Path: {api['path']}")
        print(f"   Method: {api['method']}")
        print(f"   Description: {api['description']}")
        print(f"   Authentication: {api['auth']}")
        print()

def create_api_definitions():
    """Create detailed API definitions"""
    
    api_definitions = {
        "predict_single": {
            "name": "predict-single",
            "path": "/v1/predict",
            "method": "POST",
            "description": "Predict Arabic sign language from single image",
            "request_schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Base64 encoded image (JPEG/PNG, max 10MB)"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 32,
                        "description": "Number of top predictions to return"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "default": 0.1,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum confidence threshold for predictions"
                    }
                },
                "required": ["image"]
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "class": {"type": "string", "description": "Arabic letter name"},
                                "class_index": {"type": "integer", "description": "Class index (0-31)"},
                                "confidence": {"type": "number", "description": "Prediction confidence (0-1)"},
                                "arabic_letter": {"type": "string", "description": "Arabic letter character"}
                            }
                        }
                    },
                    "model_version": {"type": "string"},
                    "processing_time_ms": {"type": "number"},
                    "timestamp": {"type": "string"}
                }
            },
            "backend_service": {
                "type": "modelarts",
                "service_name": "arsl-inference-service",
                "timeout": 30000,
                "retry_count": 2
            }
        },
        
        "predict_batch": {
            "name": "predict-batch",
            "path": "/v1/predict/batch", 
            "method": "POST",
            "description": "Predict multiple Arabic sign language images",
            "request_schema": {
                "type": "object",
                "properties": {
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object", 
                            "properties": {
                                "id": {"type": "string", "description": "Image identifier"},
                                "image": {"type": "string", "description": "Base64 encoded image"}
                            },
                            "required": ["id", "image"]
                        },
                        "maxItems": 10,
                        "description": "Array of images to process (max 10)"
                    },
                    "top_k": {
                        "type": "integer", 
                        "default": 3,
                        "minimum": 1,
                        "maximum": 32
                    }
                },
                "required": ["images"]
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "predictions": {"type": "array"},
                                "success": {"type": "boolean"},
                                "error": {"type": "string"}
                            }
                        }
                    },
                    "batch_stats": {
                        "type": "object",
                        "properties": {
                            "total_images": {"type": "integer"},
                            "successful_predictions": {"type": "integer"},
                            "failed_predictions": {"type": "integer"},
                            "total_processing_time_ms": {"type": "number"}
                        }
                    }
                }
            }
        },
        
        "health_check": {
            "name": "health-check",
            "path": "/v1/health",
            "method": "GET",
            "description": "Check API and model service health",
            "response_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                    "timestamp": {"type": "string"},
                    "services": {
                        "type": "object",
                        "properties": {
                            "api_gateway": {"type": "string", "enum": ["up", "down"]},
                            "inference_service": {"type": "string", "enum": ["up", "down"]},
                            "model": {"type": "string", "enum": ["loaded", "loading", "error"]}
                        }
                    },
                    "metrics": {
                        "type": "object",
                        "properties": {
                            "requests_per_minute": {"type": "number"},
                            "average_response_time_ms": {"type": "number"},
                            "error_rate_percent": {"type": "number"}
                        }
                    }
                }
            }
        },
        
        "model_info": {
            "name": "model-info",
            "path": "/v1/model/info",
            "method": "GET", 
            "description": "Get model information and supported classes",
            "response_schema": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "model_version": {"type": "string"},
                    "framework": {"type": "string"},
                    "input_format": {"type": "string"},
                    "classes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "name": {"type": "string"},
                                "arabic_letter": {"type": "string"}
                            }
                        }
                    },
                    "model_stats": {
                        "type": "object",
                        "properties": {
                            "training_images": {"type": "integer"},
                            "accuracy": {"type": "number"},
                            "f1_score": {"type": "number"}
                        }
                    }
                }
            }
        }
    }
    
    # Save API definitions
    api_file = Path("config/api_gateway_definitions.json")
    api_file.parent.mkdir(exist_ok=True)
    
    with open(api_file, 'w', encoding='utf-8') as f:
        json.dump(api_definitions, f, indent=2, ensure_ascii=False)
    
    print(f"📋 API definitions saved: {api_file}")
    return api_file

def print_security_guide():
    """Print API security configuration guide"""
    
    print("\n🔐 STEP 4: CONFIGURE API SECURITY")
    print("=" * 40)
    print()
    
    print("🛡️ SECURITY OPTIONS:")
    print("   1. No Authentication (for testing)")
    print("   2. API Key Authentication")
    print("   3. JWT Token Authentication")
    print("   4. OAuth 2.0")
    print("   5. Request Signature")
    print()
    
    print("🔑 RECOMMENDED: API KEY AUTHENTICATION")
    print("   • Simple to implement")
    print("   • Good for API access control")
    print("   • Supports rate limiting per key")
    print("   • Easy key management")
    print()
    
    print("⚙️ API KEY SETUP:")
    print("   1. Create API key in console")
    print("   2. Set usage quotas:")
    print("      - Requests per second: 100")
    print("      - Requests per hour: 10,000")
    print("      - Requests per day: 100,000")
    print("   3. Add X-API-Key header requirement")
    print("   4. Configure rate limiting policies")
    print()
    
    print("🚫 ADDITIONAL SECURITY:")
    security_features = [
        "CORS configuration for web clients",
        "IP whitelist/blacklist", 
        "Request size limits (max 10MB)",
        "Rate limiting by IP address",
        "Request/response logging",
        "SSL/TLS encryption (HTTPS only)"
    ]
    
    for feature in security_features:
        print(f"   • {feature}")

def create_test_client():
    """Create test client for API testing"""
    
    test_client_code = '''"""
API Test Client for Arabic Sign Language Recognition
Test the deployed API Gateway endpoints
"""

import base64
import requests
import json
from pathlib import Path
import time

class ARSLAPIClient:
    """Client for testing ARSL API endpoints"""
    
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "ARSL-API-Client/1.0"
        })
    
    def predict_single(self, image_path, top_k=3, confidence_threshold=0.1):
        """Predict single image"""
        
        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "image": image_data,
            "top_k": top_k,
            "confidence_threshold": confidence_threshold
        }
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/v1/predict",
            json=payload
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result["client_request_time_ms"] = (end_time - start_time) * 1000
            return result
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    
    def predict_batch(self, image_paths, top_k=3):
        """Predict multiple images"""
        
        images = []
        for i, image_path in enumerate(image_paths):
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            images.append({
                "id": f"image_{i}",
                "image": image_data
            })
        
        payload = {
            "images": images,
            "top_k": top_k
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/predict/batch",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    
    def health_check(self):
        """Check API health"""
        response = self.session.get(f"{self.base_url}/v1/health")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    
    def get_model_info(self):
        """Get model information"""
        response = self.session.get(f"{self.base_url}/v1/model/info")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }

# Example usage
if __name__ == "__main__":
    # Configure API client
    API_BASE_URL = "https://arsl-api.apig.af-north-1.huaweicloudapis.com"
    API_KEY = "your-api-key-here"  # Replace with actual API key
    
    client = ARSLAPIClient(API_BASE_URL, API_KEY)
    
    # Test health check
    print("Testing health check...")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Test model info
    print("\\nTesting model info...")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    # Test single prediction (if test image exists)
    test_image = "test_image.jpg"
    if Path(test_image).exists():
        print(f"\\nTesting single prediction with {test_image}...")
        result = client.predict_single(test_image)
        print(json.dumps(result, indent=2))
    else:
        print(f"\\nTest image {test_image} not found, skipping prediction test")
'''
    
    # Save test client
    client_file = Path("scripts/api_test_client.py")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(test_client_code)
    
    print(f"🧪 Test client saved: {client_file}")
    return client_file

def print_deployment_steps():
    """Print step-by-step deployment guide"""
    
    print("\n📝 STEP-BY-STEP DEPLOYMENT:")
    print("=" * 40)
    
    steps = [
        "1. Create Dedicated Gateway (5 minutes)",
        "2. Create API Group (2 minutes)",
        "3. Create API definitions (15 minutes)",
        "4. Configure backend services (10 minutes)",
        "5. Set up security policies (10 minutes)",
        "6. Test APIs (15 minutes)",
        "7. Configure monitoring (5 minutes)",
        "8. Generate API documentation (5 minutes)"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n⏱️ Total deployment time: ~67 minutes")
    print(f"🎯 Result: Public API endpoints ready for production")

def main():
    """Main function"""
    print("🌐 PHASE 4: STEP 3 - API GATEWAY DEPLOYMENT")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("Gateway: arsl-api-gateway")
    print("=" * 60)
    
    # Print API Gateway guide
    print_api_gateway_guide()
    
    # Create API definitions
    api_file = create_api_definitions()
    
    # Print security guide
    print_security_guide()
    
    # Create test client
    client_file = create_test_client()
    
    # Print deployment steps
    print_deployment_steps()
    
    print(f"\n🎯 STEP 3 SUMMARY:")
    print(f"✅ API Gateway configuration prepared")
    print(f"✅ API definitions created ({api_file})")
    print(f"✅ Security policies documented")
    print(f"✅ Test client ready ({client_file})")
    print(f"📋 Ready for manual API Gateway deployment")
    print(f"🌐 Next: Deploy APIs in API Gateway console")

if __name__ == "__main__":
    main()
'''