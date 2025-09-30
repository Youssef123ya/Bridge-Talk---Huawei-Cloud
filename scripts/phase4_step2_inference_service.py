"""
Phase 4: Step 2 - Create Real-time Inference Service
Deploy the imported model as a real-time inference service
"""

import json
from pathlib import Path

def print_inference_service_guide():
    """Print detailed guide for creating inference service"""
    
    print("🚀 STEP 2: CREATE REAL-TIME INFERENCE SERVICE")
    print("=" * 60)
    print()
    
    print("🌐 1. ACCESS REAL-TIME SERVICES:")
    print("   URL: https://console.huaweicloud.com/modelarts")
    print("   Login: yyacoup account")
    print("   Region: AF-Cairo")
    print("   Navigate: Inference Management → Real-time Services")
    print()
    
    print("🚀 2. DEPLOY SERVICE:")
    print("   • Click 'Deploy'")
    print("   • Select 'Deploy from Model'")
    print()
    
    print("📋 3. SERVICE CONFIGURATION:")
    print("   Service Name: arsl-inference-service")
    print("   Description: Arabic Sign Language Recognition API")
    print("   Resource Pool: Public resource pools")
    print()
    
    print("🤖 4. MODEL SELECTION:")
    print("   Model Source: My Models")
    print("   Model Name: arsl-recognition-inference-v1")
    print("   Model Version: 1.0.0")
    print("   Deployment Type: Real-time")
    print()
    
    print("💾 5. RESOURCE CONFIGURATION:")
    resource_options = [
        {
            "name": "CPU Standard",
            "flavor": "modelarts.vm.cpu.2u4g",
            "specs": "2 vCPUs, 4GB RAM",
            "cost": "Low",
            "throughput": "50 req/min",
            "latency": "~300ms"
        },
        {
            "name": "CPU Enhanced", 
            "flavor": "modelarts.vm.cpu.4u8g",
            "specs": "4 vCPUs, 8GB RAM",
            "cost": "Medium",
            "throughput": "100 req/min", 
            "latency": "~200ms"
        },
        {
            "name": "GPU Accelerated",
            "flavor": "modelarts.vm.gpu.t4",
            "specs": "GPU T4, 16GB RAM",
            "cost": "High",
            "throughput": "500 req/min",
            "latency": "~100ms"
        }
    ]
    
    print("   💡 Recommended: CPU Enhanced (4u8g) for cost-performance balance")
    print()
    print("   Resource Options:")
    for option in resource_options:
        print(f"   • {option['name']}: {option['specs']}")
        print(f"     - Throughput: {option['throughput']}")
        print(f"     - Latency: {option['latency']}")
        print(f"     - Cost: {option['cost']}")
        print()
    
    print("🔧 6. SCALING CONFIGURATION:")
    print("   Instance Count: 1")
    print("   Auto Scaling: Enabled")
    print("   Min Instances: 1")
    print("   Max Instances: 5")
    print("   Scale Trigger: CPU > 70%")
    print("   Scale Down Delay: 5 minutes")
    print()
    
    print("⚙️ 7. ADVANCED SETTINGS:")
    print("   Environment Variables:")
    print("     MODEL_NAME = arsl-recognition")
    print("     LOG_LEVEL = INFO")
    print("     BATCH_SIZE = 1")
    print("     MAX_BATCH_DELAY = 100ms")
    print()

def create_service_config():
    """Create service configuration file"""
    
    config = {
        "service_config": {
            "service_name": "arsl-inference-service",
            "description": "Arabic Sign Language Recognition Real-time API",
            "model_name": "arsl-recognition-inference-v1",
            "model_version": "1.0.0",
            "resource_flavor": "modelarts.vm.cpu.4u8g",
            "instance_count": 1,
            "auto_scaling": {
                "enabled": True,
                "min_instances": 1,
                "max_instances": 5,
                "scale_up_threshold": 70,
                "scale_down_threshold": 30,
                "scale_delay": 300
            }
        },
        "api_config": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Base64 encoded image (JPEG/PNG)"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of top predictions to return"
                    }
                },
                "required": ["image"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "class": {"type": "string"},
                                "class_index": {"type": "integer"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "model_version": {"type": "string"},
                    "device": {"type": "string"}
                }
            }
        }
    }
    
    # Save configuration
    config_file = Path("config/inference_service_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Service configuration saved: {config_file}")
    return config_file

def print_testing_guide():
    """Print guide for testing the inference service"""
    
    print("\n🧪 STEP 3: TEST INFERENCE SERVICE")
    print("=" * 40)
    print()
    
    print("📋 TESTING METHODS:")
    print("   1. ModelArts Console Built-in Tester")
    print("   2. REST API Testing")
    print("   3. SDK Testing")
    print("   4. Custom Test Script")
    print()
    
    print("🔬 1. CONSOLE TESTING:")
    print("   • Go to service details page")
    print("   • Click 'Test' tab")
    print("   • Upload test image or paste base64")
    print("   • Click 'Test' to see predictions")
    print()
    
    print("🌐 2. REST API TESTING:")
    print("   Endpoint: https://[service-id].modelarts.[region].ai.azure.com/predict")
    print("   Method: POST")
    print("   Content-Type: application/json")
    print()
    print("   Sample Request:")
    sample_request = {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
        "top_k": 3
    }
    print(f"   {json.dumps(sample_request, indent=2)}")
    print()
    
    print("   Sample Response:")
    sample_response = {
        "predictions": [
            {"class": "alef", "class_index": 0, "confidence": 0.94},
            {"class": "baa", "class_index": 1, "confidence": 0.04},
            {"class": "taa", "class_index": 2, "confidence": 0.02}
        ],
        "model_version": "1.0.0",
        "device": "cpu"
    }
    print(f"   {json.dumps(sample_response, indent=2)}")

def print_monitoring_setup():
    """Print monitoring and alerting setup"""
    
    print("\n📊 STEP 4: CONFIGURE MONITORING")
    print("=" * 40)
    print()
    
    print("📈 MONITORING METRICS:")
    print("   • Request Rate (requests/second)")
    print("   • Response Time (average, P95, P99)")
    print("   • Error Rate (4xx, 5xx errors)")
    print("   • CPU Utilization")
    print("   • Memory Usage")
    print("   • GPU Utilization (if GPU flavor)")
    print()
    
    print("🚨 ALERTING RULES:")
    alert_rules = [
        {"metric": "Response Time", "threshold": "> 1000ms", "duration": "5 minutes"},
        {"metric": "Error Rate", "threshold": "> 5%", "duration": "2 minutes"}, 
        {"metric": "CPU Usage", "threshold": "> 80%", "duration": "10 minutes"},
        {"metric": "Memory Usage", "threshold": "> 85%", "duration": "5 minutes"}
    ]
    
    for rule in alert_rules:
        print(f"   • {rule['metric']}: {rule['threshold']} for {rule['duration']}")
    
    print()
    print("📧 NOTIFICATION SETUP:")
    print("   • Email alerts for critical issues")
    print("   • SMS for service downtime")
    print("   • Webhook for integration with external systems")

def main():
    """Main function"""
    print("🚀 PHASE 4: STEP 2 - REAL-TIME INFERENCE SERVICE")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("Service: arsl-inference-service")
    print("=" * 60)
    
    # Print service creation guide
    print_inference_service_guide()
    
    # Create service configuration
    config_file = create_service_config()
    
    # Print testing guide
    print_testing_guide()
    
    # Print monitoring setup
    print_monitoring_setup()
    
    print(f"\n🎯 STEP 2 SUMMARY:")
    print(f"✅ Service configuration prepared")
    print(f"✅ Testing methods documented")
    print(f"✅ Monitoring setup ready")
    print(f"📋 Ready for manual service deployment")
    print(f"🌐 Next: Deploy service in ModelArts console")

if __name__ == "__main__":
    main()