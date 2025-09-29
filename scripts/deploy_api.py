#!/usr/bin/env python3
"""
Deploy inference API service for Arabic Sign Language Recognition
"""

import sys
import os
import json
import time
sys.path.append('src')

from cloud.inference_service import InferenceService
import logging

def main():
    """Deploy inference API service"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🌐 Deploying Inference API Service...")
    print("=" * 60)
    
    try:
        # Check if training is completed
        if os.path.exists("logs/training_job_info.json"):
            with open("logs/training_job_info.json", "r") as f:
                training_info = json.load(f)
            print(f"📋 Found training job: {training_info['job_name']}")
        else:
            print("⚠️  No training job info found. Proceeding with deployment...")
        
        inference = InferenceService()
        
        # API deployment configuration
        service_config = {
            "service_name": f"arsl-inference-{int(time.time())}",
            "description": "Arabic Sign Language Recognition Inference Service",
            "model_config": {
                "model_name": "arsl-recognition-v1",
                "model_path": "obs://arsl-youssef-af-cairo-2025/output/best_model.pth",
                "model_version": "1.0",
                "runtime": "python3.7",
                "framework": "PyTorch"
            },
            "deployment_config": {
                "instance_type": "modelarts.vm.cpu.2u",  # CPU instance for cost efficiency
                "instance_count": 1,
                "auto_scaling": {
                    "enabled": True,
                    "min_instances": 1,
                    "max_instances": 3,
                    "target_cpu_utilization": 70
                }
            },
            "api_config": {
                "endpoints": [
                    {"path": "/predict", "method": "POST"},
                    {"path": "/batch_predict", "method": "POST"},
                    {"path": "/health", "method": "GET"}
                ],
                "authentication": {
                    "type": "api_key",
                    "rate_limit": "1000/hour"
                }
            }
        }
        
        print("📋 Service Configuration:")
        print(f"   🏷️  Service Name: {service_config['service_name']}")
        print(f"   🔧 Instance Type: {service_config['deployment_config']['instance_type']}")
        print(f"   📊 Auto Scaling: 1-3 instances")
        print(f"   🌐 Endpoints: /predict, /batch_predict, /health")
        
        # Deploy inference service
        print("\n🚀 Deploying inference service...")
        service_id = inference.deploy_service(service_config)
        
        if service_id:
            print(f"✅ Inference service deployed successfully!")
            print(f"   🆔 Service ID: {service_id}")
            print(f"   🔗 Monitor at: https://console.huaweicloud.com/modelarts")
            
            # Get service status
            print("\n📊 Checking service status...")
            status = inference.get_service_status(service_id)
            print(f"   Status: {status}")
            
            if status in ["Running", "Deploying"]:
                print("\n🎉 API service is starting/running!")
                
                # Get API endpoints
                endpoints = inference.get_service_endpoints(service_id)
                if endpoints:
                    print("\n🌐 API Endpoints:")
                    for endpoint in endpoints:
                        print(f"   📍 {endpoint}")
                
                print("\n📋 API Usage Examples:")
                print("   📤 Single prediction:")
                print("      POST /predict")
                print("      Body: {\"image\": \"base64_encoded_image\", \"top_k\": 3}")
                
                print("\n   📦 Batch prediction:")
                print("      POST /batch_predict")
                print("      Body: {\"images\": [\"base64_1\", \"base64_2\"], \"top_k\": 3}")
                
                print("\n   🏥 Health check:")
                print("      GET /health")
                
                # Save service info
                service_info = {
                    "service_id": service_id,
                    "service_name": service_config['service_name'],
                    "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "endpoints": endpoints if 'endpoints' in locals() else [],
                    "config": service_config
                }
                
                with open("logs/inference_service_info.json", "w") as f:
                    json.dump(service_info, f, indent=2)
                    
                print(f"\n💾 Service info saved to: logs/inference_service_info.json")
                
            return True
        else:
            print("❌ Failed to deploy inference service")
            return False
            
    except Exception as e:
        print(f"\n❌ API deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)