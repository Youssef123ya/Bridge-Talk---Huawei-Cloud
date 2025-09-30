"""
Phase 4: Step 1 - Import Trained Model to ModelArts
Import the trained PyTorch model for inference deployment
"""

import json
import sys
import os
from pathlib import Path

def print_model_import_guide():
    """Print detailed guide for importing trained model"""
    
    print("📥 STEP 1: IMPORT TRAINED MODEL TO MODELARTS")
    print("=" * 60)
    print()
    
    print("🌐 1. ACCESS MODEL MANAGEMENT:")
    print("   URL: https://console.huaweicloud.com/modelarts")
    print("   Login: yyacoup account")
    print("   Region: AF-Cairo")
    print("   Navigate: Model Management → Models")
    print()
    
    print("📥 2. IMPORT MODEL:")
    print("   • Click 'Import Model'")
    print("   • Select 'Import from OBS'")
    print()
    
    print("📋 3. MODEL CONFIGURATION:")
    print("   Model Name: arsl-recognition-inference-v1")
    print("   Version: 1.0.0")
    print("   Description: Arabic Sign Language Recognition Inference Model")
    print()
    
    print("📁 4. MODEL SOURCE:")
    print("   Model File Path: obs://arsl-youssef-af-cairo-2025/output/")
    print("   Model File: best_model.pth (from training)")
    print("   AI Framework: PyTorch")
    print("   Framework Version: 1.8.0")
    print("   Runtime: python3.7")
    print()
    
    print("📝 5. INFERENCE CODE:")
    print("   Runtime Code: obs://arsl-youssef-af-cairo-2025/inference/")
    print("   Boot File: inference_service.py")
    print("   Model Class: ARSLInferenceModel")
    print()
    
    print("🔧 6. MODEL SPECIFICATIONS:")
    print("   Input Shape: [1, 1, 64, 64] (batch, channels, height, width)")
    print("   Input Type: image/jpeg, image/png")
    print("   Output: JSON with predictions and confidence scores")
    print("   Input Schema: base64 encoded image")
    print()
    
    print("📊 7. MODEL METADATA:")
    model_metadata = {
        "framework": "PyTorch",
        "version": "1.8.0",
        "input_shape": [1, 1, 64, 64],
        "num_classes": 32,
        "class_names": [
            "alef", "baa", "taa", "thaa", "jeem", "haa", "khaa", "dal",
            "thal", "raa", "zay", "seen", "sheen", "saad", "dhad", "taa_marbuta",
            "thaa_maftuh", "ain", "ghain", "fa", "qaaf", "kaaf", "lam", "meem",
            "nun", "ha", "waw", "ya", "hamza", "lam_alef", "taa_maftuha", "alef_maqsura"
        ],
        "preprocessing": "Grayscale, 64x64 resize, normalization",
        "accuracy": "85%+",
        "latency": "<200ms"
    }
    
    for key, value in model_metadata.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"   {key}: [{value[0]}, {value[1]}, ..., {value[-1]}] ({len(value)} total)")
        else:
            print(f"   {key}: {value}")
    print()

def create_inference_service_code():
    """Create inference service code for ModelArts"""
    
    print("📝 CREATING INFERENCE SERVICE CODE")
    print("=" * 40)
    
    # Create inference directory
    inf_dir = Path("src/inference")
    inf_dir.mkdir(exist_ok=True)
    
    # Inference service code
    inference_code = '''"""
ModelArts Inference Service for Arabic Sign Language Recognition
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import json
import numpy as np

class SimpleCNN(nn.Module):
    """CNN Architecture for Arabic Sign Language Recognition"""
    
    def __init__(self, num_classes=32):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class ARSLInferenceModel:
    """Arabic Sign Language Inference Model for ModelArts"""
    
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Arabic alphabet classes
        self.class_names = [
            "alef", "baa", "taa", "thaa", "jeem", "haa", "khaa", "dal",
            "thal", "raa", "zay", "seen", "sheen", "saad", "dhad", "taa_marbuta",
            "thaa_maftuh", "ain", "ghain", "fa", "qaaf", "kaaf", "lam", "meem",
            "nun", "ha", "waw", "ya", "hamza", "lam_alef", "taa_maftuha", "alef_maqsura"
        ]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.model = SimpleCNN(num_classes=32)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _preprocess_image(self, image_data):
        """Preprocess image for inference"""
        try:
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(image_data)
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, data):
        """Make prediction on input data"""
        try:
            # Extract image data
            if isinstance(data, dict):
                image_data = data.get('image', data.get('data', ''))
                top_k = data.get('top_k', 3)
            else:
                image_data = data
                top_k = 3
            
            # Preprocess image
            image_tensor = self._preprocess_image(image_data)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                predictions = []
                for i in range(top_k):
                    class_idx = top_indices[0][i].item()
                    confidence = top_probs[0][i].item()
                    class_name = self.class_names[class_idx]
                    
                    predictions.append({
                        "class": class_name,
                        "class_index": class_idx,
                        "confidence": float(confidence)
                    })
            
            result = {
                "predictions": predictions,
                "model_version": "1.0.0",
                "device": str(self.device)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "predictions": [],
                "model_version": "1.0.0"
            }

# ModelArts inference interface
def init():
    """Initialize the model (called once when service starts)"""
    global model
    model = ARSLInferenceModel("/opt/ml/model/best_model.pth", "arsl-recognition")
    return model

def inference(data):
    """Inference function (called for each request)"""
    global model
    
    # Parse input data
    if isinstance(data, str):
        data = json.loads(data)
    
    # Make prediction
    result = model.predict(data)
    
    return json.dumps(result)

'''
    
    # Save inference service
    inf_file = inf_dir / "inference_service.py"
    with open(inf_file, 'w', encoding='utf-8') as f:
        f.write(inference_code)
    
    print(f"✅ Inference service created: {inf_file}")
    
    # Create config file
    config = {
        "model_type": "PyTorch",
        "framework": "pytorch",
        "framework_version": "1.8.0",
        "python_version": "3.7",
        "model_file": "best_model.pth",
        "dependencies": [
            "torch>=1.8.0",
            "torchvision>=0.9.0",
            "pillow>=8.0.0",
            "numpy>=1.19.0"
        ]
    }
    
    config_file = inf_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Model config created: {config_file}")
    
    return inf_file, config_file

def print_upload_inference_code():
    """Print instructions to upload inference code"""
    
    print("\n📤 UPLOAD INFERENCE CODE TO OBS")
    print("=" * 40)
    print()
    print("📁 Upload these files to: obs://arsl-youssef-af-cairo-2025/inference/")
    print("   • src/inference/inference_service.py")
    print("   • src/inference/config.json")
    print()
    print("🔧 Manual Upload Steps:")
    print("   1. Go to OBS Console: https://console.huaweicloud.com/obs")
    print("   2. Open bucket: arsl-youssef-af-cairo-2025")
    print("   3. Create folder: inference/")
    print("   4. Upload inference_service.py")
    print("   5. Upload config.json")

def main():
    """Main function"""
    print("📥 PHASE 4: STEP 1 - MODEL IMPORT PREPARATION")
    print("Account: yyacoup")
    print("Region: AF-Cairo")
    print("Model: arsl-recognition-inference-v1")
    print("=" * 60)
    
    # Print import guide
    print_model_import_guide()
    
    # Create inference code
    inf_file, config_file = create_inference_service_code()
    
    # Print upload instructions
    print_upload_inference_code()
    
    print(f"\n🎯 STEP 1 SUMMARY:")
    print(f"✅ Inference service code created")
    print(f"✅ Model configuration prepared")
    print(f"📋 Ready for manual model import")
    print(f"🌐 Next: Import model in ModelArts console")

if __name__ == "__main__":
    main()