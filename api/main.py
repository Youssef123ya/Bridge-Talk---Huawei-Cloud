"""
FastAPI application for Arabic Sign Language Recognition
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import json
import os
from typing import Dict, Any
from pathlib import Path

app = FastAPI(
    title="Arabic Sign Language Recognition API",
    description="API for recognizing Arabic sign language gestures from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing
model = None
transform = None
class_names = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the trained model and class mapping"""
    global model, transform, class_names
    
    # Load class mapping
    class_mapping_path = Path("data/processed/class_mapping.json")
    if class_mapping_path.exists():
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        class_names = sorted(list(class_mapping.keys()))
    else:
        # Default Arabic alphabet classes
        class_names = [
            'alef', 'baa', 'taa', 'thaa', 'jeem', 'haa', 'khaa', 'dal', 'thal', 'raa',
            'zay', 'seen', 'sheen', 'saad', 'daad', 'tah', 'thah', 'ayn', 'ghayn', 
            'faa', 'qaf', 'kaf', 'laam', 'meem', 'noon', 'haa_end', 'waw', 'yaa'
        ]
    
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Try to load the trained model
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/best_model.pth")
    
    if os.path.exists(model_path):
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Import model registry
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from src.models.base_model import ModelRegistry
            
            # Get model from registry
            model = ModelRegistry.get_model(checkpoint['model_name'])
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✅ Loaded trained model: {checkpoint['model_name']}")
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            model = None
    
    if model is None:
        # Create a demo model for testing
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from src.models.cnn_architectures import BasicCNN
            
            model = BasicCNN(num_classes=len(class_names))
            print("⚠️  Using demo BasicCNN model (untrained)")
        except Exception as e:
            print(f"❌ Error creating demo model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load model")
    
    model.to(device)
    model.eval()
    print(f"🚀 Model loaded on {device}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    try:
        load_model()
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Arabic Sign Language Recognition API",
        "status": "running",
        "device": device,
        "model_loaded": model is not None,
        "classes": len(class_names) if class_names else 0
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "device": device,
        "num_classes": len(class_names) if class_names else 0,
        "class_names": class_names[:5] if class_names else []  # First 5 classes
    }

@app.get("/classes")
async def get_classes():
    """Get all available classes"""
    if class_names is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "classes": class_names,
        "total": len(class_names)
    }

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    """
    Predict Arabic sign language gesture from uploaded image
    """
    if model is None or transform is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, k=min(5, len(class_names)))
        
        # Prepare results
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            predictions.append({
                "class": class_names[idx.item()],
                "confidence": float(prob.item()),
                "confidence_percent": f"{float(prob.item()) * 100:.2f}%"
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "image_size": image.size,
            "device_used": device
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict Arabic sign language gestures from multiple uploaded images
    """
    if model is None or transform is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Read and process the image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Apply preprocessing
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top3_probs, top3_indices = torch.topk(probabilities, k=min(3, len(class_names)))
            
            # Prepare results
            predictions = []
            for prob, idx in zip(top3_probs[0], top3_indices[0]):
                predictions.append({
                    "class": class_names[idx.item()],
                    "confidence": float(prob.item()),
                    "confidence_percent": f"{float(prob.item()) * 100:.2f}%"
                })
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": True,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None
            })
            
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results
    }

@app.get("/model_info")
async def get_model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model.__class__.__name__,
        "device": device,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "num_classes": len(class_names) if class_names else 0,
        "input_size": [3, 224, 224],
        "model_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)