"""
Real-time inference service for Arabic Sign Language Recognition
Compatible with Huawei Cloud ModelArts and API Gateway
"""

import os
import json
import logging
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
import time

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    CNN model for Arabic Sign Language Recognition
    (Same architecture as training)
    """
    
    def __init__(self, num_classes: int = 32, dropout: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ARSLInferenceService:
    """
    Inference service for Arabic Sign Language Recognition
    """
    
    def __init__(self, model_path: str):
        """
        Initialize inference service
        
        Args:
            model_path: Path to trained model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model, self.config = self._load_model(model_path)
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Arabic alphabet mapping (customize based on your dataset)
        self.class_names = [
            'alef', 'baa', 'taa', 'thaa', 'jeem', 'haa', 'khaa', 'dal',
            'thal', 'raa', 'zay', 'seen', 'sheen', 'sad', 'dad', 'tah',
            'zah', 'ain', 'ghain', 'faa', 'qaf', 'kaf', 'lam', 'meem',
            'noon', 'heh', 'waw', 'yaa', 'lam_alef', 'taa_marbuta', 'hamza', 'space'
        ]
        
        logger.info("Inference service initialized successfully")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract configuration
            config = checkpoint.get('config', {
                'num_classes': 32,
                'image_size': 64,
                'dropout': 0.5
            })
            
            # Initialize model
            model = SimpleCNN(
                num_classes=config['num_classes'],
                dropout=config['dropout']
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model accuracy: {checkpoint.get('accuracy', 'Unknown')}")
            
            return model, config
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image from bytes
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def predict(self, image_data: bytes, top_k: int = 3) -> Dict[str, Any]:
        """
        Perform inference on image
        
        Args:
            image_data: Raw image bytes
            top_k: Number of top predictions to return
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_data)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
            
            # Format results
            predictions = []
            for i in range(top_k):
                class_idx = top_indices[i]
                confidence = float(top_probs[i])
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
                
                predictions.append({
                    'class': class_name,
                    'class_index': int(class_idx),
                    'confidence': confidence
                })
            
            inference_time = time.time() - start_time
            
            result = {
                'predictions': predictions,
                'inference_time_ms': round(inference_time * 1000, 2),
                'model_version': '1.0',
                'timestamp': time.time()
            }
            
            logger.info(f"Prediction completed in {inference_time:.3f}s: {predictions[0]['class']} ({predictions[0]['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def batch_predict(self, images_data: List[bytes], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform batch inference
        
        Args:
            images_data: List of raw image bytes
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction results
        """
        results = []
        start_time = time.time()
        
        try:
            for i, image_data in enumerate(images_data):
                result = self.predict(image_data, top_k)
                result['batch_index'] = i
                results.append(result)
            
            batch_time = time.time() - start_time
            
            logger.info(f"Batch prediction completed for {len(images_data)} images in {batch_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint
        
        Returns:
            Service health status
        """
        try:
            # Test inference with dummy data
            dummy_image = Image.new('RGB', (64, 64), color='black')
            img_bytes = BytesIO()
            dummy_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            start_time = time.time()
            result = self.predict(img_bytes, top_k=1)
            health_check_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'device': str(self.device),
                'model_classes': self.config['num_classes'],
                'health_check_time_ms': round(health_check_time * 1000, 2),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }


# Global inference service instance
inference_service = None


def init_service(model_path: str = "/opt/ml/model/model_for_inference.pth"):
    """
    Initialize the inference service (called by ModelArts)
    
    Args:
        model_path: Path to model file
    """
    global inference_service
    try:
        inference_service = ARSLInferenceService(model_path)
        logger.info("Inference service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize inference service: {e}")
        return False


def predict(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function for ModelArts inference
    
    Args:
        request_data: Request data from API Gateway
        
    Returns:
        Prediction response
    """
    global inference_service
    
    try:
        if inference_service is None:
            return {
                'error': 'Inference service not initialized',
                'status': 'error'
            }
        
        # Extract image data from request
        if 'image' in request_data:
            # Base64 encoded image
            image_b64 = request_data['image']
            image_data = base64.b64decode(image_b64)
        elif 'images' in request_data:
            # Batch prediction
            images_b64 = request_data['images']
            images_data = [base64.b64decode(img_b64) for img_b64 in images_b64]
            
            top_k = request_data.get('top_k', 3)
            return {
                'results': inference_service.batch_predict(images_data, top_k),
                'status': 'success'
            }
        else:
            return {
                'error': 'No image data provided',
                'status': 'error'
            }
        
        # Single image prediction
        top_k = request_data.get('top_k', 3)
        result = inference_service.predict(image_data, top_k)
        
        return {
            'result': result,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }


def health():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    global inference_service
    
    if inference_service is None:
        return {
            'status': 'unhealthy',
            'error': 'Inference service not initialized'
        }
    
    return inference_service.health_check()


# For local testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ARSL Inference Service')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--test_image', type=str, help='Path to test image')
    
    args = parser.parse_args()
    
    # Initialize service
    if init_service(args.model_path):
        print("Inference service initialized successfully")
        
        # Test with image if provided
        if args.test_image:
            with open(args.test_image, 'rb') as f:
                image_data = f.read()
            
            result = inference_service.predict(image_data)
            print(f"Prediction result: {json.dumps(result, indent=2)}")
        
        # Health check
        health_status = health()
        print(f"Health status: {json.dumps(health_status, indent=2)}")
    else:
        print("Failed to initialize inference service")