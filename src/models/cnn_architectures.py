"""
CNN Architectures Module for Arabic Sign Language Recognition
Provides various convolutional neural network architectures optimized for sign language classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights

from .base_model import BaseSignLanguageModel


class BasicCNN(BaseSignLanguageModel):
    """
    Basic CNN architecture for sign language recognition
    Simple and fast model suitable for initial testing
    """
    
    def __init__(self, num_classes: int = 32, dropout_rate: float = 0.5, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate feature size dynamically
        self.feature_size = self._get_conv_output_size()
        
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        # Assuming input size is 224x224 (standard image size)
        size = 224
        # After conv1 + pool: 224 -> 112
        size = size // 2
        # After conv2 + pool: 112 -> 56
        size = size // 2
        # After conv3 + pool: 56 -> 28
        size = size // 2
        
        return 128 * size * size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'BasicCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'description': 'Simple 3-layer CNN for basic sign language recognition'
        }


class AdvancedCNN(BaseSignLanguageModel):
    """
    Advanced CNN architecture with residual connections and batch normalization
    More sophisticated model with better performance
    """
    
    def __init__(self, num_classes: int = 32, dropout_rate: float = 0.3, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.dropout_rate = dropout_rate
        
        # First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128, stride=2)
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.res_block3 = self._make_residual_block(256, 512, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_residual_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'AdvancedCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'description': 'Advanced CNN with residual connections and batch normalization'
        }


class ResNet50SignLanguage(BaseSignLanguageModel):
    """
    ResNet50 architecture adapted for sign language recognition
    Uses transfer learning from ImageNet pretrained weights
    """
    
    def __init__(self, num_classes: int = 32, pretrained: bool = True, dropout_rate: float = 0.5, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # Load ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Freeze early layers if using pretrained weights
        if pretrained:
            self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze early layers for transfer learning"""
        # Freeze first two residual blocks
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'ResNet50',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'description': 'ResNet50 architecture with transfer learning for sign language recognition'
        }


class MobileNetV2SignLanguage(BaseSignLanguageModel):
    """
    MobileNetV2 architecture for efficient sign language recognition
    Lightweight model suitable for mobile deployment
    """
    
    def __init__(self, num_classes: int = 32, pretrained: bool = True, dropout_rate: float = 0.2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # Load MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'MobileNetV2',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'description': 'MobileNetV2 lightweight architecture for efficient sign language recognition'
        }


class EfficientNetB0SignLanguage(BaseSignLanguageModel):
    """
    EfficientNet-B0 architecture for sign language recognition
    Balanced efficiency and accuracy
    """
    
    def __init__(self, num_classes: int = 32, pretrained: bool = True, dropout_rate: float = 0.3, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # Load EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'EfficientNet-B0',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'description': 'EfficientNet-B0 architecture balancing efficiency and accuracy'
        }


# Architecture registry for easy access
CNN_ARCHITECTURES = {
    'cnn_basic': BasicCNN,
    'cnn_advanced': AdvancedCNN,
    'resnet50': ResNet50SignLanguage,
    'mobilenet_v2': MobileNetV2SignLanguage,
    'efficientnet_b0': EfficientNetB0SignLanguage,
}


def get_architecture(name: str) -> type:
    """
    Get architecture class by name
    
    Args:
        name: Architecture name
        
    Returns:
        Architecture class
    """
    if name not in CNN_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(CNN_ARCHITECTURES.keys())}")
    
    return CNN_ARCHITECTURES[name]


def list_architectures() -> List[str]:
    """Get list of available architectures"""
    return list(CNN_ARCHITECTURES.keys())


def get_architecture_info(name: str) -> Dict[str, Any]:
    """
    Get information about an architecture without instantiating it
    
    Args:
        name: Architecture name
        
    Returns:
        Architecture information dictionary
    """
    if name not in CNN_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}")
    
    arch_class = CNN_ARCHITECTURES[name]
    
    # Create temporary instance to get info
    temp_instance = arch_class(num_classes=32)
    info = temp_instance.get_model_info()
    
    return info