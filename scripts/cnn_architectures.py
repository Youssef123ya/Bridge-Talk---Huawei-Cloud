import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any
from .base_model import BaseModel, ModelRegistry

@ModelRegistry.register('resnet18')
@ModelRegistry.register('resnet34')
@ModelRegistry.register('resnet50')
@ModelRegistry.register('resnet101')
@ModelRegistry.register('resnet152')
class ResNetClassifier(BaseModel):
    """ResNet-based classifier for Arabic Sign Language Recognition"""

    def __init__(self, 
                 num_classes: int = 32,
                 architecture: str = 'resnet50',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 fine_tune_layers: int = -1,
                 **kwargs):
        super().__init__(num_classes, **kwargs)

        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.fine_tune_layers = fine_tune_layers

        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif architecture == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet architecture: {architecture}")

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        # Custom classifier with improved architecture
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

        # Initialize classifier weights
        self._initialize_classifier()

        # Configure fine-tuning
        if fine_tune_layers > 0:
            self._setup_fine_tuning(fine_tune_layers)

    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _setup_fine_tuning(self, fine_tune_layers: int):
        """Setup fine-tuning by freezing/unfreezing specific layers"""
        # Freeze all backbone parameters first
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze last N layers
        backbone_children = list(self.feature_extractor.children())
        for layer in backbone_children[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"🔧 Fine-tuning enabled for last {fine_tune_layers} layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone"""
        return self.feature_extractor(x)

@ModelRegistry.register('efficientnet_b0')
@ModelRegistry.register('efficientnet_b1')
@ModelRegistry.register('efficientnet_b2')
@ModelRegistry.register('efficientnet_b3')
@ModelRegistry.register('efficientnet_b4')
class EfficientNetClassifier(BaseModel):
    """EfficientNet-based classifier for Arabic Sign Language Recognition"""

    def __init__(self,
                 num_classes: int = 32,
                 architecture: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 **kwargs):
        super().__init__(num_classes, **kwargs)

        self.architecture = architecture
        self.dropout_rate = dropout_rate

        # Load pretrained EfficientNet
        if architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif architecture == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            self.feature_dim = 1280
        elif architecture == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            self.feature_dim = 1408
        elif architecture == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            self.feature_dim = 1536
        elif architecture == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            self.feature_dim = 1792
        else:
            raise ValueError(f"Unsupported EfficientNet architecture: {architecture}")

        # Extract feature extractor (everything except classifier)
        self.feature_extractor = self.backbone.features
        self.avgpool = self.backbone.avgpool

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone"""
        return self.feature_extractor(x)

@ModelRegistry.register('mobilenet_v2')
@ModelRegistry.register('mobilenet_v3_small')
@ModelRegistry.register('mobilenet_v3_large')
class MobileNetClassifier(BaseModel):
    """MobileNet-based classifier for Arabic Sign Language Recognition"""

    def __init__(self,
                 num_classes: int = 32,
                 architecture: str = 'mobilenet_v2',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 **kwargs):
        super().__init__(num_classes, **kwargs)

        self.architecture = architecture
        self.dropout_rate = dropout_rate

        # Load pretrained MobileNet
        if architecture == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.feature_dim = 1280
            self.feature_extractor = self.backbone.features
        elif architecture == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.feature_dim = 576
            self.feature_extractor = self.backbone.features
        elif architecture == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            self.feature_dim = 960
            self.feature_extractor = self.backbone.features
        else:
            raise ValueError(f"Unsupported MobileNet architecture: {architecture}")

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone"""
        return self.feature_extractor(x)

@ModelRegistry.register('custom_cnn')
class CustomArASLCNN(BaseModel):
    """Custom CNN architecture designed specifically for Arabic Sign Language Recognition"""

    def __init__(self,
                 num_classes: int = 32,
                 input_channels: int = 3,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True,
                 **kwargs):
        super().__init__(num_classes, **kwargs)

        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        # Feature extraction blocks
        self.feature_extractor = nn.Sequential(
            # Block 1
            self._make_conv_block(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Block 2
            self._make_conv_block(64, 128, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            # Block 3
            self._make_conv_block(128, 256, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(256, 256, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            # Block 4
            self._make_conv_block(256, 512, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(512, 512, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),

            # Block 5
            self._make_conv_block(512, 512, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(512, 512, kernel_size=3, stride=1, padding=1),
            self._make_conv_block(512, 512, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(512)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """Create a convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)

        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)

        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone"""
        return self.feature_extractor(x)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)

        # Apply attention
        return x * attention

@ModelRegistry.register('vision_transformer')
class VisionTransformerClassifier(BaseModel):
    """Vision Transformer-based classifier for Arabic Sign Language Recognition"""

    def __init__(self,
                 num_classes: int = 32,
                 architecture: str = 'vit_b_16',
                 pretrained: bool = True,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(num_classes, **kwargs)

        self.architecture = architecture
        self.dropout_rate = dropout_rate

        # Load pretrained Vision Transformer
        if architecture == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            self.feature_dim = 768
        elif architecture == 'vit_b_32':
            self.backbone = models.vit_b_32(pretrained=pretrained)
            self.feature_dim = 768
        elif architecture == 'vit_l_16':
            self.backbone = models.vit_l_16(pretrained=pretrained)
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT architecture: {architecture}")

        # Extract feature extractor (everything except classification head)
        self.feature_extractor = nn.Sequential(
            self.backbone.conv_proj,
            self.backbone.encoder,
        )

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.backbone.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        batch_size = x.shape[0]
        class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        # Add positional encoding
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)

        # Pass through transformer encoder
        x = self.backbone.encoder.layers(x)
        x = self.backbone.encoder.ln(x)

        # Use class token for classification
        x = x[:, 0]

        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the transformer"""
        # Patch embedding
        x = self.backbone.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token and positional encoding
        batch_size = x.shape[0]
        class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)

        # Pass through transformer encoder
        x = self.backbone.encoder.layers(x)
        x = self.backbone.encoder.ln(x)

        return x[:, 0]  # Return class token

def get_model_comparison_info() -> Dict[str, Dict[str, Any]]:
    """Get comparison information for all available models"""

    comparison = {}

    for model_name in ModelRegistry.list_models():
        try:
            # Create a small version for analysis
            model = ModelRegistry.get_model(model_name, num_classes=32)

            comparison[model_name] = {
                'parameters': model.get_num_parameters(),
                'size_mb': model.get_num_parameters() * 4 / (1024 * 1024),
                'architecture_type': model.__class__.__name__,
                'suitable_for': []
            }

            # Add suitability recommendations
            params = model.get_num_parameters()
            if params < 5_000_000:
                comparison[model_name]['suitable_for'].extend(['mobile', 'edge_devices', 'fast_inference'])
            elif params < 25_000_000:
                comparison[model_name]['suitable_for'].extend(['balanced', 'general_purpose'])
            else:
                comparison[model_name]['suitable_for'].extend(['high_accuracy', 'research'])

            del model  # Clean up

        except Exception as e:
            comparison[model_name] = {'error': str(e)}

    return comparison
