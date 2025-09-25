import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import random

class AdvancedAugmentation:
    """Advanced augmentation strategies for Arabic Sign Language data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = tuple(config.get('image_size', [224, 224]))

    def get_training_transforms(self) -> A.Compose:
        """Get comprehensive training augmentations using Albumentations"""

        transforms_list = [
            # Geometric transformations
            A.Resize(self.image_size[0], self.image_size[1]),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=self.config.get('rotation_range', 15), p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=0,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),

            # Perspective and distortion
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2),

            # Color and lighting
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # Noise and blur
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),

            # Occlusion and cutout
            A.CoarseDropout(
                max_holes=2,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            A.Cutout(num_holes=2, max_h_size=20, max_w_size=20, p=0.2),

            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]

        return A.Compose(transforms_list)

    def get_validation_transforms(self) -> A.Compose:
        """Get validation transforms (no augmentation)"""

        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def get_tta_transforms(self, num_augmentations: int = 5) -> List[A.Compose]:
        """Get Test Time Augmentation transforms"""

        tta_transforms = []

        for i in range(num_augmentations):
            transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Rotate(limit=5, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
            tta_transforms.append(transform)

        return tta_transforms

class SignLanguageSpecificAugmentation:
    """Augmentations specifically designed for sign language recognition"""

    @staticmethod
    def hand_region_focus(image: np.ndarray, bbox_expand_ratio: float = 0.1) -> np.ndarray:
        """Focus on hand regions by detecting and cropping around hands"""
        # This would integrate with hand detection models
        # For now, we'll implement a center crop with slight randomization
        h, w = image.shape[:2]

        # Center crop with random offset
        crop_size = min(h, w)
        start_x = max(0, (w - crop_size) // 2 + random.randint(-20, 20))
        start_y = max(0, (h - crop_size) // 2 + random.randint(-20, 20))

        end_x = min(w, start_x + crop_size)
        end_y = min(h, start_y + crop_size)

        return image[start_y:end_y, start_x:end_x]

    @staticmethod
    def background_blur(image: np.ndarray, blur_strength: int = 15) -> np.ndarray:
        """Blur background while keeping foreground sharp"""
        # Simple implementation - would be enhanced with segmentation
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

        # Create a simple mask (center region is foreground)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=2)

        # Blend original and blurred based on mask
        result = image * mask + blurred * (1 - mask)
        return result.astype(np.uint8)

    @staticmethod
    def lighting_variation(image: np.ndarray) -> np.ndarray:
        """Simulate various lighting conditions"""
        # Convert to HSV for better lighting control
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Random brightness adjustment
        brightness_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        # Random shadow/highlight
        shadow_strength = random.uniform(0.8, 1.0)
        highlight_strength = random.uniform(1.0, 1.2)

        # Apply non-uniform lighting
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w // 2, h // 2

        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        distance_norm = distance / max_distance

        lighting_map = shadow_strength + (highlight_strength - shadow_strength) * (1 - distance_norm)
        lighting_map = np.expand_dims(lighting_map, axis=2)

        hsv[:, :, 2] = hsv[:, :, 2] * lighting_map[:, :, 0]
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

class AdaptiveAugmentation:
    """Adaptive augmentation based on class difficulty and training progress"""

    def __init__(self, num_classes: int = 32):
        self.num_classes = num_classes
        self.class_difficulty = np.ones(num_classes)  # Initialize with equal difficulty
        self.epoch_count = 0

    def update_class_difficulty(self, class_accuracies: Dict[int, float]):
        """Update class difficulty based on current accuracies"""
        for class_idx, accuracy in class_accuracies.items():
            # Higher difficulty for lower accuracy classes
            self.class_difficulty[class_idx] = 1.0 - accuracy

    def get_adaptive_transform(self, class_idx: int, base_transform: A.Compose) -> A.Compose:
        """Get augmentation strength based on class difficulty"""
        difficulty = self.class_difficulty[class_idx]

        # Increase augmentation for difficult classes
        augmentation_strength = 0.3 + 0.4 * difficulty

        # Modify probabilities based on difficulty
        adaptive_transforms = []
        for transform in base_transform.transforms:
            if hasattr(transform, 'p'):
                # Increase probability for difficult classes
                new_p = min(1.0, transform.p * (1 + augmentation_strength))
                transform.p = new_p
            adaptive_transforms.append(transform)

        return A.Compose(adaptive_transforms)

    def curriculum_learning_schedule(self, epoch: int, max_epochs: int) -> float:
        """Implement curriculum learning - start easy, get harder"""
        self.epoch_count = epoch

        # Start with 30% augmentation strength, increase to 100%
        progress = epoch / max_epochs
        augmentation_factor = 0.3 + 0.7 * progress

        return augmentation_factor

def create_augmentation_pipeline(config: Dict[str, Any], 
                               mode: str = 'train',
                               adaptive: bool = False) -> Any:
    """Factory function to create augmentation pipelines"""

    aug = AdvancedAugmentation(config)

    if mode == 'train':
        if adaptive:
            # Return adaptive augmentation object
            return AdaptiveAugmentation(config.get('num_classes', 32))
        else:
            return aug.get_training_transforms()
    elif mode == 'val':
        return aug.get_validation_transforms()
    elif mode == 'tta':
        return aug.get_tta_transforms()
    else:
        raise ValueError(f"Unknown mode: {mode}")

def visualize_augmentations(image_path: str, 
                          config: Dict[str, Any],
                          save_dir: str = 'data/analysis/augmentations',
                          num_examples: int = 9):
    """Visualize different augmentations applied to a sample image"""
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    # Load sample image
    image = np.array(Image.open(image_path).convert('RGB'))

    # Create augmentation pipeline
    aug_pipeline = create_augmentation_pipeline(config, mode='train')

    # Generate augmented examples
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Arabic Sign Language Data Augmentation Examples', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i == 0:
            # Show original image
            ax.imshow(image)
            ax.set_title('Original')
        else:
            # Apply augmentation
            augmented = aug_pipeline(image=image)['image']

            # Convert tensor to numpy for display
            if torch.is_tensor(augmented):
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                augmented = augmented.permute(1, 2, 0).numpy()
                augmented = (augmented * std + mean)
                augmented = np.clip(augmented, 0, 1)

            ax.imshow(augmented)
            ax.set_title(f'Augmented {i}')

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'augmentation_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Augmentation examples saved to {save_dir}/augmentation_examples.png")
