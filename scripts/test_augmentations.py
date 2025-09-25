#!/usr/bin/env python3
"""
Test and visualize data augmentation strategies
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.augmentation import (
    SignLanguageAugmentation, 
    AugmentationVisualizer,
    AugmentationAnalyzer,
    create_augmentation_pipeline
)
from src.config.config import get_config

def test_augmentation_pipeline():
    """Test different augmentation strategies"""

    print("🎨 Testing Augmentation Pipeline")
    print("=" * 40)

    config = get_config()

    # Find a sample image
    data_dir = config.data.raw_data_dir
    sample_image = None

    # Try to find sample image
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break

    if not sample_image:
        print("❌ No sample images found in data directory")
        return False

    print(f"📸 Using sample image: {sample_image}")

    # Load image
    image = np.array(Image.open(sample_image).convert('RGB'))
    print(f"   Original size: {image.shape}")

    # Create different augmentation pipelines
    config_dict = config.to_dict()
    augmentations = {
        'training': create_augmentation_pipeline(config_dict, mode='train'),
        'validation': create_augmentation_pipeline(config_dict, mode='val'),
    }

    # Test each augmentation
    results = {}
    for aug_name, aug_pipeline in augmentations.items():
        print(f"\n🔄 Testing {aug_name} augmentation...")

        try:
            # Apply augmentation multiple times
            augmented_images = []
            for i in range(5):
                if aug_name == 'training':
                    # Training pipeline expects numpy array or PIL image
                    aug_result = aug_pipeline(image)
                    # Convert tensor back to numpy for visualization
                    if isinstance(aug_result, torch.Tensor):
                        aug_result = aug_result.squeeze().numpy()
                else:
                    # Validation pipeline is a torchvision transform that expects PIL
                    if isinstance(image, np.ndarray):
                        pil_image = Image.fromarray(image)
                    else:
                        pil_image = image
                    aug_result = aug_pipeline(pil_image)
                    if isinstance(aug_result, torch.Tensor):
                        aug_result = aug_result.squeeze().numpy()
                
                augmented_images.append(aug_result)

            results[aug_name] = augmented_images
            print(f"   ✅ {aug_name} augmentation successful")

        except Exception as e:
            print(f"   ❌ {aug_name} augmentation failed: {e}")
            results[aug_name] = None

    # Create visualization
    print("\n📊 Creating augmentation visualization...")
    create_augmentation_visualization(image, results, 'data/analysis/augmentation_test.png')

    # Test sign language specific augmentations
    print("\n🤟 Testing sign language specific augmentations...")
    try:
        sl_aug = SignLanguageAugmentation(
            input_size=(64, 64),
            augmentation_strength='medium',
            preserve_hands=True
        )

        # Test basic augmentation
        augmented_basic = sl_aug.apply_basic_augmentation(image)
        print(f"   ✅ Basic augmentation: {augmented_basic.shape}")

        # Test advanced augmentation
        image_array = np.array(image)
        augmented_advanced = sl_aug.apply_advanced_augmentation(image_array)
        print(f"   ✅ Advanced augmentation: {augmented_advanced.shape}")

        # Test pipeline creation
        train_pipeline = sl_aug.create_training_pipeline()
        print(f"   ✅ Training pipeline created")

    except Exception as e:
        print(f"   ❌ Sign language augmentations failed: {e}")

    # Test augmentation analysis
    print("\n🧠 Testing augmentation analysis...")
    try:
        analyzer = AugmentationAnalyzer(sl_aug)

        # Test with sample images (if they exist)
        sample_dir = Path("data/raw")
        sample_images = []
        
        # Find some sample images
        for class_dir in sample_dir.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        sample_images.append(str(img_file))
                        if len(sample_images) >= 3:
                            break
                if len(sample_images) >= 3:
                    break

        if sample_images:
            print(f"   ✅ Found {len(sample_images)} sample images for analysis")
        else:
            print("   ⚠️ No sample images found, skipping statistical analysis")

    except Exception as e:
        print(f"   ❌ Augmentation analysis failed: {e}")

    print("\n✅ Augmentation testing completed!")
    return True

def create_augmentation_visualization(original_image, results, save_path):
    """Create visualization of augmentation results"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create subplot grid
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('Arabic Sign Language - Augmentation Pipeline Test', fontsize=16)

    # Show original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Show training augmentations
    if results.get('training'):
        for i, aug_img in enumerate(results['training'][:5]):
            if torch.is_tensor(aug_img):
                aug_img = aug_img.numpy()
            
            # Handle different image formats
            if len(aug_img.shape) == 3:
                if aug_img.shape[0] in [1, 3]:  # CHW format
                    if aug_img.shape[0] == 1:  # Grayscale
                        aug_img = aug_img.squeeze(0)
                        axes[0, i+1].imshow(aug_img, cmap='gray')
                    else:  # RGB
                        aug_img = aug_img.transpose(1, 2, 0)  # CHW to HWC
                        axes[0, i+1].imshow(aug_img)
                else:  # Already HWC
                    axes[0, i+1].imshow(aug_img)
            else:  # Grayscale 2D
                axes[0, i+1].imshow(aug_img, cmap='gray')
            
            axes[0, i+1].set_title(f'Train Aug {i+1}')
            axes[0, i+1].axis('off')

    # Show validation augmentations (should be minimal)
    if results.get('validation'):
        for i, aug_img in enumerate(results['validation'][:5]):
            if torch.is_tensor(aug_img):
                aug_img = aug_img.numpy()
            
            # Handle different image formats
            if len(aug_img.shape) == 3:
                if aug_img.shape[0] in [1, 3]:  # CHW format
                    if aug_img.shape[0] == 1:  # Grayscale
                        aug_img = aug_img.squeeze(0)
                        axes[1, i+1].imshow(aug_img, cmap='gray')
                    else:  # RGB
                        aug_img = aug_img.transpose(1, 2, 0)  # CHW to HWC
                        axes[1, i+1].imshow(aug_img)
                else:  # Already HWC
                    axes[1, i+1].imshow(aug_img)
            else:  # Grayscale 2D
                axes[1, i+1].imshow(aug_img, cmap='gray')
            
            axes[1, i+1].set_title(f'Val Aug {i+1}')
            axes[1, i+1].axis('off')

    # Show sign language specific augmentations
    try:
        sl_aug = SignLanguageAugmentation(
            input_size=(64, 64),
            augmentation_strength='medium',
            preserve_hands=True
        )

        # Apply different augmentation strengths
        img_array = np.array(original_image)
        medium_aug = sl_aug.apply_advanced_augmentation(img_array.copy())
        axes[2, 0].imshow(medium_aug.squeeze(), cmap='gray')
        axes[2, 0].set_title('Medium Aug')
        axes[2, 0].axis('off')

        # Heavy augmentation
        heavy_aug = SignLanguageAugmentation(augmentation_strength='heavy')
        heavy_result = heavy_aug.apply_advanced_augmentation(img_array.copy())
        axes[2, 1].imshow(heavy_result.squeeze(), cmap='gray')
        axes[2, 1].set_title('Heavy Aug')
        axes[2, 1].axis('off')

        # Lighting variations
        for i in range(4):
            lighting_varied = sl_aug.lighting_variation(original_image)
            axes[2, i+2].imshow(lighting_varied)
            axes[2, i+2].set_title(f'Light {i+1}')
            axes[2, i+2].axis('off')

    except Exception as e:
        print(f"⚠️  Could not create SL-specific augmentations: {e}")

    # Remove empty subplots
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Augmentation visualization saved to {save_path}")

if __name__ == "__main__":
    success = test_augmentation_pipeline()
    sys.exit(0 if success else 1)
