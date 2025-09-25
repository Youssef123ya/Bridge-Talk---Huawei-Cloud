"""
Advanced Data Augmentation Module for Sign Language Recognition
Provides comprehensive augmentation strategies for improving model robustness
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional, Callable, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SignLanguageAugmentation:
    """
    Comprehensive augmentation suite designed specifically for sign language recognition
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (64, 64),
                 augmentation_strength: str = 'medium',
                 preserve_hands: bool = True):
        """
        Initialize augmentation pipeline
        
        Args:
            input_size: Target image size (height, width)
            augmentation_strength: 'light', 'medium', 'heavy'
            preserve_hands: Whether to use hand-preserving augmentations
        """
        self.input_size = input_size
        self.augmentation_strength = augmentation_strength
        self.preserve_hands = preserve_hands
        
        # Define augmentation parameters based on strength
        self.params = self._get_augmentation_params()
        
        # Initialize transform pipelines
        self.basic_transforms = self._create_basic_transforms()
        self.geometric_transforms = self._create_geometric_transforms()
        self.photometric_transforms = self._create_photometric_transforms()
        self.albumentations_pipeline = self._create_albumentations_pipeline()
        
    def _get_augmentation_params(self) -> Dict:
        """Get augmentation parameters based on strength level"""
        params = {
            'light': {
                'rotation_range': 10,
                'zoom_range': 0.1,
                'brightness_range': 0.1,
                'contrast_range': 0.1,
                'noise_prob': 0.1,
                'blur_prob': 0.1,
                'elastic_prob': 0.0
            },
            'medium': {
                'rotation_range': 20,
                'zoom_range': 0.2,
                'brightness_range': 0.2,
                'contrast_range': 0.2,
                'noise_prob': 0.2,
                'blur_prob': 0.2,
                'elastic_prob': 0.1
            },
            'heavy': {
                'rotation_range': 30,
                'zoom_range': 0.3,
                'brightness_range': 0.3,
                'contrast_range': 0.3,
                'noise_prob': 0.3,
                'blur_prob': 0.3,
                'elastic_prob': 0.2
            }
        }
        return params[self.augmentation_strength]
    
    def _create_basic_transforms(self):
        """Create basic preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Single channel normalization
        ])
    
    def _create_geometric_transforms(self):
        """Create geometric transformation pipeline"""
        transforms_list = []
        
        # Rotation
        if self.params['rotation_range'] > 0:
            transforms_list.append(
                transforms.RandomRotation(
                    degrees=self.params['rotation_range'],
                    fill=0
                )
            )
        
        # Random affine transformations
        transforms_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(1-self.params['zoom_range'], 1+self.params['zoom_range']),
                shear=5 if self.augmentation_strength != 'light' else 0,
                fill=0
            )
        )
        
        # Horizontal flip (with caution for sign language)
        if not self.preserve_hands:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        return transforms.Compose(transforms_list)
    
    def _create_photometric_transforms(self):
        """Create photometric transformation pipeline"""
        return transforms.Compose([
            transforms.ColorJitter(
                brightness=self.params['brightness_range'],
                contrast=self.params['contrast_range'],
                saturation=0,  # Grayscale images
                hue=0
            ),
            transforms.RandomGrayscale(p=1.0)  # Ensure grayscale
        ])
    
    def _create_albumentations_pipeline(self):
        """Create advanced augmentation pipeline using Albumentations"""
        augmentations = []
        
        # Geometric transformations
        augmentations.extend([
            A.Rotate(
                limit=self.params['rotation_range'],
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=self.params['zoom_range'],
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            )
        ])
        
        # Photometric transformations
        augmentations.extend([
            A.RandomBrightnessContrast(
                brightness_limit=self.params['brightness_range'],
                contrast_limit=self.params['contrast_range'],
                p=0.5
            ),
            A.GaussNoise(
                noise_scale_factor=0.1,
                p=self.params['noise_prob']
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=self.params['blur_prob']
            )
        ])
        
        # Advanced transformations
        if self.params['elastic_prob'] > 0:
            augmentations.append(
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    p=self.params['elastic_prob']
                )
            )
        
        # Grid distortion for geometric robustness
        if self.augmentation_strength == 'heavy':
            augmentations.append(
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    p=0.1
                )
            )
        
        # Final resize and normalization
        augmentations.extend([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
        
        return A.Compose(augmentations)
    
    def apply_basic_augmentation(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Apply basic augmentation pipeline"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply geometric transforms
        image = self.geometric_transforms(image)
        
        # Apply photometric transforms
        image = self.photometric_transforms(image)
        
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485], std=[0.229])(image)
        
        return image
    
    def apply_advanced_augmentation(self, image: np.ndarray) -> torch.Tensor:
        """Apply advanced augmentation using Albumentations"""
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply albumentations pipeline
        augmented = self.albumentations_pipeline(image=image)
        return augmented['image']
    
    def create_training_pipeline(self) -> Callable:
        """Create training augmentation pipeline"""
        def training_transform(image):
            if isinstance(image, Image.Image):
                image = np.array(image)
            return self.apply_advanced_augmentation(image)
        
        return training_transform
    
    def create_validation_pipeline(self) -> Callable:
        """Create validation pipeline (minimal augmentation)"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def create_test_time_augmentation(self, n_augmentations: int = 5) -> List[Callable]:
        """Create multiple augmentation pipelines for test-time augmentation"""
        pipelines = []
        
        # Original image
        pipelines.append(self.create_validation_pipeline())
        
        # Various augmentation strengths
        for i in range(n_augmentations - 1):
            pipeline = transforms.Compose([
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            pipelines.append(pipeline)
        
        return pipelines


class AugmentationVisualizer:
    """Utility class for visualizing augmentation effects"""
    
    def __init__(self, augmentation_pipeline: SignLanguageAugmentation):
        self.pipeline = augmentation_pipeline
    
    def visualize_augmentations(self, 
                              image_path: str, 
                              n_samples: int = 8,
                              save_path: Optional[str] = None):
        """
        Visualize the effects of different augmentations
        
        Args:
            image_path: Path to sample image
            n_samples: Number of augmented samples to show
            save_path: Optional path to save visualization
        """
        # Load original image
        original_image = Image.open(image_path).convert('L')
        original_array = np.array(original_image)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle('Sign Language Data Augmentation Examples', fontsize=16)
        
        # Show original
        axes[0, 0].imshow(original_array, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Generate augmented samples
        for i in range(1, n_samples + 1):
            row = i // 3
            col = i % 3
            
            # Apply augmentation
            augmented_tensor = self.pipeline.apply_advanced_augmentation(original_array)
            
            # Convert back to displayable format
            augmented_array = augmented_tensor.squeeze().numpy()
            
            # Denormalize for visualization
            augmented_array = augmented_array * 0.229 + 0.485
            augmented_array = np.clip(augmented_array, 0, 1)
            
            axes[row, col].imshow(augmented_array, cmap='gray')
            axes[row, col].set_title(f'Augmented {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_augmentation_strengths(self, 
                                     image_path: str,
                                     save_path: Optional[str] = None):
        """Compare different augmentation strength levels"""
        # Load original image
        original_image = Image.open(image_path).convert('L')
        original_array = np.array(original_image)
        
        # Create pipelines with different strengths
        strengths = ['light', 'medium', 'heavy']
        pipelines = {}
        
        for strength in strengths:
            pipelines[strength] = SignLanguageAugmentation(
                augmentation_strength=strength
            )
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Augmentation Strength Comparison', fontsize=16)
        
        # Show original in first column
        axes[0, 0].imshow(original_array, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Show different strengths
        for i, strength in enumerate(strengths):
            col = i + 1
            
            # Generate two samples for each strength
            for row in range(2):
                augmented_tensor = pipelines[strength].apply_advanced_augmentation(
                    original_array.copy()
                )
                
                # Convert back to displayable format
                augmented_array = augmented_tensor.squeeze().numpy()
                augmented_array = augmented_array * 0.229 + 0.485
                augmented_array = np.clip(augmented_array, 0, 1)
                
                axes[row, col].imshow(augmented_array, cmap='gray')
                axes[row, col].set_title(f'{strength.capitalize()} {row+1}')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AugmentationAnalyzer:
    """Analyze the statistical properties of augmented data"""
    
    def __init__(self, augmentation_pipeline: SignLanguageAugmentation):
        self.pipeline = augmentation_pipeline
    
    def analyze_augmentation_distribution(self, 
                                        image_paths: List[str],
                                        n_augmentations_per_image: int = 10,
                                        save_path: Optional[str] = None):
        """
        Analyze the statistical distribution of augmented images
        
        Args:
            image_paths: List of sample image paths
            n_augmentations_per_image: Number of augmentations per image
            save_path: Path to save analysis results
        """
        original_stats = []
        augmented_stats = []
        
        for image_path in image_paths:
            # Load original image
            original_image = Image.open(image_path).convert('L')
            original_array = np.array(original_image)
            
            # Calculate original statistics
            original_stats.append({
                'mean': np.mean(original_array),
                'std': np.std(original_array),
                'min': np.min(original_array),
                'max': np.max(original_array)
            })
            
            # Generate augmented samples
            for _ in range(n_augmentations_per_image):
                augmented_tensor = self.pipeline.apply_advanced_augmentation(
                    original_array.copy()
                )
                
                # Convert back to numpy
                augmented_array = augmented_tensor.squeeze().numpy()
                
                augmented_stats.append({
                    'mean': np.mean(augmented_array),
                    'std': np.std(augmented_array),
                    'min': np.min(augmented_array),
                    'max': np.max(augmented_array)
                })
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Augmentation Statistical Analysis', fontsize=16)
        
        # Extract statistics
        orig_means = [stat['mean'] for stat in original_stats]
        orig_stds = [stat['std'] for stat in original_stats]
        aug_means = [stat['mean'] for stat in augmented_stats]
        aug_stds = [stat['std'] for stat in augmented_stats]
        
        # Plot mean distributions
        axes[0, 0].hist(orig_means, alpha=0.7, label='Original', bins=20)
        axes[0, 0].hist(aug_means, alpha=0.7, label='Augmented', bins=20)
        axes[0, 0].set_title('Mean Pixel Value Distribution')
        axes[0, 0].legend()
        
        # Plot std distributions
        axes[0, 1].hist(orig_stds, alpha=0.7, label='Original', bins=20)
        axes[0, 1].hist(aug_stds, alpha=0.7, label='Augmented', bins=20)
        axes[0, 1].set_title('Standard Deviation Distribution')
        axes[0, 1].legend()
        
        # Scatter plot: original vs augmented means
        axes[1, 0].scatter(orig_means * len(aug_means) // len(orig_means), 
                          aug_means, alpha=0.6)
        axes[1, 0].plot([min(orig_means), max(orig_means)], 
                       [min(orig_means), max(orig_means)], 'r--')
        axes[1, 0].set_xlabel('Original Mean')
        axes[1, 0].set_ylabel('Augmented Mean')
        axes[1, 0].set_title('Mean Preservation')
        
        # Box plot comparison
        data_to_plot = [orig_means, aug_means]
        axes[1, 1].boxplot(data_to_plot, labels=['Original', 'Augmented'])
        axes[1, 1].set_title('Mean Value Box Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary statistics
        print("=== Augmentation Analysis Summary ===")
        print(f"Original images analyzed: {len(original_stats)}")
        print(f"Augmented samples generated: {len(augmented_stats)}")
        print(f"\nOriginal Mean Statistics:")
        print(f"  Mean: {np.mean(orig_means):.3f} ± {np.std(orig_means):.3f}")
        print(f"  Range: [{np.min(orig_means):.3f}, {np.max(orig_means):.3f}]")
        print(f"\nAugmented Mean Statistics:")
        print(f"  Mean: {np.mean(aug_means):.3f} ± {np.std(aug_means):.3f}")
        print(f"  Range: [{np.min(aug_means):.3f}, {np.max(aug_means):.3f}]")


def create_augmentation_suite(config: Dict) -> Dict[str, SignLanguageAugmentation]:
    """
    Create a complete augmentation suite for different training phases
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of augmentation pipelines
    """
    suite = {}
    
    # Training pipeline - stronger augmentations
    suite['train'] = SignLanguageAugmentation(
        input_size=config.get('input_size', (64, 64)),
        augmentation_strength=config.get('train_strength', 'medium'),
        preserve_hands=config.get('preserve_hands', True)
    )
    
    # Validation pipeline - minimal augmentations
    suite['val'] = SignLanguageAugmentation(
        input_size=config.get('input_size', (64, 64)),
        augmentation_strength='light',
        preserve_hands=True
    )
    
    # Test pipeline - no augmentation
    suite['test'] = SignLanguageAugmentation(
        input_size=config.get('input_size', (64, 64)),
        augmentation_strength='light',
        preserve_hands=True
    )
    
    return suite


def create_augmentation_pipeline(config: Dict, mode: str = 'train') -> Callable:
    """
    Create augmentation pipeline based on configuration and mode
    
    Args:
        config: Configuration dictionary
        mode: Pipeline mode ('train', 'val', 'test', 'tta')
        
    Returns:
        Augmentation pipeline function
    """
    # Get augmentation parameters from config
    input_size = config.get('input_size', (64, 64))
    
    if mode == 'train':
        strength = config.get('augmentation_strength', 'medium')
        preserve_hands = config.get('preserve_hands', True)
        
        augmentor = SignLanguageAugmentation(
            input_size=input_size,
            augmentation_strength=strength,
            preserve_hands=preserve_hands
        )
        return augmentor.create_training_pipeline()
    
    elif mode == 'val' or mode == 'test':
        augmentor = SignLanguageAugmentation(
            input_size=input_size,
            augmentation_strength='light',
            preserve_hands=True
        )
        return augmentor.create_validation_pipeline()
    
    elif mode == 'tta':  # Test-time augmentation
        augmentor = SignLanguageAugmentation(
            input_size=input_size,
            augmentation_strength='light',
            preserve_hands=True
        )
        n_augmentations = config.get('tta_samples', 5)
        return augmentor.create_test_time_augmentation(n_augmentations)
    
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")


def visualize_augmentations(image_path: str, 
                          config: Dict,
                          output_dir: str = "outputs/augmentation_viz",
                          n_samples: int = 8) -> None:
    """
    Visualize augmentation effects and save results
    
    Args:
        image_path: Path to sample image
        config: Configuration dictionary
        output_dir: Directory to save visualizations
        n_samples: Number of augmented samples to generate
    """
    from pathlib import Path
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create augmentor
    augmentor = SignLanguageAugmentation(
        input_size=config.get('input_size', (64, 64)),
        augmentation_strength=config.get('augmentation_strength', 'medium'),
        preserve_hands=config.get('preserve_hands', True)
    )
    
    # Create visualizer
    visualizer = AugmentationVisualizer(augmentor)
    
    # Generate visualization
    save_path = output_path / "augmentation_samples.png"
    visualizer.visualize_augmentations(
        image_path=image_path,
        n_samples=n_samples,
        save_path=str(save_path)
    )
    
    # Generate strength comparison
    comparison_path = output_path / "strength_comparison.png"
    visualizer.compare_augmentation_strengths(
        image_path=image_path,
        save_path=str(comparison_path)
    )
    
    print(f"✓ Augmentation visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Sign Language Augmentation Pipeline...")
    
    # Create augmentation pipeline
    augmentor = SignLanguageAugmentation(
        input_size=(64, 64),
        augmentation_strength='medium',
        preserve_hands=True
    )
    
    print(f"✓ Augmentation pipeline created")
    print(f"  - Input size: {augmentor.input_size}")
    print(f"  - Strength: {augmentor.augmentation_strength}")
    print(f"  - Preserve hands: {augmentor.preserve_hands}")
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    try:
        # Test basic augmentation
        augmented_basic = augmentor.apply_basic_augmentation(Image.fromarray(dummy_image))
        print(f"✓ Basic augmentation: {augmented_basic.shape}")
        
        # Test advanced augmentation
        augmented_advanced = augmentor.apply_advanced_augmentation(dummy_image)
        print(f"✓ Advanced augmentation: {augmented_advanced.shape}")
        
        # Test pipeline creation
        train_pipeline = augmentor.create_training_pipeline()
        val_pipeline = augmentor.create_validation_pipeline()
        print(f"✓ Training and validation pipelines created")
        
        # Test augmentation suite
        config = {
            'input_size': (64, 64),
            'train_strength': 'medium',
            'preserve_hands': True
        }
        suite = create_augmentation_suite(config)
        print(f"✓ Complete augmentation suite created with {len(suite)} pipelines")
        
        # Test new functions
        test_config = {
            'input_size': (64, 64),
            'augmentation_strength': 'medium',
            'preserve_hands': True,
            'tta_samples': 5
        }
        
        train_pipeline = create_augmentation_pipeline(test_config, 'train')
        val_pipeline = create_augmentation_pipeline(test_config, 'val')
        print(f"✓ New pipeline creation functions work")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
    
    print("\nAugmentation module ready for use!")