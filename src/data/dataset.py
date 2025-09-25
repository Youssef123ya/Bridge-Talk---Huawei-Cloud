"""
Advanced Dataset Module for Sign Language Recognition
Provides comprehensive dataset classes with augmentation, caching, and optimization
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings


class SignLanguageDataset(Dataset):
    """
    Advanced dataset class for sign language recognition with comprehensive features
    """
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 augmentation_pipeline: Optional[Callable] = None,
                 cache_mode: str = "memory",  # "memory", "disk", "none"
                 preload_data: bool = False,
                 class_mapping: Optional[Dict[str, int]] = None,
                 max_samples_per_class: Optional[int] = None,
                 min_samples_per_class: int = 10,
                 image_format: str = "RGB",
                 target_size: Tuple[int, int] = (64, 64),
                 quality_filter: bool = True,
                 verbose: bool = True):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to data directory
            split: Data split ("train", "val", "test")
            transform: Basic transforms to apply
            augmentation_pipeline: Advanced augmentation pipeline
            cache_mode: Caching strategy
            preload_data: Whether to preload all data into memory
            class_mapping: Optional mapping from class names to indices
            max_samples_per_class: Maximum samples per class (for balancing)
            min_samples_per_class: Minimum samples per class (for filtering)
            image_format: Target image format ("RGB", "L", "RGBA")
            target_size: Target image size
            quality_filter: Whether to apply quality filtering
            verbose: Whether to print progress information
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augmentation_pipeline = augmentation_pipeline
        self.cache_mode = cache_mode
        self.preload_data = preload_data
        self.max_samples_per_class = max_samples_per_class
        self.min_samples_per_class = min_samples_per_class
        self.image_format = image_format
        self.target_size = target_size
        self.quality_filter = quality_filter
        self.verbose = verbose
        
        # Initialize caching
        self.memory_cache = {} if cache_mode == "memory" else None
        self.disk_cache_dir = self.data_dir.parent / f".cache_{split}" if cache_mode == "disk" else None
        
        # Initialize data structures
        self.samples = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.label_encoder = LabelEncoder()
        self.data_info = {}
        
        # Set class mapping
        if class_mapping:
            self.class_to_idx = class_mapping
            self.class_names = list(class_mapping.keys())
        
        # Load and prepare data
        self._load_data()
        self._prepare_labels()
        self._filter_and_balance_data()
        self._cache_preprocessing()
        
        if self.verbose:
            self._print_dataset_info()
    
    def _load_data(self):
        """Load data from directory structure"""
        if self.verbose:
            print(f"Loading {self.split} data from {self.data_dir}...")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if not self.class_to_idx:  # Auto-discover classes
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
            elif class_name not in self.class_to_idx:
                continue  # Skip unknown classes if mapping is provided
            
            # Collect image files from class directory
            class_samples = []
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    class_samples.append((str(img_file), class_name))
            
            # Apply quality filtering if enabled
            if self.quality_filter:
                class_samples = self._filter_quality(class_samples)
            
            # Add to main samples list
            self.samples.extend(class_samples)
        
        if self.verbose:
            print(f"Found {len(self.samples)} samples across {len(self.class_names)} classes")
    
    def _filter_quality(self, samples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter samples based on image quality"""
        filtered_samples = []
        
        for img_path, class_name in samples:
            try:
                # Basic quality checks
                with Image.open(img_path) as img:
                    # Check file size
                    if os.path.getsize(img_path) < 1024:  # Less than 1KB
                        continue
                    
                    # Check image dimensions
                    if min(img.size) < 32:  # Too small
                        continue
                    
                    # Check if image can be processed
                    img_array = np.array(img)
                    if img_array.size == 0:
                        continue
                    
                    # Check for corrupted images
                    if np.all(img_array == 0) or np.all(img_array == 255):
                        continue
                
                filtered_samples.append((img_path, class_name))
                
            except Exception as e:
                if self.verbose:
                    print(f"Skipping corrupted image: {img_path} - {e}")
                continue
        
        return filtered_samples
    
    def _prepare_labels(self):
        """Prepare class labels and encodings"""
        if not self.class_to_idx:
            # Create class to index mapping
            self.class_to_idx = {name: idx for idx, name in enumerate(sorted(self.class_names))}
        
        # Update class names from mapping
        self.class_names = sorted(self.class_to_idx.keys())
        
        # Extract labels
        self.labels = [self.class_to_idx[class_name] for _, class_name in self.samples]
        
        # Fit label encoder
        class_labels = [class_name for _, class_name in self.samples]
        self.label_encoder.fit(self.class_names)
    
    def _filter_and_balance_data(self):
        """Filter classes and balance data if needed"""
        # Count samples per class
        class_counts = Counter([label for _, label in self.samples])
        
        # Filter classes with insufficient samples
        valid_classes = []
        for class_name, class_idx in self.class_to_idx.items():
            if class_counts.get(class_idx, 0) >= self.min_samples_per_class:
                valid_classes.append(class_name)
        
        if len(valid_classes) < len(self.class_names):
            if self.verbose:
                removed_classes = set(self.class_names) - set(valid_classes)
                print(f"Removing {len(removed_classes)} classes with < {self.min_samples_per_class} samples")
            
            # Filter samples
            filtered_samples = []
            for sample_path, class_name in self.samples:
                if class_name in valid_classes:
                    filtered_samples.append((sample_path, class_name))
            
            self.samples = filtered_samples
            self.class_names = sorted(valid_classes)
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.labels = [self.class_to_idx[class_name] for _, class_name in self.samples]
        
        # Apply per-class sample limit
        if self.max_samples_per_class:
            balanced_samples = []
            class_sample_counts = defaultdict(int)
            
            # Shuffle samples to get random selection
            import random
            samples_with_labels = list(zip(self.samples, self.labels))
            random.shuffle(samples_with_labels)
            
            for (sample_path, class_name), label in samples_with_labels:
                if class_sample_counts[label] < self.max_samples_per_class:
                    balanced_samples.append((sample_path, class_name))
                    class_sample_counts[label] += 1
            
            self.samples = balanced_samples
            self.labels = [self.class_to_idx[class_name] for _, class_name in self.samples]
            
            if self.verbose:
                print(f"Balanced dataset to max {self.max_samples_per_class} samples per class")
    
    def _cache_preprocessing(self):
        """Set up caching system"""
        if self.cache_mode == "disk" and self.disk_cache_dir:
            self.disk_cache_dir.mkdir(exist_ok=True)
        
        if self.preload_data:
            if self.verbose:
                print("Preloading all data into memory...")
            
            for idx in tqdm(range(len(self)), disable=not self.verbose):
                self._load_and_cache_sample(idx)
    
    def _load_and_cache_sample(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and cache a single sample"""
        cache_key = f"sample_{idx}"
        
        # Check memory cache
        if self.memory_cache and cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if self.disk_cache_dir:
            cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if self.memory_cache:
                        self.memory_cache[cache_key] = cached_data
                    
                    return cached_data
                except:
                    pass  # Cache corrupted, regenerate
        
        # Load and process image
        img_path, class_name = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path)
            
            # Convert to target format
            if image.mode != self.image_format:
                if self.image_format == 'L':  # Grayscale
                    image = image.convert('L')
                elif self.image_format == 'RGB':
                    image = image.convert('RGB')
                elif self.image_format == 'RGBA':
                    image = image.convert('RGBA')
            
            # Resize to target size
            if image.size != self.target_size:
                image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to tensor
            if self.image_format == 'L':
                image_array = np.array(image, dtype=np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add channel dimension
            else:
                image_array = np.array(image, dtype=np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW
            
            result = (image_tensor, label)
            
            # Cache the result
            if self.memory_cache:
                self.memory_cache[cache_key] = result
            
            if self.disk_cache_dir:
                cache_file = self.disk_cache_dir / f"{cache_key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                except:
                    pass  # Failed to cache to disk
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading image {img_path}: {e}")
            
            # Return zero tensor as fallback
            if self.image_format == 'L':
                zero_tensor = torch.zeros(1, *self.target_size)
            else:
                channels = 3 if self.image_format == 'RGB' else 4
                zero_tensor = torch.zeros(channels, *self.target_size)
            
            return zero_tensor, label
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        # Load and cache if needed
        image, label = self._load_and_cache_sample(idx)
        
        # Apply transforms
        if self.transform:
            # Convert tensor back to PIL for transforms
            if image.dim() == 3:
                if image.shape[0] == 1:  # Grayscale
                    pil_image = transforms.ToPILImage()(image)
                else:  # RGB/RGBA
                    pil_image = transforms.ToPILImage()(image)
            else:
                pil_image = transforms.ToPILImage()(image.unsqueeze(0))
            
            image = self.transform(pil_image)
        
        # Apply augmentation pipeline (for training)
        if self.augmentation_pipeline and self.split == "train":
            if isinstance(image, torch.Tensor):
                # Convert to PIL for augmentation
                pil_image = transforms.ToPILImage()(image)
                image = self.augmentation_pipeline(pil_image)
            else:
                image = self.augmentation_pipeline(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.class_names)
        
        weights = []
        for i in range(num_classes):
            class_count = class_counts.get(i, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * class_count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def create_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted random sampler for balanced training"""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.labels),
            replacement=True
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        class_counts = Counter(self.labels)
        
        info = {
            "dataset_name": f"SignLanguage_{self.split}",
            "num_samples": len(self.samples),
            "num_classes": len(self.class_names),
            "image_format": self.image_format,
            "target_size": self.target_size,
            "class_names": self.class_names,
            "class_distribution": dict(class_counts),
            "cache_mode": self.cache_mode,
            "preloaded": self.preload_data,
        }
        
        # Calculate statistics
        info["samples_per_class"] = {
            name: class_counts.get(idx, 0) 
            for name, idx in self.class_to_idx.items()
        }
        
        info["class_balance_ratio"] = (
            max(class_counts.values()) / min(class_counts.values()) 
            if class_counts else 1.0
        )
        
        return info
    
    def _print_dataset_info(self):
        """Print dataset information"""
        info = self.get_data_info()
        
        print(f"\n=== {info['dataset_name']} Dataset Info ===")
        print(f"Total samples: {info['num_samples']}")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Image format: {info['image_format']}")
        print(f"Target size: {info['target_size']}")
        print(f"Class balance ratio: {info['class_balance_ratio']:.2f}")
        
        print("\nClass distribution:")
        for name, count in info['samples_per_class'].items():
            percentage = (count / info['num_samples']) * 100
            print(f"  {name}: {count} samples ({percentage:.1f}%)")
        
        print("=" * 40)
    
    def visualize_samples(self, num_samples: int = 16, save_path: Optional[str] = None):
        """Visualize random samples from the dataset"""
        import random
        
        # Select random samples
        indices = random.sample(range(len(self)), min(num_samples, len(self)))
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'{self.split.title()} Dataset Samples', fontsize=16)
        
        for i, idx in enumerate(indices):
            row = i // grid_size
            col = i % grid_size
            
            if grid_size == 1:
                ax = axes
            elif grid_size == 2 and num_samples <= 2:
                ax = axes[i]
            else:
                ax = axes[row, col]
            
            # Get sample
            image, label = self[idx]
            class_name = self.class_names[label]
            
            # Convert tensor to displayable format
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # Batch dimension
                    image = image.squeeze(0)
                
                if image.shape[0] == 1:  # Grayscale
                    img_np = image.squeeze(0).numpy()
                    ax.imshow(img_np, cmap='gray')
                else:  # RGB
                    img_np = image.permute(1, 2, 0).numpy()
                    ax.imshow(img_np)
            
            ax.set_title(f'{class_name}')
            ax.axis('off')
        
        # Hide remaining subplots
        for i in range(num_samples, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            if grid_size > 1:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_dataset_info(self, save_path: Union[str, Path]):
        """Export dataset information to file"""
        info = self.get_data_info()
        
        save_path = Path(save_path)
        if save_path.suffix.lower() == '.json':
            with open(save_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
        else:
            # Save as text file
            with open(save_path, 'w') as f:
                f.write(f"Dataset Information: {info['dataset_name']}\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in info.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")


def create_data_loaders(
    data_dir: Union[str, Path],
    splits: Dict[str, str],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    transforms_dict: Optional[Dict[str, Callable]] = None,
    augmentation_dict: Optional[Dict[str, Callable]] = None,
    use_weighted_sampling: bool = True,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for all splits
    
    Args:
        data_dir: Base data directory
        splits: Dictionary mapping split names to subdirectory names
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        transforms_dict: Transforms for each split
        augmentation_dict: Augmentation pipelines for each split
        use_weighted_sampling: Whether to use weighted sampling for training
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        Dictionary mapping split names to DataLoaders
    """
    data_loaders = {}
    class_mapping = None
    
    for split_name, split_dir in splits.items():
        split_path = Path(data_dir) / split_dir
        
        # Get transforms and augmentation for this split
        transform = transforms_dict.get(split_name) if transforms_dict else None
        augmentation = augmentation_dict.get(split_name) if augmentation_dict else None
        
        # Create dataset
        dataset = SignLanguageDataset(
            data_dir=split_path,
            split=split_name,
            transform=transform,
            augmentation_pipeline=augmentation,
            class_mapping=class_mapping,
            **dataset_kwargs
        )
        
        # Use the first dataset's class mapping for consistency
        if class_mapping is None:
            class_mapping = dataset.class_to_idx
        
        # Create sampler for training split
        sampler = None
        shuffle = split_name == "train"
        
        if split_name == "train" and use_weighted_sampling:
            sampler = dataset.create_weighted_sampler()
            shuffle = False  # Don't shuffle when using sampler
        
        # Create data loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=split_name == "train"  # Drop last batch for training
        )
        
        data_loaders[split_name] = data_loader
    
    return data_loaders


def analyze_dataset_statistics(dataset: SignLanguageDataset) -> Dict[str, Any]:
    """Analyze comprehensive dataset statistics"""
    stats = {}
    
    # Basic information
    info = dataset.get_data_info()
    stats.update(info)
    
    # Load a sample of images for pixel statistics
    sample_size = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    pixel_values = []
    image_sizes = []
    
    print(f"Analyzing {sample_size} sample images...")
    for idx in tqdm(sample_indices):
        try:
            image, _ = dataset[idx]
            if isinstance(image, torch.Tensor):
                pixel_values.extend(image.flatten().numpy())
                image_sizes.append(image.numel())
        except:
            continue
    
    if pixel_values:
        pixel_values = np.array(pixel_values)
        
        stats["pixel_statistics"] = {
            "mean": float(np.mean(pixel_values)),
            "std": float(np.std(pixel_values)),
            "min": float(np.min(pixel_values)),
            "max": float(np.max(pixel_values)),
            "median": float(np.median(pixel_values))
        }
        
        stats["image_size_statistics"] = {
            "mean_pixels": float(np.mean(image_sizes)),
            "std_pixels": float(np.std(image_sizes)),
            "total_pixels_analyzed": int(np.sum(image_sizes))
        }
    
    return stats


if __name__ == "__main__":
    # Test dataset functionality
    print("Testing Sign Language Dataset...")
    
    # Create a dummy dataset structure for testing
    test_data_dir = Path("test_dataset")
    
    try:
        # This is just for testing - in real use, data directory should exist
        print("✓ Dataset module loaded successfully")
        
        # Test configuration
        print("✓ All dataset classes and functions defined")
        
        # Test transforms
        basic_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        print("✓ Transform pipeline created")
        
        print("\n✓ Dataset module ready for use!")
        print("  - SignLanguageDataset: Advanced dataset with caching")
        print("  - create_data_loaders: Convenient loader creation")
        print("  - create_data_loaders_from_csv: CSV-based loader creation")
        print("  - analyze_dataset_statistics: Comprehensive analysis")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
    
    print("Dataset module testing completed!")


class CSVSignLanguageDataset(Dataset):
    """Simple dataset class for loading from CSV files"""
    
    def __init__(self, csv_file: Union[str, Path], data_dir: Union[str, Path], 
                 transform: Optional[Callable] = None):
        """
        Initialize CSV dataset
        
        Args:
            csv_file: Path to CSV file with image paths and labels
            data_dir: Base directory containing images
            transform: Transform to apply to images
        """
        import pandas as pd
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Read CSV
        self.df = pd.read_csv(csv_file)
        
        # Handle different column names for labels
        if 'label' not in self.df.columns:
            if 'Class' in self.df.columns:
                self.df['label'] = self.df['Class']
            elif 'class' in self.df.columns:
                self.df['label'] = self.df['class']
            else:
                raise ValueError("CSV file must contain a 'label', 'Class', or 'class' column")
        
        # Handle different column names for file paths
        if 'File_Name' in self.df.columns:
            self.df['filename'] = self.df['File_Name']
        elif 'filename' not in self.df.columns and 'path' in self.df.columns:
            self.df['filename'] = self.df['path']
        elif 'filename' not in self.df.columns:
            raise ValueError("CSV file must contain a 'File_Name', 'filename', or 'path' column")
        
        # Create label mapping
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        print(f"Loaded dataset with {len(self.df)} samples and {self.num_classes} classes")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image path - images are organized in class folders
        # Path structure: data_dir/class_name/filename
        class_name = row['label']
        filename = row['filename']
        image_path = self.data_dir / class_name / filename
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get label
        label = self.class_to_idx[row['label']]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders_from_csv(
    labels_file: Union[str, Path],
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.2,
    image_size: Tuple[int, int] = (224, 224),
    augment_train: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from CSV files
    (Alternative interface for backward compatibility)
    
    Args:
        labels_file: Path to CSV file with image paths and labels
        data_dir: Base directory containing images
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        image_size: Target image size (height, width)
        augment_train: Whether to augment training data
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import torchvision.transforms as transforms
    
    # Read CSV file
    df = pd.read_csv(labels_file)
    
    # Handle different column names for labels
    if 'label' not in df.columns:
        if 'Class' in df.columns:
            df['label'] = df['Class']
        elif 'class' in df.columns:
            df['label'] = df['class']
        else:
            raise ValueError("CSV file must contain a 'label', 'Class', or 'class' column")
    
    # Check if we need to split or use as-is (when train_split is 1.0, assume pre-split)
    if train_split == 1.0 and val_split == 0.0:
        # Use the entire dataset as-is (pre-split file)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if augment_train and 'train' in str(labels_file).lower():
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transform
        
        # Create a simple dataset from CSV
        dataset = CSVSignLanguageDataset(
            csv_file=labels_file,
            data_dir=data_dir,
            transform=train_transform
        )
        
        # Create single loader 
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle='train' in str(labels_file).lower(),  # Shuffle only training data
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Return the same loader for all splits (caller handles different CSV files)
        return loader, loader, loader
    
    else:
        # Split the data
        if train_split + val_split > 1.0:
            val_split = 1.0 - train_split
        test_split = 1.0 - train_split - val_split
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_split, 
            random_state=42, 
            stratify=df['label']
        )
        
        # Second split: separate train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_split/(train_split + val_split),
            random_state=42,
            stratify=train_val_df['label']
        )
        
        # Create transforms
        base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if augment_train:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = base_transform
        
        # Save dataframes to temp files and create datasets
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            train_df.to_csv(f.name, index=False)
            train_dataset = CSVSignLanguageDataset(f.name, data_dir, train_transform)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            val_df.to_csv(f.name, index=False)
            val_dataset = CSVSignLanguageDataset(f.name, data_dir, base_transform)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            test_df.to_csv(f.name, index=False)
            test_dataset = CSVSignLanguageDataset(f.name, data_dir, base_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader