#!/usr/bin/env python3
"""
Test Data Pipeline for Arabic Sign Language Recognition Project
This script tests the data loading pipeline.
"""

import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_data_loaders
from src.utils.helpers import setup_logging, get_device, set_seed
from src.config.config import get_config
import torch

def test_data_loaders():
    """Test the data loading pipeline"""
    print("🧪 Testing Data Pipeline")
    print("=" * 40)

    # Setup
    logger = setup_logging('logs/test_pipeline.log')
    config = get_config()
    device = get_device()
    set_seed(config.get('data.random_seed', 42))

    # Configuration
    labels_file = config.get('data.labels_file', 'ArSL_Data_Labels.csv')
    data_dir = config.get('data.raw_data_path', 'data/raw/')
    batch_size = config.get('data.batch_size', 32)
    image_size = tuple(config.get('data.image_size', [224, 224]))

    print(f"📁 Labels file: {labels_file}")
    print(f"📁 Data directory: {data_dir}")
    print(f"🖼️  Image size: {image_size}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎯 Device: {device}")

    try:
        # Create data loaders
        print("\n⏳ Creating data loaders...")
        start_time = time.time()

        train_loader, val_loader, test_loader = create_data_loaders(
            labels_file=labels_file,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=config.get('data.num_workers', 4),
            train_split=config.get('data.train_split', 0.7),
            val_split=config.get('data.val_split', 0.15),
            image_size=image_size,
            random_seed=config.get('data.random_seed', 42)
        )

        creation_time = time.time() - start_time
        print(f"✅ Data loaders created in {creation_time:.2f}s")

        # Test loading a batch from each set
        print("\n🧪 Testing batch loading...")

        loaders = [
            ("Training", train_loader),
            ("Validation", val_loader), 
            ("Test", test_loader)
        ]

        for name, loader in loaders:
            try:
                start_time = time.time()
                batch = next(iter(loader))
                load_time = time.time() - start_time

                images, labels = batch
                print(f"✅ {name}:")
                print(f"   Batch shape: {images.shape}")
                print(f"   Labels shape: {labels.shape}")
                print(f"   Load time: {load_time:.3f}s")
                print(f"   Memory usage: {images.element_size() * images.nelement() / 1024**2:.1f} MB")

                # Check data ranges
                print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"   Label range: [{labels.min()}, {labels.max()}]")
                print(f"   Unique labels in batch: {len(torch.unique(labels))}")

            except Exception as e:
                print(f"❌ {name} loader failed: {e}")
                return False

        # Test multiple batches
        print("\n🔄 Testing multiple batch loading...")
        start_time = time.time()
        batch_count = 5

        for i, batch in enumerate(train_loader):
            if i >= batch_count:
                break
            images, labels = batch

        multi_batch_time = time.time() - start_time
        avg_time = multi_batch_time / batch_count

        print(f"✅ Loaded {batch_count} batches")
        print(f"   Total time: {multi_batch_time:.2f}s")
        print(f"   Average per batch: {avg_time:.3f}s")
        print(f"   Estimated epoch time: {avg_time * len(train_loader) / 60:.1f} minutes")

        # Memory test
        print("\n💾 Testing GPU memory usage...")
        if device == 'cuda':
            images = images.to(device)
            labels = labels.to(device)

            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2

            print(f"   GPU memory allocated: {memory_allocated:.1f} MB")
            print(f"   GPU memory cached: {memory_cached:.1f} MB")

            # Clear memory
            del images, labels
            torch.cuda.empty_cache()

        # Test data splits integrity
        print("\n📊 Checking data split integrity...")

        # Load class mapping
        import json
        with open('data/processed/class_mapping.json', 'r') as f:
            class_mapping = json.load(f)

        print(f"   Number of classes: {class_mapping['num_classes']}")
        print(f"   Classes: {', '.join(class_mapping['classes'][:10])}{'...' if len(class_mapping['classes']) > 10 else ''}")

        # Check if all classes are represented
        train_classes = set()
        for batch in train_loader:
            _, labels = batch
            train_classes.update(labels.numpy())
            if len(train_classes) == class_mapping['num_classes']:
                break

        print(f"   Classes in training set: {len(train_classes)}/{class_mapping['num_classes']}")

        if len(train_classes) == class_mapping['num_classes']:
            print("✅ All classes represented in training data")
        else:
            missing_classes = set(range(class_mapping['num_classes'])) - train_classes
            print(f"⚠️  Missing classes in training data: {missing_classes}")

        print("\n🎉 Data pipeline test completed successfully!")
        print("\n📝 Next steps:")
        print("1. Data pipeline is working correctly")
        print("2. All data splits have been created")
        print("3. Ready to proceed to Phase 2: Model Development")
        print("4. You can now start training models")

        return True

    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_data_pipeline()

    if success:
        print("\n✅ Phase 1 Environment Setup Completed Successfully!")
        print("\n🚀 Ready for Phase 2: Data Pipeline Development")
    else:
        print("\n❌ Phase 1 setup has issues. Please check the errors above.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
