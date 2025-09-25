#!/usr/bin/env python3
"""
Simple Data Pipeline Test for Arabic Sign Language Recognition Project
This script tests basic data loading without requiring the full project structure.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import time
from collections import Counter

class SimpleArSLDataset(Dataset):
    """Simple dataset class for testing"""
    
    def __init__(self, csv_file, data_dir, transform=None, max_samples=None):
        self.data = pd.read_csv(csv_file)
        if max_samples:
            self.data = self.data.head(max_samples)
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Create class to index mapping
        self.classes = sorted(self.data['Class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Filter out samples with missing files
        self.valid_samples = []
        print(f"🔍 Checking {len(self.data)} samples for valid files...")
        
        for idx, row in self.data.iterrows():
            file_path = self.data_dir / row['Class'] / row['File_Name']
            if file_path.exists():
                self.valid_samples.append((str(file_path), self.class_to_idx[row['Class']]))
        
        print(f"✅ Found {len(self.valid_samples)} valid samples out of {len(self.data)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        file_path, label = self.valid_samples[idx]
        
        try:
            # Load image
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            
            # Convert to tensor
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(1, 64, 64)
            return dummy_image, 0

def test_basic_data_loading():
    """Test basic data loading functionality"""
    print("🧪 Testing Basic Data Loading")
    print("=" * 40)
    
    # Configuration
    csv_file = 'data/ArSL_Data_Labels.csv'
    data_dir = 'data/raw'
    batch_size = 16
    max_samples = 1000  # Test with smaller subset first
    
    print(f"📁 CSV file: {csv_file}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔢 Max samples for testing: {max_samples}")
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    try:
        # Create dataset
        print("\n⏳ Creating dataset...")
        start_time = time.time()
        
        dataset = SimpleArSLDataset(csv_file, data_dir, max_samples=max_samples)
        
        creation_time = time.time() - start_time
        print(f"✅ Dataset created in {creation_time:.2f}s")
        print(f"   Total samples: {len(dataset):,}")
        print(f"   Number of classes: {len(dataset.classes)}")
        print(f"   Classes: {', '.join(dataset.classes[:10])}{'...' if len(dataset.classes) > 10 else ''}")
        
        # Create data loader
        print("\n⏳ Creating data loader...")
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        print(f"✅ Data loader created")
        print(f"   Number of batches: {len(dataloader)}")
        
        # Test loading a batch
        print("\n🧪 Testing batch loading...")
        start_time = time.time()
        
        batch = next(iter(dataloader))
        images, labels = batch
        
        load_time = time.time() - start_time
        
        print(f"✅ Batch loaded successfully!")
        print(f"   Batch loading time: {load_time:.3f}s")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image data type: {images.dtype}")
        print(f"   Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Labels in batch: {labels.tolist()}")
        print(f"   Unique labels in batch: {len(torch.unique(labels))}")
        
        # Test multiple batches
        print("\n🔄 Testing multiple batch loading...")
        batch_times = []
        batch_count = 5
        
        for i, (images, labels) in enumerate(dataloader):
            if i >= batch_count:
                break
            
            start_time = time.time()
            # Simulate some processing
            _ = images.mean()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
        
        avg_batch_time = np.mean(batch_times)
        print(f"✅ Loaded {batch_count} batches")
        print(f"   Average batch processing time: {avg_batch_time:.4f}s")
        print(f"   Estimated time for full epoch: {avg_batch_time * len(dataloader):.2f}s")
        
        # Test class distribution
        print("\n📊 Checking class distribution in sample...")
        all_labels = []
        sample_batches = min(10, len(dataloader))
        
        for i, (_, labels) in enumerate(dataloader):
            if i >= sample_batches:
                break
            all_labels.extend(labels.tolist())
        
        label_counts = Counter(all_labels)
        print(f"   Sampled {len(all_labels)} examples from {sample_batches} batches")
        print(f"   Classes in sample: {len(label_counts)}")
        
        # Show top 5 most common classes
        most_common = label_counts.most_common(5)
        for class_idx, count in most_common:
            class_name = dataset.classes[class_idx]
            print(f"     {class_name}: {count} samples")
        
        # Memory usage test
        print("\n💾 Testing memory usage...")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            images = images.to(device)
            memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"   GPU memory used: {memory_used:.1f} MB")
            torch.cuda.empty_cache()
        else:
            print("   GPU not available - testing with CPU")
            # Estimate CPU memory usage
            memory_mb = images.element_size() * images.nelement() / 1024**2
            print(f"   Estimated CPU memory per batch: {memory_mb:.1f} MB")
        
        print("\n🎉 Basic data pipeline test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integrity():
    """Test data file integrity"""
    print("\n🔍 Testing Data Integrity")
    print("=" * 30)
    
    csv_file = 'data/ArSL_Data_Labels.csv'
    data_dir = Path('data/raw')
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"✅ CSV loaded: {len(df)} rows")
        
        # Check for missing files
        missing_files = []
        corrupted_files = []
        
        print("🔍 Checking file integrity (sampling 100 files)...")
        
        # Sample 100 files for quick check
        sample_df = df.sample(min(100, len(df)))
        
        for _, row in sample_df.iterrows():
            file_path = data_dir / row['Class'] / row['File_Name']
            
            if not file_path.exists():
                missing_files.append(str(file_path))
            else:
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the image
                except Exception:
                    corrupted_files.append(str(file_path))
        
        print(f"📊 Integrity check results (sample of {len(sample_df)}):")
        print(f"   Missing files: {len(missing_files)}")
        print(f"   Corrupted files: {len(corrupted_files)}")
        print(f"   Valid files: {len(sample_df) - len(missing_files) - len(corrupted_files)}")
        
        if len(missing_files) > 0:
            print(f"⚠️  Sample missing files: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}")
        
        if len(corrupted_files) > 0:
            print(f"⚠️  Sample corrupted files: {corrupted_files[:3]}{'...' if len(corrupted_files) > 3 else ''}")
        
        success_rate = (len(sample_df) - len(missing_files) - len(corrupted_files)) / len(sample_df)
        print(f"✅ Success rate: {success_rate:.1%}")
        
        return success_rate > 0.9  # 90% success rate threshold
        
    except Exception as e:
        print(f"❌ Data integrity test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple Data Pipeline Test")
    print("=" * 50)
    
    # Test 1: Basic data loading
    basic_success = test_basic_data_loading()
    
    # Test 2: Data integrity
    integrity_success = test_data_integrity()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   Basic data loading: {'✅ Passed' if basic_success else '❌ Failed'}")
    print(f"   Data integrity: {'✅ Passed' if integrity_success else '❌ Failed'}")
    
    if basic_success and integrity_success:
        print("\n🎉 All tests passed! Data pipeline is ready.")
        print("\n📝 Next steps:")
        print("1. Data loading pipeline is functional")
        print("2. Dataset integrity is good")
        print("3. Ready to create more advanced data loaders")
        print("4. Ready to proceed with model training")
        return True
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)