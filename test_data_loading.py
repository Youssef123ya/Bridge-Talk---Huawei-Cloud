#!/usr/bin/env python3
"""
Test data loading functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_data_loaders_from_csv
from src.config.config import get_config

def test_data_loading():
    """Test data loading functionality"""
    
    print("🧪 Testing Data Loading")
    print("=" * 30)
    
    try:
        # Load config
        config = get_config('config/config.yaml')
        
        # Data paths
        train_csv = "data/processed/train.csv"
        val_csv = "data/processed/val.csv"
        
        print(f"📁 Train CSV: {train_csv}")
        print(f"📁 Val CSV: {val_csv}")
        
        # Check if files exist
        train_path = Path(train_csv)
        val_path = Path(val_csv)
        
        if not train_path.exists():
            print(f"❌ Train CSV not found: {train_path.absolute()}")
            return False
            
        if not val_path.exists():
            print(f"❌ Val CSV not found: {val_path.absolute()}")
            return False
            
        print("✅ CSV files found")
        
        # Check file sizes
        print(f"📊 Train CSV size: {train_path.stat().st_size / 1024:.1f} KB")
        print(f"📊 Val CSV size: {val_path.stat().st_size / 1024:.1f} KB")
        
        # Try to read first few lines
        with open(train_csv, 'r') as f:
            lines = f.readlines()[:3]
            print(f"📋 Train CSV preview:")
            for i, line in enumerate(lines):
                print(f"   Line {i}: {line.strip()}")
        
        # Try to create data loaders with smaller batch size
        print("\n🔄 Creating data loaders...")
        batch_size = 4  # Small batch for testing
        
        # Create train loader
        train_loader, _, _ = create_data_loaders_from_csv(
            labels_file=train_csv,
            data_dir=config.get('data.raw_data_path', 'data/raw/'),
            batch_size=batch_size,
            num_workers=2,  # Reduced for testing
            train_split=1.0,  # Already split
            val_split=0.0,
            image_size=(224, 224)
        )
        
        # Create val loader
        val_loader, _, _ = create_data_loaders_from_csv(
            labels_file=val_csv,
            data_dir=config.get('data.raw_data_path', 'data/raw/'),
            batch_size=batch_size,
            num_workers=2,  # Reduced for testing
            train_split=1.0,  # Already split
            val_split=0.0,
            image_size=(224, 224)
        )
        
        print(f"✅ Data loaders created")
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Val batches: {len(val_loader)}")
        
        # Try to get one batch
        print("\n🔄 Testing first batch...")
        train_iter = iter(train_loader)
        batch_images, batch_labels = next(train_iter)
        
        print(f"✅ First batch loaded")
        print(f"📊 Batch images shape: {batch_images.shape}")
        print(f"📊 Batch labels shape: {batch_labels.shape}")
        print(f"📊 Label range: {batch_labels.min().item()}-{batch_labels.max().item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Data Loading Test")
    print("=" * 40)
    
    success = test_data_loading()
    
    if success:
        print("\n✅ Data loading works!")
    else:
        print("\n❌ Data loading failed!")
    
    print("\n🎯 Testing completed!")