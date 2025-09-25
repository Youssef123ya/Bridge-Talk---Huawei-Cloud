#!/usr/bin/env python3
"""
Quick training test - verify Phase 3 training pipeline works
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_training_pipeline():
    """Test the entire Phase 3 training pipeline"""
    
    print("🚀 Phase 3 Training Pipeline Test")
    print("=" * 50)
    
    try:
        # Import all required modules
        from src.models.model_factory import ModelFactory
        from src.data.dataset import create_data_loaders_from_csv
        from src.training.trainer import create_trainer_from_config
        from src.config.config import get_config
        
        print("✅ All imports successful")
        
        # Load config
        config = get_config('config/config.yaml')
        print("✅ Config loaded")
        
        # Create small data loaders for testing
        print("\n📊 Creating test data loaders...")
        train_loader, _, _ = create_data_loaders_from_csv(
            labels_file='data/processed/train.csv',
            data_dir=config.get('data.raw_data_path', 'data/raw/'),
            batch_size=2,  # Very small batch
            num_workers=0,  # No workers for simplicity
            train_split=1.0,
            val_split=0.0,
            image_size=(64, 64)  # Smaller images
        )
        
        val_loader, _, _ = create_data_loaders_from_csv(
            labels_file='data/processed/val.csv',
            data_dir=config.get('data.raw_data_path', 'data/raw/'),
            batch_size=2,
            num_workers=0,
            train_split=1.0,
            val_split=0.0,
            image_size=(64, 64)
        )
        
        print(f"✅ Data loaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        # Create simple model
        print("\n🏗️  Creating model...")
        factory = ModelFactory()
        model_config = {
            'architecture': 'cnn_basic',
            'num_classes': 32,
            'pretrained': False
        }
        
        model = factory.create_model(model_config=model_config)
        print(f"✅ Model created: {model.__class__.__name__}")
        
        # Create trainer config
        trainer_config = {
            'optimizer': {
                'type': 'adam',
                'learning_rate': 0.01,  # Higher LR for quick test
                'weight_decay': 0.001
            },
            'device': 'cpu',
            'use_mixed_precision': False,  # Disable for CPU
            'use_early_stopping': False,
            'track_metrics': ['accuracy']
        }
        
        # Create trainer
        trainer = create_trainer_from_config(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config
        )
        print("✅ Trainer created")
        
        # Test a few training steps
        print("\n🏃 Testing training steps...")
        trainer.model.train()
        
        # Test 3 batches only
        batch_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_count >= 3:  # Only test 3 batches
                break
                
            print(f"   Batch {batch_count + 1}: {data.shape}, labels: {target.shape}")
            
            # Forward pass
            outputs = trainer.model(data)
            loss = trainer.criterion(outputs, target)
            
            # Backward pass
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            
            print(f"   Loss: {loss.item():.4f}")
            batch_count += 1
        
        print(f"✅ Successfully processed {batch_count} training batches")
        
        # Test validation step
        print("\n🔍 Testing validation...")
        trainer.model.eval()
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if val_batch_count >= 2:  # Test 2 validation batches
                    break
                    
                outputs = trainer.model(data)
                val_loss = trainer.criterion(outputs, target)
                print(f"   Val Batch {val_batch_count + 1}: Loss {val_loss.item():.4f}")
                val_batch_count += 1
        
        print(f"✅ Successfully processed {val_batch_count} validation batches")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    
    start_time = time.time()
    success = test_training_pipeline()
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n🎉 Phase 3 Training Pipeline Test PASSED! ({elapsed:.1f}s)")
        print("\n✅ All components working:")
        print("   - Model Factory ✅")
        print("   - Data Loading ✅") 
        print("   - Training Loop ✅")
        print("   - Validation Loop ✅")
        print("\n🚀 Phase 3 is ready for full training!")
    else:
        print(f"\n❌ Phase 3 Training Pipeline Test FAILED! ({elapsed:.1f}s)")