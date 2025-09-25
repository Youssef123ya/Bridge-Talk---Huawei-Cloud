#!/usr/bin/env python3
"""
Quick test of model creation and training setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import ModelFactory
from src.config.config import get_config

def test_model_creation():
    """Test model creation functionality"""
    
    print("🧪 Testing Model Creation")
    print("=" * 30)
    
    try:
        # Create model factory
        factory = ModelFactory()
        print("✅ ModelFactory created")
        
        # Test model config
        model_config = {
            'architecture': 'resnet50',
            'num_classes': 32,
            'pretrained': False,
            'dropout_rate': 0.3
        }
        
        print(f"📋 Creating model with config: {model_config}")
        
        # Create model
        model = factory.create_model(model_config=model_config)
        print(f"✅ Model created: {model.__class__.__name__}")
        
        # Get model info
        info = model.get_model_info()
        print(f"📊 Parameters: {info['total_parameters']:,}")
        print(f"📏 Size: {info['model_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    
    print("\n🧪 Testing Config Loading")
    print("=" * 30)
    
    try:
        config = get_config('config/config.yaml')
        print("✅ Config loaded")
        print(f"📋 Config type: {type(config)}")
        
        # Test get method
        batch_size = config.get('data.batch_size', 32)
        print(f"📊 Batch size: {batch_size}")
        
        # Test config property
        config_dict = config.config
        print(f"📋 Config dict keys: {list(config_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Phase 3 Component Testing")
    print("=" * 40)
    
    # Test components
    config_ok = test_config_loading()
    model_ok = test_model_creation()
    
    if config_ok and model_ok:
        print("\n✅ All components working!")
    else:
        print("\n❌ Some components failed!")
    
    print("\n🎯 Testing completed!")