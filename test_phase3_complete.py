#!/usr/bin/env python3
"""
Quick test of all Phase 3 components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test all Phase 3 imports"""
    
    print("🧪 Testing Phase 3 Component Imports")
    print("=" * 40)
    
    try:
        # Test CNN architectures
        from src.models.cnn_architectures import (
            BasicCNN, AdvancedCNN, ResNet50SignLanguage, 
            MobileNetV2SignLanguage, EfficientNetB0SignLanguage,
            CNN_ARCHITECTURES, get_architecture, list_architectures
        )
        print("✅ CNN Architectures imported successfully")
        print(f"   Available architectures: {list_architectures()}")
        
        # Test ensemble models
        from src.models.ensemble_models import (
            VotingEnsemble, StackingEnsemble, BaggingEnsemble, AdaptiveEnsemble,
            create_voting_ensemble, create_stacking_ensemble, create_bagging_ensemble
        )
        print("✅ Ensemble Models imported successfully")
        
        # Test training callbacks
        from src.training.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger,
            ProgressBar, MetricsHistory, get_default_callbacks, get_minimal_callbacks
        )
        print("✅ Training Callbacks imported successfully")
        
        # Test loss functions
        from src.training.losses import (
            FocalLoss, LabelSmoothingLoss, CenterLoss, TripletLoss,
            SupConLoss, WeightedCrossEntropyLoss, get_loss_function, LOSS_FUNCTIONS
        )
        print("✅ Loss Functions imported successfully")
        print(f"   Available losses: {list(LOSS_FUNCTIONS.keys())}")
        
        # Test optimizers
        from src.training.optimizers import (
            RAdam, AdaBound, Lookahead, SAM, get_optimizer_config,
            create_optimizer, create_scheduler, list_optimizers, list_schedulers
        )
        print("✅ Optimizers imported successfully")
        print(f"   Available optimizers: {list_optimizers()}")
        print(f"   Available schedulers: {list_schedulers()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test creating models from new architectures"""
    
    print("\n🧪 Testing Model Creation")
    print("=" * 30)
    
    try:
        from src.models.cnn_architectures import get_architecture
        
        # Test creating different architectures
        architectures = ['cnn_basic', 'cnn_advanced', 'resnet50']
        
        for arch_name in architectures:
            arch_class = get_architecture(arch_name)
            model = arch_class(num_classes=32)
            info = model.get_model_info()
            print(f"✅ {arch_name}: {info['total_parameters']:,} parameters, {info['model_size_mb']:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions():
    """Test creating different loss functions"""
    
    print("\n🧪 Testing Loss Functions")
    print("=" * 30)
    
    try:
        import torch
        from src.training.losses import get_loss_function
        
        # Test different loss functions
        losses = ['focal', 'label_smoothing', 'weighted_ce']
        
        for loss_name in losses:
            if loss_name == 'focal':
                loss_fn = get_loss_function(loss_name, gamma=2.0)
            elif loss_name == 'label_smoothing':
                loss_fn = get_loss_function(loss_name, num_classes=32, smoothing=0.1)
            elif loss_name == 'weighted_ce':
                weights = torch.ones(32)
                loss_fn = get_loss_function(loss_name, class_weights=weights)
            
            print(f"✅ {loss_name}: {loss_fn.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Phase 3 Components Test")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_imports()
    models_ok = test_model_creation()
    losses_ok = test_loss_functions()
    
    print("\n📊 Test Results")
    print("=" * 20)
    print(f"✅ Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"✅ Models: {'PASS' if models_ok else 'FAIL'}")
    print(f"✅ Losses: {'PASS' if losses_ok else 'FAIL'}")
    
    if imports_ok and models_ok and losses_ok:
        print("\n🎉 All Phase 3 components working correctly!")
        print("🚀 Phase 3 development framework is ready for use!")
    else:
        print("\n❌ Some components need attention!")
    
    print("\n🎯 Testing completed!")