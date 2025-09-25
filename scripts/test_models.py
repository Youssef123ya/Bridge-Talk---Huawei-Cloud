#!/usr/bin/env python3
"""
Test model architectures and implementations
"""

import sys
import os
from pathlib import Path
import torch
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import ModelRegistry
from src.models.model_factory import ModelFactory
from src.utils.helpers import get_device, format_time
from src.config.config import get_config

def test_all_models():
    """Test all registered model architectures"""

    print("🧪 Testing All Model Architectures")
    print("=" * 40)

    device = get_device()
    batch_size = 4
    input_size = (batch_size, 3, 224, 224)
    num_classes = 32

    print(f"Device: {device}")
    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Available models: {len(ModelRegistry.list_models())}")

    results = {}

    for model_name in ModelRegistry.list_models():
        print(f"\n🔍 Testing {model_name}...")

        try:
            # Create model
            start_time = time.time()
            model = ModelRegistry.get_model(model_name, num_classes=num_classes)
            creation_time = time.time() - start_time

            # Move to device
            model = model.to(device)
            model.eval()

            # Test forward pass
            dummy_input = torch.randn(input_size).to(device)

            with torch.no_grad():
                start_time = time.time()
                output = model(dummy_input)
                inference_time = time.time() - start_time

            # Test feature extraction
            features = model.get_features(dummy_input)

            # Get model info
            info = model.get_model_info()

            results[model_name] = {
                'status': 'success',
                'output_shape': list(output.shape),
                'feature_shape': list(features.shape),
                'parameters': info['total_parameters'],
                'size_mb': info['model_size_mb'],
                'creation_time': creation_time,
                'inference_time': inference_time,
                'fps': batch_size / inference_time
            }

            print(f"   ✅ Success")
            print(f"      Output: {output.shape}")
            print(f"      Features: {features.shape}")
            print(f"      Parameters: {info['total_parameters']:,}")
            print(f"      Size: {info['model_size_mb']:.1f} MB")
            print(f"      Inference: {inference_time*1000:.1f}ms ({batch_size/inference_time:.1f} FPS)")

        except Exception as e:
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"   ❌ Failed: {e}")

    # Summary
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    total = len(results)

    print(f"\n📊 Test Summary:")
    print(f"   Successful: {successful}/{total}")
    print(f"   Failed: {total - successful}/{total}")

    if successful > 0:
        # Find fastest and most accurate models
        successful_models = {name: data for name, data in results.items() if data['status'] == 'success'}

        fastest_model = max(successful_models.keys(), key=lambda k: successful_models[k]['fps'])
        smallest_model = min(successful_models.keys(), key=lambda k: successful_models[k]['parameters'])
        largest_model = max(successful_models.keys(), key=lambda k: successful_models[k]['parameters'])

        print(f"\n🏆 Performance Highlights:")
        print(f"   Fastest: {fastest_model} ({successful_models[fastest_model]['fps']:.1f} FPS)")
        print(f"   Smallest: {smallest_model} ({successful_models[smallest_model]['parameters']:,} params)")
        print(f"   Largest: {largest_model} ({successful_models[largest_model]['parameters']:,} params)")

    return results

def test_ensemble_models():
    """Test ensemble model creation and functionality"""

    print("\n🔗 Testing Ensemble Models")
    print("=" * 30)

    device = get_device()
    batch_size = 4
    input_size = (batch_size, 3, 224, 224)

    ensemble_configs = [
        {
            'ensemble_type': 'simple',
            'base_models': [
                {'architecture': 'mobilenet_v2', 'num_classes': 32},
                {'architecture': 'efficientnet_b0', 'num_classes': 32}
            ]
        },
        {
            'ensemble_type': 'stacked',
            'base_models': [
                {'architecture': 'resnet50', 'num_classes': 32},
                {'architecture': 'efficientnet_b0', 'num_classes': 32}
            ],
            'meta_learner_hidden_dim': 64
        }
    ]

    for i, config in enumerate(ensemble_configs):
        ensemble_type = config['ensemble_type']
        print(f"\n🧩 Testing {ensemble_type} ensemble...")

        try:
            # Create ensemble
            start_time = time.time()
            ensemble = ModelFactory.create_ensemble(config)
            creation_time = time.time() - start_time

            ensemble = ensemble.to(device)
            ensemble.eval()

            # Test forward pass
            dummy_input = torch.randn(input_size).to(device)

            with torch.no_grad():
                start_time = time.time()
                output = ensemble(dummy_input)
                inference_time = time.time() - start_time

            print(f"   ✅ {ensemble_type.title()} ensemble created successfully")
            print(f"      Output shape: {output.shape}")
            print(f"      Creation time: {creation_time:.2f}s")
            print(f"      Inference time: {inference_time*1000:.1f}ms")
            print(f"      Total parameters: {ensemble.get_num_parameters():,}")

            # Test ensemble-specific functionality
            if hasattr(ensemble, 'get_individual_predictions'):
                individual_preds = ensemble.get_individual_predictions(dummy_input)
                print(f"      Individual predictions: {len(individual_preds)} models")

            if hasattr(ensemble, 'get_prediction_confidence'):
                confidence_info = ensemble.get_prediction_confidence(dummy_input)
                print(f"      Confidence metrics available: {list(confidence_info.keys())}")

        except Exception as e:
            print(f"   ❌ {ensemble_type} ensemble failed: {e}")

def test_model_factory():
    """Test model factory functionality"""

    print("\n🏭 Testing Model Factory")
    print("=" * 25)

    # Test model recommendations
    print("\n📋 Testing model recommendations...")
    try:
        recommendations = ModelFactory.get_model_recommendations(
            dataset_size=50000,
            computational_budget='medium',
            target_accuracy=0.9
        )

        print(f"   ✅ Generated {len(recommendations)} recommendations")
        for rec in recommendations[:3]:  # Show first 3
            print(f"      {rec['name']}: Expected {rec['expected_accuracy']:.1%} accuracy")

    except Exception as e:
        print(f"   ❌ Recommendations failed: {e}")

    # Test benchmarking
    print("\n⚡ Testing model benchmarking...")
    try:
        test_configs = [
            {'architecture': 'mobilenet_v2', 'num_classes': 32},
            {'architecture': 'resnet50', 'num_classes': 32}
        ]

        device = get_device()
        benchmark_results = ModelFactory.benchmark_models(
            test_configs,
            input_size=(1, 3, 224, 224),
            device=device
        )

        print(f"   ✅ Benchmarked {len(benchmark_results)} models")
        for model_name, metrics in benchmark_results.items():
            if 'error' not in metrics:
                print(f"      {model_name}: {metrics['fps']:.1f} FPS, {metrics['parameters']:,} params")
            else:
                print(f"      {model_name}: Error - {metrics['error']}")

    except Exception as e:
        print(f"   ❌ Benchmarking failed: {e}")

def test_model_operations():
    """Test model operations like freezing, profiling, etc."""

    print("\n⚙️  Testing Model Operations")
    print("=" * 30)

    try:
        # Create a test model
        model = ModelRegistry.get_model('resnet50', num_classes=32)
        device = get_device()
        model = model.to(device)

        print("✅ Created ResNet-50 for testing operations")

        # Test model info
        info = model.get_model_info()
        print(f"   📊 Model info: {info['total_parameters']:,} parameters")

        # Test freezing
        initial_trainable = model.get_trainable_parameters()
        model.freeze_feature_extractor()
        frozen_trainable = model.get_trainable_parameters()
        model.unfreeze_feature_extractor()
        unfrozen_trainable = model.get_trainable_parameters()

        print(f"   🔒 Freezing test:")
        print(f"      Initial trainable: {initial_trainable:,}")
        print(f"      After freezing: {frozen_trainable:,}")
        print(f"      After unfreezing: {unfrozen_trainable:,}")

        # Test profiling
        print("   ⏱️  Profiling model...")
        profile = model.profile_model(input_size=(1, 3, 224, 224), device=device)
        print(f"      Inference time: {profile['avg_inference_time_ms']:.1f}ms")
        print(f"      FPS: {profile['fps']:.1f}")

        # Test model summary
        print("   📋 Testing model summary...")
        model.print_model_summary()

        print("✅ All model operations tested successfully")

    except Exception as e:
        print(f"❌ Model operations test failed: {e}")

def main():
    """Main test function"""

    print("🧪 Model Architecture Testing Suite")
    print("=" * 40)

    try:
        # Test all individual models
        model_results = test_all_models()

        # Test ensemble models
        test_ensemble_models()

        # Test model factory
        test_model_factory()

        # Test model operations
        test_model_operations()

        print("\n✅ All tests completed!")

        # Save test results
        import json
        results_file = 'logs/model_test_results.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)

        print(f"💾 Test results saved to: {results_file}")

        return True

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
