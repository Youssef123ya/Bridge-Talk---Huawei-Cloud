#!/usr/bin/env python3
"""
Phase 3: Advanced Model Development Script
This script implements comprehensive model development for Arabic Sign Language Recognition
"""

import sys
import os
from pathlib import Path
import time
import json
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import ModelFactory, ModelSelector
from src.models.base_model import ModelRegistry
from src.training.trainer import AdvancedTrainer, create_trainer_from_config
from src.data.dataset import create_data_loaders_from_csv
from src.utils.helpers import setup_logging, get_device, create_directories
from src.config.config import get_config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Phase 3: Model Development')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=ModelRegistry.list_models(),
                       help='Model architecture to use')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run model benchmarking')
    parser.add_argument('--select-model', action='store_true',
                       help='Use intelligent model selection')

    return parser.parse_args()

def benchmark_models(config):
    """Benchmark different model architectures"""

    print("🏁 Running Model Benchmarking")
    print("=" * 40)

    # Define models to benchmark
    models_to_benchmark = [
        {'architecture': 'cnn_basic', 'num_classes': 32},
        {'architecture': 'cnn_advanced', 'num_classes': 32},
        {'architecture': 'mobilenet_v2', 'num_classes': 32},
        {'architecture': 'efficientnet_b0', 'num_classes': 32},
        {'architecture': 'resnet50', 'num_classes': 32},
    ]

    # Add ensemble models
    if config.get('benchmark_ensembles', False):
        models_to_benchmark.extend([
            {
                'ensemble_type': 'simple',
                'base_models': [
                    {'architecture': 'resnet50', 'num_classes': 32},
                    {'architecture': 'efficientnet_b0', 'num_classes': 32}
                ]
            }
        ])

    device = get_device()
    results = ModelFactory.benchmark_models(
        models_to_benchmark,
        input_size=(1, 3, 224, 224),
        device=device
    )

    # Save benchmark results
    benchmark_file = 'logs/model_benchmark_results.json'
    os.makedirs(os.path.dirname(benchmark_file), exist_ok=True)

    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print results
    print("\n📊 Benchmark Results:")
    print("-" * 60)
    print(f"{'Model':<20} {'Params (M)':<12} {'Size (MB)':<12} {'FPS':<10}")
    print("-" * 60)

    for model_name, metrics in results.items():
        if 'error' not in metrics:
            params_m = metrics['parameters'] / 1_000_000
            size_mb = metrics.get('model_size_mb', metrics.get('size_mb', 0))
            fps = metrics['fps']
            print(f"{model_name:<20} {params_m:<12.2f} {size_mb:<12.1f} {fps:<10.1f}")
        else:
            print(f"{model_name:<20} ERROR: {metrics['error']}")

    print(f"\n💾 Detailed results saved to: {benchmark_file}")

    return results

def intelligent_model_selection(config):
    """Use intelligent model selection based on requirements"""

    print("🧠 Intelligent Model Selection")
    print("=" * 35)

    # Get dataset size
    dataset_size = 54049  # From ArSL dataset

    # Create model selector
    selector = ModelSelector(dataset_size=dataset_size, num_classes=32)

    # Different priority configurations
    scenarios = {
        'accuracy_focused': {
            'accuracy_priority': 0.7,
            'speed_priority': 0.2,
            'memory_priority': 0.1,
            'description': 'High accuracy research setup'
        },
        'balanced': {
            'accuracy_priority': 0.5,
            'speed_priority': 0.3,
            'memory_priority': 0.2,
            'description': 'Balanced production setup'
        },
        'speed_focused': {
            'accuracy_priority': 0.3,
            'speed_priority': 0.5,
            'memory_priority': 0.2,
            'description': 'Real-time inference focused'
        },
        'mobile_optimized': {
            'accuracy_priority': 0.4,
            'speed_priority': 0.3,
            'memory_priority': 0.3,
            'description': 'Mobile/edge deployment'
        }
    }

    recommendations = {}

    for scenario_name, priorities in scenarios.items():
        description = priorities.pop('description')  # Remove description from priorities
        print(f"\n📋 Scenario: {description}")

        selection = selector.select_optimal_model(**priorities)
        recommendations[scenario_name] = selection

        if 'error' not in selection:
            best_model = selection['recommended_model']
            scores = selection['scores']

            print(f"   🏆 Recommended: {best_model}")
            print(f"   📊 Total Score: {scores['total_score']:.3f}")
            print(f"   🎯 Accuracy Score: {scores['accuracy_score']:.3f}")
            print(f"   ⚡ Speed Score: {scores['speed_score']:.3f}")
            print(f"   💾 Memory Score: {scores['memory_score']:.3f}")
            print(f"   📈 Parameters: {scores['parameters']:,}")
        else:
            print(f"   ❌ Error: {selection['error']}")

    # Save recommendations
    recommendations_file = 'logs/model_recommendations.json'
    with open(recommendations_file, 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)

    print(f"\n💾 Recommendations saved to: {recommendations_file}")

    return recommendations

def train_single_model(model_config, training_config, data_loaders, args):
    """Train a single model with the given configuration"""

    print(f"🚀 Training Model: {model_config.get('architecture', 'unknown')}")
    print("=" * 50)

    # Create model factory
    factory = ModelFactory()
    
    # Create model
    if 'ensemble_type' in model_config:
        model = ModelFactory.create_ensemble(model_config)
    else:
        model = factory.create_model(model_config=model_config)

    print(f"✅ Created model: {model.__class__.__name__}")
    model.print_model_summary()

    # Create trainer
    trainer_config = training_config.copy()
    
    # Fix optimizer config structure - if optimizer is a string, convert to dict
    if 'optimizer' in trainer_config and isinstance(trainer_config['optimizer'], str):
        optimizer_type = trainer_config['optimizer']
        trainer_config['optimizer'] = {
            'type': optimizer_type,
            'learning_rate': trainer_config.get('learning_rate', 0.001),
            'weight_decay': trainer_config.get('weight_decay', 0.0001)
        }
    elif 'optimizer' not in trainer_config:
        trainer_config['optimizer'] = {
            'type': 'adam',
            'learning_rate': trainer_config.get('learning_rate', 0.001),
            'weight_decay': trainer_config.get('weight_decay', 0.0001)
        }

    # Override with command line arguments
    if args.epochs:
        trainer_config['epochs'] = args.epochs
    if args.learning_rate:
        trainer_config['optimizer']['learning_rate'] = args.learning_rate
    if args.device:
        trainer_config['device'] = args.device if args.device != 'auto' else get_device()
    if args.experiment_name:
        trainer_config['experiment_name'] = args.experiment_name

    train_loader, val_loader, test_loader = data_loaders

    trainer = create_trainer_from_config(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
            print(f"📂 Resumed from checkpoint: {args.resume}")
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}")

    # Train the model
    epochs = trainer_config.get('epochs', 100)
    history = trainer.train(
        epochs=epochs,
        save_best_only=trainer_config.get('save_best_only', True),
        monitor_metric=trainer_config.get('monitor_metric', 'val_accuracy')
    )

    # Create training plots
    plot_path = trainer.create_training_plots()

    # Save model information
    model_info_path = f'models/checkpoints/{trainer.experiment_name}_model_info.json'
    model.save_model_info(model_info_path)

    return {
        'model': model,
        'trainer': trainer,
        'history': history,
        'plot_path': plot_path
    }

def train_multiple_models(config, data_loaders, args):
    """Train multiple models for comparison"""

    print("🏁 Training Multiple Models")
    print("=" * 30)

    # Define models to train
    models_to_train = [
        {'architecture': 'mobilenet_v2', 'num_classes': 32},
        {'architecture': 'efficientnet_b0', 'num_classes': 32},
        {'architecture': 'resnet50', 'num_classes': 32},
    ]

    # Add custom model if available
    if 'custom_cnn' in ModelRegistry.list_models():
        models_to_train.append({'architecture': 'custom_cnn', 'num_classes': 32})

    results = {}
    training_config = config.config.get('training', {})
    training_config['epochs'] = 20  # Shorter training for comparison

    for i, model_config in enumerate(models_to_train):
        print(f"\n🔄 Training model {i+1}/{len(models_to_train)}")

        try:
            # Set unique experiment name
            arch_name = model_config['architecture']
            experiment_name = f"phase3_comparison_{arch_name}_{int(time.time())}"
            training_config['experiment_name'] = experiment_name

            result = train_single_model(
                model_config=model_config,
                training_config=training_config,
                data_loaders=data_loaders,
                args=args
            )

            results[arch_name] = {
                'final_val_accuracy': result['history']['val_acc'][-1],
                'best_val_accuracy': max(result['history']['val_acc']),
                'final_train_accuracy': result['history']['train_acc'][-1],
                'total_parameters': result['model'].get_num_parameters(),
                'experiment_name': experiment_name
            }

            print(f"✅ {arch_name} completed - Best Val Acc: {results[arch_name]['best_val_accuracy']:.2f}%")

        except Exception as e:
            print(f"❌ {arch_name} failed: {e}")
            results[arch_name] = {'error': str(e)}

    # Save comparison results
    comparison_file = 'logs/model_comparison_results.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print comparison
    print("\n📊 Training Comparison Results:")
    print("-" * 70)
    print(f"{'Model':<20} {'Best Val Acc':<15} {'Parameters':<15} {'Status'}")
    print("-" * 70)

    for model_name, metrics in results.items():
        if 'error' not in metrics:
            best_acc = metrics['best_val_accuracy']
            params = metrics['total_parameters']
            status = "✅ Success"
            print(f"{model_name:<20} {best_acc:<15.2f} {params:<15,} {status}")
        else:
            status = "❌ Failed"
            print(f"{model_name:<20} {'N/A':<15} {'N/A':<15} {status}")

    print(f"\n💾 Detailed comparison saved to: {comparison_file}")

    return results

def main():
    """Main Phase 3 execution function"""

    args = parse_args()

    print("🚀 Phase 3: Advanced Model Development")
    print("=" * 45)

    # Setup
    logger = setup_logging('logs/phase3_development.log')
    config = get_config(args.config)
    device = get_device() if args.device == 'auto' or args.device is None else args.device

    print(f"📁 Config: {args.config}")
    print(f"🎯 Device: {device}")
    print(f"🏗️  Available models: {len(ModelRegistry.list_models())}")

    # Create directories
    create_directories([
        'models/checkpoints',
        'models/weights',
        'logs/experiments',
        'logs/plots'
    ])

    # Load data if not just benchmarking
    data_loaders = None
    if not args.benchmark:
        try:
            print("\n📊 Loading data...")
            train_loader, _, _ = create_data_loaders_from_csv(
                labels_file='data/processed/train.csv',
                data_dir=config.get('data.raw_data_path', 'data/raw/'),
                batch_size=args.batch_size or config.get('data.batch_size', 32),
                num_workers=config.get('data.num_workers', 4),
                train_split=1.0,  # Already split in Phase 2
                val_split=0.0,
                image_size=tuple(config.get('data.image_size', [224, 224]))
            )

            # Also load validation data
            val_loader_separate, _, _ = create_data_loaders_from_csv(
                labels_file='data/processed/val.csv',
                data_dir=config.get('data.raw_data_path', 'data/raw/'),
                batch_size=args.batch_size or config.get('data.batch_size', 32),
                num_workers=config.get('data.num_workers', 4),
                train_split=1.0,
                val_split=0.0,
                image_size=tuple(config.get('data.image_size', [224, 224]))
            )
            
            # Load test data  
            _, _, test_loader = create_data_loaders_from_csv(
                labels_file='data/processed/test.csv',
                data_dir=config.get('data.raw_data_path', 'data/raw/'),
                batch_size=args.batch_size or config.get('data.batch_size', 32),
                num_workers=config.get('data.num_workers', 4),
                train_split=1.0,
                val_split=0.0,
                image_size=tuple(config.get('data.image_size', [224, 224]))
            )

            data_loaders = (train_loader, val_loader_separate, test_loader)
            print(f"✅ Data loaded - Train: {len(train_loader.dataset):,}, Val: {len(val_loader_separate.dataset):,}")

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            print("   Make sure Phase 2 data preparation is completed")
            return False

    start_time = time.time()

    try:
        # Execution based on arguments
        if args.benchmark:
            print("\n🏁 Running benchmarks...")
            benchmark_results = benchmark_models(config.config)

        elif args.select_model:
            print("\n🧠 Running intelligent model selection...")
            recommendations = intelligent_model_selection(config.config)

        else:
            # Regular training
            model_config = {
                'architecture': args.model,
                'num_classes': config.get('data.num_classes', 32),
                'pretrained': config.get('model.pretrained', True),
                'dropout_rate': config.get('model.dropout_rate', 0.3)
            }

            training_config = config.config.get('training', {})

            if config.get('train_multiple_models', False):
                print("\n🔄 Training multiple models...")
                results = train_multiple_models(config, data_loaders, args)
            else:
                print(f"\n🚀 Training single model: {args.model}")
                result = train_single_model(model_config, training_config, data_loaders, args)
                print(f"\n✅ Training completed!")
                print(f"   Best validation accuracy: {max(result['history']['val_acc']):.2f}%")
                print(f"   Training plots: {result['plot_path']}")

        total_time = time.time() - start_time
        print(f"\n🎉 Phase 3 completed successfully in {total_time/60:.1f} minutes!")

        print(f"\n📝 Next Steps:")
        print(f"   1. Review training results in logs/")
        print(f"   2. Analyze model performance plots")
        print(f"   3. Select best model for Phase 4 evaluation")
        print(f"   4. Consider ensemble methods for improved performance")

        return True

    except Exception as e:
        print(f"\n❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
