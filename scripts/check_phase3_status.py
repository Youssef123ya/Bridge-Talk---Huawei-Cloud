#!/usr/bin/env python3
"""
Check Phase 3 completion status
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def check_phase3_status():
    """Check Phase 3 completion status"""

    print("🔍 Phase 3 Status Check")
    print("=" * 30)

    checks = []

    # Check model implementation files
    model_files = [
        ('src/models/base_model.py', 'Base model implementation'),
        ('src/models/cnn_architectures.py', 'CNN architectures'),
        ('src/models/ensemble_models.py', 'Ensemble models'),
        ('src/models/model_factory.py', 'Model factory'),
    ]

    for file_path, description in model_files:
        if os.path.exists(file_path):
            checks.append(("✅", f"{description}: {file_path}"))
        else:
            checks.append(("❌", f"Missing {description}: {file_path}"))

    # Check training infrastructure
    training_files = [
        ('src/training/trainer.py', 'Advanced trainer'),
        ('src/training/callbacks.py', 'Training callbacks'),
        ('src/training/losses.py', 'Loss functions'),
        ('src/training/optimizers.py', 'Optimizers'),
    ]

    for file_path, description in training_files:
        if os.path.exists(file_path):
            checks.append(("✅", f"{description}: {file_path}"))
        else:
            checks.append(("❌", f"Missing {description}: {file_path}"))

    # Check script files
    script_files = [
        ('scripts/phase3_model_development.py', 'Phase 3 main script'),
        ('scripts/test_models.py', 'Model testing script'),
    ]

    for file_path, description in script_files:
        if os.path.exists(file_path):
            checks.append(("✅", f"{description}: {file_path}"))
        else:
            checks.append(("❌", f"Missing {description}: {file_path}"))

    # Check if models can be created
    try:
        from src.models.base_model import ModelRegistry
        available_models = ModelRegistry.list_models()
        if len(available_models) > 5:
            checks.append(("✅", f"Model registry: {len(available_models)} models available"))
        else:
            checks.append(("⚠️", f"Limited models: {len(available_models)} available"))
    except Exception as e:
        checks.append(("❌", f"Model registry error: {e}"))

    # Check if training can be initiated
    try:
        from src.training.trainer import AdvancedTrainer
        checks.append(("✅", "Advanced trainer can be imported"))
    except Exception as e:
        checks.append(("❌", f"Trainer import error: {e}"))

    # Check training logs/results
    if os.path.exists('logs/phase3_development.log'):
        checks.append(("✅", "Phase 3 training log found"))
    else:
        checks.append(("⚠️", "No training log found (run training first)"))

    if os.path.exists('models/checkpoints'):
        checkpoint_files = [f for f in os.listdir('models/checkpoints') if f.endswith('.pth')]
        if checkpoint_files:
            checks.append(("✅", f"Model checkpoints found: {len(checkpoint_files)} files"))
        else:
            checks.append(("⚠️", "No model checkpoints found"))
    else:
        checks.append(("⚠️", "Checkpoints directory not found"))

    # Check benchmark/test results
    test_results_files = [
        'logs/model_test_results.json',
        'logs/model_benchmark_results.json',
        'logs/model_comparison_results.json',
        'logs/model_recommendations.json'
    ]

    found_results = 0
    for results_file in test_results_files:
        if os.path.exists(results_file):
            found_results += 1

    if found_results >= 2:
        checks.append(("✅", f"Model analysis results: {found_results}/4 files found"))
    elif found_results >= 1:
        checks.append(("⚠️", f"Partial results: {found_results}/4 files found"))
    else:
        checks.append(("❌", "No model analysis results found"))

    # Print results
    for status, message in checks:
        print(f"{status} {message}")

    # Summary
    passed = sum(1 for check in checks if check[0] == "✅")
    warnings = sum(1 for check in checks if check[0] == "⚠️")
    failed = sum(1 for check in checks if check[0] == "❌")
    total = len(checks)

    print(f"\n📊 Status Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ⚠️  Warnings: {warnings}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Score: {passed}/{total}")

    if failed == 0:
        if warnings <= 3:
            print("\n🎉 Phase 3 completed successfully!")
            print("🚀 Ready for Phase 4: Evaluation & Optimization")

            # Show next steps
            print("\n📝 Recommended next steps:")
            print("   1. Run model testing: python scripts/test_models.py")
            print("   2. Train your best model: python scripts/phase3_model_development.py --model resnet50")
            print("   3. Compare multiple models: python scripts/phase3_model_development.py --benchmark")
            print("   4. Use intelligent selection: python scripts/phase3_model_development.py --select-model")

            return True
        else:
            print("\n⚠️  Phase 3 mostly complete with several warnings")
            print("📝 Review warnings above before proceeding")
            return True
    else:
        print(f"\n❌ Phase 3 incomplete. {failed} critical issues found.")
        print("🔧 Please address the missing components above")
        return False

def show_available_models():
    """Show all available model architectures"""

    try:
        from src.models.base_model import ModelRegistry

        models = ModelRegistry.list_models()

        print(f"\n🏗️  Available Model Architectures ({len(models)}):")
        print("-" * 40)

        for model in sorted(models):
            try:
                info = ModelRegistry.get_model_info(model)
                print(f"   📦 {model:<20} - {info.get('doc', 'No description')[:50]}")
            except:
                print(f"   📦 {model}")

    except Exception as e:
        print(f"❌ Could not load model registry: {e}")

def show_training_progress():
    """Show training progress if available"""

    checkpoint_dir = Path('models/checkpoints')
    if not checkpoint_dir.exists():
        print("\n⚠️  No checkpoints directory found")
        return

    checkpoints = list(checkpoint_dir.glob('*.pth'))
    if not checkpoints:
        print("\n⚠️  No model checkpoints found")
        return

    print(f"\n💾 Found {len(checkpoints)} model checkpoints:")

    for checkpoint in sorted(checkpoints):
        try:
            # Get file info
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            print(f"   📄 {checkpoint.name} ({size_mb:.1f} MB)")
        except:
            print(f"   📄 {checkpoint.name}")

if __name__ == "__main__":
    success = check_phase3_status()

    # Show additional info
    show_available_models()
    show_training_progress()

    sys.exit(0 if success else 1)
