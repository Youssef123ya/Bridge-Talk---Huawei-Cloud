#!/usr/bin/env python3
"""
Check Phase 2 completion status
"""

import os
import sys
import json
from pathlib import Path

def check_phase2_status():
    """Check Phase 2 completion status"""

    print("🔍 Phase 2 Status Check")
    print("=" * 30)

    checks = []

    # Check core data files
    data_files = [
        ('data/processed/train.csv', 'Training data split'),
        ('data/processed/val.csv', 'Validation data split'),
        ('data/processed/test.csv', 'Test data split'),
        ('data/processed/class_mapping.json', 'Class mapping'),
        ('data/processed/split_metadata.json', 'Split metadata')
    ]

    for file_path, description in data_files:
        if os.path.exists(file_path):
            checks.append(("✅", f"{description}: {file_path}"))
        else:
            checks.append(("❌", f"Missing {description}: {file_path}"))

    # Check analysis files
    analysis_files = [
        ('data/analysis/comprehensive_validation_report.txt', 'Validation report'),
        ('data/analysis/detailed_validation_results.json', 'Detailed validation'),
        ('data/analysis/data_splits_analysis.png', 'Split visualization'),
    ]

    for file_path, description in analysis_files:
        if os.path.exists(file_path):
            checks.append(("✅", f"{description}: {file_path}"))
        else:
            checks.append(("❌", f"Missing {description}: {file_path}"))

    # Check cross-validation folds
    cv_dir = 'data/splits/cv_folds'
    if os.path.exists(cv_dir):
        cv_files = [f for f in os.listdir(cv_dir) if f.endswith('.csv')]
        if len(cv_files) >= 10:  # 5 folds * 2 files each
            checks.append(("✅", f"Cross-validation folds: {len(cv_files)} files"))
        else:
            checks.append(("⚠️", f"Incomplete CV folds: {len(cv_files)} files"))
    else:
        checks.append(("❌", "Missing cross-validation folds"))

    # Check augmentation examples
    aug_file = 'data/analysis/augmentations/augmentation_examples.png'
    if os.path.exists(aug_file):
        checks.append(("✅", f"Augmentation examples: {aug_file}"))
    else:
        checks.append(("⚠️", f"Missing augmentation examples"))

    # Check data quality
    if os.path.exists('data/processed/split_metadata.json'):
        with open('data/processed/split_metadata.json', 'r') as f:
            metadata = json.load(f)

        total_samples = metadata.get('total_samples', 0)
        if total_samples > 50000:
            checks.append(("✅", f"Dataset size: {total_samples:,} samples"))
        else:
            checks.append(("⚠️", f"Small dataset: {total_samples:,} samples"))

        num_classes = metadata.get('num_classes', 0)
        if num_classes == 32:
            checks.append(("✅", f"Classes: {num_classes} (correct)"))
        else:
            checks.append(("⚠️", f"Classes: {num_classes} (expected 32)"))

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
        if warnings <= 2:
            print("\n🎉 Phase 2 completed successfully!")
            print("🚀 Ready for Phase 3: Model Development")
            return True
        else:
            print("\n⚠️  Phase 2 mostly complete with some warnings")
            print("📝 Review warnings above before proceeding")
            return True
    else:
        print(f"\n❌ Phase 2 incomplete. {failed} critical issues found.")
        print("🔧 Please run: python scripts/phase2_data_preparation.py")
        return False

if __name__ == "__main__":
    success = check_phase2_status()
    sys.exit(0 if success else 1)
