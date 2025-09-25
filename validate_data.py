#!/usr/bin/env python3
"""
Data Validation Script for Arabic Sign Language Recognition Project
This script validates the dataset and prepares it for training.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DatasetAnalyzer, validate_environment
from src.utils.helpers import setup_logging, create_directories
from src.config.config import get_config

def main():
    """Main validation function"""
    print("🔍 Arabic Sign Language Dataset Validation")
    print("=" * 50)

    # Setup logging
    logger = setup_logging('logs/validation.log')

    # Load configuration
    config = get_config()

    # Check if required files exist
    labels_file = config.get('data.labels_file', 'ArSL_Data_Labels.csv')
    data_dir = config.get('data.raw_data_path', 'data/raw/')

    if not os.path.exists(labels_file):
        print(f"❌ Labels file not found: {labels_file}")
        print("   Please copy your ArSL_Data_Labels.csv file to the data/ directory")
        return False

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please create the data/raw/ directory and copy your image files")
        return False

    # Check environment
    if not validate_environment():
        return False

    # Create analyzer
    analyzer = DatasetAnalyzer(labels_file, data_dir)

    # Run analysis
    print("\n📊 Running dataset analysis...")
    analysis_results = analyzer.analyze_dataset()

    print("\n🔍 Validating files...")
    missing_files, corrupted_files = analyzer.validate_files()

    print("\n📸 Analyzing image properties...")
    image_analysis = analyzer.analyze_image_properties(sample_size=1000)

    print("\n📊 Creating visualizations...")
    analyzer.create_visualizations()

    print("\n📋 Generating comprehensive report...")
    report = analyzer.generate_report()

    # Summary
    total_files = analysis_results['total_samples']
    usable_files = total_files - len(missing_files) - len(corrupted_files)

    print(f"\n✅ Dataset Validation Summary:")
    print(f"   Total files in CSV: {total_files:,}")
    print(f"   Missing files: {len(missing_files)}")
    print(f"   Corrupted files: {len(corrupted_files)}")
    print(f"   Usable files: {usable_files:,} ({usable_files/total_files*100:.1f}%)")
    print(f"   Classes: {analysis_results['num_classes']}")

    if len(missing_files) + len(corrupted_files) > total_files * 0.1:
        print("⚠️  Warning: More than 10% of files have issues")
        print("   Consider checking your data directory structure")

    print(f"\n📁 Results saved to:")
    print(f"   📊 Visualization: data/analysis/class_distribution.png")
    print(f"   📋 Report: data/analysis/dataset_report.txt")

    if usable_files > 50000:
        print("\n🎉 Dataset looks great! Ready for Phase 2: Data Pipeline Development")

        # Show next steps
        print("\n📝 Next steps:")
        print("1. Review the analysis report in data/analysis/")
        print("2. Adjust configuration in config/config.yaml if needed")
        print("3. Proceed to Phase 2: Data Pipeline Development")
        print("4. Run: python scripts/test_data_pipeline.py")

        return True
    else:
        print("❌ Dataset has significant issues. Please check your data setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
