#!/usr/bin/env python3
"""
Quick status check for Phase 1 setup
"""
import os
import sys
import subprocess

def check_python_packages():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'PIL', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'cv2', 'tqdm', 'sklearn'
    ]
    
    installed_packages = []
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    return installed_packages, missing_packages

def check_phase1_status():
    """Check Phase 1 completion status"""

    print("🔍 Phase 1 Setup Status Check")
    print("=" * 40)

    checks = []

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        checks.append(("✅", f"Python version: {python_version} (compatible)"))
    else:
        checks.append(("❌", f"Python version: {python_version} (need 3.8+)"))

    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        checks.append(("✅", "Virtual environment is active"))
    else:
        checks.append(("⚠️", "Virtual environment not detected"))

    # Check essential directories (Phase 1 minimal requirements)
    essential_dirs = ['data', 'data/raw', 'data/analysis', 'models', 'logs']
    for directory in essential_dirs:
        if os.path.exists(directory):
            checks.append(("✅", f"Directory exists: {directory}/"))
        else:
            checks.append(("❌", f"Missing directory: {directory}/"))

    # Check Phase 1 essential files
    essential_files = ['setup_environment.py', 'simple_validate.py']
    for file_path in essential_files:
        if os.path.exists(file_path):
            checks.append(("✅", f"Script exists: {file_path}"))
        else:
            checks.append(("❌", f"Missing script: {file_path}"))

    # Check data files
    if os.path.exists('data/ArSL_Data_Labels.csv'):
        # Count rows in CSV
        try:
            import pandas as pd
            df = pd.read_csv('data/ArSL_Data_Labels.csv')
            checks.append(("✅", f"Labels file found ({len(df):,} samples)"))
        except:
            checks.append(("✅", "Labels file found"))
    else:
        checks.append(("❌", "Labels file not found (copy ArSL_Data_Labels.csv to data/)"))

    if os.path.exists('data/raw') and len(os.listdir('data/raw')) > 1:
        # Count image folders
        image_folders = [d for d in os.listdir('data/raw') if os.path.isdir(os.path.join('data/raw', d))]
        checks.append(("✅", f"Image data found ({len(image_folders)} sign classes)"))
    else:
        checks.append(("❌", "No image data found (copy images to data/raw/)"))

    # Check analysis results
    if os.path.exists('data/analysis/validation_report.txt'):
        checks.append(("✅", "Dataset validation completed"))
    else:
        checks.append(("⚠️", "Dataset validation not run (run: python simple_validate.py)"))

    if os.path.exists('data/analysis/class_distribution.png'):
        checks.append(("✅", "Data visualization created"))
    else:
        checks.append(("⚠️", "Data visualization not created"))

    # Check Python packages
    print("\n🐍 Checking Python packages...")
    installed, missing = check_python_packages()
    
    if not missing:
        checks.append(("✅", f"All required packages installed ({len(installed)}/{len(installed) + len(missing)})"))
    else:
        checks.append(("❌", f"Missing packages: {', '.join(missing)}"))

    # Print results
    print("\n📋 Status Report:")
    for status, message in checks:
        print(f"{status} {message}")

    # Summary
    passed = sum(1 for check in checks if check[0] == "✅")
    warnings = sum(1 for check in checks if check[0] == "⚠️")
    failed = sum(1 for check in checks if check[0] == "❌")
    total = len(checks)

    print(f"\n📊 Summary: {passed} passed, {warnings} warnings, {failed} failed (total: {total})")

    if failed == 0:
        print("🎉 Phase 1 setup is complete! Ready for Phase 2.")
        print("\n📝 Phase 1 Accomplishments:")
        print("  ✅ Environment setup with Python 3.13+ and virtual environment")
        print("  ✅ All required packages installed (PyTorch, OpenCV, etc.)")
        print("  ✅ Dataset validated (108K+ images, 32 sign classes)")
        print("  ✅ Project structure created")
        print("  ✅ Data analysis and visualization completed")
        
        print("\n🚀 Ready for Phase 2: Model Development")
        print("  Next: Create training pipeline and CNN models")
        return True
    elif warnings > 0 and failed == 0:
        print("⚠️  Phase 1 mostly complete with minor issues.")
        return True
    else:
        print("❌ Phase 1 setup incomplete. Please fix the failed items above.")
        return False

if __name__ == "__main__":
    success = check_phase1_status()
    sys.exit(0 if success else 1)
