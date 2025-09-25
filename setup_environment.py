#!/usr/bin/env python3
"""
Environment Setup Script for Arabic Sign Language Recognition Project
This script handles the complete Phase 1 setup process.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0.0",
        "scikit-learn>=1.3.0",
        "psutil>=5.9.0"
    ]

    print("📦 Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
            return False

    print("🎉 All packages installed successfully!")
    return True

def verify_installation():
    """Verify that all required packages can be imported"""
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'cv2': 'OpenCV',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'sklearn': 'scikit-learn'
    }

    print("🔍 Verifying package installation...")
    failed = []

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Import failed")
            failed.append(name)

    if failed:
        print(f"\n❌ Failed packages: {', '.join(failed)}")
        return False

    print("✅ All packages verified successfully!")
    return True

def setup_project_structure():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/analysis",
        "models/checkpoints",
        "models/weights",
        "logs",
        "notebooks",
        "tests",
        "api"
    ]

    print("📁 Setting up project structure...")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}/")

    return True

def check_system_requirements():
    """Check system requirements"""
    import psutil

    print("🖥️  Checking system requirements...")

    # Check RAM
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    print(f"   RAM: {ram_gb:.1f} GB {'✅' if ram_gb >= 8 else '⚠️ (8GB+ recommended)'}")

    # Check disk space
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    print(f"   Free disk space: {free_gb:.1f} GB {'✅' if free_gb >= 10 else '⚠️ (10GB+ recommended)'}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB) ✅")
        else:
            print("   GPU: Not available (CPU training will be slower) ⚠️")
    except ImportError:
        print("   GPU: Cannot check (PyTorch not installed)")

    return True

def main():
    """Main setup function"""
    print("🚀 Arabic Sign Language Recognition - Environment Setup")
    print("=" * 60)

    # Step 1: Check Python version
    if not check_python_version():
        return False

    # Step 2: Set up project structure
    if not setup_project_structure():
        return False

    # Step 3: Install requirements
    if not install_requirements():
        return False

    # Step 4: Verify installation
    if not verify_installation():
        return False

    # Step 5: Check system requirements
    check_system_requirements()

    print("\n🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy your ArSL_Data_Labels.csv file to data/ directory")
    print("2. Extract your image files to data/raw/ directory") 
    print("3. Run: python scripts/validate_data.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
