#!/usr/bin/env python3
"""
Simple Data Validation Script for Arabic Sign Language Recognition Project
This script validates the dataset without requiring the full project structure.
"""

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import numpy as np

def validate_basic_setup():
    """Check if basic files and directories exist"""
    print("🔍 Checking basic setup...")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ data/ directory not found")
        return False
    
    # Check if labels file exists
    labels_file = 'data/ArSL_Data_Labels.csv'
    if not os.path.exists(labels_file):
        print(f"❌ Labels file not found: {labels_file}")
        return False
    
    # Check if raw data directory exists
    raw_data_dir = 'data/raw'
    if not os.path.exists(raw_data_dir):
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return False
    
    print("✅ Basic setup looks good!")
    return True

def analyze_csv_data():
    """Analyze the CSV labels file"""
    print("\n📊 Analyzing CSV data...")
    
    try:
        # Read the CSV file
        df = pd.read_csv('data/ArSL_Data_Labels.csv')
        print(f"✅ CSV file loaded successfully")
        print(f"   Total rows: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
        
        # Show column names
        print(f"   Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\n📋 First 5 rows:")
        print(df.head())
        
        # Analyze classes
        if 'class' in df.columns:
            class_counts = df['class'].value_counts()
            print(f"\n📊 Class distribution:")
            print(f"   Number of classes: {len(class_counts)}")
            print(f"   Class counts:")
            for class_name, count in class_counts.head(10).items():
                print(f"     {class_name}: {count:,} samples")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None

def check_image_folders():
    """Check image folders in data/raw"""
    print("\n📁 Checking image folders...")
    
    raw_dir = Path('data/raw')
    folders = [f for f in raw_dir.iterdir() if f.is_dir()]
    
    print(f"✅ Found {len(folders)} folders in data/raw/")
    
    folder_stats = {}
    total_images = 0
    
    for folder in sorted(folders):
        image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.JPG')) + \
                     list(folder.glob('*.png')) + list(folder.glob('*.PNG')) + \
                     list(folder.glob('*.jpeg')) + list(folder.glob('*.JPEG'))
        
        folder_stats[folder.name] = len(image_files)
        total_images += len(image_files)
        
        if len(image_files) > 0:
            print(f"   {folder.name}: {len(image_files):,} images")
    
    print(f"\n📊 Total images found: {total_images:,}")
    return folder_stats, total_images

def validate_sample_images():
    """Validate a sample of images"""
    print("\n🖼️  Validating sample images...")
    
    raw_dir = Path('data/raw')
    folders = [f for f in raw_dir.iterdir() if f.is_dir()]
    
    if not folders:
        print("❌ No image folders found")
        return False
    
    # Check a few sample images
    sample_folder = folders[0]
    image_files = list(sample_folder.glob('*.jpg')) + list(sample_folder.glob('*.JPG'))
    
    if not image_files:
        print(f"❌ No images found in {sample_folder.name}")
        return False
    
    # Try to open and validate a few sample images
    valid_images = 0
    invalid_images = 0
    
    for i, img_file in enumerate(image_files[:5]):  # Check first 5 images
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                print(f"   ✅ {img_file.name}: {width}x{height} pixels, mode: {img.mode}")
                valid_images += 1
        except Exception as e:
            print(f"   ❌ {img_file.name}: Error - {e}")
            invalid_images += 1
    
    print(f"✅ Sample validation complete: {valid_images} valid, {invalid_images} invalid")
    return valid_images > 0

def create_simple_visualization(folder_stats):
    """Create a simple visualization of the data distribution"""
    print("\n📊 Creating visualization...")
    
    try:
        # Create analysis directory if it doesn't exist
        os.makedirs('data/analysis', exist_ok=True)
        
        # Create bar plot of folder sizes
        plt.figure(figsize=(15, 8))
        folders = list(folder_stats.keys())
        counts = list(folder_stats.values())
        
        plt.subplot(1, 2, 1)
        plt.bar(folders, counts)
        plt.title('Images per Sign Class')
        plt.xlabel('Sign Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        
        # Create pie chart of top 10 classes
        plt.subplot(1, 2, 2)
        top_10 = dict(sorted(folder_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        plt.pie(top_10.values(), labels=top_10.keys(), autopct='%1.1f%%')
        plt.title('Top 10 Sign Classes Distribution')
        
        plt.tight_layout()
        plt.savefig('data/analysis/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization saved to data/analysis/class_distribution.png")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False

def generate_summary_report(df, folder_stats, total_images):
    """Generate a summary report"""
    print("\n📋 Generating summary report...")
    
    try:
        os.makedirs('data/analysis', exist_ok=True)
        
        with open('data/analysis/validation_report.txt', 'w') as f:
            f.write("Arabic Sign Language Dataset Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Summary:\n")
            f.write(f"- CSV rows: {len(df) if df is not None else 'N/A'}\n")
            f.write(f"- Image folders: {len(folder_stats)}\n")
            f.write(f"- Total images: {total_images:,}\n")
            f.write(f"- Average images per class: {total_images // len(folder_stats) if folder_stats else 0:,}\n\n")
            
            if folder_stats:
                f.write("Class Distribution:\n")
                for class_name, count in sorted(folder_stats.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {class_name}: {count:,} images\n")
        
        print("✅ Report saved to data/analysis/validation_report.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Arabic Sign Language Dataset - Simple Validation")
    print("=" * 60)
    
    # Step 1: Basic setup validation
    if not validate_basic_setup():
        return False
    
    # Step 2: Analyze CSV data
    df = analyze_csv_data()
    
    # Step 3: Check image folders
    folder_stats, total_images = check_image_folders()
    
    # Step 4: Validate sample images
    if not validate_sample_images():
        print("⚠️  Warning: Issues found with sample images")
    
    # Step 5: Create visualization
    if folder_stats:
        create_simple_visualization(folder_stats)
    
    # Step 6: Generate report
    generate_summary_report(df, folder_stats, total_images)
    
    # Final summary
    print(f"\n🎉 Validation Complete!")
    print(f"✅ Found {len(folder_stats)} sign classes")
    print(f"✅ Found {total_images:,} total images")
    
    if total_images > 50000:
        print("🎉 Dataset looks excellent! Ready for next phase.")
        print("\n📝 Next steps:")
        print("1. Review the analysis in data/analysis/")
        print("2. Check data/analysis/class_distribution.png")
        print("3. Read data/analysis/validation_report.txt")
        print("4. Ready to proceed with model development!")
    else:
        print("⚠️  Dataset is smaller than expected but still usable.")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Data validation completed successfully!")
    else:
        print("\n❌ Data validation failed. Please check your setup.")