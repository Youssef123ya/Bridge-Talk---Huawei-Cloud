#!/usr/bin/env python3
"""
Comprehensive Data Validator for Arabic Sign Language Recognition
Provides thorough validation of dataset integrity, quality, and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
import os
from pathlib import Path
from collections import Counter, defaultdict
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
from datetime import datetime
import cv2
warnings.filterwarnings('ignore')

class ComprehensiveDataValidator:
    """Comprehensive validator for image datasets"""
    
    def __init__(self, labels_file: str, data_dir: str):
        """
        Initialize the validator
        
        Args:
            labels_file: Path to CSV file with labels
            data_dir: Directory containing image data
        """
        self.labels_file = labels_file
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(labels_file)
        
        # Standardize column names
        self.df.columns = self.df.columns.str.strip()
        if 'Class' in self.df.columns:
            self.df['class'] = self.df['Class']
        if 'File_Name' in self.df.columns:
            self.df['filename'] = self.df['File_Name']
        
        self.classes = sorted(self.df['class'].unique())
        self.validation_results = {}
        
        print(f"📊 Validator initialized:")
        print(f"   Dataset: {len(self.df)} samples")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Data directory: {self.data_dir}")
        
    def validate_file_integrity(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate file existence and integrity
        
        Args:
            sample_size: Number of files to sample for integrity check (None = all)
            
        Returns:
            Dictionary with validation results
        """
        
        print("🔍 Validating file integrity...")
        
        results = {
            'missing_files': [],
            'corrupted_files': [],
            'valid_files': [],
            'file_paths': {},
            'summary': {}
        }
        
        # Check if files exist and are accessible
        total_files = len(self.df)
        sample_df = self.df.sample(n=min(sample_size or total_files, total_files), random_state=42)
        
        for idx, row in sample_df.iterrows():
            filename = row['filename']
            class_name = row['class']
            
            # Try different possible paths
            possible_paths = [
                self.data_dir / class_name / filename,
                self.data_dir / filename,
                self.data_dir / class_name.lower() / filename,
                self.data_dir / class_name.upper() / filename
            ]
            
            file_found = False
            file_path = None
            
            for path in possible_paths:
                if path.exists():
                    file_found = True
                    file_path = path
                    break
            
            if not file_found:
                results['missing_files'].append({
                    'filename': filename,
                    'class': class_name,
                    'expected_paths': [str(p) for p in possible_paths]
                })
            else:
                # Check if file can be opened
                try:
                    with Image.open(file_path) as img:
                        # Basic validation - try to load the image
                        img.verify()
                    
                    # Re-open for additional checks (verify() closes the file)
                    with Image.open(file_path) as img:
                        width, height = img.size
                        mode = img.mode
                        
                    results['valid_files'].append({
                        'filename': filename,
                        'class': class_name,
                        'path': str(file_path),
                        'width': width,
                        'height': height,
                        'mode': mode
                    })
                    
                    results['file_paths'][filename] = str(file_path)
                    
                except Exception as e:
                    results['corrupted_files'].append({
                        'filename': filename,
                        'class': class_name,
                        'path': str(file_path),
                        'error': str(e)
                    })
        
        # Calculate summary statistics
        total_checked = len(sample_df)
        valid_count = len(results['valid_files'])
        missing_count = len(results['missing_files'])
        corrupted_count = len(results['corrupted_files'])
        
        results['summary'] = {
            'total_checked': total_checked,
            'valid_files': valid_count,
            'missing_files': missing_count,
            'corrupted_files': corrupted_count,
            'success_rate': valid_count / total_checked if total_checked > 0 else 0,
            'usable_files': valid_count
        }
        
        print(f"✅ File integrity check complete:")
        print(f"   Checked: {total_checked:,} files")
        print(f"   Valid: {valid_count:,} ({valid_count/total_checked:.1%})")
        print(f"   Missing: {missing_count:,} ({missing_count/total_checked:.1%})")
        print(f"   Corrupted: {corrupted_count:,} ({corrupted_count/total_checked:.1%})")
        
        self.validation_results['file_integrity'] = results
        return results
    
    def analyze_image_properties(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze image properties like dimensions, color channels, quality
        
        Args:
            sample_size: Number of images to analyze
            
        Returns:
            Dictionary with image analysis results
        """
        
        print(f"🖼️  Analyzing image properties (sample size: {sample_size})...")
        
        # Get valid files from integrity check
        if 'file_integrity' not in self.validation_results:
            self.validate_file_integrity(sample_size)
        
        valid_files = self.validation_results['file_integrity']['valid_files']
        sample_files = valid_files[:sample_size] if len(valid_files) > sample_size else valid_files
        
        results = {
            'dimensions': {'widths': [], 'heights': [], 'aspects': []},
            'color_modes': [],
            'file_sizes': [],
            'brightness': [],
            'contrast': [],
            'sharpness': [],
            'summary': {}
        }
        
        for file_info in sample_files:
            try:
                file_path = file_info['path']
                
                # Basic properties
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    
                    results['dimensions']['widths'].append(width)
                    results['dimensions']['heights'].append(height)
                    results['dimensions']['aspects'].append(width / height)
                    results['color_modes'].append(mode)
                    
                    # File size
                    file_size = os.path.getsize(file_path)
                    results['file_sizes'].append(file_size)
                    
                    # Image quality metrics
                    if mode in ['L', 'RGB']:
                        # Convert to grayscale for consistent analysis
                        gray_img = img.convert('L')
                        
                        # Brightness (mean pixel value)
                        stat = ImageStat.Stat(gray_img)
                        brightness = stat.mean[0]
                        results['brightness'].append(brightness)
                        
                        # Contrast (standard deviation)
                        contrast = stat.stddev[0]
                        results['contrast'].append(contrast)
                        
                        # Sharpness (using Laplacian variance)
                        try:
                            img_array = np.array(gray_img)
                            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
                            results['sharpness'].append(laplacian_var)
                        except:
                            results['sharpness'].append(0)
                    
            except Exception as e:
                print(f"⚠️  Error analyzing {file_info['filename']}: {e}")
                continue
        
        # Calculate summary statistics
        if results['dimensions']['widths']:
            results['summary'] = {
                'total_analyzed': len(results['dimensions']['widths']),
                'dimensions': {
                    'width_stats': {
                        'min': min(results['dimensions']['widths']),
                        'max': max(results['dimensions']['widths']),
                        'mean': np.mean(results['dimensions']['widths']),
                        'std': np.std(results['dimensions']['widths'])
                    },
                    'height_stats': {
                        'min': min(results['dimensions']['heights']),
                        'max': max(results['dimensions']['heights']),
                        'mean': np.mean(results['dimensions']['heights']),
                        'std': np.std(results['dimensions']['heights'])
                    },
                    'aspect_stats': {
                        'min': min(results['dimensions']['aspects']),
                        'max': max(results['dimensions']['aspects']),
                        'mean': np.mean(results['dimensions']['aspects']),
                        'std': np.std(results['dimensions']['aspects'])
                    }
                },
                'color_modes': dict(Counter(results['color_modes'])),
                'file_size_stats': {
                    'min_kb': min(results['file_sizes']) / 1024,
                    'max_kb': max(results['file_sizes']) / 1024,
                    'mean_kb': np.mean(results['file_sizes']) / 1024,
                    'std_kb': np.std(results['file_sizes']) / 1024
                } if results['file_sizes'] else {},
                'quality_stats': {
                    'brightness': {
                        'min': min(results['brightness']) if results['brightness'] else 0,
                        'max': max(results['brightness']) if results['brightness'] else 0,
                        'mean': np.mean(results['brightness']) if results['brightness'] else 0
                    },
                    'contrast': {
                        'min': min(results['contrast']) if results['contrast'] else 0,
                        'max': max(results['contrast']) if results['contrast'] else 0,
                        'mean': np.mean(results['contrast']) if results['contrast'] else 0
                    },
                    'sharpness': {
                        'min': min(results['sharpness']) if results['sharpness'] else 0,
                        'max': max(results['sharpness']) if results['sharpness'] else 0,
                        'mean': np.mean(results['sharpness']) if results['sharpness'] else 0
                    }
                }
            }
        
        print(f"✅ Image properties analyzed:")
        if results['summary']:
            dims = results['summary']['dimensions']
            print(f"   Dimensions: {dims['width_stats']['mean']:.0f}x{dims['height_stats']['mean']:.0f} (avg)")
            print(f"   Color modes: {results['summary']['color_modes']}")
            print(f"   File size: {results['summary']['file_size_stats']['mean_kb']:.1f} KB (avg)")
        
        self.validation_results['image_properties'] = results
        return results
    
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """
        Analyze class distribution and balance
        
        Returns:
            Dictionary with class distribution analysis
        """
        
        print("📊 Analyzing class distribution...")
        
        class_counts = self.df['class'].value_counts()
        total_samples = len(self.df)
        
        results = {
            'class_counts': class_counts.to_dict(),
            'class_proportions': (class_counts / total_samples).to_dict(),
            'summary': {
                'num_classes': len(self.classes),
                'total_samples': total_samples,
                'min_samples': class_counts.min(),
                'max_samples': class_counts.max(),
                'mean_samples': class_counts.mean(),
                'std_samples': class_counts.std(),
                'balance_ratio': class_counts.min() / class_counts.max()
            }
        }
        
        # Class balance analysis
        mean_count = class_counts.mean()
        balanced_classes = sum(1 for count in class_counts if 0.8 <= count/mean_count <= 1.2)
        results['summary']['balanced_classes'] = balanced_classes
        results['summary']['balance_score'] = balanced_classes / len(self.classes)
        
        print(f"✅ Class distribution analyzed:")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Samples per class: {results['summary']['min_samples']}-{results['summary']['max_samples']}")
        print(f"   Balance score: {results['summary']['balance_score']:.2f}")
        
        self.validation_results['class_distribution'] = results
        return results
    
    def create_comprehensive_visualizations(self, output_dir: str):
        """
        Create comprehensive visualizations of validation results
        
        Args:
            output_dir: Directory to save visualizations
        """
        
        os.makedirs(output_dir, exist_ok=True)
        print("📊 Creating comprehensive visualizations...")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Class distribution bar plot
        if 'class_distribution' in self.validation_results:
            ax1 = fig.add_subplot(gs[0, :2])
            class_data = self.validation_results['class_distribution']
            classes = list(class_data['class_counts'].keys())
            counts = list(class_data['class_counts'].values())
            
            bars = ax1.bar(classes, counts, alpha=0.8, color='skyblue')
            ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Number of Samples')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 2. File integrity pie chart
        if 'file_integrity' in self.validation_results:
            ax2 = fig.add_subplot(gs[0, 2])
            integrity_data = self.validation_results['file_integrity']['summary']
            
            sizes = [integrity_data['valid_files'], integrity_data['missing_files'], 
                    integrity_data['corrupted_files']]
            labels = ['Valid', 'Missing', 'Corrupted']
            colors = ['lightgreen', 'lightcoral', 'lightyellow']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('File Integrity Status', fontsize=14, fontweight='bold')
        
        # 3. Image dimensions scatter plot
        if 'image_properties' in self.validation_results:
            img_data = self.validation_results['image_properties']
            
            if img_data['dimensions']['widths']:
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.scatter(img_data['dimensions']['widths'], 
                           img_data['dimensions']['heights'], 
                           alpha=0.6, s=20)
                ax3.set_title('Image Dimensions Distribution')
                ax3.set_xlabel('Width (pixels)')
                ax3.set_ylabel('Height (pixels)')
                ax3.grid(True, alpha=0.3)
                
                # 4. Aspect ratio histogram
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.hist(img_data['dimensions']['aspects'], bins=30, alpha=0.7, color='orange')
                ax4.set_title('Aspect Ratio Distribution')
                ax4.set_xlabel('Aspect Ratio (Width/Height)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
                
                # 5. File size distribution
                ax5 = fig.add_subplot(gs[1, 2])
                file_sizes_kb = [size/1024 for size in img_data['file_sizes']]
                ax5.hist(file_sizes_kb, bins=30, alpha=0.7, color='green')
                ax5.set_title('File Size Distribution')
                ax5.set_xlabel('File Size (KB)')
                ax5.set_ylabel('Frequency')
                ax5.grid(True, alpha=0.3)
        
        # 6. Quality metrics
        if 'image_properties' in self.validation_results:
            img_data = self.validation_results['image_properties']
            
            if img_data['brightness']:
                # Brightness distribution
                ax6 = fig.add_subplot(gs[2, 0])
                ax6.hist(img_data['brightness'], bins=30, alpha=0.7, color='yellow')
                ax6.set_title('Brightness Distribution')
                ax6.set_xlabel('Mean Pixel Value')
                ax6.set_ylabel('Frequency')
                ax6.grid(True, alpha=0.3)
                
                # Contrast distribution
                ax7 = fig.add_subplot(gs[2, 1])
                ax7.hist(img_data['contrast'], bins=30, alpha=0.7, color='purple')
                ax7.set_title('Contrast Distribution')
                ax7.set_xlabel('Pixel Standard Deviation')
                ax7.set_ylabel('Frequency')
                ax7.grid(True, alpha=0.3)
                
                # Sharpness distribution
                ax8 = fig.add_subplot(gs[2, 2])
                sharpness_filtered = [s for s in img_data['sharpness'] if s > 0]
                if sharpness_filtered:
                    ax8.hist(sharpness_filtered, bins=30, alpha=0.7, color='red')
                ax8.set_title('Sharpness Distribution')
                ax8.set_xlabel('Laplacian Variance')
                ax8.set_ylabel('Frequency')
                ax8.grid(True, alpha=0.3)
        
        # 7. Summary statistics table
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table data
        table_data = []
        
        if 'class_distribution' in self.validation_results:
            cd = self.validation_results['class_distribution']['summary']
            table_data.append(['Classes', f"{cd['num_classes']}"])
            table_data.append(['Total Samples', f"{cd['total_samples']:,}"])
            table_data.append(['Samples per Class', f"{cd['min_samples']}-{cd['max_samples']}"])
            table_data.append(['Balance Score', f"{cd['balance_score']:.3f}"])
        
        if 'file_integrity' in self.validation_results:
            fi = self.validation_results['file_integrity']['summary']
            table_data.append(['Success Rate', f"{fi['success_rate']:.1%}"])
            table_data.append(['Valid Files', f"{fi['valid_files']:,}"])
        
        if 'image_properties' in self.validation_results:
            ip = self.validation_results['image_properties']['summary']
            if 'dimensions' in ip:
                dims = ip['dimensions']
                avg_w = dims['width_stats']['mean']
                avg_h = dims['height_stats']['mean']
                table_data.append(['Avg Dimensions', f"{avg_w:.0f}x{avg_h:.0f}"])
        
        if table_data:
            table = ax9.table(cellText=table_data, 
                             colLabels=['Metric', 'Value'],
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.3, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            ax9.set_title('Dataset Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Comprehensive Dataset Validation Report', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save the comprehensive visualization
        plt.savefig(os.path.join(output_dir, 'comprehensive_validation_report.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comprehensive visualizations created:")
        print(f"   📊 {output_dir}/comprehensive_validation_report.png")
    
    def generate_detailed_report(self, output_dir: str):
        """
        Generate detailed text report of validation results
        
        Args:
            output_dir: Directory to save the report
        """
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'comprehensive_validation_report.txt')
        
        print("📝 Generating detailed validation report...")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE DATASET VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.labels_file}\n")
            f.write(f"Data Directory: {self.data_dir}\n\n")
            
            # File Integrity Section
            if 'file_integrity' in self.validation_results:
                f.write("FILE INTEGRITY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                fi = self.validation_results['file_integrity']['summary']
                f.write(f"Total files checked: {fi['total_checked']:,}\n")
                f.write(f"Valid files: {fi['valid_files']:,} ({fi['success_rate']:.1%})\n")
                f.write(f"Missing files: {fi['missing_files']:,}\n")
                f.write(f"Corrupted files: {fi['corrupted_files']:,}\n\n")
                
                # List problematic files
                if self.validation_results['file_integrity']['missing_files']:
                    f.write("Missing Files (first 10):\n")
                    for i, file_info in enumerate(self.validation_results['file_integrity']['missing_files'][:10]):
                        f.write(f"  {i+1}. {file_info['filename']} (class: {file_info['class']})\n")
                    f.write("\n")
                
                if self.validation_results['file_integrity']['corrupted_files']:
                    f.write("Corrupted Files (first 10):\n")
                    for i, file_info in enumerate(self.validation_results['file_integrity']['corrupted_files'][:10]):
                        f.write(f"  {i+1}. {file_info['filename']} - {file_info['error']}\n")
                    f.write("\n")
            
            # Class Distribution Section
            if 'class_distribution' in self.validation_results:
                f.write("CLASS DISTRIBUTION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                cd = self.validation_results['class_distribution']
                f.write(f"Number of classes: {cd['summary']['num_classes']}\n")
                f.write(f"Total samples: {cd['summary']['total_samples']:,}\n")
                f.write(f"Samples per class: {cd['summary']['min_samples']}-{cd['summary']['max_samples']}\n")
                f.write(f"Mean samples per class: {cd['summary']['mean_samples']:.1f}\n")
                f.write(f"Balance ratio: {cd['summary']['balance_ratio']:.3f}\n")
                f.write(f"Balance score: {cd['summary']['balance_score']:.3f}\n\n")
                
                f.write("Class breakdown:\n")
                for class_name, count in sorted(cd['class_counts'].items()):
                    percentage = count / cd['summary']['total_samples'] * 100
                    f.write(f"  {class_name}: {count:,} samples ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Image Properties Section
            if 'image_properties' in self.validation_results:
                f.write("IMAGE PROPERTIES ANALYSIS\n")
                f.write("-" * 40 + "\n")
                ip = self.validation_results['image_properties']['summary']
                f.write(f"Images analyzed: {ip['total_analyzed']:,}\n\n")
                
                if 'dimensions' in ip:
                    dims = ip['dimensions']
                    f.write("Dimensions:\n")
                    f.write(f"  Width: {dims['width_stats']['min']}-{dims['width_stats']['max']} (avg: {dims['width_stats']['mean']:.1f})\n")
                    f.write(f"  Height: {dims['height_stats']['min']}-{dims['height_stats']['max']} (avg: {dims['height_stats']['mean']:.1f})\n")
                    f.write(f"  Aspect ratio: {dims['aspect_stats']['min']:.2f}-{dims['aspect_stats']['max']:.2f} (avg: {dims['aspect_stats']['mean']:.2f})\n\n")
                
                if 'color_modes' in ip:
                    f.write("Color modes:\n")
                    for mode, count in ip['color_modes'].items():
                        f.write(f"  {mode}: {count} images\n")
                    f.write("\n")
                
                if 'file_size_stats' in ip and ip['file_size_stats']:
                    fs = ip['file_size_stats']
                    f.write("File sizes:\n")
                    f.write(f"  Range: {fs['min_kb']:.1f}-{fs['max_kb']:.1f} KB\n")
                    f.write(f"  Average: {fs['mean_kb']:.1f} KB\n\n")
                
                if 'quality_stats' in ip:
                    qs = ip['quality_stats']
                    f.write("Quality metrics:\n")
                    f.write(f"  Brightness: {qs['brightness']['min']:.1f}-{qs['brightness']['max']:.1f} (avg: {qs['brightness']['mean']:.1f})\n")
                    f.write(f"  Contrast: {qs['contrast']['min']:.1f}-{qs['contrast']['max']:.1f} (avg: {qs['contrast']['mean']:.1f})\n")
                    f.write(f"  Sharpness: {qs['sharpness']['min']:.1f}-{qs['sharpness']['max']:.1f} (avg: {qs['sharpness']['mean']:.1f})\n\n")
            
            # Recommendations Section
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            recommendations = []
            
            if 'file_integrity' in self.validation_results:
                fi = self.validation_results['file_integrity']['summary']
                if fi['success_rate'] < 0.95:
                    recommendations.append("• Address missing or corrupted files to improve data quality")
                if fi['missing_files'] > 0:
                    recommendations.append("• Check file paths and naming conventions for missing files")
            
            if 'class_distribution' in self.validation_results:
                cd = self.validation_results['class_distribution']['summary']
                if cd['balance_score'] < 0.8:
                    recommendations.append("• Consider data augmentation or resampling to balance classes")
                if cd['balance_ratio'] < 0.5:
                    recommendations.append("• Significant class imbalance detected - use stratified sampling")
            
            if 'image_properties' in self.validation_results:
                ip = self.validation_results['image_properties']['summary']
                if 'dimensions' in ip:
                    dims = ip['dimensions']
                    width_cv = dims['width_stats']['std'] / dims['width_stats']['mean']
                    height_cv = dims['height_stats']['std'] / dims['height_stats']['mean']
                    if width_cv > 0.3 or height_cv > 0.3:
                        recommendations.append("• High dimension variability - consider resizing to standard dimensions")
            
            if not recommendations:
                recommendations.append("• Dataset appears to be in good condition for training")
            
            for rec in recommendations:
                f.write(f"{rec}\n")
        
        print(f"✅ Detailed report generated: {report_path}")
        return report_path


def run_comprehensive_validation(labels_file: str, data_dir: str, 
                               output_dir: str, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Run comprehensive validation on the dataset
    
    Args:
        labels_file: Path to labels CSV file
        data_dir: Directory containing images
        output_dir: Directory to save results
        sample_size: Number of images to analyze in detail
        
    Returns:
        Dictionary with validation results
    """
    
    print("🚀 Starting comprehensive dataset validation...")
    
    # Create validator
    validator = ComprehensiveDataValidator(labels_file, data_dir)
    
    # Run all validations
    validator.validate_file_integrity(sample_size)
    validator.analyze_image_properties(sample_size)
    validator.analyze_class_distribution()
    
    # Create visualizations and reports
    os.makedirs(output_dir, exist_ok=True)
    validator.create_comprehensive_visualizations(output_dir)
    validator.generate_detailed_report(output_dir)
    
    # Save raw results
    results_path = os.path.join(output_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(validator.validation_results, f, indent=2, default=str)
    
    print("✅ Comprehensive validation completed!")
    print(f"📊 Results saved to {output_dir}/")
    
    return {
        'validator': validator,
        'results': validator.validation_results,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    # Example usage
    labels_file = "data/ArSL_Data_Labels.csv"
    data_dir = "data/raw"
    output_dir = "data/analysis"
    
    if os.path.exists(labels_file):
        results = run_comprehensive_validation(labels_file, data_dir, output_dir)
        print("✅ Validation complete!")
    else:
        print(f"❌ Labels file not found: {labels_file}")