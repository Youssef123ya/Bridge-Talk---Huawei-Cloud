import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDataValidator:
    """Comprehensive validation and quality assessment for Arabic Sign Language dataset"""

    def __init__(self, labels_file: str, data_dir: str, max_workers: int = 4):
        self.labels_file = labels_file
        self.data_dir = data_dir
        self.max_workers = max_workers
        self.df = pd.read_csv(labels_file)
        self.validation_results = {}

    def validate_file_integrity(self) -> Dict[str, Any]:
        """Validate file existence and basic integrity"""

        print("🔍 Validating file integrity...")

        results = {
            'existing_files': [],
            'missing_files': [],
            'corrupted_files': [],
            'duplicate_files': [],
            'file_hashes': {}
        }

        def check_single_file(row):
            file_path = os.path.join(self.data_dir, row['File_Name'])
            result = {
                'filename': row['File_Name'],
                'class': row['Class'],
                'exists': False,
                'readable': False,
                'hash': None,
                'file_size': 0
            }

            if os.path.exists(file_path):
                result['exists'] = True
                result['file_size'] = os.path.getsize(file_path)

                try:
                    # Try to open and verify image
                    with Image.open(file_path) as img:
                        img.verify()

                    # Calculate hash for duplicate detection
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    result['readable'] = True
                    result['hash'] = file_hash

                except Exception as e:
                    result['error'] = str(e)

            return result

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(check_single_file, row) for _, row in self.df.iterrows()]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Checking files"):
                result = future.result()

                if result['exists']:
                    if result['readable']:
                        results['existing_files'].append(result['filename'])

                        # Check for duplicates
                        file_hash = result['hash']
                        if file_hash in results['file_hashes']:
                            results['duplicate_files'].append({
                                'original': results['file_hashes'][file_hash],
                                'duplicate': result['filename']
                            })
                        else:
                            results['file_hashes'][file_hash] = result['filename']
                    else:
                        results['corrupted_files'].append(result['filename'])
                else:
                    results['missing_files'].append(result['filename'])

        # Summary statistics
        results['summary'] = {
            'total_files': len(self.df),
            'existing_files': len(results['existing_files']),
            'missing_files': len(results['missing_files']),
            'corrupted_files': len(results['corrupted_files']),
            'duplicate_files': len(results['duplicate_files']),
            'usable_files': len(results['existing_files']) - len(results['corrupted_files'])
        }

        print(f"   ✅ Existing: {results['summary']['existing_files']:,}")
        print(f"   ❌ Missing: {results['summary']['missing_files']:,}")
        print(f"   🔧 Corrupted: {results['summary']['corrupted_files']:,}")
        print(f"   📋 Duplicates: {results['summary']['duplicate_files']:,}")
        print(f"   ✅ Usable: {results['summary']['usable_files']:,}")

        return results

    def analyze_image_properties(self, sample_size: int = None) -> Dict[str, Any]:
        """Comprehensive image property analysis"""

        print("📸 Analyzing image properties...")

        if sample_size is None:
            sample_df = self.df
        else:
            sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)

        properties = {
            'dimensions': [],
            'aspect_ratios': [],
            'file_sizes': [],
            'channels': [],
            'color_stats': [],
            'brightness_stats': [],
            'contrast_stats': [],
            'blur_scores': []
        }

        def analyze_single_image(row):
            file_path = os.path.join(self.data_dir, row['File_Name'])
            result = {'filename': row['File_Name'], 'class': row['Class']}

            if not os.path.exists(file_path):
                return None

            try:
                # Basic properties
                with Image.open(file_path) as img:
                    width, height = img.size
                    channels = len(img.getbands())

                    result.update({
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height,
                        'channels': channels,
                        'file_size': os.path.getsize(file_path)
                    })

                    # Convert to RGB for analysis
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Color statistics
                    stat = ImageStat.Stat(img)
                    result.update({
                        'mean_r': stat.mean[0],
                        'mean_g': stat.mean[1], 
                        'mean_b': stat.mean[2],
                        'std_r': stat.stddev[0],
                        'std_g': stat.stddev[1],
                        'std_b': stat.stddev[2]
                    })

                    # Convert to numpy for advanced analysis
                    img_array = np.array(img)

                    # Brightness (mean luminance)
                    brightness = np.mean(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
                    result['brightness'] = brightness

                    # Contrast (std of luminance)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    contrast = np.std(gray)
                    result['contrast'] = contrast

                    # Blur detection (Laplacian variance)
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    result['blur_score'] = blur_score

                return result

            except Exception as e:
                result['error'] = str(e)
                return result

        # Process images in parallel
        valid_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(analyze_single_image, row) for _, row in sample_df.iterrows()]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing images"):
                result = future.result()
                if result and 'error' not in result:
                    valid_results.append(result)

        # Compile statistics
        if valid_results:
            for key in ['width', 'height', 'aspect_ratio', 'file_size', 'channels', 
                       'brightness', 'contrast', 'blur_score']:
                values = [r[key] for r in valid_results if key in r]
                if values:
                    properties[key.replace('_', '_') + '_stats'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75)
                    }

        # Quality assessments
        properties['quality_issues'] = {
            'low_resolution': len([r for r in valid_results 
                                 if r.get('width', 0) < 100 or r.get('height', 0) < 100]),
            'very_blurry': len([r for r in valid_results 
                              if r.get('blur_score', float('inf')) < 100]),
            'very_dark': len([r for r in valid_results 
                            if r.get('brightness', 255) < 50]),
            'very_bright': len([r for r in valid_results 
                              if r.get('brightness', 0) > 200]),
            'low_contrast': len([r for r in valid_results 
                               if r.get('contrast', float('inf')) < 20])
        }

        print(f"   📊 Analyzed {len(valid_results)} images")
        print(f"   ⚠️  Quality issues found:")
        for issue, count in properties['quality_issues'].items():
            if count > 0:
                print(f"      {issue}: {count}")

        return properties

    def detect_class_imbalance(self) -> Dict[str, Any]:
        """Detect and analyze class imbalance"""

        print("⚖️  Analyzing class balance...")

        class_counts = self.df['Class'].value_counts()

        analysis = {
            'class_counts': class_counts.to_dict(),
            'statistics': {
                'total_classes': len(class_counts),
                'mean_samples': class_counts.mean(),
                'std_samples': class_counts.std(),
                'min_samples': class_counts.min(),
                'max_samples': class_counts.max(),
                'imbalance_ratio': class_counts.max() / class_counts.min()
            },
            'imbalance_categories': {
                'severely_underrepresented': [],  # < 50% of mean
                'underrepresented': [],           # 50-80% of mean  
                'well_represented': [],           # 80-120% of mean
                'overrepresented': [],            # 120-150% of mean
                'severely_overrepresented': []    # > 150% of mean
            }
        }

        mean_samples = class_counts.mean()

        for class_name, count in class_counts.items():
            ratio = count / mean_samples

            if ratio < 0.5:
                analysis['imbalance_categories']['severely_underrepresented'].append(class_name)
            elif ratio < 0.8:
                analysis['imbalance_categories']['underrepresented'].append(class_name)
            elif ratio <= 1.2:
                analysis['imbalance_categories']['well_represented'].append(class_name)
            elif ratio <= 1.5:
                analysis['imbalance_categories']['overrepresented'].append(class_name)
            else:
                analysis['imbalance_categories']['severely_overrepresented'].append(class_name)

        # Print summary
        print(f"   📊 Classes: {analysis['statistics']['total_classes']}")
        print(f"   📈 Imbalance ratio: {analysis['statistics']['imbalance_ratio']:.2f}")
        print(f"   ⚠️  Severely underrepresented: {len(analysis['imbalance_categories']['severely_underrepresented'])}")
        print(f"   ⚠️  Severely overrepresented: {len(analysis['imbalance_categories']['severely_overrepresented'])}")

        return analysis

    def create_comprehensive_report(self, output_dir: str = 'data/analysis') -> str:
        """Generate comprehensive validation report"""

        os.makedirs(output_dir, exist_ok=True)

        print("📋 Generating comprehensive validation report...")

        # Run all validations
        file_integrity = self.validate_file_integrity()
        image_properties = self.analyze_image_properties(sample_size=2000)
        class_balance = self.detect_class_imbalance()

        # Generate report
        report_content = f"""
Arabic Sign Language Dataset - Comprehensive Validation Report
=============================================================

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {self.labels_file}
Data Directory: {self.data_dir}

SUMMARY
-------
Total files in dataset: {len(self.df):,}
Usable files: {file_integrity['summary']['usable_files']:,} ({file_integrity['summary']['usable_files']/len(self.df)*100:.1f}%)
Missing files: {file_integrity['summary']['missing_files']:,}
Corrupted files: {file_integrity['summary']['corrupted_files']:,}
Duplicate files: {file_integrity['summary']['duplicate_files']:,}

FILE INTEGRITY
--------------
✅ Existing files: {file_integrity['summary']['existing_files']:,} ({file_integrity['summary']['existing_files']/len(self.df)*100:.1f}%)
❌ Missing files: {file_integrity['summary']['missing_files']:,} ({file_integrity['summary']['missing_files']/len(self.df)*100:.1f}%)
🔧 Corrupted files: {file_integrity['summary']['corrupted_files']:,} ({file_integrity['summary']['corrupted_files']/len(self.df)*100:.1f}%)
📋 Duplicate files: {file_integrity['summary']['duplicate_files']:,}

IMAGE PROPERTIES
----------------
"""

        if 'width_stats' in image_properties:
            width_stats = image_properties['width_stats']
            height_stats = image_properties['height_stats']
            report_content += f"""
Average dimensions: {width_stats['mean']:.0f}x{height_stats['mean']:.0f} pixels
Dimension range: {width_stats['min']:.0f}x{height_stats['min']:.0f} to {width_stats['max']:.0f}x{height_stats['max']:.0f} pixels
Median dimensions: {width_stats['median']:.0f}x{height_stats['median']:.0f} pixels
"""

        if 'file_size_stats' in image_properties:
            size_stats = image_properties['file_size_stats']
            report_content += f"""
Average file size: {size_stats['mean']/1024:.1f} KB
File size range: {size_stats['min']/1024:.1f} KB to {size_stats['max']/1024:.1f} KB
"""

        quality_issues = image_properties.get('quality_issues', {})
        if quality_issues:
            report_content += f"""
QUALITY ISSUES
--------------
Low resolution images: {quality_issues.get('low_resolution', 0)}
Very blurry images: {quality_issues.get('very_blurry', 0)}
Very dark images: {quality_issues.get('very_dark', 0)}
Very bright images: {quality_issues.get('very_bright', 0)}
Low contrast images: {quality_issues.get('low_contrast', 0)}
"""

        report_content += f"""
CLASS BALANCE
-------------
Total classes: {class_balance['statistics']['total_classes']}
Average samples per class: {class_balance['statistics']['mean_samples']:.1f}
Standard deviation: {class_balance['statistics']['std_samples']:.1f}
Min/Max samples: {class_balance['statistics']['min_samples']}/{class_balance['statistics']['max_samples']}
Imbalance ratio: {class_balance['statistics']['imbalance_ratio']:.2f}

Class Distribution Categories:
- Severely underrepresented (<50% of mean): {len(class_balance['imbalance_categories']['severely_underrepresented'])}
- Underrepresented (50-80% of mean): {len(class_balance['imbalance_categories']['underrepresented'])}
- Well represented (80-120% of mean): {len(class_balance['imbalance_categories']['well_represented'])}
- Overrepresented (120-150% of mean): {len(class_balance['imbalance_categories']['overrepresented'])}
- Severely overrepresented (>150% of mean): {len(class_balance['imbalance_categories']['severely_overrepresented'])}

RECOMMENDATIONS
---------------
"""

        # Add recommendations
        recommendations = []

        if file_integrity['summary']['missing_files'] > 0:
            recommendations.append(f"• {file_integrity['summary']['missing_files']} missing files should be located and added")

        if file_integrity['summary']['corrupted_files'] > 0:
            recommendations.append(f"• {file_integrity['summary']['corrupted_files']} corrupted files should be repaired or removed")

        if file_integrity['summary']['duplicate_files'] > 0:
            recommendations.append(f"• {file_integrity['summary']['duplicate_files']} duplicate files should be reviewed")

        if class_balance['statistics']['imbalance_ratio'] > 3:
            recommendations.append(f"• Significant class imbalance detected (ratio: {class_balance['statistics']['imbalance_ratio']:.1f}) - consider data augmentation")

        if quality_issues.get('low_resolution', 0) > 100:
            recommendations.append(f"• {quality_issues['low_resolution']} low resolution images may affect model performance")

        if quality_issues.get('very_blurry', 0) > 50:
            recommendations.append(f"• {quality_issues['very_blurry']} very blurry images should be reviewed")

        if not recommendations:
            recommendations.append("• Dataset appears to be in good condition for training")

        report_content += "\n".join(recommendations)

        # Save report
        report_file = os.path.join(output_dir, 'comprehensive_validation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_content)

        # Save detailed results as JSON
        detailed_results = {
            'file_integrity': file_integrity,
            'image_properties': image_properties,
            'class_balance': class_balance
        }

        json_file = os.path.join(output_dir, 'detailed_validation_results.json')
        with open(json_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        print(f"📋 Comprehensive report saved to {report_file}")
        print(f"📊 Detailed results saved to {json_file}")

        return report_content

def run_comprehensive_validation(labels_file: str, 
                                data_dir: str, 
                                output_dir: str = 'data/analysis') -> Dict[str, Any]:
    """Run complete data validation pipeline"""

    validator = ComprehensiveDataValidator(labels_file, data_dir)
    report = validator.create_comprehensive_report(output_dir)

    return {
        'report': report,
        'validator': validator
    }
