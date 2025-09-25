#!/usr/bin/env python3
"""
Advanced Data Splitter for Arabic Sign Language Recognition
Provides stratified splitting, cross-validation, and split quality analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import Counter
import os
import json
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataSplitter:
    """Advanced data splitter with stratification and quality analysis"""
    
    def __init__(self, labels_file: str, random_seed: int = 42):
        """
        Initialize the data splitter
        
        Args:
            labels_file: Path to the CSV file with labels
            random_seed: Random seed for reproducibility
        """
        self.labels_file = labels_file
        self.random_seed = random_seed
        self.df = pd.read_csv(labels_file)
        
        # Standardize column names
        self.df.columns = self.df.columns.str.strip()
        if 'Class' in self.df.columns:
            self.df['class'] = self.df['Class']
        elif 'class' not in self.df.columns:
            raise ValueError("No 'Class' or 'class' column found in the labels file")
        
        # Create class mapping
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"📊 Loaded dataset: {len(self.df)} samples, {len(self.classes)} classes")
        
    def stratified_split(self, train_size: float = 0.7, val_size: float = 0.15, 
                        test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits
        
        Args:
            train_size: Proportion for training set
            val_size: Proportion for validation set  
            test_size: Proportion for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        
        # Validate split sizes
        total_size = train_size + val_size + test_size
        if not np.isclose(total_size, 1.0, atol=1e-3):
            raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")
        
        print(f"🔄 Creating stratified splits: {train_size:.1%}/{val_size:.1%}/{test_size:.1%}")
        
        # First split: train vs (val + test)
        sss1 = StratifiedShuffleSplit(
            n_splits=1, 
            train_size=train_size, 
            random_state=self.random_seed
        )
        
        train_idx, temp_idx = next(sss1.split(self.df, self.df['class']))
        train_df = self.df.iloc[train_idx].copy()
        temp_df = self.df.iloc[temp_idx].copy()
        
        # Second split: val vs test from remaining data
        if val_size + test_size > 0:
            val_ratio = val_size / (val_size + test_size)
            sss2 = StratifiedShuffleSplit(
                n_splits=1, 
                train_size=val_ratio, 
                random_state=self.random_seed + 1
            )
            
            val_idx, test_idx = next(sss2.split(temp_df, temp_df['class']))
            val_df = temp_df.iloc[val_idx].copy()
            test_df = temp_df.iloc[test_idx].copy()
        else:
            val_df = pd.DataFrame()
            test_df = temp_df.copy()
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True) 
        test_df = test_df.reset_index(drop=True)
        
        # Add split identifier
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        print(f"✅ Split created:")
        print(f"   Training: {len(train_df):,} samples ({len(train_df)/len(self.df):.1%})")
        print(f"   Validation: {len(val_df):,} samples ({len(val_df)/len(self.df):.1%})")
        print(f"   Test: {len(test_df):,} samples ({len(test_df)/len(self.df):.1%})")
        
        return train_df, val_df, test_df
    
    def analyze_split_quality(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                            test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the quality of data splits
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe  
            test_df: Test dataframe
            
        Returns:
            Dictionary with split quality metrics
        """
        
        print("🔍 Analyzing split quality...")
        
        analysis = {
            'train_dist': train_df['class'].value_counts().to_dict(),
            'val_dist': val_df['class'].value_counts().to_dict(),
            'test_dist': test_df['class'].value_counts().to_dict(),
            'class_balance': {},
            'split_sizes': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            }
        }
        
        # Analyze class balance across splits
        for class_name in self.classes:
            train_count = analysis['train_dist'].get(class_name, 0)
            val_count = analysis['val_dist'].get(class_name, 0)
            test_count = analysis['test_dist'].get(class_name, 0)
            total_count = train_count + val_count + test_count
            
            if total_count > 0:
                analysis['class_balance'][class_name] = {
                    'train_ratio': train_count / total_count,
                    'val_ratio': val_count / total_count,
                    'test_ratio': test_count / total_count,
                    'total_samples': total_count
                }
        
        # Calculate balance metrics
        balance_scores = []
        for class_name, ratios in analysis['class_balance'].items():
            # Ideal ratios based on split sizes
            total_samples = len(self.df)
            ideal_train = len(train_df) / total_samples
            ideal_val = len(val_df) / total_samples  
            ideal_test = len(test_df) / total_samples
            
            # Calculate deviation from ideal
            train_dev = abs(ratios['train_ratio'] - ideal_train)
            val_dev = abs(ratios['val_ratio'] - ideal_val)
            test_dev = abs(ratios['test_ratio'] - ideal_test)
            
            balance_score = 1.0 - (train_dev + val_dev + test_dev) / 3.0
            balance_scores.append(balance_score)
        
        analysis['overall_balance_score'] = np.mean(balance_scores)
        
        print(f"✅ Balance analysis complete:")
        print(f"   Overall balance score: {analysis['overall_balance_score']:.3f}")
        print(f"   Classes with perfect balance: {sum(1 for s in balance_scores if s > 0.95)}/{len(self.classes)}")
        
        return analysis
    
    def create_cross_validation_folds(self, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create stratified cross-validation folds
        
        Args:
            n_folds: Number of CV folds
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        
        print(f"🔄 Creating {n_folds}-fold cross-validation splits...")
        
        skf = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.random_seed
        )
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['class'])):
            train_fold = self.df.iloc[train_idx].copy().reset_index(drop=True)
            val_fold = self.df.iloc[val_idx].copy().reset_index(drop=True)
            
            # Add fold information
            train_fold['fold'] = fold_idx
            train_fold['split'] = 'train'
            val_fold['fold'] = fold_idx
            val_fold['split'] = 'val'
            
            folds.append((train_fold, val_fold))
            
            print(f"   Fold {fold_idx}: {len(train_fold)} train, {len(val_fold)} val")
        
        print("✅ Cross-validation folds created")
        return folds
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame, output_dir: str):
        """
        Save data splits to CSV files
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Directory to save splits
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        # Save class mapping
        class_mapping = {
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': len(self.classes)
        }
        
        with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"✅ Splits saved to {output_dir}/")
        print(f"   📄 train.csv ({len(train_df)} samples)")
        print(f"   📄 val.csv ({len(val_df)} samples)")
        print(f"   📄 test.csv ({len(test_df)} samples)")
        print(f"   📄 class_mapping.json")
    
    def visualize_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame, output_dir: str):
        """
        Create visualizations of data splits
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Directory to save visualizations
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Creating split visualizations...")
        
        # Combine data for visualization
        train_counts = train_df['class'].value_counts()
        val_counts = val_df['class'].value_counts()
        test_counts = test_df['class'].value_counts()
        
        # Create visualization dataframe
        viz_data = []
        for class_name in self.classes:
            viz_data.append({
                'class': class_name,
                'train': train_counts.get(class_name, 0),
                'val': val_counts.get(class_name, 0),
                'test': test_counts.get(class_name, 0)
            })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Split Analysis', fontsize=16, fontweight='bold')
        
        # 1. Stacked bar chart
        ax1 = axes[0, 0]
        x_pos = np.arange(len(self.classes))
        ax1.bar(x_pos, viz_df['train'], label='Train', alpha=0.8)
        ax1.bar(x_pos, viz_df['val'], bottom=viz_df['train'], label='Val', alpha=0.8)
        ax1.bar(x_pos, viz_df['test'], bottom=viz_df['train'] + viz_df['val'], label='Test', alpha=0.8)
        ax1.set_title('Samples per Class by Split')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Split proportions pie chart
        ax2 = axes[0, 1]
        split_sizes = [len(train_df), len(val_df), len(test_df)]
        split_labels = [f'Train ({len(train_df):,})', f'Val ({len(val_df):,})', f'Test ({len(test_df):,})']
        ax2.pie(split_sizes, labels=split_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Split Distribution')
        
        # 3. Class balance heatmap
        ax3 = axes[1, 0]
        balance_data = viz_df[['train', 'val', 'test']].T
        balance_data.columns = self.classes
        
        # Normalize by row to show proportions
        balance_norm = balance_data.div(balance_data.sum(axis=1), axis=0)
        
        sns.heatmap(balance_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax3)
        ax3.set_title('Class Balance Across Splits')
        ax3.set_ylabel('Split')
        
        # 4. Class distribution comparison
        ax4 = axes[1, 1]
        
        # Calculate class proportions within each split
        train_props = train_df['class'].value_counts(normalize=True).sort_index()
        val_props = val_df['class'].value_counts(normalize=True).sort_index()
        test_props = test_df['class'].value_counts(normalize=True).sort_index()
        
        x_pos = np.arange(len(self.classes))
        width = 0.25
        
        ax4.bar(x_pos - width, train_props, width, label='Train', alpha=0.8)
        ax4.bar(x_pos, val_props, width, label='Val', alpha=0.8)
        ax4.bar(x_pos + width, test_props, width, label='Test', alpha=0.8)
        
        ax4.set_title('Class Distribution Within Each Split')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Proportion')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.classes, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_splits_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics
        summary_stats = {
            'total_samples': len(self.df),
            'num_classes': len(self.classes),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'class_stats': {}
        }
        
        for class_name in self.classes:
            class_total = (train_counts.get(class_name, 0) + 
                          val_counts.get(class_name, 0) + 
                          test_counts.get(class_name, 0))
            
            summary_stats['class_stats'][class_name] = {
                'total': class_total,
                'train': train_counts.get(class_name, 0),
                'val': val_counts.get(class_name, 0),
                'test': test_counts.get(class_name, 0)
            }
        
        # Save summary (convert numpy types to native Python types)
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary_stats = convert_numpy_types(summary_stats)
        
        with open(os.path.join(output_dir, 'split_summary.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print("✅ Split visualizations created:")
        print(f"   📊 {output_dir}/data_splits_analysis.png")
        print(f"   📄 {output_dir}/split_summary.json")


def create_optimal_splits(labels_file: str, output_dir: str, random_seed: int = 42,
                         train_size: float = 0.7, val_size: float = 0.15, 
                         test_size: float = 0.15) -> Dict[str, Any]:
    """
    Convenience function to create optimal data splits
    
    Args:
        labels_file: Path to labels CSV file
        output_dir: Directory to save results
        random_seed: Random seed for reproducibility
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        
    Returns:
        Dictionary with split information and quality metrics
    """
    
    print("🚀 Creating optimal data splits...")
    
    # Create splitter
    splitter = AdvancedDataSplitter(labels_file, random_seed)
    
    # Create splits
    train_df, val_df, test_df = splitter.stratified_split(train_size, val_size, test_size)
    
    # Analyze quality
    analysis = splitter.analyze_split_quality(train_df, val_df, test_df)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    splitter.save_splits(train_df, val_df, test_df, output_dir)
    splitter.visualize_splits(train_df, val_df, test_df, output_dir)
    
    # Save analysis
    with open(os.path.join(output_dir, 'split_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    result = {
        'splitter': splitter,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'analysis': analysis
    }
    
    print("✅ Optimal splits created successfully!")
    return result


if __name__ == "__main__":
    # Example usage
    labels_file = "data/ArSL_Data_Labels.csv"
    output_dir = "data/splits"
    
    if os.path.exists(labels_file):
        result = create_optimal_splits(labels_file, output_dir)
        print(f"✅ Created splits with balance score: {result['analysis']['overall_balance_score']:.3f}")
    else:
        print(f"❌ Labels file not found: {labels_file}")