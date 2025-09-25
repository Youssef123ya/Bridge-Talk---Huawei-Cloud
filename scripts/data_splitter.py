import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import shutil
from typing import Tuple, Dict, List, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedDataSplitter:
    """Advanced data splitting with multiple strategies for Arabic Sign Language dataset"""

    def __init__(self, labels_file: str, random_seed: int = 42):
        self.labels_file = labels_file
        self.random_seed = random_seed
        self.df = pd.read_csv(labels_file)
        self.label_encoder = LabelEncoder()
        self.setup_encoder()

    def setup_encoder(self):
        """Setup label encoder for classes"""
        self.label_encoder.fit(self.df['Class'])
        self.df['class_encoded'] = self.label_encoder.transform(self.df['Class'])
        self.num_classes = len(self.label_encoder.classes_)

    def stratified_split(self, 
                        train_size: float = 0.7,
                        val_size: float = 0.15,
                        test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified split ensuring balanced class distribution"""

        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"

        print(f"🔄 Creating stratified split: {train_size:.1%}/{val_size:.1%}/{test_size:.1%}")

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            self.df,
            test_size=(val_size + test_size),
            random_state=self.random_seed,
            stratify=self.df['Class']
        )

        # Second split: val vs test
        val_test_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_test_ratio),
            random_state=self.random_seed,
            stratify=temp_df['Class']
        )

        return train_df, val_df, test_df

    def balanced_split_with_minimum_samples(self,
                                          min_train_samples: int = 100,
                                          min_val_samples: int = 20,
                                          min_test_samples: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create balanced splits ensuring minimum samples per class in each split"""

        print(f"🔄 Creating balanced split with minimum samples (train:{min_train_samples}, val:{min_val_samples}, test:{min_test_samples})")

        train_dfs = []
        val_dfs = []
        test_dfs = []

        for class_name in self.df['Class'].unique():
            class_df = self.df[self.df['Class'] == class_name].copy()
            class_size = len(class_df)

            # Check if class has enough samples
            required_samples = min_train_samples + min_val_samples + min_test_samples
            if class_size < required_samples:
                print(f"⚠️  Class '{class_name}' has only {class_size} samples (needs {required_samples})")

                # Adjust minimums proportionally
                ratio = class_size / required_samples
                adj_train = max(1, int(min_train_samples * ratio))
                adj_val = max(1, int(min_val_samples * ratio))
                adj_test = max(1, class_size - adj_train - adj_val)
            else:
                adj_train = min_train_samples
                adj_val = min_val_samples
                adj_test = min_test_samples

            # Shuffle class data
            class_df = class_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

            # Split the class data
            train_end = adj_train
            val_end = train_end + adj_val
            test_end = val_end + adj_test

            train_class = class_df[:train_end]
            val_class = class_df[train_end:val_end]
            test_class = class_df[val_end:test_end]

            # Add remaining samples to train
            if test_end < len(class_df):
                remaining = class_df[test_end:]
                train_class = pd.concat([train_class, remaining], ignore_index=True)

            train_dfs.append(train_class)
            val_dfs.append(val_class)
            test_dfs.append(test_class)

        # Combine all classes
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=self.random_seed)
        val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=self.random_seed)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=self.random_seed)

        return train_df, val_df, test_df

    def create_cross_validation_folds(self, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create stratified K-fold cross-validation splits"""

        print(f"🔄 Creating {n_folds}-fold cross-validation splits")

        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['Class'])):
            train_fold = self.df.iloc[train_idx].copy()
            val_fold = self.df.iloc[val_idx].copy()

            folds.append((train_fold, val_fold))
            print(f"   Fold {fold_idx + 1}: Train={len(train_fold)}, Val={len(val_fold)}")

        return folds

    def analyze_split_quality(self, 
                            train_df: pd.DataFrame,
                            val_df: pd.DataFrame, 
                            test_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the quality of data splits"""

        print("📊 Analyzing split quality...")

        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        analysis = {
            'split_sizes': {},
            'class_distributions': {},
            'balance_scores': {},
            'coverage': {}
        }

        for split_name, split_df in splits.items():
            # Basic statistics
            analysis['split_sizes'][split_name] = len(split_df)

            # Class distribution
            class_counts = split_df['Class'].value_counts()
            analysis['class_distributions'][split_name] = class_counts.to_dict()

            # Balance score (coefficient of variation)
            cv = class_counts.std() / class_counts.mean()
            analysis['balance_scores'][split_name] = cv

            # Class coverage
            total_classes = self.df['Class'].nunique()
            split_classes = split_df['Class'].nunique()
            coverage = split_classes / total_classes
            analysis['coverage'][split_name] = coverage

            print(f"   {split_name.upper()}: {len(split_df):,} samples, "
                  f"{split_classes}/{total_classes} classes ({coverage:.1%}), "
                  f"balance_score={cv:.3f}")

        return analysis

    def save_splits(self,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   output_dir: str = 'data/processed') -> None:
        """Save data splits to CSV files with metadata"""

        os.makedirs(output_dir, exist_ok=True)

        # Save CSV files
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        # Create class mapping
        class_mapping = {
            'class_to_idx': {cls: int(idx) for idx, cls in enumerate(self.label_encoder.classes_)},
            'idx_to_class': {int(idx): cls for idx, cls in enumerate(self.label_encoder.classes_)},
            'num_classes': self.num_classes,
            'classes': self.label_encoder.classes_.tolist()
        }

        with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
            json.dump(class_mapping, f, indent=2)

        # Save split metadata
        metadata = {
            'total_samples': len(self.df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'num_classes': self.num_classes,
            'random_seed': self.random_seed,
            'split_ratios': {
                'train': len(train_df) / len(self.df),
                'val': len(val_df) / len(self.df),
                'test': len(test_df) / len(self.df)
            }
        }

        with open(os.path.join(output_dir, 'split_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Splits saved to {output_dir}/")
        print(f"   📄 train.csv: {len(train_df):,} samples")
        print(f"   📄 val.csv: {len(val_df):,} samples") 
        print(f"   📄 test.csv: {len(test_df):,} samples")
        print(f"   📄 class_mapping.json: {self.num_classes} classes")
        print(f"   📄 split_metadata.json: Split information")

    def visualize_splits(self,
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        save_dir: str = 'data/analysis') -> None:
        """Create visualizations of the data splits"""

        os.makedirs(save_dir, exist_ok=True)

        # Prepare data for visualization
        splits_data = []
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            for _, row in split_df.iterrows():
                splits_data.append({
                    'Split': split_name,
                    'Class': row['Class'],
                    'Count': 1
                })

        viz_df = pd.DataFrame(splits_data)
        split_counts = viz_df.groupby(['Split', 'Class']).size().reset_index(name='Count')

        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Arabic Sign Language Dataset - Split Analysis', fontsize=16, fontweight='bold')

        # 1. Overall split distribution
        ax1 = axes[0, 0]
        split_sizes = [len(train_df), len(val_df), len(test_df)]
        split_labels = ['Train', 'Validation', 'Test']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        wedges, texts, autotexts = ax1.pie(split_sizes, labels=split_labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Split Distribution')

        # 2. Class distribution across splits
        ax2 = axes[0, 1]
        pivot_data = split_counts.pivot(index='Class', columns='Split', values='Count').fillna(0)
        pivot_data.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_title('Class Distribution Across Splits')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Samples')
        ax2.legend(title='Split')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Balance analysis
        ax3 = axes[1, 0]
        balance_data = []
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            class_counts = split_df['Class'].value_counts()
            cv = class_counts.std() / class_counts.mean()
            balance_data.append({'Split': split_name, 'Balance_Score': cv})

        balance_df = pd.DataFrame(balance_data)
        bars = ax3.bar(balance_df['Split'], balance_df['Balance_Score'], color=colors)
        ax3.set_title('Split Balance Scores (Lower = More Balanced)')
        ax3.set_ylabel('Coefficient of Variation')

        # Add value labels on bars
        for bar, score in zip(bars, balance_df['Balance_Score']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')

        # 4. Sample size comparison
        ax4 = axes[1, 1]
        class_sample_sizes = []
        for class_name in self.df['Class'].unique():
            class_data = {
                'Class': class_name,
                'Train': len(train_df[train_df['Class'] == class_name]),
                'Val': len(val_df[val_df['Class'] == class_name]),
                'Test': len(test_df[test_df['Class'] == class_name])
            }
            class_sample_sizes.append(class_data)

        size_df = pd.DataFrame(class_sample_sizes)
        x = np.arange(len(size_df))
        width = 0.25

        ax4.bar(x - width, size_df['Train'], width, label='Train', color=colors[0], alpha=0.8)
        ax4.bar(x, size_df['Val'], width, label='Val', color=colors[1], alpha=0.8)
        ax4.bar(x + width, size_df['Test'], width, label='Test', color=colors[2], alpha=0.8)

        ax4.set_title('Samples per Class by Split')
        ax4.set_xlabel('Class Index')
        ax4.set_ylabel('Number of Samples')
        ax4.set_xticks(x)
        ax4.set_xticklabels(range(len(size_df)))
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'data_splits_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Split visualization saved to {save_dir}/data_splits_analysis.png")

def create_optimal_splits(labels_file: str,
                         strategy: str = 'stratified',
                         **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Factory function to create optimal data splits"""

    splitter = AdvancedDataSplitter(labels_file, kwargs.get('random_seed', 42))

    if strategy == 'stratified':
        return splitter.stratified_split(
            kwargs.get('train_size', 0.7),
            kwargs.get('val_size', 0.15),
            kwargs.get('test_size', 0.15)
        )
    elif strategy == 'balanced':
        return splitter.balanced_split_with_minimum_samples(
            kwargs.get('min_train_samples', 100),
            kwargs.get('min_val_samples', 20),
            kwargs.get('min_test_samples', 20)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
