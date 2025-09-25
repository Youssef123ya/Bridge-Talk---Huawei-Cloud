"""
Visualizer Module for Arabic Sign Language Recognition Evaluation
Provides comprehensive visualization tools for model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
from pathlib import Path
import itertools


# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EvaluationVisualizer:
    """
    Comprehensive visualization class for model evaluation
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: Optional[str] = None,
                            title: str = 'Confusion Matrix',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix with enhanced visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            normalize: 'true', 'pred', 'all', or None
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: Optional[List[str]] = None,
                                 title: str = 'Classification Report',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot classification report as heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import classification_report
        
        # Get classification report as dict
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        # Extract metrics for each class
        metrics_data = []
        class_labels = []
        
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_data.append([metrics['precision'], metrics['recall'], metrics['f1-score']])
                class_labels.append(class_name)
        
        # Add average metrics
        metrics_data.append([report['macro avg']['precision'], 
                           report['macro avg']['recall'], 
                           report['macro avg']['f1-score']])
        class_labels.append('macro avg')
        
        metrics_data.append([report['weighted avg']['precision'], 
                           report['weighted avg']['recall'], 
                           report['weighted avg']['f1-score']])
        class_labels.append('weighted avg')
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data, 
                         index=class_labels,
                         columns=['Precision', 'Recall', 'F1-Score'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot heatmap
        sns.heatmap(df, annot=True, fmt='.3f', cmap='Blues', 
                   cbar_kws={'label': 'Score'}, ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('Classes')
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       title: str = 'ROC Curves',
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: Class names
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_classes = y_proba.shape[1]
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=self.colors[i % len(self.colors)],
                   label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    title: str = 'Precision-Recall Curves',
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curves for multi-class classification
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            class_names: Class names
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_classes = y_proba.shape[1]
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot PR curve for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color=self.colors[i % len(self.colors)],
                   label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]],
                            title: str = 'Training History',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history (loss and metrics over epochs)
        
        Args:
            history: Dictionary with training history
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Determine subplot layout
        metrics = list(history.keys())
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows // 2), dpi=self.dpi)
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, 'b-', linewidth=2, label=f'Training {metric}')
            
            # If validation metric exists, plot it too
            val_metric = f'val_{metric}'
            if val_metric in history:
                val_values = history[val_metric]
                ax.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation {metric}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch')
            ax.legend()
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_class_distribution(self, y: np.ndarray, class_names: Optional[List[str]] = None,
                              title: str = 'Class Distribution',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot class distribution as bar chart
        
        Args:
            y: Labels
            class_names: Class names
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Count classes
        unique, counts = np.unique(y, return_counts=True)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in unique]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create bar plot
        bars = ax.bar(class_names, counts, color=self.colors[:len(unique)])
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Samples')
        ax.set_title(title)
        
        # Rotate x-axis labels if too many classes
        if len(unique) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                            metrics: Optional[List[str]] = None,
                            title: str = 'Model Comparison',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple models across different metrics
        
        Args:
            results: Dictionary with model names as keys and metrics as values
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            # Use all available metrics
            all_metrics = set()
            for model_results in results.values():
                all_metrics.update(model_results.keys())
            metrics = list(all_metrics)
        
        # Prepare data
        models = list(results.keys())
        n_models = len(models)
        n_metrics = len(metrics)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Set width of bars
        bar_width = 0.8 / n_models
        
        # Set positions of bars on x-axis
        positions = np.arange(n_metrics)
        
        # Plot bars for each model
        for i, model in enumerate(models):
            model_values = [results[model].get(metric, 0) for metric in metrics]
            bars = ax.bar(positions + i * bar_width, model_values,
                         bar_width, label=model, color=self.colors[i % len(self.colors)])
            
            # Add value labels on bars
            for bar, value in zip(bars, model_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(positions + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_calibration_plot(self, y_true: np.ndarray, y_proba: np.ndarray,
                            n_bins: int = 10, title: str = 'Calibration Plot',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration plot (reliability diagram)
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n_bins: Number of bins
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get predicted classes and confidences
        y_pred = np.argmax(y_proba, axis=1)
        confidences = np.max(y_proba, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Compute calibration for each bin
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]), dpi=self.dpi)
        
        # Calibration plot
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
        
        # Add bin sizes as bar width
        bin_width = 1.0 / n_bins
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            if count > 0:
                ax1.bar(conf, acc, width=bin_width * 0.8, alpha=0.3, 
                       color=self.colors[0], edgecolor='black')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True)
        
        # Histogram of confidences
        ax2.hist(confidences, bins=n_bins, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Histogram')
        ax2.grid(True)
        
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: Optional[str] = None,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Standalone function to plot confusion matrix
    """
    visualizer = EvaluationVisualizer(figsize=figsize)
    return visualizer.plot_confusion_matrix(y_true, y_pred, class_names, normalize, title, save_path)


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        train_accs: Optional[List[float]] = None,
                        val_accs: Optional[List[float]] = None,
                        title: str = 'Training Curves',
                        figsize: Tuple[int, int] = (12, 5),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation curves
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy if provided
    if train_accs is not None and val_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy')
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_visible(False)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig


def create_evaluation_dashboard(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create comprehensive evaluation dashboard with all plots
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        class_names: Class names
        save_dir: Directory to save plots
        
    Returns:
        Dictionary of figures
    """
    visualizer = EvaluationVisualizer(figsize=(12, 8))
    figures = {}
    
    # Confusion matrix
    figures['confusion_matrix'] = visualizer.plot_confusion_matrix(
        y_true, y_pred, class_names, 
        save_path=Path(save_dir) / 'confusion_matrix.png' if save_dir else None
    )
    
    # Normalized confusion matrix
    figures['confusion_matrix_normalized'] = visualizer.plot_confusion_matrix(
        y_true, y_pred, class_names, normalize='true', title='Normalized Confusion Matrix',
        save_path=Path(save_dir) / 'confusion_matrix_normalized.png' if save_dir else None
    )
    
    # Classification report
    figures['classification_report'] = visualizer.plot_classification_report(
        y_true, y_pred, class_names,
        save_path=Path(save_dir) / 'classification_report.png' if save_dir else None
    )
    
    # ROC curves
    figures['roc_curves'] = visualizer.plot_roc_curves(
        y_true, y_proba, class_names,
        save_path=Path(save_dir) / 'roc_curves.png' if save_dir else None
    )
    
    # Precision-Recall curves
    figures['pr_curves'] = visualizer.plot_precision_recall_curves(
        y_true, y_proba, class_names,
        save_path=Path(save_dir) / 'pr_curves.png' if save_dir else None
    )
    
    # Class distribution
    figures['class_distribution'] = visualizer.plot_class_distribution(
        y_true, class_names,
        save_path=Path(save_dir) / 'class_distribution.png' if save_dir else None
    )
    
    # Calibration plot
    figures['calibration_plot'] = visualizer.plot_calibration_plot(
        y_true, y_proba,
        save_path=Path(save_dir) / 'calibration_plot.png' if save_dir else None
    )
    
    return figures