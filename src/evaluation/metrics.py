"""
Metrics Module for Arabic Sign Language Recognition Evaluation
Provides comprehensive metrics for classification model evaluation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings


class ConfusionMatrix:
    """
    Enhanced confusion matrix with additional analysis capabilities
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.class_names = class_names or [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        self.num_classes = len(self.class_names)
        
        # Compute confusion matrix
        self.matrix = confusion_matrix(self.y_true, self.y_pred)
        
        # Compute per-class metrics
        self._compute_per_class_metrics()
    
    def _compute_per_class_metrics(self):
        """Compute per-class precision, recall, f1-score"""
        self.precision = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        self.recall = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        self.f1 = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        self.support = np.bincount(self.y_true, minlength=self.num_classes)
    
    def get_normalized_matrix(self, normalize: str = 'true') -> np.ndarray:
        """
        Get normalized confusion matrix
        
        Args:
            normalize: 'true', 'pred', 'all', or None
            
        Returns:
            Normalized confusion matrix
        """
        if normalize == 'true':
            return self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            return self.matrix.astype('float') / self.matrix.sum(axis=0)
        elif normalize == 'all':
            return self.matrix.astype('float') / self.matrix.sum()
        else:
            return self.matrix
    
    def get_class_accuracy(self) -> np.ndarray:
        """Get per-class accuracy (diagonal values of normalized matrix)"""
        normalized = self.get_normalized_matrix('true')
        return np.diag(normalized)
    
    def get_most_confused_pairs(self, top_k: int = 5) -> List[Tuple[str, str, int]]:
        """
        Get most confused class pairs (excluding diagonal)
        
        Args:
            top_k: Number of top confused pairs to return
            
        Returns:
            List of (class1, class2, count) tuples
        """
        confused_pairs = []
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and self.matrix[i, j] > 0:
                    confused_pairs.append((
                        self.class_names[i], 
                        self.class_names[j], 
                        self.matrix[i, j]
                    ))
        
        # Sort by confusion count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confused_pairs[:top_k]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        total_samples = np.sum(self.matrix)
        correct_predictions = np.trace(self.matrix)
        
        return {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'overall_accuracy': correct_predictions / total_samples,
            'per_class_accuracy': self.get_class_accuracy(),
            'per_class_precision': self.precision,
            'per_class_recall': self.recall,
            'per_class_f1': self.f1,
            'per_class_support': self.support,
            'macro_precision': np.mean(self.precision),
            'macro_recall': np.mean(self.recall),
            'macro_f1': np.mean(self.f1),
            'weighted_precision': np.average(self.precision, weights=self.support),
            'weighted_recall': np.average(self.recall, weights=self.support),
            'weighted_f1': np.average(self.f1, weights=self.support)
        }


class ClassificationReport:
    """
    Enhanced classification report with additional metrics
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None, class_names: Optional[List[str]] = None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = y_proba
        self.class_names = class_names or [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        self.num_classes = len(self.class_names)
        
        # Compute metrics
        self.confusion_matrix = ConfusionMatrix(y_true, y_pred, class_names)
        self._compute_additional_metrics()
    
    def _compute_additional_metrics(self):
        """Compute additional metrics like AUC if probabilities are available"""
        self.metrics = {}
        
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision_macro'] = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        self.metrics['recall_macro'] = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        self.metrics['f1_macro'] = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        self.metrics['precision_weighted'] = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.metrics['recall_weighted'] = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        self.metrics['f1_weighted'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        # AUC metrics if probabilities are available
        if self.y_proba is not None:
            try:
                if self.num_classes == 2:
                    self.metrics['auc'] = roc_auc_score(self.y_true, self.y_proba[:, 1])
                    self.metrics['average_precision'] = average_precision_score(self.y_true, self.y_proba[:, 1])
                else:
                    # Multi-class AUC
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(self.y_true, classes=range(self.num_classes))
                    self.metrics['auc_macro'] = roc_auc_score(y_true_bin, self.y_proba, average='macro', multi_class='ovr')
                    self.metrics['auc_weighted'] = roc_auc_score(y_true_bin, self.y_proba, average='weighted', multi_class='ovr')
            except Exception as e:
                warnings.warn(f"Could not compute AUC metrics: {e}")
    
    def get_report_dict(self) -> Dict[str, Any]:
        """Get comprehensive report as dictionary"""
        report = {
            'overall_metrics': self.metrics,
            'per_class_metrics': {},
            'confusion_matrix_summary': self.confusion_matrix.get_summary(),
            'most_confused_pairs': self.confusion_matrix.get_most_confused_pairs()
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            report['per_class_metrics'][class_name] = {
                'precision': self.confusion_matrix.precision[i],
                'recall': self.confusion_matrix.recall[i],
                'f1_score': self.confusion_matrix.f1[i],
                'support': self.confusion_matrix.support[i],
                'accuracy': self.confusion_matrix.get_class_accuracy()[i]
            }
        
        return report
    
    def print_report(self, digits: int = 4):
        """Print formatted classification report"""
        print("Classification Report")
        print("=" * 50)
        
        # Overall metrics
        print(f"Overall Accuracy: {self.metrics['accuracy']:.{digits}f}")
        print(f"Macro F1-Score: {self.metrics['f1_macro']:.{digits}f}")
        print(f"Weighted F1-Score: {self.metrics['f1_weighted']:.{digits}f}")
        
        if 'auc_macro' in self.metrics:
            print(f"Macro AUC: {self.metrics['auc_macro']:.{digits}f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 80)
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15} "
                  f"{self.confusion_matrix.precision[i]:<10.{digits}f} "
                  f"{self.confusion_matrix.recall[i]:<10.{digits}f} "
                  f"{self.confusion_matrix.f1[i]:<10.{digits}f} "
                  f"{self.confusion_matrix.support[i]:<10}")
        
        print("-" * 80)
        print(f"{'Macro Avg':<15} "
              f"{self.metrics['precision_macro']:<10.{digits}f} "
              f"{self.metrics['recall_macro']:<10.{digits}f} "
              f"{self.metrics['f1_macro']:<10.{digits}f} "
              f"{np.sum(self.confusion_matrix.support):<10}")
        
        print(f"{'Weighted Avg':<15} "
              f"{self.metrics['precision_weighted']:<10.{digits}f} "
              f"{self.metrics['recall_weighted']:<10.{digits}f} "
              f"{self.metrics['f1_weighted']:<10.{digits}f} "
              f"{np.sum(self.confusion_matrix.support):<10}")
        
        # Most confused pairs
        print(f"\nMost Confused Class Pairs:")
        print("-" * 40)
        confused_pairs = self.confusion_matrix.get_most_confused_pairs()
        for i, (class1, class2, count) in enumerate(confused_pairs):
            print(f"{i+1}. {class1} → {class2}: {count} samples")


def compute_classification_metrics(y_true: Union[torch.Tensor, np.ndarray], 
                                 y_pred: Union[torch.Tensor, np.ndarray],
                                 y_proba: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                 class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        class_names: Class names (optional)
        
    Returns:
        Dictionary of computed metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_proba is not None and isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()
    
    # Create classification report
    report = ClassificationReport(y_true, y_pred, y_proba, class_names)
    
    return report.get_report_dict()


def compute_top_k_accuracy(y_true: Union[torch.Tensor, np.ndarray],
                          y_proba: Union[torch.Tensor, np.ndarray],
                          k: int = 5) -> float:
    """
    Compute top-k accuracy
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        k: k value for top-k accuracy
        
    Returns:
        Top-k accuracy score
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()
    
    # Get top-k predictions
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    
    # Check if true label is in top-k predictions
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute detailed per-class metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        
    Returns:
        Dictionary of per-class metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=len(class_names))
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'accuracy': per_class_acc[i],
            'support': support[i]
        }
    
    return per_class_metrics


def compute_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute macro-averaged metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of macro metrics
    """
    return {
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }


def compute_weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute weighted-averaged metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of weighted metrics
    """
    return {
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def compute_model_efficiency_metrics(model: torch.nn.Module, 
                                   input_size: Tuple[int, ...] = (3, 224, 224),
                                   device: str = 'cpu') -> Dict[str, Any]:
    """
    Compute model efficiency metrics (parameters, FLOPs, inference time)
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run inference on
        
    Returns:
        Dictionary of efficiency metrics
    """
    import time
    
    model.eval()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Inference time
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 100
    fps = 1 / avg_inference_time
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'fps': fps
    }


def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, 
                              n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute model calibration metrics
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
    """
    # Get predicted classes and confidences
    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Compute calibration metrics for each bin
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
    
    # Expected Calibration Error (ECE)
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(y_true)
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(bin_accuracies - bin_confidences))
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }