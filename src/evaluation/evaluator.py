"""
Evaluator Module for Arabic Sign Language Recognition
Provides comprehensive model evaluation and comparison tools
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
from collections import defaultdict

# Import evaluation components
from .metrics import (
    compute_classification_metrics, compute_top_k_accuracy,
    ConfusionMatrix, ClassificationReport, compute_model_efficiency_metrics,
    compute_calibration_metrics
)
from .visualizer import EvaluationVisualizer, create_evaluation_dashboard

# Import data and model components
from ..data.dataset import SignLanguageDataset
from ..models.base_model import BaseSignLanguageModel


class ModelEvaluator:
    """
    Comprehensive model evaluation class for sign language recognition
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 class_names: Optional[List[str]] = None):
        self.device = device
        self.class_names = class_names
        self.visualizer = EvaluationVisualizer()
        self.evaluation_results = {}
        
    def evaluate_model(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                      criterion: Optional[nn.Module] = None,
                      compute_efficiency: bool = True,
                      compute_calibration: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: PyTorch model to evaluate
            dataloader: Test dataloader
            criterion: Loss function (optional)
            compute_efficiency: Whether to compute efficiency metrics
            compute_calibration: Whether to compute calibration metrics
            
        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        # Inference timing
        inference_times = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / inputs.size(0))  # Per sample
                
                # Compute loss if criterion provided
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                num_batches += 1
                
                # Progress indicator
                if (batch_idx + 1) % 50 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Compute classification metrics
        classification_metrics = compute_classification_metrics(
            y_true, y_pred, y_proba, self.class_names
        )
        
        # Compute top-k accuracy
        top_5_acc = compute_top_k_accuracy(y_true, y_proba, k=5)
        
        # Compute efficiency metrics
        efficiency_metrics = {}
        if compute_efficiency:
            try:
                efficiency_metrics = compute_model_efficiency_metrics(
                    model, input_size=(3, 224, 224), device=self.device
                )
                efficiency_metrics['avg_inference_time_per_sample'] = np.mean(inference_times)
                efficiency_metrics['inference_time_std'] = np.std(inference_times)
            except Exception as e:
                warnings.warn(f"Could not compute efficiency metrics: {e}")
        
        # Compute calibration metrics
        calibration_metrics = {}
        if compute_calibration:
            try:
                calibration_metrics = compute_calibration_metrics(y_true, y_proba)
            except Exception as e:
                warnings.warn(f"Could not compute calibration metrics: {e}")
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'classification_metrics': classification_metrics,
            'top_5_accuracy': top_5_acc,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist()
            }
        }
        
        if criterion:
            results['avg_loss'] = total_loss / num_batches
        
        if efficiency_metrics:
            results['efficiency_metrics'] = efficiency_metrics
        
        if calibration_metrics:
            results['calibration_metrics'] = calibration_metrics
        
        return results
    
    def compare_models(self, models: Dict[str, nn.Module], 
                      dataloader: torch.utils.data.DataLoader,
                      criterion: Optional[nn.Module] = None,
                      save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name -> model
            dataloader: Test dataloader
            criterion: Loss function (optional)
            save_dir: Directory to save results
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            results = self.evaluate_model(model, dataloader, criterion)
            comparison_results[model_name] = results
            
            # Store for later use
            self.evaluation_results[model_name] = results
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        # Save results if directory provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save individual results
            for model_name, results in comparison_results.items():
                with open(save_path / f"{model_name}_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Save comparison summary
            with open(save_path / "comparison_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        
        return {
            'individual_results': comparison_results,
            'comparison_summary': summary
        }
    
    def _create_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary comparison of model results"""
        summary = {
            'model_rankings': {},
            'best_models': {},
            'metric_comparison': {}
        }
        
        # Extract key metrics for comparison
        key_metrics = [
            'overall_metrics.accuracy',
            'overall_metrics.f1_macro',
            'overall_metrics.f1_weighted'
        ]
        
        if 'top_5_accuracy' in list(results.values())[0]:
            key_metrics.append('top_5_accuracy')
        
        # Add efficiency metrics if available
        efficiency_metrics = ['efficiency_metrics.total_parameters', 
                            'efficiency_metrics.model_size_mb',
                            'efficiency_metrics.avg_inference_time_ms']
        
        for model_name, result in results.items():
            if 'efficiency_metrics' in result:
                key_metrics.extend(efficiency_metrics)
                break
        
        # Compare models on each metric
        for metric in key_metrics:
            metric_values = {}
            
            for model_name, result in results.items():
                try:
                    # Navigate nested dictionary
                    value = result
                    for key in metric.split('.'):
                        value = value[key]
                    metric_values[model_name] = value
                except (KeyError, TypeError):
                    continue
            
            if metric_values:
                # Sort models by metric (higher is better for most metrics)
                reverse_sort = not any(x in metric.lower() for x in ['loss', 'error', 'time', 'size', 'parameters'])
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=reverse_sort)
                
                summary['metric_comparison'][metric] = {
                    'values': metric_values,
                    'ranking': [model for model, _ in sorted_models],
                    'best': sorted_models[0][0],
                    'best_value': sorted_models[0][1]
                }
                
                summary['best_models'][metric] = sorted_models[0][0]
        
        # Overall ranking (based on accuracy and F1)
        ranking_weights = {
            'overall_metrics.accuracy': 0.4,
            'overall_metrics.f1_macro': 0.3,
            'overall_metrics.f1_weighted': 0.3
        }
        
        overall_scores = {}
        for model_name in results.keys():
            score = 0
            total_weight = 0
            
            for metric, weight in ranking_weights.items():
                if metric in summary['metric_comparison']:
                    try:
                        value = summary['metric_comparison'][metric]['values'][model_name]
                        score += value * weight
                        total_weight += weight
                    except KeyError:
                        continue
            
            if total_weight > 0:
                overall_scores[model_name] = score / total_weight
        
        if overall_scores:
            sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            summary['model_rankings']['overall'] = [model for model, _ in sorted_overall]
            summary['best_models']['overall'] = sorted_overall[0][0]
        
        return summary
    
    def generate_evaluation_report(self, model_name: str, results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_name: Name of the model
            results: Evaluation results
            save_path: Path to save the report
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"EVALUATION REPORT: {model_name}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {results.get('timestamp', 'Unknown')}")
        report_lines.append("")
        
        # Overall Metrics
        if 'classification_metrics' in results:
            overall_metrics = results['classification_metrics']['overall_metrics']
            
            report_lines.append("OVERALL PERFORMANCE")
            report_lines.append("-" * 40)
            report_lines.append(f"Accuracy: {overall_metrics['accuracy']:.4f}")
            report_lines.append(f"Macro F1-Score: {overall_metrics['f1_macro']:.4f}")
            report_lines.append(f"Weighted F1-Score: {overall_metrics['f1_weighted']:.4f}")
            
            if 'top_5_accuracy' in results:
                report_lines.append(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
            
            if 'avg_loss' in results:
                report_lines.append(f"Average Loss: {results['avg_loss']:.4f}")
            
            report_lines.append("")
        
        # Efficiency Metrics
        if 'efficiency_metrics' in results:
            eff_metrics = results['efficiency_metrics']
            
            report_lines.append("EFFICIENCY METRICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Parameters: {eff_metrics['total_parameters']:,}")
            report_lines.append(f"Trainable Parameters: {eff_metrics['trainable_parameters']:,}")
            report_lines.append(f"Model Size: {eff_metrics['model_size_mb']:.2f} MB")
            report_lines.append(f"Inference Time: {eff_metrics['avg_inference_time_ms']:.2f} ms")
            report_lines.append(f"FPS: {eff_metrics['fps']:.1f}")
            report_lines.append("")
        
        # Calibration Metrics
        if 'calibration_metrics' in results:
            cal_metrics = results['calibration_metrics']
            
            report_lines.append("CALIBRATION METRICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Expected Calibration Error: {cal_metrics['expected_calibration_error']:.4f}")
            report_lines.append(f"Maximum Calibration Error: {cal_metrics['maximum_calibration_error']:.4f}")
            report_lines.append("")
        
        # Per-Class Performance
        if 'classification_metrics' in results:
            per_class = results['classification_metrics']['per_class_metrics']
            
            report_lines.append("PER-CLASS PERFORMANCE")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            report_lines.append("-" * 80)
            
            for class_name, metrics in per_class.items():
                report_lines.append(f"{class_name:<20} "
                                  f"{metrics['precision']:<10.4f} "
                                  f"{metrics['recall']:<10.4f} "
                                  f"{metrics['f1_score']:<10.4f} "
                                  f"{metrics['support']:<10}")
            
            report_lines.append("")
        
        # Most Confused Classes
        if 'classification_metrics' in results:
            confused_pairs = results['classification_metrics']['most_confused_pairs']
            
            if confused_pairs:
                report_lines.append("MOST CONFUSED CLASS PAIRS")
                report_lines.append("-" * 40)
                for i, (class1, class2, count) in enumerate(confused_pairs[:5]):
                    report_lines.append(f"{i+1}. {class1} → {class2}: {count} samples")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def create_visualizations(self, model_name: str, results: Dict[str, Any],
                            save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create evaluation visualizations
        
        Args:
            model_name: Name of the model
            results: Evaluation results
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary of matplotlib figures
        """
        if 'predictions' not in results:
            raise ValueError("Results must contain predictions data")
        
        y_true = np.array(results['predictions']['y_true'])
        y_pred = np.array(results['predictions']['y_pred'])
        y_proba = np.array(results['predictions']['y_proba'])
        
        # Create visualization directory
        if save_dir:
            viz_dir = Path(save_dir) / f"{model_name}_visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
        else:
            viz_dir = None
        
        # Generate comprehensive dashboard
        figures = create_evaluation_dashboard(
            y_true, y_pred, y_proba, self.class_names, str(viz_dir) if viz_dir else None
        )
        
        return figures


class EnsembleEvaluator:
    """
    Specialized evaluator for ensemble models
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 class_names: Optional[List[str]] = None):
        self.device = device
        self.class_names = class_names
        self.base_evaluator = ModelEvaluator(device, class_names)
    
    def evaluate_ensemble_components(self, ensemble_model, dataloader: torch.utils.data.DataLoader,
                                   criterion: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Evaluate individual components of an ensemble model
        
        Args:
            ensemble_model: Ensemble model with individual models
            dataloader: Test dataloader
            criterion: Loss function
            
        Returns:
            Dictionary with ensemble evaluation results
        """
        results = {
            'ensemble_result': {},
            'individual_results': {},
            'component_comparison': {}
        }
        
        # Evaluate the ensemble as a whole
        print("Evaluating ensemble model...")
        results['ensemble_result'] = self.base_evaluator.evaluate_model(
            ensemble_model, dataloader, criterion
        )
        
        # Evaluate individual components if accessible
        if hasattr(ensemble_model, 'models') or hasattr(ensemble_model, 'base_models'):
            models = getattr(ensemble_model, 'models', getattr(ensemble_model, 'base_models', []))
            
            for i, model in enumerate(models):
                model_name = f"component_{i}"
                print(f"Evaluating component {i+1}/{len(models)}...")
                
                results['individual_results'][model_name] = self.base_evaluator.evaluate_model(
                    model, dataloader, criterion
                )
        
        # Create component comparison
        if results['individual_results']:
            results['component_comparison'] = self._compare_components(results)
        
        return results
    
    def _compare_components(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare ensemble components performance"""
        comparison = {
            'accuracy_comparison': {},
            'diversity_metrics': {},
            'ensemble_improvement': {}
        }
        
        # Compare accuracies
        ensemble_acc = results['ensemble_result']['classification_metrics']['overall_metrics']['accuracy']
        comparison['accuracy_comparison']['ensemble'] = ensemble_acc
        
        component_accs = []
        for comp_name, comp_result in results['individual_results'].items():
            comp_acc = comp_result['classification_metrics']['overall_metrics']['accuracy']
            comparison['accuracy_comparison'][comp_name] = comp_acc
            component_accs.append(comp_acc)
        
        # Calculate ensemble improvement
        if component_accs:
            avg_component_acc = np.mean(component_accs)
            best_component_acc = max(component_accs)
            
            comparison['ensemble_improvement'] = {
                'vs_average_component': ensemble_acc - avg_component_acc,
                'vs_best_component': ensemble_acc - best_component_acc,
                'improvement_percentage': ((ensemble_acc - best_component_acc) / best_component_acc) * 100
            }
        
        return comparison


def evaluate_single_model(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                         class_names: Optional[List[str]] = None,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                         save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for evaluating a single model
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        class_names: Class names
        device: Device to use
        save_dir: Directory to save results
        
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator(device=device, class_names=class_names)
    results = evaluator.evaluate_model(model, dataloader)
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(save_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate report
        report = evaluator.generate_evaluation_report("model", results)
        with open(save_path / "evaluation_report.txt", 'w') as f:
            f.write(report)
        
        # Create visualizations
        evaluator.create_visualizations("model", results, str(save_path))
    
    return results