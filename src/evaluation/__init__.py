"""
Evaluation Module for Arabic Sign Language Recognition
Provides comprehensive evaluation tools for model assessment and analysis
"""

from .metrics import (
    compute_classification_metrics,
    ConfusionMatrix,
    compute_top_k_accuracy,
    ClassificationReport,
    compute_per_class_metrics,
    compute_macro_metrics,
    compute_weighted_metrics,
    compute_model_efficiency_metrics,
    compute_calibration_metrics
)

from .visualizer import (
    EvaluationVisualizer,
    plot_confusion_matrix,
    plot_training_curves,
    create_evaluation_dashboard
)

from .evaluator import (
    ModelEvaluator,
    EnsembleEvaluator,
    evaluate_single_model
)

__all__ = [
    # Metrics
    'compute_classification_metrics',
    'ConfusionMatrix',
    'compute_top_k_accuracy',
    'ClassificationReport',
    'compute_per_class_metrics',
    'compute_macro_metrics',
    'compute_weighted_metrics',
    'compute_model_efficiency_metrics',
    'compute_calibration_metrics',
    
    # Visualizer
    'EvaluationVisualizer',
    'plot_confusion_matrix',
    'plot_training_curves',
    'create_evaluation_dashboard',
    
    # Evaluator
    'ModelEvaluator',
    'EnsembleEvaluator',
    'evaluate_single_model'
]

__version__ = "1.0.0"