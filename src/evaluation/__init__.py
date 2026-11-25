"""Évaluation, calibration et robustesse (branche ``feature/evaluation-robustness``).

Entrées attendues (contrat technique) : probabilités ``[N, C]`` ou passes MC
``[T, N, C]`` et étiquettes ``[N]`` ; tableaux NumPy ou tenseurs PyTorch.
"""
from .calibration import TemperatureScaler, plot_reliability_diagram
from .conformal import SplitConformalClassifier
from .cost import benchmark_inference, count_parameters, timer
from .metrics import (accuracy, brier_score, classification_metrics, evaluate_mc,
                      expected_calibration_error, maximum_calibration_error, nll,
                      reliability_bins, uncertainty_decomposition)
from .ood import auroc, aupr, fpr_at_tpr, ood_detection_metrics, uncertainty_scores
from .report import comparison_table, make_record, save_csv, save_json
from .shift import PERTURBATIONS, evaluate_under_shift, perturb, summarize_shift

__all__ = [
    "accuracy", "nll", "brier_score", "expected_calibration_error",
    "maximum_calibration_error", "reliability_bins", "classification_metrics",
    "uncertainty_decomposition", "evaluate_mc",
    "TemperatureScaler", "plot_reliability_diagram",
    "auroc", "aupr", "fpr_at_tpr", "uncertainty_scores", "ood_detection_metrics",
    "PERTURBATIONS", "perturb", "evaluate_under_shift", "summarize_shift",
    "SplitConformalClassifier",
    "timer", "benchmark_inference", "count_parameters",
    "make_record", "save_json", "save_csv", "comparison_table",
]
