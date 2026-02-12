"""Validation module for dzetsaka.

This module provides advanced validation techniques including nested
cross-validation and enhanced metrics.

Public API:
-----------
- NestedCrossValidator: Nested CV for unbiased evaluation
- ValidationMetrics: Enhanced per-class metrics
- perform_nested_cv: Convenience function for nested CV
- create_classification_summary: Text summary of results

Example:
-------
    >>> from scripts.validation import NestedCrossValidator, ValidationMetrics
    >>>
    >>> # Nested CV evaluation
    >>> validator = NestedCrossValidator(inner_cv=3, outer_cv=5)
    >>> results = validator.evaluate(X, y, 'RF', param_grid)
    >>> print(f"Mean accuracy: {results['mean_score']:.3f}")
    >>>
    >>> # Enhanced metrics
    >>> report = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
    >>> print(report['overall']['accuracy'])

"""

# Import validation components
try:
    from .nested_cv import NestedCrossValidator, perform_nested_cv

    NESTED_CV_AVAILABLE = True
except ImportError:
    NESTED_CV_AVAILABLE = False
    NestedCrossValidator = None
    perform_nested_cv = None

try:
    from .metrics import (
        ValidationMetrics,
        compute_multiclass_roc_auc,
        create_classification_summary,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    ValidationMetrics = None
    create_classification_summary = None


__all__ = [
    "METRICS_AVAILABLE",
    "NESTED_CV_AVAILABLE",
    # Nested CV
    "NestedCrossValidator",
    # Metrics
    "ValidationMetrics",
    "compute_multiclass_roc_auc",
    "create_classification_summary",
    "perform_nested_cv",
]
