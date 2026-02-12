"""Sampling techniques module for dzetsaka.

This module provides various sampling techniques for handling imbalanced
datasets and improving model training.

Public API:
-----------
- SMOTESampler: SMOTE-based oversampling
- compute_class_weights: Automatic class weight computation
- apply_smote_if_needed: Convenience function for conditional SMOTE
- check_imblearn_available: Check imbalanced-learn availability

Example:
-------
    >>> from scripts.sampling import SMOTESampler, compute_class_weights
    >>>
    >>> # Apply SMOTE
    >>> sampler = SMOTESampler(k_neighbors=5)
    >>> X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    >>>
    >>> # Or use class weights
    >>> weights = compute_class_weights(y_train, strategy="balanced")
    >>> rf = RandomForestClassifier(class_weight=weights)

"""

# Try to import sampling components
try:
    from .smote_sampler import (
        IMBLEARN_AVAILABLE,
        SMOTESampler,
        apply_smote_if_needed,
        check_imblearn_available,
    )

    SMOTE_AVAILABLE = IMBLEARN_AVAILABLE
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTESampler = None
    apply_smote_if_needed = None

    def check_imblearn_available():
        """Fallback function when imbalanced-learn is not available."""
        return False, None


# Import class weights (doesn't require imbalanced-learn)
try:
    from .class_weights import (
        apply_class_weights_to_model,
        compute_class_weights,
        compute_sample_weights,
        get_class_distribution,
        get_imbalance_ratio,
        normalize_weights,
        print_class_distribution,
        recommend_strategy,
    )

    CLASS_WEIGHTS_AVAILABLE = True
except ImportError:
    CLASS_WEIGHTS_AVAILABLE = False
    compute_class_weights = None
    apply_class_weights_to_model = None


# Define public API
__all__ = [
    "CLASS_WEIGHTS_AVAILABLE",
    "SMOTE_AVAILABLE",
    # SMOTE sampling
    "SMOTESampler",
    "apply_class_weights_to_model",
    "apply_smote_if_needed",
    "check_imblearn_available",
    # Class weights
    "compute_class_weights",
    "compute_sample_weights",
    "get_class_distribution",
    "get_imbalance_ratio",
    "normalize_weights",
    "print_class_distribution",
    "recommend_strategy",
]
