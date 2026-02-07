"""Class weight computation for handling imbalanced datasets.

This module provides utilities for computing class weights to handle
imbalanced classification problems through cost-sensitive learning.

Key Features:
-------------
- Automatic balanced weight computation
- Custom weight strategies
- Integration with all dzetsaka algorithms
- Support for RF, SVM, XGBoost, LightGBM class weights

Example Usage:
--------------
    >>> from scripts.sampling.class_weights import compute_class_weights
    >>>
    >>> # Automatic balanced weights
    >>> weights = compute_class_weights(y_train, strategy='balanced')
    >>> print(weights)  # {0: 0.5, 1: 4.5}  (class 1 is minority)
    >>>
    >>> # Use with Random Forest
    >>> rf = RandomForestClassifier(class_weight=weights)
    >>> rf.fit(X_train, y_train)

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

License:
--------
GNU General Public License v2.0 or later

"""
from typing import Dict, Optional, Union

import numpy as np

# Try to import sklearn for weight computation
try:
    from sklearn.utils.class_weight import compute_class_weight

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _normalize_labels_1d(y: np.ndarray) -> np.ndarray:
    """Normalize labels to a 1D array of scalar values.

    Handles common upstream shapes like (n, 1) and object arrays containing
    0-d/1-element numpy arrays, which otherwise break class weight utilities.
    """
    arr = np.asarray(y)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.dtype == object:
        out = np.empty(arr.shape[0], dtype=object)
        for i, val in enumerate(arr):
            if isinstance(val, np.ndarray):
                if val.size != 1:
                    raise ValueError("Labels must be scalar values per sample")
                out[i] = val.item()
            else:
                out[i] = val
        return out

    return arr


def compute_class_weights(
    y: np.ndarray,
    strategy: str = "balanced",
    custom_weights: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Compute class weights for imbalanced datasets.

    Calculates weights for each class to penalize misclassifications
    of minority classes more heavily during training.

    Parameters
    ----------
    y : np.ndarray
        Training labels, shape (n_samples,)
    strategy : str, default='balanced'
        Weight computation strategy:
        - 'balanced': n_samples / (n_classes * np.bincount(y))
          Inversely proportional to class frequencies
        - 'uniform': All classes weighted equally (weight=1.0)
        - 'custom': Use custom_weights parameter
    custom_weights : dict, optional
        Custom weights {class_label: weight}
        Only used if strategy='custom'

    Returns
    -------
    weights : dict
        Class weights {class_label: weight}
        Minority classes get higher weights

    Raises
    ------
    ValueError
        If strategy is invalid or custom_weights missing

    Example
    -------
    >>> # Dataset: 900 samples class 0, 100 samples class 1
    >>> y = np.array([0]*900 + [1]*100)
    >>> weights = compute_class_weights(y, strategy='balanced')
    >>> print(weights)
    {0: 0.556, 1: 5.0}  # Class 1 weighted 9x higher

    """
    if strategy == "custom":
        if custom_weights is None:
            raise ValueError("custom_weights must be provided when strategy='custom'")
        return custom_weights

    y = _normalize_labels_1d(y)
    unique_classes = np.unique(y)

    if strategy == "uniform":
        # All classes weighted equally
        return {int(cls): 1.0 for cls in unique_classes}

    elif strategy == "balanced":
        if SKLEARN_AVAILABLE:
            # Use sklearn's compute_class_weight
            weights_array = compute_class_weight(
                class_weight="balanced",
                classes=unique_classes,
                y=y,
            )
            return {int(cls): float(weight) for cls, weight in zip(unique_classes, weights_array)}
        else:
            # Manual balanced weight computation
            # weight = n_samples / (n_classes * class_count)
            n_samples = len(y)
            n_classes = len(unique_classes)

            weights = {}
            for cls in unique_classes:
                class_count = np.sum(y == cls)
                weight = n_samples / (n_classes * class_count)
                weights[int(cls)] = float(weight)

            return weights

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'balanced', 'uniform', or 'custom'")


def apply_class_weights_to_model(
    model_code: str,
    class_weights: Dict[int, float],
) -> Dict[str, any]:
    """Convert class weights to model-specific parameter format.

    Different algorithms use different formats for class weights.
    This function converts the unified weight dict to the appropriate format.

    Parameters
    ----------
    model_code : str
        dzetsaka classifier code (e.g., 'RF', 'SVM', 'XGB', 'LGB')
    class_weights : dict
        Class weights {class_label: weight}

    Returns
    -------
    params : dict
        Model-specific parameters to pass to classifier
        - RF/SVM/KNN/ET/GBC/LR/NB/MLP: {'class_weight': weights_dict}
        - XGB: {'scale_pos_weight': ratio}  (binary only)
        - LGB: {'class_weight': weights_dict}

    Example
    -------
    >>> weights = {0: 0.5, 1: 4.5}
    >>> params = apply_class_weights_to_model('RF', weights)
    >>> rf = RandomForestClassifier(**params)

    """
    # Scikit-learn models (accept dict directly)
    if model_code in ["RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"]:
        return {"class_weight": class_weights}

    # XGBoost (binary only - uses scale_pos_weight)
    elif model_code == "XGB":
        # For binary classification, compute scale_pos_weight
        if len(class_weights) == 2:
            classes = sorted(class_weights.keys())
            # scale_pos_weight = weight_class_1 / weight_class_0
            # or equivalently: count_class_0 / count_class_1
            weight_ratio = class_weights[classes[1]] / class_weights[classes[0]]
            return {"scale_pos_weight": weight_ratio}
        else:
            # For multiclass, XGBoost doesn't support class weights directly
            # Would need to use sample_weight in fit() instead
            return {}

    # LightGBM (accepts dict with 'balanced' string or dict)
    elif model_code == "LGB":
        return {"class_weight": class_weights}

    # GMM doesn't support class weights
    elif model_code == "GMM":
        return {}

    else:
        # Unknown model - return empty dict
        return {}


def get_imbalance_ratio(y: np.ndarray) -> float:
    """Calculate imbalance ratio (majority class / minority class).

    Parameters
    ----------
    y : np.ndarray
        Training labels

    Returns
    -------
    ratio : float
        Imbalance ratio
        1.0 = perfectly balanced
        >1.0 = imbalanced (higher = more imbalanced)

    Example
    -------
    >>> y = np.array([0]*900 + [1]*100)
    >>> ratio = get_imbalance_ratio(y)
    >>> print(ratio)  # 9.0 (class 0 is 9x larger)

    """
    y = _normalize_labels_1d(y)
    unique, counts = np.unique(y, return_counts=True)

    if len(unique) < 2:
        return 1.0

    majority_count = counts.max()
    minority_count = counts.min()

    return majority_count / minority_count


def get_class_distribution(y: np.ndarray) -> Dict[int, int]:
    """Get class distribution from labels.

    Parameters
    ----------
    y : np.ndarray
        Labels array

    Returns
    -------
    distribution : dict
        {class_label: count} dictionary

    Example
    -------
    >>> dist = get_class_distribution(y_train)
    >>> for cls, count in sorted(dist.items()):
    ...     print(f"Class {cls}: {count} samples")

    """
    y = _normalize_labels_1d(y)
    unique, counts = np.unique(y, return_counts=True)
    return {int(cls): int(count) for cls, count in zip(unique, counts)}


def recommend_strategy(
    y: np.ndarray,
    threshold: float = 2.0,
) -> str:
    """Recommend imbalance handling strategy based on imbalance ratio.

    Parameters
    ----------
    y : np.ndarray
        Training labels
    threshold : float, default=2.0
        Imbalance ratio threshold for recommending class weights

    Returns
    -------
    strategy : str
        Recommended strategy:
        - 'none': Dataset is balanced, no action needed
        - 'class_weights': Moderate imbalance, use class weights
        - 'smote': Severe imbalance, consider SMOTE + class weights

    Example
    -------
    >>> strategy = recommend_strategy(y_train)
    >>> if strategy == 'class_weights':
    ...     weights = compute_class_weights(y_train)
    >>> elif strategy == 'smote':
    ...     # Apply SMOTE then class weights
    ...     pass

    """
    ratio = get_imbalance_ratio(y)

    if ratio < threshold:
        return "none"
    elif ratio < 5.0:
        return "class_weights"
    else:
        return "smote"  # Severe imbalance


def print_class_distribution(
    y: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Print formatted class distribution.

    Parameters
    ----------
    y : np.ndarray
        Training labels
    class_names : dict, optional
        {class_id: class_name} mapping for readable output

    Example
    -------
    >>> print_class_distribution(y_train, {0: 'Water', 1: 'Forest'})
    Class Distribution:
      Water (0): 900 samples (90.0%)
      Forest (1): 100 samples (10.0%)
    Imbalance Ratio: 9.0

    """
    dist = get_class_distribution(y)
    total = len(y)

    print("Class Distribution:")
    for cls in sorted(dist.keys()):
        count = dist[cls]
        percentage = (count / total) * 100
        class_label = class_names.get(cls, str(cls)) if class_names else str(cls)
        print(f"  {class_label} ({cls}): {count} samples ({percentage:.1f}%)")

    ratio = get_imbalance_ratio(y)
    print(f"Imbalance Ratio: {ratio:.2f}")

    strategy = recommend_strategy(y)
    if strategy != "none":
        print(f"Recommended: Use {strategy}")


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """Normalize weights to sum to number of classes.

    This ensures weights are scaled consistently regardless of strategy.

    Parameters
    ----------
    weights : dict
        Class weights {class_label: weight}

    Returns
    -------
    normalized_weights : dict
        Normalized weights

    Example
    -------
    >>> weights = {0: 1.0, 1: 9.0}
    >>> normalized = normalize_weights(weights)
    >>> print(normalized)  # {0: 0.2, 1: 1.8}

    """
    total = sum(weights.values())
    n_classes = len(weights)

    if total == 0:
        # Avoid division by zero
        return {cls: 1.0 for cls in weights.keys()}

    scale_factor = n_classes / total
    return {cls: weight * scale_factor for cls, weight in weights.items()}


def compute_sample_weights(
    y: np.ndarray,
    class_weights: Dict[int, float],
) -> np.ndarray:
    """Compute per-sample weights from class weights.

    Some algorithms (like XGBoost multiclass) require sample weights
    instead of class weights.

    Parameters
    ----------
    y : np.ndarray
        Training labels
    class_weights : dict
        Class weights {class_label: weight}

    Returns
    -------
    sample_weights : np.ndarray
        Per-sample weights, shape (n_samples,)

    Example
    -------
    >>> class_weights = {0: 0.5, 1: 4.5}
    >>> sample_weights = compute_sample_weights(y_train, class_weights)
    >>> model.fit(X_train, y_train, sample_weight=sample_weights)

    """
    y = _normalize_labels_1d(y)
    sample_weights = np.zeros(len(y), dtype=np.float32)

    for cls, weight in class_weights.items():
        mask = y == cls
        sample_weights[mask] = weight

    return sample_weights
