"""SMOTE-based oversampling for imbalanced datasets.

This module provides Synthetic Minority Over-sampling Technique (SMOTE)
implementation for handling class imbalance in classification tasks.

Key Features:
-------------
- KNN-based synthetic sample generation
- Multi-class support
- Configurable sampling strategies
- Memory-efficient implementation
- Integration with dzetsaka training pipeline

Example Usage:
--------------
    >>> from scripts.sampling.smote_sampler import SMOTESampler
    >>>
    >>> sampler = SMOTESampler(k_neighbors=5, random_state=42)
    >>> X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    >>> print(f"Original: {len(y_train)}, Resampled: {len(y_resampled)}")

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

License:
--------
GNU General Public License v2.0 or later

"""
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Try to import imbalanced-learn (optional dependency)
try:
    from imblearn.over_sampling import SMOTE as ImbSMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbSMOTE = None

# Try to import from dzetsaka modules
try:
    from ..domain.exceptions import DependencyError, ValidationError
except ImportError:
    # Fallback exceptions if domain not available
    class DependencyError(Exception):  # noqa: N818
        """Dependency error fallback."""

    class ValidationError(Exception):  # noqa: N818
        """Validation error fallback."""


class SMOTESampler:
    """SMOTE-based oversampling for handling class imbalance.

    Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic
    samples for minority classes by interpolating between existing samples
    and their k-nearest neighbors.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest neighbors to use for generating synthetic samples
        Smaller values create samples closer to existing ones
        Larger values create more diverse synthetic samples
    random_state : int, default=42
        Random seed for reproducibility
    sampling_strategy : str or dict, default='auto'
        Strategy for resampling:
        - 'auto': Balance all classes to majority class size
        - 'minority': Oversample only minority class
        - 'not majority': Oversample all except majority
        - 'not minority': Oversample all except minority
        - 'all': Oversample all classes
        - dict: {class_label: n_samples} for custom targets

    Attributes
    ----------
    k_neighbors : int
        Number of neighbors for SMOTE
    random_state : int
        Random seed
    sampling_strategy : str or dict
        Resampling strategy
    smote : object
        Underlying SMOTE implementation (if imbalanced-learn available)

    Raises
    ------
    DependencyError
        If imbalanced-learn is not installed
    ValidationError
        If input data is invalid

    Example
    -------
    >>> # Handle imbalanced dataset
    >>> sampler = SMOTESampler(k_neighbors=5)
    >>> X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    >>>
    >>> # Check class distribution
    >>> unique, counts = np.unique(y_balanced, return_counts=True)
    >>> for cls, count in zip(unique, counts):
    ...     print(f"Class {cls}: {count} samples")

    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: int = 42,
        sampling_strategy: Union[str, Dict[int, int]] = "auto",
    ):
        """Initialize SMOTE sampler with parameters."""
        if not IMBLEARN_AVAILABLE:
            raise DependencyError(
                package_name="imbalanced-learn",
                reason="SMOTE requires imbalanced-learn package",
                required_version=">=0.10.0",
            )

        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

        # Create underlying SMOTE instance
        self.smote = ImbSMOTE(
            k_neighbors=k_neighbors,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples to balance dataset.

        Creates synthetic samples for minority classes by interpolating
        between existing samples and their k-nearest neighbors.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training labels, shape (n_samples,)

        Returns
        -------
        X_resampled : np.ndarray
            Features with synthetic samples added
        y_resampled : np.ndarray
            Labels with synthetic samples added

        Raises
        ------
        ValidationError
            If input data is invalid
        ValueError
            If k_neighbors is too large for minority class

        Example
        -------
        >>> # Original imbalanced data: 900 class 0, 100 class 1
        >>> X_balanced, y_balanced = sampler.fit_resample(X, y)
        >>> # Result: 900 class 0, 900 class 1 (900 synthetic samples added)

        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValidationError(
                validation_type="SMOTE input",
                reason="X cannot be None or empty",
            )

        if y is None or len(y) == 0:
            raise ValidationError(
                validation_type="SMOTE input",
                reason="y cannot be None or empty",
            )

        if len(X) != len(y):
            raise ValidationError(
                validation_type="SMOTE input",
                reason=f"X and y must have same length (X: {len(X)}, y: {len(y)})",
            )

        # Check minimum samples per class
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()

        if min_samples <= self.k_neighbors:
            # Adjust k_neighbors if necessary
            new_k_neighbors = max(1, min_samples - 1)
            if new_k_neighbors != self.k_neighbors:
                # Recreate SMOTE with adjusted k_neighbors
                self.smote = ImbSMOTE(
                    k_neighbors=new_k_neighbors,
                    random_state=self.random_state,
                    sampling_strategy=self.sampling_strategy,
                )

        try:
            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            return X_resampled, y_resampled

        except ValueError as e:
            # Handle SMOTE-specific errors
            if "k_neighbors" in str(e).lower():
                raise ValidationError(
                    validation_type="SMOTE k_neighbors",
                    reason=f"k_neighbors={self.k_neighbors} is too large for smallest class. "
                    f"Minimum class has {min_samples} samples. "
                    f"Try reducing k_neighbors to {min_samples - 1} or less.",
                ) from e
            raise

    def get_class_distribution(self, y: np.ndarray) -> Dict[int, int]:
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
        >>> dist = sampler.get_class_distribution(y_train)
        >>> print(dist)  # {0: 900, 1: 100}

        """
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def compute_imbalance_ratio(self, y: np.ndarray) -> float:
        """Compute imbalance ratio (majority/minority).

        Parameters
        ----------
        y : np.ndarray
            Labels array

        Returns
        -------
        ratio : float
            Imbalance ratio (>1 if imbalanced, 1 if balanced)

        Example
        -------
        >>> ratio = sampler.compute_imbalance_ratio(y_train)
        >>> if ratio > 1.5:
        ...     print(f"Dataset is imbalanced (ratio: {ratio:.2f})")

        """
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return 1.0

        majority_count = counts.max()
        minority_count = counts.min()

        return majority_count / minority_count

    def should_apply_smote(
        self,
        y: np.ndarray,
        threshold: float = 1.5,
    ) -> bool:
        """Determine if SMOTE should be applied based on imbalance ratio.

        Parameters
        ----------
        y : np.ndarray
            Labels array
        threshold : float, default=1.5
            Imbalance ratio threshold above which SMOTE is recommended
            1.0 = perfectly balanced
            1.5 = 50% more majority samples than minority
            2.0 = 2x more majority samples

        Returns
        -------
        should_apply : bool
            True if imbalance ratio exceeds threshold

        Example
        -------
        >>> if sampler.should_apply_smote(y_train, threshold=2.0):
        ...     X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
        ... else:
        ...     X_balanced, y_balanced = X_train, y_train

        """
        ratio = self.compute_imbalance_ratio(y)
        return ratio > threshold


def check_imblearn_available() -> Tuple[bool, Optional[str]]:
    """Check if imbalanced-learn is available and return version info.

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_available, version_string)

    Example
    -------
    >>> available, version = check_imblearn_available()
    >>> if available:
    ...     print(f"imbalanced-learn {version} is available")
    ... else:
    ...     print("Install with: pip install imbalanced-learn>=0.10.0")

    """
    if not IMBLEARN_AVAILABLE:
        return False, None

    try:
        import imblearn

        version = imblearn.__version__
        return True, version
    except AttributeError:
        return True, "unknown"


def apply_smote_if_needed(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 1.5,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Convenience function to apply SMOTE only if dataset is imbalanced.

    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    threshold : float, default=1.5
        Imbalance ratio threshold
    k_neighbors : int, default=5
        Number of neighbors for SMOTE
    random_state : int, default=42
        Random seed

    Returns
    -------
    X_result : np.ndarray
        Resampled features (or original if balanced)
    y_result : np.ndarray
        Resampled labels (or original if balanced)
    was_applied : bool
        True if SMOTE was applied

    Example
    -------
    >>> X_train, y_train, applied = apply_smote_if_needed(X, y)
    >>> if applied:
    ...     print("SMOTE was applied to balance the dataset")

    """
    if not IMBLEARN_AVAILABLE:
        # Return original data if imbalanced-learn not available
        return X, y, False

    sampler = SMOTESampler(
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    if sampler.should_apply_smote(y, threshold=threshold):
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, True
    else:
        return X, y, False
