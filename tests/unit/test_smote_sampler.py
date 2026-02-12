"""Unit tests for SMOTE sampler module.

Tests the SMOTESampler class and related utility functions.
"""

import numpy as np
import pytest

# Try to import imbalanced-learn
try:
    import imblearn

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Try to import the module under test
try:
    from scripts.sampling.smote_sampler import (
        SMOTESampler,
        apply_smote_if_needed,
        check_imblearn_available,
    )

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="Sampling module not available")


class TestCheckImblearn:
    """Test imbalanced-learn availability checking."""

    def test_check_returns_tuple(self):
        """Test that check returns a tuple of (bool, str/None)."""
        result = check_imblearn_available()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_returns_bool_first(self):
        """Test first element is boolean."""
        available, _ = check_imblearn_available()
        assert isinstance(available, bool)


@pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
class TestSMOTESamplerInit:
    """Test SMOTESampler initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        sampler = SMOTESampler()
        assert sampler.k_neighbors == 5
        assert sampler.random_state == 42
        assert sampler.sampling_strategy == "auto"

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        sampler = SMOTESampler(k_neighbors=3, random_state=0, sampling_strategy="minority")
        assert sampler.k_neighbors == 3
        assert sampler.random_state == 0
        assert sampler.sampling_strategy == "minority"

    def test_init_creates_smote_instance(self):
        """Test that underlying SMOTE instance is created."""
        sampler = SMOTESampler()
        assert sampler.smote is not None


@pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
class TestSMOTEFitResample:
    """Test SMOTE fit_resample functionality."""

    def test_basic_binary_resampling(self):
        """Test basic binary class resampling."""
        # Create imbalanced dataset: 90 class 0, 10 class 1
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)

        sampler = SMOTESampler(k_neighbors=5, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should have more samples after resampling
        assert len(y_resampled) > len(y)
        assert len(X_resampled) == len(y_resampled)

        # Classes should be more balanced
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2

        # Minority class should have increased
        minority_original = 10
        minority_new = np.sum(y_resampled == 1)
        assert minority_new > minority_original

    def test_multiclass_resampling(self):
        """Test multiclass resampling."""
        np.random.seed(42)
        X = np.random.rand(150, 4)
        y = np.array([0] * 100 + [1] * 30 + [2] * 20)

        sampler = SMOTESampler(k_neighbors=3, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should have 3 classes still
        assert len(np.unique(y_resampled)) == 3

        # Minority classes should have grown
        assert np.sum(y_resampled == 2) > 20

    def test_already_balanced_no_error(self):
        """Test that balanced data doesn't error (may still resample)."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = np.array([0] * 50 + [1] * 50)

        sampler = SMOTESampler(k_neighbors=5, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should not fail
        assert len(y_resampled) >= len(y)
        assert len(X_resampled) == len(y_resampled)

    def test_raises_on_none_input(self):
        """Test that None input raises error."""
        sampler = SMOTESampler()

        with pytest.raises(Exception):  # ValidationError
            sampler.fit_resample(None, np.array([0, 1]))

    def test_raises_on_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        sampler = SMOTESampler()

        X = np.random.rand(100, 3)
        y = np.array([0, 1, 2])  # Wrong length

        with pytest.raises(Exception):  # ValidationError
            sampler.fit_resample(X, y)

    def test_auto_adjusts_k_neighbors(self):
        """Test that k_neighbors is auto-adjusted for small classes."""
        # Create dataset with very small minority class (3 samples)
        np.random.seed(42)
        X = np.random.rand(103, 3)
        y = np.array([0] * 100 + [1] * 3)

        # k_neighbors=5 is too large for 3 samples
        sampler = SMOTESampler(k_neighbors=5, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Should not error - k_neighbors should be auto-adjusted
        assert len(y_resampled) > 103


@pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
class TestSMOTEUtilities:
    """Test SMOTE utility functions."""

    def test_get_class_distribution(self):
        """Test class distribution computation."""
        y = np.array([0, 0, 0, 1, 1, 2])
        sampler = SMOTESampler()

        dist = sampler.get_class_distribution(y)

        assert dist[0] == 3
        assert dist[1] == 2
        assert dist[2] == 1

    def test_compute_imbalance_ratio(self):
        """Test imbalance ratio computation."""
        y = np.array([0] * 900 + [1] * 100)
        sampler = SMOTESampler()

        ratio = sampler.compute_imbalance_ratio(y)

        assert abs(ratio - 9.0) < 0.01

    def test_compute_imbalance_ratio_balanced(self):
        """Test that balanced data returns ratio close to 1."""
        y = np.array([0] * 50 + [1] * 50)
        sampler = SMOTESampler()

        ratio = sampler.compute_imbalance_ratio(y)

        assert abs(ratio - 1.0) < 0.01

    def test_should_apply_smote_imbalanced(self):
        """Test SMOTE recommendation for imbalanced data."""
        y = np.array([0] * 900 + [1] * 100)  # Ratio = 9.0
        sampler = SMOTESampler()

        assert sampler.should_apply_smote(y, threshold=1.5) is True

    def test_should_apply_smote_balanced(self):
        """Test SMOTE not recommended for balanced data."""
        y = np.array([0] * 50 + [1] * 50)  # Ratio = 1.0
        sampler = SMOTESampler()

        assert sampler.should_apply_smote(y, threshold=1.5) is False


@pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
class TestApplySMOTEIfNeeded:
    """Test the convenience function apply_smote_if_needed."""

    def test_applies_when_imbalanced(self):
        """Test SMOTE is applied for imbalanced data."""
        np.random.seed(42)
        X = np.random.rand(110, 4)
        y = np.array([0] * 100 + [1] * 10)  # Ratio = 10.0

        X_result, y_result, was_applied = apply_smote_if_needed(X, y, threshold=1.5)

        assert was_applied is True
        assert len(y_result) > len(y)

    def test_skips_when_balanced(self):
        """Test SMOTE is not applied for balanced data."""
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = np.array([0] * 50 + [1] * 50)  # Ratio = 1.0

        X_result, y_result, was_applied = apply_smote_if_needed(X, y, threshold=1.5)

        assert was_applied is False
        assert len(y_result) == len(y)


class TestSMOTEEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
    def test_single_class_imbalance_ratio(self):
        """Test imbalance ratio with single class."""
        y = np.array([0, 0, 0])
        sampler = SMOTESampler()

        ratio = sampler.compute_imbalance_ratio(y)
        assert ratio == 1.0

    @pytest.mark.skipif(not IMBLEARN_AVAILABLE, reason="imbalanced-learn not installed")
    def test_binary_equal_classes(self):
        """Test with exactly equal class sizes."""
        np.random.seed(42)
        np.random.rand(100, 3)
        y = np.array([0] * 50 + [1] * 50)

        sampler = SMOTESampler()

        # Ratio should be exactly 1
        assert sampler.compute_imbalance_ratio(y) == 1.0
