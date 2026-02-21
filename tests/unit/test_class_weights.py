"""Unit tests for class weights module.

Tests class weight computation, strategy recommendations, and model integration.
"""

import numpy as np
import pytest

# Try to import the module under test
try:
    from scripts.sampling.class_weights import (
        apply_class_weights_to_model,
        compute_class_weights,
        compute_sample_weights,
        get_class_distribution,
        get_imbalance_ratio,
        normalize_weights,
        print_class_distribution,
        recommend_strategy,
    )

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="Class weights module not available")


class TestComputeClassWeights:
    """Test class weight computation."""

    def test_balanced_strategy_binary(self):
        """Test balanced strategy for binary classification."""
        # 900 class 0, 100 class 1
        y = np.array([0] * 900 + [1] * 100)
        weights = compute_class_weights(y, strategy="balanced")

        assert 0 in weights
        assert 1 in weights

        # Class 1 should have higher weight (minority)
        assert weights[1] > weights[0]

        # Class 0 weight should be close to n / (n_classes * count_0) = 1000 / (2 * 900) = 0.556
        expected_w0 = 1000 / (2 * 900)
        expected_w1 = 1000 / (2 * 100)
        assert abs(weights[0] - expected_w0) < 0.01
        assert abs(weights[1] - expected_w1) < 0.01

    def test_balanced_strategy_multiclass(self):
        """Test balanced strategy for multiclass."""
        y = np.array([0] * 500 + [1] * 300 + [2] * 200)
        weights = compute_class_weights(y, strategy="balanced")

        assert len(weights) == 3
        # Weight inversely proportional to class size
        assert weights[2] > weights[1] > weights[0]

    def test_balanced_strategy_with_column_vector_labels(self):
        """Test balanced strategy with labels shaped (n, 1)."""
        y = np.array([[0], [0], [0], [1], [1]])
        weights = compute_class_weights(y, strategy="balanced")

        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]

    def test_balanced_strategy_with_object_array_wrapped_scalars(self):
        """Test balanced strategy with object labels containing numpy scalar arrays."""
        y = np.array([np.array([0]), np.array([0]), np.array([1])], dtype=object)
        weights = compute_class_weights(y, strategy="balanced")

        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]

    def test_uniform_strategy(self):
        """Test uniform strategy (all weights = 1.0)."""
        y = np.array([0] * 90 + [1] * 10)
        weights = compute_class_weights(y, strategy="uniform")

        assert weights[0] == 1.0
        assert weights[1] == 1.0

    def test_custom_strategy(self):
        """Test custom weight strategy."""
        y = np.array([0, 1, 2])
        custom = {0: 1.0, 1: 5.0, 2: 10.0}
        weights = compute_class_weights(y, strategy="custom", custom_weights=custom)

        assert weights == custom

    def test_custom_strategy_without_weights_raises(self):
        """Test that custom strategy without weights raises error."""
        y = np.array([0, 1])

        with pytest.raises(ValueError, match="custom_weights must be provided"):
            compute_class_weights(y, strategy="custom")

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        y = np.array([0, 1])

        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_class_weights(y, strategy="invalid_strategy")


class TestApplyClassWeightsToModel:
    """Test model-specific weight parameter conversion."""

    def test_sklearn_models(self):
        """Test sklearn model parameter format."""
        weights = {0: 0.5, 1: 4.5}

        # All sklearn models should get class_weight parameter
        for code in ["RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"]:
            params = apply_class_weights_to_model(code, weights)
            assert "class_weight" in params
            assert params["class_weight"] == weights

    def test_xgb_binary(self):
        """Test XGBoost binary classification format."""
        weights = {0: 0.5, 1: 4.5}
        params = apply_class_weights_to_model("XGB", weights)

        assert "scale_pos_weight" in params
        assert abs(params["scale_pos_weight"] - (4.5 / 0.5)) < 0.01

    def test_xgb_multiclass_returns_empty(self):
        """Test XGBoost multiclass returns empty (unsupported)."""
        weights = {0: 0.5, 1: 2.0, 2: 4.0}
        params = apply_class_weights_to_model("XGB", weights)

        assert params == {}

    def test_gmm_returns_empty(self):
        """Test GMM returns empty (unsupported)."""
        weights = {0: 1.0, 1: 2.0}
        params = apply_class_weights_to_model("GMM", weights)

        assert params == {}


class TestGetImbalanceRatio:
    """Test imbalance ratio computation."""

    def test_binary_imbalanced(self):
        """Test imbalance ratio for binary imbalanced."""
        y = np.array([0] * 900 + [1] * 100)
        ratio = get_imbalance_ratio(y)
        assert abs(ratio - 9.0) < 0.01

    def test_balanced(self):
        """Test ratio for balanced data."""
        y = np.array([0] * 50 + [1] * 50)
        ratio = get_imbalance_ratio(y)
        assert abs(ratio - 1.0) < 0.01

    def test_single_class(self):
        """Test ratio with single class returns 1.0."""
        y = np.array([0, 0, 0, 0])
        ratio = get_imbalance_ratio(y)
        assert ratio == 1.0

    def test_multiclass(self):
        """Test ratio for multiclass (max/min)."""
        y = np.array([0] * 500 + [1] * 300 + [2] * 100)
        ratio = get_imbalance_ratio(y)
        assert abs(ratio - 5.0) < 0.01


class TestRecommendStrategy:
    """Test strategy recommendation logic."""

    def test_recommends_none_for_balanced(self):
        """Test 'none' recommended for balanced data."""
        y = np.array([0] * 50 + [1] * 50)  # Ratio = 1.0
        strategy = recommend_strategy(y)
        assert strategy == "none"

    def test_recommends_class_weights_for_moderate(self):
        """Test 'class_weights' for moderate imbalance."""
        y = np.array([0] * 300 + [1] * 100)  # Ratio = 3.0
        strategy = recommend_strategy(y, threshold=2.0)
        assert strategy == "class_weights"

    def test_recommends_smote_for_severe(self):
        """Test 'smote' recommended for severe imbalance."""
        y = np.array([0] * 900 + [1] * 100)  # Ratio = 9.0
        strategy = recommend_strategy(y, threshold=2.0)
        assert strategy == "smote"


class TestComputeSampleWeights:
    """Test per-sample weight computation."""

    def test_basic_sample_weights(self):
        """Test basic sample weight assignment."""
        y = np.array([0, 0, 0, 1, 1])
        class_weights = {0: 0.5, 1: 4.5}

        sample_weights = compute_sample_weights(y, class_weights)

        assert sample_weights.shape == (5,)
        assert sample_weights[0] == 0.5  # Class 0
        assert sample_weights[3] == 4.5  # Class 1

    def test_multiclass_sample_weights(self):
        """Test sample weights for multiclass."""
        y = np.array([0, 1, 2, 0, 1])
        class_weights = {0: 1.0, 1: 2.0, 2: 5.0}

        sample_weights = compute_sample_weights(y, class_weights)

        assert sample_weights[0] == 1.0  # Class 0
        assert sample_weights[1] == 2.0  # Class 1
        assert sample_weights[2] == 5.0  # Class 2

    def test_sample_weights_with_column_vector_labels(self):
        """Test sample weights with labels shaped (n, 1)."""
        y = np.array([[0], [1], [0], [1]])
        class_weights = {0: 0.5, 1: 4.5}

        sample_weights = compute_sample_weights(y, class_weights)

        assert sample_weights.shape == (4,)
        assert sample_weights[0] == 0.5
        assert sample_weights[1] == 4.5


class TestNormalizeWeights:
    """Test weight normalization."""

    def test_normalize_basic(self):
        """Test basic weight normalization."""
        weights = {0: 1.0, 1: 9.0}  # Sum = 10, 2 classes
        normalized = normalize_weights(weights)

        # Should sum to n_classes (2)
        total = sum(normalized.values())
        assert abs(total - 2.0) < 0.01

    def test_normalize_already_normalized(self):
        """Test that already normalized weights remain similar."""
        weights = {0: 1.0, 1: 1.0}  # Already balanced
        normalized = normalize_weights(weights)

        assert abs(normalized[0] - 1.0) < 0.01
        assert abs(normalized[1] - 1.0) < 0.01

    def test_normalize_zero_weights(self):
        """Test normalization with zero total weight."""
        weights = {0: 0.0, 1: 0.0}
        normalized = normalize_weights(weights)

        # Should return 1.0 for all to avoid division by zero
        assert normalized[0] == 1.0
        assert normalized[1] == 1.0


class TestGetClassDistribution:
    """Test class distribution computation."""

    def test_basic_distribution(self):
        """Test basic class distribution."""
        y = np.array([0, 0, 0, 1, 1, 2])
        dist = get_class_distribution(y)

        assert dist[0] == 3
        assert dist[1] == 2
        assert dist[2] == 1

    def test_empty_array(self):
        """Test with empty array."""
        y = np.array([])
        dist = get_class_distribution(y)

        assert dist == {}
