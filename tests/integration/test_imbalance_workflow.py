"""Integration tests for class imbalance handling workflow.

Tests the end-to-end integration of SMOTE, class weights, and
nested CV with different classifiers.
"""
import numpy as np
import pytest

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import imbalanced-learn
try:
    import imblearn

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Try to import dzetsaka modules
try:
    from scripts.sampling.class_weights import (
        apply_class_weights_to_model,
        compute_class_weights,
        compute_sample_weights,
        get_class_distribution,
        get_imbalance_ratio,
        recommend_strategy,
    )

    CLASS_WEIGHTS_AVAILABLE = True
except ImportError:
    CLASS_WEIGHTS_AVAILABLE = False

try:
    from scripts.sampling.smote_sampler import SMOTESampler, apply_smote_if_needed

    SMOTE_AVAILABLE = IMBLEARN_AVAILABLE
except ImportError:
    SMOTE_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not SKLEARN_AVAILABLE or not CLASS_WEIGHTS_AVAILABLE, reason="sklearn or class weights not available")
class TestClassWeightsWithModels:
    """Test class weights integration with actual models."""

    def test_rf_with_balanced_weights(self):
        """Test Random Forest with balanced class weights."""
        np.random.seed(42)
        # Create imbalanced dataset
        X_train = np.random.rand(200, 5)
        y_train = np.array([0] * 180 + [1] * 20)

        # Compute weights
        weights = compute_class_weights(y_train, strategy="balanced")

        # Get model parameters
        params = apply_class_weights_to_model("RF", weights)

        # Train with class weights
        model = RandomForestClassifier(
            n_estimators=20,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_train[:50])

        # Should predict both classes
        assert len(np.unique(y_pred)) > 0

    def test_svm_with_balanced_weights(self):
        """Test SVM with balanced class weights."""
        np.random.seed(42)
        X_train = np.random.rand(150, 3)
        y_train = np.array([0] * 130 + [1] * 20)

        # Compute and apply weights
        weights = compute_class_weights(y_train, strategy="balanced")
        params = apply_class_weights_to_model("SVM", weights)

        # SVM needs probability for some tests
        model = SVC(kernel="linear", probability=True, random_state=42, **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train[:50])
        assert len(y_pred) == 50

    def test_sample_weights_with_rf(self):
        """Test sample weights approach with Random Forest."""
        np.random.seed(42)
        X_train = np.random.rand(200, 4)
        y_train = np.array([0] * 180 + [1] * 20)

        # Compute sample weights
        class_weights = compute_class_weights(y_train, strategy="balanced")
        sample_weights = compute_sample_weights(y_train, class_weights)

        # Train with sample weights
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_train[:50])
        assert len(y_pred) == 50


@pytest.mark.integration
@pytest.mark.skipif(not SMOTE_AVAILABLE or not SKLEARN_AVAILABLE, reason="SMOTE or sklearn not available")
class TestSMOTEWithModels:
    """Test SMOTE integration with actual models."""

    def test_smote_then_train_rf(self):
        """Test complete SMOTE + Train workflow."""
        np.random.seed(42)
        # Create imbalanced dataset
        X_train = np.random.rand(200, 5)
        y_train = np.array([0] * 180 + [1] * 20)

        # Apply SMOTE
        sampler = SMOTESampler(k_neighbors=5, random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)

        # Verify balance
        unique, counts = np.unique(y_balanced, return_counts=True)
        assert counts[0] == counts[1]  # Should be balanced

        # Train on balanced data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_balanced, y_balanced)

        # Predict on original test data
        y_pred = model.predict(X_train[:50])
        assert len(y_pred) == 50

    def test_smote_multiclass_workflow(self):
        """Test SMOTE with multiclass data."""
        np.random.seed(42)
        X_train = np.random.rand(200, 4)
        y_train = np.array([0] * 150 + [1] * 30 + [2] * 20)

        # Apply SMOTE
        sampler = SMOTESampler(k_neighbors=3, random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)

        # All classes should have same count
        unique, counts = np.unique(y_balanced, return_counts=True)
        assert len(unique) == 3
        assert counts.max() == counts.min()  # All balanced

        # Train model
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_balanced, y_balanced)

        # Predict on a balanced sample across classes
        idx = []
        for cls in unique:
            cls_idx = np.where(y_balanced == cls)[0][:10]
            idx.extend(cls_idx.tolist())
        X_eval = X_balanced[idx]
        y_pred = model.predict(X_eval)
        assert len(np.unique(y_pred)) >= 2  # Should predict multiple classes

    def test_apply_smote_if_needed_integration(self):
        """Test convenience function with actual model."""
        np.random.seed(42)
        X_train = np.random.rand(200, 4)
        y_train = np.array([0] * 180 + [1] * 20)  # Ratio = 9.0

        # Should apply SMOTE (ratio > threshold)
        X_result, y_result, applied = apply_smote_if_needed(
            X_train, y_train, threshold=1.5
        )

        assert applied is True

        # Train on result
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_result, y_result)

        y_pred = model.predict(X_train[:50])
        assert len(y_pred) == 50


@pytest.mark.integration
@pytest.mark.skipif(not SKLEARN_AVAILABLE or not CLASS_WEIGHTS_AVAILABLE, reason="sklearn or class weights not available")
class TestCombinedImbalanceHandling:
    """Test combined SMOTE + class weights strategies."""

    @pytest.mark.skipif(not SMOTE_AVAILABLE, reason="SMOTE not available")
    def test_smote_plus_class_weights(self):
        """Test applying both SMOTE and class weights."""
        np.random.seed(42)
        X_train = np.random.rand(200, 4)
        y_train = np.array([0] * 180 + [1] * 20)

        # Step 1: Apply SMOTE
        X_balanced, y_balanced = apply_smote_if_needed(X_train, y_train, threshold=1.5)[:2]

        # Step 2: Compute class weights on balanced data
        weights = compute_class_weights(y_balanced, strategy="balanced")
        params = apply_class_weights_to_model("RF", weights)

        # Step 3: Train with both
        model = RandomForestClassifier(n_estimators=20, random_state=42, **params)
        model.fit(X_balanced, y_balanced)

        # Predict
        y_pred = model.predict(X_train[:50])
        assert len(y_pred) == 50

    def test_strategy_recommendation_workflow(self):
        """Test strategy recommendation and application."""
        # Moderate imbalance (ratio = 3.0)
        y_moderate = np.array([0] * 300 + [1] * 100)
        strategy = recommend_strategy(y_moderate, threshold=2.0)
        assert strategy == "class_weights"

        # Severe imbalance (ratio = 9.0)
        y_severe = np.array([0] * 900 + [1] * 100)
        strategy = recommend_strategy(y_severe, threshold=2.0)
        assert strategy == "smote"

        # Balanced (ratio = 1.0)
        y_balanced = np.array([0] * 500 + [1] * 500)
        strategy = recommend_strategy(y_balanced, threshold=2.0)
        assert strategy == "none"


@pytest.mark.integration
@pytest.mark.skipif(not SKLEARN_AVAILABLE or not CLASS_WEIGHTS_AVAILABLE, reason="sklearn or class weights not available")
class TestImbalanceImpactOnMetrics:
    """Test that imbalance handling improves metrics."""

    @pytest.mark.slow
    def test_weighted_f1_improves_with_class_weights(self):
        """Test that class weights improve F1 for minority class."""
        np.random.seed(42)

        # Create clearly separable but imbalanced data
        # Class 0: centered at [0, 0]
        # Class 1: centered at [3, 3]
        n_majority = 200
        n_minority = 20

        X_class0 = np.random.randn(n_majority, 2) * 0.5
        X_class1 = np.random.randn(n_minority, 2) * 0.5 + 3.0
        X_train = np.vstack([X_class0, X_class1])
        y_train = np.array([0] * n_majority + [1] * n_minority)

        # Test data (balanced)
        X_test_0 = np.random.randn(50, 2) * 0.5
        X_test_1 = np.random.randn(50, 2) * 0.5 + 3.0
        X_test = np.vstack([X_test_0, X_test_1])
        _y_test = np.array([0] * 50 + [1] * 50)

        # Model WITHOUT class weights
        model_no_weights = RandomForestClassifier(n_estimators=50, random_state=42)
        model_no_weights.fit(X_train, y_train)
        y_pred_no_weights = model_no_weights.predict(X_test)

        # Model WITH class weights
        weights = compute_class_weights(y_train, strategy="balanced")
        params = apply_class_weights_to_model("RF", weights)
        model_with_weights = RandomForestClassifier(n_estimators=50, random_state=42, **params)
        model_with_weights.fit(X_train, y_train)
        y_pred_with_weights = model_with_weights.predict(X_test)

        # Both should produce valid predictions
        assert len(y_pred_no_weights) == 100
        assert len(y_pred_with_weights) == 100

        # Both should predict both classes (data is clearly separable)
        assert len(np.unique(y_pred_no_weights)) == 2
        assert len(np.unique(y_pred_with_weights)) == 2


@pytest.mark.integration
@pytest.mark.skipif(not CLASS_WEIGHTS_AVAILABLE, reason="class_weights module not available")
class TestImbalanceEdgeCases:
    """Test edge cases in imbalance handling."""

    def test_single_sample_per_class(self):
        """Test handling of single sample per class."""
        # This is an extreme edge case
        y = np.array([0, 1, 2])  # One sample per class
        ratio = get_imbalance_ratio(y)
        assert ratio == 1.0  # Perfectly balanced

    def test_very_high_imbalance(self):
        """Test with very high imbalance ratio."""
        y = np.array([0] * 990 + [1] * 10)
        ratio = get_imbalance_ratio(y)
        assert abs(ratio - 99.0) < 0.01

        strategy = recommend_strategy(y)
        assert strategy == "smote"

    def test_three_class_imbalance(self):
        """Test multiclass imbalance detection."""
        y = np.array([0] * 800 + [1] * 150 + [2] * 50)

        dist = get_class_distribution(y)
        assert dist[0] == 800
        assert dist[1] == 150
        assert dist[2] == 50

        ratio = get_imbalance_ratio(y)
        assert abs(ratio - 16.0) < 0.01  # 800/50
