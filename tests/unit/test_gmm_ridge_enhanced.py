"""Comprehensive tests for enhanced GMM Ridge implementation.

Tests cover:
- sklearn API compatibility
- Numerical stability
- Feature importance
- Covariance diagnostics
- Backward compatibility
- Model persistence
- Edge cases
"""

import pickle
import warnings

import numpy as np
import pytest

try:
    from scripts.gmm_ridge import GMMR, ridge

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

# Check for sklearn
try:
    import sklearn
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.utils.estimator_checks import check_estimator

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="gmm_ridge module not available")


def _make_toy_data():
    """Create simple 2-class toy dataset."""
    x0 = np.array([[0.0, 0.1], [0.1, -0.1], [-0.1, 0.05]])
    x1 = np.array([[2.0, 2.1], [2.1, 1.9], [1.9, 2.05]])
    x = np.vstack([x0, x1])
    y = np.array([1, 1, 1, 2, 2, 2], dtype=np.uint16)
    return x, y


def _make_multiclass_data():
    """Create 3-class dataset."""
    np.random.seed(42)
    x0 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
    x1 = np.random.randn(20, 5) + np.array([3, 3, 0, 0, 0])
    x2 = np.random.randn(20, 5) + np.array([0, 0, 3, 3, 3])
    x = np.vstack([x0, x1, x2])
    y = np.array([1] * 20 + [2] * 20 + [3] * 20)
    return x, y


# ==================== Basic Functionality Tests ====================


def test_fit_predict_basic():
    """Test basic fit and predict workflow."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    predictions = model.predict(x)
    assert predictions.shape == (x.shape[0],)
    assert np.all(np.isin(predictions, [1, 2]))


def test_learn_predict_backward_compatibility():
    """Test that old learn() API still works."""
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    predictions = model.predict(x)
    assert predictions.shape == (x.shape[0],)


def test_fit_and_learn_produce_same_results():
    """Ensure fit() and learn() are equivalent."""
    x, y = _make_toy_data()

    model1 = GMMR(random_state=42)
    model1.fit(x, y)
    pred1 = model1.predict(x)

    model2 = GMMR(random_state=42)
    model2.learn(x, y)
    pred2 = model2.predict(x)

    np.testing.assert_array_equal(pred1, pred2)


# ==================== sklearn Compatibility Tests ====================


def test_sklearn_api_attributes():
    """Test that sklearn-required attributes are present."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    assert hasattr(model, "classes_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == x.shape[1]
    assert len(model.classes_) == 2


def test_predict_proba_returns_probabilities():
    """Test predict_proba returns valid probability distribution."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=0.1)
    model.fit(x, y)

    proba = model.predict_proba(x)

    assert proba.shape == (x.shape[0], 3)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)


def test_predict_proba_matches_predict():
    """Test that predict_proba argmax matches predict."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=0.1)
    model.fit(x, y)

    predictions = model.predict(x)
    proba = model.predict_proba(x)

    proba_predictions = model.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(predictions, proba_predictions)


def test_score_method():
    """Test that score() method works (from ClassifierMixin)."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    score = model.score(x, y)
    assert 0.0 <= score <= 1.0


def test_get_params_set_params():
    """Test get_params and set_params (from BaseEstimator)."""
    model = GMMR(tau=0.5, random_state=42)

    params = model.get_params()
    assert params["tau"] == 0.5
    assert params["random_state"] == 42

    model.set_params(tau=1.0)
    assert model.tau == 1.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
def test_cross_val_score_integration():
    """Test integration with sklearn cross_val_score."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=0.1)

    scores = cross_val_score(model, x, y, cv=3)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


# ==================== Numerical Stability Tests ====================


def test_predict_with_zero_tau_stable():
    """Test that prediction with tau=0 doesn't cause numerical issues."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predictions = model.predict(x, tau=0.0)

    assert np.all(np.isfinite(predictions))


def test_predict_with_high_tau():
    """Test prediction with very high regularization."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=1000.0)
    model.fit(x, y)

    predictions = model.predict(x)
    assert np.all(np.isfinite(predictions))


def test_predict_proba_numerical_stability():
    """Test that predict_proba doesn't overflow/underflow."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=0.01)
    model.fit(x, y)

    # Test on data far from training distribution
    x_extreme = x * 1000
    proba = model.predict_proba(x_extreme)

    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_ill_conditioned_data_handling():
    """Test handling of nearly singular covariance matrices."""
    np.random.seed(42)
    x = np.random.randn(50, 10)

    # Make last column nearly collinear with first
    x[:, -1] = x[:, 0] + 1e-8 * np.random.randn(50)

    y = np.random.randint(1, 3, 50)

    # Should not crash with appropriate regularization
    model = GMMR(tau=1e-3)
    model.fit(x, y)

    predictions = model.predict(x)
    assert np.all(np.isfinite(predictions))


def test_no_numpy_deprecation_warnings_predict():
    """Ensure no NumPy deprecation warnings during predict."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        predictions = model.predict(x)

    assert isinstance(predictions, np.ndarray)


def test_no_numpy_deprecation_warnings_bic():
    """Ensure no NumPy deprecation warnings during BIC computation."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        bic = model.BIC(x, y)

    assert np.isfinite(bic)


# ==================== Feature Importance Tests ====================


def test_get_feature_importance_variance():
    """Test variance-based feature importance."""
    x, y = _make_multiclass_data()
    model = GMMR()
    model.fit(x, y)

    importance = model.get_feature_importance(method="variance")

    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0)
    np.testing.assert_almost_equal(importance.sum(), 1.0, decimal=6)


def test_get_feature_importance_discriminative():
    """Test discriminative (Fisher) feature importance."""
    x, y = _make_multiclass_data()
    model = GMMR()
    model.fit(x, y)

    importance = model.get_feature_importance(method="discriminative")

    assert importance.shape == (x.shape[1],)
    assert np.all(importance >= 0)
    np.testing.assert_almost_equal(importance.sum(), 1.0, decimal=6)


def test_feature_importance_discriminative_higher_for_separating_features():
    """Test that discriminative features have higher importance."""
    np.random.seed(42)
    # Create data where first 2 features are discriminative
    x0 = np.random.randn(30, 5)
    x0[:, 0] += 5  # Shift class 1 on feature 0
    x0[:, 1] += 5  # Shift class 1 on feature 1

    x1 = np.random.randn(30, 5)

    x = np.vstack([x0, x1])
    y = np.array([1] * 30 + [2] * 30)

    model = GMMR()
    model.fit(x, y)

    importance = model.get_feature_importance(method="discriminative")

    # Features 0 and 1 should have highest importance
    assert importance[0] > importance[2]
    assert importance[1] > importance[2]


def test_feature_importance_invalid_method():
    """Test that invalid method raises ValueError."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    with pytest.raises(ValueError, match="Unknown method"):
        model.get_feature_importance(method="invalid_method")


# ==================== Covariance Diagnostics Tests ====================


def test_get_covariance_diagnostics():
    """Test covariance diagnostics computation."""
    x, y = _make_multiclass_data()
    model = GMMR()
    model.fit(x, y)

    diagnostics = model.get_covariance_diagnostics()

    assert "condition_numbers" in diagnostics
    assert "min_eigenvalues" in diagnostics
    assert "max_eigenvalues" in diagnostics
    assert "effective_rank" in diagnostics
    assert "explained_variance_ratio" in diagnostics

    assert diagnostics["condition_numbers"].shape == (3,)
    assert all(diagnostics["condition_numbers"] > 0)


def test_diagnostics_detect_ill_conditioning():
    """Test that diagnostics can detect ill-conditioned matrices."""
    np.random.seed(42)
    x = np.random.randn(50, 10)
    # Make highly collinear
    x[:, -1] = x[:, 0] + 1e-12 * np.random.randn(50)

    y = np.random.randint(1, 3, 50)

    model = GMMR(tau=0.0)  # No regularization to expose ill-conditioning
    model.fit(x, y)

    diagnostics = model.get_covariance_diagnostics()

    # At least one class should have high condition number
    assert any(diagnostics["condition_numbers"] > 1e8)


# ==================== Cross-Validation Tests ====================


def test_cross_validation_returns_best_tau():
    """Test cross-validation returns best tau parameter."""
    x, y = _make_multiclass_data()
    model = GMMR()

    tau_grid = np.array([1e-3, 1e-2, 1e-1, 1.0])
    best_tau, scores = model.cross_validation(x, y, tau_grid, v=3)

    assert best_tau in tau_grid
    assert scores.shape == tau_grid.shape
    assert all(0 <= s <= 100 for s in scores)


def test_cross_validation_with_sklearn():
    """Test sklearn-based cross-validation if available."""
    if not SKLEARN_AVAILABLE:
        pytest.skip("sklearn not available")

    x, y = _make_multiclass_data()
    model = GMMR(random_state=42)

    tau_grid = np.array([1e-2, 1e-1, 1.0])
    best_tau, scores = model._cross_validation_sklearn(x, y, tau_grid, v=3)

    assert best_tau in tau_grid
    assert scores.shape == tau_grid.shape


def test_cross_validation_legacy():
    """Test legacy cross-validation implementation."""
    x, y = _make_multiclass_data()
    model = GMMR(random_state=42)

    tau_grid = np.array([1e-2, 1e-1, 1.0])
    best_tau, scores = model._cross_validation_legacy(x, y, tau_grid, v=3, n_jobs=1)

    assert best_tau in tau_grid
    assert scores.shape == tau_grid.shape


# ==================== BIC Tests ====================


def test_bic_computation():
    """Test BIC computation runs without error."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=0.1)
    model.fit(x, y)

    bic = model.BIC(x, y)
    assert np.isfinite(bic)
    assert isinstance(bic, float)


def test_bic_with_different_tau():
    """Test BIC with different tau values."""
    x, y = _make_toy_data()
    model = GMMR(tau=0.1)
    model.fit(x, y)

    bic1 = model.BIC(x, y, tau=0.01)
    bic2 = model.BIC(x, y, tau=1.0)

    assert np.isfinite(bic1)
    assert np.isfinite(bic2)


# ==================== Model Persistence Tests ====================


def test_pickle_serialization():
    """Test model can be pickled and unpickled."""
    x, y = _make_multiclass_data()
    model = GMMR(tau=0.5, random_state=42)
    model.fit(x, y)

    # Pickle
    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)

    # Test predictions match
    pred_original = model.predict(x)
    pred_loaded = model_loaded.predict(x)

    np.testing.assert_array_equal(pred_original, pred_loaded)


def test_pickle_preserves_parameters():
    """Test that pickling preserves all parameters."""
    model = GMMR(tau=0.7, random_state=123, min_eigenvalue=1e-5, warn_ill_conditioned=True)

    x, y = _make_toy_data()
    model.fit(x, y)

    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)

    assert model_loaded.tau == 0.7
    assert model_loaded.random_state == 123
    assert model_loaded.min_eigenvalue == 1e-5
    assert model_loaded.warn_ill_conditioned is True


# ==================== Edge Cases & Validation Tests ====================


def test_raises_on_nan_input():
    """Test that model raises error on NaN input."""
    x, y = _make_toy_data()
    x[0, 0] = np.nan

    model = GMMR()

    # sklearn's check_X_y raises ValueError with different message
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_raises_on_inf_input():
    """Test that model raises error on Inf input."""
    x, y = _make_toy_data()
    x[1, 1] = np.inf

    model = GMMR()

    # sklearn's check_X_y raises ValueError with different message
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_warns_on_insufficient_samples():
    """Test warning when class has too few samples."""
    # 2 features but only 2 samples in class 2
    x = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]])
    y = np.array([1, 1, 2, 2])

    model = GMMR()

    with pytest.warns(UserWarning, match="has only"):
        model.fit(x, y)


def test_raises_on_mismatched_features():
    """Test error when predict features don't match training features."""
    x_train, y_train = _make_toy_data()
    model = GMMR()
    model.fit(x_train, y_train)

    x_test = np.random.randn(10, 5)  # Different number of features

    with pytest.raises(ValueError, match="features but model was trained"):
        model.predict(x_test)


def test_confidence_map_backward_compatibility():
    """Test that old confidenceMap parameter still works."""
    x, y = _make_toy_data()
    model = GMMR()
    model.fit(x, y)

    preds, confidences = model.predict(x, confidenceMap=True)

    assert preds.shape == (x.shape[0],)
    assert confidences.shape == (x.shape[0],)
    assert np.all((confidences >= 0) & (confidences <= 1))


def test_predict_with_override_tau():
    """Test that tau can be overridden during prediction."""
    x, y = _make_toy_data()
    model = GMMR(tau=0.1)
    model.fit(x, y)

    pred1 = model.predict(x)
    pred2 = model.predict(x, tau=10.0)

    # Different tau might give different predictions
    assert pred1.shape == pred2.shape


# ==================== Backward Compatibility Tests ====================


def test_ridge_alias_exists():
    """Test that ridge alias exists for factory compatibility."""
    assert ridge is GMMR


def test_ridge_factory_function():
    """Test that ridge() creates GMMR instance."""
    model = ridge(tau=0.5)
    assert isinstance(model, GMMR)
    assert model.tau == 0.5


# ==================== Integration Tests ====================


def test_full_classification_pipeline():
    """Test complete classification workflow."""
    if not SKLEARN_AVAILABLE:
        pytest.skip("sklearn not available")

    # Generate realistic dataset
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=7, n_redundant=2, n_classes=3, random_state=42,
    )
    y = y + 1  # Make labels 1, 2, 3 instead of 0, 1, 2

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = GMMR(tau=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Accuracy
    accuracy = model.score(X_test, y_test)

    # Feature importance
    importance = model.get_feature_importance()

    # Assertions
    assert accuracy > 0.5  # Should do better than random
    assert y_pred.shape == (len(X_test),)
    assert y_proba.shape == (len(X_test), 3)
    assert importance.shape == (10,)


def test_reproducibility_with_random_state():
    """Test that results are reproducible with same random_state."""
    x, y = _make_multiclass_data()

    model1 = GMMR(random_state=42)
    tau_grid = np.logspace(-3, 1, 10)
    best_tau1, scores1 = model1.cross_validation(x, y, tau_grid, v=3)

    model2 = GMMR(random_state=42)
    best_tau2, scores2 = model2.cross_validation(x, y, tau_grid, v=3)

    assert best_tau1 == best_tau2
    np.testing.assert_array_almost_equal(scores1, scores2)


# ==================== Performance Regression Tests ====================


def test_prediction_speed_reasonable():
    """Test that prediction completes in reasonable time."""
    if not SKLEARN_AVAILABLE:
        pytest.skip("sklearn not available")

    import time

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)
    y = y + 1

    model = GMMR(tau=0.1)
    model.fit(X, y)

    start = time.time()
    predictions = model.predict(X)
    elapsed = time.time() - start

    assert elapsed < 2.0  # Should complete in under 2 seconds (relaxed for CI)
    assert predictions.shape == (1000,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
