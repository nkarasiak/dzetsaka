"""Tests for GMM Ridge Phase 3 advanced features.

Tests cover:
- Incremental learning (partial_fit)
- Advanced regularization (Ledoit-Wolf, OAS)
- Streaming data workflows
"""
import numpy as np
import pytest

try:
    from scripts.gmm_ridge import GMMR

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="gmm_ridge module not available")


def _make_streaming_data(n_batches=5, batch_size=20, n_features=5, n_classes=3, random_state=42):
    """Generate synthetic streaming data."""
    np.random.seed(random_state)

    batches = []
    for _ in range(n_batches):
        X_batch = []
        y_batch = []

        for c in range(1, n_classes + 1):
            X_c = np.random.randn(batch_size // n_classes, n_features) + c * 2
            y_c = np.full(batch_size // n_classes, c)
            X_batch.append(X_c)
            y_batch.append(y_c)

        X_batch = np.vstack(X_batch)
        y_batch = np.concatenate(y_batch)

        # Shuffle
        perm = np.random.permutation(len(y_batch))
        X_batch = X_batch[perm]
        y_batch = y_batch[perm]

        batches.append((X_batch, y_batch))

    return batches


# ==================== Incremental Learning Tests ====================


def test_partial_fit_single_batch():
    """Test partial_fit with single batch."""
    X = np.random.randn(100, 5)
    y = np.random.randint(1, 4, 100)

    model = GMMR(tau=0.1)
    model.partial_fit(X, y, classes=np.array([1, 2, 3]))

    assert hasattr(model, "mean")
    assert hasattr(model, "cov")
    assert model.mean.shape == (3, 5)
    assert model.cov.shape == (3, 5, 5)


def test_partial_fit_multiple_batches():
    """Test partial_fit with multiple batches."""
    batches = _make_streaming_data(n_batches=5, batch_size=20)

    model = GMMR(tau=0.1, random_state=42)

    # First batch - must provide classes
    X1, y1 = batches[0]
    model.partial_fit(X1, y1, classes=np.array([1, 2, 3]))

    # Subsequent batches
    for X_batch, y_batch in batches[1:]:
        model.partial_fit(X_batch, y_batch)

    # Should be able to predict
    X_test = np.random.randn(10, 5)
    predictions = model.predict(X_test)

    assert predictions.shape == (10,)
    assert np.all(np.isin(predictions, [1, 2, 3]))


def test_partial_fit_convergence():
    """Test that partial_fit converges to similar result as fit."""
    np.random.seed(42)

    # Generate full dataset
    X_full = []
    y_full = []
    for c in range(1, 4):
        X_c = np.random.randn(100, 5) + c * 3
        y_c = np.full(100, c)
        X_full.append(X_c)
        y_full.append(y_c)

    X_full = np.vstack(X_full)
    y_full = np.concatenate(y_full)

    # Shuffle
    perm = np.random.permutation(len(y_full))
    X_full = X_full[perm]
    y_full = y_full[perm]

    # Train with fit
    model_batch = GMMR(tau=0.1, random_state=42)
    model_batch.fit(X_full, y_full)

    # Train with partial_fit (same data in batches)
    model_incremental = GMMR(tau=0.1, random_state=42)
    batch_size = 30

    for i in range(0, len(X_full), batch_size):
        X_batch = X_full[i : i + batch_size]
        y_batch = y_full[i : i + batch_size]

        if i == 0:
            model_incremental.partial_fit(X_batch, y_batch, classes=np.array([1, 2, 3]))
        else:
            model_incremental.partial_fit(X_batch, y_batch)

    # Predictions should be very similar
    X_test = np.random.randn(50, 5)
    pred_batch = model_batch.predict(X_test)
    pred_incremental = model_incremental.predict(X_test)

    # Allow some differences due to numerical precision
    agreement = np.mean(pred_batch == pred_incremental)
    assert agreement > 0.85  # At least 85% agreement


def test_partial_fit_requires_classes_on_first_call():
    """Test that partial_fit raises error if classes not provided on first call."""
    X = np.random.randn(50, 5)
    y = np.random.randint(1, 4, 50)

    model = GMMR()

    with pytest.raises(ValueError, match="classes must be provided"):
        model.partial_fit(X, y)


def test_partial_fit_feature_mismatch():
    """Test that partial_fit raises error on feature count mismatch."""
    X1 = np.random.randn(50, 5)
    y1 = np.random.randint(1, 4, 50)

    X2 = np.random.randn(50, 7)  # Different number of features!
    y2 = np.random.randint(1, 4, 50)

    model = GMMR()
    model.partial_fit(X1, y1, classes=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="features but model expects"):
        model.partial_fit(X2, y2)


def test_partial_fit_updates_proportions():
    """Test that partial_fit correctly updates class proportions."""
    # First batch: balanced classes
    X1 = np.vstack([np.random.randn(10, 3), np.random.randn(10, 3) + 5])
    y1 = np.array([1] * 10 + [2] * 10)

    model = GMMR(tau=0.1)
    model.partial_fit(X1, y1, classes=np.array([1, 2]))

    # Initially balanced
    assert abs(model.prop[0, 0] - 0.5) < 0.01
    assert abs(model.prop[1, 0] - 0.5) < 0.01

    # Second batch: more class 2
    X2 = np.vstack([np.random.randn(5, 3), np.random.randn(20, 3) + 5])
    y2 = np.array([1] * 5 + [2] * 20)

    model.partial_fit(X2, y2)

    # Now should be imbalanced (15 class 1, 30 class 2 = 1:2 ratio)
    assert abs(model.prop[0, 0] - 1 / 3) < 0.01
    assert abs(model.prop[1, 0] - 2 / 3) < 0.01


# ==================== Advanced Regularization Tests ====================


def test_ledoit_wolf_regularization():
    """Test Ledoit-Wolf shrinkage regularization."""
    X = np.random.randn(100, 10)
    y = np.random.randint(1, 4, 100)

    model = GMMR(tau=0.1, reg_type="ledoit_wolf", shrinkage_target="diagonal")
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (100,)


def test_oas_regularization():
    """Test OAS regularization."""
    X = np.random.randn(100, 10)
    y = np.random.randint(1, 4, 100)

    model = GMMR(tau=0.1, reg_type="oas", shrinkage_target="identity")
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (100,)


def test_regularization_types():
    """Test all regularization types."""
    X = np.random.randn(100, 5)
    y = np.random.randint(1, 4, 100)

    reg_types = ["ridge", "ledoit_wolf", "oas", "empirical"]

    for reg_type in reg_types:
        model = GMMR(tau=0.1, reg_type=reg_type)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (100,)


def test_shrinkage_targets():
    """Test all shrinkage target types."""
    X = np.random.randn(100, 5)
    y = np.random.randint(1, 4, 100)

    targets = ["diagonal", "identity", "spherical"]

    for target in targets:
        model = GMMR(tau=0.1, reg_type="ledoit_wolf", shrinkage_target=target)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (100,)


def test_invalid_reg_type():
    """Test that invalid regularization type raises error."""
    X = np.random.randn(50, 5)
    y = np.random.randint(1, 3, 50)

    model = GMMR(reg_type="invalid_method")

    with pytest.raises(ValueError, match="Unknown reg_type"):
        model.fit(X, y)


def test_invalid_shrinkage_target():
    """Test that invalid shrinkage target raises error."""
    X = np.random.randn(50, 5)
    y = np.random.randint(1, 3, 50)

    model = GMMR(reg_type="ledoit_wolf", shrinkage_target="invalid_target")

    with pytest.raises(ValueError, match="Unknown shrinkage_target"):
        model.fit(X, y)


def test_regularization_improves_stability():
    """Test that advanced regularization improves stability on ill-conditioned data."""
    np.random.seed(42)

    # Create ill-conditioned data
    X = np.random.randn(50, 10)
    X[:, -1] = X[:, 0] + 1e-10 * np.random.randn(50)  # Nearly collinear

    y = np.random.randint(1, 3, 50)

    # Empirical should struggle
    model_empirical = GMMR(reg_type="empirical", tau=0.0)
    model_empirical.fit(X, y)
    diag_empirical = model_empirical.get_covariance_diagnostics()

    # Ledoit-Wolf should be more stable
    model_lw = GMMR(reg_type="ledoit_wolf")
    model_lw.fit(X, y)
    diag_lw = model_lw.get_covariance_diagnostics()

    # Ledoit-Wolf should have better condition numbers
    assert np.mean(diag_lw["condition_numbers"]) < np.mean(diag_empirical["condition_numbers"])


# ==================== Serialization Tests for New Features ====================


def test_pickle_partial_fit_model():
    """Test that partial_fit model can be pickled."""
    import pickle

    batches = _make_streaming_data(n_batches=3)

    model = GMMR(tau=0.1)
    for i, (X_batch, y_batch) in enumerate(batches):
        if i == 0:
            model.partial_fit(X_batch, y_batch, classes=np.array([1, 2, 3]))
        else:
            model.partial_fit(X_batch, y_batch)

    # Serialize
    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)

    # Test predictions match
    X_test = np.random.randn(20, 5)
    pred_original = model.predict(X_test)
    pred_loaded = model_loaded.predict(X_test)

    np.testing.assert_array_equal(pred_original, pred_loaded)


def test_pickle_advanced_regularization():
    """Test that models with advanced regularization can be pickled."""
    import pickle

    X = np.random.randn(100, 5)
    y = np.random.randint(1, 4, 100)

    model = GMMR(reg_type="ledoit_wolf", shrinkage_target="diagonal", tau=0.1)
    model.fit(X, y)

    # Serialize
    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)

    # Check parameters preserved
    assert model_loaded.reg_type == "ledoit_wolf"
    assert model_loaded.shrinkage_target == "diagonal"

    # Test predictions match
    X_test = np.random.randn(20, 5)
    pred_original = model.predict(X_test)
    pred_loaded = model_loaded.predict(X_test)

    np.testing.assert_array_equal(pred_original, pred_loaded)


# ==================== Integration Tests ====================


def test_partial_fit_with_predict_proba():
    """Test that partial_fit works with predict_proba."""
    batches = _make_streaming_data(n_batches=3)

    model = GMMR(tau=0.1)

    for i, (X_batch, y_batch) in enumerate(batches):
        if i == 0:
            model.partial_fit(X_batch, y_batch, classes=np.array([1, 2, 3]))
        else:
            model.partial_fit(X_batch, y_batch)

    X_test = np.random.randn(20, 5)
    proba = model.predict_proba(X_test)

    assert proba.shape == (20, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_advanced_reg_with_feature_importance():
    """Test that advanced regularization works with feature importance."""
    X = np.random.randn(100, 5)
    y = np.random.randint(1, 4, 100)

    model = GMMR(reg_type="oas", shrinkage_target="identity")
    model.fit(X, y)

    importance = model.get_feature_importance(method="discriminative")

    assert importance.shape == (5,)
    assert np.allclose(importance.sum(), 1.0)


def test_partial_fit_with_cross_validation():
    """Test that partial_fit model can be used in cross-validation."""
    batches = _make_streaming_data(n_batches=5, batch_size=40)

    # Combine all batches for final evaluation
    X_full = np.vstack([X for X, _ in batches])
    y_full = np.concatenate([y for _, y in batches])

    # Train incrementally
    model = GMMR(tau=0.1)
    for i, (X_batch, y_batch) in enumerate(batches):
        if i == 0:
            model.partial_fit(X_batch, y_batch, classes=np.array([1, 2, 3]))
        else:
            model.partial_fit(X_batch, y_batch)

    # Evaluate
    score = model.score(X_full, y_full)
    assert 0.0 <= score <= 1.0


def test_streaming_workflow():
    """Test realistic streaming data workflow."""
    # Simulate real-time data stream
    n_batches = 10
    batch_size = 20

    model = GMMR(tau=0.1, random_state=42)
    accuracies = []

    for batch_idx in range(n_batches):
        # Generate new batch
        X_batch = []
        y_batch = []
        for c in range(1, 4):
            X_c = np.random.randn(batch_size // 3, 5) + c * 2
            y_c = np.full(batch_size // 3, c)
            X_batch.append(X_c)
            y_batch.append(y_c)

        X_batch = np.vstack(X_batch)
        y_batch = np.concatenate(y_batch)

        # Train
        if batch_idx == 0:
            model.partial_fit(X_batch, y_batch, classes=np.array([1, 2, 3]))
        else:
            model.partial_fit(X_batch, y_batch)

        # Evaluate on current batch
        accuracy = model.score(X_batch, y_batch)
        accuracies.append(accuracy)

    # Accuracy should improve or stay high
    # (May start at 1.0 on easy data, which is fine)
    assert accuracies[-1] >= accuracies[0] * 0.95  # At least maintain performance


# ==================== Backward Compatibility Tests ====================


def test_backward_compatibility_old_models():
    """Test that old pickled models without new attributes load correctly."""
    import pickle

    X = np.random.randn(50, 5)
    y = np.random.randint(1, 3, 50)

    # Create old-style model state (without reg_type, shrinkage_target, _M2)
    model = GMMR(tau=0.1)
    model.fit(X, y)

    state = model.__getstate__()
    del state["reg_type"]
    del state["shrinkage_target"]
    del state["_M2"]

    # Load into new model
    model_new = GMMR()
    model_new.__setstate__(state)

    # Should have defaults
    assert model_new.reg_type == "ridge"
    assert model_new.shrinkage_target == "diagonal"

    # Should still work
    predictions = model_new.predict(X)
    assert predictions.shape == (50,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
