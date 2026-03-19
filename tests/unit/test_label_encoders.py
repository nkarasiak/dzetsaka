"""Unit tests for label encoding wrappers.

Tests the XGBLabelWrapper and CBClassifierWrapper classes that handle
transparent label encoding/decoding for sparse class labels.
"""

import numpy as np
import pytest

# Try to import the module under test
try:
    from scripts.wrappers.label_encoders import (
        CBClassifierWrapper,
        SKLEARN_AVAILABLE,
        XGBLabelWrapper,
    )

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    SKLEARN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="Label encoders module not available")


def _make_linearly_separable(labels, n_per_class=30, n_features=4, seed=42):
    """Create linearly separable data for given class labels.

    Each class is centered at a different offset along the first feature axis
    to ensure the data is easily separable.
    """
    rng = np.random.RandomState(seed)
    X_parts = []
    y_parts = []
    for i, label in enumerate(labels):
        X_part = rng.randn(n_per_class, n_features) + i * 5
        X_parts.append(X_part)
        y_parts.append(np.full(n_per_class, label))
    return np.vstack(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# XGBLabelWrapper tests
# ---------------------------------------------------------------------------

XGB_AVAILABLE = False
try:
    import xgboost  # noqa: F401

    XGB_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not (SKLEARN_AVAILABLE and XGB_AVAILABLE), reason="sklearn + xgboost required")
class TestXGBLabelWrapperSparseLabels:
    """Test XGBLabelWrapper with sparse (non-continuous) labels."""

    @pytest.fixture()
    def sparse_model(self):
        """Fitted XGBLabelWrapper with sparse labels (0, 1, 3, 5)."""
        labels = [0, 1, 3, 5]
        X, y = _make_linearly_separable(labels)
        model = XGBLabelWrapper(n_estimators=20, max_depth=3, verbosity=0)
        model.fit(X, y)
        return model, X, y, labels

    def test_predict_returns_original_labels(self, sparse_model):
        """Test that predict returns original sparse labels, not encoded ones."""
        model, X, y, labels = sparse_model
        preds = model.predict(X)
        assert set(preds).issubset(set(labels))

    def test_predict_proba_shape(self, sparse_model):
        """Test that predict_proba returns shape (n_samples, n_classes)."""
        model, X, y, labels = sparse_model
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), len(labels))

    def test_classes_property(self, sparse_model):
        """Test that classes_ property returns original label values."""
        model, X, y, labels = sparse_model
        np.testing.assert_array_equal(np.sort(model.classes_), np.sort(labels))

    def test_predict_accuracy_on_training_data(self, sparse_model):
        """Test that the model achieves high accuracy on separable training data."""
        model, X, y, labels = sparse_model
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.9


@pytest.mark.skipif(not (SKLEARN_AVAILABLE and XGB_AVAILABLE), reason="sklearn + xgboost required")
class TestXGBLabelWrapperParams:
    """Test XGBLabelWrapper get_params / set_params."""

    def test_get_params_round_trip(self):
        """Test that get_params returns the params passed at init."""
        params = {"n_estimators": 50, "max_depth": 4, "verbosity": 0}
        model = XGBLabelWrapper(**params)
        retrieved = model.get_params()
        assert retrieved == params

    def test_set_params_updates(self):
        """Test that set_params updates the stored parameters."""
        model = XGBLabelWrapper(n_estimators=10, verbosity=0)
        model.set_params(n_estimators=200)
        assert model.get_params()["n_estimators"] == 200


@pytest.mark.skipif(not (SKLEARN_AVAILABLE and XGB_AVAILABLE), reason="sklearn + xgboost required")
class TestXGBLabelWrapperContinuousLabels:
    """Test XGBLabelWrapper with already-continuous labels."""

    def test_continuous_labels_work(self):
        """Test that labels (0, 1, 2) work without issues."""
        X, y = _make_linearly_separable([0, 1, 2])
        model = XGBLabelWrapper(n_estimators=20, max_depth=3, verbosity=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})
        assert np.mean(preds == y) > 0.9


# ---------------------------------------------------------------------------
# CBClassifierWrapper tests
# ---------------------------------------------------------------------------

CB_AVAILABLE = False
try:
    import catboost  # noqa: F401

    CB_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not (SKLEARN_AVAILABLE and CB_AVAILABLE), reason="sklearn + catboost required")
class TestCBClassifierWrapperSparseLabels:
    """Test CBClassifierWrapper with sparse (non-continuous) labels."""

    @pytest.fixture()
    def sparse_model(self):
        """Fitted CBClassifierWrapper with sparse labels (0, 1, 3, 5)."""
        labels = [0, 1, 3, 5]
        X, y = _make_linearly_separable(labels)
        model = CBClassifierWrapper(iterations=50, depth=3, verbose=0)
        model.fit(X, y)
        return model, X, y, labels

    def test_predict_returns_original_labels(self, sparse_model):
        """Test that predict returns original sparse labels, not encoded ones."""
        model, X, y, labels = sparse_model
        preds = model.predict(X)
        assert set(preds).issubset(set(labels))

    def test_predict_proba_shape(self, sparse_model):
        """Test that predict_proba returns shape (n_samples, n_classes)."""
        model, X, y, labels = sparse_model
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), len(labels))

    def test_classes_property(self, sparse_model):
        """Test that classes_ property returns original label values."""
        model, X, y, labels = sparse_model
        np.testing.assert_array_equal(np.sort(model.classes_), np.sort(labels))

    def test_predict_accuracy_on_training_data(self, sparse_model):
        """Test that the model achieves high accuracy on separable training data."""
        model, X, y, labels = sparse_model
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.9


@pytest.mark.skipif(not (SKLEARN_AVAILABLE and CB_AVAILABLE), reason="sklearn + catboost required")
class TestCBClassifierWrapperParams:
    """Test CBClassifierWrapper get_params / set_params."""

    def test_get_params_round_trip(self):
        """Test that get_params returns the params passed at init."""
        params = {"iterations": 100, "depth": 4, "verbose": 0}
        model = CBClassifierWrapper(**params)
        retrieved = model.get_params()
        assert retrieved == params

    def test_set_params_updates(self):
        """Test that set_params updates the stored parameters."""
        model = CBClassifierWrapper(iterations=10, verbose=0)
        model.set_params(iterations=200)
        assert model.get_params()["iterations"] == 200


@pytest.mark.skipif(not (SKLEARN_AVAILABLE and CB_AVAILABLE), reason="sklearn + catboost required")
class TestCBClassifierWrapperContinuousLabels:
    """Test CBClassifierWrapper with already-continuous labels."""

    def test_continuous_labels_work(self):
        """Test that labels (0, 1, 2) work without issues."""
        X, y = _make_linearly_separable([0, 1, 2])
        model = CBClassifierWrapper(iterations=50, depth=3, verbose=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})
        assert np.mean(preds == y) > 0.9
