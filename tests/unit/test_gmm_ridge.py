"""Unit tests for GMM ridge implementation.

Ensures NumPy deprecation warnings are not raised during predict/BIC.
"""
import warnings

import numpy as np
import pytest

try:
    from scripts.gmm_ridge import GMMR

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="gmm_ridge module not available")

try:
    from sklearn.exceptions import DataConversionWarning
except ImportError:  # pragma: no cover - sklearn optional
    DataConversionWarning = Warning


def _make_toy_data():
    # Two simple clusters in 2D.
    x0 = np.array([[0.0, 0.1], [0.1, -0.1], [-0.1, 0.05]])
    x1 = np.array([[2.0, 2.1], [2.1, 1.9], [1.9, 2.05]])
    x = np.vstack([x0, x1])
    y = np.array([1, 1, 1, 2, 2, 2], dtype=np.uint16)
    return x, y


def test_predict_no_numpy_deprecation_warning():
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    # Convert DeprecationWarning to error for this block.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        preds = model.predict(x)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (x.shape[0],)


def test_predict_confidence_map_returns_probabilities():
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    preds, probabilities = model.predict(x, confidenceMap=True)
    assert preds.shape == (x.shape[0],)
    assert probabilities.shape == (x.shape[0],)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_predict_zero_tau_stable():
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        preds = model.predict(x, tau=0.0)

    assert np.isfinite(preds).all()


def test_cross_validation_runs_without_crash():
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    tau_values = np.array([1e-3, 1.0])
    best_tau, err = model.cross_validation(x, y, tau_values, v=2)
    assert err.shape == tau_values.shape
    assert best_tau in tau_values


def test_bic_no_numpy_deprecation_warning():
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        bic = model.BIC(x, y)

    assert np.isfinite(bic)


def test_learn_accepts_column_vector_labels_without_data_conversion_warning():
    x, y = _make_toy_data()
    y_column = y.reshape(-1, 1)
    model = GMMR()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.learn(x, y_column)

    data_conversion = [w for w in caught if issubclass(w.category, DataConversionWarning)]
    assert not data_conversion
