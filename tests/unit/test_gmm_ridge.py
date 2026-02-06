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

    assert preds.shape == (x.shape[0],)


def test_bic_no_numpy_deprecation_warning():
    x, y = _make_toy_data()
    model = GMMR()
    model.learn(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        bic = model.BIC(x, y)

    assert np.isfinite(bic)
