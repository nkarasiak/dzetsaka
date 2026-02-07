"""Integration smoke tests that exercise the sample dataset under `data/sample`."""

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("osgeo")  # SHAP workflows rely on GDAL/OGR for rasterization
pytest.importorskip("shap")  # The ModelExplainer requires SHAP
pytest.importorskip("sklearn")  # Training requires scikit-learn

from sklearn.ensemble import RandomForestClassifier

from scripts.classification_pipeline import rasterize
from scripts.explainability.shap_explainer import ModelExplainer
from scripts.function_dataraster import get_samples_from_roi

pytestmark = pytest.mark.integration

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
RASTER_PATH = DATA_DIR / "map.tif"
VECTOR_PATH = next(DATA_DIR.glob("train.geoparquet*.parquet"), None)
CLASS_FIELD = "Class"
VALID_CLASS_VALUES = set(range(1, 6))


def _skip_if_missing_assets():
    """Skip tests early when the sample assets are not tracked."""
    if not RASTER_PATH.exists() or VECTOR_PATH is None or not VECTOR_PATH.exists():
        pytest.skip("Sample raster/vector assets are missing")


@pytest.fixture(scope="module")
def sample_training_data():
    """Prepare rasterized ROI and return feature/label arrays."""
    _skip_if_missing_assets()
    roi_path = None
    try:
        roi_path = rasterize(
            raster_path=str(RASTER_PATH),
            shapefile_path=str(VECTOR_PATH),
            class_field=CLASS_FIELD,
        )
    except RuntimeError as exc:
        pytest.skip(f"Sample rasterization failed: {exc}")

    try:
        samples = get_samples_from_roi(str(RASTER_PATH), roi_path)
        X, Y = samples
        yield X, Y
    finally:
        if roi_path and os.path.exists(roi_path):
            with contextlib.suppress(OSError):
                os.remove(roi_path)


def test_sample_data_loading_shapes(sample_training_data):
    """Ensure the sample data loads and returns sane shapes/labels."""
    X, Y = sample_training_data
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] > 1
    assert np.all(np.isfinite(X))

    unique_classes = set(np.unique(Y).tolist())
    assert unique_classes.issubset(VALID_CLASS_VALUES)


def test_sample_data_shap_pipeline(sample_training_data):
    """Train a small RF and run ModelExplainer on the real data subset."""
    X, Y = sample_training_data
    rng = np.random.default_rng(0)
    sample_size = min(120, len(Y))
    indices = rng.choice(len(Y), size=sample_size, replace=False)
    X_train = X[indices]
    y_train = Y[indices]

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)

    explainer = ModelExplainer(
        model=clf,
        feature_names=[f"Band_{i + 1}" for i in range(X.shape[1])],
        background_data=X_train[: min(20, len(X_train))],
    )
    shap_sample = X_train[: min(40, len(X_train))]
    importance = explainer.get_feature_importance(X_sample=shap_sample, aggregate_method="mean_abs")

    assert len(importance) == X.shape[1]
    total = sum(importance.values())
    assert abs(total - 1.0) < 1e-2
