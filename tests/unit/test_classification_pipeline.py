"""Unit tests for helpers in ``classification_pipeline``."""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub native geospatial dependencies so the module imports cleanly.
osgeo = types.ModuleType("osgeo")
osgeo.gdal = types.ModuleType("osgeo.gdal")
osgeo.ogr = types.ModuleType("osgeo.ogr")
osgeo.__spec__ = ModuleSpec("osgeo", loader=None)
osgeo.gdal.__spec__ = ModuleSpec("osgeo.gdal", loader=None)
osgeo.ogr.__spec__ = ModuleSpec("osgeo.ogr", loader=None)
sys.modules["osgeo"] = osgeo
sys.modules["osgeo.gdal"] = osgeo.gdal
sys.modules["osgeo.ogr"] = osgeo.ogr
sys.modules["gdal"] = osgeo.gdal
sys.modules["ogr"] = osgeo.ogr

STUB_MODULES = [
    "dzetsaka.scripts.function_dataraster",
    "dzetsaka.scripts.progress_bar",
]
for module_name in STUB_MODULES:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

# accuracy_index needs a working ConfusionMatrix for end-to-end tests
if "dzetsaka.scripts.accuracy_index" not in sys.modules:
    _ai_stub = types.ModuleType("dzetsaka.scripts.accuracy_index")

    class _StubConfusionMatrix:
        def __init__(self):
            self.confusion_matrix = None
            self.OA = None
            self.Kappa = None
            self.F1mean = None

        def compute_confusion_matrix(self, yp, yt):
            from sklearn.metrics import confusion_matrix as _cm, accuracy_score, f1_score, cohen_kappa_score

            self.confusion_matrix = _cm(yt, yp)
            self.OA = accuracy_score(yt, yp)
            self.Kappa = cohen_kappa_score(yt, yp)
            self.F1mean = f1_score(yt, yp, average="macro")

    _ai_stub.ConfusionMatrix = _StubConfusionMatrix
    sys.modules["dzetsaka.scripts.accuracy_index"] = _ai_stub

label_encoders_stub = types.ModuleType("dzetsaka.scripts.wrappers.label_encoders")
label_encoders_stub.XGBLabelWrapper = None
label_encoders_stub.CBClassifierWrapper = None
label_encoders_stub.SKLEARN_AVAILABLE = False
sys.modules["dzetsaka.scripts.wrappers.label_encoders"] = label_encoders_stub

classification_pipeline_spec = importlib.util.spec_from_file_location(
    "dzetsaka.scripts.classification_pipeline",
    ROOT / "scripts" / "classification_pipeline.py",
)
classification_pipeline = importlib.util.module_from_spec(classification_pipeline_spec)
sys.modules["dzetsaka.scripts.classification_pipeline"] = classification_pipeline
classification_pipeline_spec.loader.exec_module(classification_pipeline)


def _new_learn_model() -> classification_pipeline.LearnModel:
    """Create a lightweight LearnModel instance without running ``__init__``."""
    return classification_pipeline.LearnModel.__new__(classification_pipeline.LearnModel)


def test_classes_with_too_few_polygons_without_groups_returns_empty() -> None:
    helper = _new_learn_model()
    assert helper._classes_with_too_few_polygons(np.array([], dtype=int), None, min_polygons=2) == []


def test_classes_with_too_few_polygons_reports_classes_missing_groups() -> None:
    helper = _new_learn_model()
    y = np.array([1, 1, 2, 2, 2, 3])
    polygon_groups = np.array([10, 10, 20, 21, 21, 30])
    result = helper._classes_with_too_few_polygons(y, polygon_groups, min_polygons=2)
    assert result == [(1, 1), (3, 1)]


def test_classes_with_sufficient_polygons_not_reported() -> None:
    helper = _new_learn_model()
    y = np.array([1, 1, 2, 2, 2])
    polygon_groups = np.array([10, 11, 20, 21, 22])
    result = helper._classes_with_too_few_polygons(y, polygon_groups, min_polygons=2)
    assert result == []


def test_polygon_group_validator_raises_when_insufficient() -> None:
    helper = _new_learn_model()
    y = np.array([1, 1, 2, 2, 3])
    polygon_groups = np.array([10, 10, 20, 20, 30])
    with pytest.raises(classification_pipeline.PolygonCoverageInsufficientError) as exc:
        helper._ensure_polygon_group_counts(y, polygon_groups, min_polygons=2)
    assert "Classes with too few polygons" in str(exc.value)


@pytest.mark.sklearn
class TestLearnModelWithVectorTestPath:
    """Regression tests for issue #52: UnboundLocalError on ``xt`` when a
    separate validation vector (``vector_test_path``) is used."""

    @staticmethod
    def _make_data(n_per_class_train=20, n_per_class_test=7, n_features=4, n_classes=3, seed=0):
        rng = np.random.RandomState(seed)
        n_train = n_per_class_train * n_classes
        n_test = n_per_class_test * n_classes
        X_train = rng.randn(n_train, n_features)
        Y_train = np.repeat(np.arange(1, n_classes + 1), n_per_class_train).astype(float)
        X_test = rng.randn(n_test, n_features)
        Y_test = np.repeat(np.arange(1, n_classes + 1), n_per_class_test).astype(float)
        return X_train, Y_train, X_test, Y_test

    def test_vector_test_path_does_not_raise_unbound_xt(self):
        """When vector_test_path is set, xt/yt must be derived from
        self._test_data so the accuracy assessment block can run."""
        from unittest.mock import patch

        X_train, Y_train, X_test, Y_test = self._make_data()

        def fake_load(self_inner, raster_path, vector_path, class_field,
                      split_config, extra_param, feedback):
            # Simulate what happens when split_config is a .shp path:
            # training data returned normally, test data stored on self.
            self_inner._test_data = (X_test, Y_test)
            vector_test_path = "/tmp/fake_validation.shp"
            return (X_train, Y_train, None, None, None, None, vector_test_path)

        with patch.object(
            classification_pipeline.LearnModel,
            "_load_and_prepare_data",
            fake_load,
        ):
            # split_config is a .shp path string → split_value is a string,
            # so the percentage-based split branch is skipped.
            lm = classification_pipeline.LearnModel(
                raster_path="/tmp/fake.tif",
                vector_path="/tmp/fake.shp",
                class_field="Class",
                split_config="/tmp/fake_validation.shp",
                classifier="GMM",
            )

        # The model should have completed training and accuracy assessment
        assert hasattr(lm, "model")
        assert lm.model is not None

    def test_vector_test_path_with_sklearn_classifier(self):
        """Same scenario but with an sklearn classifier (RF) to cover the
        non-GMM code path."""
        from unittest.mock import patch

        sklearn = pytest.importorskip("sklearn")

        X_train, Y_train, X_test, Y_test = self._make_data()

        def fake_load(self_inner, raster_path, vector_path, class_field,
                      split_config, extra_param, feedback):
            self_inner._test_data = (X_test, Y_test)
            return (X_train, Y_train, None, None, None, None,
                    "/tmp/fake_validation.shp")

        with patch.object(
            classification_pipeline.LearnModel,
            "_load_and_prepare_data",
            fake_load,
        ):
            lm = classification_pipeline.LearnModel(
                raster_path="/tmp/fake.tif",
                vector_path="/tmp/fake.shp",
                class_field="Class",
                split_config="/tmp/fake_validation.shp",
                classifier="RF",
                extra_param={"OPTIMIZE_HYPERPARAMETERS": False},
            )

        assert hasattr(lm, "model")
        assert lm.model is not None
