"""Unit tests for helpers in ``classification_pipeline``."""

from __future__ import annotations

import importlib.util
import sys
import types
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
sys.modules["osgeo"] = osgeo
sys.modules["osgeo.gdal"] = osgeo.gdal
sys.modules["osgeo.ogr"] = osgeo.ogr
sys.modules["gdal"] = osgeo.gdal
sys.modules["ogr"] = osgeo.ogr

STUB_MODULES = [
    "dzetsaka.scripts.accuracy_index",
    "dzetsaka.scripts.function_dataraster",
    "dzetsaka.scripts.progress_bar",
]
for module_name in STUB_MODULES:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

label_encoders_stub = types.ModuleType("dzetsaka.scripts.wrappers.label_encoders")
label_encoders_stub.XGBLabelWrapper = None
label_encoders_stub.LGBLabelWrapper = None
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
