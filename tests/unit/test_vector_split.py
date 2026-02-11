"""Unit tests for ``dzetsaka.infrastructure.geo.vector_split`` helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ogr_module = types.ModuleType("ogr")
sys.modules["ogr"] = ogr_module

osgeo_module = types.ModuleType("osgeo")
setattr(osgeo_module, "ogr", ogr_module)
sys.modules["osgeo"] = osgeo_module
sys.modules["osgeo.ogr"] = ogr_module
ogr_module.Open = lambda path: None


def _reload_vector_split():
    spec = importlib.util.spec_from_file_location(
        "dzetsaka.infrastructure.geo.vector_split",
        Path("src/dzetsaka/infrastructure/geo/vector_split.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dzetsaka.infrastructure.geo.vector_split"] = module
    spec.loader.exec_module(module)
    module.ogr = ogr_module
    return module


class _Feature:
    def __init__(self, label):
        self._label = label

    def GetField(self, _):
        return self._label


class _Layer:
    def __init__(self, features):
        self._features = list(features)

    def __iter__(self):
        return iter(self._features)

    def ResetReading(self):
        pass


class _Dataset:
    def __init__(self, features):
        self._layer = _Layer(features)

    def GetLayer(self):
        return self._layer


def _patch_open(features):
    def _open(_):
        if features is None:
            return None
        return _Dataset(features)

    ogr_module.Open = _open


def test_count_polygons_per_class_returns_empty_when_dataset_missing() -> None:
    _patch_open(None)
    vector_split = _reload_vector_split()
    vector_split.OGR_BACKEND = ogr_module
    assert vector_split.count_polygons_per_class("path", "class") == {}


def test_count_polygons_per_class_aggregates_labels() -> None:
    features = [_Feature(1), _Feature(1), _Feature(2)]
    _patch_open(features)
    vector_split = _reload_vector_split()
    vector_split.OGR_BACKEND = ogr_module
    assert vector_split.count_polygons_per_class("path", "class") == {1: 2, 2: 1}


def test_count_polygons_per_class_skips_empty_labels() -> None:
    features = [_Feature(1), _Feature(None), _Feature("")]
    _patch_open(features)
    vector_split = _reload_vector_split()
    vector_split.OGR_BACKEND = ogr_module
    assert vector_split.count_polygons_per_class("path", "class") == {1: 1}
