"""Unit tests for ``dzetsaka.infrastructure.geo.vector_split`` helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ogr_module = types.ModuleType("ogr")
ogr_module.__spec__ = ModuleSpec("ogr", loader=None)
sys.modules["ogr"] = ogr_module

osgeo_module = types.ModuleType("osgeo")
osgeo_module.ogr = ogr_module
osgeo_module.__spec__ = ModuleSpec("osgeo", loader=None)
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
    """Return empty counts when OGR cannot open the input dataset."""
    _patch_open(None)
    vector_split = _reload_vector_split()
    vector_split.OGR_BACKEND = ogr_module
    assert vector_split.count_polygons_per_class("path", "class") == {}


def test_count_polygons_per_class_aggregates_labels() -> None:
    """Aggregate feature counts by class label."""
    features = [_Feature(1), _Feature(1), _Feature(2)]
    _patch_open(features)
    vector_split = _reload_vector_split()
    vector_split.OGR_BACKEND = ogr_module
    assert vector_split.count_polygons_per_class("path", "class") == {1: 2, 2: 1}


def test_count_polygons_per_class_skips_empty_labels() -> None:
    """Ignore empty and null class labels."""
    features = [_Feature(1), _Feature(None), _Feature("")]
    _patch_open(features)
    vector_split = _reload_vector_split()
    vector_split.OGR_BACKEND = ogr_module
    assert vector_split.count_polygons_per_class("path", "class") == {1: 1}


def test_stratified_split_balances_classes_and_is_deterministic() -> None:
    """Produce reproducible, balanced train/validation splits."""
    vector_split = _reload_vector_split()
    features = list(range(10))
    labels = [1] * 5 + [2] * 5

    train_1, valid_1 = vector_split._stratified_split(features, labels, train_size=0.6, test_size=0.4, random_state=7)
    train_2, valid_2 = vector_split._stratified_split(features, labels, train_size=0.6, test_size=0.4, random_state=7)

    assert train_1 == train_2
    assert valid_1 == valid_2
    assert len(train_1) == 6
    assert len(valid_1) == 4

    train_labels = [labels[i] for i in train_1]
    valid_labels = [labels[i] for i in valid_1]
    assert train_labels.count(1) == 3
    assert train_labels.count(2) == 3
    assert valid_labels.count(1) == 2
    assert valid_labels.count(2) == 2


def test_stratified_split_rejects_singleton_class() -> None:
    """Reject stratified splitting when a class has a single sample."""
    vector_split = _reload_vector_split()
    features = list(range(5))
    labels = [1, 1, 1, 1, 2]

    try:
        vector_split._stratified_split(features, labels, train_size=0.6, test_size=0.4, random_state=0)
        raised = False
    except ValueError as exc:
        raised = True
        assert "least populated class" in str(exc)

    assert raised is True
