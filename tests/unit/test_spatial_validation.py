"""Tests for the spatial validation helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from dzetsaka.qgis import spatial_validation


class _DummyLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def warning(self, message: str) -> None:
        self.records.append(("warning", message))

    def info(self, message: str) -> None:
        self.records.append(("info", message))


def _install_fake_qgis(monkeypatch, response_name: str) -> SimpleNamespace:
    buttons = SimpleNamespace(Yes=1, No=2)
    response_value = getattr(buttons, response_name)

    class DummyQMessageBox:
        StandardButton = buttons

        @staticmethod
        def question(*args, **kwargs):
            return response_value

    qgis_module = ModuleType("qgis")
    pyqt_module = ModuleType("qgis.PyQt")
    qtwidgets_module = ModuleType("qgis.PyQt.QtWidgets")
    qtwidgets_module.QMessageBox = DummyQMessageBox
    pyqt_module.QtWidgets = qtwidgets_module
    qgis_module.PyQt = pyqt_module

    monkeypatch.setitem(sys.modules, "qgis", qgis_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", pyqt_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtWidgets", qtwidgets_module)

    return buttons


def test_find_classes_with_insufficient_polygons_filters_small_classes() -> None:
    counts = {1: 1, 2: 2, 3: 0}
    result = spatial_validation.find_classes_with_insufficient_polygons(counts, min_polygons=2)
    assert result == {1: 1, 3: 0}


def test_confirm_polygon_group_split_fallbacks_when_needed(monkeypatch) -> None:
    monkeypatch.setattr(
        spatial_validation,
        "count_polygons_per_class",
        lambda *_: {1: 1, 2: 1, 3: 3},
    )
    _install_fake_qgis(monkeypatch, "Yes")
    log = _DummyLog()

    result = spatial_validation.confirm_polygon_group_split(
        None,
        vector_path="path",
        class_field="class",
        log=log,
    )

    assert result == (True, True)
    assert (
        "warning",
        "Spatial CV disabled because some classes lack enough polygons; falling back to random split.",
    ) in log.records


def test_confirm_polygon_group_split_cancels_when_user_says_no(monkeypatch) -> None:
    monkeypatch.setattr(
        spatial_validation,
        "count_polygons_per_class",
        lambda *_: {1: 1},
    )
    _install_fake_qgis(monkeypatch, "No")
    log = _DummyLog()

    result = spatial_validation.confirm_polygon_group_split(
        None,
        vector_path="path",
        class_field="class",
        log=log,
    )

    assert result == (False, False)
    assert ("info", "Spatial CV canceled by user because polygon coverage is insufficient.") in log.records


def test_confirm_polygon_group_split_handles_missing_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        spatial_validation,
        "count_polygons_per_class",
        lambda *_: {},
    )
    log = _DummyLog()

    result = spatial_validation.confirm_polygon_group_split(
        None,
        vector_path="path",
        class_field="class",
        log=log,
    )

    assert result == (True, True)
    assert ("warning", "Unable to inspect polygons for spatial CV; falling back to random split.") in log.records
