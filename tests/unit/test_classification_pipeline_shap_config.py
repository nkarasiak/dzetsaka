"""Tests around SHAP metadata generation for LearnModel."""

import importlib.util
import json
import sys
import types

import numpy as np
import pytest


def _ensure_geospatial_stubs():
    """Inject minimal GDAL/OGR/QGIS modules when unavailable."""
    if importlib.util.find_spec("osgeo") is None:
        gdal_stub = types.ModuleType("gdal")
        for const in (
            "GDT_Byte",
            "GDT_Int16",
            "GDT_UInt16",
            "GDT_Int32",
            "GDT_UInt32",
            "GDT_Float32",
            "GDT_Float64",
            "GDT_CInt16",
            "GDT_CInt32",
            "GDT_CFloat32",
            "GDT_CFloat64",
        ):
            setattr(gdal_stub, const, 0)
        gdal_stub.GA_ReadOnly = 0
        gdal_stub.GA_Update = 0

        def _no_op(*args, **kwargs):
            return None

        gdal_stub.Open = _no_op
        gdal_stub.GetDriverByName = _no_op

        ogr_stub = types.ModuleType("ogr")
        ogr_stub.Open = _no_op

        osgeo_stub = types.ModuleType("osgeo")
        osgeo_stub.gdal = gdal_stub
        osgeo_stub.ogr = ogr_stub

        sys.modules["gdal"] = gdal_stub
        sys.modules["ogr"] = ogr_stub
        sys.modules["osgeo"] = osgeo_stub

    if importlib.util.find_spec("qgis") is None:
        qtcore = types.ModuleType("qgis.PyQt.QtCore")
        class QtShape:
            WaitCursor = 0

        class QtClass:
            WaitCursor = 0
            CursorShape = QtShape

        qtcore.Qt = QtClass

        _thread = object()

        class QThreadStub:
            @staticmethod
            def currentThread():
                return _thread

        qtcore.QThread = QThreadStub

        qtgui = types.ModuleType("qgis.PyQt.QtGui")
        class QCursorStub:
            def __init__(self, *args, **kwargs):
                pass

        qtgui.QCursor = QCursorStub

        qtwidgets = types.ModuleType("qgis.PyQt.QtWidgets")

        class QApplicationStub:
            _instance = None

            def __init__(self):
                self._thread = _thread

            @classmethod
            def instance(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

            def thread(self):
                return self._thread

            @staticmethod
            def setOverrideCursor(*args, **kwargs):
                return None

        class QProgressBarStub:
            def setValue(self, _):
                pass

            def setMaximum(self, _):
                pass

        qtwidgets.QApplication = QApplicationStub
        qtwidgets.QProgressBar = QProgressBarStub

        pyqt_pkg = types.ModuleType("qgis.PyQt")
        pyqt_pkg.QtCore = qtcore
        pyqt_pkg.QtGui = qtgui
        pyqt_pkg.QtWidgets = qtwidgets

        core_package = types.ModuleType("qgis.core")

        class QgisStub:
            Info = 0
            Warning = 1
            Critical = 2
            Success = 3

        core_package.Qgis = QgisStub

        class QgsMessageLogStub:
            @staticmethod
            def logMessage(*args, **kwargs):
                pass

        core_package.QgsMessageLog = QgsMessageLogStub

        utils_package = types.ModuleType("qgis.utils")

        class _MessageWidget:
            def layout(self):
                class _Layout:
                    def addWidget(self, _widget):
                        pass

                return _Layout()

        class _MessageBar:
            def createMessage(self, *_args, **_kwargs):
                return _MessageWidget()

            def pushWidget(self, _widget):
                pass

        utils_package.iface = types.SimpleNamespace(messageBar=lambda: _MessageBar())

        qgis_package = types.ModuleType("qgis")
        qgis_package.PyQt = pyqt_pkg
        qgis_package.core = core_package
        qgis_package.utils = utils_package

        sys.modules["qgis"] = qgis_package
        sys.modules["qgis.PyQt"] = pyqt_pkg
        sys.modules["qgis.PyQt.QtCore"] = qtcore
        sys.modules["qgis.PyQt.QtGui"] = qtgui
        sys.modules["qgis.PyQt.QtWidgets"] = qtwidgets
        sys.modules["qgis.core"] = core_package
        sys.modules["qgis.utils"] = utils_package


_ensure_geospatial_stubs()

from dzetsaka.scripts.classification_pipeline import _extract_shap_settings, _write_report_bundle


def _build_summary_metrics() -> dict:
    """Return a minimal summary metric set required by _write_report_bundle."""
    return {
        "precision_per_class": [1.0],
        "recall_per_class": [1.0],
        "f1_per_class": [1.0],
        "support_per_class": [10],
        "accuracy": 1.0,
        "f1_macro": 1.0,
        "f1_weighted": 1.0,
        "f1_micro": 1.0,
        "overall_accuracy_conf": 1.0,
        "f1_mean_conf": 1.0,
    }


@pytest.mark.unit
def test_extract_shap_settings_defaults():
    enabled, sample_size = _extract_shap_settings({})
    assert enabled is False
    assert sample_size == 1000


@pytest.mark.unit
def test_run_config_reflects_shap_flags(tmp_path):
    shap_enabled, shap_sample_size = _extract_shap_settings(
        {"COMPUTE_SHAP": True, "SHAP_SAMPLE_SIZE": 123}
    )

    config_meta = {
        "classifier_code": "RF",
        "classifier_name": "Random Forest",
        "execution_date": "2026-02-07 09:00:00",
        "split_mode": "RANDOM_SPLIT",
        "split_config": 80,
        "class_field": "Class",
        "vector_path": "vector",
        "raster_path": "raster",
        "optimization_method": "none",
        "best_hyperparameters": {},
        "optuna_stats": None,
        "grid_search_combinations": None,
        "feature_importance": None,
        "shap_config": {"enabled": shap_enabled, "sample_size": shap_sample_size},
        "matrix_path": "",
    }

    report_dir = tmp_path / "bundle"
    _write_report_bundle(
        report_dir=str(report_dir),
        cm=np.array([[1]]),
        class_values=[1],
        class_names=["Class 1"],
        summary_metrics=_build_summary_metrics(),
        config_meta=config_meta,
        y_true=np.array([1]),
        y_pred=np.array([1]),
    )

    run_config_path = report_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    shap_config_output = run_config.get("shap_config", {})
    assert shap_config_output.get("enabled") is True
    assert shap_config_output.get("sample_size") == 123
