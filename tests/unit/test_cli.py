import json
import sys
import types
from importlib import import_module
from importlib.machinery import ModuleSpec

stub_accuracy = types.ModuleType("accuracy_index")
stub_gdal = types.ModuleType("gdal")
stub_ogr = types.ModuleType("ogr")
stub_gdal_array = types.ModuleType("gdal_array")
stub_osgeo = types.ModuleType("osgeo")
stub_osgeo.gdal = stub_gdal
stub_osgeo.ogr = stub_ogr
stub_osgeo.gdal_array = stub_gdal_array
stub_osgeo.__spec__ = ModuleSpec("osgeo", loader=None)
stub_gdal.__spec__ = ModuleSpec("osgeo.gdal", loader=None)
stub_ogr.__spec__ = ModuleSpec("osgeo.ogr", loader=None)
stub_gdal_array.__spec__ = ModuleSpec("osgeo.gdal_array", loader=None)
for const in (
    "GA_ReadOnly",
    "GA_Update",
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
    setattr(stub_gdal, const, 1)


class _DummyDataset:
    def __init__(self, *args, **kwargs):
        pass

    def GetRasterBand(self, *args, **kwargs):
        return None

    def GetGeoTransform(self):
        return (0, 1, 0, 0, 0, -1)

    def GetProjection(self):
        return ""


def _dummy_open(*args, **kwargs):
    return _DummyDataset()


class _DummyDriver:
    def Create(self, *args, **kwargs):
        return _DummyDataset()

    def Delete(self, *args, **kwargs):
        return None


def _dummy_driver_by_name(*args, **kwargs):
    return _DummyDriver()


stub_gdal.Dataset = _DummyDataset
stub_gdal.Open = _dummy_open
stub_gdal.GetDriverByName = _dummy_driver_by_name
sys.modules.setdefault("accuracy_index", stub_accuracy)
sys.modules.setdefault("gdal", stub_gdal)
sys.modules.setdefault("ogr", stub_ogr)
sys.modules.setdefault("osgeo", stub_osgeo)
sys.modules.setdefault("osgeo.gdal", stub_gdal)
sys.modules.setdefault("osgeo.ogr", stub_ogr)
sys.modules.setdefault("osgeo.gdal_array", stub_gdal_array)

qgis = types.ModuleType("qgis")
pyqt = types.ModuleType("qgis.PyQt")
qtcore = types.ModuleType("qgis.PyQt.QtCore")
qtgui = types.ModuleType("qgis.PyQt.QtGui")
qtwidgets = types.ModuleType("qgis.PyQt.QtWidgets")


class _DummyQt:
    class CursorShape:
        WaitCursor = 1

    WaitCursor = 1


class _DummyQCursor:
    def __init__(self, *args, **kwargs):
        pass


class _DummyQProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def setValue(self, value):
        return None

    def setMaximum(self, value):
        return None


class _DummyQApplication:
    @staticmethod
    def setOverrideCursor(cursor):
        return None


class _DummyWidget:
    def layout(self):
        class Layout:
            def addWidget(self_inner, widget):
                return None

        return Layout()


class _DummyMessageBar:
    def createMessage(self, title, text):
        return _DummyWidget()

    def pushWidget(self, widget):
        return None


class _DummyIface:
    def messageBar(self):
        return _DummyMessageBar()


qtcore.Qt = _DummyQt
qtgui.QCursor = _DummyQCursor
qtwidgets.QApplication = _DummyQApplication
qtwidgets.QProgressBar = _DummyQProgressBar
pyqt.QtCore = qtcore
pyqt.QtGui = qtgui
pyqt.QtWidgets = qtwidgets
qgis.PyQt = pyqt
iface_mod = types.ModuleType("qgis.utils")
iface_mod.iface = _DummyIface()
sys.modules.setdefault("qgis", qgis)
sys.modules.setdefault("qgis.PyQt", pyqt)
sys.modules.setdefault("qgis.PyQt.QtCore", qtcore)
sys.modules.setdefault("qgis.PyQt.QtGui", qtgui)
sys.modules.setdefault("qgis.PyQt.QtWidgets", qtwidgets)
sys.modules.setdefault("qgis.utils", iface_mod)

from dzetsaka.cli.main import main


def test_classify_command_calls_use_case(monkeypatch, tmp_path):
    captured = {}

    def fake_run_classification(**kwargs):
        captured.update(kwargs)

    module = import_module("dzetsaka.cli.main")
    monkeypatch.setattr(module, "run_classification", fake_run_classification)
    rc = main(
        [
            "classify",
            "--raster",
            "input.tif",
            "--model",
            "model.pkl",
            "--output",
            "score.tif",
            "--mask",
            "mask.tif",
            "--confidence",
            "conf.tif",
            "--nodata",
            "42",
        ],
    )

    assert rc == 0
    assert captured["raster_path"] == "input.tif"
    assert captured["model_path"] == "model.pkl"
    assert captured["output_path"] == "score.tif"
    assert captured["mask_path"] == "mask.tif"
    assert captured["confidence_map"] == "conf.tif"
    assert captured["nodata"] == 42


def test_train_command_parses_extra_params(monkeypatch, tmp_path):
    captured = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)

    module = import_module("dzetsaka.cli.main")
    monkeypatch.setattr(module, "run_training", fake_run_training)
    extra_file = tmp_path / "extras.json"
    extra_file.write_text(json.dumps({"USE_OPTUNA": True}))

    rc = main(
        [
            "train",
            "--raster",
            "train.tif",
            "--vector",
            "train.shp",
            "--model",
            "model.bin",
            "--split-config",
            "SLOO",
            "--random-seed",
            "7",
            "--matrix-path",
            "matrix.csv",
            "--classifier",
            "RF",
            "--extra",
            f"@{extra_file}",
        ],
    )

    assert rc == 0
    assert captured["split_config"] == "SLOO"
    assert captured["extra_params"] == {"USE_OPTUNA": True}
    assert captured["random_seed"] == 7
    assert captured["matrix_path"] == "matrix.csv"
    assert captured["classifier"] == "RF"
