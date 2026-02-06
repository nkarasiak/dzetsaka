"""Unit tests for the Algorithm Comparison Panel data builder.

All tests exercise ``build_comparison_data`` and the static metadata
without requiring Qt or QGIS.
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Load comparison_panel directly, bypassing ui/__init__.py which pulls qgis.PyQt.
# We first ensure guided_workflow_widget is available as a sibling stub (comparison_panel
# imports check_dependency_availability from it).
# ---------------------------------------------------------------------------

PANEL_MODULE_AVAILABLE = False

try:
    # Track which keys we inject so we can clean them up afterwards and avoid
    # polluting sys.modules for tests that run later in the same session.
    _STUB_KEYS = (
        "qgis", "qgis.PyQt", "qgis.PyQt.QtCore", "qgis.PyQt.QtWidgets", "qgis.PyQt.QtGui",
        "ui", "ui.guided_workflow_widget", "ui.comparison_panel",
    )
    _inserted_keys = [k for k in _STUB_KEYS if k not in sys.modules]

    # --- Minimal qgis.PyQt stub tree ---
    class _FakeSignal:
        """Stub for pyqtSignal."""

        def __init__(self, *a, **kw):
            pass

    class _FakeWidget:
        """Stub QWidget-like class."""

        def __init__(self, *a, **kw):
            pass

    _qgis = type(sys)("qgis")
    _pyqt = type(sys)("qgis.PyQt")
    _pyqt_core = type(sys)("qgis.PyQt.QtCore")
    _pyqt_core.pyqtSignal = _FakeSignal
    _pyqt_core.Qt = type("Qt", (), {"ItemIsEditable": 0})()
    _pyqt_widgets = type(sys)("qgis.PyQt.QtWidgets")
    for _cls_name in (
        "QCheckBox", "QComboBox", "QDialog", "QDialogButtonBox",
        "QFileDialog", "QGroupBox", "QHBoxLayout", "QLabel", "QLineEdit",
        "QMessageBox", "QPushButton", "QSpinBox", "QTableWidget", "QTableWidgetItem",
        "QTextEdit", "QVBoxLayout", "QWidget", "QWizard", "QWizardPage",
    ):
        setattr(_pyqt_widgets, _cls_name, _FakeWidget)
    _pyqt_gui = type(sys)("qgis.PyQt.QtGui")
    _pyqt_gui.QColor = _FakeWidget
    _pyqt.QtCore = _pyqt_core
    _pyqt.QtWidgets = _pyqt_widgets
    _pyqt.QtGui = _pyqt_gui
    _qgis.PyQt = _pyqt
    sys.modules["qgis"] = _qgis
    sys.modules["qgis.PyQt"] = _pyqt
    sys.modules["qgis.PyQt.QtCore"] = _pyqt_core
    sys.modules["qgis.PyQt.QtWidgets"] = _pyqt_widgets
    sys.modules["qgis.PyQt.QtGui"] = _pyqt_gui

    _UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ui"))

    # --- Load guided_workflow_widget with __package__ = 'ui' so relative imports work ---
    _workflow_spec = importlib.util.spec_from_file_location(
        "ui.guided_workflow_widget", os.path.join(_UI_DIR, "guided_workflow_widget.py")
    )
    _workflow_mod = importlib.util.module_from_spec(_workflow_spec)
    _workflow_mod.__package__ = "ui"
    sys.modules["ui.guided_workflow_widget"] = _workflow_mod
    _workflow_spec.loader.exec_module(_workflow_mod)

    # --- Create the fake 'ui' package so relative imports in comparison_panel resolve ---
    _fake_ui_pkg = type(sys)("ui")
    _fake_ui_pkg.__path__ = [_UI_DIR]
    _fake_ui_pkg.__package__ = "ui"
    _fake_ui_pkg.guided_workflow_widget = _workflow_mod
    sys.modules["ui"] = _fake_ui_pkg

    # --- Load comparison_panel ---
    _panel_spec = importlib.util.spec_from_file_location(
        "ui.comparison_panel", os.path.join(_UI_DIR, "comparison_panel.py")
    )
    _panel_mod = importlib.util.module_from_spec(_panel_spec)
    _panel_mod.__package__ = "ui"
    sys.modules["ui.comparison_panel"] = _panel_mod
    _panel_spec.loader.exec_module(_panel_mod)

    _ALGO_TABLE_DATA = _panel_mod._ALGO_TABLE_DATA
    _FEATURE_SUPPORT = _panel_mod._FEATURE_SUPPORT
    build_comparison_data = _panel_mod.build_comparison_data

    PANEL_MODULE_AVAILABLE = True

    # Clean up: remove stubs we injected so they don't leak to other tests
    for _k in _inserted_keys:
        sys.modules.pop(_k, None)
    del _workflow_mod, _panel_mod, _workflow_spec, _panel_spec, _fake_ui_pkg
except Exception:
    pass

pytestmark = pytest.mark.skipif(not PANEL_MODULE_AVAILABLE, reason="comparison_panel helpers not importable")


# ===========================================================================
# Static metadata sanity checks
# ===========================================================================


class TestAlgoTableDataStructure:
    """Verify the static _ALGO_TABLE_DATA list is well-formed."""

    def test_eleven_classifiers(self):
        """Exactly 12 classifiers in the table data."""
        assert len(_ALGO_TABLE_DATA) == 12

    def test_codes_unique(self):
        """All classifier codes are unique."""
        codes = [row[0] for row in _ALGO_TABLE_DATA]
        assert len(codes) == len(set(codes))

    def test_expected_codes_present(self):
        """All expected codes are present."""
        codes = {row[0] for row in _ALGO_TABLE_DATA}
        expected = {"GMM", "RF", "SVM", "KNN", "XGB", "LGB", "CB", "ET", "GBC", "LR", "NB", "MLP"}
        assert codes == expected

    def test_speed_values_valid(self):
        """Speed column only contains Fast, Medium or Slow."""
        valid_speeds = {"Fast", "Medium", "Slow"}
        for row in _ALGO_TABLE_DATA:
            assert row[3] in valid_speeds, f"Unexpected speed value: {row[3]}"

    def test_type_values_non_empty(self):
        """Type column is never empty."""
        for row in _ALGO_TABLE_DATA:
            assert len(row[2]) > 0


class TestFeatureSupportData:
    """Verify the _FEATURE_SUPPORT mapping."""

    def test_all_classifiers_have_feature_entries(self):
        """Every classifier code has a feature-support entry."""
        codes = {row[0] for row in _ALGO_TABLE_DATA}
        assert set(_FEATURE_SUPPORT.keys()) == codes

    def test_gmm_has_no_features(self):
        """GMM does not support optuna, shap, smote or class_weights."""
        gmm = _FEATURE_SUPPORT["GMM"]
        assert gmm["optuna"] is False
        assert gmm["shap"] is False
        assert gmm["smote"] is False
        assert gmm["class_weights"] is False

    def test_rf_supports_all_features(self):
        """Random Forest supports all four optional features."""
        rf = _FEATURE_SUPPORT["RF"]
        assert rf["optuna"] is True
        assert rf["shap"] is True
        assert rf["smote"] is True
        assert rf["class_weights"] is True

    def test_feature_keys_consistent(self):
        """Every entry has the same four keys."""
        expected_keys = {"optuna", "shap", "smote", "class_weights"}
        for code, features in _FEATURE_SUPPORT.items():
            assert set(features.keys()) == expected_keys, f"{code} has unexpected feature keys"


# ===========================================================================
# build_comparison_data
# ===========================================================================


class TestBuildComparisonData:
    """Tests for build_comparison_data()."""

    def _all_deps(self):
        """Return deps dict with everything available."""
        return {
            "sklearn": True,
            "xgboost": True,
            "lightgbm": True,
            "catboost": True,
            "optuna": True,
            "shap": True,
            "imblearn": True,
        }

    def _no_deps(self):
        """Return deps dict with nothing available."""
        return dict.fromkeys(
            ("sklearn", "xgboost", "lightgbm", "catboost", "optuna", "shap", "imblearn"), False
        )

    def test_returns_eleven_rows(self):
        """Always returns 12 rows."""
        rows = build_comparison_data(self._all_deps())
        assert len(rows) == 12

    def test_all_available_when_all_deps(self):
        """When all deps are present every row is marked available."""
        rows = build_comparison_data(self._all_deps())
        for row in rows:
            assert row[4] is True, f"{row[1]} should be available"

    def test_only_gmm_available_when_no_deps(self):
        """Only GMM is available when no packages are installed."""
        rows = build_comparison_data(self._no_deps())
        available_names = [row[1] for row in rows if row[4] is True]
        assert available_names == ["Gaussian Mixture Model"]

    def test_sklearn_classifiers_available_with_sklearn(self):
        """RF, SVM, KNN, ET, GBC, LR, NB, MLP are available with only sklearn."""
        deps = self._no_deps()
        deps["sklearn"] = True
        rows = build_comparison_data(deps)
        available = {row[0] for row in rows if row[4] is True}
        expected_available = {"GMM", "RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"}
        assert available == expected_available

    def test_xgb_available_with_xgboost(self):
        """XGB becomes available when xgboost dep is True."""
        deps = self._no_deps()
        deps["xgboost"] = True
        rows = build_comparison_data(deps)
        xgb_row = next(r for r in rows if r[0] == "XGB")
        assert xgb_row[4] is True

    def test_lgb_available_with_lightgbm(self):
        """LGB becomes available when lightgbm dep is True."""
        deps = self._no_deps()
        deps["lightgbm"] = True
        rows = build_comparison_data(deps)
        lgb_row = next(r for r in rows if r[0] == "LGB")
        assert lgb_row[4] is True

    def test_cb_available_with_catboost(self):
        """CB becomes available when catboost dep is True."""
        deps = self._no_deps()
        deps["catboost"] = True
        rows = build_comparison_data(deps)
        cb_row = next(r for r in rows if r[0] == "CB")
        assert cb_row[4] is True

    def test_feature_columns_are_yes_no_strings(self):
        """Optuna, SHAP, SMOTE, class_weights columns are 'Yes' or 'No'."""
        rows = build_comparison_data(self._all_deps())
        for row in rows:
            for col_idx in (5, 6, 7, 8):
                assert row[col_idx] in ("Yes", "No"), f"Row {row[1]} col {col_idx}: {row[col_idx]}"

    def test_row_tuple_length(self):
        """Each row tuple has exactly 9 elements."""
        rows = build_comparison_data(self._all_deps())
        for row in rows:
            assert len(row) == 9, f"Row {row[1]} has {len(row)} elements, expected 9"

    def test_speed_labels_correct_for_known_algorithms(self):
        """Spot-check known speed labels."""
        rows = build_comparison_data(self._all_deps())
        speed_map = {row[0]: row[3] for row in rows}
        assert speed_map["GMM"] == "Fast"
        assert speed_map["SVM"] == "Medium"
        assert speed_map["MLP"] == "Slow"
        assert speed_map["LGB"] == "Fast"
        assert speed_map["XGB"] == "Medium"

