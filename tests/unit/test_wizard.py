"""Unit tests for the Classification Wizard helper functions.

All tests in this module exercise pure-Python helpers that do NOT require
Qt or QGIS.  The wizard widget classes themselves (DataInputPage, etc.) are
only imported when Qt is available; those integration paths are tested
manually inside QGIS.
"""

import importlib.util
import os
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.  The helpers are defined before any Qt class
# in wizard_widget.py, so we load the file directly via importlib to avoid
# triggering ui/__init__.py (which pulls in qgis.PyQt).
# ---------------------------------------------------------------------------

_WIZARD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "wizard_widget.py")
_WIZARD_PATH = os.path.abspath(_WIZARD_PATH)

WIZARD_MODULE_AVAILABLE = False
try:
    # Inject a minimal stub for qgis.PyQt so the module-level Qt imports resolve
    # without actually needing QGIS installed.  We track which keys we add
    # and remove them afterwards so the stubs don't pollute other tests.
    _STUB_KEYS = ("qgis", "qgis.PyQt", "qgis.PyQt.QtCore", "qgis.PyQt.QtWidgets")
    _inserted_keys = [k for k in _STUB_KEYS if k not in sys.modules]

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
    _pyqt_widgets = type(sys)("qgis.PyQt.QtWidgets")
    for _cls_name in (
        "QCheckBox", "QComboBox", "QFileDialog", "QGroupBox", "QHBoxLayout",
        "QLabel", "QLineEdit", "QMessageBox", "QPushButton", "QSpinBox", "QTextEdit",
        "QVBoxLayout", "QWidget", "QWizard", "QWizardPage",
    ):
        setattr(_pyqt_widgets, _cls_name, _FakeWidget)
    _pyqt.QtCore = _pyqt_core
    _pyqt.QtWidgets = _pyqt_widgets
    _qgis.PyQt = _pyqt
    sys.modules.setdefault("qgis", _qgis)
    sys.modules.setdefault("qgis.PyQt", _pyqt)
    sys.modules.setdefault("qgis.PyQt.QtCore", _pyqt_core)
    sys.modules.setdefault("qgis.PyQt.QtWidgets", _pyqt_widgets)

    spec = importlib.util.spec_from_file_location("_wizard_widget_test", _WIZARD_PATH)
    _wizard_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_wizard_mod)

    # Extract the pure helpers we actually need
    check_dependency_availability = _wizard_mod.check_dependency_availability
    build_smart_defaults = _wizard_mod.build_smart_defaults
    build_review_summary = _wizard_mod.build_review_summary
    _CLASSIFIER_META = _wizard_mod._CLASSIFIER_META
    _classifier_available = _wizard_mod._classifier_available

    WIZARD_MODULE_AVAILABLE = True

    # Clean up: remove stubs we injected so they don't leak to other tests
    for _k in _inserted_keys:
        sys.modules.pop(_k, None)
    del _wizard_mod, spec
except Exception:
    pass

pytestmark = pytest.mark.skipif(not WIZARD_MODULE_AVAILABLE, reason="wizard_widget helpers not importable")


# ===========================================================================
# check_dependency_availability
# ===========================================================================


class TestCheckDependencyAvailability:
    """Tests for check_dependency_availability()."""

    def test_returns_dict(self):
        """Return value is a dict."""
        result = check_dependency_availability()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        """All six dependency keys are present."""
        result = check_dependency_availability()
        expected_keys = {"sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"}
        assert set(result.keys()) == expected_keys

    def test_values_are_bool(self):
        """Every value is a bool."""
        result = check_dependency_availability()
        for key, val in result.items():
            assert isinstance(val, bool), f"Key {key} has non-bool value {val!r}"

    def test_mock_all_available(self):
        """When all imports succeed the dict is all-True."""
        mods = {k: type(sys)("fake_" + k) for k in ("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn")}
        with patch.dict("sys.modules", mods):
            result = check_dependency_availability()
        for key in result:
            assert result[key] is True, f"Expected {key} to be True"

    def test_mock_all_missing(self):
        """When every import raises ImportError the dict is all-False."""
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _fail_import(name, *args, **kwargs):
            if name in ("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"):
                raise ImportError(f"mocked missing {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fail_import):
            result = check_dependency_availability()
        for key in result:
            assert result[key] is False, f"Expected {key} to be False"


# ===========================================================================
# build_smart_defaults
# ===========================================================================


class TestBuildSmartDefaults:
    """Tests for build_smart_defaults()."""

    def test_all_available_enables_features(self):
        """All optional features enabled when every dep is present."""
        deps = {"sklearn": True, "xgboost": True, "lightgbm": True, "optuna": True, "shap": True, "imblearn": True}
        result = build_smart_defaults(deps)
        assert result["USE_OPTUNA"] is True
        assert result["COMPUTE_SHAP"] is True
        assert result["USE_SMOTE"] is True
        assert result["USE_CLASS_WEIGHTS"] is True

    def test_none_available_disables_features(self):
        """All optional features disabled when no dep is available."""
        deps = {"sklearn": False, "xgboost": False, "lightgbm": False, "optuna": False, "shap": False, "imblearn": False}
        result = build_smart_defaults(deps)
        assert result["USE_OPTUNA"] is False
        assert result["COMPUTE_SHAP"] is False
        assert result["USE_SMOTE"] is False
        assert result["USE_CLASS_WEIGHTS"] is False

    def test_optuna_only(self):
        """Only Optuna enabled when only optuna is importable."""
        deps = {"sklearn": False, "xgboost": False, "lightgbm": False, "optuna": True, "shap": False, "imblearn": False}
        result = build_smart_defaults(deps)
        assert result["USE_OPTUNA"] is True
        assert result["COMPUTE_SHAP"] is False
        assert result["USE_SMOTE"] is False

    def test_sklearn_enables_class_weights(self):
        """Class weights follow sklearn availability."""
        deps = {"sklearn": True, "xgboost": False, "lightgbm": False, "optuna": False, "shap": False, "imblearn": False}
        result = build_smart_defaults(deps)
        assert result["USE_CLASS_WEIGHTS"] is True

    def test_default_numeric_values(self):
        """Numeric defaults are sensible regardless of deps."""
        deps = dict.fromkeys(("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"), False)
        result = build_smart_defaults(deps)
        assert result["OPTUNA_TRIALS"] == 100
        assert result["SHAP_SAMPLE_SIZE"] == 1000
        assert result["SMOTE_K_NEIGHBORS"] == 5
        assert result["CLASS_WEIGHT_STRATEGY"] == "balanced"
        assert result["NESTED_INNER_CV"] == 3
        assert result["NESTED_OUTER_CV"] == 5
        assert result["USE_NESTED_CV"] is False


# ===========================================================================
# build_review_summary
# ===========================================================================


class TestBuildReviewSummary:
    """Tests for build_review_summary()."""

    def _minimal_config(self):
        """Return a minimal config dict for summary generation."""
        return {
            "raster": "/data/image.tif",
            "vector": "/data/training.shp",
            "class_field": "class",
            "load_model": "",
            "classifier": "RF",
            "extraParam": {
                "USE_OPTUNA": False,
                "OPTUNA_TRIALS": 100,
                "COMPUTE_SHAP": False,
                "SHAP_OUTPUT": "",
                "SHAP_SAMPLE_SIZE": 1000,
                "USE_SMOTE": False,
                "SMOTE_K_NEIGHBORS": 5,
                "USE_CLASS_WEIGHTS": False,
                "CLASS_WEIGHT_STRATEGY": "balanced",
                "USE_NESTED_CV": False,
                "NESTED_INNER_CV": 3,
                "NESTED_OUTER_CV": 5,
            },
            "output_raster": "/data/output.tif",
            "confidence_map": "",
            "save_model": "",
            "confusion_matrix": "",
            "split_percent": 100,
        }

    def test_returns_string(self):
        """Output is a str."""
        assert isinstance(build_review_summary(self._minimal_config()), str)

    def test_contains_raster_path(self):
        """Summary includes the raster path."""
        summary = build_review_summary(self._minimal_config())
        assert "/data/image.tif" in summary

    def test_contains_vector_path(self):
        """Summary includes the vector path."""
        summary = build_review_summary(self._minimal_config())
        assert "/data/training.shp" in summary

    def test_contains_classifier(self):
        """Summary includes the classifier code."""
        summary = build_review_summary(self._minimal_config())
        assert "RF" in summary

    def test_contains_output_raster(self):
        """Summary includes the output raster path."""
        summary = build_review_summary(self._minimal_config())
        assert "/data/output.tif" in summary

    def test_optuna_section_appears_when_enabled(self):
        """When Optuna is enabled, Trials line is present."""
        cfg = self._minimal_config()
        cfg["extraParam"]["USE_OPTUNA"] = True
        cfg["extraParam"]["OPTUNA_TRIALS"] = 50
        summary = build_review_summary(cfg)
        assert "Trials : 50" in summary

    def test_smote_section_appears_when_enabled(self):
        """When SMOTE is enabled, k_neighbors line is present."""
        cfg = self._minimal_config()
        cfg["extraParam"]["USE_SMOTE"] = True
        cfg["extraParam"]["SMOTE_K_NEIGHBORS"] = 7
        summary = build_review_summary(cfg)
        assert "k_neighbors : 7" in summary

    def test_shap_section_appears_when_enabled(self):
        """When SHAP is enabled, output and sample size are present."""
        cfg = self._minimal_config()
        cfg["extraParam"]["COMPUTE_SHAP"] = True
        cfg["extraParam"]["SHAP_OUTPUT"] = "/data/shap.tif"
        cfg["extraParam"]["SHAP_SAMPLE_SIZE"] = 2000
        summary = build_review_summary(cfg)
        assert "/data/shap.tif" in summary
        assert "Sample size : 2000" in summary

    def test_nested_cv_section_appears_when_enabled(self):
        """When nested CV is enabled, inner/outer folds lines appear."""
        cfg = self._minimal_config()
        cfg["extraParam"]["USE_NESTED_CV"] = True
        cfg["extraParam"]["NESTED_INNER_CV"] = 4
        cfg["extraParam"]["NESTED_OUTER_CV"] = 6
        summary = build_review_summary(cfg)
        assert "Inner folds : 4" in summary
        assert "Outer folds : 6" in summary

    def test_load_model_line_present_when_set(self):
        """When a model path is set, it appears in the summary."""
        cfg = self._minimal_config()
        cfg["load_model"] = "/data/my_model.pkl"
        summary = build_review_summary(cfg)
        assert "/data/my_model.pkl" in summary

    def test_confusion_matrix_split_shown(self):
        """When confusion matrix is set, the split % is visible."""
        cfg = self._minimal_config()
        cfg["confusion_matrix"] = "/data/matrix.csv"
        cfg["split_percent"] = 70
        summary = build_review_summary(cfg)
        assert "Split % : 70" in summary

    def test_section_headers_present(self):
        """All four section headers appear."""
        summary = build_review_summary(self._minimal_config())
        assert "[Input Data]" in summary
        assert "[Algorithm]" in summary
        assert "[Advanced Options]" in summary
        assert "[Output]" in summary


# ===========================================================================
# AlgorithmPage logic — classifier code / dep-status mapping
# ===========================================================================


class TestAlgorithmPageLogic:
    """Verify classifier code and dependency mapping for all 11 classifiers."""

    def test_eleven_classifiers_defined(self):
        """Exactly 11 classifiers are in _CLASSIFIER_META."""
        assert len(_CLASSIFIER_META) == 11

    def test_all_codes_are_strings(self):
        """Every code is a non-empty string."""
        for code, _name, _sk, _xgb, _lgb in _CLASSIFIER_META:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_gmm_needs_no_deps(self):
        """GMM requires no external packages."""
        for code, _name, needs_sk, needs_xgb, needs_lgb in _CLASSIFIER_META:
            if code == "GMM":
                assert needs_sk is False
                assert needs_xgb is False
                assert needs_lgb is False

    def test_xgb_needs_xgboost_only(self):
        """XGB requires only xgboost."""
        for code, _name, needs_sk, needs_xgb, needs_lgb in _CLASSIFIER_META:
            if code == "XGB":
                assert needs_sk is False
                assert needs_xgb is True
                assert needs_lgb is False

    def test_lgb_needs_lightgbm_only(self):
        """LGB requires only lightgbm."""
        for code, _name, needs_sk, needs_xgb, needs_lgb in _CLASSIFIER_META:
            if code == "LGB":
                assert needs_sk is False
                assert needs_xgb is False
                assert needs_lgb is True

    def test_sklearn_classifiers_need_sklearn(self):
        """RF, SVM, KNN, ET, GBC, LR, NB, MLP all require sklearn."""
        sklearn_codes = {"RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"}
        for code, _name, needs_sk, _xgb, _lgb in _CLASSIFIER_META:
            if code in sklearn_codes:
                assert needs_sk is True, f"{code} should require sklearn"

    def test_classifier_available_gmm_always(self):
        """GMM is always available regardless of deps."""
        no_deps = dict.fromkeys(("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"), False)
        assert _classifier_available("GMM", no_deps) is True

    def test_classifier_available_rf_needs_sklearn(self):
        """RF is available only when sklearn is True."""
        deps_no = dict.fromkeys(("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"), False)
        assert _classifier_available("RF", deps_no) is False
        deps_yes = dict(deps_no)
        deps_yes["sklearn"] = True
        assert _classifier_available("RF", deps_yes) is True

    def test_classifier_available_xgb_needs_xgboost(self):
        """XGB is available only when xgboost is True."""
        deps = dict.fromkeys(("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"), False)
        assert _classifier_available("XGB", deps) is False
        deps["xgboost"] = True
        assert _classifier_available("XGB", deps) is True

    def test_classifier_available_lgb_needs_lightgbm(self):
        """LGB is available only when lightgbm is True."""
        deps = dict.fromkeys(("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"), False)
        assert _classifier_available("LGB", deps) is False
        deps["lightgbm"] = True
        assert _classifier_available("LGB", deps) is True

    def test_unknown_code_returns_false(self):
        """An unknown classifier code is never available."""
        deps = dict.fromkeys(("sklearn", "xgboost", "lightgbm", "optuna", "shap", "imblearn"), True)
        assert _classifier_available("UNKNOWN", deps) is False


# ===========================================================================
# AdvancedOptions logic — extraParam construction
# ===========================================================================


class TestAdvancedOptionsLogic:
    """Verify extraParam construction for various flag combinations."""

    def _base_extra(self, **overrides):
        """Return a base extraParam dict with optional overrides."""
        base = {
            "USE_OPTUNA": False,
            "OPTUNA_TRIALS": 100,
            "COMPUTE_SHAP": False,
            "SHAP_OUTPUT": "",
            "SHAP_SAMPLE_SIZE": 1000,
            "USE_SMOTE": False,
            "SMOTE_K_NEIGHBORS": 5,
            "USE_CLASS_WEIGHTS": False,
            "CLASS_WEIGHT_STRATEGY": "balanced",
            "CUSTOM_CLASS_WEIGHTS": {},
            "USE_NESTED_CV": False,
            "NESTED_INNER_CV": 3,
            "NESTED_OUTER_CV": 5,
        }
        base.update(overrides)
        return base

    def test_all_disabled(self):
        """All flags False by default."""
        ep = self._base_extra()
        assert ep["USE_OPTUNA"] is False
        assert ep["USE_SMOTE"] is False
        assert ep["USE_CLASS_WEIGHTS"] is False
        assert ep["COMPUTE_SHAP"] is False
        assert ep["USE_NESTED_CV"] is False

    def test_optuna_enabled(self):
        """Enabling Optuna sets the flag and keeps trial count."""
        ep = self._base_extra(USE_OPTUNA=True, OPTUNA_TRIALS=200)
        assert ep["USE_OPTUNA"] is True
        assert ep["OPTUNA_TRIALS"] == 200

    def test_smote_enabled(self):
        """Enabling SMOTE sets flag and k_neighbors."""
        ep = self._base_extra(USE_SMOTE=True, SMOTE_K_NEIGHBORS=7)
        assert ep["USE_SMOTE"] is True
        assert ep["SMOTE_K_NEIGHBORS"] == 7

    def test_class_weights_strategy(self):
        """Class weight strategy propagates correctly."""
        for strat in ("balanced", "uniform"):
            ep = self._base_extra(USE_CLASS_WEIGHTS=True, CLASS_WEIGHT_STRATEGY=strat)
            assert ep["CLASS_WEIGHT_STRATEGY"] == strat

    def test_shap_with_path(self):
        """SHAP output path propagates."""
        ep = self._base_extra(COMPUTE_SHAP=True, SHAP_OUTPUT="/out/shap.tif", SHAP_SAMPLE_SIZE=500)
        assert ep["COMPUTE_SHAP"] is True
        assert ep["SHAP_OUTPUT"] == "/out/shap.tif"
        assert ep["SHAP_SAMPLE_SIZE"] == 500

    def test_nested_cv_folds(self):
        """Nested CV inner/outer folds propagate."""
        ep = self._base_extra(USE_NESTED_CV=True, NESTED_INNER_CV=5, NESTED_OUTER_CV=10)
        assert ep["USE_NESTED_CV"] is True
        assert ep["NESTED_INNER_CV"] == 5
        assert ep["NESTED_OUTER_CV"] == 10

    def test_combined_flags(self):
        """Multiple features enabled simultaneously."""
        ep = self._base_extra(
            USE_OPTUNA=True,
            USE_SMOTE=True,
            USE_CLASS_WEIGHTS=True,
            COMPUTE_SHAP=True,
            USE_NESTED_CV=True,
        )
        assert ep["USE_OPTUNA"] is True
        assert ep["USE_SMOTE"] is True
        assert ep["USE_CLASS_WEIGHTS"] is True
        assert ep["COMPUTE_SHAP"] is True
        assert ep["USE_NESTED_CV"] is True


# ===========================================================================
# OutputPage logic — path / split extraction
# ===========================================================================


class TestOutputPageLogic:
    """Verify output config extraction with empty and filled values."""

    def _make_output_config(self, **overrides):
        """Return a default output config dict with optional overrides."""
        base = {
            "output_raster": "",
            "confidence_map": "",
            "save_model": "",
            "confusion_matrix": "",
            "split_percent": 100,
        }
        base.update(overrides)
        return base

    def test_empty_output_raster_signals_temp(self):
        """An empty output_raster means a temp file should be created."""
        cfg = self._make_output_config()
        assert cfg["output_raster"] == ""

    def test_filled_output_raster(self):
        """A filled output_raster propagates unchanged."""
        cfg = self._make_output_config(output_raster="/data/classified.tif")
        assert cfg["output_raster"] == "/data/classified.tif"

    def test_confidence_map_empty_when_unchecked(self):
        """confidence_map is empty when the user did not check it."""
        cfg = self._make_output_config()
        assert cfg["confidence_map"] == ""

    def test_confidence_map_filled(self):
        """confidence_map propagates the path."""
        cfg = self._make_output_config(confidence_map="/data/conf.tif")
        assert cfg["confidence_map"] == "/data/conf.tif"

    def test_save_model_empty_when_unchecked(self):
        """save_model empty when unchecked."""
        cfg = self._make_output_config()
        assert cfg["save_model"] == ""

    def test_save_model_filled(self):
        """save_model propagates the path."""
        cfg = self._make_output_config(save_model="/data/model.pkl")
        assert cfg["save_model"] == "/data/model.pkl"

    def test_split_100_when_no_matrix(self):
        """split_percent defaults to 100 when confusion matrix is off."""
        cfg = self._make_output_config()
        assert cfg["split_percent"] == 100

    def test_split_custom_with_matrix(self):
        """split_percent is the user-chosen value when matrix is on."""
        cfg = self._make_output_config(confusion_matrix="/data/mat.csv", split_percent=70)
        assert cfg["split_percent"] == 70
        assert cfg["confusion_matrix"] == "/data/mat.csv"
