"""Guided classification workflow for dzetsaka.

A QWizard-based step-by-step interface for configuring and launching
remote-sensing image classification. The guided workflow leads users through
input data selection, algorithm choice, advanced options, output paths,
and a final review before execution.

Pure-Python layout — no .ui file required.  The module also exposes
several helper functions (``check_dependency_availability``,
``build_smart_defaults``, ``build_review_summary``) that can be tested
without a Qt runtime.

Author:
    Nicolas Karasiak
"""

import json
import os
from pathlib import Path
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import QSettings, QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QCompleter,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWhatsThis,
    QWizard,
    QWizardPage,
)

# Import validated widgets for real-time validation feedback
from .validated_widgets import ValidatedSpinBox, ValidatedDoubleSpinBox

# Import training data quality checker
try:
    from .training_data_quality_checker import TrainingDataQualityChecker
    _QUALITY_CHECKER_AVAILABLE = True
except ImportError:
    _QUALITY_CHECKER_AVAILABLE = False

# Import theme support
try:
    from .theme_support import ThemeAwareWidget
    _THEME_SUPPORT_AVAILABLE = True
except Exception:
    _THEME_SUPPORT_AVAILABLE = False
    # Fallback: create empty mixin class
    class ThemeAwareWidget:
        """Fallback mixin when theme_support is not available."""
        def apply_theme(self):
            pass

try:
    from dzetsaka.domain.value_objects.recipe_schema_v2 import upgrade_recipe_to_v2 as _upgrade_recipe_to_v2
except Exception:
    # Test/import fallback when module is loaded outside the plugin package context.
    def _upgrade_recipe_to_v2(recipe):  # type: ignore[no-redef]
        # type: (Dict[str, object]) -> Dict[str, object]
        upgraded = dict(recipe or {})
        upgraded.setdefault("version", 1)
        upgraded.setdefault("schema_version", 2)
        upgraded.setdefault("provenance", {"source": "local", "author": "", "created_at": "", "updated_at": ""})
        upgraded.setdefault("constraints", {"offline_compatible": True, "requires_gpu": False})
        upgraded.setdefault("compat", {"min_plugin_version": "", "max_plugin_version": ""})
        upgraded.setdefault("expected_runtime_class", "medium")
        upgraded.setdefault("expected_accuracy_class", "high")
        upgraded.setdefault("dataset_fingerprint", "")
        upgraded.setdefault("signature", "")
        upgraded["schema_version"] = 2
        return upgraded

# ---------------------------------------------------------------------------
# Recipe recommendation system imports
# ---------------------------------------------------------------------------

try:
    from .recipe_recommender import RasterAnalyzer, RecipeRecommender
    from .recommendation_dialog import RecommendationDialog
    _RECOMMENDER_AVAILABLE = True
except Exception:
    _RECOMMENDER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global recipe update notification system
# ---------------------------------------------------------------------------

try:
    from qgis.PyQt.QtCore import QObject
except ImportError:
    from PyQt5.QtCore import QObject  # type: ignore


class _RecipeNotifier(QObject):
    """Global notifier for recipe updates across components."""
    recipesUpdated = pyqtSignal()


# Global instance
_recipe_notifier = _RecipeNotifier()


def notify_recipes_updated():
    # type: () -> None
    """Emit signal to notify all components that recipes were updated."""
    _recipe_notifier.recipesUpdated.emit()


# ---------------------------------------------------------------------------
# Dependency availability helpers (importable without Qt for unit tests)
# ---------------------------------------------------------------------------


def _full_dependency_bundle():
    # type: () -> List[str]
    try:
        from dzetsaka.qgis.dependency_catalog import FULL_DEPENDENCY_BUNDLE

        return list(FULL_DEPENDENCY_BUNDLE)
    except Exception:
        # Test/import fallback when running module outside plugin package context.
        return [
            "scikit-learn",
            "xgboost",
            "lightgbm",
            "catboost",
            "optuna",
            "shap",
            "seaborn",
            "imbalanced-learn",
        ]


def _full_bundle_label():
    # type: () -> str
    return ", ".join(_full_dependency_bundle())


def check_dependency_availability():
    # type: () -> Dict[str, bool]
    """Check which optional packages are importable at runtime.

    Returns
    -------
    dict[str, bool]
        Keys: ``sklearn``, ``xgboost``, ``lightgbm``, ``catboost``,
        ``optuna``, ``shap``, ``seaborn``, ``imblearn``.  Values: True when the package can be
        imported successfully.
    """
    deps = {
        "sklearn": False,
        "xgboost": False,
        "lightgbm": False,
        "catboost": False,
        "optuna": False,
        "shap": False,
        "seaborn": False,
        "imblearn": False,
    }  # type: Dict[str, bool]
    for key in deps:
        try:
            __import__(key)
            deps[key] = True
        except ImportError:
            pass
    return deps


def validate_recipe_dependencies(recipe):
    # type: (Dict[str, object]) -> tuple[bool, List[str]]
    """Validate that all dependencies required by a recipe are available.

    Parameters
    ----------
    recipe : dict
        Recipe dictionary containing classifier configuration

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, missing_packages) - is_valid is True if all dependencies
        are met, missing_packages contains names of missing packages
    """
    from .. import classifier_config

    missing = []  # type: List[str]

    # Get classifier code from recipe
    classifier = recipe.get("classifier", {})
    code = classifier.get("code", "GMM")

    # Check current dependencies
    deps = check_dependency_availability()

    # Check classifier-specific dependencies
    if code in classifier_config.SKLEARN_DEPENDENT or code in {"XGB", "LGB", "CB"}:
        if not deps.get("sklearn", False):
            missing.append("scikit-learn")

    if code in classifier_config.XGBOOST_DEPENDENT:
        if not deps.get("xgboost", False):
            missing.append("xgboost")

    if code in classifier_config.LIGHTGBM_DEPENDENT:
        if not deps.get("lightgbm", False):
            missing.append("lightgbm")

    if code in classifier_config.CATBOOST_DEPENDENT:
        if not deps.get("catboost", False):
            missing.append("catboost")

    # Check for optional features in extraParam
    extra = recipe.get("extraParam", {})
    if extra:
        if extra.get("USE_OPTUNA", False) and not deps.get("optuna", False):
            missing.append("optuna (for hyperparameter optimization)")

        if extra.get("COMPUTE_SHAP", False) and not deps.get("shap", False):
            missing.append("shap (for explainability)")

        if extra.get("USE_SMOTE", False) and not deps.get("imblearn", False):
            missing.append("imbalanced-learn (for SMOTE)")

    return len(missing) == 0, missing


def _show_issue_popup(owner, installer, title, error_type, error_message, context):
    # type: (QWidget, object, str, str, str, str) -> None
    """Show the standard issue-report popup from whichever integration is available."""
    if installer and hasattr(installer, "_show_github_issue_popup"):
        installer._show_github_issue_popup(title, error_type, error_message, context)
        return

    try:
        from dzetsaka.logging import show_error_dialog

        show_error_dialog(
            title,
            f"{error_type}: {error_message}\n\nContext: {context}",
            parent=owner,
        )
    except Exception:
        QMessageBox.warning(
            owner,
            title,
            f"{error_type}: {error_message}\n\nPlease report at https://github.com/nkarasiak/dzetsaka/issues",
        )


def build_smart_defaults(deps):
    # type: (Dict[str, bool]) -> Dict[str, object]
    """Build a pre-filled extraParam dict based on available packages.

    Each feature is enabled only when its backing package is importable.

    Parameters
    ----------
    deps : dict[str, bool]
        Output of :func:`check_dependency_availability`.

    Returns
    -------
    dict[str, object]
        ``extraParam``-compatible dictionary with sensible defaults.
    """
    return {
        "USE_OPTUNA": deps.get("optuna", False),
        "OPTUNA_TRIALS": 100,
        "COMPUTE_SHAP": deps.get("shap", False),
        "SHAP_OUTPUT": "",
        "SHAP_SAMPLE_SIZE": 1000,
        "USE_SMOTE": deps.get("imblearn", False),
        "SMOTE_K_NEIGHBORS": 5,
        "USE_CLASS_WEIGHTS": deps.get("sklearn", False),
        "CLASS_WEIGHT_STRATEGY": "balanced",
        "CUSTOM_CLASS_WEIGHTS": {},
        "USE_NESTED_CV": False,
        "NESTED_INNER_CV": 3,
        "NESTED_OUTER_CV": 5,
        "CV_MODE": "POLYGON_GROUP",
    }  # type: Dict[str, object]


def build_review_summary(config):
    # type: (Dict[str, object]) -> str
    """Produce a human-readable summary of the guided workflow configuration.

    Parameters
    ----------
    config : dict
        The full config dict that the guided workflow would emit.

    Returns
    -------
    str
        Formatted multi-line summary string.
    """
    lines = []  # type: List[str]
    lines.append("=== Classification Configuration ===")
    lines.append("")

    # --- Input Data ---
    lines.append("[Input]")
    lines.append("  Raster : " + str(config.get("raster", "<not set>")))
    lines.append("  Vector : " + str(config.get("vector", "<not set>")))
    lines.append("  Label field : " + str(config.get("class_field", "<not set>")))
    model_path = config.get("load_model", "")
    if model_path:
        lines.append("  Use existing model : " + str(model_path))
    lines.append("")

    # --- Algorithm ---
    lines.append("[Algorithm]")
    lines.append("  Classifier : " + str(config.get("classifier", "<not set>")))
    lines.append("")

    # --- Advanced Options ---
    lines.append("[Advanced Setup]")
    extra = config.get("extraParam", {})  # type: dict
    lines.append("  Optuna optimization : " + str(extra.get("USE_OPTUNA", False)))
    if extra.get("USE_OPTUNA", False):
        lines.append("    Trials : " + str(extra.get("OPTUNA_TRIALS", 100)))
    lines.append("  SMOTE : " + str(extra.get("USE_SMOTE", False)))
    if extra.get("USE_SMOTE", False):
        lines.append("    k_neighbors : " + str(extra.get("SMOTE_K_NEIGHBORS", 5)))
    lines.append("  Class weights : " + str(extra.get("USE_CLASS_WEIGHTS", False)))
    if extra.get("USE_CLASS_WEIGHTS", False):
        lines.append("    Strategy : " + str(extra.get("CLASS_WEIGHT_STRATEGY", "balanced")))
    lines.append("  SHAP explainability : " + str(extra.get("COMPUTE_SHAP", False)))
    if extra.get("COMPUTE_SHAP", False):
        lines.append("    Output : " + str(extra.get("SHAP_OUTPUT", "<temp>")))
        lines.append("    Sample size : " + str(extra.get("SHAP_SAMPLE_SIZE", 1000)))
    lines.append("  Nested CV : " + str(extra.get("USE_NESTED_CV", False)))
    if extra.get("USE_NESTED_CV", False):
        lines.append("    Inner folds : " + str(extra.get("NESTED_INNER_CV", 3)))
        lines.append("    Outer folds : " + str(extra.get("NESTED_OUTER_CV", 5)))
    cv_mode = str(extra.get("CV_MODE", "POLYGON_GROUP"))
    lines.append("  CV mode : " + cv_mode)
    lines.append("")

    # --- Output ---
    lines.append("[Output]")
    lines.append("  Classification map : " + str(config.get("output_raster", "<temp file>")))
    lines.append("  Confidence map : " + str(config.get("confidence_map", "")))
    lines.append("  Save model : " + str(config.get("save_model", "")))
    confusion_matrix_path = str(config.get("confusion_matrix", "") or "").strip()
    split_percent = int(config.get("split_percent", 100))
    if split_percent < 100 and not confusion_matrix_path:
        confusion_matrix_path = "<auto-generated near output map>"
    lines.append("  Confusion matrix : " + confusion_matrix_path)
    if split_percent < 100:
        lines.append("    Validation split % : " + str(config.get("split_percent", 50)))
    if extra.get("GENERATE_REPORT_BUNDLE", False):
        lines.append("  Report bundle : True")
        lines.append("    Folder : " + str(extra.get("REPORT_OUTPUT_DIR", "<temporary folder>")))
        lines.append("    Label column : " + str(extra.get("REPORT_LABEL_COLUMN", "")))
        lines.append("    Label map : " + str(extra.get("REPORT_LABEL_MAP", "")))
    lines.append("")
    lines.append("=== End of Configuration ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------

_RECIPE_VERSION = 1

_RECIPE_SCHEMA_TEXT = (
    "Remote recipe endpoint schema (JSON):\n"
    "- Accepts either a list of recipes or an object with a 'recipes' list.\n"
    "- Each recipe is a JSON object with:\n"
    "  - name: string (required)\n"
    "  - description: string (optional)\n"
    "  - classifier: { code: one of GMM, RF, SVM, KNN, XGB, LGB, CB, ET, GBC, LR, NB, MLP }\n"
    "  - postprocess: { confidence_map: bool, save_model: bool, confusion_matrix: bool }\n"
    "  - validation: { split_percent: 10-100, nested_cv: bool, nested_inner_cv: 2-10, nested_outer_cv: 2-10,\n"
    "                 cv_mode: RANDOM_SPLIT|POLYGON_GROUP }\n"
    "  - extraParam: (optional) extra parameters\n"
)


def _recipe_template():
    # type: () -> Dict[str, object]
    return {
        "version": _RECIPE_VERSION,
        "schema_version": 2,
        "name": "Unnamed Recipe",
        "description": "",
        "provenance": {"source": "local", "author": "", "created_at": "", "updated_at": ""},
        "constraints": {"offline_compatible": True, "requires_gpu": False},
        "compat": {"min_plugin_version": "", "max_plugin_version": ""},
        "expected_runtime_class": "medium",
        "expected_accuracy_class": "high",
        "dataset_fingerprint": "",
        "signature": "",
        "preprocessing": {},
        "features": {"bands": "all"},
        "classifier": {"code": "GMM"},
        "postprocess": {"confidence_map": False, "save_model": False, "confusion_matrix": False},
        "validation": {
            "split_percent": 100,
            "nested_cv": False,
            "nested_inner_cv": 3,
            "nested_outer_cv": 5,
            "cv_mode": "POLYGON_GROUP",
        },
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
            "CUSTOM_CLASS_WEIGHTS": {},
            "USE_NESTED_CV": False,
            "NESTED_INNER_CV": 3,
            "NESTED_OUTER_CV": 5,
            "CV_MODE": "POLYGON_GROUP",
            "GENERATE_REPORT_BUNDLE": True,
            "OPEN_REPORT_IN_BROWSER": True,
        },
    }  # type: Dict[str, object]


def build_fast_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "GMM Baseline Express",
            "description": "Probabilistic baseline that delivers the fastest response without extra dependencies.",
            "is_template": True,
            "category": "Beginner",
            "expected_runtime": "< 1 min",
            "typical_accuracy": "70-80%",
            "best_for": "Quick baseline testing and prototyping",
        }
    )
    return recipe


def build_catboost_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "CatBoost Strong Baseline (2026)",
            "description": "High-performing tree boosting baseline with balanced class weighting.",
            "classifier": {"code": "CB"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                USE_OPTUNA=False,
                COMPUTE_SHAP=False,
                USE_CLASS_WEIGHTS=True,
                CLASS_WEIGHT_STRATEGY="balanced",
            ),
            "validation": dict(_recipe_template()["validation"], split_percent=75, cv_mode="POLYGON_GROUP"),
            "postprocess": {"confidence_map": True, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def build_lightgbm_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "LightGBM Large-Scale (Optuna)",
            "description": "Optuna-tuned LightGBM with nested CV and SMOTE-ready preconditioning.",
            "is_template": True,
            "category": "Intermediate",
            "expected_runtime": "10-20 min",
            "typical_accuracy": "85-92%",
            "best_for": "Large-scale datasets with many features",
            "classifier": {"code": "LGB"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                USE_OPTUNA=True,
                OPTUNA_TRIALS=150,
                USE_CLASS_WEIGHTS=True,
                CLASS_WEIGHT_STRATEGY="balanced",
            ),
            "validation": dict(_recipe_template()["validation"], split_percent=75, cv_mode="POLYGON_GROUP"),
            "postprocess": {"confidence_map": False, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def build_best_accuracy_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "CatBoost SOTA (Optuna + Polygon CV)",
            "description": "CatBoost + Optuna with polygon grouping to maximize accuracy and fairness.",
            "is_template": True,
            "category": "Advanced",
            "expected_runtime": "20-40 min",
            "typical_accuracy": "92-96%",
            "best_for": "Maximum accuracy on production workflows",
            "classifier": {"code": "CB"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                USE_OPTUNA=True,
                OPTUNA_TRIALS=300,
                USE_CLASS_WEIGHTS=True,
                CLASS_WEIGHT_STRATEGY="balanced",
                CV_MODE="POLYGON_GROUP",
            ),
            "validation": dict(
                _recipe_template()["validation"],
                split_percent=75,
                cv_mode="POLYGON_GROUP",
                nested_cv=False,
            ),
            "postprocess": {"confidence_map": True, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def build_explainability_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "CatBoost SHAP Explainability",
            "description": "CatBoost + SHAP feature importance with confidence outputs for QA.",
            "is_template": True,
            "category": "Intermediate",
            "expected_runtime": "5-15 min",
            "typical_accuracy": "88-94%",
            "best_for": "Interpretable models and feature importance analysis",
            "classifier": {"code": "CB"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                USE_OPTUNA=False,
                COMPUTE_SHAP=True,
                SHAP_SAMPLE_SIZE=2000,
                USE_CLASS_WEIGHTS=True,
                CLASS_WEIGHT_STRATEGY="balanced",
            ),
            "validation": dict(_recipe_template()["validation"], split_percent=70),
            "postprocess": {"confidence_map": True, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def build_polygon_group_cv_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "Polygon Group CV (No Pixel Leakage)",
            "description": (
                "Group-based cross-validation by polygon identifier to reduce within-polygon spatial autocorrelation bias."
            ),
            "classifier": {"code": "CB"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                CV_MODE="POLYGON_GROUP",
                USE_OPTUNA=True,
                OPTUNA_TRIALS=120,
                USE_CLASS_WEIGHTS=True,
                CLASS_WEIGHT_STRATEGY="balanced",
            ),
            "validation": dict(
                _recipe_template()["validation"],
                cv_mode="POLYGON_GROUP",
                split_percent=75,
            ),
            "postprocess": {"confidence_map": False, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def build_imbalanced_fast_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "RF Imbalanced Guard (SMOTE + Weights)",
            "description": "Fast Random Forest with SMOTE and class weights to tame imbalance.",
            "is_template": True,
            "category": "Beginner",
            "expected_runtime": "2-5 min",
            "typical_accuracy": "82-88%",
            "best_for": "Imbalanced datasets with rare classes",
            "classifier": {"code": "RF"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                USE_SMOTE=True,
                SMOTE_K_NEIGHBORS=5,
                USE_CLASS_WEIGHTS=True,
                CLASS_WEIGHT_STRATEGY="balanced",
            ),
            "validation": dict(_recipe_template()["validation"], split_percent=70),
            "postprocess": {"confidence_map": False, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def build_nested_cv_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "Nested CV Benchmark (XGBoost)",
            "description": "Leakage-resistant benchmark with nested CV and Optuna on XGBoost.",
            "classifier": {"code": "XGB"},
            "extraParam": dict(
                _recipe_template()["extraParam"],
                USE_OPTUNA=True,
                OPTUNA_TRIALS=120,
                USE_NESTED_CV=True,
                NESTED_INNER_CV=3,
                NESTED_OUTER_CV=5,
                CV_MODE="POLYGON_GROUP",
            ),
            "validation": dict(
                _recipe_template()["validation"],
                nested_cv=True,
                nested_inner_cv=3,
                nested_outer_cv=5,
                cv_mode="POLYGON_GROUP",
                split_percent=75,
            ),
            "postprocess": {"confidence_map": False, "save_model": True, "confusion_matrix": True},
        }
    )
    return recipe


def normalize_recipe(recipe):
    # type: (Dict[str, object]) -> Dict[str, object]
    """Ensure a recipe contains all required keys."""
    recipe = _upgrade_recipe_to_v2(dict(recipe or {}))
    base = _recipe_template()
    for key, value in base.items():
        if key not in recipe:
            recipe[key] = value
    # Defensive: ensure nested keys exist
    recipe.setdefault("classifier", {"code": "GMM"})
    recipe.setdefault("preprocessing", {})
    recipe.setdefault("features", {"bands": "all"})
    recipe.setdefault("postprocess", {})
    recipe.setdefault("validation", {})
    recipe.setdefault("extraParam", {})
    recipe["validation"].setdefault("split_percent", 100)
    recipe["validation"].setdefault("nested_cv", False)
    recipe["validation"].setdefault("nested_inner_cv", 3)
    recipe["validation"].setdefault("nested_outer_cv", 5)
    recipe["validation"].setdefault("cv_mode", "POLYGON_GROUP")
    recipe["extraParam"].setdefault("USE_OPTUNA", False)
    recipe["extraParam"].setdefault("OPTUNA_TRIALS", 100)
    recipe["extraParam"].setdefault("COMPUTE_SHAP", False)
    recipe["extraParam"].setdefault("SHAP_OUTPUT", "")
    recipe["extraParam"].setdefault("SHAP_SAMPLE_SIZE", 1000)
    recipe["extraParam"].setdefault("USE_SMOTE", False)
    recipe["extraParam"].setdefault("SMOTE_K_NEIGHBORS", 5)
    recipe["extraParam"].setdefault("USE_CLASS_WEIGHTS", False)
    recipe["extraParam"].setdefault("CLASS_WEIGHT_STRATEGY", "balanced")
    recipe["extraParam"].setdefault("CUSTOM_CLASS_WEIGHTS", {})
    recipe["extraParam"].setdefault("USE_NESTED_CV", False)
    recipe["extraParam"].setdefault("NESTED_INNER_CV", 3)
    recipe["extraParam"].setdefault("NESTED_OUTER_CV", 5)
    recipe["extraParam"].setdefault("CV_MODE", "POLYGON_GROUP")
    recipe["extraParam"].setdefault("GENERATE_REPORT_BUNDLE", True)
    recipe["extraParam"].setdefault("REPORT_OUTPUT_DIR", "")
    recipe["extraParam"].setdefault("REPORT_LABEL_COLUMN", "")
    recipe["extraParam"].setdefault("REPORT_LABEL_MAP", "")
    return recipe


def _valid_classifier_codes():
    # type: () -> List[str]
    return [code for code, _n, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META]


def validate_recipe_list(payload):
    # type: (object) -> tuple
    """Validate a payload and return (recipes, errors)."""
    errors = []  # type: List[str]
    recipes = []  # type: List[Dict[str, object]]

    if isinstance(payload, dict):
        payload = payload.get("recipes", [])
    if not isinstance(payload, list):
        return [], ["Root must be a list or an object with 'recipes' list."]

    valid_codes = set(_valid_classifier_codes())

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(f"Recipe #{idx + 1}: not an object.")
            continue
        name = item.get("name", "")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"Recipe #{idx + 1}: missing or empty 'name'.")
            continue
        if "classifier" not in item:
            errors.append(f"Recipe '{name}': missing required 'classifier'.")
            continue
        classifier = item.get("classifier", {})
        if not isinstance(classifier, dict):
            errors.append(f"Recipe '{name}': classifier must be an object.")
            continue
        if "code" not in classifier:
            errors.append(f"Recipe '{name}': missing required classifier.code.")
            continue
        code = classifier.get("code")
        if code not in valid_codes:
            errors.append(f"Recipe '{name}': unknown classifier code '{code}'.")
            continue
        validation = item.get("validation", {})
        if isinstance(validation, dict):
            split = validation.get("split_percent", 100)
            if not isinstance(split, int):
                errors.append(f"Recipe '{name}': split_percent must be an integer.")
                continue
            if split < 10 or split > 100:
                errors.append(f"Recipe '{name}': split_percent must be 10-100.")
                continue
            cv_mode = validation.get("cv_mode", "POLYGON_GROUP")
            if cv_mode not in ("RANDOM_SPLIT", "POLYGON_GROUP"):
                errors.append(f"Recipe '{name}': cv_mode must be RANDOM_SPLIT or POLYGON_GROUP.")
                continue
        extra = item.get("extraParam", {})
        if isinstance(extra, dict) and "CLASS_WEIGHT_STRATEGY" in extra:
            strategy = extra.get("CLASS_WEIGHT_STRATEGY")
            if strategy not in ("balanced", "uniform"):
                errors.append(f"Recipe '{name}': CLASS_WEIGHT_STRATEGY must be 'balanced' or 'uniform'.")
                continue
        recipes.append(normalize_recipe(item))

    return recipes, errors


def format_recipe_summary(recipe):
    # type: (Dict[str, object]) -> str
    """Return a readable multi-line summary for a recipe."""
    recipe = normalize_recipe(dict(recipe))
    classifier = recipe.get("classifier", {})
    classifier_code = classifier.get("code", "GMM")
    validation = recipe.get("validation", {})
    extra = recipe.get("extraParam", {})
    lines = []
    lines.append(f"Name: {recipe.get('name', '')}")
    description = recipe.get("description", "")
    if description:
        lines.append(f"Description: {description}")
    lines.append(f"Classifier: {classifier_code}")
    lines.append(f"Optuna: {extra.get('USE_OPTUNA', False)}")
    lines.append(f"SMOTE: {extra.get('USE_SMOTE', False)}")
    lines.append(f"Class Weights: {extra.get('USE_CLASS_WEIGHTS', False)}")
    lines.append(f"SHAP: {extra.get('COMPUTE_SHAP', False)}")
    lines.append(f"Nested CV: {validation.get('nested_cv', extra.get('USE_NESTED_CV', False))}")
    lines.append(f"CV mode: {validation.get('cv_mode', extra.get('CV_MODE', 'POLYGON_GROUP'))}")
    lines.append(f"Validation split %: {validation.get('split_percent', 100)}")
    return "\n".join(lines)



def load_builtin_recipes():
    # type: () -> List[Dict[str, object]]
    """Load built-in template recipes from builtin_recipes.json."""
    builtin_path = os.path.join(os.path.dirname(__file__), "builtin_recipes.json")
    builtin_recipes = []  # type: List[Dict[str, object]]

    if os.path.exists(builtin_path):
        try:
            with open(builtin_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    builtin_recipes = data.get("recipes", [])
                elif isinstance(data, list):
                    builtin_recipes = data
        except (IOError, json.JSONDecodeError, TypeError, ValueError):
            pass  # Silently fail if built-in recipes can't be loaded

    # Normalize and mark as templates
    normalized = []
    for recipe in builtin_recipes:
        if isinstance(recipe, dict):
            normalized_recipe = normalize_recipe(recipe)
            # Ensure metadata exists and is_template is set
            if "metadata" not in normalized_recipe:
                normalized_recipe["metadata"] = {}
            normalized_recipe["metadata"]["is_template"] = True
            normalized.append(normalized_recipe)

    return normalized


def is_builtin_recipe(recipe):
    # type: (Dict[str, object]) -> bool
    """Check if a recipe is a built-in template (read-only)."""
    metadata = recipe.get("metadata", {})
    if isinstance(metadata, dict):
        return bool(metadata.get("is_template", False))
    return False


def load_recipes(settings):
    # type: (QSettings) -> List[Dict[str, object]]
    """Load recipes from QSettings merged with built-in templates."""
    # Load user recipes from settings
    raw = settings.value("/dzetsaka/recipes", "", str)
    user_recipes = []  # type: List[Dict[str, object]]
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                user_recipes = data.get("recipes", [])
            elif isinstance(data, list):
                user_recipes = data
        except (TypeError, ValueError):
            user_recipes = []

    # Load legacy default recipes (kept for backward compatibility)
    default_recipes = [
        build_fast_recipe(),
        build_lightgbm_recipe(),
        build_best_accuracy_recipe(),
        build_explainability_recipe(),
        build_imbalanced_fast_recipe(),
    ]

    # Load built-in template recipes
    builtin_recipes = load_builtin_recipes()

    # Remove old built-in names that have been replaced
    removed_builtin_names = {
        "CatBoost Strong Baseline (2026)",
        "SOTA 2026 (CatBoost + Optuna + Polygon CV)",
        "Polygon Group CV (No Pixel Leakage)",
        "Nested CV Benchmark (XGBoost)",
        "Gaussian Mixture Model (Fastest)",
        "LightGBM Large-Scale",
        "CatBoost SOTA (Optuna + Polygon CV)",
        "Explainability (CatBoost + SHAP)",
        "Imbalanced Fast (RF + SMOTE + Weights)",
        "Satellite Enterprise · CatBoost · Optuna, SHAP, Class weights, Nested CV · 65% Polygon Group",
        "Explain & Report · CatBoost · SHAP · 70% Polygon Group",
        # Visual recipe shop names replaced by the new 10-profile naming system
        "Quick Test",
        "Forest Mapping",
        "Crop Classification",
        "Urban/Rural Land Use",
        "Water Detection",
        "Hyperspectral Analysis",
        "Large Dataset Fast",
        "Publication Quality",
    }
    user_recipes = [r for r in user_recipes if str(r.get("name", "")) not in removed_builtin_names]

    # Start with built-in templates
    recipes = list(builtin_recipes)

    # Add legacy defaults if not already present
    builtin_names = {r.get("name") for r in recipes}
    for default_recipe in default_recipes:
        if default_recipe.get("name") not in builtin_names:
            recipes.append(default_recipe)

    # Add user recipes (normalized)
    if user_recipes:
        user_recipes = [normalize_recipe(r) for r in user_recipes if isinstance(r, dict)]
        # User recipes come after built-ins so they appear at the bottom
        recipes.extend(user_recipes)

    return recipes


def save_recipes(settings, recipes):
    # type: (QSettings, List[Dict[str, object]]) -> None
    """Persist recipes to QSettings (only user recipes, not built-in templates)."""
    # Filter out built-in templates - they should not be saved to user settings
    user_recipes = [r for r in recipes if not is_builtin_recipe(r)]
    settings.setValue("/dzetsaka/recipes", json.dumps(user_recipes))


def recipe_from_config(config, name, description=""):
    # type: (Dict[str, object], str, str) -> Dict[str, object]
    """Create a recipe dict from the current guided workflow config."""
    extra = config.get("extraParam", {}) or {}
    recipe = _recipe_template()
    recipe.update(
        {
            "name": name,
            "description": description,
            "classifier": {"code": config.get("classifier", "GMM")},
            "postprocess": {
                "confidence_map": bool(config.get("confidence_map")),
                "save_model": bool(config.get("save_model")),
                "confusion_matrix": bool(config.get("confusion_matrix")),
            },
            "validation": {
                "split_percent": int(config.get("split_percent", 100)),
                "nested_cv": bool(extra.get("USE_NESTED_CV", False)),
                "nested_inner_cv": int(extra.get("NESTED_INNER_CV", 3)),
                "nested_outer_cv": int(extra.get("NESTED_OUTER_CV", 5)),
                "cv_mode": str(extra.get("CV_MODE", "POLYGON_GROUP")),
            },
            "extraParam": extra,
        }
    )
    return recipe


# ---------------------------------------------------------------------------
# Classifier metadata used by the guided workflow (mirrors classifier_config)
# ---------------------------------------------------------------------------

# (code, full name, requires_sklearn, requires_xgboost, requires_lightgbm, requires_catboost)
_CLASSIFIER_META = [
    ("GMM", "Gaussian Mixture Model", False, False, False, False),
    ("RF", "Random Forest", True, False, False, False),
    ("SVM", "Support Vector Machine", True, False, False, False),
    ("KNN", "K-Nearest Neighbors", True, False, False, False),
    ("XGB", "XGBoost", True, True, False, False),
    ("LGB", "LightGBM", True, False, True, False),
    ("CB", "CatBoost", True, False, False, True),
    ("ET", "Extra Trees", True, False, False, False),
    ("GBC", "Gradient Boosting Classifier", True, False, False, False),
    ("LR", "Logistic Regression", True, False, False, False),
    ("NB", "Gaussian Naive Bayes", True, False, False, False),
    ("MLP", "Multi-layer Perceptron", True, False, False, False),
]


def _qt_align_top():
    # type: () -> object
    """Return a Qt top-alignment flag compatible with Qt5 and Qt6 APIs."""
    if hasattr(Qt, "AlignmentFlag"):
        return Qt.AlignmentFlag.AlignTop
    return Qt.AlignTop


def _qt_align_hcenter():
    # type: () -> object
    """Return a Qt horizontal-center alignment flag compatible with Qt5 and Qt6 APIs."""
    if hasattr(Qt, "AlignmentFlag"):
        return Qt.AlignmentFlag.AlignHCenter
    return Qt.AlignHCenter


def _qt_align_right():
    # type: () -> object
    """Return a Qt right-alignment flag compatible with Qt5 and Qt6 APIs."""
    if hasattr(Qt, "AlignmentFlag"):
        return Qt.AlignmentFlag.AlignRight
    return Qt.AlignRight


class _CoverPixmapLabel(QLabel):
    """QLabel that draws a pixmap using cover behavior (center-cropped)."""

    def __init__(self, parent=None):
        super(_CoverPixmapLabel, self).__init__(parent)
        self._source_pixmap = QPixmap()
        self.setScaledContents(False)

    def set_cover_pixmap(self, pixmap):
        # type: (QPixmap) -> None
        self._source_pixmap = pixmap or QPixmap()
        self.update()

    def paintEvent(self, event):
        # type: (object) -> None
        if self._source_pixmap.isNull():
            return super(_CoverPixmapLabel, self).paintEvent(event)

        painter = QPainter(self)
        target = self.rect()
        src = self._source_pixmap

        if hasattr(Qt, "AspectRatioMode"):
            aspect_mode = Qt.AspectRatioMode.KeepAspectRatioByExpanding
            transform_mode = Qt.TransformationMode.SmoothTransformation
        else:
            aspect_mode = Qt.KeepAspectRatioByExpanding
            transform_mode = Qt.SmoothTransformation
        scaled = src.scaled(target.size(), aspect_mode, transform_mode)
        x = (scaled.width() - target.width()) // 2
        y = (scaled.height() - target.height()) // 2
        painter.drawPixmap(target, scaled, scaled.rect().adjusted(x, y, -x, -y))


def _classifier_available(code, deps):
    # type: (str, Dict[str, bool]) -> bool
    """Return True when all hard dependencies for *code* are satisfied."""
    for c, _name, needs_sk, needs_xgb, needs_lgb, needs_cb in _CLASSIFIER_META:
        if c == code:
            if needs_sk and not deps.get("sklearn", False):
                return False
            if needs_xgb and not deps.get("xgboost", False):
                return False
            if needs_lgb and not deps.get("lightgbm", False):
                return False
            if needs_cb and not deps.get("catboost", False):
                return False
            return True
    return False


class RecipeShopDialog(QDialog):
    """Interactive dialog to compose a new recipe from the current dashboard context."""

    _PRESETS = [
        {
            "name": "Gaussian Mixture Model (Fastest)",
            "description": "Built-in probabilistic baseline that runs instantly with no extra dependencies.",
            "classifier": "GMM",
            "optuna": False,
            "shap": False,
            "smote": False,
            "class_weights": False,
            "nested_cv": False,
            "split": 100,
            "cv_mode": "RANDOM_SPLIT",
            "report_bundle": True,
            "open_report": True,
        },
        {
            "name": "LightGBM Large-Scale (Optuna)",
            "description": "LightGBM with Optuna, SMOTE, and nested CV tuned for large datasets.",
            "classifier": "LGB",
            "optuna": True,
            "optuna_trials": 200,
            "shap": False,
            "smote": True,
            "class_weights": True,
            "nested_cv": True,
            "nested_inner": 4,
            "nested_outer": 6,
            "split": 65,
            "cv_mode": "POLYGON_GROUP",
            "report_bundle": True,
            "open_report": True,
        },
        {
            "name": "CatBoost SOTA (Optuna + Polygon CV)",
            "description": "CatBoost + Optuna with community best-practice polygon CV for accuracy.",
            "classifier": "CB",
            "optuna": True,
            "optuna_trials": 250,
            "shap": True,
            "shap_sample": 2000,
            "smote": False,
            "class_weights": True,
            "nested_cv": True,
            "nested_inner": 4,
            "nested_outer": 6,
            "split": 65,
            "cv_mode": "POLYGON_GROUP",
            "report_bundle": True,
            "open_report": True,
        },
        {
            "name": "CatBoost SHAP Explainability",
            "description": "CatBoost with SHAP-focused output for explainability and reporting.",
            "classifier": "CB",
            "optuna": False,
            "shap": True,
            "shap_sample": 2500,
            "smote": False,
            "class_weights": True,
            "nested_cv": False,
            "split": 70,
            "cv_mode": "POLYGON_GROUP",
            "report_bundle": True,
            "open_report": True,
        },
    ]

    def __init__(self, parent=None, deps=None, installer=None, seed_recipe=None, context_text=""):
        # type: (Optional[QWidget], Optional[Dict[str, bool]], Optional[object], Optional[Dict[str, object]], str) -> None
        super(RecipeShopDialog, self).__init__(parent)
        self.setWindowTitle("Create Custom Recipe")
        self.setMinimumSize(620, 520)
        self._deps = deps or check_dependency_availability()
        self._installer = installer
        self._recipe = normalize_recipe(dict(seed_recipe or _recipe_template()))
        self._context_text = context_text.strip()
        self._previous_classifier_index = 0
        self._run_requested = False
        self._user_edited_name = False
        self._last_auto_name = ""
        self._active_preset = None
        self._active_preset_label = "Custom"
        self._build_ui()
        self._load_seed_recipe()
        self._update_summary()

    def _build_ui(self):
        # type: () -> None
        root = QVBoxLayout()
        root.setSpacing(8)

        title = QLabel("Shop your methods and build a reusable recipe.")
        title.setStyleSheet("font-weight: 600;")
        root.addWidget(title)

        if self._context_text:
            context = QLabel(self._context_text)
            context.setWordWrap(True)
            context.setStyleSheet("color: #4d4d4d;")
            root.addWidget(context)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.nameEdit = QLineEdit()
        self.nameEdit.setPlaceholderText("Example: Fast + Explainable RF")
        self.nameEdit.textChanged.connect(self._update_summary)
        self.nameEdit.textEdited.connect(self._on_name_edited)
        name_row.addWidget(self.nameEdit)
        root.addLayout(name_row)

        preset_row = QHBoxLayout()
        header = QLabel("Preset:")
        header.setStyleSheet("font-weight: 600;")
        preset_row.addWidget(header)

        self.presetCombo = QComboBox()
        self.presetCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.presetCombo.addItem("Custom recipe", None)
        for preset in self._PRESETS:
            self.presetCombo.addItem(preset.get("name", "Preset"), preset)
        self.presetCombo.currentIndexChanged.connect(self._on_preset_combo_changed)
        preset_row.addWidget(self.presetCombo)
        root.addLayout(preset_row)

        self.presetDescription = QLabel("")
        self.presetDescription.setWordWrap(True)
        self.presetDescription.setStyleSheet("color: #555;")
        root.addWidget(self.presetDescription)

        self.presetMethodsLabel = QLabel("")
        self.presetMethodsLabel.setWordWrap(True)
        self.presetMethodsLabel.setStyleSheet("color: #777; font-size: 11px;")
        root.addWidget(self.presetMethodsLabel)
        self._update_preset_info(None)

        classifier_row = QHBoxLayout()
        classifier_row.addWidget(QLabel("Core model:"))
        self.classifierCombo = QComboBox()
        for code, name, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            label = f"{name} ({code})"
            self.classifierCombo.addItem(label, code)
        self.classifierCombo.currentIndexChanged.connect(self._handle_classifier_changed)
        classifier_row.addWidget(self.classifierCombo)
        root.addLayout(classifier_row)

        method_group = QGroupBox("Method shop")
        method_layout = QGridLayout()
        method_layout.setHorizontalSpacing(12)
        method_layout.setVerticalSpacing(6)

        self.optunaCheck = QCheckBox("Hyperparameter search (Optuna)")
        self.optunaTrialsSpin = QSpinBox()
        self.optunaTrialsSpin.setRange(10, 2000)
        self.optunaTrialsSpin.setSingleStep(10)
        method_layout.addWidget(self.optunaCheck, 0, 0)
        method_layout.addWidget(QLabel("Trials"), 0, 1)
        method_layout.addWidget(self.optunaTrialsSpin, 0, 2)
        method_layout.addWidget(self._method_info_button("Run an Optuna tuning loop (requires optuna)."), 0, 3)

        self.shapCheck = QCheckBox("Explainability (SHAP)")
        self.shapSampleSpin = QSpinBox()
        self.shapSampleSpin.setRange(100, 50000)
        self.shapSampleSpin.setSingleStep(100)
        method_layout.addWidget(self.shapCheck, 1, 0)
        method_layout.addWidget(QLabel("Sample size"), 1, 1)
        method_layout.addWidget(self.shapSampleSpin, 1, 2)
        method_layout.addWidget(self._method_info_button("Compute SHAP explainability (requires shap)."), 1, 3)

        self.smoteCheck = QCheckBox("Class balancing (SMOTE)")
        self.smoteKSpin = QSpinBox()
        self.smoteKSpin.setRange(2, 30)
        method_layout.addWidget(self.smoteCheck, 2, 0)
        method_layout.addWidget(QLabel("k-neighbors"), 2, 1)
        method_layout.addWidget(self.smoteKSpin, 2, 2)
        method_layout.addWidget(
            self._method_info_button("Apply SMOTE oversampling before training (requires imbalanced-learn)."), 2, 3
        )

        self.classWeightsCheck = QCheckBox("Class weights")
        self.classWeightStrategyCombo = QComboBox()
        self.classWeightStrategyCombo.addItem("balanced")
        self.classWeightStrategyCombo.addItem("uniform")
        method_layout.addWidget(self.classWeightsCheck, 3, 0)
        method_layout.addWidget(QLabel("Strategy"), 3, 1)
        method_layout.addWidget(self.classWeightStrategyCombo, 3, 2)
        method_layout.addWidget(self._method_info_button("Use class weights when training (needs scikit-learn)."), 3, 3)

        self.nestedCvCheck = QCheckBox("Nested cross-validation")
        self.nestedInnerSpin = QSpinBox()
        self.nestedInnerSpin.setRange(2, 20)
        self.nestedOuterSpin = QSpinBox()
        self.nestedOuterSpin.setRange(2, 20)
        method_layout.addWidget(self.nestedCvCheck, 4, 0)
        method_layout.addWidget(QLabel("Inner folds"), 4, 1)
        method_layout.addWidget(self.nestedInnerSpin, 4, 2)
        method_layout.addWidget(QLabel("Outer folds"), 4, 3)
        method_layout.addWidget(self.nestedOuterSpin, 4, 4)
        method_layout.addWidget(self._method_info_button("Run nested cross-validation to avoid leakage."), 4, 5)

        self.reportBundleCheck = QCheckBox("Generate report bundle")
        self.reportBundleCheck.setChecked(True)
        self.openReportCheck = QCheckBox("Open report automatically")
        self.openReportCheck.setChecked(True)
        method_layout.addWidget(self.reportBundleCheck, 5, 0)
        method_layout.addWidget(self.openReportCheck, 5, 1, 1, 2)
        method_layout.addWidget(
            self._method_info_button("Generate the full report bundle and optionally open it immediately."), 5, 3
        )

        method_group.setLayout(method_layout)
        root.addWidget(method_group)

        self.optunaCheck.stateChanged.connect(
            lambda state: self._on_feature_toggle(self.optunaCheck, "optuna", "Optuna hyperparameter search", state)
        )
        self.shapCheck.stateChanged.connect(
            lambda state: self._on_feature_toggle(self.shapCheck, "shap", "SHAP explainability", state)
        )
        self.smoteCheck.stateChanged.connect(
            lambda state: self._on_feature_toggle(self.smoteCheck, "imbalanced-learn", "SMOTE oversampling", state)
        )
        self.classWeightsCheck.stateChanged.connect(
            lambda state: self._on_feature_toggle(
                self.classWeightsCheck, "scikit-learn", "Class weight rebalancing", state
            )
        )

        validation_group = QGroupBox("Validation recipe")
        validation_layout = QHBoxLayout()
        validation_layout.addWidget(QLabel("Train split %"))
        self.splitSpin = QSpinBox()
        self.splitSpin.setRange(10, 100)
        validation_layout.addWidget(self.splitSpin)
        validation_layout.addWidget(QLabel("CV mode"))
        self.cvModeCombo = QComboBox()
        self.cvModeCombo.addItem("RANDOM_SPLIT")
        self.cvModeCombo.addItem("POLYGON_GROUP")
        validation_layout.addWidget(self.cvModeCombo)
        idx = self.cvModeCombo.findText("POLYGON_GROUP")
        if idx >= 0:
            self.cvModeCombo.setCurrentIndex(idx)
        validation_group.setLayout(validation_layout)
        root.addWidget(validation_group)

        root.addWidget(QLabel("Live summary:"))
        self.summaryEdit = QTextEdit()
        self.summaryEdit.setReadOnly(True)
        self.summaryEdit.setMinimumHeight(130)
        root.addWidget(self.summaryEdit)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.cancelBtn = QPushButton("Cancel")
        self.cancelBtn.clicked.connect(self.reject)
        self.runBtn = QPushButton("Run")
        self.runBtn.setToolTip("Save this recipe and kick off a classification run")
        self.runBtn.clicked.connect(self._run_recipe)
        self.createBtn = QPushButton("Create recipe")
        self.createBtn.clicked.connect(self._accept_if_valid)
        button_row.addWidget(self.cancelBtn)
        button_row.addWidget(self.runBtn)
        button_row.addWidget(self.createBtn)
        root.addLayout(button_row)

        for widget in (
            self.optunaCheck,
            self.optunaTrialsSpin,
            self.shapCheck,
            self.shapSampleSpin,
            self.smoteCheck,
            self.smoteKSpin,
            self.classWeightsCheck,
            self.classWeightStrategyCombo,
            self.nestedCvCheck,
            self.nestedInnerSpin,
            self.nestedOuterSpin,
            self.reportBundleCheck,
            self.openReportCheck,
            self.splitSpin,
            self.cvModeCombo,
        ):
            if hasattr(widget, "toggled"):
                widget.toggled.connect(self._update_dynamic_state)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._update_summary)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._update_summary)

        self.setLayout(root)

    def _method_info_button(self, tooltip):
        # type: (str) -> QToolButton
        btn = QToolButton()
        btn.setText("i")
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setAutoRaise(True)
        btn.setFixedSize(18, 18)
        return btn

    def _load_seed_recipe(self):
        # type: () -> None
        self.nameEdit.setText(str(self._recipe.get("name", "")))
        classifier = self._recipe.get("classifier", {})
        code = str(classifier.get("code", "GMM"))
        idx = self.classifierCombo.findData(code)
        if idx < 0:
            idx = 0
        self.classifierCombo.blockSignals(True)
        self.classifierCombo.setCurrentIndex(idx)
        self.classifierCombo.blockSignals(False)
        self._previous_classifier_index = idx

        extra = self._recipe.get("extraParam", {})
        validation = self._recipe.get("validation", {})
        post = self._recipe.get("postprocess", {})

        self.optunaCheck.setChecked(bool(extra.get("USE_OPTUNA", False)))
        self.optunaTrialsSpin.setValue(int(extra.get("OPTUNA_TRIALS", 100)))
        self.shapCheck.setChecked(bool(extra.get("COMPUTE_SHAP", False)))
        self.shapSampleSpin.setValue(int(extra.get("SHAP_SAMPLE_SIZE", 1000)))
        self.smoteCheck.setChecked(bool(extra.get("USE_SMOTE", False)))
        self.smoteKSpin.setValue(int(extra.get("SMOTE_K_NEIGHBORS", 5)))
        self.classWeightsCheck.setChecked(bool(extra.get("USE_CLASS_WEIGHTS", False)))
        strategy = str(extra.get("CLASS_WEIGHT_STRATEGY", "balanced"))
        strategy_idx = self.classWeightStrategyCombo.findText(strategy)
        self.classWeightStrategyCombo.setCurrentIndex(max(0, strategy_idx))
        nested = bool(validation.get("nested_cv", extra.get("USE_NESTED_CV", False)))
        self.nestedCvCheck.setChecked(nested)
        self.nestedInnerSpin.setValue(int(validation.get("nested_inner_cv", extra.get("NESTED_INNER_CV", 3))))
        self.nestedOuterSpin.setValue(int(validation.get("nested_outer_cv", extra.get("NESTED_OUTER_CV", 5))))
        self.reportBundleCheck.setChecked(
            bool(extra.get("GENERATE_REPORT_BUNDLE", False) or post.get("confusion_matrix", False))
        )
        self.openReportCheck.setChecked(bool(extra.get("OPEN_REPORT_IN_BROWSER", True)))
        self.splitSpin.setValue(int(validation.get("split_percent", 75)))
        cv_mode = str(validation.get("cv_mode", extra.get("CV_MODE", "POLYGON_GROUP")))
        cv_idx = self.cvModeCombo.findText(cv_mode)
        self.cvModeCombo.setCurrentIndex(max(0, cv_idx))
        self._update_dynamic_state()
        self._run_requested = False

    def _update_dynamic_state(self):
        # type: () -> None
        self.optunaTrialsSpin.setEnabled(self.optunaCheck.isChecked() and self.optunaCheck.isEnabled())
        self.shapSampleSpin.setEnabled(self.shapCheck.isChecked() and self.shapCheck.isEnabled())
        self.smoteKSpin.setEnabled(self.smoteCheck.isChecked() and self.smoteCheck.isEnabled())
        self.classWeightStrategyCombo.setEnabled(
            self.classWeightsCheck.isChecked() and self.classWeightsCheck.isEnabled()
        )
        nested_enabled = self.nestedCvCheck.isChecked()
        self.nestedInnerSpin.setEnabled(nested_enabled)
        self.nestedOuterSpin.setEnabled(nested_enabled)
        self.openReportCheck.setEnabled(self.reportBundleCheck.isChecked())
        self._update_summary()

    def _missing_dependencies_for_code(self, code):
        # type: (str) -> tuple[List[str], List[str]]
        missing_required = []
        missing_optional = []
        for c, _name, needs_sk, needs_xgb, needs_lgb, needs_cb in _CLASSIFIER_META:
            if c != code:
                continue
            if needs_sk and not self._deps.get("sklearn", False):
                missing_required.append("scikit-learn")
            if needs_xgb and not self._deps.get("xgboost", False):
                missing_required.append("xgboost")
            if needs_lgb and not self._deps.get("lightgbm", False):
                missing_required.append("lightgbm")
            if needs_cb and not self._deps.get("catboost", False):
                missing_required.append("catboost")
            break
        if not self._deps.get("optuna", False):
            missing_optional.append("optuna")
        if not self._deps.get("shap", False):
            missing_optional.append("shap")
        if not self._deps.get("seaborn", False):
            missing_optional.append("seaborn")
        if not self._deps.get("imblearn", False):
            missing_optional.append("imblearn (SMOTE)")
        return missing_required, missing_optional

    def _classifier_name(self, code):
        # type: (str) -> str
        for c, name, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            if c == code:
                return name
        return code

    def _prompt_dependency_install(self, code, missing_required, missing_optional):
        # type: (str, List[str], List[str]) -> bool
        if not missing_required:
            return True
        classifier_name = self._classifier_name(code)
        if not self._installer or not hasattr(self._installer, "_try_install_dependencies"):
            pip_cmd = "python -m pip install dzetsaka[full]"
            QMessageBox.warning(
                self,
                "Dependencies Missing",
                (
                    f"{classifier_name} depends on {', '.join(missing_required)}.\n\n"
                    f"Install packages via PyPI (e.g., run `{pip_cmd}`) and restart QGIS."
                ),
            )
            return False
        req_list = ", ".join(missing_required)
        optional_line = ""
        if missing_optional:
            optional_line = f"Optional missing now: <code>{', '.join(missing_optional)}</code><br>"
        reply = QMessageBox.question(
            self,
            "Dependencies Missing",
            (
                "The selected classifier is not available yet.<br><br>"
                f"Required missing now: <code>{req_list}</code><br>"
                f"{optional_line}"
                f"Full bundle to install: <code>{_full_bundle_label()}</code><br><br>"
                "Install the full dzetsaka dependency bundle now?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        to_install = _full_dependency_bundle()
        if not self._installer._try_install_dependencies(to_install):
            if hasattr(self._installer, "_show_github_issue_popup"):
                self._installer._show_github_issue_popup(
                    "Dependency Installation Failed",
                    "Dependency Installation Error",
                    f"Automatic installation failed for: {', '.join(to_install)}",
                    f"Recipe shop classifier selection: {classifier_name}",
                )
            QMessageBox.warning(
                self,
                "Dependencies Missing",
                "Could not install dependencies required for the selected classifier.",
            )
            return False
        QMessageBox.information(
            self,
            "Installation Successful",
            ("Dependencies installed successfully!<br><br>Please restart QGIS to load the new libraries."),
            QMessageBox.StandardButton.Ok,
        )
        self._deps = check_dependency_availability()
        remaining, _ = self._missing_dependencies_for_code(code)
        if remaining:
            QMessageBox.warning(
                self,
                "Dependencies Missing",
                (f"Still missing packages: {', '.join(remaining)}.\nInstall them manually or restart QGIS."),
            )
            return False
        return True

    def _handle_classifier_changed(self, index):
        # type: (int) -> None
        code = str(self.classifierCombo.itemData(index) or _CLASSIFIER_META[0][0])
        missing_required, missing_optional = self._missing_dependencies_for_code(code)
        if missing_required:
            if not self._prompt_dependency_install(code, missing_required, missing_optional):
                self.classifierCombo.blockSignals(True)
                self.classifierCombo.setCurrentIndex(self._previous_classifier_index)
                self.classifierCombo.blockSignals(False)
                return
        self._previous_classifier_index = index
        self._update_summary()

    def _on_feature_toggle(self, checkbox, dependency_key, feature_name, state):
        # type: (QCheckBox, str, str, int) -> None
        if state != Qt.Checked:
            return
        dep_available = self._deps.get(dependency_key.lower(), False)
        if dep_available:
            return
        installed = self._prompt_install_for_feature(feature_name, [dependency_key])
        if not installed:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def _prompt_install_for_feature(self, feature_name, missing_required):
        # type: (str, List[str]) -> bool
        if not missing_required:
            return True
        if not self._installer or not hasattr(self._installer, "_try_install_dependencies"):
            pip_cmd = "python -m pip install dzetsaka[full]"
            QMessageBox.warning(
                self,
                "Dependencies Missing",
                (
                    f"{feature_name} requires <code>{', '.join(missing_required)}</code>.<br><br>"
                    f"Install via PyPI (e.g., `{pip_cmd}`) and restart QGIS."
                ),
            )
            return False

        req_list = ", ".join(missing_required)
        reply = QMessageBox.question(
            self,
            f"{feature_name} requires additional packages",
            (
                f"{feature_name} depends on <code>{req_list}</code>.<br>"
                f"Full bundle includes: <code>{_full_bundle_label()}</code><br><br>"
                "Install now?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        to_install = _full_dependency_bundle()
        if not self._installer._try_install_dependencies(to_install):
            QMessageBox.warning(
                self,
                "Dependencies Missing",
                (f"Could not install dependencies required for {feature_name}.\nSee logs for details."),
            )
            return False

        QMessageBox.information(
            self,
            "Installation Successful",
            ("Dependencies installed successfully!<br><br>Please restart QGIS to load the new libraries."),
            QMessageBox.StandardButton.Ok,
        )
        self._deps = check_dependency_availability()
        self._update_summary()
        return True

    def _apply_preset(self, preset):
        # type: (Optional[Dict[str, object]]) -> None
        if not isinstance(preset, dict):
            self._active_preset = None
            self._active_preset_label = "Custom"
            self._update_preset_info(None)
            return
        classifier_code = preset.get("classifier")
        if classifier_code:
            target_idx = self.classifierCombo.findData(classifier_code)
            if target_idx >= 0:
                self.classifierCombo.blockSignals(True)
                self.classifierCombo.setCurrentIndex(target_idx)
                self._previous_classifier_index = target_idx
                self.classifierCombo.blockSignals(False)
        self._active_preset = preset
        self._active_preset_label = preset.get("name", "Preset")
        self._set_checkbox(self.optunaCheck, preset.get("optuna", False))
        self.optunaTrialsSpin.setValue(int(preset.get("optuna_trials", self.optunaTrialsSpin.value())))
        self._set_checkbox(self.shapCheck, preset.get("shap", False))
        self.shapSampleSpin.setValue(int(preset.get("shap_sample", self.shapSampleSpin.value())))
        self._set_checkbox(self.smoteCheck, preset.get("smote", False))
        self.smoteKSpin.setValue(int(preset.get("smote_k", self.smoteKSpin.value())))
        self._set_checkbox(self.classWeightsCheck, preset.get("class_weights", False))
        strategy = preset.get("class_weight_strategy", self.classWeightStrategyCombo.currentText())
        idx = self.classWeightStrategyCombo.findText(strategy)
        if idx >= 0:
            self.classWeightStrategyCombo.setCurrentIndex(idx)
        self._set_checkbox(self.nestedCvCheck, preset.get("nested_cv", False))
        self.nestedInnerSpin.setValue(int(preset.get("nested_inner", self.nestedInnerSpin.value())))
        self.nestedOuterSpin.setValue(int(preset.get("nested_outer", self.nestedOuterSpin.value())))
        self.splitSpin.setValue(int(preset.get("split", self.splitSpin.value())))
        cv_mode = preset.get("cv_mode", self.cvModeCombo.currentText())
        cv_idx = self.cvModeCombo.findText(cv_mode)
        if cv_idx >= 0:
            self.cvModeCombo.setCurrentIndex(cv_idx)
        self._set_checkbox(self.reportBundleCheck, preset.get("report_bundle", False))
        self._set_checkbox(self.openReportCheck, preset.get("open_report", False))
        self._update_dynamic_state()
        self._update_preset_info(preset)

    def _update_preset_info(self, preset):
        # type: (Optional[Dict[str, object]]) -> None
        description = ""
        if isinstance(preset, dict):
            description = str(preset.get("description", "")).strip()
        self.presetDescription.setText(description)
        methods_summary = self._preset_methods_summary(preset)
        self.presetMethodsLabel.setText(methods_summary)

    def _on_preset_combo_changed(self, index):
        # type: (int) -> None
        preset = self.presetCombo.itemData(index)
        if not isinstance(preset, dict):
            self._apply_preset(None)
            return
        self._apply_preset(preset)

    def _preset_methods_summary(self, preset):
        # type: (Optional[Dict[str, object]]) -> str
        if not isinstance(preset, dict):
            return "Pick a preset to preview what it enables."
        features = []
        if preset.get("optuna") or preset.get("optuna_trials"):
            features.append("Optuna")
        if preset.get("shap"):
            features.append("SHAP")
        if preset.get("smote"):
            features.append("SMOTE")
        if preset.get("class_weights"):
            features.append("Class weights")
        if preset.get("nested_cv"):
            features.append("Nested CV")
        if preset.get("report_bundle"):
            features.append("Report bundle")
        cv = preset.get("cv_mode", "Random split").replace("_", " ").capitalize()
        split = preset.get("split", "")
        if split:
            features.append(f"{split}% split")
        if cv:
            features.append(cv)
        return " · ".join(features) if features else "Core model only."

    def _set_checkbox(self, checkbox, value):
        # type: (QCheckBox, bool) -> None
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(value))
        checkbox.blockSignals(False)

    def _on_name_edited(self, _text):
        # type: (str) -> None
        self._user_edited_name = True

    def _try_auto_fill_name(self):
        # type: () -> None
        if self._user_edited_name:
            return
        new_name = self._build_auto_name()
        if not new_name or new_name == self._last_auto_name:
            return
        current_name = self.nameEdit.text().strip()
        if current_name and current_name != self._last_auto_name:
            return
        self._last_auto_name = new_name
        self.nameEdit.blockSignals(True)
        self.nameEdit.setText(new_name)
        self.nameEdit.blockSignals(False)

    def _build_auto_name(self):
        # type: () -> str
        classifier_name = self._current_classifier_name()
        preset_label = self._active_preset_label or "Custom"
        tags = []
        if self.optunaCheck.isChecked():
            tags.append("Optuna")
        if self.shapCheck.isChecked():
            tags.append("SHAP")
        if self.smoteCheck.isChecked():
            tags.append("SMOTE")
        if self.classWeightsCheck.isChecked():
            tags.append("Class weights")
        if self.nestedCvCheck.isChecked():
            tags.append("Nested CV")
        if not tags:
            tags.append("Core")
        cv_mode = self.cvModeCombo.currentText().replace("_", " ").title()
        return f"{preset_label} · {classifier_name} · {', '.join(tags)} · {self.splitSpin.value()}% {cv_mode}"

    def _current_classifier_name(self):
        # type: () -> str
        code = self.classifierCombo.itemData(self.classifierCombo.currentIndex())
        code = str(code or _CLASSIFIER_META[0][0])
        for c, name, *_ in _CLASSIFIER_META:
            if c == code:
                return name
        return code

    def _update_preset_description(self, preset):
        # type: (Optional[Dict[str, object]]) -> None
        description = ""
        if isinstance(preset, dict):
            description = str(preset.get("description", "")).strip()
        self.presetDescription.setText(description)

    def _build_recipe(self):
        # type: () -> Dict[str, object]
        base = normalize_recipe(_recipe_template())
        classifier_code = str(self.classifierCombo.currentData() or "GMM")

        extra = dict(base.get("extraParam", {}))
        extra.update(
            {
                "USE_OPTUNA": bool(self.optunaCheck.isChecked() and self.optunaCheck.isEnabled()),
                "OPTUNA_TRIALS": int(self.optunaTrialsSpin.value()),
                "COMPUTE_SHAP": bool(self.shapCheck.isChecked() and self.shapCheck.isEnabled()),
                "SHAP_OUTPUT": "",
                "SHAP_SAMPLE_SIZE": int(self.shapSampleSpin.value()),
                "USE_SMOTE": bool(self.smoteCheck.isChecked() and self.smoteCheck.isEnabled()),
                "SMOTE_K_NEIGHBORS": int(self.smoteKSpin.value()),
                "USE_CLASS_WEIGHTS": bool(self.classWeightsCheck.isChecked() and self.classWeightsCheck.isEnabled()),
                "CLASS_WEIGHT_STRATEGY": str(self.classWeightStrategyCombo.currentText()),
                "USE_NESTED_CV": bool(self.nestedCvCheck.isChecked()),
                "NESTED_INNER_CV": int(self.nestedInnerSpin.value()),
                "NESTED_OUTER_CV": int(self.nestedOuterSpin.value()),
                "CV_MODE": str(self.cvModeCombo.currentText()),
                "GENERATE_REPORT_BUNDLE": bool(self.reportBundleCheck.isChecked()),
                "OPEN_REPORT_IN_BROWSER": bool(self.openReportCheck.isChecked()),
            }
        )

        recipe = {
            "name": self.nameEdit.text().strip(),
            "classifier": {"code": classifier_code},
            "preprocessing": {},
            "features": {"bands": "all"},
            "postprocess": {
                "confidence_map": False,
                "save_model": False,
                "confusion_matrix": bool(self.reportBundleCheck.isChecked()),
            },
            "validation": {
                "split_percent": int(self.splitSpin.value()),
                "nested_cv": bool(self.nestedCvCheck.isChecked()),
                "nested_inner_cv": int(self.nestedInnerSpin.value()),
                "nested_outer_cv": int(self.nestedOuterSpin.value()),
                "cv_mode": str(self.cvModeCombo.currentText()),
            },
            "extraParam": extra,
        }
        return normalize_recipe(recipe)

    def _update_summary(self):
        # type: () -> None
        recipe = self._build_recipe()
        self._recipe = recipe
        self.summaryEdit.setPlainText(format_recipe_summary(recipe))
        self._try_auto_fill_name()

    def _accept_if_valid(self, run_after=False):
        # type: () -> None
        name = self.nameEdit.text().strip()
        if not name:
            QMessageBox.warning(self, "Recipe name required", "Please provide a name for this recipe.")
            return
        self._recipe = self._build_recipe()
        self._run_requested = bool(run_after)
        self.accept()

    def _run_recipe(self):
        self._accept_if_valid(run_after=True)

    def selected_recipe(self):
        # type: () -> Dict[str, object]
        return normalize_recipe(dict(self._recipe))

    def run_requested(self):
        # type: () -> bool
        return bool(self._run_requested)


class RecipeGalleryDialog(QDialog):
    """Dialog that shows local recipe gallery."""

    recipeApplied = pyqtSignal(dict)
    recipesUpdated = pyqtSignal(list)

    def __init__(self, parent=None, recipes=None, remote_url=""):
        super(RecipeGalleryDialog, self).__init__(parent)
        self.setWindowTitle("Recipe Gallery")
        self.setMinimumSize(720, 420)

        self._local_recipes = recipes or []
        self._remote_url = remote_url
        self._remote_recipes = []

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Browse and manage local recipes."))

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_local_tab(), "Local")
        layout.addWidget(self.tabs)

        button_row = QHBoxLayout()
        self.useBtn = QPushButton("Use Selected")
        self.useBtn.clicked.connect(self._use_selected)
        self.closeBtn = QPushButton("Close")
        self.closeBtn.clicked.connect(self.close)
        button_row.addStretch()
        button_row.addWidget(self.useBtn)
        button_row.addWidget(self.closeBtn)
        layout.addLayout(button_row)

        self.setLayout(layout)
        self._populate_local()

    def _build_local_tab(self):
        # type: () -> QWidget
        widget = QWidget()
        layout = QVBoxLayout()

        # Filter tabs
        filter_row = QHBoxLayout()
        self.filterAllBtn = QPushButton("All")
        self.filterAllBtn.setCheckable(True)
        self.filterAllBtn.setChecked(True)
        self.filterAllBtn.clicked.connect(lambda: self._set_filter("all"))
        filter_row.addWidget(self.filterAllBtn)

        self.filterTemplatesBtn = QPushButton("Templates")
        self.filterTemplatesBtn.setCheckable(True)
        self.filterTemplatesBtn.clicked.connect(lambda: self._set_filter("templates"))
        filter_row.addWidget(self.filterTemplatesBtn)

        self.filterUserBtn = QPushButton("My Recipes")
        self.filterUserBtn.setCheckable(True)
        self.filterUserBtn.clicked.connect(lambda: self._set_filter("user"))
        filter_row.addWidget(self.filterUserBtn)

        filter_row.addStretch()
        layout.addLayout(filter_row)

        self.localList = QListWidget()
        self.localList.currentItemChanged.connect(self._on_local_selected)
        layout.addWidget(self.localList)

        self.localSummary = QTextEdit()
        self.localSummary.setReadOnly(True)
        self.localSummary.setMinimumHeight(120)
        layout.addWidget(self.localSummary)

        btn_row = QHBoxLayout()
        self.exportBtn = QPushButton("Export JSON…")
        self.exportBtn.clicked.connect(self._export_selected_local)
        btn_row.addWidget(self.exportBtn)

        self.copyTemplateBtn = QPushButton("Copy as User Recipe…")
        self.copyTemplateBtn.clicked.connect(self._copy_template_to_user)
        btn_row.addWidget(self.copyTemplateBtn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        widget.setLayout(layout)
        self._current_filter = "all"
        return widget

    def _build_remote_tab(self):
        # type: () -> QWidget
        widget = QWidget()
        layout = QVBoxLayout()

        url_row = QHBoxLayout()
        self.remoteUrlLabel = QLabel(self._remote_url or "<no remote URL>")
        url_row.addWidget(QLabel("Remote URL:"))
        url_row.addWidget(self.remoteUrlLabel)
        url_row.addStretch()
        self.setUrlBtn = QPushButton("Set URL…")
        self.setUrlBtn.clicked.connect(self._set_remote_url)
        url_row.addWidget(self.setUrlBtn)
        self.schemaBtn = QPushButton("Schema")
        self.schemaBtn.clicked.connect(self._show_schema)
        url_row.addWidget(self.schemaBtn)
        self.refreshBtn = QPushButton("Refresh")
        self.refreshBtn.clicked.connect(self._refresh_remote)
        url_row.addWidget(self.refreshBtn)
        layout.addLayout(url_row)

        self.remoteList = QListWidget()
        self.remoteList.currentItemChanged.connect(self._on_remote_selected)
        layout.addWidget(self.remoteList)

        self.remoteSummary = QTextEdit()
        self.remoteSummary.setReadOnly(True)
        self.remoteSummary.setMinimumHeight(120)
        layout.addWidget(self.remoteSummary)

        btn_row = QHBoxLayout()
        self.saveRemoteBtn = QPushButton("Save To Local")
        self.saveRemoteBtn.clicked.connect(self._save_remote_to_local)
        btn_row.addWidget(self.saveRemoteBtn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        widget.setLayout(layout)
        return widget

    def _set_filter(self, filter_type):
        # type: (str) -> None
        """Set the current filter and update button states."""
        self._current_filter = filter_type
        self.filterAllBtn.setChecked(filter_type == "all")
        self.filterTemplatesBtn.setChecked(filter_type == "templates")
        self.filterUserBtn.setChecked(filter_type == "user")
        self._populate_local()

    def _populate_local(self):
        # type: () -> None
        self.localList.clear()
        for recipe in self._local_recipes:
            is_template = recipe.get("is_template", False)

            # Apply filter
            if self._current_filter == "templates" and not is_template:
                continue
            if self._current_filter == "user" and is_template:
                continue

            name = recipe.get("name", "Unnamed Recipe")
            icon = "📦" if is_template else "⚙️"
            display_name = f"{icon} {name}"

            item = QListWidgetItem(display_name)

            # Color coding: templates with light blue, user recipes with light green
            if is_template:
                item.setBackground(QColor(230, 240, 255))  # Light blue
            else:
                item.setBackground(QColor(230, 255, 230))  # Light green

            self.localList.addItem(item)

    def _populate_remote(self):
        # type: () -> None
        self.remoteList.clear()
        for recipe in self._remote_recipes:
            name = recipe.get("name", "Unnamed Recipe")
            self.remoteList.addItem(QListWidgetItem(name))

    def _on_local_selected(self, current, _previous=None):
        # type: (QListWidgetItem, Optional[QListWidgetItem]) -> None
        if current is None:
            self.localSummary.setPlainText("")
            self.copyTemplateBtn.setEnabled(False)
            return

        # Strip emoji prefix to find recipe
        display_text = current.text()
        for prefix in ["📦 ", "⚙️ "]:
            if display_text.startswith(prefix):
                display_text = display_text[len(prefix):]
                break

        recipe = self._find_recipe(self._local_recipes, display_text)
        self.localSummary.setPlainText(format_recipe_summary(recipe) if recipe else "")

        # Enable copy button only for templates
        is_template = recipe.get("is_template", False) if recipe else False
        self.copyTemplateBtn.setEnabled(is_template)

    def _on_remote_selected(self, current, _previous=None):
        # type: (QListWidgetItem, Optional[QListWidgetItem]) -> None
        if current is None:
            self.remoteSummary.setPlainText("")
            return
        recipe = self._find_recipe(self._remote_recipes, current.text())
        self.remoteSummary.setPlainText(format_recipe_summary(recipe) if recipe else "")

    def _find_recipe(self, recipes, name):
        # type: (List[Dict[str, object]], str) -> Optional[Dict[str, object]]
        for recipe in recipes:
            if recipe.get("name") == name:
                return recipe
        return None

    def _use_selected(self):
        # type: () -> None
        item = self.localList.currentItem()
        if item is None:
            return

        # Strip emoji prefix to find recipe
        display_text = item.text()
        for prefix in ["📦 ", "⚙️ "]:
            if display_text.startswith(prefix):
                display_text = display_text[len(prefix):]
                break

        recipe = self._find_recipe(self._local_recipes, display_text)
        if recipe:
            self.recipeApplied.emit(recipe)
            self.close()
            return
        if self.tabs.currentIndex() != 0:
            item = self.remoteList.currentItem()
            if item is None:
                return
            recipe = self._find_recipe(self._remote_recipes, item.text())
        if recipe:
            self.recipeApplied.emit(recipe)
            self.close()

    def _copy_template_to_user(self):
        # type: () -> None
        """Create a user recipe copy from a selected template."""
        item = self.localList.currentItem()
        if item is None:
            return

        # Strip emoji prefix to find recipe
        display_text = item.text()
        for prefix in ["📦 ", "⚙️ "]:
            if display_text.startswith(prefix):
                display_text = display_text[len(prefix):]
                break

        recipe = self._find_recipe(self._local_recipes, display_text)
        if not recipe or not recipe.get("is_template", False):
            return

        # Ask for new name
        original_name = recipe.get("name", "Unnamed Recipe")
        new_name, ok = QInputDialog.getText(
            self,
            "Copy Template",
            f"Create a user recipe based on '{original_name}'.\n\nEnter a new name:",
            text=f"{original_name} (Custom)",
        )

        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()

        # Check if name already exists
        if any(r.get("name") == new_name for r in self._local_recipes):
            QMessageBox.warning(
                self, "Name Conflict", f"A recipe named '{new_name}' already exists. Please choose a different name."
            )
            return

        # Create copy without template flag
        user_recipe = dict(recipe)
        user_recipe["name"] = new_name
        user_recipe["is_template"] = False
        user_recipe.pop("category", None)
        user_recipe.pop("expected_runtime", None)
        user_recipe.pop("typical_accuracy", None)
        user_recipe.pop("best_for", None)
        user_recipe["description"] = f"Based on template: {original_name}"

        # Add to local recipes
        self._local_recipes.append(user_recipe)
        self.recipesUpdated.emit(self._local_recipes)
        self._populate_local()

        QMessageBox.information(
            self, "Template Copied", f"User recipe '{new_name}' created successfully from template '{original_name}'."
        )

    def _export_selected_local(self):
        # type: () -> None
        item = self.localList.currentItem()
        if item is None:
            return

        # Strip emoji prefix to find recipe
        display_text = item.text()
        for prefix in ["📦 ", "⚙️ "]:
            if display_text.startswith(prefix):
                display_text = display_text[len(prefix):]
                break

        recipe = self._find_recipe(self._local_recipes, display_text)
        if recipe is None:
            return
        path, _f = QFileDialog.getSaveFileName(self, "Export recipe", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(recipe, handle, indent=2)
        except Exception as exc:
            installer = getattr(self.parent(), "_installer", None)
            _show_issue_popup(
                self,
                installer,
                "Export failed",
                type(exc).__name__,
                str(exc),
                "Recipe gallery: export selected local recipe",
            )

    def _set_remote_url(self):
        # type: () -> None
        url, ok = QInputDialog.getText(
            self, "Remote Recipe URL", "Enter a URL that returns recipe JSON:", text=self._remote_url
        )
        if not ok:
            return
        self._remote_url = url.strip()
        self.remoteUrlLabel.setText(self._remote_url or "<no remote URL>")
        self.remoteUrlUpdated.emit(self._remote_url)

    def _show_schema(self):
        # type: () -> None
        dialog = QDialog(self)
        dialog.setWindowTitle("Remote Recipe Schema")
        dialog.setMinimumSize(520, 360)
        layout = QVBoxLayout()
        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(_RECIPE_SCHEMA_TEXT)
        layout.addWidget(info)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)
        dialog.setLayout(layout)
        try:
            dialog.exec_()
        except AttributeError:
            dialog.exec()

    def _refresh_remote(self):
        # type: () -> None
        if not self._remote_url:
            QMessageBox.information(self, "Remote gallery", "Set a remote URL to fetch shared recipes.")
            return
        try:
            with urllib.request.urlopen(self._remote_url, timeout=4) as response:
                data = response.read().decode("utf-8")
            payload = json.loads(data)
            recipes, errors = validate_recipe_list(payload)
            if errors:
                installer = getattr(self.parent(), "_installer", None)
                _show_issue_popup(
                    self,
                    installer,
                    "Remote gallery validation errors",
                    "Recipe Validation Error",
                    "\n".join(errors[:8]),
                    f"Remote URL: {self._remote_url}",
                )
            self._remote_recipes = recipes
            self._populate_remote()
            self._refresh_remote_summary()
        except Exception as exc:
            installer = getattr(self.parent(), "_installer", None)
            _show_issue_popup(
                self,
                installer,
                "Remote gallery load failed",
                type(exc).__name__,
                str(exc),
                f"Remote URL: {self._remote_url}",
            )

    def _refresh_remote_summary(self):
        # type: () -> None
        self.remoteSummary.setPlainText(
            "Select a remote recipe to see its details." if not self._remote_recipes else ""
        )

    def _save_remote_to_local(self):
        # type: () -> None
        item = self.remoteList.currentItem()
        if item is None:
            return
        recipe = self._find_recipe(self._remote_recipes, item.text())
        if recipe is None:
            return
        self._local_recipes = [r for r in self._local_recipes if r.get("name") != recipe.get("name")]
        self._local_recipes.append(recipe)
        self._populate_local()
        self.recipesUpdated.emit(self._local_recipes)


# ---------------------------------------------------------------------------
# Geometry explorer dialog
# ---------------------------------------------------------------------------


class VectorInsightDialog(QDialog):
    """Dialog showing vector geometry insight and tips."""

    def __init__(self, parent=None, vector_path="", class_field="", layer=None):
        super(VectorInsightDialog, self).__init__(parent)
        self.setWindowTitle("Geometry Explorer")
        self.setMinimumSize(640, 360)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Vector Insight & Tips"))
        self.summaryEdit = QTextEdit()
        self.summaryEdit.setReadOnly(True)
        layout.addWidget(self.summaryEdit)

        self._populate(vector_path, class_field, layer)
        self.setLayout(layout)

    def _populate(self, vector_path, class_field, layer):
        # type: (str, str, object) -> None
        summary, tips = _collect_vector_insight(vector_path, class_field, layer)
        text = summary
        if tips:
            text += "\n\nTips:\n" + "\n".join(f"- {tip}" for tip in tips)
        self.summaryEdit.setPlainText(text)


def _collect_vector_insight(vector_path, class_field, layer=None):
    # type: (str, str, object) -> tuple
    """Return summary text and tip list for a vector layer/path."""
    lines = []
    tips = []
    max_samples = 10000
    counts = Counter()
    nulls = 0
    total = None
    sampled = 0
    geom_name = "Unknown"
    fields = []

    if layer is not None:
        try:
            from qgis.core import QgsWkbTypes

            geom_name = QgsWkbTypes.displayString(layer.wkbType())
            total = layer.featureCount()
            fields = [f.name() for f in layer.fields()]
            if class_field and class_field in fields:
                for feat in layer.getFeatures():
                    if sampled >= max_samples:
                        break
                    value = feat[class_field]
                    sampled += 1
                    if value in (None, ""):
                        nulls += 1
                    else:
                        counts[str(value)] += 1
        except Exception:
            layer = None

    if layer is None and vector_path:
        try:
            from osgeo import ogr
        except ImportError:
            try:
                import ogr  # type: ignore[no-redef]
            except ImportError:
                ogr = None
        if ogr is not None:
            ds = ogr.Open(vector_path)
            if ds is not None:
                lyr = ds.GetLayer()
                if lyr is not None:
                    try:
                        geom_name = ogr.GeometryTypeToName(lyr.GetGeomType())
                    except Exception:
                        geom_name = "Unknown"
                    total = lyr.GetFeatureCount()
                    defn = lyr.GetLayerDefn()
                    fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
                    if class_field and class_field in fields:
                        for feat in lyr:
                            if sampled >= max_samples:
                                break
                            value = feat.GetField(class_field)
                            sampled += 1
                            if value in (None, ""):
                                nulls += 1
                            else:
                                counts[str(value)] += 1

    lines.append(f"Vector: {vector_path or '<from layer>'}")
    lines.append(f"Geometry: {geom_name}")
    if total is not None:
        lines.append(f"Features: {total}")
    if fields:
        lines.append(f"Fields: {', '.join(fields[:12])}{'…' if len(fields) > 12 else ''}")
    if class_field:
        lines.append(f"Label field: {class_field}")
    if counts:
        top = counts.most_common(5)
        lines.append("Top classes (sampled): " + ", ".join([f"{k}={v}" for k, v in top]))
        total_sampled = sum(counts.values()) + nulls
        if total_sampled:
            null_ratio = nulls / float(total_sampled)
            if null_ratio > 0.05:
                tips.append("More than 5% of class values are missing; fill or remove null labels.")
            if len(counts) >= 2:
                max_c = max(counts.values())
                min_c = min(counts.values())
                if min_c > 0 and max_c / float(min_c) >= 5.0:
                    tips.append("Strong class imbalance detected; consider class weights or SMOTE.")
    else:
        if class_field:
            tips.append("Label field has no readable values yet; verify field name and data types.")
        else:
            tips.append("Pick a label field to see class distribution and imbalance tips.")

    if total is not None and total > max_samples:
        tips.append(f"Stats are based on the first {max_samples} features.")

    return "\n".join(lines), tips


# ---------------------------------------------------------------------------
# Page 0 — Input & Algorithm
# ---------------------------------------------------------------------------


class DataInputPage(QWizardPage):
    """Workflow page for selecting input data and classifier."""

    def __init__(self, parent=None, deps=None, installer=None):
        """Initialise DataInputPage."""
        super(DataInputPage, self).__init__(parent)
        self.setTitle("Input and Model")
        self.setSubTitle("Select data, model source, and classifier.")

        self._deps = deps if deps is not None else check_dependency_availability()
        self._installer = installer
        self._smart_defaults_applied = False  # type: bool
        self._last_prompt_signature = None  # type: Optional[tuple]
        self._suppress_dependency_prompt = False  # type: bool
        self._recipes = []  # type: List[Dict[str, object]]

        layout = QVBoxLayout()

        # Add What's This help button to title bar
        help_btn_layout = QHBoxLayout()
        help_btn_layout.addStretch()
        help_btn = QPushButton("?")
        help_btn.setToolTip("Click for detailed help on this page")
        help_btn.setMaximumWidth(30)
        help_btn.clicked.connect(lambda: QWhatsThis.enterWhatsThisMode())
        help_btn_layout.addWidget(help_btn)
        layout.addLayout(help_btn_layout)

        # --- Recipe group ---
        recipe_group = QGroupBox("Recipe")
        recipe_layout = QGridLayout()
        recipe_layout.addWidget(QLabel("Recipe:"), 0, 0)
        self.recipeCombo = QComboBox()
        self.recipeCombo.setToolTip(
            "<b>Classification Recipe</b><br>"
            "Pre-configured classifier preset including algorithm, parameters, and advanced features.<br><br>"
            "<i>Use:</i> Load a recipe to quickly configure optimal settings for common use cases."
        )
        self.recipeCombo.setWhatsThis(
            "<h3>Classification Recipe</h3>"
            "<p>A <b>recipe</b> is a pre-configured classification workflow that bundles together:</p>"
            "<ul>"
            "<li>Algorithm selection (Random Forest, XGBoost, etc.)</li>"
            "<li>Algorithm parameters (tree depth, learning rate, etc.)</li>"
            "<li>Advanced features (Optuna, SHAP, SMOTE)</li>"
            "<li>Validation settings (CV mode, split percentage)</li>"
            "</ul>"
            "<h4>When to use:</h4>"
            "<ul>"
            "<li>You want to quickly apply proven settings for common tasks</li>"
            "<li>You're new to machine learning and want expert-recommended configurations</li>"
            "<li>You want to reproduce results from a previous classification</li>"
            "<li>You've received a recipe from a colleague or online</li>"
            "</ul>"
            "<h4>When to skip:</h4>"
            "<ul>"
            "<li>You're experimenting with custom parameters</li>"
            "<li>Your use case doesn't match available recipes</li>"
            "<li>You prefer to configure everything manually</li>"
            "</ul>"
            "<h4>Tips:</h4>"
            "<ul>"
            "<li>The <b>Gallery</b> button shows curated recipes with descriptions</li>"
            "<li>Use <b>Save Current</b> to create custom recipes from your settings</li>"
            "<li>Recipes can be shared as JSON files for reproducibility</li>"
            "</ul>"
        )
        recipe_layout.addWidget(self.recipeCombo, 0, 1, 1, 3)

        self.recipeApplyBtn = QPushButton("Apply")
        self.recipeApplyBtn.clicked.connect(self._apply_selected_recipe)
        recipe_layout.addWidget(self.recipeApplyBtn, 1, 1)

        self.recipeSaveBtn = QPushButton("Save Current…")
        self.recipeSaveBtn.clicked.connect(self._save_current_recipe)
        recipe_layout.addWidget(self.recipeSaveBtn, 1, 2)

        self.recipeGalleryBtn = QPushButton("Gallery…")
        self.recipeGalleryBtn.clicked.connect(self._open_recipe_gallery)
        recipe_layout.addWidget(self.recipeGalleryBtn, 1, 3)

        self.recipeLoadJsonBtn = QPushButton("Load JSON…")
        self.recipeLoadJsonBtn.clicked.connect(self._load_json_config)
        recipe_layout.addWidget(self.recipeLoadJsonBtn, 2, 1)

        self.recipePasteJsonBtn = QPushButton("Paste JSON…")
        self.recipePasteJsonBtn.clicked.connect(self._paste_json_config)
        recipe_layout.addWidget(self.recipePasteJsonBtn, 2, 2)

        recipe_group.setLayout(recipe_layout)
        layout.addWidget(recipe_group)

        # --- Input group ---
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()

        input_layout.addWidget(QLabel("Raster to classify (GeoTIFF):"))
        raster_row = QHBoxLayout()
        self.rasterLineEdit = QLineEdit()
        self.rasterLineEdit.setPlaceholderText("Path to raster file…")
        raster_row.addWidget(self.rasterLineEdit)
        self.rasterBrowse = QPushButton("Browse…")
        self.rasterBrowse.clicked.connect(self._browse_raster)
        raster_row.addWidget(self.rasterBrowse)
        input_layout.addLayout(raster_row)

        # Try to use QgsMapLayerComboBox when QGIS is available
        self._raster_combo = None  # type: Optional[QWidget]
        try:
            from qgis.core import QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._raster_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            if "gdal" in exclude:
                exclude.remove("gdal")
            self._raster_combo.setExcludedProviders(exclude)
            input_layout.addWidget(self._raster_combo)
            self.rasterLineEdit.setVisible(False)
            self.rasterBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass  # fallback: plain QLineEdit stays visible

        input_layout.addWidget(QLabel("Training data (vector):"))
        vector_row = QHBoxLayout()
        self.vectorLineEdit = QLineEdit()
        self.vectorLineEdit.setPlaceholderText("Path to vector file…")
        vector_row.addWidget(self.vectorLineEdit)
        self.vectorBrowse = QPushButton("Browse…")
        self.vectorBrowse.clicked.connect(self._browse_vector)
        vector_row.addWidget(self.vectorBrowse)
        input_layout.addLayout(vector_row)

        self._vector_combo = None  # type: Optional[QWidget]
        try:
            from qgis.core import QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._vector_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            if "ogr" in exclude:
                exclude.remove("ogr")
            self._vector_combo.setExcludedProviders(exclude)
            input_layout.addWidget(self._vector_combo)
            self.vectorLineEdit.setVisible(False)
            self.vectorBrowse.setVisible(False)
            self._vector_combo.currentIndexChanged.connect(self._on_vector_changed)
            if hasattr(self._vector_combo, "layerChanged"):
                self._vector_combo.layerChanged.connect(self._on_vector_changed)
        except (ImportError, AttributeError):
            pass

        input_layout.addWidget(QLabel("Label field:"))
        self.classFieldCombo = QComboBox()
        self.classFieldCombo.setToolTip(
            "<b>Class Label Field</b><br>"
            "Vector attribute field containing class codes for each training polygon/point.<br><br>"
            "<i>Required:</i> Numeric field with integer values (1, 2, 3, etc.)."
        )
        input_layout.addWidget(self.classFieldCombo)
        self.fieldStatusLabel = QLabel()
        input_layout.addWidget(self.fieldStatusLabel)

        # Add Check Data Quality button
        self.checkQualityBtn = QPushButton("Check Data Quality…")
        self.checkQualityBtn.setToolTip(
            "<b>Check Training Data Quality</b><br>"
            "Analyze your training data for common issues before classification:<br>"
            "• Class imbalance<br>"
            "• Insufficient samples<br>"
            "• Invalid geometries<br>"
            "• Spatial clustering"
        )
        # Try to set icon
        quality_icon = QIcon(":/plugins/dzetsaka/img/table.png")
        if quality_icon.isNull():
            # Fallback to vector icon if table icon not found
            quality_icon = QIcon(":/plugins/dzetsaka/img/vector.svg")
        if not quality_icon.isNull():
            self.checkQualityBtn.setIcon(quality_icon)
        self.checkQualityBtn.clicked.connect(self._check_data_quality)
        self.checkQualityBtn.setEnabled(_QUALITY_CHECKER_AVAILABLE)
        if not _QUALITY_CHECKER_AVAILABLE:
            self.checkQualityBtn.setToolTip("Training data quality checker not available (module import failed)")
        input_layout.addWidget(self.checkQualityBtn)

        self.geometryExplorerBtn = QPushButton("Geometry Explorer…")
        self.geometryExplorerBtn.clicked.connect(self._open_geometry_explorer)
        input_layout.addWidget(self.geometryExplorerBtn)

        self.loadModelCheck = QCheckBox("Use existing model")
        self.loadModelCheck.setToolTip(
            "<b>Use Existing Model</b><br>"
            "Load a previously trained classifier instead of training a new one.<br><br>"
            "<i>When to use:</i> Classify multiple rasters with same model, reuse trusted model.<br>"
            "<i>Note:</i> Model must match the number of raster bands."
        )
        self.loadModelCheck.setWhatsThis(
            "<h3>Use Existing Model</h3>"
            "<p>Load a previously trained classifier (.pkl file) to classify a raster without training a new model.</p>"
            "<h4>When to use:</h4>"
            "<ul>"
            "<li><b>Classify multiple rasters:</b> Train once, apply to multiple images with same band structure</li>"
            "<li><b>Save time:</b> Skip training when you already have a good model</li>"
            "<li><b>Reproducibility:</b> Reuse exact same model for consistent results</li>"
            "<li><b>Share models:</b> Use models trained by colleagues or from literature</li>"
            "</ul>"
            "<h4>When to skip:</h4>"
            "<ul>"
            "<li>Your training data has changed (new classes, different spectral characteristics)</li>"
            "<li>The raster has a different number of bands than the training data</li>"
            "<li>You want to experiment with different algorithms or parameters</li>"
            "</ul>"
            "<h4>Important notes:</h4>"
            "<ul>"
            "<li><b>Band matching:</b> The model expects the same number and order of bands as training data</li>"
            "<li><b>Spectral consistency:</b> Best results when raster has similar spectral characteristics (sensor, season, preprocessing)</li>"
            "<li><b>Model format:</b> dzetsaka saves models as pickled scikit-learn/XGBoost/LightGBM/CatBoost files (.pkl)</li>"
            "</ul>"
        )
        input_layout.addWidget(self.loadModelCheck)

        model_row = QHBoxLayout()
        self.modelLineEdit = QLineEdit()
        self.modelLineEdit.setPlaceholderText("Model file path…")
        self.modelLineEdit.setEnabled(False)
        self.modelLineEdit.setToolTip(
            "<b>Model File Path</b><br>"
            "Path to saved classifier model (.pkl file) from previous dzetsaka training."
        )
        model_row.addWidget(self.modelLineEdit)
        self.modelBrowse = QPushButton("Browse…")
        self.modelBrowse.setEnabled(False)
        self.modelBrowse.clicked.connect(self._browse_model)
        model_row.addWidget(self.modelBrowse)
        input_layout.addLayout(model_row)

        self.loadModelCheck.toggled.connect(self._toggle_model_mode)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- Algorithm group ---
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout()

        algo_layout.addWidget(QLabel("Classifier:"))
        self.classifierCombo = QComboBox()
        for _code, name, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            self.classifierCombo.addItem(name)
        self.classifierCombo.setWhatsThis(
            "<h3>Classifier Algorithm</h3>"
            "<p>Choose the machine learning algorithm to train your classification model. "
            "dzetsaka supports 12 algorithms with different strengths and trade-offs.</p>"
            "<h4>Popular choices:</h4>"
            "<ul>"
            "<li><b>Random Forest:</b> Balanced accuracy, fast training, good default choice for most cases</li>"
            "<li><b>XGBoost/LightGBM/CatBoost:</b> State-of-the-art accuracy, excellent with Optuna optimization</li>"
            "<li><b>SVM:</b> High accuracy for smaller datasets, slower on large datasets</li>"
            "<li><b>GMM:</b> Fast, probabilistic, good for quick exploration</li>"
            "</ul>"
            "<h4>When to use each:</h4>"
            "<ul>"
            "<li><b>Experimenting:</b> Start with Random Forest - fast, robust, good baseline</li>"
            "<li><b>Best accuracy:</b> Try XGBoost/LightGBM with Optuna optimization</li>"
            "<li><b>Small datasets (<1000 samples):</b> SVM or Random Forest</li>"
            "<li><b>Large datasets (>100k samples):</b> LightGBM, Random Forest, or Neural Network (MLP)</li>"
            "<li><b>Fast preview:</b> GMM or KNN</li>"
            "</ul>"
            "<h4>Dependencies:</h4>"
            "<ul>"
            "<li><b>Always available:</b> GMM, KNN, SVM, Random Forest (via scikit-learn)</li>"
            "<li><b>Require installation:</b> XGBoost, LightGBM, CatBoost (dzetsaka can auto-install)</li>"
            "</ul>"
            "<p><i>Tip:</i> Use the <b>Compare</b> button to see algorithm characteristics side-by-side, "
            "or click <b>Smart Defaults</b> to enable recommended advanced features.</p>"
        )
        algo_layout.addWidget(self.classifierCombo)

        self.depStatusLabel = QLabel()
        self.depStatusLabel.setVisible(False)
        algo_layout.addWidget(self.depStatusLabel)
        self.classifierCombo.currentIndexChanged.connect(self._update_dep_status)
        self._update_dep_status(0)

        btn_row = QHBoxLayout()
        self.smartDefaultsBtn = QPushButton("Smart Defaults")
        self.smartDefaultsBtn.setToolTip("Enable Optuna / SHAP / SMOTE when packages are available.")
        self.smartDefaultsBtn.clicked.connect(self._apply_smart_defaults)
        btn_row.addWidget(self.smartDefaultsBtn)

        self.compareBtn = QPushButton("Compare…")
        self.compareBtn.setToolTip("Open the algorithm comparison panel.")
        self.compareBtn.clicked.connect(self._open_comparison)
        btn_row.addWidget(self.compareBtn)
        btn_row.addStretch()
        algo_layout.addLayout(btn_row)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        layout.addStretch()

        # Register guided workflow fields
        self.registerField("raster", self.rasterLineEdit)
        self.registerField("vector", self.vectorLineEdit)
        self.registerField("loadModel", self.modelLineEdit)

        self.vectorLineEdit.editingFinished.connect(self._on_vector_path_edited)
        if self._vector_combo is not None:
            self._on_vector_changed()

        self.setLayout(layout)

    # --- internal helpers --------------------------------------------------

    def _toggle_model_mode(self, checked):
        # type: (bool) -> None
        """Enable / disable the model path widgets and vector fields."""
        self.modelLineEdit.setEnabled(checked)
        self.modelBrowse.setEnabled(checked)
        if checked:
            self.vectorLineEdit.setEnabled(False)
            self.classFieldCombo.setEnabled(False)
            if self._vector_combo is not None:
                self._vector_combo.setEnabled(False)
        else:
            self.vectorLineEdit.setEnabled(True)
            self.classFieldCombo.setEnabled(True)
            if self._vector_combo is not None:
                self._vector_combo.setEnabled(True)

    def _browse_raster(self):
        # type: () -> None
        """Open a file dialog to select a raster."""
        path, _f = QFileDialog.getOpenFileName(self, "Select raster", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.rasterLineEdit.setText(path)
            # Show recipe recommendations if available
            self._show_recipe_recommendations_for_setup_dialog(path)

    def _browse_vector(self):
        # type: () -> None
        """Open a file dialog to select a vector file."""
        path, _f = QFileDialog.getOpenFileName(
            self, "Select vector", "", "Shapefile (*.shp);;GeoPackage (*.gpkg);;All (*)"
        )
        if path:
            self.vectorLineEdit.setText(path)
            self._populate_fields_from_path(path)

    def _browse_model(self):
        # type: () -> None
        """Open a file dialog to select a saved model."""
        path, _f = QFileDialog.getOpenFileName(self, "Select model", "", "Model files (*)")
        if path:
            self.modelLineEdit.setText(path)

    def _on_vector_changed(self):
        # type: () -> None
        """Re-populate the class-field combo when the QGIS vector layer changes."""
        self.classFieldCombo.clear()
        self.fieldStatusLabel.setText("")
        if self._vector_combo is None:
            return
        layer = self._vector_combo.currentLayer()
        if layer is None:
            self.fieldStatusLabel.setText("Select a vector layer to list its fields.")
            return
        try:
            fields = layer.dataProvider().fields()
            names = [fields.at(i).name() for i in range(fields.count())]
            if names:
                self.classFieldCombo.addItems(names)
                self.fieldStatusLabel.setText("")
            else:
                self.fieldStatusLabel.setText("No fields found in the selected layer.")
        except (AttributeError, TypeError):
            self.fieldStatusLabel.setText("Unable to read fields from the selected layer.")

    def _on_vector_path_edited(self):
        # type: () -> None
        """Update class fields when a vector path is typed manually."""
        path = self.vectorLineEdit.text().strip()
        if not path:
            self.classFieldCombo.clear()
            self.fieldStatusLabel.setText("")
            return
        self._populate_fields_from_path(path)

    def _populate_fields_from_path(self, path):
        # type: (str) -> None
        """Best-effort field listing for a vector path (fallback without QGIS)."""
        self.classFieldCombo.clear()
        self.fieldStatusLabel.setText("")
        if not os.path.exists(path):
            self.fieldStatusLabel.setText("Vector path does not exist.")
            return
        try:
            from osgeo import ogr
        except ImportError:
            try:
                import ogr  # type: ignore[no-redef]
            except ImportError:
                self.fieldStatusLabel.setText("OGR is unavailable; cannot read fields from path.")
                return
        ds = ogr.Open(path)
        if ds is None:
            self.fieldStatusLabel.setText("Unable to open vector dataset.")
            return
        layer = ds.GetLayer()
        if layer is None:
            self.fieldStatusLabel.setText("No layer found in the dataset.")
            return
        field_count = layer.GetLayerDefn().GetFieldCount()
        if field_count == 0:
            self.fieldStatusLabel.setText("No fields found in the selected vector.")
            return
        for i in range(field_count):
            field_name = layer.GetLayerDefn().GetFieldDefn(i).GetName()
            self.classFieldCombo.addItem(field_name)
        self.fieldStatusLabel.setText("")

    # --- public API --------------------------------------------------------

    def get_raster_path(self):
        # type: () -> str
        """Return the raster path from combo or line-edit."""
        if self._raster_combo is not None:
            layer = self._raster_combo.currentLayer()
            if layer is not None:
                return layer.dataProvider().dataSourceUri()
        return self.rasterLineEdit.text()

    def get_vector_path(self):
        # type: () -> str
        """Return the vector path from combo or line-edit."""
        if self._vector_combo is not None:
            layer = self._vector_combo.currentLayer()
            if layer is not None:
                uri = layer.dataProvider().dataSourceUri()
                # Strip layerid=N suffix
                return uri.split("|")[0]
        return self.vectorLineEdit.text()

    def get_class_field(self):
        # type: () -> str
        """Return the currently selected class field name."""
        return self.classFieldCombo.currentText()

    def validatePage(self):
        # type: () -> bool
        """Enforce that raster is set; vector is set unless loading a model."""
        if not self.get_raster_path():
            return False
        if not self.loadModelCheck.isChecked():
            if not self.get_vector_path():
                return False
            if not self.get_class_field():
                return False
        else:
            if not self.modelLineEdit.text():
                return False
        return True

    # --- algorithm helpers -----------------------------------------------

    def _update_dep_status(self, index):
        # type: (int) -> None
        """Show required and optional dependency status for the selected classifier."""
        code = _CLASSIFIER_META[index][0]
        missing_required = []  # type: List[str]
        _code, _name, needs_sk, needs_xgb, needs_lgb, needs_cb = _CLASSIFIER_META[index]
        if needs_sk and not self._deps.get("sklearn", False):
            missing_required.append("scikit-learn")
        if needs_xgb and not self._deps.get("xgboost", False):
            missing_required.append("xgboost")
        if needs_lgb and not self._deps.get("lightgbm", False):
            missing_required.append("lightgbm")
        if needs_cb and not self._deps.get("catboost", False):
            missing_required.append("catboost")

        missing_optional = []
        if not self._deps.get("optuna", False):
            missing_optional.append("optuna")
        if not self._deps.get("shap", False):
            missing_optional.append("shap")
        if not self._deps.get("seaborn", False):
            missing_optional.append("seaborn")
        if not self._deps.get("imblearn", False):
            missing_optional.append("imblearn (SMOTE)")
        self.depStatusLabel.clear()
        self._maybe_prompt_install(missing_required, missing_optional)

    def _maybe_prompt_install(self, missing_required, missing_optional):
        # type: (List[str], List[str]) -> None
        """Offer to install missing dependencies, similar to settings panel behavior."""
        if self._suppress_dependency_prompt:
            return
        if not missing_required and not missing_optional:
            return
        if not missing_required:
            # Do not prompt optional dependency installation at classifier-selection time.
            return

        classifier_name = self.get_classifier_name()
        signature = (self.get_classifier_code(), tuple(missing_required), tuple(missing_optional))
        if signature == self._last_prompt_signature:
            return
        self._last_prompt_signature = signature

        if not self._installer or not hasattr(self._installer, "_try_install_dependencies"):
            return

        if missing_required:
            req_list = ", ".join(missing_required)
            optional_line = ""
            if missing_optional:
                opt_list = ", ".join(missing_optional)
                optional_line = f"Optional missing now: <code>{opt_list}</code><br>"
            reply = QMessageBox.question(
                self,
                "Dependencies Missing for dzetsaka",
                (
                    "To fully use dzetsaka capabilities, we recommend installing all dependencies.<br><br>"
                    f"Required missing now: <code>{req_list}</code><br>"
                    f"{optional_line}<br>"
                    f"Full bundle to install: <code>{_full_bundle_label()}</code><br><br>"
                    "Install the full dzetsaka dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                to_install = _full_dependency_bundle()
                if self._installer._try_install_dependencies(to_install):
                    QMessageBox.information(
                        self,
                        "Installation Successful",
                        f"Dependencies installed successfully!<br><br>"
                        f"<b>Important:</b> Please restart QGIS now.<br>"
                        f"Without restarting, newly installed libraries may not be loaded, "
                        f"and {classifier_name} training/classification can fail.",
                        QMessageBox.StandardButton.Ok,
                    )
                    self._deps = check_dependency_availability()
                    self._update_dep_status(self.classifierCombo.currentIndex())
                else:
                    if hasattr(self._installer, "_show_github_issue_popup"):
                        self._installer._show_github_issue_popup(
                            "Dependency Installation Failed",
                            "Dependency Installation Error",
                            f"Automatic installation failed for: {', '.join(to_install)}",
                            f"Guided classifier selection: {classifier_name}",
                        )
                    self.classifierCombo.setCurrentIndex(0)
            elif reply == QMessageBox.StandardButton.No:
                self.classifierCombo.setCurrentIndex(0)

    def _apply_smart_defaults(self):
        # type: () -> None
        """Flag that smart defaults should be applied on the Advanced page."""
        self._smart_defaults_applied = True

    def _open_comparison(self):
        # type: () -> None
        """Show the AlgorithmComparisonPanel dialog."""
        from .comparison_panel import AlgorithmComparisonPanel

        panel = AlgorithmComparisonPanel(self)
        panel.algorithmSelected.connect(self._set_algorithm_from_comparison)
        panel.show()

    def _set_algorithm_from_comparison(self, name):
        # type: (str) -> None
        """Set the combo to the algorithm chosen in the comparison panel."""
        for i, (_code, n, _sk, _xgb, _lgb, _cb) in enumerate(_CLASSIFIER_META):
            if n == name:
                self.classifierCombo.setCurrentIndex(i)
                break

    def set_classifier_by_code(self, code):
        # type: (str) -> None
        """Set the combo to the classifier matching the provided code."""
        for i, (c, _n, _sk, _xgb, _lgb, _cb) in enumerate(_CLASSIFIER_META):
            if c == code:
                self.classifierCombo.setCurrentIndex(i)
                break

    def set_dependency_prompt_suppressed(self, suppressed):
        # type: (bool) -> None
        """Enable/disable dependency-install prompt while programmatically changing classifier."""
        self._suppress_dependency_prompt = suppressed

    def set_recipe_list(self, recipes):
        # type: (List[Dict[str, object]]) -> None
        """Populate the recipe combo with provided recipes."""
        self._recipes = recipes
        self.recipeCombo.clear()
        for recipe in recipes:
            self.recipeCombo.addItem(recipe.get("name", "Unnamed Recipe"))

    def _apply_selected_recipe(self):
        # type: () -> None
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog) or not self._recipes:
            return
        name = self.recipeCombo.currentText()
        for recipe in self._recipes:
            if recipe.get("name") == name:
                parent_dialog.apply_recipe(recipe)
                break

    def _save_current_recipe(self):
        # type: () -> None
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog):
            return
        parent_dialog.save_current_recipe()

    def _open_recipe_gallery(self):
        # type: () -> None
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog):
            return
        parent_dialog.open_recipe_gallery()

    def _load_json_config(self):
        # type: () -> None
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog):
            return
        parent_dialog.import_config_from_json_file()

    def _paste_json_config(self):
        # type: () -> None
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog):
            return
        parent_dialog.import_config_from_json_paste()

    def _open_geometry_explorer(self):
        # type: () -> None
        vector_path = self.get_vector_path()
        class_field = self.get_class_field()
        layer = self._vector_combo.currentLayer() if self._vector_combo is not None else None
        if not vector_path and layer is None:
            QMessageBox.information(self, "Geometry Explorer", "Select a vector layer or enter a vector path first.")
            return
        dialog = VectorInsightDialog(self, vector_path=vector_path, class_field=class_field, layer=layer)
        try:
            dialog.exec_()
        except AttributeError:
            dialog.exec()

    def _check_data_quality(self):
        # type: () -> None
        """Open the training data quality checker dialog."""
        if not _QUALITY_CHECKER_AVAILABLE:
            QMessageBox.critical(
                self,
                "Feature Unavailable",
                "Training data quality checker is not available. Please check that all dependencies are installed."
            )
            return

        vector_path = self.get_vector_path()
        class_field = self.get_class_field()

        if not vector_path:
            QMessageBox.information(
                self,
                "Check Data Quality",
                "Please select a training vector layer first."
            )
            return

        if not class_field:
            QMessageBox.information(
                self,
                "Check Data Quality",
                "Please select a class field first."
            )
            return

        # Show status bar message
        try:
            from src.dzetsaka.infrastructure.ui.status_bar_feedback import show_quality_check_started, show_quality_check_completed
            # Try to get iface from parent plugin
            parent_widget = self.parent()
            while parent_widget is not None:
                if hasattr(parent_widget, 'iface'):
                    show_quality_check_started(parent_widget.iface)
                    break
                parent_widget = parent_widget.parent()
        except Exception:
            pass  # Fallback if status bar not available

        # Resolve checker class from module at call time to avoid stale in-session imports.
        checker_cls = TrainingDataQualityChecker
        try:
            import importlib
            from . import training_data_quality_checker as _tdq_mod

            _tdq_mod = importlib.reload(_tdq_mod)
            checker_cls = getattr(_tdq_mod, "TrainingDataQualityChecker", TrainingDataQualityChecker)
        except Exception:
            pass

        # Open the quality checker dialog
        dialog = checker_cls(
            vector_path=vector_path,
            class_field=class_field,
            parent=self
        )
        try:
            result = dialog.exec_()
        except AttributeError:
            result = dialog.exec()

        # Show completion message with issue count
        try:
            if hasattr(dialog, 'issues'):
                issue_count = len([i for i in dialog.issues if i.severity in ["critical", "error", "warning"]])
                parent_widget = self.parent()
                while parent_widget is not None:
                    if hasattr(parent_widget, 'iface'):
                        show_quality_check_completed(parent_widget.iface, issue_count)
                        break
                    parent_widget = parent_widget.parent()
        except Exception:
            pass

    def get_classifier_code(self):
        # type: () -> str
        """Return the short code of the currently selected classifier."""
        return _CLASSIFIER_META[self.classifierCombo.currentIndex()][0]

    def get_classifier_name(self):
        # type: () -> str
        """Return the full name of the currently selected classifier."""
        return _CLASSIFIER_META[self.classifierCombo.currentIndex()][1]

    def smart_defaults_requested(self):
        # type: () -> bool
        """Return True if the user clicked Smart Defaults on this page."""
        return self._smart_defaults_applied

    def _show_recipe_recommendations_for_setup_dialog(self, raster_path):
        # type: (str) -> None
        """Show recipe recommendations for setup-dialog context.

        Parameters
        ----------
        raster_path : str
            Path to the raster file to analyze

        """
        # Check if recommender is available and enabled
        if not _RECOMMENDER_AVAILABLE:
            return

        settings = QSettings()
        if not settings.value("/dzetsaka/show_recommendations", True, bool):
            return

        # Get recipes from the setup dialog
        setup_dialog = self.wizard()
        if not setup_dialog or not hasattr(setup_dialog, "_recipes"):
            return

        recipes = getattr(setup_dialog, "_recipes", [])
        if not recipes:
            return

        try:
            # Analyze the raster
            analyzer = RasterAnalyzer()
            raster_info = analyzer.analyze_raster(raster_path)

            # Check for errors
            if raster_info.get("error"):
                return

            # Get recommendations
            recommender = RecipeRecommender()
            recommendations = recommender.recommend(raster_info, recipes)

            # Only show dialog if we have good recommendations
            if not recommendations or recommendations[0][1] < 40:
                return

            # Show recommendation dialog
            dialog = RecommendationDialog(recommendations, raster_info, self)
            dialog.recipeSelected.connect(lambda recipe: self._apply_recipe_to_setup_dialog(recipe))
            dialog.exec_()

        except Exception:
            # Silently fail - recommendations are a nice-to-have feature
            pass

    def _apply_recipe_to_setup_dialog(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply a recommended recipe to the setup dialog.

        Parameters
        ----------
        recipe : Dict[str, object]
            Recipe dictionary to apply

        """
        try:
            setup_dialog = self.wizard()
            if setup_dialog and hasattr(setup_dialog, "apply_recipe"):
                setup_dialog.apply_recipe(recipe)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Page 2 — Advanced Options
# ---------------------------------------------------------------------------


class AdvancedOptionsPage(QWizardPage):
    """Workflow page for Optuna, imbalance, explainability and validation."""

    def __init__(self, parent=None, deps=None):
        """Initialise AdvancedOptionsPage."""
        super(AdvancedOptionsPage, self).__init__(parent)
        self.setTitle("Advanced Setup")
        self.setSubTitle("Configure optimization, explainability, and validation.")

        self._deps = deps if deps is not None else check_dependency_availability()

        # Create main layout container
        main_layout = QVBoxLayout()

        # Add What's This help button
        help_btn_layout = QHBoxLayout()
        help_btn_layout.addStretch()
        help_btn = QPushButton("?")
        help_btn.setToolTip("Click for detailed help on this page")
        help_btn.setMaximumWidth(30)
        help_btn.clicked.connect(lambda: QWhatsThis.enterWhatsThisMode())
        help_btn_layout.addWidget(help_btn)
        main_layout.addLayout(help_btn_layout)

        layout = QGridLayout()
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        # --- Optimization group ---
        opt_group = QGroupBox("Optimization")
        opt_layout = QGridLayout()
        opt_layout.setContentsMargins(8, 8, 8, 8)
        opt_layout.setHorizontalSpacing(10)
        opt_layout.setVerticalSpacing(6)

        self.optunaCheck = QCheckBox("Use Optuna hyperparameter optimization")
        self.optunaCheck.setEnabled(self._deps.get("optuna", False))
        self.optunaCheck.setToolTip(
            "<b>Optuna Hyperparameter Optimization</b><br>"
            "Automatically searches for optimal algorithm parameters using Bayesian optimization. "
            "Significantly improves accuracy but increases training time.<br><br>"
            "<i>When to use:</i> Want best accuracy, have time for longer training (5-30 min), "
            "unsure about parameter values.<br>"
            "<i>When to skip:</i> Quick exploration, proven parameters from past runs."
        )
        self.optunaCheck.setWhatsThis(
            "<h3>Optuna Hyperparameter Optimization</h3>"
            "<p>Optuna automatically searches for the best algorithm parameters (hyperparameters) using "
            "<b>Bayesian optimization</b> - a smart search strategy that learns from previous trials.</p>"
            "<h4>What it does:</h4>"
            "<ul>"
            "<li>Tests multiple parameter combinations (trials) during training</li>"
            "<li>Learns which parameter ranges work best for your data</li>"
            "<li>Converges toward optimal settings without exhaustive grid search</li>"
            "<li>Typically improves accuracy by 2-10% compared to default parameters</li>"
            "</ul>"
            "<h4>When to use:</h4>"
            "<ul>"
            "<li><b>Best accuracy is critical:</b> Research, production models, competitive benchmarks</li>"
            "<li><b>Unsure about parameters:</b> Don't know optimal tree depth, learning rate, etc.</li>"
            "<li><b>Have time for training:</b> 5-30 min extra depending on trials (100 trials ≈ 10-15 min)</li>"
            "<li><b>Complex datasets:</b> Many classes, high-dimensional features, imbalanced data</li>"
            "</ul>"
            "<h4>When to skip:</h4>"
            "<ul>"
            "<li><b>Quick exploration:</b> Testing different algorithms, rapid prototyping</li>"
            "<li><b>Known good parameters:</b> Reusing parameters from previous successful runs</li>"
            "<li><b>Time constraints:</b> Need results in <5 minutes</li>"
            "<li><b>Very small datasets:</b> <500 samples may overfit during search</li>"
            "</ul>"
            "<h4>Trade-offs:</h4>"
            "<ul>"
            "<li><b>Pro:</b> Often significant accuracy gains (2-10%)</li>"
            "<li><b>Pro:</b> Removes guesswork from parameter tuning</li>"
            "<li><b>Con:</b> Training time multiplied by number of trials (10-1000x)</li>"
            "<li><b>Con:</b> Risk of overfitting if validation split is too small</li>"
            "</ul>"
            "<p><i>Recommended:</i> Use 100 trials for balanced speed/accuracy. "
            "Increase to 300+ for critical applications. Use <50 for quick experiments.</p>"
        )
        opt_layout.addWidget(self.optunaCheck, 0, 0, 1, 2)

        self.optunaInfoLabel = QLabel(
            "Optuna runs multiple trials to search better hyperparameters.\n"
            "Trials range: 10-1000 (more trials = slower, often more stable best params).\n"
            "Search ranges are classifier-specific and handled automatically."
        )
        self.optunaInfoLabel.setWordWrap(True)
        opt_layout.addWidget(self.optunaInfoLabel, 1, 0, 1, 2)

        trials_label = QLabel("Trials:")
        self.optunaTrials = ValidatedSpinBox(
            validator_fn=lambda v: 10 <= v <= 1000,
            warning_threshold=300,
            time_estimator_fn=lambda v: f"{v * 0.05:.0f}-{v * 0.15:.0f} min" if v > 0 else "0 min"
        )
        self.optunaTrials.setRange(10, 1000)
        self.optunaTrials.setValue(100)
        self.optunaTrials.setEnabled(False)
        self.optunaTrials.setToolTip(
            "<b>Optuna Trials</b><br>"
            "Number of hyperparameter combinations to test. "
            "More trials = better accuracy but slower training.<br><br>"
            "<i>Typical values:</i> Quick: 10-50, Balanced: 100-200, Thorough: 300-1000"
        )
        opt_layout.addWidget(trials_label, 2, 0)
        opt_layout.addWidget(self.optunaTrials, 2, 1)

        self.optunaCheck.toggled.connect(self.optunaTrials.setEnabled)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group, 0, 0)

        # --- Imbalance group ---
        imb_group = QGroupBox("Imbalance Handling")
        imb_layout = QGridLayout()
        imb_layout.setContentsMargins(8, 8, 8, 8)
        imb_layout.setHorizontalSpacing(10)
        imb_layout.setVerticalSpacing(6)

        self.smoteCheck = QCheckBox("SMOTE oversampling")
        self.smoteCheck.setEnabled(self._deps.get("imblearn", False))
        self.smoteCheck.setToolTip(
            "<b>SMOTE Oversampling</b><br>"
            "Synthetic Minority Over-sampling Technique. Creates synthetic samples for underrepresented classes "
            "to balance training data.<br><br>"
            "<i>When to use:</i> Severe class imbalance (ratio >10:1), minority class has <100 samples.<br>"
            "<i>When to skip:</i> Balanced classes, very small datasets (<50 samples per class)."
        )
        self.smoteCheck.setWhatsThis(
            "<h3>SMOTE Class Balancing</h3>"
            "<p><b>SMOTE</b> (Synthetic Minority Over-sampling Technique) creates synthetic training samples for "
            "underrepresented classes by interpolating between existing minority samples and their nearest neighbors.</p>"
            "<h4>What it does:</h4>"
            "<ul>"
            "<li>Identifies minority classes (those with fewer samples)</li>"
            "<li>Generates synthetic samples by interpolating between k-nearest neighbors</li>"
            "<li>Balances class distribution before training the classifier</li>"
            "<li>Improves model's ability to detect rare classes</li>"
            "</ul>"
            "<h4>When to use:</h4>"
            "<ul>"
            "<li><b>Severe class imbalance:</b> Ratio >10:1 (e.g., 1000 forest samples vs. 50 wetland samples)</li>"
            "<li><b>Rare class detection critical:</b> Missing rare classes is unacceptable (e.g., endangered habitats)</li>"
            "<li><b>Sufficient minority samples:</b> At least 50-100 samples in smallest class (needed for k-neighbors)</li>"
            "<li><b>Continuous features:</b> Works best with spectral data, not categorical attributes</li>"
            "</ul>"
            "<h4>When to skip:</h4>"
            "<ul>"
            "<li><b>Balanced classes:</b> All classes have similar sample counts (ratio <3:1)</li>"
            "<li><b>Very small datasets:</b> <50 samples per class (risk of overfitting synthetic samples)</li>"
            "<li><b>High-dimensional data:</b> Many bands may create unrealistic synthetic samples</li>"
            "<li><b>Use class weights instead:</b> Simpler alternative for moderate imbalance</li>"
            "</ul>"
            "<h4>Trade-offs:</h4>"
            "<ul>"
            "<li><b>Pro:</b> Improves recall for minority classes (fewer missed detections)</li>"
            "<li><b>Pro:</b> Better than simple duplication or downsampling</li>"
            "<li><b>Con:</b> Can introduce synthetic noise or unrealistic samples</li>"
            "<li><b>Con:</b> May overfit if k_neighbors is too small or dataset is tiny</li>"
            "<li><b>Con:</b> Training time increases (more samples to process)</li>"
            "</ul>"
            "<h4>k_neighbors parameter:</h4>"
            "<ul>"
            "<li><b>Default (5):</b> Good for most cases</li>"
            "<li><b>Small (2-3):</b> For very small minority classes (<100 samples)</li>"
            "<li><b>Large (7-10):</b> For large datasets with smooth class boundaries</li>"
            "<li><b>Constraint:</b> Must be less than smallest class sample count</li>"
            "</ul>"
            "<p><i>Alternative:</i> For moderate imbalance (2:1 to 10:1), try <b>Class Weights</b> first - "
            "it's simpler and faster, with no risk of synthetic sample artifacts.</p>"
        )
        imb_layout.addWidget(self.smoteCheck, 0, 0, 1, 2)

        k_label = QLabel("k_neighbors:")
        self.smoteK = QSpinBox()
        self.smoteK.setRange(1, 20)
        self.smoteK.setValue(5)
        self.smoteK.setEnabled(False)
        self.smoteK.setToolTip(
            "<b>SMOTE k_neighbors</b><br>"
            "Number of nearest neighbors used to generate synthetic samples.<br><br>"
            "<i>Typical values:</i> Default: 5, Small datasets (<100 samples): 3, Large datasets: 5-10<br>"
            "<i>Note:</i> Must be less than the number of samples in the smallest class."
        )
        imb_layout.addWidget(k_label, 1, 0)
        imb_layout.addWidget(self.smoteK, 1, 1)
        self.smoteCheck.toggled.connect(self.smoteK.setEnabled)

        self.classWeightCheck = QCheckBox("Use class weights")
        self.classWeightCheck.setEnabled(self._deps.get("sklearn", False))
        self.classWeightCheck.setToolTip(
            "<b>Class Weights</b><br>"
            "Assigns higher penalty to misclassifying minority classes. "
            "Lighter alternative to SMOTE, works directly during training.<br><br>"
            "<i>When to use:</i> Moderate class imbalance (2:1 to 10:1).<br>"
            "<i>When to skip:</i> Balanced classes, already using SMOTE."
        )
        imb_layout.addWidget(self.classWeightCheck, 2, 0, 1, 2)

        strat_label = QLabel("Strategy:")
        self.weightStrategyCombo = QComboBox()
        self.weightStrategyCombo.addItems(["balanced", "uniform"])
        self.weightStrategyCombo.setEnabled(False)
        self.weightStrategyCombo.setToolTip(
            "<b>Class Weight Strategy</b><br>"
            "<i>Balanced:</i> Automatically adjusts weights inversely proportional to class frequencies.<br>"
            "<i>Uniform:</i> All classes weighted equally (no balancing).<br><br>"
            "<i>Recommended:</i> Use 'balanced' for most imbalanced datasets."
        )
        imb_layout.addWidget(strat_label, 3, 0)
        imb_layout.addWidget(self.weightStrategyCombo, 3, 1)
        self.classWeightCheck.toggled.connect(self.weightStrategyCombo.setEnabled)

        imb_group.setLayout(imb_layout)
        layout.addWidget(imb_group, 0, 1)

        # --- Explainability group ---
        exp_group = QGroupBox("Explainability")
        exp_layout = QGridLayout()
        exp_layout.setContentsMargins(8, 8, 8, 8)
        exp_layout.setHorizontalSpacing(10)
        exp_layout.setVerticalSpacing(6)

        self.shapCheck = QCheckBox("Compute SHAP feature importance")
        self.shapCheck.setEnabled(self._deps.get("shap", False))
        self.shapCheck.setToolTip(
            "<b>SHAP Feature Importance</b><br>"
            "Computes SHapley Additive exPlanations to show which raster bands contribute most to predictions. "
            "Generates feature importance raster and summary plots.<br><br>"
            "<i>Use case:</i> Understanding model decisions, identifying important spectral bands.<br>"
            "<i>Note:</i> Adds 10-30% to processing time depending on sample size."
        )
        self.shapCheck.setWhatsThis(
            "<h3>SHAP Feature Importance (Explainability)</h3>"
            "<p><b>SHAP</b> (SHapley Additive exPlanations) explains <i>why</i> the model makes predictions "
            "by computing how much each raster band contributes to each classification decision.</p>"
            "<h4>What it generates:</h4>"
            "<ul>"
            "<li><b>Feature importance raster:</b> Multi-band GeoTIFF showing spatial contribution of each input band</li>"
            "<li><b>Summary plots:</b> Bar charts and beeswarm plots showing global feature importance</li>"
            "<li><b>Per-class importance:</b> Which bands matter most for each land cover class</li>"
            "</ul>"
            "<h4>When to use:</h4>"
            "<ul>"
            "<li><b>Model interpretation:</b> Understanding which spectral bands drive classifications</li>"
            "<li><b>Scientific research:</b> Identifying key vegetation indices, atmospheric windows, etc.</li>"
            "<li><b>Debugging:</b> Finding if model relies on artifacts or unexpected bands</li>"
            "<li><b>Feature selection:</b> Discovering which bands are redundant or uninformative</li>"
            "<li><b>Stakeholder communication:</b> Explaining model decisions to non-technical users</li>"
            "</ul>"
            "<h4>When to skip:</h4>"
            "<ul>"
            "<li><b>Pure production:</b> Only need classification output, not explanations</li>"
            "<li><b>Time-critical workflows:</b> Adds 10-30% to processing time</li>"
            "<li><b>Well-understood models:</b> Already know which bands are important from past runs</li>"
            "<li><b>Simple datasets:</b> Obvious band importance (e.g., only 3-4 bands)</li>"
            "</ul>"
            "<h4>Technical details:</h4>"
            "<ul>"
            "<li><b>Method:</b> Uses game-theory Shapley values from cooperative game theory</li>"
            "<li><b>Sample size:</b> 1000 samples = balanced speed/accuracy, 5000+ = thorough but slower</li>"
            "<li><b>Overhead:</b> ~2-5 min for 1000 samples, ~10-20 min for 5000 samples</li>"
            "<li><b>Compatible with:</b> All tree-based models (RF, XGB, LGB, CB, ET, GBC)</li>"
            "</ul>"
            "<p><i>Use case example:</i> In vegetation mapping, SHAP might reveal that NIR and Red Edge bands "
            "are most important for forest/grassland distinction, while SWIR bands matter for urban areas.</p>"
        )
        exp_layout.addWidget(self.shapCheck, 0, 0, 1, 3)

        shap_label = QLabel("Output:")
        self.shapOutput = QLineEdit()
        self.shapOutput.setPlaceholderText("Path to SHAP raster…")
        self.shapOutput.setEnabled(False)
        self.shapOutput.setToolTip(
            "<b>SHAP Output Raster</b><br>"
            "Path where SHAP feature importance raster will be saved. "
            "Each band shows the importance of corresponding input band for the classification."
        )
        self.shapBrowse = QPushButton("Browse…")
        self.shapBrowse.setEnabled(False)
        self.shapBrowse.clicked.connect(self._browse_shap_output)
        exp_layout.addWidget(shap_label, 1, 0)
        exp_layout.addWidget(self.shapOutput, 1, 1)
        exp_layout.addWidget(self.shapBrowse, 1, 2)

        shap_sample_label = QLabel("Sample size:")
        self.shapSampleSize = ValidatedSpinBox(
            validator_fn=lambda v: 100 <= v <= 50000,
            warning_threshold=5000,
            time_estimator_fn=lambda v: f"{v * 0.002:.0f}-{v * 0.005:.0f} min" if v > 0 else "0 min"
        )
        self.shapSampleSize.setRange(100, 50000)
        self.shapSampleSize.setValue(1000)
        self.shapSampleSize.setEnabled(False)
        self.shapSampleSize.setToolTip(
            "<b>SHAP Sample Size</b><br>"
            "Number of samples used to compute feature importance. "
            "Higher = more accurate explanations but slower computation.<br><br>"
            "<i>Typical values:</i> Fast: 100-500, Balanced: 1000-2000, Thorough: 5000+"
        )
        exp_layout.addWidget(shap_sample_label, 2, 0)
        exp_layout.addWidget(self.shapSampleSize, 2, 1)

        def _toggle_shap(checked):
            # type: (bool) -> None
            self.shapOutput.setEnabled(checked)
            self.shapBrowse.setEnabled(checked)
            self.shapSampleSize.setEnabled(checked)

        self.shapCheck.toggled.connect(_toggle_shap)
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group, 1, 0)

        # --- Validation group ---
        val_group = QGroupBox("Validation")
        val_layout = QGridLayout()
        val_layout.setContentsMargins(8, 8, 8, 8)
        val_layout.setHorizontalSpacing(10)
        val_layout.setVerticalSpacing(6)

        self.nestedCVCheck = QCheckBox("Nested cross-validation")
        self.nestedCVCheck.setToolTip(
            "<b>Nested Cross-Validation</b><br>"
            "Uses two levels of cross-validation: inner loop for hyperparameter tuning, "
            "outer loop for unbiased performance estimation.<br><br>"
            "<i>When to use:</i> Small datasets (<5000 samples), need unbiased accuracy estimate.<br>"
            "<i>Note:</i> Significantly increases training time (multiply by inner_folds × outer_folds)."
        )
        val_layout.addWidget(self.nestedCVCheck, 0, 0, 1, 2)

        inner_label = QLabel("Inner folds:")
        self.innerFolds = QSpinBox()
        self.innerFolds.setRange(2, 10)
        self.innerFolds.setValue(3)
        self.innerFolds.setEnabled(False)
        self.innerFolds.setToolTip(
            "<b>Inner Folds</b><br>"
            "Number of cross-validation folds for hyperparameter tuning in nested CV.<br><br>"
            "<i>Typical values:</i> 3-5 folds<br>"
            "<i>Note:</i> Higher values = more robust tuning but longer training time."
        )
        val_layout.addWidget(inner_label, 1, 0)
        val_layout.addWidget(self.innerFolds, 1, 1)

        outer_label = QLabel("Outer folds:")
        self.outerFolds = QSpinBox()
        self.outerFolds.setRange(2, 10)
        self.outerFolds.setValue(5)
        self.outerFolds.setEnabled(False)
        self.outerFolds.setToolTip(
            "<b>Outer Folds</b><br>"
            "Number of cross-validation folds for performance estimation in nested CV.<br><br>"
            "<i>Typical values:</i> 5-10 folds<br>"
            "<i>Note:</i> Higher values = more reliable accuracy estimate but longer training time."
        )
        val_layout.addWidget(outer_label, 2, 0)
        val_layout.addWidget(self.outerFolds, 2, 1)

        def _toggle_nested(checked):
            # type: (bool) -> None
            self.innerFolds.setEnabled(checked)
            self.outerFolds.setEnabled(checked)

        self.nestedCVCheck.toggled.connect(_toggle_nested)

        cv_mode_label = QLabel("Validation mode:")
        self.cvModeCombo = QComboBox()
        self.cvModeCombo.addItem("Random split", "RANDOM_SPLIT")
        self.cvModeCombo.addItem("Polygon group CV (default, recommended)", "POLYGON_GROUP")
        self.cvModeCombo.setCurrentIndex(1)  # Default to POLYGON_GROUP
        self.cvModeCombo.setToolTip(
            "Use polygon-group CV to avoid pixel leakage within the same polygon "
            "when spatial autocorrelation is strong."
        )
        val_layout.addWidget(cv_mode_label, 3, 0)
        val_layout.addWidget(self.cvModeCombo, 3, 1)

        self.cvInfoLabel = QLabel(
            "Random split may overestimate accuracy when many pixels belong to the same polygon.\n"
            "Polygon group CV splits by polygons per label (no extra group field required).\n"
            "In this mode, split percentage is interpreted as TRAIN percent (e.g. 75 = 75% train, 25% valid)."
        )
        self.cvInfoLabel.setWordWrap(True)
        val_layout.addWidget(self.cvInfoLabel, 4, 0, 1, 2)
        val_group.setLayout(val_layout)
        layout.addWidget(val_group, 1, 1)

        main_layout.addLayout(layout)
        self.setLayout(main_layout)

    # --- internal helpers --------------------------------------------------

    def _browse_shap_output(self):
        # type: () -> None
        """File dialog for SHAP output raster."""
        path, _f = QFileDialog.getSaveFileName(self, "SHAP output raster", "", "GeoTIFF (*.tif)")
        if path:
            self.shapOutput.setText(path)

    # --- public API --------------------------------------------------------

    def apply_smart_defaults(self, defaults):
        # type: (Dict[str, object]) -> None
        """Programmatically fill widgets from a smart-defaults dict."""
        self.optunaCheck.setChecked(bool(defaults.get("USE_OPTUNA", False)))
        self.optunaTrials.setValue(int(defaults.get("OPTUNA_TRIALS", 100)))
        self.smoteCheck.setChecked(bool(defaults.get("USE_SMOTE", False)))
        self.smoteK.setValue(int(defaults.get("SMOTE_K_NEIGHBORS", 5)))
        self.classWeightCheck.setChecked(bool(defaults.get("USE_CLASS_WEIGHTS", False)))
        self.shapCheck.setChecked(bool(defaults.get("COMPUTE_SHAP", False)))
        self.shapSampleSize.setValue(int(defaults.get("SHAP_SAMPLE_SIZE", 1000)))
        self._set_cv_mode(str(defaults.get("CV_MODE", "POLYGON_GROUP")))

    def apply_recipe(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply recipe settings to advanced options."""
        recipe = normalize_recipe(dict(recipe))
        extra = recipe.get("extraParam", {})
        validation = recipe.get("validation", {})
        self.optunaCheck.setChecked(bool(extra.get("USE_OPTUNA", False)))
        self.optunaTrials.setValue(int(extra.get("OPTUNA_TRIALS", 100)))
        self.smoteCheck.setChecked(bool(extra.get("USE_SMOTE", False)))
        self.smoteK.setValue(int(extra.get("SMOTE_K_NEIGHBORS", 5)))
        self.classWeightCheck.setChecked(bool(extra.get("USE_CLASS_WEIGHTS", False)))
        self.weightStrategyCombo.setCurrentText(str(extra.get("CLASS_WEIGHT_STRATEGY", "balanced")))
        self.shapCheck.setChecked(bool(extra.get("COMPUTE_SHAP", False)))
        self.shapSampleSize.setValue(int(extra.get("SHAP_SAMPLE_SIZE", 1000)))
        self.shapOutput.setText(str(extra.get("SHAP_OUTPUT", "")))

        nested = bool(validation.get("nested_cv", extra.get("USE_NESTED_CV", False)))
        self.nestedCVCheck.setChecked(nested)
        self.innerFolds.setValue(int(validation.get("nested_inner_cv", extra.get("NESTED_INNER_CV", 3))))
        self.outerFolds.setValue(int(validation.get("nested_outer_cv", extra.get("NESTED_OUTER_CV", 5))))
        cv_mode = str(validation.get("cv_mode", extra.get("CV_MODE", "POLYGON_GROUP")))
        self._set_cv_mode(cv_mode)

    def get_extra_params(self):
        # type: () -> Dict[str, object]
        """Collect the extraParam dict from the current widget states."""
        return {
            "USE_OPTUNA": self.optunaCheck.isChecked(),
            "OPTUNA_TRIALS": self.optunaTrials.value(),
            "COMPUTE_SHAP": self.shapCheck.isChecked(),
            "SHAP_OUTPUT": self.shapOutput.text(),
            "SHAP_SAMPLE_SIZE": self.shapSampleSize.value(),
            "USE_SMOTE": self.smoteCheck.isChecked(),
            "SMOTE_K_NEIGHBORS": self.smoteK.value(),
            "USE_CLASS_WEIGHTS": self.classWeightCheck.isChecked(),
            "CLASS_WEIGHT_STRATEGY": self.weightStrategyCombo.currentText(),
            "CUSTOM_CLASS_WEIGHTS": {},
            "USE_NESTED_CV": self.nestedCVCheck.isChecked(),
            "NESTED_INNER_CV": self.innerFolds.value(),
            "NESTED_OUTER_CV": self.outerFolds.value(),
            "CV_MODE": str(self.cvModeCombo.currentData() or "POLYGON_GROUP"),
        }  # type: Dict[str, object]

    def _set_cv_mode(self, mode):
        # type: (str) -> None
        normalized = "RANDOM_SPLIT" if str(mode).upper() == "RANDOM_SPLIT" else "POLYGON_GROUP"
        idx = self.cvModeCombo.findData(normalized)
        if idx < 0:
            idx = 0
        self.cvModeCombo.setCurrentIndex(idx)


# ---------------------------------------------------------------------------
# Page 3 — Output Configuration
# ---------------------------------------------------------------------------


class SplitPreviewDialog(QDialog):
    """Dialog showing train/test split distribution preview."""

    def __init__(self, vector_path, class_field, split_percent, cv_mode, parent=None):
        # type: (str, str, int, str, QWidget) -> None
        """Initialize SplitPreviewDialog.

        Args:
            vector_path: Path to training vector file
            class_field: Name of the class field
            split_percent: Percentage for training (in POLYGON_GROUP) or test (in RANDOM_SPLIT)
            cv_mode: CV mode - "RANDOM_SPLIT" or "POLYGON_GROUP"
            parent: Parent widget
        """
        super(SplitPreviewDialog, self).__init__(parent)
        self.setWindowTitle("Split Preview")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        # Header
        header_label = QLabel(f"<h3>Train/Test Split Preview</h3>")
        layout.addWidget(header_label)

        info_label = QLabel(
            f"<b>Mode:</b> {cv_mode}<br>"
            f"<b>Split:</b> {split_percent}% train, {100 - split_percent}% test<br>"
            f"<b>Vector:</b> {vector_path}"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Compute split preview
        try:
            train_counts, test_counts, warnings = self._compute_split_preview(
                vector_path, class_field, split_percent, cv_mode
            )
        except Exception as e:
            error_label = QLabel(f"<font color='red'>Error computing split: {str(e)}</font>")
            error_label.setWordWrap(True)
            layout.addWidget(error_label)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            layout.addWidget(close_btn)
            self.setLayout(layout)
            return

        # Results table
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Class", "Train Count", "Test Count"])
        table.horizontalHeader().setStretchLastSection(True)

        classes = sorted(set(train_counts.keys()) | set(test_counts.keys()))
        table.setRowCount(len(classes))

        for i, cls in enumerate(classes):
            train_count = train_counts.get(cls, 0)
            test_count = test_counts.get(cls, 0)

            table.setItem(i, 0, QTableWidgetItem(str(cls)))
            table.setItem(i, 1, QTableWidgetItem(str(train_count)))
            table.setItem(i, 2, QTableWidgetItem(str(test_count)))

        layout.addWidget(table)

        # Warnings section
        if warnings:
            warnings_label = QLabel("<b>Warnings:</b>")
            warnings_label.setStyleSheet("color: orange; font-weight: bold;")
            layout.addWidget(warnings_label)

            warnings_text = QTextEdit()
            warnings_text.setReadOnly(True)
            warnings_text.setMaximumHeight(150)
            warnings_text.setHtml("<ul>" + "".join([f"<li>{w}</li>" for w in warnings]) + "</ul>")
            layout.addWidget(warnings_text)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def _compute_split_preview(self, vector_path, class_field, split_percent, cv_mode):
        # type: (str, str, int, str) -> Tuple[Dict[Any, int], Dict[Any, int], List[str]]
        """Compute train/test class distribution.

        Returns:
            Tuple of (train_counts, test_counts, warnings)
        """
        from dzetsaka.infrastructure.geo.vector_split import count_polygons_per_class, split_vector_stratified

        # Get total class counts
        class_counts = count_polygons_per_class(vector_path, class_field)

        if not class_counts:
            raise RuntimeError("Unable to read vector file or no features found")

        # Perform split using the same logic as classification pipeline
        train_path, test_path = split_vector_stratified(
            vector_path, class_field, train_percent=split_percent, use_percent=True
        )

        # Count classes in train/test splits
        train_counts = count_polygons_per_class(train_path, class_field)
        test_counts = count_polygons_per_class(test_path, class_field)

        # Generate warnings
        warnings = []

        # Check for class imbalance
        all_counts = list(train_counts.values()) + list(test_counts.values())
        if all_counts:
            max_count = max(all_counts)
            min_count = min(all_counts)
            if min_count > 0 and max_count / min_count > 10:
                warnings.append(
                    f"<b>Severe class imbalance detected:</b> Ratio {max_count}:{min_count} (>10:1). "
                    "Consider using SMOTE for class balancing."
                )

        # Check for small test sets
        small_test_classes = [cls for cls, count in test_counts.items() if count < 5]
        if small_test_classes:
            warnings.append(
                f"<b>Small test sets:</b> Classes {', '.join(map(str, small_test_classes))} have <5 test samples. "
                "Results may be unreliable."
            )

        # Check for classes missing in train or test
        missing_in_train = set(test_counts.keys()) - set(train_counts.keys())
        missing_in_test = set(train_counts.keys()) - set(test_counts.keys())
        if missing_in_train:
            warnings.append(
                f"<b>Classes missing in training:</b> {', '.join(map(str, missing_in_train))}. "
                "Model will not learn these classes."
            )
        if missing_in_test:
            warnings.append(
                f"<b>Classes missing in test:</b> {', '.join(map(str, missing_in_test))}. "
                "Accuracy for these classes cannot be evaluated."
            )

        # Note about polygon group mode
        if cv_mode == "POLYGON_GROUP":
            warnings.append(
                "<b>Info:</b> POLYGON_GROUP mode ensures pixels from the same polygon stay together, "
                "preventing spatial leakage. This is a more realistic validation for spatial data."
            )

        return train_counts, test_counts, warnings


class OutputConfigPage(QWizardPage):
    """Workflow page for specifying output paths and optional outputs."""

    def __init__(self, parent=None):
        """Initialise OutputConfigPage."""
        super(OutputConfigPage, self).__init__(parent)
        self.setTitle("Output and Review")
        self.setSubTitle("Set output paths and confirm settings.")

        layout = QVBoxLayout()

        out_group = QGroupBox("Outputs")
        out_layout = QGridLayout()
        out_layout.setContentsMargins(8, 8, 8, 8)
        out_layout.setHorizontalSpacing(10)
        out_layout.setVerticalSpacing(6)

        out_layout.addWidget(QLabel("Classification map:"), 0, 0)
        self.outRasterEdit = QLineEdit()
        self.outRasterEdit.setPlaceholderText("<temporary file>")
        out_layout.addWidget(self.outRasterEdit, 0, 1)
        self.outRasterBrowse = QPushButton("Browse…")
        self.outRasterBrowse.clicked.connect(self._browse_out_raster)
        out_layout.addWidget(self.outRasterBrowse, 0, 2)

        self.confidenceCheck = QCheckBox("Generate confidence map")
        self.confidenceCheck.setToolTip(
            "<b>Confidence Map</b><br>"
            "Generates a raster showing the classifier's confidence (probability) for each pixel's predicted class.<br><br>"
            "<i>Use case:</i> Identify uncertain regions, threshold predictions, quality assessment.<br>"
            "<i>Output:</i> Float raster with values 0.0-1.0 (or 0-100%)."
        )
        out_layout.addWidget(self.confidenceCheck, 1, 0, 1, 3)
        out_layout.addWidget(QLabel("Confidence map:"), 2, 0)
        self.confMapEdit = QLineEdit()
        self.confMapEdit.setPlaceholderText("Path to confidence map…")
        self.confMapEdit.setEnabled(False)
        self.confMapEdit.setToolTip(
            "<b>Confidence Map Output Path</b><br>"
            "Path where confidence raster will be saved. Leave empty to auto-generate next to classification map."
        )
        out_layout.addWidget(self.confMapEdit, 2, 1)
        self.confMapBrowse = QPushButton("Browse…")
        self.confMapBrowse.setEnabled(False)
        self.confMapBrowse.clicked.connect(self._browse_conf_map)
        out_layout.addWidget(self.confMapBrowse, 2, 2)

        def _toggle_conf(checked):
            # type: (bool) -> None
            self.confMapEdit.setEnabled(checked)
            self.confMapBrowse.setEnabled(checked)

        self.confidenceCheck.toggled.connect(_toggle_conf)

        self.saveModelCheck = QCheckBox("Save trained model")
        self.saveModelCheck.setToolTip(
            "<b>Save Trained Model</b><br>"
            "Saves the trained classifier to disk for later reuse without retraining.<br><br>"
            "<i>Use case:</i> Classify multiple rasters with same model, share model with colleagues.<br>"
            "<i>Format:</i> Pickled scikit-learn/XGBoost/LightGBM/CatBoost model file."
        )
        out_layout.addWidget(self.saveModelCheck, 3, 0, 1, 3)
        out_layout.addWidget(QLabel("Model file:"), 4, 0)
        self.saveModelEdit = QLineEdit()
        self.saveModelEdit.setPlaceholderText("Model file path…")
        self.saveModelEdit.setEnabled(False)
        self.saveModelEdit.setToolTip(
            "<b>Model File Path</b><br>"
            "Path where trained model will be saved (.pkl file)."
        )
        out_layout.addWidget(self.saveModelEdit, 4, 1)
        self.saveModelBrowse = QPushButton("Browse…")
        self.saveModelBrowse.setEnabled(False)
        self.saveModelBrowse.clicked.connect(self._browse_save_model)
        out_layout.addWidget(self.saveModelBrowse, 4, 2)

        def _toggle_save_model(checked):
            # type: (bool) -> None
            self.saveModelEdit.setEnabled(checked)
            self.saveModelBrowse.setEnabled(checked)

        self.saveModelCheck.toggled.connect(_toggle_save_model)

        self.matrixCheck = QCheckBox("Save confusion matrix")
        self.matrixCheck.setToolTip(
            "<b>Confusion Matrix</b><br>"
            "Generates accuracy assessment table showing predicted vs. actual classes.<br><br>"
            "<i>Metrics included:</i> Overall accuracy, per-class precision/recall/F1, kappa coefficient.<br>"
            "<i>Note:</i> Requires validation split - portion of training data held out for testing."
        )
        out_layout.addWidget(self.matrixCheck, 5, 0, 1, 3)
        out_layout.addWidget(QLabel("Matrix CSV:"), 6, 0)
        self.matrixEdit = QLineEdit()
        self.matrixEdit.setPlaceholderText("Path to CSV…")
        self.matrixEdit.setEnabled(False)
        out_layout.addWidget(self.matrixEdit, 6, 1)
        self.matrixBrowse = QPushButton("Browse…")
        self.matrixBrowse.setEnabled(False)
        self.matrixBrowse.clicked.connect(self._browse_matrix)
        out_layout.addWidget(self.matrixBrowse, 6, 2)

        out_layout.addWidget(QLabel("Validation split (%):"), 7, 0)
        self.splitSpinBox = ValidatedSpinBox(
            validator_fn=lambda v: 10 <= v <= 90,
            warning_threshold=None  # No warning needed for split percentage
        )
        self.splitSpinBox.setRange(10, 90)
        self.splitSpinBox.setValue(50)
        self.splitSpinBox.setEnabled(False)
        self.splitSpinBox.setToolTip(
            "<b>Validation Split Percentage</b><br>"
            "Percentage of training data used for validation (accuracy assessment).<br><br>"
            "<i>Typical values:</i> 20-30% for large datasets, 40-50% for small datasets.<br>"
            "<i>Note:</i> In POLYGON_GROUP mode, this is interpreted as TRAIN percent "
            "(e.g., 75 = 75% train, 25% validation)."
        )
        out_layout.addWidget(self.splitSpinBox, 7, 1)

        self.previewSplitBtn = QPushButton("Preview Split...")
        self.previewSplitBtn.setEnabled(False)
        self.previewSplitBtn.setToolTip(
            "<b>Preview Split Distribution</b><br>"
            "Shows how training samples will be distributed across train/test sets.<br><br>"
            "Helps identify potential issues:<br>"
            "• Class imbalance (ratio >10:1)<br>"
            "• Small test sets (<5 samples)<br>"
            "• Polygon group leakage (in POLYGON_GROUP mode)"
        )
        self.previewSplitBtn.clicked.connect(self._preview_split)
        out_layout.addWidget(self.previewSplitBtn, 7, 2)

        def _toggle_matrix(checked):
            # type: (bool) -> None
            self.matrixEdit.setEnabled(checked)
            self.matrixBrowse.setEnabled(checked)
            self.splitSpinBox.setEnabled(checked)
            self.previewSplitBtn.setEnabled(checked)

        self.matrixCheck.toggled.connect(_toggle_matrix)

        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # Review summary
        report_group = QGroupBox("Classification Report")
        report_layout = QGridLayout()
        report_layout.setContentsMargins(8, 8, 8, 8)
        report_layout.setHorizontalSpacing(10)
        report_layout.setVerticalSpacing(6)

        self.reportBundleCheck = QCheckBox("Generate classification report (HTML + CSV/JSON + heatmaps)")
        self.reportBundleCheck.setToolTip(
            "<b>Classification Report Bundle</b><br>"
            "Generates comprehensive report with confusion matrix, accuracy metrics, "
            "visualization heatmaps, and run metadata (algorithm, parameters, timestamps).<br><br>"
            "<i>Formats:</i> HTML (interactive), CSV (metrics), JSON (machine-readable), PNG (heatmaps).<br>"
            "<i>Use case:</i> Documentation, quality assessment, reproducibility."
        )
        report_layout.addWidget(self.reportBundleCheck, 0, 0, 1, 3)

        report_layout.addWidget(QLabel("Report output folder (optional):"), 1, 0)
        self.reportFolderEdit = QLineEdit()
        self.reportFolderEdit.setPlaceholderText("Leave empty to create report next to output map")
        self.reportFolderEdit.setEnabled(False)
        self.reportFolderEdit.setToolTip(
            "<b>Report Output Folder</b><br>"
            "Custom folder for classification report files. "
            "Leave empty to auto-create folder next to classification output map."
        )
        report_layout.addWidget(self.reportFolderEdit, 1, 1)
        self.reportFolderBrowse = QPushButton("Browse…")
        self.reportFolderBrowse.setEnabled(False)
        self.reportFolderBrowse.clicked.connect(self._browse_report_folder)
        report_layout.addWidget(self.reportFolderBrowse, 1, 2)

        report_layout.addWidget(QLabel("Label name column (optional):"), 2, 0)
        self.reportLabelColumnEdit = QLineEdit()
        self.reportLabelColumnEdit.setPlaceholderText("e.g. class_name")
        self.reportLabelColumnEdit.setEnabled(False)
        self.reportLabelColumnEdit.setToolTip(
            "<b>Label Name Column</b><br>"
            "Vector field containing human-readable class names (e.g., 'Forest', 'Water').<br><br>"
            "<i>Use case:</i> Makes confusion matrix more readable by showing names instead of numeric codes."
        )
        report_layout.addWidget(self.reportLabelColumnEdit, 2, 1, 1, 2)

        report_layout.addWidget(QLabel("Label mapping (optional):"), 3, 0)
        self.reportLabelMapEdit = QLineEdit()
        self.reportLabelMapEdit.setPlaceholderText("1:Forest,2:Water")
        self.reportLabelMapEdit.setEnabled(False)
        self.reportLabelMapEdit.setToolTip(
            "<b>Label Mapping</b><br>"
            "Manual mapping of class codes to names in format: code:name,code:name<br><br>"
            "<i>Example:</i> 1:Forest,2:Water,3:Urban<br>"
            "<i>Note:</i> Alternative to Label Name Column if vector lacks a name field."
        )
        report_layout.addWidget(self.reportLabelMapEdit, 3, 1, 1, 2)

        self.reportInfoLabel = QLabel(
            "Report bundle includes confusion matrices (numeric + labeled), per-class precision/recall/F1, "
            "heatmap PNG, and a Markdown/JSON run summary with algorithm, split/CV mode, and hyperparameters. "
            "When matrix CSV path is empty, dzetsaka writes it automatically near the output map."
        )
        self.reportInfoLabel.setWordWrap(True)
        self.reportInfoLabel.setEnabled(False)
        report_layout.addWidget(self.reportInfoLabel, 4, 0, 1, 3)

        def _toggle_report_bundle(checked):
            # type: (bool) -> None
            self.reportFolderEdit.setEnabled(checked)
            self.reportFolderBrowse.setEnabled(checked)
            self.reportLabelColumnEdit.setEnabled(checked)
            self.reportLabelMapEdit.setEnabled(checked)
            self.reportInfoLabel.setEnabled(checked)

        self.reportBundleCheck.toggled.connect(_toggle_report_bundle)
        _toggle_report_bundle(False)
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)

        # Review summary
        review_group = QGroupBox("Review Summary")
        review_group.setCheckable(True)
        review_group.setChecked(False)
        review_layout = QVBoxLayout()
        review_layout.setContentsMargins(8, 8, 8, 8)
        review_layout.setSpacing(6)
        review_btn_row = QHBoxLayout()
        self.refreshReviewBtn = QPushButton("Refresh Summary")
        self.refreshReviewBtn.clicked.connect(self._refresh_review)
        review_btn_row.addStretch()
        review_btn_row.addWidget(self.refreshReviewBtn)
        review_layout.addLayout(review_btn_row)
        self.reviewEdit = QTextEdit()
        self.reviewEdit.setReadOnly(True)
        self.reviewEdit.setMinimumHeight(120)
        self.reviewEdit.setMaximumHeight(220)
        review_layout.addWidget(self.reviewEdit)
        review_group.setLayout(review_layout)
        layout.addWidget(review_group)

        def _toggle_review(checked):
            # type: (bool) -> None
            self.refreshReviewBtn.setVisible(checked)
            self.reviewEdit.setVisible(checked)
            if checked:
                self._refresh_review()

        review_group.toggled.connect(_toggle_review)
        _toggle_review(False)

        # Live refresh when outputs change
        self.outRasterEdit.textChanged.connect(self._refresh_review)
        self.confidenceCheck.toggled.connect(self._refresh_review)
        self.confMapEdit.textChanged.connect(self._refresh_review)
        self.saveModelCheck.toggled.connect(self._refresh_review)
        self.saveModelEdit.textChanged.connect(self._refresh_review)
        self.matrixCheck.toggled.connect(self._refresh_review)
        self.matrixEdit.textChanged.connect(self._refresh_review)
        self.splitSpinBox.valueChanged.connect(self._refresh_review)

        layout.addStretch()
        self.setLayout(layout)

    # --- internal helpers --------------------------------------------------

    def _browse_out_raster(self):
        # type: () -> None
        """Browse for the output raster path."""
        path, _f = QFileDialog.getSaveFileName(self, "Classification map", "", "GeoTIFF (*.tif)")
        if path:
            self.outRasterEdit.setText(path)

    def _browse_conf_map(self):
        # type: () -> None
        """Browse for the confidence map path."""
        path, _f = QFileDialog.getSaveFileName(self, "Confidence map", "", "GeoTIFF (*.tif)")
        if path:
            self.confMapEdit.setText(path)

    def _browse_save_model(self):
        # type: () -> None
        """Browse for the model save path."""
        path, _f = QFileDialog.getSaveFileName(self, "Save model", "", "Model files (*)")
        if path:
            self.saveModelEdit.setText(path)

    def _browse_matrix(self):
        # type: () -> None
        """Browse for the confusion matrix CSV path."""
        path, _f = QFileDialog.getSaveFileName(self, "Confusion matrix", "", "CSV (*.csv)")
        if path:
            self.matrixEdit.setText(path)

    def _browse_report_folder(self):
        # type: () -> None
        """Browse for report output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select report output folder", "")
        if path:
            self.reportFolderEdit.setText(path)

    def _preview_split(self):
        # type: () -> None
        """Preview train/test split distribution before running classification."""
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog):
            return

        # Get required parameters from setup dialog
        vector_path = parent_dialog.dataPage.field("vector") or parent_dialog.dataPage.vectorLineEdit.text()
        class_field = parent_dialog.dataPage.get_class_field()
        split_percent = self.splitSpinBox.value()
        cv_mode = parent_dialog.advPage.cvModeCombo.currentText()

        # Validate inputs
        if not vector_path or not class_field:
            QMessageBox.warning(self, "Missing Information", "Please select training vector and class field first.")
            return

        # Show preview dialog
        try:
            dialog = SplitPreviewDialog(vector_path, class_field, split_percent, cv_mode, self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to compute split preview:\n{str(e)}")

    def _refresh_review(self):
        # type: () -> None
        """Refresh the review summary based on current guided workflow state."""
        parent_dialog = self.window()
        if not isinstance(parent_dialog, ClassificationSetupDialog):
            return
        config = parent_dialog.collect_config()
        self.reviewEdit.setPlainText(build_review_summary(config))

    def initializePage(self):
        # type: () -> None
        """Populate the review summary when entering the page."""
        self._refresh_review()

    # --- public API --------------------------------------------------------

    def get_output_config(self):
        # type: () -> Dict[str, object]
        """Return all output-page settings as a dict."""
        out_raster = self.outRasterEdit.text()
        if not out_raster:
            out_raster = ""  # signals 'use temp' to the caller

        conf_map = self.confMapEdit.text() if self.confidenceCheck.isChecked() else ""
        save_model = self.saveModelEdit.text() if self.saveModelCheck.isChecked() else ""
        matrix = self.matrixEdit.text() if self.matrixCheck.isChecked() else ""
        split = self.splitSpinBox.value() if self.matrixCheck.isChecked() else 100

        return {
            "output_raster": out_raster,
            "confidence_map": conf_map,
            "save_model": save_model,
            "confusion_matrix": matrix,
            "split_percent": split,
        }  # type: Dict[str, object]

    def get_report_options(self):
        # type: () -> Dict[str, object]
        """Return report bundle options for advanced post-run reporting."""
        return {
            "GENERATE_REPORT_BUNDLE": self.reportBundleCheck.isChecked(),
            "REPORT_OUTPUT_DIR": self.reportFolderEdit.text().strip(),
            "REPORT_LABEL_COLUMN": self.reportLabelColumnEdit.text().strip(),
            "REPORT_LABEL_MAP": self.reportLabelMapEdit.text().strip(),
        }

    def apply_recipe(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply recipe settings to output toggles."""
        recipe = normalize_recipe(dict(recipe))
        post = recipe.get("postprocess", {})
        validation = recipe.get("validation", {})
        extra = recipe.get("extraParam", {})

        self.confidenceCheck.setChecked(bool(post.get("confidence_map", False)))
        self.saveModelCheck.setChecked(bool(post.get("save_model", False)))

        split = int(validation.get("split_percent", 100))
        matrix = bool(post.get("confusion_matrix", False)) or split < 100
        self.matrixCheck.setChecked(matrix)
        self.splitSpinBox.setValue(split)
        self.reportBundleCheck.setChecked(bool(extra.get("GENERATE_REPORT_BUNDLE", False)))
        self.reportFolderEdit.setText(str(extra.get("REPORT_OUTPUT_DIR", "")))
        self.reportLabelColumnEdit.setText(str(extra.get("REPORT_LABEL_COLUMN", "")))
        self.reportLabelMapEdit.setText(str(extra.get("REPORT_LABEL_MAP", "")))


# ---------------------------------------------------------------------------
# Main Guided Dialog
# ---------------------------------------------------------------------------


class ClassificationSetupDialog(ThemeAwareWidget, QWizard):
    """Step-by-step guided classification dialog for dzetsaka.

    Emits ``classificationRequested`` with the assembled config dict
    when the user clicks Finish (labelled "Run Classification").
    """

    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None, installer=None, close_on_accept=True):
        """Initialise ClassificationSetupDialog with all 3 pages."""
        super(ClassificationSetupDialog, self).__init__(parent)
        self.setWindowTitle("dzetsaka Classification")

        # Apply theme-aware styling
        if _THEME_SUPPORT_AVAILABLE:
            self.apply_theme()
        self._settings = QSettings()
        self._deps = check_dependency_availability()
        self._installer = installer
        self._close_on_accept = bool(close_on_accept)
        self._recipes = load_recipes(self._settings)
        self._remote_recipe_url = self._settings.value("/dzetsaka/recipesRemoteUrl", "", str)

        # Pages
        self.dataPage = DataInputPage(deps=self._deps, installer=self._installer)
        self.advPage = AdvancedOptionsPage(deps=self._deps)
        self.outputPage = OutputConfigPage()

        self.addPage(self.dataPage)  # index 0
        self.addPage(self.advPage)  # index 1
        self.addPage(self.outputPage)  # index 2

        self.dataPage.set_recipe_list(self._recipes)

        # Connect to global recipe update signal
        _recipe_notifier.recipesUpdated.connect(self.reload_recipes_from_settings)

        # Override the Finish button text
        try:
            finish_button = QWizard.FinishButton
        except AttributeError:
            finish_button = QWizard.WizardButton.FinishButton
        self.setButtonText(finish_button, "Run Classification")

        try:
            dialog_style = QWizard.ModernStyle
        except AttributeError:
            dialog_style = QWizard.WizardStyle.ModernStyle
        self.setWizardStyle(dialog_style)

        # Setup keyboard shortcuts
        self._setup_dialog_shortcuts()

    def _setup_dialog_shortcuts(self):
        # type: () -> None
        """Setup keyboard shortcuts for the setup dialog."""
        # Shortcut for Check Data Quality: Ctrl+Shift+Q
        if _QUALITY_CHECKER_AVAILABLE and hasattr(self.dataPage, 'checkQualityBtn'):
            quality_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
            quality_shortcut.activated.connect(self.dataPage._check_data_quality)

        # Shortcut for Next page: Ctrl+Right
        next_shortcut = QShortcut(QKeySequence("Ctrl+Right"), self)
        next_shortcut.activated.connect(self.next)

        # Shortcut for Previous page: Ctrl+Left
        back_shortcut = QShortcut(QKeySequence("Ctrl+Left"), self)
        back_shortcut.activated.connect(self.back)

    # --- page-transition hook ---------------------------------------------

    def validateCurrentPage(self):
        # type: () -> bool
        """Handle smart-defaults propagation when leaving the first page."""
        current = self.currentId()
        # Leaving Input/Algorithm page (index 0) -> entering AdvancedOptionsPage
        if current == 0:
            if self.dataPage.smart_defaults_requested():
                defaults = build_smart_defaults(self._deps)
                self.advPage.apply_smart_defaults(defaults)
                # Reset flag so it fires only once
                self.dataPage._smart_defaults_applied = False
        return super(ClassificationSetupDialog, self).validateCurrentPage()

    # --- recipe helpers ---------------------------------------------------

    def _update_recipes(self, recipes):
        # type: (List[Dict[str, object]]) -> None
        self._recipes = recipes
        save_recipes(self._settings, self._recipes)
        self.dataPage.set_recipe_list(self._recipes)
        # Notify other components
        notify_recipes_updated()

    def reload_recipes_from_settings(self):
        # type: () -> None
        """Reload recipes from QSettings (e.g., after external updates)."""
        self._recipes = load_recipes(self._settings)
        self.dataPage.set_recipe_list(self._recipes)

    def apply_recipe(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply a recipe to the guided workflow UI with dependency validation."""
        recipe = normalize_recipe(dict(recipe))

        # Validate recipe dependencies
        is_valid, missing = validate_recipe_dependencies(recipe)

        if not is_valid:
            # Show warning dialog with missing dependencies
            classifier = recipe.get("classifier", {})
            recipe_name = recipe.get("name", "this recipe")
            classifier_name = classifier.get("name", classifier.get("code", "selected classifier"))
            missing_list = ", ".join(missing)

            reply = QMessageBox.question(
                self,
                "Dependencies Missing for dzetsaka",
                (
                    "To fully use dzetsaka capabilities, we recommend installing all dependencies.<br><br>"
                    f"Recipe: <code>{recipe_name}</code><br>"
                    f"Missing now: <code>{missing_list}</code><br><br>"
                    f"Full bundle to install: <code>{_full_bundle_label()}</code><br><br>"
                    "Install the full dzetsaka dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                to_install = _full_dependency_bundle()

                # Check if installer is available
                if self._installer and hasattr(self._installer, "_try_install_dependencies"):
                    if self._installer._try_install_dependencies(to_install):
                        QMessageBox.information(
                            self,
                            "Installation Successful",
                            f"Dependencies installed successfully!<br><br>"
                            "<b>Important:</b> Please restart QGIS now.<br>"
                            "Without restarting, newly installed libraries may not be loaded, "
                            "and training/classification can fail.",
                            QMessageBox.StandardButton.Ok,
                        )
                        # Re-check after installation
                        is_valid_now, still_missing = validate_recipe_dependencies(recipe)
                        if not is_valid_now:
                            _show_issue_popup(
                                self,
                                self._installer,
                                "Dependencies Still Missing",
                                "Dependency Validation Error",
                                "Some dependencies are still missing after auto-install.",
                                f"Recipe: {recipe.get('name', 'unnamed')}; Missing: {', '.join(still_missing)}",
                            )
                    else:
                        _show_issue_popup(
                            self,
                            self._installer,
                            "Installation Failed",
                            "Dependency Installation Error",
                            "Failed to install dependencies automatically.",
                            f"Recipe: {recipe.get('name', 'unnamed')}; Requested: {', '.join(to_install)}",
                        )
                else:
                    QMessageBox.information(
                        self,
                        "Manual Installation Required",
                        "Please install the full dependency bundle manually using:\n"
                        f"pip install {' '.join(to_install)}",
                    )
            else:
                # User chose not to install, show warning
                _show_issue_popup(
                    self,
                    self._installer,
                    "Recipe May Not Work",
                    "Missing Dependencies",
                    "User chose not to install required dependencies.",
                    f"Recipe: {recipe.get('name', 'unnamed')}; Missing: {', '.join(missing)}",
                )

        # Apply the recipe to UI pages
        classifier = recipe.get("classifier", {})
        code = classifier.get("code", "GMM")
        self.dataPage.set_dependency_prompt_suppressed(True)
        try:
            self.dataPage.set_classifier_by_code(code)
        finally:
            self.dataPage.set_dependency_prompt_suppressed(False)
        self.advPage.apply_recipe(recipe)
        self.outputPage.apply_recipe(recipe)

    def save_current_recipe(self):
        # type: () -> None
        """Save the current guided workflow configuration as a recipe."""
        name, ok = QInputDialog.getText(self, "Save Recipe", "Recipe name:")
        if not ok or not name.strip():
            return

        # Check if trying to overwrite a template
        for r in self._recipes:
            if r.get("name") == name.strip() and r.get("is_template", False):
                QMessageBox.warning(
                    self,
                    "Cannot Overwrite Template",
                    f"'{name.strip()}' is a built-in template and cannot be overwritten.\n\n"
                    "Please choose a different name for your custom recipe.",
                )
                return

        description, _ok = QInputDialog.getText(self, "Save Recipe", "Description (optional):")
        config = self.collect_config()
        recipe = recipe_from_config(config, name.strip(), description.strip())
        # Ensure it's marked as user recipe (remove template metadata)
        if "metadata" in recipe and isinstance(recipe["metadata"], dict):
            recipe["metadata"]["is_template"] = False

        if any(r.get("name") == recipe.get("name") for r in self._recipes):
            reply = QMessageBox.question(
                self,
                "Overwrite recipe?",
                f"A recipe named '{recipe.get('name')}' already exists. Overwrite it?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        updated = [r for r in self._recipes if r.get("name") != recipe.get("name")]
        updated.append(recipe)
        self._update_recipes(updated)

    def open_recipe_gallery(self):
        # type: () -> None
        """Open the local recipe gallery."""
        dialog = RecipeGalleryDialog(
            parent=self,
            recipes=list(self._recipes),
        )
        dialog.recipeApplied.connect(self.apply_recipe)
        dialog.recipesUpdated.connect(self._update_recipes)
        try:
            dialog.exec_()
        except AttributeError:
            dialog.exec()

    def _set_remote_recipe_url(self, url):
        # type: (str) -> None
        self._remote_recipe_url = url
        self._settings.setValue("/dzetsaka/recipesRemoteUrl", url)

    def _apply_config_dict(self, config):
        # type: (Dict[str, object]) -> None
        """Apply a full classification config dictionary to Expert mode controls."""
        raster = str(config.get("raster", "") or "")
        vector = str(config.get("vector", "") or "")
        class_field = str(config.get("class_field", "") or "")
        load_model = str(config.get("load_model", "") or "")
        classifier_code = str(config.get("classifier", "GMM") or "GMM")

        self.dataPage.rasterLineEdit.setText(raster)
        self.dataPage.vectorLineEdit.setText(vector)
        if vector:
            self.dataPage._populate_fields_from_path(vector)
        if class_field:
            idx = self.dataPage.classFieldCombo.findText(class_field)
            if idx < 0:
                self.dataPage.classFieldCombo.addItem(class_field)
                idx = self.dataPage.classFieldCombo.findText(class_field)
            if idx >= 0:
                self.dataPage.classFieldCombo.setCurrentIndex(idx)
        self.dataPage.loadModelCheck.setChecked(bool(load_model))
        self.dataPage.modelLineEdit.setText(load_model)
        self.dataPage.set_dependency_prompt_suppressed(True)
        try:
            self.dataPage.set_classifier_by_code(classifier_code)
        finally:
            self.dataPage.set_dependency_prompt_suppressed(False)

        extra = config.get("extraParam", {})
        if not isinstance(extra, dict):
            extra = {}
        split_percent = config.get("split_percent", 100)
        try:
            split_percent = int(split_percent)
        except (TypeError, ValueError):
            split_percent = 100
        pseudo_recipe = normalize_recipe(
            {
                "classifier": {"code": classifier_code},
                "validation": {
                    "split_percent": split_percent,
                    "nested_cv": bool(extra.get("USE_NESTED_CV", False)),
                    "nested_inner_cv": int(extra.get("NESTED_INNER_CV", 3)),
                    "nested_outer_cv": int(extra.get("NESTED_OUTER_CV", 5)),
                    "cv_mode": str(extra.get("CV_MODE", "POLYGON_GROUP")),
                },
                "postprocess": {
                    "confidence_map": bool(config.get("confidence_map", "")),
                    "save_model": bool(config.get("save_model", "")),
                    "confusion_matrix": bool(config.get("confusion_matrix", "")) or split_percent < 100,
                },
                "extraParam": extra,
            }
        )
        self.advPage.apply_recipe(pseudo_recipe)
        self.outputPage.apply_recipe(pseudo_recipe)

        self.outputPage.outRasterEdit.setText(str(config.get("output_raster", "") or ""))
        self.outputPage.confMapEdit.setText(str(config.get("confidence_map", "") or ""))
        self.outputPage.saveModelEdit.setText(str(config.get("save_model", "") or ""))
        self.outputPage.matrixEdit.setText(str(config.get("confusion_matrix", "") or ""))

    def _coerce_external_payload_to_config(self, payload):
        # type: (Dict[str, object]) -> Dict[str, object]
        """Normalize imported JSON payload into a config dict accepted by Expert mode."""
        if any(k in payload for k in ("raster", "vector", "class_field", "classifier", "extraParam")):
            return dict(payload)

        # Sprint 1 artifact support: run_manifest.json
        if payload.get("artifact") == "dzetsaka_run_manifest" and isinstance(payload.get("run"), dict):
            run = payload.get("run", {})
            return {
                "raster": str(run.get("raster_path", "") or ""),
                "vector": str(run.get("vector_path", "") or ""),
                "class_field": str(run.get("class_field", "") or ""),
                "load_model": "",
                "classifier": str(run.get("classifier_code", "GMM") or "GMM"),
                "extraParam": {
                    "CV_MODE": str(run.get("split_mode", "RANDOM_SPLIT") or "RANDOM_SPLIT"),
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
                    "GENERATE_REPORT_BUNDLE": False,
                    "REPORT_OUTPUT_DIR": "",
                    "REPORT_LABEL_COLUMN": "",
                    "REPORT_LABEL_MAP": "",
                },
                "output_raster": "",
                "confidence_map": "",
                "save_model": "",
                "confusion_matrix": "",
                "split_percent": int(run.get("split_config", 100) or 100),
            }

        # Fallback: report bundle run_config.json style
        classifier_code = str(payload.get("classifier_code", "GMM") or "GMM")
        split_mode = str(payload.get("split_mode", "RANDOM_SPLIT") or "RANDOM_SPLIT")
        split_config = payload.get("split_config", 100)
        split_percent = 100
        try:
            split_percent = int(split_config)
        except (TypeError, ValueError):
            split_percent = 75
        return {
            "raster": str(payload.get("raster_path", "") or ""),
            "vector": str(payload.get("vector_path", "") or ""),
            "class_field": str(payload.get("class_field", "") or ""),
            "load_model": "",
            "classifier": classifier_code,
            "extraParam": {
                "CV_MODE": split_mode,
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
                "GENERATE_REPORT_BUNDLE": False,
                "REPORT_OUTPUT_DIR": "",
                "REPORT_LABEL_COLUMN": "",
                "REPORT_LABEL_MAP": "",
            },
            "output_raster": "",
            "confidence_map": "",
            "save_model": "",
            "confusion_matrix": str(payload.get("matrix_path", "") or ""),
            "split_percent": split_percent,
        }

    def _import_payload(self, payload):
        # type: (Dict[str, object]) -> None
        """Import recipe/config JSON payload and apply it to Expert mode."""
        # Recipe JSON
        if "postprocess" in payload and "validation" in payload and "classifier" in payload:
            self.apply_recipe(payload)
            return
        config = self._coerce_external_payload_to_config(payload)
        self._apply_config_dict(config)

    def import_config_from_json_file(self):
        # type: () -> None
        """Load and apply a JSON recipe/config file."""
        path, _f = QFileDialog.getOpenFileName(self, "Load configuration JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            QMessageBox.warning(self, "JSON load error", f"Failed to read JSON file:\n{exc!s}")
            return
        if isinstance(payload, dict) and "recipes" in payload and isinstance(payload.get("recipes"), list):
            recipes = payload.get("recipes") or []
            if not recipes:
                QMessageBox.warning(self, "JSON load error", "No recipe found in this file.")
                return
            payload = recipes[0]
        elif isinstance(payload, list):
            if not payload:
                QMessageBox.warning(self, "JSON load error", "JSON list is empty.")
                return
            payload = payload[0]
        if not isinstance(payload, dict):
            QMessageBox.warning(
                self, "JSON load error", "JSON root must be an object, recipe list, or recipes payload."
            )
            return
        self._import_payload(payload)

    def import_config_from_json_paste(self):
        # type: () -> None
        """Paste and apply JSON recipe/config."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Paste JSON Configuration",
            "Paste a recipe JSON, full config JSON, or run_config.json here:",
        )
        if not ok or not text.strip():
            return
        try:
            payload = json.loads(text)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Could not parse JSON:\n{exc!s}")
            return
        if isinstance(payload, dict) and "recipes" in payload and isinstance(payload.get("recipes"), list):
            recipes = payload.get("recipes") or []
            if not recipes:
                QMessageBox.warning(self, "JSON import error", "No recipe found in pasted payload.")
                return
            payload = recipes[0]
        elif isinstance(payload, list):
            if not payload:
                QMessageBox.warning(self, "JSON import error", "JSON list is empty.")
                return
            payload = payload[0]
        if not isinstance(payload, dict):
            QMessageBox.warning(self, "JSON import error", "Payload must be a JSON object.")
            return
        self._import_payload(payload)

    # --- config assembly --------------------------------------------------

    def collect_config(self):
        # type: () -> Dict[str, object]
        """Assemble the full config dict from all guided workflow pages."""
        config = {}  # type: Dict[str, object]

        # Input data
        config["raster"] = self.dataPage.get_raster_path()
        config["vector"] = self.dataPage.get_vector_path()
        config["class_field"] = self.dataPage.get_class_field()
        config["load_model"] = self.dataPage.modelLineEdit.text()

        # Algorithm
        config["classifier"] = self.dataPage.get_classifier_code()

        # Advanced options
        config["extraParam"] = self.advPage.get_extra_params()
        config["extraParam"].update(self.outputPage.get_report_options())

        # Output
        config.update(self.outputPage.get_output_config())
        if config["extraParam"].get("GENERATE_REPORT_BUNDLE", False) and int(config.get("split_percent", 100)) >= 100:
            config["split_percent"] = 75

        return config

    # --- finish handler ---------------------------------------------------

    def accept(self):
        # type: () -> None
        """Emit the config signal and close the dialog."""
        config = self.collect_config()
        self.classificationRequested.emit(config)
        if self._close_on_accept:
            super(ClassificationSetupDialog, self).accept()


class RecipeHubDialog(QDialog):
    """Quad-layout recipe hub inspired by e-commerce mega menus."""

    def __init__(self, parent=None, recipes=None, current_recipe_name="", compact_mode=True):
        # type: (Optional[QWidget], Optional[List[Dict[str, object]]], str, bool) -> None
        super(RecipeHubDialog, self).__init__(parent)
        self.setWindowTitle("Recipe Hub")
        self._compact_mode = bool(compact_mode)
        if self._compact_mode:
            self.resize(680, 430)
            self.setMinimumSize(640, 400)
        else:
            self.resize(960, 620)
            self.setMinimumSize(760, 460)
        self._command = "apply"
        self._selected_recipe_name = ""
        self._current_recipe_name = str(current_recipe_name or "").strip()
        self._active_category = ""
        self._active_chip = "ALL"
        self._preset_focus = ""
        self._recipes = [normalize_recipe(dict(r)) for r in (recipes or []) if isinstance(r, dict)]
        self._category_items = {}  # type: Dict[str, List[Dict[str, object]]]

        root = QGridLayout(self)
        if self._compact_mode:
            root.setContentsMargins(4, 4, 4, 4)
            root.setHorizontalSpacing(4)
            root.setVerticalSpacing(4)
        else:
            root.setContentsMargins(10, 10, 10, 10)
            root.setHorizontalSpacing(8)
            root.setVerticalSpacing(8)

        self.headerTitle = QLabel("Recipe Hub")
        self.headerTitle.setObjectName("recipeHubTitle")
        self.headerSubtitle = QLabel("Browse templates like a web mega menu: pick category, preview recipe, apply.")
        self.headerSubtitle.setObjectName("recipeHubSubtitle")
        self.fastPresetBtn = QToolButton()
        self.fastPresetBtn.setText("Fast")
        self.fastPresetBtn.setObjectName("hubQuickPreset")
        self.qualityPresetBtn = QToolButton()
        self.qualityPresetBtn.setText("Highest quality")
        self.qualityPresetBtn.setObjectName("hubQuickPreset")
        self.searchEdit = QLineEdit()
        self.searchEdit.setPlaceholderText("Search recipes...")
        self.searchEdit.setClearButtonEnabled(True)
        self.searchEdit.setObjectName("hubSearch")
        self.headerTools = QWidget()
        header_tools_layout = QHBoxLayout(self.headerTools)
        header_tools_layout.setContentsMargins(0, 0, 0, 0)
        header_tools_layout.setSpacing(4)
        header_tools_layout.addWidget(self.fastPresetBtn)
        header_tools_layout.addWidget(self.qualityPresetBtn)
        header_tools_layout.addWidget(self.searchEdit)
        if self._compact_mode:
            self.headerSubtitle.setVisible(False)
        root.addWidget(self.headerTitle, 0, 0, 1, 1)
        root.addWidget(self.headerTools, 0, 1, 1, 1, _qt_align_right())
        root.addWidget(self.headerSubtitle, 1, 0, 1, 2)

        self.topBar = QFrame()
        self.topBar.setObjectName("hubTopBar")
        top_layout = QHBoxLayout(self.topBar)
        top_layout.setContentsMargins(8, 6, 8, 6)
        top_layout.setSpacing(6)
        self.statsLabel = QLabel("")
        self.statsLabel.setObjectName("hubStatPill")
        top_layout.addWidget(self.statsLabel)
        self.chipButtons = {}  # type: Dict[str, QToolButton]
        for key, label in [
            ("ALL", "All"),
            ("OPTUNA", "Optuna"),
            ("SHAP", "SHAP"),
            ("SMOTE", "SMOTE"),
            ("CUSTOM", "Custom"),
        ]:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setObjectName("hubChip")
            btn.clicked.connect(lambda _checked=False, k=key: self._set_active_chip(k))
            self.chipButtons[key] = btn
            top_layout.addWidget(btn)
        top_layout.addStretch()
        self.sortCombo = QComboBox()
        self.sortCombo.addItems(["Featured", "Name (A-Z)", "Fastest", "Highest Accuracy"])
        self.sortCombo.setObjectName("hubSort")
        top_layout.addWidget(self.sortCombo)
        root.addWidget(self.topBar, 2, 0, 1, 2)
        if self._compact_mode:
            self.topBar.setVisible(False)

        categories_group = QGroupBox("Categories")
        categories_group.setObjectName("hubPanel")
        categories_layout = QVBoxLayout(categories_group)
        categories_layout.setContentsMargins(4, 8, 4, 4)
        self.categoryList = QListWidget()
        self.categoryList.setObjectName("hubList")
        self.categoryList.setSpacing(1 if self._compact_mode else 4)
        self.categoryList.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.categoryList.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        categories_layout.addWidget(self.categoryList)
        root.addWidget(categories_group, 3, 0)
        if self._compact_mode:
            categories_group.setMaximumWidth(220)

        recipes_group = QGroupBox("Recipes")
        recipes_group.setObjectName("hubPanel")
        recipes_layout = QVBoxLayout(recipes_group)
        recipes_layout.setContentsMargins(4, 8, 4, 4)
        self.recipeList = QListWidget()
        self.recipeList.setObjectName("hubList")
        self.recipeList.setSpacing(1 if self._compact_mode else 4)
        recipes_layout.addWidget(self.recipeList)
        root.addWidget(recipes_group, 3, 1)

        actions_group = QGroupBox("Quick Actions")
        actions_group.setObjectName("hubPanel")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setContentsMargins(4, 8, 4, 4)
        self.createBtn = QPushButton("Create / Customize...")
        self.importBtn = QPushButton("Import Recipe...")
        self.exportBtn = QPushButton("Export Selected")
        self.createBtn.setObjectName("hubPrimary")
        self.importBtn.setObjectName("hubSecondary")
        self.exportBtn.setObjectName("hubSecondary")
        actions_layout.addWidget(self.createBtn)
        actions_layout.addWidget(self.importBtn)
        actions_layout.addWidget(self.exportBtn)
        actions_layout.addStretch()
        root.addWidget(actions_group, 4, 0)
        if self._compact_mode:
            actions_group.setMaximumWidth(220)

        details_group = QGroupBox("Recipe Details")
        details_group.setObjectName("hubPanel")
        details_layout = QVBoxLayout(details_group)
        details_layout.setContentsMargins(4, 8, 4, 4)
        self.detailsLabel = QLabel("Select a recipe to view details.")
        self.detailsLabel.setWordWrap(True)
        self.detailsLabel.setTextFormat(Qt.RichText)
        self.detailsLabel.setObjectName("detailsCard")
        details_layout.addWidget(self.detailsLabel)
        footer = QHBoxLayout()
        footer.addStretch()
        self.applyBtn = QPushButton("Apply Recipe")
        self.cancelBtn = QPushButton("Close")
        self.applyBtn.setObjectName("hubPrimary")
        self.cancelBtn.setObjectName("hubGhost")
        self.applyBtn.setEnabled(False)
        if self._compact_mode:
            self.createBtn.setFixedHeight(26)
            self.importBtn.setFixedHeight(26)
            self.exportBtn.setFixedHeight(26)
            self.applyBtn.setFixedHeight(26)
            self.cancelBtn.setFixedHeight(26)
        footer.addWidget(self.cancelBtn)
        footer.addWidget(self.applyBtn)
        details_layout.addLayout(footer)
        root.addWidget(details_group, 4, 1)
        if self._compact_mode:
            actions_group.setMaximumHeight(150)
            details_group.setMaximumHeight(190)

        root.setColumnStretch(0, 3)
        root.setColumnStretch(1, 7)
        root.setRowStretch(3, 5 if self._compact_mode else 7)
        root.setRowStretch(4, 2 if self._compact_mode else 6)

        self.categoryList.currentRowChanged.connect(self._on_category_changed)
        self.searchEdit.textChanged.connect(self._on_search_changed)
        self.fastPresetBtn.clicked.connect(self._select_fast_recipe)
        self.qualityPresetBtn.clicked.connect(self._select_highest_quality_recipe)
        self.sortCombo.currentIndexChanged.connect(self._on_sort_changed)
        self.recipeList.currentItemChanged.connect(self._on_recipe_changed)
        self.recipeList.itemDoubleClicked.connect(lambda _item: self._apply_selection())
        self.applyBtn.clicked.connect(self._apply_selection)
        self.cancelBtn.clicked.connect(self.reject)

        self.createBtn.clicked.connect(lambda: self._finish_with_command("create"))
        self.importBtn.clicked.connect(lambda: self._finish_with_command("import"))
        self.exportBtn.clicked.connect(lambda: self._finish_with_command("export"))
        self._alt_f4_shortcut = QShortcut(QKeySequence("Alt+F4"), self)
        self._alt_f4_shortcut.activated.connect(self.reject)

        self._apply_web_style()
        self._populate_categories()
        self._set_active_chip("ALL")
        self._restore_initial_selection()

    def command(self):
        # type: () -> str
        return self._command

    def selected_recipe_name(self):
        # type: () -> str
        return self._selected_recipe_name

    def _finish_with_command(self, command_name):
        # type: (str) -> None
        self._command = command_name
        self.accept()

    def _populate_categories(self):
        # type: () -> None
        self.categoryList.clear()
        self._category_items = {}
        self._categories = []  # type: List[str]
        templates_by_category = {"Beginner": [], "Intermediate": [], "Advanced": []}
        user_recipes = []
        for recipe in self._recipes:
            if is_builtin_recipe(recipe):
                metadata = recipe.get("metadata", {})
                category = metadata.get("category", "intermediate") if isinstance(metadata, dict) else "intermediate"
                category = str(category).capitalize()
                if category not in templates_by_category:
                    category = "Intermediate"
                templates_by_category[category].append(recipe)
            else:
                user_recipes.append(recipe)

        all_recipes = list(self._recipes)
        self._categories.append("All")
        self._category_items["All"] = all_recipes
        item = QListWidgetItem(f"All ({len(all_recipes)})")
        item.setSizeHint(QSize(0, 24 if self._compact_mode else 34))
        self.categoryList.addItem(item)

        for label in ("Beginner", "Intermediate", "Advanced"):
            count = len(templates_by_category[label])
            if count > 0:
                self._categories.append(label)
                self._category_items[label] = templates_by_category[label]
                item = QListWidgetItem(f"{label} ({count})")
                item.setSizeHint(QSize(0, 24 if self._compact_mode else 34))
                self.categoryList.addItem(item)
        if user_recipes:
            self._categories.append("My Recipes")
            self._category_items["My Recipes"] = user_recipes
            item = QListWidgetItem(f"My Recipes ({len(user_recipes)})")
            item.setSizeHint(QSize(0, 24 if self._compact_mode else 34))
            self.categoryList.addItem(item)

        if self.categoryList.count() > 0:
            self.categoryList.setCurrentRow(0)
        self._autosize_category_list()

    def _autosize_category_list(self):
        # type: () -> None
        if self.categoryList.count() <= 0:
            return
        default_row_height = 24 if self._compact_mode else 34
        row_heights = []
        for idx in range(self.categoryList.count()):
            hint = self.categoryList.sizeHintForRow(idx)
            row_heights.append(hint if hint > 0 else default_row_height)
        content_height = sum(row_heights)
        spacing = self.categoryList.spacing()
        frame = self.categoryList.frameWidth() * 2
        visible = content_height + (spacing * max(0, self.categoryList.count() - 1)) + frame + 8
        # Keep a little headroom to avoid clipping/auto-scroll jitter on selection.
        self.categoryList.setMinimumHeight(visible)
        self.categoryList.setMaximumHeight(visible + default_row_height)

    def _recipes_for_category(self, category):
        # type: (str) -> List[Dict[str, object]]
        if category not in self._category_items:
            return list(self._category_items.get("All", []))
        return list(self._category_items.get(category, []))

    def _recipe_feature_tags(self, recipe):
        # type: (Dict[str, object]) -> List[str]
        extra = recipe.get("extraParam", {})
        tags = []
        if extra.get("USE_OPTUNA"):
            tags.append("Optuna")
        if extra.get("COMPUTE_SHAP"):
            tags.append("SHAP")
        if extra.get("USE_SMOTE"):
            tags.append("SMOTE")
        if extra.get("USE_CLASS_WEIGHTS"):
            tags.append("Class weights")
        if extra.get("USE_NESTED_CV") or recipe.get("validation", {}).get("nested_cv"):
            tags.append("Nested CV")
        return tags

    def _classifier_name(self, recipe):
        # type: (Dict[str, object]) -> str
        classifier = recipe.get("classifier", {})
        classifier_code = str(classifier.get("code", "GMM"))
        for code, label, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            if code == classifier_code:
                return label
        return classifier_code

    def _accuracy_rank(self, recipe):
        # type: (Dict[str, object]) -> int
        value = str(recipe.get("expected_accuracy_class", "")).lower()
        if "very_high" in value or "highest" in value:
            return 4
        if "high" in value:
            return 3
        if "medium" in value:
            return 2
        if "low" in value or "baseline" in value:
            return 1
        return 2

    def _runtime_rank(self, recipe):
        # type: (Dict[str, object]) -> int
        value = str(recipe.get("expected_runtime_class", "")).lower()
        if "fast" in value:
            return 1
        if "slow" in value:
            return 3
        return 2

    def _sort_recipes(self, recipes):
        # type: (List[Dict[str, object]]) -> List[Dict[str, object]]
        mode = self.sortCombo.currentText().strip().lower()
        items = list(recipes)
        if mode.startswith("name"):
            return sorted(items, key=lambda r: str(r.get("name", "")).lower())
        if mode.startswith("fastest"):
            return sorted(items, key=lambda r: (self._runtime_rank(r), str(r.get("name", "")).lower()))
        if mode.startswith("highest"):
            return sorted(items, key=lambda r: (-self._accuracy_rank(r), str(r.get("name", "")).lower()))
        return sorted(
            items,
            key=lambda r: (
                0 if is_builtin_recipe(r) else 1,
                self._runtime_rank(r),
                -self._accuracy_rank(r),
                str(r.get("name", "")).lower(),
            ),
        )

    def _passes_chip_filter(self, recipe):
        # type: (Dict[str, object]) -> bool
        if self._active_chip == "ALL":
            return True
        tags = {t.upper() for t in self._recipe_feature_tags(recipe)}
        if self._active_chip == "CUSTOM":
            return not is_builtin_recipe(recipe)
        return self._active_chip in tags

    def _set_active_chip(self, key, clear_preset=True):
        # type: (str, bool) -> None
        if clear_preset:
            self._preset_focus = ""
        self._active_chip = key
        for chip_key, btn in self.chipButtons.items():
            btn.setChecked(chip_key == key)
        self._rebuild_recipe_list()

    def _apply_preset_focus(self, recipes):
        # type: (List[Dict[str, object]]) -> List[Dict[str, object]]
        if not recipes:
            return []
        if self._preset_focus == "FAST":
            best_runtime = min(self._runtime_rank(r) for r in recipes)
            return [r for r in recipes if self._runtime_rank(r) == best_runtime]
        if self._preset_focus == "QUALITY":
            best_quality = max(self._accuracy_rank(r) for r in recipes)
            return [r for r in recipes if self._accuracy_rank(r) == best_quality]
        return recipes

    def _update_stats_pill(self):
        # type: () -> None
        search_term = self.searchEdit.text().strip()
        if search_term:
            total = len(self._recipes)
            category = "All"
        else:
            total = len(self._recipes_for_category(self._active_category)) if self._active_category else 0
            category = self._active_category if self._active_category else "None"
        shown = self.recipeList.count()
        chip = self.chipButtons.get(self._active_chip)
        chip_label = chip.text() if chip else "All"
        preset_label = ""
        if self._preset_focus == "FAST":
            preset_label = " | Preset: Fast"
        elif self._preset_focus == "QUALITY":
            preset_label = " | Preset: Highest quality"
        self.statsLabel.setText(f"{shown} / {total} recipes | {category} | Filter: {chip_label}{preset_label}")

    def _build_recipe_card_widget(self, recipe):
        # type: (Dict[str, object]) -> QWidget
        # Legacy helper kept for compatibility. Current UI uses title-only rows.
        card = QFrame()
        card.setObjectName("recipeCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        name = str(recipe.get("name", "Unnamed Recipe"))
        classifier_name = self._classifier_name(recipe)
        metadata = recipe.get("metadata", {})
        category = metadata.get("category", "Custom") if isinstance(metadata, dict) else "Custom"
        description = str(recipe.get("description", "No description available"))
        description = description if len(description) <= 130 else description[:127] + "..."
        runtime = str(recipe.get("expected_runtime_class", "medium")).capitalize()
        accuracy = str(recipe.get("expected_accuracy_class", "high")).capitalize()
        tags = self._recipe_feature_tags(recipe)

        top = QLabel(
            f"<span style='font-size:14px; font-weight:700; color:#102a43;'>{name}</span> "
            f"<span style='color:#5f748a;'>({classifier_name})</span>"
        )
        top.setTextFormat(Qt.RichText)
        layout.addWidget(top)

        meta = QLabel(
            f"<span style='color:#37516a;'>Category: {category}</span>  •  "
            f"<span style='color:#37516a;'>Runtime: {runtime}</span>  •  "
            f"<span style='color:#37516a;'>Accuracy: {accuracy}</span>"
        )
        meta.setTextFormat(Qt.RichText)
        layout.addWidget(meta)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#334e68;")
        layout.addWidget(desc)

        badges = QLabel("  ".join([f"#{t.replace(' ', '')}" for t in tags]) if tags else "#Baseline")
        badges.setStyleSheet("color:#0b74de; font-weight:600;")
        layout.addWidget(badges)
        return card

    def _rebuild_recipe_list(self):
        # type: () -> None
        self.recipeList.clear()
        self.applyBtn.setEnabled(False)
        self.detailsLabel.setText("Select a recipe to view details.")
        if not self._active_category or self._active_category not in self._category_items:
            self._active_category = "All"
        search_term = self.searchEdit.text().strip().lower()
        base_recipes = list(self._recipes) if search_term else self._recipes_for_category(self._active_category)
        recipes = self._sort_recipes(base_recipes)
        recipes = self._apply_preset_focus(recipes)
        for recipe in recipes:
            if not self._passes_chip_filter(recipe):
                continue
            name = str(recipe.get("name", "Unnamed Recipe"))
            desc = str(recipe.get("description", ""))
            classifier_code = str(recipe.get("classifier", {}).get("code", "GMM")).lower()
            tags = [tag.lower() for tag in self._recipe_feature_tags(recipe)]
            blob = f"{name} {desc} {classifier_code} {' '.join(tags)}".lower()
            if search_term and search_term not in blob:
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            item.setToolTip(desc)
            item.setSizeHint(QSize(0, 24 if self._compact_mode else 32))
            self.recipeList.addItem(item)
        if self.recipeList.count() > 0:
            self.recipeList.setCurrentRow(0)
        else:
            self.detailsLabel.setText("No recipes match this filter.")
        self._update_stats_pill()

    def _on_category_changed(self, row):
        # type: (int) -> None
        self._preset_focus = ""
        if row < 0 or row >= len(getattr(self, "_categories", [])):
            self._active_category = "All"
            self._rebuild_recipe_list()
            return
        self._active_category = self._categories[row]
        self._rebuild_recipe_list()

    def _on_search_changed(self, _text):
        # type: (str) -> None
        self._preset_focus = ""
        # Search always operates on All recipes; keep All selected while filtering.
        if self.searchEdit.text().strip() and self.categoryList.count() > 0 and self.categoryList.currentRow() != 0:
            self.categoryList.setCurrentRow(0)
            return
        self._rebuild_recipe_list()

    def _on_sort_changed(self, _index):
        # type: (int) -> None
        self._preset_focus = ""
        self._rebuild_recipe_list()

    def _ensure_all_category_selected(self):
        # type: () -> None
        if self.categoryList.count() <= 0:
            return
        if self.categoryList.currentRow() != 0:
            self.categoryList.setCurrentRow(0)

    def _select_fast_recipe(self):
        # type: () -> None
        self._ensure_all_category_selected()
        self.searchEdit.clear()
        idx = self.sortCombo.findText("Fastest")
        if idx >= 0:
            self.sortCombo.setCurrentIndex(idx)
        self._preset_focus = "FAST"
        self._set_active_chip("ALL", clear_preset=False)
        self._rebuild_recipe_list()
        if self.recipeList.count() > 0:
            self.recipeList.setCurrentRow(0)

    def _select_highest_quality_recipe(self):
        # type: () -> None
        self._ensure_all_category_selected()
        self.searchEdit.clear()
        idx = self.sortCombo.findText("Highest Accuracy")
        if idx >= 0:
            self.sortCombo.setCurrentIndex(idx)
        self._preset_focus = "QUALITY"
        self._set_active_chip("ALL", clear_preset=False)
        self._rebuild_recipe_list()
        if self.recipeList.count() > 0:
            self.recipeList.setCurrentRow(0)

    def _details_html(self, recipe):
        # type: (Dict[str, object]) -> str
        name = str(recipe.get("name", "Unnamed Recipe"))
        description = str(recipe.get("description", "No description available"))
        classifier = recipe.get("classifier", {})
        classifier_code = str(classifier.get("code", "GMM"))
        classifier_name = classifier_code
        for code, label, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            if code == classifier_code:
                classifier_name = label
                break
        metadata = recipe.get("metadata", {})
        category = metadata.get("category", "Custom") if isinstance(metadata, dict) else "Custom"
        extra = recipe.get("extraParam", {})
        features = []
        if extra.get("USE_OPTUNA"):
            features.append("Optuna")
        if extra.get("COMPUTE_SHAP"):
            features.append("SHAP")
        if extra.get("USE_SMOTE"):
            features.append("SMOTE")
        if extra.get("USE_CLASS_WEIGHTS"):
            features.append("Class weights")
        if extra.get("USE_NESTED_CV") or recipe.get("validation", {}).get("nested_cv"):
            features.append("Nested CV")
        features_text = ", ".join(features) if features else "Baseline"
        guidance = ""
        if classifier_code == "CB":
            guidance = (
                "<br><span style='color:#2f6f44;'>"
                "CatBoost typically delivers stronger accuracy than Random Forest with efficient CPU usage."
                "</span>"
            )
        return (
            f"<b>{name}</b><br>"
            f"<span style='color:#2f4f6a;'><b>{classifier_name}</b> | {category}</span><br>"
            f"<span style='color:#5f748a;'>Features: {features_text}</span><br><br>"
            f"{description}{guidance}<br><br>"
            f"<span style='color:#7c8ea1;'>Tip: double-click recipe card to apply instantly.</span>"
        )

    def _on_recipe_changed(self, current, _previous):
        # type: (Optional[QListWidgetItem], Optional[QListWidgetItem]) -> None
        if current is None:
            self._selected_recipe_name = ""
            self.applyBtn.setEnabled(False)
            self.detailsLabel.setText("Select a recipe to view details.")
            return
        recipe_name = str(current.data(Qt.UserRole) or "").strip()
        self._selected_recipe_name = recipe_name
        self.applyBtn.setEnabled(bool(recipe_name))
        selected = None
        for recipe in self._recipes:
            if str(recipe.get("name", "")).strip() == recipe_name:
                selected = recipe
                break
        if selected is None:
            self.detailsLabel.setText("Select a recipe to view details.")
            return
        self.detailsLabel.setText(self._details_html(selected))

    def _restore_initial_selection(self):
        # type: () -> None
        # Always start on All Recipes; optionally pre-select current recipe in that list.
        if self.categoryList.count() > 0:
            self.categoryList.setCurrentRow(0)
        if not self._current_recipe_name or self.recipeList.count() <= 0:
            return
        target = self._current_recipe_name
        for idx in range(self.recipeList.count()):
            item = self.recipeList.item(idx)
            if item and str(item.data(Qt.UserRole) or "").strip() == target:
                self.recipeList.setCurrentRow(idx)
                break

    def _apply_selection(self):
        # type: () -> None
        if not self._selected_recipe_name:
            return
        self._command = "apply"
        self.accept()

    def _apply_web_style(self):
        # type: () -> None
        if self._compact_mode:
            # Compact mode: mimic QGIS plugin-manager palette and density.
            self.setStyleSheet(
                """
                QDialog {
                    background: #efefef;
                }
                QLabel#recipeHubTitle {
                    color: #111111;
                    font-size: 14px;
                    font-weight: 700;
                }
                QLabel#recipeHubSubtitle {
                    color: #555555;
                    font-size: 11px;
                }
                QGroupBox {
                    background: #f7f7f7;
                    border: 0;
                    border-radius: 0;
                    margin-top: 6px;
                    color: #222222;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 6px;
                    padding: 0 3px;
                }
                QLineEdit#hubSearch, QComboBox#hubSort {
                    background: #ffffff;
                    border: 0;
                    border-radius: 0;
                    padding: 2px 6px;
                    color: #222222;
                }
                QToolButton#hubQuickPreset {
                    background: #f4f4f4;
                    color: #222222;
                    border: 0;
                    padding: 2px 7px;
                    min-height: 22px;
                }
                QToolButton#hubQuickPreset:hover {
                    background: #eaeaea;
                }
                QListWidget#hubList {
                    background: #ffffff;
                    border: 0;
                    outline: 0;
                }
                QListWidget#hubList::item {
                    border: 0;
                    padding: 2px 6px;
                    color: #222222;
                }
                QListWidget#hubList::item:selected {
                    background: #d9d9d9;
                    color: #111111;
                }
                QListWidget#hubList::item:hover {
                    background: #ececec;
                }
                QLabel#detailsCard {
                    background: #ffffff;
                    border: 0;
                    border-radius: 0;
                    padding: 6px;
                    color: #222222;
                }
                QPushButton {
                    background: #f4f4f4;
                    color: #222222;
                    border: 0;
                    border-radius: 0;
                    padding: 2px 8px;
                    min-height: 22px;
                }
                QPushButton:hover {
                    background: #eaeaea;
                }
                QPushButton:pressed {
                    background: #dddddd;
                }
                """
            )
            return

        self.setStyleSheet(
            """
            QDialog {
                background: #f3f3f3;
            }
            QLabel#recipeHubTitle {
                color: #202020;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#recipeHubSubtitle {
                color: #4a4a4a;
                font-size: 11px;
            }
            QFrame#hubTopBar {
                background: #f3f3f3;
                border: 1px solid #d0d0d0;
                border-radius: 0;
            }
            QLabel#hubStatPill {
                color: #333333;
                background: #f3f3f3;
                border: 1px solid #d0d0d0;
                border-radius: 0;
                padding: 3px 8px;
                font-weight: 500;
            }
            QToolButton#hubChip {
                background: #f8f8f8;
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 0;
                padding: 3px 8px;
            }
            QToolButton#hubChip:checked {
                background: #e6edf7;
                border: 1px solid #8ca9cc;
                color: #1f3f5f;
                font-weight: 700;
            }
            QComboBox#hubSort {
                border: 1px solid #c6c6c6;
                border-radius: 0;
                padding: 4px 8px;
                background: #ffffff;
            }
            QGroupBox#hubPanel {
                border: 1px solid #d0d0d0;
                border-radius: 0;
                margin-top: 6px;
                background: #ffffff;
                font-weight: 600;
                color: #2a2a2a;
            }
            QGroupBox#hubPanel::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 6px;
            }
            QLineEdit#hubSearch {
                border: 1px solid #c6c6c6;
                border-radius: 0;
                padding: 3px 6px;
                background: #ffffff;
                min-width: 200px;
                max-width: 240px;
            }
            QToolButton#hubQuickPreset {
                background: #f8f8f8;
                color: #252525;
                border: 1px solid #c6c6c6;
                border-radius: 0;
                padding: 1px 6px;
                min-height: 22px;
            }
            QLineEdit#hubSearch:focus {
                border: 1px solid #2196f3;
                background: #ffffff;
            }
            QListWidget#hubList {
                border: 1px solid #d0d0d0;
                background: #ffffff;
                padding: 0;
                outline: 0;
            }
            QListWidget#hubList::item {
                background: #ffffff;
                border: 0;
                border-bottom: 1px solid #ececec;
                border-radius: 0;
                padding: 2px 8px;
                color: #252525;
            }
            QListWidget#hubList::item:hover {
                background: #f3f3f3;
            }
            QListWidget#hubList::item:selected {
                background: #e6edf7;
                border: 0;
                color: #1f3f5f;
                font-weight: 700;
            }
            QFrame#recipeCard {
                background: transparent;
                border: 0;
            }
            QLabel#detailsCard {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 0;
                padding: 6px;
                color: #252525;
            }
            QPushButton#hubPrimary {
                background: #f8f8f8;
                color: #252525;
                border: 1px solid #c6c6c6;
                border-radius: 0;
                padding: 1px 6px;
                font-weight: 600;
                min-height: 22px;
            }
            QPushButton#hubPrimary:hover {
                background: #f0f0f0;
            }
            QPushButton#hubSecondary, QPushButton#hubGhost {
                background: #f8f8f8;
                color: #252525;
                border: 1px solid #c6c6c6;
                border-radius: 0;
                padding: 1px 6px;
                min-height: 22px;
            }
            QPushButton#hubSecondary:hover, QPushButton#hubGhost:hover {
                background: #f0f0f0;
                border: 1px solid #b8b8b8;
            }
            """
        )


class QuickClassificationPanel(QWidget):
    """Compact dashboard for common classification tasks."""

    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None, installer=None):
        super(QuickClassificationPanel, self).__init__(parent)
        self._deps = check_dependency_availability()
        self._installer = installer
        self._last_prompt_signature = None  # type: Optional[tuple]
        self._recipes = load_recipes(QSettings())
        self._quick_split_percent = 75
        self._open_report_in_browser = True
        self._previous_recipe_index = 0

        # Connect to global recipe update signal
        _recipe_notifier.recipesUpdated.connect(self.reload_recipes_from_settings)

        self._setup_ui()

        # Setup keyboard shortcuts for power users
        self._setup_shortcuts()

    def _setup_ui(self):
        # type: () -> None
        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.setAlignment(_qt_align_top())

        raster_row = QHBoxLayout()
        raster_row.setSpacing(3)
        raster_row.addWidget(
            self._icon_label(
                "modern/ux_raster.png",
                "Raster to classify",
                fallback_resource=":/plugins/dzetsaka/img/raster.svg",
            )
        )
        self.rasterLineEdit = QLineEdit()
        self.rasterLineEdit.setPlaceholderText("Path to raster file…")
        self.rasterLineEdit.setToolTip(
            "<b>Raster to Classify</b><br>"
            "Multi-band raster image (e.g., satellite imagery) to be classified.<br><br>"
            "<i>Supported formats:</i> GeoTIFF (.tif), IMG, other GDAL-supported formats."
        )
        raster_row.addWidget(self.rasterLineEdit)
        self.rasterBrowse = QPushButton("Browse…")
        self.rasterBrowse.clicked.connect(self._browse_raster)
        raster_row.addWidget(self.rasterBrowse)
        root.addLayout(raster_row)

        self._raster_combo = None
        try:
            from qgis.core import QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._raster_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            if "gdal" in exclude:
                exclude.remove("gdal")
            self._raster_combo.setExcludedProviders(exclude)
            raster_row.addWidget(self._raster_combo)
            self.rasterLineEdit.setVisible(False)
            self.rasterBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass

        vector_row = QHBoxLayout()
        vector_row.setSpacing(3)
        vector_row.addWidget(
            self._icon_label(
                "modern/ux_vector.png",
                "Training vector layer",
                fallback_resource=":/plugins/dzetsaka/img/vector.svg",
            )
        )
        self.vectorLineEdit = QLineEdit()
        self.vectorLineEdit.setPlaceholderText("Path to vector file…")
        self.vectorLineEdit.setToolTip(
            "<b>Training Vector Layer</b><br>"
            "Polygon/point shapefile with labeled samples for training the classifier.<br><br>"
            "<i>Supported formats:</i> Shapefile (.shp), GeoPackage (.gpkg).<br>"
            "<i>Required:</i> Must contain a field with class labels (numeric codes)."
        )
        vector_row.addWidget(self.vectorLineEdit)
        self.vectorBrowse = QPushButton("Browse…")
        self.vectorBrowse.clicked.connect(self._browse_vector)
        vector_row.addWidget(self.vectorBrowse)
        root.addLayout(vector_row)

        self._vector_combo = None
        try:
            from qgis.core import QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._vector_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            if "ogr" in exclude:
                exclude.remove("ogr")
            self._vector_combo.setExcludedProviders(exclude)
            self._vector_combo.currentIndexChanged.connect(self._on_vector_changed)
            if hasattr(self._vector_combo, "layerChanged"):
                self._vector_combo.layerChanged.connect(self._on_vector_changed)
            vector_row.addWidget(self._vector_combo)
            self.vectorLineEdit.setVisible(False)
            self.vectorBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass

        field_row = QHBoxLayout()
        field_row.setSpacing(3)
        field_row.addWidget(
            self._icon_label(
                "modern/ux_label.png",
                "Class label field",
                fallback_resource=":/plugins/dzetsaka/img/column.svg",
            )
        )
        self.classFieldCombo = QComboBox()
        self.classFieldCombo.setToolTip(
            "<b>Class Label Field</b><br>"
            "Vector attribute field containing class codes (numeric values) for each training polygon/point.<br><br>"
            "<i>Example values:</i> 1 (Forest), 2 (Water), 3 (Urban), etc."
        )
        field_row.addWidget(self.classFieldCombo)
        root.addLayout(field_row)

        self.fieldStatusLabel = QLabel("")
        self.fieldStatusLabel.setVisible(False)
        root.addWidget(self.fieldStatusLabel)

        # Prepare Check Data Quality button (placed next to main run action)
        self.checkQualityBtn = QPushButton("Check Data Quality")
        self.checkQualityBtn.setToolTip(
            "<b>Check Training Data Quality</b><br>"
            "Analyze training data for common issues:<br>"
            "• Class imbalance (>10:1 ratio)<br>"
            "• Insufficient samples (<30 per class)<br>"
            "• Invalid geometries<br>"
            "• Spatial clustering"
        )
        # Try to set icon
        quality_icon = QIcon(":/plugins/dzetsaka/img/table.png")
        if quality_icon.isNull():
            quality_icon = QIcon(":/plugins/dzetsaka/img/vector.svg")
        if not quality_icon.isNull():
            self.checkQualityBtn.setIcon(quality_icon)
        self.checkQualityBtn.clicked.connect(self._check_data_quality)
        self.checkQualityBtn.setEnabled(_QUALITY_CHECKER_AVAILABLE)
        if not _QUALITY_CHECKER_AVAILABLE:
            self.checkQualityBtn.setToolTip("Quality checker not available (module import failed)")
        self.checkQualityBtn.setObjectName("quickInlineAction")
        self.checkQualityBtn.setText("Quality…")
        self.checkQualityBtn.setMinimumHeight(24)
        self.checkQualityBtn.setMaximumWidth(110)
        field_row.addWidget(self.checkQualityBtn)

        self.vectorLineEdit.editingFinished.connect(self._on_vector_path_edited)
        self._on_vector_changed()

        classifier_row = QHBoxLayout()
        classifier_row.setSpacing(3)
        classifier_row.addWidget(
            self._icon_label(
                "modern/ux_classifier.png",
                "Recipe (classifier presets)",
                fallback_resource=":/plugins/dzetsaka/img/filter.png",
            )
        )
        self.recipeCombo = QComboBox()
        self.recipeCombo.setEditable(True)
        self.recipeCombo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.recipeCombo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.recipeCombo.setMinimumContentsLength(20)
        self.recipeCombo.setMaximumWidth(360)
        self.recipeCombo.currentIndexChanged.connect(self._apply_selected_recipe)
        if self.recipeCombo.lineEdit() is not None:
            self.recipeCombo.lineEdit().setPlaceholderText("Select or search recipe...")
        self.recipeCombo.setToolTip(
            "<b>Classification Recipe</b><br>"
            "Pre-configured classifier preset including algorithm, parameters, and advanced features.<br><br>"
            "<i>Type to search recipes</i>, then choose one from the dropdown.<br>"
            "<i>Browse Hub</i> opens the full recipe catalog and management tools."
        )
        classifier_row.addWidget(self.recipeCombo, 1)

        self.recipeHubBtn = QToolButton()
        self.recipeHubBtn.setText("Browse Hub")
        self.recipeHubBtn.setToolTip(
            "<b>Recipe Hub</b><br>"
            "Open the modern quad recipe hub with categories, recipe list, actions, and details."
        )
        self.recipeHubBtn.clicked.connect(self._open_recipe_hub)
        classifier_row.addWidget(self.recipeHubBtn)
        root.addLayout(classifier_row)

        # Keep classifier combo as internal state (not shown in default dashboard).
        self.classifierCombo = QComboBox()
        for _code, name, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            self.classifierCombo.addItem(name)

        self.depStatusLabel = QLabel()
        self.depStatusLabel.setVisible(False)
        root.addWidget(self.depStatusLabel)
        self._update_dep_status(self.classifierCombo.currentIndex())

        run_row = QHBoxLayout()
        run_row.setSpacing(3)
        self.reportCheck = QCheckBox("Report")
        self.reportCheck.setChecked(True)
        self.reportCheck.setToolTip("Generate a full classification report bundle (HTML + CSV/JSON + heatmaps).")
        run_row.addWidget(self.reportCheck)
        run_row.addStretch()
        self.runButton = QPushButton("Run Classification")
        self.runButton.setObjectName("quickPrimaryAction")
        self.runButton.setDefault(True)
        self.runButton.setAutoDefault(True)
        self.runButton.setMinimumHeight(30)
        self.runButton.setToolTip(
            "<b>Run Classification</b><br>"
            "Train the selected model with your labeled vector data, then classify the raster."
        )
        run_icon_path = self._icon_asset_path("modern/ux_run.png")
        run_icon = QIcon(run_icon_path)
        if run_icon.isNull():
            run_icon = QIcon(":/plugins/dzetsaka/img/icon.png")
        self.runButton.setIcon(run_icon)
        self.runButton.clicked.connect(self._emit_config)
        run_row.addWidget(self.runButton)
        root.addLayout(run_row)

        # Make primary action visually obvious in quick mode.
        self.setStyleSheet(
            """
            QPushButton#quickPrimaryAction {
                background-color: #2b7cd3;
                color: white;
                border: 1px solid #1f5fa0;
                border-radius: 3px;
                font-weight: 700;
                padding: 4px 14px;
            }
            QPushButton#quickPrimaryAction:hover {
                background-color: #2369b3;
            }
            QPushButton#quickPrimaryAction:pressed {
                background-color: #1b548f;
            }
            QPushButton#quickInlineAction {
                background-color: #f6f6f6;
                color: #2f2f2f;
                border: 1px solid #c8c8c8;
                border-radius: 3px;
                font-weight: 400;
                padding: 2px 8px;
            }
            """
        )

        self.setLayout(root)
        self._refresh_recipe_combo()

    def _icon_asset_path(self, icon_path):
        # type: (str) -> str
        if icon_path.startswith(":/"):
            return icon_path
        plugin_root = self._plugin_root_dir()
        return os.path.normpath(os.path.join(plugin_root, "img", icon_path))

    def _plugin_root_dir(self):
        # type: () -> str
        try:
            import dzetsaka as _dzetsaka_pkg

            return str(Path(_dzetsaka_pkg.__file__).resolve().parent)
        except Exception:
            here = Path(__file__).resolve().parent
            if here.name == "__pycache__":
                here = here.parent
            return str(here.parent)

    def _resource_to_file_path(self, resource_path):
        # type: (str) -> Optional[str]
        prefix = ":/plugins/dzetsaka/"
        if not resource_path.startswith(prefix):
            return None
        rel = resource_path[len(prefix):]
        plugin_root = self._plugin_root_dir()
        return os.path.normpath(os.path.join(plugin_root, rel))

    def _icon_label(self, icon_path, tooltip, fallback_resource=None):
        # type: (str, str, Optional[str]) -> QLabel
        icon_label = QLabel()
        icon_label.setFixedSize(15, 15)
        icon_label.setToolTip(tooltip)
        candidates = []
        if fallback_resource:
            candidates.append(fallback_resource)
            fs_fallback = self._resource_to_file_path(fallback_resource)
            if fs_fallback:
                candidates.append(fs_fallback)
        if icon_path.startswith(":/"):
            candidates.append(icon_path)
            fs_path = self._resource_to_file_path(icon_path)
            if fs_path:
                candidates.append(fs_path)
        else:
            resource_path = f":/plugins/dzetsaka/img/{icon_path}"
            candidates.append(resource_path)
            fs_from_resource = self._resource_to_file_path(resource_path)
            if fs_from_resource:
                candidates.append(fs_from_resource)
            candidates.append(self._icon_asset_path(icon_path))

        pix = QPixmap()
        selected_candidate = None
        for candidate in candidates:
            candidate_pix = QPixmap(candidate)
            if not candidate_pix.isNull():
                pix = candidate_pix
                selected_candidate = candidate
                break
        icon_label.setPixmap(pix)
        icon_label.setScaledContents(True)
        return icon_label

    def _setup_shortcuts(self):
        # type: () -> None
        """Setup keyboard shortcuts for power users."""
        # Shortcut for Check Data Quality: Ctrl+Shift+Q
        if _QUALITY_CHECKER_AVAILABLE and hasattr(self, 'checkQualityBtn'):
            quality_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
            quality_shortcut.activated.connect(self._check_data_quality)
            # Update button tooltip to mention shortcut
            current_tooltip = self.checkQualityBtn.toolTip()
            if "Ctrl+Shift+Q" not in current_tooltip:
                self.checkQualityBtn.setToolTip(
                    current_tooltip + "<br><br><i>Keyboard shortcut: Ctrl+Shift+Q</i>"
                )

        # Shortcut for Run Classification: Ctrl+Return
        if hasattr(self, 'runButton'):
            run_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
            run_shortcut.activated.connect(self._emit_config)
            # Update button tooltip to mention shortcut
            current_tooltip = self.runButton.toolTip()
            if "Ctrl+Return" not in current_tooltip:
                self.runButton.setToolTip(
                    current_tooltip + "<br><br><i>Keyboard shortcut: Ctrl+Return</i>"
                )

    def _create_feature_badge(self, name, description, icon_path, enabled):
        # type: (str, str, str, bool) -> QToolButton
        """Create a checkable badge button for an advanced feature.

        Parameters
        ----------
        name : str
            Feature name (e.g., 'optuna', 'shap', 'smote')
        description : str
            Full description for tooltip
        icon_path : str
            Path to icon image
        enabled : bool
            Whether the feature dependencies are available

        Returns
        -------
        QToolButton
            Configured badge button
        """
        badge = QToolButton()
        badge.setText(name.upper())
        badge.setCheckable(True)
        badge.setChecked(False)
        badge.setMinimumHeight(32)
        badge.setMinimumWidth(80)

        # Try to load icon
        full_icon_path = self._icon_asset_path(icon_path)
        icon = QIcon(full_icon_path)

        # Fallback to a generic icon or disabled state icon
        if icon.isNull():
            fallback_icon_path = self._icon_asset_path("modern/ux_quick.png")
            icon = QIcon(fallback_icon_path)

        if not icon.isNull():
            badge.setIcon(icon)
            # Qt5/Qt6 compatibility for ToolButtonStyle
            if hasattr(Qt, "ToolButtonStyle"):
                badge.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            else:
                badge.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Set tooltip and state
        if enabled:
            tooltip = f"<b>{name.upper()}</b><br>{description}<br><br><i>Click to toggle</i>"
            badge.setStyleSheet("")  # Normal style
        else:
            tooltip = f"<b>{name.upper()}</b><br>{description}<br><br><i>Not available - click to install dependencies</i>"
            badge.setStyleSheet("QToolButton { color: gray; }")

        badge.setToolTip(tooltip)
        # Always keep badge enabled so clicks work even if dependencies missing initially
        badge.setEnabled(True)

        # Connect click handler that checks current dependency status (not cached)
        badge.clicked.connect(lambda checked, n=name: self._on_feature_badge_clicked(n, checked))

        return badge

    def _configure_feature(self, feature_name):
        # type: (str) -> None
        """Open configuration dialog for an advanced feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature ('optuna', 'shap', 'smote')
        """
        # Placeholder for configuration dialogs - to be implemented
        QMessageBox.information(
            self,
            f"{feature_name.upper()} Configuration",
            f"Configuration dialog for {feature_name.upper()} will be implemented here.\n\n"
            f"This will allow you to customize:\n"
            f"- Optuna: trials count, optimization metric, search space\n"
            f"- SHAP: explanation depth, sample size, visualization options\n"
            f"- SMOTE: sampling strategy, k-neighbors, random state"
        )

    def _prompt_install_dependencies(self, features):
        # type: (List[str]) -> None
        """Prompt user to install missing dependencies for advanced features.

        Parameters
        ----------
        features : list of str
            List of feature names requiring installation
        """
        if not self._installer or not hasattr(self._installer, "_try_install_dependencies"):
            QMessageBox.warning(
                self,
                "Installation Not Available",
                "Dependency installer is not available in this context.\n\n"
                "Please use the plugin settings to install dependencies."
            )
            return

        # Map feature names to package names
        feature_map = {
            "optuna": "optuna",
            "shap": "shap",
            "smote": "imbalanced-learn"
        }

        packages = [feature_map.get(f, f) for f in features if f in feature_map]
        package_list = ", ".join(packages)

        reply = QMessageBox.question(
            self,
            "Install Advanced Features",
            f"The following packages are required:<br><br>"
            f"<code>{package_list}</code><br><br>"
            f"Install the full dzetsaka dependency bundle now?<br>"
            f"(includes all ML algorithms and advanced features)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Install full bundle
            if self._installer._try_install_dependencies([]):
                QMessageBox.information(
                    self,
                    "Installation Successful",
                    "Dependencies installed successfully.\n\nPlease restart QGIS to load new libraries.",
                )
                # Refresh dependency status
                self._deps = check_dependency_availability()
            else:
                QMessageBox.warning(
                    self,
                    "Installation Failed",
                    "Failed to install dependencies. Please check the QGIS message log for details."
                )

    def _on_feature_badge_clicked(self, feature_name, checked):
        # type: (str, bool) -> None
        """Handle feature badge clicks with fresh dependency check.

        Parameters
        ----------
        feature_name : str
            Feature name (optuna, shap, smote)
        checked : bool
            Whether the badge was checked or unchecked
        """
        # Check current dependency status (not cached)
        current_deps = check_dependency_availability()

        # Map feature names to dependency keys
        dep_map = {
            "optuna": "optuna",
            "shap": "shap",
            "smote": "imblearn"
        }

        dep_key = dep_map.get(feature_name)
        if not dep_key:
            return

        # If dependency is missing, prompt installation
        if not current_deps.get(dep_key, False):
            # Uncheck the badge since we can't use it
            sender = self.sender()
            if sender:
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)

            # Prompt for installation
            self._prompt_install_dependencies([feature_name])
            return

        # Dependency is available, update state
        self._on_feature_badge_toggled(checked)

    def _on_feature_badge_toggled(self, checked):
        # type: (bool) -> None
        """Update internal recipe extra params when feature badges are toggled.

        Parameters
        ----------
        checked : bool
            Whether the badge was checked or unchecked
        """
        # Ensure _recipe_extra_params exists
        if not hasattr(self, "_recipe_extra_params"):
            self._recipe_extra_params = {}

        # Update the corresponding parameter based on which badge was toggled
        sender = self.sender()
        if sender == self.optunaBadge:
            self._recipe_extra_params["USE_OPTUNA"] = checked
        elif sender == self.shapBadge:
            self._recipe_extra_params["COMPUTE_SHAP"] = checked
        elif sender == self.smoteBadge:
            self._recipe_extra_params["USE_SMOTE"] = checked

    def _browse_raster(self):
        # type: () -> None
        path, _f = QFileDialog.getOpenFileName(self, "Select raster", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.rasterLineEdit.setText(path)
            # Show recipe recommendations based on raster characteristics
            self._show_recipe_recommendations(path)

    def _browse_vector(self):
        # type: () -> None
        path, _f = QFileDialog.getOpenFileName(
            self, "Select vector", "", "Shapefile (*.shp);;GeoPackage (*.gpkg);;All (*)"
        )
        if path:
            self.vectorLineEdit.setText(path)
            self._populate_fields_from_path(path)

    def _check_data_quality(self):
        # type: () -> None
        """Open the training data quality checker dialog."""
        if not _QUALITY_CHECKER_AVAILABLE:
            QMessageBox.critical(
                self,
                "Feature Unavailable",
                "Training data quality checker is not available. Please check that all dependencies are installed."
            )
            return

        vector_path = self._get_vector_path()
        class_field = self.classFieldCombo.currentText().strip()

        if not vector_path:
            QMessageBox.information(
                self,
                "Check Data Quality",
                "Please select a training vector layer first."
            )
            return

        if not class_field:
            QMessageBox.information(
                self,
                "Check Data Quality",
                "Please select a class field first."
            )
            return

        # Show status bar message
        try:
            from src.dzetsaka.infrastructure.ui.status_bar_feedback import show_quality_check_started, show_quality_check_completed
            parent_widget = self.parent()
            while parent_widget is not None:
                if hasattr(parent_widget, 'iface'):
                    show_quality_check_started(parent_widget.iface)
                    break
                parent_widget = parent_widget.parent()
        except Exception:
            pass

        # Resolve checker class from module at call time to avoid stale in-session imports.
        checker_cls = TrainingDataQualityChecker
        try:
            import importlib
            from . import training_data_quality_checker as _tdq_mod

            _tdq_mod = importlib.reload(_tdq_mod)
            checker_cls = getattr(_tdq_mod, "TrainingDataQualityChecker", TrainingDataQualityChecker)
        except Exception:
            pass

        # Open the quality checker dialog
        dialog = checker_cls(
            vector_path=vector_path,
            class_field=class_field,
            parent=self
        )
        try:
            result = dialog.exec_()
        except AttributeError:
            result = dialog.exec()

        # Show completion message
        try:
            if hasattr(dialog, 'issues'):
                issue_count = len([i for i in dialog.issues if i.severity in ["critical", "error", "warning"]])
                parent_widget = self.parent()
                while parent_widget is not None:
                    if hasattr(parent_widget, 'iface'):
                        show_quality_check_completed(parent_widget.iface, issue_count)
                        break
                    parent_widget = parent_widget.parent()
        except Exception:
            pass

    def _on_vector_changed(self):
        # type: () -> None
        self.classFieldCombo.clear()
        self.fieldStatusLabel.setText("")
        self.fieldStatusLabel.setVisible(False)
        if self._vector_combo is None:
            return
        layer = self._vector_combo.currentLayer()
        if layer is None:
            self.fieldStatusLabel.setText("Select a vector layer to list its fields.")
            self.fieldStatusLabel.setVisible(True)
            return
        try:
            fields = layer.dataProvider().fields()
            names = [fields.at(i).name() for i in range(fields.count())]
            if names:
                self.classFieldCombo.addItems(names)
                self.fieldStatusLabel.setText("")
                self.fieldStatusLabel.setVisible(False)
            else:
                self.fieldStatusLabel.setText("No fields found in selected layer.")
                self.fieldStatusLabel.setVisible(True)
        except (AttributeError, TypeError):
            self.fieldStatusLabel.setText("Unable to read fields from selected layer.")
            self.fieldStatusLabel.setVisible(True)

    def _on_vector_path_edited(self):
        # type: () -> None
        path = self.vectorLineEdit.text().strip()
        if not path:
            self.classFieldCombo.clear()
            self.fieldStatusLabel.setText("")
            self.fieldStatusLabel.setVisible(False)
            return
        self._populate_fields_from_path(path)

    def _populate_fields_from_path(self, path):
        # type: (str) -> None
        self.classFieldCombo.clear()
        self.fieldStatusLabel.setText("")
        self.fieldStatusLabel.setVisible(False)
        if not os.path.exists(path):
            self.fieldStatusLabel.setText("Vector path does not exist.")
            self.fieldStatusLabel.setVisible(True)
            return
        try:
            from osgeo import ogr
        except ImportError:
            try:
                import ogr  # type: ignore[no-redef]
            except ImportError:
                self.fieldStatusLabel.setText("OGR unavailable; cannot list fields.")
                self.fieldStatusLabel.setVisible(True)
                return
        ds = ogr.Open(path)
        if ds is None:
            self.fieldStatusLabel.setText("Unable to open vector dataset.")
            self.fieldStatusLabel.setVisible(True)
            return
        layer = ds.GetLayer()
        if layer is None:
            self.fieldStatusLabel.setText("No layer found in dataset.")
            self.fieldStatusLabel.setVisible(True)
            return
        dfn = layer.GetLayerDefn()
        count = dfn.GetFieldCount()
        if count == 0:
            self.fieldStatusLabel.setText("No fields found in vector dataset.")
            self.fieldStatusLabel.setVisible(True)
            return
        for i in range(count):
            self.classFieldCombo.addItem(dfn.GetFieldDefn(i).GetName())

    def _get_raster_path(self):
        # type: () -> str
        if self._raster_combo is not None:
            layer = self._raster_combo.currentLayer()
            if layer is not None:
                return layer.dataProvider().dataSourceUri()
        return self.rasterLineEdit.text().strip()

    def _get_vector_path(self):
        # type: () -> str
        if self._vector_combo is not None:
            layer = self._vector_combo.currentLayer()
            if layer is not None:
                return layer.dataProvider().dataSourceUri().split("|")[0]
        return self.vectorLineEdit.text().strip()

    def _get_classifier_code(self):
        # type: () -> str
        return _CLASSIFIER_META[self.classifierCombo.currentIndex()][0]

    def _get_classifier_name(self):
        # type: () -> str
        return _CLASSIFIER_META[self.classifierCombo.currentIndex()][1]

    def _get_missing_dependencies(self, index=None):
        # type: (Optional[int]) -> tuple[List[str], List[str]]
        if index is None:
            index = self.classifierCombo.currentIndex()

        missing_required = []  # type: List[str]
        _code, _name, needs_sk, needs_xgb, needs_lgb, needs_cb = _CLASSIFIER_META[index]
        if needs_sk and not self._deps.get("sklearn", False):
            missing_required.append("scikit-learn")
        if needs_xgb and not self._deps.get("xgboost", False):
            missing_required.append("xgboost")
        if needs_lgb and not self._deps.get("lightgbm", False):
            missing_required.append("lightgbm")
        if needs_cb and not self._deps.get("catboost", False):
            missing_required.append("catboost")

        missing_optional = []
        if not self._deps.get("optuna", False):
            missing_optional.append("optuna")
        if not self._deps.get("shap", False):
            missing_optional.append("shap")
        if not self._deps.get("imblearn", False):
            missing_optional.append("imblearn (SMOTE)")
        return missing_required, missing_optional

    def _update_dep_status(self, index):
        # type: (int) -> None
        code = _CLASSIFIER_META[index][0]
        missing_required, missing_optional = self._get_missing_dependencies(index)
        if hasattr(self, "depStatusLabel"):
            self.depStatusLabel.clear()
        self._maybe_prompt_install(missing_required, missing_optional)

    def _maybe_prompt_install(self, missing_required, missing_optional):
        # type: (List[str], List[str]) -> None
        if not missing_required:
            return

        signature = (self._get_classifier_code(), tuple(missing_required), tuple(missing_optional))
        if signature == self._last_prompt_signature:
            return
        self._last_prompt_signature = signature

        classifier_name = self._get_classifier_name()

        if self._installer and hasattr(self._installer, "_try_install_dependencies"):
            req_list = ", ".join(missing_required)
            optional_line = ""
            if missing_optional:
                optional_line = f"Optional missing now: <code>{', '.join(missing_optional)}</code><br>"
            reply = QMessageBox.question(
                self,
                "Dependencies Missing for dzetsaka",
                (
                    "The selected classifier needs missing runtime dependencies.<br><br>"
                    f"Required missing now: <code>{req_list}</code><br>"
                    f"{optional_line}<br>"
                    f"Full bundle to install: <code>{_full_bundle_label()}</code><br><br>"
                    "Install the full dzetsaka dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                to_install = _full_dependency_bundle()
                if self._installer._try_install_dependencies(to_install):
                    QMessageBox.information(
                        self,
                        "Installation Successful",
                        "Dependencies installed successfully.\n\nPlease restart QGIS to load new libraries.",
                    )
                    self._deps = check_dependency_availability()
                    self._update_dep_status(self.classifierCombo.currentIndex())
                    self._update_summary()
                else:
                    self.classifierCombo.setCurrentIndex(0)
            else:
                self.classifierCombo.setCurrentIndex(0)
            return

        QMessageBox.warning(
            self,
            "Dependencies Missing",
            (
                f"Selected classifier {classifier_name} cannot run right now.\n\n"
                f"Missing runtime dependencies: {', '.join(missing_required)}\n\n"
                "Install dependencies from the dashboard installer and restart QGIS."
            ),
        )
        self.classifierCombo.setCurrentIndex(0)

    def _build_recipe_tooltip(self, recipe):
        # type: (Dict[str, object]) -> str
        """Build a detailed tooltip for a recipe."""

        def _accuracy_label(value):
            # type: (object) -> str
            if value is None:
                return ""
            text = str(value).strip().lower()
            if not text:
                return ""
            if "highest" in text:
                return "Highest"
            if "high" in text:
                return "High"
            if "medium" in text:
                return "Medium"
            if "low" in text or "baseline" in text:
                return "Baseline"
            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
            if nums:
                avg = sum(nums) / len(nums)
                if avg >= 92:
                    return "Highest"
                if avg >= 85:
                    return "High"
                if avg >= 75:
                    return "Medium"
                return "Baseline"
            return ""

        name = recipe.get("name", "Unnamed Recipe")
        is_template = is_builtin_recipe(recipe)
        metadata = recipe.get("metadata", {})
        category = metadata.get("category", "Custom") if isinstance(metadata, dict) else "Custom"
        description = recipe.get("description", "No description available")

        # Get classifier info
        classifier = recipe.get("classifier", {})
        classifier_code = str(classifier.get("code", "GMM"))
        classifier_name = classifier_code
        for code, name_str, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            if code == classifier_code:
                classifier_name = name_str
                break

        # Build feature list
        extra = recipe.get("extraParam", {})
        features = []
        if extra.get("USE_OPTUNA"):
            features.append("Optuna")
        if extra.get("COMPUTE_SHAP"):
            features.append("SHAP")
        if extra.get("USE_SMOTE"):
            features.append("SMOTE")
        if extra.get("USE_CLASS_WEIGHTS"):
            features.append("Class Weights")
        if extra.get("USE_NESTED_CV") or recipe.get("validation", {}).get("nested_cv"):
            features.append("Nested CV")

        # Build tooltip
        icon = "📦" if is_template else "⚙️"
        tooltip_parts = [f"{icon} {name}"]

        if is_template:
            tooltip_parts.append("")
            tooltip_parts.append(f"Category: {category}")

        tooltip_parts.append(f"Algorithm: {classifier_name}")

        if features:
            tooltip_parts.append(f"Features: {', '.join(features)}")

        if is_template:
            accuracy_range = metadata.get("expected_accuracy_range", "") if isinstance(metadata, dict) else ""
            accuracy_class = (
                recipe.get("expected_accuracy_class", "")
                or (metadata.get("expected_accuracy_class", "") if isinstance(metadata, dict) else "")
            )
            accuracy_profile = _accuracy_label(accuracy_class) or _accuracy_label(accuracy_range)
            use_cases = metadata.get("use_cases", []) if isinstance(metadata, dict) else []
            best_for = ", ".join(use_cases) if use_cases else ""

            if accuracy_profile:
                tooltip_parts.append(f"Accuracy profile: {accuracy_profile}")
            if best_for:
                tooltip_parts.append("")
                tooltip_parts.append(f"Example contexts: {best_for}")
                tooltip_parts.append("Use-case tags are guidance only, not strict algorithm constraints.")
            if classifier_code == "CB":
                tooltip_parts.append("")
                tooltip_parts.append(
                    "Note: CatBoost often outperforms Random Forest on tabular remote-sensing tasks "
                    "while staying compute-efficient on CPU."
                )

        # Add required dependencies
        deps_required = []
        for code, _name, needs_sk, needs_xgb, needs_lgb, needs_cb in _CLASSIFIER_META:
            if code == classifier_code:
                if needs_sk:
                    deps_required.append("scikit-learn")
                if needs_xgb:
                    deps_required.append("XGBoost")
                if needs_lgb:
                    deps_required.append("LightGBM")
                if needs_cb:
                    deps_required.append("CatBoost")
                break
        if extra.get("USE_OPTUNA"):
            deps_required.append("Optuna")
        if extra.get("COMPUTE_SHAP"):
            deps_required.append("SHAP")
        if extra.get("USE_SMOTE"):
            deps_required.append("imbalanced-learn")

        if deps_required:
            tooltip_parts.append("")
            tooltip_parts.append(f"Requires: {', '.join(set(deps_required))}")

        return "\n".join(tooltip_parts)

    def _quick_extra_params(self):
        # type: () -> Dict[str, object]
        report_enabled = bool(self.reportCheck.isChecked()) if hasattr(self, "reportCheck") else True
        # Start with defaults
        defaults = {
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
            "GENERATE_REPORT_BUNDLE": report_enabled,
            "REPORT_OUTPUT_DIR": "",
            "REPORT_LABEL_COLUMN": "",
            "REPORT_LABEL_MAP": "",
            # In Quick mode, "Report" implies opening the generated HTML report.
            "OPEN_REPORT_IN_BROWSER": report_enabled,
        }
        # Merge recipe's extraParam if a recipe was applied
        if hasattr(self, "_recipe_extra_params") and self._recipe_extra_params:
            defaults.update(self._recipe_extra_params)
            # Override report settings based on current UI state
            defaults["GENERATE_REPORT_BUNDLE"] = report_enabled
            defaults["OPEN_REPORT_IN_BROWSER"] = report_enabled

        return defaults

    def _apply_recipe_by_name(self, recipe_name):
        # type: (str) -> None
        target = str(recipe_name or "").strip()
        if not target:
            return
        for i in range(self.recipeCombo.count()):
            if str(self.recipeCombo.itemData(i) or "").strip() == target:
                self.recipeCombo.setCurrentIndex(i)
                return

    def _open_recipe_hub(self):
        # type: () -> None
        current_name = str(self.recipeCombo.currentData() or "").strip()
        dialog = RecipeHubDialog(self, recipes=list(self._recipes), current_recipe_name=current_name)
        try:
            accepted = dialog.exec_()
        except AttributeError:
            accepted = dialog.exec()
        if accepted != 1:
            return

        command = dialog.command()
        if command == "apply":
            self._apply_recipe_by_name(dialog.selected_recipe_name())
        elif command == "create":
            self._open_recipe_shop()
        elif command == "import":
            self._import_recipe()
        elif command == "export":
            self._export_recipe()

    def _refresh_recipe_combo(self, preferred_name=""):
        # type: (str) -> None
        current_name = preferred_name.strip() if preferred_name else ""
        if not current_name:
            current_name = self.recipeCombo.currentText().strip() if self.recipeCombo.count() else ""
        # Strip emoji prefix if present
        for prefix in ["📦 ", "⚙️ "]:
            if current_name.startswith(prefix):
                current_name = current_name[len(prefix):]
                break

        self.recipeCombo.blockSignals(True)
        self.recipeCombo.clear()

        for recipe in self._recipes:
            name = str(recipe.get("name", "Unnamed Recipe"))
            self.recipeCombo.addItem(name, name)
            tooltip = self._build_recipe_tooltip(recipe)
            self.recipeCombo.setItemData(self.recipeCombo.count() - 1, tooltip, Qt.ToolTipRole)

        # Enable in-field fuzzy search over recipe names.
        try:
            completer = QCompleter(self.recipeCombo.model(), self.recipeCombo)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.recipeCombo.setCompleter(completer)
        except Exception:
            pass

        restore_index = 0
        if current_name:
            # Search for matching recipe by data (original name without emoji)
            for i in range(self.recipeCombo.count()):
                if self.recipeCombo.itemData(i) == current_name:
                    restore_index = i
                    break

        self.recipeCombo.setCurrentIndex(restore_index)
        self._previous_recipe_index = restore_index
        self.recipeCombo.blockSignals(False)
        self._apply_selected_recipe(restore_index)

    def reload_recipes_from_settings(self, preferred_name=""):
        # type: (str) -> None
        """Reload recipes from QSettings (e.g., after external updates) and refresh combo."""
        self._recipes = load_recipes(QSettings())
        self._refresh_recipe_combo(preferred_name=preferred_name)

    def _build_recipe_seed_from_quick_state(self):
        # type: () -> Dict[str, object]
        return normalize_recipe(
            {
                "name": "",
                "description": "",
                "classifier": {"code": self._get_classifier_code()},
                "preprocessing": {},
                "features": {"bands": "all"},
                "postprocess": {
                    "confidence_map": False,
                    "save_model": False,
                    "confusion_matrix": bool(hasattr(self, "reportCheck") and self.reportCheck.isChecked()),
                },
                "validation": {
                    "split_percent": int(self._quick_split_percent),
                    "nested_cv": False,
                    "nested_inner_cv": 3,
                    "nested_outer_cv": 5,
                    "cv_mode": "POLYGON_GROUP",
                },
                "extraParam": self._quick_extra_params(),
            }
        )

    def _recipe_context_summary(self):
        # type: () -> str
        raster = self._get_raster_path() or "<not selected>"
        vector = self._get_vector_path() or "<not selected>"
        class_field = self.classFieldCombo.currentText().strip() or "<not selected>"
        return f"Using current dashboard data: raster={raster} | vector={vector} | class field={class_field}"

    def _open_recipe_shop(self):
        # type: () -> None
        dialog = RecipeShopDialog(
            self,
            deps=self._deps,
            installer=self._installer,
            seed_recipe=self._build_recipe_seed_from_quick_state(),
            context_text=self._recipe_context_summary(),
        )
        try:
            accepted = dialog.exec_()
        except AttributeError:
            accepted = dialog.exec()
        if accepted != 1:
            return

        recipe = dialog.selected_recipe()
        name = str(recipe.get("name", "")).strip()
        if not name:
            return

        # Check if trying to overwrite a template
        existing_recipe = next((r for r in self._recipes if str(r.get("name", "")).strip() == name), None)
        if existing_recipe and is_builtin_recipe(existing_recipe):
            QMessageBox.warning(
                self,
                "Cannot Overwrite Template",
                f"'{name}' is a built-in template and cannot be overwritten.\n\n"
                "Please choose a different name for your custom recipe.",
            )
            return

        already_exists = any(str(r.get("name", "")).strip() == name for r in self._recipes)
        if already_exists:
            overwrite = QMessageBox.question(
                self,
                "Overwrite recipe?",
                f"A recipe named '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                return

        # Ensure it's marked as user recipe (remove template metadata)
        if "metadata" in recipe and isinstance(recipe["metadata"], dict):
            recipe["metadata"]["is_template"] = False
        updated = [r for r in self._recipes if str(r.get("name", "")).strip() != name]
        updated.append(recipe)
        self._recipes = updated
        save_recipes(QSettings(), self._recipes)
        # Notify other components
        notify_recipes_updated()
        self._refresh_recipe_combo(preferred_name=name)
        if dialog.run_requested():
            self._emit_config()

    def _apply_selected_recipe(self, _index=None):
        # type: (Optional[int]) -> None
        if not self._recipes or self.recipeCombo.count() == 0:
            return
        current_value = self.recipeCombo.currentData()
        name = str(current_value or "").strip()
        if not name:
            return
        selected = None
        for recipe in self._recipes:
            if str(recipe.get("name", "")).strip() == name:
                selected = normalize_recipe(dict(recipe))
                break
        if selected is None:
            return
        self._previous_recipe_index = self.recipeCombo.currentIndex()
        classifier = selected.get("classifier", {})
        classifier_code = str(classifier.get("code", "GMM"))
        for i, (code, _n, _sk, _xgb, _lgb, _cb) in enumerate(_CLASSIFIER_META):
            if code == classifier_code:
                self.classifierCombo.setCurrentIndex(i)
                break
        self._update_dep_status(self.classifierCombo.currentIndex())

        extra = selected.get("extraParam", {})
        validation = selected.get("validation", {})
        post = selected.get("postprocess", {})

        # Store recipe's extraParam so it's used when emitting config
        self._recipe_extra_params = dict(extra) if extra else {}

        # Update feature badges based on recipe settings
        if hasattr(self, "optunaBadge"):
            use_optuna = bool(extra.get("USE_OPTUNA", False))
            self.optunaBadge.setChecked(use_optuna)

        if hasattr(self, "shapBadge"):
            use_shap = bool(extra.get("COMPUTE_SHAP", False))
            self.shapBadge.setChecked(use_shap)

        if hasattr(self, "smoteBadge"):
            use_smote = bool(extra.get("USE_SMOTE", False))
            self.smoteBadge.setChecked(use_smote)

        report_enabled = bool(extra.get("GENERATE_REPORT_BUNDLE", False) or post.get("confusion_matrix", False))
        if hasattr(self, "reportCheck"):
            self.reportCheck.setChecked(report_enabled)
        self._open_report_in_browser = bool(extra.get("OPEN_REPORT_IN_BROWSER", True))

        try:
            split_percent = int(validation.get("split_percent", 75))
        except (TypeError, ValueError):
            split_percent = 75
        self._quick_split_percent = max(10, min(100, split_percent))

    def _export_recipe(self):
        # type: () -> None
        """Export current recipe to a .dzrecipe or .json file."""
        # Get current recipe from combo or build from current state
        recipe = None
        current_value = self.recipeCombo.currentData()
        name = str(current_value or "").strip()
        for r in self._recipes:
            if str(r.get("name", "")).strip() == name:
                recipe = normalize_recipe(dict(r))
                break

        # If no recipe selected or "Add custom" is selected, build from current UI state
        if recipe is None:
            recipe = self._build_recipe_seed_from_quick_state()
            recipe["name"] = "Custom Recipe"

        # Ask for export location
        default_name = str(recipe.get("name", "recipe")).replace(" ", "_") + ".dzrecipe"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Recipe",
            default_name,
            "dzetsaka Recipe (*.dzrecipe);;JSON Files (*.json)"
        )
        if not path:
            return

        # Write recipe to file
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(recipe, f, indent=2, ensure_ascii=False)

            QMessageBox.information(
                self,
                "Recipe Exported",
                f"Recipe exported successfully to:<br><br><code>{path}</code><br><br>"
                "You can share this file with colleagues or import it later."
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export Failed",
                f"Failed to export recipe:<br><br>{exc!s}"
            )

    def _import_recipe(self):
        # type: () -> None
        """Import recipe from a .dzrecipe or .json file."""
        # Ask for file to import
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import Recipe",
            "",
            "dzetsaka Recipe (*.dzrecipe);;JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        # Read and parse JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                recipe = json.load(f)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Import Failed",
                f"Failed to read recipe file:<br><br>{exc!s}"
            )
            return

        # Handle wrapped format (recipes array)
        if isinstance(recipe, dict) and "recipes" in recipe and isinstance(recipe.get("recipes"), list):
            recipes_list = recipe.get("recipes") or []
            if not recipes_list:
                QMessageBox.warning(self, "Import Failed", "No recipes found in file.")
                return
            recipe = recipes_list[0]
        elif isinstance(recipe, list):
            if not recipe:
                QMessageBox.warning(self, "Import Failed", "Recipe list is empty.")
                return
            recipe = recipe[0]

        if not isinstance(recipe, dict):
            QMessageBox.warning(
                self,
                "Import Failed",
                "Recipe file must contain a JSON object or recipe list."
            )
            return

        # Validate schema - check required fields
        if "name" not in recipe:
            recipe["name"] = "Imported Recipe"
        if "classifier" not in recipe or not isinstance(recipe.get("classifier"), dict):
            QMessageBox.warning(
                self,
                "Invalid Recipe",
                "Recipe is missing required 'classifier' field."
            )
            return
        if "extraParam" not in recipe:
            recipe["extraParam"] = {}

        # Normalize recipe (adds missing fields, upgrades to v2 if needed)
        recipe = normalize_recipe(recipe)

        # Check dependencies
        is_valid, missing = validate_recipe_dependencies(recipe)
        if not is_valid and missing:
            reply = QMessageBox.question(
                self,
                "Dependencies Missing",
                (
                    f"This recipe requires missing dependencies:<br><br>"
                    f"<code>{', '.join(missing)}</code><br><br>"
                    f"Full bundle to install: <code>{_full_bundle_label()}</code><br><br>"
                    "Would you like to install the full dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self._installer and hasattr(self._installer, "_try_install_dependencies"):
                    to_install = _full_dependency_bundle()
                    if self._installer._try_install_dependencies(to_install):
                        QMessageBox.information(
                            self,
                            "Installation Successful",
                            "Dependencies installed successfully.<br><br>"
                            "<b>Important:</b> Please restart QGIS to load new libraries."
                        )
                        self._deps = check_dependency_availability()
                        self._update_dep_status(self.classifierCombo.currentIndex())
                    else:
                        QMessageBox.warning(
                            self,
                            "Installation Failed",
                            "Failed to install dependencies. You can still import the recipe,<br>"
                            "but some features may not work until dependencies are installed."
                        )
                else:
                    QMessageBox.warning(
                        self,
                        "Cannot Install",
                        "Dependency installer not available. Install manually with:<br><br>"
                        "<code>python -m pip install dzetsaka[full]</code><br><br>"
                        "Then restart QGIS."
                    )

        # Check if recipe with same name exists
        name = str(recipe.get("name", "Imported Recipe")).strip()
        already_exists = any(str(r.get("name", "")).strip() == name for r in self._recipes)
        if already_exists:
            overwrite = QMessageBox.question(
                self,
                "Recipe Exists",
                f"A recipe named '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                # Ask for new name
                new_name, ok = QInputDialog.getText(
                    self,
                    "Rename Recipe",
                    "Enter a new name for the imported recipe:",
                    text=f"{name} (imported)"
                )
                if not ok or not new_name.strip():
                    return
                name = new_name.strip()
                recipe["name"] = name

        # Add to recipe list
        updated = [r for r in self._recipes if str(r.get("name", "")).strip() != name]
        updated.append(recipe)
        self._recipes = updated
        save_recipes(QSettings(), self._recipes)
        # Notify other components
        notify_recipes_updated()
        self._refresh_recipe_combo(preferred_name=name)

        QMessageBox.information(
            self,
            "Recipe Imported",
            f"Recipe '{name}' imported successfully!<br><br>"
            "The recipe is now available in the recipe dropdown."
        )

    def _emit_config(self):
        # type: () -> None
        try:
            raster = self._get_raster_path()
            if not raster:
                QMessageBox.warning(self, "Missing Input", "Please select an input raster.")
                return

            if bool(getattr(self, "reportCheck", None) and self.reportCheck.isChecked()):
                deps = check_dependency_availability()
                missing_bundle = []
                for dep_key in ("sklearn", "xgboost", "lightgbm", "catboost", "optuna", "shap", "imblearn"):
                    # seaborn is used by report heatmaps.
                    if dep_key == "imblearn":
                        if not deps.get(dep_key, False):
                            missing_bundle.append(dep_key)
                        continue
                    if not deps.get(dep_key, False):
                        missing_bundle.append(dep_key)
                if not deps.get("seaborn", False):
                    missing_bundle.append("seaborn")

                if missing_bundle:
                    if self._installer and hasattr(self._installer, "_try_install_dependencies"):
                        reply = QMessageBox.question(
                            self,
                            "Report Dependencies Missing",
                            (
                                "Report mode requires the full dzetsaka dependency bundle.<br><br>"
                                f"Missing now: <code>{', '.join(missing_bundle)}</code><br>"
                                f"Bundle to install: <code>{_full_bundle_label()}</code><br><br>"
                                "Install now?<br><br>"
                                "<i>Choose No to continue without report mode.</i>"
                            ),
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            if not self._installer._try_install_dependencies(_full_dependency_bundle()):
                                QMessageBox.warning(
                                    self,
                                    "Dependencies Missing",
                                    "Could not install full dependency bundle required for report mode.",
                                )
                                return
                            self._deps = check_dependency_availability()
                        else:
                            # Allow baseline execution by disabling report mode for this run.
                            self.reportCheck.setChecked(False)
                    else:
                        reply = QMessageBox.question(
                            self,
                            "Dependencies Missing",
                            (
                                "Report mode requires the full dependency bundle, but installer is unavailable.\n\n"
                                f"Required bundle: {_full_bundle_label()}\n\n"
                                "Continue without report mode?"
                            ),
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes,
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.reportCheck.setChecked(False)
                        else:
                            return

            missing_required, _missing_optional = self._get_missing_dependencies()
            if missing_required:
                QMessageBox.warning(
                    self,
                    "Dependencies Missing",
                    (
                        f"Selected classifier {self._get_classifier_name()} cannot run right now.\n\n"
                        f"Missing runtime dependencies: {', '.join(missing_required)}\n\n"
                        "Install dependencies from the dashboard installer and restart QGIS."
                    ),
                )
                return

            vector = self._get_vector_path()
            class_field = self.classFieldCombo.currentText().strip()

            if not vector or not class_field:
                QMessageBox.warning(
                    self,
                    "Missing Training Data",
                    "Express mode requires training data and a label field.",
                )
                return

            config = {
                "raster": raster,
                "vector": vector,
                "class_field": class_field,
                "load_model": "",
                "classifier": self._get_classifier_code(),
                "extraParam": self._quick_extra_params(),
                "output_raster": "",
                "confidence_map": "",
                "save_model": "",
                "confusion_matrix": "",
                "split_percent": self._quick_split_percent,
            }
            self.classificationRequested.emit(config)
        except Exception as exc:
            _show_issue_popup(
                owner=self,
                installer=self._installer,
                title="Quick Run Failed",
                error_type="Runtime Error",
                error_message=f"Unexpected error while preparing Quick Run: {exc!s}",
                context="classification_workflow_ui._emit_config",
            )

    def _show_recipe_recommendations(self, raster_path):
        # type: (str) -> None
        """Show recipe recommendations based on raster characteristics.

        Parameters
        ----------
        raster_path : str
            Path to the raster file to analyze

        """
        # Check if recommender is available and enabled
        if not _RECOMMENDER_AVAILABLE:
            return

        settings = QSettings()
        if not settings.value("/dzetsaka/show_recommendations", True, bool):
            return

        # Check if we have any recipes
        if not self._recipes:
            return

        try:
            # Analyze the raster
            analyzer = RasterAnalyzer()
            raster_info = analyzer.analyze_raster(raster_path)

            # Check for errors
            if raster_info.get("error"):
                # Silently fail - don't interrupt user workflow
                return

            # Get recommendations
            recommender = RecipeRecommender()
            recommendations = recommender.recommend(raster_info, self._recipes)

            # Only show dialog if we have good recommendations
            if not recommendations or recommendations[0][1] < 40:
                # No good recommendations, skip
                return

            # Show recommendation dialog
            dialog = RecommendationDialog(recommendations, raster_info, self)
            dialog.recipeSelected.connect(self._apply_recommended_recipe)
            dialog.exec_()

        except Exception:
            # Silently fail - recommendations are a nice-to-have feature
            pass

    def _apply_recommended_recipe(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply a recommended recipe to the current configuration.

        Parameters
        ----------
        recipe : Dict[str, object]
            Recipe dictionary to apply

        """
        try:
            # Find the recipe in the combo box
            recipe_name = recipe.get("name", "")
            for i in range(self.recipeCombo.count()):
                if self.recipeCombo.itemData(i) == recipe_name:
                    self.recipeCombo.setCurrentIndex(i)
                    # This will trigger recipe application through _apply_selected_recipe.
                    break

        except Exception:
            # If something goes wrong, just skip
            pass

class ClassificationDashboardDock(QDockWidget):
    """Dockable dashboard with Quick Run and Advanced Setup modes."""

    closingRequested = pyqtSignal()
    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None, installer=None):
        super(ClassificationDashboardDock, self).__init__(parent)
        self.setWindowTitle("dzetsaka Classifier")
        self.setObjectName("DzetsakaClassificationDashboardDock")
        self.setMinimumWidth(220)
        self.setMinimumHeight(260)
        self.resize(260, 320)
        self._installer = installer

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        banner_container = QWidget()
        banner_container.setMinimumHeight(60)
        banner_container.setMaximumHeight(60)
        banner_stack = QStackedLayout(banner_container)
        if hasattr(QStackedLayout, "StackingMode"):
            banner_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        else:
            banner_stack.setStackingMode(QStackedLayout.StackAll)

        self.banner = _CoverPixmapLabel()
        self.banner.setMinimumHeight(60)
        self.banner.setMaximumHeight(60)
        self.banner.setAlignment(_qt_align_hcenter())
        if hasattr(QSizePolicy, "Policy"):
            self.banner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        banner_pixmap = QPixmap(":/plugins/dzetsaka/img/parcguyane.jpg")
        if not banner_pixmap.isNull():
            self.banner.set_cover_pixmap(banner_pixmap)
            self.banner.setToolTip("dzetsaka")
        banner_stack.addWidget(self.banner)

        layout.addWidget(banner_container)

        self.stack = QStackedWidget()
        self.quickPanel = QuickClassificationPanel(installer=installer)
        self.quickPanel.setToolTip("Express: essential inputs and one-click run.")
        self.quickPanel.classificationRequested.connect(self.classificationRequested)

        self.stack.addWidget(self.quickPanel)
        layout.addWidget(self.stack)

        self.setWidget(container)
        self.stack.setCurrentWidget(self.quickPanel)
        self.setMinimumHeight(260)

    def closeEvent(self, event):
        self.closingRequested.emit()
        event.accept()

