"""Classification Wizard for dzetsaka.

A QWizard-based step-by-step interface for configuring and launching
remote-sensing image classification. The wizard guides the user through
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
import tempfile
import urllib.request
from collections import Counter
from typing import Dict, List, Optional

from qgis.PyQt.QtCore import QSettings, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
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
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

# ---------------------------------------------------------------------------
# Dependency availability helpers (importable without Qt for unit tests)
# ---------------------------------------------------------------------------


def check_dependency_availability():
    # type: () -> Dict[str, bool]
    """Check which optional packages are importable at runtime.

    Returns
    -------
    dict[str, bool]
        Keys: ``sklearn``, ``xgboost``, ``lightgbm``, ``catboost``,
        ``optuna``, ``shap``, ``imblearn``.  Values: True when the package can be
        imported successfully.
    """
    deps = {
        "sklearn": False,
        "xgboost": False,
        "lightgbm": False,
        "catboost": False,
        "optuna": False,
        "shap": False,
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
    if code in classifier_config.SKLEARN_DEPENDENT:
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

        if extra.get("USE_SHAP", False) and not deps.get("shap", False):
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
        from ..logging_utils import show_error_dialog

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
    }  # type: Dict[str, object]


def build_review_summary(config):
    # type: (Dict[str, object]) -> str
    """Produce a human-readable summary of the wizard configuration.

    Parameters
    ----------
    config : dict
        The full config dict that the wizard would emit.

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
    lines.append("")

    # --- Output ---
    lines.append("[Output]")
    lines.append("  Classification map : " + str(config.get("output_raster", "<temp file>")))
    lines.append("  Confidence map : " + str(config.get("confidence_map", "")))
    lines.append("  Save model : " + str(config.get("save_model", "")))
    lines.append("  Confusion matrix : " + str(config.get("confusion_matrix", "")))
    if config.get("confusion_matrix", ""):
        lines.append("    Validation split % : " + str(config.get("split_percent", 50)))
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
    "  - validation: { split_percent: 10-100, nested_cv: bool, nested_inner_cv: 2-10, nested_outer_cv: 2-10 }\n"
    "  - extraParam: (optional) wizard extra parameters\n"
)


def _recipe_template():
    # type: () -> Dict[str, object]
    return {
        "version": _RECIPE_VERSION,
        "name": "Unnamed Recipe",
        "description": "",
        "preprocessing": {},
        "features": {"bands": "all"},
        "classifier": {"code": "GMM"},
        "postprocess": {"confidence_map": False, "save_model": False, "confusion_matrix": False},
        "validation": {
            "split_percent": 100,
            "nested_cv": False,
            "nested_inner_cv": 3,
            "nested_outer_cv": 5,
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
        },
    }  # type: Dict[str, object]


def build_fast_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "Fast (Default)",
            "description": "Minimal dependencies, quick baseline for immediate success.",
        }
    )
    return recipe


def build_catboost_recipe():
    # type: () -> Dict[str, object]
    recipe = _recipe_template()
    recipe.update(
        {
            "name": "CatBoost Quick",
            "description": "Great for heterogeneous, tabular features (handles categorical values well).",
            "classifier": {"code": "CB"},
            "extraParam": dict(_recipe_template()["extraParam"], USE_OPTUNA=False, COMPUTE_SHAP=False),
        }
    )
    return recipe


def normalize_recipe(recipe):
    # type: (Dict[str, object]) -> Dict[str, object]
    """Ensure a recipe contains all required keys."""
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
            errors.append(f"Recipe #{idx+1}: not an object.")
            continue
        name = item.get("name", "")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"Recipe #{idx+1}: missing or empty 'name'.")
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
        extra = item.get("extraParam", {})
        if isinstance(extra, dict) and "CLASS_WEIGHT_STRATEGY" in extra:
            strategy = extra.get("CLASS_WEIGHT_STRATEGY")
            if strategy not in ("balanced", "uniform"):
                errors.append(
                    f"Recipe '{name}': CLASS_WEIGHT_STRATEGY must be 'balanced' or 'uniform'."
                )
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
    lines.append(f"Validation split %: {validation.get('split_percent', 100)}")
    return "\n".join(lines)


def load_recipes(settings):
    # type: (QSettings) -> List[Dict[str, object]]
    """Load recipes from QSettings; seed with defaults when missing."""
    raw = settings.value("/dzetsaka/recipes", "", str)
    recipes = []  # type: List[Dict[str, object]]
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                recipes = data.get("recipes", [])
            elif isinstance(data, list):
                recipes = data
        except (TypeError, ValueError):
            recipes = []
    if not recipes:
        recipes = [build_fast_recipe(), build_catboost_recipe()]
    else:
        recipes = [normalize_recipe(r) for r in recipes if isinstance(r, dict)]
        names = {r.get("name") for r in recipes}
        if "Fast (Default)" not in names:
            recipes.insert(0, build_fast_recipe())
        if "CatBoost Quick" not in names:
            recipes.append(build_catboost_recipe())
    return recipes


def save_recipes(settings, recipes):
    # type: (QSettings, List[Dict[str, object]]) -> None
    """Persist recipes to QSettings."""
    settings.setValue("/dzetsaka/recipes", json.dumps(recipes))


def recipe_from_config(config, name, description=""):
    # type: (Dict[str, object], str, str) -> Dict[str, object]
    """Create a recipe dict from the current wizard config."""
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
            },
            "extraParam": extra,
        }
    )
    return recipe


# ---------------------------------------------------------------------------
# Classifier metadata used by the wizard (mirrors classifier_config)
# ---------------------------------------------------------------------------

# (code, full name, requires_sklearn, requires_xgboost, requires_lightgbm, requires_catboost)
_CLASSIFIER_META = [
    ("GMM", "Gaussian Mixture Model", False, False, False, False),
    ("RF", "Random Forest", True, False, False, False),
    ("SVM", "Support Vector Machine", True, False, False, False),
    ("KNN", "K-Nearest Neighbors", True, False, False, False),
    ("XGB", "XGBoost", False, True, False, False),
    ("LGB", "LightGBM", False, False, True, False),
    ("CB", "CatBoost", False, False, False, True),
    ("ET", "Extra Trees", True, False, False, False),
    ("GBC", "Gradient Boosting Classifier", True, False, False, False),
    ("LR", "Logistic Regression", True, False, False, False),
    ("NB", "Gaussian Naive Bayes", True, False, False, False),
    ("MLP", "Multi-layer Perceptron", True, False, False, False),
]


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


# ---------------------------------------------------------------------------
# Recipe gallery dialog
# ---------------------------------------------------------------------------


class RecipeGalleryDialog(QDialog):
    """Dialog that shows local and remote recipe galleries."""

    recipeApplied = pyqtSignal(dict)
    recipesUpdated = pyqtSignal(list)
    remoteUrlUpdated = pyqtSignal(str)

    def __init__(self, parent=None, recipes=None, remote_url=""):
        super(RecipeGalleryDialog, self).__init__(parent)
        self.setWindowTitle("Recipe Gallery")
        self.setMinimumSize(720, 420)

        self._local_recipes = recipes or []
        self._remote_url = remote_url
        self._remote_recipes = []

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Browse and share segmentation recipes (local + remote)."))

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_local_tab(), "Local")
        self.tabs.addTab(self._build_remote_tab(), "Remote")
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
        self._refresh_remote_summary()

    def _build_local_tab(self):
        # type: () -> QWidget
        widget = QWidget()
        layout = QVBoxLayout()
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
        btn_row.addStretch()
        layout.addLayout(btn_row)

        widget.setLayout(layout)
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

    def _populate_local(self):
        # type: () -> None
        self.localList.clear()
        for recipe in self._local_recipes:
            name = recipe.get("name", "Unnamed Recipe")
            self.localList.addItem(QListWidgetItem(name))

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
            return
        recipe = self._find_recipe(self._local_recipes, current.text())
        self.localSummary.setPlainText(format_recipe_summary(recipe) if recipe else "")

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
        if self.tabs.currentIndex() == 0:
            item = self.localList.currentItem()
            if item is None:
                return
            recipe = self._find_recipe(self._local_recipes, item.text())
        else:
            item = self.remoteList.currentItem()
            if item is None:
                return
            recipe = self._find_recipe(self._remote_recipes, item.text())
        if recipe:
            self.recipeApplied.emit(recipe)
            self.close()

    def _export_selected_local(self):
        # type: () -> None
        item = self.localList.currentItem()
        if item is None:
            return
        recipe = self._find_recipe(self._local_recipes, item.text())
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
            QMessageBox.information(
                self, "Remote gallery", "Set a remote URL to fetch shared recipes."
            )
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
    """Wizard page for selecting input data and classifier."""

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

        # --- Recipe group ---
        recipe_group = QGroupBox("Recipe")
        recipe_layout = QGridLayout()
        recipe_layout.addWidget(QLabel("Recipe:"), 0, 0)
        self.recipeCombo = QComboBox()
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
        input_layout.addWidget(self.classFieldCombo)
        self.fieldStatusLabel = QLabel()
        input_layout.addWidget(self.fieldStatusLabel)
        self.geometryExplorerBtn = QPushButton("Geometry Explorer…")
        self.geometryExplorerBtn.clicked.connect(self._open_geometry_explorer)
        input_layout.addWidget(self.geometryExplorerBtn)

        self.loadModelCheck = QCheckBox("Use existing model")
        input_layout.addWidget(self.loadModelCheck)

        model_row = QHBoxLayout()
        self.modelLineEdit = QLineEdit()
        self.modelLineEdit.setPlaceholderText("Model file path…")
        self.modelLineEdit.setEnabled(False)
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
        algo_layout.addWidget(self.classifierCombo)

        self.depStatusLabel = QLabel()
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

        # Register wizard fields
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
            self.classFieldCombo.addItem(layer.GetLayerDefn().GetFieldDefn(i).GetName())
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
        available = _classifier_available(code, self._deps)
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

        lines = []
        if available:
            lines.append("<span style='color:#166534;'>Required: OK</span>")
        else:
            req = ", ".join(missing_required) if missing_required else "Unknown"
            lines.append(f"<span style='color:#b91c1c;'>Missing: {req}</span>")
        if missing_optional:
            opt = ", ".join(missing_optional)
            lines.append(f"<span style='color:#92400e;'>Optional missing: {opt}</span>")
        else:
            lines.append("<span style='color:#166534;'>Optional: OK</span>")

        self.depStatusLabel.setText("<br>".join(lines))
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

        package_map = {
            "scikit-learn": "scikit-learn",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "optuna": "optuna",
            "shap": "shap",
            "imblearn (SMOTE)": "imbalanced-learn",
        }

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
                    "Install the full dzetsaka dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                to_install = [
                    "scikit-learn",
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "optuna",
                    "shap",
                    "imbalanced-learn",
                ]
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
                            f"Wizard classifier selection: {classifier_name}",
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
        wizard = self.wizard()  # type: Optional[ClassificationWizard]
        if wizard is None or not self._recipes:
            return
        name = self.recipeCombo.currentText()
        for recipe in self._recipes:
            if recipe.get("name") == name:
                wizard.apply_recipe(recipe)
                break

    def _save_current_recipe(self):
        # type: () -> None
        wizard = self.wizard()  # type: Optional[ClassificationWizard]
        if wizard is None:
            return
        wizard.save_current_recipe()

    def _open_recipe_gallery(self):
        # type: () -> None
        wizard = self.wizard()  # type: Optional[ClassificationWizard]
        if wizard is None:
            return
        wizard.open_recipe_gallery()

    def _open_geometry_explorer(self):
        # type: () -> None
        vector_path = self.get_vector_path()
        class_field = self.get_class_field()
        layer = self._vector_combo.currentLayer() if self._vector_combo is not None else None
        if not vector_path and layer is None:
            QMessageBox.information(
                self, "Geometry Explorer", "Select a vector layer or enter a vector path first."
            )
            return
        dialog = VectorInsightDialog(self, vector_path=vector_path, class_field=class_field, layer=layer)
        try:
            dialog.exec_()
        except AttributeError:
            dialog.exec()

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


# ---------------------------------------------------------------------------
# Page 2 — Advanced Options
# ---------------------------------------------------------------------------


class AdvancedOptionsPage(QWizardPage):
    """Wizard page for Optuna, imbalance, explainability and validation."""

    def __init__(self, parent=None, deps=None):
        """Initialise AdvancedOptionsPage."""
        super(AdvancedOptionsPage, self).__init__(parent)
        self.setTitle("Advanced Setup")
        self.setSubTitle("Configure optimization, explainability, and validation.")

        self._deps = deps if deps is not None else check_dependency_availability()

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
        opt_layout.addWidget(self.optunaCheck, 0, 0, 1, 2)

        trials_label = QLabel("Trials:")
        self.optunaTrials = QSpinBox()
        self.optunaTrials.setRange(10, 1000)
        self.optunaTrials.setValue(100)
        self.optunaTrials.setEnabled(False)
        opt_layout.addWidget(trials_label, 1, 0)
        opt_layout.addWidget(self.optunaTrials, 1, 1)

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
        imb_layout.addWidget(self.smoteCheck, 0, 0, 1, 2)

        k_label = QLabel("k_neighbors:")
        self.smoteK = QSpinBox()
        self.smoteK.setRange(1, 20)
        self.smoteK.setValue(5)
        self.smoteK.setEnabled(False)
        imb_layout.addWidget(k_label, 1, 0)
        imb_layout.addWidget(self.smoteK, 1, 1)
        self.smoteCheck.toggled.connect(self.smoteK.setEnabled)

        self.classWeightCheck = QCheckBox("Use class weights")
        self.classWeightCheck.setEnabled(self._deps.get("sklearn", False))
        imb_layout.addWidget(self.classWeightCheck, 2, 0, 1, 2)

        strat_label = QLabel("Strategy:")
        self.weightStrategyCombo = QComboBox()
        self.weightStrategyCombo.addItems(["balanced", "uniform"])
        self.weightStrategyCombo.setEnabled(False)
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
        exp_layout.addWidget(self.shapCheck, 0, 0, 1, 3)

        shap_label = QLabel("Output:")
        self.shapOutput = QLineEdit()
        self.shapOutput.setPlaceholderText("Path to SHAP raster…")
        self.shapOutput.setEnabled(False)
        self.shapBrowse = QPushButton("Browse…")
        self.shapBrowse.setEnabled(False)
        self.shapBrowse.clicked.connect(self._browse_shap_output)
        exp_layout.addWidget(shap_label, 1, 0)
        exp_layout.addWidget(self.shapOutput, 1, 1)
        exp_layout.addWidget(self.shapBrowse, 1, 2)

        shap_sample_label = QLabel("Sample size:")
        self.shapSampleSize = QSpinBox()
        self.shapSampleSize.setRange(100, 50000)
        self.shapSampleSize.setValue(1000)
        self.shapSampleSize.setEnabled(False)
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
        val_layout.addWidget(self.nestedCVCheck, 0, 0, 1, 2)

        inner_label = QLabel("Inner folds:")
        self.innerFolds = QSpinBox()
        self.innerFolds.setRange(2, 10)
        self.innerFolds.setValue(3)
        self.innerFolds.setEnabled(False)
        val_layout.addWidget(inner_label, 1, 0)
        val_layout.addWidget(self.innerFolds, 1, 1)

        outer_label = QLabel("Outer folds:")
        self.outerFolds = QSpinBox()
        self.outerFolds.setRange(2, 10)
        self.outerFolds.setValue(5)
        self.outerFolds.setEnabled(False)
        val_layout.addWidget(outer_label, 2, 0)
        val_layout.addWidget(self.outerFolds, 2, 1)

        def _toggle_nested(checked):
            # type: (bool) -> None
            self.innerFolds.setEnabled(checked)
            self.outerFolds.setEnabled(checked)

        self.nestedCVCheck.toggled.connect(_toggle_nested)
        val_group.setLayout(val_layout)
        layout.addWidget(val_group, 1, 1)

        self.setLayout(layout)

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
        }  # type: Dict[str, object]


# ---------------------------------------------------------------------------
# Page 3 — Output Configuration
# ---------------------------------------------------------------------------


class OutputConfigPage(QWizardPage):
    """Wizard page for specifying output paths and optional outputs."""

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
        out_layout.addWidget(self.confidenceCheck, 1, 0, 1, 3)
        out_layout.addWidget(QLabel("Confidence map:"), 2, 0)
        self.confMapEdit = QLineEdit()
        self.confMapEdit.setPlaceholderText("Path to confidence map…")
        self.confMapEdit.setEnabled(False)
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
        out_layout.addWidget(self.saveModelCheck, 3, 0, 1, 3)
        out_layout.addWidget(QLabel("Model file:"), 4, 0)
        self.saveModelEdit = QLineEdit()
        self.saveModelEdit.setPlaceholderText("Model file path…")
        self.saveModelEdit.setEnabled(False)
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
        self.splitSpinBox = QSpinBox()
        self.splitSpinBox.setRange(10, 90)
        self.splitSpinBox.setValue(50)
        self.splitSpinBox.setEnabled(False)
        out_layout.addWidget(self.splitSpinBox, 7, 1)

        def _toggle_matrix(checked):
            # type: (bool) -> None
            self.matrixEdit.setEnabled(checked)
            self.matrixBrowse.setEnabled(checked)
            self.splitSpinBox.setEnabled(checked)

        self.matrixCheck.toggled.connect(_toggle_matrix)

        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

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

    def _refresh_review(self):
        # type: () -> None
        """Refresh the review summary based on current wizard state."""
        wizard = self.wizard()  # type: Optional[ClassificationWizard]
        if wizard is None:
            return
        config = wizard.collect_config()
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

    def apply_recipe(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply recipe settings to output toggles."""
        recipe = normalize_recipe(dict(recipe))
        post = recipe.get("postprocess", {})
        validation = recipe.get("validation", {})

        self.confidenceCheck.setChecked(bool(post.get("confidence_map", False)))
        self.saveModelCheck.setChecked(bool(post.get("save_model", False)))

        split = int(validation.get("split_percent", 100))
        matrix = bool(post.get("confusion_matrix", False)) or split < 100
        self.matrixCheck.setChecked(matrix)
        self.splitSpinBox.setValue(split)


# ---------------------------------------------------------------------------
# Main Wizard
# ---------------------------------------------------------------------------


class ClassificationWizard(QWizard):
    """Step-by-step classification wizard for dzetsaka.

    Emits ``classificationRequested`` with the assembled config dict
    when the user clicks Finish (labelled "Run Classification").
    """

    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None, installer=None, close_on_accept=True):
        """Initialise ClassificationWizard with all 3 pages."""
        super(ClassificationWizard, self).__init__(parent)
        self.setWindowTitle("dzetsaka Classification")
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

        self.addPage(self.dataPage)      # index 0
        self.addPage(self.advPage)       # index 1
        self.addPage(self.outputPage)    # index 2

        self.dataPage.set_recipe_list(self._recipes)

        # Override the Finish button text
        try:
            finish_button = QWizard.FinishButton
        except AttributeError:
            finish_button = QWizard.WizardButton.FinishButton
        self.setButtonText(finish_button, "Run Classification")

        try:
            wizard_style = QWizard.ModernStyle
        except AttributeError:
            wizard_style = QWizard.WizardStyle.ModernStyle
        self.setWizardStyle(wizard_style)

    # --- page-transition hook ---------------------------------------------

    def validateCurrentPage(self):
        # type: () -> bool
        """Handle smart-defaults propagation when leaving the first page."""
        current = self.currentId()
        # Leaving Input/Algorithm page (index 0) -> entering AdvancedOptionsPage
        if current == 0 and self.dataPage.smart_defaults_requested():
            defaults = build_smart_defaults(self._deps)
            self.advPage.apply_smart_defaults(defaults)
            # Reset flag so it fires only once
            self.dataPage._smart_defaults_applied = False
        return super(ClassificationWizard, self).validateCurrentPage()

    # --- recipe helpers ---------------------------------------------------

    def _update_recipes(self, recipes):
        # type: (List[Dict[str, object]]) -> None
        self._recipes = recipes
        save_recipes(self._settings, self._recipes)
        self.dataPage.set_recipe_list(self._recipes)

    def apply_recipe(self, recipe):
        # type: (Dict[str, object]) -> None
        """Apply a recipe to the wizard UI with dependency validation."""
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
                    "Install the full dzetsaka dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                to_install = [
                    "scikit-learn",
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "optuna",
                    "shap",
                    "imbalanced-learn",
                ]

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
        """Save the current wizard configuration as a recipe."""
        name, ok = QInputDialog.getText(self, "Save Recipe", "Recipe name:")
        if not ok or not name.strip():
            return
        description, _ok = QInputDialog.getText(self, "Save Recipe", "Description (optional):")
        config = self.collect_config()
        recipe = recipe_from_config(config, name.strip(), description.strip())
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
        """Open the recipe gallery (local + remote)."""
        dialog = RecipeGalleryDialog(
            parent=self,
            recipes=list(self._recipes),
            remote_url=self._remote_recipe_url,
        )
        dialog.recipeApplied.connect(self.apply_recipe)
        dialog.recipesUpdated.connect(self._update_recipes)
        dialog.remoteUrlUpdated.connect(self._set_remote_recipe_url)
        try:
            dialog.exec_()
        except AttributeError:
            dialog.exec()

    def _set_remote_recipe_url(self, url):
        # type: (str) -> None
        self._remote_recipe_url = url
        self._settings.setValue("/dzetsaka/recipesRemoteUrl", url)

    # --- config assembly --------------------------------------------------

    def collect_config(self):
        # type: () -> Dict[str, object]
        """Assemble the full config dict from all wizard pages."""
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

        # Output
        config.update(self.outputPage.get_output_config())

        return config

    # --- finish handler ---------------------------------------------------

    def accept(self):
        # type: () -> None
        """Emit the config signal and close the wizard."""
        config = self.collect_config()
        self.classificationRequested.emit(config)
        if self._close_on_accept:
            super(ClassificationWizard, self).accept()


class QuickClassificationPanel(QWidget):
    """Compact dashboard for common classification tasks."""

    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super(QuickClassificationPanel, self).__init__(parent)
        self._deps = check_dependency_availability()
        self._setup_ui()

    def _setup_ui(self):
        # type: () -> None
        root = QVBoxLayout()

        intro = QLabel(
            "Quick run uses the standard workflow with essential options only. "
            "Switch to Advanced setup for tuning and expert controls."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()

        input_layout.addWidget(QLabel("Raster to classify:"))
        raster_row = QHBoxLayout()
        self.rasterLineEdit = QLineEdit()
        self.rasterLineEdit.setPlaceholderText("Path to raster file…")
        raster_row.addWidget(self.rasterLineEdit)
        self.rasterBrowse = QPushButton("Browse…")
        self.rasterBrowse.clicked.connect(self._browse_raster)
        raster_row.addWidget(self.rasterBrowse)
        input_layout.addLayout(raster_row)

        self._raster_combo = None
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
            pass

        self.loadModelCheck = QCheckBox("Use existing model")
        self.loadModelCheck.toggled.connect(self._toggle_model_mode)
        input_layout.addWidget(self.loadModelCheck)

        model_row = QHBoxLayout()
        self.modelLineEdit = QLineEdit()
        self.modelLineEdit.setPlaceholderText("Model file path…")
        self.modelLineEdit.setEnabled(False)
        model_row.addWidget(self.modelLineEdit)
        self.modelBrowse = QPushButton("Browse…")
        self.modelBrowse.setEnabled(False)
        self.modelBrowse.clicked.connect(self._browse_model)
        model_row.addWidget(self.modelBrowse)
        input_layout.addLayout(model_row)

        input_layout.addWidget(QLabel("Training data (vector):"))
        vector_row = QHBoxLayout()
        self.vectorLineEdit = QLineEdit()
        self.vectorLineEdit.setPlaceholderText("Path to vector file…")
        vector_row.addWidget(self.vectorLineEdit)
        self.vectorBrowse = QPushButton("Browse…")
        self.vectorBrowse.clicked.connect(self._browse_vector)
        vector_row.addWidget(self.vectorBrowse)
        input_layout.addLayout(vector_row)

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
            input_layout.addWidget(self._vector_combo)
            self.vectorLineEdit.setVisible(False)
            self.vectorBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass

        input_layout.addWidget(QLabel("Label field:"))
        self.classFieldCombo = QComboBox()
        input_layout.addWidget(self.classFieldCombo)
        self.fieldStatusLabel = QLabel("")
        input_layout.addWidget(self.fieldStatusLabel)
        self.vectorLineEdit.editingFinished.connect(self._on_vector_path_edited)
        self._on_vector_changed()

        input_layout.addWidget(QLabel("Classifier:"))
        self.classifierCombo = QComboBox()
        for _code, name, _sk, _xgb, _lgb, _cb in _CLASSIFIER_META:
            self.classifierCombo.addItem(name)
        input_layout.addWidget(self.classifierCombo)

        input_group.setLayout(input_layout)
        root.addWidget(input_group)

        output_group = QGroupBox("Output")
        output_layout = QGridLayout()
        output_layout.addWidget(QLabel("Classification map (optional):"), 0, 0)
        self.outRasterEdit = QLineEdit()
        self.outRasterEdit.setPlaceholderText("<temporary file>")
        output_layout.addWidget(self.outRasterEdit, 0, 1)
        self.outRasterBrowse = QPushButton("Browse…")
        self.outRasterBrowse.clicked.connect(self._browse_out_raster)
        output_layout.addWidget(self.outRasterBrowse, 0, 2)

        self.confidenceCheck = QCheckBox("Generate confidence map")
        self.confidenceCheck.toggled.connect(self._toggle_confidence)
        output_layout.addWidget(self.confidenceCheck, 1, 0, 1, 3)

        output_layout.addWidget(QLabel("Confidence map path:"), 2, 0)
        self.confMapEdit = QLineEdit()
        self.confMapEdit.setEnabled(False)
        self.confMapEdit.setPlaceholderText("Path to confidence map…")
        output_layout.addWidget(self.confMapEdit, 2, 1)
        self.confMapBrowse = QPushButton("Browse…")
        self.confMapBrowse.setEnabled(False)
        self.confMapBrowse.clicked.connect(self._browse_conf_map)
        output_layout.addWidget(self.confMapBrowse, 2, 2)

        output_group.setLayout(output_layout)
        root.addWidget(output_group)

        run_row = QHBoxLayout()
        run_row.addStretch()
        self.runButton = QPushButton("Run classification")
        self.runButton.clicked.connect(self._emit_config)
        run_row.addWidget(self.runButton)
        root.addLayout(run_row)

        root.addStretch()
        self.setLayout(root)

    def _toggle_model_mode(self, checked):
        # type: (bool) -> None
        self.modelLineEdit.setEnabled(checked)
        self.modelBrowse.setEnabled(checked)
        self.vectorLineEdit.setEnabled(not checked)
        self.classFieldCombo.setEnabled(not checked)
        if self._vector_combo is not None:
            self._vector_combo.setEnabled(not checked)

    def _toggle_confidence(self, checked):
        # type: (bool) -> None
        self.confMapEdit.setEnabled(checked)
        self.confMapBrowse.setEnabled(checked)

    def _browse_raster(self):
        # type: () -> None
        path, _f = QFileDialog.getOpenFileName(self, "Select raster", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.rasterLineEdit.setText(path)

    def _browse_vector(self):
        # type: () -> None
        path, _f = QFileDialog.getOpenFileName(
            self, "Select vector", "", "Shapefile (*.shp);;GeoPackage (*.gpkg);;All (*)"
        )
        if path:
            self.vectorLineEdit.setText(path)
            self._populate_fields_from_path(path)

    def _browse_model(self):
        # type: () -> None
        path, _f = QFileDialog.getOpenFileName(self, "Select model", "", "Model files (*)")
        if path:
            self.modelLineEdit.setText(path)

    def _browse_out_raster(self):
        # type: () -> None
        path, _f = QFileDialog.getSaveFileName(self, "Classification map", "", "GeoTIFF (*.tif)")
        if path:
            self.outRasterEdit.setText(path)

    def _browse_conf_map(self):
        # type: () -> None
        path, _f = QFileDialog.getSaveFileName(self, "Confidence map", "", "GeoTIFF (*.tif)")
        if path:
            self.confMapEdit.setText(path)

    def _on_vector_changed(self):
        # type: () -> None
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
                self.fieldStatusLabel.setText("No fields found in selected layer.")
        except (AttributeError, TypeError):
            self.fieldStatusLabel.setText("Unable to read fields from selected layer.")

    def _on_vector_path_edited(self):
        # type: () -> None
        path = self.vectorLineEdit.text().strip()
        if not path:
            self.classFieldCombo.clear()
            self.fieldStatusLabel.setText("")
            return
        self._populate_fields_from_path(path)

    def _populate_fields_from_path(self, path):
        # type: (str) -> None
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
                self.fieldStatusLabel.setText("OGR unavailable; cannot list fields.")
                return
        ds = ogr.Open(path)
        if ds is None:
            self.fieldStatusLabel.setText("Unable to open vector dataset.")
            return
        layer = ds.GetLayer()
        if layer is None:
            self.fieldStatusLabel.setText("No layer found in dataset.")
            return
        dfn = layer.GetLayerDefn()
        count = dfn.GetFieldCount()
        if count == 0:
            self.fieldStatusLabel.setText("No fields found in vector dataset.")
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

    def _quick_extra_params(self):
        # type: () -> Dict[str, object]
        return {
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

    def _emit_config(self):
        # type: () -> None
        raster = self._get_raster_path()
        if not raster:
            QMessageBox.warning(self, "Missing Input", "Please select an input raster.")
            return

        load_model = self.modelLineEdit.text().strip() if self.loadModelCheck.isChecked() else ""
        vector = self._get_vector_path()
        class_field = self.classFieldCombo.currentText().strip()

        if not load_model and (not vector or not class_field):
            QMessageBox.warning(
                self,
                "Missing Training Data",
                "Quick run requires training data and a label field when no model is loaded.",
            )
            return

        config = {
            "raster": raster,
            "vector": vector,
            "class_field": class_field,
            "load_model": load_model,
            "classifier": self._get_classifier_code(),
            "extraParam": self._quick_extra_params(),
            "output_raster": self.outRasterEdit.text().strip(),
            "confidence_map": self.confMapEdit.text().strip() if self.confidenceCheck.isChecked() else "",
            "save_model": "",
            "confusion_matrix": "",
            "split_percent": 100,
        }
        self.classificationRequested.emit(config)


class ClassificationDashboardDock(QDockWidget):
    """Dockable dashboard with Quick Run and Advanced Setup modes."""

    closingPlugin = pyqtSignal()
    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None, installer=None):
        super(ClassificationDashboardDock, self).__init__(parent)
        self.setWindowTitle("dzetsaka: Classification")
        self.setObjectName("DzetsakaClassificationDashboardDock")
        self.setMinimumWidth(430)
        self.setMinimumHeight(520)
        self.setMaximumWidth(700)

        container = QWidget()
        layout = QVBoxLayout(container)

        header = QHBoxLayout()
        header.addWidget(QLabel("Mode:"))
        self.modeCombo = QComboBox()
        self.modeCombo.addItems(["Quick run", "Advanced setup"])
        header.addWidget(self.modeCombo)
        header.addStretch()
        layout.addLayout(header)

        self.modeHint = QLabel("Quick run: essential inputs and one-click run.")
        layout.addWidget(self.modeHint)

        self.stack = QStackedWidget()
        self.quickPanel = QuickClassificationPanel()
        self.quickPanel.classificationRequested.connect(self.classificationRequested)

        self.advancedWizard = ClassificationWizard(
            parent=container,
            installer=installer,
            close_on_accept=False,
        )
        self.advancedWizard.classificationRequested.connect(self.classificationRequested)

        self.stack.addWidget(self.quickPanel)
        self.stack.addWidget(self.advancedWizard)
        layout.addWidget(self.stack)

        self.modeCombo.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed(0)

        self.setWidget(container)

    def _on_mode_changed(self, index):
        # type: (int) -> None
        self.stack.setCurrentIndex(index)
        if index == 0:
            self.modeHint.setText("Quick run: essential inputs and one-click run.")
        else:
            self.modeHint.setText("Advanced setup: full workflow with detailed optimization and outputs.")

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()
