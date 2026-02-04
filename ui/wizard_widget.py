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

import os
import tempfile
from typing import Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
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
        Keys: ``sklearn``, ``xgboost``, ``lightgbm``, ``optuna``,
        ``shap``, ``imblearn``.  Values: True when the package can be
        imported successfully.
    """
    deps = {
        "sklearn": False,
        "xgboost": False,
        "lightgbm": False,
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
    lines.append("[Input Data]")
    lines.append("  Raster : " + str(config.get("raster", "<not set>")))
    lines.append("  Vector : " + str(config.get("vector", "<not set>")))
    lines.append("  Class field : " + str(config.get("class_field", "<not set>")))
    model_path = config.get("load_model", "")
    if model_path:
        lines.append("  Load model : " + str(model_path))
    lines.append("")

    # --- Algorithm ---
    lines.append("[Algorithm]")
    lines.append("  Classifier : " + str(config.get("classifier", "<not set>")))
    lines.append("")

    # --- Advanced Options ---
    lines.append("[Advanced Options]")
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
    lines.append("  Output raster : " + str(config.get("output_raster", "<temp file>")))
    lines.append("  Confidence map : " + str(config.get("confidence_map", "")))
    lines.append("  Save model : " + str(config.get("save_model", "")))
    lines.append("  Confusion matrix : " + str(config.get("confusion_matrix", "")))
    if config.get("confusion_matrix", ""):
        lines.append("    Split % : " + str(config.get("split_percent", 50)))
    lines.append("")
    lines.append("=== End of Configuration ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classifier metadata used by the wizard (mirrors classifier_config)
# ---------------------------------------------------------------------------

# (code, full name, requires_sklearn, requires_xgboost, requires_lightgbm)
_CLASSIFIER_META = [
    ("GMM", "Gaussian Mixture Model", False, False, False),
    ("RF", "Random Forest", True, False, False),
    ("SVM", "Support Vector Machine", True, False, False),
    ("KNN", "K-Nearest Neighbors", True, False, False),
    ("XGB", "XGBoost", False, True, False),
    ("LGB", "LightGBM", False, False, True),
    ("ET", "Extra Trees", True, False, False),
    ("GBC", "Gradient Boosting Classifier", True, False, False),
    ("LR", "Logistic Regression", True, False, False),
    ("NB", "Gaussian Naive Bayes", True, False, False),
    ("MLP", "Multi-layer Perceptron", True, False, False),
]


def _classifier_available(code, deps):
    # type: (str, Dict[str, bool]) -> bool
    """Return True when all hard dependencies for *code* are satisfied."""
    for c, _name, needs_sk, needs_xgb, needs_lgb in _CLASSIFIER_META:
        if c == code:
            if needs_sk and not deps.get("sklearn", False):
                return False
            if needs_xgb and not deps.get("xgboost", False):
                return False
            if needs_lgb and not deps.get("lightgbm", False):
                return False
            return True
    return False


# ---------------------------------------------------------------------------
# Page 0 — Input Data
# ---------------------------------------------------------------------------


class DataInputPage(QWizardPage):
    """Wizard page for selecting input raster, vector and class field."""

    def __init__(self, parent=None):
        """Initialise DataInputPage."""
        super(DataInputPage, self).__init__(parent)
        self.setTitle("Input Data")
        self.setSubTitle("Select the raster image and training vector layer.")

        layout = QVBoxLayout()

        # --- Raster ---
        layout.addWidget(QLabel("Input raster (GeoTIFF):"))
        raster_row = QHBoxLayout()
        self.rasterLineEdit = QLineEdit()
        self.rasterLineEdit.setPlaceholderText("Path to raster file…")
        raster_row.addWidget(self.rasterLineEdit)
        self.rasterBrowse = QPushButton("Browse…")
        self.rasterBrowse.clicked.connect(self._browse_raster)
        raster_row.addWidget(self.rasterBrowse)
        layout.addLayout(raster_row)

        # Try to use QgsMapLayerComboBox when QGIS is available
        self._raster_combo = None  # type: Optional[QWidget]
        try:
            from qgis.core import QgsMapLayerFilterModel, QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._raster_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            exclude.remove("gdal")
            self._raster_combo.setExcludedProviders(exclude)
            layout.addWidget(self._raster_combo)
            self.rasterLineEdit.setVisible(False)
            self.rasterBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass  # fallback: plain QLineEdit stays visible

        # --- Vector ---
        layout.addWidget(QLabel("Training vector (Shapefile / GeoPackage):"))
        vector_row = QHBoxLayout()
        self.vectorLineEdit = QLineEdit()
        self.vectorLineEdit.setPlaceholderText("Path to vector file…")
        vector_row.addWidget(self.vectorLineEdit)
        self.vectorBrowse = QPushButton("Browse…")
        self.vectorBrowse.clicked.connect(self._browse_vector)
        vector_row.addWidget(self.vectorBrowse)
        layout.addLayout(vector_row)

        self._vector_combo = None  # type: Optional[QWidget]
        try:
            from qgis.core import QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._vector_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            exclude.remove("ogr")
            self._vector_combo.setExcludedProviders(exclude)
            layout.addWidget(self._vector_combo)
            self.vectorLineEdit.setVisible(False)
            self.vectorBrowse.setVisible(False)
            self._vector_combo.currentIndexChanged.connect(self._on_vector_changed)
        except (ImportError, AttributeError):
            pass

        # --- Class field ---
        layout.addWidget(QLabel("Class field:"))
        self.classFieldCombo = QComboBox()
        layout.addWidget(self.classFieldCombo)

        # --- Load model mode ---
        layout.addWidget(QLabel(""))
        self.loadModelCheck = QCheckBox("Load an existing model (skip training)")
        layout.addWidget(self.loadModelCheck)

        model_row = QHBoxLayout()
        self.modelLineEdit = QLineEdit()
        self.modelLineEdit.setPlaceholderText("Path to saved model…")
        self.modelLineEdit.setEnabled(False)
        model_row.addWidget(self.modelLineEdit)
        self.modelBrowse = QPushButton("Browse…")
        self.modelBrowse.setEnabled(False)
        self.modelBrowse.clicked.connect(self._browse_model)
        model_row.addWidget(self.modelBrowse)
        layout.addLayout(model_row)

        self.loadModelCheck.toggled.connect(self._toggle_model_mode)

        # Register wizard fields
        self.registerField("raster", self.rasterLineEdit)
        self.registerField("vector", self.vectorLineEdit)
        self.registerField("loadModel", self.modelLineEdit)

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
        if self._vector_combo is None:
            return
        layer = self._vector_combo.currentLayer()
        if layer is None:
            return
        try:
            fields = layer.dataProvider().fields()
            self.classFieldCombo.addItems([fields.at(i).name() for i in range(fields.count())])
        except (AttributeError, TypeError):
            pass

    def _populate_fields_from_path(self, path):
        # type: (str) -> None
        """Best-effort field listing for a vector path (fallback without QGIS)."""
        self.classFieldCombo.clear()
        try:
            from osgeo import ogr
        except ImportError:
            try:
                import ogr  # type: ignore[no-redef]
            except ImportError:
                return
        ds = ogr.Open(path)
        if ds is None:
            return
        layer = ds.GetLayer()
        if layer is None:
            return
        for i in range(layer.GetLayerDefn().GetFieldCount()):
            self.classFieldCombo.addItem(layer.GetLayerDefn().GetFieldDefn(i).GetName())

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


# ---------------------------------------------------------------------------
# Page 1 — Algorithm
# ---------------------------------------------------------------------------


class AlgorithmPage(QWizardPage):
    """Wizard page for selecting the classification algorithm."""

    # Signal emitted when the user clicks "Use Selected" in the comparison panel
    algorithmFromComparison = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialise AlgorithmPage."""
        super(AlgorithmPage, self).__init__(parent)
        self.setTitle("Algorithm")
        self.setSubTitle("Choose a classification algorithm.")

        self._deps = check_dependency_availability()
        self._comparison_panel = None  # type: Optional[object]

        layout = QVBoxLayout()

        # Classifier combo
        layout.addWidget(QLabel("Classifier:"))
        self.classifierCombo = QComboBox()
        for _code, name, _sk, _xgb, _lgb in _CLASSIFIER_META:
            self.classifierCombo.addItem(name)
        layout.addWidget(self.classifierCombo)

        # Dependency status label
        self.depStatusLabel = QLabel()
        layout.addWidget(self.depStatusLabel)

        self.classifierCombo.currentIndexChanged.connect(self._update_dep_status)
        self._update_dep_status(0)

        # Buttons row
        btn_row = QHBoxLayout()
        self.smartDefaultsBtn = QPushButton("Smart Defaults")
        self.smartDefaultsBtn.setToolTip("Enable Optuna / SHAP / SMOTE when packages are available.")
        self.smartDefaultsBtn.clicked.connect(self._apply_smart_defaults)
        btn_row.addWidget(self.smartDefaultsBtn)

        self.compareBtn = QPushButton("Compare…")
        self.compareBtn.setToolTip("Open the algorithm comparison panel.")
        self.compareBtn.clicked.connect(self._open_comparison)
        btn_row.addWidget(self.compareBtn)
        layout.addLayout(btn_row)

        # Store the smart defaults for later retrieval by the wizard
        self._smart_defaults_applied = False  # type: bool

        self.setLayout(layout)

    # --- internal helpers --------------------------------------------------

    def _update_dep_status(self, index):
        # type: (int) -> None
        """Colour the dep-status label green or red based on availability."""
        code = _CLASSIFIER_META[index][0]
        available = _classifier_available(code, self._deps)
        if available:
            self.depStatusLabel.setText("<span style='color:green;'>All dependencies satisfied.</span>")
        else:
            missing = []  # type: List[str]
            _code, _name, needs_sk, needs_xgb, needs_lgb = _CLASSIFIER_META[index]
            if needs_sk and not self._deps.get("sklearn", False):
                missing.append("scikit-learn")
            if needs_xgb and not self._deps.get("xgboost", False):
                missing.append("xgboost")
            if needs_lgb and not self._deps.get("lightgbm", False):
                missing.append("lightgbm")
            self.depStatusLabel.setText(
                "<span style='color:red;'>Missing: " + ", ".join(missing) + "</span>"
            )

    def _apply_smart_defaults(self):
        # type: () -> None
        """Flag that smart defaults should be applied on the Advanced page."""
        self._smart_defaults_applied = True

    def _open_comparison(self):
        # type: () -> None
        """Show the AlgorithmComparisonPanel dialog."""
        from .comparison_panel import AlgorithmComparisonPanel

        self._comparison_panel = AlgorithmComparisonPanel(self)
        self._comparison_panel.algorithmSelected.connect(self._set_algorithm_from_comparison)
        self._comparison_panel.show()

    def _set_algorithm_from_comparison(self, name):
        # type: (str) -> None
        """Set the combo to the algorithm chosen in the comparison panel."""
        for i, (_code, n, _sk, _xgb, _lgb) in enumerate(_CLASSIFIER_META):
            if n == name:
                self.classifierCombo.setCurrentIndex(i)
                break

    # --- public API --------------------------------------------------------

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

    def __init__(self, parent=None):
        """Initialise AdvancedOptionsPage."""
        super(AdvancedOptionsPage, self).__init__(parent)
        self.setTitle("Advanced Options")
        self.setSubTitle("Configure optional enhancements.")

        self._deps = check_dependency_availability()

        layout = QVBoxLayout()

        # --- Optimization group ---
        opt_group = QGroupBox("Optimization")
        opt_layout = QVBoxLayout()

        self.optunaCheck = QCheckBox("Use Optuna hyperparameter optimization")
        self.optunaCheck.setEnabled(self._deps.get("optuna", False))
        opt_layout.addWidget(self.optunaCheck)

        trials_row = QHBoxLayout()
        trials_row.addWidget(QLabel("  Trials:"))
        self.optunaTrials = QSpinBox()
        self.optunaTrials.setRange(10, 1000)
        self.optunaTrials.setValue(100)
        self.optunaTrials.setEnabled(False)
        trials_row.addWidget(self.optunaTrials)
        opt_layout.addLayout(trials_row)

        self.optunaCheck.toggled.connect(self.optunaTrials.setEnabled)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # --- Imbalance group ---
        imb_group = QGroupBox("Imbalance Handling")
        imb_layout = QVBoxLayout()

        self.smoteCheck = QCheckBox("SMOTE oversampling")
        self.smoteCheck.setEnabled(self._deps.get("imblearn", False))
        imb_layout.addWidget(self.smoteCheck)

        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("  k_neighbors:"))
        self.smoteK = QSpinBox()
        self.smoteK.setRange(1, 20)
        self.smoteK.setValue(5)
        self.smoteK.setEnabled(False)
        k_row.addWidget(self.smoteK)
        imb_layout.addLayout(k_row)
        self.smoteCheck.toggled.connect(self.smoteK.setEnabled)

        self.classWeightCheck = QCheckBox("Use class weights")
        self.classWeightCheck.setEnabled(self._deps.get("sklearn", False))
        imb_layout.addWidget(self.classWeightCheck)

        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("  Strategy:"))
        self.weightStrategyCombo = QComboBox()
        self.weightStrategyCombo.addItems(["balanced", "uniform"])
        self.weightStrategyCombo.setEnabled(False)
        strat_row.addWidget(self.weightStrategyCombo)
        imb_layout.addLayout(strat_row)
        self.classWeightCheck.toggled.connect(self.weightStrategyCombo.setEnabled)

        imb_group.setLayout(imb_layout)
        layout.addWidget(imb_group)

        # --- Explainability group ---
        exp_group = QGroupBox("Explainability")
        exp_layout = QVBoxLayout()

        self.shapCheck = QCheckBox("Compute SHAP feature importance")
        self.shapCheck.setEnabled(self._deps.get("shap", False))
        exp_layout.addWidget(self.shapCheck)

        shap_path_row = QHBoxLayout()
        shap_path_row.addWidget(QLabel("  Output:"))
        self.shapOutput = QLineEdit()
        self.shapOutput.setPlaceholderText("Path to SHAP raster…")
        self.shapOutput.setEnabled(False)
        shap_path_row.addWidget(self.shapOutput)
        self.shapBrowse = QPushButton("Browse…")
        self.shapBrowse.setEnabled(False)
        self.shapBrowse.clicked.connect(self._browse_shap_output)
        shap_path_row.addWidget(self.shapBrowse)
        exp_layout.addLayout(shap_path_row)

        shap_sample_row = QHBoxLayout()
        shap_sample_row.addWidget(QLabel("  Sample size:"))
        self.shapSampleSize = QSpinBox()
        self.shapSampleSize.setRange(100, 50000)
        self.shapSampleSize.setValue(1000)
        self.shapSampleSize.setEnabled(False)
        shap_sample_row.addWidget(self.shapSampleSize)
        exp_layout.addLayout(shap_sample_row)

        def _toggle_shap(checked):
            # type: (bool) -> None
            self.shapOutput.setEnabled(checked)
            self.shapBrowse.setEnabled(checked)
            self.shapSampleSize.setEnabled(checked)

        self.shapCheck.toggled.connect(_toggle_shap)
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)

        # --- Validation group ---
        val_group = QGroupBox("Validation")
        val_layout = QVBoxLayout()

        self.nestedCVCheck = QCheckBox("Nested cross-validation")
        val_layout.addWidget(self.nestedCVCheck)

        inner_row = QHBoxLayout()
        inner_row.addWidget(QLabel("  Inner folds:"))
        self.innerFolds = QSpinBox()
        self.innerFolds.setRange(2, 10)
        self.innerFolds.setValue(3)
        self.innerFolds.setEnabled(False)
        inner_row.addWidget(self.innerFolds)
        val_layout.addLayout(inner_row)

        outer_row = QHBoxLayout()
        outer_row.addWidget(QLabel("  Outer folds:"))
        self.outerFolds = QSpinBox()
        self.outerFolds.setRange(2, 10)
        self.outerFolds.setValue(5)
        self.outerFolds.setEnabled(False)
        outer_row.addWidget(self.outerFolds)
        val_layout.addLayout(outer_row)

        def _toggle_nested(checked):
            # type: (bool) -> None
            self.innerFolds.setEnabled(checked)
            self.outerFolds.setEnabled(checked)

        self.nestedCVCheck.toggled.connect(_toggle_nested)
        val_group.setLayout(val_layout)
        layout.addWidget(val_group)

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
        self.setTitle("Output")
        self.setSubTitle("Set output paths and optional outputs.")

        layout = QVBoxLayout()

        # Output raster
        layout.addWidget(QLabel("Output raster (leave blank for temp file):"))
        out_row = QHBoxLayout()
        self.outRasterEdit = QLineEdit()
        self.outRasterEdit.setPlaceholderText("<temporary file>")
        out_row.addWidget(self.outRasterEdit)
        self.outRasterBrowse = QPushButton("Browse…")
        self.outRasterBrowse.clicked.connect(self._browse_out_raster)
        out_row.addWidget(self.outRasterBrowse)
        layout.addLayout(out_row)

        # Confidence map
        self.confidenceCheck = QCheckBox("Generate confidence map")
        layout.addWidget(self.confidenceCheck)
        conf_row = QHBoxLayout()
        self.confMapEdit = QLineEdit()
        self.confMapEdit.setPlaceholderText("Path to confidence map…")
        self.confMapEdit.setEnabled(False)
        conf_row.addWidget(self.confMapEdit)
        self.confMapBrowse = QPushButton("Browse…")
        self.confMapBrowse.setEnabled(False)
        self.confMapBrowse.clicked.connect(self._browse_conf_map)
        conf_row.addWidget(self.confMapBrowse)
        layout.addLayout(conf_row)

        def _toggle_conf(checked):
            # type: (bool) -> None
            self.confMapEdit.setEnabled(checked)
            self.confMapBrowse.setEnabled(checked)

        self.confidenceCheck.toggled.connect(_toggle_conf)

        # Save model
        self.saveModelCheck = QCheckBox("Save trained model")
        layout.addWidget(self.saveModelCheck)
        model_row = QHBoxLayout()
        self.saveModelEdit = QLineEdit()
        self.saveModelEdit.setPlaceholderText("Path to save model…")
        self.saveModelEdit.setEnabled(False)
        model_row.addWidget(self.saveModelEdit)
        self.saveModelBrowse = QPushButton("Browse…")
        self.saveModelBrowse.setEnabled(False)
        self.saveModelBrowse.clicked.connect(self._browse_save_model)
        model_row.addWidget(self.saveModelBrowse)
        layout.addLayout(model_row)

        def _toggle_save_model(checked):
            # type: (bool) -> None
            self.saveModelEdit.setEnabled(checked)
            self.saveModelBrowse.setEnabled(checked)

        self.saveModelCheck.toggled.connect(_toggle_save_model)

        # Confusion matrix
        self.matrixCheck = QCheckBox("Save confusion matrix")
        layout.addWidget(self.matrixCheck)
        matrix_row = QHBoxLayout()
        self.matrixEdit = QLineEdit()
        self.matrixEdit.setPlaceholderText("Path to CSV…")
        self.matrixEdit.setEnabled(False)
        matrix_row.addWidget(self.matrixEdit)
        self.matrixBrowse = QPushButton("Browse…")
        self.matrixBrowse.setEnabled(False)
        self.matrixBrowse.clicked.connect(self._browse_matrix)
        matrix_row.addWidget(self.matrixBrowse)
        layout.addLayout(matrix_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("  Train/test split %:"))
        self.splitSpinBox = QSpinBox()
        self.splitSpinBox.setRange(10, 90)
        self.splitSpinBox.setValue(50)
        self.splitSpinBox.setEnabled(False)
        split_row.addWidget(self.splitSpinBox)
        layout.addLayout(split_row)

        def _toggle_matrix(checked):
            # type: (bool) -> None
            self.matrixEdit.setEnabled(checked)
            self.matrixBrowse.setEnabled(checked)
            self.splitSpinBox.setEnabled(checked)

        self.matrixCheck.toggled.connect(_toggle_matrix)

        self.setLayout(layout)

    # --- internal helpers --------------------------------------------------

    def _browse_out_raster(self):
        # type: () -> None
        """Browse for the output raster path."""
        path, _f = QFileDialog.getSaveFileName(self, "Output raster", "", "GeoTIFF (*.tif)")
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


# ---------------------------------------------------------------------------
# Page 4 — Review & Run
# ---------------------------------------------------------------------------


class ReviewPage(QWizardPage):
    """Read-only summary page generated from all previous pages."""

    def __init__(self, parent=None):
        """Initialise ReviewPage."""
        super(ReviewPage, self).__init__(parent)
        self.setTitle("Review & Run")
        self.setSubTitle("Verify your settings before starting the classification.")

        layout = QVBoxLayout()
        self.reviewEdit = QTextEdit()
        self.reviewEdit.setReadOnly(True)
        layout.addWidget(self.reviewEdit)
        self.setLayout(layout)

    def initializePage(self):
        # type: () -> None
        """Called by QWizard before showing this page; regenerates the summary."""
        wizard = self.wizard()  # type: Optional[ClassificationWizard]
        if wizard is None:
            return
        config = wizard.collect_config()
        self.reviewEdit.setPlainText(build_review_summary(config))


# ---------------------------------------------------------------------------
# Main Wizard
# ---------------------------------------------------------------------------


class ClassificationWizard(QWizard):
    """Step-by-step classification wizard for dzetsaka.

    Emits ``classificationRequested`` with the assembled config dict
    when the user clicks Finish (labelled "Run Classification").
    """

    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialise ClassificationWizard with all 5 pages."""
        super(ClassificationWizard, self).__init__(parent)
        self.setWindowTitle("dzetsaka Classification Wizard")

        # Pages
        self.dataPage = DataInputPage()
        self.algoPage = AlgorithmPage()
        self.advPage = AdvancedOptionsPage()
        self.outputPage = OutputConfigPage()
        self.reviewPage = ReviewPage()

        self.addPage(self.dataPage)      # index 0
        self.addPage(self.algoPage)      # index 1
        self.addPage(self.advPage)       # index 2
        self.addPage(self.outputPage)    # index 3
        self.addPage(self.reviewPage)    # index 4

        # Override the Finish button text
        self.setButtonText(QWizard.FinishButton, "Run Classification")

        self.setWizardStyle(QWizard.ModernStyle)

    # --- page-transition hook ---------------------------------------------

    def validateCurrentPage(self):
        # type: () -> bool
        """Handle smart-defaults propagation when leaving the Algorithm page."""
        current = self.currentId()
        # Leaving AlgorithmPage (index 1) -> entering AdvancedOptionsPage
        if current == 1 and self.algoPage.smart_defaults_requested():
            defaults = build_smart_defaults(check_dependency_availability())
            self.advPage.apply_smart_defaults(defaults)
            # Reset flag so it fires only once
            self.algoPage._smart_defaults_applied = False
        return super(ClassificationWizard, self).validateCurrentPage()

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
        config["classifier"] = self.algoPage.get_classifier_code()

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
        super(ClassificationWizard, self).accept()
