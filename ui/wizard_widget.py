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

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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
# Page 0 — Input & Algorithm
# ---------------------------------------------------------------------------


class DataInputPage(QWizardPage):
    """Wizard page for selecting input data and classifier."""

    def __init__(self, parent=None, deps=None, installer=None):
        """Initialise DataInputPage."""
        super(DataInputPage, self).__init__(parent)
        self.setTitle("Input & Algorithm")
        self.setSubTitle("Select training data and choose a classifier.")

        self._deps = deps if deps is not None else check_dependency_availability()
        self._installer = installer
        self._smart_defaults_applied = False  # type: bool
        self._last_prompt_signature = None  # type: Optional[tuple]

        layout = QVBoxLayout()

        # --- Input group ---
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()

        input_layout.addWidget(QLabel("Input raster (GeoTIFF):"))
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

        input_layout.addWidget(QLabel("Training vector (Shapefile / GeoPackage):"))
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

        input_layout.addWidget(QLabel("Class field:"))
        self.classFieldCombo = QComboBox()
        input_layout.addWidget(self.classFieldCombo)
        self.fieldStatusLabel = QLabel()
        self.fieldStatusLabel.setStyleSheet("color: #666;")
        input_layout.addWidget(self.fieldStatusLabel)

        self.loadModelCheck = QCheckBox("Load an existing model (skip training)")
        input_layout.addWidget(self.loadModelCheck)

        model_row = QHBoxLayout()
        self.modelLineEdit = QLineEdit()
        self.modelLineEdit.setPlaceholderText("Path to saved model…")
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
        for _code, name, _sk, _xgb, _lgb in _CLASSIFIER_META:
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
        _code, _name, needs_sk, needs_xgb, needs_lgb = _CLASSIFIER_META[index]
        if needs_sk and not self._deps.get("sklearn", False):
            missing_required.append("scikit-learn")
        if needs_xgb and not self._deps.get("xgboost", False):
            missing_required.append("xgboost")
        if needs_lgb and not self._deps.get("lightgbm", False):
            missing_required.append("lightgbm")

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
        if not missing_required and not missing_optional:
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
            "optuna": "optuna",
            "shap": "shap",
            "imblearn (SMOTE)": "imbalanced-learn",
        }

        required_msg = ""
        for dep in missing_required:
            required_msg += f"{dep} is missing.<br>"
            if dep in package_map:
                required_msg += f"Install with: <code>pip install {package_map[dep]}</code><br><br>"

        optional_msg = ""
        for dep in missing_optional:
            optional_msg += f"{dep} is missing (optional).<br>"
            if dep in package_map:
                optional_msg += f"Install with: <code>pip install {package_map[dep]}</code><br><br>"

        if missing_required:
            error_message = "<b>Required dependencies:</b><br>" + required_msg
            if missing_optional:
                error_message += "<b>Optional enhancements:</b><br>" + optional_msg
            reply = QMessageBox.question(
                self,
                f"Missing Dependencies for {classifier_name}",
                f"{error_message}<br>"
                f"<b>🧪 Experimental Feature:</b><br>"
                f"Would you like dzetsaka to try installing the missing dependencies automatically?<br><br>"
                f"<b>Note:</b> This is experimental and may not work in all QGIS environments.<br>"
                f"Please wait — installing dependencies can take up to 2 minutes.<br>"
                f"Click 'Yes' to try auto-install, 'No' to install manually, or 'Cancel' to use GMM.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                to_install = [package_map.get(dep, dep) for dep in (missing_required + missing_optional)]
                if self._installer._try_install_dependencies(to_install):
                    QMessageBox.information(
                        self,
                        "Installation Successful",
                        f"Dependencies installed successfully!<br><br>"
                        f"<b>Note:</b> If {classifier_name} doesn't work immediately, "
                        f"please restart QGIS to ensure the new libraries are properly loaded.",
                        QMessageBox.Ok,
                    )
                    self._deps = check_dependency_availability()
                    self._update_dep_status(self.classifierCombo.currentIndex())
                else:
                    self.classifierCombo.setCurrentIndex(0)
            elif reply == QMessageBox.Cancel:
                self.classifierCombo.setCurrentIndex(0)

        elif missing_optional:
            reply = QMessageBox.question(
                self,
                "Optional Enhancements Available",
                f"{optional_msg}"
                f"<b>Note:</b> {classifier_name} works without them using standard options.<br><br>"
                f"Please wait — installing dependencies can take up to 2 minutes.<br>"
                f"Would you like to install them automatically?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                to_install = [package_map.get(dep, dep) for dep in missing_optional]
                self._installer._try_install_dependencies(to_install)
                self._deps = check_dependency_availability()
                self._update_dep_status(self.classifierCombo.currentIndex())

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
        for i, (_code, n, _sk, _xgb, _lgb) in enumerate(_CLASSIFIER_META):
            if n == name:
                self.classifierCombo.setCurrentIndex(i)
                break

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
        self.setTitle("Advanced Options")
        self.setSubTitle("Configure optional enhancements.")

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
        self.setTitle("Output & Review")
        self.setSubTitle("Set output paths and confirm settings.")

        layout = QVBoxLayout()

        out_group = QGroupBox("Output Files")
        out_layout = QGridLayout()
        out_layout.setContentsMargins(8, 8, 8, 8)
        out_layout.setHorizontalSpacing(10)
        out_layout.setVerticalSpacing(6)

        out_layout.addWidget(QLabel("Output raster:"), 0, 0)
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
        out_layout.addWidget(QLabel("Model path:"), 4, 0)
        self.saveModelEdit = QLineEdit()
        self.saveModelEdit.setPlaceholderText("Path to save model…")
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

        out_layout.addWidget(QLabel("Train/test split %:"), 7, 0)
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


# ---------------------------------------------------------------------------
# Main Wizard
# ---------------------------------------------------------------------------


class ClassificationWizard(QWizard):
    """Step-by-step classification wizard for dzetsaka.

    Emits ``classificationRequested`` with the assembled config dict
    when the user clicks Finish (labelled "Run Classification").
    """

    classificationRequested = pyqtSignal(dict)

    def __init__(self, parent=None, installer=None):
        """Initialise ClassificationWizard with all 3 pages."""
        super(ClassificationWizard, self).__init__(parent)
        self.setWindowTitle("dzetsaka Classification Wizard")
        self._deps = check_dependency_availability()
        self._installer = installer

        self.setStyleSheet(
            """
            QWizard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f7f7fb, stop:1 #eef1f7);
                font-family: "Segoe UI Variable", "Plus Jakarta Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 12px;
                color: #0f172a;
            }
            QWizardPage {
                background: transparent;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e6e8f0;
                border-radius: 12px;
                margin-top: 12px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #111827;
                font-weight: 600;
            }
            QLabel {
                color: #0f172a;
            }
            QLineEdit, QComboBox, QSpinBox, QTextEdit {
                background: #ffffff;
                border: 1px solid #d7dbe6;
                border-radius: 8px;
                padding: 6px 8px;
                selection-background-color: #fde68a;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
                border: 1px solid #b45309;
            }
            QPushButton {
                background: #f3f4f6;
                color: #111827;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 7px 12px;
            }
            QPushButton:hover {
                background: #e5e7eb;
            }
            QPushButton:disabled {
                background: #e5e7eb;
                color: #9ca3af;
            }
            QCheckBox {
                padding: 2px 0;
            }
            QTextEdit {
                background: #f9fafb;
            }
            """
        )

        # Pages
        self.dataPage = DataInputPage(deps=self._deps, installer=self._installer)
        self.advPage = AdvancedOptionsPage(deps=self._deps)
        self.outputPage = OutputConfigPage()

        self.addPage(self.dataPage)      # index 0
        self.addPage(self.advPage)       # index 1
        self.addPage(self.outputPage)    # index 2

        # Override the Finish button text
        self.setButtonText(QWizard.FinishButton, "Run Classification")

        self.setWizardStyle(QWizard.ModernStyle)

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
        super(ClassificationWizard, self).accept()
