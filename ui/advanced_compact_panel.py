import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class AdvancedCompactPanel(QWidget):
    """Compact intent-driven advanced panel for small QGIS dock usage."""

    classificationRequested = pyqtSignal(dict)
    openExpertModeRequested = pyqtSignal()

    def __init__(self, parent=None, deps=None, classifier_meta=None):
        super(AdvancedCompactPanel, self).__init__(parent)
        self._deps = deps or {}
        self._classifier_meta = list(classifier_meta or [])
        self._setting_goal = False
        self._setup_ui()
        self._apply_goal_defaults(self.goalCombo.currentText())
        self._update_status()

    def _setup_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        # Express-like compact essentials (always visible, no framed box)
        essentials_widget = QWidget()
        essentials_layout = QVBoxLayout(essentials_widget)
        essentials_layout.setContentsMargins(0, 0, 0, 0)
        essentials_layout.setSpacing(2)

        goal_row = QHBoxLayout()
        goal_row.setContentsMargins(0, 0, 0, 0)
        goal_row.setSpacing(2)
        goal_row.addWidget(QLabel("Goal:"))
        self.goalCombo = QComboBox()
        self.goalCombo.addItems([
            "Fast map now",
            "Best accuracy",
            "Explain results",
            "Imbalanced classes",
        ])
        self.goalCombo.currentTextChanged.connect(self._apply_goal_defaults)
        goal_row.addWidget(self.goalCombo)
        essentials_layout.addLayout(goal_row)

        raster_row = QHBoxLayout()
        raster_row.setContentsMargins(0, 0, 0, 0)
        raster_row.setSpacing(2)
        raster_row.addWidget(
            self._icon_label(
                "modern/ux_raster.png",
                "Raster to classify",
                fallback_resource=":/plugins/dzetsaka/img/raster.svg",
            )
        )
        self.rasterEdit = QLineEdit()
        self.rasterEdit.setPlaceholderText("Path to raster file...")
        self.rasterEdit.setToolTip("Raster to classify")
        self.rasterEdit.textChanged.connect(self._update_status)
        raster_row.addWidget(self.rasterEdit)
        self.rasterBrowse = QPushButton("Browse...")
        self.rasterBrowse.clicked.connect(self._browse_raster)
        raster_row.addWidget(self.rasterBrowse)
        self._raster_combo = None
        try:
            from qgis.core import QgsProviderRegistry
            from qgis.gui import QgsMapLayerComboBox

            self._raster_combo = QgsMapLayerComboBox()
            exclude = QgsProviderRegistry.instance().providerList()
            if "gdal" in exclude:
                exclude.remove("gdal")
            self._raster_combo.setExcludedProviders(exclude)
            if hasattr(self._raster_combo, "layerChanged"):
                self._raster_combo.layerChanged.connect(self._update_status)
            self._raster_combo.currentIndexChanged.connect(self._update_status)
            self._raster_combo.setToolTip("Raster to classify")
            raster_row.addWidget(self._raster_combo)
            self.rasterEdit.setVisible(False)
            self.rasterBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass
        essentials_layout.addLayout(raster_row)

        vector_row = QHBoxLayout()
        vector_row.setContentsMargins(0, 0, 0, 0)
        vector_row.setSpacing(2)
        vector_row.addWidget(
            self._icon_label(
                "modern/ux_vector.png",
                "Training vector layer",
                fallback_resource=":/plugins/dzetsaka/img/vector.svg",
            )
        )
        self.vectorEdit = QLineEdit()
        self.vectorEdit.setPlaceholderText("Path to training vector...")
        self.vectorEdit.setToolTip("Training vector layer")
        self.vectorEdit.textChanged.connect(self._update_status)
        self.vectorEdit.editingFinished.connect(self._on_vector_path_edited)
        vector_row.addWidget(self.vectorEdit)
        self.vectorBrowse = QPushButton("Browse...")
        self.vectorBrowse.clicked.connect(self._browse_vector)
        vector_row.addWidget(self.vectorBrowse)
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
            self._vector_combo.setToolTip("Training vector layer")
            vector_row.addWidget(self._vector_combo)
            self.vectorEdit.setVisible(False)
            self.vectorBrowse.setVisible(False)
        except (ImportError, AttributeError):
            pass
        essentials_layout.addLayout(vector_row)

        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(2)
        label_row.addWidget(
            self._icon_label(
                "modern/ux_label.png",
                "Class label field",
                fallback_resource=":/plugins/dzetsaka/img/column.svg",
            )
        )
        self.classFieldCombo = QComboBox()
        self.classFieldCombo.setToolTip("Class label field")
        self.classFieldCombo.currentIndexChanged.connect(self._update_status)
        label_row.addWidget(self.classFieldCombo)
        essentials_layout.addLayout(label_row)

        self.fieldStatusLabel = QLabel("")
        self.fieldStatusLabel.setVisible(False)
        essentials_layout.addWidget(self.fieldStatusLabel)

        classifier_row = QHBoxLayout()
        classifier_row.setContentsMargins(0, 0, 0, 0)
        classifier_row.setSpacing(2)
        classifier_row.addWidget(
            self._icon_label(
                "modern/ux_classifier.png",
                "Classifier algorithm",
                fallback_resource=":/plugins/dzetsaka/img/filter.png",
            )
        )
        self.classifierCombo = QComboBox()
        self.classifierCombo.setToolTip("Classifier algorithm")
        for _code, name, _sk, _xgb, _lgb, _cb in self._classifier_meta:
            self.classifierCombo.addItem(name)
        self.classifierCombo.currentIndexChanged.connect(self._update_status)
        classifier_row.addWidget(self.classifierCombo)
        essentials_layout.addLayout(classifier_row)

        root.addWidget(essentials_widget)

        # Optional model mode in a collapsed section
        model_content = QWidget()
        model_layout = QGridLayout(model_content)
        self.loadModelCheck = QCheckBox("Use existing model instead of training data")
        self.loadModelCheck.toggled.connect(self._toggle_model_mode)
        self.loadModelCheck.toggled.connect(self._update_status)
        model_layout.addWidget(self.loadModelCheck, 0, 0, 1, 3)

        model_layout.addWidget(QLabel("Model:"), 1, 0)
        self.modelEdit = QLineEdit()
        self.modelEdit.setPlaceholderText("Path to trained model...")
        self.modelEdit.setEnabled(False)
        self.modelEdit.textChanged.connect(self._update_status)
        model_layout.addWidget(self.modelEdit, 1, 1)
        self.modelBrowse = QPushButton("Browse...")
        self.modelBrowse.setEnabled(False)
        self.modelBrowse.clicked.connect(self._browse_model)
        model_layout.addWidget(self.modelBrowse, 1, 2)

        # Quality and outputs (collapsed by default)
        q_content = QWidget()
        q_layout = QGridLayout(q_content)

        self.confidenceCheck = QCheckBox("Generate confidence map")
        self.confidenceCheck.toggled.connect(self._toggle_confidence)
        q_layout.addWidget(self.confidenceCheck, 0, 0, 1, 3)

        q_layout.addWidget(QLabel("Confidence path:"), 1, 0)
        self.confidenceEdit = QLineEdit()
        self.confidenceEdit.setEnabled(False)
        self.confidenceEdit.setPlaceholderText("Optional confidence map path...")
        q_layout.addWidget(self.confidenceEdit, 1, 1)
        self.confidenceBrowse = QPushButton("Browse...")
        self.confidenceBrowse.setEnabled(False)
        self.confidenceBrowse.clicked.connect(self._browse_confidence)
        q_layout.addWidget(self.confidenceBrowse, 1, 2)

        self.validationCheck = QCheckBox("Validation report (confusion matrix)")
        self.validationCheck.toggled.connect(self._toggle_validation)
        q_layout.addWidget(self.validationCheck, 2, 0, 1, 3)

        q_layout.addWidget(QLabel("Matrix CSV:"), 3, 0)
        self.matrixEdit = QLineEdit()
        self.matrixEdit.setEnabled(False)
        self.matrixEdit.setPlaceholderText("Optional matrix CSV path...")
        q_layout.addWidget(self.matrixEdit, 3, 1)
        self.matrixBrowse = QPushButton("Browse...")
        self.matrixBrowse.setEnabled(False)
        self.matrixBrowse.clicked.connect(self._browse_matrix)
        q_layout.addWidget(self.matrixBrowse, 3, 2)

        q_layout.addWidget(QLabel("Validation split (%):"), 4, 0)
        self.splitSpin = QSpinBox()
        self.splitSpin.setRange(10, 90)
        self.splitSpin.setValue(50)
        self.splitSpin.setEnabled(False)
        q_layout.addWidget(self.splitSpin, 4, 1)

        self.saveModelCheck = QCheckBox("Save trained model")
        self.saveModelCheck.toggled.connect(self._toggle_save_model)
        q_layout.addWidget(self.saveModelCheck, 5, 0, 1, 3)

        q_layout.addWidget(QLabel("Model output:"), 6, 0)
        self.saveModelEdit = QLineEdit()
        self.saveModelEdit.setEnabled(False)
        self.saveModelEdit.setPlaceholderText("Optional output model path...")
        q_layout.addWidget(self.saveModelEdit, 6, 1)
        self.saveModelBrowse = QPushButton("Browse...")
        self.saveModelBrowse.setEnabled(False)
        self.saveModelBrowse.clicked.connect(self._browse_save_model)
        q_layout.addWidget(self.saveModelBrowse, 6, 2)

        # Expert options (collapsed by default)
        expert_content = QWidget()
        expert_layout = QGridLayout(expert_content)

        self.optunaCheck = QCheckBox("Use Optuna")
        self.optunaCheck.setEnabled(bool(self._deps.get("optuna", False)))
        expert_layout.addWidget(self.optunaCheck, 0, 0, 1, 2)
        expert_layout.addWidget(QLabel("Trials:"), 1, 0)
        self.optunaTrials = QSpinBox()
        self.optunaTrials.setRange(10, 1000)
        self.optunaTrials.setValue(100)
        self.optunaTrials.setEnabled(False)
        expert_layout.addWidget(self.optunaTrials, 1, 1)
        self.optunaCheck.toggled.connect(self.optunaTrials.setEnabled)

        self.smoteCheck = QCheckBox("SMOTE oversampling")
        self.smoteCheck.setEnabled(bool(self._deps.get("imblearn", False)))
        expert_layout.addWidget(self.smoteCheck, 2, 0, 1, 2)
        expert_layout.addWidget(QLabel("SMOTE k:"), 3, 0)
        self.smoteK = QSpinBox()
        self.smoteK.setRange(1, 20)
        self.smoteK.setValue(5)
        self.smoteK.setEnabled(False)
        expert_layout.addWidget(self.smoteK, 3, 1)
        self.smoteCheck.toggled.connect(self.smoteK.setEnabled)

        self.classWeightCheck = QCheckBox("Use class weights")
        self.classWeightCheck.setEnabled(bool(self._deps.get("sklearn", False)))
        expert_layout.addWidget(self.classWeightCheck, 4, 0, 1, 2)

        self.shapCheck = QCheckBox("Explainability (SHAP)")
        self.shapCheck.setEnabled(bool(self._deps.get("shap", False)))
        expert_layout.addWidget(self.shapCheck, 5, 0, 1, 2)
        expert_layout.addWidget(QLabel("SHAP sample:"), 6, 0)
        self.shapSampleSize = QSpinBox()
        self.shapSampleSize.setRange(100, 50000)
        self.shapSampleSize.setValue(1000)
        self.shapSampleSize.setEnabled(False)
        expert_layout.addWidget(self.shapSampleSize, 6, 1)
        self.shapCheck.toggled.connect(self.shapSampleSize.setEnabled)

        self.nestedCVCheck = QCheckBox("Nested CV")
        expert_layout.addWidget(self.nestedCVCheck, 7, 0, 1, 2)

        self._build_advanced_dialog(model_content, q_content, expert_content)

        # Run bar
        bar_row = QHBoxLayout()
        self.statusLabel = QLabel("")
        bar_row.addWidget(self.statusLabel)
        bar_row.addStretch()

        self.advancedOptionsButton = QPushButton("Advanced options...")
        self.advancedOptionsButton.clicked.connect(self._open_advanced_options)
        bar_row.addWidget(self.advancedOptionsButton)

        self.expertModeButton = QPushButton("Expert mode...")
        self.expertModeButton.clicked.connect(self.openExpertModeRequested)
        bar_row.addWidget(self.expertModeButton)

        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(self._emit_config)
        bar_row.addWidget(self.runButton)
        root.addLayout(bar_row)

        self.setLayout(root)
        self._toggle_expert_controls(True)
        self._on_vector_changed()

    def _build_advanced_dialog(self, model_content, q_content, expert_content):
        self.advancedDialog = QDialog(self)
        self.advancedDialog.setWindowTitle("Guided - Advanced options")
        self.advancedDialog.resize(640, 460)

        dialog_layout = QVBoxLayout(self.advancedDialog)
        dialog_layout.setContentsMargins(8, 8, 8, 8)
        dialog_layout.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(model_content, "Model")
        tabs.addTab(q_content, "Quality & Outputs")
        tabs.addTab(expert_content, "Expert options")
        dialog_layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.advancedDialog.close)
        buttons.accepted.connect(self.advancedDialog.close)
        dialog_layout.addWidget(buttons)

    def _open_advanced_options(self):
        self.advancedDialog.show()
        self.advancedDialog.raise_()
        self.advancedDialog.activateWindow()

    def _toggle_model_mode(self, checked):
        self.modelEdit.setEnabled(checked)
        self.modelBrowse.setEnabled(checked)
        self.vectorEdit.setEnabled(not checked)
        self.vectorBrowse.setEnabled(not checked)
        if self._vector_combo is not None:
            self._vector_combo.setEnabled(not checked)
        self.classFieldCombo.setEnabled(not checked)
        self.fieldStatusLabel.setEnabled(not checked)

    def _toggle_confidence(self, checked):
        self.confidenceEdit.setEnabled(checked)
        self.confidenceBrowse.setEnabled(checked)

    def _toggle_validation(self, checked):
        self.matrixEdit.setEnabled(checked)
        self.matrixBrowse.setEnabled(checked)
        self.splitSpin.setEnabled(checked)

    def _toggle_save_model(self, checked):
        self.saveModelEdit.setEnabled(checked)
        self.saveModelBrowse.setEnabled(checked)

    def _toggle_expert_controls(self, enabled):
        for w in (
            self.optunaCheck,
            self.optunaTrials,
            self.smoteCheck,
            self.smoteK,
            self.classWeightCheck,
            self.shapCheck,
            self.shapSampleSize,
            self.nestedCVCheck,
        ):
            w.setEnabled(enabled and w.isEnabled())
        if enabled:
            self.optunaCheck.setEnabled(bool(self._deps.get("optuna", False)))
            self.smoteCheck.setEnabled(bool(self._deps.get("imblearn", False)))
            self.classWeightCheck.setEnabled(bool(self._deps.get("sklearn", False)))
            self.shapCheck.setEnabled(bool(self._deps.get("shap", False)))
            self.optunaTrials.setEnabled(self.optunaCheck.isChecked())
            self.smoteK.setEnabled(self.smoteCheck.isChecked())
            self.shapSampleSize.setEnabled(self.shapCheck.isChecked())

    def _browse_raster(self):
        path, _f = QFileDialog.getOpenFileName(self, "Select raster", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.rasterEdit.setText(path)

    def _browse_vector(self):
        path, _f = QFileDialog.getOpenFileName(
            self,
            "Select vector",
            "",
            "Shapefile (*.shp);;GeoPackage (*.gpkg);;All (*)",
        )
        if path:
            self.vectorEdit.setText(path)
            self._populate_fields_from_path(path)

    def _browse_model(self):
        path, _f = QFileDialog.getOpenFileName(self, "Select model", "", "Model files (*)")
        if path:
            self.modelEdit.setText(path)

    def _browse_confidence(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save confidence map", "", "GeoTIFF (*.tif)")
        if path:
            self.confidenceEdit.setText(path)

    def _browse_matrix(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save confusion matrix", "", "CSV (*.csv)")
        if path:
            self.matrixEdit.setText(path)

    def _browse_save_model(self):
        path, _f = QFileDialog.getSaveFileName(self, "Save model", "", "Model files (*)")
        if path:
            self.saveModelEdit.setText(path)

    def _code_for_index(self, index):
        if index < 0 or index >= len(self._classifier_meta):
            return "GMM"
        return self._classifier_meta[index][0]

    def _get_raster_path(self):
        if self._raster_combo is not None:
            layer = self._raster_combo.currentLayer()
            if layer is not None:
                return layer.dataProvider().dataSourceUri()
        return self.rasterEdit.text().strip()

    def _get_vector_path(self):
        if self._vector_combo is not None:
            layer = self._vector_combo.currentLayer()
            if layer is not None:
                return layer.dataProvider().dataSourceUri().split("|")[0]
        return self.vectorEdit.text().strip()

    def _icon_asset_path(self, icon_path):
        if icon_path.startswith(":/"):
            return icon_path
        return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "img", icon_path))

    def _icon_label(self, icon_path, tooltip, fallback_resource=None):
        icon_label = QLabel()
        icon_label.setFixedSize(15, 15)
        icon_label.setToolTip(tooltip)
        pix = QPixmap(self._icon_asset_path(icon_path))
        if pix.isNull() and fallback_resource:
            pix = QPixmap(fallback_resource)
        icon_label.setPixmap(pix)
        icon_label.setScaledContents(True)
        return icon_label

    def _on_vector_changed(self):
        self.classFieldCombo.clear()
        self.fieldStatusLabel.setText("")
        self.fieldStatusLabel.setVisible(False)
        if self._vector_combo is None:
            self._on_vector_path_edited()
            return
        layer = self._vector_combo.currentLayer()
        if layer is None:
            self.fieldStatusLabel.setText("Select a vector layer to list its fields.")
            self.fieldStatusLabel.setVisible(True)
            self._update_status()
            return
        try:
            fields = layer.dataProvider().fields()
            names = [fields.at(i).name() for i in range(fields.count())]
            if names:
                self.classFieldCombo.addItems(names)
            else:
                self.fieldStatusLabel.setText("No fields found in selected layer.")
                self.fieldStatusLabel.setVisible(True)
        except (AttributeError, TypeError):
            self.fieldStatusLabel.setText("Unable to read fields from selected layer.")
            self.fieldStatusLabel.setVisible(True)
        self._update_status()

    def _on_vector_path_edited(self):
        if self._vector_combo is not None:
            self._on_vector_changed()
            return
        path = self.vectorEdit.text().strip()
        if not path:
            self.classFieldCombo.clear()
            self.fieldStatusLabel.setText("")
            self.fieldStatusLabel.setVisible(False)
            self._update_status()
            return
        self._populate_fields_from_path(path)

    def _populate_fields_from_path(self, path):
        self.classFieldCombo.clear()
        self.fieldStatusLabel.setText("")
        self.fieldStatusLabel.setVisible(False)
        dataset_path = path.split("|")[0].strip()
        if not os.path.exists(dataset_path):
            self.fieldStatusLabel.setText("Vector path does not exist.")
            self.fieldStatusLabel.setVisible(True)
            self._update_status()
            return
        try:
            from osgeo import ogr
        except ImportError:
            try:
                import ogr  # type: ignore[no-redef]
            except ImportError:
                self.fieldStatusLabel.setText("OGR unavailable; cannot list fields.")
                self.fieldStatusLabel.setVisible(True)
                self._update_status()
                return
        ds = ogr.Open(dataset_path)
        if ds is None:
            self.fieldStatusLabel.setText("Unable to open vector dataset.")
            self.fieldStatusLabel.setVisible(True)
            self._update_status()
            return
        layer = ds.GetLayer()
        if layer is None:
            self.fieldStatusLabel.setText("No layer found in dataset.")
            self.fieldStatusLabel.setVisible(True)
            self._update_status()
            return
        dfn = layer.GetLayerDefn()
        count = dfn.GetFieldCount()
        if count == 0:
            self.fieldStatusLabel.setText("No fields found in vector dataset.")
            self.fieldStatusLabel.setVisible(True)
            self._update_status()
            return
        for i in range(count):
            self.classFieldCombo.addItem(dfn.GetFieldDefn(i).GetName())
        self._update_status()

    def _set_classifier_by_code(self, code):
        for i, meta in enumerate(self._classifier_meta):
            if meta[0] == code:
                self.classifierCombo.setCurrentIndex(i)
                return

    def _classifier_available(self, code):
        for c, _name, needs_sk, needs_xgb, needs_lgb, needs_cb in self._classifier_meta:
            if c != code:
                continue
            if needs_sk and not self._deps.get("sklearn", False):
                return False
            if needs_xgb and not self._deps.get("xgboost", False):
                return False
            if needs_lgb and not self._deps.get("lightgbm", False):
                return False
            if needs_cb and not self._deps.get("catboost", False):
                return False
            return True
        return False

    def _missing_required_deps(self, code):
        missing = []
        for c, _name, needs_sk, needs_xgb, needs_lgb, needs_cb in self._classifier_meta:
            if c != code:
                continue
            if needs_sk and not self._deps.get("sklearn", False):
                missing.append("scikit-learn")
            if needs_xgb and not self._deps.get("xgboost", False):
                missing.append("xgboost")
            if needs_lgb and not self._deps.get("lightgbm", False):
                missing.append("lightgbm")
            if needs_cb and not self._deps.get("catboost", False):
                missing.append("catboost")
        return missing

    def _first_available(self, preferred_codes):
        for code in preferred_codes:
            if self._classifier_available(code):
                return code
        return "GMM"

    def _apply_goal_defaults(self, goal):
        if self._setting_goal:
            return
        self._setting_goal = True

        self.optunaCheck.setChecked(False)
        self.smoteCheck.setChecked(False)
        self.classWeightCheck.setChecked(False)
        self.shapCheck.setChecked(False)
        self.nestedCVCheck.setChecked(False)
        self.confidenceCheck.setChecked(False)
        self.validationCheck.setChecked(False)

        if goal == "Fast map now":
            self._set_classifier_by_code(self._first_available(["GMM", "RF"]))
        elif goal == "Best accuracy":
            self._set_classifier_by_code(self._first_available(["CB", "XGB", "LGB", "RF", "ET", "GMM"]))
            if self._deps.get("optuna", False):
                self.optunaCheck.setChecked(True)
            if self._deps.get("sklearn", False):
                self.classWeightCheck.setChecked(True)
            self.validationCheck.setChecked(True)
        elif goal == "Explain results":
            self._set_classifier_by_code(self._first_available(["CB", "RF", "ET", "GBC", "GMM"]))
            if self._deps.get("shap", False):
                self.shapCheck.setChecked(True)
            self.confidenceCheck.setChecked(True)
        elif goal == "Imbalanced classes":
            self._set_classifier_by_code(self._first_available(["GMM", "RF", "ET"]))
            if self._deps.get("imblearn", False):
                self.smoteCheck.setChecked(True)
            if self._deps.get("sklearn", False):
                self.classWeightCheck.setChecked(True)
            self.validationCheck.setChecked(True)

        self._setting_goal = False
        self._update_status()

    def _current_classifier_code(self):
        return self._code_for_index(self.classifierCombo.currentIndex())

    def _update_status(self):
        issues = []
        if not self._get_raster_path():
            issues.append("raster")

        if self.loadModelCheck.isChecked():
            if not self.modelEdit.text().strip():
                issues.append("model")
        else:
            if not self._get_vector_path():
                issues.append("vector")
            if not self.classFieldCombo.currentText().strip():
                issues.append("label")

        missing_deps = self._missing_required_deps(self._current_classifier_code())
        if missing_deps:
            issues.append("deps")

        if not issues:
            self.statusLabel.setText("Ready")
            return

        if "deps" in issues:
            self.statusLabel.setText("Missing dependencies")
            return

        self.statusLabel.setText("Missing: " + ", ".join([i for i in issues if i != "deps"]))

    def _validate_before_run(self):
        if not self._get_raster_path():
            QMessageBox.warning(self, "Missing input", "Please select a raster.")
            return False

        if self.loadModelCheck.isChecked():
            if not self.modelEdit.text().strip():
                QMessageBox.warning(self, "Missing model", "Please provide an existing model path.")
                return False
        else:
            if not self._get_vector_path() or not self.classFieldCombo.currentText().strip():
                QMessageBox.warning(
                    self,
                    "Missing training data",
                    "Please provide vector data and label field, or switch to existing model mode.",
                )
                return False

        missing_deps = self._missing_required_deps(self._current_classifier_code())
        if missing_deps:
            QMessageBox.warning(
                self,
                "Dependencies missing",
                "Selected classifier is unavailable. Missing: " + ", ".join(missing_deps),
            )
            return False

        return True

    def _emit_config(self):
        if not self._validate_before_run():
            return

        confidence_map = ""
        if self.confidenceCheck.isChecked():
            confidence_map = self.confidenceEdit.text().strip() or tempfile.mktemp(".tif")

        matrix_path = ""
        if self.validationCheck.isChecked():
            matrix_path = self.matrixEdit.text().strip()

        save_model = ""
        if self.saveModelCheck.isChecked():
            save_model = self.saveModelEdit.text().strip()

        config = {
            "raster": self._get_raster_path(),
            "vector": "" if self.loadModelCheck.isChecked() else self._get_vector_path(),
            "class_field": "" if self.loadModelCheck.isChecked() else self.classFieldCombo.currentText().strip(),
            "load_model": self.modelEdit.text().strip() if self.loadModelCheck.isChecked() else "",
            "classifier": self._current_classifier_code(),
            "extraParam": {
                "USE_OPTUNA": self.optunaCheck.isChecked(),
                "OPTUNA_TRIALS": self.optunaTrials.value(),
                "COMPUTE_SHAP": self.shapCheck.isChecked(),
                "SHAP_OUTPUT": "",
                "SHAP_SAMPLE_SIZE": self.shapSampleSize.value(),
                "USE_SMOTE": self.smoteCheck.isChecked(),
                "SMOTE_K_NEIGHBORS": self.smoteK.value(),
                "USE_CLASS_WEIGHTS": self.classWeightCheck.isChecked(),
                "CLASS_WEIGHT_STRATEGY": "balanced",
                "CUSTOM_CLASS_WEIGHTS": {},
                "USE_NESTED_CV": self.nestedCVCheck.isChecked(),
                "NESTED_INNER_CV": 3,
                "NESTED_OUTER_CV": 5,
            },
            "output_raster": "",
            "confidence_map": confidence_map,
            "save_model": save_model,
            "confusion_matrix": matrix_path,
            "split_percent": self.splitSpin.value() if self.validationCheck.isChecked() else 100,
        }

        self.classificationRequested.emit(config)
