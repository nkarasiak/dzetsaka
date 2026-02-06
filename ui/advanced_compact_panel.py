import os
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class AdvancedCompactPanel(QWidget):
    """Compact intent-driven advanced panel for small QGIS dock usage."""

    classificationRequested = pyqtSignal(dict)
    openFullWizardRequested = pyqtSignal()

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
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Goal
        goal_row = QHBoxLayout()
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
        root.addLayout(goal_row)

        # Core data
        data_group = QGroupBox("Core Inputs")
        data_layout = QGridLayout()

        data_layout.addWidget(QLabel("Raster:"), 0, 0)
        self.rasterEdit = QLineEdit()
        self.rasterEdit.setPlaceholderText("Path to raster file...")
        self.rasterEdit.textChanged.connect(self._update_status)
        data_layout.addWidget(self.rasterEdit, 0, 1)
        self.rasterBrowse = QPushButton("Browse...")
        self.rasterBrowse.clicked.connect(self._browse_raster)
        data_layout.addWidget(self.rasterBrowse, 0, 2)

        self.loadModelCheck = QCheckBox("Use existing model")
        self.loadModelCheck.toggled.connect(self._toggle_model_mode)
        self.loadModelCheck.toggled.connect(self._update_status)
        data_layout.addWidget(self.loadModelCheck, 1, 0, 1, 3)

        data_layout.addWidget(QLabel("Model:"), 2, 0)
        self.modelEdit = QLineEdit()
        self.modelEdit.setPlaceholderText("Path to trained model...")
        self.modelEdit.setEnabled(False)
        self.modelEdit.textChanged.connect(self._update_status)
        data_layout.addWidget(self.modelEdit, 2, 1)
        self.modelBrowse = QPushButton("Browse...")
        self.modelBrowse.setEnabled(False)
        self.modelBrowse.clicked.connect(self._browse_model)
        data_layout.addWidget(self.modelBrowse, 2, 2)

        data_layout.addWidget(QLabel("Vector:"), 3, 0)
        self.vectorEdit = QLineEdit()
        self.vectorEdit.setPlaceholderText("Path to training vector...")
        self.vectorEdit.textChanged.connect(self._update_status)
        data_layout.addWidget(self.vectorEdit, 3, 1)
        self.vectorBrowse = QPushButton("Browse...")
        self.vectorBrowse.clicked.connect(self._browse_vector)
        data_layout.addWidget(self.vectorBrowse, 3, 2)

        data_layout.addWidget(QLabel("Label field:"), 4, 0)
        self.classFieldEdit = QLineEdit()
        self.classFieldEdit.setPlaceholderText("Class attribute name...")
        self.classFieldEdit.textChanged.connect(self._update_status)
        data_layout.addWidget(self.classFieldEdit, 4, 1, 1, 2)

        data_layout.addWidget(QLabel("Classifier:"), 5, 0)
        self.classifierCombo = QComboBox()
        for _code, name, _sk, _xgb, _lgb, _cb in self._classifier_meta:
            self.classifierCombo.addItem(name)
        self.classifierCombo.currentIndexChanged.connect(self._update_status)
        data_layout.addWidget(self.classifierCombo, 5, 1, 1, 2)

        data_group.setLayout(data_layout)
        root.addWidget(data_group)

        # Quality and outputs
        q_group = QGroupBox("Quality and Outputs")
        q_layout = QGridLayout()

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

        q_group.setLayout(q_layout)
        root.addWidget(q_group)

        # Expert options
        expert_group = QGroupBox("Expert options")
        expert_group.setCheckable(True)
        expert_group.setChecked(False)
        expert_layout = QGridLayout()

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

        expert_group.toggled.connect(self._toggle_expert_controls)
        expert_group.setLayout(expert_layout)
        root.addWidget(expert_group)

        # Run bar
        bar_row = QHBoxLayout()
        self.statusLabel = QLabel("")
        bar_row.addWidget(self.statusLabel)
        bar_row.addStretch()

        self.fullWizardButton = QPushButton("Full wizard...")
        self.fullWizardButton.clicked.connect(self.openFullWizardRequested)
        bar_row.addWidget(self.fullWizardButton)

        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(self._emit_config)
        bar_row.addWidget(self.runButton)
        root.addLayout(bar_row)

        self.setLayout(root)

    def _toggle_model_mode(self, checked):
        self.modelEdit.setEnabled(checked)
        self.modelBrowse.setEnabled(checked)
        self.vectorEdit.setEnabled(not checked)
        self.vectorBrowse.setEnabled(not checked)
        self.classFieldEdit.setEnabled(not checked)

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
            self._set_classifier_by_code(self._first_available(["RF", "ET", "XGB", "LGB", "CB", "GMM"]))
            if self._deps.get("optuna", False):
                self.optunaCheck.setChecked(True)
            if self._deps.get("sklearn", False):
                self.classWeightCheck.setChecked(True)
            self.validationCheck.setChecked(True)
        elif goal == "Explain results":
            self._set_classifier_by_code(self._first_available(["RF", "ET", "GBC", "GMM"]))
            if self._deps.get("shap", False):
                self.shapCheck.setChecked(True)
            self.confidenceCheck.setChecked(True)
        elif goal == "Imbalanced classes":
            self._set_classifier_by_code(self._first_available(["RF", "ET", "GMM"]))
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
        if not self.rasterEdit.text().strip():
            issues.append("raster")

        if self.loadModelCheck.isChecked():
            if not self.modelEdit.text().strip():
                issues.append("model")
        else:
            if not self.vectorEdit.text().strip():
                issues.append("vector")
            if not self.classFieldEdit.text().strip():
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
        if not self.rasterEdit.text().strip():
            QMessageBox.warning(self, "Missing input", "Please select a raster.")
            return False

        if self.loadModelCheck.isChecked():
            if not self.modelEdit.text().strip():
                QMessageBox.warning(self, "Missing model", "Please provide an existing model path.")
                return False
        else:
            if not self.vectorEdit.text().strip() or not self.classFieldEdit.text().strip():
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
            matrix_path = self.matrixEdit.text().strip() or tempfile.mktemp(".csv")

        save_model = ""
        if self.saveModelCheck.isChecked():
            save_model = self.saveModelEdit.text().strip()

        config = {
            "raster": self.rasterEdit.text().strip(),
            "vector": "" if self.loadModelCheck.isChecked() else self.vectorEdit.text().strip(),
            "class_field": "" if self.loadModelCheck.isChecked() else self.classFieldEdit.text().strip(),
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
