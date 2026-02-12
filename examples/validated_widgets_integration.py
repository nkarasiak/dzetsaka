"""Example integration of validated widgets into dzetsaka GUI.

This example demonstrates how to replace existing QSpinBox widgets
with ValidatedSpinBox widgets in the guided workflow to provide
real-time validation feedback.

This is a reference implementation that can be adapted into
ui/classification_workflow_ui.py.
"""

from qgis.PyQt.QtWidgets import QCheckBox, QGridLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from ui.validated_widgets import ValidatedSpinBox


class OptimizationMethodsWidget(QWidget):
    """Example widget showing validated spinboxes for optimization methods.

    This demonstrates how to integrate ValidatedSpinBox into the
    existing dzetsaka UI patterns for the optimization panel.
    """

    def __init__(self, parent=None):
        """Initialize the optimization methods widget.

        Args:
            parent: Parent widget (optional)

        """
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Optuna optimization section
        optuna_group = self._create_optuna_section()
        layout.addWidget(optuna_group)

        # SHAP explainability section
        shap_group = self._create_shap_section()
        layout.addWidget(shap_group)

        # SMOTE class balancing section
        smote_group = self._create_smote_section()
        layout.addWidget(smote_group)

        # Nested CV section
        nested_cv_group = self._create_nested_cv_section()
        layout.addWidget(nested_cv_group)

        layout.addStretch()

    def _create_optuna_section(self):
        """Create Optuna optimization section with validated spinbox.

        Returns:
            QGroupBox: Group box containing Optuna controls

        """
        group = QGroupBox("Hyperparameter Optimization")
        layout = QGridLayout(group)

        # Enable/disable checkbox
        self.optunaCheck = QCheckBox("Enable Optuna optimization")
        layout.addWidget(self.optunaCheck, 0, 0, 1, 2)

        # Trials spinbox with validation
        trials_label = QLabel("Number of trials:")
        self.optunaTrials = ValidatedSpinBox(
            validator_fn=lambda v: 10 <= v <= 2000,
            warning_threshold=500,
            time_estimator_fn=self._estimate_optuna_time,
        )
        self.optunaTrials.setRange(10, 2000)
        self.optunaTrials.setValue(100)
        self.optunaTrials.setSingleStep(10)
        self.optunaTrials.setEnabled(False)
        self.optunaTrials.setToolTip(
            "Number of Optuna trials to run. Higher values find better hyperparameters but take longer to compute.",
        )

        layout.addWidget(trials_label, 1, 0)
        layout.addWidget(self.optunaTrials, 1, 1)

        # Connect checkbox to enable/disable spinbox
        self.optunaCheck.toggled.connect(self.optunaTrials.setEnabled)

        return group

    def _create_shap_section(self):
        """Create SHAP explainability section with validated spinbox.

        Returns:
            QGroupBox: Group box containing SHAP controls

        """
        group = QGroupBox("Model Explainability")
        layout = QGridLayout(group)

        # Enable/disable checkbox
        self.shapCheck = QCheckBox("Enable SHAP explainability")
        layout.addWidget(self.shapCheck, 0, 0, 1, 2)

        # Sample size spinbox with validation
        sample_label = QLabel("Sample size:")
        self.shapSampleSize = ValidatedSpinBox(
            validator_fn=lambda v: 100 <= v <= 50000,
            warning_threshold=10000,
            time_estimator_fn=self._estimate_shap_time,
        )
        self.shapSampleSize.setRange(100, 50000)
        self.shapSampleSize.setValue(1000)
        self.shapSampleSize.setSingleStep(100)
        self.shapSampleSize.setEnabled(False)
        self.shapSampleSize.setToolTip(
            "Number of samples to use for SHAP value computation. "
            "Larger samples provide more accurate explanations but take longer.",
        )

        layout.addWidget(sample_label, 1, 0)
        layout.addWidget(self.shapSampleSize, 1, 1)

        # Connect checkbox to enable/disable spinbox
        self.shapCheck.toggled.connect(self.shapSampleSize.setEnabled)

        return group

    def _create_smote_section(self):
        """Create SMOTE class balancing section with validated spinbox.

        Returns:
            QGroupBox: Group box containing SMOTE controls

        """
        group = QGroupBox("Class Balancing")
        layout = QGridLayout(group)

        # Enable/disable checkbox
        self.smoteCheck = QCheckBox("Enable SMOTE oversampling")
        layout.addWidget(self.smoteCheck, 0, 0, 1, 2)

        # k-neighbors spinbox with validation
        k_label = QLabel("k-neighbors:")
        self.smoteK = ValidatedSpinBox(
            validator_fn=lambda v: 1 <= v <= 20,
            warning_threshold=15,  # High k might smooth too much
        )
        self.smoteK.setRange(1, 20)
        self.smoteK.setValue(5)
        self.smoteK.setEnabled(False)
        self.smoteK.setToolTip(
            "Number of nearest neighbors for SMOTE. Lower values create more "
            "distinct synthetic samples; higher values create smoother distributions.",
        )

        layout.addWidget(k_label, 1, 0)
        layout.addWidget(self.smoteK, 1, 1)

        # Connect checkbox to enable/disable spinbox
        self.smoteCheck.toggled.connect(self.smoteK.setEnabled)

        return group

    def _create_nested_cv_section(self):
        """Create nested cross-validation section with validated spinboxes.

        Returns:
            QGroupBox: Group box containing nested CV controls

        """
        group = QGroupBox("Nested Cross-Validation")
        layout = QGridLayout(group)

        # Enable/disable checkbox
        self.nestedCVCheck = QCheckBox("Enable nested cross-validation")
        layout.addWidget(self.nestedCVCheck, 0, 0, 1, 2)

        # Inner folds spinbox with validation
        inner_label = QLabel("Inner folds:")
        self.innerFolds = ValidatedSpinBox(
            validator_fn=lambda v: 2 <= v <= 10,
            warning_threshold=7,  # Many folds = long runtime
        )
        self.innerFolds.setRange(2, 10)
        self.innerFolds.setValue(3)
        self.innerFolds.setEnabled(False)
        self.innerFolds.setToolTip("Number of inner cross-validation folds for hyperparameter tuning.")

        layout.addWidget(inner_label, 1, 0)
        layout.addWidget(self.innerFolds, 1, 1)

        # Outer folds spinbox with validation
        outer_label = QLabel("Outer folds:")
        self.outerFolds = ValidatedSpinBox(
            validator_fn=lambda v: 2 <= v <= 10,
            warning_threshold=7,
        )
        self.outerFolds.setRange(2, 10)
        self.outerFolds.setValue(5)
        self.outerFolds.setEnabled(False)
        self.outerFolds.setToolTip("Number of outer cross-validation folds for performance estimation.")

        layout.addWidget(outer_label, 2, 0)
        layout.addWidget(self.outerFolds, 2, 1)

        # Connect checkbox to enable/disable spinboxes
        self.nestedCVCheck.toggled.connect(self.innerFolds.setEnabled)
        self.nestedCVCheck.toggled.connect(self.outerFolds.setEnabled)

        return group

    @staticmethod
    def _estimate_optuna_time(trials):
        """Estimate Optuna optimization time based on number of trials.

        Args:
            trials: Number of Optuna trials

        Returns:
            str: Time estimate string

        """
        # Rough estimates: 0.1-0.3 minutes per trial on average
        min_time = trials * 0.1
        max_time = trials * 0.3

        if max_time < 1:
            return f"{min_time * 60:.0f}-{max_time * 60:.0f} sec"
        if max_time < 60:
            return f"{min_time:.0f}-{max_time:.0f} min"
        return f"{min_time / 60:.1f}-{max_time / 60:.1f} hr"

    @staticmethod
    def _estimate_shap_time(samples):
        """Estimate SHAP computation time based on sample size.

        Args:
            samples: Number of samples for SHAP

        Returns:
            str: Time estimate string

        """
        # Rough estimates: 0.01-0.05 seconds per sample
        min_time = samples * 0.01
        max_time = samples * 0.05

        if max_time < 60:
            return f"{min_time:.0f}-{max_time:.0f} sec"
        return f"{min_time / 60:.1f}-{max_time / 60:.1f} min"


# Example usage in QuickClassificationPanel
class QuickClassificationPanelExample(QWidget):
    """Example showing integration into QuickClassificationPanel.

    This demonstrates the minimal changes needed to integrate
    validated widgets into the existing dashboard panel.
    """

    def __init__(self, parent=None):
        """Initialize the example panel.

        Args:
            parent: Parent widget (optional)

        """
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Add the optimization methods widget
        self.optimizationWidget = OptimizationMethodsWidget()
        layout.addWidget(self.optimizationWidget)


# Example: Direct replacement in existing code
def example_replacement():
    """Show before/after for direct replacement.

    This is a conceptual example showing the minimal diff needed
    to upgrade existing code.
    """
    # BEFORE (original code in classification_workflow_ui.py):
    # -------------------------------------------------
    # from qgis.PyQt.QtWidgets import QSpinBox
    #
    # self.optunaTrials = QSpinBox()
    # self.optunaTrials.setRange(10, 2000)
    # self.optunaTrials.setValue(100)
    # self.optunaTrials.setSingleStep(10)

    # AFTER (with validated widget):
    # -------------------------------------------------
    # from ui.validated_widgets import ValidatedSpinBox
    #
    # self.optunaTrials = ValidatedSpinBox(
    #     validator_fn=lambda v: 10 <= v <= 2000,
    #     warning_threshold=500,
    #     time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
    # )
    # self.optunaTrials.setRange(10, 2000)
    # self.optunaTrials.setValue(100)
    # self.optunaTrials.setSingleStep(10)
    # self.optunaTrials.setToolTip("Number of Optuna optimization trials")
