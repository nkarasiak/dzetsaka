"""Minimal working example of validated widgets.

This is a standalone script that demonstrates the validated widgets
without requiring the full dzetsaka plugin context.

Run with: python examples/validated_widgets_minimal.py

Note: Requires QGIS Python environment with qgis.PyQt available.
"""

import sys

from qgis.PyQt.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# Import validated widgets
from ui.validated_widgets import ValidatedDoubleSpinBox, ValidatedLineEdit, ValidatedSpinBox


class MinimalDemo(QWidget):
    """Minimal demonstration of validated widgets."""

    def __init__(self):
        """Initialize the demo window."""
        super().__init__()
        self.setWindowTitle("Validated Widgets Demo")
        self.setMinimumWidth(500)
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Validated Widgets Demo</h2>")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Try changing the values to see validation feedback:\n"
            "• Green/default border = valid\n"
            "• Orange border = warning (valid but high)\n"
            "• Red border = invalid\n"
            "• Hover to see tooltip with details",
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # ValidatedSpinBox examples
        spinbox_group = self._create_spinbox_examples()
        layout.addWidget(spinbox_group)

        # ValidatedDoubleSpinBox example
        double_spinbox_group = self._create_double_spinbox_example()
        layout.addWidget(double_spinbox_group)

        # ValidatedLineEdit example
        lineedit_group = self._create_lineedit_example()
        layout.addWidget(lineedit_group)

        layout.addStretch()

    def _create_spinbox_examples(self):
        """Create ValidatedSpinBox examples.

        Returns:
            QGroupBox: Group containing spinbox examples

        """
        group = QGroupBox("ValidatedSpinBox Examples")
        layout = QGridLayout(group)

        # Example 1: With warning and time estimate
        layout.addWidget(QLabel("<b>With warning & time estimate:</b>"), 0, 0, 1, 2)
        layout.addWidget(QLabel("Optuna trials:"), 1, 0)

        trials = ValidatedSpinBox(
            validator_fn=lambda v: 10 <= v <= 2000,
            warning_threshold=500,
            time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min",
        )
        trials.setRange(10, 2000)
        trials.setValue(100)
        trials.setToolTip("Number of Optuna trials (warning at 500+)")
        layout.addWidget(trials, 1, 1)

        # Example 2: Only warning, no time estimate
        layout.addWidget(QLabel("<b>With warning only:</b>"), 2, 0, 1, 2)
        layout.addWidget(QLabel("CV folds:"), 3, 0)

        folds = ValidatedSpinBox(
            validator_fn=lambda v: 2 <= v <= 10,
            warning_threshold=7,
        )
        folds.setRange(2, 10)
        folds.setValue(5)
        folds.setToolTip("Number of cross-validation folds (warning at 7+)")
        layout.addWidget(folds, 3, 1)

        # Example 3: Custom validation (even numbers only)
        layout.addWidget(QLabel("<b>Custom validation (even only):</b>"), 4, 0, 1, 2)
        layout.addWidget(QLabel("Even number:"), 5, 0)

        even = ValidatedSpinBox(
            validator_fn=lambda v: v % 2 == 0,
        )
        even.setRange(0, 100)
        even.setValue(10)
        even.setToolTip("Only even numbers allowed")
        layout.addWidget(even, 5, 1)

        return group

    def _create_double_spinbox_example(self):
        """Create ValidatedDoubleSpinBox example.

        Returns:
            QGroupBox: Group containing double spinbox example

        """
        group = QGroupBox("ValidatedDoubleSpinBox Example")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Learning rate:"), 0, 0)

        learning_rate = ValidatedDoubleSpinBox(
            validator_fn=lambda v: 0.0001 <= v <= 1.0,
            warning_threshold=0.5,
        )
        learning_rate.setRange(0.0001, 1.0)
        learning_rate.setDecimals(4)
        learning_rate.setValue(0.01)
        learning_rate.setToolTip("Learning rate (warning at 0.5+)")
        layout.addWidget(learning_rate, 0, 1)

        help_text = QLabel("Try: 0.01 (valid), 0.8 (warning), or manually type 2.0 then clear to see invalid state")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(help_text, 1, 0, 1, 2)

        return group

    def _create_lineedit_example(self):
        """Create ValidatedLineEdit example.

        Returns:
            QGroupBox: Group containing line edit example

        """
        group = QGroupBox("ValidatedLineEdit Example")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Username:"), 0, 0)

        username = ValidatedLineEdit(
            validator_fn=lambda text: len(text) >= 3,
            warning_fn=lambda text: "Username contains spaces" if " " in text else None,
        )
        username.setText("validuser")
        username.setToolTip("Username (min 3 characters)")
        layout.addWidget(username, 0, 1)

        help_text = QLabel("Try: 'abc' (valid), 'user name' (warning), 'ab' (invalid)")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(help_text, 1, 0, 1, 2)

        return group


def main():
    """Run the demo application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    demo = MinimalDemo()
    demo.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
