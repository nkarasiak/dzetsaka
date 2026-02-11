"""Custom widget classes with real-time validation and visual feedback.

This module provides enhanced Qt widgets that perform real-time validation
with color-coded visual feedback and dynamic tooltips. These widgets integrate
seamlessly with the existing dzetsaka UI while providing better user experience
through immediate validation feedback.

Features:
    - Color-coded borders: red (invalid), orange (warning), default (valid)
    - Real-time validation on value changes
    - Dynamic tooltip updates with warnings and time estimates
    - Compatible with QGIS PyQt5/PyQt6 abstraction layer

Author:
    Nicolas Karasiak
"""

from typing import Callable, Optional

from qgis.PyQt.QtWidgets import QDoubleSpinBox, QLineEdit, QSpinBox


class ValidatedSpinBox(QSpinBox):
    """QSpinBox with real-time validation and visual feedback.

    Provides color-coded borders and dynamic tooltips based on validation
    rules. Useful for preventing user errors before form submission.

    Args:
        parent: Parent widget (optional)
        validator_fn: Function that takes value and returns True if valid
        warning_threshold: Value above which to show warning (optional)
        time_estimator_fn: Function that takes value and returns time estimate string (optional)

    Example:
        >>> optunaTrials = ValidatedSpinBox(
        ...     validator_fn=lambda v: 10 <= v <= 2000,
        ...     warning_threshold=500,
        ...     time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min",
        ... )
        >>> optunaTrials.setRange(10, 2000)
        >>> optunaTrials.setValue(100)
    """

    def __init__(
        self,
        parent=None,
        validator_fn: Optional[Callable[[int], bool]] = None,
        warning_threshold: Optional[int] = None,
        time_estimator_fn: Optional[Callable[[int], str]] = None,
    ):
        """Initialize the validated spin box.

        Args:
            parent: Parent widget (optional)
            validator_fn: Function that takes value and returns True if valid
            warning_threshold: Value above which to show warning (optional)
            time_estimator_fn: Function that takes value and returns time estimate string (optional)
        """
        super().__init__(parent)
        self._validator_fn = validator_fn
        self._warning_threshold = warning_threshold
        self._time_estimator_fn = time_estimator_fn
        self._original_tooltip = ""

        # Store original stylesheet to restore when valid
        self._base_stylesheet = self.styleSheet()

        # Connect to value change signal for real-time validation
        self.valueChanged.connect(self._on_value_changed)

    def setToolTip(self, tooltip: str) -> None:
        """Override setToolTip to store original tooltip.

        Args:
            tooltip: The tooltip text to set
        """
        self._original_tooltip = tooltip
        super().setToolTip(tooltip)

    def _on_value_changed(self, value: int) -> None:
        """Handle value changes and update validation state.

        Args:
            value: The new spinbox value
        """
        self._update_validation_state(value)

    def _update_validation_state(self, value: int) -> None:
        """Update visual feedback based on validation rules.

        Args:
            value: The current value to validate
        """
        tooltip_parts = []
        if self._original_tooltip:
            tooltip_parts.append(self._original_tooltip)

        # Check if value is valid according to custom validator
        is_valid = True
        if self._validator_fn is not None:
            is_valid = self._validator_fn(value)

        if not is_valid:
            # Invalid: red border
            self.setStyleSheet(self._base_stylesheet + "\n" + "QSpinBox { border: 2px solid #e74c3c; }")
            tooltip_parts.append("⚠️ Invalid value")
        elif self._warning_threshold is not None and value >= self._warning_threshold:
            # Warning: orange border
            self.setStyleSheet(self._base_stylesheet + "\n" + "QSpinBox { border: 2px solid #f39c12; }")
            tooltip_parts.append(f"⚠️ High value (≥{self._warning_threshold})")

            # Add time estimate if available
            if self._time_estimator_fn is not None:
                time_estimate = self._time_estimator_fn(value)
                tooltip_parts.append(f"⏱️ Estimated time: {time_estimate}")
        else:
            # Valid: restore default appearance
            self.setStyleSheet(self._base_stylesheet)

            # Still show time estimate if available, even for valid values
            if self._time_estimator_fn is not None:
                time_estimate = self._time_estimator_fn(value)
                tooltip_parts.append(f"⏱️ Estimated time: {time_estimate}")

        # Update tooltip with all relevant information
        super().setToolTip("\n".join(tooltip_parts))


class ValidatedDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with real-time validation and visual feedback.

    Provides color-coded borders and dynamic tooltips based on validation
    rules. Useful for floating-point parameters that need validation.

    Args:
        parent: Parent widget (optional)
        validator_fn: Function that takes value and returns True if valid
        warning_threshold: Value above which to show warning (optional)
        time_estimator_fn: Function that takes value and returns time estimate string (optional)

    Example:
        >>> learningRate = ValidatedDoubleSpinBox(
        ...     validator_fn=lambda v: 0.0001 <= v <= 1.0,
        ...     warning_threshold=0.5,
        ... )
        >>> learningRate.setRange(0.0001, 1.0)
        >>> learningRate.setValue(0.01)
    """

    def __init__(
        self,
        parent=None,
        validator_fn: Optional[Callable[[float], bool]] = None,
        warning_threshold: Optional[float] = None,
        time_estimator_fn: Optional[Callable[[float], str]] = None,
    ):
        """Initialize the validated double spin box.

        Args:
            parent: Parent widget (optional)
            validator_fn: Function that takes value and returns True if valid
            warning_threshold: Value above which to show warning (optional)
            time_estimator_fn: Function that takes value and returns time estimate string (optional)
        """
        super().__init__(parent)
        self._validator_fn = validator_fn
        self._warning_threshold = warning_threshold
        self._time_estimator_fn = time_estimator_fn
        self._original_tooltip = ""

        # Store original stylesheet to restore when valid
        self._base_stylesheet = self.styleSheet()

        # Connect to value change signal for real-time validation
        self.valueChanged.connect(self._on_value_changed)

    def setToolTip(self, tooltip: str) -> None:
        """Override setToolTip to store original tooltip.

        Args:
            tooltip: The tooltip text to set
        """
        self._original_tooltip = tooltip
        super().setToolTip(tooltip)

    def _on_value_changed(self, value: float) -> None:
        """Handle value changes and update validation state.

        Args:
            value: The new spinbox value
        """
        self._update_validation_state(value)

    def _update_validation_state(self, value: float) -> None:
        """Update visual feedback based on validation rules.

        Args:
            value: The current value to validate
        """
        tooltip_parts = []
        if self._original_tooltip:
            tooltip_parts.append(self._original_tooltip)

        # Check if value is valid according to custom validator
        is_valid = True
        if self._validator_fn is not None:
            is_valid = self._validator_fn(value)

        if not is_valid:
            # Invalid: red border
            self.setStyleSheet(self._base_stylesheet + "\n" + "QDoubleSpinBox { border: 2px solid #e74c3c; }")
            tooltip_parts.append("⚠️ Invalid value")
        elif self._warning_threshold is not None and value >= self._warning_threshold:
            # Warning: orange border
            self.setStyleSheet(self._base_stylesheet + "\n" + "QDoubleSpinBox { border: 2px solid #f39c12; }")
            tooltip_parts.append(f"⚠️ High value (≥{self._warning_threshold})")

            # Add time estimate if available
            if self._time_estimator_fn is not None:
                time_estimate = self._time_estimator_fn(value)
                tooltip_parts.append(f"⏱️ Estimated time: {time_estimate}")
        else:
            # Valid: restore default appearance
            self.setStyleSheet(self._base_stylesheet)

            # Still show time estimate if available, even for valid values
            if self._time_estimator_fn is not None:
                time_estimate = self._time_estimator_fn(value)
                tooltip_parts.append(f"⏱️ Estimated time: {time_estimate}")

        # Update tooltip with all relevant information
        super().setToolTip("\n".join(tooltip_parts))


class ValidatedLineEdit(QLineEdit):
    """QLineEdit with real-time validation and visual feedback.

    Provides color-coded borders and dynamic tooltips based on custom
    validation rules. Useful for text inputs that need format validation.

    Args:
        parent: Parent widget (optional)
        validator_fn: Function that takes text and returns True if valid
        warning_fn: Function that takes text and returns warning message or None (optional)

    Example:
        >>> pathEdit = ValidatedLineEdit(
        ...     validator_fn=lambda text: os.path.exists(text) or text == "",
        ...     warning_fn=lambda text: "File already exists" if os.path.exists(text) else None,
        ... )
    """

    def __init__(
        self,
        parent=None,
        validator_fn: Optional[Callable[[str], bool]] = None,
        warning_fn: Optional[Callable[[str], Optional[str]]] = None,
    ):
        """Initialize the validated line edit.

        Args:
            parent: Parent widget (optional)
            validator_fn: Function that takes text and returns True if valid
            warning_fn: Function that takes text and returns warning message or None (optional)
        """
        super().__init__(parent)
        self._validator_fn = validator_fn
        self._warning_fn = warning_fn
        self._original_tooltip = ""

        # Store original stylesheet to restore when valid
        self._base_stylesheet = self.styleSheet()

        # Connect to text change signal for real-time validation
        self.textChanged.connect(self._on_text_changed)

    def setToolTip(self, tooltip: str) -> None:
        """Override setToolTip to store original tooltip.

        Args:
            tooltip: The tooltip text to set
        """
        self._original_tooltip = tooltip
        super().setToolTip(tooltip)

    def _on_text_changed(self, text: str) -> None:
        """Handle text changes and update validation state.

        Args:
            text: The new text content
        """
        self._update_validation_state(text)

    def _update_validation_state(self, text: str) -> None:
        """Update visual feedback based on validation rules.

        Args:
            text: The current text to validate
        """
        tooltip_parts = []
        if self._original_tooltip:
            tooltip_parts.append(self._original_tooltip)

        # Check if text is valid according to custom validator
        is_valid = True
        if self._validator_fn is not None:
            is_valid = self._validator_fn(text)

        # Check for warnings
        warning_msg = None
        if self._warning_fn is not None:
            warning_msg = self._warning_fn(text)

        if not is_valid:
            # Invalid: red border
            self.setStyleSheet(self._base_stylesheet + "\n" + "QLineEdit { border: 2px solid #e74c3c; }")
            tooltip_parts.append("⚠️ Invalid input")
        elif warning_msg:
            # Warning: orange border
            self.setStyleSheet(self._base_stylesheet + "\n" + "QLineEdit { border: 2px solid #f39c12; }")
            tooltip_parts.append(f"⚠️ {warning_msg}")
        else:
            # Valid: restore default appearance
            self.setStyleSheet(self._base_stylesheet)

        # Update tooltip with all relevant information
        super().setToolTip("\n".join(tooltip_parts))
