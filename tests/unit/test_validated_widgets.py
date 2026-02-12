"""Unit tests for validated widgets module.

Tests the real-time validation and visual feedback mechanisms
for custom widget classes.
"""

import pytest

# Skip if QGIS environment is not available
qtwidgets = pytest.importorskip("qgis.PyQt.QtWidgets")
if not hasattr(qtwidgets, "QApplication"):
    pytest.skip("QGIS QtWidgets QApplication is unavailable", allow_module_level=True)
QApplication = qtwidgets.QApplication

from ui.validated_widgets import ValidatedDoubleSpinBox, ValidatedLineEdit, ValidatedSpinBox


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication instance for widget testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_validated_spinbox_valid_value(qapp):
    """Test ValidatedSpinBox with valid value."""
    widget = ValidatedSpinBox(
        validator_fn=lambda v: 10 <= v <= 100,
        warning_threshold=80,
    )
    widget.setRange(10, 100)
    widget.setValue(50)

    # Should not have red or orange border (valid value below warning threshold)
    stylesheet = widget.styleSheet()
    assert "#e74c3c" not in stylesheet  # No red border
    assert "#f39c12" not in stylesheet  # No orange border


def test_validated_spinbox_warning_value(qapp):
    """Test ValidatedSpinBox with warning value."""
    widget = ValidatedSpinBox(
        validator_fn=lambda v: 10 <= v <= 100,
        warning_threshold=80,
    )
    widget.setRange(10, 100)
    widget.setValue(90)

    # Should have orange border (warning)
    stylesheet = widget.styleSheet()
    assert "#f39c12" in stylesheet  # Orange border
    assert "⚠️ High value" in widget.toolTip()


def test_validated_spinbox_invalid_value(qapp):
    """Test ValidatedSpinBox with invalid value."""
    widget = ValidatedSpinBox(
        validator_fn=lambda v: v % 2 == 0,  # Only even numbers valid
    )
    widget.setRange(0, 100)
    widget.setValue(51)  # Odd number

    # Should have red border (invalid)
    stylesheet = widget.styleSheet()
    assert "#e74c3c" in stylesheet  # Red border
    assert "⚠️ Invalid value" in widget.toolTip()


def test_validated_spinbox_time_estimator(qapp):
    """Test ValidatedSpinBox with time estimator function."""
    widget = ValidatedSpinBox(
        validator_fn=lambda v: 10 <= v <= 2000,
        warning_threshold=500,
        time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min",
    )
    widget.setRange(10, 2000)
    widget.setValue(100)

    # Should show time estimate in tooltip
    tooltip = widget.toolTip()
    assert "⏱️ Estimated time:" in tooltip
    assert "10-30 min" in tooltip


def test_validated_double_spinbox_valid_value(qapp):
    """Test ValidatedDoubleSpinBox with valid value."""
    widget = ValidatedDoubleSpinBox(
        validator_fn=lambda v: 0.0001 <= v <= 1.0,
        warning_threshold=0.5,
    )
    widget.setRange(0.0001, 1.0)
    widget.setValue(0.1)

    # Should not have red or orange border
    stylesheet = widget.styleSheet()
    assert "#e74c3c" not in stylesheet
    assert "#f39c12" not in stylesheet


def test_validated_double_spinbox_warning_value(qapp):
    """Test ValidatedDoubleSpinBox with warning value."""
    widget = ValidatedDoubleSpinBox(
        validator_fn=lambda v: 0.0001 <= v <= 1.0,
        warning_threshold=0.5,
    )
    widget.setRange(0.0001, 1.0)
    widget.setValue(0.8)

    # Should have orange border
    stylesheet = widget.styleSheet()
    assert "#f39c12" in stylesheet
    assert "⚠️ High value" in widget.toolTip()


def test_validated_lineedit_valid_text(qapp):
    """Test ValidatedLineEdit with valid text."""
    widget = ValidatedLineEdit(
        validator_fn=lambda text: len(text) >= 3,
    )
    widget.setText("valid text")

    # Should not have red or orange border
    stylesheet = widget.styleSheet()
    assert "#e74c3c" not in stylesheet
    assert "#f39c12" not in stylesheet


def test_validated_lineedit_invalid_text(qapp):
    """Test ValidatedLineEdit with invalid text."""
    widget = ValidatedLineEdit(
        validator_fn=lambda text: len(text) >= 3,
    )
    widget.setText("ab")  # Too short

    # Should have red border
    stylesheet = widget.styleSheet()
    assert "#e74c3c" in stylesheet
    assert "⚠️ Invalid input" in widget.toolTip()


def test_validated_lineedit_warning(qapp):
    """Test ValidatedLineEdit with warning function."""
    widget = ValidatedLineEdit(
        validator_fn=lambda text: True,  # Always valid
        warning_fn=lambda text: "Contains special chars" if not text.isalnum() else None,
    )
    widget.setText("hello_world")

    # Should have orange border
    stylesheet = widget.styleSheet()
    assert "#f39c12" in stylesheet
    assert "⚠️ Contains special chars" in widget.toolTip()


def test_validated_spinbox_tooltip_preservation(qapp):
    """Test that original tooltips are preserved."""
    widget = ValidatedSpinBox(
        validator_fn=lambda v: v > 0,
    )
    widget.setToolTip("Original tooltip text")
    widget.setRange(0, 100)
    widget.setValue(50)

    # Original tooltip should be preserved
    tooltip = widget.toolTip()
    assert "Original tooltip text" in tooltip


def test_validated_spinbox_no_validator(qapp):
    """Test ValidatedSpinBox without validator (should work normally)."""
    widget = ValidatedSpinBox()  # No validator
    widget.setRange(0, 100)
    widget.setValue(50)

    # Should not have any border styling
    stylesheet = widget.styleSheet()
    assert "#e74c3c" not in stylesheet
    assert "#f39c12" not in stylesheet
