# Validated Widgets Usage Guide

This guide demonstrates how to use the custom validated widgets in the dzetsaka QGIS plugin to provide real-time validation feedback to users.

## Overview

The `ui/validated_widgets.py` module provides three custom widget classes:

1. **ValidatedSpinBox** - Integer input with validation
2. **ValidatedDoubleSpinBox** - Float input with validation
3. **ValidatedLineEdit** - Text input with validation

All widgets provide:
- **Color-coded borders**: Red (invalid), orange (warning), default (valid)
- **Dynamic tooltips**: Updated in real-time with validation messages
- **Time estimates**: Optional time estimation for long-running operations
- **QGIS compatibility**: Uses `qgis.PyQt` for PyQt5/PyQt6 compatibility

## Basic Usage

### ValidatedSpinBox

```python
from ui.validated_widgets import ValidatedSpinBox

# Simple validation: value must be between 10 and 2000
optunaTrials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000
)
optunaTrials.setRange(10, 2000)
optunaTrials.setValue(100)
```

### With Warning Threshold

```python
# Show orange warning when value is high
optunaTrials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500  # Orange border when v >= 500
)
optunaTrials.setRange(10, 2000)
optunaTrials.setValue(100)
```

### With Time Estimator

```python
# Show estimated runtime in tooltip
optunaTrials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
)
optunaTrials.setRange(10, 2000)
optunaTrials.setValue(100)
# Tooltip will show: "⏱️ Estimated time: 10-30 min"
```

### ValidatedDoubleSpinBox

```python
from ui.validated_widgets import ValidatedDoubleSpinBox

# Validation for learning rate
learningRate = ValidatedDoubleSpinBox(
    validator_fn=lambda v: 0.0001 <= v <= 1.0,
    warning_threshold=0.5  # Warn if learning rate is high
)
learningRate.setRange(0.0001, 1.0)
learningRate.setDecimals(4)
learningRate.setValue(0.01)
```

### ValidatedLineEdit

```python
from ui.validated_widgets import ValidatedLineEdit
import os

# Validation for file paths
pathEdit = ValidatedLineEdit(
    validator_fn=lambda text: os.path.exists(text) or text == "",
    warning_fn=lambda text: "File already exists - will be overwritten"
                           if os.path.exists(text) and text != ""
                           else None
)
pathEdit.setText("")
```

## Integration Examples

### Replacing Existing QSpinBox in Guided Workflow

**Before:**
```python
from qgis.PyQt.QtWidgets import QSpinBox

self.optunaTrialsSpin = QSpinBox()
self.optunaTrialsSpin.setRange(10, 2000)
self.optunaTrialsSpin.setSingleStep(10)
self.optunaTrialsSpin.setValue(100)
```

**After:**
```python
from ui.validated_widgets import ValidatedSpinBox

self.optunaTrialsSpin = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
)
self.optunaTrialsSpin.setRange(10, 2000)
self.optunaTrialsSpin.setSingleStep(10)
self.optunaTrialsSpin.setValue(100)
self.optunaTrialsSpin.setToolTip("Number of Optuna optimization trials")
```

### Custom Validation Logic

```python
# Only allow even numbers
evenNumberSpin = ValidatedSpinBox(
    validator_fn=lambda v: v % 2 == 0
)
evenNumberSpin.setRange(2, 100)
evenNumberSpin.setValue(10)

# Only allow prime numbers
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primeSpin = ValidatedSpinBox(
    validator_fn=is_prime
)
primeSpin.setRange(2, 100)
primeSpin.setValue(7)
```

### Complex Warning Conditions

```python
# Multiple warning thresholds with dynamic messages
def time_estimate(trials):
    if trials < 50:
        return f"{trials * 0.05:.0f}-{trials * 0.1:.0f} sec"
    elif trials < 500:
        return f"{trials * 0.1:.0f}-{trials * 0.3:.0f} min"
    else:
        return f"{trials * 0.2:.0f}-{trials * 0.5:.0f} min"

optunaTrials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=time_estimate
)
```

## Visual Feedback States

### Valid State (Default)
- **Border**: Default Qt styling
- **Tooltip**: Original tooltip + optional time estimate
- **Example**: Value within normal range

### Warning State (Orange)
- **Border**: 2px solid orange (#f39c12)
- **Tooltip**: Original tooltip + warning message + optional time estimate
- **Example**: Value >= warning_threshold

### Invalid State (Red)
- **Border**: 2px solid red (#e74c3c)
- **Tooltip**: Original tooltip + "⚠️ Invalid value"
- **Example**: validator_fn returns False

## Best Practices

### 1. Set Tooltips After Widget Creation
```python
widget = ValidatedSpinBox(validator_fn=lambda v: v > 0)
widget.setRange(1, 100)
widget.setToolTip("Number of trees in random forest")  # Set after creation
```

### 2. Use Meaningful Validation Functions
```python
# Good: Clear validation logic
validator_fn=lambda v: 10 <= v <= 2000

# Bad: Unclear magic numbers
validator_fn=lambda v: v > 5 and v < 1000 and v % 10 == 0
```

### 3. Provide Helpful Time Estimates
```python
# Good: Realistic estimates based on actual measurements
time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"

# Bad: Vague or misleading estimates
time_estimator_fn=lambda v: "a few minutes"
```

### 4. Use Warning Thresholds for Performance-Critical Parameters
```python
# Warn users when they select values that might cause long runtimes
optunaTrials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,  # > 500 trials might take a long time
    time_estimator_fn=lambda v: f"{v * 0.2:.0f}-{v * 0.5:.0f} min"
)
```

### 5. Combine Validation with Qt Range Constraints
```python
# Use both Qt range and custom validator for comprehensive validation
widget = ValidatedSpinBox(
    validator_fn=lambda v: v % 10 == 0  # Must be multiple of 10
)
widget.setRange(10, 1000)  # Qt enforces hard limits
widget.setSingleStep(10)  # Makes it easier to get valid values
```

## Migration Checklist

To migrate existing spinboxes to validated widgets:

- [ ] Import `ValidatedSpinBox` or `ValidatedDoubleSpinBox`
- [ ] Replace `QSpinBox()` with `ValidatedSpinBox()`
- [ ] Add `validator_fn` for custom validation logic
- [ ] Add `warning_threshold` if applicable
- [ ] Add `time_estimator_fn` for long-running operations
- [ ] Test validation states (valid, warning, invalid)
- [ ] Verify tooltip updates work correctly

## Testing

Run the unit tests to verify widget behavior:

```bash
pytest tests/unit/test_validated_widgets.py -v
```

Test coverage includes:
- Valid values
- Warning values
- Invalid values
- Time estimator functionality
- Tooltip preservation
- Widgets without validators (normal operation)

## Troubleshooting

### Validation not triggering
- Ensure `valueChanged` signal is not blocked
- Check that validator_fn is callable and returns bool

### Tooltips not updating
- Call `setToolTip()` after widget creation
- Verify original tooltip is stored before validation updates

### Styling conflicts
- Check for global stylesheets that override widget styles
- Ensure base stylesheet is preserved and extended, not replaced

### Performance issues
- Avoid expensive operations in validator_fn (it runs on every value change)
- Cache validation results if computation is expensive
- Consider debouncing for text inputs with complex validation
