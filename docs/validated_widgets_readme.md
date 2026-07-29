# Validated Widgets Module

Custom Qt widget classes with real-time validation and visual feedback for the dzetsaka QGIS plugin.

## Quick Start

```python
from ui.validated_widgets import ValidatedSpinBox

# Create a spinbox with validation
trials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
)
trials.setRange(10, 2000)
trials.setValue(100)
```

## Features

### Visual Feedback States

| State | Border Color | When |
|-------|-------------|------|
| **Valid** | Default | `validator_fn` returns `True` and value below `warning_threshold` |
| **Warning** | Orange (#f39c12) | `validator_fn` returns `True` but value >= `warning_threshold` |
| **Invalid** | Red (#e74c3c) | `validator_fn` returns `False` |

### Dynamic Tooltips

Tooltips are automatically updated with:
- Original tooltip text (preserved)
- Validation warnings
- Time estimates (if provided)

Example tooltip for warning state:
```
Number of Optuna trials to run
⚠️ High value (≥500)
⏱️ Estimated time: 50-150 min
```

## Widget Classes

### ValidatedSpinBox

Integer input with validation feedback.

**Constructor Parameters:**
- `parent` (optional): Parent widget
- `validator_fn` (optional): Function `(int) -> bool` for validation
- `warning_threshold` (optional): Value above which to show warning
- `time_estimator_fn` (optional): Function `(int) -> str` for time estimates

**Example:**
```python
optuna_trials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
)
optuna_trials.setRange(10, 2000)
optuna_trials.setValue(100)
optuna_trials.setToolTip("Number of optimization trials")
```

### ValidatedDoubleSpinBox

Float input with validation feedback.

**Constructor Parameters:**
- `parent` (optional): Parent widget
- `validator_fn` (optional): Function `(float) -> bool` for validation
- `warning_threshold` (optional): Value above which to show warning
- `time_estimator_fn` (optional): Function `(float) -> str` for time estimates

**Example:**
```python
learning_rate = ValidatedDoubleSpinBox(
    validator_fn=lambda v: 0.0001 <= v <= 1.0,
    warning_threshold=0.5
)
learning_rate.setRange(0.0001, 1.0)
learning_rate.setDecimals(4)
learning_rate.setValue(0.01)
```

### ValidatedLineEdit

Text input with validation feedback.

**Constructor Parameters:**
- `parent` (optional): Parent widget
- `validator_fn` (optional): Function `(str) -> bool` for validation
- `warning_fn` (optional): Function `(str) -> Optional[str]` returning warning message

**Example:**
```python
import os

path_edit = ValidatedLineEdit(
    validator_fn=lambda text: os.path.exists(text) or text == "",
    warning_fn=lambda text: "File will be overwritten"
                           if os.path.exists(text) and text != ""
                           else None
)
```

## Integration Examples

### Replacing QSpinBox

**Before:**
```python
from qgis.PyQt.QtWidgets import QSpinBox

self.trials = QSpinBox()
self.trials.setRange(10, 2000)
self.trials.setValue(100)
```

**After:**
```python
from ui.validated_widgets import ValidatedSpinBox

self.trials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
)
self.trials.setRange(10, 2000)
self.trials.setValue(100)
self.trials.setToolTip("Number of trials")
```

### Custom Validation Logic

```python
# Only accept even numbers
even_only = ValidatedSpinBox(
    validator_fn=lambda v: v % 2 == 0
)

# Accept multiples of 10
multiples_of_10 = ValidatedSpinBox(
    validator_fn=lambda v: v % 10 == 0
)

# Complex validation with multiple conditions
complex_validation = ValidatedSpinBox(
    validator_fn=lambda v: v >= 10 and v <= 1000 and v % 5 == 0
)
```

### Time Estimation Functions

```python
def estimate_optuna_time(trials):
    """Estimate Optuna runtime."""
    min_time = trials * 0.1
    max_time = trials * 0.3

    if max_time < 1:
        return f"{min_time * 60:.0f}-{max_time * 60:.0f} sec"
    elif max_time < 60:
        return f"{min_time:.0f}-{max_time:.0f} min"
    else:
        return f"{min_time / 60:.1f}-{max_time / 60:.1f} hr"

trials = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=estimate_optuna_time
)
```

## Testing

Run unit tests:
```bash
pytest tests/unit/test_validated_widgets.py -v
```

Note: Tests require QGIS Python environment and will be skipped if QGIS is not available.

## Design Principles

1. **Non-intrusive**: Works as drop-in replacement for standard Qt widgets
2. **Real-time feedback**: Validates on every value change
3. **Performance-conscious**: Validation functions should be fast
4. **Accessible**: Uses color + icons + text for color-blind accessibility
5. **Preserves Qt patterns**: Compatible with existing signal/slot connections

## Performance Considerations

- Validation runs on every `valueChanged`/`textChanged` signal
- Keep validator functions lightweight (avoid I/O, heavy computation)
- For expensive validation, consider debouncing or caching results

## Color Scheme

Colors chosen for accessibility and consistency with Qt styling:

- **Red** (#e74c3c): Error state (invalid input)
- **Orange** (#f39c12): Warning state (valid but potentially problematic)
- **Default**: Uses Qt's default widget styling

These colors are also distinguishable for common types of color blindness.

## Files

- **`ui/validated_widgets.py`**: Widget implementations
- **`tests/unit/test_validated_widgets.py`**: Unit tests
- **`docs/validated_widgets_usage.md`**: Detailed usage guide
- **`examples/validated_widgets_integration.py`**: Integration examples

## See Also

- [Detailed Usage Guide](../docs/validated_widgets_usage.md)
- [Integration Examples](../examples/validated_widgets_integration.py)
- [Unit Tests](../tests/unit/test_validated_widgets.py)
