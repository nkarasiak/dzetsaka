# Validated Widgets Quick Reference

## Import

```python
from ui.validated_widgets import ValidatedSpinBox, ValidatedDoubleSpinBox, ValidatedLineEdit
```

## ValidatedSpinBox

```python
widget = ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,         # Required range check
    warning_threshold=500,                           # Show orange when v >= 500
    time_estimator_fn=lambda v: f"{v*0.1:.0f} min"  # Show time estimate
)
widget.setRange(10, 2000)
widget.setValue(100)
widget.setToolTip("Your tooltip here")
```

### Visual States

| Value | validator_fn | warning_threshold | Border | Tooltip |
|-------|--------------|-------------------|---------|---------|
| 50 | True | - | Default | Original |
| 600 | True | 500 | Orange | Original + ⚠️ High value + ⏱️ Time |
| 5 | False | - | Red | Original + ⚠️ Invalid |

## ValidatedDoubleSpinBox

```python
widget = ValidatedDoubleSpinBox(
    validator_fn=lambda v: 0.0001 <= v <= 1.0,
    warning_threshold=0.5
)
widget.setRange(0.0001, 1.0)
widget.setDecimals(4)
widget.setValue(0.01)
```

## ValidatedLineEdit

```python
widget = ValidatedLineEdit(
    validator_fn=lambda text: len(text) >= 3,
    warning_fn=lambda text: "Already exists" if os.path.exists(text) else None
)
widget.setText("example")
```

## Common Patterns

### Only validation (no warning)
```python
ValidatedSpinBox(validator_fn=lambda v: v > 0)
```

### Only warning (all values valid)
```python
ValidatedSpinBox(warning_threshold=1000)
```

### Validation + Warning
```python
ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500
)
```

### Validation + Warning + Time
```python
ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v*0.1:.0f} min"
)
```

## Time Estimator Helpers

```python
# Simple linear estimate
time_estimator_fn=lambda v: f"{v*0.1:.0f}-{v*0.3:.0f} min"

# Adaptive units
def estimate_time(value):
    seconds = value * 0.5
    if seconds < 60:
        return f"{seconds:.0f} sec"
    elif seconds < 3600:
        return f"{seconds/60:.0f} min"
    else:
        return f"{seconds/3600:.1f} hr"

time_estimator_fn=estimate_time
```

## Migration Checklist

- [ ] Import `ValidatedSpinBox` instead of `QSpinBox`
- [ ] Add `validator_fn` parameter
- [ ] Add `warning_threshold` if needed
- [ ] Add `time_estimator_fn` for long operations
- [ ] Keep all existing `setRange()`, `setValue()`, etc. calls
- [ ] Add/update `setToolTip()` after widget creation
- [ ] Test all three states: valid, warning, invalid

## Examples by Use Case

### Optuna trials
```python
ValidatedSpinBox(
    validator_fn=lambda v: 10 <= v <= 2000,
    warning_threshold=500,
    time_estimator_fn=lambda v: f"{v*0.1:.0f}-{v*0.3:.0f} min"
)
```

### SHAP sample size
```python
ValidatedSpinBox(
    validator_fn=lambda v: 100 <= v <= 50000,
    warning_threshold=10000,
    time_estimator_fn=lambda v: f"{v*0.01:.0f} sec"
)
```

### K-neighbors (SMOTE)
```python
ValidatedSpinBox(
    validator_fn=lambda v: 1 <= v <= 20,
    warning_threshold=15
)
```

### CV folds
```python
ValidatedSpinBox(
    validator_fn=lambda v: 2 <= v <= 10,
    warning_threshold=7
)
```

### Learning rate
```python
ValidatedDoubleSpinBox(
    validator_fn=lambda v: 0.0001 <= v <= 1.0,
    warning_threshold=0.5
)
```

### File path
```python
ValidatedLineEdit(
    validator_fn=lambda text: text.endswith('.tif') or text == "",
    warning_fn=lambda text: "File exists" if os.path.exists(text) else None
)
```
