# Enhanced Progress Reporting

This document describes the enhanced progress reporting system implemented in `src/dzetsaka/qgis/task_runner.py`.

## Overview

The enhanced progress reporting system provides detailed, multi-stage progress feedback during classification tasks with:
- Main task label showing current major operation
- Progress bar with percentage
- Sub-task information (e.g., "Testing parameter set 3/10")
- Automatic time estimation (elapsed + remaining)

## Components

### 1. EnhancedProgressWidget

A Qt widget that displays comprehensive progress information.

**Features:**
- Bold main task label (e.g., "Training Random Forest...")
- Progress bar with percentage display
- Sub-task label in gray, smaller font
- Time estimate label showing elapsed and remaining time
- Automatic time calculations based on progress rate

**Usage:**
```python
from dzetsaka.qgis.task_runner import EnhancedProgressWidget

# Create widget
widget = EnhancedProgressWidget()
widget.show()

# Update progress
widget.set_main_task("Training model...")
widget.set_progress(50, 100, "Testing parameter set 5/10")
```

### 2. TaskFeedbackAdapter (Enhanced)

The feedback adapter now includes intelligent message parsing to automatically detect stages and extract sub-task information.

**Stage Detection:**
The adapter recognizes common patterns in progress messages and maps them to predefined stages:

| Stage | Keywords | Progress Range |
|-------|----------|----------------|
| Loading data | loading, reading, opening | 0-5% |
| Training model | learning, training, fitting | 5-60% |
| Computing SHAP values | shap, explainability, explaining | 60-70% |
| Classifying raster | predicting, classifying, inference | 70-95% |
| Generating report | report, generating, writing output | 95-100% |

**Pattern Matching:**
- Optuna trials: `"trial 5/10"` → "Testing trial 5/10"
- Parameter sets: `"parameter set 3/20"` → "Testing parameter set 3/20"
- Band processing: `"4-band image"` → "Processing 4-band image"

### 3. ClassificationTask (Enhanced)

The `ClassificationTask` class now accepts an optional `enhanced_widget` parameter.

**Changes:**
```python
task = ClassificationTask(
    description="My classification",
    # ... existing parameters ...
    enhanced_widget=my_widget,  # NEW: optional progress widget
)
```

When provided, the task automatically:
1. Initializes the widget on task start
2. Updates main task label at each stage
3. Passes the widget to the feedback adapter
4. Shows completion message at 100%

## Integration Guide

### Minimal Integration

Add enhanced progress to existing code with just 3 lines:

```python
# 1. Create the widget
from dzetsaka.qgis.task_runner import EnhancedProgressWidget
progress_widget = EnhancedProgressWidget()
progress_widget.show()

# 2. Create task as usual, add enhanced_widget parameter
task = ClassificationTask(
    # ... all your existing parameters ...
    enhanced_widget=progress_widget,  # <-- Add this line
)

# 3. Add to task manager as usual
from qgis.core import QgsApplication
QgsApplication.taskManager().addTask(task)
```

### Full Dialog Integration

For a complete progress dialog with cancel button:

```python
from dzetsaka.qgis.progress_widget_example import ClassificationDialogWithProgress

dialog = ClassificationDialogWithProgress()
dialog.start_classification(
    do_training=True,
    raster_path="/path/to/raster.tif",
    vector_path="/path/to/training.shp",
    class_field="class",
    model_path="/path/to/model.pkl",
    # ... other parameters ...
)
```

See `src/dzetsaka/qgis/progress_widget_example.py` for complete example code.

## Progress Stages

The system uses predefined stages to structure progress reporting:

```python
PROGRESS_STAGES = {
    "loading": {"name": "Loading data", "start": 0, "end": 5},
    "training": {"name": "Training model", "start": 5, "end": 60},
    "shap": {"name": "Computing SHAP values", "start": 60, "end": 70},
    "classifying": {"name": "Classifying raster", "start": 70, "end": 95},
    "report": {"name": "Generating report", "start": 95, "end": 100},
}
```

These stages provide consistent progress ranges and ensure the progress bar moves smoothly through the workflow.

## Time Estimation

The widget automatically calculates and displays time estimates:

- **During execution:** Shows elapsed time and estimated remaining time
  - Format: "Elapsed: 45s | ~30s remaining"
  - Uses linear extrapolation based on current progress rate
- **On completion:** Shows total elapsed time
  - Format: "Completed in 75s"

Time formatting:
- Under 60 seconds: Shows seconds (e.g., "~45s remaining")
- Over 60 seconds: Shows minutes (e.g., "~2m remaining")

## Testing

Comprehensive unit tests are provided in `tests/unit/test_enhanced_progress_widget.py`:

```bash
# Run tests
pytest tests/unit/test_enhanced_progress_widget.py

# Run with coverage
pytest tests/unit/test_enhanced_progress_widget.py --cov=dzetsaka.qgis.task_runner
```

**Test coverage includes:**
- Widget initialization and basic operations
- Progress clamping (values stay in 0-100 range)
- Time estimation accuracy
- Stage detection for all keywords
- Pattern matching (trials, parameters, bands)
- Sub-task extraction and display
- Reset functionality

## Architecture

### Message Flow

```
Classification Pipeline
        ↓
  feedback.setProgressText("Learning... trial 5/10")
        ↓
  TaskFeedbackAdapter._parse_and_update_progress()
        ↓
  [Detects "learning" → stage="training"]
  [Parses "trial 5/10" → sub_task="Testing trial 5/10"]
  [Calculates progress: 5 + (5/10 * 55) = 32.5%]
        ↓
  EnhancedProgressWidget.set_progress(32.5, 100, "Testing trial 5/10")
        ↓
  [Updates progress bar to 32%]
  [Shows sub-task label]
  [Calculates time estimate]
```

### Design Decisions

1. **Keyword-based detection:** Simple, robust, works with existing messages
2. **Stage ranges:** Provide smooth progress even without explicit updates
3. **Linear time estimation:** Simple and accurate for most workflows
4. **Optional integration:** Backward compatible with existing code

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive time estimation:** Use exponential smoothing for better accuracy
2. **Custom stage definitions:** Allow users to define workflow-specific stages
3. **Progress persistence:** Save progress state for long-running tasks
4. **Cancellation feedback:** Show cancellation progress
5. **Nested progress:** Support sub-task progress bars for parallel operations
6. **Historical tracking:** Remember typical durations for better estimates

## API Reference

### EnhancedProgressWidget

```python
class EnhancedProgressWidget(QWidget):
    def __init__(self, parent=None)
    def set_main_task(self, text: str)
    def set_progress(self, value: float, total: float = 100, sub_task_text: str = "")
    def reset()
```

### TaskFeedbackAdapter

```python
class TaskFeedbackAdapter:
    def __init__(self, task, enhanced_widget=None)
    def setProgress(self, value: float)
    def setProgressText(self, text: str)
```

### ClassificationTask

```python
class ClassificationTask(QgsTask):
    def __init__(
        self,
        description: str,
        *,
        # ... existing parameters ...
        enhanced_widget: EnhancedProgressWidget | None = None,
    )
```

## See Also

- `src/dzetsaka/qgis/task_runner.py` - Main implementation
- `src/dzetsaka/qgis/progress_widget_example.py` - Usage examples
- `tests/unit/test_enhanced_progress_widget.py` - Test suite
- `ui/install_progress_dialog.py` - Similar progress dialog for dependency installation
