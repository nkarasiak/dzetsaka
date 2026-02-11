# Results Explorer Dock

The Results Explorer Dock is an interactive widget that automatically opens after a classification task completes, providing immediate insights into the classification results without requiring external HTML reports.

## Overview

The Results Explorer provides comprehensive visualization and analysis of classification results through multiple tabs:

1. **Overview** - Key statistics and file paths
2. **Distribution** - Visual class distribution (histogram and pie chart)
3. **Accuracy** - Confusion matrix and metrics (if validation was performed)
4. **Explainability** - SHAP feature importance (if SHAP analysis was enabled)

## Features

### Tab 1: Overview

Displays essential classification information:
- **Algorithm Name**: The ML algorithm used (e.g., Random Forest, XGBoost)
- **Runtime**: Execution time in seconds or minutes
- **Number of Classes**: Count of distinct classes in the output
- **Total Pixels**: Total number of pixels classified
- **Timestamp**: When the classification completed
- **File Paths**: Input raster, output raster, and model file paths (selectable for easy copying)

### Tab 2: Distribution

Visual representation of class distribution:
- **Histogram**: Bar chart showing pixel counts per class
  - Color-coded bars using matplotlib's tab10 colormap
  - Value labels on each bar
  - Grid for easy reading
- **Pie Chart**: Proportional view of class distribution
  - Percentages shown for slices > 2%
  - Labels for significant classes
  - Same color scheme as histogram

**Requirements**: Requires matplotlib. If not available, displays installation instructions.

### Tab 3: Accuracy (Conditional)

Only shown if a confusion matrix was generated during validation:
- **Overall Metrics**:
  - Overall Accuracy (OA)
  - Kappa Coefficient
- **Confusion Matrix Table**:
  - Color-coded cells: green gradient for correct predictions (diagonal), red gradient for errors
  - Intensity proportional to count magnitude
  - Row headers: True class labels
  - Column headers: Predicted class labels

### Tab 4: Explainability (Conditional)

Only shown if SHAP analysis was performed:
- **SHAP Visualization**: Feature importance plot showing which bands contribute most to predictions
- **Info Text**: Explanation of SHAP values
- **Image Display**: Automatically scaled to fit dock width

## Quick Actions

Bottom section provides convenient actions:

### Apply Color Scheme
Applies a predefined color ramp to the classification result layer in QGIS:
- Uses matplotlib tab10 colors if available
- Falls back to HSV color rotation
- Updates layer renderer and triggers repaint
- Shows success confirmation

### Export to GeoPackage
Converts the classification result to GeoPackage format:
- Prompts for output path with suggested filename
- Uses GDAL to perform conversion
- Shows success message with output path

### Open Full Report (Conditional)
Opens the complete HTML report in the default web browser:
- Only shown if HTML report was generated
- Opens using system default browser

## Auto-Open Behavior

The Results Explorer automatically opens after successful classification completion. This behavior is controlled by a QSettings preference:

```python
# User preference (default: True)
QSettings().value("/dzetsaka/autoOpenResultsExplorer", True, type=bool)
```

Users can disable auto-open by setting this to False.

## Integration

### Task Completion Flow

1. Classification task completes successfully
2. `task_launcher.py::on_task_success()` collects result data:
   - Algorithm name
   - Runtime (calculated from task start time)
   - Output/input paths
   - Matrix path (if validation performed)
   - SHAP path (if found in report directory)
   - Report path (if HTML report generated)
3. Calls `open_results_explorer()` to create and display dock
4. Dock is added to QGIS interface as a dockable widget

### Result Data Structure

```python
result_data = {
    "algorithm": str,           # Full algorithm name
    "runtime_seconds": float,   # Execution time
    "output_path": str,         # Classification output
    "input_path": str,          # Input raster
    "model_path": str,          # Saved model
    "timestamp": str,           # ISO format timestamp
    "matrix_path": str,         # Optional: confusion matrix CSV
    "confidence_path": str,     # Optional: confidence map
    "shap_path": str,           # Optional: SHAP visualization
    "report_path": str,         # Optional: HTML report
    "class_counts": dict,       # Optional: {class_id: pixel_count}
}
```

### Class Count Computation

If `class_counts` is not provided, the dock computes it on-demand:
1. Opens output raster with GDAL
2. Reads band 1 as numpy array
3. Computes unique values and counts
4. Filters out nodata and unreasonable values (< 0 or > 10000)

## Code Structure

### Main Components

- **`ui/results_explorer_dock.py`**: Main dock widget implementation
  - `ResultsExplorerDock`: QDockWidget subclass
  - `open_results_explorer()`: Factory function to create and show dock

- **`src/dzetsaka/qgis/task_launcher.py`**: Integration point
  - `_build_result_data()`: Collects classification results
  - Modified `on_task_success()`: Triggers dock opening

### Dependencies

- **Required**: PyQt5/6, QGIS, GDAL
- **Optional**: matplotlib (for distribution charts), numpy (for class counting)

## User Customization

### Disabling Auto-Open

Users can disable automatic opening via QGIS Settings:

```python
from qgis.PyQt.QtCore import QSettings

settings = QSettings()
settings.setValue("/dzetsaka/autoOpenResultsExplorer", False)
```

This setting persists across QGIS sessions.

### Manual Opening

The dock can be manually opened for saved results:

```python
from ui.results_explorer_dock import open_results_explorer

result_data = {
    "algorithm": "Random Forest",
    "output_path": "/path/to/output.tif",
    # ... other fields
}

dock = open_results_explorer(result_data, parent=None, iface=iface)
if dock and iface:
    iface.addDockWidget(Qt.RightDockWidgetArea, dock)
```

## Testing

Tests are located in `tests/unit/test_results_explorer_dock.py`:

- `test_results_explorer_import()`: Verifies module imports
- `test_build_result_data()`: Tests result data building
- `test_result_data_no_optional_fields()`: Tests with minimal data
- `test_result_data_with_report_dir()`: Tests SHAP/report detection
- `test_results_explorer_dock_creation()`: Tests dock creation

All tests are marked with `@pytest.mark.qgis` and skip when QGIS is unavailable.

## Future Enhancements

Potential improvements:
- Export confusion matrix as image
- Interactive class selection to highlight in map
- Per-class statistics (min/max/mean confidence)
- Time-series comparison for multiple classifications
- Integration with trust artifacts (run_manifest.json, trust_card.json)
