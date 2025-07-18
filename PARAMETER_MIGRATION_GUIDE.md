# Parameter Migration Guide

This document outlines the parameter name changes in the refactored `mainfunction.py` and provides guidance for migrating to the new parameter names while maintaining backward compatibility.

## Overview

The refactoring removes Hungarian notation prefixes (`in`, `out`, `io`) from parameter names to improve code readability and follow modern Python conventions. **All old parameter names remain functional** through backward compatibility aliases.

## Parameter Name Mapping

### `learnModel` Class

| Old Parameter Name | New Parameter Name | Description |
|-------------------|-------------------|-------------|
| `inRaster` | `raster_path` | Input raster image path or numpy array |
| `inVector` | `vector_path` | Training vector/shapefile path or numpy array |
| `inField` | `class_field` | Column name containing class numbers |
| `outModel` | `model_path` | Output model file path |
| `inSplit` | `split_config` | Training split configuration (int, float, or 'SLOO'/'STAND') |
| `inSeed` | `random_seed` | Random seed for reproducibility |
| `outMatrix` | `matrix_path` | Confusion matrix output file path |
| `inClassifier` | `classifier` | Classifier type ('GMM', 'RF', 'SVM', 'KNN') |

### `classifyImage` Class

| Old Parameter Name | New Parameter Name | Description |
|-------------------|-------------------|-------------|
| `inRaster` | `raster_path` | Input raster image path |
| `inModel` | `model_path` | Trained model file path |
| `outRaster` | `output_path` | Output classification raster path |
| `inMask` | `mask_path` | Optional mask raster path |

### Helper Functions

| Old Parameter Name | New Parameter Name | Description |
|-------------------|-------------------|-------------|
| `inRaster` | `raster_path` | Reference raster path |
| `inShape` | `shapefile_path` | Vector shapefile path |
| `inField` | `class_field` | Attribute field to rasterize |

## Migration Examples

### Example 1: Using New Parameter Names (Recommended)

```python
# New way - recommended
model = learnModel(
    raster_path="path/to/raster.tif",
    vector_path="path/to/training.shp", 
    class_field="Class",
    model_path="path/to/model.pkl",
    split_config=80,
    random_seed=42,
    classifier="RF"
)

classifier = classifyImage()
classifier.initPredict(
    raster_path="path/to/raster.tif",
    model_path="path/to/model.pkl",
    output_path="path/to/output.tif"
)
```

### Example 2: Using Old Parameter Names (Still Works)

```python
# Old way - still functional but deprecated
model = learnModel(
    inRaster="path/to/raster.tif",      # Works but triggers deprecation warning
    inVector="path/to/training.shp",   # Works but triggers deprecation warning
    inField="Class",                   # Works but triggers deprecation warning
    outModel="path/to/model.pkl",      # Works but triggers deprecation warning
    inSplit=80,                        # Works but triggers deprecation warning
    inSeed=42,                         # Works but triggers deprecation warning
    inClassifier="RF"                  # Works but triggers deprecation warning
)

classifier = classifyImage()
classifier.initPredict(
    inRaster="path/to/raster.tif",     # Works but triggers deprecation warning
    inModel="path/to/model.pkl",       # Works but triggers deprecation warning
    outRaster="path/to/output.tif"     # Works but triggers deprecation warning
)
```

### Example 3: Mixed Parameter Usage

```python
# Mixed usage - new parameters take precedence
model = learnModel(
    raster_path="new_raster.tif",      # This will be used
    inRaster="old_raster.tif",         # This will be ignored
    vector_path="training.shp",        # New parameter
    inField="Class",                   # Old parameter (used since class_field not specified)
    classifier="RF"                    # New parameter
)
```

## Deprecation Warnings

When using old parameter names, you'll see deprecation warnings like:

```
DeprecationWarning: Parameter 'inRaster' is deprecated. Use the new parameter name instead.
```

These warnings help identify code that needs updating but don't break functionality.

## Migration Strategy

### Phase 1: Update New Code (Immediate)
- Use new parameter names for all new code
- Update documentation and examples

### Phase 2: Gradual Migration (Recommended)
- Update existing code gradually during maintenance
- Fix deprecation warnings as they appear
- Test thoroughly after each update

### Phase 3: Complete Migration (Future)
- Eventually remove old parameter support (in a major version update)
- This would be communicated well in advance

## Benefits of New Parameter Names

1. **Improved Readability**: `raster_path` is clearer than `inRaster`
2. **Modern Python Style**: Follows PEP 8 naming conventions
3. **Better IDE Support**: More descriptive names improve autocompletion
4. **Reduced Cognitive Load**: No need to remember prefix meanings
5. **Consistency**: Uniform naming scheme across the codebase

## Configuration Constants

The refactoring also introduces configuration constants for better maintainability:

```python
CLASSIFIER_CONFIGS = {
    'RF': {'param_grid': {...}, 'n_splits': 5},
    'SVM': {'param_grid': {...}, 'n_splits': 3},
    'KNN': {'param_grid': {...}, 'n_splits': 3}
}

MAX_MEMORY_MB = 512
MIN_CROSS_VALIDATION_SPLITS = 2
```

## Testing Backward Compatibility

Run the included test to verify backward compatibility:

```bash
python test_backward_compat.py
```

This test verifies:
- New parameter names work correctly
- Old parameter names still function
- Proper deprecation warnings are issued
- Parameter precedence rules work as expected

## Support

If you encounter any issues during migration or have questions about the new parameter names, please:

1. Check this migration guide
2. Run the backward compatibility test
3. Review the updated docstrings in the code
4. File an issue if problems persist

The backward compatibility will be maintained for several releases to ensure a smooth transition.