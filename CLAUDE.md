# dzetsaka QGIS Plugin Development Guide

This file provides essential context for AI assistants working in the dzetsaka repository.

## Quick Reference

### Essential Commands

**Build & Package:**
- `make plugin-package` - Create QGIS plugin .zip for distribution
- `make build` - Build Python package

**Code Quality:**
- `make lint` - Run ruff linting checks
- `make format` - Auto-format code with ruff (includes --fix)
- `make typecheck` - Run mypy type checking
- `make quality` - Run both lint and typecheck
- `make pre-commit` - Run pre-commit hooks on all files

**Testing:**
- `make test` - Run all tests with pytest
- `make test-coverage` - Run tests with HTML coverage report
- `make quick-test` - Run non-QGIS tests only (faster)
- `make test-verbose` - Run tests with detailed output

**Development Setup:**
- `make setup-dev` - Complete dev setup (clean, install-dev, pre-commit-install)
- `make install-dev` - Install with development dependencies
- `make install-full` - Install with all optional dependencies
- `make ci` - Simulate CI environment (quality + test)

**Shortcuts:**
- `make dev-check` - Format, lint, typecheck, and test in one command

### Code Style Standards

- **Line length:** 120 characters (ruff/black configured)
- **Python version:** 3.8+ minimum (QGIS 3.0+ requirement)
- **QGIS conventions:** Non-lowercase function/variable names allowed (N802, N806 disabled)
- **Never edit generated files:** `resources.py`, `ui/*.py` (Qt Designer generated)
- **Memory constraints:** 512MB limit for raster processing operations

## Architecture Overview

### Plugin Entry Flow

```
__init__.py (QGIS entry point)
    └─> dzetsaka.py (main plugin class)
        ├─> dzetsaka_provider.py (QGIS Processing Framework provider)
        │   └─> processing/*.py (individual algorithm implementations)
        └─> ui/dzetsaka_dockwidget.py (interactive GUI)
```

### Dual Interface System

dzetsaka provides two complementary interfaces:

1. **Interactive Dock Widget** (`ui/dzetsaka_dockwidget.py`)
   - Traditional point-and-click interface
   - Real-time parameter adjustment
   - Visual feedback and progress bars
   - Direct user interaction model

2. **QGIS Processing Framework** (`processing/*.py`)
   - Batch processing capabilities
   - Model builder integration
   - Command-line accessible
   - Scriptable workflows

Both interfaces use the same core classification engine.

### Core ML Engine

**Location:** `scripts/mainfunction.py` (1500+ lines - central to all classification)

**Primary Classes:**
- `LearnModel` - Training and model creation
- `ClassifyImage` - Raster classification with trained models

**Key Features:**
- Memory-efficient block-based raster processing
- Automatic mask detection (e.g., `image.tif` → `image_mask.tif`)
- Model persistence (save/load for reuse across images)
- Confidence map generation
- Progress callback system

### Centralized Configuration System

**Location:** `classifier_config.py`

Single source of truth for all 11 algorithms containing:
- Algorithm definitions and metadata
- Dependency mappings (built-in vs. external)
- Hyperparameter grids for optimization
- UI parameter configurations
- Utility functions for classifier management

This centralized approach eliminates scattered configuration and ensures consistency across both UI and processing interfaces.

### Three-Tier Algorithm Architecture

**Tier 1: Built-in (No Dependencies)**
- **GMM (Gaussian Mixture Model)** - Fast baseline classifier, no external deps

**Tier 2: Scikit-learn Based (8 Algorithms)**
- Random Forest (RF), SVM, KNN
- Extra Trees (ET), Gradient Boosting (GBC)
- Logistic Regression (LR), Naive Bayes (NB), MLP
- Shared dependency: scikit-learn
- Automatic hyperparameter optimization via GridSearchCV

**Tier 3: Specialized (Advanced Gradient Boosting)**
- **XGBoost** - State-of-the-art gradient boosting
- **LightGBM** - Fast gradient boosting framework
- Separate package dependencies
- Custom wrapper classes for label encoding

### Dependency Management System

**Validation:** `sklearn_validator.py`
- Runtime dependency checking
- Version compatibility validation
- Import fallback mechanisms

**Auto-Installation System** (v4.2.0+):
- Detects missing packages automatically
- One-click installation via pip subprocess
- Real-time progress logging in QGIS
- Handles: scikit-learn, xgboost, lightgbm
- Eliminates manual pip commands for users

### Label Encoding System

**Problem:** XGBoost/LightGBM require continuous class labels (0,1,2,3...) but users often have sparse labels (0,1,3,5...)

**Solution:** Transparent wrapper classes in `scripts/mainfunction.py`
- `XGBClassifierWrapper` - Auto-encodes/decodes labels for XGBoost
- `LGBMClassifierWrapper` - Auto-encodes/decodes labels for LightGBM
- Seamless workflow - users never see the encoding
- Maintains compatibility with existing code
- Proper pickling support for model persistence

### Backward Compatibility System

**Parameter Migration Decorator** (from v4.1.0):
- Removes Hungarian notation prefixes (in/out) from parameters
- Old names: `inRaster`, `outRaster` → New names: `raster`, `output_raster`
- `@migrate_parameters` decorator ensures old code still works
- Deprecation warnings guide users to new names
- See migration guide in `scripts/mainfunction.py` docstrings

### File Organization

**Core Logic:**
- `scripts/mainfunction.py` - ML training and classification engine
- `scripts/function_dataraster.py` - Raster I/O operations (GDAL wrappers)
- `scripts/function_vector.py` - Vector I/O operations (OGR wrappers)
- `classifier_config.py` - Algorithm definitions and configuration

**QGIS Integration:**
- `dzetsaka.py` - Main plugin class (toolbar, menus, initialization)
- `dzetsaka_provider.py` - Processing framework provider registration
- `processing/*.py` - Individual processing algorithm implementations
- `ui/dzetsaka_dockwidget.py` - Main GUI dock widget

**UI Components:**
- `ui/*.ui` - Qt Designer files (source of truth for UI)
- `ui/*.py` - Generated Python from .ui files (NEVER edit manually)
- `resources.py` - Generated Qt resource file (NEVER edit manually)

**Configuration:**
- `metadata.txt` - QGIS plugin metadata (version, description, dependencies)
- `pyproject.toml` - Python package configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

## Algorithm Support Details

### Hyperparameter Optimization

All algorithms except GMM use automatic grid search with cross-validation:

**Random Forest:** 5-fold CV, tunes `n_estimators` and `max_features`
**SVM:** 3-fold CV, tunes `gamma` (0.25-4.0) and `C` (0.1-100)
**KNN:** 3-fold CV, tunes `n_neighbors` (1-17)
**XGBoost:** 3-fold CV, tunes `n_estimators`, `max_depth`, `learning_rate`
**LightGBM:** 3-fold CV, tunes `n_estimators`, `num_leaves`, `learning_rate`
**Extra Trees:** 3-fold CV, tunes `n_estimators` and `max_features`
**Gradient Boosting:** 3-fold CV, tunes `n_estimators` and `max_depth`
**Logistic Regression:** 3-fold CV, tunes `C` and `penalty`
**MLP:** 3-fold CV, tunes `hidden_layer_sizes` and `learning_rate`
**Naive Bayes:** Optimal defaults (no tuning needed)

Grid configurations are defined in `classifier_config.py`.

### Sparse Label Handling

dzetsaka automatically handles non-continuous class labels:

**Scenario:** Training data has classes 0, 1, 3, 5 (missing 2 and 4)

**Scikit-learn algorithms:** Work natively with sparse labels
**XGBoost/LightGBM:** Transparent encoding/decoding via wrapper classes

Users never need to manually encode labels or worry about compatibility.

## Development Tips

### Working with UI Files

1. Edit `.ui` files in Qt Designer
2. Regenerate Python files: `pyrcc5 -o resources.py resources.qrc`
3. Never manually edit generated `.py` files in `ui/` directory

### Plugin Packaging

`make plugin-package` creates a `.zip` file for QGIS plugin repository:
- Excludes dev files (.git, tests, docs, build artifacts)
- Includes metadata.txt and readme.md
- Ready for direct upload to QGIS plugin repository

### Testing Strategy

- `make quick-test` - Fast iteration without QGIS environment
- `make test` - Full test suite (requires QGIS)
- `make test-coverage` - HTML coverage report in `htmlcov/`

### Memory Considerations

Raster processing uses block-based iteration to stay within 512MB limit:
- See `scripts/function_dataraster.py` for block size calculations
- Large images processed in chunks, not all-at-once
- Memory-efficient NumPy operations preferred

### Error Reporting

v4.2.0+ includes GitHub issue integration:
- Automatic error templates with system info
- Links to create issues from error dialogs
- Detailed logging for debugging

## Important Notes

- **QGIS 3.0+ minimum:** Python 3.8+ required
- **Auto-install is a key feature:** Highlight in user-facing changes
- **11 algorithms supported:** Core differentiator from other QGIS classifiers
- **Hyperparameter tuning:** Automatic optimization sets dzetsaka apart
- **Model persistence:** Save trained models for reuse across multiple images

## Plugin Metadata

- **Name:** dzetsaka : Classification tool
- **Current Version:** 4.2.2
- **Author:** Nicolas Karasiak
- **Category:** Raster
- **Repository:** https://github.com/nkarasiak/dzetsaka
- **Issue Tracker:** https://github.com/nkarasiak/dzetsaka/issues
- **DOI:** 10.5281/zenodo.2552284

## Recent Major Changes

**v4.2.2:** Critical bug fixes for imports, progress bars, and resource paths
**v4.2.0:** 11 algorithms, auto-install system, hyperparameter optimization
**v4.1.0:** Parameter migration, memory optimizations, comprehensive refactoring
