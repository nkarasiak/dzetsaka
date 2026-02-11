# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dzetsaka is a QGIS plugin for raster classification supporting 12 machine learning algorithms (GMM, RF, SVM, KNN, XGBoost, LightGBM, CatBoost, Extra Trees, GBC, LR, NB, MLP). It features automatic dependency installation, recipe-based workflows, Optuna optimization, SHAP explainability, and class imbalance handling.

## Development Commands

### Testing
```bash
# Run all tests
make test
pytest

# Run quick tests (excluding QGIS-dependent tests)
make quick-test
pytest tests/ -k "not qgis" --disable-warnings

# Run specific test types
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests
pytest -m sklearn               # Tests requiring scikit-learn
pytest -m xgboost               # Tests requiring XGBoost

# Run with coverage
make test-coverage
pytest --cov=dzetsaka --cov-report=html --cov-report=term

# Run single test file
pytest tests/unit/test_guided_workflow.py
```

### Linting and Formatting
```bash
# Format code with ruff (primary formatter)
make format
ruff format .
ruff check --fix .

# Lint only (no auto-fix)
make lint
ruff check .

# Type checking
make typecheck
mypy dzetsaka/ --ignore-missing-imports

# Run all quality checks
make quality     # lint + typecheck
make dev-check   # format + lint + typecheck + test
```

### Installation
```bash
# Development setup (installs package in editable mode)
make install-dev
pip install -e ".[dev,test,docs]"

# Install with all ML dependencies
make install-full
pip install -e ".[full,dev,test,docs]"

# Install pre-commit hooks
make pre-commit-install
```

### Building
```bash
# Build Python package
make build
python -m build

# Create QGIS plugin package (.zip)
make plugin-package
python tools/build_plugin.py --output dzetsaka.zip
```

## Architecture

### High-Level Structure

The codebase follows a **hybrid architecture**:
- **Legacy monolithic layer** (`scripts/classification_pipeline.py`, `ui/` widgets) for backward compatibility
- **Modern hexagonal architecture** (`src/dzetsaka/`) being introduced incrementally

### Key Architectural Layers

#### 1. Presentation Layer (`src/dzetsaka/presentation/qgis/`)
- **QGIS plugin runtime** (`plugin_runtime.py`, `ui_init.py`, `runtime_bootstrap.py`)
- **Dashboard UI** (`dashboard_dock.py`, `dashboard_execution.py`)
- **Guided workflow** (`ui/classification_workflow_ui.py`) - main classification wizard
- **Dependency management** (`dependency_installer.py`, `dependency_catalog.py`)
- **Settings and configuration** (`config_runtime.py`, `settings_handlers.py`)

#### 2. Domain Layer (`src/dzetsaka/domain/`)
- **Value objects** for recipes (`recipe_schema_v2.py`)
- **Entities** for classification models
- Business logic isolated from QGIS/UI concerns

#### 3. Application Layer (`src/dzetsaka/application/`)
- **Use cases** (`train_model.py`, `classify_raster.py`)
- **Ports** for abstractions
- **DTOs** for data transfer

#### 4. Infrastructure Layer (`src/dzetsaka/infrastructure/`)
- **ML adapters** (`ml/sklearn/`, `ml/xgboost/`, `ml/lightgbm/`, `ml/catboost/`)
- **Geo adapters** (`geo/gdal/`, `geo/qgis/`)

#### 5. Legacy Core (`scripts/`)
- **`classification_pipeline.py`** - main ML pipeline (LearnModel, ClassifyImage, ConfusionMatrix)
- **Feature modules**: accuracy_index, function_dataraster, progress_bar
- **Advanced features**: optimization/optuna_optimizer, explainability/shap_explainer, sampling/smote_sampler

### Critical UI Components

#### Guided Workflow Widget (`ui/classification_workflow_ui.py`)
- **ClassificationSetupDialog**: Full wizard for new users
- **QuickClassificationPanel**: Simplified panel for dashboard
- **ClassificationDashboardDock**: Main dockable panel that opens on plugin load
- Contains recipe management, parameter selection, and classification execution

#### Recipe System
- **Recipe schema v2** (`src/dzetsaka/domain/value_objects/recipe_schema_v2.py`): New versioned format
- **Migration**: v1 recipes auto-migrate to v2 on load
- **Storage**: Recipes contain full classification configuration (algorithm, params, preprocessing)

### Dependency Installation Flow

**Recommended Async Approach (QgsTask):**
1. User selects algorithm requiring missing deps (e.g., XGBoost)
2. `plugin._try_install_dependencies_async()` is called
3. Creates `DependencyInstallTask` (extends QgsTask)
4. Submits to QGIS task manager (non-blocking)
5. Runs pip installation in background thread using subprocess
6. Shows completion dialog when finished
7. No UI freezing, user can continue working in QGIS

**Legacy Synchronous Approach (QEventLoop):**
1. `dependency_installer.py::try_install_dependencies()` uses QProcess + QEventLoop
2. Shows `InstallProgressDialog` (non-modal, live output)
3. Tries multiple Python launcher candidates (handles QGIS Python quirks on Windows)
4. Installs full bundle: scikit-learn, xgboost, lightgbm, catboost, optuna, shap, imbalanced-learn
5. Uses runtime constraints to avoid breaking numpy/scipy/pandas versions

**Note:** New code should use `_try_install_dependencies_async()` which follows QGIS best practices.

### Plugin Initialization Flow

1. **QGIS loads plugin** → `__init__.py::classFactory()` → `DzetsakaGUI.__init__()`
2. **Runtime bootstrap** (`runtime_bootstrap.py::initialize_runtime_state()`)
   - Loads settings, checks first installation
   - Sets `_open_welcome_on_init` and `_auto_open_dashboard_on_init` flags
3. **GUI initialization** (`plugin_runtime.py::initGui()` → `ui_init.py::init_gui()`)
   - Registers processing provider
   - Creates menu actions and toolbar icons
   - **Auto-opens dashboard on first install** (QTimer.singleShot 1200ms)
4. **Dashboard opens** (`dashboard_dock.py::open_dashboard_dock()`)

### Classification Execution Flow

1. User configures in dashboard → fills params in `QuickClassificationPanel`
2. Click "Run" → `_emit_config()` → `configReady` signal
3. Dashboard catches signal → `dashboard_execution.py::_on_config_ready()`
4. Validates inputs → calls `plugin_runtime.py::_validate_classification_request()`
5. Executes classification → `scripts/classification_pipeline.py::LearnModel()` + `ClassifyImage()`
6. Generates report bundle (HTML + CSV + heatmaps) if enabled
7. Emits `run_manifest.json` and `trust_card.json` for reproducibility

## Important Conventions

### UI File Editing
- **Generated UI files** (`ui/welcome.py`, `ui/install_progress_dialog.py`): Can be edited directly
- **Custom widgets** (`ui/classification_workflow_ui.py`): Hand-coded, NOT generated
- QGIS uses PyQt6 with compatibility for PyQt5 (use try/except for enum values)

### Reading Files
When modifying UI code in `ui/classification_workflow_ui.py`:
1. **ALWAYS read the file first** before editing (it's 4000+ lines)
2. Use offset/limit for large files or grep for specific sections
3. Search for class definitions, combo boxes, or methods before editing

### Tool Usage for File Operations
- Use **Edit** tool for modifying existing files (not sed/awk)
- Use **Read** tool for viewing files (not cat/head/tail)
- Use **Grep** tool for searching content (not bash grep)
- Use **Glob** tool for finding files (not find/ls)

### Code Style
- **Formatting**: Ruff (primary), Black (fallback)
- **Line length**: 120 characters
- **Imports**: Order by stdlib → third-party → qgis → local
- **Naming**: Follow QGIS conventions (camelCase for Qt widgets, snake_case for Python)
- **Docstrings**: Google style with type hints

### Algorithm Support
Each algorithm has a classifier code:
- `1`: GMM, `2`: SVM, `3`: RF, `4`: KNN
- `5`: XGB, `6`: LGB, `7`: CB
- `8`: ET, `9`: GBC, `10`: LR, `11`: NB, `12`: MLP

Check `classifier_config.py` for capability matrix (which algorithms need which dependencies).

## Testing Strategy

### Test Organization
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Multi-component tests
├── algorithms/     # Algorithm-specific tests
└── fixtures/       # Shared test data
```

### Test Markers
Use pytest markers to categorize tests:
```python
@pytest.mark.sklearn      # Requires scikit-learn
@pytest.mark.xgboost      # Requires XGBoost
@pytest.mark.qgis         # Requires QGIS environment
@pytest.mark.slow         # Long-running tests
```

### Running Subset of Tests
```bash
pytest -m "not qgis"              # Skip QGIS-dependent tests
pytest -m "sklearn and not slow"  # Fast sklearn tests
pytest tests/unit/test_recipe_schema_v2.py::test_v1_migration  # Single test
```

## Common Patterns

### Adding a New Algorithm
1. Add entry to `classifier_config.py::CLASSIFIER_NAMES` and dependency checks
2. Add to `scripts/classification_pipeline.py::LearnModel()` train logic
3. Update `_CLASSIFIER_META` in `ui/classification_workflow_ui.py`
4. Add dependency to `dependency_catalog.py::FULL_DEPENDENCY_BUNDLE`
5. Write unit tests in `tests/algorithms/`

### Adding a New UI Parameter
1. Add widget in `ui/classification_workflow_ui.py::QuickClassificationPanel.__init__()`
2. Connect signal if interactive
3. Read value in `_quick_extra_params()` or `_emit_config()`
4. Update `dashboard_execution.py::_on_config_ready()` to pass to classification
5. Handle in `classification_pipeline.py::LearnModel()` or `ClassifyImage()`

### Recipe Workflow
1. **Save recipe**: User clicks "Save Current…" → serializes current UI state to JSON
2. **Load recipe**: User selects from combo → `_apply_selected_recipe()` → populates UI
3. **Migration**: Old v1 recipes auto-convert to v2 schema on load
4. **Validation**: Schema validation before save, dependency checks before run

## Key Files Reference

- **Main plugin entry**: `__init__.py::classFactory()`
- **Plugin runtime**: `src/dzetsaka/presentation/qgis/plugin_runtime.py::DzetsakaGUI`
- **Classification engine**: `scripts/classification_pipeline.py`
- **Dashboard UI**: `ui/classification_workflow_ui.py`
- **Dependency installer**: `src/dzetsaka/presentation/qgis/dependency_installer.py`
- **Recipe schema**: `src/dzetsaka/domain/value_objects/recipe_schema_v2.py`
- **Config**: `pyproject.toml`, `classifier_config.py`

## QGIS Best Practices Compliance

### Background Processing ✅
- **Classification tasks**: Use `QgsTask` via `ClassificationTask` (no UI freezing)
- **Dependency installation**:
  - New: `DependencyInstallTask` using subprocess in background thread
  - Legacy: QEventLoop + QProcess (still available for compatibility)
- **Task cancellation**: Properly supports `isCanceled()` checks
- **Progress reporting**: Uses feedback objects and `setProgress()`

### Signal Management ✅
- All signals properly disconnected in `unload_plugin()`
- Prevents memory leaks and dangling connections
- Handles Qt5/Qt6 compatibility for signal disconnection

### Resource Cleanup ✅
- Proper widget closing before unload
- Processing provider deregistration
- Toolbar icon and menu cleanup
- Signal disconnection (exception-safe)

### Processing Framework ✅
- Implements `QgsProcessingAlgorithm` for all processing tools
- Proper parameter handling and feedback integration
- Error handling with user-friendly messages

## Current Development Focus

See `PLAN.md` for the active roadmap. Current phase: implementing recipe schema v2, trust artifacts, and local recommendation system for reproducible, shareable classification workflows.


