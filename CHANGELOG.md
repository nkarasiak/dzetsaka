# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.2] - 2025-08-28

### Fixed
- 🐛 **Import errors**: Fixed `splitTrain` and `trainAlgorithm` class name imports in dzetsaka_provider.py
- 🐛 **Progress bar module**: Fixed `progress_bar.progressBar` attribute error by correcting case to `progress_bar.ProgressBar`
- 🐛 **Missing toolbar icons**: Restored Qt resources import in dzetsaka.py to properly load embedded icon data
- 🐛 **Resource paths**: Fixed incorrect resource path in sieve_area.py (`:/plugins/dzetsaka/icon.png` → `:/plugins/dzetsaka/img/icon.png`)
- 🧹 **Code quality**: Applied ruff linting fixes including docstrings for dummy classes and proper exception chaining

### Improved
- 📝 **Documentation**: Added proper docstrings to sklearn fallback classes (BaseEstimator, ClassifierMixin, LabelEncoder)
- 🔧 **Error handling**: Enhanced exception chaining with `raise ... from e` pattern for better debugging

## [4.2.0] - 2025-07-16

### Added
- 🎯 **7 new machine learning algorithms**: XGBoost, LightGBM, Extra Trees, Gradient Boosting, Logistic Regression, Naive Bayes, Multi-layer Perceptron
- 🚀 **Automatic dependency installation system**: One-click install of scikit-learn, XGBoost, LightGBM with real-time progress
- ⚡ **Automatic hyperparameter optimization**: Cross-validation grid search for all algorithms with optimized parameter ranges
- 🔧 **Smart sparse label handling**: Automatically handles missing class labels (e.g., classes 0,1,3 with missing 2)
- 📊 **GitHub issue integration**: Automatic error reporting templates with system information for better bug reports
- 📈 **Real-time installation tracking**: Live pip output streaming in QGIS log with progress indicators

### Improved  
- 🎨 **Better log levels**: Changed from WARNING to INFO for normal operations, reducing user confusion
- 📝 **Enhanced error handling**: Specific exception types with detailed user guidance and recovery suggestions
- 🏃 **Optimized hyperparameter grids**: Reduced grid search combinations by 60-70% while maintaining coverage
- 💾 **Model serialization**: Fixed pickling issues for XGBoost/LightGBM wrapper classes
- 🔄 **Parameter delegation**: Proper hyperparameter passing for wrapped algorithms

### Fixed
- 🐛 **XGBoost/LightGBM label encoding**: Automatic sparse label handling with proper inverse transformation
- 🐛 **Model file handling**: Comprehensive error handling for corrupted/missing model files  
- 🐛 **Import compatibility**: Fixed `Qgs` vs `Qgis` import issues for proper log levels
- 🐛 **Wrapper initialization**: Resolved parameter delegation issues during hyperparameter optimization

### Technical Details
- **Global wrapper classes**: Moved XGBLabelWrapper and LGBLabelWrapper to module level for proper serialization
- **Enhanced pip integration**: Cross-platform Python executable detection with QGIS environment handling
- **Comprehensive validation**: Added specific error types (FileNotFoundError, PickleError, ValueError) with actionable messages

## [4.1.2] - 2025-07-16

### Added
- **Enhanced error reporting** - Added GitHub issue reporting guidance for unexpected errors
- **Debug information generator** - Automatic logging of system configuration for troubleshooting
- **Comprehensive error handling** - Added try-catch wrappers for both training and classification processes

### Improved
- **Error messages** - Now include direct links to GitHub issues with guidance on what information to include
- **System diagnostics** - Automatic detection and logging of QGIS, Python, OS versions and library availability

## [4.1.1] - 2025-07-16

### Fixed
- **UI classifier integration** - Fixed UnboundLocalError for `inClassifier` variable during classification
- **Centralized classifier configuration** - Created `classifier_config.py` to eliminate duplication across codebase
- **Enhanced dependency validation** - Improved UI validation for sklearn, XGBoost, and LightGBM dependencies
- **Parameter mapping fixes** - Updated function calls to use new parameter names (`raster_path`, `vector_path`, etc.)
- **UI consistency** - Fixed missing classifier items in settings dropdown (now shows all 11 algorithms)
- **Validation scope** - Updated all processing algorithms to support the full classifier set

### Changed
- **Single source of truth** - All classifier lists now reference `classifier_config.py`
- **Better error messages** - More informative dependency error dialogs with installation instructions
- **Robust detection** - Improved library availability checking for all supported algorithms

## [4.1.0] - 2025-07-15

### Added
- **New backward compatibility decorator system** - Elegant solution for handling deprecated parameters
- **Extended classifier support** - Added 7 new machine learning algorithms:
  - XGBoost (XGB) - High-performance gradient boosting
  - LightGBM (LGB) - Fast gradient boosting with lower memory usage
  - Extra Trees (ET) - Extremely randomized trees for variance reduction
  - Gradient Boosting Classifier (GBC) - Sklearn gradient boosting implementation
  - Logistic Regression (LR) - Linear classification with regularization
  - Gaussian Naive Bayes (NB) - Probabilistic classifier
  - Multi-layer Perceptron (MLP) - Neural network classifier
- Comprehensive parameter migration guide (`PARAMETER_MIGRATION_GUIDE.md`)
- Enhanced sklearn validation with detailed algorithm availability checking
- Configuration constants for classifier settings and memory optimization
- Extensive type hints throughout `scripts/mainfunction.py`
- Memory optimization for large multi-band image processing with adaptive block sizing
- Comprehensive test coverage for sklearn validation functionality

### Changed
- **BREAKING CHANGE (with seamless backward compatibility)**: Modernized parameter naming convention
  - Removed Hungarian notation prefixes for cleaner, more intuitive API
  - `inRaster` → `raster_path`, `inVector` → `vector_path`, `inField` → `class_field`
  - `outModel` → `model_path`, `inSplit` → `split_config`, `inSeed` → `random_seed`
  - `outMatrix` → `matrix_path`, `inClassifier` → `classifier`
  - `inMask` → `mask_path`, `outRaster` → `output_path`, `inShape` → `shapefile_path`
- **Architectural improvements**:
  - Implemented `@backward_compatible` decorator to eliminate parameter duplication
  - Refactored 1279-line `learnModel.__init__` into focused, single-responsibility methods
  - Replaced manual parameter resolution with elegant decorator-based approach
  - Streamlined function signatures by removing deprecated parameter definitions
- **Code quality enhancements**:
  - Applied ruff formatting and linting for consistent code style
  - Enhanced docstrings with comprehensive parameter documentation
  - Improved class inheritance patterns (removed outdated `object` inheritance)

### Enhanced
- **Sklearn integration**: Robust validation system with detailed error messages and installation instructions
- **Classifier ecosystem**: Automatic dependency detection for XGBoost and LightGBM packages
- **Parameter optimization**: Comprehensive hyperparameter grids for all new classifiers
- **UI integration**: Enhanced QGIS interface with all 11 classifiers in dropdown menus and real-time validation feedback
- **Error handling**: Specific exception types with contextual error messages and troubleshooting guidance
- **Memory management**: Optimized processing for large datasets with adaptive resource allocation
- **Developer experience**: Cleaner API with better type hints and documentation

### Fixed
- Resolved all ruff linting issues (undefined variables, ambiguous names, unused imports)
- Enhanced resource cleanup and file handle management
- Improved model loading with comprehensive validation and error recovery
- Fixed potential memory leaks in image processing workflows
- Corrected parameter scope issues in cross-validation methods

### Technical Architecture
- **Decorator pattern**: Centralized backward compatibility logic with automatic deprecation warnings
- **Method decomposition**: Split large methods following single responsibility principle
- **Helper methods**: `_validate_inputs()`, `_load_and_prepare_data()`, `_setup_progress_feedback()`
- **Configuration constants**: Centralized settings for classifiers and memory management
- **Validation system**: Comprehensive sklearn availability and algorithm compatibility checking

### Backward Compatibility
- **100% compatible**: All existing code continues to work without modification
- **Automatic parameter mapping**: Old parameter names transparently mapped to new ones
- **Deprecation warnings**: Helpful guidance for migrating to new parameter names
- **Gradual migration**: Users can adopt new names at their own pace
- Implemented automatic parameter resolution with precedence rules
- Memory optimization reduces usage by up to 50% for large multi-band images
- Enhanced cross-validation parameter validation and adjustment

## [4.0.0] - 2025-07-12

### Added
- New environment initialization command
- Comprehensive error handling improvements
- Enhanced documentation across the codebase

### Fixed
- Improve error handling and add comprehensive documentation
- Resolve multiple critical issues in classification and processing
- Fixed various typos in codebase

### Changed
- **BREAKING CHANGE**: Major version bump due to significant architectural improvements
- Enhanced stability and reliability of classification algorithms
- Improved processing workflows and error handling

## [3.70] - 2024-XX-XX

### Fixed
- Fix bug with new gdal import from osgeo

## [3.7] - 2024-XX-XX

### Fixed
- Fix bug #31

## [3.64] - 2024-XX-XX

### Added
- Add closing filter in the processing toolbox
- Median and closing filter functionality

## [3.63] - 2024-XX-XX

### Fixed
- Fix bug in train algorithm (split was percent of train not of validation)

## [3.62] - 2024-XX-XX

### Fixed
- Fix bug when loading cursor was not removed after unsuccessful learning

## [3.61] - 2024-XX-XX

### Fixed
- Fix bug #19 with self.addAlgorithm(alg)

## [3.6] - 2024-XX-XX

### Added
- Add confidence map in processing
- Add median filter and shannon entropy
- Move dzetsaka icons to extension toolbar

### Fixed
- Fix bug with GMM confidence map

## [3.5.1] - 2023-XX-XX

### Fixed
- Fix bug in algorithm with vector files

## [3.5] - 2023-XX-XX

### Fixed
- Fix bug with gpkg files in train algorithm provider

### Changed
- Update to install scikit-learn on windows

## [3.4.8] - 2023-XX-XX

### Fixed
- Fix bug when classes >44

## [3.4.7] - 2023-XX-XX

### Added
- Support more than 255 classes to predict

## [3.4.6] - 2023-XX-XX

### Fixed
- Fixes in processing

## [3.4.5] - 2023-XX-XX

### Fixed
- Fix bug #17, if model is loaded, do not search for a vector file

## [3.4.4] - 2023-XX-XX

### Added
- Add version modification

## [3.4.3] - 2023-XX-XX

### Added
- Create LICENSE

## [3.4.2] - 2023-XX-XX

### Changed
- Version update

## [3.4.1] - 2023-XX-XX

### Changed
- Version update

## [3.4] - 2023-XX-XX

### Added
- Welcome help functionality

## [3.3.1] - 2023-XX-XX

### Changed
- Use gdal instead of osgeo.gdal

## [3.3] - 2023-XX-XX

### Fixed
- Fix bugs with processing toolbox

### Changed
- Various improvements

## [3.2] - 2023-XX-XX

### Changed
- Beta version with experimental processing providers

## [3.1] - 2023-XX-XX

### Added
- Experimental processing providers

## [3.0.2] - 2022-XX-XX

### Added
- Progress bar for GUI

### Fixed
- Try correcting closing dock

## [3.0.1] - 2022-XX-XX

### Fixed
- Minor fixes

## [3.0.0] - 2022-XX-XX

### Changed
- **BREAKING CHANGE**: Major version release - dzetsaka v3.0
- Replace scipy by numpy
- Major fixes and new tools for python3

### Added
- New processing functions
- Confirmation box if different projection
- Manage push message
- Rewrite feedback & resample SITS

## [2.5.1] - 2021-XX-XX

### Fixed
- Correct bug with confusion matrix (if split<100%)

## [2.5] - 2021-XX-XX

### Fixed
- v2.4.5 verification step and bug split

## [2.4.4] - 2021-XX-XX

### Changed
- Adapt for model_selection in sklearn >= 0.18
- KFold v.18 & .20

## [2.4.3] - 2021-XX-XX

### Changed
- Sklearn validation updates

## [2.4.2] - 2021-XX-XX

### Changed
- Version update

## [2.4.1] - 2021-XX-XX

### Added
- Split train/validation functionality

## [2.4] - 2021-XX-XX

### Added
- DTW (Dynamic Time Warping) functionality

## [2.3.1] - 2020-XX-XX

### Changed
- Version update

## [2.3] - 2020-XX-XX

### Changed
- Version update

## [2.2] - 2020-XX-XX

### Added
- Confidence functionality
- Saved directory feature

## [2.1.2] - 2020-XX-XX

### Fixed
- Correct bug saving classification
- Bug with output file

## [2.1.1] - 2020-XX-XX

### Changed
- Version update

## [2.1] - 2020-XX-XX

### Added
- Processing support

## [2.0.3] - 2019-XX-XX

### Changed
- Version update

## [2.0.2] - 2019-XX-XX

### Fixed
- Bug with loading model

## [2.0.1] - 2019-XX-XX

### Changed
- Version update

## [2.0] - 2019-XX-XX

### Changed
- **BREAKING CHANGE**: Major version release

## [1.2] - 2018-XX-XX

### Changed
- Version update

## [1.1] - 2018-XX-XX

### Added
- Processing error management

## [1.0.1] - 2018-XX-XX

### Added
- Mask support

## [1.0] - 2018-XX-XX

### Added
- Verification process
- Initial stable release

## [0.1] - 2018-XX-XX

### Added
- Initial release of dzetsaka classification tool