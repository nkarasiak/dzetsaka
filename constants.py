"""Central configuration constants for dzetsaka.

This module contains all magic numbers and configuration constants used
throughout the dzetsaka plugin. Centralizing these values makes the codebase
more maintainable and allows for easy configuration changes.
"""

# Memory Management
MEMORY_LIMIT_MB = 512  # Maximum memory for raster processing blocks
MEMORY_SAFETY_MARGIN = 0.9  # Use 90% of available memory as safety margin

# Cross-Validation Configuration
DEFAULT_CV_FOLDS = {
    "RF": 5,  # Random Forest uses 5-fold CV
    "SVM": 3,  # SVM uses 3-fold CV
    "KNN": 3,  # K-Nearest Neighbors uses 3-fold CV
    "XGB": 3,  # XGBoost uses 3-fold CV
    "CB": 3,  # CatBoost uses 3-fold CV
    "ET": 3,  # Extra Trees uses 3-fold CV
    "GBC": 3,  # Gradient Boosting uses 3-fold CV
    "LR": 3,  # Logistic Regression uses 3-fold CV
    "MLP": 3,  # Multi-Layer Perceptron uses 3-fold CV
    "NB": 3,  # Naive Bayes uses 3-fold CV
}

# Training Validation
MIN_SAMPLES_PER_CLASS = 5  # Minimum samples required per class for training
MIN_TOTAL_SAMPLES = 10  # Minimum total samples for any ML operation
DEFAULT_TRAIN_SPLIT_PERCENT = 50  # Default percentage for train/validation split

# Raster Processing
DEFAULT_BLOCK_SIZE = 256  # Default block size for raster processing (pixels)
MIN_BLOCK_SIZE = 64  # Minimum block size
MAX_BLOCK_SIZE = 2048  # Maximum block size
NODATA_VALUE = -9999  # Standard NoData value for output rasters

# Progress Reporting
PROGRESS_UPDATE_INTERVAL = 5  # Update progress every N% completion
MIN_OPERATION_TIME_FOR_PROGRESS = 5  # Show progress bar for operations > 5 seconds

# UI Configuration
MAX_RECENT_RECIPES = 10  # Maximum number of recent recipes to remember
MAX_COMPARISON_RESULTS = 20  # Maximum number of results in comparison panel
ERROR_DIALOG_DEDUPE_TIMEOUT = 5000  # Milliseconds to deduplicate error dialogs

# File Extensions
MODEL_FILE_EXTENSION = ".npz"  # Extension for saved models
MATRIX_FILE_EXTENSION = ".csv"  # Extension for confusion matrices
RECIPE_FILE_EXTENSION = ".json"  # Extension for workflow recipe files

# Hyperparameter Optimization
OPTUNA_N_TRIALS_DEFAULT = 100  # Default number of Optuna trials
OPTUNA_TIMEOUT_DEFAULT = 3600  # Default timeout in seconds (1 hour)
OPTUNA_N_JOBS_DEFAULT = -1  # Use all available cores

# SMOTE Sampling
SMOTE_K_NEIGHBORS_DEFAULT = 5  # Default k_neighbors for SMOTE
SMOTE_SAMPLING_STRATEGY_DEFAULT = "auto"  # Default sampling strategy

# Logging
LOG_RETENTION_DAYS = 7  # Days to retain log files
MAX_LOG_FILE_SIZE_MB = 10  # Maximum size for individual log files

# CRS Handling
CRS_TOLERANCE_METERS = 0.01  # Tolerance for CRS comparison in meters

# Default Algorithm Settings
DEFAULT_ALGORITHM = "RF"  # Random Forest as default
RANDOM_STATE = 42  # Fixed random state for reproducibility

# Error Messages
ERROR_MSG_MISSING_SKLEARN = (
    "Missing dependency: scikit-learn. "
    "Please install it using the plugin's dependency installer or run: "
    "pip install scikit-learn"
)
ERROR_MSG_MISSING_XGBOOST = (
    "Missing dependency: xgboost. Please install it using the plugin's dependency installer or run: pip install xgboost"
)
ERROR_MSG_MISSING_CATBOOST = (
    "Missing dependency: catboost. "
    "Please install it using the plugin's dependency installer or run: "
    "pip install catboost"
)

# GitHub Integration
GITHUB_REPO = "nkarasiak/dzetsaka"
GITHUB_ISSUES_URL = f"https://github.com/{GITHUB_REPO}/issues"
GITHUB_NEW_ISSUE_URL = f"{GITHUB_ISSUES_URL}/new"

# Documentation URLs (GitHub)
DOCS_BASE_URL = "https://github.com/nkarasiak/dzetsaka/blob/master/docs"
DOCS_USER_GUIDE_URL = f"{DOCS_BASE_URL}/USER_GUIDE.md"
DOCS_ALGORITHMS_URL = f"{DOCS_BASE_URL}/ALGORITHMS.md"
DOCS_TROUBLESHOOTING_URL = f"{DOCS_USER_GUIDE_URL}#common-issues"
DOCS_QUICKSTART_URL = f"{DOCS_USER_GUIDE_URL}#quick-start"

# Version Information
PLUGIN_MIN_QGIS_VERSION = "3.0"  # Minimum QGIS version required
PLUGIN_MIN_PYTHON_VERSION = "3.8"  # Minimum Python version required
