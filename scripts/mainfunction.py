"""Core machine learning functions for dzetsaka classification.

This module contains the main implementation of machine learning algorithms,
model training, image classification, and accuracy assessment for dzetsaka.
It provides a comprehensive interface between the QGIS GUI and the underlying
classification algorithms.

Key Components:
--------------
- LearnModel: Train classification models with various algorithms
- ClassifyImage: Classify entire raster images using trained models
- ConfusionMatrix: Compute accuracy statistics from classifications
- Backward compatibility decorators for parameter name changes
- Advanced cross-validation methods (SLOO, STAND)
- Label encoding wrappers for XGBoost and LightGBM

Supported Algorithms:
--------------------
- GMM: Gaussian Mixture Model (built-in)
- RF: Random Forest (sklearn)
- SVM: Support Vector Machine (sklearn)
- KNN: K-Nearest Neighbors (sklearn)
- XGB: XGBoost (requires xgboost package)
- LGB: LightGBM (requires lightgbm package)
- ET: Extra Trees (sklearn)
- GBC: Gradient Boosting Classifier (sklearn)
- LR: Logistic Regression (sklearn)
- NB: Gaussian Naive Bayes (sklearn)
- MLP: Multi-layer Perceptron (sklearn)

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>
Based on original work by Mathieu Fauvel

License:
--------
GNU General Public License v2.0 or later

"""

try:
    # when using it from QGIS 3
    from qgis.core import QgsMessageLog

    from . import accuracy_index as ai
    from . import function_dataraster as dataraster
    from . import progress_bar
except BaseException:
    import accuracy_index as ai
    import function_dataraster as dataraster
    import progress_bar

import contextlib
import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from osgeo import gdal, ogr

# Try to import Optuna optimizer (optional)
try:
    from .optimization.optuna_optimizer import OptunaOptimizer

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None

# Try to import SHAP explainer (optional)
try:
    from .explainability.shap_explainer import ModelExplainer, SHAP_AVAILABLE
except ImportError:
    SHAP_AVAILABLE = False
    ModelExplainer = None

# Try to import sampling techniques (Phase 3)
try:
    from .sampling.smote_sampler import SMOTESampler, apply_smote_if_needed, IMBLEARN_AVAILABLE
    from .sampling.class_weights import (
        compute_class_weights,
        apply_class_weights_to_model,
        compute_sample_weights,
        recommend_strategy,
    )

    SAMPLING_AVAILABLE = True
except ImportError:
    SAMPLING_AVAILABLE = False
    IMBLEARN_AVAILABLE = False
    SMOTESampler = None
    apply_smote_if_needed = None
    compute_class_weights = None

# Try to import validation techniques (Phase 3)
try:
    from .validation.nested_cv import NestedCrossValidator, perform_nested_cv
    from .validation.metrics import ValidationMetrics, create_classification_summary

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    NestedCrossValidator = None
    ValidationMetrics = None

# Import sklearn modules (optional dependency)
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    # Create dummy classes when sklearn is not available
    class BaseEstimator:
        """Dummy BaseEstimator class when sklearn is not available."""

    class ClassifierMixin:
        """Dummy ClassifierMixin class when sklearn is not available."""

    class LabelEncoder:
        """Dummy LabelEncoder class when sklearn is not available."""

        def fit(self, y):
            """Dummy fit method."""
            return self

        def transform(self, y):
            """Dummy transform method."""
            return y

        def fit_transform(self, y):
            """Dummy fit_transform method."""
            return y

    confusion_matrix = None
    SKLEARN_AVAILABLE = False

from .. import classifier_config


# Backward compatibility decorator
def backward_compatible(**parameter_mapping):
    """Decorator to handle backward compatibility for function parameters.

    Parameters
    ----------
    **parameter_mapping : dict
        Mapping from old parameter names to new parameter names
        Example: backward_compatible(inRaster='raster_path', inVector='vector_path')

    """

    def decorator(func):
        import functools
        import warnings

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a copy of kwargs to avoid modifying the original
            new_kwargs = kwargs.copy()

            # Process parameter mapping
            for old_param, new_param in parameter_mapping.items():
                if old_param in kwargs and new_param not in kwargs:
                    # Move old parameter to new parameter name
                    new_kwargs[new_param] = new_kwargs.pop(old_param)
                    warnings.warn(
                        f"Parameter '{old_param}' is deprecated. Use '{new_param}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                elif old_param in kwargs and new_param in kwargs:
                    # Both old and new parameters provided - remove old one and warn
                    new_kwargs.pop(old_param)
                    warnings.warn(
                        f"Both '{old_param}' and '{new_param}' provided. Using '{new_param}' and ignoring '{old_param}'.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            return func(*args, **new_kwargs)

        return wrapper

    return decorator


# Label encoding wrapper for XGBoost and LightGBM


class XGBLabelWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for XGBoost that handles sparse label encoding/decoding."""

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.label_encoder = LabelEncoder()
        self.xgb_classifier = None

    def fit(self, X, y):
        """Fit the XGBoost classifier with label encoding."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not found. Install with: pip install xgboost") from None

        y_encoded = self.label_encoder.fit_transform(y)
        self.xgb_classifier = XGBClassifier(**self.xgb_params)
        self.xgb_classifier.fit(X, y_encoded)
        return self

    def predict(self, X):
        """Predict labels using the fitted classifier."""
        y_encoded = self.xgb_classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        """Predict class probabilities using the fitted classifier."""
        return self.xgb_classifier.predict_proba(X)

    @property
    def classes_(self):
        """Get class labels."""
        return self.label_encoder.classes_

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Return XGBoost parameters directly
        return self.xgb_params.copy()

    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.xgb_params.update(params)
        if self.xgb_classifier is not None:
            self.xgb_classifier.set_params(**params)
        return self


class LGBLabelWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for LightGBM that handles sparse label encoding/decoding."""

    def __init__(self, **lgb_params):
        self.lgb_params = lgb_params
        self.label_encoder = LabelEncoder()
        self.lgb_classifier = None

    def fit(self, X, y):
        """Fit the LightGBM classifier with label encoding."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("LightGBM not found. Install with: pip install lightgbm") from None

        y_encoded = self.label_encoder.fit_transform(y)
        self.lgb_classifier = LGBMClassifier(**self.lgb_params)
        self.lgb_classifier.fit(X, y_encoded)
        return self

    def predict(self, X):
        """Predict class labels for samples."""
        y_encoded = self.lgb_classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        """Predict class probabilities for samples."""
        return self.lgb_classifier.predict_proba(X)

    @property
    def classes_(self):
        """Get class labels."""
        return self.label_encoder.classes_

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        # Return LightGBM parameters directly
        return self.lgb_params.copy()

    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.lgb_params.update(params)
        if self.lgb_classifier is not None:
            self.lgb_classifier.set_params(**params)
        return self


# Configuration constants
CLASSIFIER_CONFIGS = {
    "RF": {
        "param_grid": {
            "n_estimators": [100],
            "max_features": lambda x_shape: range(1, max(2, x_shape), max(1, int(x_shape / 3))),
        },
        "n_splits": 3,
    },
    "SVM": {
        "param_grid": {"gamma": 2.0 ** np.arange(-2, 3), "C": 10.0 ** np.arange(-1, 3)},
        "n_splits": 3,
    },
    "KNN": {"param_grid": {"n_neighbors": [1, 3, 10]}, "n_splits": 3},
    "XGB": {
        "param_grid": {
            "n_estimators": [100],
            "max_depth": [9],
            "learning_rate": [0.01],
        },
        "n_splits": 3,
    },
    "LGB": {
        "param_grid": {
            "n_estimators": [50, 200],
            "num_leaves": [31, 100],
            "learning_rate": [0.01, 0.2],
        },
        "n_splits": 3,
    },
    "ET": {
        "param_grid": {
            "n_estimators": [50, 200],
            "max_features": lambda x_shape: range(1, max(2, x_shape), max(1, int(x_shape / 2))),
        },
        "n_splits": 3,
    },
    "GBC": {
        "param_grid": {
            "n_estimators": [50, 200],
            "max_depth": [3, 7],
            "learning_rate": [0.01, 0.2],
        },
        "n_splits": 3,
    },
    "LR": {
        "param_grid": {
            "C": 10.0 ** np.arange(-2, 3, 2),  # [-2, 0, 2] -> [0.01, 1, 100]
            "solver": ["liblinear", "lbfgs"],
        },
        "n_splits": 3,
    },
    "NB": {
        "param_grid": {"var_smoothing": 10.0 ** np.arange(-9, -3, 3)},  # [-9, -6, -3]
        "n_splits": 3,
    },
    "MLP": {
        "param_grid": {
            "hidden_layer_sizes": [(50,), (100, 50)],  # Keep simple and complex
            "alpha": [0.0001, 0.01],  # Keep low and high regularization
            "learning_rate_init": [0.001, 0.01],
        },
        "n_splits": 3,
    },
}

MAX_MEMORY_MB = 512
MIN_CROSS_VALIDATION_SPLITS = 2


class LearnModel:
    """Machine learning model training class for dzetsaka classification."""

    @backward_compatible(
        inRaster="raster_path",
        inVector="vector_path",
        inField="class_field",
        outModel="model_path",
        inSplit="split_config",
        inSeed="random_seed",
        outMatrix="matrix_path",
        inClassifier="classifier",
    )
    def __init__(
        self,
        raster_path: Union[str, np.ndarray] = None,
        vector_path: Union[str, np.ndarray] = None,
        class_field: str = "Class",
        model_path: Optional[str] = None,
        split_config: Union[int, float, str] = 100,
        random_seed: int = 0,
        matrix_path: Optional[str] = None,
        classifier: str = "GMM",
        extraParam: Optional[Dict[str, Any]] = None,
        feedback=None,
    ):
        """Learn model with a shapefile and a raster image.

        Parameters
        ----------
        raster_path : str or np.ndarray
            Filtered image path or numpy array
        vector_path : str or np.ndarray
            Training shapefile path or numpy array
        class_field : str, default="Class"
            Column name where class numbers are stored
        split_config : int, float, or str, default=100
            Training split percentage, 'SLOO', or 'STAND'
        random_seed : int, default=0
            Random seed for reproducibility
        model_path : str, optional
            Model output file path
        matrix_path : str, optional
            Confusion matrix output file path
        classifier : str, default="GMM"
            Classifier type. Available options:
            - 'GMM': Gaussian Mixture Model (built-in, no dependencies)
            - 'RF': Random Forest (sklearn)
            - 'SVM': Support Vector Machine (sklearn)
            - 'KNN': K-Nearest Neighbors (sklearn)
            - 'XGB': XGBoost (requires: pip install xgboost)
            - 'LGB': LightGBM (requires: pip install lightgbm)
            - 'ET': Extra Trees (sklearn)
            - 'GBC': Gradient Boosting Classifier (sklearn)
            - 'LR': Logistic Regression (sklearn)
            - 'NB': Gaussian Naive Bayes (sklearn)
            - 'MLP': Multi-layer Perceptron (sklearn)
        extraParam : dict, optional
            Additional parameters for advanced configurations:

            Optimization (Phase 1):
            - 'USE_OPTUNA': bool, default=False
                Use Optuna for hyperparameter optimization (2-10x faster than GridSearchCV)
            - 'OPTUNA_TRIALS': int, default=100
                Number of optimization trials for Optuna

            Explainability (Phase 2):
            - 'COMPUTE_SHAP': bool, default=False
                Compute SHAP feature importance after training (requires shap>=0.41.0)
            - 'SHAP_OUTPUT': str, optional
                Path to save feature importance raster (e.g., 'importance.tif')
            - 'SHAP_SAMPLE_SIZE': int, default=1000
                Number of pixels to sample for SHAP computation

            Imbalance Handling (Phase 3):
            - 'USE_SMOTE': bool, default=False
                Apply SMOTE oversampling for imbalanced datasets (requires imbalanced-learn)
            - 'SMOTE_K_NEIGHBORS': int, default=5
                Number of neighbors for SMOTE
            - 'USE_CLASS_WEIGHTS': bool, default=False
                Apply class weights for cost-sensitive learning
            - 'CLASS_WEIGHT_STRATEGY': str, default='balanced'
                Weight strategy ('balanced', 'custom')
            - 'CUSTOM_CLASS_WEIGHTS': dict, optional
                Custom weights {class_label: weight}

            Validation (Phase 3):
            - 'USE_NESTED_CV': bool, default=False
                Use nested cross-validation for unbiased evaluation
            - 'NESTED_INNER_CV': int, default=3
                Inner CV folds for parameter tuning
            - 'NESTED_OUTER_CV': int, default=5
                Outer CV folds for evaluation

            Other:
            - 'param_grid': dict, optional
                Custom parameter grid for GridSearchCV (overrides defaults)
        feedback : object, optional
            Feedback object for progress reporting

        Returns
        -------
        None
            Stores model in self.model, scaling parameters in self.M and self.m

        Notes
        -----
        Backward compatibility is maintained through the @backward_compatible decorator.
        Old parameter names (inRaster, inVector, etc.) are automatically mapped to new names.

        """
        # Validate required parameters
        if raster_path is None:
            raise ValueError("raster_path is required")
        if vector_path is None:
            raise ValueError("vector_path is required")

        # Initialize and validate parameters
        self._validate_inputs(raster_path, vector_path, classifier, feedback)
        extraParam = extraParam or {}

        # Setup progress tracking
        total = 100 / 10
        progress = self._setup_progress_feedback(feedback)

        # Load and prepare data
        try:
            X, Y, coords, distanceArray, STDs, vector_test_path = self._load_and_prepare_data(
                raster_path,
                vector_path,
                class_field,
                split_config,
                extraParam,
                feedback,
            )

        except Exception as e:
            self._handle_data_loading_error(e, class_field, feedback, progress)
            return None

        [n, d] = X.shape
        C = int(Y.max())
        SPLIT = split_config

        # Cleanup handled in _load_and_prepare_data method
        # os.remove(filename)
        # os.rmdir(temp_folder)

        # Phase 3: Class imbalance handling
        self._handle_class_imbalance(X, Y, classifier, extraParam, feedback)

        # Apply SMOTE oversampling if enabled
        if extraParam.get("USE_SMOTE", False):
            X, Y = self._apply_smote(X, Y, extraParam, feedback)

        # Compute class weights if enabled
        self.class_weights_ = None
        if extraParam.get("USE_CLASS_WEIGHTS", False):
            self.class_weights_ = self._compute_weights(Y, extraParam, feedback)

        [n, d] = X.shape
        C = int(Y.max())

        # Scale the data
        X, M, m = self.scale(X)

        pushFeedback(int(1 * total), feedback=feedback)
        if feedback == "gui":
            progress.prgBar.setValue(10)
        # Learning process take split of groundthruth pixels for training and
        # the remaining for testing

        try:
            if isinstance(SPLIT, (int, float)):
                if SPLIT < 100:
                    # Random selection of the sample
                    x = np.array([]).reshape(0, d)
                    y = np.array([]).reshape(0, 1)
                    xt = np.array([]).reshape(0, d)
                    yt = np.array([]).reshape(0, 1)

                    np.random.seed(random_seed)  # Set the random generator state
                    for i in range(C):
                        t = np.where((i + 1) == Y)[0]
                        nc = t.size
                        ns = int(nc * (SPLIT / float(100)))
                        rp = np.random.permutation(nc)
                        x = np.concatenate((X[t[rp[0:ns]], :], x))
                        xt = np.concatenate((X[t[rp[ns:]], :], xt))
                        y = np.concatenate((Y[t[rp[0:ns]]], y))
                        yt = np.concatenate((Y[t[rp[ns:]]], yt))

                else:
                    x, y = X, Y
                    self.x = x
                    self.y = y
            else:
                x, y = X, Y
                self.x = x
                self.y = y
        except BaseException:
            pushFeedback("Problem while learning if SPLIT <1", feedback=feedback)

        pushFeedback(int(2 * total), feedback=feedback)
        if feedback == "gui":
            progress.prgBar.setValue(20)

        pushFeedback("Starting model training process...", feedback=feedback)
        pushFeedback(
            "Training phase in progress. This may take several minutes depending on your data size and classifier settings. The progress bar may appear to pause during hyperparameter optimization - this is normal.",
            feedback=feedback,
        )

        if feedback == "gui":
            progress.prgBar.setValue(25)
        # Train Classifier
        if classifier == "GMM":
            try:
                from . import gmm_ridge as gmmr
            except BaseException:
                import gmm_ridge as gmmr

            try:
                # tau=10.0**sp.arange(-8,8,0.5)
                model = gmmr.GMMR()
                model.learn(x, y)
                # htau,err = model.cross_validation(x,y,tau)
                # model.tau = htau
            except BaseException:
                pushFeedback("Cannot train with GMM", feedback=feedback)
        else:
            # from sklearn import neighbors
            # from sklearn.svm import SVC
            # from sklearn.ensemble import RandomForestClassifier

            # model_selection = True
            try:
                from sklearn.model_selection import GridSearchCV, StratifiedKFold

                joblib = __import__("joblib")  # Test for joblib dependency
            except ImportError as e:
                if "joblib" in str(e):
                    pushFeedback(
                        "Missing dependency: joblib. Please install with: pip install joblib",
                        feedback=feedback,
                    )
                    return None
                else:
                    pushFeedback(
                        "Missing scikit-learn dependency for {classifier}. Please install with: pip install scikit-learn. Error: {e}",
                        feedback=feedback,
                    )
                    return None

            try:
                if extraParam and "param_algo" in extraParam:
                    param_algo = extraParam["param_algo"]

                # AS Qgis in Windows doensn't manage multiprocessing, force to
                # use 1 thread for not linux system

                if SPLIT == "STAND":
                    label = np.copy(Y)

                    if extraParam:
                        SLOO = extraParam.get("SLOO", False)
                        maxIter = extraParam.get("maxIter", 5)
                    else:
                        SLOO = False
                        maxIter = 5

                    try:
                        from .function_vector import StandCV
                    except ImportError:
                        from function_vector import StandCV

                    rawCV = StandCV(label, STDs, maxIter, SLOO, seed=random_seed)
                    print(rawCV)
                    cvDistance = []
                    for tr, vl in rawCV:
                        # sts.append(stat)
                        cvDistance.append((tr, vl))

                if SPLIT == "SLOO":
                    # Compute CV for Learning later

                    label = np.copy(Y)
                    if extraParam:
                        if "distance" in extraParam:
                            distance = extraParam["distance"]
                        else:
                            pushFeedback("You need distance in extraParam", feedback=feedback)

                        minTrain = float(extraParam["minTrain"]) if "minTrain" in extraParam else -1

                        SLOO = extraParam.get("SLOO", True)

                        maxIter = extraParam.get("maxIter", False)

                        otherLevel = extraParam.get("otherLevel", False)
                    # sts = []
                    cvDistance = []

                    """
                    rawCV = DistanceCV(distanceArray,label,distanceThresold=distance,minTrain=minTrain,SLOO=SLOO,maxIter=maxIter,verbose=False,stats=False)

                    """
                    # feedback.setProgressText('distance is '+str(extraParam['distance']))
                    pushFeedback("label is " + str(label.shape), feedback=feedback)
                    pushFeedback(
                        "distance array shape is " + str(distanceArray.shape),
                        feedback=feedback,
                    )
                    pushFeedback("minTrain is " + str(minTrain), feedback=feedback)
                    pushFeedback("SLOO is " + str(SLOO), feedback=feedback)
                    pushFeedback("maxIter is " + str(maxIter), feedback=feedback)

                    # Import distanceCV dynamically when needed
                    try:
                        from function_vector import DistanceCV
                    except ImportError:
                        from .function_vector import DistanceCV

                    rawCV = DistanceCV(
                        distanceArray,
                        label,
                        distanceThresold=distance,
                        minTrain=minTrain,
                        SLOO=SLOO,
                        maxIter=maxIter,
                        stats=False,
                    )

                    pushFeedback("Computing SLOO Cross Validation", feedback=feedback)

                    for tr, vl in rawCV:
                        pushFeedback("Training size is " + str(tr.shape), feedback=feedback)
                        pushFeedback("Validation size is " + str(vl.shape), feedback=feedback)
                        # sts.append(stat)
                        cvDistance.append((tr, vl))
                    """
                    for tr,vl,stat in rawCV :
                        sts.append(stat)
                        cvDistance.append((tr,vl))
                    """
                    #

                if classifier == "RF":
                    from sklearn.ensemble import RandomForestClassifier

                    config = CLASSIFIER_CONFIGS["RF"]
                    param_grid = config["param_grid"].copy()
                    # Handle lambda function for max_features
                    if callable(param_grid.get("max_features")):
                        try:
                            max_features_range = param_grid["max_features"](x.shape[1])
                            # Convert to list to avoid range object issues
                            param_grid["max_features"] = list(max_features_range)
                            pushFeedback(
                                f"RF max_features range: {param_grid['max_features']}",
                                feedback=feedback,
                            )
                        except Exception as e:
                            pushFeedback(
                                f"Error generating max_features range for RF: {e}",
                                feedback=feedback,
                            )
                            # Fallback to safe values
                            param_grid["max_features"] = [1, min(x.shape[1], 3)]

                    if "param_algo" in locals():
                        classifier = RandomForestClassifier(random_state=random_seed, **param_algo)
                    else:
                        classifier = RandomForestClassifier(random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "SVM":
                    from sklearn.svm import SVC

                    config = CLASSIFIER_CONFIGS["SVM"]
                    param_grid = config["param_grid"]

                    if "param_algo" in locals():
                        classifier = SVC(probability=True, random_state=random_seed, **param_algo)
                        print("Found param algo : " + str(param_algo))
                    else:
                        classifier = SVC(probability=True, kernel="rbf", random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "KNN":
                    from sklearn import neighbors

                    config = CLASSIFIER_CONFIGS["KNN"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = neighbors.KNeighborsClassifier(**param_algo)
                    else:
                        classifier = neighbors.KNeighborsClassifier()

                    n_splits = config["n_splits"]

                elif classifier == "XGB":
                    import importlib.util

                    if importlib.util.find_spec("xgboost") is None:
                        pushFeedback(
                            "XGBoost not found. Install with: pip install xgboost",
                            feedback=feedback,
                        )
                        return None

                    config = CLASSIFIER_CONFIGS["XGB"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = XGBLabelWrapper(
                            random_state=random_seed,
                            eval_metric="logloss",
                            **param_algo,
                        )
                    else:
                        classifier = XGBLabelWrapper(random_state=random_seed, eval_metric="logloss")
                    n_splits = config["n_splits"]

                elif classifier == "LGB":
                    import importlib.util

                    if importlib.util.find_spec("lightgbm") is None:
                        pushFeedback(
                            "LightGBM not found. Install with: pip install lightgbm",
                            feedback=feedback,
                        )
                        return None

                    config = CLASSIFIER_CONFIGS["LGB"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = LGBLabelWrapper(random_state=random_seed, verbose=-1, **param_algo)
                    else:
                        classifier = LGBLabelWrapper(random_state=random_seed, verbose=-1)
                    n_splits = config["n_splits"]

                elif classifier == "ET":
                    from sklearn.ensemble import ExtraTreesClassifier

                    config = CLASSIFIER_CONFIGS["ET"]
                    param_grid = config["param_grid"].copy()
                    # Handle lambda function for max_features
                    if callable(param_grid.get("max_features")):
                        try:
                            max_features_range = param_grid["max_features"](x.shape[1])
                            # Convert to list to avoid range object issues
                            param_grid["max_features"] = list(max_features_range)
                            pushFeedback(
                                f"ET max_features range: {param_grid['max_features']}",
                                feedback=feedback,
                            )
                        except Exception as e:
                            pushFeedback(
                                f"Error generating max_features range for ET: {e}",
                                feedback=feedback,
                            )
                            # Fallback to safe values
                            param_grid["max_features"] = [1, min(x.shape[1], 3)]

                    if "param_algo" in locals():
                        classifier = ExtraTreesClassifier(random_state=random_seed, **param_algo)
                    else:
                        classifier = ExtraTreesClassifier(random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "GBC":
                    from sklearn.ensemble import GradientBoostingClassifier

                    config = CLASSIFIER_CONFIGS["GBC"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = GradientBoostingClassifier(random_state=random_seed, **param_algo)
                    else:
                        classifier = GradientBoostingClassifier(random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "LR":
                    from sklearn.linear_model import LogisticRegression

                    config = CLASSIFIER_CONFIGS["LR"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = LogisticRegression(random_state=random_seed, max_iter=1000, **param_algo)
                    else:
                        classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
                    n_splits = config["n_splits"]

                elif classifier == "NB":
                    from sklearn.naive_bayes import GaussianNB

                    config = CLASSIFIER_CONFIGS["NB"]
                    param_grid = config["param_grid"]
                    classifier = GaussianNB(**param_algo) if "param_algo" in locals() else GaussianNB()
                    n_splits = config["n_splits"]

                elif classifier == "MLP":
                    from sklearn.neural_network import MLPClassifier

                    config = CLASSIFIER_CONFIGS["MLP"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = MLPClassifier(random_state=random_seed, max_iter=500, **param_algo)
                    else:
                        classifier = MLPClassifier(random_state=random_seed, max_iter=500)
                    n_splits = config["n_splits"]

            except ImportError as e:
                pushFeedback(
                    "Import error for classifier " + classifier + ": " + str(e),
                    feedback=feedback,
                )
                if feedback == "gui":
                    progress.reset()
                return None
            except Exception as e:
                pushFeedback(
                    "Error initializing classifier " + classifier + ": " + str(e),
                    feedback=feedback,
                )
                if feedback == "gui":
                    progress.reset()
                return None

            if feedback == "gui":
                progress.prgBar.setValue(30)

            y.shape = (y.size,)

            # Validate training data before proceeding
            if x.shape[0] == 0 or y.shape[0] == 0:
                pushFeedback(
                    "Error: No training data found. Check your training samples.",
                    feedback=feedback,
                )
                if feedback == "gui":
                    progress.reset()
                return None

            if x.shape[0] != y.shape[0]:
                pushFeedback(
                    "Error: Mismatch between feature data and labels. Check your training data.",
                    feedback=feedback,
                )
                if feedback == "gui":
                    progress.reset()
                return None

            # Check for any NaN or infinite values
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                pushFeedback(
                    "Warning: NaN or infinite values detected in training data. These will be handled automatically.",
                    feedback=feedback,
                )
                x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)

            # Check if all classes have sufficient samples for cross-validation
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples = np.min(class_counts)
            if min_samples < n_splits:
                n_splits = max(2, min_samples)
                pushFeedback(
                    f"Adjusting cross-validation splits to {n_splits} due to small class sizes",
                    feedback=feedback,
                )

            # Initialize cross-validation after validation and potential n_splits adjustment
            cv = StratifiedKFold(n_splits=n_splits) if isinstance(SPLIT, int) else cvDistance  # .split(x,y)

            # Check if Optuna should be used for hyperparameter optimization
            use_optuna = extraParam.get("USE_OPTUNA", False) if extraParam else False
            optuna_trials = extraParam.get("OPTUNA_TRIALS", 100) if extraParam else 100

            if use_optuna and OPTUNA_AVAILABLE:
                # Use Optuna for faster hyperparameter optimization
                pushFeedback(
                    f"Using Optuna optimization with {optuna_trials} trials (2-10x faster than GridSearchCV)...",
                    feedback=feedback,
                )

                try:
                    # Get classifier code from classifier name
                    classifier_code = str(inClassifier).upper()

                    # Create Optuna optimizer
                    optimizer = OptunaOptimizer(
                        classifier_code=classifier_code, n_trials=optuna_trials, random_seed=inSeed, verbose=False
                    )

                    # Run optimization
                    best_params = optimizer.optimize(X=x, y=y, cv=cv, scoring="f1_weighted")

                    pushFeedback(
                        f"Optuna optimization completed. Best score: {optimizer.study.best_value:.4f}",
                        feedback=feedback,
                    )
                    pushFeedback(f"Best parameters: {best_params}", feedback=feedback)

                    # Recreate classifier with best parameters
                    if classifier_code == "GMM":
                        from . import gmm_ridge

                        model = gmm_ridge.ridge()
                    elif classifier_code == "RF":
                        from sklearn.ensemble import RandomForestClassifier

                        model = RandomForestClassifier(**best_params)
                    elif classifier_code == "SVM":
                        from sklearn.svm import SVC

                        model = SVC(**best_params, probability=True)
                    elif classifier_code == "KNN":
                        from sklearn.neighbors import KNeighborsClassifier

                        model = KNeighborsClassifier(**best_params)
                    elif classifier_code == "XGB":
                        from xgboost import XGBClassifier

                        model = XGBClassifierWrapper(**best_params)
                    elif classifier_code == "LGB":
                        from lightgbm import LGBMClassifier

                        model = LGBMClassifierWrapper(**best_params)
                    elif classifier_code == "ET":
                        from sklearn.ensemble import ExtraTreesClassifier

                        model = ExtraTreesClassifier(**best_params)
                    elif classifier_code == "GBC":
                        from sklearn.ensemble import GradientBoostingClassifier

                        model = GradientBoostingClassifier(**best_params)
                    elif classifier_code == "LR":
                        from sklearn.linear_model import LogisticRegression

                        model = LogisticRegression(**best_params)
                    elif classifier_code == "NB":
                        from sklearn.naive_bayes import GaussianNB

                        model = GaussianNB(**best_params)
                    elif classifier_code == "MLP":
                        from sklearn.neural_network import MLPClassifier

                        model = MLPClassifier(**best_params)
                    else:
                        pushFeedback(
                            f"Unknown classifier code: {classifier_code}. Falling back to GridSearchCV.",
                            feedback=feedback,
                        )
                        use_optuna = False

                    if use_optuna:
                        # Fit final model with best parameters
                        model.fit(x, y)
                        pushFeedback("Final model training completed with Optuna parameters", feedback=feedback)

                except Exception as e:
                    pushFeedback(
                        f"Optuna optimization failed: {str(e)}. Falling back to GridSearchCV.", feedback=feedback
                    )
                    use_optuna = False

            elif use_optuna and not OPTUNA_AVAILABLE:
                pushFeedback(
                    "Optuna is not installed. Falling back to GridSearchCV. "
                    "Install Optuna with: pip install optuna",
                    feedback=feedback,
                )
                use_optuna = False

            # Use GridSearchCV if Optuna is not used
            if not use_optuna:
                if extraParam and "param_grid" in extraParam:
                    param_grid = extraParam["param_grid"]

                    pushFeedback(
                        "Custom param for Grid Search CV has been found : " + str(param_grid),
                        feedback=feedback,
                    )

                # Provide feedback about potentially long training time for SVM
                if classifier == "SVM":
                    pushFeedback(
                        "Training SVM with GridSearchCV - this may take several minutes...",
                        feedback=feedback,
                    )

                try:
                    grid = GridSearchCV(classifier, param_grid=param_grid, cv=cv)
                    grid.fit(x, y)

                    pushFeedback("GridSearchCV completed, fitting final model...", feedback=feedback)
                    model = grid.best_estimator_
                    model.fit(x, y)
                except MemoryError:
                    pushFeedback(
                        "Memory error during training. Try reducing the image size or using fewer training samples.",
                        feedback=feedback,
                    )
                    if feedback == "gui":
                        progress.reset()
                    return None
                except ValueError as e:
                    pushFeedback(
                        "Data validation error: "
                        + str(e)
                        + ". Check your training data for issues like empty classes or invalid values.",
                        feedback=feedback,
                    )
                    if feedback == "gui":
                        progress.reset()
                    return None
                except Exception as e:
                    pushFeedback("Training error: " + str(e), feedback=feedback)
                    if feedback == "gui":
                        progress.reset()
                    return None

            if isinstance(SPLIT, str):
                CM = []
                testIndex = []
                # Get saveDir from extraParam if available
                saveDir = extraParam.get("saveDir", tempfile.gettempdir())
                for train_index, test_index in cv:
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    X_pred = model.predict(X_test)
                    CM.append(confusion_matrix(y_test, X_pred))
                    testIndex.append(test_index)
                for i, _j in enumerate(CM):
                    if SPLIT == "SLOO":
                        # np.savetxt((saveDir+'matrix/'+str(distance)+'_'+str(inField)+'_'+str(minTrain)+'_'+str(i)+'.csv'),CM[i],delimiter=',',fmt='%.d')
                        np.savetxt(
                            os.path.join(
                                saveDir,
                                "matrix/"
                                + str(distance)
                                + "_"
                                + str(class_field)
                                + "_"
                                + str(minTrain)
                                + "_"
                                + str(i)
                                + ".csv",
                            ),
                            CM[i],
                            delimiter=",",
                            fmt="%.d",
                        )
                        if otherLevel is not False:
                            otherLevelFolder = os.path.join(saveDir, "matrix/level3/")
                            if not os.path.exists(otherLevelFolder):
                                os.makedirs(otherLevelFolder)
                            bigCM = np.zeros([14, 14], dtype=np.byte)

                            arr = CM[i]
                            curLevel = otherLevel[testIndex[i]]
                            curLevel = np.sort(curLevel, axis=0)
                            for lvl in range(curLevel.shape[0]):
                                bigCM[
                                    curLevel.astype(int) - 1,
                                    curLevel[lvl].astype(int) - 1,
                                ] = arr[:, lvl].reshape(-1, 1)
                            np.savetxt(
                                os.path.join(
                                    otherLevelFolder,
                                    str(distance)
                                    + "_"
                                    + str(class_field)
                                    + "_"
                                    + str(minTrain)
                                    + "_"
                                    + str(i)
                                    + ".csv",
                                ),
                                bigCM,
                                delimiter=",",
                                fmt="%.d",
                            )

                    elif SPLIT == "STAND":
                        # np.savetxt((saveDir+'matrix/stand_'+str(inField)+'_'+str(i)+'.csv'),CM[i],delimiter=',',fmt='%.d')
                        np.savetxt(
                            os.path.join(
                                saveDir,
                                "matrix/stand_" + str(class_field) + "_" + str(i) + ".csv",
                            ),
                            CM[i],
                            delimiter=",",
                            fmt="%.d",
                        )

        pushFeedback(int(9 * total), feedback=feedback)

        # Assess the quality of the model
        if feedback == "gui":
            progress.prgBar.setValue(90)

        if (vector_test_path or isinstance(SPLIT, int)) and (SPLIT != 100 or vector_test_path):
            # from sklearn.metrics import cohen_kappa_score,accuracy_score,f1_score
            # if  inClassifier == 'GMM':
            #          = model.predict(xt)[0]
            # else:
            yp = model.predict(xt)
            CONF = ai.ConfusionMatrix()
            CONF.compute_confusion_matrix(yp, yt)

            if matrix_path is not None:
                if not os.path.exists(os.path.dirname(matrix_path)):
                    os.makedirs(os.path.dirname(matrix_path))
                np.savetxt(
                    matrix_path,
                    CONF.confusion_matrix,
                    delimiter=",",
                    header="Columns=prediction,Lines=reference.",
                    fmt="%1.4d",
                )

            if classifier != "GMM":
                for key in param_grid:
                    message = "best " + key + " : " + str(grid.best_params_[key])
                    if feedback == "gui":
                        QgsMessageLog.logMessage(message)
                    elif feedback:
                        feedback.setProgressText(message)
                    else:
                        print(message)

            """
                self.kappa = cohen_kappa_score(yp,yt)
                self.f1 = f1_score(yp,yt,average='micro')
                self.oa = accuracy_score(yp,yt)
                """
            res = {
                "Overall Accuracy": CONF.OA,
                "Kappa": CONF.Kappa,
                "f1": CONF.F1mean,
            }

            for estim in res:
                pushFeedback(estim + " : " + str(res[estim]), feedback=feedback)

        # Update progress after model training completion
        if feedback == "gui":
            progress.prgBar.setValue(95)
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgress(95)

        # Save Tree model
        self.model = model
        self.M = M
        self.m = m

        # SHAP explainability computation (optional)
        if extraParam.get("COMPUTE_SHAP", False):
            self._compute_shap_importance(
                model=model,
                raster_path=raster_path,
                X_train=x if 'x' in locals() else X,
                feature_names=None,  # Will generate Band_1, Band_2, etc.
                shap_output_path=extraParam.get("SHAP_OUTPUT"),
                sample_size=extraParam.get("SHAP_SAMPLE_SIZE", 1000),
                feedback=feedback,
            )

        if model_path is not None:
            # Debug: log what we're saving
            pushFeedback(
                f"Saving model with classifier: {classifier} (type: {type(classifier)})",
                feedback=feedback,
            )
            with open(model_path, "wb") as output:
                pickle.dump([model, M, m, str(classifier)], output)  # Ensure classifier is saved as string

        pushFeedback(int(10 * total), feedback=feedback)
        if feedback == "gui":
            progress.reset()
            progress = None
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgress(100)

    def scale(self, x, M=None, m=None):
        """!@brief Function that standardize the data.

        Input:
            x: the data
            M: the Max vector
            m: the Min vector
        Output:
            x: the standardize data
            M: the Max vector
            m: the Min vector
        """
        [n, d] = x.shape
        if np.float64 != x.dtype.type:
            x = x.astype("float")

        # Initialization of the output
        xs = np.empty_like(x)

        # get the parameters of the scaling
        M, m = np.amax(x, axis=0), np.amin(x, axis=0)
        den = M - m
        for i in range(d):
            if den[i] != 0:
                xs[:, i] = 2 * (x[:, i] - m[i]) / den[i] - 1
            else:
                xs[:, i] = x[:, i]

        return xs, M, m

    def _compute_shap_importance(
        self,
        model: Any,
        raster_path: Union[str, np.ndarray],
        X_train: np.ndarray,
        feature_names: Optional[List[str]],
        shap_output_path: Optional[str],
        sample_size: int,
        feedback,
    ) -> None:
        """Compute SHAP feature importance and optionally create raster output.

        Parameters
        ----------
        model : Any
            Trained classifier model
        raster_path : str or np.ndarray
            Path to raster or numpy array
        X_train : np.ndarray
            Training data for background samples
        feature_names : List[str], optional
            Names of features (bands)
        shap_output_path : str, optional
            Path to save importance raster
        sample_size : int
            Number of samples for SHAP computation
        feedback : object
            Feedback interface for progress reporting

        """
        if not SHAP_AVAILABLE:
            pushFeedback(
                "SHAP is not installed. Install with: pip install shap>=0.41.0",
                feedback=feedback,
            )
            return

        try:
            pushFeedback("Computing SHAP feature importance...", feedback=feedback)

            # Generate feature names if not provided
            n_features = X_train.shape[1]
            if feature_names is None:
                feature_names = [f"Band_{i+1}" for i in range(n_features)]

            # Create ModelExplainer
            explainer = ModelExplainer(
                model=model,
                feature_names=feature_names,
                background_data=X_train[:min(100, len(X_train))],  # Use up to 100 samples as background
            )

            # Compute feature importance on training sample
            sample_for_shap = X_train[:min(sample_size, len(X_train))]
            importance = explainer.get_feature_importance(
                X_sample=sample_for_shap,
                aggregate_method="mean_abs",
            )

            # Log importance scores
            pushFeedback("\nFeature Importance (SHAP values):", feedback=feedback)
            for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
                pushFeedback(f"  {feat}: {score:.4f}", feedback=feedback)

            # Store importance in model metadata
            self.feature_importance = importance

            # Create importance raster if output path provided and raster_path is a file
            if shap_output_path and isinstance(raster_path, str):
                pushFeedback(f"Generating feature importance raster: {shap_output_path}", feedback=feedback)

                # Create progress callback for SHAP raster generation
                def shap_progress_callback(pct):
                    if feedback is not None and hasattr(feedback, "setProgress"):
                        # Map SHAP progress (0-100) to remaining progress window
                        feedback.setProgress(int(95 + pct * 0.05))

                explainer.create_importance_raster(
                    raster_path=raster_path,
                    output_path=shap_output_path,
                    sample_size=sample_size,
                    progress_callback=shap_progress_callback if feedback else None,
                )

                pushFeedback(f"Feature importance raster saved to: {shap_output_path}", feedback=feedback)

        except Exception as e:
            pushFeedback(
                f"Warning: SHAP computation failed: {e!s}\nContinuing without SHAP analysis.",
                feedback=feedback,
            )
            # Don't raise - continue with normal workflow

    def _handle_class_imbalance(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        classifier: str,
        extraParam: Dict[str, Any],
        feedback,
    ) -> None:
        """Analyze class imbalance and log recommendations.

        Parameters
        ----------
        X : np.ndarray
            Training features
        Y : np.ndarray
            Training labels
        classifier : str
            Classifier code
        extraParam : dict
            Extra parameters
        feedback : object
            Feedback interface

        """
        if not SAMPLING_AVAILABLE:
            return

        # Check imbalance ratio
        unique, counts = np.unique(Y, return_counts=True)
        if len(unique) < 2:
            return

        ratio = counts.max() / counts.min()

        # Log class distribution
        pushFeedback("Class distribution:", feedback=feedback)
        for cls, count in zip(unique, counts):
            pct = (count / len(Y)) * 100
            pushFeedback(f"  Class {int(cls)}: {int(count)} samples ({pct:.1f}%)", feedback=feedback)
        pushFeedback(f"  Imbalance ratio: {ratio:.2f}", feedback=feedback)

        # Recommend strategy if not already configured
        if not extraParam.get("USE_SMOTE", False) and not extraParam.get("USE_CLASS_WEIGHTS", False):
            strategy = recommend_strategy(Y)
            if strategy == "smote":
                pushFeedback(
                    "Warning: Dataset is severely imbalanced. "
                    "Consider enabling USE_SMOTE=True or USE_CLASS_WEIGHTS=True.",
                    feedback=feedback,
                )
            elif strategy == "class_weights":
                pushFeedback(
                    "Note: Dataset is moderately imbalanced. "
                    "Consider enabling USE_CLASS_WEIGHTS=True for better performance.",
                    feedback=feedback,
                )

    def _apply_smote(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        extraParam: Dict[str, Any],
        feedback,
    ) -> tuple:
        """Apply SMOTE oversampling if conditions are met.

        Parameters
        ----------
        X : np.ndarray
            Training features
        Y : np.ndarray
            Training labels
        extraParam : dict
            Extra parameters
        feedback : object
            Feedback interface

        Returns
        -------
        X_resampled : np.ndarray
            Resampled features
        Y_resampled : np.ndarray
            Resampled labels

        """
        if not SAMPLING_AVAILABLE or not IMBLEARN_AVAILABLE:
            pushFeedback(
                "Warning: SMOTE requires imbalanced-learn. Install with: pip install imbalanced-learn>=0.10.0",
                feedback=feedback,
            )
            return X, Y

        k_neighbors = extraParam.get("SMOTE_K_NEIGHBORS", 5)

        try:
            pushFeedback("Applying SMOTE oversampling...", feedback=feedback)
            original_size = len(Y)

            sampler = SMOTESampler(k_neighbors=k_neighbors, random_state=0)
            X_resampled, Y_resampled = sampler.fit_resample(X, Y)

            new_size = len(Y_resampled)
            pushFeedback(
                f"SMOTE complete: {original_size} -> {new_size} samples ({new_size - original_size} synthetic)",
                feedback=feedback,
            )

            return X_resampled, Y_resampled

        except Exception as e:
            pushFeedback(
                f"Warning: SMOTE failed: {e!s}. Continuing with original data.",
                feedback=feedback,
            )
            return X, Y

    def _compute_weights(
        self,
        Y: np.ndarray,
        extraParam: Dict[str, Any],
        feedback,
    ) -> Optional[Dict[int, float]]:
        """Compute class weights for cost-sensitive learning.

        Parameters
        ----------
        Y : np.ndarray
            Training labels
        extraParam : dict
            Extra parameters
        feedback : object
            Feedback interface

        Returns
        -------
        weights : dict or None
            Class weights or None if computation fails

        """
        if not SAMPLING_AVAILABLE:
            return None

        strategy = extraParam.get("CLASS_WEIGHT_STRATEGY", "balanced")
        custom_weights = extraParam.get("CUSTOM_CLASS_WEIGHTS", None)

        try:
            weights = compute_class_weights(Y, strategy=strategy, custom_weights=custom_weights)

            pushFeedback("Class weights computed:", feedback=feedback)
            for cls, weight in sorted(weights.items()):
                pushFeedback(f"  Class {cls}: {weight:.4f}", feedback=feedback)

            return weights

        except Exception as e:
            pushFeedback(
                f"Warning: Class weight computation failed: {e!s}. "
                "Continuing without class weights.",
                feedback=feedback,
            )
            return None

    def _validate_inputs(
        self,
        raster_path: Union[str, np.ndarray],
        vector_path: Union[str, np.ndarray],
        classifier: str,
        feedback,
    ) -> None:
        """Validate input parameters."""
        valid_classifiers = classifier_config.CLASSIFIER_CODES
        if classifier not in valid_classifiers:
            raise ValueError(f"Invalid classifier: {classifier}. Must be one of {valid_classifiers}")

        if isinstance(raster_path, np.ndarray) and not isinstance(vector_path, np.ndarray):
            msg = "You have to give an array for labels when using array for raster"
            pushFeedback(msg, feedback=feedback)
            raise ValueError(msg)

    def _setup_progress_feedback(self, feedback):
        """Setup progress feedback based on feedback type."""
        if feedback == "gui":
            return progress_bar.ProgressBar("Loading...", 100)
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgressText("Loading...")
            feedback.setProgress(0)
            return None
        return None

    def _load_and_prepare_data(
        self,
        raster_path: Union[str, np.ndarray],
        vector_path: Union[str, np.ndarray],
        class_field: str,
        split_config: Union[int, float, str],
        extraParam: Dict[str, Any],
        feedback,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[str],
    ]:
        """Load and prepare training data."""
        needXY = True
        coords = None
        distanceArray = None
        STDs = None
        vector_test_path = None

        pushFeedback("Learning model...", feedback=feedback)
        pushFeedback(0, feedback=feedback)

        # Handle numpy array inputs
        if isinstance(raster_path, np.ndarray):
            needXY = False
            X = raster_path
            if isinstance(vector_path, np.ndarray):
                Y = vector_path
            else:
                raise ValueError("Label array required when using raster array")
        else:
            # Handle vector inputs and extra parameters
            X, Y = None, None

            # Setup save directory if specified
            if "saveDir" in extraParam:
                self._setup_save_directory(extraParam["saveDir"])

            # Check for test vector
            if isinstance(split_config, str) and split_config.endswith((".shp", ".sqlite")):
                vector_test_path = split_config

            # Handle special ROI reading
            if extraParam.get("readROIFromVector", False):
                X, Y = self._read_roi_from_vector(vector_path, extraParam, class_field, feedback)
                needXY = False
                coords = extraParam.get("coords")

            # Standard rasterization approach
            if needXY:
                ROI = rasterize(raster_path, vector_path, class_field)

                if split_config == "SLOO":
                    X, Y, coords, distanceArray = self._prepare_sloo_data(raster_path, ROI, extraParam, feedback)
                elif split_config == "STAND":
                    X, Y, STDs = self._prepare_stand_data(
                        raster_path, vector_path, ROI, class_field, extraParam, feedback
                    )
                else:
                    X, Y = dataraster.get_samples_from_roi(raster_path, ROI)

                # Handle test vector if specified
                if vector_test_path:
                    ROIt = rasterize(raster_path, vector_test_path, class_field)
                    Xt, yt = dataraster.get_samples_from_roi(raster_path, ROIt)
                    # Store test data for later use
                    self._test_data = (Xt, yt)

        return X, Y, coords, distanceArray, STDs, vector_test_path

    def _setup_save_directory(self, saveDir: str) -> None:
        """Create save directory and subdirectories."""
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        matrix_dir = os.path.join(saveDir, "matrix/")
        if not os.path.exists(matrix_dir):
            os.makedirs(matrix_dir)

    def _read_roi_from_vector(self, vector_path, extraParam, class_field, feedback):
        """Read ROI data from vector using custom function."""
        try:
            from function_vector import readROIFromVector

            return readROIFromVector(vector_path, extraParam["readROIFromVector"], class_field)
        except ImportError:
            msg = "Problem when importing readFieldVector from functions in dzetsaka"
            pushFeedback(msg, feedback=feedback)
            raise

    def _prepare_sloo_data(self, raster_path, ROI, extraParam, feedback):
        """Prepare data for Spatial Leave-One-Out cross-validation."""
        try:
            from function_vector import distMatrix
        except ImportError:
            from .function_vector import distMatrix

        if extraParam.get("readROIFromVector", False):
            coords = extraParam.get("coords")
            if coords is None:
                pushFeedback("Can't read coords array", feedback=feedback)
                raise ValueError("Coordinates not found in extraParam")
            X, Y = None, None  # Will be set elsewhere
        else:
            X, Y, coords = dataraster.get_samples_from_roi(raster_path, ROI, getCoords=True)

        distanceArray = distMatrix(coords)
        return X, Y, coords, distanceArray

    def _prepare_stand_data(self, raster_path, vector_path, ROI, class_field, extraParam, feedback):
        """Prepare data for stand-based cross-validation."""
        inStand = extraParam.get("inStand", "stand")
        STAND = rasterize(raster_path, vector_path, inStand)
        X, Y, STDs = dataraster.get_samples_from_roi(raster_path, ROI, STAND)
        return X, Y, STDs

    def _handle_data_loading_error(self, error: Exception, class_field: str, feedback, progress) -> None:
        """Handle data loading errors with appropriate error messages."""
        if isinstance(error, ValueError) and ("could not convert" in str(error) or "invalid literal" in str(error)):
            msg = (
                f"Data type error: Unable to convert class values to numbers.\n"
                f"Please ensure your {class_field} field contains only integer values (1, 2, 3, etc.)\n"
                f"Error details: {error}"
            )
        else:
            msg = (
                f"Problem with getting samples from ROI: {error}\n"
                "Common causes:\n"
                "- Shapefile and raster have different projections\n"
                "- Invalid geometry in shapefile\n"
                f"- Field '{class_field}' contains non-numeric values\n"
                "- Memory issues with large datasets"
            )

        pushFeedback(msg, feedback=feedback)
        if progress and hasattr(progress, "reset"):
            progress.reset()


class ClassifyImage:
    """!@brief Classify image with learn clasifier and learned model.

    Create a raster file, fill hole from your give class (inClassForest), convert to a vector,
    remove parcel size which are under a certain size (defined in inMinSize) and save it to shp.

        Input :
            inRaster : Filtered image name ('sample_filtered.tif',str)
            inModel : Output name of the filtered file ('training.shp',str)
            outShpFile : Output name of vector files ('sample.shp',str)
            inMinSize : min size in acre for the forest, ex 6 means all polygons below 6000 m2 (int)
            TODO inMask : Mask size where no classification is done                                     |||| NOT YET IMPLEMENTED
            inField : Column name where are stored class number (str)
            inNODATA : if NODATA (int)
            inClassForest : Classification number of the forest class (int)

        Output :
            SHP file with deleted polygon below inMinSize

    """

    @backward_compatible(
        inRaster="raster_path",
        inModel="model_path",
        outRaster="output_path",
        inMask="mask_path",
    )
    def initPredict(
        self,
        raster_path: Optional[str] = None,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        confidenceMap: Optional[str] = None,
        confidenceMapPerClass: Optional[str] = None,
        NODATA: int = 0,
        feedback=None,
    ) -> Optional[str]:
        """Initialize prediction with raster and model paths."""
        if not raster_path:
            raise ValueError("raster_path (or inRaster) is required")
        if not model_path:
            raise ValueError("model_path (or inModel) is required")
        if not output_path:
            raise ValueError("output_path (or outRaster) is required")

        # Load model
        try:
            tree, M, m, classifier = self._load_model(model_path, feedback)
        except FileNotFoundError as e:
            pushFeedback(
                f"Model file not found: {model_path}\n"
                f"Please check that the file exists and the path is correct.\n"
                f"Error details: {e}",
                feedback=feedback,
            )
            return None
        except (pickle.UnpicklingError, pickle.PickleError) as e:
            pushFeedback(
                f"Model file is corrupted or incompatible: {model_path}\n"
                f"The model file may have been created with a different version of dzetsaka or Python.\n"
                f"Try retraining your model or use a different model file.\n"
                f"Error details: {e}",
                feedback=feedback,
            )
            return None
        except ValueError as e:
            pushFeedback(
                f"Invalid model file format: {model_path}\n"
                f"The model file structure is not recognized by dzetsaka.\n"
                f"Error details: {e}",
                feedback=feedback,
            )
            return None
        except Exception as e:
            error_details = f"Unexpected error while loading model {model_path}\n"
            error_details += f"Error type: {type(e).__name__}\n"
            error_details += f"Error details: {e}\n"
            error_details += "Please check the QGIS log for more details and consider reporting this issue."

            pushFeedback(error_details, feedback=feedback)

            # Show GitHub issue popup for unexpected errors
            if feedback == "gui":
                self._show_github_issue_popup(
                    "Model Loading Error",
                    f"Error type: {type(e).__name__}",
                    str(e),
                    f"Model path: {model_path}",
                )
            return None

        # Create temporary directory for processing
        try:
            temp_folder = tempfile.mkdtemp()
            os.path.join(temp_folder, "temp.tif")
        except Exception as e:
            pushFeedback(f"Cannot create temp file: {e}", feedback=feedback)
            return None
            # Process the data
        # Validate model components
        if not all(var is not None for var in [tree, M, m, classifier]):
            pushFeedback("Model variables not properly loaded", feedback=feedback)
            return None
        # try:
        predictedImage = self.predict_image(
            raster_path,
            output_path,
            tree,
            mask_path,
            confidenceMap,
            confidenceMapPerClass=confidenceMapPerClass,
            NODATA=NODATA,
            SCALE=[M, m],
            classifier=classifier,
            feedback=feedback,
        )
        # except:
        #   QgsMessageLog.logMessage("Problem while predicting "+inRaster+" in temp"+rasterTemp)

        return predictedImage

    def _load_model(self, model_path: str, feedback) -> Tuple[Any, np.ndarray, np.ndarray, str]:
        """Load pickled model with proper error handling.

        Parameters
        ----------
        model_path : str
            Path to the pickled model file
        feedback : object
            Feedback interface for error reporting

        Returns
        -------
        tuple
            (model, M, m, classifier) where M and m are scaling parameters

        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        pickle.UnpicklingError
            If model file is corrupted

        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as model_file:
                model_data = pickle.load(model_file)

            # Validate model data structure
            if not isinstance(model_data, (list, tuple)) or len(model_data) != 4:
                raise ValueError("Invalid model file format. Expected 4 components.")

            tree, M, m, classifier = model_data

            # Debug: log what we loaded
            pushFeedback(
                f"Loaded model data: tree={type(tree)}, M={type(M)}, m={type(m)}, classifier='{classifier}' (type: {type(classifier)})",
                feedback=feedback,
            )

            # Basic validation of components
            if tree is None:
                raise ValueError("Model is None")
            if not isinstance(M, np.ndarray) or not isinstance(m, np.ndarray):
                raise ValueError("Scaling parameters M and m must be numpy arrays")
            if not isinstance(classifier, str):
                raise ValueError(f"Classifier must be a string, got {type(classifier)}: {classifier}")

            return tree, M, m, classifier

        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Corrupted model file: {e}") from e
        except Exception as e:
            raise Exception(f"Failed to load model: {e}") from e

    def scale(
        self,
        x: np.ndarray,
        M: Optional[np.ndarray] = None,
        m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """!@brief Function that standardize the data.

        Input:
            x: the data
            M: the Max vector
            m: the Min vector
        Output:
            x: the standardize data
            M: the Max vector
            m: the Min vector
        """
        [n, d] = x.shape
        if np.float64 != x.dtype.type:
            x = x.astype("float")

        # Initialization of the output
        xs = np.empty_like(x)

        # get the parameters of the scaling
        if M is None:
            M, m = np.amax(x, axis=0), np.amin(x, axis=0)

        den = M - m
        for i in range(d):
            if den[i] != 0:
                xs[:, i] = 2 * (x[:, i] - m[i]) / den[i] - 1
            else:
                xs[:, i] = x[:, i]

        return xs

    def predict_image(
        self,
        raster_path: str,
        output_path: str,
        model=None,
        mask_path: Optional[str] = None,
        confidenceMap: Optional[str] = None,
        confidenceMapPerClass: Optional[str] = None,
        NODATA: int = 0,
        SCALE: Optional[List[np.ndarray]] = None,
        classifier: str = "GMM",
        feedback=None,
    ) -> str:
        """Classify the whole raster image using per-block image analysis.

        Parameters
        ----------
        raster_path : str
            Input raster image path
        output_path : str
            Output classification raster path
        model : object
            Trained classification model
        mask_path : str, optional
            Mask raster path
        confidenceMap : str, optional
            Confidence map output path
        confidenceMapPerClass : str, optional
            Per-class confidence map output path
        NODATA : int, default=0
            No data value
        SCALE : list of np.ndarray, optional
            Scaling parameters [M, m]
        classifier : str, default="GMM"
            Classifier type
        feedback : object, optional
            Feedback interface

        Returns
        -------
        str
            Path to output raster

        """
        # Open Raster and get additional information
        raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if raster is None:
            raise RuntimeError(f"Cannot open raster: {raster_path}")

        if mask_path is None:
            mask = None
        else:
            mask = gdal.Open(mask_path, gdal.GA_ReadOnly)
            if mask is None:
                raise RuntimeError(f"Cannot open mask: {mask_path}")
            # Check size
            if (raster.RasterXSize != mask.RasterXSize) or (raster.RasterYSize != mask.RasterYSize):
                # fix_print_with_import
                print("Image and mask should be of the same size")
                exit()
        if SCALE is not None:
            M, m = np.asarray(SCALE[0]), np.asarray(SCALE[1])

        # Get the size of the image
        d = raster.RasterCount
        nc = raster.RasterXSize
        nl = raster.RasterYSize

        # Provide feedback for multi-band images
        if d > 3:
            pushFeedback(
                f"Processing {d}-band image. This may take longer than standard RGB images.",
                feedback=feedback,
            )

        # Optimize block size for memory efficiency
        x_block_size, y_block_size = self._calculate_optimal_block_size(raster, d, feedback)

        # Get the geoinformation
        GeoTransform = raster.GetGeoTransform()
        Projection = raster.GetProjection()

        # Initialize the output
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        driver = gdal.GetDriverByName("GTiff")

        dtype = gdal.GDT_UInt16 if np.amax(model.classes_) > 255 else gdal.GDT_Byte

        dst_ds = driver.Create(output_path, nc, nl, 1, dtype)
        dst_ds.SetGeoTransform(GeoTransform)
        dst_ds.SetProjection(Projection)
        out = dst_ds.GetRasterBand(1)

        if classifier != "GMM":
            nClass = len(model.classes_)
        if confidenceMap:
            dst_confidenceMap = driver.Create(confidenceMap, nc, nl, 1, gdal.GDT_Int16)
            dst_confidenceMap.SetGeoTransform(GeoTransform)
            dst_confidenceMap.SetProjection(Projection)
            out_confidenceMap = dst_confidenceMap.GetRasterBand(1)

        if confidenceMapPerClass:
            dst_confidenceMapPerClass = driver.Create(confidenceMapPerClass, nc, nl, nClass, gdal.GDT_Int16)
            dst_confidenceMapPerClass.SetGeoTransform(GeoTransform)
            dst_confidenceMapPerClass.SetProjection(Projection)

        # Perform the classification

        total = nl

        if d > 3:
            pushFeedback(f"Predicting model for {d}-band image...")
        else:
            pushFeedback("Predicting model...")

        if feedback == "gui":
            progress_text = f"Predicting model ({d} bands)..." if d > 3 else "Predicting model..."
            progress = progress_bar.ProgressBar(progress_text, 100)
        elif feedback is not None and hasattr(feedback, "setProgress"):
            # Handle batch processing feedback
            progress_text = f"Predicting model for {d}-band image..." if d > 3 else "Predicting model..."
            feedback.setProgressText(progress_text)
            feedback.setProgress(0)

        for i in range(0, nl, y_block_size):
            if "lastBlock" not in locals():
                lastBlock = i
            if int(lastBlock / total * 100) != int(i / total * 100):
                lastBlock = i
                pct = int(i / total * 100)
                pushFeedback(pct, feedback=feedback)
                if feedback == "gui":
                    progress.prgBar.setValue(pct)

            lines = y_block_size if i + y_block_size < nl else nl - i  # Check for size consistency in Y
            for j in range(0, nc, x_block_size):  # Check for size consistency in X
                cols = x_block_size if j + x_block_size < nc else nc - j

                # Load the data efficiently
                X = self._load_block_data(raster, d, j, i, cols, lines, feedback)
                if X is None:
                    continue

                # Do the prediction
                band_temp = raster.GetRasterBand(1)
                nodata_temp = band_temp.GetNoDataValue()
                if nodata_temp is None:
                    nodata_temp = -9999

                if mask is None:
                    band_temp = raster.GetRasterBand(1)
                    mask_temp = band_temp.ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                    temp_nodata = np.where(mask_temp != nodata_temp)[0]
                    # t = np.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
                    t = np.where(X[:, 0] != nodata_temp)[0]
                    yp = np.zeros((cols * lines,))
                    # K = np.zeros((cols*lines,))
                    if confidenceMapPerClass or (confidenceMap and classifier != "GMM"):
                        K = np.zeros((cols * lines, nClass))
                        K[:, :] = -1
                    else:
                        K = np.zeros(cols * lines)
                        K[:] = -1

                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                    t = np.where((mask_temp != 0) & (X[:, 0] != nodata_temp))[0]
                    yp = np.zeros((cols * lines,))
                    yp[:] = NODATA
                    # K = np.zeros((cols*lines,))
                    if confidenceMapPerClass or (confidenceMap and classifier != "GMM"):
                        K = np.ones((cols * lines, nClass))
                        K = np.negative(K)
                    else:
                        K = np.zeros(cols * lines)
                        K = np.negative(K)

                # TODO: Change this part accorindgly ...
                if t.size > 0:
                    if confidenceMap and classifier == "GMM":
                        yp[t], K[t] = model.predict(self.scale(X[t, :], M=M, m=m), None, confidenceMap)

                    elif confidenceMap or (confidenceMapPerClass and classifier != "GMM"):
                        yp[t] = model.predict(self.scale(X[t, :], M=M, m=m))
                        K[t, :] = model.predict_proba(self.scale(X[t, :], M=M, m=m)) * 100

                    else:
                        yp[t] = model.predict(self.scale(X[t, :], M=M, m=m))

                        # QgsMessageLog.logMessage('amax from predict proba is : '+str(sp.amax(model.predict.proba(self.scale(X[t,:],M=M,m=m)),axis=1)))

                # Write the data
                out.WriteArray(yp.reshape(lines, cols), j, i)
                out.SetNoDataValue(NODATA)
                out.FlushCache()

                if confidenceMap and classifier == "GMM":
                    K *= 100
                    out_confidenceMap.WriteArray(K.reshape(lines, cols), j, i)
                    out_confidenceMap.SetNoDataValue(-1)
                    out_confidenceMap.FlushCache()

                if confidenceMap and classifier != "GMM":
                    Kconf = np.amax(K, axis=1)
                    out_confidenceMap.WriteArray(Kconf.reshape(lines, cols), j, i)
                    out_confidenceMap.SetNoDataValue(-1)
                    out_confidenceMap.FlushCache()

                if confidenceMapPerClass and classifier != "GMM":
                    for band in range(nClass):
                        gdalBand = band + 1
                        out_confidenceMapPerClass = dst_confidenceMapPerClass.GetRasterBand(gdalBand)
                        out_confidenceMapPerClass.SetNoDataValue(-1)
                        out_confidenceMapPerClass.WriteArray(K[:, band].reshape(lines, cols), j, i)
                        out_confidenceMapPerClass.FlushCache()

                # Explicit memory cleanup
                del X, yp
                if "K" in locals():
                    del K

        # Clean/Close variables
        if feedback == "gui":
            progress.reset()
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgress(100)

        raster = None
        dst_ds = None
        return output_path

    def _calculate_optimal_block_size(self, raster, num_bands: int, feedback) -> Tuple[int, int]:
        """Calculate optimal block size for memory efficiency."""
        # Get default block size
        band = raster.GetRasterBand(1)
        block_sizes = band.GetBlockSize()
        x_block_size = block_sizes[0]
        y_block_size = block_sizes[1]
        del band

        # Memory optimization for large multi-band images
        if num_bands > 3:
            pixel_size_bytes = 8 * num_bands  # Assume 8 bytes per pixel per band
            max_pixels_per_block = (MAX_MEMORY_MB * 1024 * 1024) // pixel_size_bytes

            current_block_pixels = x_block_size * y_block_size
            if current_block_pixels > max_pixels_per_block:
                scale_factor = (max_pixels_per_block / current_block_pixels) ** 0.5
                x_block_size = max(32, int(x_block_size * scale_factor))
                y_block_size = max(32, int(y_block_size * scale_factor))
                pushFeedback(
                    f"Adjusted block size to {x_block_size}x{y_block_size} for memory optimization",
                    feedback=feedback,
                )

        return x_block_size, y_block_size

    def _load_block_data(
        self,
        raster,
        num_bands: int,
        x_offset: int,
        y_offset: int,
        cols: int,
        lines: int,
        feedback,
    ) -> Optional[np.ndarray]:
        """Load raster block data with memory optimization."""
        try:
            # Use memory-efficient data type for multi-band images
            dtype = np.float32 if num_bands > 3 else np.float64
            X = np.empty((cols * lines, num_bands), dtype=dtype)

            for band_idx in range(num_bands):
                band_data = raster.GetRasterBand(band_idx + 1).ReadAsArray(x_offset, y_offset, cols, lines)
                if band_data is None:
                    pushFeedback(f"Error reading band {band_idx + 1}", feedback=feedback)
                    return None
                X[:, band_idx] = band_data.reshape(cols * lines)

                # Free band_data immediately
                del band_data

            return X

        except MemoryError:
            pushFeedback(
                "Memory error loading block data. Consider reducing block size or using fewer bands.",
                feedback=feedback,
            )
            return None
        except Exception as e:
            pushFeedback(f"Error loading block data: {e}", feedback=feedback)
            return None

    def _show_github_issue_popup(self, error_title, error_type, error_message, context):
        """Show a popup with GitHub issue template for copy/paste."""
        try:
            import platform

            from PyQt5.QtGui import QFont
            from PyQt5.QtWidgets import (
                QDialog,
                QHBoxLayout,
                QLabel,
                QPushButton,
                QTextEdit,
                QVBoxLayout,
            )
            from qgis.core import QgsApplication

            # Get system information
            qgis_version = QgsApplication.applicationVersion()
            python_version = platform.python_version()
            os_info = f"{platform.system()} {platform.release()}"

            # Create GitHub issue template
            github_template = f"""## Bug Report: {error_title}

**Error Type:** {error_type}

**Error Message:**
```
{error_message}
```

**Context:**
{context}

**Environment:**
- QGIS Version: {qgis_version}
- Python Version: {python_version}
- Operating System: {os_info}
- dzetsaka Version: 4.1.2

**Steps to Reproduce:**
1. [Please describe the steps that led to this error]
2.
3.

**Expected Behavior:**
[What you expected to happen]

**Additional Information:**
[Any additional context, screenshots, or logs that might help]

**Log Output:**
```
[Please paste relevant log output from QGIS Message Log]
```
"""

            # Create dialog
            dialog = QDialog()
            dialog.setWindowTitle("GitHub Issue Template - dzetsaka")
            dialog.setModal(True)
            dialog.resize(700, 600)

            layout = QVBoxLayout()

            # Title
            title_label = QLabel("Copy this template to report the issue on GitHub:")
            title_font = QFont()
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)

            # Text area with template
            text_edit = QTextEdit()
            text_edit.setPlainText(github_template)
            text_edit.selectAll()  # Pre-select all text for easy copying
            layout.addWidget(text_edit)

            # Buttons
            button_layout = QHBoxLayout()

            copy_button = QPushButton("Copy to Clipboard")
            copy_button.clicked.connect(lambda: self._copy_to_clipboard(github_template))

            github_button = QPushButton("Open GitHub Issues")
            github_button.clicked.connect(lambda: self._open_github_issues())

            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.close)

            button_layout.addWidget(copy_button)
            button_layout.addWidget(github_button)
            button_layout.addStretch()
            button_layout.addWidget(close_button)

            layout.addLayout(button_layout)
            dialog.setLayout(layout)

            dialog.exec_()

        except Exception as e:
            # Fallback if popup fails
            pushFeedback(f"Could not show GitHub issue popup: {e}", feedback="gui")

    def _copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        try:
            from PyQt5.QtWidgets import QApplication

            clipboard = QApplication.clipboard()
            clipboard.setText(text)

            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.information(None, "Copied", "GitHub issue template copied to clipboard!")
        except Exception as e:
            pushFeedback(f"Could not copy to clipboard: {e}", feedback="gui")

    def _open_github_issues(self):
        """Open dzetsaka GitHub issues page."""
        try:
            import webbrowser

            webbrowser.open("https://github.com/nkarasiak/dzetsaka/issues")
        except Exception as e:
            pushFeedback(f"Could not open GitHub: {e}", feedback="gui")


class ConfusionMatrix:
    """Class for computing confusion matrix statistics from raster predictions."""

    def __init__(self):
        self.confusion_matrix: Optional[np.ndarray] = None
        self.OA: Optional[float] = None
        self.Kappa: Optional[float] = None

    @backward_compatible(inRaster="raster_path", inShape="shapefile_path", inField="class_field")
    def computeStatistics(
        self,
        raster_path: Optional[str] = None,
        shapefile_path: Optional[str] = None,
        class_field: Optional[str] = None,
        feedback=None,
    ) -> None:
        """Compute confusion matrix statistics.

        Parameters
        ----------
        raster_path : str
            Path to prediction raster
        shapefile_path : str
            Path to reference shapefile
        class_field : str
            Field name containing reference classes
        feedback : object, optional
            Feedback interface for progress reporting

        Notes
        -----
        Backward compatibility is maintained through the @backward_compatible decorator.

        """
        if not raster_path:
            raise ValueError("raster_path (or inRaster) is required")
        if not shapefile_path:
            raise ValueError("shapefile_path (or inShape) is required")
        if not class_field:
            raise ValueError("class_field (or inField) is required")

        try:
            rasterized = rasterize(raster_path, shapefile_path, class_field)
            Yp, Yt = dataraster.get_samples_from_roi(raster_path, rasterized)
            CONF = ai.ConfusionMatrix()
            CONF.compute_confusion_matrix(Yp, Yt)
            self.confusion_matrix = CONF.confusion_matrix
            self.Kappa = CONF.Kappa
            self.OA = CONF.OA

            # Clean up temporary raster
            with contextlib.suppress(OSError):
                os.remove(rasterized)

        except Exception as e:
            error_msg = f"Error during statistics calculation: {e}"
            pushFeedback(error_msg, feedback=feedback)
            raise RuntimeError(error_msg) from e


@backward_compatible(inRaster="raster_path", inShape="shapefile_path", inField="class_field")
def rasterize(
    raster_path: Optional[str] = None,
    shapefile_path: Optional[str] = None,
    class_field: Optional[str] = None,
) -> str:
    """Rasterize vector data to match raster extent and resolution.

    Parameters
    ----------
    raster_path : str
        Reference raster path
    shapefile_path : str
        Vector shapefile path
    class_field : str
        Attribute field to rasterize

    Returns
    -------
    str
        Path to temporary rasterized file

    Notes
    -----
    Backward compatibility is maintained through the @backward_compatible decorator.

    """
    if not raster_path:
        raise ValueError("raster_path (or inRaster) is required")
    if not shapefile_path:
        raise ValueError("shapefile_path (or inShape) is required")
    if not class_field:
        raise ValueError("class_field (or inField) is required")

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    filename = tempfile.mktemp(".tif")

    try:
        data = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if data is None:
            raise RuntimeError(f"Cannot open raster: {raster_path}")

        shp = ogr.Open(shapefile_path)
        if shp is None:
            raise RuntimeError(f"Cannot open shapefile: {shapefile_path}")

        lyr = shp.GetLayer()
        if lyr is None:
            raise RuntimeError(f"Cannot access layer in shapefile: {shapefile_path}")

        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(filename, data.RasterXSize, data.RasterYSize, 1, gdal.GDT_UInt16)

        if dst_ds is None:
            raise RuntimeError(f"Cannot create output raster: {filename}")

        dst_ds.SetGeoTransform(data.GetGeoTransform())
        dst_ds.SetProjection(data.GetProjection())

        OPTIONS = f"ATTRIBUTE={class_field}"
        result = gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=[OPTIONS])

        if result != gdal.CE_None:
            raise RuntimeError(f"Rasterization failed for field {class_field}")

    except Exception as e:
        # Clean up on error
        if os.path.exists(filename):
            with contextlib.suppress(OSError):
                os.remove(filename)
        raise RuntimeError(f"Rasterization error: {e}") from e
    finally:
        # Ensure GDAL objects are properly closed
        data, dst_ds, shp, lyr = None, None, None, None

    return filename


def pushFeedback(message, feedback=None) -> None:
    """Push feedback message to appropriate interface.

    Parameters
    ----------
    message : str, int, or float
        Message to display or progress value
    feedback : object, optional
        Feedback interface object

    """
    isNum = isinstance(message, (float, int))

    if feedback and feedback is not True:
        if feedback == "gui":
            if not isNum:
                QgsMessageLog.logMessage(str(message))
        else:
            if isNum:
                feedback.setProgress(message)
            else:
                feedback.setProgressText(message)
    else:
        if not isNum:
            print(str(message))
        """
        else:
            print(52*"=")
            print(((int(message/2)-3)*'-'+(str(message)+'%')))
            print(52*"=")
        """


if __name__ == "__main__":
    # Example using new parameter names
    RASTER_PATH = "/mnt/DATA/demo/map.tif"
    VECTOR_PATH = "/mnt/DATA/demo/train.shp"
    CLASS_FIELD = "Class"
    MODEL_PATH = "/mnt/DATA/demo/test/model.RF"
    SPLIT_PERCENT = 50
    MATRIX_PATH = "/mnt/DATA/demo/test/matrix.csv"
    CLASSIFIER_TYPE = "RF"
    CONFIDENCE_PATH = "/mnt/DATA/demo/test/confidence.tif"
    MASK_PATH = None
    OUTPUT_PATH = "/mnt/DATA/demo/test/class.tif"

    # Using new parameter names
    temp = LearnModel(
        raster_path=RASTER_PATH,
        vector_path=VECTOR_PATH,
        class_field=CLASS_FIELD,
        model_path=MODEL_PATH,
        split_config=SPLIT_PERCENT,
        random_seed=0,
        matrix_path=MATRIX_PATH,
        classifier=CLASSIFIER_TYPE,
        extraParam=None,
        feedback=None,
    )
    print("learned")

    temp = ClassifyImage()
    temp.initPredict(
        raster_path=RASTER_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        mask_path=MASK_PATH,
        confidenceMap=CONFIDENCE_PATH,
    )
    print("classified")

    # Example showing backward compatibility still works
    print("\n=== Testing Backward Compatibility ===")
    temp_old = LearnModel(
        inRaster=RASTER_PATH,  # Old parameter name
        inVector=VECTOR_PATH,  # Old parameter name
        inField=CLASS_FIELD,  # Old parameter name
        outModel=MODEL_PATH,  # Old parameter name
        inSplit=SPLIT_PERCENT,  # Old parameter name
        inSeed=0,  # Old parameter name
        outMatrix=MATRIX_PATH,  # Old parameter name
        inClassifier=CLASSIFIER_TYPE,  # Old parameter name
        extraParam=None,
        feedback=None,
    )
    print("backward compatibility test passed")

    # Advanced testing examples
    Test = "SLOO"

    if Test == "STAND":
        extra_param = {
            "inStand": "Stand",
            "saveDir": "/tmp/test1/",
            "maxIter": 5,
            "SLOO": False,
        }
        LearnModel(
            raster_path=RASTER_PATH,
            vector_path=VECTOR_PATH,
            class_field=CLASS_FIELD,
            model_path=MODEL_PATH,
            split_config="STAND",
            random_seed=0,
            matrix_path=None,
            classifier=CLASSIFIER_TYPE,
            feedback=None,
            extraParam=extra_param,
        )

    if Test == "SLOO":
        RASTER_PATH = "/mnt/DATA/Test/DA/SITS/SITS_2013.tif"
        VECTOR_PATH = "/mnt/DATA/Test/DA/ROI_2154.sqlite"
        CLASS_FIELD = "level1"

        extra_param = {"distance": 100, "maxIter": 5, "saveDir": "/tmp/"}
        LearnModel(
            raster_path=RASTER_PATH,
            vector_path=VECTOR_PATH,
            class_field=CLASS_FIELD,
            model_path=MODEL_PATH,
            split_config="SLOO",
            random_seed=0,
            matrix_path=None,
            classifier=CLASSIFIER_TYPE,
            feedback=None,
            extraParam=extra_param,
        )
