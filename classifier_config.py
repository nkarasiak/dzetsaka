"""Centralized classifier configuration for dzetsaka.

This module provides a centralized configuration system for all machine learning
classifiers supported by dzetsaka. It defines classifier codes, names, dependencies,
and utility functions to manage classifier selection and validation.

The module supports 12 different machine learning algorithms:
- Gaussian Mixture Model (GMM) - Built-in, no dependencies
- Random Forest (RF) - Requires scikit-learn
- Support Vector Machine (SVM) - Requires scikit-learn
- K-Nearest Neighbors (KNN) - Requires scikit-learn
- XGBoost (XGB) - Requires XGBoost package
- LightGBM (LGB) - Requires LightGBM package
- CatBoost (CB) - Requires CatBoost package
- Extra Trees (ET) - Requires scikit-learn
- Gradient Boosting Classifier (GBC) - Requires scikit-learn
- Logistic Regression (LR) - Requires scikit-learn
- Gaussian Naive Bayes (NB) - Requires scikit-learn
- Multi-layer Perceptron (MLP) - Requires scikit-learn

Example:
    >>> get_classifier_code("Random Forest")
    'RF'
    >>> requires_sklearn("RF")
    True
    >>> is_valid_classifier("XGB")
    True

Author:
    Nicolas Karasiak

"""

# Classifier short codes (used in backend)
CLASSIFIER_CODES = [
    "GMM",
    "RF",
    "SVM",
    "KNN",
    "XGB",
    "LGB",
    "CB",
    "ET",
    "GBC",
    "LR",
    "NB",
    "MLP",
]

# Full classifier names (used in UI)
CLASSIFIER_NAMES = [
    "Gaussian Mixture Model",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Extra Trees",
    "Gradient Boosting Classifier",
    "Logistic Regression",
    "Gaussian Naive Bayes",
    "Multi-layer Perceptron",
]

# Mapping from codes to names
CODE_TO_NAME = dict(zip(CLASSIFIER_CODES, CLASSIFIER_NAMES))

# Mapping from names to codes
NAME_TO_CODE = dict(zip(CLASSIFIER_NAMES, CLASSIFIER_CODES))

# Sklearn-dependent classifiers (all except GMM)
SKLEARN_DEPENDENT = ["RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"]

# Special dependency classifiers
XGBOOST_DEPENDENT = ["XGB"]
LIGHTGBM_DEPENDENT = ["LGB"]
CATBOOST_DEPENDENT = ["CB"]

# UI display names (for processing algorithms)
UI_DISPLAY_NAMES = [
    "Gaussian Mixture Model",
    "Random-Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Extra Trees",
    "Gradient Boosting Classifier",
    "Logistic Regression",
    "Gaussian Naive Bayes",
    "Multi-layer Perceptron",
]


def get_classifier_code(name):
    """Get classifier short code from full name.

    Parameters
    ----------
    name : str
        Full classifier name (e.g., "Random Forest")

    Returns
    -------
    str
        Classifier short code (e.g., "RF"). Returns "GMM" if name not found.

    Examples
    --------
    >>> get_classifier_code("Random Forest")
    'RF'
    >>> get_classifier_code("Unknown Classifier")
    'GMM'

    """
    return NAME_TO_CODE.get(name, "GMM")


def get_classifier_name(code):
    """Get classifier full name from short code.

    Parameters
    ----------
    code : str
        Classifier short code (e.g., "RF")

    Returns
    -------
    str
        Full classifier name (e.g., "Random Forest").
        Returns "Gaussian Mixture Model" if code not found.

    Examples
    --------
    >>> get_classifier_name("RF")
    'Random Forest'
    >>> get_classifier_name("unknown")
    'Gaussian Mixture Model'

    """
    return CODE_TO_NAME.get(code.upper(), "Gaussian Mixture Model")


def is_valid_classifier(code):
    """Check if classifier code is valid.

    Parameters
    ----------
    code : str
        Classifier short code to validate

    Returns
    -------
    bool
        True if code is valid, False otherwise

    Examples
    --------
    >>> is_valid_classifier("RF")
    True
    >>> is_valid_classifier("INVALID")
    False

    """
    return code.upper() in CLASSIFIER_CODES


def requires_sklearn(code):
    """Check if classifier requires scikit-learn package.

    Parameters
    ----------
    code : str
        Classifier short code

    Returns
    -------
    bool
        True if classifier requires scikit-learn, False otherwise

    Examples
    --------
    >>> requires_sklearn("RF")
    True
    >>> requires_sklearn("GMM")
    False

    """
    return code.upper() in SKLEARN_DEPENDENT


def requires_xgboost(code):
    """Check if classifier requires XGBoost package.

    Parameters
    ----------
    code : str
        Classifier short code

    Returns
    -------
    bool
        True if classifier requires XGBoost, False otherwise

    Examples
    --------
    >>> requires_xgboost("XGB")
    True
    >>> requires_xgboost("RF")
    False

    """
    return code.upper() in XGBOOST_DEPENDENT


def requires_lightgbm(code):
    """Check if classifier requires LightGBM package.

    Parameters
    ----------
    code : str
        Classifier short code

    Returns
    -------
    bool
        True if classifier requires LightGBM, False otherwise

    Examples
    --------
    >>> requires_lightgbm("LGB")
    True
    >>> requires_lightgbm("RF")
    False

    """
    return code.upper() in LIGHTGBM_DEPENDENT


def requires_catboost(code):
    """Check if classifier requires CatBoost package.

    Parameters
    ----------
    code : str
        Classifier short code

    Returns
    -------
    bool
        True if classifier requires CatBoost, False otherwise

    Examples
    --------
    >>> requires_catboost("CB")
    True
    >>> requires_catboost("RF")
    False

    """
    return code.upper() in CATBOOST_DEPENDENT
