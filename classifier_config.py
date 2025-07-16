# -*- coding: utf-8 -*-
"""
Centralized classifier configuration for dzetsaka.

This module defines all supported classifiers in one place to avoid duplication
and ensure consistency across the codebase.
"""

# Classifier short codes (used in backend)
CLASSIFIER_CODES = ["GMM", "RF", "SVM", "KNN", "XGB", "LGB", "ET", "GBC", "LR", "NB", "MLP"]

# Full classifier names (used in UI)
CLASSIFIER_NAMES = [
    "Gaussian Mixture Model",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "XGBoost",
    "LightGBM",
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

# UI display names (for processing algorithms)
UI_DISPLAY_NAMES = [
    "Gaussian Mixture Model",
    "Random-Forest", 
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "XGBoost",
    "LightGBM",
    "Extra Trees",
    "Gradient Boosting Classifier",
    "Logistic Regression",
    "Gaussian Naive Bayes",
    "Multi-layer Perceptron",
]

def get_classifier_code(name):
    """Get classifier code from name."""
    return NAME_TO_CODE.get(name, "GMM")

def get_classifier_name(code):
    """Get classifier name from code."""
    return CODE_TO_NAME.get(code.upper(), "Gaussian Mixture Model")

def is_valid_classifier(code):
    """Check if classifier code is valid."""
    return code.upper() in CLASSIFIER_CODES

def requires_sklearn(code):
    """Check if classifier requires sklearn."""
    return code.upper() in SKLEARN_DEPENDENT

def requires_xgboost(code):
    """Check if classifier requires XGBoost."""
    return code.upper() in XGBOOST_DEPENDENT

def requires_lightgbm(code):
    """Check if classifier requires LightGBM."""
    return code.upper() in LIGHTGBM_DEPENDENT