# -*- coding: utf-8 -*-
"""
Machine Learning Utilities

Shared utilities for ML functionality including backward compatibility
decorators and classifier wrappers.
"""

import functools
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


def backward_compatible(**parameter_mapping):
    """
    Decorator to handle backward compatibility for function parameters.

    Parameters
    ----------
    **parameter_mapping : dict
        Mapping from old parameter names to new parameter names
        Example: backward_compatible(inRaster='raster_path', inVector='vector_path')
    """

    def decorator(func):
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


class XGBLabelWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for XGBoost that handles sparse label encoding/decoding."""

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.label_encoder = LabelEncoder()
        self.xgb_classifier = None

    def fit(self, X, y):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not found. Install with: pip install xgboost")

        y_encoded = self.label_encoder.fit_transform(y)
        self.xgb_classifier = XGBClassifier(**self.xgb_params)
        self.xgb_classifier.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded = self.xgb_classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        return self.xgb_classifier.predict_proba(X)

    @property
    def classes_(self):
        return self.label_encoder.classes_

    def get_params(self, deep=True):
        return self.xgb_params.copy() if not deep else self.xgb_params.copy()

    def set_params(self, **params):
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
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("LightGBM not found. Install with: pip install lightgbm")

        y_encoded = self.label_encoder.fit_transform(y)
        self.lgb_classifier = LGBMClassifier(**self.lgb_params)
        self.lgb_classifier.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded = self.lgb_classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        return self.lgb_classifier.predict_proba(X)

    @property
    def classes_(self):
        return self.label_encoder.classes_

    def get_params(self, deep=True):
        return self.lgb_params.copy() if not deep else self.lgb_params.copy()

    def set_params(self, **params):
        self.lgb_params.update(params)
        if self.lgb_classifier is not None:
            self.lgb_classifier.set_params(**params)
        return self