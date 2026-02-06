"""Label encoding wrappers for XGBoost, LightGBM, and CatBoost.

These wrapper classes transparently handle sparse label encoding/decoding,
allowing models to work with non-continuous class labels (e.g., 1, 3, 5, 7)
without manual preprocessing.

Author:
    Nicolas Karasiak
"""

import numpy as np

# Check if sklearn is available for label encoding
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
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

    SKLEARN_AVAILABLE = False


# Label encoding wrapper classes
# Only define these classes if sklearn is available
XGBLabelWrapper = None
LGBLabelWrapper = None
CBClassifierWrapper = None

if SKLEARN_AVAILABLE:

    class XGBLabelWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper for XGBoost that handles sparse label encoding/decoding.

        This wrapper automatically encodes sparse class labels (e.g., 0, 1, 3, 5)
        to continuous labels (0, 1, 2, 3) required by XGBoost, and decodes
        predictions back to original labels.

        Parameters
        ----------
        **xgb_params : dict
            Parameters to pass to XGBClassifier

        Examples
        --------
        >>> model = XGBLabelWrapper(n_estimators=100, max_depth=5)
        >>> # Train with sparse labels (0, 1, 3, 5)
        >>> model.fit(X_train, y_train)
        >>> # Predictions automatically decoded to original labels
        >>> predictions = model.predict(X_test)
        """

        def __init__(self, **xgb_params):
            self.xgb_params = xgb_params
            self.label_encoder = LabelEncoder()
            self.xgb_classifier = None

        def fit(self, X, y):
            """Fit the XGBoost classifier with label encoding.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data
            y : array-like of shape (n_samples,)
                Target values (can be sparse/non-continuous)

            Returns
            -------
            self
                Fitted estimator
            """
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError("XGBoost not found. Install with: pip install xgboost") from None

            y_encoded = self.label_encoder.fit_transform(y)
            self.xgb_classifier = XGBClassifier(**self.xgb_params)
            self.xgb_classifier.fit(X, y_encoded)
            return self

        def predict(self, X):
            """Predict labels using the fitted classifier.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Samples to predict

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted class labels (in original sparse encoding)
            """
            y_encoded = self.xgb_classifier.predict(X)
            return self.label_encoder.inverse_transform(y_encoded)

        def predict_proba(self, X):
            """Predict class probabilities using the fitted classifier.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Samples to predict

            Returns
            -------
            proba : ndarray of shape (n_samples, n_classes)
                Class probabilities
            """
            return self.xgb_classifier.predict_proba(X)

        @property
        def classes_(self):
            """Get class labels in original encoding."""
            return self.label_encoder.classes_

        def get_params(self, deep=True):
            """Get parameters for this estimator."""
            return self.xgb_params.copy()

        def set_params(self, **params):
            """Set parameters for this estimator."""
            self.xgb_params.update(params)
            if self.xgb_classifier is not None:
                self.xgb_classifier.set_params(**params)
            return self

    class LGBLabelWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper for LightGBM that handles sparse label encoding/decoding.

        This wrapper automatically encodes sparse class labels (e.g., 0, 1, 3, 5)
        to continuous labels (0, 1, 2, 3) required by LightGBM, and decodes
        predictions back to original labels.

        Parameters
        ----------
        **lgb_params : dict
            Parameters to pass to LGBMClassifier

        Examples
        --------
        >>> model = LGBLabelWrapper(n_estimators=100, num_leaves=31)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        """

        def __init__(self, **lgb_params):
            self.lgb_params = lgb_params
            self.label_encoder = LabelEncoder()
            self.lgb_classifier = None

        def fit(self, X, y):
            """Fit the LightGBM classifier with label encoding.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data
            y : array-like of shape (n_samples,)
                Target values (can be sparse/non-continuous)

            Returns
            -------
            self
                Fitted estimator
            """
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                raise ImportError("LightGBM not found. Install with: pip install lightgbm") from None

            y_encoded = self.label_encoder.fit_transform(y)
            self.lgb_classifier = LGBMClassifier(**self.lgb_params)
            self.lgb_classifier.fit(X, y_encoded)
            return self

        def predict(self, X):
            """Predict class labels for samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Samples to predict

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted class labels (in original sparse encoding)
            """
            y_encoded = self.lgb_classifier.predict(X)
            return self.label_encoder.inverse_transform(y_encoded)

        def predict_proba(self, X):
            """Predict class probabilities for samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Samples to predict

            Returns
            -------
            proba : ndarray of shape (n_samples, n_classes)
                Class probabilities
            """
            return self.lgb_classifier.predict_proba(X)

        @property
        def classes_(self):
            """Get class labels in original encoding."""
            return self.label_encoder.classes_

        def get_params(self, deep=True):
            """Get parameters for this estimator."""
            return self.lgb_params.copy()

        def set_params(self, **params):
            """Set parameters for this estimator."""
            self.lgb_params.update(params)
            if self.lgb_classifier is not None:
                self.lgb_classifier.set_params(**params)
            return self

    class CBClassifierWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper for CatBoost that handles sparse label encoding/decoding.

        This wrapper automatically encodes sparse class labels (e.g., 0, 1, 3, 5)
        to continuous labels (0, 1, 2, 3) required by CatBoost, and decodes
        predictions back to original labels.

        Parameters
        ----------
        **cb_params : dict
            Parameters to pass to CatBoostClassifier

        Examples
        --------
        >>> model = CBClassifierWrapper(iterations=100, depth=6)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        """

        def __init__(self, **cb_params):
            self.cb_params = cb_params
            self.label_encoder = LabelEncoder()
            self.cb_classifier = None

        def fit(self, X, y):
            """Fit the CatBoost classifier with label encoding.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data
            y : array-like of shape (n_samples,)
                Target values (can be sparse/non-continuous)

            Returns
            -------
            self
                Fitted estimator
            """
            try:
                from catboost import CatBoostClassifier
            except ImportError:
                raise ImportError("CatBoost not found. Install with: pip install catboost") from None

            y_encoded = self.label_encoder.fit_transform(y)
            self.cb_classifier = CatBoostClassifier(**self.cb_params)
            self.cb_classifier.fit(X, y_encoded)
            return self

        def predict(self, X):
            """Predict class labels for samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Samples to predict

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted class labels (in original sparse encoding)
            """
            y_encoded = self.cb_classifier.predict(X)
            y_encoded = np.array(y_encoded).reshape(-1)
            return self.label_encoder.inverse_transform(y_encoded)

        def predict_proba(self, X):
            """Predict class probabilities for samples.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Samples to predict

            Returns
            -------
            proba : ndarray of shape (n_samples, n_classes)
                Class probabilities
            """
            return self.cb_classifier.predict_proba(X)

        @property
        def classes_(self):
            """Get class labels in original encoding."""
            return self.label_encoder.classes_

        def get_params(self, deep=True):
            """Get parameters for this estimator."""
            return self.cb_params.copy()

        def set_params(self, **params):
            """Set parameters for this estimator."""
            self.cb_params.update(params)
            if self.cb_classifier is not None:
                self.cb_classifier.set_params(**params)
            return self
