"""Gaussian Mixture Model with Ridge Regularization.

This module provides a GMM classifier with ridge regularization for improved
numerical stability. It follows scikit-learn's estimator API for compatibility
with the broader ML ecosystem.

Author: Mathieu Fauvel (original implementation)
Enhanced: Nicolas Karasiak (sklearn compatibility, numerical stability, performance)

Example
-------
>>> from scripts.gmm_ridge import GMMR
>>> import numpy as np
>>> X = np.random.randn(100, 5)
>>> y = np.random.randint(1, 4, 100)
>>> model = GMMR(tau=0.1, random_state=42)
>>> model.fit(X, y)
>>> predictions = model.predict(X)
>>> probabilities = model.predict_proba(X)

"""
# -*- coding: utf-8 -*-

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy import linalg

# Try to import sklearn components for enhanced functionality
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Minimal stubs for when sklearn is not available
    class BaseEstimator:
        """Stub for sklearn.base.BaseEstimator."""

        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            """Get parameters for this estimator."""
            return {key: getattr(self, key) for key in self._get_param_names()}

        def set_params(self, **params) -> "BaseEstimator":
            """Set parameters for this estimator."""
            for key, value in params.items():
                setattr(self, key, value)
            return self

        @classmethod
        def _get_param_names(cls):
            """Get parameter names for the estimator."""
            import inspect

            init_signature = inspect.signature(cls.__init__)
            parameters = [
                p.name for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            return sorted(parameters)

    class ClassifierMixin:
        """Stub for sklearn.base.ClassifierMixin."""

        def score(self, X, y):
            """Return accuracy score."""
            predictions = self.predict(X)
            return np.mean(predictions == y)

    def check_X_y(X, y, **kwargs):
        """Minimal validation stub."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        return X, y

    def check_array(X, **kwargs):
        """Minimal validation stub."""
        return np.asarray(X, dtype=np.float64)

    def check_is_fitted(estimator, attributes):
        """Minimal fitted check stub."""
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise RuntimeError(f"Model not fitted. Missing attribute: {attr}")


# Multiprocessing support for cross-validation
import multiprocessing as mp


def _predict_worker(tau: float, model: "GMMR", xT: np.ndarray, yT: np.ndarray) -> np.ndarray:
    """Worker function for parallel cross-validation.

    Parameters
    ----------
    tau : float
        Regularization parameter to test
    model : GMMR
        Trained GMM model
    xT : np.ndarray
        Test samples
    yT : np.ndarray
        True labels

    Returns
    -------
    err : np.ndarray
        Accuracy scores for each tau value
    """
    err = np.zeros(tau.size)
    for j, t in enumerate(tau):
        yp = model.predict(xT, tau=t)
        if isinstance(yp, tuple):
            yp = yp[0]
        eq = np.where(yp.ravel() == yT.ravel())[0]
        err[j] = eq.size * 100.0 / yT.size
    return err


class GMMR(BaseEstimator, ClassifierMixin):
    """Gaussian Mixture Model with Ridge Regularization.

    This classifier models each class as a multivariate Gaussian distribution
    and uses ridge regularization to handle ill-conditioned covariance matrices.
    It's particularly useful for high-dimensional data with limited samples.

    Parameters
    ----------
    tau : float, default=0.0
        Ridge regularization parameter. Higher values add more regularization
        to the covariance matrices, improving numerical stability but potentially
        reducing discriminative power. Typical range: [1e-6, 10.0].
    random_state : int or None, default=None
        Random seed for reproducibility in cross-validation.
    min_eigenvalue : float, default=1e-6
        Minimum eigenvalue threshold for numerical stability. Eigenvalues below
        this threshold are clipped to prevent numerical issues.
    warn_ill_conditioned : bool, default=False
        Whether to warn about ill-conditioned covariance matrices during training.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels
    n_features_in_ : int
        Number of features seen during fit
    ni : ndarray of shape (n_classes, 1)
        Number of samples per class
    prop : ndarray of shape (n_classes, 1)
        Class proportions (priors)
    mean : ndarray of shape (n_classes, n_features)
        Class means
    cov : ndarray of shape (n_classes, n_features, n_features)
        Class covariance matrices
    Q : ndarray of shape (n_classes, n_features, n_features)
        Eigenvector matrices for each class
    L : ndarray of shape (n_classes, n_features)
        Eigenvalues for each class (sorted descending)

    Examples
    --------
    >>> import numpy as np
    >>> from scripts.gmm_ridge import GMMR
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(1, 4, 100)
    >>> model = GMMR(tau=0.1)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> y_proba = model.predict_proba(X)

    Notes
    -----
    This implementation uses eigendecomposition for efficient inversion of
    covariance matrices and improved numerical stability. Ridge regularization
    is applied in eigenspace, which is more stable than direct matrix regularization.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
           Springer. Chapter 2.3 (The Gaussian Distribution).
    """

    def __init__(
        self,
        tau: float = 0.0,
        random_state: Optional[int] = None,
        min_eigenvalue: float = 1e-6,
        warn_ill_conditioned: bool = False,
    ):
        self.tau = tau
        self.random_state = random_state
        self.min_eigenvalue = min_eigenvalue
        self.warn_ill_conditioned = warn_ill_conditioned

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMR":
        """Fit the Gaussian Mixture Model.

        This is the sklearn-compatible entry point. Internally calls learn().

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples
        y : array-like of shape (n_samples,)
            Target class labels

        Returns
        -------
        self : object
            Returns self for method chaining
        """
        return self.learn(X, y)

    def learn(self, x: np.ndarray, y: np.ndarray) -> "GMMR":
        """Learn the GMM parameters from training data.

        Estimates the mean, covariance, and prior probability for each class
        using maximum likelihood estimation. Performs eigendecomposition of
        covariance matrices for efficient prediction.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Training samples
        y : np.ndarray of shape (n_samples,)
            Target class labels

        Returns
        -------
        self : object
            Returns self for method chaining

        Raises
        ------
        ValueError
            If input contains NaN/Inf values or if any class has too few samples
        """
        # Input validation
        x, y = check_X_y(x, y, dtype=np.float64)

        # Check for non-finite values
        if not np.all(np.isfinite(x)):
            raise ValueError("Input X contains NaN or Inf values")

        unique_labels = np.unique(y)
        if not np.all(np.isfinite(unique_labels)):
            raise ValueError("Non-finite class labels detected in training data")

        C = unique_labels.shape[0]  # Number of classes
        n = x.shape[0]  # Number of samples
        d = x.shape[1]  # Number of features

        # Store for sklearn compatibility
        self.n_features_in_ = d

        # Check for sufficient samples per class
        min_samples_required = d + 1  # Need at least d+1 samples for reliable covariance
        for label in unique_labels:
            n_samples = np.sum(y == label)
            if n_samples < min_samples_required:
                warnings.warn(
                    f"Class {label} has only {n_samples} samples but needs "
                    f"at least {min_samples_required} for reliable covariance estimation. "
                    f"Consider increasing regularization (tau) or collecting more samples.",
                    UserWarning,
                )

        # Initialize storage
        self.ni = np.empty((C, 1))
        self.prop = np.empty((C, 1))
        self.mean = np.empty((C, d))
        self.cov = np.empty((C, d, d))
        self.Q = np.empty((C, d, d))
        self.L = np.empty((C, d))
        self.classnum = unique_labels.astype("uint16")
        self.classes_ = self.classnum

        # Learn parameters for each class
        for c, cR in enumerate(unique_labels):
            j = np.where(y == cR)[0]

            self.ni[c] = float(j.size)
            self.prop[c] = self.ni[c] / n
            self.mean[c, :] = np.mean(x[j, :], axis=0)

            # Compute covariance (biased estimator for consistency with ridge)
            self.cov[c, :, :] = np.cov(x[j, :], bias=1, rowvar=0)

            # Eigendecomposition for efficient inversion
            L, Q = linalg.eigh(self.cov[c, :, :])

            # Sort eigenvalues/eigenvectors in descending order
            idx = L.argsort()[::-1]
            self.L[c, :] = L[idx]
            self.Q[c, :, :] = Q[:, idx]

            # Check condition number and warn if ill-conditioned
            if self.warn_ill_conditioned and self.L[c, 0] > 0:
                condition_number = self.L[c, 0] / max(self.L[c, -1], 1e-15)
                if condition_number > 1e10:
                    warnings.warn(
                        f"Class {cR} has ill-conditioned covariance matrix "
                        f"(condition number: {condition_number:.2e}). "
                        f"Consider increasing tau parameter.",
                        UserWarning,
                    )

        return self

    def predict(
        self, xt: np.ndarray, tau: Optional[float] = None, confidenceMap: Optional[bool] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict class labels for samples.

        Parameters
        ----------
        xt : np.ndarray of shape (n_samples, n_features)
            Samples to classify
        tau : float, optional
            Override regularization parameter for this prediction.
            If None, uses self.tau.
        confidenceMap : bool, optional
            If True, also returns confidence scores (probability of predicted class).
            Deprecated: use predict_proba() instead.

        Returns
        -------
        yp : np.ndarray of shape (n_samples,)
            Predicted class labels
        confidences : np.ndarray of shape (n_samples,), optional
            Confidence scores (only returned if confidenceMap=True)

        Notes
        -----
        This implementation uses numerically stable computation via:
        1. Eigendecomposition-based matrix inversion
        2. Log-sum-exp trick for probability computation
        3. Careful handling of overflow/underflow
        """
        # Validation
        check_is_fitted(self, ["mean", "cov", "classes_"])
        xt = check_array(xt, dtype=np.float64)

        if xt.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {xt.shape[1]} features but model was trained on {self.n_features_in_} features"
            )

        MAX = np.finfo(np.float64).max
        E_MAX = np.log(MAX)

        nt = xt.shape[0]  # Number of test samples
        C = self.ni.shape[0]  # Number of classes
        d = xt.shape[1]  # Number of features

        # Determine regularization
        TAU = self.tau if tau is None else tau
        TAU = float(TAU)
        regularization = max(TAU, 0.0)

        # Pre-compute inverse covariance matrices and constants
        invCovs = np.empty((C, d, d))
        csts = np.empty(C)

        for c in range(C):
            # Regularized eigenvalues
            Lr = self.L[c, :] + regularization

            # Ensure numerical stability
            Lr = np.maximum(Lr, self.min_eigenvalue)

            # Compute inverse via eigendecomposition: Sigma^{-1} = Q * diag(1/L) * Q^T
            temp = self.Q[c, :, :] * (1.0 / Lr)
            invCovs[c] = np.dot(temp, self.Q[c, :, :].T)

            # Log determinant for normalization constant
            logdet = np.sum(np.log(Lr))
            csts[c] = logdet - 2.0 * np.log(self.prop[c].item())

        # Compute discriminant function for all samples and classes
        K = np.empty((nt, C))
        for c in range(C):
            delta = xt - self.mean[c]
            projection = np.dot(delta, invCovs[c])
            K[:, c] = np.sum(delta * projection, axis=1) + csts[c]

        # Predict class with minimum discriminant value
        yp_idx = np.argmin(K, axis=1)
        yp = self.classnum[yp_idx]

        # Return predictions only if confidence map not requested
        if not confidenceMap:
            return yp

        # Compute confidence scores (posterior probabilities)
        # Use log-sum-exp trick for numerical stability
        logits = -0.5 * K
        logits = np.clip(logits, -E_MAX, E_MAX)

        # Softmax in stable form
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Extract confidence for predicted class
        confidences = proba[np.arange(nt), yp_idx]

        return yp, confidences

    def predict_proba(self, X: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to classify
        tau : float, optional
            Override regularization parameter. If None, uses self.tau.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample. Each row sums to 1.

        Examples
        --------
        >>> model = GMMR(tau=0.1).fit(X_train, y_train)
        >>> proba = model.predict_proba(X_test)
        >>> print(proba.sum(axis=1))  # All close to 1.0
        """
        check_is_fitted(self, ["mean", "cov", "classes_"])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features but model was trained on {self.n_features_in_} features")

        MAX = np.finfo(np.float64).max
        E_MAX = np.log(MAX)

        nt = X.shape[0]
        C = self.ni.shape[0]
        d = X.shape[1]

        # Determine regularization
        TAU = self.tau if tau is None else tau
        TAU = float(TAU)
        regularization = max(TAU, 0.0)

        # Pre-compute inverse covariance matrices and constants
        invCovs = np.empty((C, d, d))
        csts = np.empty(C)

        for c in range(C):
            Lr = self.L[c, :] + regularization
            Lr = np.maximum(Lr, self.min_eigenvalue)
            temp = self.Q[c, :, :] * (1.0 / Lr)
            invCovs[c] = np.dot(temp, self.Q[c, :, :].T)
            logdet = np.sum(np.log(Lr))
            csts[c] = logdet - 2.0 * np.log(self.prop[c].item())

        # Compute discriminant values
        K = np.empty((nt, C))
        for c in range(C):
            delta = X - self.mean[c]
            projection = np.dot(delta, invCovs[c])
            K[:, c] = np.sum(delta * projection, axis=1) + csts[c]

        # Convert to probabilities using log-sum-exp trick
        logits = -0.5 * K
        logits = np.clip(logits, -E_MAX, E_MAX)

        # Stable softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return proba

    def compute_inverse_logdet(self, c: int, tau: float) -> Tuple[np.ndarray, float]:
        """Compute inverse covariance matrix and log determinant for a class.

        Parameters
        ----------
        c : int
            Class index
        tau : float
            Regularization parameter

        Returns
        -------
        invCov : np.ndarray of shape (n_features, n_features)
            Inverse covariance matrix
        logdet : float
            Log determinant of the regularized covariance matrix
        """
        Lr = self.L[c, :] + tau
        Lr = np.maximum(Lr, self.min_eigenvalue)
        temp = self.Q[c, :, :] * (1.0 / Lr)
        invCov = np.dot(temp, self.Q[c, :, :].T)
        logdet = np.sum(np.log(Lr))
        return invCov, logdet

    def BIC(self, x: np.ndarray, y: np.ndarray, tau: Optional[float] = None) -> float:
        """Compute the Bayesian Information Criterion.

        BIC is used for model selection, with lower values indicating better models.
        It balances model fit against complexity.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Samples
        y : np.ndarray of shape (n_samples,)
            True labels
        tau : float, optional
            Regularization parameter. If None, uses self.tau.

        Returns
        -------
        bic : float
            Bayesian Information Criterion value

        Notes
        -----
        BIC = log-likelihood + penalty
        where penalty = (number of parameters) * log(n_samples) / 2
        """
        check_is_fitted(self, ["mean", "cov"])

        # Get information from the data
        C, d = self.mean.shape
        n = x.shape[0]

        # Use instance tau if not provided
        TAU = self.tau if tau is None else tau

        # Penalization term: number of parameters * log(n)
        # Parameters: C means (C*d) + C covariances (C*d*(d+1)/2) + C-1 priors
        P = C * (d * (d + 3) / 2) + (C - 1)
        P *= np.log(n)

        # Compute the log-likelihood
        L = 0.0
        for c in range(C):
            # Get samples for this class
            j = np.where(y == self.classnum[c])[0]
            if j.size == 0:
                continue

            xi = x[j, :]
            invCov, logdet = self.compute_inverse_logdet(c, TAU)

            # Constant term
            cst = logdet - 2.0 * np.log(self.prop[c].item())

            # Center the data
            xi_centered = xi - self.mean[c, :]

            # Mahalanobis distance
            temp = np.dot(invCov, xi_centered.T).T
            K = np.sum(xi_centered * temp, axis=1) + cst
            L += np.sum(K)

        return L + P

    def cross_validation(
        self, x: np.ndarray, y: np.ndarray, tau: np.ndarray, v: int = 5, n_jobs: int = -1
    ) -> Tuple[float, np.ndarray]:
        """Perform cross-validation to find optimal tau parameter.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Training samples
        y : np.ndarray of shape (n_samples,)
            Target labels
        tau : np.ndarray of shape (n_tau_values,)
            Array of regularization values to test
        v : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs. -1 means using all processors.
            Only used if multiprocessing is available.

        Returns
        -------
        best_tau : float
            Regularization parameter with highest cross-validation accuracy
        err : np.ndarray of shape (n_tau_values,)
            Cross-validation accuracy for each tau value

        Examples
        --------
        >>> tau_grid = np.logspace(-6, 2, 20)
        >>> best_tau, scores = model.cross_validation(X, y, tau_grid, v=5)
        >>> print(f"Best tau: {best_tau}, Best score: {scores.max()}")
        """
        # Use sklearn if available for better stratification
        if SKLEARN_AVAILABLE:
            return self._cross_validation_sklearn(x, y, tau, v)
        else:
            return self._cross_validation_legacy(x, y, tau, v, n_jobs)

    def _cross_validation_sklearn(
        self, x: np.ndarray, y: np.ndarray, tau: np.ndarray, v: int
    ) -> Tuple[float, np.ndarray]:
        """Cross-validation using sklearn's StratifiedKFold."""
        from sklearn.base import clone

        cv = StratifiedKFold(n_splits=v, shuffle=True, random_state=self.random_state)

        num_tau = tau.size
        err = np.zeros(num_tau)

        for tau_idx, tau_val in enumerate(tau):
            fold_scores = []

            for train_idx, test_idx in cv.split(x, y):
                # Create and train model on fold
                fold_model = clone(self)
                fold_model.tau = tau_val
                fold_model.fit(x[train_idx], y[train_idx])

                # Evaluate on validation fold
                y_pred = fold_model.predict(x[test_idx])
                accuracy = np.mean(y_pred == y[test_idx]) * 100.0
                fold_scores.append(accuracy)

            err[tau_idx] = np.mean(fold_scores)

        best_idx = err.argmax()
        return tau[best_idx], err

    def _cross_validation_legacy(
        self, x: np.ndarray, y: np.ndarray, tau: np.ndarray, v: int, n_jobs: int
    ) -> Tuple[float, np.ndarray]:
        """Legacy cross-validation with multiprocessing."""
        num_tau = tau.size

        # Create stratified folds manually
        unique_labels = np.unique(y)
        fold_indices = [[] for _ in range(v)]

        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            n_samples = len(label_indices)
            fold_size = n_samples // v

            # Shuffle indices
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(label_indices)
            else:
                np.random.shuffle(label_indices)

            # Distribute to folds
            for fold_idx in range(v):
                start = fold_idx * fold_size
                end = (fold_idx + 1) * fold_size if fold_idx < v - 1 else n_samples
                fold_indices[fold_idx].extend(label_indices[start:end])

        # Train models for each fold
        err = np.zeros(num_tau)
        model_cv = []

        for i in range(v):
            test_idx = np.array(fold_indices[i])
            train_idx = np.concatenate([fold_indices[j] for j in range(v) if j != i])

            fold_model = GMMR(tau=self.tau, random_state=self.random_state)
            fold_model.learn(x[train_idx], y[train_idx])
            model_cv.append(fold_model)

        # Evaluate with multiprocessing
        if n_jobs != 1:
            try:
                pool = mp.Pool(processes=None if n_jobs == -1 else n_jobs)
                processes = []

                for i in range(v):
                    test_idx = np.array(fold_indices[i])
                    processes.append(pool.apply_async(_predict_worker, args=(tau, model_cv[i], x[test_idx], y[test_idx])))

                pool.close()
                pool.join()

                for p in processes:
                    err += p.get()

                err /= v
            except Exception:
                # Fallback to sequential if multiprocessing fails
                for i in range(v):
                    test_idx = np.array(fold_indices[i])
                    err += _predict_worker(tau, model_cv[i], x[test_idx], y[test_idx])
                err /= v
        else:
            # Sequential evaluation
            for i in range(v):
                test_idx = np.array(fold_indices[i])
                err += _predict_worker(tau, model_cv[i], x[test_idx], y[test_idx])
            err /= v

        return tau[err.argmax()], err

    def get_feature_importance(self, method: str = "variance") -> np.ndarray:
        """Compute feature importance scores.

        Parameters
        ----------
        method : {'variance', 'discriminative'}, default='variance'
            Method for computing importance:
            - 'variance': Total variance across all classes (weighted by priors)
            - 'discriminative': Between-class vs within-class variance ratio (Fisher criterion)

        Returns
        -------
        importance : np.ndarray of shape (n_features,)
            Feature importance scores normalized to sum to 1.0

        Raises
        ------
        ValueError
            If method is not recognized

        Examples
        --------
        >>> model = GMMR().fit(X, y)
        >>> importance = model.get_feature_importance(method='discriminative')
        >>> print(f"Most important feature: {importance.argmax()}")
        """
        check_is_fitted(self, ["mean", "cov", "prop"])

        if method == "variance":
            # Weighted average of variances across classes
            total_var = np.zeros(self.mean.shape[1])
            for c in range(len(self.prop)):
                total_var += self.prop[c, 0] * np.diag(self.cov[c, :, :])
            importance = total_var / (total_var.sum() + 1e-10)

        elif method == "discriminative":
            # Fisher criterion: between-class variance / within-class variance
            overall_mean = np.average(self.mean, axis=0, weights=self.prop.ravel())

            # Between-class scatter
            between = np.zeros(self.mean.shape[1])
            for c in range(len(self.prop)):
                diff = self.mean[c, :] - overall_mean
                between += self.prop[c, 0] * (diff**2)

            # Within-class scatter
            within = np.zeros(self.mean.shape[1])
            for c in range(len(self.prop)):
                within += self.prop[c, 0] * np.diag(self.cov[c, :, :])

            # Fisher ratio
            importance = between / (within + 1e-10)
            importance = importance / (importance.sum() + 1e-10)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'variance' or 'discriminative'")

        return importance

    def get_covariance_diagnostics(self) -> Dict[str, np.ndarray]:
        """Compute diagnostic information about covariance matrices.

        Returns
        -------
        diagnostics : dict
            Dictionary containing:
            - 'condition_numbers': Condition number per class (max_eigenvalue / min_eigenvalue)
            - 'min_eigenvalues': Minimum eigenvalue per class
            - 'max_eigenvalues': Maximum eigenvalue per class
            - 'effective_rank': Number of eigenvalues above threshold per class
            - 'explained_variance_ratio': Proportion of variance explained by each eigenvalue

        Examples
        --------
        >>> model = GMMR().fit(X, y)
        >>> diag = model.get_covariance_diagnostics()
        >>> print(f"Condition numbers: {diag['condition_numbers']}")
        >>> if any(diag['condition_numbers'] > 1e10):
        ...     print("Warning: Some classes have ill-conditioned covariances!")
        """
        check_is_fitted(self, ["L", "classes_"])

        C = len(self.classes_)
        condition_numbers = np.zeros(C)
        min_eigenvalues = np.zeros(C)
        max_eigenvalues = np.zeros(C)
        effective_rank = np.zeros(C, dtype=int)

        for c in range(C):
            max_eig = self.L[c, 0]
            min_eig = self.L[c, -1]

            condition_numbers[c] = max_eig / max(min_eig, 1e-15)
            min_eigenvalues[c] = min_eig
            max_eigenvalues[c] = max_eig
            effective_rank[c] = np.sum(self.L[c, :] > self.min_eigenvalue)

        # Variance explained ratios
        explained_variance_ratio = []
        for c in range(C):
            total_var = np.sum(self.L[c, :])
            explained_variance_ratio.append(self.L[c, :] / (total_var + 1e-10))

        return {
            "condition_numbers": condition_numbers,
            "min_eigenvalues": min_eigenvalues,
            "max_eigenvalues": max_eigenvalues,
            "effective_rank": effective_rank,
            "explained_variance_ratio": explained_variance_ratio,
        }

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickle serialization."""
        return {
            "tau": self.tau,
            "random_state": self.random_state,
            "min_eigenvalue": self.min_eigenvalue,
            "warn_ill_conditioned": self.warn_ill_conditioned,
            "n_features_in_": getattr(self, "n_features_in_", None),
            "ni": getattr(self, "ni", None),
            "prop": getattr(self, "prop", None),
            "mean": getattr(self, "mean", None),
            "cov": getattr(self, "cov", None),
            "Q": getattr(self, "Q", None),
            "L": getattr(self, "L", None),
            "classnum": getattr(self, "classnum", None),
            "classes_": getattr(self, "classes_", None),
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state for pickle deserialization."""
        self.tau = state["tau"]
        self.random_state = state["random_state"]
        self.min_eigenvalue = state["min_eigenvalue"]
        self.warn_ill_conditioned = state["warn_ill_conditioned"]
        self.n_features_in_ = state["n_features_in_"]
        self.ni = state["ni"]
        self.prop = state["prop"]
        self.mean = state["mean"]
        self.cov = state["cov"]
        self.Q = state["Q"]
        self.L = state["L"]
        self.classnum = state["classnum"]
        self.classes_ = state["classes_"]


# Backward compatibility alias for factory pattern
ridge = GMMR
