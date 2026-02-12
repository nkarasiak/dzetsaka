"""Nested cross-validation for unbiased model evaluation.

This module implements nested cross-validation, which provides unbiased
performance estimates by separating hyperparameter tuning from model evaluation.

Architecture:
-------------
- Outer loop: Model evaluation (test performance)
- Inner loop: Hyperparameter tuning (parameter selection)

This separation ensures that test data is never used for parameter tuning,
providing realistic performance estimates.

Example Usage:
--------------
    >>> from scripts.validation.nested_cv import NestedCrossValidator
    >>>
    >>> validator = NestedCrossValidator(inner_cv=3, outer_cv=5)
    >>> results = validator.evaluate(X, y, "RF", param_grid)
    >>> print(f"Mean accuracy: {results['mean_score']:.3f} ± {results['std_score']:.3f}")

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

License:
--------
GNU General Public License v2.0 or later

"""

from typing import Any, Callable, Dict, Optional

import numpy as np

# Try to import sklearn
try:
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import Optuna
try:
    from ..optimization.optuna_optimizer import OptunaOptimizer

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None

# Try to import dzetsaka modules
try:
    from ..domain.exceptions import DependencyError, ValidationError
except ImportError:
    # Fallback exceptions
    class DependencyError(Exception):
        """Dependency error fallback."""

    class ValidationError(Exception):
        """Validation error fallback."""


class NestedCrossValidator:
    """Nested cross-validation for unbiased model evaluation.

    Performs nested CV with:
    - Inner loop: Hyperparameter tuning (finds best parameters)
    - Outer loop: Model evaluation (estimates true performance)

    This approach prevents information leakage from test data into
    hyperparameter selection, providing unbiased performance estimates.

    Parameters
    ----------
    inner_cv : int, default=3
        Number of folds for inner loop (hyperparameter tuning)
        Smaller values = faster but less reliable tuning
        Larger values = slower but more reliable tuning
    outer_cv : int, default=5
        Number of folds for outer loop (model evaluation)
        This determines how many independent test sets are used
    random_state : int, default=42
        Random seed for reproducibility
    use_optuna : bool, default=False
        Use Optuna for hyperparameter tuning (faster than GridSearchCV)
    n_trials : int, default=50
        Number of Optuna trials per inner fold (only if use_optuna=True)

    Attributes
    ----------
    inner_cv : int
        Inner CV folds
    outer_cv : int
        Outer CV folds
    random_state : int
        Random seed
    use_optuna : bool
        Whether to use Optuna
    n_trials : int
        Optuna trials per fold

    Example
    -------
    >>> # Standard nested CV with GridSearchCV
    >>> validator = NestedCrossValidator(inner_cv=3, outer_cv=5)
    >>> param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]}
    >>> results = validator.evaluate(X, y, "RF", param_grid)
    >>>
    >>> # Faster nested CV with Optuna
    >>> validator = NestedCrossValidator(use_optuna=True, n_trials=50)
    >>> results = validator.evaluate(X, y, "RF", param_grid)

    """

    def __init__(
        self,
        inner_cv: int = 3,
        outer_cv: int = 5,
        random_state: int = 42,
        use_optuna: bool = False,
        n_trials: int = 50,
    ):
        """Initialize nested cross-validator."""
        if not SKLEARN_AVAILABLE:
            raise DependencyError(
                package_name="scikit-learn",
                reason="Nested CV requires scikit-learn",
                required_version=">=1.0.0",
            )

        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.random_state = random_state
        self.use_optuna = use_optuna
        self.n_trials = n_trials

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier_code: str,
        param_grid: Dict[str, Any],
        scoring: str = "accuracy",
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Perform nested cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training labels, shape (n_samples,)
        classifier_code : str
            Classifier code (e.g., 'RF', 'SVM', 'XGB')
        param_grid : dict
            Parameter grid for hyperparameter tuning
        scoring : str, default='accuracy'
            Scoring metric ('accuracy', 'f1_macro', 'roc_auc', etc.)
        progress_callback : callable, optional
            Callback function for progress updates (receives percentage 0-100)

        Returns
        -------
        results : dict
            - 'outer_scores': List of outer fold scores
            - 'mean_score': Mean performance across outer folds
            - 'std_score': Standard deviation across outer folds
            - 'best_params_per_fold': Best parameters from each outer fold
            - 'inner_scores_per_fold': Inner CV scores per outer fold

        Raises
        ------
        ValidationError
            If input data is invalid

        Example
        -------
        >>> results = validator.evaluate(X, y, "RF", param_grid)
        >>> print(f"Performance: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
        >>> print(f"Best params: {results['best_params_per_fold'][0]}")

        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValidationError("Nested CV input", "X cannot be None or empty")

        if y is None or len(y) == 0:
            raise ValidationError("Nested CV input", "y cannot be None or empty")

        if len(X) != len(y):
            raise ValidationError("Nested CV input", f"X and y must have same length (X: {len(X)}, y: {len(y)})")

        # Import classifier factory
        try:
            from ..factories.classifier_factory import ClassifierFactory

            factory = ClassifierFactory()
        except ImportError as err:
            raise DependencyError(
                package_name="dzetsaka.factories",
                reason="Classifier factory required for nested CV",
            ) from err

        # Setup outer CV
        outer_cv = StratifiedKFold(
            n_splits=self.outer_cv,
            shuffle=True,
            random_state=self.random_state,
        )

        # Storage for results
        outer_scores = []
        best_params_per_fold = []
        inner_scores_per_fold = []

        total_folds = self.outer_cv
        current_fold = 0

        # Outer loop: Model evaluation
        for current_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner loop: Hyperparameter tuning
            if self.use_optuna and OPTUNA_AVAILABLE:
                # Use Optuna for faster tuning
                best_params, inner_score = self._tune_with_optuna(
                    X_train,
                    y_train,
                    classifier_code,
                    param_grid,
                    scoring,
                )
            else:
                # Use GridSearchCV
                best_params, inner_score = self._tune_with_gridsearch(
                    X_train,
                    y_train,
                    classifier_code,
                    param_grid,
                    scoring,
                )

            best_params_per_fold.append(best_params)
            inner_scores_per_fold.append(inner_score)

            # Train model with best parameters on outer train set
            model = factory.create(classifier_code, **best_params)
            model.fit(X_train, y_train)

            # Evaluate on outer test set
            score = model.score(X_test, y_test)
            outer_scores.append(score)

            # Progress callback
            if progress_callback:
                progress_pct = int((current_fold / total_folds) * 100)
                progress_callback(progress_pct)

        # Compute summary statistics
        return {
            "outer_scores": outer_scores,
            "mean_score": float(np.mean(outer_scores)),
            "std_score": float(np.std(outer_scores)),
            "best_params_per_fold": best_params_per_fold,
            "inner_scores_per_fold": inner_scores_per_fold,
        }


    def _tune_with_gridsearch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier_code: str,
        param_grid: Dict[str, Any],
        scoring: str,
    ) -> tuple[Dict[str, Any], float]:
        """Tune hyperparameters using GridSearchCV.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        classifier_code : str
            Classifier code
        param_grid : dict
            Parameter grid
        scoring : str
            Scoring metric

        Returns
        -------
        best_params : dict
            Best parameters found
        best_score : float
            Best cross-validation score

        """
        from ..factories.classifier_factory import ClassifierFactory

        factory = ClassifierFactory()

        # Create base model
        base_model = factory.create(classifier_code)

        # Setup inner CV
        inner_cv = StratifiedKFold(
            n_splits=self.inner_cv,
            shuffle=True,
            random_state=self.random_state,
        )

        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,  # Use all cores
        )

        grid_search.fit(X, y)

        return grid_search.best_params_, grid_search.best_score_

    def _tune_with_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier_code: str,
        param_grid: Dict[str, Any],
        scoring: str,
    ) -> tuple[Dict[str, Any], float]:
        """Tune hyperparameters using Optuna.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        classifier_code : str
            Classifier code
        param_grid : dict
            Parameter grid
        scoring : str
            Scoring metric

        Returns
        -------
        best_params : dict
            Best parameters found
        best_score : float
            Best cross-validation score

        """
        optimizer = OptunaOptimizer(
            n_trials=self.n_trials,
            n_splits=self.inner_cv,
            random_state=self.random_state,
        )

        best_params, best_score = optimizer.optimize(
            X=X,
            y=y,
            classifier_code=classifier_code,
            param_grid=param_grid,
            scoring=scoring,
        )

        return best_params, best_score


def perform_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier_code: str,
    param_grid: Dict[str, Any],
    inner_cv: int = 3,
    outer_cv: int = 5,
    use_optuna: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Convenience function for nested cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    classifier_code : str
        Classifier code (e.g., 'RF', 'SVM')
    param_grid : dict
        Parameter grid for tuning
    inner_cv : int, default=3
        Inner CV folds
    outer_cv : int, default=5
        Outer CV folds
    use_optuna : bool, default=False
        Use Optuna instead of GridSearchCV
    random_state : int, default=42
        Random seed

    Returns
    -------
    results : dict
        Nested CV results

    Example
    -------
    >>> results = perform_nested_cv(X, y, "RF", param_grid)
    >>> print(f"Performance: {results['mean_score']:.3f}")

    """
    validator = NestedCrossValidator(
        inner_cv=inner_cv,
        outer_cv=outer_cv,
        random_state=random_state,
        use_optuna=use_optuna,
    )

    return validator.evaluate(X, y, classifier_code, param_grid)
