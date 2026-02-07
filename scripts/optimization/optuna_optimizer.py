"""Optuna-based hyperparameter optimization for dzetsaka classifiers.

This module provides Bayesian hyperparameter optimization using Optuna,
replacing the exhaustive GridSearchCV with intelligent trial-based optimization.

Benefits over GridSearchCV:
- 2-10x faster optimization through intelligent sampling
- Better parameter combinations through Bayesian optimization
- Early stopping of poor trials via pruning
- Parallel trial execution support
- Progress visualization and logging

Example:
    >>> optimizer = OptunaOptimizer("RF", n_trials=100)
    >>> best_params = optimizer.optimize(X_train, y_train, cv=5)
    >>> clf = RandomForestClassifier(**best_params)

Author:
    Nicolas Karasiak

"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

# Conditional imports
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


from ...logging_utils import QgisLogger


class OptunaOptimizer:
    """Bayesian hyperparameter optimization using Optuna.

    This class provides intelligent hyperparameter search for all supported
    classifiers using Optuna's Tree-structured Parzen Estimator (TPE) algorithm.

    Parameters
    ----------
        classifier_code : str
        Classifier code (e.g., "RF", "SVM", "XGB", "CB")
    n_trials : int, default=100
        Number of optimization trials to run
    timeout : Optional[int], default=None
        Time limit in seconds (None for no limit)
    random_seed : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all cores)
    verbose : bool, default=False
        Enable verbose logging

    Attributes
    ----------
    study : optuna.Study
        Optuna study object containing optimization results

    """

    def __init__(
        self,
        classifier_code: str,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        random_seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install it with: pip install optuna")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Install it with: pip install scikit-learn")

        self.classifier_code = classifier_code.upper()
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.study: Optional[optuna.Study] = None
        self.log = QgisLogger(tag="Dzetsaka/Optuna")

        # Configure Optuna logging
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Union[int, Any] = 5,
        scoring: str = "f1_weighted",
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run Optuna optimization and return best parameters.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix (n_samples, n_features)
        y : np.ndarray
            Training labels (n_samples,)
        cv : Union[int, Any], default=5
            Cross-validation strategy (int or sklearn CV splitter)
        scoring : str, default="f1_weighted"
            Scoring metric for optimization
        groups : np.ndarray, optional
            Group labels for GroupKFold or StratifiedGroupKFold CV splitters

        Returns
        -------
        Dict[str, Any]
            Dictionary of best hyperparameters found

        """
        # Create cross-validation strategy
        if isinstance(cv, int):
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_seed)
        else:
            cv_splitter = cv

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            # Suggest hyperparameters based on classifier
            params = self._suggest_params(trial, self.classifier_code)

            # Create classifier with suggested parameters
            try:
                clf = self._create_classifier(self.classifier_code, params)
            except Exception as e:
                self.log.warning(f"Failed to create classifier with params {params}: {e}")
                raise optuna.TrialPruned() from e

            # Evaluate with cross-validation
            try:
                scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring=scoring, n_jobs=1, groups=groups)
                mean_score = scores.mean()

                # Report intermediate value for pruning
                trial.report(mean_score, step=0)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return mean_score

            except Exception as e:
                self.log.warning(f"Trial failed with params {params}: {e}")
                raise optuna.TrialPruned() from e

        # Create study with TPE sampler and median pruner
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        # Run optimization
        self.log.info(
            f"Starting Optuna optimization for {self.classifier_code}: {self.n_trials} trials, scoring={scoring}"
        )

        self.study.optimize(
            objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs, show_progress_bar=self.verbose
        )

        self.log.info(
            f"Optimization complete. Best score: {self.study.best_value:.4f}, Best params: {self.study.best_params}"
        )

        return self.study.best_params

    def _suggest_params(self, trial: optuna.Trial, classifier_code: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a given classifier.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for suggesting parameters
        classifier_code : str
            Classifier code (e.g., "RF", "SVM")

        Returns
        -------
        Dict[str, Any]
            Dictionary of suggested hyperparameters

        """
        if classifier_code == "RF":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": self.random_seed,
            }

        elif classifier_code == "SVM":
            return {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
                if trial.suggest_categorical("gamma_type", ["fixed", "float"]) == "fixed"
                else trial.suggest_float("gamma_value", 0.001, 1.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                "random_state": self.random_seed,
            }

        elif classifier_code == "KNN":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
                "leaf_size": trial.suggest_int("leaf_size", 10, 50),
            }

        elif classifier_code == "XGB":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": self.random_seed,
                "eval_metric": "logloss",
            }

        elif classifier_code == "LGB":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "random_state": self.random_seed,
                "verbose": -1,
            }

        elif classifier_code == "CB":
            return {
                "iterations": trial.suggest_int("iterations", 50, 500, step=50),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "loss_function": "MultiClass",
                "random_seed": self.random_seed,
                "verbose": False,
                "allow_writing_files": False,
            }

        elif classifier_code == "ET":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": self.random_seed,
            }

        elif classifier_code == "GBC":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": self.random_seed,
            }

        elif classifier_code == "LR":
            return {
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
                "random_state": self.random_seed,
            }

        elif classifier_code == "MLP":
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden_layer_sizes = tuple(trial.suggest_int(f"layer_{i}_size", 50, 300, step=50) for i in range(n_layers))

            return {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
                "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
                "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
                "random_state": self.random_seed,
            }

        elif classifier_code == "NB":
            # Naive Bayes has minimal hyperparameters
            return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-6, log=True)}

        else:
            raise ValueError(f"Unknown classifier code: {classifier_code}")

    def _create_classifier(self, classifier_code: str, params: Dict[str, Any]) -> Any:
        """Create classifier instance with given parameters.

        Parameters
        ----------
        classifier_code : str
            Classifier code (e.g., "RF", "SVM")
        params : Dict[str, Any]
            Hyperparameters for the classifier

        Returns
        -------
        Any
            Classifier instance

        """
        # Handle SVM gamma parameter special case
        if classifier_code == "SVM" and "gamma_value" in params:
            params["gamma"] = params.pop("gamma_value")
            params.pop("gamma_type", None)

        # Handle LR solver/penalty compatibility
        if classifier_code == "LR":
            if params["penalty"] == "elasticnet" and params["solver"] != "saga":
                params["solver"] = "saga"
            elif params["penalty"] == "l1" and params["solver"] not in ["liblinear", "saga"]:
                params["solver"] = "liblinear"
            elif params["penalty"] is None:
                params["solver"] = "lbfgs"

        if classifier_code == "RF":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**params)

        elif classifier_code == "SVM":
            from sklearn.svm import SVC

            return SVC(**params)

        elif classifier_code == "KNN":
            from sklearn.neighbors import KNeighborsClassifier

            return KNeighborsClassifier(**params)

        elif classifier_code == "XGB":
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(**params)
            except ImportError as e:
                raise ImportError("XGBoost is not installed. Install it with: pip install xgboost") from e

        elif classifier_code == "LGB":
            try:
                from lightgbm import LGBMClassifier

                return LGBMClassifier(**params)
            except ImportError as e:
                raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm") from e

        elif classifier_code == "CB":
            try:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(**params)
            except ImportError as e:
                raise ImportError("CatBoost is not installed. Install it with: pip install catboost") from e

        elif classifier_code == "ET":
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(**params)

        elif classifier_code == "GBC":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(**params)

        elif classifier_code == "LR":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**params)

        elif classifier_code == "MLP":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(**params)

        elif classifier_code == "NB":
            from sklearn.naive_bayes import GaussianNB

            return GaussianNB(**params)

        else:
            raise ValueError(f"Unknown classifier code: {classifier_code}")

    def get_optimization_history(self) -> Optional[Dict[str, Any]]:
        """Get optimization history and statistics.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing optimization statistics, or None if not optimized yet

        """
        if self.study is None:
            return None

        return {
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial": self.study.best_trial.number,
            "n_complete": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }
