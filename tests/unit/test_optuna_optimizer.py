"""Unit tests for Optuna optimizer.

These tests verify the OptunaOptimizer class functionality including:
- Parameter suggestion for different classifiers
- Classifier creation
- Optimization workflow
- Error handling

Author:
    Nicolas Karasiak

"""

import numpy as np
import pytest

# Skip all tests if optuna is not available
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Skip all tests if sklearn is not available
try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

skip_if_no_optuna = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
skip_if_no_sklearn = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")


@skip_if_no_optuna
@skip_if_no_sklearn
class TestOptunaOptimizer:
    """Test suite for OptunaOptimizer class."""

    def test_optimizer_initialization(self):
        """Test OptunaOptimizer can be initialized."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer("RF", n_trials=10)
        assert optimizer.classifier_code == "RF"
        assert optimizer.n_trials == 10

    def test_suggest_params_rf(self):
        """Test parameter suggestion for Random Forest."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer("RF", n_trials=10)
        study = optuna.create_study()
        trial = study.ask()
        params = optimizer._suggest_params(trial, "RF")

        # Check required parameters are present
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "max_features" in params
        assert "random_state" in params

    def test_suggest_params_svm(self):
        """Test parameter suggestion for SVM."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer("SVM", n_trials=10)
        study = optuna.create_study()
        trial = study.ask()
        params = optimizer._suggest_params(trial, "SVM")

        # Check required parameters are present
        assert "C" in params
        assert "kernel" in params
        assert "random_state" in params

    def test_create_classifier_rf(self):
        """Test classifier creation for Random Forest."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer("RF", n_trials=10)
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }
        clf = optimizer._create_classifier("RF", params)

        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(clf, RandomForestClassifier)
        assert clf.n_estimators == 100
        assert clf.max_depth == 10

    def test_create_classifier_svm(self):
        """Test classifier creation for SVM."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer("SVM", n_trials=10)
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "random_state": 42,
        }
        clf = optimizer._create_classifier("SVM", params)

        from sklearn.svm import SVC

        assert isinstance(clf, SVC)
        assert clf.C == 1.0
        assert clf.kernel == "rbf"

    def test_optimize_workflow(self):
        """Test full optimization workflow with synthetic data."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        # Create synthetic data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)

        # Run optimization with few trials
        optimizer = OptunaOptimizer("RF", n_trials=5, random_seed=42)
        best_params = optimizer.optimize(X, y, cv=2)

        # Check optimization completed
        assert best_params is not None
        assert "n_estimators" in best_params
        assert optimizer.study is not None
        assert optimizer.study.best_value > 0

    def test_invalid_classifier_code(self):
        """Test error handling for invalid classifier code."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer("INVALID_CODE", n_trials=10)
        study = optuna.create_study()
        trial = study.ask()

        with pytest.raises(ValueError, match="Unknown classifier code"):
            optimizer._suggest_params(trial, "INVALID_CODE")

    def test_optimization_history(self):
        """Test getting optimization history."""
        from scripts.optimization.optuna_optimizer import OptunaOptimizer

        # Create synthetic data
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)

        # Run optimization
        optimizer = OptunaOptimizer("RF", n_trials=3, random_seed=42)
        optimizer.optimize(X, y, cv=2)

        # Get history
        history = optimizer.get_optimization_history()
        assert history is not None
        assert "n_trials" in history
        assert "best_value" in history
        assert "best_params" in history
        assert history["n_trials"] >= 3


@skip_if_no_optuna
def test_optuna_import():
    """Test that optuna can be imported."""
    import optuna

    assert hasattr(optuna, "create_study")
    assert hasattr(optuna, "Trial")


@skip_if_no_sklearn
def test_sklearn_import():
    """Test that sklearn can be imported."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    assert RandomForestClassifier is not None
    assert cross_val_score is not None


@skip_if_no_optuna
@skip_if_no_sklearn
def test_module_constants():
    """Test module-level constants."""
    from scripts.optimization.optuna_optimizer import OPTUNA_AVAILABLE, SKLEARN_AVAILABLE

    # These should be boolean
    assert isinstance(OPTUNA_AVAILABLE, bool)
    assert isinstance(SKLEARN_AVAILABLE, bool)
