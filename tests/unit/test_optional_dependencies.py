"""Smoke tests for optional stacks (Optuna, SHAP, CatBoost)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

optuna_spec = importlib.util.find_spec("optuna")
shap_spec = importlib.util.find_spec("shap")
catboost_spec = importlib.util.find_spec("catboost")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dzetsaka_script_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


optuna_optimizer_mod = _load_dzetsaka_script_module(
    "dzetsaka.scripts.optimization.optuna_optimizer", Path("scripts") / "optimization" / "optuna_optimizer.py",
)


@pytest.mark.skipif(optuna_spec is None, reason="optuna not installed")
def test_optuna_optimizer_runs_minimal_trial() -> None:
    OptunaOptimizer = optuna_optimizer_mod.OptunaOptimizer
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=40, n_features=5, n_informative=3, n_classes=2, random_state=0)
    optimizer = OptunaOptimizer(classifier_code="RF", n_trials=2, random_seed=0, verbose=False)
    best_params = optimizer.optimize(X=X, y=y, cv=2, scoring="f1_weighted")
    assert isinstance(best_params, dict)


@pytest.mark.skipif(shap_spec is None, reason="shap not installed")
def test_shap_tree_explainer_runs() -> None:
    import shap
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=40, n_features=5, n_informative=3, random_state=1)
    model = RandomForestClassifier(n_estimators=5, random_state=1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X[:2])
    assert values is not None


@pytest.mark.skipif(catboost_spec is None, reason="catboost not installed")
def test_catboost_classifier_exposes_sklearn_tags() -> None:
    OptunaOptimizer = optuna_optimizer_mod.OptunaOptimizer

    optimizer = OptunaOptimizer(classifier_code="CB", n_trials=1, random_seed=0, verbose=False)
    clf = optimizer._create_classifier(
        "CB", {"iterations": 1, "depth": 1, "learning_rate": 0.1, "loss_function": "MultiClass"},
    )
    assert hasattr(clf, "__sklearn_tags__") or hasattr(clf.__class__, "__sklearn_tags__")
