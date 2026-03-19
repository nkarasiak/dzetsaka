"""Single source of truth for dzetsaka optional dependency bundles."""

from __future__ import annotations

FULL_DEPENDENCY_BUNDLE = [
    "scikit-learn",
    "xgboost",
    "catboost",
    "optuna",
    "shap",
    "seaborn",
    "imbalanced-learn",
]

def full_bundle_label() -> str:
    """Human-readable label for full dependency bundle."""
    return ", ".join(FULL_DEPENDENCY_BUNDLE)
