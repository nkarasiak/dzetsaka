"""Single source of truth for dzetsaka optional dependency bundles."""

from __future__ import annotations

FULL_DEPENDENCY_BUNDLE = [
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "shap",
    "seaborn",
    "imbalanced-learn",
]

# Used in guided/settings dialogs where the runtime key differs from pip package name.
RUNTIME_TO_PIP_PACKAGE = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "optuna": "optuna",
    "shap": "shap",
    "seaborn": "seaborn",
    "imblearn": "imbalanced-learn",
    "imblearn (SMOTE)": "imbalanced-learn",
    "scikit-learn": "scikit-learn",
    "imbalanced-learn": "imbalanced-learn",
}


def full_bundle_label() -> str:
    """Human-readable label for full dependency bundle."""
    return ", ".join(FULL_DEPENDENCY_BUNDLE)
