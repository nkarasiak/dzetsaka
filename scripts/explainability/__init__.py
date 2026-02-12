"""Model explainability module for dzetsaka.

This module provides SHAP-based feature importance and model explanation
capabilities for all dzetsaka classification algorithms.

Public API:
-----------
- ModelExplainer: Main class for computing SHAP values and feature importance
- check_shap_available: Check if SHAP library is installed
- SHAP_AVAILABLE: Boolean flag indicating SHAP availability

Example:
-------
    >>> from scripts.explainability import ModelExplainer, SHAP_AVAILABLE
    >>>
    >>> if SHAP_AVAILABLE:
    ...     explainer = ModelExplainer(model, feature_names=["B1", "B2", "B3"])
    ...     importance = explainer.get_feature_importance(X_test)
    ... else:
    ...     print("Install SHAP with: pip install shap>=0.41.0")

"""

# Try to import SHAP explainer components
try:
    from .shap_explainer import SHAP_AVAILABLE, ModelExplainer, check_shap_available

    __all__ = ["SHAP_AVAILABLE", "ModelExplainer", "check_shap_available"]
except ImportError:
    # SHAP not available - provide graceful fallback
    SHAP_AVAILABLE = False

    def check_shap_available():
        """Fallback function when SHAP is not available."""
        return False, None

    # Don't export ModelExplainer if SHAP unavailable
    __all__ = ["SHAP_AVAILABLE", "check_shap_available"]
