"""Public ui package exports with lazy imports.

This keeps `import ui.recipe_recommender` usable in non-QGIS environments by
avoiding eager imports of QGIS-heavy modules at package import time.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ClassificationDashboardDock",
    "ClassificationSetupDialog",
    "check_dependency_availability",
    "AlgorithmComparisonPanel",
    "ValidatedSpinBox",
    "ValidatedDoubleSpinBox",
    "ValidatedLineEdit",
    "RasterAnalyzer",
    "RecipeRecommender",
    "RecommendationDialog",
]

_EXPORT_TO_MODULE = {
    "ClassificationDashboardDock": ".dashboard_widget",
    "ClassificationSetupDialog": ".dashboard_widget",
    "check_dependency_availability": ".dashboard_widget",
    "AlgorithmComparisonPanel": ".comparison_panel",
    "ValidatedSpinBox": ".validated_widgets",
    "ValidatedDoubleSpinBox": ".validated_widgets",
    "ValidatedLineEdit": ".validated_widgets",
    "RasterAnalyzer": ".recipe_recommender",
    "RecipeRecommender": ".recipe_recommender",
    "RecommendationDialog": ".recommendation_dialog",
}
_KNOWN_SUBMODULES = {
    "classification_workflow_ui",
    "dashboard_widget",
    "comparison_panel",
    "validated_widgets",
    "recipe_recommender",
    "recommendation_dialog",
    "results_explorer_dock",
}


def __getattr__(name: str):
    if name in _KNOWN_SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


