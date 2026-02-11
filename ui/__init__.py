from .dashboard_widget import (
    ClassificationSetupDialog,
    ClassificationDashboardDock,
    check_dependency_availability,
)
from .comparison_panel import AlgorithmComparisonPanel
from .results_explorer_dock import ResultsExplorerDock, open_results_explorer
from .validated_widgets import ValidatedDoubleSpinBox, ValidatedLineEdit, ValidatedSpinBox

# Optional recommendation system (may not have GDAL available)
try:
    from .recipe_recommender import RasterAnalyzer, RecipeRecommender
    from .recommendation_dialog import RecommendationDialog

    __all__ = [
        "ClassificationDashboardDock",
        "ClassificationSetupDialog",
        "check_dependency_availability",
        "AlgorithmComparisonPanel",
        "ValidatedSpinBox",
        "ValidatedDoubleSpinBox",
        "ValidatedLineEdit",
        "ResultsExplorerDock",
        "open_results_explorer",
        "RasterAnalyzer",
        "RecipeRecommender",
        "RecommendationDialog",
    ]
except ImportError:
    __all__ = [
        "ClassificationDashboardDock",
        "ClassificationSetupDialog",
        "check_dependency_availability",
        "AlgorithmComparisonPanel",
        "ValidatedSpinBox",
        "ValidatedDoubleSpinBox",
        "ValidatedLineEdit",
        "ResultsExplorerDock",
        "open_results_explorer",
    ]


