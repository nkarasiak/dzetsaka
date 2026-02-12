from .dashboard_widget import (
    ClassificationSetupDialog,
    ClassificationDashboardDock,
    check_dependency_availability,
)
from .comparison_panel import AlgorithmComparisonPanel
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
    ]


