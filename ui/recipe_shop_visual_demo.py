"""Demo script for visual recipe shop dialog.

Shows how to integrate VisualRecipeShopDialog with existing dzetsaka code.
Can be run standalone for testing or imported into guided_workflow_widget.py.

Usage:
    python -m ui.recipe_shop_visual_demo
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qgis.PyQt.QtWidgets import QApplication
from ui.recipe_shop_visual import show_visual_recipe_shop
from ui.guided_workflow_widget import load_builtin_recipes


def check_dependency_availability():
    """Check which ML dependencies are available.

    Returns:
        Dict of dependency availability
    """
    deps = {}

    try:
        import sklearn
        deps["sklearn"] = True
    except ImportError:
        deps["sklearn"] = False

    try:
        import xgboost
        deps["xgboost"] = True
    except ImportError:
        deps["xgboost"] = False

    try:
        import lightgbm
        deps["lightgbm"] = True
    except ImportError:
        deps["lightgbm"] = False

    try:
        import catboost
        deps["catboost"] = True
    except ImportError:
        deps["catboost"] = False

    try:
        import optuna
        deps["optuna"] = True
    except ImportError:
        deps["optuna"] = False

    try:
        import shap
        deps["shap"] = True
    except ImportError:
        deps["shap"] = False

    try:
        import imblearn
        deps["imblearn"] = True
    except ImportError:
        deps["imblearn"] = False

    return deps


def demo_visual_recipe_shop():
    """Run visual recipe shop demo."""
    # Load built-in recipes
    recipes = load_builtin_recipes()

    # Check dependencies
    available_deps = check_dependency_availability()

    print(f"Loaded {len(recipes)} recipes")
    print(f"Available dependencies: {[k for k, v in available_deps.items() if v]}")

    # Create Qt application
    app = QApplication(sys.argv)

    # Show dialog
    selected_recipe = show_visual_recipe_shop(recipes, available_deps)

    if selected_recipe:
        print("\nSelected recipe:")
        print(f"  Name: {selected_recipe.get('name')}")
        print(f"  Algorithm: {selected_recipe.get('classifier', {}).get('name')}")
        print(f"  Category: {selected_recipe.get('metadata', {}).get('category')}")
        return 0
    else:
        print("\nNo recipe selected (dialog canceled)")
        return 1


if __name__ == "__main__":
    sys.exit(demo_visual_recipe_shop())
