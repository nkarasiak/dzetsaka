# Visual Recipe Shop Integration Guide

This document explains how to integrate the new `VisualRecipeShopDialog` into the existing dzetsaka guided workflow.

## Overview

The `VisualRecipeShopDialog` provides a modern, card-based UI for browsing and selecting recipes. It's designed to replace or complement the existing text-based recipe selection.

## Files Created

1. **`ui/recipe_shop_visual.py`** - Main dialog implementation
2. **`ui/recipe_shop_visual_demo.py`** - Standalone demo script
3. **`ui/recipe_shop_visual_integration.md`** - This integration guide

## Quick Integration

### Option 1: Replace Existing Recipe Shop Button

In `guided_workflow_widget.py`, find the recipe shop button (search for "Browse Recipe Shop" or similar) and replace its click handler:

```python
# Add import at top of file
from ui.recipe_shop_visual import show_visual_recipe_shop

# In QuickClassificationPanel or ClassificationDashboardDock class:
def _open_visual_recipe_shop(self):
    """Open visual recipe shop dialog."""
    # Get current recipes
    recipes = load_recipes(QSettings())

    # Check dependencies
    available_deps = self._check_dependencies()  # Or use existing dependency check

    # Show dialog
    selected_recipe = show_visual_recipe_shop(recipes, available_deps, self)

    if selected_recipe:
        # Apply recipe to UI
        self._apply_selected_recipe(selected_recipe)

        # Optionally show confirmation
        QMessageBox.information(
            self,
            "Recipe Applied",
            f"Applied recipe: {selected_recipe.get('name')}"
        )
```

### Option 2: Add as New Button

Add a new "Visual Recipe Shop" button alongside the existing shop:

```python
# In _init_ui() or wherever buttons are created:
visual_shop_btn = QPushButton("🏪 Visual Recipe Shop")
visual_shop_btn.clicked.connect(self._open_visual_recipe_shop)
layout.addWidget(visual_shop_btn)
```

### Option 3: Make It the Default

Replace the old recipe selection combo box with a button that opens the visual shop:

```python
# Instead of QComboBox for recipes:
recipe_shop_btn = QPushButton("Browse Recipes...")
recipe_shop_btn.setMinimumHeight(40)
recipe_shop_btn.clicked.connect(self._open_visual_recipe_shop)

# Add small label showing currently selected recipe
self.current_recipe_label = QLabel("No recipe selected")
self.current_recipe_label.setStyleSheet("color: gray; font-style: italic;")

layout.addWidget(recipe_shop_btn)
layout.addWidget(self.current_recipe_label)
```

## Dependency Checking

The dialog requires a dictionary of dependency availability. Here's a helper function:

```python
def _check_dependencies(self):
    """Check which ML dependencies are available.

    Returns:
        Dict[str, bool]: Dependency availability
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
```

Or reuse existing dependency checks:

```python
from ui.guided_workflow_widget import check_dependency_availability

# Later in code:
available_deps = check_dependency_availability()
```

## Recipe Application

The dialog emits a `recipeSelected` signal and returns the selected recipe. Apply it like this:

```python
def _apply_selected_recipe(self, recipe):
    """Apply recipe to current UI state."""
    # Extract values from recipe
    classifier = recipe.get("classifier", {})
    classifier_code = classifier.get("code", "GMM")

    extra_params = recipe.get("extraParam", {})
    validation = recipe.get("validation", {})

    # Update UI elements
    # (This depends on your specific UI structure)
    if hasattr(self, "algorithm_combo"):
        # Find index of classifier code
        for i in range(self.algorithm_combo.count()):
            if self.algorithm_combo.itemData(i) == classifier_code:
                self.algorithm_combo.setCurrentIndex(i)
                break

    # Update checkboxes
    if hasattr(self, "optuna_checkbox"):
        self.optuna_checkbox.setChecked(extra_params.get("USE_OPTUNA", False))

    if hasattr(self, "shap_checkbox"):
        self.shap_checkbox.setChecked(extra_params.get("COMPUTE_SHAP", False))

    if hasattr(self, "smote_checkbox"):
        self.smote_checkbox.setChecked(extra_params.get("USE_SMOTE", False))

    # Update split percent
    if hasattr(self, "split_spinbox"):
        self.split_spinbox.setValue(validation.get("split_percent", 70))

    # Store full recipe for later reference
    self._current_recipe = recipe

    # Update current recipe label if it exists
    if hasattr(self, "current_recipe_label"):
        recipe_name = recipe.get("name", "Custom")
        self.current_recipe_label.setText(f"Current: {recipe_name}")
```

## Testing the Integration

### Standalone Test

Run the demo script to test the dialog independently:

```bash
cd dzetsaka
python -m ui.recipe_shop_visual_demo
```

### In QGIS

1. Install the plugin as usual
2. Open the dzetsaka dashboard
3. Click the new "Visual Recipe Shop" button
4. Browse recipes, click cards, and apply

## Features

### User Features

- **Visual cards**: Each recipe shown as a colorful card with emoji
- **Category filtering**: Filter by Beginner, Intermediate, Advanced, or My Recipes
- **Search**: Real-time search by name, algorithm, or keyword
- **Dependency indicators**: Shows which packages are missing
- **Recipe preview**: Click card to see full description
- **Quick apply**: Click "Apply" button on card for instant application

### Developer Features

- **Self-contained**: Single file with no external dependencies (except Qt)
- **Reusable**: Can be used in any Qt application
- **Extensible**: Easy to add new card fields or filters
- **Responsive**: Automatically adjusts card grid on window resize
- **Theme-aware**: Works with light and dark Qt themes

## Customization

### Change Card Appearance

Edit constants at top of `recipe_shop_visual.py`:

```python
CARD_WIDTH = 180  # Make cards wider
CARD_HEIGHT = 220  # Make cards taller

CATEGORY_COLORS = {
    "beginner": "#your_color",  # Change background colors
    # ...
}
```

### Add New Filters

Add buttons and filter logic in `_filter_category()`:

```python
self.algorithm_filter_btn = QPushButton("XGBoost Only")
self.algorithm_filter_btn.setCheckable(True)
self.algorithm_filter_btn.clicked.connect(
    lambda: self._filter_algorithm("XGB")
)
```

### Show More Card Info

Edit `RecipeCard._init_ui()` to add more labels:

```python
# Add expected bands
metadata = self.recipe.get("metadata", {})
bands_text = metadata.get("expected_bands", "any")
bands_label = QLabel(f"Bands: {bands_text}")
layout.addWidget(bands_label)
```

## Migration Path

### Phase 1: Parallel (Safe)
- Keep existing recipe combo box
- Add "Visual Shop" button next to it
- Let users choose which interface they prefer

### Phase 2: Primary (Recommended)
- Make visual shop the default
- Keep old combo as fallback (hidden by default)
- Add setting to toggle between views

### Phase 3: Complete (Optional)
- Remove old recipe UI completely
- Visual shop becomes only interface
- Simplify codebase

## Known Limitations

1. **No drag-and-drop**: Cards don't support reordering (yet)
2. **No inline editing**: Must use "Create Copy" to modify recipes
3. **Fixed grid**: Not a Pinterest-style masonry layout
4. **No animations**: Card transitions are instant (could add Qt animations)

## Future Enhancements

Potential improvements for future versions:

- **Favorites/Pins**: Pin frequently used recipes to top
- **Recent recipes**: Show recently used recipes in separate tab
- **Recipe rating**: Let users rate/star recipes
- **Recipe sharing**: Export/import recipes via URL or file
- **Preview mode**: Show example classification results
- **Tooltips**: Rich tooltips with more recipe details on hover
- **Keyboard navigation**: Arrow keys to navigate cards

## Support

For questions or issues with the visual recipe shop:

1. Check this integration guide
2. Look at `recipe_shop_visual_demo.py` for examples
3. Review the docstrings in `recipe_shop_visual.py`
4. Open an issue on the dzetsaka GitHub repository
