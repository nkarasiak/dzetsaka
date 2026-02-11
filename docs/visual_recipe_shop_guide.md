# Visual Recipe Shop - User Guide

## Overview

The **Visual Recipe Shop** is a modern, card-based UI for browsing and selecting pre-configured classification workflows in dzetsaka. It provides an intuitive, visual way to explore recipes optimized for different remote sensing tasks.

## Features

### 🎨 Modern Card-Based Interface

Recipes are displayed as beautiful visual cards in a scrollable grid layout, making it easy to browse and compare different workflows at a glance.

### 🏷️ Category Filtering

Filter recipes by expertise level:
- **Beginner** - Simple, fast workflows for getting started
- **Intermediate** - Production-ready workflows with balanced features
- **Advanced** - Comprehensive pipelines for research and publication
- **My Recipes** - User-created custom recipes

### 🔍 Smart Search

Search recipes by:
- Recipe name
- Algorithm type (RF, XGB, SVM, etc.)
- Features (OPTUNA, SHAP, SMOTE)
- Keywords in description

### 📊 Visual Recipe Cards

Each recipe card displays:
- **Category Badge** - Color-coded by difficulty level
- **Algorithm Emoji** - Visual identifier (🌲 for RF, 🚀 for XGB, etc.)
- **Recipe Name** - Clear, descriptive title
- **Runtime Estimate** - ⚡ Fast (<3 min), ⏱️ Medium (5-15 min), ⏳ Slow (30-90 min)
- **Accuracy Range** - Expected classification accuracy (70-80%, 80-92%, 90-98%)
- **Algorithm Summary** - Main algorithm + key features
- **Feature Tags** - OPTUNA, SHAP, SMOTE, NESTED CV, etc.
- **Apply Button** - Quick one-click application

### ✅ Dependency Indicators

The details panel shows:
- ✓ All dependencies available (ready to use)
- ⚠ Missing dependencies (with specific package names)

## Built-in Recipes

### Beginner Recipes

#### 🔵 Quick Test
- **Runtime:** <1 minute
- **Accuracy:** 70-80%
- **Algorithm:** GMM (Gaussian Mixture Model)
- **Use Cases:** Quick sanity checks, workflow testing, baseline comparisons
- **Dependencies:** None (uses built-in GMM)

#### 💧 Water Detection
- **Runtime:** 1-3 minutes
- **Accuracy:** 70-85%
- **Algorithm:** GMM
- **Use Cases:** Water body detection, flood mapping, lake/river extraction
- **Features:** Confidence maps, fast binary classification

### Intermediate Recipes

#### 🌲 Forest Mapping (Landsat)
- **Runtime:** 3-8 minutes
- **Accuracy:** 75-88%
- **Algorithm:** Random Forest + SMOTE + Class Weights
- **Use Cases:** Forest classification, tree species mapping, deforestation monitoring
- **Features:**
  - SMOTE for imbalanced classes
  - Polygon group CV (prevents spatial leakage)
  - Full reporting bundle

#### 🌾 Crop Classification (Sentinel-2)
- **Runtime:** 8-15 minutes
- **Accuracy:** 80-92%
- **Algorithm:** XGBoost + Optuna + SHAP
- **Use Cases:** Agricultural crop type mapping, Sentinel-2 analysis
- **Features:**
  - 150 Optuna trials for hyperparameter optimization
  - SHAP explainability (1500 samples)
  - Polygon group CV
  - Full reporting

#### 🏙️ Urban/Rural Land Use
- **Runtime:** 5-12 minutes
- **Accuracy:** 78-89%
- **Algorithm:** SVM with class weights
- **Use Cases:** Urban land cover, built-up area extraction, high-res imagery
- **Features:**
  - Balanced class weights
  - Polygon group CV
  - High-resolution optimized

#### ⚡ Large Dataset Fast
- **Runtime:** 3-8 minutes
- **Accuracy:** 78-90%
- **Algorithm:** LightGBM
- **Use Cases:** Large rasters (>1GB), speed-critical workflows, production pipelines
- **Features:**
  - Histogram-based gradient boosting
  - Minimal overhead
  - Fast training on big data

### Advanced Recipes

#### 🌈 Hyperspectral Analysis
- **Runtime:** 10-20 minutes
- **Accuracy:** 82-94%
- **Algorithm:** Extra Trees + SHAP
- **Use Cases:** Hyperspectral classification, band importance analysis
- **Features:**
  - SHAP for band importance (2000 samples)
  - Optimized for >10 bands
  - Polygon group CV
  - Full reporting

#### 🏆 Publication Quality
- **Runtime:** 30-90 minutes
- **Accuracy:** 90-98%
- **Algorithm:** CatBoost + Optuna (300 trials) + SHAP + Nested CV
- **Use Cases:** Scientific publications, rigorous validation, benchmark comparisons
- **Features:**
  - 300 Optuna trials (extensive optimization)
  - Nested CV (3 inner × 5 outer folds)
  - SHAP explainability (2500 samples)
  - Polygon group CV
  - Full reproducibility artifacts
  - State-of-the-art accuracy

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 🏪 dzetsaka Recipe Shop                              [X]    │
├─────────────────────────────────────────────────────────────┤
│ [All] [Beginner] [Intermediate] [Advanced] [My Recipes]    │
├─────────────────────────────────────────────────────────────┤
│ Search: [_______________________________] 🔍                │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│ │  BEGINNER    │  │ INTERMEDIATE │  │ INTERMEDIATE │       │
│ │              │  │              │  │              │       │
│ │      🔵      │  │      🌲      │  │      🌾      │       │
│ │              │  │              │  │              │       │
│ │  Quick Test  │  │   Forest     │  │     Crop     │       │
│ │              │  │   Mapping    │  │ Classification│      │
│ │  ⚡ <1 min    │  │  ⏱️ 3-8 min   │  │  ⏱️ 8-15 min  │      │
│ │  📊 70-80%   │  │  📈 75-88%   │  │  📈 80-92%   │       │
│ │              │  │              │  │              │       │
│ │  Gaussian    │  │  Random      │  │  XGBoost +   │       │
│ │  Mixture     │  │  Forest +    │  │  Optuna      │       │
│ │              │  │  SMOTE       │  │              │       │
│ │              │  │              │  │              │       │
│ │  SMOTE       │  │  WEIGHTS     │  │ OPTUNA, SHAP │       │
│ │              │  │              │  │              │       │
│ │   [Apply]    │  │   [Apply]    │  │   [Apply]    │       │
│ └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│ │     More cards in scrollable grid...                     │
│ └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Selected: Crop Classification (Sentinel-2)                  │
│ Description: XGBoost with Optuna hyperparameter            │
│ optimization and SHAP explainability. Optimized for        │
│ agricultural crop type mapping with 10-13 band Sentinel-2  │
│ imagery.                                                    │
│                                                             │
│ Requirements: ✓ All dependencies available                  │
│                                                             │
│ [Apply Recipe] [Create Copy] [View Details]        [Close] │
└─────────────────────────────────────────────────────────────┘
```

## Visual Design

### Color Scheme

**Category Colors:**
- **Beginner:** Light green (#e8f5e9) with green border (#4caf50)
- **Intermediate:** Light blue (#e3f2fd) with blue border (#2196f3)
- **Advanced:** Light orange (#fff3e0) with orange border (#ff9800)
- **My Recipes:** Light purple (#f3e5f5) with purple border (#9c27b0)

**Selection:**
- Selected card: Blue highlight border (#2196f3, 3px)
- Hover effect: Slightly darker background, 2px border

### Typography

- **Category badge:** 9pt, bold, uppercase
- **Recipe name:** 11pt, bold, center-aligned
- **Runtime/Accuracy:** 9pt, with emoji indicators
- **Algorithm summary:** 9pt, italic
- **Feature tags:** 8pt, gray text

### Card Dimensions

- **Card size:** 180×220 pixels (fixed)
- **Border radius:** 8px (rounded corners)
- **Grid spacing:** 12px between cards
- **Cards per row:** 3-5 (responsive, based on window width)

### Algorithm Emojis

- 🔵 GMM - Gaussian Mixture Model
- 🌲 RF - Random Forest
- ⚔️ SVM - Support Vector Machine
- 👥 KNN - K-Nearest Neighbors
- 🚀 XGB - XGBoost
- ⚡ LGB - LightGBM
- 🐈 CB - CatBoost
- 🌳 ET - Extra Trees
- 📊 GBC - Gradient Boosting Classifier
- 📉 LR - Logistic Regression
- 🔮 NB - Naive Bayes
- 🧠 MLP - Multi-Layer Perceptron

## How to Use

### Opening the Recipe Shop

**Option 1: From Dashboard**
1. Open dzetsaka dashboard
2. Click "🏪 Browse Recipe Shop" button

**Option 2: From Code**
```python
from ui.recipe_shop_visual import show_visual_recipe_shop
from ui.classification_workflow_ui import load_builtin_recipes

# Load recipes
recipes = load_builtin_recipes()

# Check dependencies
available_deps = {
    "sklearn": True,
    "xgboost": True,
    "lightgbm": False,
    "catboost": False,
    "optuna": True,
    "shap": True,
    "imblearn": True,
}

# Show dialog
selected_recipe = show_visual_recipe_shop(recipes, available_deps)

if selected_recipe:
    print(f"Selected: {selected_recipe['name']}")
```

### Browsing Recipes

1. **Filter by Category:** Click category buttons (All, Beginner, Intermediate, Advanced, My Recipes)
2. **Search:** Type in search box to filter by name, algorithm, or keywords
3. **View Card:** Click a card to see full details in bottom panel
4. **Check Dependencies:** Selected recipe shows dependency status

### Applying a Recipe

**Method 1: Quick Apply**
- Click "Apply" button directly on recipe card

**Method 2: From Selection**
1. Click recipe card to select
2. Review details in bottom panel
3. Click "Apply Recipe" button

### Additional Actions

- **Create Copy:** Duplicate selected recipe to customize (appears in "My Recipes")
- **View Details:** See full recipe JSON configuration
- **Close:** Cancel without applying

## Integration with Workflow

When a recipe is applied:

1. All classification parameters are updated to match the recipe
2. Algorithm selection changes to recipe's algorithm
3. Validation split, CV mode, and advanced features are configured
4. User sees confirmation dialog with recipe name
5. Current recipe label updates to show selected recipe

## Customization

### Creating Custom Recipes

1. Configure classification parameters in main panel
2. Click "Save Current as Recipe..."
3. Enter recipe name and description
4. Recipe appears in "My Recipes" tab

### Recipe Schema

Recipes use schema v2 with the following structure:

```json
{
  "name": "Recipe Name",
  "description": "Description...",
  "schema_version": 2,
  "expected_runtime_class": "fast|medium|slow",
  "expected_accuracy_class": "medium|high|very_high",
  "classifier": {
    "code": "XGB",
    "name": "XGBoost"
  },
  "extraParam": {
    "USE_OPTUNA": true,
    "COMPUTE_SHAP": true,
    "USE_SMOTE": false,
    ...
  },
  "metadata": {
    "category": "beginner|intermediate|advanced|custom",
    "is_template": false,
    "use_cases": ["..."],
    ...
  }
}
```

## Keyboard Shortcuts

- **Enter/Double-click:** Apply selected recipe
- **Escape:** Close dialog
- **Ctrl+F:** Focus search box (if implemented)

## Accessibility

- Card selection visible via border highlighting
- Dependency warnings clearly marked with ⚠ symbol
- Color-coded categories with text labels
- High contrast text on all backgrounds

## Performance

- **Responsive layout:** Cards automatically reflow on window resize
- **Efficient filtering:** Search and category filters update instantly
- **Lazy loading:** Cards created on-demand for large recipe collections

## Technical Details

### Files

- `ui/recipe_shop_visual.py` - Main dialog implementation
- `ui/recipe_shop_visual_demo.py` - Standalone demo script
- `ui/recipe_shop_visual_example_integration.py` - Integration examples
- `ui/recipe_shop_visual_styles.qss` - Optional enhanced stylesheet
- `ui/builtin_recipes.json` - Built-in recipe templates

### Classes

- **`VisualRecipeShopDialog`** - Main dialog window
- **`RecipeCard`** - Individual recipe card widget
- **`show_visual_recipe_shop()`** - Convenience function

### Dependencies

- QGIS PyQt (PyQt5/PyQt6 compatible)
- No external dependencies beyond QGIS

## Troubleshooting

### Recipe Shop Not Opening

- Check that `ui/recipe_shop_visual.py` exists
- Verify import: `from ui.recipe_shop_visual import show_visual_recipe_shop`
- Check console for import errors

### No Recipes Showing

- Verify `ui/builtin_recipes.json` exists and is valid JSON
- Check that recipes have required fields (name, classifier, metadata)
- Use demo script to test: `python -m ui.recipe_shop_visual_demo`

### Cards Not Displaying Correctly

- Check window size (minimum 700×600 recommended)
- Verify card layout is not corrupted
- Try resizing window to trigger card reflow

### Missing Dependencies Warning

- Install required packages: `pip install scikit-learn xgboost lightgbm catboost optuna shap imbalanced-learn`
- Use dzetsaka's built-in dependency installer
- Some recipes work without all dependencies (check recipe requirements)

## Future Enhancements

Potential improvements for future versions:

- **Recipe ratings:** User ratings and favorites
- **Recipe sharing:** Export/import recipes as files
- **Community recipes:** Browse online recipe repository
- **Recipe versioning:** Track recipe history and changes
- **Preview mode:** Estimate runtime/accuracy before applying
- **Comparison view:** Side-by-side recipe comparison
- **Dark mode:** Full dark theme support
- **Animations:** Smooth transitions and hover effects

## Examples

### Example 1: Quick Forest Mapping

```python
# User wants fast forest classification
# 1. Opens Recipe Shop
# 2. Clicks "Intermediate" tab
# 3. Searches "forest"
# 4. Clicks "Forest Mapping (Landsat)" card
# 5. Reviews: Runtime 3-8 min, Accuracy 75-88%, RF+SMOTE
# 6. Checks: ✓ All dependencies available
# 7. Clicks "Apply Recipe"
# 8. Starts classification
```

### Example 2: Publication-Quality Results

```python
# Researcher needs rigorous validation for paper
# 1. Opens Recipe Shop
# 2. Clicks "Advanced" tab
# 3. Finds "Publication Quality" card
# 4. Reviews comprehensive features:
#    - CatBoost algorithm
#    - 300 Optuna trials
#    - Nested CV (3×5 folds)
#    - SHAP explainability
#    - Full reporting
# 5. Sees: ⚠ Missing catboost
# 6. Installs catboost via dzetsaka dependency installer
# 7. Returns to Recipe Shop
# 8. Applies "Publication Quality" recipe
# 9. Starts long-running (30-90 min) comprehensive classification
```

### Example 3: Custom Recipe from Template

```python
# User wants to customize "Crop Classification"
# 1. Opens Recipe Shop
# 2. Finds "Crop Classification" card
# 3. Clicks "Create Copy"
# 4. Copy appears in "My Recipes" tab
# 5. Closes Recipe Shop
# 6. Modifies parameters in main panel:
#    - Increases Optuna trials to 300
#    - Changes split to 80%
# 7. Saves modified version
# 8. Custom recipe now available in "My Recipes"
```

## Support

For issues or questions:
- Check CLAUDE.md for development guidelines
- Review ui/recipe_shop_visual_integration.md for integration details
- Run demo: `python -m ui.recipe_shop_visual_demo`
- Check recipe schema: `src/dzetsaka/domain/value_objects/recipe_schema_v2.py`

---

**Version:** 5.0.0
**Last Updated:** 2026-02-09
**Author:** Nicolas Karasiak / dzetsaka team

