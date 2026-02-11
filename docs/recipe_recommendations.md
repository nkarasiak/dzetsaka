# Recipe Recommendation System

## Overview

The recipe recommendation system analyzes raster characteristics and suggests the most suitable classification recipes based on detected patterns. This helps users quickly select optimal configurations for their data without deep ML expertise.

## How It Works

### 1. Raster Analysis

When a user selects a raster file, the system analyzes:

- **Band Count**: Detects sensor type (RGB, multispectral, hyperspectral)
- **File Size**: Recommends faster algorithms for large files
- **Filename Patterns**: Detects sensors (Sentinel-2, Landsat, etc.)
- **Metadata**: Resolution, CRS, dimensions

### 2. Pattern Matching

The recommender scores each available recipe based on:

#### Sensor Detection
- **Sentinel-2**: 10-13 bands or "sentinel"/"s2" in filename
- **Landsat 8/9**: 7-11 bands or "landsat"/"LC08" in filename
- **Hyperspectral**: >20 bands
- **RGB/Multispectral**: 3-4 bands

#### Performance Optimization
- **Large files (>1 GB)**: Boost fast algorithms (RF, XGB, LGB), penalize slow (SVM, MLP)
- **Medium files (500 MB - 1 GB)**: Slight preference for efficient algorithms

#### Land Cover Specificity
- **Agriculture**: Recipes mentioning "crop", "agri", "farm"
- **Forest**: Recipes mentioning "forest", "tree", "vegetation"
- **Urban**: Recipes mentioning "urban", "city", "built"
- **Water**: Recipes mentioning "water", "aquatic"

#### Special Features
- **Imbalanced data**: Recipes with SMOTE enabled
- **Explainability**: Recipes with SHAP enabled
- **Speed vs. Accuracy**: Recipes optimized for speed or accuracy

### 3. Confidence Scoring

Scores range from 0-100%:

- **95-100%**: Excellent match (e.g., 12 bands + "sentinel" + "crop")
- **80-94%**: Good match (e.g., 12 bands but no filename hints)
- **60-79%**: Possible match (e.g., filename hints but unusual band count)
- **40-59%**: Low confidence
- **<40%**: Not recommended (not shown)

### 4. User Interface

The recommendation dialog displays:

- **Top 5 recommendations** sorted by confidence
- **Star ratings** (⭐⭐⭐⭐⭐) based on score
- **Confidence bar** with visual progress indicator
- **Explanation** of why each recipe was recommended
- **Performance expectations** (runtime, accuracy)
- **One-click apply** to load the recipe

## Integration Points

### Quick Classification Panel (Dashboard)

When user browses for a raster file:
```python
def _browse_raster(self):
    path, _f = QFileDialog.getOpenFileName(...)
    if path:
        self.rasterLineEdit.setText(path)
        self._show_recipe_recommendations(path)
```

### Guided Workflow (Wizard)

When user selects a raster in DataInputPage:
```python
def _browse_raster(self):
    path, _f = QFileDialog.getOpenFileName(...)
    if path:
        self.rasterLineEdit.setText(path)
        self._show_recipe_recommendations_for_setup_dialog(path)
```

## User Control

Users can:

- **View recommendations**: Automatically shown when selecting raster
- **Disable recommendations**: Checkbox in dialog ("Don't show again")
- **Re-enable**: Via plugin settings (stored in QSettings)
- **Ignore recommendations**: Just close the dialog and continue

## Technical Details

### Core Components

#### `RasterAnalyzer` (ui/recipe_recommender.py)
- Analyzes raster files using GDAL
- Detects sensor type, band count, file size
- Extracts hints from filename and metadata

#### `RecipeRecommender` (ui/recipe_recommender.py)
- Scores recipes based on raster characteristics
- Applies heuristics for sensor, file size, land cover
- Generates confidence scores and explanations

#### `RecommendationDialog` (ui/recommendation_dialog.py)
- Displays top recommendations with UI
- Shows confidence bars, star ratings, explanations
- Allows one-click recipe application

### Error Handling

The system is designed to fail silently:
- If GDAL can't read the file → no recommendations shown
- If no good matches found → dialog not displayed
- If imports fail → feature gracefully disabled

This ensures recommendations never interrupt the user workflow.

## Configuration

### Settings (QSettings)

```python
# Enable/disable recommendations
QSettings().setValue("/dzetsaka/show_recommendations", True)

# Check if enabled
enabled = QSettings().value("/dzetsaka/show_recommendations", True, bool)
```

### Customizing Scoring Logic

Edit `RecipeRecommender._score_recipe()` in `ui/recipe_recommender.py` to:
- Add new sensor detection patterns
- Adjust confidence thresholds
- Modify scoring weights

## Examples

### Example 1: Sentinel-2 Agriculture

**Input:**
- Raster: `sentinel2_crop_classification_2023.tif`
- Bands: 12
- Size: 450 MB

**Analysis:**
- Detected sensor: sentinel2 ✓
- Land cover: agriculture ✓
- File size: medium ✓

**Top Recommendation:**
- Recipe: "Sentinel-2 Crop Classification"
- Score: 95%
- Reason: "Perfect Sentinel-2 match (12 bands) • Optimized for agriculture classification • Efficient for medium-large files"

### Example 2: Large Hyperspectral

**Input:**
- Raster: `hyperspectral_forest_mapping.tif`
- Bands: 64
- Size: 3.2 GB

**Analysis:**
- Detected sensor: unknown
- Band count: hyperspectral (>20) ✓
- File size: large ✓

**Top Recommendation:**
- Recipe: "Fast Random Forest"
- Score: 78%
- Reason: "Hyperspectral imagery (64 bands) • Fast algorithm suitable for large files (3200 MB) • Optimized for speed"

### Example 3: RGB Image

**Input:**
- Raster: `rgb_orthophoto.tif`
- Bands: 3
- Size: 120 MB

**Analysis:**
- Detected sensor: unknown
- Band count: RGB ✓
- File size: small ✓

**Top Recommendation:**
- Recipe: "Fast Random Forest"
- Score: 55%
- Reason: "RGB or 4-band multispectral imagery • Optimized for speed"

## Testing

Run tests with:

```bash
# Core recommendation logic (no QGIS required)
pytest tests/unit/test_recommender_core.py -v

# Full integration tests (requires QGIS)
pytest tests/unit/test_recipe_recommender.py -v
```

## Future Enhancements

Potential improvements:

1. **Machine Learning**: Train a model on successful classifications
2. **User Feedback**: Learn from which recommendations users accept
3. **Remote Repository**: Fetch community-curated recipes
4. **Performance Tracking**: Suggest recipes based on past runtime
5. **Dataset Fingerprinting**: Match against known dataset types
6. **Cloud Service**: API for professional recommendations

## Troubleshooting

### Recommendations not showing

1. Check if enabled: `QSettings().value("/dzetsaka/show_recommendations", True, bool)`
2. Verify GDAL available: `from osgeo import gdal`
3. Check file readable: Ensure raster path is valid
4. Debug mode: Check logs for silent failures

### Wrong recommendations

1. Verify raster metadata is correct
2. Check filename patterns in `RasterAnalyzer.sensor_patterns`
3. Adjust scoring weights in `RecipeRecommender._score_recipe()`
4. Add custom detection rules for your data types

### Performance issues

1. Reduce `max_recommendations` in dialog (currently 5)
2. Add file size threshold to skip analysis for huge files
3. Cache raster analysis results temporarily
4. Async analysis for very large files

## License

Part of the dzetsaka QGIS plugin.
Copyright (c) 2024 Nicolas Karasiak

