# Recipe Recommendation UI Guide

## User Flow

### Step 1: User Selects Raster

User browses for a raster file in either:
- Quick Classification Panel (dashboard)
- Data Input Page (guided wizard)

```
┌─────────────────────────────────────────┐
│  Raster to Classify                     │
│  ┌───────────────────────────────────┐  │
│  │ sentinel2_crop_2023.tif           │  │
│  └───────────────────────────────────┘  │
│                          [Browse...]     │
└─────────────────────────────────────────┘
```

### Step 2: Automatic Analysis

System analyzes the raster (silently, <1 second):
- Opens file with GDAL
- Counts bands
- Checks file size
- Parses filename for hints
- Detects sensor type
- Identifies land cover keywords

### Step 3: Recommendation Dialog Appears

If good matches found (score ≥40%), dialog opens:

```
╔═══════════════════════════════════════════════════════════════╗
║  Recipe Recommendations for Your Raster                       ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Bands: 12  •  Size: 1024 × 768 pixels  •  File: 450.3 MB   ║
║  Sensor: Sentinel-2  •  Type: Agriculture                    ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │  #1  Sentinel-2 Crop Classification        ⭐⭐⭐⭐⭐  │    ║
║  │                                                      │    ║
║  │  Excellent: 95% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░                │    ║
║  │                                                      │    ║
║  │  Optimized for Sentinel-2 agriculture mapping       │    ║
║  │                                                      │    ║
║  │  Why recommended:                                   │    ║
║  │  Perfect Sentinel-2 match (12 bands) •              │    ║
║  │  Optimized for agriculture classification •         │    ║
║  │  Efficient for medium-large files                   │    ║
║  │                                                      │    ║
║  │  Runtime: Medium (~10-30 min)  •  Accuracy: High   │    ║
║  │  Algorithm: XGB                                      │    ║
║  │                                                      │    ║
║  │            [Apply This Recipe]                      │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                                                               ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │  #2  Fast Random Forest                    ⭐⭐⭐⭐    │    ║
║  │                                                      │    ║
║  │  Good: 85% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░                     │    ║
║  │                                                      │    ║
║  │  Quick classification with Random Forest            │    ║
║  │                                                      │    ║
║  │  Why recommended:                                   │    ║
║  │  Fast algorithm suitable for large files •          │    ║
║  │  Optimized for speed                                │    ║
║  │                                                      │    ║
║  │  Runtime: Fast (~minutes)  •  Accuracy: Medium      │    ║
║  │  Algorithm: RF                                       │    ║
║  │                                                      │    ║
║  │            [Apply This Recipe]                      │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                                                               ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │  #3  High Accuracy XGBoost                 ⭐⭐⭐      │    ║
║  │                                                      │    ║
║  │  Fair: 72% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░                     │    ║
║  │                                                      │    ║
║  │  Best accuracy with XGBoost ensemble                │    ║
║  │                                                      │    ║
║  │  Why recommended:                                   │    ║
║  │  Optimized for accuracy • Includes model            │    ║
║  │  explainability (SHAP)                              │    ║
║  │                                                      │    ║
║  │  Runtime: Medium (~10-30 min)  •  Accuracy: High    │    ║
║  │  Algorithm: XGB                                      │    ║
║  │                                                      │    ║
║  │            [Apply This Recipe]                      │    ║
║  └─────────────────────────────────────────────────────┘    ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  [ ] Don't show recommendations again                        ║
║                                                               ║
║              [Show All Recipes]      [Close]                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## UI Components Explained

### Header Section

```
Bands: 12  •  Size: 1024 × 768 pixels  •  File: 450.3 MB
Sensor: Sentinel-2  •  Type: Agriculture
```

Shows analyzed raster characteristics:
- **Bands**: Number of spectral bands
- **Size**: Raster dimensions in pixels
- **File**: File size in MB
- **Sensor**: Detected sensor type (if any)
- **Type**: Detected land cover type (if any)

### Recommendation Cards

Each card contains:

1. **Rank Badge**: #1, #2, #3 (blue background)
2. **Recipe Name**: Bold, prominent
3. **Star Rating**: ⭐⭐⭐⭐⭐ (1-5 stars based on score)
4. **Confidence Bar**: Visual progress bar with percentage
   - Green: 80-100% (good matches)
   - Amber: 60-79% (fair matches)
   - Orange: 40-59% (low confidence)
5. **Recipe Description**: Italicized summary
6. **Why Recommended**: Bullet-pointed explanation
7. **Performance Info**: Runtime and accuracy expectations
8. **Algorithm**: ML algorithm used
9. **Apply Button**: One-click to load the recipe

### Footer Options

- **Don't show again checkbox**: Disables recommendations (saved to settings)
- **Show All Recipes button**: Opens full recipe list (future feature)
- **Close button**: Dismiss dialog without applying

## Color Scheme

### Confidence Colors

```python
score >= 80:  color = "#4CAF50"  # Green
score >= 60:  color = "#FFC107"  # Amber
score >= 40:  color = "#FF9800"  # Orange
```

### Card Styling

```css
Normal:
  background: white
  border: 1px solid #ddd

Hover:
  background: #f9fff9 (light green tint)
  border: 1px solid #4CAF50
```

### Rank Badge

```css
background: #2196F3 (blue)
color: white
border-radius: 12px
```

## Behavior

### Dialog Appearance

The dialog:
- ✅ Appears automatically after raster selection
- ✅ Is modal (blocks interaction until closed)
- ✅ Can be closed/dismissed without action
- ✅ Only shows if confidence ≥40% for top result
- ✅ Shows maximum 5 recommendations
- ✅ Fails silently if analysis errors occur

### Applying a Recipe

When user clicks "Apply This Recipe":
1. Recipe is loaded into UI
2. All parameters are populated
3. Dialog closes
4. User can proceed with classification

### Disabling Recommendations

When user checks "Don't show again":
1. Setting saved to QSettings: `/dzetsaka/show_recommendations` = False
2. Future raster selections won't trigger dialog
3. Can be re-enabled in plugin settings (future feature)

## Responsive Design

### Minimum Size
- Width: 600px
- Height: 500px

### Scrolling
- Header: Fixed
- Recommendations: Scrollable area
- Footer: Fixed

### Card Layout
- Cards stack vertically
- Consistent spacing (8px between cards)
- Padding: 10px inside each card

## Error States

### No Recommendations

If no good matches (all scores <40%):
- Dialog doesn't appear
- User proceeds normally
- No interruption to workflow

### GDAL Error

If raster can't be analyzed:
- Error silently caught
- Dialog doesn't appear
- User proceeds normally

### Missing GDAL

If GDAL not available:
- Import fails
- Feature disabled
- No UI shown
- No error messages

## Accessibility

### Keyboard Navigation
- Tab through cards and buttons
- Enter to apply selected recipe
- Escape to close dialog

### Screen Readers
- All labels have tooltips
- Progress bars have aria-labels
- Buttons clearly labeled

### Color Blindness
- Don't rely on color alone
- Percentages shown numerically
- Text explanations provided

## Example Scenarios

### Scenario 1: Perfect Match

```
Input: sentinel2_ndvi_2023.tif (12 bands, 200 MB)

Dialog shows:
#1: Sentinel-2 NDVI Recipe      ⭐⭐⭐⭐⭐ (98%)
#2: Vegetation Mapping          ⭐⭐⭐⭐  (87%)
#3: Fast Random Forest          ⭐⭐⭐⭐  (82%)
```

### Scenario 2: Good Match

```
Input: landsat8_forest.tif (11 bands, 800 MB)

Dialog shows:
#1: Landsat Forest Classifier   ⭐⭐⭐⭐  (89%)
#2: Fast Random Forest          ⭐⭐⭐⭐  (83%)
#3: High Accuracy SVM           ⭐⭐⭐   (71%)
```

### Scenario 3: Fair Match

```
Input: aerial_photo.tif (4 bands, 50 MB)

Dialog shows:
#1: Fast Random Forest          ⭐⭐⭐   (65%)
#2: Multispectral Classifier    ⭐⭐⭐   (62%)
#3: Quick RGB Mapping           ⭐⭐    (58%)
```

### Scenario 4: No Good Matches

```
Input: unknown_data.tif (1 band, 10 MB)

Result: Dialog doesn't appear (all scores <40%)
User proceeds with manual selection
```

## User Feedback

### Positive Indicators
- ⭐⭐⭐⭐⭐ Excellent match
- Green progress bar
- "Perfect match" text
- High percentage (95%+)

### Neutral Indicators
- ⭐⭐⭐ Fair match
- Amber progress bar
- "Possible match" text
- Medium percentage (60-79%)

### Low Confidence
- ⭐⭐ or ⭐ Low confidence
- Orange progress bar
- "Low confidence" text
- Low percentage (40-59%)

## Implementation Notes

### Performance
- Analysis: <1 second for most rasters
- Dialog render: Instant
- No blocking operations
- Async GDAL calls (non-blocking)

### Memory
- Minimal overhead
- Raster not loaded into memory
- Only metadata analyzed
- Dialog cleans up on close

### Error Handling
- All exceptions caught
- Silent failures (no popups)
- Graceful degradation
- No workflow interruption

## Future UI Enhancements

### Version 2.0 Ideas
1. **Collapsible cards**: Show more recipes without scrolling
2. **Preview thumbnails**: Show visual representation of recipes
3. **Performance graphs**: Historical runtime/accuracy charts
4. **User ratings**: Community-driven confidence scores
5. **Quick filters**: Filter by speed, accuracy, complexity
6. **Comparison view**: Side-by-side recipe comparison
7. **Smart search**: Filter recommendations by keyword
8. **Recipe details**: Expandable sections with full parameters
9. **Tutorial mode**: Explain each recommendation in detail
10. **Dark mode**: Match QGIS theme

## Customization

### For Developers

Customize appearance in `ui/recommendation_dialog.py`:

```python
# Card styling
card.setStyleSheet("""
    QWidget {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
""")

# Confidence bar colors
if score >= 80:
    color = "#4CAF50"  # Change to your color
```

### For Users

Settings available in plugin configuration (future):
- Show/hide recommendations
- Maximum recommendations to display
- Minimum confidence threshold
- Preferred algorithms to boost

## Summary

The recommendation UI provides:
- ✅ Clear, visual confidence indicators
- ✅ Detailed explanations for each recommendation
- ✅ One-click recipe application
- ✅ Non-intrusive, dismissible design
- ✅ Graceful error handling
- ✅ Accessible, keyboard-navigable
- ✅ Professional appearance
- ✅ Fast, responsive performance

The goal is to help users quickly identify the best recipe for their data without overwhelming them with technical details or interrupting their workflow.
