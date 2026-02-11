# dzetsaka v5.0 - New Features User Guide

Welcome to the dzetsaka v5.0 user guide! This release introduces powerful new features to enhance your classification workflows, improve data quality, and boost productivity.

## 🎯 Table of Contents

1. [Training Data Quality Checker](#training-data-quality-checker)
2. [Batch Classification](#batch-classification)
3. [Confidence Map Analysis](#confidence-map-analysis)
4. [Keyboard Shortcuts](#keyboard-shortcuts)
5. [Dark Mode Support](#dark-mode-support)
6. [Quick Start Examples](#quick-start-examples)

---

## Training Data Quality Checker

### What is it?

The Training Data Quality Checker analyzes your training vector data **before** classification to identify potential issues that could affect accuracy. Think of it as a "pre-flight check" for your data.

### Why use it?

Running a classification can take 5-30 minutes depending on your settings. The Quality Checker takes only seconds and can save you from wasting time on classifications that are doomed to fail or produce poor results.

### How to access

**Option 1: Before Classification (Proactive)**
1. Open the dzetsaka dashboard or wizard
2. Select your training vector and class field
3. Click the **"Check Data Quality"** button
4. Or press **`Ctrl+Shift+Q`** (keyboard shortcut)

**Option 2: After Classification (Iterative Improvement)**
1. After classification completes, open the **Results Explorer**
2. In the **Quick Actions** section, click **"Check Training Data"**
3. Review issues and improve your training data
4. Re-run classification with improved data

### What it checks

| Check | Severity | What it means |
|-------|----------|---------------|
| **Class Imbalance** | ⚠️ Warning | One class has >10x more samples than another. Can bias the model. |
| **Insufficient Samples** | 🔴 Error | A class has <30 training samples. Too few for reliable training. |
| **Invalid Geometries** | 🔴 Error | Corrupted or malformed polygon/point geometries. |
| **Spatial Clustering** | ℹ️ Info | Training samples are geographically clustered. Consider polygon-based cross-validation. |
| **Duplicate Features** | ⚠️ Warning | Multiple samples with identical attributes. May indicate data collection errors. |

### Understanding the report

After running the check, you'll see a color-coded report:

```
🔴 ERROR: Insufficient samples in class "Water"
   Found only 15 samples. Recommended minimum: 30 samples per class.

   Recommendation: Collect more training samples for this class or merge with a similar class.

⚠️ WARNING: Severe class imbalance detected
   Class "Forest" has 450 samples, but class "Urban" has only 30 samples (15:1 ratio).

   Recommendation: Balance your classes by collecting more "Urban" samples or
   enabling SMOTE (Synthetic Minority Oversampling) in Advanced Options.

ℹ️ INFO: Spatial clustering detected
   Training samples are concentrated in the northwest region.

   Recommendation: Enable "Polygon-based cross-validation" to account for spatial autocorrelation.
```

### Actionable recommendations

Each issue includes specific recommendations:

| Issue | Recommendation |
|-------|----------------|
| **Insufficient samples** | Collect more training data for underrepresented classes |
| **Class imbalance** | Enable SMOTE in Advanced Options, or balance manually |
| **Invalid geometries** | Fix geometries in QGIS (Vector > Geometry Tools > Fix Geometries) |
| **Spatial clustering** | Use Polygon-based Cross-Validation, or collect spatially diverse samples |
| **Duplicates** | Review and remove duplicate features |

### Exporting the report

Click **"Export Report..."** to save the quality check results as a text file for:
- Documentation
- Sharing with colleagues
- Before/after comparisons

---

## Batch Classification

### What is it?

Batch Classification allows you to classify **multiple rasters** using the **same trained model**. Perfect for:
- 📅 **Time-series analysis** (classify multiple dates)
- 🌍 **Large-scale projects** (classify multiple tiles)
- ⚙️ **Operational workflows** (daily/weekly classification runs)

### How to access

**Menu:** `Plugins > dzetsaka > Batch Classification...`

### Workflow

#### 1. **Queue Your Rasters**

Click **"Add Files..."** to select multiple raster files, or **"Add Folder..."** to add all rasters from a directory.

Your queue will look like:
```
✓ sentinel2_2023_01_15.tif
✓ sentinel2_2023_02_15.tif
✓ sentinel2_2023_03_15.tif
✓ sentinel2_2023_04_15.tif
```

**Tip:** Files are processed in the order listed. You can remove individual files or clear the entire queue.

#### 2. **Select Your Model**

Click **"Browse Model..."** and select a previously saved `.pkl` model file.

**Important:** The model must match the number of bands in your rasters!

#### 3. **Choose Output Directory**

Click **"Browse Output..."** to select where classified rasters will be saved.

**Output Naming:** Results are automatically named based on input files:
- Input: `sentinel2_2023_01_15.tif`
- Output: `sentinel2_2023_01_15_classified.tif`

#### 4. **Configure Options**

- ✅ **Auto-add to Map:** Automatically add results to QGIS
- ✅ **Generate Summary:** Create CSV summary report

#### 5. **Run Batch**

Click **"Start Batch"** and watch the progress:

```
Processing raster 2 of 4...
sentinel2_2023_02_15.tif

Overall Progress: [████████████░░░░░░] 50%
```

**Controls:**
- ⏸️ **Pause:** Pause processing (current raster will complete)
- ▶️ **Resume:** Continue processing
- ⏹️ **Stop:** Stop batch (results so far are kept)

#### 6. **Review Summary**

After completion, a summary CSV is generated:

| Raster | Status | Runtime | Output Path |
|--------|--------|---------|-------------|
| sentinel2_2023_01_15.tif | ✅ Success | 12.3s | output/sentinel2_2023_01_15_classified.tif |
| sentinel2_2023_02_15.tif | ✅ Success | 11.8s | output/sentinel2_2023_02_15_classified.tif |
| sentinel2_2023_03_15.tif | ❌ Failed | - | Error: Band count mismatch |
| sentinel2_2023_04_15.tif | ⏸️ Paused | - | - |

### Best Practices

1. **Test First:** Run a single raster before batching 100 files
2. **Check Band Count:** Ensure all rasters have the same number of bands
3. **Save Model:** Train once, reuse the model for the entire batch
4. **Monitor Progress:** The dialog is non-modal - you can use QGIS while it runs

---

## Confidence Map Analysis

### What is it?

Confidence maps show **how certain** the model is about each pixel's classification. Higher confidence = more reliable prediction.

### How to enable

**During Classification:**
1. In Advanced Options, check **"Generate confidence map"**
2. Run classification

**After Classification:**
The Results Explorer automatically opens with a **Confidence** tab if a confidence map was created.

### Understanding the Analysis

#### Histogram

The histogram shows the distribution of confidence values across all pixels:

```
     Pixel Count
         │
    6000 │     ▂▄█▆▃
    4000 │   ▂█████▆▃
    2000 │ ▁▄█████████▅▂
       0 │─────────────────────
         0%  20%  40%  60%  80%  100%
                 Confidence

         🔴 Low  🟡 Medium  🟢 High
```

#### Statistics Summary

```
Mean confidence:     76.3%
Median confidence:   82.1%
Std. deviation:      0.18

Range:              12.5% - 99.8%
25th percentile:    65.2%
75th percentile:    89.7%

Confidence Distribution:
🔴 Low (<50%):      8.2% (12,450 pixels)
🟡 Medium (50-80%): 31.5% (47,820 pixels)
🟢 High (>80%):     60.3% (91,530 pixels)
```

### Interpreting Confidence Values

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| **>80% (High)** | Strong model agreement. Likely correct. | ✅ Trust these predictions |
| **50-80% (Medium)** | Moderate agreement. Generally reliable. | ⚠️ Validate critical areas |
| **<50% (Low)** | Weak agreement. High risk of error. | 🔴 Review carefully, consider retraining |

### What causes low confidence?

1. **Insufficient training data** in that region
2. **Class confusion** (similar spectral signatures)
3. **Mixed pixels** at class boundaries
4. **Unique conditions** not present in training data

### Improving confidence

- ✅ Add more training samples in low-confidence areas
- ✅ Use higher-resolution imagery
- ✅ Add additional spectral bands (e.g., NDVI, NDWI)
- ✅ Review and refine class definitions
- ✅ Apply post-processing filters

### Exporting Statistics

Click **"Export Statistics..."** to save confidence metrics as CSV for:
- Accuracy reporting
- Before/after comparisons
- Quality assurance documentation

---

## Keyboard Shortcuts

Power users can speed up their workflow with keyboard shortcuts:

### Quick Classification Panel (Dashboard)

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+Q` | Check Data Quality |
| `Ctrl+Return` | Run Classification |

### Guided Wizard

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+Q` | Check Data Quality (any page) |
| `Ctrl+Right` | Next page |
| `Ctrl+Left` | Previous page |

### Tips

- Tooltips show keyboard shortcuts when you hover over buttons
- Shortcuts work even when button is disabled (will show error message)
- Use shortcuts to navigate wizard without lifting hands from keyboard

---

## Dark Mode Support

### Automatic Theme Detection

dzetsaka v5.0 automatically adapts to your QGIS theme:

**Light Mode:**
- White backgrounds
- Dark text
- Blue accents

**Dark Mode:**
- Dark gray backgrounds
- Light text
- Bright blue accents

### What's themed

- ✅ All dialogs (wizard, dashboard, batch, quality checker)
- ✅ Results explorer
- ✅ Welcome wizard
- ✅ Matplotlib charts (confidence histograms, accuracy plots)

### Switching themes

1. Go to `Settings > Options` in QGIS
2. Select `General > UI Theme`
3. Choose your preferred theme
4. Restart QGIS
5. dzetsaka dialogs will automatically match!

**Supported themes:**
- Default (light)
- Night Mapping (dark)
- Blend of Gray (dark)
- Any custom theme

---

## Quick Start Examples

### Example 1: First-Time Classification with Quality Check

```
Step 1: Open dzetsaka dashboard
Step 2: Load training vector (e.g., training_samples.shp)
Step 3: Select class field (e.g., "class_id")
Step 4: Press Ctrl+Shift+Q to check data quality
Step 5: Review and fix any errors
Step 6: Select raster and recipe
Step 7: Press Ctrl+Return to run classification
Step 8: Review results in Results Explorer
```

### Example 2: Batch Time-Series Classification

```
Step 1: Train model on one date
Step 2: Save model (.pkl file)
Step 3: Open Batch Classification (Plugins > dzetsaka > Batch Classification)
Step 4: Add all time-series rasters
Step 5: Select saved model
Step 6: Choose output directory
Step 7: Enable "Auto-add to Map" and "Generate Summary"
Step 8: Click "Start Batch"
Step 9: Review summary CSV when complete
```

### Example 3: Iterative Quality Improvement

```
Step 1: Run initial classification
Step 2: In Results Explorer, review confidence map
Step 3: Identify low-confidence areas (red zones in histogram)
Step 4: Click "Check Training Data" in Quick Actions
Step 5: Address identified issues (add samples, fix imbalance)
Step 6: Re-run classification
Step 7: Compare confidence histograms (should show improvement!)
Step 8: Repeat until satisfied with accuracy and confidence
```

---

## Frequently Asked Questions

### Q: How many training samples do I need per class?

**A:** Minimum 30, but more is better. Aim for 50-100 samples per class for reliable results. Use the Quality Checker to verify!

### Q: What if my model takes too long to train?

**A:**
- Reduce Optuna trials (default 300 → try 50-100)
- Use a faster algorithm (RF instead of XGBoost)
- Reduce training sample size (randomly sample if you have thousands)
- Check that SMOTE is disabled if you don't need it

### Q: Can I use batch classification with different models?

**A:** No, batch classification uses one model for all rasters. This ensures consistency across your time-series or tiles.

### Q: How do I interpret a confidence map visually?

**A:** Load the confidence map in QGIS and apply a diverging color ramp:
1. Right-click confidence layer > Properties
2. Go to Symbology
3. Choose "Singleband pseudocolor"
4. Select color ramp "RdYlGn" (red-yellow-green)
5. Red areas = low confidence, yellow = medium, green = high

### Q: Do keyboard shortcuts work in all contexts?

**A:** Keyboard shortcuts work when the corresponding dialog is active. For example, Ctrl+Shift+Q only works when the dashboard or wizard is open.

---

## Troubleshooting

### Quality Checker says "module import failed"

**Solution:** Install the training_data_quality_checker dependencies:
```python
pip install numpy scipy
```

### Batch Classification fails with "Band count mismatch"

**Solution:** Ensure all rasters have the same number of bands. Check with:
```python
from osgeo import gdal
ds = gdal.Open("your_raster.tif")
print(f"Bands: {ds.RasterCount}")
```

### Confidence map shows all 100% confidence

**Solution:** This usually means:
- You used a very simple classification task
- Model is overfitted
- Check that confidence map generation is actually enabled

### Theme looks wrong after update

**Solution:** Restart QGIS. Qt caches styles, and a restart ensures the new theme system loads correctly.

---

## Getting Help

- **GitHub Issues:** https://github.com/nkarasiak/dzetsaka/issues
- **Documentation:** https://dzetsaka.readthedocs.io
- **Email:** nicolas.karasiak@irstea.fr

---

## Changelog

### v5.0 - New Features

- ✨ Training Data Quality Checker
- ✨ Batch Classification
- ✨ Confidence Map Analysis
- ✨ Keyboard Shortcuts
- ✨ Automatic Dark Mode Support
- ✨ Results Explorer enhancements
- ✨ Validated input fields with real-time feedback
- 🐛 100+ bug fixes and improvements

---

**Happy Classifying! 🚀**
