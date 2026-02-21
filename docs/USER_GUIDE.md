# dzetsaka User Guide

Welcome to dzetsaka! This guide will help you get started with remote sensing image classification in QGIS.

## Table of Contents

- [Quick Start](#quick-start)
- [Interface Overview](#interface-overview)
- [Basic Workflow](#basic-workflow)
- [Using the Wizard](#using-the-wizard)
- [Algorithm Selection](#algorithm-selection)
- [Advanced Features](#advanced-features)
- [Tips & Best Practices](#tips--best-practices)

## Quick Start

### 5-Minute Classification

1. **Load your data** into QGIS:
   - Raster layer (satellite/aerial image)
   - Vector layer with training polygons (each polygon labeled with a class number)

2. **Open dzetsaka**: `Plugins → dzetsaka → Classification tool`

3. **Configure**:
   - Input raster: Select your image
   - Training layer: Select your vector file
   - Field: Select column with class numbers (e.g., 1=forest, 2=water, 3=urban)
   - Algorithm: Choose "Random Forest" (good default)
   - Output raster: Choose save location

4. **Click "Train & Classify"** - Done!

## Interface Overview

### Classic Dock Widget

The main interface with all options visible:

- **Input Data Section**: Select raster, training vector, and class field
- **Algorithm Section**: Choose from 11 ML algorithms
- **Advanced Options** (collapsible): Hyperparameters, validation split, optimization
- **Output Section**: Save locations for model, confusion matrix, confidence map
- **Action Buttons**: Train, Classify, Train & Classify

### Wizard Interface

Step-by-step guided workflow:

- **Page 1 - Data**: Select inputs
- **Page 2 - Algorithm**: Choose classifier
- **Page 3 - Advanced**: Optional features (Optuna, SMOTE, SHAP)
- **Page 4 - Output**: Save locations
- **Page 5 - Review**: Confirm and execute

Access: Click the **Wizard** button in the dock widget

## Basic Workflow

### 1. Prepare Training Data

**Vector layer requirements:**
- Polygon or point features
- Integer field with class labels (1, 2, 3, etc.)
- At least 5-10 samples per class (more is better)
- Classes should be mutually exclusive

**Example attribute table:**
```
| ID | Class | Name   |
|----|-------|--------|
| 1  | 1     | Forest |
| 2  | 1     | Forest |
| 3  | 2     | Water  |
| 4  | 3     | Urban  |
```

### 2. Train a Model

**Option A: Train Only**
1. Configure inputs
2. Set validation split (e.g., 50% for training, 50% for validation)
3. Click **Train**
4. Review confusion matrix
5. Save model (.npz file)

**Option B: Train & Classify**
1. Configure inputs and outputs
2. Click **Train & Classify**
3. Classified raster is automatically created

### 3. Classify New Images

Use a saved model on different images:

1. Click **Classify** tab
2. Select saved model (.npz file)
3. Select new raster to classify
4. Click **Classify**

**Requirements:**
- New image must have same number of bands as training image
- Same band order and wavelengths recommended

## Using the Wizard

### Recipe System

Save your configurations as **recipes** for reuse:

1. **Save Recipe**:
   - Configure wizard
   - Click "Save Recipe" button
   - Enter name and description
   - Recipe saved for future use

2. **Load Recipe**:
   - Click "Load Recipe" button
   - Select from gallery
   - Configuration applied instantly

3. **Share Recipes**:
   - Export as JSON file
   - Share with colleagues
   - Import on other machines

### Built-in Recipes

- **Fast (Default)**: Random Forest with sensible defaults
- **CatBoost Quick**: High accuracy with minimal tuning
- **Precision Mode**: Extra Trees with cross-validation
- *(Download more from recipe gallery)*

## Algorithm Selection

### Decision Guide

**Choose GMM if:**
- You have limited samples
- No dependencies installed
- Need very fast classification
- Baseline comparison needed

**Choose Random Forest if:**
- General-purpose classification (best default)
- You have 100+ samples per class
- Balanced accuracy and speed needed

**Choose SVM if:**
- You have limited samples (50-100 per class)
- High accuracy required
- Processing time is not critical

**Choose XGBoost/CatBoost if:**
- Maximum accuracy required
- Large training dataset available (1000+ samples)
- Have time for hyperparameter tuning

**Choose KNN if:**
- Simple, interpretable model needed
- Irregular class boundaries

### Algorithm Comparison

| Algorithm | Speed | Accuracy | Memory | Samples Needed |
|-----------|-------|----------|--------|----------------|
| GMM       | ⚡⚡⚡  | ⭐⭐     | 💾     | 10+            |
| RF        | ⚡⚡   | ⭐⭐⭐⭐  | 💾💾   | 100+           |
| SVM       | ⚡     | ⭐⭐⭐⭐  | 💾💾   | 50+            |
| XGB       | ⚡     | ⭐⭐⭐⭐⭐ | 💾💾💾 | 500+           |
| CB        | ⚡⚡   | ⭐⭐⭐⭐⭐ | 💾💾   | 500+           |

## Advanced Features

### Hyperparameter Optimization

**Automatic Grid Search** (default for most algorithms):
- 3-5 fold cross-validation
- Algorithm-specific parameter grids
- No configuration needed

**Optuna Optimization** (advanced):
1. Install optuna: `pip install optuna`
2. Enable in Advanced Options
3. Set number of trials (default: 100)
4. Algorithm explores parameter space intelligently

### Confidence Mapping

Generate confidence scores for each pixel:

1. Enable "Confidence Map" in output section
2. Output shows probability/confidence per pixel
3. Use for uncertainty analysis
4. Filter low-confidence areas

**Interpretation:**
- High values (80-100): Model is confident
- Low values (0-30): Model is uncertain
- Useful for identifying mixed pixels

### SMOTE Sampling

Balance imbalanced datasets:

1. Install imbalanced-learn: `pip install imbalanced-learn`
2. Enable in Advanced Options
3. Automatically generates synthetic samples for minority classes

**When to use:**
- One class has 10x more samples than others
- Minority class is being ignored by classifier

### SHAP Explainability

Understand model predictions:

1. Install SHAP: `pip install shap`
2. Use Processing algorithm: "Explain Model (SHAP)"
3. Generates feature importance maps
4. Shows which bands contribute most to each class

### Spatial Cross-Validation

Avoid spatial autocorrelation bias:

**SLOO (Spatial Leave-One-Out):**
- Leave entire spatial blocks out
- Better accuracy estimation
- Use Processing algorithm: "Learn with Spatial Sampling"

**STAND (Stratified Spatial):**
- Spatial stratification + CV
- Use Processing algorithm: "Learn with STAND CV"

## Tips & Best Practices

### Training Data Quality

✅ **Do:**
- Collect diverse samples across the study area
- Include edge cases and variations
- Label consistently
- Balance class representation

❌ **Don't:**
- Cluster all samples in one area
- Mix spectral dates in training
- Use fewer than 5 samples per class
- Mislabel ambiguous pixels

### Model Selection

- Start with **Random Forest** - best general-purpose algorithm
- Try **CatBoost** if RF is insufficient
- Use **GMM** for quick tests
- Save time with **recipes** for repeated tasks

### Performance Optimization

**For large rasters:**
- Classification processes in memory-efficient blocks (512MB limit)
- Use mask to exclude areas (e.g., water, clouds)
- Consider tiling very large images

**For faster training:**
- Limit training samples (10,000 max often sufficient)
- Use Extra Trees instead of XGBoost for faster training
- Disable cross-validation for quick tests

### Validation

Always validate your results:
1. Set aside 30-50% for validation
2. Review confusion matrix
3. Check overall accuracy and kappa
4. Examine per-class accuracies
5. Visually inspect classification errors

### Common Issues

**"Import Error: sklearn"**
- Solution: Install scikit-learn via Settings → Dependencies

**"Different projections" error**
- Solution: Reproject vector to match raster CRS

**Low accuracy (<60%)**
- Check: Training data quality
- Check: Sufficient samples per class
- Try: Different algorithm or hyperparameter optimization

**Classification shows only one class**
- Check: Class labels are integers (not strings)
- Check: All classes have samples
- Try: SMOTE if classes are imbalanced

## Next Steps

- Explore the **Processing Toolbox** (Ctrl+Alt+T) for batch workflows
- Build **QGIS Models** combining dzetsaka with other tools
- Check **GitHub issues** for latest updates and community tips
- Read algorithm-specific docs in `/docs/algorithms/`

---

**Need help?** Open an issue at: https://github.com/nkarasiak/dzetsaka/issues
