# UI Update Guide - New Classifiers in QGIS Interface

This guide explains how to ensure the QGIS plugin interface shows all 11 available classifiers after updating to dzetsaka v4.1.0.

## 🔧 What Was Updated

The following UI files have been updated to include 7 new classifiers:

### Updated Files:
- ✅ `ui/dzetsaka_dock.ui` - Main classification panel
- ✅ `ui/settings_dock.ui` - Settings panel  
- ✅ `dzetsaka.py` - Main plugin logic
- ✅ `scripts/sklearn_validator.py` - Validation system

### New Classifiers Added to UI:
1. **XGBoost** - High-performance gradient boosting
2. **LightGBM** - Fast gradient boosting  
3. **Extra Trees** - Extremely randomized trees
4. **Gradient Boosting Classifier** - Sklearn gradient boosting
5. **Logistic Regression** - Linear classification
6. **Gaussian Naive Bayes** - Probabilistic classifier
7. **Multi-layer Perceptron** - Neural network

## 🖥️ Accessing New Classifiers in QGIS

### Method 1: Settings Panel
1. Open QGIS with dzetsaka plugin loaded
2. Go to **Plugins → dzetsaka → Settings** (or find the dzetsaka settings dock)
3. In the **Classifier** dropdown, you should now see all 11 options:
   - Gaussian Mixture Model
   - Random Forest
   - Support Vector Machine  
   - K-Nearest Neighbors
   - **XGBoost** ⭐ NEW
   - **LightGBM** ⭐ NEW
   - **Extra Trees** ⭐ NEW
   - **Gradient Boosting Classifier** ⭐ NEW
   - **Logistic Regression** ⭐ NEW
   - **Gaussian Naive Bayes** ⭐ NEW
   - **Multi-layer Perceptron** ⭐ NEW

### Method 2: Main Classification Panel
1. Open the main dzetsaka classification dock
2. Look for the **Classifier** dropdown in the classification parameters
3. Select any of the 11 available classifiers

## 🔍 Troubleshooting

### Problem: Only 4 Classifiers Showing
If you only see the original 4 classifiers, the Python UI files need to be regenerated:

**Solution A - Automatic (Recommended):**
```bash
cd /path/to/dzetsaka
python regenerate_ui.py
```

**Solution B - Manual:**
```bash
# Install PyQt5 tools if not available
pip install PyQt5-tools

# Regenerate Python UI files
pyuic5 ui/dzetsaka_dock.ui -o ui/dzetsaka_dock.py
pyuic5 ui/settings_dock.ui -o ui/settings_dock.py
```

**Solution C - Alternative Method:**
```bash
# Using Python module directly
python -m PyQt5.uic.pyuic ui/dzetsaka_dock.ui -o ui/dzetsaka_dock.py
python -m PyQt5.uic.pyuic ui/settings_dock.ui -o ui/settings_dock.py
```

### Problem: Classifier Shows as Unavailable
When you select a new classifier and get an error message:

1. **For XGBoost/LightGBM**: Install the required packages
   ```bash
   pip install xgboost lightgbm
   ```

2. **For sklearn classifiers**: Ensure scikit-learn is installed
   ```bash
   pip install scikit-learn
   ```

3. **Check validation**: The plugin will automatically detect what's available and show helpful error messages with installation instructions.

## ✅ Verification Steps

### 1. Check UI Files Updated
Run the verification script:
```bash
python regenerate_ui.py
```

### 2. Test in QGIS
1. Load dzetsaka plugin in QGIS
2. Open settings panel
3. Count classifiers in dropdown (should be 11)
4. Try selecting different classifiers
5. Check that validation messages appear for missing dependencies

### 3. Test Classification
1. Select a new classifier (e.g., XGBoost)
2. Run a classification with training data
3. Verify the new classifier is actually used in training

## 🎯 Expected Behavior

### ✅ When Working Correctly:
- Dropdown shows all 11 classifiers
- Selecting unavailable classifiers shows helpful error messages
- Available classifiers work normally
- Plugin automatically detects and validates dependencies

### ⚠️ Common Issues:
- **UI not updated**: Python files need regeneration
- **Import errors**: Missing packages (install with pip)
- **Settings not saved**: QGIS cache issue (restart QGIS)

## 📧 Support

If you continue having issues:

1. **Check the logs**: QGIS → View → Panels → Log Messages
2. **Verify files**: Ensure all UI files are updated correctly  
3. **Test dependencies**: Run `python test_new_classifiers.py`
4. **Report issues**: Include QGIS version, Python version, and error messages

## 🚀 Advanced Usage

Once working, you can:
- **Switch classifiers easily** for different datasets
- **Use automatic hyperparameter tuning** (built into each classifier)  
- **Compare performance** between different algorithms
- **Leverage specialized classifiers** for specific use cases

The new classifiers integrate seamlessly with all existing dzetsaka features including cross-validation, confidence mapping, and batch processing!