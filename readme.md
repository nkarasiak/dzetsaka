# dzetsaka : classification tool
[![DOI](https://zenodo.org/badge/59029116.svg)](https://zenodo.org/badge/latestdoi/59029116)

![Inselberg in Guiana Amazonian Park](https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/guyane.jpg)

dzetsaka <img src="https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/icon.png" alt="dzetsaka logo" width="30px"/> is very fast and easy to use but also a **powerful classification plugin for Qgis**. Initially based on Gaussian Mixture Model classifier developed by [Mathieu Fauvel](http://fauvel.mathieu.free.fr), this plugin now supports **12 machine learning algorithms** including advanced gradient boosting methods like XGBoost, LightGBM, and CatBoost. This plugin is a more generalist tool than [Historical Map](https://github.com/lennepkade/HistoricalMap) which was dedicated to classify forests from old maps.
This plugin has by developped by [Nicolas Karasiak](https://github.com/nkarasiak/dzetsaka).

## QGIS vs core runtime

The QGIS plugin UI now lives under `src/dzetsaka/qgis`, and everything that previously imported from `dzetsaka.presentation.qgis` is rerouted through a tiny shim (`src/dzetsaka/presentation/qgis/__init__.py`). The heavy ML logic (classification, training, SHAP, Optuna, SMOTE) is part of the shared `dzetsaka` package and can run without QGIS, which opens the door to CLI or batch usage. See `docs/runtime_split.md` for the high-level architecture and to understand how imports resolve.

## CLI usage

Install dzetsaka with `pip install -e .` (or build/distribute the wheel) and call the CLI commands:

```
dzetsaka classify --raster input.tif --model model.pkl --output classification.tif
dzetsaka train --raster train.tif --vector train.shp --model model.pkl
```

Both commands accept the same `--nodata`, `--confidence`, `--classifier`, and `--matrix-path` arguments that the QGIS UI exposes and print progress feedback to stdout. Supply JSON for `--extra` or point to a file with `@extras.json` to activate SHAP explainability, Optuna, SMOTE, or any other advanced flag recognized by `scripts/classification_pipeline.py`.

You can [download samples](https://github.com/lennepkade/dzetsaka/archive/docs.zip) to test the plugin on your own.

## What does dzetsaka mean ?
As this tool was developped during my work in the Guiana Amazonian Park to classify different kind of vegetation, I gave an Teko name (a native-american language from a nation which lives in french Guiana) which represent the objects we use to see the world through, such as satellites, microscope, camera... 

## Discover dzetsaka
`dzetsaka : Classification tool` runs with scipy library. You can download package like [Spider by Anaconda](https://docs.continuum.io/anaconda/) for a very easy setup. 

Then, as this plugin is very simple, you will just need two things for making a good classification : 
- A **raster**
- A **shapefile** which contains your **ROI** (Region Of Interest)

The shapefile must have a column which contains your classification numbers *(1,3,4...)*. Otherwise if you use text or anything else it certainly won't work.

## 🎯 Supported Algorithms

dzetsaka now supports **12 powerful machine learning algorithms**:

### **Core Algorithms** (built-in)
- **Gaussian Mixture Model (GMM)** - Fast baseline classifier
- **Random Forest (RF)** - Robust ensemble method
- **Support Vector Machine (SVM)** - High-accuracy classifier
- **K-Nearest Neighbors (KNN)** - Simple distance-based classifier

### **Advanced Algorithms** ⭐ NEW
- **XGBoost (XGB)** - State-of-the-art gradient boosting
- **LightGBM (LGB)** - Fast gradient boosting framework
- **CatBoost (CB)** - Gradient boosting with strong defaults
- **Extra Trees (ET)** - Extremely randomized trees
- **Gradient Boosting Classifier (GBC)** - Scikit-learn gradient boosting
- **Logistic Regression (LR)** - Linear probabilistic classifier
- **Naive Bayes (NB)** - Fast probabilistic classifier
- **Multi-layer Perceptron (MLP)** - Neural network classifier

### **🚀 Automatic Dependency Installation**

**NEW FEATURE**: dzetsaka can now automatically install missing dependencies!

When you select an algorithm that requires additional packages (XGBoost, LightGBM, CatBoost), dzetsaka will:
1. **Detect missing dependencies** automatically
2. **Offer to install them** with one click
3. **Handle the installation process** in the background
4. **Provide real-time progress** in the QGIS log

**Supported auto-installation**:
- ✅ scikit-learn (for RF, SVM, KNN, ET, GBC, LR, NB, MLP)
- ✅ XGBoost (for XGB classifier)
- ✅ LightGBM (for LGB classifier)
- ✅ CatBoost (for CB classifier)

**No more manual pip commands!** Just select your algorithm and let dzetsaka handle the rest.

## Manual Installation (if needed)

### On Linux
Simply open terminal and type: 
`python3 -m pip install scikit-learn -U --user`

### On macOS
**Method 1 - Using QGIS Python console (Recommended):**
1. Open QGIS
2. Go to Plugins → Python Console
3. Type: `import subprocess; subprocess.check_call(["/Applications/QGIS.app/Contents/MacOS/bin/pip3", "install", "scikit-learn", "--user"])`

**Method 2 - Using Terminal:**
If you have Python 3 installed globally:
`python3 -m pip install scikit-learn -U --user`

**Note:** On some macOS systems, you may also need to install joblib separately:
`python3 -m pip install joblib -U --user`

After installation, restart QGIS to ensure the libraries are properly loaded.

### On Windows
**For QGIS 3.20 and higher:** 
Open OsGeo shell, then :

`o4w_env`

`python3 -m pip install scikit-learn -U --user`

**For Qgis 3.18 and lower**: 
Open OsGeo shell, then :

`py3_env.bat`

`python3 -m pip install scikit-learn -U --user`

Thanks to Alexander Bruy for the tip.

**For Qgis 2**:
In the OsGeo setup, search for PIP and install it. Then you have few more steps to do. In the explorer, search for OsGeo4W Shell, right click to open it as an administrator. Now use pip in OsGeo Shell like on Linux. Just type :<br/>
`pip install scikit-learn`

If you do not have pip installed, open osgeo4w-setup-x86_64.exe, select Advanced install and install *pip*.


You can now use **all 12 machine learning algorithms** including XGBoost, LightGBM, and CatBoost!

## 🔧 Algorithm Parameters & Performance

### **Hyperparameter Optimization**
dzetsaka automatically optimizes algorithm parameters using **cross-validation grid search**:

**Core Algorithms:**
- **Random Forest (RF)**: 5-fold CV, optimizes n_estimators and max_features
- **SVM**: 3-fold CV, optimizes gamma (0.25-4.0) and C (0.1-100)
- **KNN**: 3-fold CV, optimizes n_neighbors (1-17)
- **GMM**: No tuning (fastest baseline)

**Advanced Algorithms** ⭐:
- **XGBoost (XGB)**: 3-fold CV, optimizes n_estimators (50-200), max_depth (3-9), learning_rate (0.01-0.2)
- **LightGBM (LGB)**: 3-fold CV, optimizes n_estimators (50-200), num_leaves (31-100), learning_rate (0.01-0.2)
- **CatBoost (CB)**: 3-fold CV, optimizes iterations, depth, learning_rate, l2_leaf_reg
- **Extra Trees (ET)**: 3-fold CV, optimizes n_estimators and max_features
- **Gradient Boosting (GBC)**: 3-fold CV, optimizes n_estimators and max_depth
- **Logistic Regression (LR)**: 3-fold CV, optimizes C and penalty
- **Naive Bayes (NB)**: Uses optimal default parameters
- **MLP**: 3-fold CV, optimizes hidden_layer_sizes and learning_rate

### **🎯 Label Handling**
dzetsaka automatically handles **sparse class labels** (e.g., classes 0, 1, 3 - missing class 2):
- **Core algorithms**: Work natively with sparse labels
- **XGBoost/LightGBM/CatBoost**: Automatic label encoding/decoding for compatibility
- **Seamless workflow**: No manual preprocessing required

### Custom Parameters
Advanced users can provide custom parameters through the processing interface using the parameter grid functionality.

## Tips

- If your raster is *spot6scene.tif*, you can create your mask under the name *spot6scene_mask.tif* and the script will detect it automatically.
- If you want to keep your spectral ROI model from an image, you can save your model to use it on another image.

Online dev documentation is available throught the [doxygen branch](https://rawgit.com/lennepkade/dzetsaka/doxygen/index.html).

## Like us, use us ? Cite us !

If you use dzetsaka in your research and find it useful, please cite Dzetsaka using the following bibtex reference:

```
@misc{karasiak2016dzetsaka,
title={Dzetsaka Qgis Classification plugin},
author={Karasiak, Nicolas},
url={https://github.com/nkarasiak/dzetsaka},
year={2016},
doi={10.5281/zenodo.2552284}
}
```

### Thanks to...
I would like to thank the [Guiana Amazonian Park](http://www.parc-amazonien-guyane.fr/) for their trust in my work, and the Master 2 Geomatics [Sigma](http://sigma.univ-toulouse.fr/en/welcome.html) for their excellent lessons in geomatics.

![Sponsors of Qgis](https://cdn.rawgit.com/lennepkade/dzetsaka/master/img/logo.png)
