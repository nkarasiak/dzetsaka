# dzetsaka: classification tool

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3463523.svg)](https://doi.org/10.5281/zenodo.3463523)

![Inselberg in Guiana Amazonian Park](https://raw.githubusercontent.com/nkarasiak/dzetsaka/main/img/guyane.jpg)

dzetsaka <img src="https://raw.githubusercontent.com/nkarasiak/dzetsaka/main/img/icon.png" alt="dzetsaka logo" width="30px"/> is a raster classification plugin for QGIS. It started around the Gaussian Mixture Model classifier written by [Mathieu Fauvel](http://fauvel.mathieu.free.fr) and now ships 11 machine learning algorithms, hyperparameter search, SHAP explanations, and class imbalance handling. It is maintained by [Nicolas Karasiak](https://github.com/nkarasiak/dzetsaka).

If you only need to classify forests on old maps, [Historical Map](https://github.com/lennepkade/HistoricalMap) is the narrower tool for that job.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## What you need

Two inputs are enough:

- a **raster**
- a **shapefile** holding your ROI (Region Of Interest)

The shapefile needs a column with class numbers (1, 3, 4...). Text labels will not work.

Most algorithms need scikit-learn. GMM runs without it. You can [download samples](https://github.com/lennepkade/dzetsaka/archive/docs.zip) to try the plugin on real data.

## Supported algorithms

Available without extra packages beyond scikit-learn:

- Gaussian Mixture Model (GMM), the fastest baseline and the only one with no scikit-learn requirement
- Random Forest (RF)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Extra Trees (ET)
- Gradient Boosting Classifier (GBC)
- Logistic Regression (LR)
- Naive Bayes (NB)
- Multi-layer Perceptron (MLP)

Available once their own library is installed:

- XGBoost (XGB)
- CatBoost (CB)

### Automatic dependency installation

When you pick an algorithm whose packages are missing, dzetsaka detects it, offers to install them, and runs pip in the background while printing progress to the QGIS log. This covers scikit-learn, XGBoost, and CatBoost, so in most cases you never have to open a terminal.

## Installing dependencies by hand

### Linux

```
python3 -m pip install scikit-learn -U --user
```

### macOS

From the QGIS Python console (Plugins > Python Console):

```
import subprocess; subprocess.check_call(["/Applications/QGIS.app/Contents/MacOS/bin/pip3", "install", "scikit-learn", "--user"])
```

Or from a terminal, if you have a system Python 3:

```
python3 -m pip install scikit-learn -U --user
```

Some macOS setups also need joblib installed separately (`python3 -m pip install joblib -U --user`). Restart QGIS afterwards so the libraries are picked up.

### Windows

Open the OSGeo4W shell, then run `o4w_env` (QGIS 3.20 and above) or `py3_env.bat` (QGIS 3.18 and below), followed by:

```
python3 -m pip install scikit-learn -U --user
```

Thanks to Alexander Bruy for the tip.

For QGIS 2, install PIP through the OSGeo4W setup (Advanced install), open the OSGeo4W Shell as administrator, and run `pip install scikit-learn`.

## Hyperparameter tuning

dzetsaka tunes parameters with cross-validated grid search:

- RF: 5-fold CV over n_estimators and max_features
- SVM: 3-fold CV over gamma (0.25 to 4.0) and C (0.1 to 100)
- KNN: 3-fold CV over n_neighbors (1 to 17)
- XGB: 3-fold CV over n_estimators (50 to 200), max_depth (3 to 9), learning_rate (0.01 to 0.2)
- CB: 3-fold CV over iterations, depth, learning_rate, l2_leaf_reg
- ET: 3-fold CV over n_estimators and max_features
- GBC: 3-fold CV over n_estimators and max_depth
- LR: 3-fold CV over C and penalty
- MLP: 3-fold CV over hidden_layer_sizes and learning_rate
- GMM and NB are not tuned

Optuna-based search is available for larger parameter spaces. If you want to define your own grid, use the parameter grid field in the processing interface.

### Sparse class labels

Classes do not have to be contiguous. If your labels are 0, 1, 3 with no 2, the scikit-learn algorithms handle it directly, and XGBoost and CatBoost get their labels encoded and decoded around training so you do not have to renumber anything.

## QGIS UI and core runtime

The QGIS plugin UI lives under `src/dzetsaka/qgis`. Anything that used to import from `dzetsaka.presentation.qgis` goes through a small shim at `src/dzetsaka/presentation/qgis/__init__.py`. The ML code (classification, training, SHAP, Optuna, SMOTE) sits in the shared `dzetsaka` package and runs without QGIS, which is what makes CLI and batch use possible. `docs/runtime_split.md` explains how the imports resolve.

## CLI usage

Install with `pip install -e .` (or build a wheel), then:

```
dzetsaka classify --raster input.tif --model model.pkl --output classification.tif
dzetsaka train --raster train.tif --vector train.shp --model model.pkl
```

Both commands take the same `--nodata`, `--confidence`, `--classifier`, and `--matrix-path` arguments as the QGIS UI, and print progress to stdout. Pass JSON to `--extra`, or point at a file with `@extras.json`, to switch on SHAP, Optuna, SMOTE, or any other flag that `scripts/classification_pipeline.py` recognizes.

## Tips

- If your raster is *spot6scene.tif*, name your mask *spot6scene_mask.tif* and the script will find it on its own.
- Save your model if you want to reuse a spectral ROI on another image.

Development documentation is generated on the [doxygen branch](https://rawgit.com/lennepkade/dzetsaka/doxygen/index.html).

## What does dzetsaka mean?

I wrote this tool while working in the Guiana Amazonian Park, classifying kinds of vegetation, so I named it in Teko, a native-american language spoken by a nation living in French Guiana. It refers to the objects we look at the world through: satellites, microscopes, cameras.

## Citing dzetsaka

If dzetsaka is useful in your research, please cite it:

```
@misc{karasiak2016dzetsaka,
title={Dzetsaka Qgis Classification plugin},
author={Karasiak, Nicolas},
url={https://github.com/nkarasiak/dzetsaka},
year={2016},
doi={10.5281/zenodo.2552284}
}
```

## Thanks to...

Thanks to the [Guiana Amazonian Park](http://www.parc-amazonien-guyane.fr/) for trusting my work, and to the Master 2 Geomatics [Sigma](http://sigma.univ-toulouse.fr/en/welcome.html) for their lessons in geomatics.

![Sponsors of QGIS](https://raw.githubusercontent.com/nkarasiak/dzetsaka/main/img/logo.png)
