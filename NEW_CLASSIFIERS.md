# New Machine Learning Classifiers in dzetsaka v4.1.0

dzetsaka v4.1.0 introduces **7 new powerful machine learning classifiers** to give users more options for land cover classification. These classifiers complement the existing GMM, Random Forest, SVM, and KNN algorithms.

## 🆕 New Classifiers

### 1. **XGBoost (XGB)**
- **Type**: Gradient Boosting Framework
- **Strengths**: High performance, handles missing values, built-in regularization
- **Best for**: Complex datasets with mixed feature types
- **Installation**: `pip install xgboost`
- **Parameters**: n_estimators, max_depth, learning_rate

### 2. **LightGBM (LGB)**
- **Type**: Fast Gradient Boosting
- **Strengths**: Fast training, low memory usage, high accuracy
- **Best for**: Large datasets where speed is important
- **Installation**: `pip install lightgbm`
- **Parameters**: n_estimators, num_leaves, learning_rate

### 3. **Extra Trees (ET)**
- **Type**: Extremely Randomized Trees
- **Strengths**: Reduced overfitting, faster training than Random Forest
- **Best for**: High-dimensional data with noise
- **Installation**: Included with scikit-learn
- **Parameters**: n_estimators, max_features

### 4. **Gradient Boosting Classifier (GBC)**
- **Type**: Sklearn Gradient Boosting
- **Strengths**: Good performance, interpretable feature importance
- **Best for**: Structured data with moderate size
- **Installation**: Included with scikit-learn
- **Parameters**: n_estimators, max_depth, learning_rate

### 5. **Logistic Regression (LR)**
- **Type**: Linear Classification
- **Strengths**: Fast, interpretable, works well with linear relationships
- **Best for**: Linearly separable classes, baseline models
- **Installation**: Included with scikit-learn
- **Parameters**: C (regularization), solver

### 6. **Gaussian Naive Bayes (NB)**
- **Type**: Probabilistic Classifier
- **Strengths**: Fast, works well with small datasets, handles categorical features
- **Best for**: Text classification, small datasets, baseline models
- **Installation**: Included with scikit-learn
- **Parameters**: var_smoothing

### 7. **Multi-layer Perceptron (MLP)**
- **Type**: Neural Network
- **Strengths**: Can learn complex non-linear relationships
- **Best for**: Complex patterns, non-linear relationships
- **Installation**: Included with scikit-learn
- **Parameters**: hidden_layer_sizes, alpha, learning_rate_init

## 📦 Installation

### Core classifiers (included with scikit-learn):
```bash
pip install scikit-learn
```

### Optional high-performance classifiers:
```bash
pip install xgboost lightgbm
```

### Install everything at once:
```bash
pip install scikit-learn xgboost lightgbm
```

## 🎯 Classifier Selection Guide

| Dataset Characteristics | Recommended Classifiers |
|------------------------|------------------------|
| **Small dataset (< 1000 samples)** | NB, LR, KNN |
| **Large dataset (> 10000 samples)** | XGB, LGB, RF |
| **High-dimensional data** | ET, RF, LR |
| **Linear relationships** | LR, SVM |
| **Complex non-linear patterns** | XGB, LGB, MLP |
| **Speed is critical** | LGB, NB, LR |
| **Maximum accuracy** | XGB, LGB, RF |
| **Interpretability needed** | RF, ET, LR |

## 🔧 Usage Examples

### Using XGBoost:
```python
from scripts.mainfunction import learnModel

# Train with XGBoost
model = learnModel(
    raster_path="image.tif",
    vector_path="training.shp", 
    class_field="class",
    classifier="XGB",  # Use XGBoost
    model_path="model_xgb.pkl"
)
```

### Using LightGBM:
```python
model = learnModel(
    raster_path="image.tif",
    vector_path="training.shp",
    class_field="class", 
    classifier="LGB",  # Use LightGBM
    model_path="model_lgb.pkl"
)
```

### Custom parameters:
```python
# Custom hyperparameters
custom_params = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05
}

model = learnModel(
    raster_path="image.tif",
    vector_path="training.shp",
    class_field="class",
    classifier="XGB",
    extraParam={"param_algo": custom_params},
    model_path="model_custom.pkl"
)
```

## ⚙️ Automatic Hyperparameter Tuning

All classifiers come with pre-configured parameter grids for automatic hyperparameter optimization via GridSearchCV:

- **XGBoost**: n_estimators [50, 100, 200], max_depth [3, 6, 9], learning_rate [0.01, 0.1, 0.2]
- **LightGBM**: n_estimators [50, 100, 200], num_leaves [31, 50, 100], learning_rate [0.01, 0.1, 0.2]
- **Extra Trees**: n_estimators [50, 100, 200], max_features (adaptive to data dimensions)
- **Gradient Boosting**: n_estimators [50, 100, 200], max_depth [3, 5, 7], learning_rate [0.01, 0.1, 0.2]
- **Logistic Regression**: C [0.001, 0.01, 0.1, 1, 10, 100], solver ["liblinear", "lbfgs"]
- **Naive Bayes**: var_smoothing [1e-12 to 1e-3]
- **MLP**: hidden_layer_sizes [(50,), (100,), (50,50), (100,50)], alpha [0.0001, 0.001, 0.01]

## 🔍 Validation and Error Handling

The system automatically:
- ✅ Checks if required packages are installed
- ✅ Provides helpful installation instructions
- ✅ Validates algorithm availability before training
- ✅ Shows clear error messages with solutions

## 🔄 Backward Compatibility

All new classifiers integrate seamlessly with existing dzetsaka workflows:
- ✅ Same API as existing classifiers
- ✅ Compatible with all cross-validation methods (SLOO, STAND)
- ✅ Works with existing confidence mapping
- ✅ Supports all extraParam configurations

## 📊 Performance Comparison

For remote sensing applications, preliminary tests suggest:

1. **XGBoost/LightGBM**: Best overall accuracy for complex landscapes
2. **Extra Trees**: Good balance of speed and accuracy 
3. **Random Forest**: Still excellent, now with more alternatives
4. **Logistic Regression**: Fast baseline, good for simple classifications
5. **Naive Bayes**: Fastest training, decent for preliminary analysis
6. **MLP**: Good for very complex patterns, requires more data

## 🚀 Getting Started

1. **Update dzetsaka to v4.1.0**
2. **Install optional packages**: `pip install xgboost lightgbm`
3. **Try different classifiers** on your data to find the best performer
4. **Use the same workflow** - just change the `classifier` parameter!

For detailed examples and advanced usage, see the [Parameter Migration Guide](PARAMETER_MIGRATION_GUIDE.md) and updated documentation.