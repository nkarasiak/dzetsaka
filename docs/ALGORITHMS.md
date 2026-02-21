# dzetsaka Algorithm Reference

Complete guide to all 11 machine learning algorithms supported by dzetsaka.

## Quick Comparison Table

| Algorithm | Type | Speed | Accuracy | Dependencies | Best For |
|-----------|------|-------|----------|--------------|----------|
| GMM | Probabilistic | ⚡⚡⚡ | ⭐⭐ | None | Quick tests, baselines |
| Random Forest | Ensemble | ⚡⚡ | ⭐⭐⭐⭐ | scikit-learn | General purpose (best default) |
| SVM | Kernel-based | ⚡ | ⭐⭐⭐⭐ | scikit-learn | Small datasets, high accuracy |
| KNN | Instance-based | ⚡⚡ | ⭐⭐⭐ | scikit-learn | Simple boundaries |
| XGBoost | Gradient Boosting | ⚡ | ⭐⭐⭐⭐⭐ | xgboost | Maximum accuracy |
| CatBoost | Gradient Boosting | ⚡⚡ | ⭐⭐⭐⭐⭐ | catboost | Best default boosting |
| Extra Trees | Ensemble | ⚡⚡⚡ | ⭐⭐⭐ | scikit-learn | Fast ensemble |
| Gradient Boosting | Ensemble | ⚡ | ⭐⭐⭐⭐ | scikit-learn | Controlled overfitting |
| Logistic Regression | Linear | ⚡⚡⚡ | ⭐⭐⭐ | scikit-learn | Linear separability |
| Naive Bayes | Probabilistic | ⚡⚡⚡ | ⭐⭐⭐ | scikit-learn | Independent features |
| MLP | Neural Network | ⚡ | ⭐⭐⭐⭐ | scikit-learn | Complex patterns |

---

## Detailed Algorithm Descriptions

### 1. Gaussian Mixture Model (GMM)

**Code:** `GMM`
**Dependencies:** None (built-in)
**Hyperparameters:** Automatically tuned

**Description:**
Probabilistic model that represents each class as a mixture of Gaussian distributions. Assumes pixel values within each class follow a normal distribution.

**Strengths:**
- No external dependencies required
- Very fast training and prediction
- Provides probability estimates naturally
- Good baseline for comparison

**Weaknesses:**
- Lower accuracy than other methods
- Assumes Gaussian distribution (may not fit all data)
- Sensitive to outliers

**Use When:**
- Testing workflow quickly
- No dependencies available
- Establishing baseline accuracy
- Data follows normal distribution

**Default Configuration:**
- Covariance type: Full
- Number of components: Automatic (per class)

---

### 2. Random Forest (RF)

**Code:** `RF`
**Dependencies:** scikit-learn
**Cross-Validation:** 5-fold

**Description:**
Ensemble of decision trees trained on random subsets of data and features. Each tree votes, and the majority wins. One of the most popular algorithms for remote sensing.

**Strengths:**
- Excellent accuracy for most tasks
- Handles high-dimensional data well
- Resistant to overfitting
- Provides feature importance
- Works with minimal tuning

**Weaknesses:**
- Slower than single models
- Higher memory usage
- Can be biased toward majority classes

**Use When:**
- General-purpose classification (RECOMMENDED DEFAULT)
- Have 100+ samples per class
- Need good accuracy without much tuning
- Want to understand feature importance

**Tuned Hyperparameters:**
- `n_estimators`: 16, 32, 64, 128, 256, 512
- `max_features`: 5, 10, 20, 30, 40
- `min_samples_split`: 2, 3, 4, 5

**Typical Values:**
- Best n_estimators: 128-256
- Best max_features: Square root of total features

---

### 3. Support Vector Machine (SVM)

**Code:** `SVM`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Finds optimal hyperplane that maximizes the margin between classes. Uses RBF kernel to handle non-linear boundaries.

**Strengths:**
- Very high accuracy
- Works well with limited samples (50-100 per class)
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)

**Weaknesses:**
- Slow training on large datasets
- Sensitive to parameter tuning
- Longer prediction time

**Use When:**
- High accuracy is critical
- Have limited training samples
- Classes are well-separated
- Willing to wait for training

**Tuned Hyperparameters:**
- `gamma`: 0.25, 0.5, 1.0, 2.0, 4.0
- `C`: 0.1, 1, 10, 100

**Typical Values:**
- Best gamma: 0.5-1.0
- Best C: 10-100

---

### 4. K-Nearest Neighbors (KNN)

**Code:** `KNN`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Classifies pixels based on majority vote of K nearest training samples. Simple instance-based learning.

**Strengths:**
- Very simple and interpretable
- No training phase (lazy learning)
- Naturally handles multi-modal classes
- Good for irregular boundaries

**Weaknesses:**
- Slow prediction on large datasets
- Sensitive to noisy training data
- Memory intensive (stores all training data)
- Performance degrades in high dimensions

**Use When:**
- Need simple, explainable model
- Classes have irregular boundaries
- Training data is clean
- Dataset is not too large (<10,000 pixels)

**Tuned Hyperparameters:**
- `n_neighbors`: 1, 3, 5, 7, 9, 11, 13, 15, 17

**Typical Values:**
- Best n_neighbors: 5-9

---

### 5. XGBoost

**Code:** `XGB`
**Dependencies:** xgboost
**Cross-Validation:** 3-fold

**Description:**
Extreme Gradient Boosting - builds trees sequentially, each correcting errors of previous ones. State-of-the-art performance.

**Strengths:**
- Often highest accuracy
- Handles missing values
- Built-in regularization
- Feature importance scores
- Competitive performance

**Weaknesses:**
- Requires more samples (500+)
- Longer training time
- More hyperparameters to tune
- Can overfit with default settings

**Use When:**
- Maximum accuracy required
- Have large training dataset (500+ samples per class)
- Willing to invest time in optimization
- Participating in competitions

**Tuned Hyperparameters:**
- `n_estimators`: 50, 100, 200
- `max_depth`: 3, 5, 7
- `learning_rate`: 0.01, 0.1, 0.3

**Typical Values:**
- Best n_estimators: 100-200
- Best max_depth: 3-5
- Best learning_rate: 0.1

---

### 6. CatBoost (CB)

**Code:** `CB`
**Dependencies:** catboost
**Cross-Validation:** 3-fold

**Description:**
Gradient boosting by Yandex with strong defaults and categorical feature handling. Often works well out-of-the-box.

**Strengths:**
- Excellent default parameters
- Minimal tuning required
- Very competitive accuracy
- Robust to overfitting
- Best for users new to boosting

**Weaknesses:**
- Slower than XGBoost in some scenarios
- Higher memory usage
- Less control over tree structure

**Use When:**
- Want boosting without much tuning (RECOMMENDED)
- Need high accuracy with minimal effort
- Have moderate dataset size
- New to gradient boosting

**Tuned Hyperparameters:**
- `iterations`: 100, 200, 300
- `depth`: 4, 6, 8
- `learning_rate`: 0.01, 0.1, 0.3

**Typical Values:**
- Best iterations: 100-200
- Best depth: 6
- Best learning_rate: 0.1

---

### 7. Extra Trees (ET)

**Code:** `ET`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Extremely Randomized Trees - similar to Random Forest but with more randomization in split selection. Faster training.

**Strengths:**
- Faster training than Random Forest
- Less overfitting
- Good with noisy data
- Similar accuracy to RF

**Weaknesses:**
- Slightly lower accuracy than RF
- Less popular (fewer resources)

**Use When:**
- Want RF-like performance with faster training
- Dealing with noisy training data
- Need quick ensemble method

**Tuned Hyperparameters:**
- `n_estimators`: 50, 100, 200, 300
- `max_features`: 5, 10, 20, 30

**Typical Values:**
- Best n_estimators: 100-200

---

### 8. Gradient Boosting Classifier (GBC)

**Code:** `GBC`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Scikit-learn's gradient boosting implementation. More controlled than XGBoost but slower.

**Strengths:**
- High accuracy
- Built-in scikit-learn (no extra package)
- Good regularization control
- Less prone to overfitting than XGBoost

**Weaknesses:**
- Much slower than XGBoost
- Limited to smaller datasets
- Fewer advanced features

**Use When:**
- Want boosting without extra dependencies
- Have small-medium dataset
- Prefer scikit-learn ecosystem

**Tuned Hyperparameters:**
- `n_estimators`: 50, 100, 150
- `max_depth`: 3, 4, 5

---

### 9. Logistic Regression (LR)

**Code:** `LR`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Linear model with logistic function. Fast and simple, works when classes are linearly separable.

**Strengths:**
- Very fast training and prediction
- Low memory usage
- Good for linearly separable data
- Provides probability estimates

**Weaknesses:**
- Assumes linear boundaries
- Lower accuracy for complex data
- Sensitive to feature scaling

**Use When:**
- Classes are linearly separable
- Need very fast classification
- Working with many samples (>10,000)
- Want simple, interpretable model

**Tuned Hyperparameters:**
- `C`: 0.01, 0.1, 1, 10, 100
- `penalty`: l1, l2

---

### 10. Gaussian Naive Bayes (NB)

**Code:** `NB`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Probabilistic classifier based on Bayes' theorem assuming independence between features.

**Strengths:**
- Very fast
- Works well with small datasets
- Naturally multi-class
- Provides probabilities

**Weaknesses:**
- Assumes feature independence (rarely true)
- Lower accuracy than ensemble methods
- Sensitive to feature correlations

**Use When:**
- Need very fast classification
- Features are relatively independent
- Have limited samples
- Want probabilistic output

**Tuned Hyperparameters:**
- `var_smoothing`: Automatic

---

### 11. Multi-Layer Perceptron (MLP)

**Code:** `MLP`
**Dependencies:** scikit-learn
**Cross-Validation:** 3-fold

**Description:**
Feed-forward neural network with hidden layers. Can learn complex non-linear patterns.

**Strengths:**
- Handles complex patterns
- Flexible architecture
- Can approximate any function
- Good for very complex data

**Weaknesses:**
- Slow training
- Sensitive to initialization
- Requires more samples
- Many hyperparameters
- Risk of overfitting

**Use When:**
- Have very complex, non-linear patterns
- Large dataset available
- Other methods fail
- Willing to experiment with architecture

**Tuned Hyperparameters:**
- `hidden_layer_sizes`: (50,), (100,), (100, 50)
- `learning_rate`: constant, adaptive

**Typical Values:**
- Best hidden_layer_sizes: (100,)

---

## Algorithm Selection Flowchart

```
START
  |
  ├─ Need maximum accuracy?
  │   ├─ Yes → Have 500+ samples/class?
  │   │   ├─ Yes → Try CatBoost or XGBoost
  │   │   └─ No → Try SVM or Random Forest
  │   └─ No → Continue
  |
  ├─ No dependencies available?
  │   └─ Yes → Use GMM
  |
  ├─ Limited samples (<100/class)?
  │   └─ Yes → Try SVM or KNN
  |
  ├─ Need fast classification?
  │   └─ Yes → Try Extra Trees or Logistic Regression
  |
  └─ Default → Use Random Forest (best general purpose)
```

---

## Performance Tips

### Memory Management
- All algorithms process rasters in blocks (512MB limit)
- Large models (RF, XGB) use more memory
- Use masks to exclude unnecessary areas

### Training Speed
- GMM, NB, LR: Very fast (<1 minute)
- RF, ET, KNN: Fast (1-5 minutes)
- SVM, MLP: Moderate (5-15 minutes)
- XGB, CB, GBC: Slow (15-60 minutes with tuning)

### Prediction Speed
- GMM, LR, NB: Very fast
- RF, ET, XGB, CB: Fast
- KNN: Slow (stores all training data)
- SVM, MLP: Moderate

---

## Further Reading

- **Random Forest:** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- **XGBoost:** Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
- **SVM:** Mountrakis et al. (2011). Support vector machines in remote sensing.
- **CatBoost:** Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features.

---

**Questions?** Open an issue: https://github.com/nkarasiak/dzetsaka/issues
