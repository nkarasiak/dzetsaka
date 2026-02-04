# Phase 3 Kickoff: Class Imbalance & Nested CV (Weeks 6-7)

## Context

**Previous Phases**:
- ✅ Phase 1 (v4.3.0): Optuna optimization - 2-10x faster training
- ✅ Phase 2 (v4.4.0): SHAP explainability - model interpretability

**Current Phase**:
- 🚀 Phase 3 (v4.5.0): Class Imbalance & Nested CV - robust validation

## Phase 3 Objectives

**Target Version**: 4.5.0
**Timeline**: Weeks 6-7 (10 days)
**Goal**: Handle imbalanced datasets and provide unbiased model evaluation

### Deliverables

#### Week 6: Class Imbalance Handling

1. **SMOTE Implementation** (`scripts/sampling/smote_sampler.py`)
   - Synthetic Minority Over-sampling Technique
   - Generate synthetic samples for minority classes
   - Integration with training pipeline
   - Support for multi-class imbalance

2. **Class Weights Module** (`scripts/sampling/class_weights.py`)
   - Automatic class weight computation
   - Balanced vs. custom weight strategies
   - Integration with all supported algorithms
   - Cost-sensitive learning support

3. **Stratified Sampling** (`scripts/sampling/stratified_sampler.py`)
   - Ensure proportional representation in splits
   - Cross-validation with stratification
   - Support for spatial stratification

#### Week 7: Nested Cross-Validation

4. **Nested CV Module** (`scripts/validation/nested_cv.py`)
   - Inner loop: Hyperparameter optimization
   - Outer loop: Model evaluation
   - Unbiased performance estimates
   - Proper train/validation/test separation

5. **Enhanced Metrics** (`scripts/validation/metrics.py`)
   - Per-class precision, recall, F1
   - ROC curves and AUC computation
   - Learning curves for overfitting detection
   - Confusion matrix improvements

6. **Integration & Documentation**
   - Update LearnModel with new parameters
   - Processing algorithms for batch validation
   - Comprehensive documentation
   - Usage examples

## Implementation Details

### 1. SMOTE Implementation

```python
# scripts/sampling/smote_sampler.py

class SMOTESampler:
    """SMOTE-based oversampling for imbalanced datasets."""

    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: Union[str, dict] = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for minority classes.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        sampling_strategy : str or dict
            'auto': Balance to majority class
            'minority': Oversample only minority class
            dict: Custom per-class targets

        Returns
        -------
        X_resampled : np.ndarray
            Features with synthetic samples
        y_resampled : np.ndarray
            Labels with synthetic samples
        """
```

**Key Features**:
- KNN-based synthetic sample generation
- Multi-class support
- Configurable sampling strategies
- Integration with existing pipeline

### 2. Class Weights Module

```python
# scripts/sampling/class_weights.py

def compute_class_weights(
    y: np.ndarray,
    strategy: str = 'balanced',
    custom_weights: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.

    Parameters
    ----------
    y : np.ndarray
        Training labels
    strategy : str
        'balanced': n_samples / (n_classes * np.bincount(y))
        'custom': Use custom_weights
    custom_weights : dict, optional
        Custom weights per class

    Returns
    -------
    weights : dict
        Class weights {class_id: weight}
    """
```

**Integration Points**:
- Random Forest: `class_weight` parameter
- SVM: `class_weight` parameter
- XGBoost: `scale_pos_weight` parameter
- LightGBM: `class_weight` parameter

### 3. Nested Cross-Validation

```python
# scripts/validation/nested_cv.py

class NestedCrossValidator:
    """Nested cross-validation for unbiased model evaluation."""

    def __init__(
        self,
        inner_cv: int = 3,
        outer_cv: int = 5,
        random_state: int = 42
    ):
        self.inner_cv = inner_cv  # Hyperparameter tuning
        self.outer_cv = outer_cv  # Model evaluation

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier_code: str,
        param_grid: dict
    ) -> Dict[str, Any]:
        """
        Perform nested CV and return unbiased metrics.

        Returns
        -------
        results : dict
            - 'outer_scores': List of outer fold scores
            - 'mean_score': Mean performance
            - 'std_score': Standard deviation
            - 'best_params': Best params from each outer fold
        """
```

### 4. Enhanced Metrics Module

```python
# scripts/validation/metrics.py

class ValidationMetrics:
    """Enhanced validation metrics for classification."""

    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """Compute precision, recall, F1 per class."""

    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        output_path: str
    ) -> None:
        """Generate ROC curve plots."""

    @staticmethod
    def plot_learning_curves(
        model,
        X: np.ndarray,
        y: np.ndarray,
        output_path: str
    ) -> None:
        """Generate learning curves for overfitting detection."""
```

## New Parameters in extraParam

### SMOTE Parameters
- **`USE_SMOTE`**: `bool`, default=False - Enable SMOTE oversampling
- **`SMOTE_K_NEIGHBORS`**: `int`, default=5 - Number of neighbors for SMOTE
- **`SMOTE_STRATEGY`**: `str`, default='auto' - Sampling strategy

### Class Weights Parameters
- **`USE_CLASS_WEIGHTS`**: `bool`, default=False - Enable class weighting
- **`CLASS_WEIGHT_STRATEGY`**: `str`, default='balanced' - Weight computation strategy
- **`CUSTOM_CLASS_WEIGHTS`**: `dict`, optional - Custom weights per class

### Nested CV Parameters
- **`USE_NESTED_CV`**: `bool`, default=False - Enable nested cross-validation
- **`NESTED_INNER_CV`**: `int`, default=3 - Inner CV folds
- **`NESTED_OUTER_CV`**: `int`, default=5 - Outer CV folds

### Metrics Parameters
- **`COMPUTE_ROC`**: `bool`, default=False - Generate ROC curves
- **`ROC_OUTPUT`**: `str`, optional - Path for ROC curve plot
- **`COMPUTE_LEARNING_CURVES`**: `bool`, default=False - Generate learning curves
- **`LEARNING_CURVES_OUTPUT`**: `str`, optional - Path for learning curves plot

## Directory Structure

```
dzetsaka/
├── scripts/
│   ├── sampling/                    # NEW: Sampling techniques
│   │   ├── __init__.py
│   │   ├── smote_sampler.py         # SMOTE implementation
│   │   ├── class_weights.py         # Class weight computation
│   │   └── stratified_sampler.py    # Stratified sampling
│   ├── validation/                  # NEW: Validation methods
│   │   ├── __init__.py
│   │   ├── nested_cv.py             # Nested cross-validation
│   │   └── metrics.py               # Enhanced metrics
│   └── mainfunction.py              # MODIFIED: Integration
├── processing/
│   └── nested_cv_algorithm.py       # NEW: Processing algorithm
├── tests/
│   ├── unit/
│   │   ├── test_smote_sampler.py    # NEW: SMOTE tests
│   │   ├── test_class_weights.py    # NEW: Weights tests
│   │   └── test_nested_cv.py        # NEW: Nested CV tests
│   └── integration/
│       └── test_imbalance_workflow.py # NEW: Integration tests
└── PHASE3_SUMMARY.md                # Summary document
```

## Dependencies

Add to `pyproject.toml`:
```toml
# Imbalanced learning
imbalanced = [
    "imbalanced-learn>=0.10.0",  # SMOTE and other techniques
]

# Visualization for metrics
visualization = [
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]

# Full with all features
full = [
    "scikit-learn>=1.0.0",
    "xgboost>=1.5.0",
    "lightgbm>=3.2.0",
    "optuna>=3.0.0",
    "shap>=0.41.0",
    "imbalanced-learn>=0.10.0",  # NEW
    "matplotlib>=3.5.0",          # NEW
    "seaborn>=0.11.0",            # NEW
]
```

## Expected Outputs

### User-Facing
1. **Balanced Training**: Automatic handling of imbalanced datasets
2. **Reliable Metrics**: Unbiased performance estimates via nested CV
3. **Rich Diagnostics**: Per-class metrics, ROC curves, learning curves
4. **Processing Algorithms**: Batch validation workflows

### Technical
1. **New Modules**:
   - `scripts/sampling/` (~400 lines)
   - `scripts/validation/` (~500 lines)
2. **Modified**: `scripts/mainfunction.py` (integration)
3. **New Algorithm**: `processing/nested_cv_algorithm.py` (~300 lines)
4. **Tests**: ~800 lines of unit and integration tests
5. **Docs**: Usage guide and examples

## Success Criteria

✅ SMOTE works with all algorithms
✅ Class weights properly applied
✅ Nested CV provides unbiased estimates
✅ Per-class metrics computed correctly
✅ ROC curves generated for multiclass
✅ Learning curves detect overfitting
✅ Performance acceptable (<2x slowdown)
✅ Documentation complete
✅ Tests pass (>70% coverage)
✅ Backward compatible

## Estimated Timeline

**Day 1-2**: SMOTE implementation and class weights
**Day 3-4**: Integration with mainfunction.py
**Day 5-6**: Nested CV implementation
**Day 7-8**: Enhanced metrics (ROC, learning curves)
**Day 9-10**: Testing, documentation, polish

## Known Challenges

1. **SMOTE Memory Usage**: Large datasets may require chunked processing
   - Solution: Sample-based SMOTE with configurable limits

2. **Nested CV Time**: Outer × Inner CV multiplies training time
   - Solution: Use with Optuna for faster hyperparameter tuning
   - Recommend: 3×5 CV for reasonable time

3. **Multiclass ROC**: Need one-vs-rest approach
   - Solution: Use sklearn's multiclass ROC utilities

4. **Spatial Autocorrelation**: Standard CV may overestimate performance
   - Solution: Offer spatial CV option in future version

## Reference Implementations

- **imbalanced-learn**: SMOTE and class balancing
- **sklearn.model_selection**: Nested CV patterns
- **sklearn.metrics**: ROC curves, classification reports

---

**Ready for Phase 3!** Let's build robust validation and imbalance handling! 🚀

**Version**: 4.5.0 (target)
**Author**: Nicolas Karasiak
**Contributors**: Claude Sonnet 4.5
