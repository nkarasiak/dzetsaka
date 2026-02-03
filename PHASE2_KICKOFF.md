# Phase 2 Kickoff: SHAP & Explainability (Weeks 4-5)

## Context

This document provides everything needed to start Phase 2 implementation in a new chat session.

## Phase 1 Status: ✅ COMPLETE

**Version**: 4.3.0
**Completed**: 2026-02-03
**Summary**: See `PHASE1_SUMMARY.md` for full details

**Key Deliverables**:
- ⚡ Optuna optimization (2-10x faster training)
- 🏗️ Classifier factory pattern
- 🛡️ Custom exception hierarchy
- 📦 Clean architecture (optimization/, domain/, factories/)

## Phase 2 Objectives: SHAP & Explainability

**Target Version**: 4.4.0
**Timeline**: Weeks 4-5 (10 days)
**Goal**: Add model explainability to UI and Processing framework

### Deliverables

#### Week 4: SHAP Core Implementation

1. **Create SHAP Explainer Module** (`scripts/explainability/shap_explainer.py`)
   - `ModelExplainer` class with SHAP integration
   - TreeExplainer for tree-based models (RF, XGB, LGB, ET, GBC)
   - KernelExplainer fallback for other models (SVM, KNN, LR, NB, MLP)
   - Feature importance computation
   - Raster output generation

2. **Extend LearnModel for SHAP** (`scripts/mainfunction.py`)
   - Add optional SHAP value computation during training
   - Store feature importance in model metadata
   - New parameter: `COMPUTE_SHAP` in extraParam

#### Week 5: UI & Processing Integration

3. **UI Integration** (`ui/dzetsaka_dockwidget.py`)
   - New checkbox: "Generate feature importance map"
   - Output file selector for importance raster
   - Connect to training workflow
   - Progress feedback during SHAP computation

4. **Processing Algorithm** (`processing/explain_model_algorithm.py`)
   - New algorithm: "Explain Model (SHAP)"
   - Inputs: trained model (.model file) + raster (.tif)
   - Outputs: feature importance raster (.tif)
   - Batch processing ready

5. **Documentation & Examples**
   - Usage guide for SHAP feature importance
   - Example workflows
   - Update CHANGELOG.md for v4.4.0
   - Update metadata.txt

## Implementation Details

### SHAP Explainer Architecture

```python
# scripts/explainability/shap_explainer.py

class ModelExplainer:
    """Generate feature importance using SHAP values."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = self._create_explainer()

    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type."""
        if hasattr(self.model, 'tree_'):
            # Tree-based models: RF, XGB, LGB, ET, GBC
            return shap.TreeExplainer(self.model)
        else:
            # Other models: SVM, KNN, LR, NB, MLP
            return shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(X_background, 100)
            )

    def get_feature_importance(self, X_sample: np.ndarray) -> Dict[str, float]:
        """Calculate SHAP-based feature importance."""
        shap_values = self.explainer.shap_values(X_sample)
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, importance))

    def create_importance_raster(
        self,
        raster_path: str,
        output_path: str,
        sample_size: int = 1000
    ):
        """Generate raster showing per-pixel feature importance."""
        # Implementation: read raster, compute SHAP, write output
```

### UI Integration

**New Controls in Dock Widget**:
```python
# ui/dzetsaka_dockwidget.py

self.checkbox_shap = QCheckBox("Generate feature importance map")
self.file_shap_output = QgsFileWidget()
self.file_shap_output.setStorageMode(QgsFileWidget.SaveFile)
self.file_shap_output.setFilter("GeoTIFF (*.tif)")

# Connect to training workflow
extraParam["COMPUTE_SHAP"] = self.checkbox_shap.isChecked()
extraParam["SHAP_OUTPUT"] = self.file_shap_output.filePath()
```

### Processing Algorithm

**New File**: `processing/explain_model_algorithm.py`

```python
class ExplainModelAlgorithm(QgsProcessingAlgorithm):
    """Generate SHAP feature importance map from trained model."""

    INPUT_MODEL = 'INPUT_MODEL'
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_IMPORTANCE = 'OUTPUT_IMPORTANCE'
    SAMPLE_SIZE = 'SAMPLE_SIZE'

    def processAlgorithm(self, parameters, context, feedback):
        # Load model
        model_path = self.parameterAsFile(parameters, self.INPUT_MODEL)
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Load raster
        raster_path = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER)

        # Create explainer
        explainer = ModelExplainer(
            model=model_data['model'],
            feature_names=model_data.get('feature_names', [])
        )

        # Generate importance raster
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_IMPORTANCE)
        explainer.create_importance_raster(raster_path, output_path)

        return {self.OUTPUT_IMPORTANCE: output_path}
```

## Testing Strategy

### Unit Tests
- `tests/unit/test_shap_explainer.py` - Test ModelExplainer class
- Test TreeExplainer creation for tree models
- Test KernelExplainer fallback
- Test feature importance computation
- Test raster I/O operations

### Integration Tests
- `tests/integration/test_shap_workflow.py` - End-to-end SHAP workflow
- Train model + compute SHAP in one workflow
- Processing algorithm execution
- UI checkbox integration

### Manual Testing
1. Train RF model with SHAP checkbox enabled
2. Verify feature importance raster is generated
3. Visualize in QGIS (darker = more important)
4. Run Processing algorithm on existing model
5. Batch process multiple rasters

## Dependencies

Add to `pyproject.toml`:
```toml
# Model explainability (SHAP)
explainability = [
    "shap>=0.41.0",
]
```

Already added in Phase 1, just need to implement!

## Expected Outputs

### User-Facing
1. **UI Checkbox**: Users can enable SHAP during training
2. **Feature Importance Raster**: Shows which bands matter most
3. **Processing Algorithm**: Batch generate SHAP maps
4. **Better Understanding**: Users know why model makes predictions

### Technical
1. **New Module**: `scripts/explainability/shap_explainer.py` (~300 lines)
2. **Modified**: `scripts/mainfunction.py` (integrate SHAP)
3. **New Algorithm**: `processing/explain_model_algorithm.py` (~200 lines)
4. **Modified**: `ui/dzetsaka_dockwidget.py` (add checkbox)
5. **Tests**: Unit and integration tests
6. **Docs**: Usage guide and examples

## Success Criteria

✅ SHAP explainer module works for all 11 algorithms
✅ UI checkbox generates feature importance raster
✅ Processing algorithm works in batch mode
✅ Performance acceptable (< 2x training time overhead)
✅ Output rasters visualizable in QGIS
✅ Documentation complete
✅ Tests pass (>70% coverage)

## Estimated Timeline

**Day 1-2**: SHAP explainer module
**Day 3-4**: Integration with mainfunction.py
**Day 5-6**: Processing algorithm
**Day 7-8**: UI integration
**Day 9-10**: Testing, documentation, polish

## Known Challenges

1. **Memory Usage**: SHAP can be memory-intensive for large samples
   - Solution: Use sample_size parameter (default: 1000 pixels)

2. **Computation Time**: KernelExplainer slow for non-tree models
   - Solution: Use TreeExplainer when possible, warn for slow models

3. **Multiclass Models**: SHAP returns values per class
   - Solution: Aggregate across classes (mean absolute SHAP)

4. **Background Data**: KernelExplainer needs background dataset
   - Solution: Sample from training data, store in model file

## Reference Implementation

See `PHASE1_SUMMARY.md` for Phase 1 architecture that Phase 2 builds upon:
- Custom exceptions (`domain/exceptions.py`)
- Factory pattern (`factories/classifier_factory.py`)
- Optuna optimization (`scripts/optimization/optuna_optimizer.py`)

## Starter Prompt for New Chat

Copy this to start Phase 2:

```
I want to implement Phase 2 of the dzetsaka enhancement plan: SHAP & Explainability.

Phase 1 (Optuna optimization) is complete - see PHASE1_SUMMARY.md for details.

Please read PHASE2_KICKOFF.md for Phase 2 objectives and implementation details.

Let's start with creating the SHAP explainer module (scripts/explainability/shap_explainer.py).
```

---

**Ready for Phase 2!** Start a new chat, use the prompt above, and let's build SHAP explainability! 🚀
