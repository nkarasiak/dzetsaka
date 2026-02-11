# Validated Widgets Integration Checklist

This checklist guides the integration of validated widgets into the dzetsaka guided workflow UI.

## Phase 1: Core Integration (High Priority)

### Optimization Panel (`QuickClassificationPanel`)

- [ ] **Optuna Trials** (`self.optunaTrials`)
  - Current: `QSpinBox` (range 10-1000)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 10 <= v <= 2000`
  - Warning threshold: `500`
  - Time estimator: `lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"`
  - Location: `~line 2640` in `classification_workflow_ui.py`

- [ ] **SHAP Sample Size** (`self.shapSampleSize`)
  - Current: `QSpinBox` (range 100-50000)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 100 <= v <= 50000`
  - Warning threshold: `10000`
  - Time estimator: `lambda v: f"{v * 0.01:.0f}-{v * 0.05:.0f} sec"`
  - Location: `~line 2753` in `classification_workflow_ui.py`

- [ ] **SMOTE k-neighbors** (`self.smoteK`)
  - Current: `QSpinBox` (range 1-20)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 1 <= v <= 20`
  - Warning threshold: `15`
  - Time estimator: None
  - Location: `~line 2676` in `classification_workflow_ui.py`

- [ ] **Nested CV - Inner Folds** (`self.innerFolds`)
  - Current: `QSpinBox` (range 2-10)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 2 <= v <= 10`
  - Warning threshold: `7`
  - Time estimator: None
  - Location: `~line 2795` in `classification_workflow_ui.py`

- [ ] **Nested CV - Outer Folds** (`self.outerFolds`)
  - Current: `QSpinBox` (range 2-10)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 2 <= v <= 10`
  - Warning threshold: `7`
  - Time estimator: None
  - Location: `~line 2809` in `classification_workflow_ui.py`

- [ ] **Validation Split** (`self.splitSpinBox`)
  - Current: `QSpinBox` (range 10-90)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 10 <= v <= 90`
  - Warning threshold: `80` (too little validation data)
  - Time estimator: None
  - Location: `~line 3035` in `classification_workflow_ui.py`

## Phase 2: Advanced Method Panel

### Recipe Shop Panel (`RecipeShopPanel`)

- [ ] **Optuna Trials** (`self.optunaTrialsSpin`)
  - Current: `QSpinBox` (range 10-2000)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 10 <= v <= 2000`
  - Warning threshold: `500`
  - Time estimator: `lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"`
  - Location: `~line 1034` in `classification_workflow_ui.py`

- [ ] **SHAP Sample** (`self.shapSampleSpin`)
  - Current: `QSpinBox` (range 100-50000)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 100 <= v <= 50000`
  - Warning threshold: `10000`
  - Time estimator: `lambda v: f"{v * 0.01:.0f}-{v * 0.05:.0f} sec"`
  - Location: `~line 1043` in `classification_workflow_ui.py`

- [ ] **SMOTE k-neighbors** (`self.smoteKSpin`)
  - Current: `QSpinBox` (range 2-30)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 2 <= v <= 30`
  - Warning threshold: `20`
  - Time estimator: None
  - Location: `~line 1052` in `classification_workflow_ui.py`

- [ ] **Nested CV - Inner** (`self.nestedInnerSpin`)
  - Current: `QSpinBox` (range 2-20)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 2 <= v <= 20`
  - Warning threshold: `10`
  - Time estimator: None
  - Location: `~line 1071` in `classification_workflow_ui.py`

- [ ] **Nested CV - Outer** (`self.nestedOuterSpin`)
  - Current: `QSpinBox` (range 2-20)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 2 <= v <= 20`
  - Warning threshold: `10`
  - Time estimator: None
  - Location: `~line 1073` in `classification_workflow_ui.py`

- [ ] **Train Split %** (`self.splitSpin`)
  - Current: `QSpinBox` (range 10-100)
  - Replace with: `ValidatedSpinBox`
  - Validator: `lambda v: 10 <= v <= 100`
  - Warning threshold: `90` (too much training data)
  - Time estimator: None
  - Location: `~line 1113` in `classification_workflow_ui.py`

## Phase 3: Algorithm-Specific Parameters (Future)

Consider adding validated widgets for algorithm-specific parameters:

- [ ] **Random Forest - n_estimators**
  - Validator: `lambda v: 10 <= v <= 1000`
  - Warning threshold: `500`

- [ ] **Random Forest - max_depth**
  - Validator: `lambda v: v > 0` (or None for unlimited)
  - Warning threshold: `50`

- [ ] **SVM - C parameter**
  - Use `ValidatedDoubleSpinBox`
  - Validator: `lambda v: v > 0`
  - Warning threshold: `100`

- [ ] **XGBoost - n_estimators**
  - Validator: `lambda v: 10 <= v <= 2000`
  - Warning threshold: `1000`

## Implementation Steps

### For Each Widget

1. **Import the validated widget class**
   ```python
   from ui.validated_widgets import ValidatedSpinBox
   ```

2. **Replace widget initialization**

   Before:
   ```python
   self.optunaTrials = QSpinBox()
   self.optunaTrials.setRange(10, 1000)
   self.optunaTrials.setValue(100)
   ```

   After:
   ```python
   self.optunaTrials = ValidatedSpinBox(
       validator_fn=lambda v: 10 <= v <= 2000,
       warning_threshold=500,
       time_estimator_fn=lambda v: f"{v * 0.1:.0f}-{v * 0.3:.0f} min"
   )
   self.optunaTrials.setRange(10, 2000)
   self.optunaTrials.setValue(100)
   ```

3. **Add or update tooltip**
   ```python
   self.optunaTrials.setToolTip("Number of Optuna optimization trials")
   ```

4. **Test all states**
   - Valid state (default styling)
   - Warning state (orange border + tooltip)
   - Invalid state (red border + tooltip)

5. **Verify signal connections still work**
   - `valueChanged` connections
   - `toggled` connections from checkboxes
   - `setEnabled()` calls

## Testing Strategy

### Manual Testing

For each integrated widget:

1. **Valid state**
   - Set value in normal range
   - Verify default border
   - Check tooltip shows original text + time estimate

2. **Warning state**
   - Set value at/above warning threshold
   - Verify orange border
   - Check tooltip shows warning message

3. **Invalid state** (if validator allows)
   - Type invalid value manually (if possible)
   - Verify red border
   - Check tooltip shows "Invalid value"

4. **Integration**
   - Verify checkbox enable/disable works
   - Verify value is read correctly in `_emit_config()`
   - Test recipe save/load preserves values

### Automated Testing

- [ ] Add UI tests for validated widgets in `tests/unit/test_guided_workflow.py`
- [ ] Test that existing signal connections still work
- [ ] Test that recipe serialization/deserialization works
- [ ] Test that values are correctly passed to classification pipeline

## Rollout Plan

### Stage 1: Single Widget Pilot
- Integrate one widget (suggest `self.optunaTrials`)
- Test thoroughly
- Gather user feedback
- Fix any issues

### Stage 2: Core Widgets
- Integrate all Phase 1 widgets
- Test interactions between widgets
- Update documentation

### Stage 3: Recipe Shop
- Integrate Phase 2 widgets
- Ensure consistency with Phase 1
- Test recipe save/load

### Stage 4: Algorithm Parameters (Future)
- Design parameter configuration UI
- Integrate Phase 3 widgets
- Consider advanced validation (parameter dependencies)

## Compatibility Notes

### Qt Version Compatibility
- Widgets use `qgis.PyQt` imports for PyQt5/PyQt6 compatibility
- Test on both Qt5 and Qt6 if possible

### QGIS Version Compatibility
- Minimum QGIS version: 3.0 (current dzetsaka requirement)
- Test on QGIS LTR and latest stable

### Backward Compatibility
- Validated widgets are drop-in replacements for standard Qt widgets
- No breaking changes to existing API
- Existing signal/slot connections remain compatible

## Documentation Updates

After integration:

- [ ] Update `CLAUDE.md` with validated widgets usage
- [ ] Add screenshots to `docs/validated_widgets_usage.md`
- [ ] Update user-facing documentation/help
- [ ] Add examples to QGIS plugin help dialog

## Success Criteria

- [ ] All targeted widgets replaced successfully
- [ ] No regressions in existing functionality
- [ ] Validation provides helpful feedback to users
- [ ] Time estimates are reasonably accurate
- [ ] Recipe save/load works correctly
- [ ] All tests pass
- [ ] Code review approved
- [ ] User acceptance testing completed

## Known Limitations

1. **Real-time validation overhead**
   - Validators run on every value change
   - Keep validator functions lightweight

2. **Tooltip accessibility**
   - Consider adding aria-labels for screen readers
   - Color alone should not convey information

3. **Time estimates are approximate**
   - Actual runtime depends on hardware, data size, etc.
   - Consider adding calibration mechanism

## Future Enhancements

- [ ] Add validation history visualization
- [ ] Implement progressive validation (only warn after user stops typing)
- [ ] Add "why?" buttons to explain validation rules
- [ ] Integrate with Qt's built-in validator system
- [ ] Add animation for state transitions
- [ ] Support custom color schemes for themes

