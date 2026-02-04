# Phase 4 & 5 Kickoff: Wizard UI + Polish

## Phases Complete
- ✅ Phase 1 (v4.3.0): Optuna optimization — see PHASE1_SUMMARY.md
- ✅ Phase 2 (v4.4.0): SHAP explainability — see PHASE2_SUMMARY.md
- ✅ Phase 3 (v4.5.0): Class imbalance & nested CV — see PHASE3_SUMMARY.md

## Current Version: 4.5.0

## Architecture at a Glance
```
scripts/
├── optimization/       # Phase 1: Optuna optimizer
├── explainability/     # Phase 2: SHAP (ModelExplainer)
├── sampling/           # Phase 3: SMOTE + class_weights
├── validation/         # Phase 3: nested_cv + metrics
└── mainfunction.py     # LearnModel / ClassifyImage (central)

processing/
├── train.py            # Train algorithm
├── classify.py         # Classify algorithm
├── explain_model.py    # Phase 2: SHAP
└── nested_cv_algorithm.py  # Phase 3: Nested CV

domain/exceptions.py    # Phase 1: Custom exceptions
factories/classifier_factory.py  # Phase 1: Factory pattern
```

## extraParam keys (all phases)
```python
# Phase 1
"USE_OPTUNA", "OPTUNA_TRIALS"
# Phase 2
"COMPUTE_SHAP", "SHAP_OUTPUT", "SHAP_SAMPLE_SIZE"
# Phase 3
"USE_SMOTE", "SMOTE_K_NEIGHBORS",
"USE_CLASS_WEIGHTS", "CLASS_WEIGHT_STRATEGY", "CUSTOM_CLASS_WEIGHTS",
"USE_NESTED_CV", "NESTED_INNER_CV", "NESTED_OUTER_CV"
```

## Phase 4 Targets (v4.6.0) — Wizard UI
1. Step-by-step classification wizard in the dock widget
2. Real-time parameter validation
3. Interactive model comparison panel
4. One-click "smart defaults" that auto-select Optuna/SMOTE/SHAP

## Phase 5 Targets (v5.0.0) — Polish & Testing
1. Comprehensive test suite (>80% coverage)
2. Performance profiling & benchmarks
3. User documentation & tutorials
4. Final release preparation

## Key Files to Read Before Starting
- `CLAUDE.md` — project conventions & commands
- `scripts/mainfunction.py` — central training/classification engine
- `ui/dzetsaka_dockwidget.py` — current dock widget (where wizard goes)
- `classifier_config.py` — all 11 algorithm definitions
- `pyproject.toml` — build config, linting rules, dependencies
