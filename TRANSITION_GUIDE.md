# Phase Transition Guide

This guide explains how to transition between implementation phases in the dzetsaka enhancement project.

## Current Status

**✅ Phase 1 Complete**: Speed & Foundation (v4.3.0)
- Optuna optimization
- Custom exceptions
- Classifier factory
- See `PHASE1_SUMMARY.md` for details

**⏭️ Next**: Phase 2 - SHAP & Explainability (v4.4.0)
- See `PHASE2_KICKOFF.md` for objectives

## How to Transition to Phase 2

### Step 1: Commit Phase 1

Run the commit script:

**Windows**:
```bash
commit_phase1.bat
```

**Linux/Mac**:
```bash
chmod +x commit_phase1.sh
./commit_phase1.sh
```

**Or manually**:
```bash
git add .
git commit -m "feat: Phase 1 complete - Optuna optimization (v4.3.0)"
```

### Step 2: Start New Chat

1. Open Claude Code
2. Start a **new chat** (fresh context)
3. Copy and paste this prompt:

```
I want to implement Phase 2 of the dzetsaka enhancement plan: SHAP & Explainability.

Phase 1 (Optuna optimization) is complete - see PHASE1_SUMMARY.md for details.

Please read PHASE2_KICKOFF.md for Phase 2 objectives and implementation details.

Let's start with creating the SHAP explainer module (scripts/explainability/shap_explainer.py).
```

### Step 3: Validate and Continue

The new chat will:
1. Read `PHASE1_SUMMARY.md` to understand what's been done
2. Read `PHASE2_KICKOFF.md` for Phase 2 objectives
3. Start implementing SHAP explainability
4. Ask for your validation at key checkpoints

## Why This Approach?

**Benefits of Phase-by-Phase Development**:

✅ **Clean Context** - Fresh chat for each phase (no token limits)
✅ **Focused Work** - Each phase has clear objectives
✅ **Self-Documenting** - Each phase documents what it did
✅ **Easy Handoff** - New chat reads previous phase docs
✅ **Modular Development** - Phases can be reviewed independently
✅ **Version Control** - Each phase is a clean commit

**Inspired by Plan Mode**:
- Plan mode creates separate sessions for planning vs. implementation
- Similarly, we create separate sessions for each implementation phase
- Each phase is self-contained with clear inputs/outputs

## File Structure

```
dzetsaka/
├── PHASE1_SUMMARY.md          # Phase 1 completion summary
├── PHASE2_KICKOFF.md           # Phase 2 objectives and plan
├── TRANSITION_GUIDE.md         # This file
├── commit_phase1.bat           # Windows commit script
├── commit_phase1.sh            # Linux/Mac commit script
├── scripts/
│   ├── optimization/           # Phase 1: Optuna
│   └── explainability/         # Phase 2: SHAP (to be created)
├── domain/
│   └── exceptions.py           # Phase 1: Exceptions
├── factories/
│   └── classifier_factory.py   # Phase 1: Factory
└── ...
```

## Phase Timeline

| Phase | Weeks | Status | Version | Focus |
|-------|-------|--------|---------|-------|
| **1** | 1-3 | ✅ Complete | 4.3.0 | Speed & Foundation |
| **2** | 4-5 | 🔄 Next | 4.4.0 | SHAP & Explainability |
| **3** | 6-7 | ⏳ Pending | 4.5.0 | Class Imbalance & Nested CV |
| **4** | 8-10 | ⏳ Pending | 5.0.0 | Wizard UI |
| **5** | 11-12 | ⏳ Pending | 5.0.0 | Polish & Documentation |

## What Each Phase Delivers

### Phase 1: Speed & Foundation ✅
- Optuna optimization (2-10x faster)
- Custom exception hierarchy
- Classifier factory pattern
- Clean architecture foundation

### Phase 2: SHAP & Explainability 🔄
- SHAP feature importance
- UI checkbox for importance maps
- Processing algorithm for batch SHAP
- Model interpretability

### Phase 3: Class Imbalance & Nested CV ⏳
- Automatic class weighting
- SMOTE oversampling
- Nested cross-validation
- Unbiased accuracy estimates

### Phase 4: Wizard UI ⏳
- Step-by-step guided workflow
- Real-time validation
- Pre-execution checklist
- Modern user experience

### Phase 5: Polish & Documentation ⏳
- Comprehensive test suite (>80% coverage)
- Performance optimization
- Complete documentation
- Video tutorials

## Validation Checkpoints

At the end of each phase, validate:

1. **Code Quality**: All files pass `ruff check`
2. **Functionality**: Features work as expected
3. **Tests**: Unit tests pass
4. **Documentation**: README and CHANGELOG updated
5. **Backward Compatibility**: Old code still works
6. **Performance**: Benchmarks meet targets

## Need Help?

If you encounter issues:

1. **Check Phase Summary**: Read `PHASE{N}_SUMMARY.md` for context
2. **Check Kickoff Doc**: Read `PHASE{N}_KICKOFF.md` for plan
3. **Check Git Log**: See commit history for changes
4. **Start Fresh Chat**: Describe issue and reference phase docs

## Best Practices

✅ **Do**:
- Commit after each phase
- Start new chat for each phase
- Read previous phase docs
- Validate before moving on
- Document as you go

❌ **Don't**:
- Skip validation checkpoints
- Mix multiple phases in one chat
- Forget to commit work
- Rush through phases
- Skip documentation

---

**Ready for Phase 2?** Run `commit_phase1.bat` and start a new chat! 🚀
