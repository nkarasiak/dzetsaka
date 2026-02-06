# Dzetsaka Architecture Refactor Blueprint (2026)

## Purpose

This document defines the target architecture for a major refactor of the dzetsaka QGIS plugin.
It is the reference for planning, implementation, and review of all refactor-related pull requests.

Goals:

- Modernize project structure using 2025 Python best practices.
- Keep QGIS plugin compatibility while improving maintainability.
- Isolate domain logic from QGIS/GDAL/UI concerns.
- Reduce legacy technical debt (root-level modules, monolithic files, naming drift).

## Current Pain Points

- Mixed concerns at repository root (`dzetsaka.py`, provider, resources, utilities).
- Legacy naming and monolithic modules (example: `scripts/mainfunction.py`).
- UI/generated files mixed with runtime logic.
- Weak architectural boundaries between domain, processing, UI, and infrastructure.
- Hard-to-test flows due to heavy coupling with QGIS runtime classes.

## Target Architecture

```text
dzetsaka/
  pyproject.toml
  README.md
  CHANGELOG.md
  Makefile
  src/
    dzetsaka/
      __init__.py
      application/
        dto/
        ports/
        use_cases/
          train_model.py
          classify_raster.py
          explain_model.py
      domain/
        entities/
        value_objects/
        services/
        exceptions.py
      infrastructure/
        ml/
          sklearn/
          xgboost/
          lightgbm/
          catboost/
        geo/
          gdal/
          qgis/
        persistence/
      presentation/
        qgis/
          plugin.py
          provider.py
          processing/
            algorithms/
          ui/
            docks/
            dialogs/
            viewmodels/
      assets/
        ui/
        qrc/
        icons/
      _generated/
        ui/
        resources.py
  plugin/
    __init__.py
    metadata.txt
    icon.png
  tests/
    unit/
      domain/
      application/
      infrastructure/
    integration/
      processing/
    qgis/
  tools/
    build_plugin.py
    compile_ui.py
    compile_resources.py
  dist/
```

## Architectural Rules

1. `domain` must not import `qgis`, `osgeo`, or UI modules.
2. `application` orchestrates use cases via interfaces (`ports`), not concrete frameworks.
3. `infrastructure` implements adapters for ML/GDAL/QGIS and external systems.
4. `presentation/qgis` is a thin adapter layer between QGIS events and application use cases.
5. Generated code (`_generated`) is not manually edited.
6. `plugin/__init__.py` only exposes `classFactory`.

## Migration Strategy

## Migration Status

- 2026-02-06: Phase 1 started.
- 2026-02-06: `src/dzetsaka/...` skeleton created.
- 2026-02-06: Root `classFactory` now bridges to `src/dzetsaka/presentation/qgis/plugin.py` with fallback.
- 2026-02-06: Processing provider moved to `src/dzetsaka/presentation/qgis/provider.py` with root compatibility shim.
- 2026-02-06: `ClassificationTask` now calls bridge use cases (`services/use_case_bridge.py`) instead of directly calling `scripts/mainfunction.py`.
- 2026-02-06: Initial use-case wrappers added in `src/dzetsaka/application/use_cases/` (`train_model.py`, `classify_raster.py`).
- 2026-02-06: QGIS plugin runtime duplicated to `src/dzetsaka/presentation/qgis/plugin_runtime.py`; `src/.../plugin.py` now loads this runtime first.
- 2026-02-06: Root `dzetsaka.py` converted to a lazy compatibility shim that proxies runtime classes from `src/.../plugin_runtime.py`.
- 2026-02-06: Added architecture guardrail tests in `tests/unit/test_architecture_guardrails.py` (domain must remain framework-free).
- 2026-02-06: Extracted task orchestration from plugin runtime into `src/dzetsaka/presentation/qgis/task_runner.py` and wired runtime to import it.
- 2026-02-06: Converted `dzetsaka_provider.py` into a lazy shim that proxies to `src/dzetsaka/presentation/qgis/provider.py`.
- 2026-02-06: Refactored `services/use_case_bridge.py` to lazy-load migrated use cases and legacy `mainfunction` fallback at call-time.
- 2026-02-06: Added shared runtime module loader utility (`services/runtime_loader.py`) and reused it across plugin/provider bridges.
- 2026-02-06: Extracted dependency installation flow from plugin runtime to `src/dzetsaka/presentation/qgis/dependency_installer.py`.
- 2026-02-06: Extracted debug information builder from plugin runtime to `src/dzetsaka/presentation/qgis/debug_info.py`.
- 2026-02-06: Extracted classification input validation from plugin runtime to `src/dzetsaka/presentation/qgis/input_validation.py`.
- 2026-02-06: Extracted classifier runtime dependency checks to `src/dzetsaka/presentation/qgis/classifier_runtime.py`.
- 2026-02-06: Extracted background task launch/lifecycle handling to `src/dzetsaka/presentation/qgis/task_launcher.py`.
- 2026-02-06: Extracted output naming and issue popup helpers to `src/dzetsaka/presentation/qgis/output_naming.py` and `src/dzetsaka/presentation/qgis/issue_reporting.py`.
- 2026-02-06: Extracted shared runtime helpers (`is_module_importable`, config write) to `src/dzetsaka/presentation/qgis/runtime_utils.py`.
- 2026-02-06: Extended architecture guardrails to block new `presentation/qgis` helper modules from importing legacy `scripts/mainfunction`.
- 2026-02-06: Extracted wizard-driven execution flow to `src/dzetsaka/presentation/qgis/wizard_execution.py`.
- 2026-02-06: Extracted plugin metadata version read helper to `src/dzetsaka/presentation/qgis/metadata_utils.py`.
- 2026-02-06: Extracted scikit-learn runtime health check to `src/dzetsaka/presentation/qgis/runtime_checks.py`.
- 2026-02-06: Extracted wizard dock lifecycle helpers to `src/dzetsaka/presentation/qgis/wizard_dock.py`.
- 2026-02-06: Extracted small UI helpers (`rememberLastSaveDir`, welcome widget) to `src/dzetsaka/presentation/qgis/ui_utils.py`.
- 2026-02-06: Extracted action/icon helpers to `src/dzetsaka/presentation/qgis/actions_utils.py`.
- 2026-02-06: Extracted translation helper to `src/dzetsaka/presentation/qgis/localization.py`.
- 2026-02-06: Extracted QGIS GUI init wiring (`initGui`) to `src/dzetsaka/presentation/qgis/ui_init.py`.
- 2026-02-06: Extracted plugin unload cleanup routine to `src/dzetsaka/presentation/qgis/unload_utils.py`.
- 2026-02-06: Extracted remaining major `plugin_runtime.py` UI/workflow methods into helper modules (`main_dock_runtime.py`, `file_form_handlers.py`, `settings_handlers.py`, `main_panel_execution.py`, `config_runtime.py`).
- 2026-02-06: Removed `services/use_case_bridge.py`; QGIS task/processing flows now call application use-cases directly.
- 2026-02-06: Migrated active processing algorithms to `src/dzetsaka/presentation/qgis/processing/algorithms/` (`train`, `classify`, `split_train_validation`, `nested_cv_algorithm`, `explain_model`) with root compatibility shims.
- 2026-02-06: Dropped unused experimental processing registration and removed obsolete legacy processing modules (`closing_filter`, `median_filter`, `shannon_entropy`, `domain_adaptation`, `learn_with_spatial_sampling`, `learn_with_stand_cv`, `resample_image_same_date`, `sieve_area`).
- 2026-02-06: Added `tools/build_plugin.py` and wired `make plugin-package` to use it for reproducible plugin zip builds.
- 2026-02-06: Converted root `zip_file.py` into a compatibility wrapper delegating packaging to `tools/build_plugin.py`.
- 2026-02-06: Extracted `DzetsakaGUI` startup/bootstrap wiring from `plugin_runtime.py` to `src/dzetsaka/presentation/qgis/runtime_bootstrap.py` to keep runtime entrypoint thin.
- 2026-02-06: Finalized root `processing/` compatibility scope to algorithm shim modules only; removed stale `processing/metadata_helpers.py` shim.
- 2026-02-06: Finalized runtime/use-case coupling: no `services/use_case_bridge.py` dependency remains in runtime or processing flows.
- 2026-02-06: Added/updated architecture guardrails enforcing removed bridge module, removed experimental processing modules, and canonical packaging flow (`tools/build_plugin.py` + Makefile target).

### Phase 0: Baseline and Guardrails

- Stabilize CI (`ruff`, `mypy`, `pytest`).
- Add/confirm architecture lint rules (import boundaries by convention).
- Freeze behavior with regression tests for core workflows.

Exit criteria:

- Existing behavior validated by tests.
- Refactor branch has reliable CI signal.

### Phase 1: Introduce `src/` and New Entrypoints

- Create `src/dzetsaka` package and mirror minimal runtime entrypoints.
- Move plugin entry logic to `presentation/qgis/plugin.py`.
- Keep compatibility shims in old paths temporarily.

Exit criteria:

- Plugin still loads in QGIS.
- No behavior change for end users.

### Phase 2: Decompose Monoliths

- Split `scripts/mainfunction.py` into focused use cases:
  - training
  - inference
  - evaluation/explainability
- Move raster/vector helpers to `infrastructure/geo`.
- Move classifier integration to `infrastructure/ml`.

Exit criteria:

- Old monolithic path no longer needed in runtime flow.
- Use-case modules have direct unit tests.

### Phase 3: Presentation and Processing Cleanup

- Move processing algorithms under `presentation/qgis/processing/algorithms`.
- Separate UI controllers/viewmodels from widget definitions.
- Keep `.ui` sources in `assets/ui`, generated wrappers in `_generated/ui`.

Exit criteria:

- Clear separation between UI and business logic.
- Processing providers call use cases, not low-level script internals.

### Phase 4: Packaging and Distribution

- Add `tools/build_plugin.py` to assemble final QGIS plugin zip.
- Ensure plugin package contains required metadata/assets/generated files.
- Remove obsolete root-level compatibility modules once migration is complete.

Exit criteria:

- Reproducible `dzetsaka.zip` artifact from automated build.
- Clean root structure with no legacy entrypoint clutter.

## Mapping: Old -> New

- `dzetsaka.py` -> `src/dzetsaka/presentation/qgis/plugin.py`
- `dzetsaka_provider.py` -> `src/dzetsaka/presentation/qgis/provider.py`
- `scripts/mainfunction.py` -> `src/dzetsaka/application/use_cases/*`
- `scripts/function_dataraster.py` -> `src/dzetsaka/infrastructure/geo/*`
- `factories/classifier_factory.py` -> `src/dzetsaka/application/ports` + `src/dzetsaka/infrastructure/ml/*`
- `ui/*.ui` -> `src/dzetsaka/assets/ui/`
- generated `ui/*.py` -> `src/dzetsaka/_generated/ui/`
- `resources.py` -> `src/dzetsaka/_generated/resources.py`

## PR Strategy

Use small, reversible PRs with explicit scope:

1. Skeleton + entrypoints
2. Use-case extraction (train/classify)
3. Infrastructure adapters
4. Processing and UI migration
5. Packaging/build pipeline
6. Cleanup and legacy removal

Each PR must include:

- migration notes
- test evidence
- rollback strategy (if applicable)

## Definition of Done

- Plugin behavior preserved for existing workflows.
- Core logic testable outside QGIS UI runtime.
- Import boundaries respected (`domain` framework-free).
- Packaging produces a valid QGIS plugin artifact.
- Legacy root-level architecture removed or minimized to compatibility stubs.

## Risks and Mitigations

- Risk: break QGIS plugin loading.
  - Mitigation: preserve thin compatibility layer until final phase.
- Risk: regressions in training/classification pipelines.
  - Mitigation: expand integration tests before moving logic.
- Risk: generated file drift.
  - Mitigation: script-based generation in `tools/` and CI check.

## Governance

- This file is the source of truth for refactor direction.
- Any major deviation must be recorded here in a dated "Decision Update" section.
