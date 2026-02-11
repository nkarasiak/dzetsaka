# Runtime split & CLI

dzetsaka now ships with two coexisting runtimes:

1. `dzetsaka.qgis`: the QGIS plugin integration that imports `qgis.core`, `qgis.PyQt`, and the processing UI. It lives under `src/dzetsaka/qgis` and exposes `DzetsakaGUI`, `DzetsakaProvider`, processing algorithms, dashboards, etc. The legacy path `dzetsaka.presentation.qgis` is a thin shim that points to the new package so existing code continues to work.
2. `dzetsaka`: the core package that contains the ML application layers (`application/use_cases/`, `infrastructure/`), shared configuration helpers, and a CLI surface. It deliberately avoids importing QGIS, instead relying on `dzetsaka.logging` and feedback adapters that can be swapped (Python logger for CLI, QGIS logger for the plugin). 

Both runtimes share:

- `dzetsaka.logging`: exposes `Logger`, `Reporter`, issue popup helpers, and registration hooks for wiring in either the CLI or QGIS-level logging backend.
- `application/use_cases` + `infrastructure`: keep training/classification/SHAP logic centralized, so changes are reflected wherever the ML logic runs.
- `scripts/classification_pipeline.py`: still the canonical implementation of the classifiers; both runtimes import from it indirectly (via `infrastructure/ml/ml_pipeline_adapter.py`) so dependency installation, SHAP, Optuna, and heuristic helpers stay in sync.

### CLI entry point

Declared via `[project.scripts]` in `pyproject.toml`, `dzetsaka.cli:main` exposes:

```
dzetsaka classify  --raster input.tif --model model.pkl --output map.tif [--mask mask.tif] [--confidence conf.tif]
dzetsaka train     --raster train.tif --vector train.shp --model model.pkl [--split-config 80] [--classifier RF] [--extra '{"USE_OPTUNA": true}']
```

- `--extra` accepts raw JSON or `@path/to/file.json` to reuse the same extra flags (Optuna, SHAP, SMOTE, etc.) that `scripts/classification_pipeline` understands.
- `_CLIProgress` prints progress to stdout and implements `setProgress`/`setProgressText` so the existing feedback hooks work.
- Exceptions are routed through `dzetsaka.logging.show_error_dialog` so downstream helpers (and the CLI test harness) can fail fast with meaningful info.

### Packaging notes

- `pyproject.toml` still makes setuptools find `dzetsaka*`, so both `dzetsaka` and `dzetsaka.qgis` are packaged.
- The plugin’s compatibility layer (`dzetsaka.py`, `dzetsaka_provider.py`) now point to `src/dzetsaka/qgis`.
- The CLI runner is installed alongside the wheel, so users installing the package once can switch between CLI batches and opening the plugin in QGIS.

For more background, refer to `docs/runtime_split.md`; to run the CLI, see `readme.md`.
