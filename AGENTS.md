# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python QGIS plugin with most runtime code at the repository root and in feature folders.
- Core entry points: `dzetsaka.py`, `dzetsaka_provider.py`
- Processing and ML workflows: `processing/`, `scripts/`, `services/`, `factories/`, `domain/`
- UI components: `ui/` (`.ui` designer files and generated Python wrappers)
- Configuration and metadata: `pyproject.toml`, `metadata.txt`, `config/`
- Tests: `tests/unit/` and `tests/integration/`
- Documentation and assets: `docs/`, `img/`

## Build, Test, and Development Commands
Use the Makefile targets when possible:
- `make install-dev` installs editable package plus dev/test/docs dependencies.
- `make lint` runs Ruff checks.
- `make format` runs Ruff formatter and autofixes lint issues.
- `make typecheck` runs MyPy.
- `make test` runs the full pytest suite.
- `make quick-test` runs tests excluding QGIS-marked tests.
- `make docs` builds Sphinx docs from `docs/`.
- `make plugin-package` builds `dzetsaka.zip` for QGIS plugin distribution.

## Coding Style & Naming Conventions
- Python: 4-space indentation, max line length 120.
- Formatting/linting: Ruff is primary (`ruff format`, `ruff check`); Black is configured as fallback.
- Type checking: MyPy with strict settings (missing imports ignored for QGIS/ML deps).
- Naming: prefer `snake_case` for modules/functions/tests; keep existing QGIS-compatible naming where already used.
- Do not manually edit generated files unless required: `resources.py`, generated files in `ui/`.

## Testing Guidelines
- Framework: `pytest` with markers (`unit`, `integration`, `qgis`, `slow`, etc.).
- Naming: `test_*.py` files and `test_*` functions.
- Run targeted tests with markers, e.g. `pytest -m "not qgis"` or `pytest tests/unit/`.
- Coverage is enabled by default via pytest config (`--cov=dzetsaka`, HTML/XML reports).

## Commit & Pull Request Guidelines
- Prefer Conventional Commit style seen in history: `feat: ...`, `fix: ...` (imperative, concise).
- Keep commits focused and logically grouped; avoid mixing refactor and feature changes.
- PRs should include:
  - Clear summary of behavior changes
  - Linked issue(s) when applicable
  - Test evidence (`make test`, marker-based runs, or targeted command output)
  - UI screenshots/GIFs for changes under `ui/`
