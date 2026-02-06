"""Architecture guardrail tests for the ongoing refactor."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = REPO_ROOT / "src" / "dzetsaka" / "domain"
PRESENTATION_QGIS_ROOT = REPO_ROOT / "src" / "dzetsaka" / "presentation" / "qgis"

FORBIDDEN_ROOT_IMPORTS = {"qgis", "osgeo", "PyQt6", "PyQt5", "PySide6", "PySide2"}
EXPECTED_PROCESSING_SHIMS = {
    "__init__.py",
    "classify.py",
    "explain_model.py",
    "nested_cv_algorithm.py",
    "split_train_validation.py",
    "train.py",
}
REMOVED_EXPERIMENTAL_PROCESSING = {
    "closing_filter.py",
    "domain_adaptation.py",
    "learn_with_spatial_sampling.py",
    "learn_with_stand_cv.py",
    "median_filter.py",
    "resample_image_same_date.py",
    "shannon_entropy.py",
    "sieve_area.py",
}


def _import_roots_from_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
            elif node.level and node.level > 0:
                roots.add(".")
    return roots


def test_domain_layer_has_no_framework_imports() -> None:
    """Domain layer must remain framework-independent."""
    assert DOMAIN_ROOT.exists(), "Expected domain layer scaffold at src/dzetsaka/domain"

    py_files = [p for p in DOMAIN_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    violations: list[str] = []

    for file_path in py_files:
        roots = _import_roots_from_file(file_path)
        forbidden = sorted(roots.intersection(FORBIDDEN_ROOT_IMPORTS))
        if forbidden:
            violations.append(f"{file_path.relative_to(REPO_ROOT)} imports {', '.join(forbidden)}")

    assert not violations, "Domain framework coupling detected:\n" + "\n".join(violations)


def test_compat_shim_has_no_eager_qgis_import() -> None:
    """Root shim must stay importable outside QGIS runtime."""
    shim_path = REPO_ROOT / "dzetsaka.py"
    roots = _import_roots_from_file(shim_path)
    assert "qgis" not in roots, f"{shim_path.relative_to(REPO_ROOT)} imports qgis eagerly"


def test_provider_shim_has_no_eager_qgis_import() -> None:
    """Provider shim must stay importable outside QGIS runtime."""
    shim_path = REPO_ROOT / "dzetsaka_provider.py"
    roots = _import_roots_from_file(shim_path)
    assert "qgis" not in roots, f"{shim_path.relative_to(REPO_ROOT)} imports qgis eagerly"


def test_presentation_helpers_do_not_import_pipeline_monolith() -> None:
    """Presentation helper modules must not couple to the scripts pipeline monolith."""
    assert PRESENTATION_QGIS_ROOT.exists(), "Expected presentation/qgis layer scaffold"

    helper_files = [
        p
        for p in PRESENTATION_QGIS_ROOT.glob("*.py")
        if p.name not in {"plugin_runtime.py", "__init__.py", "plugin.py", "provider.py"}
    ]
    violations: list[str] = []
    for file_path in helper_files:
        roots = _import_roots_from_file(file_path)
        text = file_path.read_text(encoding="utf-8")
        if "scripts.classification_pipeline" in text or "from dzetsaka.scripts import classification_pipeline" in text:
            violations.append(str(file_path.relative_to(REPO_ROOT)))
        if "scripts" in roots and "classification_pipeline" in text:
            violations.append(str(file_path.relative_to(REPO_ROOT)))

    assert not violations, "Presentation helper leaked scripts pipeline dependency:\n" + "\n".join(violations)


def test_use_case_bridge_module_removed() -> None:
    """Refactor completion: bridge module should remain removed."""
    bridge_path = REPO_ROOT / "services" / "use_case_bridge.py"
    assert not bridge_path.exists(), "Unexpected legacy bridge module found at services/use_case_bridge.py"


def test_processing_root_contains_only_required_compat_shims() -> None:
    """Root processing package should keep only compatibility modules still needed for import stability."""
    processing_root = REPO_ROOT / "processing"
    assert processing_root.exists(), "Expected root processing compatibility package"

    py_files = {path.name for path in processing_root.glob("*.py")}
    assert py_files == EXPECTED_PROCESSING_SHIMS, f"Unexpected processing root files: {sorted(py_files)}"


def test_removed_experimental_processing_modules_stay_deleted() -> None:
    """Dropped experimental processing modules must not reappear."""
    processing_root = REPO_ROOT / "processing"
    for filename in sorted(REMOVED_EXPERIMENTAL_PROCESSING):
        assert not (processing_root / filename).exists(), f"Deprecated processing module restored: {filename}"


def test_packaging_flow_is_canonicalized() -> None:
    """Packaging should be driven by tools/build_plugin.py with root zip script as compatibility only."""
    build_script = REPO_ROOT / "tools" / "build_plugin.py"
    makefile = REPO_ROOT / "Makefile"
    zip_wrapper = REPO_ROOT / "zip_file.py"

    assert build_script.exists(), "Missing canonical packaging script tools/build_plugin.py"
    makefile_text = makefile.read_text(encoding="utf-8")
    assert "python tools/build_plugin.py --output dzetsaka.zip" in makefile_text

    wrapper_text = zip_wrapper.read_text(encoding="utf-8")
    assert "from tools.build_plugin import build_plugin_zip" in wrapper_text
