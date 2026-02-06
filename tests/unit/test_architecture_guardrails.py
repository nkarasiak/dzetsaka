"""Architecture guardrail tests for the ongoing refactor."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = REPO_ROOT / "src" / "dzetsaka" / "domain"
PRESENTATION_QGIS_ROOT = REPO_ROOT / "src" / "dzetsaka" / "presentation" / "qgis"

FORBIDDEN_ROOT_IMPORTS = {"qgis", "osgeo", "PyQt6", "PyQt5", "PySide6", "PySide2"}


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


def test_presentation_helpers_do_not_import_legacy_mainfunction() -> None:
    """New presentation helper modules should not couple back to legacy monolith."""
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
        if "scripts.mainfunction" in text or "from dzetsaka.scripts import mainfunction" in text:
            violations.append(str(file_path.relative_to(REPO_ROOT)))
        if "scripts" in roots and "mainfunction" in text:
            violations.append(str(file_path.relative_to(REPO_ROOT)))

    assert not violations, "Presentation helper leaked legacy mainfunction dependency:\n" + "\n".join(violations)
