"""QGIS plugin initialization entrypoint for dzetsaka.

This module intentionally stays thin and delegates plugin loading to the new
architecture entrypoint when present, while preserving a fallback to the legacy
`dzetsaka.py` implementation for safe incremental migration.
"""

from __future__ import annotations

from pathlib import Path

from .services.runtime_loader import load_module_from_path


def _load_new_entrypoint():
    """Load the new entrypoint module from `src/` if available."""
    root_dir = Path(__file__).resolve().parent
    entrypoint = root_dir / "src" / "dzetsaka" / "presentation" / "qgis" / "plugin.py"
    if not entrypoint.exists():
        return None

    try:
        return load_module_from_path("_dzetsaka_new_entrypoint", entrypoint)
    except Exception:
        return None


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """QGIS-required plugin factory function."""
    module = _load_new_entrypoint()
    if module is not None and hasattr(module, "class_factory"):
        return module.class_factory(iface)

    from .dzetsaka import DzetsakaGUI

    return DzetsakaGUI(iface)
