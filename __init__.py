"""QGIS plugin initialization entrypoint for dzetsaka.

This module intentionally stays thin and delegates plugin loading to the new
architecture entrypoint when present, while preserving a fallback to the legacy
`dzetsaka.py` implementation for safe incremental migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .services.runtime_loader import load_module_from_path

# Make `src/dzetsaka` visible as part of the plugin package so imports like
# `dzetsaka.presentation...` resolve in QGIS plugin runtime.
_ROOT_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE_DIR = _ROOT_DIR / "src" / "dzetsaka"
if _SRC_PACKAGE_DIR.exists():
    if str(_SRC_PACKAGE_DIR) not in __path__:
        __path__.append(str(_SRC_PACKAGE_DIR))
    src_parent = str(_SRC_PACKAGE_DIR.parent)
    if src_parent not in sys.path:
        sys.path.insert(0, src_parent)


def _load_new_entrypoint():
    """Load the new entrypoint module from `src/` if available."""
    entrypoint = _SRC_PACKAGE_DIR / "presentation" / "qgis" / "plugin.py"
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
