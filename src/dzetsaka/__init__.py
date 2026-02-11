"""dzetsaka source package.

This package can be imported directly by QGIS plugin loader depending on
``sys.path`` ordering.  Expose a ``classFactory`` bridge here so plugin load
works whether QGIS resolves ``dzetsaka`` to repository root package or this
``src/dzetsaka`` package.
"""

from __future__ import annotations

import sys
from pathlib import Path


# When QGIS resolves the plugin package from `plugins/dzetsaka/src`, expose the
# plugin root as part of this package so submodules located at repository root
# remain importable (e.g. `dzetsaka.ui`, `dzetsaka.services`, `dzetsaka.dzetsaka`).
_SRC_PKG_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _SRC_PKG_DIR.parent.parent
if _PLUGIN_ROOT.exists():
    if str(_PLUGIN_ROOT) not in __path__:
        __path__.append(str(_PLUGIN_ROOT))
    if str(_PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(_PLUGIN_ROOT))


def classFactory(iface):  # pylint: disable=invalid-name
    """QGIS plugin factory bridge for direct ``src/dzetsaka`` imports."""
    from .qgis.plugin import classFactory as _factory

    return _factory(iface)

