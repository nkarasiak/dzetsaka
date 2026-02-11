from __future__ import annotations

from pathlib import Path

__all__ = []

# Allow legacy imports under dzetsaka.presentation.qgis to resolve to src/dzetsaka/qgis
current_dir = Path(__file__).resolve().parent
qgis_package = current_dir.parent.parent / "qgis"
if str(qgis_package) not in __path__:
    __path__.append(str(qgis_package))
