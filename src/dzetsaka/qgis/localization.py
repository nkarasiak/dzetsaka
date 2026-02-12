"""Localization helpers for QGIS presentation layer."""

from __future__ import annotations

from qgis.PyQt.QtCore import QCoreApplication


def tr(message: str) -> str:
    """Translate a message string in dzetsaka context."""
    return QCoreApplication.translate("dzetsaka", message)
