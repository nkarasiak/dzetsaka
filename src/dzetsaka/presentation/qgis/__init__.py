"""QGIS presentation adapters for dzetsaka."""

from __future__ import annotations

__all__ = ["class_factory", "get_provider_class"]


def class_factory(iface):
    """Lazy wrapper to avoid importing QGIS modules at package import time."""
    from .plugin import class_factory as _class_factory

    return _class_factory(iface)


def get_provider_class():
    """Lazy wrapper to avoid importing QGIS modules at package import time."""
    from .provider import get_provider_class as _get_provider_class

    return _get_provider_class()
