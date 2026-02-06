"""QGIS presentation adapters for dzetsaka."""

from .plugin import class_factory
from .provider import get_provider_class

__all__ = ["class_factory", "get_provider_class"]
