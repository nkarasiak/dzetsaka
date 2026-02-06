"""QGIS presentation-layer plugin entrypoint."""

from __future__ import annotations

from pathlib import Path

from dzetsaka.services.runtime_loader import load_module_from_path


def _load_runtime_class():
    """Load DzetsakaGUI runtime class from the local src module."""
    runtime_path = Path(__file__).resolve().parent / "plugin_runtime.py"
    try:
        module = load_module_from_path("_dzetsaka_plugin_runtime", runtime_path)
    except Exception:
        return None
    return getattr(module, "DzetsakaGUI", None)


def class_factory(iface):
    """Factory used by the root plugin `classFactory` bridge."""
    runtime_cls = _load_runtime_class()
    if runtime_cls is not None:
        return runtime_cls(iface)

    from dzetsaka.dzetsaka import DzetsakaGUI

    return DzetsakaGUI(iface)


def classFactory(iface):  # pylint: disable=invalid-name
    """QGIS-compatible factory alias."""
    return class_factory(iface)
