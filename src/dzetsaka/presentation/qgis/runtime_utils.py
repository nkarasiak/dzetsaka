"""Small runtime utilities for QGIS presentation layer."""

from __future__ import annotations

import importlib


def is_module_importable(module_name: str) -> bool:
    """Return True if a module can be imported in current runtime."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def write_plugin_config(config_path: str, config_obj, section: str, option: str, value: str) -> None:
    """Write one configuration key/value to the plugin config file."""
    with open(config_path, "w") as config_file:
        config_obj.set(section, option, value)
        config_obj.write(config_file)

