"""Action and icon helpers for QGIS plugin UI wiring."""

from __future__ import annotations

import os

from qgis.PyQt.QtGui import QAction, QIcon


def add_action(
    plugin,
    icon_path,
    text,
    callback,
    enabled_flag=True,
    add_to_menu=True,
    add_to_toolbar=True,
    status_tip=None,
    whats_this=None,
    parent=None,
):
    """Add a toolbar/menu action and register it in plugin actions list."""
    icon = QIcon(icon_path)
    action = QAction(icon, text, parent)
    action.triggered.connect(callback)
    action.setEnabled(enabled_flag)

    if status_tip is not None:
        action.setStatusTip(status_tip)

    if whats_this is not None:
        action.setWhatsThis(whats_this)

    if add_to_menu:
        plugin.iface.addPluginToMenu(plugin.menu, action)

    plugin.actions.append(action)
    return action


def get_icon_path(plugin_dir: str, icon_name: str) -> str:
    """Get icon path, trying Qt resource first, then filesystem fallback."""
    resource_path = f":/plugins/dzetsaka/img/{icon_name}"
    file_path = os.path.join(plugin_dir, "img", icon_name)

    icon = QIcon(resource_path)
    if not icon.isNull():
        return resource_path
    return file_path

