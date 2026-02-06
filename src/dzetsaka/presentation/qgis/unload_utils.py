"""Unload/cleanup helpers for QGIS plugin lifecycle."""

from __future__ import annotations

from qgis.core import QgsApplication


def unload_plugin(plugin) -> None:
    """Remove plugin UI/menu/toolbar artifacts from QGIS."""
    plugin.pluginIsActive = False
    if plugin.dock_widget is not None:
        plugin.dock_widget.close()
    if plugin.dashboard_dock is not None:
        plugin.dashboard_dock.close()

    QgsApplication.processingRegistry().removeProvider(plugin.provider)

    for action in plugin.actions:
        plugin.iface.removeToolBarIcon(action)
        plugin.iface.removePluginMenu(plugin.tr("&dzetsaka"), action)




