"""Unload/cleanup helpers for QGIS plugin lifecycle."""

from __future__ import annotations

from qgis.core import QgsApplication


def unload_plugin(plugin) -> None:
    """Remove plugin UI/menu/toolbar artifacts from QGIS."""
    plugin.pluginIsActive = False
    if plugin.dockwidget is not None:
        plugin.dockwidget.close()
    if plugin.wizarddock is not None:
        plugin.wizarddock.close()

    QgsApplication.processingRegistry().removeProvider(plugin.provider)

    for action in plugin.actions:
        plugin.iface.removeToolBarIcon(action)
        plugin.iface.removePluginMenu(plugin.tr("&dzetsaka"), action)

