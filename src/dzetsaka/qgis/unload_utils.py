"""Unload/cleanup helpers for QGIS plugin lifecycle."""

from __future__ import annotations

from qgis.core import QgsApplication


def unload_plugin(plugin) -> None:
    """Remove plugin UI/menu/toolbar artifacts from QGIS."""
    plugin.pluginIsActive = False

    # Disconnect signals before closing widgets
    if plugin.dashboard_dock is not None:
        try:
            plugin.dashboard_dock.classificationRequested.disconnect(plugin.execute_dashboard_config)
        except (TypeError, RuntimeError):
            pass  # Signal was not connected or already disconnected
        try:
            plugin.dashboard_dock.closingRequested.disconnect(plugin.on_close_dashboard_dock)
        except (TypeError, RuntimeError):
            pass
        plugin.dashboard_dock.close()

    # Disconnect toolbar action signal
    if plugin.dashboard_toolbar_action is not None:
        try:
            plugin.dashboard_toolbar_action.triggered.disconnect(plugin.open_dashboard)
        except (TypeError, RuntimeError):
            pass

    QgsApplication.processingRegistry().removeProvider(plugin.provider)

    for action in plugin.actions:
        plugin.iface.removeToolBarIcon(action)
        plugin.iface.removePluginMenu(plugin.tr("&dzetsaka"), action)

