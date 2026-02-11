"""Dashboard dock lifecycle helpers."""

from __future__ import annotations


def open_dashboard_dock(plugin, left_dock_area) -> None:
    """Open the dockable classification dashboard (Quick/Advanced)."""
    from dzetsaka import ui

    if plugin.dashboard_dock is None:
        plugin.dashboard_dock = ui.ClassificationDashboardDock(plugin.iface.mainWindow(), installer=plugin)
        plugin.dashboard_dock.classificationRequested.connect(plugin.execute_dashboard_config)
        plugin.dashboard_dock.closingRequested.connect(plugin.on_close_dashboard_dock)
        plugin.iface.addDockWidget(left_dock_area, plugin.dashboard_dock)

    plugin.dashboard_dock.show()
    plugin.dashboard_dock.raise_()


def close_dashboard_dock(plugin) -> None:
    """Track dashboard dock closing state."""
    if plugin.dashboard_dock is not None:
        plugin.dashboard_dock.hide()
