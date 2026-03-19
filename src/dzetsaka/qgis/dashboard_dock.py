"""Dashboard dock lifecycle helpers."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt


def open_dashboard_dock(plugin) -> None:
    """Open the dockable classification dashboard (Quick/Advanced)."""
    from dzetsaka import ui

    if plugin.dashboard_dock is None:
        plugin.dashboard_dock = ui.ClassificationDashboardDock(plugin.iface.mainWindow(), installer=plugin)
        plugin.dashboard_dock.classificationRequested.connect(plugin.execute_dashboard_config)
        plugin.dashboard_dock.closingRequested.connect(plugin.on_close_dashboard_dock)
        # Use Qt.DockWidgetArea(int) constructor to ensure the C++ enum is
        # correctly formed — avoids "invalid 'area' argument" warnings on
        # some PyQt5/Qt5 builds where the Python enum wrapper is not
        # recognised by QMainWindow::addDockWidget.
        try:
            area = Qt.DockWidgetArea(0x1)  # LeftDockWidgetArea
        except TypeError:
            area = Qt.LeftDockWidgetArea
        plugin.iface.addDockWidget(area, plugin.dashboard_dock)

    plugin.dashboard_dock.show()
    plugin.dashboard_dock.raise_()


def close_dashboard_dock(plugin) -> None:
    """Track dashboard dock closing state."""
    if plugin.dashboard_dock is not None:
        plugin.dashboard_dock.hide()
