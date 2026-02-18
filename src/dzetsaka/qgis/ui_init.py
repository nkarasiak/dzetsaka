"""QGIS GUI initialization helpers."""

from __future__ import annotations

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from dzetsaka.qgis.about_dialog import show_about_dialog


def init_gui(plugin) -> None:
    """Create menu entries and toolbar icons inside QGIS GUI."""
    QgsApplication.processingRegistry().addProvider(plugin.provider)

    icon_path = plugin.get_icon_path("icon.png")
    plugin.add_action(
        icon_path,
        text=plugin.tr("classifier dashboard"),
        callback=plugin.open_dashboard,
        parent=plugin.iface.mainWindow(),
    )

    # Add Batch Classification menu item
    batch_icon_path = plugin.get_icon_path("raster.svg")
    if not batch_icon_path:
        batch_icon_path = icon_path  # Fallback to default icon
    plugin.add_action(
        batch_icon_path,
        text=plugin.tr("Batch Classification..."),
        callback=plugin.open_batch_classification,
        parent=plugin.iface.mainWindow(),
        add_to_toolbar=False,  # Don't add to toolbar, only menu
    )

    # Add About action (menu only)
    plugin.add_action(
        icon_path,
        text=plugin.tr("About dzetsaka"),
        callback=lambda: show_about_dialog(plugin),
        parent=plugin.iface.mainWindow(),
        add_to_toolbar=False,
    )

    plugin.dashboard_toolbar_action = QAction(
        QIcon(plugin.get_icon_path("icon.png")),
        "dzetsaka classifier dashboard",
        plugin.iface.mainWindow(),
    )
    plugin.dashboard_toolbar_action.triggered.connect(plugin.open_dashboard)
    plugin.iface.addToolBarIcon(plugin.dashboard_toolbar_action)
    plugin.actions.append(plugin.dashboard_toolbar_action)
    if plugin._auto_open_dashboard_on_init:
        plugin._auto_open_dashboard_on_init = False
        QTimer.singleShot(1200, plugin.open_dashboard)
