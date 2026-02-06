"""QGIS GUI initialization helpers."""

from __future__ import annotations

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtGui import QAction, QIcon


def init_gui(plugin) -> None:
    """Create menu entries and toolbar icons inside QGIS GUI."""
    QgsApplication.processingRegistry().addProvider(plugin.provider)

    icon_path = plugin.get_icon_path("icon.png")
    plugin.add_action(
        icon_path,
        text=plugin.tr("welcome message"),
        callback=plugin.showWelcomeWidget,
        add_to_toolbar=False,
        parent=plugin.iface.mainWindow(),
    )

    icon_path = plugin.get_icon_path("icon.png")
    plugin.add_action(
        icon_path,
        text=plugin.tr("classifier dashboard"),
        callback=plugin.run_wizard,
        parent=plugin.iface.mainWindow(),
    )

    plugin.dockIcon = QAction(
        QIcon(plugin.get_icon_path("icon.png")),
        "dzetsaka classifier dashboard",
        plugin.iface.mainWindow(),
    )
    plugin.dockIcon.triggered.connect(plugin.run_wizard)
    plugin.iface.addToolBarIcon(plugin.dockIcon)
    plugin.actions.append(plugin.dockIcon)

    if plugin._open_welcome_on_init:
        plugin._open_welcome_on_init = False
        plugin.settings.setValue("/dzetsaka/onboardingShownVersion", plugin.plugin_version)
        QTimer.singleShot(400, plugin.showWelcomeWidget)
    if plugin._open_dashboard_on_init:
        plugin._open_dashboard_on_init = False
        QTimer.singleShot(800, plugin.run_wizard)

