"""Wizard dock lifecycle helpers."""

from __future__ import annotations


def open_wizard_dock(plugin, left_dock_area) -> None:
    """Open the dockable classification dashboard (Quick/Advanced)."""
    from dzetsaka import ui

    if plugin.wizarddock is None:
        plugin.wizarddock = ui.ClassificationDashboardDock(plugin.iface.mainWindow(), installer=plugin)
        plugin.wizarddock.classificationRequested.connect(plugin.execute_wizard_config)
        plugin.wizarddock.closingPlugin.connect(plugin.onCloseWizardDock)
        plugin.iface.addDockWidget(left_dock_area, plugin.wizarddock)

    plugin.wizarddock.show()
    plugin.wizarddock.raise_()


def close_wizard_dock(plugin) -> None:
    """Track dashboard dock closing state."""
    if plugin.wizarddock is not None:
        plugin.wizarddock.hide()

