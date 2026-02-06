"""Small UI utility helpers for dzetsaka plugin runtime."""

from __future__ import annotations

from dzetsaka import ui


def remember_last_save_dir(plugin, file_name: str) -> None:
    """Remember the last directory used for saving/loading files."""
    if file_name != "":
        plugin.lastSaveDir = file_name
        plugin.settings.setValue("/dzetsaka/lastSaveDir", plugin.lastSaveDir)


def show_welcome_widget(plugin) -> None:
    """Display the welcome widget and persist onboarding flag."""
    plugin.welcomeWidget = ui.welcomeWidget()
    plugin.welcomeWidget.show()
    plugin.settings.setValue("/dzetsaka/firstInstallation", False)

