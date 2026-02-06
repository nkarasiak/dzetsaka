"""Startup/bootstrap helpers for QGIS plugin runtime state."""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QDialog

from dzetsaka.dzetsaka_provider import DzetsakaProvider
from dzetsaka.logging_utils import QgisLogger


def initialize_runtime_state(gui, iface) -> None:
    """Initialize DzetsakaGUI runtime state without embedding setup details in plugin_runtime."""
    gui.iface = iface
    gui.log = QgisLogger(tag="Dzetsaka/Core")

    QDialog.__init__(gui)
    gui.settings = QSettings()
    gui.loadConfig()

    gui.provider = DzetsakaProvider()
    gui.plugin_dir = str(Path(__file__).resolve().parents[4])
    gui.plugin_version = gui._read_plugin_version()
    shown_version = gui.settings.value("/dzetsaka/onboardingShownVersion", "", str) or ""
    should_show_onboarding = shown_version != gui.plugin_version
    gui._open_welcome_on_init = bool(gui.firstInstallation or should_show_onboarding)
    gui._open_dashboard_on_init = bool(gui.firstInstallation or should_show_onboarding)

    gui.actions = []
    gui.menu = gui.tr("&dzetsaka")
    gui.pluginIsActive = False
    gui.dockwidget = None
    gui.wizarddock = None
    gui._active_classification_task = None
    gui.lastSaveDir = ""
