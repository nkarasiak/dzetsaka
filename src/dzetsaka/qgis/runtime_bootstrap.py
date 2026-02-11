"""Startup/bootstrap helpers for QGIS plugin runtime state."""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QDialog

from dzetsaka.dzetsaka_provider import DzetsakaProvider
from dzetsaka.qgis.logging import QgisLogger, register_qgis_logging


def initialize_runtime_state(gui, iface) -> None:
    """Initialize DzetsakaGUI runtime state without embedding setup details in plugin_runtime."""
    gui.iface = iface
    register_qgis_logging()
    gui.log = QgisLogger(tag="Dzetsaka")

    QDialog.__init__(gui)
    gui.settings = QSettings()
    gui.loadConfig()

    gui.provider = DzetsakaProvider()
    gui.plugin_dir = str(Path(__file__).resolve().parents[4])
    gui.plugin_version = gui._read_plugin_version()
    shown_version = gui.settings.value("/dzetsaka/onboardingShownVersion", "", str) or ""
    gui._open_dashboard_on_init = True

    gui.actions = []
    gui.menu = gui.tr("&dzetsaka")
    gui.pluginIsActive = False
    gui.dock_widget = None
    gui.dashboard_dock = None
    gui._active_classification_task = None
    gui.lastSaveDir = ""
