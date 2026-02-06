"""Configuration loading helpers for plugin runtime."""

from __future__ import annotations

from dzetsaka import classifier_config
from dzetsaka.logging_utils import show_error_dialog


def load_config(gui) -> None:
    """Load plugin runtime configuration from QSettings."""
    try:
        gui.classifiers = classifier_config.CLASSIFIER_NAMES

        gui.classifier = gui.settings.value("/dzetsaka/classifier", "", str)
        if not gui.classifier:
            gui.classifier = gui.classifiers[0]
            gui.settings.setValue("/dzetsaka/classifier", gui.classifier)

        gui.maskSuffix = gui.DEFAULT_MASK_SUFFIX
        first_install_raw = gui.settings.value("/dzetsaka/firstInstallation", None)
        if first_install_raw is None:
            gui.firstInstallation = True
            gui.settings.setValue("/dzetsaka/firstInstallation", True)
        elif isinstance(first_install_raw, bool):
            gui.firstInstallation = first_install_raw
        else:
            gui.firstInstallation = str(first_install_raw).strip().lower() in ("1", "true", "yes", "on")
    except BaseException:
        gui.log.error("Failed to open config file " + gui.configFile)
        show_error_dialog(
            "dzetsaka Configuration Error",
            "Failed to load configuration. Check the QGIS log for details.",
            parent=gui.iface.mainWindow(),
        )
