import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

from . import dzetsaka_dock
from . import settings_dock

from . import welcome


class dzetsakaDockWidget(QtWidgets.QDockWidget, dzetsaka_dock.Ui_DockWidget):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(dzetsakaDockWidget, self).__init__(parent)
        self.setupUi(self)
        # Emphasize the wizard call-to-action inside the dock.
        if hasattr(self, "messageBanner"):
            self.messageBanner.setText(
                "<b>New dashboard available.</b><br>"
                "<a href=\"open_wizard\">Open Quick run / Advanced setup →</a>"
            )

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


class settings_dock(QtWidgets.QDockWidget, settings_dock.Ui_settingsDock):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(settings_dock, self).__init__(parent)
        self.setupUi(self)

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


class welcomeWidget(QtWidgets.QDockWidget, welcome.Ui_DockWidget):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(welcomeWidget, self).__init__(parent)
        self.setupUi(self)

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


# Wizard UI (Phase 4)
from .wizard_widget import ClassificationWizard
from .wizard_widget import ClassificationDashboardDock
from .comparison_panel import AlgorithmComparisonPanel

# Installation progress dialog
from .install_progress_dialog import InstallProgressDialog
