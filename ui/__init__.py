import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import pyqtSignal

# Import resources for icons
try:
    from .. import resources
except ImportError:
    pass

from . import dzetsaka_dock
from . import settings_dock

from . import welcome
from .anniversary_widget import AnniversaryDialog, AnniversaryManager

# Load main widget


class dzetsakaDockWidget(QtWidgets.QDockWidget, dzetsaka_dock.Ui_DockWidget):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(dzetsakaDockWidget, self).__init__(parent)
        self.setupUi(self)

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
