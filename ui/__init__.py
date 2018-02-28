
import os

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import pyqtSignal

from . import dzetsaka_dock
from . import settings_dock


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
