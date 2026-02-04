import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

from . import dzetsaka_dock
from . import settings_dock

from . import welcome

# Load main widget

_DZETSAKA_THEME = """
QDockWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #f7f7fb, stop:1 #eef1f7);
    font-family: "Segoe UI Variable", "Plus Jakarta Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
QWidget#dockWidgetContents {
    background: transparent;
}
QLabel {
    color: #0f172a;
}
QLabel#messageBanner {
    background: #fef9f3;
    border: 1px solid #f3e2d3;
    border-radius: 10px;
    padding: 10px 12px;
    color: #4b2e14;
}
QLineEdit, QComboBox, QSpinBox, QTextEdit {
    background: #ffffff;
    border: 1px solid #d7dbe6;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #fde68a;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
    border: 1px solid #b45309;
}
QToolButton, QPushButton {
    background: #f3f4f6;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 7px 12px;
}
QToolButton:hover, QPushButton:hover {
    background: #e5e7eb;
}
QToolButton:disabled, QPushButton:disabled {
    background: #e5e7eb;
    color: #9ca3af;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e6e8f0;
    border-radius: 12px;
    margin-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #111827;
    font-weight: 600;
}
QCheckBox {
    padding: 2px 0;
}
"""


def _apply_dzetsaka_theme(widget):
    widget.setStyleSheet(_DZETSAKA_THEME)


class dzetsakaDockWidget(QtWidgets.QDockWidget, dzetsaka_dock.Ui_DockWidget):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(dzetsakaDockWidget, self).__init__(parent)
        self.setupUi(self)
        _apply_dzetsaka_theme(self)
        # Emphasize the wizard call-to-action inside the dock.
        if hasattr(self, "messageBanner"):
            self.messageBanner.setText(
                "<b>New workflow available.</b><br>"
                "<a href=\"open_wizard\">Open the step-by-step wizard →</a>"
            )

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


class settings_dock(QtWidgets.QDockWidget, settings_dock.Ui_settingsDock):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(settings_dock, self).__init__(parent)
        self.setupUi(self)
        _apply_dzetsaka_theme(self)

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


class welcomeWidget(QtWidgets.QDockWidget, welcome.Ui_DockWidget):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(welcomeWidget, self).__init__(parent)
        self.setupUi(self)
        _apply_dzetsaka_theme(self)

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


# Wizard UI (Phase 4)
from .wizard_widget import ClassificationWizard
from .comparison_panel import AlgorithmComparisonPanel
