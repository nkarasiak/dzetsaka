import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

from . import dzetsaka_dock

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
                "<a href=\"open_wizard\">Open Express / Guided classifier →</a>"
            )

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


class welcomeWidget(QtWidgets.QDockWidget, welcome.Ui_DockWidget):
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(welcomeWidget, self).__init__(parent)
        self.setupUi(self)
        if hasattr(self, "label"):
            pix = QtGui.QPixmap(":/plugins/dzetsaka/img/parcguyane.jpg")
            if pix.isNull():
                pix = QtGui.QPixmap(os.path.join(os.path.dirname(__file__), "..", "img", "parcguyane.jpg"))
            if not pix.isNull():
                self.label.setPixmap(pix)
        if hasattr(self, "label_2"):
            self.label_2.setText(
                "<html><head/><body>"
                "<p><span style=\" font-weight:600;\">Thanks for installing dzetsaka.</span></p>"
                "<p>For documentation and updates, visit "
                "<a href=\"https://github.com/nkarasiak/dzetsaka\">"
                "<span style=\" font-weight:600; text-decoration: underline; color:#046c08;\">"
                "https://github.com/nkarasiak/dzetsaka"
                "</span></a>.</p>"
                "<p>You can also download a sample dataset to test and understand how dzetsaka works: "
                "<a href=\"https://github.com/lennepkade/dzetsaka/archive/docs.zip\">"
                "<span style=\" font-weight:600; text-decoration: underline; color:#046c08;\">"
                "https://github.com/lennepkade/dzetsaka/archive/docs.zip"
                "</span></a>.</p>"
                "<p>Have fun,<br/>Nicolas Karasiak</p>"
                "</body></html>"
            )

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()


# Wizard UI (Phase 4)
from .wizard_widget import ClassificationWizard
from .wizard_widget import ClassificationDashboardDock
from .comparison_panel import AlgorithmComparisonPanel

# Installation progress dialog
from .install_progress_dialog import InstallProgressDialog
