from PyQt5.QtWidgets import (QProgressBar, QApplication)
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from qgis.utils import iface


class progressBar(object):
    """!@brief Manage progressBar and loading cursor.
    Allow to add a progressBar in Qgis and to change cursor to loading
    input:
         -inMsg : Message to show to the user (str)
         -inMax : The steps of the script (int)
    output:
        nothing but changing cursor and print progressBar inside Qgis
    """

    def __init__(self, inMsg=' Loading...', inMaxStep=1):
        """
        """
        # Save reference to the QGIS interface
        # initialize progressBar
        # QApplication.processEvents() # Help to keep UI alive
        self.iface = iface

        widget = iface.messageBar().createMessage('Please wait  ', inMsg)

        prgBar = QProgressBar()
        self.prgBar = prgBar

        widget.layout().addWidget(self.prgBar)
        iface.messageBar().pushWidget(widget)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # if Max 0 and value 0, no progressBar, only cursor loading
        # default is set to 0
        prgBar.setValue(1)
        # set Maximum for progressBar
        prgBar.setMaximum(inMaxStep)

    def addStep(self, step=1):
        """!@brief Add a step to the progressBar
        addStep() simply add +1 to current value of the progressBar
        addStep(3) will add 3 steps
        """
        plusOne = self.prgBar.value() + step
        self.prgBar.setValue(plusOne)

    def reset(self):
        """!@brief Simply remove progressBar and reset cursor
        """
        # Remove progressBar and back to default cursor
        self.iface.messageBar().clearWidgets()
        self.iface.mapCanvas().refresh()
        QApplication.restoreOverrideCursor()
