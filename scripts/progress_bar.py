from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QProgressBar
from qgis.utils import iface


class ProgressBar:
    """Manage progressBar and loading cursor.

    Allow to add a progressBar in Qgis and to change cursor to loading.

    Parameters
    ----------
    inMsg : str
        Message to show to the user
    inMax : int
        The steps of the script

    Notes
    -----
    Changes cursor and prints progressBar inside Qgis.

    """

    def __init__(self, inMsg=" Loading...", inMaxStep=1):
        """Initialize the progress bar.

        Parameters
        ----------
        inMsg : str, optional
            Message to display (default: " Loading...")
        inMaxStep : int, optional
            Maximum number of steps (default: 1)

        """
        # Save reference to the QGIS interface
        # initialize progressBar
        # QApplication.processEvents() # Help to keep UI alive
        self.iface = iface

        widget = iface.messageBar().createMessage("Please wait  ", inMsg)

        prgBar = QProgressBar()
        self.prgBar = prgBar

        widget.layout().addWidget(self.prgBar)
        iface.messageBar().pushWidget(widget)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # if Max 0 and value 0, no progressBar, only cursor loading
        prgBar.setValue(0)
        # set Maximum for progressBar - ensure integer conversion for Python 3.12+ compatibility
        prgBar.setMaximum(int(inMaxStep))

    def addStep(self, step=1):
        """Add a step to the progressBar.

        addStep() simply add +1 to current value of the progressBar.
        addStep(3) will add 3 steps.

        Parameters
        ----------
        step : int, optional
            Number of steps to add (default: 1)

        """
        plusOne = self.prgBar.value() + step
        self.prgBar.setValue(plusOne)

    def reset(self):
        """!@brief Simply remove progressBar and reset cursor."""
        # Remove progressBar and back to default cursor
        self.iface.messageBar().clearWidgets()
        self.iface.mapCanvas().refresh()
        QApplication.restoreOverrideCursor()
