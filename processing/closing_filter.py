"""Closing Filter Algorithm for dzetsaka.

This module provides morphological closing filter operations for raster processing
within the dzetsaka QGIS plugin framework.
"""

__author__ = "Nicolas Karasiak"
__date__ = "2018-02-24"
__copyright__ = "(C) 2018 by Nicolas Karasiak"

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = "$Format:%H$"


# from ... import dzetsaka.scripts.function_dataraster as dataraster

# from PyQt4.QtGui import QIcon
# from PyQt4.QtCore import QSettings


import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

from ..scripts import function_dataraster as dataraster

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# EX
"""
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.outputs import OutputRaster
"""


class ClosingFilterAlgorithm(QgsProcessingAlgorithm):
    """Morphological closing filter algorithm for raster processing.

    Applies morphological closing operations to raster images, which consists
    of a dilation followed by an erosion. This is useful for filling small
    holes and gaps in classified images.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    CLOSING_SIZE = "CLOSING_SIZE"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Define the inputs and output of the algorithm.

        Along with some other properties.
        """
        # We add the input vector layer. It can have any kind of geometry
        # It is a mandatory (not optional) one, hence the False argument
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

        # We add a raster as output
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr("Output raster")))
        # add num

        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLOSING_SIZE,
                self.tr("Size of the closing filter"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=3,
            )
        )

    def name(self):
        """Return the algorithm name used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Closing filter"

    def processAlgorithm(self, parameters, context, feedback):
        """Here is where the processing itself takes place."""
        INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        # INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_RASTER = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        CLOSING_SIZE = self.parameterAsInt(parameters, self.CLOSING_SIZE, context)

        """
        MEDIAN_ITER = self.parameterAsInt(parameters, self.MEDIAN_ITER, context)
        MEDIAN_SIZE = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)
        # First we create the output layer. The output value entered by
        # the user is a string containing a filename, so we can use it
        # directly

        #from scipy import ndimage
        #import gdal
        """
        INPUT_RASTER_src = INPUT_RASTER.source()

        # feedback.pushInfo(str(OUTPUT_RASTER))
        # QgsMessageLog.logMessage('output is: '+str(OUTPUT_RASTER))

        from scipy.ndimage.morphology import grey_closing

        data, im = dataraster.open_data_band(INPUT_RASTER_src)

        proj = data.GetProjection()
        geo = data.GetGeoTransform()
        d = data.RasterCount

        total = 100 / (d * 1)

        outFile = dataraster.create_empty_tiff(OUTPUT_RASTER, im, d, geo, proj)

        for i in range(d):
            # Read data from the right band
            # pbNow+=1
            # pb.setValue(pbNow)

            tempBand = data.GetRasterBand(i + 1).ReadAsArray()

            tempBand = grey_closing(tempBand, size=(CLOSING_SIZE, CLOSING_SIZE))
            # tempBand = tempBand
            feedback.setProgress(int(i * total))

            # Save bandand outFile
            out = outFile.GetRasterBand(i + 1)
            out.WriteArray(tempBand)
            out.FlushCache()
            tempBand = None

        return {self.OUTPUT_RASTER: OUTPUT_RASTER}

        # return OUTPUT_RASTER

    def tr(self, string):
        """Translate string using Qt translation API."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return ClosingFilterAlgorithm()

    def displayName(self):
        """Return the translated algorithm name.

        Should be used for any user-visible display of the algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """Return the name of the group this algorithm belongs to.

        This string should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """Return the unique ID of the group this algorithm belongs to.

        This string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Raster tool"
