"""Median Filter Algorithm for dzetsaka.

This module provides morphological median filter operations for noise reduction
and smoothing in classified raster images.
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

from PyQt5.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtGui import QIcon

from ..scripts import function_dataraster as dataraster

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# EX
"""
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.outputs import OutputRaster
"""


class MedianFilterAlgorithm(QgsProcessingAlgorithm):
    """Median filter algorithm for noise reduction in raster images.

    Applies median filtering to smooth classified images and reduce
    salt-and-pepper noise while preserving edges and boundaries.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    MEDIAN_SIZE = "MEDIAN_SIZE"
    MEDIAN_ITER = "MEDIAN_ITER"

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
                self.MEDIAN_ITER,
                self.tr("Number of iteration for median filter"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=3,
            )
        )

        # add num
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MEDIAN_SIZE,
                self.tr("Window size of median filter"),
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
        return "Median filter"

    def processAlgorithm(self, parameters, context, feedback):
        """Here is where the processing itself takes place."""
        INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        # INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_RASTER = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        MEDIAN_ITER = self.parameterAsInt(parameters, self.MEDIAN_ITER, context)
        MEDIAN_SIZE = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)
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

        from scipy import ndimage

        data, im = dataraster.open_data_band(INPUT_RASTER_src)

        proj = data.GetProjection()
        geo = data.GetGeoTransform()
        d = data.RasterCount

        total = 100 / (d * MEDIAN_ITER)

        outFile = dataraster.create_empty_tiff(OUTPUT_RASTER, im, d, geo, proj)

        iterPos = 0

        for i in range(d):
            # Read data from the right band
            # pbNow+=1
            # pb.setValue(pbNow)
            iterPos += 1
            tempBand = data.GetRasterBand(i + 1).ReadAsArray()

            for j in range(MEDIAN_ITER):
                tempBand = ndimage.filters.median_filter(tempBand, size=(MEDIAN_SIZE, MEDIAN_SIZE))
                # tempBand = tempBand
                iterPos += j
                feedback.setProgress(int(iterPos * total))

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
        return MedianFilterAlgorithm()

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
