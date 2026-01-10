"""Resample Image Same Date Algorithm for dzetsaka.

This module provides temporal resampling functionality to match image dates
for time series analysis in remote sensing applications.
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
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

# from ..scripts import function_dataraster as dataraster
from ..scripts.resample_same_date_as_source import resampleWithSameDateAsSource

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# EX
"""
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.outputs import OutputRaster
"""


class ResampleImageSameDateAsSource(QgsProcessingAlgorithm):
    """Resample images to match source image dates.

    Resamples target images to align with source image acquisition dates
    for consistent temporal analysis in time series processing.

    All Processing algorithms should extend the GeoAlgorithm class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    SOURCE_RASTER = "SOURCE_RASTER"
    TARGET_RASTER = "TARGET_RASTER"
    OUTPUT_RASTER = "OUTPUT_RASTER"

    TARGET_DATES = "TARGET_DATES"
    SOURCE_DATES = "SOURCE_DATES"

    N_SPECTRAL_BAND = "N_SPECTRAL_BAND"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Here we define the inputs and output of the algorithm, along.

        with some other properties.
        """
        # We add the input vector layer. It can have any kind of geometry
        # It is a mandatory (not optional) one, hence the False argument
        self.addParameter(QgsProcessingParameterRasterLayer(self.SOURCE_RASTER, self.tr("Source raster")))

        self.addParameter(QgsProcessingParameterRasterLayer(self.TARGET_RASTER, self.tr("Target raster")))

        # We add a raster as output
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr("Output raster")))

        self.addParameter(QgsProcessingParameterFile(self.SOURCE_DATES, self.tr("Source dates (csv)"), extension="csv"))

        self.addParameter(QgsProcessingParameterFile(self.TARGET_DATES, self.tr("Target dates (csv)"), extension="csv"))

        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_SPECTRAL_BAND,
                self.tr("Number of spectral bands in your SITS (e.g. 4 if B,R,G,IR)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4,
                minValue=1,
            )
        )

        # add num

    def name(self):
        """Returns the algorithm name, used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Resample SITS dates"

    def processAlgorithm(self, parameters, context, feedback):
        """Here is where the processing itself takes place."""
        SOURCE_RASTER = self.parameterAsRasterLayer(parameters, self.SOURCE_RASTER, context)
        TARGET_RASTER = self.parameterAsRasterLayer(parameters, self.TARGET_RASTER, context)

        N_SPECTRAL_BAND = self.parameterAsInt(parameters, self.N_SPECTRAL_BAND, context)

        SOURCE_DATES = self.parameterAsFile(parameters, self.SOURCE_DATES, context)
        TARGET_DATES = self.parameterAsFile(parameters, self.TARGET_DATES, context)

        OUTPUT_RASTER = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)

        SOURCE_RASTER_src = SOURCE_RASTER.source()
        TARGET_RASTER_src = TARGET_RASTER.source()

        libOk = True
        libErrors = []
        commandBashToTest = ["otbcli_BandMath", "gdalbuildvrt"]
        for command in commandBashToTest:
            if os.system(command) != 256:
                libOk = False
                libErrors.append(command)

        # learn model
        if libOk:
            resampleWithSameDateAsSource(
                SOURCE_RASTER_src,
                TARGET_RASTER_src,
                SOURCE_DATES,
                TARGET_DATES,
                N_SPECTRAL_BAND,
                OUTPUT_RASTER,
                feedback,
            )
            return {self.OUTPUT_RASTER: OUTPUT_RASTER}

        else:
            return {"Missing library": f"Error importing {libErrors}"}
            # QMessageBox.about(None, "Missing library", "Please install scikit-learn library to use"+str(SELECTED_ALGORITHM))

        # return OUTPUT_RASTER

    def tr(self, string):
        """Translate string using Qt's translation system."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return ResampleImageSameDateAsSource()

    def displayName(self):
        """Returns the translated algorithm name, which should be used for any user-visible display.

        The algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """Returns the name of the group this algorithm belongs to.

        This string should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """Returns the unique ID of the group this algorithm belongs to.

        This string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Raster tool"
