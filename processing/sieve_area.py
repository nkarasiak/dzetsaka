"""Sieve Area Algorithm for dzetsaka.

This module provides sieve filtering functionality to remove small pixel groups
in classification results based on area thresholds.
"""

import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

try:
    from osgeo import gdal
except ImportError:
    import gdal

from ..logging_utils import QgisLogger, show_error_dialog
from ..scripts import function_dataraster as dataraster
from . import metadata_helpers

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class SieveAreaAlgorithm(QgsProcessingAlgorithm):
    """Sieve filter algorithm for removing small pixel groups by area.

    Applies GDAL sieve filter to remove small connected regions from
    raster classification results based on area threshold.
    """

    INPUT_RASTER = "INPUT_RASTER"
    SIZE_HA = "SIZE_HA"
    INPUT_CONNECTIVITY = "INPUT_CONNECTIVITY"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    CONNECTIVITY = ["4", "8"]

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def name(self):
        """Returns the algorithm name, used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Sieve raster by area"

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

        # SIEVE SIZE
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SIZE_HA,
                self.tr("Sieve size (0.5 for 0.5ha in metrics)"),
                type=QgsProcessingParameterNumber.Double,
                minValue=0.0,
                defaultValue=0.5,
            )
        )

        # CONNECTIVITY
        self.addParameter(
            QgsProcessingParameterEnum(
                self.INPUT_CONNECTIVITY,
                "Connectivity",
                self.CONNECTIVITY,
                defaultValue=0,
            )
        )

        # OUTPUT RASTER
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr("Output raster")))

    def processAlgorithm(self, parameters, context, feedback):
        """Process the sieve filter algorithm."""
        log = QgisLogger(tag="Dzetsaka/Processing/Sieve")

        try:
            INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
            OUTPUT_RASTER = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
            SIZE_HA = self.parameterAsDouble(parameters, self.SIZE_HA, context)
            INPUT_CONNECTIVITY = self.parameterAsEnum(parameters, self.INPUT_CONNECTIVITY, context)

            connectivity = int(self.CONNECTIVITY[INPUT_CONNECTIVITY])

            # convert hectare to square meters
            SIZE_M2 = int(SIZE_HA * 10000)

            # Open source dataset
            datasrc = gdal.Open(INPUT_RASTER.source())
            if datasrc is None:
                error_msg = f"Failed to open input raster: {INPUT_RASTER.source()}"
                feedback.reportError(error_msg)
                log.error(error_msg)
                show_error_dialog("dzetsaka Sieve Error", error_msg)
                return {}

            d = datasrc.RasterCount

            # Create output dataset
            drv = gdal.GetDriverByName("GTiff")
            dst_ds = drv.Create(
                OUTPUT_RASTER,
                datasrc.RasterXSize,
                datasrc.RasterYSize,
                d,
                gdal.GDT_Byte,
            )

            dst_ds.SetGeoTransform(datasrc.GetGeoTransform())
            dst_ds.SetProjection(datasrc.GetProjection())

            # Get pixel size to calculate number of pixels to sieve
            pixelSize = datasrc.GetGeoTransform()[1]  # get pixel size
            pixelSieve = int(SIZE_M2 / (pixelSize * pixelSize))  # get number of pixels to sieve

            log.info(f"Pixel to sieve: {pixelSieve}")
            feedback.pushInfo(f"Sieve threshold: {pixelSieve} pixels ({SIZE_HA} ha)")

            # Process each band
            for i in range(d):
                feedback.setProgress(int((i / d) * 100))
                feedback.pushInfo(f"Processing band {i + 1} of {d}")

                srcband = datasrc.GetRasterBand(i + 1)
                dstband = dst_ds.GetRasterBand(i + 1)

                gdal.SieveFilter(srcband, None, dstband, pixelSieve, connectivity)

                srcband = None
                dstband = None

            # Close datasets
            dst_ds = None
            datasrc = None

            feedback.setProgress(100)
            return {self.OUTPUT_RASTER: OUTPUT_RASTER}

        except Exception as e:
            error_msg = f"Sieve filter failed: {e!s}"
            feedback.reportError(error_msg)
            log.exception("Sieve filter algorithm failed", e)
            show_error_dialog("dzetsaka Sieve Error", error_msg)
            return {}

    def tr(self, string):
        """Translate string using Qt's translation system."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return SieveAreaAlgorithm()

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
        return metadata_helpers.get_group_id()

    def helpUrl(self):
        """Returns a URL to the algorithm's help/documentation."""
        return metadata_helpers.get_help_url("sieve")

    def tags(self):
        """Returns tags for the algorithm for better searchability."""
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("postprocessing")
        return common + specific
