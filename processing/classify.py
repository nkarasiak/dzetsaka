"""Dzetsaka Classification Algorithm for QGIS Processing.

This module provides the classification algorithm for the QGIS Processing framework,
allowing users to classify raster images using trained machine learning models.
"""

import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

from ..logging_utils import QgisLogger, show_error_dialog
from ..scripts.mainfunction import ClassifyImage
from . import metadata_helpers

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class ClassifyAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm for raster image classification.

    This algorithm applies trained machine learning models to classify raster images
    using any of the supported algorithms (GMM, RF, SVM, KNN, XGB, LGB, etc.).
    """

    INPUT_RASTER = "INPUT_RASTER"
    INPUT_MASK = "INPUT_MASK"
    INPUT_MODEL = "INPUT_MODEL"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    CONFIDENCE_RASTER = "CONFIDENCE_RASTER"

    def name(self):
        """Return the algorithm name used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Predict model (classification map)"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        # inputs
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, self.tr("Mask raster"), optional=True))

        self.addParameter(QgsProcessingParameterFile(self.INPUT_MODEL, self.tr("Model learned")))

        # output
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr("Output raster"), optional=False)
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(self.CONFIDENCE_RASTER, self.tr("Confidence raster"), optional=True)
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process the classification algorithm."""
        log = QgisLogger(tag="Dzetsaka/Processing/Classify")

        try:
            INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
            INPUT_MASK = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context)
            INPUT_MODEL = self.parameterAsFile(parameters, self.INPUT_MODEL, context)

            OUTPUT_RASTER = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
            CONFIDENCE_RASTER = self.parameterAsOutputLayer(parameters, self.CONFIDENCE_RASTER, context)

            # Validate model file exists
            if not os.path.exists(INPUT_MODEL):
                error_msg = f"Model file not found: {INPUT_MODEL}"
                feedback.reportError(error_msg)
                show_error_dialog("dzetsaka Classify Error", error_msg)
                return {}

            # Log classification parameters
            feedback.pushInfo(f"Input raster: {INPUT_RASTER.source()}")
            feedback.pushInfo(f"Model file: {INPUT_MODEL}")
            feedback.pushInfo(f"Output raster: {OUTPUT_RASTER}")
            if CONFIDENCE_RASTER:
                feedback.pushInfo(f"Confidence raster: {CONFIDENCE_RASTER}")

            # Retrieve algo from code
            worker = ClassifyImage()
            # classify
            mask = None if INPUT_MASK is None else INPUT_MASK.source()
            worker.initPredict(
                INPUT_RASTER.source(),
                INPUT_MODEL,
                OUTPUT_RASTER,
                mask,
                confidenceMap=CONFIDENCE_RASTER,
                feedback=feedback,
            )

            return {self.OUTPUT_RASTER: OUTPUT_RASTER}

        except FileNotFoundError as e:
            error_msg = f"File not found: {e!s}"
            feedback.reportError(error_msg)
            log.exception("Classification failed - file not found", e)
            show_error_dialog("dzetsaka Classify Error", error_msg)
            return {}
        except Exception as e:
            error_msg = f"Classification failed: {e!s}"
            feedback.reportError(error_msg)
            log.exception("Classification algorithm failed", e)
            show_error_dialog("dzetsaka Classify Error", error_msg)
            return {}

    def tr(self, string):
        """Translate string using Qt translation API."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return ClassifyAlgorithm()

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
        return metadata_helpers.get_group_id()

    def helpUrl(self):
        """Returns a URL to the algorithm's help/documentation."""
        return metadata_helpers.get_help_url("classify")

    def tags(self):
        """Returns tags for the algorithm for better searchability."""
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("classification")
        return common + specific
