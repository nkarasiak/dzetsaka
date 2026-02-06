"""Dzetsaka Classification Algorithm for QGIS Processing."""

from __future__ import annotations

import os

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

from dzetsaka.logging_utils import QgisLogger, show_error_dialog
from dzetsaka.processing import metadata_helpers
from dzetsaka.scripts.mainfunction import ClassifyImage

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 7)))


class ClassifyAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm for raster image classification."""

    INPUT_RASTER = "INPUT_RASTER"
    INPUT_MASK = "INPUT_MASK"
    INPUT_MODEL = "INPUT_MODEL"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    CONFIDENCE_RASTER = "CONFIDENCE_RASTER"

    def name(self):
        return "Predict model (classification map)"

    def icon(self):
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, self.tr("Mask raster"), optional=True))
        self.addParameter(QgsProcessingParameterFile(self.INPUT_MODEL, self.tr("Model learned")))
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, self.tr("Output raster"), optional=False)
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.CONFIDENCE_RASTER, self.tr("Confidence raster"), optional=True)
        )

    def processAlgorithm(self, parameters, context, feedback):
        log = QgisLogger(tag="Dzetsaka/Processing/Classify")

        try:
            input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
            input_mask = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context)
            input_model = self.parameterAsFile(parameters, self.INPUT_MODEL, context)
            output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
            confidence_raster = self.parameterAsOutputLayer(parameters, self.CONFIDENCE_RASTER, context)

            if not os.path.exists(input_model):
                error_msg = f"Model file not found: {input_model}"
                feedback.reportError(error_msg)
                show_error_dialog("dzetsaka Classify Error", error_msg)
                return {}

            feedback.pushInfo(f"Input raster: {input_raster.source()}")
            feedback.pushInfo(f"Model file: {input_model}")
            feedback.pushInfo(f"Output raster: {output_raster}")
            if confidence_raster:
                feedback.pushInfo(f"Confidence raster: {confidence_raster}")

            worker = ClassifyImage()
            mask = None if input_mask is None else input_mask.source()
            worker.initPredict(
                input_raster.source(),
                input_model,
                output_raster,
                mask,
                confidenceMap=confidence_raster,
                feedback=feedback,
            )

            return {self.OUTPUT_RASTER: output_raster}

        except FileNotFoundError as exc:
            error_msg = f"File not found: {exc!s}"
            feedback.reportError(error_msg)
            log.exception("Classification failed - file not found", exc)
            show_error_dialog("dzetsaka Classify Error", error_msg)
            return {}
        except Exception as exc:
            error_msg = f"Classification failed: {exc!s}"
            feedback.reportError(error_msg)
            log.exception("Classification algorithm failed", exc)
            show_error_dialog("dzetsaka Classify Error", error_msg)
            return {}

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return ClassifyAlgorithm()

    def displayName(self):
        return self.tr(self.name())

    def group(self):
        return self.tr(self.groupId())

    def groupId(self):
        return metadata_helpers.get_group_id()

    def helpUrl(self):
        return metadata_helpers.get_help_url("classify")

    def tags(self):
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("classification")
        return common + specific
