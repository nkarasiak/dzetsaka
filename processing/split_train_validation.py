"""Split Train Validation Algorithm for dzetsaka.

This module provides functionality to split training datasets into training
and validation subsets for robust model evaluation.
"""

import os
from typing import ClassVar

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsMessageLog,
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
)

from ..scripts import function_vector
from ..scripts.function_dataraster import get_layer_source_path

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class SplitTrain(QgsProcessingAlgorithm):
    """Algorithm for splitting training datasets into train/validation sets.

    Provides functionality to split vector datasets for training and validation
    purposes using either percentage or count-based methods.
    """

    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    METHOD = "METHOD"
    METHOD_VALUES: ClassVar = ["Percent", "Count value"]
    VALUE = "VALUE"
    OUTPUT_VALIDATION = "OUTPUT_VALIDATION"
    OUTPUT_TRAIN = "OUTPUT_TRAIN"

    def name(self):
        """Returns the algorithm name, used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Split train and validation"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LAYER,
                "Input layer",
            )
        )
        # TABLE / COLUMN
        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_COLUMN,
                "Field (column must have classification number (e.g. '1' forest, '2' water...))",
                parentLayerParameterName=self.INPUT_LAYER,
                optional=False,
            )
        )  # save model

        # Train algorithm

        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                "Select method for splitting dataset",
                self.METHOD_VALUES,
                0,
            )
        )

        # SPLIT %

        self.addParameter(
            QgsProcessingParameterNumber(
                self.VALUE,
                self.tr("Select 50 for 50% if PERCENT method. Else, value represents whole size of test sample."),
                type=QgsProcessingParameterNumber.Integer,
                minValue=1,
                maxValue=99999,
                defaultValue=50,
            )
        )

        # SAVE AS
        # SAVE MODEL
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT_VALIDATION, self.tr("Output validation")))

        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT_TRAIN, self.tr("Output train")))

    def processAlgorithm(self, parameters, context, feedback):
        """Process the algorithm with given parameters."""
        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        METHOD = self.parameterAsEnums(parameters, self.METHOD, context)

        OUTPUT_TRAIN = self.parameterAsOutputLayer(parameters, self.OUTPUT_TRAIN, context)
        OUTPUT_VALIDATION = self.parameterAsOutputLayer(parameters, self.OUTPUT_VALIDATION, context)

        VALUE = self.parameterAsInt(parameters, self.VALUE, context)
        # Retrieve algo from code
        selectedMETHOD = self.METHOD_VALUES[METHOD[0]]

        percent = selectedMETHOD == "Percent"

        libOk = True

        try:
            pass
        except BaseException:
            libOk = False

        if libOk:
            function_vector.RandomInSubset(
                get_layer_source_path(INPUT_LAYER),
                str(INPUT_COLUMN[0]),
                OUTPUT_VALIDATION,
                OUTPUT_TRAIN,
                VALUE,
                percent,
            )
            return {
                self.OUTPUT_TRAIN: OUTPUT_TRAIN,
                self.OUTPUT_VALIDATION: OUTPUT_VALIDATION,
            }
        else:
            # QMessageBox(None, "Please install scikit-learn library")
            QgsMessageLog.logMessage("Please install scikit-learn library")

    def tr(self, string):
        """Translate string using Qt's translation system."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return SplitTrain()

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
        return "Vector manipulation"
