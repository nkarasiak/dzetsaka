"""Split Train Validation Algorithm for dzetsaka.

This module provides functionality to split training datasets into training
and validation subsets for robust model evaluation.
"""

import os
from typing import ClassVar

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
)

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon

from dzetsaka.infrastructure.geo.vector_split import split_vector_stratified
from dzetsaka.logging import show_error_dialog
from dzetsaka.qgis.logging import QgisLogger
from dzetsaka.qgis.processing import metadata_helpers
from dzetsaka.scripts.function_dataraster import get_layer_source_path

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 7)))


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
            ),
        )
        # TABLE / COLUMN
        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_COLUMN,
                "Field (column must have classification number (e.g. '1' forest, '2' water...))",
                parentLayerParameterName=self.INPUT_LAYER,
                optional=False,
            ),
        )  # save model

        # Train algorithm

        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                "Select method for splitting dataset",
                self.METHOD_VALUES,
                0,
            ),
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
            ),
        )

        # SAVE AS
        # SAVE MODEL
        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT_VALIDATION, self.tr("Output validation")))

        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT_TRAIN, self.tr("Output train")))

    def processAlgorithm(self, parameters, context, feedback):
        """Process the algorithm with given parameters."""
        log = QgisLogger(tag="Dzetsaka/Processing/SplitTrainValidation")

        try:
            INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
            INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
            METHOD = self.parameterAsEnums(parameters, self.METHOD, context)

            OUTPUT_TRAIN = self.parameterAsOutputLayer(parameters, self.OUTPUT_TRAIN, context)
            OUTPUT_VALIDATION = self.parameterAsOutputLayer(parameters, self.OUTPUT_VALIDATION, context)

            VALUE = self.parameterAsInt(parameters, self.VALUE, context)
            # Retrieve algo from code
            selectedMETHOD = self.METHOD_VALUES[METHOD[0]]

            use_percent = selectedMETHOD == "Percent"

            # Convert from validation-centric to train-centric parameter
            # Old RandomInSubset used VALUE for validation size
            # New split_vector_stratified uses train_percent for training size
            # VALUE is validation size. For percentage mode, convert to train percent.
            # For absolute mode we keep VALUE and delegate exact handling downstream.
            train_value = 100 - VALUE if use_percent else VALUE
            # We'll need to adjust the logic in the call

            try:
                # Call new scikit-learn based splitting function
                if use_percent:
                    train_path, valid_path = split_vector_stratified(
                        vector_path=get_layer_source_path(INPUT_LAYER),
                        class_field=str(INPUT_COLUMN[0]),
                        train_percent=train_value,
                        train_output=OUTPUT_TRAIN,
                        validation_output=OUTPUT_VALIDATION,
                        use_percent=True,
                    )
                else:
                    # For count mode, we need total features to calculate train count
                    # Let's read the total count first
                    try:
                        from osgeo import ogr
                    except ImportError:
                        import ogr  # type: ignore[no-redef]

                    ds = ogr.Open(get_layer_source_path(INPUT_LAYER))
                    if ds is None:
                        raise RuntimeError("Unable to open input vector layer")
                    lyr = ds.GetLayer()
                    total_features = lyr.GetFeatureCount()
                    ds = None

                    train_count = max(1, total_features - VALUE)
                    train_path, valid_path = split_vector_stratified(
                        vector_path=get_layer_source_path(INPUT_LAYER),
                        class_field=str(INPUT_COLUMN[0]),
                        train_percent=train_count,
                        train_output=OUTPUT_TRAIN,
                        validation_output=OUTPUT_VALIDATION,
                        use_percent=False,
                    )

                feedback.pushInfo(f"Training samples written to: {train_path}")
                feedback.pushInfo(f"Validation samples written to: {valid_path}")

                return {
                    self.OUTPUT_TRAIN: OUTPUT_TRAIN,
                    self.OUTPUT_VALIDATION: OUTPUT_VALIDATION,
                }

            except ImportError:
                error_msg = "scikit-learn is required for train/test splitting. Install with: pip install scikit-learn"
                feedback.reportError(error_msg)
                log.error(error_msg)
                show_error_dialog("dzetsaka Split Train/Validation Error", error_msg)
                return {}

        except Exception as e:
            error_msg = f"Split train/validation failed: {e!s}"
            feedback.reportError(error_msg)
            log.exception("Split train/validation algorithm failed", e)
            show_error_dialog("dzetsaka Split Train/Validation Error", error_msg)
            return {}

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
        return metadata_helpers.get_group_id()

    def helpUrl(self):
        """Returns a URL to the algorithm's help/documentation."""
        return metadata_helpers.get_help_url("split_train_validation")

    def tags(self):
        """Returns tags for the algorithm for better searchability."""
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("preprocessing")
        return common + specific
