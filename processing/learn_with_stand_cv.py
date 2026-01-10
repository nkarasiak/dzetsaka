"""STAND Cross-Validation Algorithm for dzetsaka.

This module provides STAND (Spatially and Temporally Adaptive Non-parametric
Distance) cross-validation methods for robust model evaluation.
"""

import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
)

from .. import classifier_config
from ..scripts import mainfunction
from ..scripts.function_dataraster import get_layer_source_path

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class TrainSTANDAlgorithm(QgsProcessingAlgorithm):
    """STAND Cross-Validation training algorithm.

    Implements STAND (Spatially and Temporally Adaptive Non-parametric Distance)
    cross-validation for robust model evaluation with spatial and temporal constraints.
    """

    INPUT_RASTER = "INPUT_RASTER"
    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = classifier_config.UI_DISPLAY_NAMES
    TRAIN_ALGORITHMS_CODE = classifier_config.CLASSIFIER_CODES
    SLOO = "SLOO"
    STAND_COLUMN = "STAND_COLUMN"
    MAXITER = "MAXITER"
    PARAMGRID = "PARAMGRID"
    # MINTRAIN = "MINTRAIN"
    # SPLIT_PERCENT= 'SPLIT_PERCENT'
    OUTPUT_MODEL = "OUTPUT_MODEL"
    # OUTPUT_MATRIX = "OUTPUT_MATRIX"
    SAVEDIR = "SAVEDIR"

    def shortHelpString(self):
        """Return the short help string for the algorithm."""
        return self.tr(
            "Learn with Cross Validation with Spatial Leave-One-Out stand to better learn and estimate prediction."
        )

    """
    def helpUrl(self):
        return "http://pot.readthedocs.io/en/stable/all.html#module-ot.da"
    """

    def name(self):
        """Return the algorithm name used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Train algorithm (CV per stand/polygon)"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        # The name that the user will see in the toolbox

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

        # SLOO
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SLOO,
                self.tr("Check for Leave-One-Out validation. Uncheck for 50\\50."),
                defaultValue=True,
            )
        )

        # Train algorithm

        self.addParameter(QgsProcessingParameterEnum(self.TRAIN, "Select algorithm to train", self.TRAIN_ALGORITHMS, 0))

        # ROI
        # VECTOR
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

        self.addParameter(
            QgsProcessingParameterField(
                self.STAND_COLUMN,
                "Stand number (column must have unique id per stand)",
                parentLayerParameterName=self.INPUT_LAYER,
                optional=False,
            )
        )  # save model

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXITER,
                self.tr("Maximum iteration (default : 5)"),
                type=QgsProcessingParameterNumber.Integer,
                minValue=1,
                maxValue=99999,
                defaultValue=5,
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.PARAMGRID,
                self.tr("Parameters for the hyperparameters of the algorithm"),
                optional=True,
            )
        )
        # SAVE AS
        # SAVE MODEL
        self.addParameter(
            QgsProcessingParameterFileDestination(self.OUTPUT_MODEL, self.tr("Output model (to use for classifying)"))
        )
        """
        # SAVE CONFUSION MATRIX
        self.addParameter(
        QgsProcessingParameterFileDestination(
            self.OUTPUT_MATRIX,
            self.tr("Output confusion matrix"),
            fileFilter='csv'))#,
            #ext='csv'))
        """
        # SAVE DIR
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.SAVEDIR, self.tr("Directory to save every confusion matrix"))
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process the STAND cross-validation algorithm."""
        INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)

        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        STAND_COLUMN = self.parameterAsFields(parameters, self.STAND_COLUMN, context)
        # SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)
        # TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
        # INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_MODEL = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
        # OUTPUT_MATRIX = self.parameterAsFileOutput(parameters, self.OUTPUT_MATRIX, context)

        SAVEDIR = self.parameterAsFileOutput(parameters, self.SAVEDIR, context)
        # Retrieve algo from code
        SLOO = self.parameterAsBool(parameters, self.SLOO, context)

        extraParam = {}

        extraParam["SLOO"] = SLOO
        # extraParam['maxIter']=False
        # extraParam['param_grid'] = dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6))

        MAXITER = self.parameterAsInt(parameters, self.MAXITER, context)

        if MAXITER == 0:
            MAXITER = False
        extraParam["maxIter"] = MAXITER

        PARAMGRID = self.parameterAsString(parameters, self.PARAMGRID, context)
        if PARAMGRID != "":
            extraParam["param_grid"] = eval(PARAMGRID)

        if not SAVEDIR.endswith("/"):
            SAVEDIR += "/"

        extraParam["saveDir"] = SAVEDIR

        extraParam["inStand"] = STAND_COLUMN[0]

        TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)

        # Retrieve algo from code
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]

        # eval(PARAM_GRID

        # learn model
        mainfunction.LearnModel(
            INPUT_RASTER.source(),
            get_layer_source_path(INPUT_LAYER),
            INPUT_COLUMN[0],
            OUTPUT_MODEL,
            "STAND",
            0,
            None,
            SELECTED_ALGORITHM,
            feedback=feedback,
            extraParam=extraParam,
        )
        return {self.SAVEDIR: SAVEDIR, self.OUTPUT_MODEL: OUTPUT_MODEL}

    def tr(self, string):
        """Translate string using Qt translation API."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return TrainSTANDAlgorithm()

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
        return "Classification tool"
