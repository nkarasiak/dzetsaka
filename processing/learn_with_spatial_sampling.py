"""Spatial Leave-One-Out Cross-Validation Algorithm for dzetsaka.

This module provides spatial cross-validation methods for robust model
evaluation in remote sensing classification tasks.
"""

import os

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
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
from ..logging_utils import QgisLogger, show_error_dialog
from ..scripts import mainfunction
from . import metadata_helpers

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class TrainSLOOAlgorithm(QgsProcessingAlgorithm):
    """Spatial Leave-One-Out Cross-Validation training algorithm.

    Implements spatial cross-validation to provide robust model evaluation
    that accounts for spatial autocorrelation in remote sensing data.
    """

    INPUT_RASTER = "INPUT_RASTER"
    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = classifier_config.UI_DISPLAY_NAMES
    TRAIN_ALGORITHMS_CODE = classifier_config.CLASSIFIER_CODES

    DISTANCE = "DISTANCE"
    MAXITER = "MAXITER"
    PARAMGRID = "PARAMGRID"
    MINTRAIN = "MINTRAIN"
    # SPLIT_PERCENT= 'SPLIT_PERCENT'
    OUTPUT_MODEL = "OUTPUT_MODEL"
    # OUTPUT_MATRIX = "OUTPUT_MATRIX"
    MAX_ITER = "MAX_ITER"
    SAVEDIR = "SAVEDIR"

    def shortHelpString(self):
        """Return the short help string for the algorithm."""
        return self.tr(
            "Spatial sampling to better learn and estimate prediction.. \n \n \
                       SLOO : Spatial Leave-One-Out Cross Validation. \n \
                       \
                       <h3>Classifier (paramgrid)</h3> \n \
                       Param grid can be fit for each algorithm : \n \
                       <h4>Random-Forest</h4> \n \
                       e.g. : dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier \n \
                       \n \
                       <h4>SVM</h4> \
                       e.g. : dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(-2,5)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html "
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
        return "Train algorithm (CV with SLOO)"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        # The name that the user will see in the toolbox

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

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

        # DISTANCE
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE,
                self.tr("Distance in pixels"),
                type=QgsProcessingParameterNumber.Integer,
                minValue=0,
                maxValue=99999,
                defaultValue=100,
            )
        )

        # MAX ITER
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXITER,
                self.tr("Maximum iteration (default : 0 e.g. class with min effective)"),
                type=QgsProcessingParameterNumber.Integer,
                minValue=0,
                maxValue=99999,
                defaultValue=0,
            )
        )
        #
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
        """Process the spatial leave-one-out training algorithm."""
        log = QgisLogger(tag="Dzetsaka/Processing/TrainSLOO")

        try:
            INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
            INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)

            INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
            # SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)
            # TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
            # INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
            OUTPUT_MODEL = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
            # OUTPUT_MATRIX = self.parameterAsFileOutput(parameters, self.OUTPUT_MATRIX, context)

            SAVEDIR = self.parameterAsFileOutput(parameters, self.SAVEDIR, context)
            # Retrieve algo from code
            extraParam = {}
            # extraParam['maxIter']=False
            # extraParam['param_grid'] = dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6))

            MAXITER = self.parameterAsInt(parameters, self.MAXITER, context)
            DISTANCE = self.parameterAsInt(parameters, self.DISTANCE, context)

            extraParam["distance"] = DISTANCE

            if MAXITER == 0:
                MAXITER = False
            extraParam["maxIter"] = MAXITER

            PARAMGRID = self.parameterAsString(parameters, self.PARAMGRID, context)
            if PARAMGRID != "":
                extraParam["param_grid"] = eval(PARAMGRID)

            if not SAVEDIR.endswith("/"):
                SAVEDIR += "/"

            extraParam["saveDir"] = SAVEDIR

            TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)

            # Retrieve algo from code
            SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]

            # eval(PARAM_GRID)

            # QgsMessageLog.logMessage(str(eval(PARAMGRID)))

            mainfunction.LearnModel(
                INPUT_RASTER.source(),
                INPUT_LAYER.source(),
                INPUT_COLUMN[0],
                OUTPUT_MODEL,
                "SLOO",
                0,
                None,
                SELECTED_ALGORITHM,
                feedback=feedback,
                extraParam=extraParam,
            )
            return {self.SAVEDIR: SAVEDIR, self.OUTPUT_MODEL: OUTPUT_MODEL}

        except Exception as e:
            error_msg = f"Spatial Leave-One-Out training failed: {e!s}"
            feedback.reportError(error_msg)
            log.exception("Spatial Leave-One-Out training algorithm failed", e)
            show_error_dialog("dzetsaka Train SLOO Error", error_msg)
            return {}

    def tr(self, string):
        """Translate string using Qt translation API."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return TrainSLOOAlgorithm()

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
        return metadata_helpers.get_help_url("sloo")

    def tags(self):
        """Returns tags for the algorithm for better searchability."""
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("training")
        return common + specific
