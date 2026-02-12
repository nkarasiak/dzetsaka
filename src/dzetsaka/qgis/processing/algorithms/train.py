"""Training Algorithm for dzetsaka."""

import ast
import os

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
)

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon

from dzetsaka import classifier_config
from dzetsaka.application.use_cases.train_model import run_training
from dzetsaka.logging import show_error_dialog
from dzetsaka.qgis.logging import QgisLogger
from dzetsaka.qgis.processing import metadata_helpers
from dzetsaka.scripts.function_dataraster import get_layer_source_path

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 7)))


class TrainAlgorithm(QgsProcessingAlgorithm):
    """Main training algorithm for machine learning model training.

    Provides training functionality for various classification algorithms
    including hyperparameter optimization and model validation.
    """

    INPUT_RASTER = "INPUT_RASTER"
    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = classifier_config.UI_DISPLAY_NAMES
    TRAIN_ALGORITHMS_CODE = classifier_config.CLASSIFIER_CODES
    SPLIT_PERCENT = "SPLIT_PERCENT"
    OUTPUT_MODEL = "OUTPUT_MODEL"
    OUTPUT_MATRIX = "OUTPUT_MATRIX"
    PARAMGRID = "PARAMGRID"

    def shortHelpString(self):
        """Return the short help string for this algorithm."""
        return self.tr(
            "Train classifier.\n \n \
                       Parameters for Cross Validation can be fit using a dictionnary.\n \
                       \
                       <h3>Classifier (paramgrid)</h3> \n \
                       Param grid can be fit for each algorithm : \n \
                       <h4>Random-Forest</h4> \n \
                       e.g. : dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier \n \
                       \n \
                       <h4>SVM</h4> \
                       e.g. : dict(gamma=2.0**np.arange(-4,4), C=10.0**np.arange(-2,5)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html "
        )

    def name(self):
        """Returns the algorithm name, used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Train algorithm"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        # The name that the user will see in the toolbox

        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

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

        # Train algorithm

        self.addParameter(QgsProcessingParameterEnum(self.TRAIN, "Select algorithm to train", self.TRAIN_ALGORITHMS, 0))

        # SPLIT %

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SPLIT_PERCENT,
                self.tr("Pixels (%) to keep for validation."),
                type=QgsProcessingParameterNumber.Integer,
                minValue=0,
                maxValue=100,
                defaultValue=50,
            )
        )

        # SAVE AS
        # SAVE MODEL
        self.addParameter(
            QgsProcessingParameterFileDestination(self.OUTPUT_MODEL, self.tr("Output model (to use for classifying)"))
        )

        # SAVE CONFUSION MATRIX
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MATRIX, self.tr("Output confusion matrix"), fileFilter="csv"
            )
        )  # ,
        # ext='csv'))
        # PARAM GRID
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAMGRID,
                self.tr("Parameters for the hyperparameters of the algorithm"),
                optional=True,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process the algorithm with given parameters."""
        log = QgisLogger(tag="Dzetsaka/Processing/Train")

        try:
            INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
            INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
            INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
            SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)

            SPLIT_PERCENT = 100 - SPLIT_PERCENT  # if 30 means 30% of valid per class, 70% of train

            TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
            OUTPUT_MODEL = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
            OUTPUT_MATRIX = self.parameterAsFileOutput(parameters, self.OUTPUT_MATRIX, context)

            # Retrieve algo from code
            SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]
            log.info(str(SELECTED_ALGORITHM))

            PARAMGRID = self.parameterAsString(parameters, self.PARAMGRID, context)
            if PARAMGRID != "":
                extraParam = {}
                try:
                    # Use ast.literal_eval() for safe evaluation of parameter grid
                    # Only supports Python literals (dict, list, str, int, float, bool, None)
                    extraParam["param_grid"] = ast.literal_eval(PARAMGRID)
                except (ValueError, SyntaxError) as e:
                    error_msg = (
                        f"Invalid parameter grid syntax: {e}\n"
                        f"Expected a Python dictionary literal, e.g.: "
                        f"{{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]}}"
                    )
                    feedback.reportError(error_msg)
                    show_error_dialog("dzetsaka Train Error", error_msg)
                    return {}
            else:
                extraParam = None

            if SELECTED_ALGORITHM == "RF" or SELECTED_ALGORITHM == "SVM" or SELECTED_ALGORITHM == "KNN":
                import importlib.util

                if importlib.util.find_spec("joblib") is None:
                    error_msg = (
                        "Missing dependency: joblib. "
                        "Please install joblib package (e.g., pip install joblib or your system's package manager)"
                    )
                    feedback.reportError(error_msg)
                    show_error_dialog("dzetsaka Train Error", error_msg)
                    return {}
                if importlib.util.find_spec("sklearn") is None:
                    error_msg = "You need to install scikit-learn library and its dependencies"
                    feedback.reportError(error_msg)
                    show_error_dialog("dzetsaka Train Error", error_msg)
                    return {}

            run_training(
                raster_path=INPUT_RASTER.source(),
                vector_path=get_layer_source_path(INPUT_LAYER),
                class_field=INPUT_COLUMN[0],
                model_path=OUTPUT_MODEL,
                split_config=SPLIT_PERCENT,
                random_seed=0,
                matrix_path=OUTPUT_MATRIX,
                classifier=SELECTED_ALGORITHM,
                extra_params=extraParam,
                feedback=feedback,
            )
            return {self.OUTPUT_MATRIX: OUTPUT_MATRIX, self.OUTPUT_MODEL: OUTPUT_MODEL}

        except Exception as e:
            error_msg = f"Training failed: {e!s}"
            feedback.reportError(error_msg)
            log.exception("Training algorithm failed", e)
            show_error_dialog("dzetsaka Train Error", error_msg)
            return {}

    def tr(self, string):
        """Translate string using Qt's translation system."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return TrainAlgorithm()

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
        return metadata_helpers.get_help_url("train")

    def tags(self):
        """Returns tags for the algorithm for better searchability."""
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("training")
        return common + specific
