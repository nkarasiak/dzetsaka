"""SHAP Model Explanation Algorithm for dzetsaka.

This module provides the processing algorithm for generating SHAP-based
feature importance maps from trained models.
"""

import os
import pickle  # nosec B403

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon

from dzetsaka.logging import show_error_dialog
from dzetsaka.qgis.processing import metadata_helpers

# Try to import SHAP explainer
try:
    from dzetsaka.scripts.explainability.shap_explainer import SHAP_AVAILABLE, ModelExplainer
except ImportError:
    SHAP_AVAILABLE = False
    ModelExplainer = None

plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 7)))


class ExplainModelAlgorithm(QgsProcessingAlgorithm):
    """Generate SHAP feature importance map from trained model.

    This algorithm takes a trained dzetsaka model and a raster image,
    and produces a feature importance map showing which bands/features
    are most important for the classification.
    """

    INPUT_MODEL = "INPUT_MODEL"
    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_IMPORTANCE = "OUTPUT_IMPORTANCE"
    SAMPLE_SIZE = "SAMPLE_SIZE"

    def shortHelpString(self):
        """Return the short help string for this algorithm."""
        return self.tr(
            """Generate feature importance map using SHAP (SHapley Additive exPlanations).

<h3>What is SHAP?</h3>
SHAP explains model predictions by computing the contribution of each feature
(band) to the model's output. This helps you understand which bands are most
important for classification.

<h3>Requirements</h3>
- SHAP library must be installed: pip install shap>=0.41.0
- A trained dzetsaka model (.model file)
- The same raster image used for training

<h3>Output</h3>
Multi-band raster where each band shows the importance (0-100) of the
corresponding input band. Higher values = more important for classification.

<h3>Sample Size</h3>
Number of pixels to sample for SHAP computation. Larger values are more
accurate but slower. Recommended: 500-2000 for most cases.

<h3>Supported Algorithms</h3>
All dzetsaka algorithms are supported:
- Tree models (RF, XGB, ET, GBC): Fast TreeExplainer
- Other models (SVM, KNN, LR, NB, MLP): Slower KernelExplainer

<h3>Performance</h3>
- Random Forest: ~10-30 seconds
- XGBoost: ~15-40 seconds
- SVM/MLP: ~2-5 minutes (slower but provides insights)

<h3>Tips</h3>
- Start with sample_size=500 for quick testing
- Use sample_size=1000-2000 for production
- Visualize output in QGIS with "Singleband pseudocolor" style
- Darker/higher values = more important features
""",
        )

    def name(self):
        """Returns the algorithm name, used for identifying the algorithm.

        This string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Explain model (SHAP)"

    def icon(self):
        """Return the algorithm icon."""
        return QIcon(os.path.join(plugin_path, "icon.png"))

    def initAlgorithm(self, config=None):
        """Initialize the algorithm parameters."""
        # Input model file
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_MODEL,
                self.tr("Trained model file (.model)"),
                extension="model",
            ),
        )

        # Input raster (same one used for training)
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr("Input raster")))

        # Sample size for SHAP computation
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAMPLE_SIZE,
                self.tr("Number of pixels to sample for SHAP computation"),
                type=QgsProcessingParameterNumber.Integer,
                minValue=100,
                maxValue=10000,
                defaultValue=1000,
            ),
        )

        # Output importance raster
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT_IMPORTANCE, self.tr("Output feature importance raster")),
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Process the algorithm with given parameters."""
        # Check if SHAP is available
        if not SHAP_AVAILABLE:
            feedback.reportError(
                "SHAP library is not installed. "
                "Please install it using: pip install shap>=0.41.0\n"
                "Or install dzetsaka with explainability support: pip install dzetsaka[explainability]",
            )
            show_error_dialog(
                "dzetsaka Explain Model Error",
                "SHAP library is not installed. Install shap>=0.41.0 or dzetsaka[explainability].",
            )
            return {}

        # Get parameters
        model_path = self.parameterAsFile(parameters, self.INPUT_MODEL, context)
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        sample_size = self.parameterAsInt(parameters, self.SAMPLE_SIZE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_IMPORTANCE, context)

        # Log parameters
        feedback.pushInfo(f"Model file: {model_path}")
        feedback.pushInfo(f"Input raster: {raster_layer.source()}")
        feedback.pushInfo(f"Sample size: {sample_size}")
        feedback.pushInfo(f"Output path: {output_path}")

        # Load model
        feedback.pushInfo("Loading model...")
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)  # nosec B301

            # Extract model components
            if isinstance(model_data, (list, tuple)) and len(model_data) >= 4:
                model = model_data[0]
                model_data[1]  # Max scaling values (not used for SHAP)
                model_data[2]  # Min scaling values (not used for SHAP)
                classifier_code = model_data[3]
                feedback.pushInfo(f"Loaded {classifier_code} model successfully")
            else:
                feedback.reportError("Invalid model file format")
                show_error_dialog("dzetsaka Explain Model Error", "Invalid model file format.")
                return {}

        except FileNotFoundError:
            feedback.reportError(f"Model file not found: {model_path}")
            show_error_dialog("dzetsaka Explain Model Error", f"Model file not found: {model_path}")
            return {}
        except Exception as e:
            feedback.reportError(f"Failed to load model: {e!s}")
            show_error_dialog("dzetsaka Explain Model Error", f"Failed to load model: {e!s}")
            return {}

        # Get raster information
        raster_path = raster_layer.source()
        n_bands = raster_layer.bandCount()
        feedback.pushInfo(f"Raster has {n_bands} bands")

        # Generate feature names
        feature_names = [f"Band_{i + 1}" for i in range(n_bands)]

        # Create ModelExplainer
        feedback.pushInfo("Creating SHAP explainer...")
        try:
            explainer = ModelExplainer(
                model=model,
                feature_names=feature_names,
                background_data=None,  # Will be sampled from raster
            )
        except Exception as e:
            feedback.reportError(f"Failed to create explainer: {e!s}")
            show_error_dialog("dzetsaka Explain Model Error", f"Failed to create explainer: {e!s}")
            return {}

        # Generate importance raster
        feedback.pushInfo("Computing SHAP feature importance...")
        feedback.pushInfo("This may take a few minutes depending on model complexity and sample size...")

        try:
            # Create progress callback
            def progress_callback(pct):
                feedback.setProgress(int(pct))

            # Generate importance raster
            explainer.create_importance_raster(
                raster_path=raster_path,
                output_path=output_path,
                sample_size=sample_size,
                progress_callback=progress_callback,
            )

            feedback.pushInfo(f"Feature importance raster created: {output_path}")
            feedback.pushInfo("\nTo visualize in QGIS:")
            feedback.pushInfo("1. Load the output raster")
            feedback.pushInfo("2. Right-click > Properties > Symbology")
            feedback.pushInfo("3. Choose 'Singleband pseudocolor'")
            feedback.pushInfo("4. Higher values (lighter colors) = more important features")

            return {self.OUTPUT_IMPORTANCE: output_path}

        except Exception as e:
            feedback.reportError(f"SHAP computation failed: {e!s}")
            feedback.reportError("This may be due to:")
            feedback.reportError("- Insufficient memory (try reducing sample_size)")
            feedback.reportError("- Model incompatibility")
            feedback.reportError("- Raster dimension mismatch")
            show_error_dialog(
                "dzetsaka Explain Model Error",
                "SHAP computation failed. This may be due to insufficient memory, model incompatibility, "
                "or raster dimension mismatch. Check the QGIS log for details.",
            )
            return {}

    def tr(self, string):
        """Translate string using Qt's translation system."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        """Create a new instance of this algorithm."""
        return ExplainModelAlgorithm()

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
        return metadata_helpers.get_help_url("explain_model")

    def tags(self):
        """Returns tags for the algorithm for better searchability."""
        common = metadata_helpers.get_common_tags()
        specific = metadata_helpers.get_algorithm_specific_tags("explainability")
        return common + specific
