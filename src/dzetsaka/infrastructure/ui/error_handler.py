"""Enhanced error handling with user-friendly guidance.

Provides detailed error messages with actionable recommendations for common issues.

Author:
    Nicolas Karasiak
"""

from typing import Optional, Tuple

try:
    from qgis.PyQt.QtWidgets import QMessageBox

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


class ErrorGuide:
    """User-friendly error messages with actionable guidance."""

    @staticmethod
    def no_training_vector_selected(parent=None) -> None:
        """Show error when no training vector is selected."""
        if not _QT_AVAILABLE:
            return

        QMessageBox.information(
            parent,
            "No Training Vector Selected",
            "<h3>Please select a training vector layer</h3>"
            "<p>The training vector contains your labeled samples for training the classifier.</p>"
            "<p><b>How to fix:</b></p>"
            "<ul>"
            "<li>Click the <b>Browse</b> button to select a shapefile or GeoPackage</li>"
            "<li>Or select a loaded layer from the dropdown (if using QGIS)</li>"
            "</ul>"
            "<p><i>Supported formats:</i> Shapefile (.shp), GeoPackage (.gpkg)</p>",
        )

    @staticmethod
    def no_class_field_selected(parent=None) -> None:
        """Show error when no class field is selected."""
        if not _QT_AVAILABLE:
            return

        QMessageBox.information(
            parent,
            "No Class Field Selected",
            "<h3>Please select a class label field</h3>"
            "<p>The class field contains the numeric codes identifying each class in your training data.</p>"
            "<p><b>How to fix:</b></p>"
            "<ol>"
            "<li>Select your training vector first</li>"
            "<li>Choose the field containing class codes from the dropdown</li>"
            "</ol>"
            "<p><b>Requirements:</b></p>"
            "<ul>"
            "<li>Field must contain <b>numeric values</b> (integers)</li>"
            "<li>Example values: 1 (Forest), 2 (Water), 3 (Urban)</li>"
            "<li>Do NOT use text labels ('forest', 'water')</li>"
            "</ul>",
        )

    @staticmethod
    def insufficient_training_samples(class_name: str, sample_count: int, minimum: int = 30, parent=None) -> None:
        """Show error for insufficient training samples.

        Parameters
        ----------
        class_name : str
            Name or code of the class
        sample_count : int
            Number of samples found
        minimum : int
            Minimum required samples
        parent : QWidget, optional
            Parent widget
        """
        if not _QT_AVAILABLE:
            return

        QMessageBox.critical(
            parent,
            "Insufficient Training Samples",
            f"<h3>Not enough training samples for class '{class_name}'</h3>"
            f"<p>Found <b>{sample_count}</b> samples, but need at least <b>{minimum}</b>.</p>"
            "<p><b>Why this matters:</b></p>"
            "<p>With too few samples, the model cannot learn the characteristics of this class, "
            "leading to poor classification accuracy.</p>"
            "<p><b>How to fix:</b></p>"
            "<ol>"
            "<li><b>Collect more samples:</b> Add at least {need_more} more labeled samples for this class</li>"
            "<li><b>Merge classes:</b> If appropriate, combine similar classes</li>"
            "<li><b>Remove class:</b> If this class is not important, remove it from training data</li>"
            "</ol>"
            "<p><i>Recommended:</i> Aim for 50-100 samples per class for reliable results.</p>".format(
                need_more=minimum - sample_count
            ),
        )

    @staticmethod
    def severe_class_imbalance(
        dominant_class: str, dominant_count: int, minority_class: str, minority_count: int, ratio: float, parent=None
    ) -> None:
        """Show warning for severe class imbalance.

        Parameters
        ----------
        dominant_class : str
            Name of class with most samples
        dominant_count : int
            Number of samples in dominant class
        minority_class : str
            Name of class with fewest samples
        minority_count : int
            Number of samples in minority class
        ratio : float
            Imbalance ratio
        parent : QWidget, optional
            Parent widget
        """
        if not _QT_AVAILABLE:
            return

        QMessageBox.warning(
            parent,
            "Severe Class Imbalance Detected",
            f"<h3>Your training data has severe class imbalance</h3>"
            f"<p>Class '<b>{dominant_class}</b>' has <b>{dominant_count}</b> samples<br>"
            f"Class '<b>{minority_class}</b>' has only <b>{minority_count}</b> samples<br>"
            f"<b>Imbalance ratio: {ratio:.1f}:1</b></p>"
            "<p><b>Why this matters:</b></p>"
            "<p>The classifier will be biased toward the dominant class and may ignore the minority class.</p>"
            "<p><b>How to fix:</b></p>"
            "<ol>"
            "<li><b>Balance manually:</b> Add more samples for minority classes or remove some from dominant class</li>"
            "<li><b>Use SMOTE:</b> Enable 'Use SMOTE' in Advanced Options to synthetically balance classes</li>"
            "<li><b>Adjust class weights:</b> Some algorithms support weighted classes (see algorithm documentation)</li>"
            "</ol>"
            "<p><b>Quick fix:</b> Check the 'Use SMOTE' checkbox in Advanced Options before running classification.</p>",
        )

    @staticmethod
    def model_file_not_found(model_path: str, parent=None) -> None:
        """Show error when model file is not found.

        Parameters
        ----------
        model_path : str
            Path to missing model file
        parent : QWidget, optional
            Parent widget
        """
        if not _QT_AVAILABLE:
            return

        QMessageBox.critical(
            parent,
            "Model File Not Found",
            f"<h3>Cannot find model file</h3>"
            f"<p><b>Path:</b><br><code>{model_path}</code></p>"
            "<p><b>Possible causes:</b></p>"
            "<ul>"
            "<li>File was moved or deleted</li>"
            "<li>File path contains invalid characters</li>"
            "<li>Network drive is disconnected</li>"
            "<li>Insufficient permissions to access file</li>"
            "</ul>"
            "<p><b>How to fix:</b></p>"
            "<ol>"
            "<li>Verify the file exists at the specified location</li>"
            "<li>Click 'Browse Model' to select the correct file</li>"
            "<li>If the model was deleted, retrain and save a new model</li>"
            "</ol>",
        )

    @staticmethod
    def band_count_mismatch(raster_path: str, raster_bands: int, model_bands: int, parent=None) -> None:
        """Show error when raster band count doesn't match model.

        Parameters
        ----------
        raster_path : str
            Path to raster file
        raster_bands : int
            Number of bands in raster
        model_bands : int
            Number of bands model expects
        parent : QWidget, optional
            Parent widget
        """
        if not _QT_AVAILABLE:
            return

        QMessageBox.critical(
            parent,
            "Band Count Mismatch",
            f"<h3>Raster band count doesn't match model</h3>"
            f"<p><b>Raster:</b> {raster_path}<br>"
            f"<b>Raster bands:</b> {raster_bands}<br>"
            f"<b>Model expects:</b> {model_bands} bands</p>"
            "<p><b>Why this happens:</b></p>"
            "<p>The model was trained on imagery with a different number of bands. "
            "Models can only classify rasters with the same band configuration.</p>"
            "<p><b>How to fix:</b></p>"
            "<ol>"
            "<li><b>Use matching raster:</b> Find a raster with {model_bands} bands</li>"
            "<li><b>Retrain model:</b> Train a new model using {raster_bands}-band imagery</li>"
            "<li><b>Stack/subset bands:</b> Use QGIS to create a {model_bands}-band raster</li>"
            "</ol>"
            "<p><i>Tip:</i> Always use the same satellite/sensor for training and classification.</p>",
        )

    @staticmethod
    def dependency_missing(dependency_name: str, feature_name: str, parent=None) -> Tuple[bool, str]:
        """Show error for missing dependency with install option.

        Parameters
        ----------
        dependency_name : str
            Name of missing dependency (e.g., 'XGBoost', 'optuna')
        feature_name : str
            Feature that requires the dependency
        parent : QWidget, optional
            Parent widget

        Returns
        -------
        bool
            True if user wants to install
        str
            Installation command or empty string
        """
        if not _QT_AVAILABLE:
            return False, ""

        dependency_info = {
            "xgboost": ("XGBoost", "pip install xgboost"),
            "catboost": ("CatBoost", "pip install catboost"),
            "optuna": ("Optuna", "pip install optuna"),
            "shap": ("SHAP", "pip install shap"),
            "imblearn": ("imbalanced-learn", "pip install imbalanced-learn"),
        }

        dep_lower = dependency_name.lower()
        full_name, install_cmd = dependency_info.get(dep_lower, (dependency_name, f"pip install {dependency_name}"))

        reply = QMessageBox.question(
            parent,
            f"{full_name} Required",
            f"<h3>{feature_name} requires {full_name}</h3>"
            f"<p><b>{full_name}</b> is not installed in your Python environment.</p>"
            "<p><b>Would you like to install it now?</b></p>"
            f"<p><i>Installation command:</i><br><code>{install_cmd}</code></p>"
            "<p><b>Note:</b> Installation may take a few minutes. "
            "The plugin will attempt to install automatically.</p>",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        should_install = reply == QMessageBox.StandardButton.Yes
        return should_install, install_cmd if should_install else ""

    @staticmethod
    def show_helpful_error(error_type: str, **kwargs) -> None:
        """Show a helpful error message based on error type.

        Parameters
        ----------
        error_type : str
            Type of error (key for error messages)
        **kwargs : dict
            Additional parameters for the error message
        """
        error_handlers = {
            "no_vector": ErrorGuide.no_training_vector_selected,
            "no_class_field": ErrorGuide.no_class_field_selected,
            "insufficient_samples": ErrorGuide.insufficient_training_samples,
            "class_imbalance": ErrorGuide.severe_class_imbalance,
            "model_not_found": ErrorGuide.model_file_not_found,
            "band_mismatch": ErrorGuide.band_count_mismatch,
        }

        handler = error_handlers.get(error_type)
        if handler:
            parent = kwargs.pop("parent", None)
            handler(parent=parent, **kwargs)
        else:
            # Fallback for unknown error types
            if _QT_AVAILABLE:
                QMessageBox.warning(kwargs.get("parent"), "Error", f"An error occurred: {error_type}")


# Convenience function
def show_error_with_help(error_type: str, parent=None, **kwargs):
    """Convenience function to show error with help.

    Parameters
    ----------
    error_type : str
        Type of error
    parent : QWidget, optional
        Parent widget
    **kwargs : dict
        Additional error parameters
    """
    ErrorGuide.show_helpful_error(error_type, parent=parent, **kwargs)
