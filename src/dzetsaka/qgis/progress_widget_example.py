"""Example usage of EnhancedProgressWidget with ClassificationTask.

This module demonstrates how to integrate the enhanced progress reporting
widget with dzetsaka classification tasks.
"""

from qgis.PyQt.QtWidgets import QDialog, QPushButton, QVBoxLayout

from dzetsaka.qgis.task_runner import ClassificationTask, EnhancedProgressWidget


class ClassificationDialogWithProgress(QDialog):
    """Example dialog showing enhanced progress during classification.

    This dialog demonstrates how to:
    1. Create an EnhancedProgressWidget
    2. Pass it to a ClassificationTask
    3. Display real-time progress with sub-tasks and time estimates
    """

    def __init__(self, parent=None):
        """Initialize the dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget

        """
        super().__init__(parent)
        self.setWindowTitle("Classification Progress")
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)

        # Set up UI
        layout = QVBoxLayout(self)

        # Create the enhanced progress widget
        self.progress_widget = EnhancedProgressWidget(self)
        layout.addWidget(self.progress_widget)

        # Add cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)
        layout.addWidget(self.cancel_button)

        # Store reference to the task
        self.task = None

    def start_classification(
        self,
        do_training=True,
        raster_path="",
        vector_path="",
        class_field="",
        model_path="",
        split_config=None,
        random_seed=42,
        matrix_path="",
        classifier="3",
        output_path="",
        mask_path=None,
        confidence_map=None,
        nodata=-9999,
        extra_params=None,
    ):
        """Start a classification task with enhanced progress reporting.

        Parameters
        ----------
        do_training : bool
            Whether to perform training
        raster_path : str
            Path to input raster
        vector_path : str
            Path to training vector
        class_field : str
            Field name containing class labels
        model_path : str
            Path to save/load model
        split_config : Any
            Train/validation split configuration
        random_seed : int
            Random seed for reproducibility
        matrix_path : str
            Path to save confusion matrix
        classifier : str
            Classifier code
        output_path : str
            Path for classification output
        mask_path : str, optional
            Path to mask raster
        confidence_map : str, optional
            Path for confidence map output
        nodata : int
            No data value
        extra_params : dict, optional
            Additional classifier parameters

        """
        # Create the task with enhanced progress widget
        self.task = ClassificationTask(
            description="Classification with enhanced progress",
            do_training=do_training,
            raster_path=raster_path,
            vector_path=vector_path,
            class_field=class_field,
            model_path=model_path,
            split_config=split_config,
            random_seed=random_seed,
            matrix_path=matrix_path,
            classifier=classifier,
            output_path=output_path,
            mask_path=mask_path,
            confidence_map=confidence_map,
            nodata=nodata,
            extra_params=extra_params,
            on_success=self._on_success,
            on_error=self._on_error,
            enhanced_widget=self.progress_widget,  # Pass the progress widget
        )

        # Add task to QGIS task manager
        from qgis.core import QgsApplication

        QgsApplication.taskManager().addTask(self.task)

        # Show the dialog
        self.show()

    def _on_success(self, output_path, confidence_map):
        """Handle successful completion.

        Parameters
        ----------
        output_path : str
            Path to classification output
        confidence_map : str
            Path to confidence map (if generated)

        """
        self.cancel_button.setText("Close")
        # You can add code here to load the results into QGIS
        print(f"Classification complete! Output: {output_path}")

    def _on_error(self, title, message):
        """Handle error during classification.

        Parameters
        ----------
        title : str
            Error title
        message : str
            Error message

        """
        self.cancel_button.setText("Close")
        print(f"Error: {title} - {message}")
        # You can add code here to show error dialog

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.task and not self.task.isCanceled():
            # Cancel the task
            self.task.cancel()
            self.progress_widget.set_main_task("Cancelling...")
        else:
            # Close dialog
            self.accept()


# Example usage from existing code
def launch_classification_with_enhanced_progress(classification_params):
    """Launch classification with enhanced progress dialog.

    This is an example of how to integrate the enhanced progress widget
    into existing classification code.

    Parameters
    ----------
    classification_params : dict
        Dictionary containing all classification parameters

    """
    dialog = ClassificationDialogWithProgress()
    dialog.start_classification(**classification_params)
    return dialog


# Simpler integration: Just pass the widget to the task
def simple_integration_example():
    """Example of minimal integration with existing task creation code.

    This shows how to add enhanced progress to existing code with minimal changes.
    """
    # Create the progress widget
    progress_widget = EnhancedProgressWidget()
    progress_widget.show()

    # Create your task as usual, just add the enhanced_widget parameter
    task = ClassificationTask(
        description="My classification",
        do_training=True,
        raster_path="/path/to/raster.tif",
        vector_path="/path/to/training.shp",
        class_field="class",
        model_path="/path/to/model.pkl",
        split_config=None,
        random_seed=42,
        matrix_path="/path/to/matrix.csv",
        classifier="3",
        output_path="/path/to/output.tif",
        mask_path=None,
        confidence_map=None,
        nodata=-9999,
        extra_params=None,
        on_success=lambda out, conf: print(f"Done: {out}"),
        on_error=lambda title, msg: print(f"Error: {title}"),
        enhanced_widget=progress_widget,  # <-- Just add this line!
    )

    # Add to task manager as usual
    from qgis.core import QgsApplication

    QgsApplication.taskManager().addTask(task)

    return progress_widget, task
