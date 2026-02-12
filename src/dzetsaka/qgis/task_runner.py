"""QGIS task orchestration for dzetsaka training/classification."""

from __future__ import annotations

import contextlib
import os
import re
import time
import traceback
from typing import Any, Callable

from qgis.core import Qgis, QgsMessageLog, QgsTask
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from dzetsaka import classifier_config
from dzetsaka.application.use_cases.classify_raster import run_classification
from dzetsaka.application.use_cases.train_model import run_training
from dzetsaka.scripts.classification_pipeline import (
    OptunaOptimizationError,
    PolygonCoverageInsufficientError,
)


class EnhancedProgressWidget(QWidget):
    """Enhanced progress widget showing main task, progress bar, sub-task, and time estimates.

    Displays:
    - Main task label (e.g., "Training Random Forest...")
    - Progress bar with percentage
    - Sub-task label in smaller, gray font (e.g., "Testing parameter set 3/10")
    - Time estimate label in gray font (e.g., "~45s remaining")
    """

    def __init__(self, parent=None):
        """Initialize the enhanced progress widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget

        """
        super().__init__(parent)
        self.start_time = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Main task label (bold, normal size)
        self.main_label = QLabel("Initializing...")
        font = self.main_label.font()
        font.setBold(True)
        self.main_label.setFont(font)
        layout.addWidget(self.main_label)

        # Progress bar with percentage
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        try:
            self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except AttributeError:
            self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        # Sub-task label (smaller, gray font)
        self.sub_task_label = QLabel("")
        self.sub_task_label.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addWidget(self.sub_task_label)

        # Time estimate label (gray font)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addWidget(self.time_label)

    def set_main_task(self, text):
        """Set the main task description.

        Parameters
        ----------
        text : str
            Main task description

        """
        self.main_label.setText(text)

    def set_progress(self, value, total=100, sub_task_text=""):
        """Update progress with optional sub-task information.

        Parameters
        ----------
        value : float
            Current progress value
        total : float, optional
            Total progress value (default: 100)
        sub_task_text : str, optional
            Sub-task description (default: "")

        """
        if self.start_time is None:
            self.start_time = time.time()

        # Update progress bar
        percentage = int((value / total) * 100) if total > 0 else 0
        percentage = min(100, max(0, percentage))
        self.progress_bar.setValue(percentage)

        # Update sub-task label
        if sub_task_text:
            self.sub_task_label.setText(sub_task_text)
            self.sub_task_label.setVisible(True)
        else:
            self.sub_task_label.setVisible(False)

        # Calculate and update time estimate
        if percentage > 0 and percentage < 100:
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / (percentage / 100.0)
            remaining = estimated_total - elapsed

            if remaining > 0:
                if remaining < 60:
                    time_text = f"~{int(remaining)}s remaining"
                else:
                    minutes = int(remaining / 60)
                    time_text = f"~{minutes}m remaining"

                elapsed_text = f"Elapsed: {int(elapsed)}s"
                self.time_label.setText(f"{elapsed_text} | {time_text}")
                self.time_label.setVisible(True)
            else:
                self.time_label.setVisible(False)
        elif percentage == 100:
            if self.start_time:
                elapsed = time.time() - self.start_time
                self.time_label.setText(f"Completed in {int(elapsed)}s")
                self.time_label.setVisible(True)
        else:
            self.time_label.setVisible(False)

    def reset(self):
        """Reset the widget to initial state."""
        self.start_time = time.time()
        self.progress_bar.setValue(0)
        self.sub_task_label.setText("")
        self.sub_task_label.setVisible(False)
        self.time_label.setText("")
        self.time_label.setVisible(False)


# Progress stage definitions for classification workflow
PROGRESS_STAGES = {
    "loading": {"name": "Loading data", "start": 0, "end": 5},
    "training": {"name": "Training model", "start": 5, "end": 60},
    "shap": {"name": "Computing SHAP values", "start": 60, "end": 70},
    "classifying": {"name": "Classifying raster", "start": 70, "end": 95},
    "report": {"name": "Generating report", "start": 95, "end": 100},
}


class TaskFeedbackAdapter:
    """Feedback adapter with enhanced progress reporting and stage detection.

    Parses progress messages to extract stage information and update the enhanced
    progress widget with sub-task details.
    """

    def __init__(self, task, enhanced_widget=None):
        """Initialize the feedback adapter.

        Parameters
        ----------
        task : QgsTask
            The QGIS task to report progress to
        enhanced_widget : EnhancedProgressWidget, optional
            Enhanced progress widget for detailed progress display

        """
        self.task = task
        self.enhanced_widget = enhanced_widget
        self._last_text = ""
        self._current_stage = None
        self._stage_progress = 0

    def setProgress(self, value):
        """Set progress value.

        Parameters
        ----------
        value : float
            Progress value (0-100)

        """
        with contextlib.suppress(Exception):
            self.task.setProgress(float(value))
            if self.enhanced_widget:
                self.enhanced_widget.set_progress(float(value), 100)

    def setProgressText(self, text):
        """Set progress text and parse for stage/sub-task information.

        Parameters
        ----------
        text : str
            Progress message text

        """
        message = str(text)
        if message and message != self._last_text:
            self._last_text = message
            self.task.status_message = message

            # Update task description
            if hasattr(self.task, "setDescription"):
                with contextlib.suppress(Exception):
                    self.task.setDescription(message)

            # Parse message for stage and sub-task information
            if self.enhanced_widget:
                self._parse_and_update_progress(message)

    def _parse_and_update_progress(self, message):
        """Parse progress message and update enhanced widget.

        Detects common patterns in progress messages to extract stage and sub-task
        information, then updates the enhanced progress widget accordingly.

        Parameters
        ----------
        message : str
            Progress message to parse

        """
        message_lower = message.lower()

        # Detect stage based on message content
        stage = None
        sub_task = ""

        # Loading stage
        if any(keyword in message_lower for keyword in ["loading", "reading", "opening"]):
            stage = "loading"
            sub_task = message

        # Training stage
        elif any(keyword in message_lower for keyword in ["learning", "training", "fitting"]):
            stage = "training"

            # Check for Optuna optimization patterns
            trial_match = re.search(r"trial[s]?\s+(\d+)/(\d+)", message_lower)
            param_match = re.search(r"parameter[s]?\s+set[s]?\s+(\d+)/(\d+)", message_lower)

            if trial_match:
                current, total = trial_match.groups()
                sub_task = f"Testing trial {current}/{total}"
                # Calculate progress within training stage
                trial_progress = (int(current) / int(total)) * 100
                stage_range = PROGRESS_STAGES["training"]["end"] - PROGRESS_STAGES["training"]["start"]
                progress = PROGRESS_STAGES["training"]["start"] + (trial_progress * stage_range / 100)
                self.enhanced_widget.set_progress(progress, 100, sub_task)
                return
            if param_match:
                current, total = param_match.groups()
                sub_task = f"Testing parameter set {current}/{total}"
                # Calculate progress within training stage
                param_progress = (int(current) / int(total)) * 100
                stage_range = PROGRESS_STAGES["training"]["end"] - PROGRESS_STAGES["training"]["start"]
                progress = PROGRESS_STAGES["training"]["start"] + (param_progress * stage_range / 100)
                self.enhanced_widget.set_progress(progress, 100, sub_task)
                return
            sub_task = message

        # SHAP stage
        elif any(keyword in message_lower for keyword in ["shap", "explainability", "explaining"]):
            stage = "shap"
            sub_task = message

        # Classification/prediction stage
        elif any(keyword in message_lower for keyword in ["predicting", "classifying", "inference"]):
            stage = "classifying"

            # Check for batch processing patterns
            band_match = re.search(r"(\d+)-band image", message_lower)
            sub_task = f"Processing {band_match.group(1)}-band image" if band_match else message

        # Report generation stage
        elif any(keyword in message_lower for keyword in ["report", "generating", "writing output"]):
            stage = "report"
            sub_task = message

        # Update progress based on detected stage
        if stage and stage in PROGRESS_STAGES:
            self._current_stage = stage
            stage_info = PROGRESS_STAGES[stage]

            # Set main task label
            self.enhanced_widget.set_main_task(stage_info["name"] + "...")

            # Use middle of stage range as default progress if no specific progress calculated
            default_progress = (stage_info["start"] + stage_info["end"]) / 2
            self.enhanced_widget.set_progress(default_progress, 100, sub_task)


class ClassificationTask(QgsTask):
    """QGIS background task for training + classification with enhanced progress reporting."""

    def __init__(
        self,
        description: str,
        *,
        do_training: bool,
        raster_path: str,
        vector_path: str | None,
        class_field: str | None,
        model_path: str,
        split_config: Any,
        random_seed: int,
        matrix_path: str | None,
        classifier: str,
        output_path: str,
        mask_path: str | None,
        confidence_map: str | None,
        nodata: int,
        extra_params: dict | None,
        on_success: Callable[[str, str], None],
        on_error: Callable[[str, str], None],
        enhanced_widget: EnhancedProgressWidget | None = None,
    ):
        """Initialize the classification task.

        Parameters
        ----------
        description : str
            Task description
        do_training : bool
            Whether to perform training
        raster_path : str
            Path to input raster
        vector_path : str, optional
            Path to training vector
        class_field : str, optional
            Field name containing class labels
        model_path : str
            Path to save/load model
        split_config : Any
            Train/validation split configuration
        random_seed : int
            Random seed for reproducibility
        matrix_path : str, optional
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
        on_success : callable
            Callback for successful completion
        on_error : callable
            Callback for error
        enhanced_widget : EnhancedProgressWidget, optional
            Enhanced progress widget for detailed progress display

        """
        super().__init__(description)
        self.do_training = do_training
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.class_field = class_field
        self.model_path = model_path
        self.split_config = split_config
        self.random_seed = random_seed
        self.matrix_path = matrix_path
        self.classifier = classifier
        self.output_path = output_path
        self.mask_path = mask_path
        self.confidence_map = confidence_map
        self.nodata = nodata
        self.extra_params = extra_params or {}
        self.on_success = on_success
        self.on_error = on_error
        self.enhanced_widget = enhanced_widget
        self.error_title = ""
        self.error_message = ""
        self.status_message = ""

    def _set_error(self, title: str, message: str) -> None:
        self.error_title = title
        self.error_message = message

    def run(self) -> bool:
        """Execute training and/or classification in the background task.

        Returns
        -------
        bool
            True if successful, False otherwise

        """
        try:
            classifier_name = classifier_config.get_classifier_name(self.classifier)
            feedback = TaskFeedbackAdapter(self, self.enhanced_widget)

            # Initialize enhanced widget if provided
            if self.enhanced_widget:
                self.enhanced_widget.reset()
                self.enhanced_widget.set_main_task(f"Initializing {classifier_name}...")

            if self.do_training:
                self.status_message = f"Training {classifier_name}..."
                self.setProgress(1)

                if self.enhanced_widget:
                    self.enhanced_widget.set_main_task(f"Training {classifier_name}...")
                    self.enhanced_widget.set_progress(
                        PROGRESS_STAGES["training"]["start"], 100, "Preparing training data...",
                    )

                run_training(
                    raster_path=self.raster_path,
                    vector_path=self.vector_path,
                    class_field=self.class_field,
                    model_path=self.model_path,
                    split_config=self.split_config,
                    random_seed=self.random_seed,
                    matrix_path=self.matrix_path,
                    classifier=self.classifier,
                    extra_params=self.extra_params,
                    feedback=feedback,
                )

                if self.isCanceled():
                    return False

                if not self.model_path or not os.path.exists(self.model_path):
                    self._set_error(
                        "dzetsaka Training Error",
                        (
                            "Training did not produce a valid model file.<br><br>"
                            f"Expected model path: <code>{self.model_path}</code><br><br>"
                            "This often happens when a dependency failed to initialize. "
                            "If dependencies were installed in this session, restart QGIS and retry."
                        ),
                    )
                    return False

            if self.isCanceled():
                return False

            self.status_message = f"Running inference with {classifier_name}..."
            self.setProgress(max(self.progress(), PROGRESS_STAGES["classifying"]["start"]))

            if self.enhanced_widget:
                self.enhanced_widget.set_main_task("Classifying raster...")
                self.enhanced_widget.set_progress(
                    PROGRESS_STAGES["classifying"]["start"], 100, "Loading model and preparing data...",
                )

            prediction_result = run_classification(
                raster_path=self.raster_path,
                model_path=self.model_path,
                output_path=self.output_path,
                mask_path=self.mask_path,
                confidence_map=self.confidence_map,
                nodata=self.nodata,
                feedback=feedback,
            )
            if prediction_result is None:
                self._set_error(
                    "dzetsaka Classification Error",
                    (
                        "Classification failed: model prediction did not complete.<br><br>"
                        "Check the QGIS log for root cause details. "
                        "If dependencies were installed during this session, restart QGIS first."
                    ),
                )
                return False

            self.setProgress(100)
            if self.enhanced_widget:
                self.enhanced_widget.set_progress(100, 100, "Complete!")

            return True
        except OptunaOptimizationError as exc:
            error_message = str(exc)
            QgsMessageLog.logMessage(
                f"Optuna optimization error: {error_message}",
                "Dzetsaka",
                Qgis.Warning,
            )
            self._set_error("dzetsaka Optuna Optimization Failed", error_message)
            return False

        except PolygonCoverageInsufficientError as exc:
            error_message = str(exc)
            QgsMessageLog.logMessage(
                f"Spatial CV canceled: {error_message}",
                "Dzetsaka",
                Qgis.Warning,
            )
            self._set_error("dzetsaka Spatial CV Error", error_message)
            return False

        except Exception as exc:
            # Get full traceback for detailed error reporting
            tb_str = traceback.format_exc()
            error_msg = f"Unexpected error: {exc!s}\n\nFull traceback:\n{tb_str}"

            # Log to QGIS message log with full traceback
            QgsMessageLog.logMessage(f"Classification task error:\n{tb_str}", "Dzetsaka", Qgis.Critical)

            self._set_error("dzetsaka Error", error_msg)
            return False

    def finished(self, result: bool) -> None:
        """Handle task completion in the main thread."""
        if result:
            self.on_success(self.output_path, self.confidence_map or "")
            return

        if self.isCanceled():
            self.on_error(
                "dzetsaka Task Cancelled",
                "Training/classification task was cancelled.",
            )
            return

        title = self.error_title or "dzetsaka Task Error"
        message = self.error_message or "Background classification task failed."
        self.on_error(title, message)
