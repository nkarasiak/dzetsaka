"""QGIS task orchestration for dzetsaka training/classification."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Optional

from qgis.core import QgsTask

from dzetsaka.application.use_cases.classify_raster import run_classification
from dzetsaka.application.use_cases.train_model import run_training
from dzetsaka import classifier_config


class TaskFeedbackAdapter:
    """Minimal feedback adapter compatible with training/inference hooks."""

    def __init__(self, task):
        self.task = task
        self._last_text = ""

    def setProgress(self, value):
        with contextlib.suppress(Exception):
            self.task.setProgress(float(value))

    def setProgressText(self, text):
        message = str(text)
        if message and message != self._last_text:
            self._last_text = message
            self.task.status_message = message
            if hasattr(self.task, "setDescription"):
                with contextlib.suppress(Exception):
                    self.task.setDescription(message)


class ClassificationTask(QgsTask):
    """QGIS background task for training + classification."""

    def __init__(
        self,
        description: str,
        *,
        do_training: bool,
        raster_path: str,
        vector_path: Optional[str],
        class_field: Optional[str],
        model_path: str,
        split_config: Any,
        random_seed: int,
        matrix_path: Optional[str],
        classifier: str,
        output_path: str,
        mask_path: Optional[str],
        confidence_map: Optional[str],
        nodata: int,
        extra_params: Optional[dict],
        on_success: Callable[[str, str], None],
        on_error: Callable[[str, str], None],
    ):
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
        self.error_title = ""
        self.error_message = ""
        self.status_message = ""

    def _set_error(self, title: str, message: str) -> None:
        self.error_title = title
        self.error_message = message

    def run(self) -> bool:
        """Execute training and/or classification in the background task."""
        try:
            classifier_name = classifier_config.get_classifier_name(self.classifier)
            feedback = TaskFeedbackAdapter(self)

            if self.do_training:
                self.status_message = f"Training {classifier_name}..."
                self.setProgress(1)
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
            self.setProgress(max(self.progress(), 80))
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
            return True
        except Exception as exc:
            self._set_error("dzetsaka Error", f"Unexpected error: {exc!s}")
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
