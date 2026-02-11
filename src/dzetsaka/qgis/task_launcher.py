"""Launch and lifecycle handling for background classification tasks."""

from __future__ import annotations

import os
from datetime import datetime

from qgis.core import QgsApplication, QgsTask
from qgis.PyQt.QtWidgets import QMessageBox

from dzetsaka import classifier_config
from dzetsaka.qgis.task_runner import ClassificationTask


def start_classification_task(
    plugin,
    *,
    description,
    do_training,
    raster_path,
    vector_path,
    class_field,
    model_path,
    split_config,
    random_seed,
    matrix_path,
    classifier,
    output_path,
    mask_path,
    confidence_map,
    nodata,
    extra_params,
    error_context,
    success_prefix,
) -> None:
    """Submit a background classification task to the QGIS task manager."""
    if plugin._active_classification_task is not None:
        task_active = False
        try:
            status = plugin._active_classification_task.status()
            try:
                done_statuses = {QgsTask.TaskStatus.Complete, QgsTask.TaskStatus.Terminated}
            except AttributeError:
                done_statuses = {QgsTask.Complete, QgsTask.Terminated}
            task_active = status not in done_statuses
        except Exception:
            task_active = False

        if task_active:
            QMessageBox.information(
                plugin.iface.mainWindow(),
                "Task Already Running",
                "A dzetsaka classification task is already running in the QGIS Task Manager. "
                "Please wait for it to finish before starting a new one.",
                QMessageBox.StandardButton.Ok,
            )
            return

    def on_task_error(title, message):
        plugin._active_classification_task = None
        config_info = plugin._get_debug_info()
        plugin.log.error(f"{title}: {message}")
        plugin.log.info("Configuration for issue reporting:")
        plugin.log.info(config_info)
        if "Optuna" in title:
            QMessageBox.warning(
                plugin.iface.mainWindow(),
                title,
                message,
                QMessageBox.StandardButton.Ok,
            )
            return
        plugin._show_github_issue_popup(
            error_title=title,
            error_type="Runtime Error",
            error_message=message,
            context=error_context,
        )

    # Store task reference for callbacks
    task_ref = {"task": None}

    def on_task_success(out_raster, out_confidence):
        plugin.log.info(f"[{success_prefix}] Classification completed successfully")
        plugin.iface.addRasterLayer(out_raster)
        if out_confidence:
            plugin.iface.addRasterLayer(out_confidence)

        # Collect results data for the results explorer
        current_task = task_ref.get("task") or plugin._active_classification_task
        result_data = _build_result_data(
            classifier=classifier,
            output_path=out_raster,
            input_path=raster_path,
            matrix_path=matrix_path,
            model_path=model_path,
            confidence_path=out_confidence,
            extra_params=extra_params,
            start_time=getattr(current_task, "_start_time", None) if current_task else None,
        )

        # Open results explorer dock
        try:
            from ui.results_explorer_dock import open_results_explorer

            dock = open_results_explorer(result_data, parent=plugin.iface.mainWindow(), iface=plugin.iface)
            if dock and plugin.iface:
                plugin.iface.addDockWidget(1, dock)  # Qt.RightDockWidgetArea = 2, Qt.LeftDockWidgetArea = 1
        except Exception as e:
            plugin.log.warning(f"Failed to open results explorer: {e}")

        # Clean up reference
        plugin._active_classification_task = None

    task = ClassificationTask(
        description,
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
        on_success=on_task_success,
        on_error=on_task_error,
    )
    # Store start time for runtime calculation
    task._start_time = datetime.now()

    # Store task reference for callbacks
    task_ref["task"] = task
    plugin._active_classification_task = task
    QgsApplication.taskManager().addTask(task)
    plugin.log.info(f"Task submitted to QGIS task manager: {description}")


def _build_result_data(
    classifier: str,
    output_path: str,
    input_path: str,
    matrix_path: str | None,
    model_path: str,
    confidence_path: str | None,
    extra_params: dict,
    start_time: datetime | None,
) -> dict:
    """Build result data dictionary for results explorer.

    Parameters
    ----------
    classifier : str
        Classifier code (e.g., "RF", "XGB")
    output_path : str
        Path to output classification raster
    input_path : str
        Path to input raster
    matrix_path : str, optional
        Path to confusion matrix CSV
    model_path : str
        Path to saved model
    confidence_path : str, optional
        Path to confidence map
    extra_params : dict
        Extra parameters including report paths
    start_time : datetime, optional
        Task start time for runtime calculation

    Returns
    -------
    dict
        Result data dictionary for ResultsExplorerDock

    """
    result_data = {
        "algorithm": classifier_config.get_classifier_name(classifier),
        "output_path": output_path,
        "input_path": input_path,
        "model_path": model_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Calculate runtime
    if start_time:
        runtime = (datetime.now() - start_time).total_seconds()
        result_data["runtime_seconds"] = runtime
    else:
        result_data["runtime_seconds"] = 0.0

    # Add matrix path if it exists
    if matrix_path and os.path.exists(matrix_path):
        result_data["matrix_path"] = matrix_path

    # Add confidence map if available
    if confidence_path:
        result_data["confidence_path"] = confidence_path

    # Look for SHAP visualization
    report_dir = extra_params.get("REPORT_OUTPUT_DIR", "")
    if report_dir and os.path.exists(report_dir):
        # Look for SHAP images
        for filename in os.listdir(report_dir):
            if "shap" in filename.lower() and filename.endswith((".png", ".jpg", ".jpeg")):
                result_data["shap_path"] = os.path.join(report_dir, filename)
                break

        # Look for HTML report
        for filename in os.listdir(report_dir):
            if filename.endswith(".html"):
                result_data["report_path"] = os.path.join(report_dir, filename)
                break

    # Class counts will be computed on-demand by the dock if not provided
    result_data["class_counts"] = {}

    return result_data

