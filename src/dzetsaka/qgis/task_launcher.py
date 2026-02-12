"""Launch and lifecycle handling for background classification tasks."""

from __future__ import annotations

from datetime import datetime

from qgis.core import QgsApplication, QgsTask
from qgis.PyQt.QtWidgets import QMessageBox

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

    def on_task_success(out_raster, out_confidence):
        plugin.log.info(f"[{success_prefix}] Classification completed successfully")
        plugin.iface.addRasterLayer(out_raster)
        if out_confidence:
            plugin.iface.addRasterLayer(out_confidence)

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

    plugin._active_classification_task = task
    QgsApplication.taskManager().addTask(task)
    plugin.log.info(f"Task submitted to QGIS task manager: {description}")

