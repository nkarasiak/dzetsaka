"""Input validation helpers for QGIS-triggered classification runs."""

from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QMessageBox


def validate_classification_request(
    plugin,
    *,
    raster_path,
    do_training,
    vector_path=None,
    class_field=None,
    model_path=None,
    source_label="Classification",
) -> bool:
    """Validate required inputs before launching a classification task."""
    errors = []

    raster_path = (raster_path or "").strip()
    vector_path = (vector_path or "").strip()
    class_field = (class_field or "").strip()
    model_path = (model_path or "").strip()

    if not raster_path:
        errors.append("Raster to classify is required.")
    else:
        raster_fs_path = raster_path.split("|")[0]
        if raster_fs_path and not os.path.exists(raster_fs_path):
            errors.append(f"Raster to classify was not found: {raster_fs_path}")

    if do_training:
        if not vector_path:
            errors.append("Training data (vector) is required when no model is loaded.")
        else:
            vector_fs_path = vector_path.split("|")[0]
            if vector_fs_path and not os.path.exists(vector_fs_path):
                errors.append(f"Training data (vector) was not found: {vector_fs_path}")
        if not class_field:
            errors.append("Label field is required when training a new model.")
    else:
        if not model_path:
            errors.append("A model path is required when loading an existing model.")
        elif not os.path.exists(model_path):
            errors.append(f"Model file was not found: {model_path}")

    if errors:
        details = "<br>".join(f"- {line}" for line in errors)
        QMessageBox.warning(
            plugin.iface.mainWindow(),
            f"{source_label} Input Error",
            f"Please fix the following before running:<br><br>{details}",
            QMessageBox.StandardButton.Ok,
        )
        plugin.log.warning(f"{source_label} validation failed: {' | '.join(errors)}")
        return False
    return True
