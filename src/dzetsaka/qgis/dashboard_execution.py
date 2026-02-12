"""Dashboard-driven training/classification execution flow."""

from __future__ import annotations

import os
import tempfile

from dzetsaka.infrastructure.geo.vector_split import split_vector_stratified
from dzetsaka.qgis.spatial_validation import confirm_polygon_group_split


def execute_dashboard_config(plugin, config) -> None:
    """Run training and classification driven by dashboard config dict."""
    try:
        raster_path = config.get("raster", "")
        vector_path = config.get("vector", "")
        class_field = config.get("class_field", "")
        classifier_code = str(config.get("classifier", "GMM"))
        extra_param = config.get("extraParam", None)
        if not isinstance(extra_param, dict):
            extra_param = {}

        load_model = config.get("load_model", "")
        if load_model:
            model_path = load_model
        else:
            save_model = config.get("save_model", "")
            if save_model:
                model_path = save_model
            else:
                fd, tmp_model_path = tempfile.mkstemp(suffix="." + classifier_code)
                os.close(fd)
                model_path = tmp_model_path

        matrix_path = config.get("confusion_matrix", "") or None
        split_percent = config.get("split_percent", 100)

        nodata_value = -9999
        random_seed = 0

        do_training = not load_model
        if not plugin._validate_classification_request(
            raster_path=raster_path,
            do_training=do_training,
            vector_path=vector_path if do_training else None,
            class_field=class_field if do_training else None,
            model_path=model_path if not do_training else None,
            source_label="Dashboard",
        ):
            return
        if not plugin._ensure_classifier_runtime_ready(
            classifier_code, source_label="Dashboard", fallback_to_gmm=False,
        ):
            return

        output_raster_path = config.get("output_raster", "")
        if not output_raster_path:
            temp_folder = tempfile.mkdtemp()
            output_raster_path = os.path.join(temp_folder, plugin._default_output_name(raster_path, classifier_code))

        # Reporting defaults: if enabled and folder is empty, use output-map folder/name.
        if bool(extra_param.get("GENERATE_REPORT_BUNDLE", False)):
            output_base = os.path.splitext(os.path.basename(output_raster_path))[0]
            output_dir = os.path.dirname(output_raster_path) or os.getcwd()
            if not str(extra_param.get("REPORT_OUTPUT_DIR", "")).strip():
                extra_param["REPORT_OUTPUT_DIR"] = os.path.join(output_dir, f"{output_base}_report")
            if "OPEN_REPORT_IN_BROWSER" not in extra_param:
                extra_param["OPEN_REPORT_IN_BROWSER"] = True
            if not matrix_path:
                matrix_path = os.path.join(output_dir, f"{output_base}_confusion_matrix.csv")
            try:
                split_percent_int = int(split_percent)
            except Exception:
                split_percent_int = 100
            if split_percent_int >= 100:
                split_percent = 75

        if not matrix_path:
            split_percent = 100

        split_config = split_percent
        cv_mode = str(extra_param.get("CV_MODE", "RANDOM_SPLIT")).upper()
        if cv_mode == "POLYGON_GROUP" and matrix_path:
            proceed, fallback_to_random = confirm_polygon_group_split(
                plugin.iface.mainWindow(),
                vector_path=vector_path,
                class_field=class_field,
                log=plugin.log,
            )
            if not proceed:
                return
            train_percent = int(split_percent)
            if train_percent < 1 or train_percent > 99:
                train_percent = 75
            try:
                train_vector, valid_vector = split_vector_stratified(vector_path, class_field, train_percent)
                vector_path = train_vector
                split_config = valid_vector
            except Exception as exc:
                plugin.log.warning(f"[Dashboard] Polygon label split failed, fallback to random split: {exc!s}")
                split_config = split_percent
            if fallback_to_random:
                extra_param["CV_MODE"] = "RANDOM_SPLIT"

        confidence_map = config.get("confidence_map", "") or None

        plugin.log.info(
            f"[Dashboard] Starting {'training and ' if do_training else ''}classification with {classifier_code}",
        )
        plugin._start_classification_task(
            description=f"dzetsaka Dashboard: {classifier_code} classification",
            do_training=do_training,
            raster_path=raster_path,
            vector_path=vector_path if do_training else None,
            class_field=class_field if do_training else None,
            model_path=model_path,
            split_config=split_config,
            random_seed=random_seed,
            matrix_path=matrix_path,
            classifier=classifier_code,
            output_path=output_raster_path,
            mask_path=None,
            confidence_map=confidence_map,
            nodata=nodata_value,
            extra_params=extra_param,
            error_context="Dashboard classification workflow",
            success_prefix="Dashboard",
        )
    except Exception as exc:
        plugin._show_github_issue_popup(
            error_title="Dashboard Configuration Error",
            error_type="Runtime Error",
            error_message=f"Unexpected error in dashboard execution: {exc!s}",
            context="dashboard_execution.execute_dashboard_config",
        )
