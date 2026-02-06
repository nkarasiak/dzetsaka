"""Wizard-driven training/classification execution flow."""

from __future__ import annotations

import os
import tempfile


def execute_wizard_config(plugin, config) -> None:
    """Run training and classification driven by wizard config dict."""
    in_raster = config.get("raster", "")
    in_shape = config.get("vector", "")
    in_field = config.get("class_field", "")
    in_classifier = str(config.get("classifier", "GMM"))
    extra_param = config.get("extraParam", None)

    load_model = config.get("load_model", "")
    if load_model:
        model = load_model
    else:
        save_model = config.get("save_model", "")
        model = save_model if save_model else tempfile.mktemp("." + in_classifier)

    out_matrix = config.get("confusion_matrix", "") or None
    in_split = config.get("split_percent", 100)
    if not out_matrix:
        in_split = 100

    nodata = -9999
    in_seed = 0

    do_training = not load_model
    if not plugin._validate_classification_request(
        raster_path=in_raster,
        do_training=do_training,
        vector_path=in_shape if do_training else None,
        class_field=in_field if do_training else None,
        model_path=model if not do_training else None,
        source_label="Wizard",
    ):
        return
    if not plugin._ensure_classifier_runtime_ready(
        in_classifier, source_label="Wizard", fallback_to_gmm=False
    ):
        return

    out_raster = config.get("output_raster", "")
    if not out_raster:
        temp_folder = tempfile.mkdtemp()
        out_raster = os.path.join(temp_folder, plugin._default_output_name(in_raster, in_classifier))

    confidence_map = config.get("confidence_map", "") or None

    plugin.log.info(
        f"[Wizard] Starting {'training and ' if do_training else ''}classification with {in_classifier}"
    )
    plugin._start_classification_task(
        description=f"dzetsaka Wizard: {in_classifier} classification",
        do_training=do_training,
        raster_path=in_raster,
        vector_path=in_shape if do_training else None,
        class_field=in_field if do_training else None,
        model_path=model,
        split_config=in_split,
        random_seed=in_seed,
        matrix_path=out_matrix,
        classifier=in_classifier,
        output_path=out_raster,
        mask_path=None,
        confidence_map=confidence_map,
        nodata=nodata,
        extra_params=extra_param,
        error_context="Wizard classification workflow",
        success_prefix="Wizard",
    )

