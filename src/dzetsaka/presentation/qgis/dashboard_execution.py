"""Dashboard-driven training/classification execution flow."""

from __future__ import annotations

import os
import tempfile


def _build_polygon_label_split(vector_path, class_field, train_percent):
    # type: (str, str, int) -> tuple[str, str]
    """Create train/validation vectors by splitting polygons stratified by label."""
    try:
        from osgeo import ogr
    except ImportError:
        import ogr  # type: ignore[no-redef]

    from sklearn.model_selection import train_test_split

    ds = ogr.Open(vector_path)
    if ds is None:
        raise RuntimeError(f"Unable to open vector dataset: {vector_path}")
    lyr = ds.GetLayer()
    if lyr is None:
        raise RuntimeError(f"No layer found in vector dataset: {vector_path}")

    features = []
    labels = []
    srs = lyr.GetSpatialRef()
    defn = lyr.GetLayerDefn()
    field_names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    for feat in lyr:
        label = feat.GetField(class_field)
        if label in (None, ""):
            continue
        features.append(feat.Clone())
        labels.append(label)
    ds = None

    if len(features) < 2:
        raise RuntimeError("Not enough polygons for polygon-group split.")

    validation_percent = max(1, min(99, 100 - int(train_percent)))
    test_size = validation_percent / 100.0
    valid_feats, train_feats = train_test_split(features, test_size=test_size, stratify=labels, random_state=0)

    out_dir = tempfile.mkdtemp(prefix="dzetsaka_poly_split_")
    train_path = os.path.join(out_dir, "train.shp")
    valid_path = os.path.join(out_dir, "valid.shp")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("OGR Shapefile driver unavailable.")

    def _write_subset(path, subset):
        if os.path.exists(path):
            driver.DeleteDataSource(path)
        out_ds = driver.CreateDataSource(path)
        if out_ds is None:
            raise RuntimeError(f"Unable to create output vector: {path}")
        out_lyr = out_ds.CreateLayer("subset", srs, defn.GetGeomType())
        for field_name in field_names:
            src_field = defn.GetFieldDefn(defn.GetFieldIndex(field_name))
            out_lyr.CreateField(src_field)
        out_defn = out_lyr.GetLayerDefn()
        for src_feat in subset:
            out_feat = ogr.Feature(out_defn)
            geom = src_feat.GetGeometryRef()
            if geom is not None:
                out_feat.SetGeometry(geom.Clone())
            for field_name in field_names:
                out_feat.SetField(field_name, src_feat.GetField(field_name))
            out_lyr.CreateFeature(out_feat)
            out_feat = None
        out_ds = None

    _write_subset(train_path, train_feats)
    _write_subset(valid_path, valid_feats)
    return train_path, valid_path


def execute_dashboard_config(plugin, config) -> None:
    """Run training and classification driven by dashboard config dict."""
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
        model_path = save_model if save_model else tempfile.mktemp("." + classifier_code)

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
        classifier_code, source_label="Dashboard", fallback_to_gmm=False
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
        train_percent = int(split_percent)
        if train_percent < 1 or train_percent > 99:
            train_percent = 75
        try:
            train_vector, valid_vector = _build_polygon_label_split(vector_path, class_field, train_percent)
            vector_path = train_vector
            split_config = valid_vector
        except Exception as exc:
            plugin.log.warning(f"[Dashboard] Polygon label split failed, fallback to random split: {exc!s}")
            split_config = split_percent

    confidence_map = config.get("confidence_map", "") or None

    plugin.log.info(
        f"[Dashboard] Starting {'training and ' if do_training else ''}classification with {classifier_code}"
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
