"""Vector dataset train/validation splitting utilities.

This module provides functionality to split vector datasets for machine learning
training and validation using scikit-learn's stratified splitting.
"""

from __future__ import annotations

import os
import tempfile

try:
    from osgeo import ogr
except ImportError:
    import ogr  # type: ignore[no-redef]

from sklearn.model_selection import train_test_split


def split_vector_stratified(
    vector_path: str,
    class_field: str,
    train_percent: int | float,
    train_output: str | None = None,
    validation_output: str | None = None,
    use_percent: bool = True,
) -> tuple[str, str]:
    """Split vector dataset into train/validation subsets using stratified sampling.

    Creates two new shapefiles by splitting the input vector layer's features
    while maintaining class distribution (stratified split). Uses scikit-learn's
    train_test_split for robust, reproducible splitting.

    Args:
        vector_path: Path to input vector file (shapefile, GeoJSON, etc.)
        class_field: Name of the field containing class labels
        train_percent: If use_percent=True, percentage (1-99) for training.
            If use_percent=False, absolute number of samples for training.
        train_output: Optional path for training output shapefile.
            If None, creates in temp directory.
        validation_output: Optional path for validation output shapefile.
            If None, creates in temp directory.
        use_percent: If True, train_percent is a percentage.
            If False, train_percent is an absolute count.

    Returns:
        Tuple of (train_path, validation_path) - paths to output shapefiles

    Raises:
        RuntimeError: If vector cannot be opened, has no features, or OGR operations fail
        ValueError: If train_percent is not in valid range

    Example:
        >>> # Using temp directory with percentage (default)
        >>> train_shp, valid_shp = split_vector_stratified(
        ...     "training_data.shp",
        ...     "class",
        ...     train_percent=70
        ... )
        >>> # Specifying output paths
        >>> train_shp, valid_shp = split_vector_stratified(
        ...     "training_data.shp",
        ...     "class",
        ...     train_percent=70,
        ...     train_output="/path/to/train.shp",
        ...     validation_output="/path/to/valid.shp"
        ... )
        >>> # Using absolute count
        >>> train_shp, valid_shp = split_vector_stratified(
        ...     "training_data.shp",
        ...     "class",
        ...     train_percent=100,
        ...     use_percent=False
        ... )
    """
    if use_percent and not (1 <= train_percent <= 99):
        raise ValueError(f"train_percent must be between 1 and 99 when use_percent=True, got {train_percent}")

    # Open input vector
    ds = ogr.Open(vector_path)
    if ds is None:
        raise RuntimeError(f"Unable to open vector dataset: {vector_path}")

    lyr = ds.GetLayer()
    if lyr is None:
        raise RuntimeError(f"No layer found in vector dataset: {vector_path}")

    # Extract features and labels
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

    ds = None  # Close dataset

    if len(features) < 2:
        raise RuntimeError("Not enough features for splitting (need at least 2 features).")

    # Calculate split parameters based on mode
    if use_percent:
        # train_percent is percentage - convert to train_size for sklearn
        train_size = train_percent / 100.0
        validation_size = 1.0 - train_size
    else:
        # train_percent is absolute count
        train_size = int(train_percent)
        validation_size = None  # Let sklearn calculate based on train_size

    # Perform stratified split using scikit-learn
    if validation_size is not None:
        train_feats, validation_feats = train_test_split(
            features,
            train_size=train_size,
            test_size=validation_size,
            stratify=labels,
            random_state=0,  # Fixed seed for reproducibility
        )
    else:
        train_feats, validation_feats = train_test_split(
            features,
            train_size=train_size,
            stratify=labels,
            random_state=0,  # Fixed seed for reproducibility
        )

    # Determine output paths
    if train_output and validation_output:
        train_path = train_output
        valid_path = validation_output
    else:
        # Create output directory in temp
        out_dir = tempfile.mkdtemp(prefix="dzetsaka_vector_split_")
        train_path = train_output or os.path.join(out_dir, "train.shp")
        valid_path = validation_output or os.path.join(out_dir, "validation.shp")

    # Get shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("OGR Shapefile driver unavailable.")

    # Write train and validation shapefiles
    _write_shapefile_subset(driver, train_path, train_feats, srs, defn, field_names)
    _write_shapefile_subset(driver, valid_path, validation_feats, srs, defn, field_names)

    return train_path, valid_path


def _write_shapefile_subset(
    driver: ogr.Driver,
    output_path: str,
    features: list,
    srs: ogr.SpatialReference,
    defn: ogr.FeatureDefn,
    field_names: list[str],
) -> None:
    """Write a subset of features to a new shapefile.

    Args:
        driver: OGR driver for shapefile creation
        output_path: Path for output shapefile
        features: List of OGR features to write
        srs: Spatial reference system
        defn: Feature definition from source layer
        field_names: List of field names to copy

    Raises:
        RuntimeError: If shapefile creation fails
    """
    # Remove existing file if present
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)

    # Create new datasource
    out_ds = driver.CreateDataSource(output_path)
    if out_ds is None:
        raise RuntimeError(f"Unable to create output vector: {output_path}")

    # Create layer with same spatial reference and geometry type
    out_lyr = out_ds.CreateLayer("subset", srs, defn.GetGeomType())

    # Add fields
    for field_name in field_names:
        src_field = defn.GetFieldDefn(defn.GetFieldIndex(field_name))
        out_lyr.CreateField(src_field)

    out_defn = out_lyr.GetLayerDefn()

    # Copy features
    for src_feat in features:
        out_feat = ogr.Feature(out_defn)

        # Copy geometry
        geom = src_feat.GetGeometryRef()
        if geom is not None:
            out_feat.SetGeometry(geom.Clone())

        # Copy attributes
        for field_name in field_names:
            out_feat.SetField(field_name, src_feat.GetField(field_name))

        out_lyr.CreateFeature(out_feat)
        out_feat = None

    out_ds = None  # Close and flush
