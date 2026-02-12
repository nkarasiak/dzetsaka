"""Vector dataset train/validation splitting utilities.

This module provides functionality to split vector datasets for machine learning
training and validation using stratified splitting.
"""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any

try:
    from osgeo import ogr
except ImportError:
    import ogr  # type: ignore[no-redef]

OGR_BACKEND: Any | None = None
_DEFAULT_OGR = ogr


def _get_ogr():
    return OGR_BACKEND or _DEFAULT_OGR


def count_polygons_per_class(vector_path: str, class_field: str) -> dict[Any, int]:
    """Count how many vector features per class exist in a layer."""
    ds = _get_ogr().Open(vector_path)
    if ds is None:
        return {}

    layer = ds.GetLayer()
    class_counts: dict[Any, int] = {}
    for feature in layer:
        label = feature.GetField(class_field)
        if label in (None, ""):
            continue
        class_counts[label] = class_counts.get(label, 0) + 1

    ds = None
    return class_counts


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
    while maintaining class distribution (stratified split).

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
        >>> train_shp, valid_shp = split_vector_stratified("training_data.shp", "class", train_percent=70)
        >>> # Specifying output paths
        >>> train_shp, valid_shp = split_vector_stratified(
        ...     "training_data.shp",
        ...     "class",
        ...     train_percent=70,
        ...     train_output="/path/to/train.shp",
        ...     validation_output="/path/to/valid.shp",
        ... )
        >>> # Using absolute count
        >>> train_shp, valid_shp = split_vector_stratified(
        ...     "training_data.shp", "class", train_percent=100, use_percent=False
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
        # train_percent is percentage - convert to train_size ratio
        train_size = train_percent / 100.0
        validation_size = 1.0 - train_size
    else:
        # train_percent is absolute count
        train_size = int(train_percent)
        validation_size = None  # Let splitter calculate based on train_size

    # Perform reproducible stratified split.
    train_feats, validation_feats = _stratified_split(
        features=features,
        labels=labels,
        train_size=train_size,
        test_size=validation_size,
        random_state=0,
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


def _stratified_split(
    features: list[Any],
    labels: list[Any],
    train_size: float | int,
    test_size: float | int | None = None,
    random_state: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Split features into stratified train/test subsets.

    This keeps plugin startup independent from optional ML dependencies while
    preserving deterministic, class-balanced splitting for vector sampling.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if len(features) < 2:
        raise ValueError("Need at least two samples to split")

    n_samples = len(features)
    n_train, n_test = _resolve_split_sizes(
        n_samples=n_samples,
        train_size=train_size,
        test_size=test_size,
    )

    class_to_indices: dict[Any, list[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)

    if len(class_to_indices) < 2:
        raise ValueError("Need at least two classes for stratified splitting")

    min_class_count = min(len(indices) for indices in class_to_indices.values())
    if min_class_count < 2:
        raise ValueError("The least populated class has fewer than 2 samples")

    n_classes = len(class_to_indices)
    if n_train < n_classes or n_test < n_classes:
        raise ValueError("train/test split too small to keep all classes in both subsets")

    train_counts = _allocate_train_counts(class_to_indices, n_train, n_samples)
    rng = random.Random(random_state)  # nosec B311

    train_indices: list[int] = []
    test_indices: list[int] = []
    for label, indices in class_to_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        split_at = train_counts[label]
        train_indices.extend(shuffled[:split_at])
        test_indices.extend(shuffled[split_at:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_features = [features[idx] for idx in train_indices]
    test_features = [features[idx] for idx in test_indices]
    return train_features, test_features


def _resolve_split_sizes(
    n_samples: int,
    train_size: float | int,
    test_size: float | int | None,
) -> tuple[int, int]:
    """Resolve validated absolute train/test sizes."""
    n_train = _resolve_size(train_size, n_samples, "train_size")
    if test_size is None:
        n_test = n_samples - n_train
    else:
        n_test = _resolve_size(test_size, n_samples, "test_size")
        if n_train + n_test != n_samples:
            raise ValueError("train_size and test_size must sum to total number of samples")

    if n_train <= 0 or n_test <= 0:
        raise ValueError("train and test subsets must both be non-empty")
    return n_train, n_test


def _resolve_size(size: float | int, n_samples: int, name: str) -> int:
    """Convert a ratio/count split parameter to an absolute integer count."""
    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} as float must be in (0, 1), got {size}")
        resolved = round(size * n_samples)
    else:
        resolved = int(size)

    if not 0 < resolved < n_samples:
        raise ValueError(f"{name} must be between 1 and {n_samples - 1}, got {size}")
    return resolved


def _allocate_train_counts(class_to_indices: dict[Any, list[int]], n_train: int, n_total: int) -> dict[Any, int]:
    """Allocate train counts per class using largest remainder rounding."""
    counts = {label: len(indices) for label, indices in class_to_indices.items()}
    base: dict[Any, int] = {}
    remainders: list[tuple[float, Any]] = []

    for label, count in counts.items():
        expected = (count * n_train) / n_total
        assigned = int(expected)
        assigned = max(1, min(count - 1, assigned))
        base[label] = assigned
        remainders.append((expected - int(expected), label))

    assigned_total = sum(base.values())
    delta = n_train - assigned_total

    if delta > 0:
        for _, label in sorted(remainders, reverse=True):
            if delta == 0:
                break
            if base[label] < counts[label] - 1:
                base[label] += 1
                delta -= 1
    elif delta < 0:
        for _, label in sorted(remainders):
            if delta == 0:
                break
            if base[label] > 1:
                base[label] -= 1
                delta += 1

    if delta != 0:
        for label, count in counts.items():
            if delta == 0:
                break
            while delta > 0 and base[label] < count - 1:
                base[label] += 1
                delta -= 1
            while delta < 0 and base[label] > 1:
                base[label] -= 1
                delta += 1

    if delta != 0:
        raise ValueError("Unable to satisfy stratified split constraints for the requested sizes")

    return base
