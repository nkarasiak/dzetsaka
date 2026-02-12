"""Unit tests for split preview functionality."""

import pytest

try:
    from osgeo import ogr
except ImportError:
    ogr = None  # type: ignore[assignment]

from dzetsaka.infrastructure.geo.vector_split import count_polygons_per_class, split_vector_stratified


@pytest.mark.skipif(ogr is None or not hasattr(ogr, "GetDriverByName"), reason="OGR/GDAL not available")
def test_split_preview_basic(tmp_path):
    """Test basic split preview computation."""
    # Create a simple test shapefile with 3 classes
    shapefile_path = str(tmp_path / "test.shp")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(shapefile_path)
    layer = ds.CreateLayer("test", geom_type=ogr.wkbPoint)

    # Add class field
    field_defn = ogr.FieldDefn("class", ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Add features: 10 of class 1, 10 of class 2, 10 of class 3
    for class_id in [1, 2, 3]:
        for i in range(10):
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("class", class_id)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(i, class_id)
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feature = None

    ds = None  # Close dataset

    # Test count function
    class_counts = count_polygons_per_class(shapefile_path, "class")
    assert class_counts == {1: 10, 2: 10, 3: 10}

    # Test split function
    train_path, test_path = split_vector_stratified(shapefile_path, "class", train_percent=75, use_percent=True)

    train_counts = count_polygons_per_class(train_path, "class")
    test_counts = count_polygons_per_class(test_path, "class")

    # Verify stratified split maintains class distribution (75% train, 25% test)
    # Total per class: 10, so 7-8 train, 2-3 test
    assert train_counts[1] + test_counts[1] == 10
    assert train_counts[2] + test_counts[2] == 10
    assert train_counts[3] + test_counts[3] == 10

    # Verify approximate split ratio
    assert 6 <= train_counts[1] <= 8
    assert 2 <= test_counts[1] <= 4


@pytest.mark.skipif(ogr is None or not hasattr(ogr, "GetDriverByName"), reason="OGR/GDAL not available")
def test_split_preview_imbalanced(tmp_path):
    """Test split preview with imbalanced classes."""
    shapefile_path = str(tmp_path / "test_imbalanced.shp")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(shapefile_path)
    layer = ds.CreateLayer("test", geom_type=ogr.wkbPoint)

    # Add class field
    field_defn = ogr.FieldDefn("class", ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Add features: 100 of class 1, 10 of class 2 (10:1 imbalance)
    for class_id, count in [(1, 100), (2, 10)]:
        for i in range(count):
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("class", class_id)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(i, class_id)
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feature = None

    ds = None

    # Test count function
    class_counts = count_polygons_per_class(shapefile_path, "class")
    assert class_counts[1] == 100
    assert class_counts[2] == 10

    # Verify imbalance detection logic
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    assert max_count / min_count == 10.0  # Exactly 10:1 ratio


@pytest.mark.skipif(ogr is None or not hasattr(ogr, "GetDriverByName"), reason="OGR/GDAL not available")
def test_split_preview_small_test_set(tmp_path):
    """Test split preview with small test set."""
    shapefile_path = str(tmp_path / "test_small.shp")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(shapefile_path)
    layer = ds.CreateLayer("test", geom_type=ogr.wkbPoint)

    field_defn = ogr.FieldDefn("class", ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Add only 10 features per class
    for class_id in [1, 2]:
        for i in range(10):
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("class", class_id)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(i, class_id)
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feature = None

    ds = None

    # Split with 80% train, 20% test = 2 samples per class in test
    train_path, test_path = split_vector_stratified(shapefile_path, "class", train_percent=80, use_percent=True)

    test_counts = count_polygons_per_class(test_path, "class")

    # Verify small test set
    assert test_counts[1] == 2  # <5 samples
    assert test_counts[2] == 2  # <5 samples
