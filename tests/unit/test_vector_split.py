"""Tests for vector splitting functionality."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("sklearn")

try:
    from osgeo import ogr
except ImportError:
    ogr = pytest.importorskip("ogr")

from dzetsaka.infrastructure.geo.vector_split import split_vector_stratified


def create_test_shapefile(path: str, num_features: int = 100) -> None:
    """Create a simple test shapefile with features."""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(path):
        driver.DeleteDataSource(path)

    ds = driver.CreateDataSource(path)
    lyr = ds.CreateLayer("test", None, ogr.wkbPoint)

    # Add class field
    field_defn = ogr.FieldDefn("class", ogr.OFTInteger)
    lyr.CreateField(field_defn)

    # Create features with balanced classes
    for i in range(num_features):
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetField("class", i % 3 + 1)  # Classes 1, 2, 3

        # Create simple point geometry
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(i), float(i))
        feat.SetGeometry(point)

        lyr.CreateFeature(feat)
        feat = None

    ds = None


def count_features_in_shapefile(path: str) -> int:
    """Count features in a shapefile."""
    ds = ogr.Open(path)
    if ds is None:
        return 0
    lyr = ds.GetLayer()
    count = lyr.GetFeatureCount()
    ds = None
    return count


class TestVectorSplit:
    """Test vector splitting functionality."""

    def test_split_percentage_mode(self):
        """Test splitting with percentage mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.shp")
            create_test_shapefile(input_path, num_features=100)

            # Split 70% train, 30% validation
            train_path, valid_path = split_vector_stratified(
                vector_path=input_path,
                class_field="class",
                train_percent=70,
                use_percent=True,
            )

            # Verify files exist
            assert os.path.exists(train_path)
            assert os.path.exists(valid_path)

            # Verify split ratio (approximately)
            train_count = count_features_in_shapefile(train_path)
            valid_count = count_features_in_shapefile(valid_path)

            assert train_count + valid_count == 100
            # Allow some tolerance due to stratification rounding
            assert 65 <= train_count <= 75
            assert 25 <= valid_count <= 35

    def test_split_count_mode(self):
        """Test splitting with absolute count mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.shp")
            create_test_shapefile(input_path, num_features=100)

            # Split with 60 samples for training
            train_path, valid_path = split_vector_stratified(
                vector_path=input_path,
                class_field="class",
                train_percent=60,
                use_percent=False,
            )

            # Verify files exist
            assert os.path.exists(train_path)
            assert os.path.exists(valid_path)

            # Verify split
            train_count = count_features_in_shapefile(train_path)
            valid_count = count_features_in_shapefile(valid_path)

            assert train_count + valid_count == 100
            assert train_count == 60
            assert valid_count == 40

    def test_split_with_custom_output_paths(self):
        """Test splitting with custom output paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.shp")
            create_test_shapefile(input_path, num_features=50)

            train_out = os.path.join(tmpdir, "custom_train.shp")
            valid_out = os.path.join(tmpdir, "custom_valid.shp")

            train_path, valid_path = split_vector_stratified(
                vector_path=input_path,
                class_field="class",
                train_percent=80,
                train_output=train_out,
                validation_output=valid_out,
            )

            # Verify correct paths returned
            assert train_path == train_out
            assert valid_path == valid_out

            # Verify files exist
            assert os.path.exists(train_out)
            assert os.path.exists(valid_out)

    def test_invalid_percentage_raises_error(self):
        """Test that invalid percentages raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.shp")
            create_test_shapefile(input_path, num_features=50)

            with pytest.raises(ValueError, match="train_percent must be between 1 and 99"):
                split_vector_stratified(
                    vector_path=input_path,
                    class_field="class",
                    train_percent=100,
                    use_percent=True,
                )

    def test_missing_vector_raises_error(self):
        """Test that missing vector file raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unable to open vector dataset"):
            split_vector_stratified(
                vector_path="/nonexistent/path.shp",
                class_field="class",
                train_percent=70,
            )

    def test_stratification_maintains_class_distribution(self):
        """Test that stratified split maintains class distribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.shp")
            create_test_shapefile(input_path, num_features=90)  # 30 per class

            train_path, valid_path = split_vector_stratified(
                vector_path=input_path,
                class_field="class",
                train_percent=70,
            )

            # Count classes in each split
            def get_class_counts(path):
                ds = ogr.Open(path)
                lyr = ds.GetLayer()
                counts = {1: 0, 2: 0, 3: 0}
                for feat in lyr:
                    class_val = feat.GetField("class")
                    counts[class_val] += 1
                ds = None
                return counts

            train_counts = get_class_counts(train_path)
            valid_counts = get_class_counts(valid_path)

            # Each class should have roughly same proportion in train/valid
            # With 30 samples per class and 70/30 split: ~21 train, ~9 valid per class
            for cls in [1, 2, 3]:
                assert 18 <= train_counts[cls] <= 24  # Allow some tolerance
                assert 6 <= valid_counts[cls] <= 12
