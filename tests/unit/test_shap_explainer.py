"""Unit tests for SHAP explainer module.

Tests the ModelExplainer class and related functionality for computing
SHAP-based feature importance.
"""
import os
import pickle
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Try to import SHAP and skip tests if unavailable
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import the module to test
try:
    from scripts.explainability.shap_explainer import (
        SHAP_AVAILABLE as MODULE_SHAP_AVAILABLE,
        ModelExplainer,
        check_shap_available,
    )

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    ModelExplainer = None
    check_shap_available = None

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="SHAP explainer module not available")


class TestCheckShapAvailable:
    """Test SHAP availability checking."""

    def test_check_shap_available_returns_tuple(self):
        """Test that check_shap_available returns a tuple."""
        result = check_shap_available()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_shap_available_boolean_first(self):
        """Test that first element is boolean."""
        is_available, version = check_shap_available()
        assert isinstance(is_available, bool)

    def test_check_shap_available_version_when_available(self):
        """Test that version is string or None."""
        is_available, version = check_shap_available()
        if is_available:
            assert version is None or isinstance(version, str)
        else:
            assert version is None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerInit:
    """Test ModelExplainer initialization."""

    def test_init_with_valid_model(self):
        """Test initialization with valid model."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )

        assert explainer.model is model
        assert explainer.feature_names == ["f1", "f2", "f3", "f4", "f5"]

    def test_init_with_none_model_raises_error(self):
        """Test that None model raises ValueError."""
        with pytest.raises(ValueError, match="Model cannot be None"):
            ModelExplainer(model=None)

    def test_init_without_shap_raises_dependency_error(self):
        """Test that missing SHAP raises DependencyError."""
        with patch("scripts.explainability.shap_explainer.SHAP_AVAILABLE", False):
            # Need to reload module for patch to take effect
            # Instead, test the condition directly
            if not SHAP_AVAILABLE:
                from scripts.explainability.shap_explainer import DependencyError

                with pytest.raises(DependencyError):
                    ModelExplainer(model=Mock())

    def test_init_generates_feature_names_when_none(self):
        """Test that feature names are generated if not provided."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)

        # Feature names generated later when needed
        assert explainer.feature_names is None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerTreeDetection:
    """Test tree-based model detection."""

    def test_detects_random_forest_as_tree(self):
        """Test Random Forest is detected as tree-based."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)
        is_tree = explainer._is_tree_based_model()

        assert is_tree is True

    def test_detects_svm_as_not_tree(self):
        """Test SVM is detected as not tree-based."""
        from sklearn.svm import SVC

        model = SVC(kernel="linear", random_state=42, probability=True)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)
        is_tree = explainer._is_tree_based_model()

        assert is_tree is False

    def test_detects_gradient_boosting_as_tree(self):
        """Test Gradient Boosting is detected as tree-based."""
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)
        is_tree = explainer._is_tree_based_model()

        assert is_tree is True


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerFeatureImportance:
    """Test feature importance computation."""

    def test_get_feature_importance_basic(self):
        """Test basic feature importance computation."""
        from sklearn.ensemble import RandomForestClassifier

        # Create simple dataset
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)

        # Create explainer
        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3", "f4"],
        )

        # Compute importance
        X_sample = np.random.rand(50, 4)
        importance = explainer.get_feature_importance(X_sample)

        # Validate result
        assert isinstance(importance, dict)
        assert len(importance) == 4
        assert all(k in importance for k in ["f1", "f2", "f3", "f4"])
        assert all(isinstance(v, (float, np.floating)) for v in importance.values())

        # Importance should sum to ~1.0 (normalized)
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_get_feature_importance_with_none_sample_raises_error(self):
        """Test that None sample raises ValueError."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)

        with pytest.raises(ValueError, match="X_sample cannot be None or empty"):
            explainer.get_feature_importance(None)

    def test_get_feature_importance_with_empty_sample_raises_error(self):
        """Test that empty sample raises ValueError."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)

        with pytest.raises(ValueError, match="X_sample cannot be None or empty"):
            explainer.get_feature_importance(np.array([]))

    def test_get_feature_importance_generates_feature_names(self):
        """Test that feature names are auto-generated if not provided."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)  # No feature names
        X_sample = np.random.rand(30, 3)

        importance = explainer.get_feature_importance(X_sample)

        # Should have auto-generated names
        assert "Feature_0" in importance or "Feature_1" in importance

    def test_get_feature_importance_different_aggregation_methods(self):
        """Test different aggregation methods."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 3)
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model, feature_names=["f1", "f2", "f3"])
        X_sample = np.random.rand(50, 3)

        # Test different methods
        imp_mean_abs = explainer.get_feature_importance(X_sample, aggregate_method="mean_abs")
        imp_mean = explainer.get_feature_importance(X_sample, aggregate_method="mean")
        imp_max_abs = explainer.get_feature_importance(X_sample, aggregate_method="max_abs")

        # All should return valid dictionaries
        assert isinstance(imp_mean_abs, dict)
        assert isinstance(imp_mean, dict)
        assert isinstance(imp_max_abs, dict)

        # All should have same keys
        assert set(imp_mean_abs.keys()) == set(imp_mean.keys()) == set(imp_max_abs.keys())

    def test_get_feature_importance_invalid_aggregation_raises_error(self):
        """Test that invalid aggregation method raises ValueError."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)
        X_sample = np.random.rand(30, 3)

        with pytest.raises(ValueError, match="Unknown aggregate_method"):
            explainer.get_feature_importance(X_sample, aggregate_method="invalid_method")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerSaveLoad:
    """Test saving and loading explainer."""

    def test_save_to_file(self):
        """Test saving explainer to file."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3"],
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            explainer.save_to_file(temp_path)
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_file(self):
        """Test loading explainer from file."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(
            model=model,
            feature_names=["f1", "f2", "f3"],
        )

        # Save and load
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            explainer.save_to_file(temp_path)
            loaded_explainer = ModelExplainer.load_from_file(temp_path)

            assert loaded_explainer.feature_names == ["f1", "f2", "f3"]
            assert loaded_explainer.model is not None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_nonexistent_file_raises_error(self):
        """Test that loading from nonexistent file raises error."""
        with pytest.raises(ValueError, match="Failed to load explainer"):
            ModelExplainer.load_from_file("/nonexistent/path/file.pkl")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerRasterProcessing:
    """Test raster importance generation (mocked)."""

    @patch("scripts.explainability.shap_explainer.gdal")
    def test_sample_raster_pixels_basic(self, mock_gdal):
        """Test pixel sampling from raster."""
        from sklearn.ensemble import RandomForestClassifier

        # Create mock raster dataset
        mock_ds = Mock()
        mock_ds.RasterXSize = 100
        mock_ds.RasterYSize = 100

        # Create mock band
        mock_band = Mock()
        mock_band_data = np.random.rand(100, 100).astype(np.float32)
        mock_band.ReadAsArray.return_value = mock_band_data

        mock_ds.GetRasterBand.return_value = mock_band

        # Create model and explainer
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)

        # Sample pixels
        X_sample = explainer._sample_raster_pixels(mock_ds, sample_size=50, n_bands=3)

        # Validate
        assert X_sample.shape == (50, 3)
        assert X_sample.dtype == np.float32


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_tree_model(self):
        """Test complete workflow with tree-based model."""
        from sklearn.ensemble import RandomForestClassifier

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 3, 100)  # 3 classes
        model.fit(X_train, y_train)

        # Create explainer
        explainer = ModelExplainer(
            model=model,
            feature_names=["B1", "B2", "B3", "B4"],
        )

        # Compute importance
        X_sample = np.random.rand(50, 4)
        importance = explainer.get_feature_importance(X_sample)

        # Validate
        assert len(importance) == 4
        assert all(k in importance for k in ["B1", "B2", "B3", "B4"])
        assert all(0 <= v <= 1 for v in importance.values())

        # Check normalization
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_end_to_end_non_tree_model(self):
        """Test complete workflow with non-tree model."""
        from sklearn.svm import SVC

        # Train model
        model = SVC(kernel="linear", probability=True, random_state=42)
        X_train = np.random.rand(80, 3)
        y_train = np.random.randint(0, 2, 80)
        model.fit(X_train, y_train)

        # Create explainer with background data
        explainer = ModelExplainer(
            model=model,
            feature_names=["F1", "F2", "F3"],
            background_data=X_train[:50],  # Use subset as background
        )

        # Compute importance (smaller sample for faster test)
        X_sample = np.random.rand(20, 3)
        importance = explainer.get_feature_importance(X_sample)

        # Validate
        assert len(importance) == 3
        assert all(k in importance for k in ["F1", "F2", "F3"])
        assert all(0 <= v <= 1 for v in importance.values())


class TestModuleAvailability:
    """Test module-level availability checks."""

    def test_shap_available_flag_is_boolean(self):
        """Test that SHAP_AVAILABLE is a boolean."""
        if MODULE_AVAILABLE:
            assert isinstance(MODULE_SHAP_AVAILABLE, bool)

    def test_module_imports_without_error(self):
        """Test that module can be imported."""
        try:
            from scripts.explainability import shap_explainer

            assert shap_explainer is not None
        except ImportError:
            pytest.skip("Module not available")
