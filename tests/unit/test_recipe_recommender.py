"""Unit tests for the recipe recommendation engine."""

import pytest

from ui.recipe_recommender import RasterAnalyzer, RecipeRecommender


class TestRasterAnalyzer:
    """Tests for RasterAnalyzer class."""

    def test_extract_filename_hints(self):
        """Test filename hint extraction."""
        analyzer = RasterAnalyzer()

        # Test Sentinel detection
        hints = analyzer._extract_filename_hints("sentinel2_crop_2023.tif")
        assert "sentinel" in hints
        assert "crop" in hints

        # Test Landsat detection
        hints = analyzer._extract_filename_hints("LC08_L1TP_forest.tif")
        assert "landsat" in hints
        assert "forest" in hints

        # Test NDVI
        hints = analyzer._extract_filename_hints("ndvi_image.tif")
        assert "ndvi" in hints

    def test_detect_sensor_sentinel(self):
        """Test Sentinel-2 sensor detection."""
        analyzer = RasterAnalyzer()

        # Various Sentinel-2 naming patterns
        assert analyzer._detect_sensor("sentinel2_data.tif") == "sentinel2"
        assert analyzer._detect_sensor("s2a_20230101.tif") == "sentinel2"
        assert analyzer._detect_sensor("S2B_T32TQN_20230615.tif") == "sentinel2"

    def test_detect_sensor_landsat(self):
        """Test Landsat sensor detection."""
        analyzer = RasterAnalyzer()

        # Landsat 8 patterns
        assert analyzer._detect_sensor("LC08_L1TP_042034_20230101.tif") == "landsat8"
        assert analyzer._detect_sensor("landsat8_image.tif") == "landsat8"

        # Landsat 9 patterns
        assert analyzer._detect_sensor("LC09_L1TP_042034_20230101.tif") == "landsat9"

    def test_detect_landcover_type(self):
        """Test land cover type detection."""
        analyzer = RasterAnalyzer()

        assert analyzer._detect_landcover_type("agriculture_data.tif") == "agriculture"
        assert analyzer._detect_landcover_type("forest_classification.tif") == "forest"
        assert analyzer._detect_landcover_type("urban_area.tif") == "urban"
        assert analyzer._detect_landcover_type("water_bodies.tif") == "water"
        assert analyzer._detect_landcover_type("random_image.tif") == "unknown"

    def test_analyze_nonexistent_file(self):
        """Test analysis of non-existent file."""
        analyzer = RasterAnalyzer()
        result = analyzer.analyze_raster("/nonexistent/path/to/file.tif")

        assert result["error"] is not None
        assert result["band_count"] == 0


class TestRecipeRecommender:
    """Tests for RecipeRecommender class."""

    @pytest.fixture
    def sample_recipes(self):
        """Create sample recipes for testing."""
        return [
            {
                "name": "Fast Random Forest",
                "description": "Quick classification with Random Forest",
                "classifier": "RF",
                "extraParam": {},
                "expected_runtime_class": "fast",
                "expected_accuracy_class": "medium",
            },
            {
                "name": "Sentinel-2 Crop Classification",
                "description": "Optimized for Sentinel-2 agriculture mapping",
                "classifier": "XGB",
                "extraParam": {},
                "expected_runtime_class": "medium",
                "expected_accuracy_class": "high",
            },
            {
                "name": "High Accuracy SVM",
                "description": "Best accuracy with SVM",
                "classifier": "SVM",
                "extraParam": {},
                "expected_runtime_class": "slow",
                "expected_accuracy_class": "high",
            },
            {
                "name": "Imbalanced Data Handler",
                "description": "Uses SMOTE for imbalanced classes",
                "classifier": "RF",
                "extraParam": {"SMOTE": True},
                "expected_runtime_class": "medium",
                "expected_accuracy_class": "high",
            },
        ]

    def test_recommend_sentinel_data(self, sample_recipes):
        """Test recommendations for Sentinel-2 data."""
        recommender = RecipeRecommender()

        raster_info = {
            "band_count": 12,
            "file_size_mb": 500.0,
            "detected_sensor": "sentinel2",
            "landcover_type": "agriculture",
            "filename_hints": ["sentinel", "crop"],
        }

        recommendations = recommender.recommend(raster_info, sample_recipes)

        # Should have recommendations
        assert len(recommendations) > 0

        # Top recommendation should be the Sentinel-2 specific recipe
        top_recipe, top_score, top_reason = recommendations[0]
        assert "Sentinel" in top_recipe["name"]
        assert top_score > 80.0  # High confidence
        assert "Sentinel" in top_reason or "12" in top_reason

    def test_recommend_large_file(self, sample_recipes):
        """Test recommendations penalize slow algorithms for large files."""
        recommender = RecipeRecommender()

        raster_info = {
            "band_count": 4,
            "file_size_mb": 2000.0,  # 2 GB file
            "detected_sensor": "unknown",
            "landcover_type": "unknown",
            "filename_hints": [],
        }

        recommendations = recommender.recommend(raster_info, sample_recipes)

        # Find SVM recipe (should be penalized)
        svm_found = False
        rf_found = False
        for recipe, score, reason in recommendations:
            if recipe["classifier"] == "SVM":
                svm_found = True
                # SVM should have lower score due to file size
                assert "slow" in reason.lower() or "large" in reason.lower()
            elif recipe["classifier"] == "RF":
                rf_found = True
                rf_score = score

        assert svm_found
        assert rf_found

    def test_recommend_hyperspectral(self, sample_recipes):
        """Test recommendations for hyperspectral data."""
        recommender = RecipeRecommender()

        raster_info = {
            "band_count": 50,  # Many bands
            "file_size_mb": 300.0,
            "detected_sensor": "unknown",
            "landcover_type": "unknown",
            "filename_hints": ["hyperspectral"],
        }

        recommendations = recommender.recommend(raster_info, sample_recipes)

        # Should have recommendations
        assert len(recommendations) > 0

        # Top recommendation should mention hyperspectral or many bands
        _top_recipe, top_score, top_reason = recommendations[0]
        assert "hyperspectral" in top_reason.lower() or "band" in top_reason.lower()

    def test_score_recipe_empty_info(self, sample_recipes):
        """Test scoring with minimal raster info."""
        recommender = RecipeRecommender()

        raster_info = {
            "band_count": 0,
            "file_size_mb": 0.0,
            "detected_sensor": "unknown",
            "landcover_type": "unknown",
            "filename_hints": [],
        }

        score, reasons = recommender._score_recipe(raster_info, sample_recipes[0])

        # Should have base score at minimum
        assert score >= 30.0
        assert isinstance(reasons, list)

    def test_get_confidence_class(self):
        """Test confidence class categorization."""
        recommender = RecipeRecommender()

        assert recommender.get_confidence_class(100) == "Excellent match"
        assert recommender.get_confidence_class(85) == "Good match"
        assert recommender.get_confidence_class(70) == "Possible match"
        assert recommender.get_confidence_class(45) == "Low confidence"
        assert recommender.get_confidence_class(20) == "Not recommended"

    def test_get_star_rating(self):
        """Test star rating generation."""
        recommender = RecipeRecommender()

        assert recommender.get_star_rating(100) == "⭐⭐⭐⭐⭐"
        assert recommender.get_star_rating(80) == "⭐⭐⭐⭐"
        assert recommender.get_star_rating(60) == "⭐⭐⭐"
        assert recommender.get_star_rating(40) == "⭐⭐"
        assert recommender.get_star_rating(10) == "⭐"

    def test_empty_recipes_list(self):
        """Test recommendation with no recipes available."""
        recommender = RecipeRecommender()

        raster_info = {
            "band_count": 12,
            "file_size_mb": 500.0,
            "detected_sensor": "sentinel2",
            "landcover_type": "unknown",
            "filename_hints": [],
        }

        recommendations = recommender.recommend(raster_info, [])
        assert recommendations == []
