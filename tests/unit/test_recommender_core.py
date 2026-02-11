"""Unit tests for the core recommendation engine logic (QGIS-independent)."""

import sys
from pathlib import Path

# Add ui directory to path for direct import
ui_dir = Path(__file__).parent.parent.parent / "ui"
sys.path.insert(0, str(ui_dir))

import pytest


def test_import_recommender():
    """Test that recommender modules can be imported."""
    try:
        from recipe_recommender import RasterAnalyzer, RecipeRecommender

        assert RasterAnalyzer is not None
        assert RecipeRecommender is not None
    except ImportError as e:
        pytest.fail(f"Could not import recommender: {e}")


def test_analyzer_filename_hints():
    """Test filename hint extraction."""
    from recipe_recommender import RasterAnalyzer

    analyzer = RasterAnalyzer()

    # Test Sentinel detection
    hints = analyzer._extract_filename_hints("sentinel2_crop_2023.tif")
    assert "sentinel" in hints
    assert "crop" in hints

    # Test Landsat detection (landsat keyword must be explicit in filename)
    hints = analyzer._extract_filename_hints("landsat8_l1tp_forest.tif")
    assert "landsat" in hints
    assert "forest" in hints


def test_sensor_detection():
    """Test sensor detection from filename."""
    from recipe_recommender import RasterAnalyzer

    analyzer = RasterAnalyzer()

    # Sentinel-2
    assert analyzer._detect_sensor("sentinel2_data.tif") == "sentinel2"
    assert analyzer._detect_sensor("s2a_20230101.tif") == "sentinel2"

    # Landsat
    assert analyzer._detect_sensor("lc08_l1tp_042034.tif") == "landsat8"
    assert analyzer._detect_sensor("landsat8_image.tif") == "landsat8"

    # Unknown
    assert analyzer._detect_sensor("random_file.tif") == "unknown"


def test_landcover_detection():
    """Test land cover type detection."""
    from recipe_recommender import RasterAnalyzer

    analyzer = RasterAnalyzer()

    assert analyzer._detect_landcover_type("agriculture_data.tif") == "agriculture"
    assert analyzer._detect_landcover_type("forest_classification.tif") == "forest"
    assert analyzer._detect_landcover_type("urban_area.tif") == "urban"
    assert analyzer._detect_landcover_type("random_image.tif") == "unknown"


def test_score_recipe():
    """Test recipe scoring logic."""
    from recipe_recommender import RecipeRecommender

    recommender = RecipeRecommender()

    recipe = {
        "name": "Fast Random Forest",
        "description": "Quick classification",
        "classifier": "RF",
        "extraParam": {},
    }

    raster_info = {
        "band_count": 12,
        "file_size_mb": 500.0,
        "detected_sensor": "sentinel2",
        "landcover_type": "unknown",
        "filename_hints": ["sentinel"],
    }

    score, reasons = recommender._score_recipe(raster_info, recipe)

    # Should have positive score
    assert score > 0
    assert isinstance(reasons, list)
    # Should detect Sentinel-2
    assert any("sentinel" in r.lower() or "12" in r.lower() for r in reasons)


def test_confidence_classes():
    """Test confidence class descriptions."""
    from recipe_recommender import RecipeRecommender

    recommender = RecipeRecommender()

    assert "Excellent" in recommender.get_confidence_class(95)
    assert "Good" in recommender.get_confidence_class(85)
    assert "Possible" in recommender.get_confidence_class(70)
    assert "Low" in recommender.get_confidence_class(45)


def test_star_ratings():
    """Test star rating generation."""
    from recipe_recommender import RecipeRecommender

    recommender = RecipeRecommender()

    # Should generate 1-5 stars
    assert len(recommender.get_star_rating(100)) == 5
    assert len(recommender.get_star_rating(80)) == 4
    assert len(recommender.get_star_rating(60)) == 3
    assert len(recommender.get_star_rating(40)) == 2
    assert len(recommender.get_star_rating(10)) >= 1  # At least 1 star


def test_recommend_with_recipes():
    """Test full recommendation flow."""
    from recipe_recommender import RecipeRecommender

    recommender = RecipeRecommender()

    recipes = [
        {
            "name": "Sentinel Crop",
            "description": "Optimized for Sentinel-2 agriculture",
            "classifier": "XGB",
            "extraParam": {},
        },
        {
            "name": "Fast RF",
            "description": "Quick classification",
            "classifier": "RF",
            "extraParam": {},
        },
    ]

    raster_info = {
        "band_count": 12,
        "file_size_mb": 300.0,
        "detected_sensor": "sentinel2",
        "landcover_type": "agriculture",
        "filename_hints": ["sentinel", "crop"],
    }

    recommendations = recommender.recommend(raster_info, recipes)

    # Should have recommendations
    assert len(recommendations) > 0

    # Should be sorted by score
    if len(recommendations) > 1:
        assert recommendations[0][1] >= recommendations[1][1]

    # Each recommendation should be a tuple
    for recipe, score, reason in recommendations:
        assert isinstance(recipe, dict)
        assert isinstance(score, float)
        assert isinstance(reason, str)
        assert 0 <= score <= 100


def test_empty_recipes():
    """Test with no recipes available."""
    from recipe_recommender import RecipeRecommender

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


def test_large_file_penalty():
    """Test that slow algorithms are penalized for large files."""
    from recipe_recommender import RecipeRecommender

    recommender = RecipeRecommender()

    svm_recipe = {
        "name": "SVM",
        "description": "Support Vector Machine",
        "classifier": "SVM",
        "extraParam": {},
    }

    rf_recipe = {
        "name": "Random Forest",
        "description": "Fast Random Forest",
        "classifier": "RF",
        "extraParam": {},
    }

    large_file_info = {
        "band_count": 4,
        "file_size_mb": 2000.0,  # 2 GB
        "detected_sensor": "unknown",
        "landcover_type": "unknown",
        "filename_hints": [],
    }

    svm_score, _svm_reasons = recommender._score_recipe(large_file_info, svm_recipe)
    rf_score, _rf_reasons = recommender._score_recipe(large_file_info, rf_recipe)

    # RF should have higher score for large files
    assert rf_score > svm_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
