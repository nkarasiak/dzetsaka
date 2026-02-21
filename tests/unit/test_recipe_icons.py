"""Unit tests for recipe icons and styling helpers."""

import sys
from pathlib import Path

# Add ui directory to path to import recipe_icons directly without triggering __init__.py
ui_path = Path(__file__).parent.parent.parent / "ui"
sys.path.insert(0, str(ui_path))

import recipe_icons

# Import the functions we need
get_recipe_icon = recipe_icons.get_recipe_icon
get_category_style = recipe_icons.get_category_style
get_category_icon = recipe_icons.get_category_icon
get_feature_badges = recipe_icons.get_feature_badges
format_runtime = recipe_icons.format_runtime
format_accuracy = recipe_icons.format_accuracy
get_algorithm_description = recipe_icons.get_algorithm_description
get_use_case_icon = recipe_icons.get_use_case_icon
get_difficulty_level = recipe_icons.get_difficulty_level
format_difficulty_stars = recipe_icons.format_difficulty_stars
get_recipe_card_stylesheet = recipe_icons.get_recipe_card_stylesheet
lighten_color = recipe_icons.lighten_color
get_badge_stylesheet = recipe_icons.get_badge_stylesheet
ALGORITHM_ICONS = recipe_icons.ALGORITHM_ICONS
CATEGORY_COLORS = recipe_icons.CATEGORY_COLORS
FEATURE_BADGES = recipe_icons.FEATURE_BADGES


class TestRecipeIcon:
    """Test recipe icon retrieval."""

    def test_get_recipe_icon_gmm(self):
        """Test icon for GMM algorithm."""
        recipe = {"classifier": {"code": "GMM"}}
        assert get_recipe_icon(recipe) == "🎯"

    def test_get_recipe_icon_rf(self):
        """Test icon for Random Forest algorithm."""
        recipe = {"classifier": {"code": "RF"}}
        assert get_recipe_icon(recipe) == "🌲"

    def test_get_recipe_icon_xgb(self):
        """Test icon for XGBoost algorithm."""
        recipe = {"classifier": {"code": "XGB"}}
        assert get_recipe_icon(recipe) == "🚀"

    def test_get_recipe_icon_unknown(self):
        """Test icon for unknown algorithm."""
        recipe = {"classifier": {"code": "UNKNOWN"}}
        assert get_recipe_icon(recipe) == "📦"

    def test_get_recipe_icon_empty(self):
        """Test icon for empty recipe."""
        recipe = {}
        assert get_recipe_icon(recipe) == "📦"


class TestCategoryStyle:
    """Test category styling functions."""

    def test_get_category_style_beginner(self):
        """Test style for beginner category."""
        style = get_category_style("beginner")
        assert style["bg"] == "#e8f5e9"
        assert style["border"] == "#4caf50"
        assert style["text"] == "#2e7d32"

    def test_get_category_style_intermediate(self):
        """Test style for intermediate category."""
        style = get_category_style("intermediate")
        assert style["bg"] == "#e3f2fd"
        assert style["border"] == "#2196f3"

    def test_get_category_style_advanced(self):
        """Test style for advanced category."""
        style = get_category_style("advanced")
        assert style["bg"] == "#fff3e0"
        assert style["border"] == "#ff9800"

    def test_get_category_style_case_insensitive(self):
        """Test category style is case insensitive."""
        style1 = get_category_style("BEGINNER")
        style2 = get_category_style("beginner")
        assert style1 == style2

    def test_get_category_style_unknown(self):
        """Test fallback for unknown category."""
        style = get_category_style("unknown")
        assert style == CATEGORY_COLORS["beginner"]

    def test_get_category_icon(self):
        """Test category icon retrieval."""
        assert get_category_icon("beginner") == "🌱"
        assert get_category_icon("intermediate") == "🔬"
        assert get_category_icon("advanced") == "🎓"
        assert get_category_icon("experimental") == "🧪"


class TestFeatureBadges:
    """Test feature badge extraction."""

    def test_get_feature_badges_empty(self):
        """Test badges for recipe with no features."""
        recipe = {}
        badges = get_feature_badges(recipe)
        assert badges == []

    def test_get_feature_badges_optuna(self):
        """Test Optuna badge detection."""
        recipe = {"extraParam": {"USE_OPTUNA": True}}
        badges = get_feature_badges(recipe)
        assert len(badges) == 1
        assert badges[0]["text"] == "OPTUNA"
        assert badges[0]["color"] == "#9c27b0"

    def test_get_feature_badges_shap(self):
        """Test SHAP badge detection."""
        recipe = {"extraParam": {"COMPUTE_SHAP": True}}
        badges = get_feature_badges(recipe)
        assert len(badges) == 1
        assert badges[0]["text"] == "SHAP"

    def test_get_feature_badges_smote(self):
        """Test SMOTE badge detection."""
        recipe = {"extraParam": {"USE_SMOTE": True}}
        badges = get_feature_badges(recipe)
        assert len(badges) == 1
        assert badges[0]["text"] == "SMOTE"

    def test_get_feature_badges_nested_cv(self):
        """Test nested CV badge detection."""
        recipe = {"validation": {"nested_cv": True}}
        badges = get_feature_badges(recipe)
        assert len(badges) == 1
        assert badges[0]["text"] == "NESTED CV"

    def test_get_feature_badges_nested_cv_legacy(self):
        """Test nested CV badge from legacy extraParam."""
        recipe = {"extraParam": {"USE_NESTED_CV": True}}
        badges = get_feature_badges(recipe)
        assert any(b["text"] == "NESTED CV" for b in badges)

    def test_get_feature_badges_spatial_cv(self):
        """Test spatial CV badge detection."""
        recipe = {"validation": {"spatial_cv": True}}
        badges = get_feature_badges(recipe)
        assert len(badges) == 1
        assert badges[0]["text"] == "SPATIAL CV"

    def test_get_feature_badges_multiple(self):
        """Test multiple badges."""
        recipe = {
            "extraParam": {"USE_OPTUNA": True, "COMPUTE_SHAP": True, "USE_SMOTE": True},
            "validation": {"nested_cv": True, "spatial_cv": True},
        }
        badges = get_feature_badges(recipe)
        assert len(badges) == 5
        badge_texts = [b["text"] for b in badges]
        assert "OPTUNA" in badge_texts
        assert "SHAP" in badge_texts
        assert "SMOTE" in badge_texts
        assert "NESTED CV" in badge_texts
        assert "SPATIAL CV" in badge_texts


class TestRuntimeFormatting:
    """Test runtime formatting."""

    def test_format_runtime_none(self):
        """Test runtime formatting with None."""
        result = format_runtime(None)
        assert result == "⏱️ Unknown"

    def test_format_runtime_fast_seconds(self):
        """Test runtime formatting for fast (< 1 min)."""
        result = format_runtime(0.5)
        assert "⚡" in result
        assert "Fast" in result
        assert "30s" in result

    def test_format_runtime_fast_minutes(self):
        """Test runtime formatting for fast (< 5 min)."""
        result = format_runtime(3)
        assert "⚡" in result
        assert "Fast" in result
        assert "3m" in result

    def test_format_runtime_medium(self):
        """Test runtime formatting for medium (5-30 min)."""
        result = format_runtime(15)
        assert "⏱️" in result
        assert "Medium" in result
        assert "15m" in result

    def test_format_runtime_slow_minutes(self):
        """Test runtime formatting for slow (> 30 min)."""
        result = format_runtime(45)
        assert "🕐" in result
        assert "Slow" in result
        assert "45m" in result

    def test_format_runtime_slow_hours(self):
        """Test runtime formatting for slow (> 1 hour)."""
        result = format_runtime(90)
        assert "🕐" in result
        assert "Slow" in result
        assert "1.5h" in result


class TestAccuracyFormatting:
    """Test accuracy formatting."""

    def test_format_accuracy_none(self):
        """Test accuracy formatting with None."""
        result = format_accuracy(None)
        assert result == "📊 Unknown"

    def test_format_accuracy_low(self):
        """Test accuracy formatting for low (< 70%)."""
        result = format_accuracy(65)
        assert "📊" in result
        assert "Low" in result
        assert "65%" in result

    def test_format_accuracy_medium(self):
        """Test accuracy formatting for medium (70-85%)."""
        result = format_accuracy(78)
        assert "📊" in result
        assert "Medium" in result
        assert "78%" in result

    def test_format_accuracy_high(self):
        """Test accuracy formatting for high (85-95%)."""
        result = format_accuracy(90)
        assert "📊" in result
        assert "High" in result
        assert "90%" in result

    def test_format_accuracy_very_high(self):
        """Test accuracy formatting for very high (> 95%)."""
        result = format_accuracy(97)
        assert "🎯" in result
        assert "Very High" in result
        assert "97%" in result


class TestAlgorithmDescription:
    """Test algorithm description retrieval."""

    def test_get_algorithm_description_gmm(self):
        """Test GMM description."""
        desc = get_algorithm_description("GMM")
        assert "Gaussian Mixture Model" in desc
        assert "Probabilistic" in desc

    def test_get_algorithm_description_rf(self):
        """Test Random Forest description."""
        desc = get_algorithm_description("RF")
        assert "Random Forest" in desc
        assert "decision trees" in desc

    def test_get_algorithm_description_xgb(self):
        """Test XGBoost description."""
        desc = get_algorithm_description("XGB")
        assert "XGBoost" in desc
        assert "Gradient boosting" in desc

    def test_get_algorithm_description_mlp(self):
        """Test MLP description."""
        desc = get_algorithm_description("MLP")
        assert "Multi-Layer Perceptron" in desc
        assert "Neural network" in desc

    def test_get_algorithm_description_unknown(self):
        """Test unknown algorithm description."""
        desc = get_algorithm_description("UNKNOWN")
        assert "Unknown algorithm" in desc


class TestUseCaseIcon:
    """Test use case icon retrieval."""

    def test_get_use_case_icon_agriculture(self):
        """Test agriculture icon."""
        assert get_use_case_icon("agriculture") == "🌾"

    def test_get_use_case_icon_forestry(self):
        """Test forestry icon."""
        assert get_use_case_icon("forestry") == "🌲"

    def test_get_use_case_icon_urban(self):
        """Test urban icon."""
        assert get_use_case_icon("urban") == "🏙️"

    def test_get_use_case_icon_water(self):
        """Test water icon."""
        assert get_use_case_icon("water") == "💧"

    def test_get_use_case_icon_unknown(self):
        """Test unknown use case fallback."""
        assert get_use_case_icon("unknown") == "📦"


class TestDifficultyLevel:
    """Test difficulty level functions."""

    def test_get_difficulty_level_beginner(self):
        """Test beginner difficulty level."""
        assert get_difficulty_level("beginner") == 1

    def test_get_difficulty_level_intermediate(self):
        """Test intermediate difficulty level."""
        assert get_difficulty_level("intermediate") == 2

    def test_get_difficulty_level_advanced(self):
        """Test advanced difficulty level."""
        assert get_difficulty_level("advanced") == 3

    def test_get_difficulty_level_experimental(self):
        """Test experimental difficulty level."""
        assert get_difficulty_level("experimental") == 4

    def test_get_difficulty_level_unknown(self):
        """Test unknown category defaults to beginner."""
        assert get_difficulty_level("unknown") == 1

    def test_format_difficulty_stars_beginner(self):
        """Test star formatting for beginner."""
        stars = format_difficulty_stars("beginner")
        assert stars == "⭐☆☆☆"

    def test_format_difficulty_stars_intermediate(self):
        """Test star formatting for intermediate."""
        stars = format_difficulty_stars("intermediate")
        assert stars == "⭐⭐☆☆"

    def test_format_difficulty_stars_advanced(self):
        """Test star formatting for advanced."""
        stars = format_difficulty_stars("advanced")
        assert stars == "⭐⭐⭐☆"

    def test_format_difficulty_stars_experimental(self):
        """Test star formatting for experimental."""
        stars = format_difficulty_stars("experimental")
        assert stars == "⭐⭐⭐⭐"


class TestColorHelpers:
    """Test color manipulation helpers."""

    def test_lighten_color_basic(self):
        """Test color lightening."""
        result = lighten_color("#000000", 0.5)
        assert result == "#7f7f7f"

    def test_lighten_color_already_light(self):
        """Test lightening already light color."""
        result = lighten_color("#ffffff", 0.5)
        assert result == "#ffffff"

    def test_lighten_color_small_factor(self):
        """Test small lightening factor."""
        result = lighten_color("#e8f5e9", 0.1)
        # Should be slightly lighter than original
        assert result != "#e8f5e9"
        # Should still be greenish
        assert result.startswith("#")
        assert len(result) == 7


class TestStylesheets:
    """Test stylesheet generation."""

    def test_get_recipe_card_stylesheet_beginner(self):
        """Test recipe card stylesheet for beginner."""
        stylesheet = get_recipe_card_stylesheet("beginner")
        assert "QFrame" in stylesheet
        assert "#e8f5e9" in stylesheet  # bg color
        assert "#4caf50" in stylesheet  # border color
        assert "border-radius" in stylesheet

    def test_get_recipe_card_stylesheet_advanced(self):
        """Test recipe card stylesheet for advanced."""
        stylesheet = get_recipe_card_stylesheet("advanced")
        assert "#fff3e0" in stylesheet  # bg color
        assert "#ff9800" in stylesheet  # border color

    def test_get_badge_stylesheet(self):
        """Test badge stylesheet generation."""
        badge = {"text": "OPTUNA", "color": "#9c27b0"}
        stylesheet = get_badge_stylesheet(badge)
        assert "QLabel" in stylesheet
        assert "#9c27b0" in stylesheet
        assert "border-radius" in stylesheet
        assert "white" in stylesheet


class TestConstants:
    """Test constant dictionaries."""

    def test_algorithm_icons_complete(self):
        """Test all 11 algorithms have icons."""
        expected_algorithms = ["GMM", "RF", "SVM", "KNN", "XGB", "CB", "ET", "GBC", "LR", "NB", "MLP"]
        for algo in expected_algorithms:
            assert algo in ALGORITHM_ICONS
            assert len(ALGORITHM_ICONS[algo]) > 0

    def test_category_colors_complete(self):
        """Test all categories have color schemes."""
        expected_categories = ["beginner", "intermediate", "advanced", "experimental"]
        for category in expected_categories:
            assert category in CATEGORY_COLORS
            assert "bg" in CATEGORY_COLORS[category]
            assert "border" in CATEGORY_COLORS[category]
            assert "text" in CATEGORY_COLORS[category]

    def test_feature_badges_complete(self):
        """Test all feature badges defined."""
        expected_badges = ["optuna", "shap", "smote", "nested_cv", "spatial_cv"]
        for badge in expected_badges:
            assert badge in FEATURE_BADGES
            assert "text" in FEATURE_BADGES[badge]
            assert "color" in FEATURE_BADGES[badge]
