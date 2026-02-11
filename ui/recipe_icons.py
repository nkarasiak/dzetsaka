"""Visual icons and styling for recipe cards.

This module provides icon mappings, color schemes, and helper functions
for displaying recipes in the recipe shop UI.
"""

from typing import Dict, List, Any, Optional


# Recipe category colors
CATEGORY_COLORS = {
    "beginner": {"bg": "#e8f5e9", "border": "#4caf50", "text": "#2e7d32"},
    "intermediate": {"bg": "#e3f2fd", "border": "#2196f3", "text": "#1565c0"},
    "advanced": {"bg": "#fff3e0", "border": "#ff9800", "text": "#e65100"},
    "experimental": {"bg": "#f3e5f5", "border": "#9c27b0", "text": "#6a1b9a"},
}

# Algorithm icons (emoji)
ALGORITHM_ICONS = {
    "GMM": "🎯",
    "RF": "🌲",
    "SVM": "⚡",
    "KNN": "🎲",
    "XGB": "🚀",
    "LGB": "⚡",
    "CB": "🏆",
    "ET": "🌳",
    "GBC": "📈",
    "LR": "📏",
    "NB": "🎰",
    "MLP": "🧠",
}

# Feature badges
FEATURE_BADGES = {
    "optuna": {"text": "OPTUNA", "color": "#9c27b0"},
    "shap": {"text": "SHAP", "color": "#ff5722"},
    "smote": {"text": "SMOTE", "color": "#00bcd4"},
    "nested_cv": {"text": "NESTED CV", "color": "#795548"},
    "spatial_cv": {"text": "SPATIAL CV", "color": "#4caf50"},
}

# Runtime icons
RUNTIME_ICONS = {
    "fast": "⚡",      # < 5 min
    "medium": "⏱️",   # 5-30 min
    "slow": "🕐",     # > 30 min
}

# Accuracy icons
ACCURACY_ICONS = {
    "low": "📊",      # < 70%
    "medium": "📊",   # 70-85%
    "high": "📊",     # 85-95%
    "very_high": "🎯", # > 95%
}

# Category icons
CATEGORY_ICONS = {
    "beginner": "🌱",
    "intermediate": "🔬",
    "advanced": "🎓",
    "experimental": "🧪",
}

# Use case icons
USE_CASE_ICONS = {
    "agriculture": "🌾",
    "forestry": "🌲",
    "urban": "🏙️",
    "water": "💧",
    "land_cover": "🗺️",
    "change_detection": "📊",
    "general": "📦",
}


def get_recipe_icon(recipe: Dict[str, Any]) -> str:
    """Get icon for a recipe based on its algorithm.

    Args:
        recipe: Recipe dictionary

    Returns:
        Emoji icon string
    """
    classifier = recipe.get("classifier", {})
    code = classifier.get("code")
    if not code:
        return "📦"
    return ALGORITHM_ICONS.get(code, "📦")


def get_category_style(category: str) -> Dict[str, str]:
    """Get color scheme for a category.

    Args:
        category: Category name (beginner, intermediate, advanced, experimental)

    Returns:
        Dict with bg, border, and text color codes
    """
    return CATEGORY_COLORS.get(category.lower(), CATEGORY_COLORS["beginner"])


def get_category_icon(category: str) -> str:
    """Get icon for a category.

    Args:
        category: Category name

    Returns:
        Emoji icon string
    """
    return CATEGORY_ICONS.get(category.lower(), "📦")


def get_feature_badges(recipe: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract feature badges from recipe.

    Args:
        recipe: Recipe dictionary

    Returns:
        List of badge dicts with text and color
    """
    badges = []
    extra = recipe.get("extraParam", {})
    validation = recipe.get("validation", {})

    if extra.get("USE_OPTUNA"):
        badges.append(FEATURE_BADGES["optuna"])
    if extra.get("COMPUTE_SHAP"):
        badges.append(FEATURE_BADGES["shap"])
    if extra.get("USE_SMOTE"):
        badges.append(FEATURE_BADGES["smote"])
    if extra.get("USE_NESTED_CV") or validation.get("nested_cv"):
        badges.append(FEATURE_BADGES["nested_cv"])
    if validation.get("spatial_cv"):
        badges.append(FEATURE_BADGES["spatial_cv"])

    return badges


def format_runtime(runtime_minutes: Optional[float]) -> str:
    """Format runtime for display.

    Args:
        runtime_minutes: Estimated runtime in minutes

    Returns:
        Formatted string with icon
    """
    if runtime_minutes is None:
        return "⏱️ Unknown"

    if runtime_minutes < 5:
        icon = RUNTIME_ICONS["fast"]
        label = "Fast"
    elif runtime_minutes < 30:
        icon = RUNTIME_ICONS["medium"]
        label = "Medium"
    else:
        icon = RUNTIME_ICONS["slow"]
        label = "Slow"

    if runtime_minutes < 1:
        time_str = f"{int(runtime_minutes * 60)}s"
    elif runtime_minutes < 60:
        time_str = f"{int(runtime_minutes)}m"
    else:
        time_str = f"{runtime_minutes / 60:.1f}h"

    return f"{icon} {label} (~{time_str})"


def format_accuracy(accuracy_percent: Optional[float]) -> str:
    """Format expected accuracy for display.

    Args:
        accuracy_percent: Expected accuracy percentage (0-100)

    Returns:
        Formatted string with icon
    """
    if accuracy_percent is None:
        return "📊 Unknown"

    if accuracy_percent < 70:
        icon = ACCURACY_ICONS["low"]
        label = "Low"
    elif accuracy_percent < 85:
        icon = ACCURACY_ICONS["medium"]
        label = "Medium"
    elif accuracy_percent < 95:
        icon = ACCURACY_ICONS["high"]
        label = "High"
    else:
        icon = ACCURACY_ICONS["very_high"]
        label = "Very High"

    return f"{icon} {label} (~{accuracy_percent:.0f}%)"


def get_algorithm_description(algorithm_code: str) -> str:
    """Get friendly description of an algorithm.

    Args:
        algorithm_code: Algorithm code (GMM, RF, SVM, etc.)

    Returns:
        Human-readable description
    """
    descriptions = {
        "GMM": "Gaussian Mixture Model - Probabilistic clustering",
        "RF": "Random Forest - Ensemble of decision trees",
        "SVM": "Support Vector Machine - Margin-based classifier",
        "KNN": "K-Nearest Neighbors - Instance-based learning",
        "XGB": "XGBoost - Gradient boosting with regularization",
        "LGB": "LightGBM - Fast gradient boosting framework",
        "CB": "CatBoost - Gradient boosting for categorical features",
        "ET": "Extra Trees - Randomized decision tree ensemble",
        "GBC": "Gradient Boosting Classifier - Additive modeling",
        "LR": "Logistic Regression - Linear probabilistic model",
        "NB": "Naive Bayes - Probabilistic classifier",
        "MLP": "Multi-Layer Perceptron - Neural network",
    }
    return descriptions.get(algorithm_code, "Unknown algorithm")


def get_use_case_icon(use_case: str) -> str:
    """Get icon for a use case.

    Args:
        use_case: Use case identifier

    Returns:
        Emoji icon string
    """
    return USE_CASE_ICONS.get(use_case.lower(), USE_CASE_ICONS["general"])


def get_difficulty_level(category: str) -> int:
    """Get numeric difficulty level from category.

    Args:
        category: Category name

    Returns:
        Difficulty level (1-4)
    """
    levels = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3,
        "experimental": 4,
    }
    return levels.get(category.lower(), 1)


def format_difficulty_stars(category: str) -> str:
    """Format difficulty as star rating.

    Args:
        category: Category name

    Returns:
        Star string (e.g., "⭐⭐⭐")
    """
    level = get_difficulty_level(category)
    return "⭐" * level + "☆" * (4 - level)


def get_recipe_card_stylesheet(category: str) -> str:
    """Get stylesheet for a recipe card.

    Args:
        category: Recipe category

    Returns:
        CSS stylesheet string for QFrame
    """
    style = get_category_style(category)
    return f"""
        QFrame {{
            background-color: {style['bg']};
            border: 2px solid {style['border']};
            border-radius: 8px;
            padding: 12px;
        }}
        QFrame:hover {{
            border: 3px solid {style['border']};
            background-color: {lighten_color(style['bg'], 0.05)};
        }}
        QLabel {{
            color: {style['text']};
            background-color: transparent;
            border: none;
        }}
    """


def lighten_color(hex_color: str, factor: float = 0.1) -> str:
    """Lighten a hex color by a factor.

    Args:
        hex_color: Hex color string (e.g., "#e8f5e9")
        factor: Lightening factor (0-1)

    Returns:
        Lightened hex color string
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))

    return f"#{r:02x}{g:02x}{b:02x}"


def get_badge_stylesheet(badge: Dict[str, str]) -> str:
    """Get stylesheet for a feature badge.

    Args:
        badge: Badge dict with text and color

    Returns:
        CSS stylesheet string for QLabel
    """
    return f"""
        QLabel {{
            background-color: {badge['color']};
            color: white;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: bold;
        }}
    """


# Main stylesheet for recipe shop dialog
RECIPE_SHOP_STYLESHEET = """
/* Main dialog */
QDialog {
    background-color: #f5f5f5;
}

/* Search bar */
QLineEdit {
    padding: 8px 12px;
    border: 2px solid #bdbdbd;
    border-radius: 6px;
    background-color: white;
    font-size: 13px;
}

QLineEdit:focus {
    border: 2px solid #2196f3;
}

/* Category filter tabs */
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background-color: white;
}

QTabBar::tab {
    background-color: #eeeeee;
    color: #616161;
    padding: 8px 16px;
    border: 1px solid #e0e0e0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: white;
    color: #1976d2;
    font-weight: bold;
}

QTabBar::tab:hover {
    background-color: #e3f2fd;
}

/* Scroll area */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    border: none;
    background-color: #f5f5f5;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #bdbdbd;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9e9e9e;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}

/* Buttons */
QPushButton {
    background-color: #2196f3;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1976d2;
}

QPushButton:pressed {
    background-color: #1565c0;
}

QPushButton:disabled {
    background-color: #e0e0e0;
    color: #9e9e9e;
}

/* Secondary button */
QPushButton[objectName="secondaryButton"] {
    background-color: #757575;
}

QPushButton[objectName="secondaryButton"]:hover {
    background-color: #616161;
}

/* Recipe card container */
QFrame[objectName="recipeCard"] {
    margin: 8px;
}

/* Labels */
QLabel[objectName="recipeName"] {
    font-size: 15px;
    font-weight: bold;
}

QLabel[objectName="recipeDescription"] {
    font-size: 12px;
    color: #616161;
}

QLabel[objectName="recipeMetadata"] {
    font-size: 11px;
    color: #757575;
}

/* Group boxes */
QGroupBox {
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    color: #424242;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    background-color: #f5f5f5;
}

/* Combo boxes */
QComboBox {
    border: 2px solid #bdbdbd;
    border-radius: 6px;
    padding: 6px 12px;
    background-color: white;
    font-size: 12px;
}

QComboBox:hover {
    border: 2px solid #2196f3;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border: 2px solid #757575;
    width: 6px;
    height: 6px;
    border-top: none;
    border-right: none;
    transform: rotate(-45deg);
}

/* Tooltips */
QToolTip {
    background-color: #424242;
    color: white;
    border: 1px solid #212121;
    border-radius: 4px;
    padding: 6px;
    font-size: 12px;
}
"""
