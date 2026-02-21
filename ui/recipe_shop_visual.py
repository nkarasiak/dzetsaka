"""Visual Recipe Shop dialog for dzetsaka with modern card-based UI.

A modern, visually appealing recipe browser showing recipes as cards in a scrollable
grid layout. Features category filtering, search, dependency indicators, and
detailed recipe preview.

Author:
    Nicolas Karasiak
"""

import json
import os
from typing import Any, Dict, List, Optional

from qgis.PyQt.QtCore import Qt, pyqtSignal, QSize
from qgis.PyQt.QtGui import QColor, QFont, QPalette, QIcon
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
    QGridLayout,
    QTabBar,
    QTextEdit,
    QMessageBox,
    QSizePolicy,
)


# Recipe card dimensions and styling
CARD_WIDTH = 180
CARD_HEIGHT = 220
CARD_BORDER_RADIUS = 8
CARD_SHADOW_COLOR = "#00000020"

# Category colors
CATEGORY_COLORS = {
    "beginner": "#e8f5e9",     # Light green
    "intermediate": "#e3f2fd", # Light blue
    "advanced": "#fff3e0",      # Light orange
    "custom": "#f3e5f5",        # Light purple
}

CATEGORY_BORDER_COLORS = {
    "beginner": "#4caf50",
    "intermediate": "#2196f3",
    "advanced": "#ff9800",
    "custom": "#9c27b0",
}

# Runtime class to emoji and text
RUNTIME_MAP = {
    "fast": ("⚡", "<1-3 min"),
    "medium": ("⏱️", "5-15 min"),
    "slow": ("⏳", "30-90 min"),
}

# Accuracy class to emoji and text
ACCURACY_MAP = {
    "medium": ("📊", "70-80%"),
    "high": ("📈", "80-92%"),
    "very_high": ("🎯", "90-98%"),
}

# Algorithm to emoji
ALGORITHM_EMOJI = {
    "GMM": "🔵",
    "RF": "🌲",
    "SVM": "⚔️",
    "KNN": "👥",
    "XGB": "🚀",
    "CB": "🐈",
    "ET": "🌳",
    "GBC": "📊",
    "LR": "📉",
    "NB": "🔮",
    "MLP": "🧠",
}


def _get_algorithm_code(recipe: Dict[str, Any]) -> str:
    """Extract algorithm code from recipe."""
    classifier = recipe.get("classifier", {})
    if isinstance(classifier, dict):
        return classifier.get("code", "GMM")
    return "GMM"


def _get_algorithm_name(recipe: Dict[str, Any]) -> str:
    """Extract algorithm name from recipe."""
    classifier = recipe.get("classifier", {})
    if isinstance(classifier, dict):
        return classifier.get("name", "Gaussian Mixture Model")
    return "Gaussian Mixture Model"


def _get_recipe_category(recipe: Dict[str, Any]) -> str:
    """Get recipe category from metadata."""
    metadata = recipe.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("category", "intermediate")
    return "intermediate"


def _get_runtime_class(recipe: Dict[str, Any]) -> str:
    """Get runtime class from recipe."""
    return recipe.get("expected_runtime_class", "medium")


def _get_accuracy_class(recipe: Dict[str, Any]) -> str:
    """Get accuracy class from recipe."""
    return recipe.get("expected_accuracy_class", "high")


def _get_feature_tags(recipe: Dict[str, Any]) -> List[str]:
    """Extract feature tags from recipe (e.g., OPTUNA, SHAP, SMOTE)."""
    tags = []
    extra_params = recipe.get("extraParam", {})

    if isinstance(extra_params, dict):
        if extra_params.get("USE_OPTUNA"):
            tags.append("OPTUNA")
        if extra_params.get("COMPUTE_SHAP"):
            tags.append("SHAP")
        if extra_params.get("USE_SMOTE"):
            tags.append("SMOTE")
        if extra_params.get("USE_CLASS_WEIGHTS"):
            tags.append("WEIGHTS")
        if extra_params.get("USE_NESTED_CV"):
            tags.append("NESTED CV")

    return tags


def _check_dependencies_available(recipe: Dict[str, Any], available_deps: Dict[str, bool]) -> tuple:
    """Check if recipe dependencies are available.

    Returns:
        (all_met, missing_list)
    """
    algo_code = _get_algorithm_code(recipe)
    extra_params = recipe.get("extraParam", {})

    missing = []

    # Check algorithm dependencies
    if algo_code in ["RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"]:
        if not available_deps.get("sklearn", False):
            missing.append("scikit-learn")
    elif algo_code == "XGB":
        if not available_deps.get("xgboost", False):
            missing.append("xgboost")
    elif algo_code == "CB":
        if not available_deps.get("catboost", False):
            missing.append("catboost")

    # Check feature dependencies
    if isinstance(extra_params, dict):
        if extra_params.get("USE_OPTUNA") and not available_deps.get("optuna", False):
            missing.append("optuna")
        if extra_params.get("COMPUTE_SHAP") and not available_deps.get("shap", False):
            missing.append("shap")
        if extra_params.get("USE_SMOTE") and not available_deps.get("imblearn", False):
            missing.append("imbalanced-learn")

    return len(missing) == 0, missing


class RecipeCard(QFrame):
    """Visual card widget representing a single recipe."""

    clicked = pyqtSignal(dict)
    applyRequested = pyqtSignal(dict)

    def __init__(self, recipe: Dict[str, Any], available_deps: Dict[str, bool], parent=None):
        super().__init__(parent)
        self.recipe = recipe
        self.available_deps = available_deps
        self._selected = False

        self._init_ui()
        self._update_style()

    def _init_ui(self):
        """Initialize card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Category badge
        category = _get_recipe_category(self.recipe)
        category_label = QLabel(category.upper())
        category_font = QFont()
        category_font.setPointSize(9)
        category_font.setBold(True)
        category_label.setFont(category_font)
        category_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        category_label.setStyleSheet(f"""
            background-color: {CATEGORY_BORDER_COLORS.get(category, '#999999')};
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
        """)
        layout.addWidget(category_label)

        # Algorithm emoji (large)
        algo_code = _get_algorithm_code(self.recipe)
        emoji = ALGORITHM_EMOJI.get(algo_code, "📦")
        emoji_label = QLabel(emoji)
        emoji_font = QFont()
        emoji_font.setPointSize(32)
        emoji_label.setFont(emoji_font)
        emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        layout.addWidget(emoji_label)

        # Recipe name (bold, 2 lines max)
        name = self.recipe.get("name", "Untitled Recipe")
        name_label = QLabel(name)
        name_font = QFont()
        name_font.setPointSize(11)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setMaximumHeight(40)
        layout.addWidget(name_label)

        # Runtime estimate
        runtime_class = _get_runtime_class(self.recipe)
        runtime_emoji, runtime_text = RUNTIME_MAP.get(runtime_class, ("⏱️", "5-15 min"))
        runtime_label = QLabel(f"{runtime_emoji} {runtime_text}")
        runtime_font = QFont()
        runtime_font.setPointSize(9)
        runtime_label.setFont(runtime_font)
        runtime_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        layout.addWidget(runtime_label)

        # Accuracy range
        accuracy_class = _get_accuracy_class(self.recipe)
        accuracy_emoji, accuracy_text = ACCURACY_MAP.get(accuracy_class, ("📊", "80-90%"))
        accuracy_label = QLabel(f"{accuracy_emoji} {accuracy_text}")
        accuracy_font = QFont()
        accuracy_font.setPointSize(9)
        accuracy_label.setFont(accuracy_font)
        accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        layout.addWidget(accuracy_label)

        # Algorithm summary
        algo_name = _get_algorithm_name(self.recipe)
        algo_summary = algo_name.split()[0]  # Take first word
        tags = _get_feature_tags(self.recipe)
        if tags:
            algo_summary += f" + {tags[0]}"

        algo_label = QLabel(algo_summary)
        algo_font = QFont()
        algo_font.setPointSize(9)
        algo_font.setItalic(True)
        algo_label.setFont(algo_font)
        algo_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        layout.addWidget(algo_label)

        # Feature tags (pill-shaped badges)
        if len(tags) > 1:
            tags_text = ", ".join(tags[:2])  # Show max 2 tags
            tags_label = QLabel(tags_text)
            tags_font = QFont()
            tags_font.setPointSize(8)
            tags_label.setFont(tags_font)
            tags_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
            tags_label.setStyleSheet("color: #666666;")
            layout.addWidget(tags_label)

        layout.addStretch()

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(lambda: self.applyRequested.emit(self.recipe))
        layout.addWidget(apply_btn)

        # Set fixed size
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)

    def _update_style(self):
        """Update card styling based on selection and category."""
        category = _get_recipe_category(self.recipe)
        bg_color = CATEGORY_COLORS.get(category, "#f5f5f5")

        if self._selected:
            border_color = "#2196f3"
            border_width = 3
            shadow = "0px 4px 8px rgba(0, 0, 0, 0.2)"
        else:
            border_color = CATEGORY_BORDER_COLORS.get(category, "#cccccc")
            border_width = 1
            shadow = "0px 2px 4px rgba(0, 0, 0, 0.1)"

        self.setStyleSheet(f"""
            RecipeCard {{
                background-color: {bg_color};
                border: {border_width}px solid {border_color};
                border-radius: {CARD_BORDER_RADIUS}px;
            }}
            RecipeCard:hover {{
                background-color: {bg_color};
                border: 2px solid {border_color};
            }}
        """)

    def setSelected(self, selected: bool):
        """Set card selection state."""
        self._selected = selected
        self._update_style()

    def mousePressEvent(self, event):
        """Handle mouse click."""
        self.clicked.emit(self.recipe)
        super().mousePressEvent(event)


class VisualRecipeShopDialog(QDialog):
    """Modern visual recipe shop with card-based UI."""

    recipeSelected = pyqtSignal(dict)

    def __init__(self, recipes: List[Dict[str, Any]], available_deps: Dict[str, bool], parent=None):
        super().__init__(parent)
        self.recipes = recipes
        self.available_deps = available_deps
        self.filtered_recipes = recipes.copy()
        self.selected_recipe = None
        self.cards = []
        self.current_filter = "all"

        self.setWindowTitle("dzetsaka Recipe Shop")
        self.resize(900, 700)

        self._init_ui()
        self._populate_cards()

    def _init_ui(self):
        """Initialize dialog UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header with icon and title
        header_layout = QHBoxLayout()
        header_label = QLabel("🏪 dzetsaka Recipe Shop")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Filter tabs
        filter_layout = QHBoxLayout()

        self.all_btn = QPushButton("All")
        self.all_btn.setCheckable(True)
        self.all_btn.setChecked(True)
        self.all_btn.clicked.connect(lambda: self._filter_category("all"))
        filter_layout.addWidget(self.all_btn)

        self.beginner_btn = QPushButton("Beginner")
        self.beginner_btn.setCheckable(True)
        self.beginner_btn.clicked.connect(lambda: self._filter_category("beginner"))
        filter_layout.addWidget(self.beginner_btn)

        self.intermediate_btn = QPushButton("Intermediate")
        self.intermediate_btn.setCheckable(True)
        self.intermediate_btn.clicked.connect(lambda: self._filter_category("intermediate"))
        filter_layout.addWidget(self.intermediate_btn)

        self.advanced_btn = QPushButton("Advanced")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.clicked.connect(lambda: self._filter_category("advanced"))
        filter_layout.addWidget(self.advanced_btn)

        self.custom_btn = QPushButton("My Recipes")
        self.custom_btn.setCheckable(True)
        self.custom_btn.clicked.connect(lambda: self._filter_category("custom"))
        filter_layout.addWidget(self.custom_btn)

        filter_layout.addStretch()
        main_layout.addLayout(filter_layout)

        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter by name, algorithm, or keyword...")
        self.search_input.textChanged.connect(self._apply_search_filter)
        search_layout.addWidget(self.search_input)

        main_layout.addLayout(search_layout)

        # Scrollable card grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff if hasattr(Qt, 'ScrollBarPolicy') else Qt.ScrollBarAlwaysOff)

        self.card_container = QWidget()
        self.card_layout = QGridLayout(self.card_container)
        self.card_layout.setSpacing(12)
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignTop if hasattr(Qt, 'AlignmentFlag') else Qt.AlignTop)

        scroll_area.setWidget(self.card_container)
        main_layout.addWidget(scroll_area, 1)

        # Selected recipe details panel
        details_frame = QFrame()
        details_frame.setFrameStyle(QFrame.Shape.StyledPanel if hasattr(QFrame, 'Shape') else QFrame.StyledPanel)
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(12, 12, 12, 12)

        self.selected_name_label = QLabel("Selected: None")
        selected_font = QFont()
        selected_font.setBold(True)
        self.selected_name_label.setFont(selected_font)
        details_layout.addWidget(self.selected_name_label)

        self.selected_desc_label = QLabel("Click a recipe card to see details.")
        self.selected_desc_label.setWordWrap(True)
        self.selected_desc_label.setMaximumHeight(60)
        details_layout.addWidget(self.selected_desc_label)

        self.selected_deps_label = QLabel("")
        details_layout.addWidget(self.selected_deps_label)

        main_layout.addWidget(details_frame)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.apply_btn = QPushButton("Apply Recipe")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_selected_recipe)
        button_layout.addWidget(self.apply_btn)

        self.copy_btn = QPushButton("Create Copy")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._create_copy)
        button_layout.addWidget(self.copy_btn)

        self.details_btn = QPushButton("View Details")
        self.details_btn.setEnabled(False)
        self.details_btn.clicked.connect(self._view_details)
        button_layout.addWidget(self.details_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(close_btn)

        main_layout.addLayout(button_layout)

    def _populate_cards(self):
        """Populate card grid with recipe cards."""
        # Clear existing cards
        for card in self.cards:
            card.setParent(None)
            card.deleteLater()
        self.cards.clear()

        # Calculate cards per row based on dialog width
        available_width = self.width() - 48  # Account for margins and scrollbar
        cards_per_row = max(1, available_width // (CARD_WIDTH + 12))

        # Create cards
        row = 0
        col = 0
        for recipe in self.filtered_recipes:
            card = RecipeCard(recipe, self.available_deps, self.card_container)
            card.clicked.connect(self._on_card_clicked)
            card.applyRequested.connect(self._apply_recipe)

            self.card_layout.addWidget(card, row, col)
            self.cards.append(card)

            col += 1
            if col >= cards_per_row:
                col = 0
                row += 1

    def _filter_category(self, category: str):
        """Filter recipes by category."""
        self.current_filter = category

        # Update button states
        self.all_btn.setChecked(category == "all")
        self.beginner_btn.setChecked(category == "beginner")
        self.intermediate_btn.setChecked(category == "intermediate")
        self.advanced_btn.setChecked(category == "advanced")
        self.custom_btn.setChecked(category == "custom")

        # Apply filter
        self._apply_filters()

    def _apply_search_filter(self):
        """Apply search text filter."""
        self._apply_filters()

    def _apply_filters(self):
        """Apply both category and search filters."""
        search_text = self.search_input.text().lower()

        self.filtered_recipes = []
        for recipe in self.recipes:
            # Category filter
            recipe_category = _get_recipe_category(recipe)
            if self.current_filter != "all":
                if self.current_filter == "custom":
                    # Custom recipes are user-created (not templates)
                    metadata = recipe.get("metadata", {})
                    is_template = metadata.get("is_template", False) if isinstance(metadata, dict) else False
                    if is_template:
                        continue
                elif recipe_category != self.current_filter:
                    continue

            # Search filter
            if search_text:
                name = recipe.get("name", "").lower()
                desc = recipe.get("description", "").lower()
                algo = _get_algorithm_name(recipe).lower()
                tags = " ".join(_get_feature_tags(recipe)).lower()

                if search_text not in name and search_text not in desc and search_text not in algo and search_text not in tags:
                    continue

            self.filtered_recipes.append(recipe)

        # Repopulate cards
        self._populate_cards()

    def _on_card_clicked(self, recipe: Dict[str, Any]):
        """Handle card click."""
        self.selected_recipe = recipe

        # Update card selection states
        for card in self.cards:
            card.setSelected(card.recipe == recipe)

        # Update details panel
        name = recipe.get("name", "Untitled Recipe")
        desc = recipe.get("description", "No description available.")

        self.selected_name_label.setText(f"Selected: {name}")
        self.selected_desc_label.setText(desc)

        # Check dependencies
        deps_met, missing = _check_dependencies_available(recipe, self.available_deps)
        if deps_met:
            self.selected_deps_label.setText("Requirements: ✓ All dependencies available")
            self.selected_deps_label.setStyleSheet("color: green;")
        else:
            missing_text = ", ".join(missing)
            self.selected_deps_label.setText(f"Requirements: ⚠ Missing: {missing_text}")
            self.selected_deps_label.setStyleSheet("color: orange;")

        # Enable buttons
        self.apply_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self.details_btn.setEnabled(True)

    def _apply_recipe(self, recipe: Dict[str, Any]):
        """Apply recipe and close dialog."""
        self.selected_recipe = recipe
        self.recipeSelected.emit(recipe)
        self.accept()

    def _apply_selected_recipe(self):
        """Apply currently selected recipe."""
        if self.selected_recipe:
            self._apply_recipe(self.selected_recipe)

    def _create_copy(self):
        """Create an editable copy of selected recipe."""
        if not self.selected_recipe:
            return

        QMessageBox.information(
            self,
            "Create Copy",
            "This feature will create an editable copy of the selected recipe.\n"
            "The copy will appear in 'My Recipes' and can be customized.",
        )

    def _view_details(self):
        """Show detailed recipe information."""
        if not self.selected_recipe:
            return

        # Create details dialog
        details_dialog = QDialog(self)
        details_dialog.setWindowTitle(f"Recipe Details: {self.selected_recipe.get('name', 'Untitled')}")
        details_dialog.resize(600, 500)

        layout = QVBoxLayout(details_dialog)

        # Show full recipe JSON
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(json.dumps(self.selected_recipe, indent=2))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(details_dialog.close)
        layout.addWidget(close_btn)

        details_dialog.exec()

    def resizeEvent(self, event):
        """Handle window resize to adjust card layout."""
        super().resizeEvent(event)
        self._populate_cards()


# Convenience function to show the dialog
def show_visual_recipe_shop(recipes: List[Dict[str, Any]], available_deps: Dict[str, bool], parent=None) -> Optional[Dict[str, Any]]:
    """Show visual recipe shop dialog and return selected recipe.

    Args:
        recipes: List of recipe dictionaries
        available_deps: Dict of dependency availability (e.g., {"sklearn": True, "xgboost": False})
        parent: Parent widget

    Returns:
        Selected recipe dict or None if canceled
    """
    dialog = VisualRecipeShopDialog(recipes, available_deps, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted if hasattr(QDialog, 'DialogCode') else QDialog.Accepted:
        return dialog.selected_recipe
    return None
