"""Recommendation dialog for displaying recipe suggestions.

This module provides a user-friendly dialog that shows recommended recipes
based on the analyzed raster characteristics, with confidence scores and
explanations for each recommendation.

Author:
    Nicolas Karasiak
"""

from typing import Any, Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import QSettings, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class RecommendationDialog(QDialog):
    """Dialog that displays recommended recipes with confidence scores."""

    recipeSelected = pyqtSignal(dict)  # Emits the selected recipe

    def __init__(
        self,
        recommendations: List[Tuple[Dict[str, Any], float, str]],
        raster_info: Dict[str, Any],
        parent: Optional[QWidget] = None,
    ):
        """Initialize the recommendation dialog.

        Parameters
        ----------
        recommendations : List[Tuple[Dict[str, Any], float, str]]
            List of (recipe, confidence_score, reason) tuples
        raster_info : Dict[str, Any]
            Information about the analyzed raster
        parent : Optional[QWidget]
            Parent widget

        """
        super(RecommendationDialog, self).__init__(parent)
        self.recommendations = recommendations
        self.raster_info = raster_info
        self.selected_recipe = None

        self.setWindowTitle("Recipe Recommendations")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Header with raster info
        header = self._create_header()
        layout.addWidget(header)

        # Scrollable area for recommendations
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        # Add recommendation cards (top 3 or all if fewer)
        max_recommendations = min(5, len(self.recommendations))
        for i in range(max_recommendations):
            recipe, score, reason = self.recommendations[i]
            card = self._create_recommendation_card(recipe, score, reason, i)
            scroll_layout.addWidget(card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Footer with options
        footer = self._create_footer()
        layout.addWidget(footer)

        self.setLayout(layout)

    def _create_header(self) -> QWidget:
        """Create the header with raster information.

        Returns
        -------
        QWidget
            Header widget

        """
        header = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("<b>Recommended Recipes for Your Raster</b>")
        title.setStyleSheet("font-size: 14pt; padding: 5px;")
        header_layout.addWidget(title)

        # Raster info summary
        info_text = self._format_raster_info()
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #555; padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
        header_layout.addWidget(info_label)

        header.setLayout(header_layout)
        return header

    def _format_raster_info(self) -> str:
        """Format raster information for display.

        Returns
        -------
        str
            Formatted info text

        """
        band_count = self.raster_info.get("band_count", 0)
        file_size_mb = self.raster_info.get("file_size_mb", 0.0)
        detected_sensor = self.raster_info.get("detected_sensor", "unknown")
        landcover_type = self.raster_info.get("landcover_type", "unknown")
        width = self.raster_info.get("width", 0)
        height = self.raster_info.get("height", 0)

        parts = []

        # Basic info
        parts.append(f"<b>Bands:</b> {band_count}")

        if width and height:
            parts.append(f"<b>Size:</b> {width} × {height} pixels")

        if file_size_mb > 0:
            parts.append(f"<b>File:</b> {file_size_mb:.1f} MB")

        # Detected characteristics
        if detected_sensor != "unknown":
            sensor_display = detected_sensor.replace("_", " ").title()
            parts.append(f"<b>Sensor:</b> {sensor_display}")

        if landcover_type != "unknown":
            parts.append(f"<b>Type:</b> {landcover_type.capitalize()}")

        return " &nbsp;•&nbsp; ".join(parts)

    def _create_recommendation_card(
        self, recipe: Dict[str, Any], score: float, reason: str, index: int
    ) -> QWidget:
        """Create a card for a single recipe recommendation.

        Parameters
        ----------
        recipe : Dict[str, Any]
            Recipe dictionary
        score : float
            Confidence score (0-100)
        reason : str
            Explanation for the recommendation
        index : int
            Card index (0-based)

        Returns
        -------
        QWidget
            Recommendation card widget

        """
        card = QWidget()
        card.setStyleSheet(
            """
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
            QWidget:hover {
                border: 1px solid #4CAF50;
                background-color: #f9fff9;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Header row: rank, name, and confidence
        header_row = QHBoxLayout()

        # Rank badge
        rank_label = QLabel(f"#{index + 1}")
        rank_label.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "padding: 5px 10px; border-radius: 12px; font-weight: bold;"
        )
        rank_label.setFixedWidth(40)
        header_row.addWidget(rank_label)

        # Recipe name
        recipe_name = recipe.get("name", "Unnamed Recipe")
        name_label = QLabel(f"<b>{recipe_name}</b>")
        name_label.setWordWrap(True)
        header_row.addWidget(name_label, 1)

        # Star rating
        stars = self._get_star_rating(score)
        star_label = QLabel(stars)
        star_label.setStyleSheet("font-size: 16pt;")
        header_row.addWidget(star_label)

        layout.addLayout(header_row)

        # Confidence bar
        confidence_bar = self._create_confidence_bar(score)
        layout.addWidget(confidence_bar)

        # Recipe description
        description = recipe.get("description", "")
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #555; font-style: italic;")
            layout.addWidget(desc_label)

        # Why recommended
        reason_label = QLabel(f"<b>Why recommended:</b> {reason}")
        reason_label.setWordWrap(True)
        reason_label.setStyleSheet("color: #333; margin-top: 5px;")
        layout.addWidget(reason_label)

        # Expected performance (if available)
        perf_text = self._format_performance(recipe)
        if perf_text:
            perf_label = QLabel(perf_text)
            perf_label.setWordWrap(True)
            perf_label.setStyleSheet("color: #666; font-size: 9pt;")
            layout.addWidget(perf_label)

        # Apply button
        apply_btn = QPushButton("Apply This Recipe")
        apply_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        )
        apply_btn.clicked.connect(lambda: self._on_apply_recipe(recipe))
        layout.addWidget(apply_btn)

        card.setLayout(layout)
        return card

    def _create_confidence_bar(self, score: float) -> QWidget:
        """Create a visual confidence bar.

        Parameters
        ----------
        score : float
            Confidence score (0-100)

        Returns
        -------
        QWidget
            Confidence bar widget

        """
        container = QWidget()
        container.setFixedHeight(30)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Confidence label
        confidence_class = self._get_confidence_class(score)
        conf_label = QLabel(f"{confidence_class}: {score:.0f}%")
        conf_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(conf_label)

        # Progress bar-like visual
        bar_container = QWidget()
        bar_container.setFixedHeight(20)
        bar_container.setStyleSheet("background-color: #eee; border-radius: 10px;")
        bar_layout = QHBoxLayout(bar_container)
        bar_layout.setContentsMargins(2, 2, 2, 2)

        # Filled portion
        fill = QWidget()
        fill_width = int((score / 100.0) * 200)  # Max width 200px
        fill.setFixedWidth(fill_width)
        fill.setFixedHeight(16)

        # Color based on score
        if score >= 80:
            color = "#4CAF50"  # Green
        elif score >= 60:
            color = "#FFC107"  # Amber
        else:
            color = "#FF9800"  # Orange

        fill.setStyleSheet(f"background-color: {color}; border-radius: 8px;")
        bar_layout.addWidget(fill)
        bar_layout.addStretch()

        layout.addWidget(bar_container)

        container.setLayout(layout)
        return container

    def _format_performance(self, recipe: Dict[str, Any]) -> str:
        """Format expected performance information.

        Parameters
        ----------
        recipe : Dict[str, Any]
            Recipe dictionary

        Returns
        -------
        str
            Formatted performance text

        """
        parts = []

        runtime_class = recipe.get("expected_runtime_class", "")
        if runtime_class:
            runtime_map = {
                "fast": "Fast (~minutes)",
                "medium": "Medium (~10-30 min)",
                "slow": "Slow (30+ min)",
            }
            runtime_text = runtime_map.get(runtime_class.lower(), runtime_class)
            parts.append(f"<b>Runtime:</b> {runtime_text}")

        accuracy_class = recipe.get("expected_accuracy_class", "")
        if accuracy_class:
            accuracy_map = {
                "high": "High accuracy",
                "medium": "Medium accuracy",
                "low": "Lower accuracy",
            }
            accuracy_text = accuracy_map.get(accuracy_class.lower(), accuracy_class)
            parts.append(f"<b>Accuracy:</b> {accuracy_text}")

        classifier = recipe.get("classifier", "")
        if classifier:
            parts.append(f"<b>Algorithm:</b> {classifier}")

        return " &nbsp;•&nbsp; ".join(parts)

    def _get_star_rating(self, score: float) -> str:
        """Get star rating from score.

        Parameters
        ----------
        score : float
            Confidence score (0-100)

        Returns
        -------
        str
            Star rating string

        """
        stars = int(round(score / 20.0))
        return "⭐" * max(1, min(5, stars))

    def _get_confidence_class(self, score: float) -> str:
        """Get confidence class description.

        Parameters
        ----------
        score : float
            Confidence score (0-100)

        Returns
        -------
        str
            Confidence class

        """
        if score >= 95:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Low"

    def _create_footer(self) -> QWidget:
        """Create the footer with action buttons.

        Returns
        -------
        QWidget
            Footer widget

        """
        footer = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)

        # "Don't show again" checkbox
        self.dont_show_checkbox = QCheckBox("Don't show recommendations again")
        self.dont_show_checkbox.setToolTip("You can re-enable this in plugin settings")
        layout.addWidget(self.dont_show_checkbox)

        # Button box
        button_box = QDialogButtonBox()

        # Show all recipes button
        show_all_btn = QPushButton("Show All Recipes")
        show_all_btn.clicked.connect(self._on_show_all)
        button_box.addButton(show_all_btn, QDialogButtonBox.ActionRole)

        # Close button
        close_btn = button_box.addButton(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.reject)

        layout.addWidget(button_box)

        footer.setLayout(layout)
        return footer

    def _on_apply_recipe(self, recipe: Dict[str, Any]):
        """Handle recipe application.

        Parameters
        ----------
        recipe : Dict[str, Any]
            Selected recipe

        """
        self.selected_recipe = recipe

        # Save preference if checkbox is checked
        if self.dont_show_checkbox.isChecked():
            settings = QSettings()
            settings.setValue("/dzetsaka/show_recommendations", False)

        self.recipeSelected.emit(recipe)
        self.accept()

    def _on_show_all(self):
        """Handle show all recipes button click."""
        # Just close and let the main UI handle showing all recipes
        self.reject()

    def should_show_recommendations(self) -> bool:
        """Check if recommendations should be shown based on settings.

        Returns
        -------
        bool
            True if recommendations should be shown

        """
        settings = QSettings()
        return settings.value("/dzetsaka/show_recommendations", True, bool)

    @staticmethod
    def set_recommendations_enabled(enabled: bool):
        """Enable or disable showing recommendations.

        Parameters
        ----------
        enabled : bool
            Whether to show recommendations

        """
        settings = QSettings()
        settings.setValue("/dzetsaka/show_recommendations", enabled)
