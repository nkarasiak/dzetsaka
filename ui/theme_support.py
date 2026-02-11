"""Theme-aware styling support for dzetsaka UI components.

Provides utilities for detecting and adapting to QGIS dark/light themes,
ensuring the plugin UI looks professional in all theme configurations.

Features:
    - Automatic theme detection using QPalette
    - Theme-aware base classes for dialogs and widgets
    - Dynamic stylesheet generation for dark/light modes
    - Color scheme utilities for charts and visualizations

Usage:
    from ui.theme_support import ThemeAwareWidget, get_current_theme, get_theme_colors

    class MyDialog(ThemeAwareWidget, QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.apply_theme()  # Automatically applies appropriate theme

Author:
    Nicolas Karasiak
"""

from typing import Dict, Tuple

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPalette, QColor
from qgis.PyQt.QtWidgets import QApplication, QWidget


class Theme:
    """Theme constants and detection."""

    DARK = "dark"
    LIGHT = "light"


def get_current_theme() -> str:
    """Detect current QGIS theme (dark or light).

    Returns
    -------
    str
        "dark" if dark theme is active, "light" otherwise
    """
    palette = QApplication.instance().palette()
    window_color = palette.color(QPalette.ColorRole.Window)

    # Calculate lightness (0-255, higher = lighter)
    lightness = window_color.lightness()

    # Threshold at 128 (middle of 0-255 range)
    return Theme.DARK if lightness < 128 else Theme.LIGHT


def get_theme_colors(theme: str = None) -> Dict[str, str]:
    """Get color scheme for specified theme.

    Parameters
    ----------
    theme : str, optional
        Theme name ("dark" or "light"). If None, auto-detects current theme.

    Returns
    -------
    dict
        Dictionary mapping color role names to hex color strings
    """
    if theme is None:
        theme = get_current_theme()

    if theme == Theme.DARK:
        return {
            # Background colors
            "background": "#2b2b2b",
            "background_alt": "#3c3c3c",
            "surface": "#323232",
            # Text colors
            "text": "#e0e0e0",
            "text_secondary": "#b0b0b0",
            "text_disabled": "#707070",
            # Border colors
            "border": "#555555",
            "border_light": "#666666",
            # Status colors
            "primary": "#4a9eff",
            "success": "#4caf50",
            "warning": "#ff9800",
            "error": "#e74c3c",
            "info": "#2196f3",
            # UI element colors
            "button_bg": "#424242",
            "button_hover": "#4a4a4a",
            "input_bg": "#383838",
            "selection": "#0078d4",
        }
    else:  # Light theme
        return {
            # Background colors
            "background": "#ffffff",
            "background_alt": "#f5f5f5",
            "surface": "#fafafa",
            # Text colors
            "text": "#212121",
            "text_secondary": "#666666",
            "text_disabled": "#9e9e9e",
            # Border colors
            "border": "#cccccc",
            "border_light": "#e0e0e0",
            # Status colors
            "primary": "#1976d2",
            "success": "#388e3c",
            "warning": "#f57c00",
            "error": "#d32f2f",
            "info": "#0288d1",
            # UI element colors
            "button_bg": "#e0e0e0",
            "button_hover": "#d0d0d0",
            "input_bg": "#ffffff",
            "selection": "#0078d4",
        }


def get_chart_colors(theme: str = None) -> Tuple[str, ...]:
    """Get color palette for charts and visualizations.

    Parameters
    ----------
    theme : str, optional
        Theme name ("dark" or "light"). If None, auto-detects current theme.

    Returns
    -------
    tuple of str
        Tuple of hex color strings suitable for charts
    """
    if theme is None:
        theme = get_current_theme()

    if theme == Theme.DARK:
        # Colors that stand out on dark backgrounds
        return (
            "#4a9eff",  # Blue
            "#4caf50",  # Green
            "#ff9800",  # Orange
            "#e74c3c",  # Red
            "#9c27b0",  # Purple
            "#00bcd4",  # Cyan
            "#ffeb3b",  # Yellow
            "#795548",  # Brown
        )
    else:
        # Colors that work on light backgrounds
        return (
            "#1976d2",  # Blue
            "#388e3c",  # Green
            "#f57c00",  # Orange
            "#d32f2f",  # Red
            "#7b1fa2",  # Purple
            "#0097a7",  # Cyan
            "#fbc02d",  # Yellow
            "#5d4037",  # Brown
        )


class ThemeAwareWidget:
    """Mixin class for widgets that adapt to QGIS theme.

    Usage:
        class MyWidget(ThemeAwareWidget, QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.apply_theme()
    """

    def apply_theme(self):
        """Apply theme-appropriate styling to the widget."""
        theme = get_current_theme()
        colors = get_theme_colors(theme)

        # Generate stylesheet
        stylesheet = self._generate_stylesheet(theme, colors)

        # Apply to widget
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(stylesheet)

    def _generate_stylesheet(self, theme: str, colors: Dict[str, str]) -> str:
        """Generate CSS stylesheet for the current theme.

        Parameters
        ----------
        theme : str
            Theme name ("dark" or "light")
        colors : dict
            Color scheme dictionary

        Returns
        -------
        str
            CSS stylesheet string
        """
        return f"""
            QWidget {{
                background-color: {colors['background']};
                color: {colors['text']};
            }}

            QGroupBox {{
                border: 1px solid {colors['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: {colors['surface']};
                font-weight: bold;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: {colors['text']};
            }}

            QLabel {{
                color: {colors['text']};
                background-color: transparent;
            }}

            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 4px;
                color: {colors['text']};
                selection-background-color: {colors['selection']};
            }}

            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border: 1px solid {colors['primary']};
            }}

            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
                color: {colors['text_disabled']};
                background-color: {colors['background_alt']};
            }}

            QPushButton {{
                background-color: {colors['button_bg']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 6px 12px;
                color: {colors['text']};
            }}

            QPushButton:hover {{
                background-color: {colors['button_hover']};
                border: 1px solid {colors['border_light']};
            }}

            QPushButton:pressed {{
                background-color: {colors['border']};
            }}

            QPushButton:disabled {{
                color: {colors['text_disabled']};
                background-color: {colors['background_alt']};
            }}

            QComboBox {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 4px;
                color: {colors['text']};
            }}

            QComboBox:hover {{
                border: 1px solid {colors['primary']};
            }}

            QComboBox:disabled {{
                color: {colors['text_disabled']};
                background-color: {colors['background_alt']};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                selection-background-color: {colors['selection']};
                color: {colors['text']};
            }}

            QSpinBox, QDoubleSpinBox {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 4px;
                color: {colors['text']};
            }}

            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {colors['primary']};
            }}

            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                color: {colors['text_disabled']};
                background-color: {colors['background_alt']};
            }}

            QCheckBox, QRadioButton {{
                color: {colors['text']};
                spacing: 5px;
            }}

            QCheckBox:disabled, QRadioButton:disabled {{
                color: {colors['text_disabled']};
            }}

            QTableWidget, QListWidget, QTreeWidget {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['border']};
                color: {colors['text']};
                selection-background-color: {colors['selection']};
                alternate-background-color: {colors['background_alt']};
            }}

            QTableWidget::item, QListWidget::item, QTreeWidget::item {{
                padding: 4px;
            }}

            QHeaderView::section {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                padding: 4px;
                color: {colors['text']};
                font-weight: bold;
            }}

            QScrollBar:vertical, QScrollBar:horizontal {{
                background: {colors['background_alt']};
                border: 1px solid {colors['border']};
            }}

            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background: {colors['button_bg']};
                border-radius: 4px;
                min-height: 20px;
                min-width: 20px;
            }}

            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background: {colors['button_hover']};
            }}

            QScrollBar::add-line, QScrollBar::sub-line {{
                background: none;
                border: none;
            }}

            QProgressBar {{
                border: 1px solid {colors['border']};
                border-radius: 4px;
                text-align: center;
                background-color: {colors['background_alt']};
                color: {colors['text']};
            }}

            QProgressBar::chunk {{
                background-color: {colors['primary']};
                border-radius: 3px;
            }}

            QTabWidget::pane {{
                border: 1px solid {colors['border']};
                background-color: {colors['surface']};
            }}

            QTabBar::tab {{
                background-color: {colors['background_alt']};
                border: 1px solid {colors['border']};
                padding: 8px 16px;
                color: {colors['text']};
            }}

            QTabBar::tab:selected {{
                background-color: {colors['surface']};
                border-bottom-color: {colors['surface']};
                font-weight: bold;
            }}

            QTabBar::tab:hover {{
                background-color: {colors['button_hover']};
            }}

            QToolTip {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 4px;
            }}
        """

    def get_current_theme(self) -> str:
        """Get current theme name.

        Returns
        -------
        str
            "dark" or "light"
        """
        return get_current_theme()

    def get_theme_color(self, color_name: str) -> str:
        """Get specific theme color by name.

        Parameters
        ----------
        color_name : str
            Color name (e.g., "primary", "error", "text")

        Returns
        -------
        str
            Hex color string
        """
        colors = get_theme_colors()
        return colors.get(color_name, "#000000")


def apply_matplotlib_theme(ax, theme: str = None):
    """Apply theme styling to matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to style
    theme : str, optional
        Theme name ("dark" or "light"). If None, auto-detects.
    """
    if theme is None:
        theme = get_current_theme()

    colors = get_theme_colors(theme)

    # Set background colors
    ax.set_facecolor(colors["surface"])
    ax.figure.patch.set_facecolor(colors["background"])

    # Set text colors
    ax.title.set_color(colors["text"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])

    # Set tick colors
    ax.tick_params(colors=colors["text"], which="both")

    # Set spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["border"])

    # Set grid color
    ax.grid(True, alpha=0.3, color=colors["border_light"])
