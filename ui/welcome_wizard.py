"""First-run welcome wizard for dzetsaka plugin.

This wizard provides a comprehensive onboarding experience for new users,
including feature overview, dependency installation, and quick start options.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QFont, QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

# Import theme support
try:
    from .theme_support import ThemeAwareWidget
    _THEME_SUPPORT_AVAILABLE = True
except Exception:
    _THEME_SUPPORT_AVAILABLE = False
    # Fallback: create empty mixin class
    class ThemeAwareWidget:
        """Fallback mixin when theme_support is not available."""
        def apply_theme(self):
            pass


class WelcomeWizard(ThemeAwareWidget, QWizard):
    """First-run welcome wizard for dzetsaka plugin.

    Provides a 3-page guided experience:
    1. Feature overview with visual showcase
    2. Optional dependency installation
    3. Quick start options (sample data or user data)

    The wizard sets a QSettings flag when completed to prevent showing
    on subsequent plugin loads.
    """

    # Wizard page IDs
    PAGE_OVERVIEW = 0
    PAGE_DEPENDENCIES = 1
    PAGE_QUICKSTART = 2

    def __init__(self, plugin, parent=None):
        """Initialize the welcome wizard.

        Parameters
        ----------
        plugin : DzetsakaGUI
            Parent plugin instance for accessing settings, logger, etc.
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.plugin = plugin

        # Apply theme-aware styling
        if _THEME_SUPPORT_AVAILABLE:
            self.apply_theme()

        # Configure wizard appearance
        self.setWindowTitle("Welcome to dzetsaka")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)
        self.setMinimumSize(700, 500)

        # Add pages
        self.setPage(self.PAGE_OVERVIEW, OverviewPage(self.plugin))
        self.setPage(self.PAGE_DEPENDENCIES, DependencyCheckPage(self.plugin))
        self.setPage(self.PAGE_QUICKSTART, QuickStartPage(self.plugin))

        # Set starting page
        self.setStartId(self.PAGE_OVERVIEW)

        # Connect finished signal to mark wizard as completed
        self.finished.connect(self._on_wizard_finished)

    def _on_wizard_finished(self, result):
        """Handle wizard completion.

        Parameters
        ----------
        result : int
            QDialog.DialogCode result (Accepted or Rejected)
        """
        if result == QWizard.DialogCode.Accepted:
            # Mark wizard as completed in settings
            self.plugin.settings.setValue("/dzetsaka/welcomeCompleted", True)
            self.plugin.log.info("Welcome wizard completed successfully")


class OverviewPage(QWizardPage):
    """First wizard page: Feature overview and welcome message."""

    def __init__(self, plugin):
        """Initialize the overview page.

        Parameters
        ----------
        plugin : DzetsakaGUI
            Parent plugin instance
        """
        super().__init__()
        self.plugin = plugin

        self.setTitle("Welcome to dzetsaka")
        self.setSubTitle("A powerful QGIS plugin for machine learning-based raster classification")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create scrollable area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Introduction text
        intro_label = QLabel(
            "<p style='font-size: 11pt;'>"
            "dzetsaka helps you classify satellite imagery and other raster data using "
            "state-of-the-art machine learning algorithms. Whether you're mapping land cover, "
            "identifying crops, or analyzing multispectral data, dzetsaka provides the tools "
            "you need for accurate classification."
            "</p>"
        )
        intro_label.setWordWrap(True)
        content_layout.addWidget(intro_label)

        # Feature highlights section
        features_label = QLabel("<b style='font-size: 12pt;'>Key Features</b>")
        content_layout.addWidget(features_label)

        # Feature grid
        features = [
            ("🤖", "12 ML Algorithms", "GMM, RF, SVM, KNN, XGBoost, LightGBM, CatBoost, Extra Trees, Gradient Boosting, Logistic Regression, Naive Bayes, MLP"),
            ("⚡", "Automatic Optimization", "Optuna-powered hyperparameter tuning with cross-validation"),
            ("🔍", "Model Explainability", "SHAP values to understand feature importance and predictions"),
            ("⚖️", "Class Imbalance Handling", "SMOTE and other techniques for balanced training"),
            ("📊", "Comprehensive Reports", "Confusion matrices, accuracy metrics, and visual heatmaps"),
            ("🎯", "Recipe System", "Save and share complete classification workflows"),
        ]

        for emoji, title, description in features:
            feature_widget = self._create_feature_item(emoji, title, description)
            content_layout.addWidget(feature_widget)

        # Sample image (if available)
        self._add_sample_image(content_layout)

        # Add stretch to push content to top
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

    def _create_feature_item(self, emoji, title, description):
        """Create a feature highlight item.

        Parameters
        ----------
        emoji : str
            Emoji icon for the feature
        title : str
            Feature title
        description : str
            Feature description

        Returns
        -------
        QWidget
            Feature item widget
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)

        # Emoji label
        emoji_label = QLabel(emoji)
        emoji_font = QFont()
        emoji_font.setPointSize(20)
        emoji_label.setFont(emoji_font)
        emoji_label.setFixedWidth(40)
        layout.addWidget(emoji_label)

        # Text layout
        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)

        title_label = QLabel(f"<b>{title}</b>")
        text_layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666;")
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        return widget

    def _add_sample_image(self, layout):
        """Add sample classification image if available.

        Parameters
        ----------
        layout : QVBoxLayout
            Layout to add image to
        """
        # Try to find sample classification result image
        plugin_dir = Path(self.plugin.plugin_dir)
        sample_images = [
            plugin_dir / "docs" / "images" / "classification_example.png",
            plugin_dir / "docs" / "classification_result.png",
            plugin_dir / "images" / "sample.png",
        ]

        for image_path in sample_images:
            if image_path.exists():
                try:
                    pixmap = QPixmap(str(image_path))
                    if not pixmap.isNull():
                        image_label = QLabel()
                        # Scale to reasonable size while maintaining aspect ratio
                        scaled_pixmap = pixmap.scaledToWidth(
                            500,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        image_label.setPixmap(scaled_pixmap)
                        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                        layout.addSpacing(10)
                        sample_header = QLabel("<b>Example Classification Result</b>")
                        sample_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        layout.addWidget(sample_header)
                        layout.addWidget(image_label)
                        break
                except Exception as e:
                    self.plugin.log.warning(f"Could not load sample image: {e}")


class DependencyCheckPage(QWizardPage):
    """Second wizard page: Dependency status and installation."""

    def __init__(self, plugin):
        """Initialize the dependency check page.

        Parameters
        ----------
        plugin : DzetsakaGUI
            Parent plugin instance
        """
        super().__init__()
        self.plugin = plugin
        self._installation_in_progress = False

        self.setTitle("Install Dependencies")
        self.setSubTitle("Install optional Python libraries for advanced features")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Explanation text
        explanation = QLabel(
            "<p>dzetsaka uses several Python libraries for different algorithms. "
            "Core features work without additional installations, but installing "
            "the full bundle unlocks all capabilities.</p>"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Dependency status section
        status_label = QLabel("<b>Dependency Status</b>")
        layout.addWidget(status_label)

        # Status list container
        self.status_container = QWidget()
        self.status_layout = QVBoxLayout(self.status_container)
        self.status_layout.setSpacing(8)
        layout.addWidget(self.status_container)

        # Populate dependency status
        self._update_dependency_status()

        # Add stretch
        layout.addStretch()

        # Installation buttons section
        button_layout = QHBoxLayout()

        self.install_button = QPushButton("Install Full Bundle")
        self.install_button.setMinimumHeight(40)
        font = self.install_button.font()
        font.setPointSize(10)
        font.setBold(True)
        self.install_button.setFont(font)
        self.install_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.install_button.clicked.connect(self._on_install_clicked)
        button_layout.addWidget(self.install_button)

        self.skip_checkbox = QCheckBox("Skip for now (can install later from dashboard)")
        button_layout.addWidget(self.skip_checkbox)

        layout.addLayout(button_layout)

        # Progress indicator (hidden initially)
        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

    def _update_dependency_status(self):
        """Update the dependency status display."""
        from dzetsaka.qgis.dependency_catalog import FULL_DEPENDENCY_BUNDLE

        # Clear existing status items
        while self.status_layout.count():
            child = self.status_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Check each dependency
        dependency_info = {
            "scikit-learn": ("sklearn", "Core algorithms: RF, SVM, KNN, ET, GBC, LR, NB, MLP"),
            "xgboost": ("xgboost", "XGBoost gradient boosting algorithm"),
            "lightgbm": ("lightgbm", "LightGBM gradient boosting algorithm"),
            "catboost": ("catboost", "CatBoost gradient boosting algorithm"),
            "optuna": ("optuna", "Hyperparameter optimization"),
            "shap": ("shap", "Model explainability with SHAP values"),
            "seaborn": ("seaborn", "Enhanced report visualizations"),
            "imbalanced-learn": ("imblearn", "Class imbalance handling (SMOTE)"),
        }

        for package in FULL_DEPENDENCY_BUNDLE:
            if package in dependency_info:
                import_name, description = dependency_info[package]
                installed = self._is_package_installed(import_name)
                status_widget = self._create_status_item(package, description, installed)
                self.status_layout.addWidget(status_widget)

    def _is_package_installed(self, module_name):
        """Check if a package is installed.

        Parameters
        ----------
        module_name : str
            Module name to check

        Returns
        -------
        bool
            True if package is installed and importable
        """
        return importlib.util.find_spec(module_name) is not None

    def _create_status_item(self, package, description, installed):
        """Create a dependency status item.

        Parameters
        ----------
        package : str
            Package name
        description : str
            Package description
        installed : bool
            Whether package is installed

        Returns
        -------
        QWidget
            Status item widget
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(15, 5, 15, 5)

        # Status icon
        status_icon = QLabel("✓" if installed else "○")
        status_icon.setStyleSheet(
            f"color: {'#4CAF50' if installed else '#999999'}; "
            f"font-size: 16pt; font-weight: bold;"
        )
        status_icon.setFixedWidth(30)
        layout.addWidget(status_icon)

        # Package info
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        name_label = QLabel(f"<b>{package}</b>")
        name_label.setStyleSheet(f"color: {'#000000' if installed else '#666666'};")
        text_layout.addWidget(name_label)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 9pt;")
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        # Status text
        status_text = QLabel("Installed" if installed else "Not installed")
        status_text.setStyleSheet(
            f"color: {'#4CAF50' if installed else '#FF9800'}; "
            f"font-weight: bold;"
        )
        layout.addWidget(status_text)

        return widget

    def _on_install_clicked(self):
        """Handle install button click."""
        if self._installation_in_progress:
            return

        from dzetsaka.qgis.dependency_catalog import FULL_DEPENDENCY_BUNDLE

        self._installation_in_progress = True
        self.install_button.setEnabled(False)
        self.install_button.setText("Installing...")
        self.progress_label.setText(
            "Installing dependencies... This may take several minutes. "
            "The wizard will remain responsive, and you'll be notified when complete."
        )
        self.progress_label.show()

        def on_installation_complete(success):
            """Handle installation completion.

            Parameters
            ----------
            success : bool
                Whether installation succeeded
            """
            self._installation_in_progress = False
            self.install_button.setEnabled(True)

            if success:
                self.install_button.setText("Installation Complete")
                self.install_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 4px;
                        padding: 8px 16px;
                    }
                """)
                self.progress_label.setText(
                    "<b>✓ Installation complete!</b> All dependencies are now available. "
                    "Note: You may need to restart QGIS for all changes to take effect."
                )
                self.progress_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

                # Refresh status display
                QTimer.singleShot(1000, self._update_dependency_status)
            else:
                self.install_button.setText("Install Full Bundle")
                self.progress_label.setText(
                    "Installation encountered issues. You can try again or skip for now. "
                    "Dependencies can also be installed later from the dashboard."
                )
                self.progress_label.setStyleSheet("color: #FF9800; font-weight: bold;")

        # Start async installation
        try:
            self.plugin._try_install_dependencies_async(
                FULL_DEPENDENCY_BUNDLE,
                on_installation_complete
            )
        except Exception as e:
            self.plugin.log.error(f"Error starting dependency installation: {e}")
            self._installation_in_progress = False
            self.install_button.setEnabled(True)
            self.install_button.setText("Install Full Bundle")
            self.progress_label.setText(f"Error: {e}")
            self.progress_label.setStyleSheet("color: #f44336; font-weight: bold;")

    def validatePage(self):
        """Validate page before moving to next.

        Returns
        -------
        bool
            True if can proceed to next page
        """
        # Allow proceeding even if installation is in progress
        # (async installation continues in background)
        return True


class QuickStartPage(QWizardPage):
    """Third wizard page: Quick start options."""

    # Signals
    loadSampleDataRequested = pyqtSignal()

    def __init__(self, plugin):
        """Initialize the quick start page.

        Parameters
        ----------
        plugin : DzetsakaGUI
            Parent plugin instance
        """
        super().__init__()
        self.plugin = plugin

        self.setTitle("Get Started")
        self.setSubTitle("Choose how you'd like to begin using dzetsaka")

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Introduction
        intro_label = QLabel(
            "<p>You're all set! Choose one of the options below to start classifying.</p>"
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        # Option 1: Sample data
        sample_widget = self._create_option_widget(
            "🎓",
            "Try Sample Data",
            "Load example raster and training data to explore dzetsaka's features",
            self._on_sample_data_clicked,
            "#2196F3"
        )
        layout.addWidget(sample_widget)

        # Option 2: User data
        user_widget = self._create_option_widget(
            "🚀",
            "Use My Data",
            "Close this wizard and open the dashboard to classify your own data",
            self._on_user_data_clicked,
            "#4CAF50"
        )
        layout.addWidget(user_widget)

        # Quick tips section
        layout.addSpacing(20)
        tips_label = QLabel("<b>Quick Tips</b>")
        layout.addWidget(tips_label)

        tips = [
            "Use the Recipe system to save and share complete classification workflows",
            "Enable Optuna optimization for automatic hyperparameter tuning",
            "Check SHAP explainability to understand which features drive predictions",
            "Use spatial validation to avoid overestimating accuracy with spatial data",
        ]

        tips_widget = QWidget()
        tips_layout = QVBoxLayout(tips_widget)
        tips_layout.setSpacing(8)
        tips_layout.setContentsMargins(15, 10, 15, 10)

        for tip in tips:
            tip_label = QLabel(f"• {tip}")
            tip_label.setWordWrap(True)
            tip_label.setStyleSheet("color: #666;")
            tips_layout.addWidget(tip_label)

        layout.addWidget(tips_widget)

        # Add stretch to push content to top
        layout.addStretch()

    def _create_option_widget(self, emoji, title, description, callback, color):
        """Create a quick start option widget.

        Parameters
        ----------
        emoji : str
            Emoji icon
        title : str
            Option title
        description : str
            Option description
        callback : callable
            Click callback function
        color : str
            Button color (hex)

        Returns
        -------
        QWidget
            Option widget
        """
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 6px;
                padding: 15px;
            }
        """)

        layout = QHBoxLayout(widget)
        layout.setSpacing(15)

        # Emoji
        emoji_label = QLabel(emoji)
        emoji_font = QFont()
        emoji_font.setPointSize(28)
        emoji_label.setFont(emoji_font)
        emoji_label.setFixedWidth(50)
        emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(emoji_label)

        # Text content
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)

        title_label = QLabel(f"<b style='font-size: 12pt;'>{title}</b>")
        text_layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666;")
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        # Button
        button = QPushButton("Choose")
        button.setMinimumHeight(35)
        button.setMinimumWidth(100)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
        """)
        button.clicked.connect(callback)
        layout.addWidget(button)

        return widget

    def _darken_color(self, hex_color):
        """Darken a hex color by 10%.

        Parameters
        ----------
        hex_color : str
            Hex color string (e.g., "#4CAF50")

        Returns
        -------
        str
            Darkened hex color
        """
        # Simple darkening: multiply each RGB component by 0.9
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r, g, b = int(r * 0.9), int(g * 0.9), int(b * 0.9)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_sample_data_clicked(self):
        """Handle sample data button click."""
        self.plugin.log.info("User selected: Try Sample Data")

        # Try to load sample data
        sample_loaded = self._load_sample_data()

        if sample_loaded:
            # Close wizard and open dashboard
            self.wizard().accept()
            QTimer.singleShot(300, self.plugin.open_dashboard)
        else:
            # Show info message and close anyway
            from qgis.PyQt.QtWidgets import QMessageBox
            _ok = getattr(getattr(QMessageBox, "StandardButton", None), "Ok", None) or QMessageBox.Ok  # type: ignore[attr-defined]
            QMessageBox.information(
                self,
                "Sample Data",
                "Sample data files not found. The dashboard will open where you can load your own data.",
                _ok,
            )
            self.wizard().accept()
            QTimer.singleShot(300, self.plugin.open_dashboard)

    def _on_user_data_clicked(self):
        """Handle user data button click."""
        self.plugin.log.info("User selected: Use My Data")

        # Close wizard and open dashboard
        self.wizard().accept()
        QTimer.singleShot(300, self.plugin.open_dashboard)

    def _load_sample_data(self):
        """Try to load sample data into QGIS.

        Returns
        -------
        bool
            True if sample data was loaded successfully
        """
        try:
            from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

            plugin_dir = Path(self.plugin.plugin_dir)
            sample_dir = plugin_dir / "data" / "sample"

            if not sample_dir.exists():
                self.plugin.log.warning(f"Sample data directory not found: {sample_dir}")
                return False

            # Look for sample raster
            raster_files = list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.tiff"))
            if raster_files:
                raster_path = str(raster_files[0])
                raster_layer = QgsRasterLayer(raster_path, "Sample Raster")
                if raster_layer.isValid():
                    QgsProject.instance().addMapLayer(raster_layer)
                    self.plugin.log.info(f"Loaded sample raster: {raster_path}")
                else:
                    self.plugin.log.warning(f"Invalid raster layer: {raster_path}")

            # Look for sample vector (training data)
            vector_files = (
                list(sample_dir.glob("*.shp")) +
                list(sample_dir.glob("*.gpkg")) +
                list(sample_dir.glob("*.geojson")) +
                list(sample_dir.glob("*.geoparquet*"))
            )
            if vector_files:
                vector_path = str(vector_files[0])
                vector_layer = QgsVectorLayer(vector_path, "Training Data", "ogr")
                if vector_layer.isValid():
                    QgsProject.instance().addMapLayer(vector_layer)
                    self.plugin.log.info(f"Loaded sample vector: {vector_path}")
                else:
                    self.plugin.log.warning(f"Invalid vector layer: {vector_path}")

            # Return True if we found at least one file
            return bool(raster_files or vector_files)

        except Exception as e:
            self.plugin.log.error(f"Error loading sample data: {e}")
            return False
