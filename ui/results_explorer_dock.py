"""Interactive results explorer dock for classification results.

This module provides a comprehensive, interactive dock widget that displays
classification results including overview stats, class distribution, accuracy metrics,
and explainability visualizations. The dock auto-opens after classification completes
to provide immediate insights without opening external HTML reports.

Author:
    Nicolas Karasiak
"""

import os
import webbrowser
from typing import Dict, Optional

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QColor, QPixmap
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from osgeo import gdal
except ImportError:
    try:
        import gdal
    except ImportError:
        gdal = None

try:
    from .confidence_analysis_widget import ConfidenceAnalysisWidget
    CONFIDENCE_WIDGET_AVAILABLE = True
except ImportError:
    CONFIDENCE_WIDGET_AVAILABLE = False

try:
    from .training_data_quality_checker import TrainingDataQualityChecker
    QUALITY_CHECKER_AVAILABLE = True
except ImportError:
    QUALITY_CHECKER_AVAILABLE = False

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


class ResultsExplorerDock(ThemeAwareWidget, QDockWidget):
    """Interactive dock widget for exploring classification results.

    Displays comprehensive information about a completed classification including:
    - Overview: algorithm, runtime, paths, timestamp
    - Distribution: histogram and pie chart of class pixel counts
    - Accuracy: confusion matrix and metrics (if available)
    - Explainability: SHAP visualizations (if available)

    Parameters
    ----------
    classification_result : dict
        Dictionary containing classification results with keys:
        - 'algorithm': str - algorithm name
        - 'runtime_seconds': float - execution time
        - 'output_path': str - path to output raster
        - 'input_path': str - path to input raster (optional)
        - 'matrix_path': str - path to confusion matrix CSV (optional)
        - 'shap_path': str - path to SHAP visualization (optional)
        - 'class_counts': dict - {class_id: pixel_count} (optional, computed if missing)
        - 'timestamp': str - completion timestamp (optional)
    parent : QWidget, optional
        Parent widget
    iface : QgisInterface, optional
        QGIS interface for layer operations

    """

    def __init__(self, classification_result: Dict, parent=None, iface=None):
        """Initialize the results explorer dock."""
        super(ResultsExplorerDock, self).__init__(parent)
        self.result = classification_result
        self.iface = iface

        # Apply theme-aware styling
        if _THEME_SUPPORT_AVAILABLE:
            self.apply_theme()

        # Set dock properties
        self.setWindowTitle("Classification Results")
        self.setObjectName("DzetsakaResultsExplorerDock")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        # Main container
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Create tabs
        self.tab_widget = QTabWidget()
        self._create_overview_tab()
        self._create_distribution_tab()

        # Conditionally add tabs based on available data
        if self.result.get("matrix_path") and os.path.exists(self.result.get("matrix_path", "")):
            self._create_accuracy_tab()

        if self.result.get("shap_path") and os.path.exists(self.result.get("shap_path", "")):
            self._create_explainability_tab()

        if self.result.get("confidence_path") and os.path.exists(self.result.get("confidence_path", "")):
            self._create_confidence_tab()

        self.main_layout.addWidget(self.tab_widget)

        # Quick actions section
        self._create_quick_actions()

        # Set main widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.main_widget)
        self.setWidget(scroll_area)

    def _create_overview_tab(self):
        """Create the overview tab with key-value information."""
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        overview_layout.setContentsMargins(10, 10, 10, 10)

        # Key information group
        info_group = QGroupBox("Classification Information")
        info_layout = QFormLayout(info_group)

        # Algorithm
        algorithm_name = self.result.get("algorithm", "Unknown")
        info_layout.addRow("Algorithm:", QLabel(f"<b>{algorithm_name}</b>"))

        # Runtime
        runtime = self.result.get("runtime_seconds", 0)
        if runtime > 60:
            runtime_str = f"{runtime / 60:.1f} minutes ({runtime:.1f}s)"
        else:
            runtime_str = f"{runtime:.1f} seconds"
        info_layout.addRow("Runtime:", QLabel(runtime_str))

        # Number of classes
        class_counts = self.result.get("class_counts", {})
        if not class_counts and self.result.get("output_path"):
            class_counts = self._compute_class_counts(self.result.get("output_path"))
            self.result["class_counts"] = class_counts
        num_classes = len(class_counts)
        info_layout.addRow("Number of Classes:", QLabel(str(num_classes)))

        # Total pixels classified
        total_pixels = sum(class_counts.values()) if class_counts else 0
        info_layout.addRow("Total Pixels:", QLabel(f"{total_pixels:,}"))

        # Timestamp
        timestamp = self.result.get("timestamp", "Not recorded")
        info_layout.addRow("Completed:", QLabel(timestamp))

        overview_layout.addWidget(info_group)

        # Paths group
        paths_group = QGroupBox("File Paths")
        paths_layout = QFormLayout(paths_group)

        # Input raster
        input_path = self.result.get("input_path", "Not specified")
        input_label = QLabel(input_path)
        input_label.setWordWrap(True)
        input_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        paths_layout.addRow("Input Raster:", input_label)

        # Output raster
        output_path = self.result.get("output_path", "Not specified")
        output_label = QLabel(output_path)
        output_label.setWordWrap(True)
        output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        paths_layout.addRow("Output Raster:", output_label)

        # Model path (if available)
        model_path = self.result.get("model_path", "")
        if model_path:
            model_label = QLabel(model_path)
            model_label.setWordWrap(True)
            model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            paths_layout.addRow("Model File:", model_label)

        overview_layout.addWidget(paths_group)

        # Add spacer
        overview_layout.addStretch()

        self.tab_widget.addTab(overview_widget, "Overview")

    def _create_distribution_tab(self):
        """Create the distribution tab with histogram and pie chart."""
        distribution_widget = QWidget()
        distribution_layout = QVBoxLayout(distribution_widget)
        distribution_layout.setContentsMargins(10, 10, 10, 10)

        class_counts = self.result.get("class_counts", {})
        if not class_counts:
            # Show message if no data available
            no_data_label = QLabel("No class distribution data available.")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            distribution_layout.addWidget(no_data_label)
            self.tab_widget.addTab(distribution_widget, "Distribution")
            return

        if not MATPLOTLIB_AVAILABLE:
            # Show message if matplotlib not available
            warning_label = QLabel(
                "Matplotlib is not available. Install matplotlib to view distribution charts:\n"
                "pip install matplotlib"
            )
            warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warning_label.setStyleSheet("color: orange;")
            distribution_layout.addWidget(warning_label)
            self.tab_widget.addTab(distribution_widget, "Distribution")
            return

        # Set matplotlib backend
        matplotlib.use("Qt5Agg")

        # Create figure with two subplots
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)

        # Sort classes by ID
        sorted_classes = sorted(class_counts.items())
        class_ids = [str(cls_id) for cls_id, _ in sorted_classes]
        pixel_counts = [count for _, count in sorted_classes]

        # Histogram (top)
        ax1 = fig.add_subplot(2, 1, 1)
        colors = plt.cm.tab10.colors[: len(class_ids)]
        bars = ax1.bar(class_ids, pixel_counts, color=colors, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Class ID")
        ax1.set_ylabel("Pixel Count")
        ax1.set_title("Class Distribution - Histogram")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Pie chart (bottom)
        ax2 = fig.add_subplot(2, 1, 2)
        total = sum(pixel_counts)
        percentages = [(count / total) * 100 for count in pixel_counts]

        # Only show labels for slices > 2%
        labels = [
            f"Class {cls_id}\n({pct:.1f}%)" if pct > 2 else ""
            for cls_id, pct in zip(class_ids, percentages)
        ]

        ax2.pie(
            pixel_counts,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
            startangle=90,
        )
        ax2.set_title("Class Distribution - Pie Chart")

        fig.tight_layout()

        distribution_layout.addWidget(canvas)
        self.tab_widget.addTab(distribution_widget, "Distribution")

    def _create_accuracy_tab(self):
        """Create the accuracy tab with confusion matrix and metrics."""
        accuracy_widget = QWidget()
        accuracy_layout = QVBoxLayout(accuracy_widget)
        accuracy_layout.setContentsMargins(10, 10, 10, 10)

        matrix_path = self.result.get("matrix_path", "")
        if not matrix_path or not os.path.exists(matrix_path):
            no_data_label = QLabel("Confusion matrix file not found.")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            accuracy_layout.addWidget(no_data_label)
            self.tab_widget.addTab(accuracy_widget, "Accuracy")
            return

        # Parse confusion matrix CSV
        try:
            matrix_data, metrics = self._parse_confusion_matrix(matrix_path)

            # Metrics group (at top)
            if metrics:
                metrics_group = QGroupBox("Overall Metrics")
                metrics_layout = QFormLayout(metrics_group)

                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        value_str = f"{metric_value:.4f}"
                    else:
                        value_str = str(metric_value)
                    metrics_layout.addRow(f"{metric_name}:", QLabel(f"<b>{value_str}</b>"))

                accuracy_layout.addWidget(metrics_group)

            # Confusion matrix table
            if matrix_data:
                matrix_group = QGroupBox("Confusion Matrix")
                matrix_layout = QVBoxLayout(matrix_group)

                table = QTableWidget()
                num_classes = len(matrix_data)
                table.setRowCount(num_classes)
                table.setColumnCount(num_classes)

                # Set headers
                class_labels = list(matrix_data.keys())
                table.setHorizontalHeaderLabels([f"Pred {cls}" for cls in class_labels])
                table.setVerticalHeaderLabels([f"True {cls}" for cls in class_labels])

                # Fill table with color-coded values
                max_value = max(
                    max(row_data.values()) for row_data in matrix_data.values() if row_data
                )

                for i, true_class in enumerate(class_labels):
                    for j, pred_class in enumerate(class_labels):
                        value = matrix_data[true_class].get(pred_class, 0)
                        item = QTableWidgetItem(str(value))
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                        # Color code: diagonal (correct) = green, off-diagonal (errors) = red
                        if i == j:
                            # Correct predictions - green gradient
                            intensity = int(200 * (value / max_value)) if max_value > 0 else 0
                            color = QColor(255 - intensity, 255, 255 - intensity)
                        else:
                            # Errors - red gradient
                            intensity = int(200 * (value / max_value)) if max_value > 0 else 0
                            color = QColor(255, 255 - intensity, 255 - intensity)

                        item.setBackground(color)
                        table.setItem(i, j, item)

                # Adjust table size
                table.resizeColumnsToContents()
                table.resizeRowsToContents()

                matrix_layout.addWidget(table)
                accuracy_layout.addWidget(matrix_group)

        except Exception as e:
            error_label = QLabel(f"Error parsing confusion matrix: {e}")
            error_label.setStyleSheet("color: red;")
            accuracy_layout.addWidget(error_label)

        accuracy_layout.addStretch()
        self.tab_widget.addTab(accuracy_widget, "Accuracy")

    def _create_explainability_tab(self):
        """Create the explainability tab with SHAP visualizations."""
        explainability_widget = QWidget()
        explainability_layout = QVBoxLayout(explainability_widget)
        explainability_layout.setContentsMargins(10, 10, 10, 10)

        shap_path = self.result.get("shap_path", "")
        if not shap_path or not os.path.exists(shap_path):
            no_data_label = QLabel("SHAP visualization not found.")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            explainability_layout.addWidget(no_data_label)
            self.tab_widget.addTab(explainability_widget, "Explainability")
            return

        # Info label
        info_label = QLabel(
            "SHAP (SHapley Additive exPlanations) values show which features (bands) "
            "are most important for the model's predictions."
        )
        info_label.setWordWrap(True)
        explainability_layout.addWidget(info_label)

        # Display SHAP image
        try:
            pixmap = QPixmap(shap_path)
            if not pixmap.isNull():
                image_label = QLabel()
                # Scale to fit width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                explainability_layout.addWidget(image_label)
            else:
                error_label = QLabel("Could not load SHAP visualization image.")
                error_label.setStyleSheet("color: red;")
                explainability_layout.addWidget(error_label)
        except Exception as e:
            error_label = QLabel(f"Error loading SHAP visualization: {e}")
            error_label.setStyleSheet("color: red;")
            explainability_layout.addWidget(error_label)

        explainability_layout.addStretch()
        self.tab_widget.addTab(explainability_widget, "Explainability")

    def _create_confidence_tab(self):
        """Create the confidence analysis tab with confidence map visualization."""
        if not CONFIDENCE_WIDGET_AVAILABLE:
            confidence_widget = QWidget()
            confidence_layout = QVBoxLayout(confidence_widget)
            confidence_layout.setContentsMargins(10, 10, 10, 10)
            no_module_label = QLabel("Confidence analysis widget not available (module import failed).")
            no_module_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            confidence_layout.addWidget(no_module_label)
            self.tab_widget.addTab(confidence_widget, "Confidence")
            return

        confidence_path = self.result.get("confidence_path", "")
        if not confidence_path or not os.path.exists(confidence_path):
            confidence_widget = QWidget()
            confidence_layout = QVBoxLayout(confidence_widget)
            confidence_layout.setContentsMargins(10, 10, 10, 10)
            no_data_label = QLabel("Confidence map file not found.")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            confidence_layout.addWidget(no_data_label)
            self.tab_widget.addTab(confidence_widget, "Confidence")
            return

        # Create confidence analysis widget
        try:
            confidence_widget = ConfidenceAnalysisWidget(
                confidence_raster_path=confidence_path,
                parent=self,
                iface=self.iface
            )
            self.tab_widget.addTab(confidence_widget, "Confidence")
        except Exception as e:
            # Fallback if widget creation fails
            confidence_widget = QWidget()
            confidence_layout = QVBoxLayout(confidence_widget)
            confidence_layout.setContentsMargins(10, 10, 10, 10)
            error_label = QLabel(f"Error loading confidence analysis: {e}")
            error_label.setStyleSheet("color: red;")
            error_label.setWordWrap(True)
            confidence_layout.addWidget(error_label)
            self.tab_widget.addTab(confidence_widget, "Confidence")

    def _create_quick_actions(self):
        """Create quick actions section with useful buttons."""
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)

        # Apply Color Scheme button
        color_btn = QPushButton("Apply Color Scheme")
        color_btn.setToolTip("Apply a predefined color ramp to the classification result layer")
        color_btn.clicked.connect(self._apply_color_scheme)
        actions_layout.addWidget(color_btn)

        # Export to GeoPackage button
        export_btn = QPushButton("Export to GeoPackage")
        export_btn.setToolTip("Save the classification result as a GeoPackage")
        export_btn.clicked.connect(self._export_to_geopackage)
        actions_layout.addWidget(export_btn)

        # Check Training Data Quality button (if training data available)
        if QUALITY_CHECKER_AVAILABLE and self.result.get("training_vector") and self.result.get("class_field"):
            quality_btn = QPushButton("Check Training Data")
            quality_btn.setToolTip(
                "Analyze training data quality for iterative improvement<br>"
                "Check for class imbalance, insufficient samples, and spatial issues"
            )
            quality_btn.clicked.connect(self._check_training_data)
            actions_layout.addWidget(quality_btn)

        # Open Full Report button (if available)
        report_path = self.result.get("report_path", "")
        if report_path and os.path.exists(report_path):
            report_btn = QPushButton("Open Full Report")
            report_btn.setToolTip("Open the complete HTML report in your web browser")
            report_btn.clicked.connect(self._open_full_report)
            actions_layout.addWidget(report_btn)

        self.main_layout.addWidget(actions_group)

    def _apply_color_scheme(self):
        """Apply a predefined color scheme to the result layer."""
        if not self.iface:
            QMessageBox.warning(
                self,
                "Not Available",
                "Color scheme application requires QGIS interface.",
            )
            return

        output_path = self.result.get("output_path", "")
        if not output_path:
            QMessageBox.warning(
                self,
                "No Output Path",
                "Cannot apply color scheme: output path not specified.",
            )
            return

        # Find the layer in QGIS
        from qgis.core import QgsProject

        layers = QgsProject.instance().mapLayersByName(os.path.basename(output_path))
        if not layers:
            # Try to find by path
            for layer in QgsProject.instance().mapLayers().values():
                if hasattr(layer, "source") and layer.source() == output_path:
                    layers = [layer]
                    break

        if not layers:
            QMessageBox.information(
                self,
                "Layer Not Found",
                "The classification result layer is not loaded in QGIS. "
                "Please load the layer first.",
            )
            return

        try:
            from qgis.core import (
                QgsColorRampShader,
                QgsRasterShader,
                QgsSingleBandPseudoColorRenderer,
            )

            layer = layers[0]
            class_counts = self.result.get("class_counts", {})
            if not class_counts:
                QMessageBox.information(
                    self,
                    "No Class Data",
                    "Cannot apply color scheme: class information not available.",
                )
                return

            # Create color ramp
            shader = QgsRasterShader()
            color_ramp = QgsColorRampShader()
            color_ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)

            # Generate color ramp items
            sorted_classes = sorted(class_counts.keys())
            color_list = []

            # Use tab10 colormap colors
            if MATPLOTLIB_AVAILABLE:
                colors = plt.cm.tab10.colors
                for i, class_id in enumerate(sorted_classes):
                    color_idx = i % len(colors)
                    r, g, b = [int(c * 255) for c in colors[color_idx][:3]]
                    color_list.append(
                        QgsColorRampShader.ColorRampItem(
                            class_id,
                            QColor(r, g, b),
                            f"Class {class_id}",
                        )
                    )
            else:
                # Fallback to simple color scheme
                for class_id in sorted_classes:
                    # Simple hue rotation
                    hue = int((class_id * 360 / len(sorted_classes)) % 360)
                    color_list.append(
                        QgsColorRampShader.ColorRampItem(
                            class_id,
                            QColor.fromHsv(hue, 200, 200),
                            f"Class {class_id}",
                        )
                    )

            color_ramp.setColorRampItemList(color_list)
            shader.setRasterShaderFunction(color_ramp)

            # Apply renderer
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()

            QMessageBox.information(
                self,
                "Success",
                "Color scheme applied successfully!",
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to apply color scheme: {e}",
            )

    def _export_to_geopackage(self):
        """Export the classification result to GeoPackage format."""
        output_path = self.result.get("output_path", "")
        if not output_path or not os.path.exists(output_path):
            QMessageBox.warning(
                self,
                "No Output File",
                "Cannot export: output raster not found.",
            )
            return

        try:
            from qgis.PyQt.QtWidgets import QFileDialog

            # Prompt for output path
            gpkg_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export to GeoPackage",
                os.path.splitext(output_path)[0] + ".gpkg",
                "GeoPackage (*.gpkg)",
            )

            if not gpkg_path:
                return

            # Use GDAL to convert
            if gdal is None:
                QMessageBox.warning(
                    self,
                    "GDAL Not Available",
                    "GDAL is required to export to GeoPackage.",
                )
                return

            translate_options = gdal.TranslateOptions(format="GPKG")
            gdal.Translate(gpkg_path, output_path, options=translate_options)

            QMessageBox.information(
                self,
                "Success",
                f"Classification result exported to:\n{gpkg_path}",
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Export Failed",
                f"Failed to export to GeoPackage: {e}",
            )

    def _open_full_report(self):
        """Open the full HTML report in the default web browser."""
        report_path = self.result.get("report_path", "")
        if not report_path or not os.path.exists(report_path):
            QMessageBox.warning(
                self,
                "Report Not Found",
                "Full HTML report not found.",
            )
            return

        try:
            webbrowser.open(f"file://{report_path}")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open report: {e}",
            )

    def _check_training_data(self):
        """Open training data quality checker for iterative improvement."""
        training_vector = self.result.get("training_vector", "")
        class_field = self.result.get("class_field", "")

        if not training_vector or not os.path.exists(training_vector):
            QMessageBox.warning(
                self,
                "Training Data Not Found",
                "Training vector path not available or file not found.",
            )
            return

        if not class_field:
            QMessageBox.warning(
                self,
                "Missing Class Field",
                "Class field information not available.",
            )
            return

        # Open quality checker dialog
        dialog = TrainingDataQualityChecker(
            vector_path=training_vector,
            class_field=class_field,
            parent=self
        )
        try:
            dialog.exec_()
        except AttributeError:
            dialog.exec()

    def _compute_class_counts(self, raster_path: str) -> Dict[int, int]:
        """Compute class pixel counts from a raster file.

        Parameters
        ----------
        raster_path : str
            Path to the classification raster

        Returns
        -------
        dict
            Dictionary mapping class IDs to pixel counts

        """
        if not gdal:
            return {}

        try:
            ds = gdal.Open(raster_path)
            if not ds:
                return {}

            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()

            if array is None:
                return {}

            # Count unique values
            import numpy as np

            unique, counts = np.unique(array, return_counts=True)

            # Filter out nodata values (typically negative or very large)
            nodata = band.GetNoDataValue()
            class_counts = {}
            for value, count in zip(unique, counts):
                if nodata is not None and value == nodata:
                    continue
                if value < 0 or value > 10000:  # Skip unreasonable values
                    continue
                class_counts[int(value)] = int(count)

            return class_counts

        except Exception:
            return {}

    def _parse_confusion_matrix(self, csv_path: str):
        """Parse confusion matrix CSV file.

        Parameters
        ----------
        csv_path : str
            Path to confusion matrix CSV file

        Returns
        -------
        tuple
            (matrix_data, metrics) where matrix_data is a nested dict
            {true_class: {pred_class: count}} and metrics is a dict
            of overall metrics

        """
        import csv

        matrix_data = {}
        metrics = {}

        try:
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

                if not rows:
                    return matrix_data, metrics

                # Try to find overall accuracy and kappa
                for row in rows:
                    if not row:
                        continue
                    first_cell = row[0].lower().strip()

                    if "overall accuracy" in first_cell or "oa" == first_cell:
                        if len(row) > 1:
                            try:
                                metrics["Overall Accuracy"] = float(row[1])
                            except (ValueError, IndexError):
                                pass

                    elif "kappa" in first_cell:
                        if len(row) > 1:
                            try:
                                metrics["Kappa Coefficient"] = float(row[1])
                            except (ValueError, IndexError):
                                pass

                # Find the confusion matrix section
                # Look for row with class headers
                matrix_start_idx = -1
                for i, row in enumerate(rows):
                    if len(row) > 2 and any(
                        str(cell).strip().replace(".", "").isdigit() for cell in row[1:]
                    ):
                        matrix_start_idx = i
                        break

                if matrix_start_idx >= 0:
                    header_row = rows[matrix_start_idx]
                    class_labels = [str(cell).strip() for cell in header_row[1:] if cell]

                    # Parse matrix rows
                    for row in rows[matrix_start_idx + 1 :]:
                        if not row or len(row) < 2:
                            continue

                        true_class = str(row[0]).strip()
                        if not true_class or not any(c.isdigit() for c in true_class):
                            continue

                        matrix_data[true_class] = {}
                        for j, value in enumerate(row[1 : len(class_labels) + 1]):
                            if j < len(class_labels):
                                try:
                                    matrix_data[true_class][class_labels[j]] = int(float(value))
                                except (ValueError, IndexError):
                                    matrix_data[true_class][class_labels[j]] = 0

        except Exception as exc:
            _ = exc

        return matrix_data, metrics


def open_results_explorer(result_data: Dict, parent=None, iface=None) -> Optional[ResultsExplorerDock]:
    """Open the results explorer dock with classification results.

    This function checks the QSettings to see if auto-open is enabled,
    creates the results explorer dock, and displays it.

    Parameters
    ----------
    result_data : dict
        Classification result data dictionary
    parent : QWidget, optional
        Parent widget
    iface : QgisInterface, optional
        QGIS interface for layer operations

    Returns
    -------
    ResultsExplorerDock or None
        The created dock widget, or None if auto-open is disabled

    """
    # Check if auto-open is enabled
    settings = QSettings()
    auto_open = settings.value("/dzetsaka/autoOpenResultsExplorer", True, type=bool)

    if not auto_open:
        return None

    # Create and show the dock
    dock = ResultsExplorerDock(result_data, parent=parent, iface=iface)
    dock.show()

    return dock
