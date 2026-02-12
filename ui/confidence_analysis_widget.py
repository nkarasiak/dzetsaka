"""Confidence map visualization and analysis widget.

Provides interactive analysis of classification confidence maps, including histogram
visualization, statistics, interpretation guidance, and tools for highlighting areas
of low confidence that may require additional attention or validation.

Features:
    - Histogram of confidence values with binning
    - Summary statistics (mean, median, std, percentiles)
    - Low/high confidence area identification
    - Interpretation guidance for users
    - Export confidence statistics to CSV
    - Create vector layer of low-confidence areas

Author:
    Nicolas Karasiak
"""

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    matplotlib.use("Qt5Agg")  # Ensure Qt backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from osgeo import gdal
except ImportError:
    try:
        import gdal  # type: ignore[no-redef]
    except ImportError:
        gdal = None  # type: ignore[assignment]

# Import theme support
try:
    from .theme_support import ThemeAwareWidget, apply_matplotlib_theme
    _THEME_SUPPORT_AVAILABLE = True
except Exception:
    _THEME_SUPPORT_AVAILABLE = False
    # Fallback: create empty mixin class
    class ThemeAwareWidget:
        """Fallback mixin when theme_support is not available."""
        def apply_theme(self):
            pass
    def apply_matplotlib_theme(ax, theme=None):
        """Fallback function."""
        pass


class ConfidenceAnalysisWidget(ThemeAwareWidget, QWidget):
    """Widget for analyzing and visualizing classification confidence maps."""

    def __init__(self, confidence_raster_path: str, parent=None, iface=None):
        """Initialize confidence analysis widget.

        Parameters
        ----------
        confidence_raster_path : str
            Path to confidence map raster (values 0.0-1.0 or 0-100)
        parent : QWidget, optional
            Parent widget
        iface : QgisInterface, optional
            QGIS interface for creating layers
        """
        super().__init__(parent)
        self.confidence_path = confidence_raster_path
        self.iface = iface

        # Apply theme-aware styling
        if _THEME_SUPPORT_AVAILABLE:
            self.apply_theme()

        # Confidence statistics (computed lazily)
        self.stats: Optional[Dict] = None
        self.confidence_data: Optional[np.ndarray] = None

        self._setup_ui()

        # Compute statistics on initialization
        self._compute_statistics()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header_label = QLabel(f"<h3>📊 Confidence Map Analysis</h3>")
        layout.addWidget(header_label)

        file_label = QLabel(f"<b>File:</b> {Path(self.confidence_path).name}")
        file_label.setWordWrap(True)
        layout.addWidget(file_label)

        # Statistics section
        self.stats_group = self._create_statistics_section()
        layout.addWidget(self.stats_group)

        # Histogram section (if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            self.histogram_canvas = self._create_histogram_section()
            layout.addWidget(self.histogram_canvas, 1)  # Stretch to fill
        else:
            no_plot_label = QLabel(
                "<i>Matplotlib not available. Install matplotlib for histogram visualization.</i>"
            )
            no_plot_label.setStyleSheet("color: #999;")
            layout.addWidget(no_plot_label)

        # Interpretation guidance section
        guidance_group = self._create_guidance_section()
        layout.addWidget(guidance_group)

        # Action buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)

    def _create_statistics_section(self) -> QGroupBox:
        """Create statistics display section."""
        group = QGroupBox("Confidence Statistics")
        layout = QVBoxLayout(group)

        self.stats_label = QLabel("<i>Computing statistics...</i>")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 10pt;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        return group

    def _create_histogram_section(self) -> QWidget:
        """Create histogram visualization section using matplotlib."""
        # Create matplotlib figure
        fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = fig.add_subplot(111)

        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(300)

        return canvas

    def _create_guidance_section(self) -> QGroupBox:
        """Create interpretation guidance section."""
        group = QGroupBox("Interpretation Guide")
        layout = QVBoxLayout(group)

        guidance_text = QTextEdit()
        guidance_text.setReadOnly(True)
        guidance_text.setMaximumHeight(200)
        guidance_text.setHtml(
            """
            <h4>What does confidence mean?</h4>
            <p>Confidence represents the model's certainty in its predictions. Higher values indicate more confident classifications.</p>

            <h4>Interpreting confidence values:</h4>
            <ul>
                <li><b>High confidence (>80%):</b> Strong agreement. These pixels are likely correctly classified.</li>
                <li><b>Medium confidence (50-80%):</b> Moderate agreement. Generally reliable but may benefit from validation.</li>
                <li><b>Low confidence (<50%):</b> Weak agreement. High risk of misclassification - review these areas carefully.</li>
            </ul>

            <h4>What to do with low-confidence areas:</h4>
            <ul>
                <li>Collect additional training samples in these regions</li>
                <li>Use higher-resolution imagery or additional spectral bands</li>
                <li>Consider these areas for field validation</li>
                <li>Apply post-processing filters or manual editing</li>
            </ul>

            <h4>Common causes of low confidence:</h4>
            <ul>
                <li>Insufficient or unrepresentative training data</li>
                <li>Class confusion (similar spectral signatures)</li>
                <li>Mixed pixels at class boundaries</li>
                <li>Unique conditions not present in training data</li>
            </ul>
            """
        )
        layout.addWidget(guidance_text)

        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        export_stats_btn = QPushButton("Export Statistics...")
        export_stats_btn.setToolTip("Export confidence statistics to CSV file")
        export_stats_btn.clicked.connect(self._export_statistics)
        layout.addWidget(export_stats_btn)

        if gdal is not None:
            highlight_low_btn = QPushButton("Highlight Low Confidence Areas...")
            highlight_low_btn.setToolTip("Create vector layer showing areas with confidence below threshold")
            highlight_low_btn.clicked.connect(self._highlight_low_confidence)
            layout.addWidget(highlight_low_btn)

        layout.addStretch()

        return layout

    def _compute_statistics(self):
        """Compute confidence statistics from raster."""
        if gdal is None:
            self.stats_label.setText("<b>Error:</b> GDAL not available for reading confidence raster.")
            self.stats_label.setStyleSheet("color: #e74c3c;")
            return

        try:
            # Open raster
            dataset = gdal.Open(self.confidence_path)
            if dataset is None:
                raise RuntimeError(f"Cannot open raster: {self.confidence_path}")

            # Read first band
            band = dataset.GetRasterBand(1)
            if band is None:
                raise RuntimeError("Cannot read raster band")

            # Read data
            data = band.ReadAsArray()
            if data is None:
                raise RuntimeError("Cannot read raster data")

            # Get nodata value
            nodata = band.GetNoDataValue()

            # Mask nodata values
            if nodata is not None:
                valid_mask = data != nodata
            else:
                valid_mask = np.ones_like(data, dtype=bool)

            confidence_values = data[valid_mask].flatten()

            # Check if values are 0-100 or 0.0-1.0
            max_val = np.max(confidence_values)
            if max_val > 1.5:  # Likely 0-100 scale
                confidence_values = confidence_values / 100.0

            # Store for later use
            self.confidence_data = confidence_values

            # Compute statistics
            self.stats = {
                "mean": float(np.mean(confidence_values)),
                "median": float(np.median(confidence_values)),
                "std": float(np.std(confidence_values)),
                "min": float(np.min(confidence_values)),
                "max": float(np.max(confidence_values)),
                "p25": float(np.percentile(confidence_values, 25)),
                "p75": float(np.percentile(confidence_values, 75)),
                "total_pixels": len(confidence_values),
                "low_conf_pct": float(np.sum(confidence_values < 0.5) / len(confidence_values) * 100),
                "medium_conf_pct": float(
                    np.sum((confidence_values >= 0.5) & (confidence_values < 0.8)) / len(confidence_values) * 100
                ),
                "high_conf_pct": float(np.sum(confidence_values >= 0.8) / len(confidence_values) * 100),
            }

            # Update display
            self._update_statistics_display()

            # Update histogram
            if MATPLOTLIB_AVAILABLE:
                self._update_histogram()

        except Exception as e:
            self.stats_label.setText(f"<b>Error computing statistics:</b> {str(e)}")
            self.stats_label.setStyleSheet("color: #e74c3c;")

    def _update_statistics_display(self):
        """Update statistics label with computed values."""
        if not self.stats:
            return

        stats_html = f"""
        <table style='font-family: monospace;'>
        <tr><td><b>Mean confidence:</b></td><td>{self.stats['mean']:.2%}</td></tr>
        <tr><td><b>Median confidence:</b></td><td>{self.stats['median']:.2%}</td></tr>
        <tr><td><b>Std. deviation:</b></td><td>{self.stats['std']:.3f}</td></tr>
        <tr><td><b>Range:</b></td><td>{self.stats['min']:.2%} - {self.stats['max']:.2%}</td></tr>
        <tr><td><b>25th percentile:</b></td><td>{self.stats['p25']:.2%}</td></tr>
        <tr><td><b>75th percentile:</b></td><td>{self.stats['p75']:.2%}</td></tr>
        </table>
        <br>
        <b>Confidence Distribution:</b>
        <table style='font-family: monospace;'>
        <tr><td>🔴 Low (<50%):</td><td><b>{self.stats['low_conf_pct']:.1f}%</b> ({int(self.stats['total_pixels'] * self.stats['low_conf_pct'] / 100):,} pixels)</td></tr>
        <tr><td>🟡 Medium (50-80%):</td><td><b>{self.stats['medium_conf_pct']:.1f}%</b> ({int(self.stats['total_pixels'] * self.stats['medium_conf_pct'] / 100):,} pixels)</td></tr>
        <tr><td>🟢 High (>80%):</td><td><b>{self.stats['high_conf_pct']:.1f}%</b> ({int(self.stats['total_pixels'] * self.stats['high_conf_pct'] / 100):,} pixels)</td></tr>
        </table>
        """

        self.stats_label.setText(stats_html)
        self.stats_label.setStyleSheet("")

    def _update_histogram(self):
        """Update histogram visualization."""
        if not MATPLOTLIB_AVAILABLE or self.confidence_data is None:
            return

        try:
            self.ax.clear()

            # Apply theme to matplotlib
            if _THEME_SUPPORT_AVAILABLE:
                apply_matplotlib_theme(self.ax)

            # Create histogram with color-coded bins
            bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
            counts, edges, patches = self.ax.hist(
                self.confidence_data, bins=bins, edgecolor="black", linewidth=0.5
            )

            # Color code bins: red (<0.5), yellow (0.5-0.8), green (>0.8)
            for i, patch in enumerate(patches):
                bin_center = (edges[i] + edges[i + 1]) / 2
                if bin_center < 0.5:
                    patch.set_facecolor("#e74c3c")  # Red
                elif bin_center < 0.8:
                    patch.set_facecolor("#f39c12")  # Orange/yellow
                else:
                    patch.set_facecolor("#27ae60")  # Green

            self.ax.set_xlabel("Confidence", fontsize=11, fontweight="bold")
            self.ax.set_ylabel("Pixel Count", fontsize=11, fontweight="bold")
            self.ax.set_title("Confidence Distribution", fontsize=12, fontweight="bold")

            # Add gridlines for readability
            self.ax.grid(True, alpha=0.3, linestyle="--")

            # Format x-axis as percentages
            self.ax.set_xlim(0, 1)
            from matplotlib.ticker import FuncFormatter

            self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0%}"))

            # Add reference lines
            self.ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Low/Medium threshold")
            self.ax.axvline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Medium/High threshold")

            self.ax.legend()

            # Redraw canvas
            self.ax.figure.canvas.draw()

        except Exception as e:
            print(f"Error updating histogram: {e}")

    def _export_statistics(self):
        """Export confidence statistics to CSV file."""
        if not self.stats:
            QMessageBox.warning(self, "No Statistics", "No statistics available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Confidence Statistics",
            f"confidence_stats_{Path(self.confidence_path).stem}.csv",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(["Statistic", "Value"])

                # Write statistics
                writer.writerow(["Confidence Raster", self.confidence_path])
                writer.writerow([])
                writer.writerow(["Summary Statistics", ""])
                writer.writerow(["Mean", f"{self.stats['mean']:.4f}"])
                writer.writerow(["Median", f"{self.stats['median']:.4f}"])
                writer.writerow(["Std. Deviation", f"{self.stats['std']:.4f}"])
                writer.writerow(["Minimum", f"{self.stats['min']:.4f}"])
                writer.writerow(["Maximum", f"{self.stats['max']:.4f}"])
                writer.writerow(["25th Percentile", f"{self.stats['p25']:.4f}"])
                writer.writerow(["75th Percentile", f"{self.stats['p75']:.4f}"])
                writer.writerow([])
                writer.writerow(["Distribution", ""])
                writer.writerow(["Total Pixels", f"{self.stats['total_pixels']:,}"])
                writer.writerow(["Low Confidence (<50%)", f"{self.stats['low_conf_pct']:.2f}%"])
                writer.writerow(["Medium Confidence (50-80%)", f"{self.stats['medium_conf_pct']:.2f}%"])
                writer.writerow(["High Confidence (>80%)", f"{self.stats['high_conf_pct']:.2f}%"])

            QMessageBox.information(self, "Export Successful", f"Statistics exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export statistics:\n{str(e)}")

    def _highlight_low_confidence(self):
        """Create vector layer highlighting low confidence areas."""
        # This is a placeholder - full implementation would:
        # 1. Threshold the confidence raster at user-defined level (e.g., <0.5)
        # 2. Vectorize low-confidence regions using gdal.Polygonize()
        # 3. Add resulting vector layer to QGIS map
        # 4. Style with red/orange color for visibility

        QMessageBox.information(
            self,
            "Feature Coming Soon",
            "Highlight Low Confidence Areas feature will be implemented in a future update.\n\n"
            "For now, you can:\n"
            "1. Load the confidence raster in QGIS\n"
            "2. Use Raster > Raster Calculator to threshold (confidence < 0.5)\n"
            "3. Use Raster > Conversion > Polygonize to create vector areas",
        )
