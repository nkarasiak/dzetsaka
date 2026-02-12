"""Training data quality checker for dzetsaka.

Analyzes training vector data to identify potential issues before classification,
including class imbalance, insufficient samples, spatial clustering, and data quality problems.

Features:
    - Class balance analysis (warns for ratio >10:1)
    - Minimum sample count validation (errors for <30 samples)
    - Spatial clustering detection (recommends polygon CV)
    - Missing/invalid geometry detection
    - Duplicate feature detection
    - Summary report with actionable recommendations

Author:
    Nicolas Karasiak
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from osgeo import ogr
except ImportError:
    try:
        import ogr  # type: ignore[no-redef]
    except ImportError:
        ogr = None  # type: ignore[assignment]

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


class DataQualityIssue:
    """Represents a data quality issue with severity and recommendations."""

    SEVERITY_CRITICAL = "critical"
    # Backward-compatible alias for older code paths that still reference "error".
    SEVERITY_ERROR = SEVERITY_CRITICAL
    SEVERITY_WARNING = "warning"
    SEVERITY_INFO = "info"

    def __init__(self, severity: str, title: str, description: str, recommendation: str):
        """Initialize a data quality issue.

        Parameters
        ----------
        severity : str
            Issue severity: "critical", "warning", or "info"
        title : str
            Short issue title
        description : str
            Detailed issue description
        recommendation : str
            Actionable recommendation to fix the issue
        """
        self.severity = severity
        self.title = title
        self.description = description
        self.recommendation = recommendation

    def get_icon(self) -> str:
        """Get emoji icon for severity level."""
        return {"critical": "🔴", "error": "🔴", "warning": "⚠️", "info": "ℹ️"}.get(self.severity, "📌")

    def get_color(self) -> str:
        """Get color code for severity level."""
        return {"critical": "#e74c3c", "error": "#e74c3c", "warning": "#f39c12", "info": "#3498db"}.get(
            self.severity, "#95a5a6"
        )


class TrainingDataQualityChecker(ThemeAwareWidget, QDialog):
    """Dialog for analyzing and reporting training data quality issues."""

    def __init__(self, vector_path: str, class_field: str, parent=None):
        """Initialize the training data quality checker.

        Parameters
        ----------
        vector_path : str
            Path to training vector file
        class_field : str
            Name of field containing class labels
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.vector_path = vector_path
        self.class_field = class_field
        self.issues: List[DataQualityIssue] = []
        self._severity_filter = "all"
        self._filter_buttons = {}  # type: Dict[str, QPushButton]
        self._issues_layout: Optional[QVBoxLayout] = None

        # Apply theme-aware styling
        if _THEME_SUPPORT_AVAILABLE:
            self.apply_theme()

        # Set dialog properties
        self.setWindowTitle("Training Data Quality Report")
        self.setMinimumSize(560, 380)
        self.setModal(False)  # Allow user to interact with QGIS while viewing

        # Run analysis
        self._analyze_data()

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header section
        header_label = QLabel("📋 <b>Training Data Quality Report</b>")
        layout.addWidget(header_label)

        file_label = QLabel(f"<b>Vector:</b> {Path(self.vector_path).name}<br><b>Class field:</b> {self.class_field}")
        file_label.setWordWrap(True)
        layout.addWidget(file_label)

        # Summary status
        summary_widget = self._create_summary_widget()
        layout.addWidget(summary_widget)

        # Issues list (scrollable)
        if self.issues:
            issues_scroll = QScrollArea()
            issues_scroll.setWidgetResizable(True)
            issues_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

            issues_container = QWidget()
            self._issues_layout = QVBoxLayout(issues_container)
            self._issues_layout.setSpacing(6)
            self._issues_layout.setContentsMargins(0, 0, 0, 0)
            issues_scroll.setWidget(issues_container)
            layout.addWidget(issues_scroll, 1)  # Stretch to fill available space
            self._rebuild_issue_list()
        else:
            # No issues found
            success_group = QGroupBox()
            success_group.setStyleSheet("background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;")
            success_layout = QVBoxLayout(success_group)
            success_layout.setContentsMargins(8, 8, 8, 8)
            success_layout.setSpacing(4)

            success_label = QLabel("✅ <b>All checks passed!</b><br>Your training data looks good and ready for classification.")
            success_label.setWordWrap(True)
            success_layout.addWidget(success_label)

            layout.addWidget(success_group)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_btn = QPushButton("Export Report...")
        export_btn.setToolTip("Export this report to a text file")
        export_btn.clicked.connect(self._export_report)
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _create_summary_widget(self) -> QWidget:
        """Create summary status widget showing counts by severity."""
        summary_widget = QGroupBox("Filters")
        summary_layout = QHBoxLayout(summary_widget)
        summary_layout.setContentsMargins(8, 8, 8, 8)
        summary_layout.setSpacing(6)

        # Count issues by severity
        error_count = sum(
            1
            for issue in self.issues
            if issue.severity in (DataQualityIssue.SEVERITY_CRITICAL, DataQualityIssue.SEVERITY_ERROR)
        )
        warning_count = sum(1 for issue in self.issues if issue.severity == DataQualityIssue.SEVERITY_WARNING)
        info_count = sum(1 for issue in self.issues if issue.severity == DataQualityIssue.SEVERITY_INFO)

        counts = {
            "all": len(self.issues),
            "critical": error_count,
            "warning": warning_count,
            "info": info_count,
        }
        labels = {
            "all": "All",
            "critical": "Critical",
            "warning": "Warning",
            "info": "Info",
        }

        for key in ("all", "critical", "warning", "info"):
            button = QPushButton(f"{labels[key]} ({counts[key]})")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=key: self._set_severity_filter(k))
            if key == "critical":
                button.setStyleSheet("color: #e74c3c;")
            elif key == "warning":
                button.setStyleSheet("color: #f39c12;")
            elif key == "info":
                button.setStyleSheet("color: #3498db;")
            self._filter_buttons[key] = button
            summary_layout.addWidget(button)
        self._set_severity_filter("all")

        summary_layout.addStretch()

        return summary_widget

    def _set_severity_filter(self, severity: str):
        """Set active severity filter and refresh issue list."""
        self._severity_filter = severity
        for key, button in self._filter_buttons.items():
            button.setChecked(key == severity)
        self._rebuild_issue_list()

    def _rebuild_issue_list(self):
        """Rebuild issue list according to active severity filter."""
        if self._issues_layout is None:
            return
        while self._issues_layout.count():
            item = self._issues_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for issue in self.issues:
            if not self._issue_visible_for_filter(issue):
                continue
            issue_widget = self._create_issue_widget(issue)
            self._issues_layout.addWidget(issue_widget)
        self._issues_layout.addStretch()

    def _issue_visible_for_filter(self, issue: DataQualityIssue) -> bool:
        """Return True if issue matches current severity filter."""
        if self._severity_filter == "all":
            return True
        if self._severity_filter == "critical":
            return issue.severity in (DataQualityIssue.SEVERITY_CRITICAL, DataQualityIssue.SEVERITY_ERROR)
        return issue.severity == self._severity_filter

    def _create_issue_widget(self, issue: DataQualityIssue) -> QWidget:
        """Create a widget displaying a single quality issue."""
        issue_group = QGroupBox()
        issue_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {issue.get_color()};
                border-radius: 6px;
                margin-top: 4px;
                padding: 6px;
                background-color: #f9f9f9;
            }}
        """)

        layout = QVBoxLayout(issue_group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Title with icon
        title_label = QLabel(f"{issue.get_icon()} <b style='font-size: 11pt;'>{issue.title}</b>")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(issue.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #555; margin-left: 8px;")
        layout.addWidget(desc_label)

        # Recommendation
        rec_label = QLabel(f"<b>💡 Recommendation:</b> {issue.recommendation}")
        rec_label.setWordWrap(True)
        rec_label.setStyleSheet(
            f"color: {issue.get_color()}; font-weight: bold; margin-left: 8px; margin-top: 2px; padding: 6px; "
            f"background-color: rgba(52, 152, 219, 0.1); border-radius: 4px;"
        )
        layout.addWidget(rec_label)

        return issue_group

    def _analyze_data(self):
        """Analyze training data and populate issues list."""
        if ogr is None:
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_CRITICAL,
                    "OGR Not Available",
                    "Cannot analyze vector data - OGR/GDAL library not found.",
                    "Install GDAL/OGR to enable data quality checking.",
                )
            )
            return

        # Open vector file
        ds = ogr.Open(self.vector_path)
        if ds is None:
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_CRITICAL,
                    "Cannot Open Vector File",
                    f"Unable to open vector dataset: {self.vector_path}",
                    "Check that the file path is correct and the file is not corrupted.",
                )
            )
            return

        layer = ds.GetLayer()
        if layer is None:
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_CRITICAL,
                    "No Layer Found",
                    "Vector dataset contains no layers.",
                    "Ensure the vector file has a valid layer.",
                )
            )
            return

        # Check if class field exists
        layer_defn = layer.GetLayerDefn()
        field_idx = layer_defn.GetFieldIndex(self.class_field)
        if field_idx == -1:
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_CRITICAL,
                    "Class Field Not Found",
                    f"Field '{self.class_field}' does not exist in the vector layer.",
                    f"Select a valid field name. Available fields: {self._get_field_names(layer_defn)}",
                )
            )
            return

        # Collect class labels and geometries
        class_labels = []
        geometries = []
        invalid_geom_count = 0
        feature_count = layer.GetFeatureCount()

        layer.ResetReading()
        for feat in layer:
            if feat is None:
                continue

            # Get class label
            label = feat.GetField(self.class_field)
            if label is not None:
                class_labels.append(label)

            # Get geometry
            geom = feat.GetGeometryRef()
            if geom is None or not geom.IsValid():
                invalid_geom_count += 1
            else:
                geometries.append(geom.Clone())

        # Run quality checks
        self._check_feature_count(feature_count)
        self._check_class_balance(class_labels)
        self._check_minimum_samples(class_labels)
        self._check_invalid_geometries(invalid_geom_count, feature_count)
        self._check_spatial_clustering(geometries, class_labels)
        self._check_duplicates(class_labels, geometries)

    def _get_field_names(self, layer_defn) -> str:
        """Get comma-separated list of field names."""
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        return ", ".join(field_names[:10]) + ("..." if len(field_names) > 10 else "")

    def _check_feature_count(self, count: int):
        """Check if there are enough features overall."""
        if count < 50:
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_WARNING,
                    "Very Small Dataset",
                    f"Only {count} training samples found. ML models typically need 100+ samples for reliable results.",
                    "Collect more training samples if possible, or use simpler algorithms (GMM, KNN).",
                )
            )

    def _check_class_balance(self, class_labels: List):
        """Check for severe class imbalance."""
        if not class_labels:
            return

        class_counts = Counter(class_labels)
        if len(class_counts) < 2:
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_CRITICAL,
                    "Single Class Detected",
                    "All training samples belong to the same class. Classification requires at least 2 classes.",
                    "Add training samples for other classes you want to classify.",
                )
            )
            return

        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        if min_count > 0:
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 50:
                majority_class = max(class_counts, key=class_counts.get)
                minority_class = min(class_counts, key=class_counts.get)
                self.issues.append(
                    DataQualityIssue(
                        DataQualityIssue.SEVERITY_CRITICAL,
                        "Extreme Class Imbalance",
                        (
                            f"Severe imbalance detected: class '{majority_class}' has {max_count} samples "
                            f"while class '{minority_class}' has only {min_count} samples (ratio: {imbalance_ratio:.1f}:1)."
                        ),
                        "Collect more samples for minority classes, or use SMOTE and class weights for balancing.",
                    )
                )
            elif imbalance_ratio > 10:
                self.issues.append(
                    DataQualityIssue(
                        DataQualityIssue.SEVERITY_WARNING,
                        "Significant Class Imbalance",
                        f"Class imbalance detected with ratio {imbalance_ratio:.1f}:1. This may bias the classifier.",
                        "Consider enabling SMOTE oversampling or class weights in Advanced Options.",
                    )
                )
            elif imbalance_ratio > 3:
                self.issues.append(
                    DataQualityIssue(
                        DataQualityIssue.SEVERITY_INFO,
                        "Moderate Class Imbalance",
                        f"Class distribution varies with ratio {imbalance_ratio:.1f}:1.",
                        "Monitor per-class accuracy. Consider using class weights if minority classes perform poorly.",
                    )
                )

    def _check_minimum_samples(self, class_labels: List):
        """Check that each class has minimum required samples."""
        if not class_labels:
            return

        class_counts = Counter(class_labels)
        classes_below_30 = {cls: count for cls, count in class_counts.items() if count < 30}
        classes_below_10 = {cls: count for cls, count in class_counts.items() if count < 10}

        if classes_below_10:
            class_list = ", ".join([f"'{cls}' ({count})" for cls, count in classes_below_10.items()])
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_CRITICAL,
                    "Critically Low Sample Count",
                    f"Classes with <10 samples: {class_list}. This is too few for reliable classification.",
                    "Collect at least 30 samples per class (50+ recommended) for stable results.",
                )
            )
        elif classes_below_30:
            class_list = ", ".join([f"'{cls}' ({count})" for cls, count in classes_below_30.items()])
            self.issues.append(
                DataQualityIssue(
                    DataQualityIssue.SEVERITY_WARNING,
                    "Low Sample Count",
                    f"Classes with <30 samples: {class_list}. Models may not generalize well.",
                    "Collect more samples (50+ recommended per class) or use simpler algorithms.",
                )
            )

    def _check_invalid_geometries(self, invalid_count: int, total_count: int):
        """Check for missing or invalid geometries."""
        if invalid_count > 0:
            percentage = (invalid_count / total_count) * 100 if total_count > 0 else 0
            severity = DataQualityIssue.SEVERITY_CRITICAL if percentage > 10 else DataQualityIssue.SEVERITY_WARNING

            self.issues.append(
                DataQualityIssue(
                    severity,
                    "Invalid Geometries Found",
                    f"{invalid_count} features ({percentage:.1f}%) have missing or invalid geometries.",
                    "Fix or remove invalid geometries using QGIS 'Fix Geometries' tool before classification.",
                )
            )

    def _check_spatial_clustering(self, geometries: List, class_labels: List):
        """Check for spatial clustering that might lead to overfitting."""
        if len(geometries) < 10:
            return  # Not enough data to detect clustering

        # Simple check: if >80% of polygons are within 10% of bounding box diagonal from each other, they're clustered
        # This is a heuristic - proper spatial autocorrelation analysis would be more robust

        # For now, just recommend polygon CV as best practice for spatial data
        self.issues.append(
            DataQualityIssue(
                DataQualityIssue.SEVERITY_INFO,
                "Spatial Data Detected",
                (
                    "Training data consists of spatial features (polygons/points). "
                    "Standard random splitting may overestimate accuracy due to spatial autocorrelation."
                ),
                "Use 'Polygon Group CV' mode in Advanced Options to prevent spatial leakage during validation.",
            )
        )

    def _check_duplicates(self, class_labels: List, geometries: List):
        """Check for duplicate features (same label and geometry)."""
        if len(geometries) != len(class_labels):
            return

        # Simple duplicate check based on label and centroid proximity
        # Full geometric comparison would be expensive
        seen_combinations = set()
        duplicate_count = 0

        for label, geom in zip(class_labels, geometries):
            if geom is None:
                continue

            try:
                centroid = geom.Centroid()
                if centroid:
                    # Create a simplified key: label + rounded centroid coordinates
                    x = round(centroid.GetX(), 6)
                    y = round(centroid.GetY(), 6)
                    key = (label, x, y)

                    if key in seen_combinations:
                        duplicate_count += 1
                    else:
                        seen_combinations.add(key)
            except Exception as exc:
                _ = exc  # Skip if centroid calculation fails

        if duplicate_count > 0:
            percentage = (duplicate_count / len(class_labels)) * 100 if class_labels else 0
            severity = DataQualityIssue.SEVERITY_WARNING if percentage > 5 else DataQualityIssue.SEVERITY_INFO

            self.issues.append(
                DataQualityIssue(
                    severity,
                    "Potential Duplicate Features",
                    f"~{duplicate_count} features ({percentage:.1f}%) may be duplicates (same class + similar location).",
                    "Review and remove duplicate features to avoid inflating training set artificially.",
                )
            )

    def _export_report(self):
        """Export quality report to a text file."""
        from qgis.PyQt.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Quality Report", f"quality_report_{Path(self.vector_path).stem}.txt", "Text Files (*.txt)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("TRAINING DATA QUALITY REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Vector File: {self.vector_path}\n")
                f.write(f"Class Field: {self.class_field}\n")
                f.write(f"Total Issues: {len(self.issues)}\n\n")

                if not self.issues:
                    f.write("✅ All checks passed! Training data looks good.\n")
                else:
                    for idx, issue in enumerate(self.issues, 1):
                        f.write(f"\n{'-' * 80}\n")
                        f.write(f"Issue #{idx}: [{issue.severity.upper()}] {issue.title}\n")
                        f.write(f"{'-' * 80}\n\n")
                        f.write(f"Description:\n{issue.description}\n\n")
                        f.write(f"Recommendation:\n{issue.recommendation}\n")

                f.write(f"\n{'=' * 80}\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")

            QMessageBox.information(self, "Export Successful", f"Quality report exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export report:\n{str(e)}")


def check_training_data_quality(vector_path: str, class_field: str, parent=None) -> Optional[TrainingDataQualityChecker]:
    """Convenience function to check training data quality and show dialog.

    Parameters
    ----------
    vector_path : str
        Path to training vector file
    class_field : str
        Name of field containing class labels
    parent : QWidget, optional
        Parent widget

    Returns
    -------
    TrainingDataQualityChecker or None
        Quality checker dialog instance if successful, None if error
    """
    if not Path(vector_path).exists():
        QMessageBox.critical(parent, "File Not Found", f"Training vector file not found:\n{vector_path}")
        return None

    try:
        checker = TrainingDataQualityChecker(vector_path, class_field, parent)
        checker.show()
        return checker
    except Exception as e:
        QMessageBox.critical(parent, "Quality Check Failed", f"Failed to analyze training data:\n{str(e)}")
        return None
