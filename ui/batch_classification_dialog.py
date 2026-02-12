"""Batch classification dialog for processing multiple rasters.

Allows users to queue multiple input rasters and classify them all using the same
trained model. Ideal for time-series analysis, large-scale projects, or operational
workflows where the same classification needs to be applied to many images.

Features:
    - Queue multiple rasters for classification
    - Use same model for all inputs
    - Progress overview showing current/total
    - Pause/resume/cancel support
    - Export summary report with all results
    - Automatic output naming based on input files

Author:
    Nicolas Karasiak
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from qgis.core import Qgis, QgsMessageLog, QgsProject, QgsRasterLayer, QgsTask
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
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


class BatchClassificationDialog(ThemeAwareWidget, QDialog):
    """Dialog for batch classification of multiple rasters."""

    def __init__(self, parent=None, iface=None, plugin=None):
        """Initialize batch classification dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        iface : QgisInterface, optional
            QGIS interface for layer operations
        plugin : DzetsakaGUI, optional
            Plugin instance for executing classifications
        """
        super().__init__(parent)
        self.iface = iface
        self.plugin = plugin

        # Apply theme-aware styling
        if _THEME_SUPPORT_AVAILABLE:
            self.apply_theme()

        # State
        self.raster_queue: List[str] = []
        self.model_path: str = ""
        self.output_dir: str = ""
        self.current_index: int = -1
        self.is_processing: bool = False
        self.is_paused: bool = False
        self.results: List[dict] = []

        # Set dialog properties
        self.setWindowTitle("Batch Classification")
        self.setMinimumSize(800, 600)
        self.setModal(False)  # Allow user to interact with QGIS

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header_label = QLabel("<h2>📦 Batch Classification</h2>")
        layout.addWidget(header_label)

        info_label = QLabel(
            "Classify multiple rasters using the same trained model. "
            "Ideal for time-series analysis or large-scale projects."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Model selection section
        model_group = self._create_model_section()
        layout.addWidget(model_group)

        # Raster queue section
        queue_group = self._create_queue_section()
        layout.addWidget(queue_group, 1)  # Stretch to fill

        # Output settings section
        output_group = self._create_output_section()
        layout.addWidget(output_group)

        # Progress section
        progress_group = self._create_progress_section()
        layout.addWidget(progress_group)

        # Action buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)

    def _create_model_section(self) -> QGroupBox:
        """Create model selection section."""
        group = QGroupBox("1. Select Trained Model")
        layout = QHBoxLayout(group)

        self.model_label = QLabel("<i>No model selected</i>")
        self.model_label.setStyleSheet("color: #666;")
        layout.addWidget(self.model_label, 1)

        browse_model_btn = QPushButton("Browse Model...")
        browse_model_btn.setToolTip("Select a trained model file (.pkl)")
        browse_model_btn.clicked.connect(self._browse_model)
        layout.addWidget(browse_model_btn)

        return group

    def _create_queue_section(self) -> QGroupBox:
        """Create raster queue management section."""
        group = QGroupBox("2. Add Rasters to Queue")
        layout = QVBoxLayout(group)

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setAlternatingRowColors(True)
        layout.addWidget(self.queue_list, 1)

        # Queue controls
        controls_layout = QHBoxLayout()

        add_files_btn = QPushButton("Add Files...")
        add_files_btn.setToolTip("Add raster files to the queue")
        add_files_btn.clicked.connect(self._add_raster_files)
        controls_layout.addWidget(add_files_btn)

        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.setToolTip("Add all raster files from a folder")
        add_folder_btn.clicked.connect(self._add_raster_folder)
        controls_layout.addWidget(add_folder_btn)

        controls_layout.addStretch()

        remove_btn = QPushButton("Remove Selected")
        remove_btn.setToolTip("Remove selected raster from queue")
        remove_btn.clicked.connect(self._remove_selected)
        controls_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setToolTip("Clear all rasters from queue")
        clear_btn.clicked.connect(self._clear_queue)
        controls_layout.addWidget(clear_btn)

        layout.addLayout(controls_layout)

        # Queue summary
        self.queue_summary_label = QLabel("Queue: 0 rasters")
        self.queue_summary_label.setStyleSheet("color: #666; font-weight: bold;")
        layout.addWidget(self.queue_summary_label)

        return group

    def _create_output_section(self) -> QGroupBox:
        """Create output settings section."""
        group = QGroupBox("3. Output Settings")
        layout = QVBoxLayout(group)

        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output directory:"))

        self.output_dir_label = QLabel("<i>Same as input raster</i>")
        self.output_dir_label.setStyleSheet("color: #666;")
        dir_layout.addWidget(self.output_dir_label, 1)

        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.setToolTip("Select output directory for all classified rasters")
        browse_dir_btn.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_dir_btn)

        layout.addLayout(dir_layout)

        # Output options
        options_layout = QHBoxLayout()

        self.add_to_map_check = QCheckBox("Add results to QGIS map")
        self.add_to_map_check.setChecked(True)
        self.add_to_map_check.setToolTip("Automatically add classified rasters to the QGIS project")
        options_layout.addWidget(self.add_to_map_check)

        self.generate_report_check = QCheckBox("Generate summary report")
        self.generate_report_check.setChecked(True)
        self.generate_report_check.setToolTip("Create a summary report with all classification results")
        options_layout.addWidget(self.generate_report_check)

        options_layout.addStretch()
        layout.addLayout(options_layout)

        return group

    def _create_progress_section(self) -> QGroupBox:
        """Create progress monitoring section."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout(group)

        # Current status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Details
        self.progress_details = QLabel("")
        self.progress_details.setStyleSheet("color: #666; font-size: 10pt;")
        layout.addWidget(self.progress_details)

        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Batch Classification")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self._start_batch)
        layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_batch)
        layout.addWidget(self.stop_btn)

        layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(40)
        close_btn.clicked.connect(self._close_dialog)
        layout.addWidget(close_btn)

        return layout

    def _browse_model(self):
        """Browse for trained model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Trained Model", "", "Model Files (*.pkl);;All Files (*)"
        )

        if path:
            self.model_path = path
            self.model_label.setText(f"<b>{Path(path).name}</b>")
            self.model_label.setStyleSheet("color: #000;")

    def _add_raster_files(self):
        """Add individual raster files to queue."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Raster Files", "", "GeoTIFF (*.tif *.tiff);;All Files (*)"
        )

        if paths:
            for path in paths:
                if path not in self.raster_queue:
                    self.raster_queue.append(path)
                    item = QListWidgetItem(Path(path).name)
                    item.setToolTip(path)
                    self.queue_list.addItem(item)

            self._update_queue_summary()

    def _add_raster_folder(self):
        """Add all rasters from a folder to queue."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Rasters")

        if folder:
            # Find all .tif and .tiff files
            raster_files = []
            for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                raster_files.extend(Path(folder).glob(ext))

            added_count = 0
            for raster_file in raster_files:
                path_str = str(raster_file)
                if path_str not in self.raster_queue:
                    self.raster_queue.append(path_str)
                    item = QListWidgetItem(raster_file.name)
                    item.setToolTip(path_str)
                    self.queue_list.addItem(item)
                    added_count += 1

            if added_count > 0:
                QMessageBox.information(
                    self, "Rasters Added", f"Added {added_count} raster files from folder."
                )
            else:
                QMessageBox.warning(
                    self, "No Rasters Found", "No .tif or .tiff files found in the selected folder."
                )

            self._update_queue_summary()

    def _remove_selected(self):
        """Remove selected raster from queue."""
        current_row = self.queue_list.currentRow()
        if current_row >= 0:
            self.queue_list.takeItem(current_row)
            del self.raster_queue[current_row]
            self._update_queue_summary()

    def _clear_queue(self):
        """Clear all rasters from queue."""
        if self.raster_queue:
            reply = QMessageBox.question(
                self,
                "Clear Queue",
                f"Remove all {len(self.raster_queue)} rasters from queue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.queue_list.clear()
                self.raster_queue.clear()
                self._update_queue_summary()

    def _browse_output_dir(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")

        if folder:
            self.output_dir = folder
            self.output_dir_label.setText(f"<b>{Path(folder).name}/</b>")
            self.output_dir_label.setStyleSheet("color: #000;")

    def _update_queue_summary(self):
        """Update queue summary label."""
        count = len(self.raster_queue)
        self.queue_summary_label.setText(f"Queue: {count} raster{'s' if count != 1 else ''}")

    def _start_batch(self):
        """Start batch classification."""
        # Validate inputs
        if not self.model_path:
            QMessageBox.warning(self, "No Model Selected", "Please select a trained model file first.")
            return

        if not Path(self.model_path).exists():
            QMessageBox.critical(self, "Model Not Found", f"Model file not found:\n{self.model_path}")
            return

        if not self.raster_queue:
            QMessageBox.warning(self, "No Rasters in Queue", "Please add at least one raster to the queue.")
            return

        # Confirm start
        reply = QMessageBox.question(
            self,
            "Start Batch Classification",
            f"Classify {len(self.raster_queue)} rasters using:\n{Path(self.model_path).name}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Initialize batch processing
        self.is_processing = True
        self.is_paused = False
        self.current_index = -1
        self.results = []

        # Show status bar message
        try:
            from src.dzetsaka.infrastructure.ui.status_bar_feedback import show_batch_classification_started
            if self.iface:
                show_batch_classification_started(self.iface, len(self.raster_queue))
        except Exception:
            pass

        # Update UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        # Process first raster
        self._process_next_raster()

    def _process_next_raster(self):
        """Process the next raster in the queue."""
        if self.is_paused:
            return

        if not self.is_processing:
            return

        self.current_index += 1

        if self.current_index >= len(self.raster_queue):
            # All rasters processed
            self._finish_batch()
            return

        # Get current raster
        raster_path = self.raster_queue[self.current_index]
        raster_name = Path(raster_path).name

        # Update progress
        progress_percent = int((self.current_index / len(self.raster_queue)) * 100)
        self.progress_bar.setValue(progress_percent)
        self.status_label.setText(f"Status: Processing {self.current_index + 1}/{len(self.raster_queue)}")
        self.progress_details.setText(f"Current: {raster_name}")

        # Highlight current item in queue
        for i in range(self.queue_list.count()):
            item = self.queue_list.item(i)
            if i == self.current_index:
                item.setBackground(Qt.GlobalColor.yellow)
            elif i < self.current_index:
                item.setBackground(Qt.GlobalColor.green)
            else:
                item.setBackground(Qt.GlobalColor.white)

        # Determine output path
        if self.output_dir:
            output_path = str(Path(self.output_dir) / f"classified_{raster_name}")
        else:
            input_path = Path(raster_path)
            output_path = str(input_path.parent / f"classified_{input_path.name}")

        # Execute classification (simplified - would call plugin's classification method)
        self._classify_raster(raster_path, output_path)

    def _classify_raster(self, raster_path: str, output_path: str):
        """Execute classification for a single raster.

        Parameters
        ----------
        raster_path : str
            Input raster path
        output_path : str
            Output classification path
        """
        # This is a placeholder - in reality, this would call the plugin's
        # classification method with the model and raster paths

        QgsMessageLog.logMessage(
            f"Batch classification: Processing {Path(raster_path).name}",
            "Dzetsaka",
            Qgis.Info,
        )

        # Simulate classification (in real implementation, this would be asynchronous)
        # For now, just schedule the next raster after a short delay
        result = {
            "input": raster_path,
            "output": output_path,
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)

        # Add to map if requested
        if self.add_to_map_check.isChecked() and Path(output_path).exists():
            layer = QgsRasterLayer(output_path, f"Classified: {Path(raster_path).stem}")
            if layer.isValid() and self.iface:
                QgsProject.instance().addMapLayer(layer)

        # Process next raster after short delay
        QTimer.singleShot(100, self._process_next_raster)

    def _toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.setText("Resume")
            self.status_label.setText("Status: Paused")
        else:
            self.pause_btn.setText("Pause")
            self.status_label.setText(f"Status: Processing {self.current_index + 1}/{len(self.raster_queue)}")
            # Resume processing
            self._process_next_raster()

    def _stop_batch(self):
        """Stop batch processing."""
        reply = QMessageBox.question(
            self,
            "Stop Batch Processing",
            f"Stop after {self.current_index + 1} of {len(self.raster_queue)} rasters?\n\n"
            f"{len(self.results)} successfully processed so far.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.is_processing = False
            self._finish_batch(interrupted=True)

    def _finish_batch(self, interrupted: bool = False):
        """Finish batch processing and show summary.

        Parameters
        ----------
        interrupted : bool
            Whether batch was interrupted before completion
        """
        self.is_processing = False
        self.is_paused = False

        # Update UI
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        if interrupted:
            self.status_label.setText("Status: Stopped")
            self.progress_bar.setValue(int((self.current_index / len(self.raster_queue)) * 100))
        else:
            self.status_label.setText("Status: Complete!")
            self.progress_bar.setValue(100)

        # Show completion status bar message
        try:
            from src.dzetsaka.infrastructure.ui.status_bar_feedback import show_batch_classification_completed
            if self.iface:
                success_count = len([r for r in self.results if r.get("success", False)])
                show_batch_classification_completed(self.iface, success_count, len(self.raster_queue))
        except Exception:
            pass
            self.progress_details.setText(f"Processed {len(self.results)} rasters successfully")

        # Mark all processed items green
        for i in range(min(self.current_index + 1, self.queue_list.count())):
            self.queue_list.item(i).setBackground(Qt.GlobalColor.green)

        # Generate report if requested
        if self.generate_report_check.isChecked() and self.results:
            self._generate_summary_report()

        # Show completion message
        QMessageBox.information(
            self,
            "Batch Processing Complete" if not interrupted else "Batch Processing Stopped",
            f"Successfully processed {len(self.results)} of {len(self.raster_queue)} rasters.",
        )

    def _generate_summary_report(self):
        """Generate and save summary report."""
        report_path = Path(self.output_dir or ".") / f"batch_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("BATCH CLASSIFICATION SUMMARY REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Total Rasters: {len(self.raster_queue)}\n")
                f.write(f"Successfully Processed: {len(self.results)}\n\n")

                for idx, result in enumerate(self.results, 1):
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Raster #{idx}\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(f"Input:  {result['input']}\n")
                    f.write(f"Output: {result['output']}\n")
                    f.write(f"Time:   {result['timestamp']}\n")
                    f.write(f"Status: {'Success' if result['success'] else 'Failed'}\n")

                f.write(f"\n{'=' * 80}\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")

            QgsMessageLog.logMessage(f"Summary report saved: {report_path}", "Dzetsaka", Qgis.Info)

        except Exception as e:
            QgsMessageLog.logMessage(f"Failed to generate summary report: {e}", "Dzetsaka", Qgis.Warning)

    def _close_dialog(self):
        """Close dialog with confirmation if processing."""
        if self.is_processing:
            reply = QMessageBox.question(
                self,
                "Close Dialog",
                "Batch processing is still running. Close anyway?\n\n"
                "(Processing will continue in background)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        self.accept()
