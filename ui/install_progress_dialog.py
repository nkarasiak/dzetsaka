"""Custom progress dialog for package installation.

Default behavior is progress-first: logs stay hidden unless the user
explicitly expands details.
"""

from qgis.PyQt.QtCore import Qt, pyqtSlot
from qgis.PyQt.QtGui import QFont, QTextCursor
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPlainTextEdit,
    QPushButton,
    QHBoxLayout,
)


class InstallProgressDialog(QDialog):
    """Progress dialog for package installation with optional details."""

    _COLLAPSED_MIN_HEIGHT = 150
    _COLLAPSED_TARGET_HEIGHT = 190
    _EXPANDED_MIN_HEIGHT = 380
    _EXPANDED_TARGET_HEIGHT = 420

    def __init__(self, parent=None, total_packages=1):
        """Initialize the installation progress dialog.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        total_packages : int
            Total number of packages to install
        """
        super().__init__(parent)
        self.total_packages = total_packages
        self.current_package = 0
        self._cancelled = False
        self._details_visible = False

        self.setWindowTitle("Installing Dependencies")
        self.setMinimumWidth(520)
        self.setMinimumHeight(self._COLLAPSED_MIN_HEIGHT)

        # Keep non-modal to avoid perceived UI freezes while install runs.
        try:
            modality = Qt.WindowModality.NonModal
        except AttributeError:
            modality = Qt.NonModal
        self.setWindowModality(modality)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel("Preparing installation...")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.total_packages)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Output text area (hidden by default)
        self.output_label = QLabel("Installation Output:")
        self.output_label.setVisible(False)
        layout.addWidget(self.output_label)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Set monospace font for terminal-like appearance
        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Courier New", 9)
        if not font.exactMatch():
            font = QFont("monospace", 9)
        self.output_text.setFont(font)

        self.output_text.setVisible(False)
        layout.addWidget(self.output_text)

        # Button layout
        button_layout = QHBoxLayout()

        self.details_button = QPushButton("Show details")
        self.details_button.clicked.connect(self._toggle_details)
        button_layout.addWidget(self.details_button)

        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def set_current_package(self, package_name, index):
        """Update the current package being installed.

        Parameters
        ----------
        package_name : str
            Name of the package being installed
        index : int
            Zero-based index of the package (0 to total_packages-1)
        """
        self.current_package = index
        self.status_label.setText(f"Installing {package_name}... ({index + 1}/{self.total_packages})")
        self.progress_bar.setValue(index)

    def append_output(self, text):
        """Append text to the output area.

        Parameters
        ----------
        text : str
            Text to append
        """
        if text:
            self.output_text.appendPlainText(text.rstrip())
            # Auto-scroll to bottom
            self.output_text.moveCursor(QTextCursor.MoveOperation.End)

    def mark_package_complete(self):
        """Mark the current package as complete and increment progress."""
        self.progress_bar.setValue(self.current_package + 1)

    def mark_complete(self, success=True):
        """Mark the entire installation as complete.

        Parameters
        ----------
        success : bool
            Whether the installation succeeded
        """
        if success:
            self.status_label.setText("Installation complete!")
            self.cancel_button.setText("Close")
        else:
            self.status_label.setText("Installation failed or cancelled.")
            self.cancel_button.setText("Close")

        self.progress_bar.setValue(self.total_packages)

    def was_cancelled(self):
        """Check if the user cancelled the installation.

        Returns
        -------
        bool
            True if cancelled, False otherwise
        """
        return self._cancelled

    @pyqtSlot()
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        if self.cancel_button.text() == "Close":
            self.accept()
        else:
            self._cancelled = True
            self.status_label.setText("Cancelling installation...")
            self.cancel_button.setEnabled(False)

    @pyqtSlot()
    def _toggle_details(self):
        """Toggle visibility of installation log output."""
        self._details_visible = not self._details_visible
        self.output_label.setVisible(self._details_visible)
        self.output_text.setVisible(self._details_visible)
        self.details_button.setText("Hide details" if self._details_visible else "Show details")
        if self._details_visible:
            self.setMinimumHeight(self._EXPANDED_MIN_HEIGHT)
            self.resize(max(self.width(), 600), max(self.height(), self._EXPANDED_TARGET_HEIGHT))
        else:
            self.setMinimumHeight(self._COLLAPSED_MIN_HEIGHT)
            # Explicitly shrink back after the user hides details.
            self.resize(max(self.width(), 520), self._COLLAPSED_TARGET_HEIGHT)
