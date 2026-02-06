"""Custom progress dialog for package installation with live output display.

This dialog provides:
- Live pip output streaming
- Responsive UI during installation
- Functional cancel button
- Auto-scrolling terminal-like output
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
    """Progress dialog for package installation with live output."""

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

        self.setWindowTitle("Installing Dependencies")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        # Set window modality - WindowModal allows continued QGIS interaction
        try:
            modality = Qt.WindowModality.WindowModal
        except AttributeError:
            modality = Qt.WindowModal
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

        # Output text area
        output_label = QLabel("Installation Output:")
        layout.addWidget(output_label)

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

        layout.addWidget(self.output_text)

        # Button layout
        button_layout = QHBoxLayout()
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
