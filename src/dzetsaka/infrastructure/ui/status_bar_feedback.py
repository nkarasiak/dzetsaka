"""Status bar feedback utilities for dzetsaka.

Provides consistent status bar messaging throughout the plugin for better user feedback.

Author:
    Nicolas Karasiak
"""

from typing import Optional

try:
    from qgis.core import Qgis, QgsMessageLog
    _QGIS_AVAILABLE = True
except ImportError:
    _QGIS_AVAILABLE = False


class StatusBarMessenger:
    """Utility class for consistent status bar messaging."""

    # Message durations (in seconds)
    DURATION_SHORT = 3
    DURATION_MEDIUM = 5
    DURATION_LONG = 10

    def __init__(self, iface=None):
        """Initialize status bar messenger.

        Parameters
        ----------
        iface : QgisInterface, optional
            QGIS interface instance
        """
        self.iface = iface

    def success(self, message: str, duration: int = DURATION_MEDIUM):
        """Show success message in status bar.

        Parameters
        ----------
        message : str
            Success message to display
        duration : int, optional
            Display duration in seconds
        """
        if self.iface and hasattr(self.iface, "messageBar"):
            if _QGIS_AVAILABLE:
                self.iface.messageBar().pushMessage(
                    "dzetsaka",
                    message,
                    level=Qgis.Success,
                    duration=duration
                )
        self._log_message(message, "INFO")

    def info(self, message: str, duration: int = DURATION_MEDIUM):
        """Show info message in status bar.

        Parameters
        ----------
        message : str
            Info message to display
        duration : int, optional
            Display duration in seconds
        """
        if self.iface and hasattr(self.iface, "messageBar"):
            if _QGIS_AVAILABLE:
                self.iface.messageBar().pushMessage(
                    "dzetsaka",
                    message,
                    level=Qgis.Info,
                    duration=duration
                )
        self._log_message(message, "INFO")

    def warning(self, message: str, duration: int = DURATION_LONG):
        """Show warning message in status bar.

        Parameters
        ----------
        message : str
            Warning message to display
        duration : int, optional
            Display duration in seconds
        """
        if self.iface and hasattr(self.iface, "messageBar"):
            if _QGIS_AVAILABLE:
                self.iface.messageBar().pushMessage(
                    "dzetsaka",
                    message,
                    level=Qgis.Warning,
                    duration=duration
                )
        self._log_message(message, "WARNING")

    def error(self, message: str, duration: int = DURATION_LONG):
        """Show error message in status bar.

        Parameters
        ----------
        message : str
            Error message to display
        duration : int, optional
            Display duration in seconds
        """
        if self.iface and hasattr(self.iface, "messageBar"):
            if _QGIS_AVAILABLE:
                self.iface.messageBar().pushMessage(
                    "dzetsaka",
                    message,
                    level=Qgis.Critical,
                    duration=duration
                )
        self._log_message(message, "ERROR")

    def clear(self):
        """Clear all messages from status bar."""
        if self.iface and hasattr(self.iface, "messageBar"):
            self.iface.messageBar().clearWidgets()

    def _log_message(self, message: str, level: str):
        """Log message to QGIS message log.

        Parameters
        ----------
        message : str
            Message to log
        level : str
            Log level (INFO, WARNING, ERROR)
        """
        if _QGIS_AVAILABLE:
            qgis_level = {
                "INFO": Qgis.Info,
                "WARNING": Qgis.Warning,
                "ERROR": Qgis.Critical
            }.get(level, Qgis.Info)
            QgsMessageLog.logMessage(message, "dzetsaka", level=qgis_level)


# Convenience functions for common messages
def show_quality_check_started(iface):
    """Show message when quality check starts."""
    messenger = StatusBarMessenger(iface)
    messenger.info("Analyzing training data quality...", duration=2)


def show_quality_check_completed(iface, issue_count: int):
    """Show message when quality check completes.

    Parameters
    ----------
    iface : QgisInterface
        QGIS interface
    issue_count : int
        Number of issues found
    """
    messenger = StatusBarMessenger(iface)
    if issue_count == 0:
        messenger.success("✓ Training data quality check passed! No issues found.", duration=5)
    elif issue_count == 1:
        messenger.warning("⚠ Training data quality check found 1 issue. Review recommendations.", duration=5)
    else:
        messenger.warning(f"⚠ Training data quality check found {issue_count} issues. Review recommendations.", duration=5)


def show_batch_classification_started(iface, raster_count: int):
    """Show message when batch classification starts.

    Parameters
    ----------
    iface : QgisInterface
        QGIS interface
    raster_count : int
        Number of rasters to process
    """
    messenger = StatusBarMessenger(iface)
    messenger.info(f"Starting batch classification of {raster_count} rasters...", duration=3)


def show_batch_classification_progress(iface, current: int, total: int):
    """Show batch classification progress.

    Parameters
    ----------
    iface : QgisInterface
        QGIS interface
    current : int
        Current raster number
    total : int
        Total number of rasters
    """
    messenger = StatusBarMessenger(iface)
    progress_pct = int((current / total) * 100)
    messenger.info(f"Batch classification: {current}/{total} complete ({progress_pct}%)", duration=2)


def show_batch_classification_completed(iface, success_count: int, total_count: int):
    """Show message when batch classification completes.

    Parameters
    ----------
    iface : QgisInterface
        QGIS interface
    success_count : int
        Number of successful classifications
    total_count : int
        Total number of rasters attempted
    """
    messenger = StatusBarMessenger(iface)
    if success_count == total_count:
        messenger.success(f"✓ Batch classification complete! {success_count}/{total_count} rasters processed successfully.", duration=10)
    else:
        failed_count = total_count - success_count
        messenger.warning(f"⚠ Batch classification complete with errors: {success_count}/{total_count} successful, {failed_count} failed.", duration=10)


def show_confidence_analysis_ready(iface):
    """Show message when confidence analysis is ready."""
    messenger = StatusBarMessenger(iface)
    messenger.info("Confidence map analysis ready. Open the HTML report for details.", duration=5)


def show_dialog_opened(iface, dialog_name: str):
    """Show message when a dialog opens.

    Parameters
    ----------
    iface : QgisInterface
        QGIS interface
    dialog_name : str
        Name of the dialog
    """
    messenger = StatusBarMessenger(iface)
    messenger.info(f"Opening {dialog_name}...", duration=2)
