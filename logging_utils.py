"""Centralized logging utilities for Dzetsaka.

All logging is routed to the QGIS Message Log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import traceback

try:
    from qgis.core import Qgis, QgsMessageLog
except Exception:  # pragma: no cover - QGIS not available in tests
    Qgis = None
    QgsMessageLog = None


DEFAULT_LOG_TAG = "Dzetsaka"
_shown_error_dialogs: set[str] = set()


def show_error_dialog(title: str, message: Any, parent: Optional[Any] = None) -> None:
    """Show a best-effort error dialog when a GUI is available."""
    text = _format_message(message)
    if not text:
        return

    key = f"{title}|{text}"
    if key in _shown_error_dialogs:
        return
    _shown_error_dialogs.add(key)

    try:
        from qgis.PyQt.QtWidgets import QApplication, QMessageBox
    except Exception:
        return

    app = QApplication.instance()
    if app is None:
        return

    parent_widget = parent or app.activeWindow()
    QMessageBox.warning(parent_widget, title, text, QMessageBox.StandardButton.Ok)


def _format_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    return repr(message)


@dataclass(frozen=True)
class QgisLogger:
    """QGIS Message Log adapter with level-aware helpers."""

    tag: str = DEFAULT_LOG_TAG

    def _log(self, level: Optional[int], message: Any) -> None:
        if QgsMessageLog is None:
            return

        text = _format_message(message)
        if Qgis is None or level is None:
            QgsMessageLog.logMessage(text, tag=self.tag)
        else:
            QgsMessageLog.logMessage(text, tag=self.tag, level=level)

    def info(self, message: Any) -> None:
        self._log(Qgis.Info if Qgis is not None else None, message)

    def warning(self, message: Any) -> None:
        self._log(Qgis.Warning if Qgis is not None else None, message)

    def error(self, message: Any) -> None:
        self._log(Qgis.Critical if Qgis is not None else None, message)

    def exception(self, message: Any, exc: Optional[BaseException] = None) -> None:
        if exc is None:
            details = traceback.format_exc()
        else:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.error(f"{_format_message(message)}\n{details}")


@dataclass
class Reporter:
    """Message logger with optional progress reporting."""

    logger: QgisLogger
    feedback: Optional[Any] = None

    @classmethod
    def from_feedback(cls, feedback: Optional[Any], tag: str = DEFAULT_LOG_TAG) -> "Reporter":
        return cls(logger=QgisLogger(tag=tag), feedback=feedback)

    def info(self, message: Any) -> None:
        self.logger.info(message)

    def warning(self, message: Any) -> None:
        self.logger.warning(message)

    def error(self, message: Any) -> None:
        self.logger.error(message)
        show_error_dialog("dzetsaka Error", message)

    def exception(self, message: Any, exc: Optional[BaseException] = None) -> None:
        self.logger.exception(message, exc)
        if exc is None:
            show_error_dialog("dzetsaka Error", message)
        else:
            show_error_dialog("dzetsaka Error", f"{_format_message(message)}\n{exc!s}")

    def progress(self, value: float | int) -> None:
        if self.feedback is not None and hasattr(self.feedback, "setProgress"):
            self.feedback.setProgress(int(value))
