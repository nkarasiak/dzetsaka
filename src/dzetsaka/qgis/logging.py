"""QGIS-specific logging adapters for dzetsaka."""

from __future__ import annotations

import contextlib
import traceback
import webbrowser
from dataclasses import dataclass
from typing import Any

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from dzetsaka.logging import (
    DEFAULT_LOG_TAG,
    Logger,
    build_issue_template,
    register_error_handler,
    register_issue_popup_handler,
    register_logger_factory,
)


def _format_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    return repr(message)


@dataclass(frozen=True)
class QgisLogger(Logger):
    """Adapter using the QGIS message log."""

    tag: str = DEFAULT_LOG_TAG

    def _log(self, level: int | None, message: Any) -> None:
        text = _format_message(message)
        if QgsMessageLog is None:
            return
        if level is None or Qgis is None:
            QgsMessageLog.logMessage(text, tag=self.tag)
            return
        QgsMessageLog.logMessage(text, tag=self.tag, level=level)

    def info(self, message: Any) -> None:
        self._log(Qgis.Info if Qgis is not None else None, message)

    def warning(self, message: Any) -> None:
        self._log(Qgis.Warning if Qgis is not None else None, message)

    def error(self, message: Any) -> None:
        self._log(Qgis.Critical if Qgis is not None else None, message)

    def exception(self, message: Any, exc: BaseException | None = None) -> None:
        if exc is None:
            details = traceback.format_exc()
        else:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.error(f"{_format_message(message)}\n{details}")


def _active_window() -> Any | None:
    try:
        app = QApplication.instance()
        if app is None:
            return None
        return app.activeWindow()
    except Exception:
        return None


def _create_issue_dialog(
    parent: Any | None,
    title: str,
    template: str,
) -> None:
    dialog = QDialog(parent if parent is not None else _active_window())
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.resize(760, 520)
    layout = QVBoxLayout(dialog)

    summary = QLabel(f"<b>{title}</b>")
    summary.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(summary)

    report_box = QPlainTextEdit()
    report_box.setReadOnly(True)
    report_box.setPlainText(template)
    layout.addWidget(report_box)

    row = QHBoxLayout()
    copy_button = QPushButton("Copy Report")
    open_button = QPushButton("Open GitHub Issues")
    close_button = QPushButton("Close")
    row.addWidget(copy_button)
    row.addWidget(open_button)
    row.addStretch()
    row.addWidget(close_button)
    layout.addLayout(row)

    def _copy_to_clipboard() -> None:
        clipboard = QApplication.instance().clipboard()
        clipboard.setText(template)
        QMessageBox.information(dialog, "Copied", "Error report copied to clipboard.")

    copy_button.clicked.connect(_copy_to_clipboard)
    open_button.clicked.connect(lambda: webbrowser.open("https://github.com/nkarasiak/dzetsaka/issues"))
    close_button.clicked.connect(dialog.close)

    with contextlib.suppress(Exception):
        dialog.exec()


def _show_issue_popup(
    error_title: str,
    error_type: str,
    error_message: Any,
    context: str | None,
    parent: Any | None,
) -> None:
    template = build_issue_template(
        error_title=error_title,
        error_type=error_type,
        error_message=error_message,
        context=context or "",
    )
    _create_issue_dialog(parent, error_title, template)


def _show_error_dialog(title: str, message: Any, context: str | None) -> None:
    _show_issue_popup(title, "Runtime Error", message, context, _active_window())


def register_qgis_logging() -> None:
    """Wire up the QGIS logger, error handler, and issue popup."""

    def _logger_factory(tag: str) -> Logger:
        return QgisLogger(tag=tag)

    register_logger_factory(_logger_factory)
    register_issue_popup_handler(_show_issue_popup)
    register_error_handler(_show_error_dialog)
