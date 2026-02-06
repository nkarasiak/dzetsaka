"""Centralized logging utilities for Dzetsaka.

All logging is routed to the QGIS Message Log.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import platform
import sys
import traceback
import urllib.parse

try:
    from qgis.core import Qgis, QgsMessageLog
except Exception:  # pragma: no cover - QGIS not available in tests
    Qgis = None
    QgsMessageLog = None


DEFAULT_LOG_TAG = "Dzetsaka"
_shown_error_dialogs: set[str] = set()
_recent_log_lines: deque[str] = deque(maxlen=1200)


def _level_name(level: Optional[int]) -> str:
    if Qgis is None or level is None:
        return "INFO"
    if level == Qgis.Critical:
        return "CRITICAL"
    if level == Qgis.Warning:
        return "WARNING"
    if level == Qgis.Success:
        return "SUCCESS"
    return "INFO"


def _append_recent_log(tag: str, level: Optional[int], message: Any) -> None:
    text = _format_message(message).strip()
    if not text:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lvl = _level_name(level)
    for line in text.splitlines():
        _recent_log_lines.append(f"{ts} [{lvl}] {tag}: {line}")


def get_recent_log_output(max_lines: int = 400) -> str:
    """Return recent dzetsaka log lines captured in this session."""
    if max_lines <= 0 or not _recent_log_lines:
        return ""
    lines = list(_recent_log_lines)[-max_lines:]
    return "\n".join(lines)


def get_system_info() -> str:
    """Collect system information for error reporting."""
    info_lines = []

    # Python version
    info_lines.append(f"Python: {sys.version}")

    # Operating System
    info_lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")

    # QGIS version
    try:
        from qgis.core import Qgis as QgisVersion

        info_lines.append(f"QGIS: {QgisVersion.QGIS_VERSION}")
    except Exception:
        info_lines.append("QGIS: Unknown")

    # Plugin version
    try:
        from . import metadata

        version = metadata.get("general", "version", fallback="Unknown")
        info_lines.append(f"dzetsaka: {version}")
    except Exception:
        info_lines.append("dzetsaka: Unknown")

    # Dependencies
    dependencies = {}
    for pkg in ["sklearn", "xgboost", "lightgbm", "catboost", "numpy", "gdal"]:
        try:
            if pkg == "sklearn":
                import sklearn

                dependencies[pkg] = getattr(sklearn, "__version__", "Unknown version")
            elif pkg == "xgboost":
                import xgboost

                dependencies[pkg] = getattr(xgboost, "__version__", "Unknown version")
            elif pkg == "lightgbm":
                import lightgbm

                dependencies[pkg] = getattr(lightgbm, "__version__", "Unknown version")
            elif pkg == "catboost":
                import catboost

                dependencies[pkg] = getattr(catboost, "__version__", "Unknown version")
            elif pkg == "numpy":
                import numpy

                dependencies[pkg] = getattr(numpy, "__version__", "Unknown version")
            elif pkg == "gdal":
                from osgeo import gdal

                dependencies[pkg] = getattr(gdal, "__version__", "Unknown version")
        except ImportError:
            dependencies[pkg] = "Not installed"
        except Exception as e:
            dependencies[pkg] = f"Error reading version: {e!s}"

    info_lines.append("\nDependencies:")
    for pkg, version in dependencies.items():
        info_lines.append(f"  - {pkg}: {version}")

    return "\n".join(info_lines)


def create_github_issue_url(title: str, body: str) -> str:
    """Create a pre-filled GitHub issue URL."""
    try:
        from . import constants

        github_url = constants.GITHUB_NEW_ISSUE_URL
    except ImportError:
        # Fallback if constants module not available
        github_url = "https://github.com/nkarasiak/dzetsaka/issues/new"

    params = {"title": title, "body": body}
    query_string = urllib.parse.urlencode(params)
    return f"{github_url}?{query_string}"


def build_issue_template(
    error_title: str,
    error_type: str,
    error_message: Any,
    context: str = "",
    max_log_lines: int = 1200,
) -> str:
    """Build a markdown issue template including environment and recent logs."""
    msg = _format_message(error_message)
    system_info = get_system_info()
    log_output = get_recent_log_output(max_lines=max_log_lines) or "[No dzetsaka logs captured in this session]"
    return f"""## Bug Report: {error_title}

**Error Type:** {error_type}

**Error Message:**
```
{msg}
```

**Context:**
{context or "N/A"}

**Environment:**
{system_info}

**Steps to Reproduce:**
1. [Please describe the steps that led to this error]
2.
3.

**Expected Behavior:**
[What you expected to happen]

**Additional Information:**
[Any additional context, screenshots, or logs that might help]

**dzetsaka Log Output (recent):**
```
{log_output}
```
"""


def show_issue_popup(
    error_title: str,
    error_type: str,
    error_message: Any,
    context: str = "",
    parent: Optional[Any] = None,
) -> None:
    """Show a compact, copy-ready issue popup."""
    text = _format_message(error_message)
    if not text:
        return

    key = f"{error_title}|{error_type}|{text}|{context}"
    if key in _shown_error_dialogs:
        return
    _shown_error_dialogs.add(key)

    try:
        import webbrowser

        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import (
            QApplication,
            QDialog,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QVBoxLayout,
        )
    except Exception:
        return

    app = QApplication.instance()
    if app is None:
        return
    parent_widget = parent if parent is not None else app.activeWindow()
    github_template = build_issue_template(error_title, error_type, error_message, context=context)

    try:
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle("dzetsaka Error Report")
        dialog.setModal(True)
        dialog.resize(760, 520)

        layout = QVBoxLayout(dialog)

        summary = QLabel(f"<b>{error_title}</b><br>{text[:220]}")
        summary.setWordWrap(True)
        summary.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(summary)

        info = QLabel(
            "Use <b>Copy Report</b>, then open GitHub issues. The report already includes recent dzetsaka logs."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(info)

        report_box = QPlainTextEdit()
        report_box.setReadOnly(True)
        report_box.setPlainText(github_template)
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
            clipboard = app.clipboard()
            clipboard.setText(github_template)
            QMessageBox.information(parent_widget, "Copied", "Error report copied to clipboard.")

        copy_button.clicked.connect(_copy_to_clipboard)
        open_button.clicked.connect(lambda: webbrowser.open("https://github.com/nkarasiak/dzetsaka/issues"))
        close_button.clicked.connect(dialog.close)

        if hasattr(dialog, "exec"):
            dialog.exec()
        else:
            dialog.exec_()
    except Exception:
        try:
            from qgis.PyQt.QtWidgets import QMessageBox

            QMessageBox.warning(
                parent_widget,
                error_title,
                f"{text}\n\nPlease report: https://github.com/nkarasiak/dzetsaka/issues",
            )
        except Exception:
            return


def show_error_dialog(title: str, message: Any, parent: Optional[Any] = None) -> None:
    """Backward-compatible alias to the standardized issue popup."""
    show_issue_popup(
        error_title=title,
        error_type="Runtime Error",
        error_message=message,
        context="Raised through logging_utils.show_error_dialog()",
        parent=parent,
    )


def _is_main_qt_thread() -> bool:
    """Return True if current execution is on the Qt GUI thread."""
    try:
        from qgis.PyQt.QtCore import QThread
        from qgis.PyQt.QtWidgets import QApplication
    except Exception:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    app_thread = app.thread()
    current_thread = QThread.currentThread()
    return app_thread is not None and current_thread is not None and app_thread == current_thread



def _format_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    return repr(message)


@dataclass(frozen=True)
class QgisLogger:
    """QGIS Message Log adapter with level-aware helpers."""

    tag: str = DEFAULT_LOG_TAG

    def _log(self, level: Optional[int], message: Any) -> None:
        _append_recent_log(self.tag, level, message)
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
        if _is_main_qt_thread():
            show_error_dialog("dzetsaka Error", message)

    def exception(self, message: Any, exc: Optional[BaseException] = None) -> None:
        self.logger.exception(message, exc)
        if _is_main_qt_thread():
            if exc is None:
                show_error_dialog("dzetsaka Error", message)
            else:
                show_error_dialog("dzetsaka Error", f"{_format_message(message)}\n{exc!s}")

    def progress(self, value: float | int) -> None:
        if self.feedback is not None and hasattr(self.feedback, "setProgress"):
            self.feedback.setProgress(int(value))
