"""Core logging and issue helpers for dzetsaka."""

from __future__ import annotations

import logging
import platform
import sys
import traceback
import urllib.parse
from collections import deque
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

DEFAULT_LOG_TAG = "Dzetsaka"
_recent_log_lines: deque[str] = deque(maxlen=1200)


class Logger(Protocol):
    """Simple logger protocol that mirrors qgis logging semantics."""

    def info(self, message: Any) -> None: ...

    def warning(self, message: Any) -> None: ...

    def error(self, message: Any) -> None: ...

    def exception(self, message: Any, exc: BaseException | None = None) -> None: ...


class FeedbackProtocol(Protocol):
    """Minimal feedback interface used by dzetsaka workflows."""

    def setProgress(self, value: float | int) -> None: ...

    def setProgressText(self, message: str) -> None: ...


LoggerFactory = Callable[[str], Logger]
IssuePopupHandler = Callable[[str, str, Any, Optional[str], Optional[Any]], None]
ErrorHandler = Callable[[str, Any, Optional[str]], None]


def _format_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    return repr(message)


def _level_name(level: int | None) -> str:
    if level is None:
        return "INFO"
    if level == logging.CRITICAL:
        return "CRITICAL"
    if level == logging.WARNING:
        return "WARNING"
    if level == logging.INFO:
        return "INFO"
    if level == logging.DEBUG:
        return "DEBUG"
    return "INFO"


def record_log_entry(tag: str, level: int | None, message: Any) -> None:
    """Capture recent log entries for issue reporting."""
    text = _format_message(message)
    if not text:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _recent_log_lines.append(f"{timestamp} [{_level_name(level)}] {tag}: {text}")


def get_recent_log_output(max_lines: int = 400) -> str:
    """Return recent log lines captured in the current process."""
    if max_lines <= 0 or not _recent_log_lines:
        return ""
    lines = list(_recent_log_lines)[-max_lines:]
    return "\n".join(lines)


def _read_plugin_metadata_value(key: str, default: str = "Unknown") -> str:
    plugin_root = Path(__file__).resolve().parents[2]
    metadata_path = plugin_root / "metadata.txt"
    if not metadata_path.exists():
        return default
    parser = ConfigParser()
    try:
        with metadata_path.open(encoding="utf-8") as fh:
            parser.read_file(fh)
        return parser.get("general", key, fallback=default)
    except Exception:
        return default


def get_system_info() -> str:
    """Gather system/environment information for issue reports."""
    info_lines: list[str] = []
    info_lines.append(f"Python: {sys.version}")
    info_lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    try:
        from qgis.core import Qgis as QgisVersion

        info_lines.append(f"QGIS: {QgisVersion.QGIS_VERSION}")
    except Exception:
        info_lines.append("QGIS: Unknown")

    version = _read_plugin_metadata_value("version")
    info_lines.append(f"dzetsaka: {version}")

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
        except Exception as exc:  # pragma: no cover
            dependencies[pkg] = f"Error reading version: {exc!s}"

    info_lines.append("\nDependencies:")
    for pkg, version in dependencies.items():
        info_lines.append(f"  - {pkg}: {version}")

    return "\n".join(info_lines)


def create_github_issue_url(title: str, body: str) -> str:
    """Return a GitHub issue URL prefilled with title/body."""
    try:
        from . import constants

        github_url = constants.GITHUB_NEW_ISSUE_URL
    except (ImportError, ModuleNotFoundError):
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
    """Compose a markdown issue template containing system info and logs."""
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


def _default_issue_popup_handler(
    error_title: str,
    error_type: str,
    error_message: Any,
    context: str | None,
    parent: Any | None,
) -> None:
    template = build_issue_template(error_title, error_type, error_message, context=context or "")
    print("\n".join(["---", f"[dzetsaka issue] {error_title}", template, "---"]))


def _default_error_handler(title: str, message: Any, context: str | None) -> None:
    print(f"{title}: {_format_message(message)}", file=sys.stderr)


_issue_popup_handler: IssuePopupHandler = _default_issue_popup_handler
_error_handler: ErrorHandler = _default_error_handler


def _logger_factory(tag):
    return PythonLogger(tag)


def register_issue_popup_handler(handler: IssuePopupHandler) -> None:
    global _issue_popup_handler
    _issue_popup_handler = handler


def register_error_handler(handler: ErrorHandler) -> None:
    global _error_handler
    _error_handler = handler


def register_logger_factory(factory: LoggerFactory) -> None:
    global _logger_factory
    _logger_factory = factory


def create_logger(tag: str = DEFAULT_LOG_TAG) -> Logger:
    return _logger_factory(tag)


def show_issue_popup(
    *,
    error_title: str,
    error_type: str,
    error_message: Any,
    context: str | None = "",
    parent: Any | None = None,
) -> None:
    _issue_popup_handler(error_title, error_type, error_message, context, parent)


def show_error_dialog(title: str, message: Any) -> None:
    _error_handler(title, message, None)


@dataclass
class PythonLogger:
    tag: str = DEFAULT_LOG_TAG

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.tag)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def _log(self, level: int, message: Any) -> None:
        record_log_entry(self.tag, level, message)
        if level == logging.CRITICAL:
            self._logger.critical(_format_message(message))
        elif level == logging.WARNING:
            self._logger.warning(_format_message(message))
        elif level == logging.DEBUG:
            self._logger.debug(_format_message(message))
        elif level == logging.INFO:
            self._logger.info(_format_message(message))
        else:
            self._logger.log(level, _format_message(message))

    def info(self, message: Any) -> None:
        self._log(logging.INFO, message)

    def warning(self, message: Any) -> None:
        self._log(logging.WARNING, message)

    def error(self, message: Any) -> None:
        self._log(logging.CRITICAL, message)

    def exception(self, message: Any, exc: BaseException | None = None) -> None:
        details = _format_message(message)
        if exc is None:
            details += "\n" + traceback.format_exc()
        else:
            details += "\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._log(logging.CRITICAL, details)


@dataclass
class Reporter:
    logger: Logger
    feedback: FeedbackProtocol | None = None
    error_handler: ErrorHandler = _default_error_handler

    @classmethod
    def from_feedback(cls, feedback: FeedbackProtocol | None, tag: str = DEFAULT_LOG_TAG) -> Reporter:
        logger = _logger_factory(tag)
        return cls(logger=logger, feedback=feedback, error_handler=_error_handler)

    def info(self, message: Any) -> None:
        self.logger.info(message)

    def warning(self, message: Any) -> None:
        self.logger.warning(message)

    def error(self, message: Any) -> None:
        self.logger.error(message)
        self.error_handler("dzetsaka Error", message, None)

    def exception(self, message: Any, exc: BaseException | None = None) -> None:
        self.logger.exception(message, exc)
        self.error_handler("dzetsaka Error", message, None)

    def progress(self, value: float | int) -> None:
        if self.feedback is not None and hasattr(self.feedback, "setProgress"):
            self.feedback.setProgress(int(value))
