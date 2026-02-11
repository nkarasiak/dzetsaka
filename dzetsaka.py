"""Compatibility shim for the migrated QGIS plugin runtime.

Canonical runtime implementation lives in:
`src/dzetsaka/presentation/qgis/plugin_runtime.py`.
"""

from __future__ import annotations

from pathlib import Path

from dzetsaka.services.runtime_loader import load_module_from_path

def _load_runtime_module():
    runtime_path = (
        Path(__file__).resolve().parent / "src" / "dzetsaka" / "qgis" / "plugin_runtime.py"
    )
    return load_module_from_path("_dzetsaka_plugin_runtime", runtime_path)


_runtime = None
_runtime_error = None


def _get_runtime_module():
    """Load and cache runtime module on demand."""
    global _runtime, _runtime_error
    if _runtime is not None:
        return _runtime
    if _runtime_error is not None:
        raise _runtime_error
    try:
        _runtime = _load_runtime_module()
        return _runtime
    except Exception as exc:  # pragma: no cover - depends on QGIS runtime
        _runtime_error = exc
        raise


class DzetsakaGUI:
    """Lazy proxy to runtime DzetsakaGUI."""

    def __new__(cls, *args, **kwargs):
        runtime_cls = _get_runtime_module().DzetsakaGUI
        return runtime_cls(*args, **kwargs)


class ClassificationTask:
    """Lazy proxy to runtime ClassificationTask."""

    def __new__(cls, *args, **kwargs):
        runtime_cls = _get_runtime_module().ClassificationTask
        return runtime_cls(*args, **kwargs)


def __getattr__(name):
    """Expose optional runtime symbols lazily for compatibility."""
    if name in {"_TaskFeedbackAdapter", "_LEFT_DOCK_AREA"}:
        runtime = _get_runtime_module()
        if hasattr(runtime, name):
            return getattr(runtime, name)
    raise AttributeError(name)
