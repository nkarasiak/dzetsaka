"""Compatibility shim for the migrated processing provider runtime."""

from __future__ import annotations

from pathlib import Path

from dzetsaka.services.runtime_loader import load_module_from_path


def _load_provider_class():
    provider_path = (
        Path(__file__).resolve().parent / "src" / "dzetsaka" / "presentation" / "qgis" / "provider.py"
    )
    module = load_module_from_path("_dzetsaka_provider_runtime", provider_path)
    runtime_cls = getattr(module, "DzetsakaProvider", None)
    if runtime_cls is None:
        raise ImportError("DzetsakaProvider class not found in migrated provider runtime")
    return runtime_cls


_provider_cls = None
_provider_error = None


def _get_provider_class():
    global _provider_cls, _provider_error
    if _provider_cls is not None:
        return _provider_cls
    if _provider_error is not None:
        raise _provider_error
    try:
        _provider_cls = _load_provider_class()
        return _provider_cls
    except Exception as exc:  # pragma: no cover - depends on QGIS runtime
        _provider_error = exc
        raise


class DzetsakaProvider:
    """Lazy proxy to migrated provider class."""

    def __new__(cls, *args, **kwargs):
        runtime_cls = _get_provider_class()
        return runtime_cls(*args, **kwargs)
