"""Tests for _run_async_install dispatch logic.

Uses importlib to load classification_workflow_ui.py directly (bypassing
ui/__init__.py) with auto-stubbing QGIS/Qt modules.
"""

import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# Stub infrastructure
# ---------------------------------------------------------------------------

_WORKFLOW_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ui", "classification_workflow_ui.py"),
)


class _StubMeta(type):
    """Metaclass that auto-creates sub-stubs for any missing attribute."""

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Return a new stub class so chained access (e.g. Qt.ItemDataRole.UserRole) works.
        sub = _StubMeta(name, (), {})
        setattr(cls, name, sub)
        return sub


class _StubBase(metaclass=_StubMeta):
    """Usable as any Qt class, enum, signal, constant, etc."""

    def __init__(self, *a, **kw):
        pass

    def __call__(self, *a, **kw):
        return _StubBase()

    def __bool__(self):
        return True

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __int__(self):
        return 0


class _AutoStubModule(ModuleType):
    """Module that auto-creates _StubBase for any missing attribute."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        stub = _StubMeta(name, (_StubBase,), {})
        object.__setattr__(self, name, stub)
        return stub


# Keys we inject into sys.modules; track originals for cleanup.
_STUB_KEYS: list[str] = []
_originals: dict[str, ModuleType | None] = {}


def _inject(name: str, mod: ModuleType) -> None:
    _STUB_KEYS.append(name)
    _originals[name] = sys.modules.get(name)
    sys.modules[name] = mod


def _cleanup() -> None:
    for name in _STUB_KEYS:
        prev = _originals.get(name)
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev
    _STUB_KEYS.clear()
    _originals.clear()


# -- Build stubs --
_qgis = _AutoStubModule("qgis")
_pyqt = _AutoStubModule("qgis.PyQt")
_qtcore = _AutoStubModule("qgis.PyQt.QtCore")
_qtgui = _AutoStubModule("qgis.PyQt.QtGui")
_qtwidgets = _AutoStubModule("qgis.PyQt.QtWidgets")
_core = _AutoStubModule("qgis.core")
_gui = _AutoStubModule("qgis.gui")
_utils = _AutoStubModule("qgis.utils")

_pyqt.QtCore = _qtcore
_pyqt.QtGui = _qtgui
_pyqt.QtWidgets = _qtwidgets
_qgis.PyQt = _pyqt
_qgis.core = _core
_qgis.gui = _gui
_qgis.utils = _utils

for _name, _mod in [
    ("qgis", _qgis),
    ("qgis.PyQt", _pyqt),
    ("qgis.PyQt.QtCore", _qtcore),
    ("qgis.PyQt.QtGui", _qtgui),
    ("qgis.PyQt.QtWidgets", _qtwidgets),
    ("qgis.core", _core),
    ("qgis.gui", _gui),
    ("qgis.utils", _utils),
]:
    _inject(_name, _mod)

# Relative-import stubs for the ui package.
_ui_pkg = _AutoStubModule("ui")
_ui_pkg.__path__ = [os.path.dirname(_WORKFLOW_PATH)]
_inject("ui", _ui_pkg)

for _sub in ("validated_widgets", "theme_support", "training_data_quality_checker"):
    _inject(f"ui.{_sub}", _AutoStubModule(f"ui.{_sub}"))

# ---------------------------------------------------------------------------
# Load the module under test.
# ---------------------------------------------------------------------------
MODULE_AVAILABLE = False
_run_async_install = None

try:
    _spec = importlib.util.spec_from_file_location(
        "ui.classification_workflow_ui",
        _WORKFLOW_PATH,
        submodule_search_locations=[],
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["ui.classification_workflow_ui"] = _mod
    _spec.loader.exec_module(_mod)
    _run_async_install = _mod._run_async_install
    MODULE_AVAILABLE = True
except Exception:
    import traceback

    traceback.print_exc()
finally:
    sys.modules.pop("ui.classification_workflow_ui", None)
    sys.modules.pop("ui", None)
    for _sub in ("validated_widgets", "theme_support", "training_data_quality_checker"):
        sys.modules.pop(f"ui.{_sub}", None)
    _cleanup()

pytestmark = pytest.mark.skipif(
    not MODULE_AVAILABLE,
    reason="classification_workflow_ui not importable without QGIS",
)


class TestRunAsyncInstall:
    """Verify that _run_async_install dispatches to the correct installer method."""

    PACKAGES = ["scikit-learn", "xgboost"]

    # -- Branch 1: async installer available --

    def test_prefers_async_installer(self):
        installer = Mock(spec=["_try_install_dependencies_async"])
        callback = Mock()

        _run_async_install(installer, self.PACKAGES, on_complete=callback)

        installer._try_install_dependencies_async.assert_called_once_with(self.PACKAGES, on_complete=callback)

    def test_async_installer_without_callback(self):
        installer = Mock(spec=["_try_install_dependencies_async"])

        _run_async_install(installer, self.PACKAGES, on_complete=None)

        installer._try_install_dependencies_async.assert_called_once_with(self.PACKAGES, on_complete=None)

    def test_async_takes_precedence_over_sync(self):
        """When both methods exist, async is preferred."""
        installer = Mock(spec=["_try_install_dependencies_async", "_try_install_dependencies"])
        callback = Mock()

        _run_async_install(installer, self.PACKAGES, on_complete=callback)

        installer._try_install_dependencies_async.assert_called_once()
        installer._try_install_dependencies.assert_not_called()

    # -- Branch 2: legacy sync installer fallback --

    def test_falls_back_to_sync_installer(self):
        installer = Mock(spec=["_try_install_dependencies"])
        installer._try_install_dependencies.return_value = True
        callback = Mock()

        _run_async_install(installer, self.PACKAGES, on_complete=callback)

        installer._try_install_dependencies.assert_called_once_with(self.PACKAGES)
        callback.assert_called_once_with(True)

    def test_sync_installer_forwards_failure(self):
        installer = Mock(spec=["_try_install_dependencies"])
        installer._try_install_dependencies.return_value = False
        callback = Mock()

        _run_async_install(installer, self.PACKAGES, on_complete=callback)

        callback.assert_called_once_with(False)

    def test_sync_installer_without_callback(self):
        installer = Mock(spec=["_try_install_dependencies"])
        installer._try_install_dependencies.return_value = True

        # Should not raise even though on_complete is None.
        _run_async_install(installer, self.PACKAGES, on_complete=None)

        installer._try_install_dependencies.assert_called_once_with(self.PACKAGES)

    # -- Branch 3: no installer method available --

    def test_no_installer_calls_callback_with_false(self):
        installer = Mock(spec=[])  # no relevant methods
        callback = Mock()

        _run_async_install(installer, self.PACKAGES, on_complete=callback)

        callback.assert_called_once_with(False)

    def test_no_installer_no_callback(self):
        installer = Mock(spec=[])

        # Should not raise.
        _run_async_install(installer, self.PACKAGES, on_complete=None)
