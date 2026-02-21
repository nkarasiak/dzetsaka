"""Tests for QGIS best practices compliance."""

from __future__ import annotations

import os
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("qgis")
qgis_core = pytest.importorskip("qgis.core")
if not getattr(qgis_core, "QgsApplication", None) or not getattr(qgis_core, "QgsTask", None):
    pytest.skip("Real QGIS core API is unavailable", allow_module_level=True)


@pytest.mark.unit
def test_signal_disconnection_in_unload():
    """Test that all signals are properly disconnected during plugin unload."""
    from dzetsaka.qgis.unload_utils import unload_plugin

    # Create mock plugin
    plugin = Mock()
    plugin.pluginIsActive = True

    # Create mock widgets with signals
    plugin.dashboard_dock = Mock()
    plugin.dashboard_dock.classificationRequested = Mock()
    plugin.dashboard_dock.classificationRequested.disconnect = Mock()
    plugin.dashboard_dock.closingRequested = Mock()
    plugin.dashboard_dock.closingRequested.disconnect = Mock()
    plugin.dashboard_dock.close = Mock()

    plugin.dashboard_toolbar_action = Mock()
    plugin.dashboard_toolbar_action.triggered = Mock()
    plugin.dashboard_toolbar_action.triggered.disconnect = Mock()

    plugin.provider = Mock()
    plugin.actions = []
    plugin.iface = Mock()
    plugin.tr = Mock(return_value="&dzetsaka")

    # Mock QgsApplication
    with patch("dzetsaka.qgis.unload_utils.QgsApplication") as mock_qgs_app:
        mock_registry = Mock()
        mock_qgs_app.processingRegistry.return_value = mock_registry

        # Call unload
        unload_plugin(plugin)

        # Verify signals were disconnected
        plugin.dashboard_dock.classificationRequested.disconnect.assert_called_once()
        plugin.dashboard_dock.closingRequested.disconnect.assert_called_once()
        plugin.dashboard_toolbar_action.triggered.disconnect.assert_called_once()

        # Verify widgets were closed
        plugin.dashboard_dock.close.assert_called_once()

        # Verify cleanup
        assert plugin.pluginIsActive is False
        mock_registry.removeProvider.assert_called_once_with(plugin.provider)


@pytest.mark.unit
def test_signal_disconnection_handles_exceptions():
    """Test that signal disconnection handles missing signals gracefully."""
    from dzetsaka.qgis.unload_utils import unload_plugin

    # Create mock plugin with signals that raise exceptions
    plugin = Mock()
    plugin.pluginIsActive = True

    plugin.dashboard_dock = Mock()
    plugin.dashboard_dock.classificationRequested = Mock()
    plugin.dashboard_dock.classificationRequested.disconnect = Mock(side_effect=TypeError("Signal not connected"))
    plugin.dashboard_dock.closingRequested = Mock()
    plugin.dashboard_dock.closingRequested.disconnect = Mock(side_effect=RuntimeError("Signal already disconnected"))
    plugin.dashboard_dock.close = Mock()

    plugin.dashboard_toolbar_action = None  # Test None action

    plugin.provider = Mock()
    plugin.actions = []
    plugin.iface = Mock()
    plugin.tr = Mock(return_value="&dzetsaka")

    with patch("dzetsaka.qgis.unload_utils.QgsApplication"):
        # Should not raise exception
        unload_plugin(plugin)

        # Verify close was still called despite disconnect errors
        plugin.dashboard_dock.close.assert_called_once()


@pytest.mark.unit
def test_dependency_install_task_structure():
    """Test that DependencyInstallTask follows QgsTask pattern correctly."""
    from dzetsaka.qgis.dependency_install_task import DependencyInstallTask

    # Create mock logger
    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.error = Mock()
    mock_logger.warning = Mock()

    # Create task
    task = DependencyInstallTask(
        description="Test installation",
        packages=["test-package"],
        plugin_logger=mock_logger,
    )

    # Verify it has required QgsTask methods
    assert hasattr(task, "run"), "Task must have run() method"
    assert hasattr(task, "finished"), "Task must have finished() method"
    assert callable(task.run), "run() must be callable"
    assert callable(task.finished), "finished() must be callable"

    # Verify it has progress tracking
    assert hasattr(task, "setProgress"), "Task must support setProgress()"
    assert hasattr(task, "isCanceled"), "Task must support isCanceled()"


@pytest.mark.unit
def test_classification_task_checks_cancellation():
    """Test that ClassificationTask properly checks for cancellation."""
    from dzetsaka.qgis.task_runner import ClassificationTask

    # Create task with minimal parameters
    task = ClassificationTask(
        description="Test classification",
        do_training=False,
        raster_path="/tmp/test.tif",
        vector_path=None,
        class_field=None,
        model_path="/tmp/model.pkl",
        split_config=None,
        random_seed=0,
        matrix_path=None,
        classifier="1",
        output_path="/tmp/output.tif",
        mask_path=None,
        confidence_map=None,
        nodata=-9999,
        extra_params={},
        on_success=Mock(),
        on_error=Mock(),
    )

    # Mock isCanceled to return True
    task.isCanceled = Mock(return_value=True)

    # Run should return False when cancelled
    with patch("dzetsaka.qgis.task_runner.run_classification"):
        result = task.run()
        assert result is False, "Task should return False when cancelled"


@pytest.mark.unit
def test_async_dependency_installer_uses_callback():
    """Test that async dependency installer uses callback pattern."""
    from dzetsaka.qgis.dependency_installer import try_install_dependencies_async

    # Create mock plugin
    plugin = Mock()
    plugin.log = Mock()
    plugin.log.info = Mock()
    plugin.log.warning = Mock()
    plugin.iface = Mock()
    plugin.iface.mainWindow = Mock(return_value=Mock())

    callback_called = {"value": False}

    def on_complete(success):
        callback_called["value"] = True

    # Mock the task creation and submission
    with patch("dzetsaka.qgis.dependency_installer.DependencyInstallTask"), patch(
        "dzetsaka.qgis.dependency_installer.QgsApplication",
    ):
        try_install_dependencies_async(plugin, ["test-package"], on_complete)

        # Verify logger was called (task was created)
        assert plugin.log.info.called, "Logger should be called during setup"


@pytest.mark.unit
def test_runtime_constraints_fallbacks_to_imported_versions_when_metadata_missing():
    """Scientific stack versions should still be pinned when metadata is unavailable."""
    from dzetsaka.qgis.dependency_installer import _build_runtime_constraints_file

    plugin_logger = Mock()
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "numpy":
            return SimpleNamespace(__version__="1.26.4")
        if name == "scipy":
            return SimpleNamespace(__version__="1.11.4")
        if name == "pandas":
            return SimpleNamespace(__version__="2.2.3")
        return original_import(name, globals, locals, fromlist, level)

    with patch("importlib.metadata.version", side_effect=importlib_metadata.PackageNotFoundError):
        with patch("builtins.__import__", side_effect=fake_import):
            constraints_file, args = _build_runtime_constraints_file(plugin_logger)

    assert constraints_file is not None
    assert args == ["-c", constraints_file]

    with open(constraints_file, encoding="utf-8") as f:
        constraints = f.read()

    assert "numpy==1.26.4" in constraints
    assert "scipy==1.11.4" in constraints
    assert "pandas==2.2.3" in constraints

    os.remove(constraints_file)


@pytest.mark.unit
def test_dependency_install_task_finished_calls_on_finished_callback():
    """Test that DependencyInstallTask.finished() invokes the on_finished callback."""
    from dzetsaka.qgis.dependency_install_task import DependencyInstallTask

    mock_logger = Mock()
    callback_results = []

    task = DependencyInstallTask(
        description="Test installation",
        packages=["test-package"],
        plugin_logger=mock_logger,
        on_finished=lambda result: callback_results.append(result),
    )

    # Simulate successful completion
    task.finished(True)
    assert callback_results == [True]

    # Simulate failed completion
    task.finished(False)
    assert callback_results == [True, False]


@pytest.mark.unit
def test_dependency_install_task_finished_without_callback():
    """Test that DependencyInstallTask.finished() works when no callback is provided."""
    from dzetsaka.qgis.dependency_install_task import DependencyInstallTask

    mock_logger = Mock()

    task = DependencyInstallTask(
        description="Test installation",
        packages=["test-package"],
        plugin_logger=mock_logger,
    )

    # Should not raise when no callback is set
    task.finished(True)
    task.finished(False)
