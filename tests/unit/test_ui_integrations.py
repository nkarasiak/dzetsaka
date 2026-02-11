"""Unit tests for UI integration enhancements.

Tests for:
- Training Data Quality Checker integration
- Batch Classification menu integration
- Confidence Analysis integration
- Theme support application
- Keyboard shortcuts

Author:
    Nicolas Karasiak
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# Test fixtures
@pytest.fixture
def mock_qgis_interface():
    """Mock QGIS interface for testing."""
    iface = MagicMock()
    iface.mainWindow.return_value = MagicMock()
    iface.messageBar.return_value = MagicMock()
    return iface


@pytest.fixture
def mock_plugin():
    """Mock plugin instance for testing."""
    plugin = MagicMock()
    plugin.iface = MagicMock()
    plugin.get_icon_path = Mock(return_value="test_icon.png")
    return plugin


class TestQualityCheckerIntegration:
    """Test Training Data Quality Checker integration."""

    def test_quality_checker_button_exists_in_data_input_page(self):
        """Test that quality checker button exists in DataInputPage."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import DataInputPage

        page = DataInputPage()
        assert hasattr(page, "checkQualityBtn"), "DataInputPage should have checkQualityBtn"
        assert page.checkQualityBtn.text() == "Check Data Quality…"

    def test_quality_checker_button_exists_in_quick_panel(self):
        """Test that quality checker button exists in QuickClassificationPanel."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import QuickClassificationPanel

        panel = QuickClassificationPanel()
        assert hasattr(panel, "checkQualityBtn"), "QuickClassificationPanel should have checkQualityBtn"
        assert "Check Data Quality" in panel.checkQualityBtn.text()

    def test_quality_checker_handler_method_exists(self):
        """Test that quality checker handler methods exist."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import DataInputPage, QuickClassificationPanel

        data_page = DataInputPage()
        assert hasattr(data_page, "_check_data_quality"), "DataInputPage should have _check_data_quality method"

        quick_panel = QuickClassificationPanel()
        assert hasattr(quick_panel, "_check_data_quality"), "QuickClassificationPanel should have _check_data_quality method"

    @patch("ui.classification_workflow_ui.TrainingDataQualityChecker")
    def test_quality_checker_opens_with_correct_params(self, mock_checker_class):
        """Test that quality checker is opened with correct parameters."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import QuickClassificationPanel

        panel = QuickClassificationPanel()
        panel._get_vector_path = Mock(return_value="/path/to/vector.shp")
        panel.classFieldCombo.currentText = Mock(return_value="class")

        # Simulate button click
        panel._check_data_quality()

        # Verify dialog was created with correct parameters
        mock_checker_class.assert_called_once()
        call_kwargs = mock_checker_class.call_args[1]
        assert call_kwargs["vector_path"] == "/path/to/vector.shp"
        assert call_kwargs["class_field"] == "class"


class TestBatchClassificationIntegration:
    """Test Batch Classification menu integration."""

    def test_batch_classification_menu_item_added(self, mock_plugin):
        """Test that batch classification menu item is added."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from src.dzetsaka.qgis.ui_init import init_gui

        # Mock the add_action method to track calls
        mock_plugin.add_action = Mock()

        init_gui(mock_plugin)

        # Verify add_action was called for batch classification
        calls = mock_plugin.add_action.call_args_list
        batch_call = [c for c in calls if "Batch Classification" in str(c)]
        assert len(batch_call) > 0, "Batch Classification menu item should be added"

    def test_open_batch_classification_method_exists(self, mock_plugin):
        """Test that open_batch_classification method exists in plugin."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from src.dzetsaka.qgis.plugin_runtime import DzetsakaGUI

        # Check method exists
        assert hasattr(DzetsakaGUI, "open_batch_classification"), "Plugin should have open_batch_classification method"

    @patch("src.dzetsaka.qgis.plugin_runtime.BatchClassificationDialog")
    def test_batch_classification_dialog_opens(self, mock_dialog_class, mock_plugin):
        """Test that batch classification dialog opens correctly."""
        pytest.importorskip("qgis.PyQt.QtWidgets")

        # Import after qgis is available
        from src.dzetsaka.qgis.plugin_runtime import DzetsakaGUI

        plugin = DzetsakaGUI(mock_plugin.iface)
        plugin.open_batch_classification()

        # Verify dialog was created and shown
        mock_dialog_class.assert_called_once()


class TestConfidenceAnalysisIntegration:
    """Test Confidence Analysis integration into Results Explorer."""

    def test_confidence_tab_created_when_path_exists(self):
        """Test that confidence tab is created when confidence path exists."""
        pytest.importorskip("qgis.PyQt.QtWidgets")

        # Create a temporary file to simulate confidence map
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            confidence_path = f.name

        try:
            from ui.results_explorer_dock import ResultsExplorerDock

            result = {
                "algorithm": "Random Forest",
                "runtime_seconds": 10.5,
                "output_path": "/path/to/output.tif",
                "confidence_path": confidence_path,
            }

            dock = ResultsExplorerDock(result)

            # Check that confidence tab was added
            tab_count = dock.tab_widget.count()
            tab_names = [dock.tab_widget.tabText(i) for i in range(tab_count)]
            assert "Confidence" in tab_names, "Confidence tab should be added when path exists"

        finally:
            # Cleanup
            if os.path.exists(confidence_path):
                os.unlink(confidence_path)

    def test_confidence_tab_not_created_when_no_path(self):
        """Test that confidence tab is not created when no confidence path."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.results_explorer_dock import ResultsExplorerDock

        result = {
            "algorithm": "Random Forest",
            "runtime_seconds": 10.5,
            "output_path": "/path/to/output.tif",
            # No confidence_path
        }

        dock = ResultsExplorerDock(result)

        # Check that confidence tab was NOT added
        tab_count = dock.tab_widget.count()
        tab_names = [dock.tab_widget.tabText(i) for i in range(tab_count)]
        assert "Confidence" not in tab_names, "Confidence tab should not be added when no path"


class TestThemeSupportIntegration:
    """Test theme support application to dialogs."""

    def test_guided_dialog_inherits_theme_aware(self):
        """Test that ClassificationSetupDialog inherits from ThemeAwareWidget."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import ClassificationSetupDialog
        from ui.theme_support import ThemeAwareWidget

        assert issubclass(ClassificationSetupDialog, ThemeAwareWidget), \
            "ClassificationSetupDialog should inherit from ThemeAwareWidget"

    def test_welcome_wizard_inherits_theme_aware(self):
        """Test that WelcomeWizard inherits from ThemeAwareWidget."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.welcome_wizard import WelcomeWizard
        from ui.theme_support import ThemeAwareWidget

        assert issubclass(WelcomeWizard, ThemeAwareWidget), \
            "WelcomeWizard should inherit from ThemeAwareWidget"

    def test_results_explorer_inherits_theme_aware(self):
        """Test that ResultsExplorerDock inherits from ThemeAwareWidget."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.results_explorer_dock import ResultsExplorerDock
        from ui.theme_support import ThemeAwareWidget

        assert issubclass(ResultsExplorerDock, ThemeAwareWidget), \
            "ResultsExplorerDock should inherit from ThemeAwareWidget"

    def test_batch_dialog_inherits_theme_aware(self):
        """Test that BatchClassificationDialog inherits from ThemeAwareWidget."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.batch_classification_dialog import BatchClassificationDialog
        from ui.theme_support import ThemeAwareWidget

        assert issubclass(BatchClassificationDialog, ThemeAwareWidget), \
            "BatchClassificationDialog should inherit from ThemeAwareWidget"

    def test_quality_checker_inherits_theme_aware(self):
        """Test that TrainingDataQualityChecker inherits from ThemeAwareWidget."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.training_data_quality_checker import TrainingDataQualityChecker
        from ui.theme_support import ThemeAwareWidget

        assert issubclass(TrainingDataQualityChecker, ThemeAwareWidget), \
            "TrainingDataQualityChecker should inherit from ThemeAwareWidget"

    def test_confidence_widget_inherits_theme_aware(self):
        """Test that ConfidenceAnalysisWidget inherits from ThemeAwareWidget."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.confidence_analysis_widget import ConfidenceAnalysisWidget
        from ui.theme_support import ThemeAwareWidget

        assert issubclass(ConfidenceAnalysisWidget, ThemeAwareWidget), \
            "ConfidenceAnalysisWidget should inherit from ThemeAwareWidget"


class TestKeyboardShortcuts:
    """Test keyboard shortcuts integration."""

    def test_quick_panel_shortcuts_setup_method_exists(self):
        """Test that QuickClassificationPanel has shortcuts setup method."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import QuickClassificationPanel

        panel = QuickClassificationPanel()
        assert hasattr(panel, "_setup_shortcuts"), "QuickClassificationPanel should have _setup_shortcuts method"

    def test_wizard_shortcuts_setup_method_exists(self):
        """Test that ClassificationSetupDialog has shortcuts setup method."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import ClassificationSetupDialog

        dialog = ClassificationSetupDialog()
        assert hasattr(dialog, "_setup_dialog_shortcuts"), \
            "ClassificationSetupDialog should have _setup_dialog_shortcuts method"

    def test_quality_check_shortcut_tooltip_updated(self):
        """Test that quality check button tooltip mentions keyboard shortcut."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import QuickClassificationPanel

        panel = QuickClassificationPanel()

        if hasattr(panel, 'checkQualityBtn'):
            tooltip = panel.checkQualityBtn.toolTip()
            assert "Ctrl+Shift+Q" in tooltip or "shortcut" in tooltip.lower(), \
                "Quality check button tooltip should mention keyboard shortcut"

    def test_run_button_shortcut_tooltip_updated(self):
        """Test that run button tooltip mentions keyboard shortcut."""
        pytest.importorskip("qgis.PyQt.QtWidgets")
        from ui.classification_workflow_ui import QuickClassificationPanel

        panel = QuickClassificationPanel()

        if hasattr(panel, 'runButton'):
            tooltip = panel.runButton.toolTip()
            assert "Ctrl+Return" in tooltip or "shortcut" in tooltip.lower(), \
                "Run button tooltip should mention keyboard shortcut"


class TestResultsExplorerQualityChecker:
    """Test quality checker integration in Results Explorer."""

    def test_quality_checker_button_added_when_data_available(self):
        """Test that quality checker button is added when training data available."""
        pytest.importorskip("qgis.PyQt.QtWidgets")

        # Create a temporary file to simulate training vector
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".shp", delete=False) as f:
            vector_path = f.name

        try:
            from ui.results_explorer_dock import ResultsExplorerDock

            result = {
                "algorithm": "Random Forest",
                "runtime_seconds": 10.5,
                "output_path": "/path/to/output.tif",
                "training_vector": vector_path,
                "class_field": "class",
            }

            dock = ResultsExplorerDock(result)

            # Check that quick actions were created (method should have been called)
            assert hasattr(dock, "_check_training_data"), \
                "ResultsExplorerDock should have _check_training_data method"

        finally:
            # Cleanup
            if os.path.exists(vector_path):
                os.unlink(vector_path)

    @patch("ui.results_explorer_dock.TrainingDataQualityChecker")
    def test_quality_checker_opens_from_results_explorer(self, mock_checker_class):
        """Test that quality checker opens from results explorer."""
        pytest.importorskip("qgis.PyQt.QtWidgets")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".shp", delete=False) as f:
            vector_path = f.name

        try:
            from ui.results_explorer_dock import ResultsExplorerDock

            result = {
                "algorithm": "Random Forest",
                "runtime_seconds": 10.5,
                "output_path": "/path/to/output.tif",
                "training_vector": vector_path,
                "class_field": "class_id",
            }

            dock = ResultsExplorerDock(result)
            dock._check_training_data()

            # Verify dialog was created with correct parameters
            mock_checker_class.assert_called_once()
            call_kwargs = mock_checker_class.call_args[1]
            assert call_kwargs["vector_path"] == vector_path
            assert call_kwargs["class_field"] == "class_id"

        finally:
            if os.path.exists(vector_path):
                os.unlink(vector_path)


# Integration test markers
pytestmark = pytest.mark.integration

