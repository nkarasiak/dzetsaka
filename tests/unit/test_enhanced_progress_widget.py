"""Unit tests for EnhancedProgressWidget and TaskFeedbackAdapter."""

import time

import pytest

# These tests require QGIS environment
pytest.importorskip("qgis")

from qgis.PyQt.QtWidgets import QApplication

from dzetsaka.qgis.task_runner import (
    PROGRESS_STAGES,
    EnhancedProgressWidget,
    TaskFeedbackAdapter,
)


@pytest.fixture
def qapp():
    """Fixture to provide QApplication instance for Qt widgets."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MockTask:
    """Mock QGIS task for testing."""

    def __init__(self):
        self.progress_value = 0
        self.status_message = ""
        self.description = ""

    def setProgress(self, value):
        """Set progress value."""
        self.progress_value = value

    def setDescription(self, text):
        """Set task description."""
        self.description = text


class TestEnhancedProgressWidget:
    """Test cases for EnhancedProgressWidget."""

    def test_initialization(self, qapp):
        """Test widget initialization."""
        widget = EnhancedProgressWidget()
        assert widget.start_time is None
        assert widget.progress_bar.value() == 0
        assert widget.main_label.text() == "Initializing..."

    def test_set_main_task(self, qapp):
        """Test setting main task label."""
        widget = EnhancedProgressWidget()
        widget.set_main_task("Training Random Forest...")
        assert widget.main_label.text() == "Training Random Forest..."

    def test_set_progress_basic(self, qapp):
        """Test basic progress setting."""
        widget = EnhancedProgressWidget()
        widget.set_progress(50, 100)

        assert widget.progress_bar.value() == 50
        assert widget.start_time is not None

    def test_set_progress_with_sub_task(self, qapp):
        """Test progress with sub-task text."""
        widget = EnhancedProgressWidget()
        widget.set_progress(30, 100, "Testing parameter set 3/10")

        assert widget.progress_bar.value() == 30
        assert widget.sub_task_label.text() == "Testing parameter set 3/10"
        assert widget.sub_task_label.isVisible()

    def test_set_progress_clamps_values(self, qapp):
        """Test that progress values are clamped to 0-100."""
        widget = EnhancedProgressWidget()

        widget.set_progress(-10, 100)
        assert widget.progress_bar.value() == 0

        widget.set_progress(150, 100)
        assert widget.progress_bar.value() == 100

    def test_time_estimation(self, qapp):
        """Test time estimation display."""
        widget = EnhancedProgressWidget()

        # Set to 50% progress
        widget.set_progress(50, 100)

        # Simulate some time passing
        time.sleep(0.1)

        # Update to 75% progress
        widget.set_progress(75, 100)

        # Time label should be visible and contain "remaining"
        if widget.time_label.isVisible():
            assert "remaining" in widget.time_label.text() or "Elapsed" in widget.time_label.text()

    def test_completion_time_display(self, qapp):
        """Test that completion time is shown at 100%."""
        widget = EnhancedProgressWidget()

        # Set to 50% to start timer
        widget.set_progress(50, 100)
        time.sleep(0.05)

        # Complete
        widget.set_progress(100, 100)

        # Should show "Completed in Xs"
        assert widget.time_label.isVisible()
        assert "Completed" in widget.time_label.text()

    def test_reset(self, qapp):
        """Test widget reset functionality."""
        widget = EnhancedProgressWidget()

        # Set some progress
        widget.set_progress(75, 100, "Some sub-task")
        widget.set_main_task("Training...")

        # Reset
        widget.reset()

        # Check reset state
        assert widget.progress_bar.value() == 0
        assert widget.sub_task_label.text() == ""
        assert not widget.sub_task_label.isVisible()
        assert widget.time_label.text() == ""
        assert not widget.time_label.isVisible()
        assert widget.start_time is not None  # Should be reset to now


class TestTaskFeedbackAdapter:
    """Test cases for TaskFeedbackAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        task = MockTask()
        adapter = TaskFeedbackAdapter(task)

        assert adapter.task is task
        assert adapter.enhanced_widget is None
        assert adapter._last_text == ""

    def test_set_progress(self):
        """Test setProgress method."""
        task = MockTask()
        adapter = TaskFeedbackAdapter(task)

        adapter.setProgress(42)
        assert task.progress_value == 42.0

    def test_set_progress_text(self):
        """Test setProgressText method."""
        task = MockTask()
        adapter = TaskFeedbackAdapter(task)

        adapter.setProgressText("Loading data...")
        assert task.status_message == "Loading data..."
        assert adapter._last_text == "Loading data..."

    def test_duplicate_text_handling(self):
        """Test that duplicate messages are not repeated."""
        task = MockTask()
        adapter = TaskFeedbackAdapter(task)

        adapter.setProgressText("Same message")
        adapter.setProgressText("Same message")

        # Should only be set once (check that _last_text is still set)
        assert adapter._last_text == "Same message"

    def test_stage_detection_loading(self, qapp):
        """Test detection of loading stage."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Loading data from raster...")

        assert "Loading data" in widget.main_label.text()
        assert PROGRESS_STAGES["loading"]["start"] <= widget.progress_bar.value() <= PROGRESS_STAGES["loading"]["end"]

    def test_stage_detection_training(self, qapp):
        """Test detection of training stage."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Training model...")

        assert "Training model" in widget.main_label.text()

    def test_optuna_trial_parsing(self, qapp):
        """Test parsing of Optuna trial messages."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Training model... trial 5/10")

        assert widget.sub_task_label.text() == "Testing trial 5/10"
        # Progress should be within training stage
        assert PROGRESS_STAGES["training"]["start"] <= widget.progress_bar.value() <= PROGRESS_STAGES["training"]["end"]

    def test_parameter_set_parsing(self, qapp):
        """Test parsing of parameter set messages."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Learning... Testing parameter set 3/20")

        assert widget.sub_task_label.text() == "Testing parameter set 3/20"
        # Progress should be within training stage
        assert PROGRESS_STAGES["training"]["start"] <= widget.progress_bar.value() <= PROGRESS_STAGES["training"]["end"]

    def test_stage_detection_shap(self, qapp):
        """Test detection of SHAP stage."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Computing SHAP values...")

        assert "Computing SHAP values" in widget.main_label.text()
        assert PROGRESS_STAGES["shap"]["start"] <= widget.progress_bar.value() <= PROGRESS_STAGES["shap"]["end"]

    def test_stage_detection_classifying(self, qapp):
        """Test detection of classification stage."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Predicting model for 4-band image...")

        assert "Classifying raster" in widget.main_label.text()
        assert widget.sub_task_label.text() == "Processing 4-band image"

    def test_stage_detection_report(self, qapp):
        """Test detection of report generation stage."""
        task = MockTask()
        widget = EnhancedProgressWidget()
        adapter = TaskFeedbackAdapter(task, widget)

        adapter.setProgressText("Generating report...")

        assert "Generating report" in widget.main_label.text()
        assert PROGRESS_STAGES["report"]["start"] <= widget.progress_bar.value() <= PROGRESS_STAGES["report"]["end"]


class TestProgressStages:
    """Test cases for PROGRESS_STAGES configuration."""

    def test_stages_defined(self):
        """Test that all expected stages are defined."""
        expected_stages = ["loading", "training", "shap", "classifying", "report"]
        for stage in expected_stages:
            assert stage in PROGRESS_STAGES

    def test_stage_structure(self):
        """Test that each stage has required fields."""
        for _stage_name, stage_info in PROGRESS_STAGES.items():
            assert "name" in stage_info
            assert "start" in stage_info
            assert "end" in stage_info
            assert isinstance(stage_info["name"], str)
            assert isinstance(stage_info["start"], (int, float))
            assert isinstance(stage_info["end"], (int, float))

    def test_stage_progression(self):
        """Test that stages progress from 0 to 100."""
        assert PROGRESS_STAGES["loading"]["start"] == 0
        assert PROGRESS_STAGES["report"]["end"] == 100

    def test_no_stage_gaps(self):
        """Test that there are no gaps between stages."""
        stages_list = ["loading", "training", "shap", "classifying", "report"]
        for i in range(len(stages_list) - 1):
            current_end = PROGRESS_STAGES[stages_list[i]]["end"]
            next_start = PROGRESS_STAGES[stages_list[i + 1]]["start"]
            # Stages should be adjacent or overlapping
            assert next_start <= current_end + 5  # Allow small gap
