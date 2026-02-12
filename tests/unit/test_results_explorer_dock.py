"""Unit tests for results explorer dock."""

import pytest


@pytest.mark.qgis
def test_results_explorer_import():
    """Test that ResultsExplorerDock can be imported."""
    pytest.importorskip("qgis")
    from ui.results_explorer_dock import ResultsExplorerDock, open_results_explorer

    assert ResultsExplorerDock is not None
    assert open_results_explorer is not None


@pytest.mark.qgis
def test_build_result_data():
    """Test result data building helper."""
    pytest.importorskip("qgis")
    from datetime import datetime

    from dzetsaka.qgis.task_launcher import _build_result_data

    result = _build_result_data(
        classifier="RF",
        output_path="/path/to/output.tif",
        input_path="/path/to/input.tif",
        matrix_path="/path/to/matrix.csv",
        model_path="/path/to/model.pkl",
        confidence_path="/path/to/confidence.tif",
        extra_params={},
        start_time=datetime.now(),
    )

    assert result["algorithm"] == "Random Forest"
    assert result["output_path"] == "/path/to/output.tif"
    assert result["input_path"] == "/path/to/input.tif"
    assert result["model_path"] == "/path/to/model.pkl"
    assert "runtime_seconds" in result
    assert "timestamp" in result
    assert result["runtime_seconds"] >= 0


@pytest.mark.qgis
def test_result_data_no_optional_fields():
    """Test result data building with no optional fields."""
    pytest.importorskip("qgis")
    from dzetsaka.qgis.task_launcher import _build_result_data

    result = _build_result_data(
        classifier="GMM",
        output_path="/path/to/output.tif",
        input_path="/path/to/input.tif",
        matrix_path=None,
        model_path="/path/to/model.pkl",
        confidence_path=None,
        extra_params={},
        start_time=None,
    )

    assert result["algorithm"] == "Gaussian Mixture Model"
    assert "matrix_path" not in result
    assert "confidence_path" not in result
    assert result["runtime_seconds"] == 0.0


@pytest.mark.qgis
def test_result_data_with_report_dir(tmp_path):
    """Test result data building with report directory."""
    pytest.importorskip("qgis")

    from dzetsaka.qgis.task_launcher import _build_result_data

    # Create fake report directory
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "shap_importance.png").touch()
    (report_dir / "classification_report.html").touch()

    result = _build_result_data(
        classifier="XGB",
        output_path="/path/to/output.tif",
        input_path="/path/to/input.tif",
        matrix_path=None,
        model_path="/path/to/model.pkl",
        confidence_path=None,
        extra_params={"REPORT_OUTPUT_DIR": str(report_dir)},
        start_time=None,
    )

    assert "shap_path" in result
    assert "shap_importance.png" in result["shap_path"]
    assert "report_path" in result
    assert "classification_report.html" in result["report_path"]


@pytest.mark.qgis
def test_results_explorer_dock_creation():
    """Test creating results explorer dock with minimal data."""
    pytest.importorskip("qgis")
    from ui.results_explorer_dock import ResultsExplorerDock

    result_data = {
        "algorithm": "Random Forest",
        "runtime_seconds": 42.5,
        "output_path": "/path/to/output.tif",
        "input_path": "/path/to/input.tif",
        "timestamp": "2024-01-01 12:00:00",
        "class_counts": {},
    }

    dock = ResultsExplorerDock(result_data, parent=None, iface=None)
    assert dock.windowTitle() == "Classification Results"
    assert dock.result == result_data
