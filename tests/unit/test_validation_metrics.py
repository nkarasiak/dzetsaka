"""Unit tests for validation metrics module.

Tests the ValidationMetrics class and standalone metric functions
for classification evaluation.
"""

import numpy as np
import pytest

# Try to import the module under test
try:
    from scripts.validation.metrics import (
        SKLEARN_AVAILABLE,
        ValidationMetrics,
        compute_multiclass_roc_auc,
        create_classification_summary,
    )

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    SKLEARN_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not MODULE_AVAILABLE, reason="Validation metrics module not available"),
    pytest.mark.sklearn,
]


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestComputePerClassMetricsPerfect:
    """Test compute_per_class_metrics with perfect predictions."""

    def test_perfect_predictions_accuracy(self):
        """Test that perfect predictions yield accuracy of 1.0."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        result = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
        assert result["overall"]["accuracy"] == 1.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestComputePerClassMetricsStructure:
    """Test the return structure of compute_per_class_metrics."""

    def test_return_has_required_keys(self):
        """Test that result dict contains all required top-level keys."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
        assert "overall" in result
        assert "per_class" in result
        assert "confusion_matrix" in result
        assert "classification_report" in result

    def test_overall_metrics_present(self):
        """Test that overall dict contains accuracy, macro_f1, weighted_f1."""
        y_true = np.array([0, 1, 0, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        result = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
        overall = result["overall"]
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            assert key in overall, f"Missing key: {key}"
            assert isinstance(overall[key], float)

    def test_per_class_has_expected_keys(self):
        """Test that each per-class entry has precision, recall, f1, support."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
        for class_name, metrics in result["per_class"].items():
            for key in ("precision", "recall", "f1", "support"):
                assert key in metrics, f"Missing key '{key}' for class '{class_name}'"

    def test_with_class_names_parameter(self):
        """Test that custom class_names are used as per-class dict keys."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        names = ["Water", "Forest"]
        result = ValidationMetrics.compute_per_class_metrics(y_true, y_pred, class_names=names)
        assert "Water" in result["per_class"]
        assert "Forest" in result["per_class"]

    def test_multiclass_three_plus_classes(self):
        """Test compute_per_class_metrics with 4 classes."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 1, 1, 2, 0])
        result = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
        assert len(result["per_class"]) == 4
        assert result["overall"]["accuracy"] == pytest.approx(0.75)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestPlotRocCurves:
    """Test ROC curve plotting for binary and multiclass cases."""

    def test_binary_roc_returns_auc_and_creates_file(self, tmp_path):
        """Test binary ROC: returns AUC dict with key 1, creates output file."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)

        output_file = tmp_path / "roc_binary.png"
        auc_scores = ValidationMetrics.plot_roc_curves(
            y, y_proba, ["Neg", "Pos"], str(output_file)
        )

        assert output_file.exists()
        assert 1 in auc_scores
        assert 0.0 <= auc_scores[1] <= 1.0
        # Trained on same data, AUC should be high
        assert auc_scores[1] > 0.9

    def test_multiclass_roc_returns_auc_per_class(self, tmp_path):
        """Test multiclass ROC: returns AUC dict with one entry per class."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(
            n_samples=300, n_features=10, n_classes=3,
            n_informative=6, random_state=42,
        )
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)

        output_file = tmp_path / "roc_multi.png"
        auc_scores = ValidationMetrics.plot_roc_curves(
            y, y_proba, ["A", "B", "C"], str(output_file)
        )

        assert output_file.exists()
        assert len(auc_scores) == 3
        for cls_id, auc_val in auc_scores.items():
            assert 0.0 <= auc_val <= 1.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestCreateClassificationSummary:
    """Test the create_classification_summary function."""

    def test_returns_string_with_expected_sections(self):
        """Test that summary is a string containing key section headers."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        summary = create_classification_summary(y_true, y_pred)
        assert isinstance(summary, str)
        assert "CLASSIFICATION PERFORMANCE SUMMARY" in summary
        assert "Overall Metrics" in summary
        assert "Per-Class Metrics" in summary
        assert "Accuracy" in summary


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestComputeMulticlassRocAuc:
    """Test the compute_multiclass_roc_auc standalone function."""

    def test_known_probabilities(self):
        """Test ROC AUC with near-perfect probabilities gives high score."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(
            n_samples=300, n_features=10, n_classes=3,
            n_informative=6, random_state=42,
        )
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)

        auc_score = compute_multiclass_roc_auc(y, y_proba, average="macro")
        assert isinstance(auc_score, float)
        assert 0.0 <= auc_score <= 1.0
        # Trained on same data, should be high
        assert auc_score > 0.9
