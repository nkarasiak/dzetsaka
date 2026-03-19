"""Unit tests for accuracy index module.

Tests confusion matrix computation, overall accuracy, and F1 scores
using both ConfusionMatrix (from label arrays) and StatsFromConfusionMatrix (from matrix).
"""

import numpy as np
import pytest

from scripts.accuracy_index import ConfusionMatrix, StatsFromConfusionMatrix


class TestConfusionMatrixPerfect:
    """Test ConfusionMatrix with perfect predictions."""

    def test_perfect_predictions_oa(self):
        """Test that perfect predictions yield OA of 1.0."""
        yp = np.array([1, 2, 3, 1, 2, 3])
        yr = np.array([1, 2, 3, 1, 2, 3])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.OA == pytest.approx(1.0)

    def test_perfect_predictions_f1mean(self):
        """Test that perfect predictions yield F1mean of 1.0."""
        yp = np.array([1, 2, 1, 2])
        yr = np.array([1, 2, 1, 2])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.F1mean == pytest.approx(1.0)


class TestConfusionMatrixTwoClass:
    """Test ConfusionMatrix with a 2-class scenario."""

    def test_two_class_known_oa(self):
        """Test OA for a known 2-class confusion scenario.

        Predictions: [1,1,1,1,2,2,2,2,2,2]
        Reference:   [1,1,1,2,2,2,2,2,1,1]
        Correct: positions 0,1,2 (class 1) and 4,5,6,7 (class 2) = 7/10 = 0.7
        """
        yp = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        yr = np.array([1, 1, 1, 2, 2, 2, 2, 2, 1, 1])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.OA == pytest.approx(0.7)

    def test_two_class_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape for 2-class problem."""
        yp = np.array([1, 1, 2, 2])
        yr = np.array([1, 2, 1, 2])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.confusion_matrix.shape == (2, 2)


class TestConfusionMatrixMultiClass:
    """Test ConfusionMatrix with a multi-class scenario."""

    def test_three_class_oa(self):
        """Test OA for a 3-class scenario with known correct count."""
        # 6 samples, 4 correct -> OA = 4/6
        yp = np.array([1, 2, 3, 1, 2, 3])
        yr = np.array([1, 2, 3, 2, 2, 1])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.OA == pytest.approx(4.0 / 6.0)

    def test_three_class_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape for 3-class problem."""
        yp = np.array([1, 2, 3, 1])
        yr = np.array([1, 2, 3, 3])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.confusion_matrix.shape == (3, 3)


class TestStatsFromConfusionMatrixPerfect:
    """Test StatsFromConfusionMatrix with a perfect diagonal matrix."""

    def test_diagonal_matrix_oa(self):
        """Test that a diagonal confusion matrix yields OA of 1.0."""
        mat = np.diag([10, 20, 15])
        stats = StatsFromConfusionMatrix(mat)

        assert stats.OA == pytest.approx(1.0)

    def test_diagonal_matrix_all_f1_are_one(self):
        """Test that a diagonal confusion matrix yields per-class F1 all equal to 1.0."""
        mat = np.diag([10, 20, 15])
        stats = StatsFromConfusionMatrix(mat)

        for f1 in stats.F1:
            assert f1 == pytest.approx(1.0)


class TestStatsFromConfusionMatrixKnown:
    """Test StatsFromConfusionMatrix with known off-diagonal values."""

    def test_known_matrix_oa(self):
        """Test OA from a known confusion matrix.

        Matrix:
            [[50, 5],
             [10, 35]]
        Total = 100, correct = 85, OA = 0.85
        """
        mat = np.array([[50, 5], [10, 35]])
        stats = StatsFromConfusionMatrix(mat)

        assert stats.OA == pytest.approx(0.85)

    def test_known_matrix_f1mean(self):
        """Test F1mean from a known confusion matrix."""
        mat = np.array([[50, 5], [10, 35]])
        stats = StatsFromConfusionMatrix(mat)

        # F1 class 0: 2*50 / (55+60) = 100/115
        # F1 class 1: 2*35 / (45+40) = 70/85
        expected_f1mean = np.mean([100.0 / 115.0, 70.0 / 85.0])
        assert stats.F1mean == pytest.approx(expected_f1mean, abs=1e-4)

    def test_per_class_f1_correct_length(self):
        """Test that per-class F1 list has one entry per class."""
        mat = np.array([[30, 5, 2], [3, 40, 1], [2, 4, 25]])
        stats = StatsFromConfusionMatrix(mat)

        assert len(stats.F1) == 3


class TestStatsFromConfusionMatrixTwoClass:
    """Test StatsFromConfusionMatrix with a 2-class scenario."""

    def test_two_class_metrics_in_valid_range(self):
        """Test that OA and F1mean are in valid ranges for 2-class case."""
        mat = np.array([[40, 10], [5, 45]])
        stats = StatsFromConfusionMatrix(mat)

        assert 0.0 <= stats.OA <= 1.0
        assert 0.0 <= stats.F1mean <= 1.0


class TestConsistencyBetweenClasses:
    """Test that ConfusionMatrix and StatsFromConfusionMatrix agree."""

    def test_same_metrics_for_same_data(self):
        """Test that both classes produce the same OA and F1mean."""
        yp = np.array([1, 1, 2, 2, 3, 3, 1, 2])
        yr = np.array([1, 2, 2, 3, 3, 3, 1, 1])

        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        stats = StatsFromConfusionMatrix(cm.confusion_matrix)

        assert cm.OA == pytest.approx(stats.OA)
        assert cm.F1mean == pytest.approx(stats.F1mean)


class TestSingleClassEdgeCase:
    """Test edge case with a single class (1x1 confusion matrix)."""

    def test_single_class_oa(self):
        """Test that a single-class prediction yields OA of 1.0."""
        yp = np.array([1, 1, 1, 1])
        yr = np.array([1, 1, 1, 1])
        cm = ConfusionMatrix()
        cm.compute_confusion_matrix(yp, yr)

        assert cm.OA == pytest.approx(1.0)

    def test_single_class_stats_from_matrix(self):
        """Test StatsFromConfusionMatrix with a 1x1 matrix."""
        mat = np.array([[10]])
        stats = StatsFromConfusionMatrix(mat)

        assert stats.OA == pytest.approx(1.0)
        assert len(stats.F1) == 1
        assert stats.F1[0] == pytest.approx(1.0)
