"""Root-level compatibility shims for migrated QGIS processing algorithms."""

from __future__ import annotations

from .classify import ClassifyAlgorithm
from .explain_model import ExplainModelAlgorithm
from .nested_cv_algorithm import NestedCVAlgorithm
from .split_train_validation import SplitTrain
from .train import TrainAlgorithm

__all__ = [
    "ClassifyAlgorithm",
    "ExplainModelAlgorithm",
    "NestedCVAlgorithm",
    "SplitTrain",
    "TrainAlgorithm",
]
