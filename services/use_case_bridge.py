"""Bridge to application use-case modules."""

from __future__ import annotations

from dzetsaka.application.use_cases.classify_raster import run_classification as run_classification_use_case
from dzetsaka.application.use_cases.train_model import run_training as run_training_use_case


def run_training(**kwargs):
    """Execute model training via the train-model use case."""
    return run_training_use_case(**kwargs)


def run_classification(**kwargs):
    """Execute inference via the classify-raster use case."""
    return run_classification_use_case(**kwargs)
