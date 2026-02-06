"""Dzetsaka Processing Provider for QGIS Processing Framework.

Phase 1 implementation: provider logic is hosted in the new architecture path
while still relying on existing processing algorithm modules.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from dzetsaka.presentation.qgis.processing.algorithms.classify import ClassifyAlgorithm
from dzetsaka.processing.split_train_validation import SplitTrain
from dzetsaka.processing.train import TrainAlgorithm

try:
    from dzetsaka.processing.explain_model import ExplainModelAlgorithm

    EXPLAIN_MODEL_AVAILABLE = True
except ImportError:
    EXPLAIN_MODEL_AVAILABLE = False

try:
    from dzetsaka.processing.nested_cv_algorithm import NestedCVAlgorithm

    NESTED_CV_ALGORITHM_AVAILABLE = True
except ImportError:
    NESTED_CV_ALGORITHM_AVAILABLE = False


class DzetsakaProvider(QgsProcessingProvider):
    """Processing provider for dzetsaka algorithms."""

    def __init__(self, providerType="Standard"):
        super().__init__()
        self.providerType = providerType

    def icon(self):
        """Add provider icon."""
        here = Path(__file__).resolve()
        repo_icon = here.parents[4] / "icon.png"
        asset_icon = here.parents[2] / "assets" / "icons" / "icon.png"
        icon_path = repo_icon if repo_icon.exists() else asset_icon
        return QIcon(str(icon_path))

    def unload(self):
        """Unload provider."""

    def loadAlgorithms(self):
        """Load algorithms exposed by this provider."""
        self.addAlgorithm(TrainAlgorithm())
        self.addAlgorithm(ClassifyAlgorithm())
        self.addAlgorithm(SplitTrain())

        if EXPLAIN_MODEL_AVAILABLE:
            self.addAlgorithm(ExplainModelAlgorithm())

        if NESTED_CV_ALGORITHM_AVAILABLE:
            self.addAlgorithm(NestedCVAlgorithm())

        if self.providerType == "Experimental":
            from dzetsaka.processing.closing_filter import ClosingFilterAlgorithm
            from dzetsaka.processing.domain_adaptation import DomainAdaptation
            from dzetsaka.processing.median_filter import MedianFilterAlgorithm
            from dzetsaka.processing.shannon_entropy import ShannonAlgorithm

            self.addAlgorithm(ClosingFilterAlgorithm())
            self.addAlgorithm(MedianFilterAlgorithm())
            self.addAlgorithm(DomainAdaptation())
            self.addAlgorithm(ShannonAlgorithm())

    def id(self):
        """Return the unique provider id."""
        return "dzetsaka"

    def name(self):
        """Return the provider name."""
        return self.tr("dzetsaka")

    def longName(self):
        """Return the longer version of the provider name."""
        return self.name()


def get_provider_class():
    """Factory helper used by compatibility shims."""
    return DzetsakaProvider
