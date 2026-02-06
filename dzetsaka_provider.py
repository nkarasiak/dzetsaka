"""Compatibility bridge for Dzetsaka QGIS processing provider.

This module is kept for backward compatibility while provider implementation is
migrated to `src/dzetsaka/presentation/qgis/provider.py`.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_new_provider_class():
    """Load DzetsakaProvider from new architecture path, if available."""
    root_dir = Path(__file__).resolve().parent
    provider_path = root_dir / "src" / "dzetsaka" / "presentation" / "qgis" / "provider.py"
    if not provider_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("_dzetsaka_new_provider", provider_path)
    if spec is None or spec.loader is None:
        return None

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "DzetsakaProvider", None)


DzetsakaProvider = _load_new_provider_class()

if DzetsakaProvider is None:
    from qgis.core import QgsProcessingProvider
    from qgis.PyQt.QtGui import QIcon

    from .processing.classify import ClassifyAlgorithm
    from .processing.split_train_validation import SplitTrain
    from .processing.train import TrainAlgorithm

    try:
        from .processing.explain_model import ExplainModelAlgorithm

        EXPLAIN_MODEL_AVAILABLE = True
    except ImportError:
        EXPLAIN_MODEL_AVAILABLE = False

    try:
        from .processing.nested_cv_algorithm import NestedCVAlgorithm

        NESTED_CV_ALGORITHM_AVAILABLE = True
    except ImportError:
        NESTED_CV_ALGORITHM_AVAILABLE = False

    plugin_path = os.path.dirname(__file__)

    class DzetsakaProvider(QgsProcessingProvider):
        """Processing provider for dzetsaka algorithms."""

        def __init__(self, providerType="Standard"):
            super().__init__()
            self.providerType = providerType

        def icon(self):
            iconPath = os.path.join(plugin_path, "icon.png")
            return QIcon(os.path.join(iconPath))

        def unload(self):
            """Unload provider."""

        def loadAlgorithms(self):
            self.addAlgorithm(TrainAlgorithm())
            self.addAlgorithm(ClassifyAlgorithm())
            self.addAlgorithm(SplitTrain())

            if EXPLAIN_MODEL_AVAILABLE:
                self.addAlgorithm(ExplainModelAlgorithm())

            if NESTED_CV_ALGORITHM_AVAILABLE:
                self.addAlgorithm(NestedCVAlgorithm())

            if self.providerType == "Experimental":
                from .processing.closing_filter import ClosingFilterAlgorithm
                from .processing.median_filter import MedianFilterAlgorithm

                self.addAlgorithm(ClosingFilterAlgorithm())
                self.addAlgorithm(MedianFilterAlgorithm())

                from .processing.domain_adaptation import DomainAdaptation
                from .processing.shannon_entropy import ShannonAlgorithm

                self.addAlgorithm(DomainAdaptation())
                self.addAlgorithm(ShannonAlgorithm())

        def id(self):
            return "dzetsaka"

        def name(self):
            return self.tr("dzetsaka")

        def longName(self):
            return self.name()

