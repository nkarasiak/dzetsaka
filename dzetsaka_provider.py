"""Dzetsaka Processing Provider for QGIS Processing Framework.

This module provides the processing provider that registers dzetsaka algorithms
with QGIS Processing framework, making them available in the Processing Toolbox.
"""

__author__ = "Nicolas Karasiak"
__date__ = "2018-02-24"
__copyright__ = "(C) 2018 by Nicolas Karasiak"

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = "$Format:%H$"
import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .processing.classify import ClassifyAlgorithm
from .processing.split_train_validation import splitTrain

# from .moduleName_algorithm import classNameAlgorithm
# from .processing.moduleName_algorithm import classNameAlgorithm
from .processing.train import trainAlgorithm

plugin_path = os.path.dirname(__file__)

"""
import sys
sys.setrecursionlimit(10000) # 10000 is an example, try with different values
"""


class DzetsakaProvider(QgsProcessingProvider):
    """Processing provider for dzetsaka algorithms.

    This class registers dzetsaka's machine learning algorithms with the
    QGIS Processing framework, making them available in the Processing Toolbox.
    """

    def __init__(self, providerType="Standard"):
        """Initialize the dzetsaka processing provider.

        Parameters
        ----------
        providerType : str
            Type of provider (default: "Standard")

        """
        QgsProcessingProvider.__init__(self)

        # Load algorithms
        # ,learnWithSpatialSampling()]#,ClassifyAlgorithm(),splitTrain()]
        self.providerType = providerType

    def icon(self):
        """Add icon."""
        iconPath = os.path.join(plugin_path, "icon.png")

        return QIcon(os.path.join(iconPath))

    def unload(self):
        """Unload the provider.

        Any tear-down steps required by the provider should be implemented here.
        """

    def loadAlgorithms(self):
        """Loads all algorithms belonging to this provider."""
        self.addAlgorithm(trainAlgorithm())
        self.addAlgorithm(ClassifyAlgorithm())
        self.addAlgorithm(splitTrain())
        # self.addAlgorithm(trainSTANDalgorithm())

        if self.providerType == "Experimental":
            from .processing.closing_filter import ClosingFilterAlgorithm
            from .processing.median_filter import MedianFilterAlgorithm

            self.addAlgorithm(ClosingFilterAlgorithm())
            self.addAlgorithm(MedianFilterAlgorithm())

            from .processing.domain_adaptation import DomainAdaptation
            from .processing.shannon_entropy import shannonAlgorithm

            self.addAlgorithm(DomainAdaptation())
            self.addAlgorithm(shannonAlgorithm())

    def id(self):
        """Return the unique provider id.

        Used for identifying the provider. This string should be a unique, short,
        character only string, eg "qgis" or "gdal". This string should not be localised.
        """
        return "dzetsaka"

    def name(self):
        """Return the provider name.

        Used to describe the provider within the GUI.
        This string should be short (e.g. "Lastools") and localised.
        """
        return self.tr("dzetsaka")

    def longName(self):
        """Return the longer version of the provider name.

        Can include extra details such as version numbers. E.g. "Lastools LIDAR tools
        (version 2.2.1)". This string should be localised. The default
        implementation returns the same string as name().
        """
        return self.name()
