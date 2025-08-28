"""Dzetsaka QGIS Plugin Initialization.

This module initializes the dzetsaka plugin for QGIS, making it available in the
QGIS plugin system. Dzetsaka is a powerful classification plugin supporting 11
machine learning algorithms for remote sensing image classification.

The plugin provides:
- 11 classification algorithms (GMM, RF, SVM, KNN, XGB, LGB, ET, GBC, LR, NB, MLP)
- Advanced accuracy assessment with confusion matrices
- Interactive training sample collection
- Batch processing capabilities
- Export functionality for various formats

Author:
    Nicolas Karasiak <karasiak.nicolas@gmail.com>

License:
    GNU General Public License v2 or later (GPLv2+)

Created:
    2018-02-24
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load DzetsakaGUI class from file classNameGUI.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .dzetsaka import DzetsakaGUI

    return DzetsakaGUI(iface)
