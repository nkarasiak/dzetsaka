# -*- coding: utf-8 -*-
"""
/***************************************************************************
 dzetsaka
                                 A QGIS plugin
 Fast and Easy Classification
                             -------------------
        begin                : 2016-05-13
        copyright            : (C) 2016 by Nicolaï Van Lennepkade
        email                : karasiak.nicolas@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load dzetsaka class from file dzetsaka.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .dzetsaka import dzetsaka
    return dzetsaka(iface)
