# -*- coding: utf-8 -*-

"""
/***************************************************************************
 className
                                 A QGIS plugin
 description
                              -------------------
        begin                : 2016-12-03
        copyright            : (C) 2016 by Nico
        email                : nico@nico
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""


#import dzetsaka.scripts.function_dataraster as dataraster

from dzetsaka.scripts.filters import filtersFunction

from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber

from processing.core.outputs import OutputRaster

class historicalFilterAlgorithm(GeoAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_RASTER = "OUTPUT_RASTER"
    MEDIAN_SIZE = 'MEDIAN_SIZE'
    MEDIAN_ITER = 'MEDIAN_ITER'
    CLOSING_SIZE = 'CLOSING_SIZE'
        
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/iconHM.png")
        
    def defineCharacteristics(self):

        # The name that the user will see in the toolbox
        self.name = 'Filter map'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Historical Map Process'

        self.addParameter(
        ParameterRaster(
            self.INPUT_RASTER,
            self.tr('Input raster'),
            False))
    
        # add num
        self.addParameter(
        ParameterNumber(
            self.CLOSING_SIZE,
            self.tr('Window size of closing filter'),
            minValue=3,
            default=5))
        
        # add num
        self.addParameter(
        ParameterNumber(
            self.MEDIAN_SIZE,
            self.tr('Window size of median filter'),
            minValue=3,
            default=5))
        
        # add num
        self.addParameter(
        ParameterNumber(
            self.MEDIAN_ITER,
            self.tr('Number of iteration for median filter'),
            minValue=1,
            default=3))

        # We add a vector layer as output
        self.addOutput(
        OutputRaster(
            self.OUTPUT_RASTER,
            self.tr('Output raster')))


    def processAlgorithm(self, progress):


        INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_RASTER = self.getOutputValue(self.OUTPUT_RASTER)
        CLOSING_SIZE = self.getParameterValue(self.CLOSING_SIZE)
        MEDIAN_ITER = self.getParameterValue(self.MEDIAN_ITER)
        MEDIAN_SIZE = self.getParameterValue(self.MEDIAN_SIZE)

        worker = filtersFunction()
        worker.historicalMapFilter(INPUT_RASTER,OUTPUT_RASTER,CLOSING_SIZE,MEDIAN_SIZE,MEDIAN_ITER)
        
        

        