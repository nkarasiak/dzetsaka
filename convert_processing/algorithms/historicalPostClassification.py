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


from dzetsaka.scripts.filters import filtersFunction

from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.parameters import ParameterVector
from processing.core.outputs import OutputRaster
from processing.core.parameters import ParameterTableField
from processing.core.parameters import ParameterBoolean
from processing.core.parameters import ParameterSelection
from processing.core.outputs import OutputFile
from processing.core.outputs import OutputVector
from qgis.core import QgsMessageLog

class historicalPostClassAlgorithm(GeoAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    SIZE_HA = 'SIZE_HA'
    CLASS_NUMBER = 'CLASS_NUMBER'
    OUTPUT_VECTOR = "OUTPUT_VECTOR"
    
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/iconHM.png")
        
    def defineCharacteristics(self):

        # The name that the user will see in the toolbox
        self.name = 'Post Classification'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Historical Map Process'	

        self.addParameter(
        ParameterRaster(
            self.INPUT_RASTER,
            self.tr('Input raster'),
            False))
        
        self.addParameter(
        ParameterNumber(
            self.SIZE_HA,
            self.tr('Sieve size (0.5 for 0.5ha)'),
            minValue=0,default=0.5)) 
            
        self.addParameter(
        ParameterNumber(
            self.CLASS_NUMBER,
            self.tr('Class number you want to keep'),
            minValue=0,default=1))
            
        self.addOutput(
        OutputVector(
            self.OUTPUT_VECTOR,
            self.tr("Final layer")))

    def processAlgorithm(self, progress):
        
        
       INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)     
       SIZE_HA = self.getParameterValue(self.SIZE_HA)
       CLASS_NUMBER = self.getParameterValue(self.CLASS_NUMBER)
       OUTPUT_VECTOR = self.getOutputValue(self.OUTPUT_VECTOR)
       
       
       # Get pixel Size
       from osgeo import gdal
       # convert ha to m²
       SIZE_HA = int(SIZE_HA*1000)
        
       datasrc=gdal.Open(INPUT_RASTER)
       pixelSize = datasrc.GetGeoTransform()[1] #get pixel size
       pixelSieve = int(SIZE_HA/(pixelSize*pixelSize)) #get number of pixel to sieve
       datasrc = None
       
       worker = filtersFunction()
       QgsMessageLog.logMessage('input : '+str(INPUT_RASTER))
       QgsMessageLog.logMessage('pixel Sieve : '+str(pixelSieve))
       QgsMessageLog.logMessage('class num : '+str(CLASS_NUMBER))
       QgsMessageLog.logMessage('output vector : '+str(OUTPUT_VECTOR))
       
       worker.historicalMapPostClass(INPUT_RASTER,int(pixelSieve),int(CLASS_NUMBER),OUTPUT_VECTOR)

        
        

        