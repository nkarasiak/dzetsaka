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

from dzetsaka.scripts.mainfunction import classifyImage

from qgis.PyQt.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.outputs import OutputRaster
from processing.core.parameters import ParameterFile
from qgis.core import QgsMessageLog

class classifyAlgorithm(GeoAlgorithm):

    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_MASK = "INPUT_MASK"
    INPUT_MODEL = "INPUT_MODEL"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/icon.png")
        
    def defineCharacteristics(self):
        """Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # The name that the user will see in the toolbox
        self.name = 'Classify model'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Learning and classification'	

        self.addParameter(
        ParameterRaster(
            self.INPUT_RASTER,
            self.tr('Input raster'),
            False))

        self.addParameter(
        ParameterRaster(
            self.INPUT_MASK,
            self.tr('Input mask'),
            True))
        
        self.addParameter(
        ParameterFile(
            self.INPUT_MODEL,
            self.tr('Input model'),
            False))

        self.addOutput(
        OutputRaster(
            self.OUTPUT_RASTER,
            self.tr('Output raster (classification)')))

    def processAlgorithm(self, progress):
        """Here is where the processing itself takes place."""

        INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        INPUT_MODEL = self.getParameterValue(self.INPUT_MODEL)
        INPUT_MASK = self.getParameterValue(self.INPUT_MASK)
        OUTPUT_RASTER = self.getOutputValue(self.OUTPUT_RASTER)
        
        worker = classifyImage()
        #classify
        worker.initPredict(INPUT_RASTER,INPUT_MODEL,OUTPUT_RASTER,INPUT_MASK)
        
        

        