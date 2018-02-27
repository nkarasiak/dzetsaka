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

from qgis.PyQt.QtGui import QIcon
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QMessageBox

from qgis.core import (QgsMessageLog,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterRasterDestination)

import os

#from ..scripts.mainfunction import classifyImage


pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

class classifyAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_MASK = "INPUT_MASK"
    INPUT_MODEL = "INPUT_MODEL"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    
    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Predict model (classification map)'
    
    def icon(self):

        return QIcon(os.path.join(pluginPath,'img','icon.png'))
        
    def initAlgorithm(self,config=None):

        # inputs       
        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input raster')
            )   
        )

  
        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.INPUT_MASK,
                self.tr('Mask raster')
            )   
        )
        
        self.addParameter(
                QgsProcessingParameterFile(
                self.INPUT_MODEL,
                self.tr('Model learned')
            )   
        )
        
        # output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER,
                self.tr('Output raster')
            )
        )    
        
    def processAlgorithm(self, parameters,context,feedback):

        INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        INPUT_MASK = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context)         
        INPUT_MODEL = self.parameterAsFile(parameters, self.INPUT_MODEL, context)
    
        OUTPUT_RASTER = self.parameterAsRasterDestination(parameters, self.OUTPUT_RASTER, context)        
        # Retrieve algo from code        
        
        worker = classifyImage()
        #classify
        worker.initPredict(INPUT_RASTER.source(),INPUT_MODEL,OUTPUT_RASTER,INPUT_MASK)

        return {'Classification' : str(OUTPUT_RASTER)}

        
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return classifyAlgorithm()
    
    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self.name())
    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Classification tools'