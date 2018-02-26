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


from builtins import str

from qgis.PyQt.QtGui import QIcon
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QMessageBox

from qgis.core import (QgsMessageLog,
                       
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFileDestination)

import os
from ..scripts import function_dataraster as dataraster
from ..scripts import mainfunction

pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

class trainAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_LAYER = 'INPUT_LAYER'
    INPUT_COLUMN = 'INPUT_COLUMN'
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = ['Gaussian Mixture Model','Random-Forest','K-Nearest Neighbors','Support Vector Machine']
    TRAIN_ALGORITHMS_CODE = ['GMM','RF','KNN','SVM']
    SPLIT_PERCENT= 'SPLIT_PERCENT'
    OUTPUT_MODEL = "OUTPUT_MODEL"
    OUTPUT_MATRIX = "OUTPUT_MATRIX"
    
    
    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'train algorithm'
    
    def icon(self):

        return QIcon(os.path.join(pluginPath,'img','icon.png'))
        
    def initAlgorithm(self,config=None):

        # The name that the user will see in the toolbox
        
        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input raster')
            )   
        )

        # ROI
        # VECTOR
        self.addParameter(
        QgsProcessingParameterVectorLayer(
            self.INPUT_LAYER,
            'Input layer',
            ))
        # TABLE / COLUMN 
        self.addParameter(
        QgsProcessingParameterField(
            self.INPUT_COLUMN,
            'Field (column must have classification number (e.g. \'1\' forest, \'2\' water...))',
            parentLayerParameterName = self.INPUT_LAYER,
            optional=False)) # save model
        
        # Train algorithm
        
        self.addParameter(
        QgsProcessingParameterEnum(
        self.TRAIN,"Select algorithm to train",
        self.TRAIN_ALGORITHMS, 0))
        
        # SPLIT %
        
        self.addParameter(
        QgsProcessingParameterNumber(
            self.SPLIT_PERCENT,
            self.tr('Pixels (0.5 for 50%) to keep for classification'),
            type=QgsProcessingParameterNumber.Integer,
            minValue=0,maxValue=100,defaultValue=50))
        
        # SAVE AS
        # SAVE MODEL
        self.addParameter(
        QgsProcessingParameterFileDestination(
            self.OUTPUT_MODEL,
            self.tr("Output model (to use for classifying)")))
            
        # SAVE CONFUSION MATRIX
        self.addParameter(
        QgsProcessingParameterFileDestination(
            self.OUTPUT_MATRIX,
            self.tr("Output confusion matrix"),
            fileFilter='csv'))#,
            #ext='csv'))
        
        
    def processAlgorithm(self, parameters,context,feedback):

        INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        INPUT_RASTER_src = INPUT_RASTER.source()

        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        INPUT_LAYER_src = INPUT_LAYER.source()
        
        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)
        TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
        #INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_MODEL = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
        OUTPUT_MATRIX = self.parameterAsFileOutput(parameters, self.OUTPUT_MATRIX, context)


        # Retrieve algo from code        
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]
        QgsMessageLog.logMessage(str(SELECTED_ALGORITHM))        
       
        
        libOk = True
        
        if SELECTED_ALGORITHM=='RF' or SELECTED_ALGORITHM=='SVM' or SELECTED_ALGORITHM=='KNN':
            try:
                import sklearn
            except:
                libOk = False
                
        # learn model
        if libOk:
            mainfunction.learnModel(INPUT_RASTER.source(),INPUT_LAYER.source(),INPUT_COLUMN[0],OUTPUT_MODEL,SPLIT_PERCENT,0,OUTPUT_MATRIX,SELECTED_ALGORITHM)
        else:
            QMessageBox.information(None, "Please install scikit-learn library to use:", str(SELECTED_ALGORITHM)) 
        
        return {str(OUTPUT_MATRIX) : str(OUTPUT_MODEL)}

        
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return trainAlgorithm()
    
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
        return 'algoGroup'