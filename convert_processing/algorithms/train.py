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


import dzetsaka.scripts.mainfunction as mainfunction

from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.parameters import ParameterVector
#from processing.core.outputs import OutputRaster
from processing.core.parameters import ParameterTableField
#from processing.core.parameters import ParameterBoolean
from processing.core.parameters import ParameterSelection
from processing.core.outputs import OutputFile
from PyQt4.QtGui import QMessageBox
#import sys

class trainAlgorithm(GeoAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_LAYER = 'INPUT_LAYER'
    INPUT_COLUMN = 'INPUT_COLUMN'
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = ['Gaussian Mixture Model','Random-Forest','K-Nearest Neighbors','Support Vector Machine']
    TRAIN_ALGORITHMS_CODE = ['GMM','RF','KNN','SVM']
    SPLIT_PERCENT= 'SPLIT_PERCENT'
    OUTPUT_MODEL = "OUTPUT_MODEL"
    OUTPUT_MATRIX = "OUTPUT_MATRIX"
    
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/icon.png")
        
    def defineCharacteristics(self):

        # The name that the user will see in the toolbox
        self.name = 'Train algorithm'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Learning and classification'	


        self.addParameter(
        ParameterRaster(
            self.INPUT_RASTER,
            self.tr('Input raster'),
            False))
        
        # ROI
        # VECTOR
        self.addParameter(
        ParameterVector(
            self.INPUT_LAYER,
            'Input layer',
            [ParameterVector.VECTOR_TYPE_ANY], False))
            
        # TABLE / COLUMN 
        self.addParameter(
        ParameterTableField(
            self.INPUT_COLUMN,
            'Field (column must have classification number (e.g. \'1\' forest, \'2\' water...))',
            self.INPUT_LAYER, optional=False)) # save model
        
        # Train algorithm
        self.addParameter(
        ParameterSelection(
        self.TRAIN,"Select algorithm to train",self.TRAIN_ALGORITHMS, 0))
        
        # SPLIT %
        self.addParameter(
        ParameterNumber(
            self.SPLIT_PERCENT,
            self.tr('Pixels (0.5 for 50%) to keep for classification'),
            minValue=0,maxValue=1,default=0.5))
            
        # SAVE AS
        # SAVE MODEL
        self.addOutput(
        OutputFile(
            self.OUTPUT_MODEL,
            self.tr("Output model (to use for classifying)")))
            
        # SAVE CONFUSION MATRIX
        self.addOutput(
        OutputFile(
            self.OUTPUT_MATRIX,
            self.tr("Output confusion matrix"),
            ext='csv'))

    def processAlgorithm(self, progress):

        INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        INPUT_LAYER = self.getParameterValue(self.INPUT_LAYER)
        INPUT_COLUMN = self.getParameterValue(self.INPUT_COLUMN)
        OUTPUT_MODEL = self.getOutputValue(self.OUTPUT_MODEL)
        OUTPUT_MATRIX = self.getOutputValue(self.OUTPUT_MATRIX)
        SPLIT_PERCENT = self.getParameterValue(self.SPLIT_PERCENT)
        TRAIN = self.getParameterValue(self.TRAIN)
        
        # Retrieve algo from code
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN]
        
        libOk = True
        
        if SELECTED_ALGORITHM=='RF' or SELECTED_ALGORITHM=='SVM' or SELECTED_ALGORITHM=='KNN':
            try:
                import sklearn
            except:
                libOk = False
                
        # learn model
        if libOk:
            mainfunction.learnModel(INPUT_RASTER,INPUT_LAYER,INPUT_COLUMN,OUTPUT_MODEL,SPLIT_PERCENT,0,OUTPUT_MATRIX,SELECTED_ALGORITHM)
        else:
            QMessageBox.information(None, "Please install scikit-learn library to use:", str(SELECTED_ALGORITHM)) 

        
        

        