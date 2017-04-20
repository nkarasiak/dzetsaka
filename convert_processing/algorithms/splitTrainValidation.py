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


from dzetsaka.scripts.function_vector import randomInSubset

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
#from processing.core.outputs import OutputFile
from processing.core.outputs import OutputVector
from PyQt4.QtGui import QMessageBox
#import sys
from qgis.core import QgsMessageLog

class splitTrainValidation(GeoAlgorithm):
    INPUT_LAYER = 'INPUT_LAYER'
    INPUT_COLUMN = 'INPUT_COLUMN'
    METHOD = "METHOD"
    METHOD_VALUES = ['Percent','Count value']
    VALUE = 'VALUE'
    OUTPUT_VALIDATION = "OUTPUT_VALIDATION"
    OUTPUT_TRAIN = "OUTPUT_TRAIN"
    
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/icon.png")
        
    def defineCharacteristics(self):

        # The name that the user will see in the toolbox
        self.name = 'Split train and validation within subsets'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Vector manipulation'	
        
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
            'Field (with int values)',
            self.INPUT_LAYER, optional=False)) # save model
        
        # Train algorithm
        self.addParameter(
        ParameterSelection(
        self.METHOD,"Select method (percent or number)",self.METHOD_VALUES, 0))
        
        # SPLIT %
        self.addParameter(
        ParameterNumber(
            self.VALUE,
            self.tr('Select 50 for 50% if PERCENT method. Else, value represents whole size of test sample.'),
            minValue=0,maxValue=9999,default=50))
            
        # SAVE AS
        # SAVE MODEL
        self.addOutput(
        OutputVector(
            self.OUTPUT_VALIDATION,
            self.tr("Output validation vector")))
            
        # SAVE CONFUSION MATRIX
        self.addOutput(
        OutputVector(
            self.OUTPUT_TRAIN,
            self.tr("Output training vector")))

    def checkParameterValuesBeforeExecuting(self):
        message = False
        """ GET VARIABLES """
        try:
            from sklearn.model_selection import train_test_split
        except:
            message = 'You must install scikit-learn library in Osgeo'
      
        if message:                
            #QgsMessageLog.logMessage('error is :'+str(message))
            return self.tr(message)
        else:
            pass
        
    def processAlgorithm(self, progress):

        INPUT_LAYER = self.getParameterValue(self.INPUT_LAYER)
        INPUT_COLUMN = self.getParameterValue(self.INPUT_COLUMN)
        OUTPUT_VALIDATION = self.getOutputValue(self.OUTPUT_VALIDATION)
        OUTPUT_TRAIN = self.getOutputValue(self.OUTPUT_TRAIN)
        VALUE = self.getParameterValue(self.VALUE)
        METHOD = self.getParameterValue(self.METHOD)
        
        
        # Retrieve algo from code
        selectedMETHOD  = self.METHOD_VALUES[METHOD]
        
        if selectedMETHOD == 'Percent' :
            percent = True
        else:
            percent = False
            
        #QgsMessageLog.logMessage(str(INPUT_LAYER)+' '+str(INPUT_COLUMN)+' '+str(OUTPUT_VALIDATION)+' '+str(OUTPUT_TRAIN)+' '+str(VALUE)+' '+str(percent))
        
        randomInSubset(INPUT_LAYER,str(INPUT_COLUMN),OUTPUT_VALIDATION,OUTPUT_TRAIN,VALUE,percent)
        