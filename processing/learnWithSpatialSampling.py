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
#from PyQt5.QtWidgets import QMessageBox
import numpy as np
from qgis.core import (QgsMessageLog,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterString,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterFileDestination)

import os

from ..scripts import mainfunction

pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

class trainSLOOAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_LAYER = 'INPUT_LAYER'
    INPUT_COLUMN = 'INPUT_COLUMN'
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = ['Random-Forest','K-Nearest Neighbors','Support Vector Machine']
    TRAIN_ALGORITHMS_CODE = ['RF','KNN','SVM']

    DISTANCE = "DISTANCE"
    MAXITER = "MAXITER"
    PARAMGRID = "PARAMGRID"
    MINTRAIN = "MINTRAIN"
    #SPLIT_PERCENT= 'SPLIT_PERCENT'
    OUTPUT_MODEL = "OUTPUT_MODEL"
    # OUTPUT_MATRIX = "OUTPUT_MATRIX"
    MAX_ITER = "MAX_ITER"
    SAVEDIR = "SAVEDIR"
    
    def shortHelpString(self):
        return self.tr("Spatial sampling to better learn and estimate prediction.. \n \n \
                       SLOO : Spatial Leave-One-Out Cross Validation. \n \
                       \
                       <h3>Classifier (paramgrid)</h3> \n \
                       Param grid can be fit for each algorithm : \n \
                       <h4>Random-Forest</h4> \n \
                       e.g. : dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier \n \
                       \n \
                       <h4>SVM</h4> \
                       e.g. : dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(-2,5)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html ")
        
    """
    def helpUrl(self):
        return "http://pot.readthedocs.io/en/stable/all.html#module-ot.da"
    """
    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Train algorithm (CV with SLOO)'
    
    def icon(self):

        return QIcon(os.path.join(pluginPath,'icon.png'))
        
    def initAlgorithm(self,config=None):

        # The name that the user will see in the toolbox
        
        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr('Input raster')
            )   
        )

        
        # Train algorithm
        
        self.addParameter(
        QgsProcessingParameterEnum(
        self.TRAIN,"Select algorithm to train",
        self.TRAIN_ALGORITHMS, 0))
        
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
                        
        # DISTANCE
        self.addParameter(
        QgsProcessingParameterNumber(
            self.DISTANCE,
            self.tr('Distance in pixels'),
            type=QgsProcessingParameterNumber.Integer,
            minValue=0,maxValue=99999,defaultValue=100))
        
        # MAX ITER
        self.addParameter(
        QgsProcessingParameterNumber(
            self.MAXITER,
            self.tr('Maximum iteration (default : 0 e.g. class with min effective)'),
            type=QgsProcessingParameterNumber.Integer,
            minValue=0,maxValue=99999,defaultValue=0))
        #
        self.addParameter(QgsProcessingParameterString(
                self.PARAMGRID,
                self.tr('Parameters for the hyperparameters of the algorithm'),
                optional=True))
        # SAVE AS
        # SAVE MODEL
        self.addParameter(
        QgsProcessingParameterFileDestination(
            self.OUTPUT_MODEL,
            self.tr("Output model (to use for classifying)")))
        """  
        # SAVE CONFUSION MATRIX
        self.addParameter(
        QgsProcessingParameterFileDestination(
            self.OUTPUT_MATRIX,
            self.tr("Output confusion matrix"),
            fileFilter='csv'))#,
            #ext='csv'))
        """    
        # SAVE DIR
        self.addParameter(
        QgsProcessingParameterFolderDestination(
            self.SAVEDIR,
            self.tr("Directory to save every confusion matrix")))
        
        
    def processAlgorithm(self, parameters,context,feedback):

        INPUT_RASTER = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        
        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        # SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)
        #TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
        #INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_MODEL = self.parameterAsFileOutput(parameters, self.OUTPUT_MODEL, context)
        #OUTPUT_MATRIX = self.parameterAsFileOutput(parameters, self.OUTPUT_MATRIX, context)
      
        SAVEDIR = self.parameterAsFileOutput(parameters, self.SAVEDIR, context)
        # Retrieve algo from code        
        extraParam = {}
        #extraParam['maxIter']=False
        #extraParam['param_grid'] = dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6))
        
        MAXITER = self.parameterAsInt(parameters, self.MAXITER, context)
        DISTANCE = self.parameterAsInt(parameters, self.DISTANCE, context)
        
        extraParam['distance'] = DISTANCE
        
        if MAXITER == 0:
            MAXITER = False
        extraParam['maxIter'] = MAXITER
           
        PARAMGRID = self.parameterAsString(parameters, self.PARAMGRID, context)
        if PARAMGRID != '':
            extraParam['param_grid'] = eval(PARAMGRID)
         
        
        if not SAVEDIR.endswith('/'):
            SAVEDIR += '/'
        
        extraParam['saveDir'] = SAVEDIR
        
        
        TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)

        # Retrieve algo from code        
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]
        
        #eval(PARAM_GRID)
               
        #QgsMessageLog.logMessage(str(eval(PARAMGRID)))
        
        mainfunction.learnModel(INPUT_RASTER.source(),INPUT_LAYER.source(),INPUT_COLUMN[0],OUTPUT_MODEL,'SLOO',0,None,SELECTED_ALGORITHM,feedback=feedback,extraParam=extraParam)
        return {self.SAVEDIR: SAVEDIR, self.OUTPUT_MODEL: OUTPUT_MODEL}
        
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return trainSLOOAlgorithm()
    
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
        return 'Classification tool'
