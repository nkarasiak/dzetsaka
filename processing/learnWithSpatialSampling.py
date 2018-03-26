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
#from PyQt5.QtWidgets import QMessageBox

from qgis.core import (QgsMessageLog,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterField,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterString,
                       QgsProcessingParameterRasterDestination)

import os
from ..scripts import domainAdaptation as DA

from ..scripts import function_dataraster as dataraster

pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

class learnWithSpatialSampling(QgsProcessingAlgorithm):
    SOURCE_RASTER = 'SOURCE_RASTER'
    SOURCE_LAYER = 'SOURCE_LAYER'
    SOURCE_COLUMN = 'SOURCE_COLUMN'
    TARGET_RASTER = 'TARGET_RASTER'
    TARGET_LAYER = 'TARGET_LAYER'
    TARGET_COLUMN = 'TARGET_COLUMN'

    MASK = 'MASK'
    
    PARAMS = 'PARAMS'
    
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = ['Earth Mover\'s Distance','Sinkhorn algorithm','Sinkhorn algorithm + l1 class regularization','Sinkhorn algorithm + l1l2 class regularization']
    TRAIN_ALGORITHMS_CODE = ['EMDTransport','SinkhornTransport','SinkhornLpl1Transport','SinkhornL1l2Transport']
    
    TRANSPORTED_IMAGE = 'TRANSPORTED_IMAGE'
    
    
    
    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Learn with spatial sampling'
    
    def icon(self):

        return QIcon(os.path.join(pluginPath,'icon.png'))
        
    def initAlgorithm(self,config=None):

        # The name that the user will see in the toolbox
        # Raster
        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.SOURCE_RASTER,
                self.tr('Source raster')
            )   
        )

        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.TARGET_RASTER,
                self.tr('Target raster')
            )   
        )

        # ROI SOURCE
        self.addParameter(
        QgsProcessingParameterVectorLayer(
            self.SOURCE_LAYER,
            'Source layer',
            ))
        
        # TABLE / COLUMN 
        self.addParameter(
        QgsProcessingParameterField(
            self.SOURCE_COLUMN,
            'Source field (column must have classification number (e.g. \'1\' forest, \'2\' water...))',
            parentLayerParameterName = self.SOURCE_LAYER,
            optional=False)) # save model
        
        # ROI TARGET (Optional)
    
        self.addParameter(
        QgsProcessingParameterVectorLayer(
            self.TARGET_LAYER,
            'Target layer', optional=False
            ))
        # TABLE / COLUMN 
        self.addParameter(
        QgsProcessingParameterField(
            self.TARGET_COLUMN,
            'Optional : Target field',
            parentLayerParameterName = self.TARGET_LAYER,
            optional=True)) # save model
        
        
        # Mask
        self.addParameter(
                QgsProcessingParameterRasterLayer(
                self.MASK,
                self.tr('Mask image (0 to mask)'),
                optional=True
            )   
        )
        
        self.addParameter(
        QgsProcessingParameterEnum(
        self.TRAIN,"Select algorithm to transport",
        self.TRAIN_ALGORITHMS, 0))
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.TRANSPORTED_IMAGE,
                self.tr('Transported image')
            )
        )    
      
        self.addParameter(
        QgsProcessingParameterString(
                self.PARAMS,
                self.tr('Parameters for the algorithm'),
                defaultValue='norm="loglog", metric="sqeuclidean"')
        )
        

    def processAlgorithm(self, parameters,context,feedback):

        SOURCE_RASTER = self.parameterAsRasterLayer(parameters, self.SOURCE_RASTER, context)
        SOURCE_LAYER = self.parameterAsVectorLayer(parameters, self.SOURCE_LAYER, context)
        SOURCE_COLUMN = self.parameterAsFields(parameters, self.SOURCE_COLUMN, context)
        
        TARGET_RASTER = self.parameterAsRasterLayer(parameters, self.TARGET_RASTER, context)
        TARGET_LAYER = self.parameterAsVectorLayer(parameters, self.TARGET_LAYER, context)
        TARGET_COLUMN = self.parameterAsFields(parameters, self.TARGET_COLUMN, context)
        
        TRANSPORTED_IMAGE = self.parameterAsOutputLayer(parameters, self.TRANSPORTED_IMAGE, context)

        
        TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
        #INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        
        MASK = self.parameterAsRasterLayer(parameters, self.MASK, context)
        
        PARAMS = self.parameterAsString(parameters, self.PARAMS, context)

        # Retrieve algo from code        
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]
       
        
        libOk = True

        
        try:
            import pyplot
        except:
            libOk = False
            msg = 'Please install pyplot library : "pip install pyplot"'
        # learn model
        if libOk :
            extraParam={}
            extraParam['minTrain'] = 0.5
            extraParam['maxIter']=False
            extraParam['distance'] = 500
            if classifier == 'RF':
                extraParam['param_grid'] = dict(n_estimators=2**sp.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6))
                extraParam['feature_importances_by_class'] = True

            inSplit = 'SLOO',extraParam=extraParam
            if classifier == 'RF':
                if 'feature_importances_by_class' in extraParam.keys():
                    for idx,method in enumerate(['mean','std','amin','amax']):
                        sp.savetxt(extraParam['saveDir']+'featimp_'+level+'_'+method+'.csv',trainmodel.feature_importances_[idx,:,:],delimiter=',',fmt='%.6g')
                elif 'feature_importances' in extraParam.keys():
                    sp.savetxt(extraParam['saveDir']+'featimp_'+level+'.csv',trainmodel.feature_importances_,delimiter=',',fmt='%.6g')
            
            return {'Transported image' : str(TRANSPORTED_IMAGE)}
        

        else:
            return {'Error' : msg}
            #QMessageBox.about(None, "Missing library", "Please install scikit-learn library to use"+str(SELECTED_ALGORITHM))        

        
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return learnWithSpatialSampling()
    
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
