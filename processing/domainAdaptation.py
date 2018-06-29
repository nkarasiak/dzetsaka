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

from qgis.core import (QgsProcessingAlgorithm,
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

class domainAdaptation(QgsProcessingAlgorithm):
    SOURCE_RASTER = 'SOURCE_RASTER'
    SOURCE_LAYER = 'SOURCE_LAYER'
    SOURCE_COLUMN = 'SOURCE_COLUMN'
    TARGET_RASTER = 'TARGET_RASTER'
    TARGET_LAYER = 'TARGET_LAYER'
    TARGET_COLUMN = 'TARGET_COLUMN'

    MASK = 'MASK'
    
    PARAMS = 'PARAMS'
    
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = ['Mapping Transport','Earth Mover\'s Distance','Sinkhorn Algorithm','Sinkhorn algorithm + l1 class regularization','Sinkhorn algorithm + l1l2 class regularization']
    TRAIN_ALGORITHMS_CODE = ['MappingTransport','EMDTransport','SinkhornTransport','SinkhornLpl1Transport','SinkhornL1l2Transport']
    
    TRANSPORTED_IMAGE = 'TRANSPORTED_IMAGE'
    
    
    
    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Domain Adaptation'
    
    def shortHelpString(self):
        return self.tr("Domain Adaptation for raster images using Python Optimal Transport library. <br>\
                       Help can be found on Python Optimal Transport documentation : http://pot.readthedocs.io/en/stable/all.html#module-ot.da <br>\
                       <br> Extra parameters for L1L2 sinkhorn algorithm can be for example : dict(norm=\"loglog\",reg_e=1e-1, reg_cl=2e0, max_iter=20). <br>\
                       For Gaussian : dict(norm=\"loglog\",mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10)")

    def helpUrl(self):
        return "http://pot.readthedocs.io/en/stable/all.html#module-ot.da"
    
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
      
        self.addParameter(QgsProcessingParameterString(
                self.PARAMS,
                self.tr('Parameters for the algorithm'),
                defaultValue='dict(norm="loglog", metric="sqeuclidean")',
                	)
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

        if MASK :
            MASK = MASK.source()
        
        ## Convert param str to param dictionnary
        msg =''
        try:
            PARAMSdict=eval(PARAMS)
            
        except:
            msg += 'Unable to identify parameters. Use dict(name=value, name=othervalue). \n'
            
        try:
            getattr(__import__("ot").da,SELECTED_ALGORITHM)
        except:
            msg += 'Please install POT library : "pip install POT" \n'
        # learn model
        
        if msg =='' :
            feedback.setProgress(1)
            feedback.setProgressText('Computing ROI values')
            import tempfile
            tempROI = tempfile.mktemp(suffix='.tif')
            
            #feedback.setProgressText('Params are : in dict '+str(dict(PARAMS)))
            
            dataraster.rasterize(SOURCE_RASTER.source(),SOURCE_LAYER.source(),SOURCE_COLUMN[0],tempROI)
            
            feedback.setProgress(2)
            Xs,ys = dataraster.get_samples_from_roi(SOURCE_RASTER.source(),tempROI)
        
            if TARGET_COLUMN == [] :
                TARGET_COLUMN = None
            else:
                TARGET_COLUMN = TARGET_COLUMN[0]
                
            feedback.setProgress(5)
            dataraster.rasterize(TARGET_RASTER.source(),TARGET_LAYER.source(),TARGET_COLUMN,tempROI)
            
            feedback.setProgress(8)
            
            Xt,yt = dataraster.get_samples_from_roi(TARGET_RASTER.source(),tempROI)
            
            os.remove(tempROI)                                                                                          
            
                
    

            ###
            transferModel = DA.rasterOT(params=PARAMSdict,transportAlgorithm=SELECTED_ALGORITHM,feedback=feedback)
            transferModel.learnTransfer(Xs,ys,Xt,None)
            
            transferModel.predictTransfer(SOURCE_RASTER.source(),TRANSPORTED_IMAGE,mask=MASK,NODATA=-10000)
    

            """
            transferModel = DA.learnTransfer(Xs,ys,Xt,yt,SELECTED_ALGORITHM,params=PARAMSdict,feedback=feedback)
        
            DA.predictTransfer(transferModel,SOURCE_RASTER.source(),TRANSPORTED_IMAGE,mask=MASK,NODATA=-10000,feedback=feedback)
            """
            return {self.TRANSPORTED_IMAGE: TRANSPORTED_IMAGE}

        else:
            return {'Error' : msg}
            
        
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return domainAdaptation()
    
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
        return 'Raster tool'
