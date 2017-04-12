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
import scipy as sp
from dzetsaka.scripts.dtw import DTW, getSizes

from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.parameters import ParameterFile
from processing.core.parameters import ParameterMultipleInput
from processing.core.outputs import OutputDirectory
from processing.tools import dataobjects

from qgis.core import QgsMessageLog
from PyQt4.QtGui import QMessageBox
#from qgis.core import (QgsProcessingAlgorithm,QgsRasterLayer)

class dtwAlgorithm(GeoAlgorithm):
    REF_RASTER = 'REF_RASTER'
    REF_CSV = 'REF_CSV'
    SYNC_RASTER = 'SYNC_RASTER'
    SYNC_CSV = 'SYNC_CSV'
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    N_SAMPLES = 'N_SAMPLES'
    N_SPECTRAL_BANDS = 'N_SPECTRAL_BANDS'
    MASK_RASTER = 'MASK_RASTER'
    NO_DATA = 'NO_DATA'
        
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/icon.png")
        
    def defineCharacteristics(self):

        # The name that the user will see in the toolbox
        self.name = 'Dynamic Time Warping (DTW)'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Image manipulation'

        self.addParameter(
        ParameterRaster(
            self.REF_RASTER,
            self.tr('Reference image'),
            False))

        self.addParameter(
        ParameterFile(
            self.REF_CSV,
            self.tr('Reference variable (1 csv with 1 value per date).  Delimiter is comma : \',\'.'),ext='csv',
            optional=False))        
        ##
        self.addParameter(
            ParameterMultipleInput(
            self.SYNC_RASTER,
            self.tr('Image(s) to sync'),
            ParameterMultipleInput.TYPE_RASTER,True))
            
        #
        self.addParameter(
        ParameterFile(
            self.SYNC_CSV,
            self.tr('Sync variable (1 csv with 1 value per date). Respect the same order as the sync raster list. Delimiter is comma : \',\'.'),
            ext='csv',
            optional=False))        
        
        # add num
        self.addParameter(
        ParameterNumber(
            self.N_SAMPLES,
            self.tr('Number of dates to resample (minimum is the number of dates of your largest dataset). -1 to use the minimum.'),
            minValue=-1,
            default=-1))
        
        # add num
        self.addParameter(
        ParameterNumber(
            self.N_SPECTRAL_BANDS,
            self.tr('Number of spectral bands used for each date'),
            minValue=1,
            default=4))
            
        #MASK 
        self.addParameter(
        ParameterRaster(
            self.MASK_RASTER,
            self.tr('Mask image. Each pixel > 0 is classed as nodata.'),
            True))
            
        self.addParameter(
        ParameterNumber(
            self.NO_DATA,
            self.tr('No data value'),
            minValue=-10000,
            default=-10000))

        # We add a vector layer as output
        self.addOutput(
        OutputDirectory(
            self.OUTPUT_FOLDER,
            self.tr('Output folder')))

    def processAlgorithm(self, progress):


        REF_RASTER = self.getParameterValue(self.REF_RASTER)
        SYNC_RASTER = self.getParameterValue(self.SYNC_RASTER)
        N_SAMPLES = self.getParameterValue(self.N_SAMPLES)
        N_SPECTRAL_BANDS = self.getParameterValue(self.N_SPECTRAL_BANDS)
        
        REF_CSV = self.getParameterValue(self.REF_CSV)
        SYNC_CSV = self.getParameterValue(self.SYNC_CSV)
        OUTPUT_FOLDER = self.getOutputValue(self.OUTPUT_FOLDER)
        MASK_RASTER = self.getParameterValue(self.MASK_RASTER)
        
        refCsv = sp.loadtxt(REF_CSV,float,delimiter=',')
        
        
        QgsMessageLog.logMessage("MASK_RASTER is  "+str(MASK_RASTER))
        
        
        SYNC_CSV = SYNC_CSV.split(';')
        syncCsvList = [sp.loadtxt(i,float,delimiter=',') for i in SYNC_CSV]
        
        SYNC_RASTER = SYNC_RASTER.split(';')
        
        CSVs = syncCsvList[:]
        CSVs.insert(0,refCsv)
        
        QgsMessageLog.logMessage("CSVs is is "+str(CSVs))
        """
        VERIFY TEST
        """
        
        message = False
                
        r1,x1,y1,d1 = getSizes(REF_RASTER)        
        if r1 is None:
            message =  'Impossible to open '+str(REF_RASTER)
        
    
        for r in SYNC_RASTER:
    
            r2,x2,y2,d2 = getSizes(r)
            if r2 is None:
                message = 'Impossible to open ' +str(r)
            
            elif (x1 != x2) or (y1 != y2):
                message = 'Sync image and ref should be of the same size'
    
        if MASK_RASTER:
            rm,xm,ym,dm = getSizes(MASK_RASTER)
            if (x1 != xm) or (y1 != ym):
                message = "Ref image and mask should be the same size"
    
        """
        RUN IF OK
        """
        
        if N_SAMPLES == -1:
            N_SAMPLES = max([ref.shape[0] for ref in CSVs])
        
        if message:
            QMessageBox.information(None, message) 
        else:
            DTW(REF_RASTER,refCsv,SYNC_RASTER,syncCsvList,OUTPUT_FOLDER,mask=MASK_RASTER,n_color_bands=N_SPECTRAL_BANDS,n_samples=N_SAMPLES)

        
        

        
        
