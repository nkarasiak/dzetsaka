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
from builtins import range
import dzetsaka.scripts.function_dataraster as dataraster


from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.parameters import ParameterSelection
from processing.core.outputs import OutputRaster

class sieveAreaAlgorithm(GeoAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_LAYER = 'INPUT_LAYER'
    INPUT_COLUMN = 'INPUT_COLUMN'
    SIZE_HA= 'SIZE_HA'
    OUTPUT_RASTER = "OUTPUT_RASTER"
    CONNECTIVITY = ['4','8']
    INPUT_CONNECTIVITY = "INPUT_CONNECTIVITY"
    
    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/icon.png")
        
    def defineCharacteristics(self):

        # The name that the user will see in the toolbox
        self.name = 'Sieve raster by area (with multiband support)'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Filtering'

        self.addParameter(
        ParameterRaster(
            self.INPUT_RASTER,
            self.tr('Input raster'),
            False))
    
        # SIEVE SIZE
        self.addParameter(
        ParameterNumber(
            self.SIZE_HA,
            self.tr('Sieve size (0.5 for 0.5ha in metrics)'),
            default=0.5))

        # CONNECTIVITY
        self.addParameter(
        ParameterSelection(
            self.INPUT_CONNECTIVITY,
            "Connectivity",
            self.CONNECTIVITY,
            0))
       
        # OUTPUT RASTER
        self.addOutput(
        OutputRaster(
            self.OUTPUT_RASTER,
            self.tr("Output raster")))
            


    def processAlgorithm(self, progress):

        INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_MODEL = self.getOutputValue(self.OUTPUT_RASTER)
        SIZE_HA = self.getParameterValue(self.SIZE_HA)    
        INPUT_CONNECTIVITY = self.getParameterValue(self.INPUT_CONNECTIVITY)
        
        int(self.CONNECTIVITY[INPUT_CONNECTIVITY])
        
        # convert meter to ha
        SIZE_HA = int((SIZE_HA)*1000)
        
        from osgeo import gdal
    
    
        # begin sieve
                        
        datasrc = gdal.Open(INPUT_RASTER)
        srcband = datasrc.GetRasterBand(1)
        data,im = dataraster.open_data_band(INPUT_RASTER)        
        
        drv = gdal.GetDriverByName('GTiff')
        d = datasrc.RasterCount
        dst_ds = drv.Create(OUTPUT_MODEL,datasrc.RasterXSize,datasrc.RasterXSize,d,gdal.GDT_Byte)
        
        dst_ds.SetGeoTransform(datasrc.GetGeoTransform())
        dst_ds.SetProjection(datasrc.GetProjection())
    
        
        def sieve(srcband,dstband,sieveSize):
            gdal.SieveFilter(srcband,None,dstband,SIZE_HA,INPUT_CONNECTIVITY)
        
        pixelSize = datasrc.GetGeoTransform()[1] #get pixel size
        pixelSieve = int(SIZE_HA/(pixelSize*pixelSize)) #get number of pixel to sieve
        
        from qgis.core import QgsMessageLog
        QgsMessageLog.logMessage('pixel to sieve : '+str(pixelSieve))
        
        for i in range(d):
            srcband=datasrc.GetRasterBand(i+1)
            dstband=dst_ds.GetRasterBand(i+1)
            
            sieve(srcband,dstband,pixelSieve)
            srcband = None
            dstband = None

        dst_ds = None # close destination band
        
        
        
        

        