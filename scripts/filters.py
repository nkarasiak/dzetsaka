# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 08:18:26 2016

@author: lennepkade
"""
import function_dataraster as dataraster
import os
import tempfile
import scipy as sp
from scipy import ndimage
from osgeo import gdal, ogr, osr
from mainfunction import progressBar
from PyQt4 import QtCore
from qgis.core import QgsMessageLog

class filtersFunction():  
    """!@brief Filter a raster with median and closing filter.
    
    Filter class to isolate the forest, delete dark lines and fonts from Historical Map
    
    Input :
        inImage : image name to filter ('text.tif',str)
        outRaster : raster name of the filtered file (str)
        inShapeGrey : Size for the grey closing convolution matrix (odd number, int)
        inShapeMedian : Size for the median convolution matrix (odd  number, int)
        
    Output :
        Nothing except a raster file (outRaster)
        
    """

        
    def historicalMapFilter(self,inImage,outRaster,inShapeGrey,inShapeMedian,iterMedian):
        # open data with Gdal
        try:
            data,im=dataraster.open_data_band(inImage)
        except:
            print 'Cannot open image'
        
        # get proj,geo and dimension (d) from data
        proj = data.GetProjection()
        geo = data.GetGeoTransform()
        d = data.RasterCount
        
        if outRaster=='':
            outRaster = tempfile.mktemp('.tif')
        # Progress Bar
        maxStep=d+d*iterMedian
        try:
            filterProgress=progressBar(' Filtering...',maxStep)
        except:
            print 'Failed loading progress Bar'
        
        # Try all, if error close filterProgress        
        try:            
            # create empty geotiff with d dimension, geotransform & projection
            
            try:
                outFile=dataraster.create_empty_tiff(outRaster,im,d,geo,proj)
            except:
                print 'Cannot write empty image '+outRaster
            
            # fill outFile with filtered band
            for i in range(d):
                # Read data from the right band
                try:
                    filterProgress.addStep()
                    temp = data.GetRasterBand(i+1).ReadAsArray()
                    
                except:
                    print 'Cannot get rasterband'+i
                    QgsMessageLog.logMessage("Problem reading band "+str(i)+" from image "+inImage)
                # Filter with greyclosing, then with median filter
            
                try:
                    temp = ndimage.morphology.grey_closing(temp,size=(inShapeGrey,inShapeGrey))
                except:
                    print 'Cannot filter with Grey_Closing'
                    QgsMessageLog.logMessage("Problem with Grey Closing")
    
                for j in range(iterMedian):
                    try:
                        filterProgress.addStep()
                        temp = ndimage.filters.median_filter(temp,size=(inShapeMedian,inShapeMedian))
                    except:
                        print 'Cannot filter with Median'
                        QgsMessageLog.logMessage("Problem with median filter")
                    
                # Save bandand outFile
                try:
                    out=outFile.GetRasterBand(i+1)
                    out.WriteArray(temp)
                    out.FlushCache()
                    temp = None
                except:
                    QgsMessageLog.logMessage("Cannot save band"+str(i)+" on image" + outRaster)
                    
            filterProgress.reset()
        except:
            filterProgress.reset()
    
    def historicalMapPostClass(self,inRaster,sieveSize,inClassNumber,outShp):        
        
        #Progress=progressBar(' Post-classification...',10)
            
        rasterTemp = tempfile.mktemp('.tif')
                
        datasrc = gdal.Open(inRaster)
        srcband = datasrc.GetRasterBand(1)
        data,im=dataraster.open_data_band(inRaster)        

        drv = gdal.GetDriverByName('GTiff')
        dst_ds = drv.Create(rasterTemp,datasrc.RasterXSize,datasrc.RasterXSize,1,srcband.DataType)
        dst_ds.SetGeoTransform(datasrc.GetGeoTransform())
        dst_ds.SetProjection(datasrc.GetProjection())
    
        dstband=dst_ds.GetRasterBand(1)

        def sieve(srcband,dstband,sieveSize):
            gdal.SieveFilter(srcband,None,dstband,sieveSize)
        
        sieve(srcband,dstband,sieveSize)
        #Progress.addStep(2)
        dst_ds =None
        
        def reclass(rasterTemp,inClassNumber):
            dst_ds = gdal.Open(rasterTemp,gdal.GA_Update)
            srcband = dst_ds.GetRasterBand(1).ReadAsArray()
            # All data which is not forest is set to 0, so we fill all for the forest only, because it's a binary fill holes.            
            # Set selected class as 1                   
            srcband[srcband !=inClassNumber]=0
            srcband[srcband ==inClassNumber]=1
            #QgsMessageLog.logMessage("Cannot sieve")
                
            dst_ds.GetRasterBand(1).WriteArray(srcband)
            dst_ds.FlushCache()
            return rasterTemp
        
        
        rasterTemp = reclass(rasterTemp,inClassNumber)
        #Progress.addStep(1)
        
        def polygonize(rasterTemp,outShp):
            
            sourceRaster = gdal.Open(rasterTemp)
            band = sourceRaster.GetRasterBand(1)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            # If shapefile already exist, delete it
            if os.path.exists(outShp):
                driver.DeleteDataSource(outShp)
                
            outDatasource = driver.CreateDataSource(outShp)            
            # get proj from raster            
            srs = osr.SpatialReference()
            srs.ImportFromWkt(sourceRaster.GetProjectionRef())
            # create layer with proj
            outLayer = outDatasource.CreateLayer(outShp,srs)
            # Add class column (1,2...) to shapefile
      
            newField = ogr.FieldDefn('Class', ogr.OFTInteger)
            outLayer.CreateField(newField)
            print('vectorizing')
            gdal.Polygonize(band, None,outLayer, 0,[],callback=None)  
            print('vectorized')
            outDatasource.Destroy()
            sourceRaster=None
            band=None
                    #    QgsMessageLog.logMessage("Cannot vectorize "+rasterTemp)
            
            ioShpFile = ogr.Open(outShp, update = 1)
            
            
            lyr = ioShpFile.GetLayerByIndex(0)
            nbFeatures = lyr.GetFeatureCount()
            lyr.ResetReading()    

            for i in lyr:
                lyr.SetFeature(i)
            # if area is less than inMinSize or if it isn't forest, remove polygon 
                if i.GetField('Class')!=1:
                    lyr.DeleteFeature(i.GetFID())        
            ioShpFile.Destroy()
            
            #historicalProgress.reset()
            
            return outShp
        
        outShp = polygonize(rasterTemp,outShp)  
        #Progress.addStep(7)
        #Progress.reset()
        return outShp
        
        
    def filters(self,inImage,outRaster,inFilter,inFilterSize,inFilterIter):
        # open data with Gdal        
        self.processed = 0
        try:
            data,im=dataraster.open_data_band(inImage)
        except:
            print 'Cannot open image'

        # get proj,geo and dimension (d) from data
        proj = data.GetProjection()
        geo = data.GetGeoTransform()
        d = data.RasterCount
        
        # Progress Bar
        maxStep=d*inFilterIter
        filterProgress=progressBar(' Filtering...',maxStep)
        
        # Try all, if error close filterProgress        
                
        # create empty geotiff with d dimension, geotransform & projection
        
        try:
            outFile=dataraster.create_empty_tiff(outRaster,im,d,geo,proj)
        except:
            print 'Cannot write empty image '+outRaster
        
        # fill outFile with filtered band
        for i in range(d):
            # Read data from the right band
            filterProgress.addStep()
            
            temp = data.GetRasterBand(i+1).ReadAsArray()
                
            # Filter with greyclosing, then with median filter
            for j in range(inFilterIter):
                filterProgress.addStep()

                try:
                    if inFilter=='Closing':
                        temp = ndimage.morphology.grey_closing(temp,size=(inFilterSize,inFilterSize))
                    elif inFilter=='Dilation':
                        temp = ndimage.morphology.grey_dilation(temp,size=(inFilterSize,inFilterSize))
                    elif inFilter=='Opening':
                        temp = ndimage.morphology.grey_opening(temp,size=(inFilterSize,inFilterSize))
                    elif inFilter=='Erosion':
                        temp = ndimage.morphology.grey_erosion(temp,size=(inFilterSize,inFilterSize))
                    elif inFilter=='Median':
                        temp = ndimage.filters.median_filter(temp,size=(inFilterSize,inFilterSize))
                  
                except:
                    QgsMessageLog.logMessage("Cannot perform " +inFilter)                    
            # Save bandand outFile
            try:
                out=outFile.GetRasterBand(i+1)
                out.WriteArray(temp)
                out.FlushCache()
                temp = None
            except:
                QgsMessageLog.logMessage("Cannot save band"+str(i)+" on image" + outRaster)
        filterProgress.reset()
        print outRaster
   
     
class optimizeFilters(QtCore.QObject):

    def __init__(self,inRaster,outRaster,inFilter,inFilterSize,inFilterIter, parent = None):
        QtCore.QObject.__init__(self, parent)
        self.killed = False
        self.inRaster = inRaster
        self.outRaster = outRaster
        self.inFilter = inFilter
        self.inFilterSize = inFilterSize
        self.inFilterIter = inFilterIter
        
    def run(self):
        ret = False
        try:
            worker = filtersFunction()
            #self.status.emit('Task started!')    
            filterProgress=progressBar(' Filtering...',2)
            filterProgress.addStep()
            worker.filters(self.inRaster,self.outRaster,self.inFilter,self.inFilterSize,self.inFilterIter)
            
            print('done')
            
            #self.status.emit('Task ended!')
            if self.killed is False:
                ret = True
        except Exception, e:
            import traceback
            # forward the exception upstream
            self.error.emit(e, traceback.format_exc())
        self.finished.emit(ret)
    
    def kill(self):
        self.killed = True
    
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(bool)
    error = QtCore.pyqtSignal(Exception, basestring)
    status = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
        

if __name__=="__main__":
    inRaster = '/tmp/tmp0BeSgc/map_class.tif'
    outVector = '/tmp/processingfe206327dd344fa0bd47a73e280660e6/15a60020fa0c4715b2730c81c87f1f13/OUTPUTVECTOR.shp'
    
    worker=filtersFunction()
    #worker.filters(inRaster,outRaster,inFilter,10,1)
    worker.historicalMapPostClass(inRaster,31,1,outVector)
    #worker.historicalMapReclass('/tmp/nana2.tif',2)
    
