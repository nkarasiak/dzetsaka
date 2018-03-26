#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:24:38 2018

@author: nkarasiak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:43:14 2018

@author: nkarasiak
"""

if __name__ == '__main__':
    import function_dataraster as dataraster
else:
    from . import function_dataraster as dataraster
    
import gdal
import tempfile
import ot
import os
#from sklearn import preprocessing

import numpy as np


def learnTransfer(Xs,ys,Xt,yt=None,transportAlgorithm='EMDTransport',params=None,feedback=None):
    """
    Learn domain adaptation model.
    
    Parameters
    ----------
    Xs : array_like
        Source domain array.
    ys : array_like
        Label source array (1d).
    Xt: array_like
        Target domain array.
    yt: array_like, optional
        Label target array (1d).
    Algorithm : str, optional
        item in list : ['EMDTransport','SinkhornTransport','SinkhornLpl1Transport','SinkhornL1l2Transport']
        
    Returns
    -------
    transportmodel : object
        The output model

    """
    if feedback:
        feedback.setProgress(10)
        feedback.setProgressText('Learning Optimal Transport with '+str(transportAlgorithm))
    transportFunction = getattr(__import__("ot").da,transportAlgorithm)
    
    if params is None:
        transportModel = transportFunction()
    else:
        transportModel = transportFunction(**params)
    
    # check if label is 1d
    if len(ys.shape)>1:
        ys = ys[:,0]
    if yt is not None:
        if len(yt.shape)>1:
            yt = yt[:,0]

    # learn transport        
    transportModel.fit(Xs,ys=ys,Xt=Xt,yt=yt)    
    
    if feedback:
        feedback.setProgress(20)
        
    return transportModel


def predictTransfer(model,imageTarget,outRaster,mask=None,NODATA=-9999,feedback=None):
    """
    Predict model using domain adaptation.
    
    Parameters
    ----------
    model : object
        Model generated from learnTransfer function.
    imageSource : str
        Path of image to adapt (source image)
    outRaster : str
        Path of tiff image to save as.
    mask: str, optional
        Path of raster mask.
    NODATA : int, optional
        Default -9999
    feedback : object, optional
        For Qgis Processing. Default is None.
        
    Returns
    -------
    outRaster : str
        Return the path of the predicted image.

    """
    if feedback:
        feedback.setProgressText('Now transporting '+str(os.path.basename(imageTarget)))
        
    dataSrc = gdal.Open(imageTarget)
    # Get the size of the image
    d  = dataSrc.RasterCount
    nc = dataSrc.RasterXSize
    nl = dataSrc.RasterYSize

    # Get the geoinformation
    GeoTransform = dataSrc.GetGeoTransform()
    Projection = dataSrc.GetProjection()

    # Get block size
    band = dataSrc.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    gdal_dt = band.DataType
    

    del band

    ## Initialize the output
    driver = gdal.GetDriverByName('GTiff')
    
    dst_ds = driver.Create(outRaster, nc,nl, d, gdal_dt)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)
    
    ## Perform the classification
    if mask is not None:
        maskData = gdal.Open(mask,gdal.GA_ReadOnly)

    total = nl*y_block_size
    
    
    total = 80/(int(nl/y_block_size))

    for i in range(0,nl,y_block_size):
        # feedback for Qgis
        if feedback:
            feedback.setProgress(int(i * total)+20)    
            

        if i + y_block_size < nl: # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0,nc,x_block_size): # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # Load the data and Do the prediction
            X = np.empty((cols*lines,d))
            for ind in range(d):
                X[:,ind] = dataSrc.GetRasterBand(int(ind+1)).ReadAsArray(j, i, cols, lines).reshape(cols*lines)


            # Do the prediction
            
            if mask is None:
                mask_temp = dataSrc.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols*lines)
            else:
                mask_temp = maskData.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols*lines)
                
            # check if nodata 
            t = np.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
            
            # transform array, default has nodata value
            yp = np.empty((cols*lines,d))
            yp[:,:] = NODATA
            # yp = np.nan((cols*lines,d))
            # K = np.zeros((cols*lines,))

            # TODO: Change this part accorindgly ...
            #if t.size > 0:
            if t.size > 0 :
                yp[t,:] = model.transform(X[t,:])
                
            for ind in range(d):         
                out = dst_ds.GetRasterBand(ind+1)
                # Write the data
                ypTemp = yp[:,ind]
                out.WriteArray(ypTemp.reshape(lines,cols),j,i)
                out.SetNoDataValue(NODATA)
                out.FlushCache()

            del X,yp
    
    return outRaster

if __name__ == '__main__':
    
    imageSource = "/mnt/DATA/Test/DA/SENTINEL_20170516.tif"
    vectorSource = "/mnt/DATA/Test/DA/ROI.sqlite"
    labelSourceField = "level3"
    
    imageTarget = "/mnt/DATA/Test/DA/SENTINEL_20171013.tif"
    vectorTarget = "/mnt/DATA/Test/DA/ROI.sqlite"
    labelTargetField = "level3"
    
    mask = None# "/mnt/DATA/Test/DA/woodmask.tif"
    
    tempROI = tempfile.mktemp(suffix='.tif')
    
    dataraster.rasterize(imageSource,vectorSource,labelSourceField,tempROI)
    
    Xs,ys = dataraster.get_samples_from_roi(imageSource,tempROI)


    dataraster.rasterize(imageTarget,vectorTarget,labelTargetField,tempROI)

    Xt,yt = dataraster.get_samples_from_roi(imageTarget,tempROI)
    
    os.remove(tempROI)

    
    transportAlgorithms = ['EMDTransport','SinkhornTransport','SinkhornLpl1Transport','SinkhornL1l2Transport']
    #
    transportAlgorithm = transportAlgorithms[3]

    params = dict(norm='loglog',mapping='barycentric',reg_e=1e-1,reg_cl=2e0,max_iter=20,verbose=False)
    
    #params = {'max_iter': 20.0, 'norm': 'loglog', 'reg_cl': 2.0, 'reg_e': 0.1}


    transferModel = learnTransfer(Xs,ys,Xt,yt,transportAlgorithm,params=params,feedback=None)
   
    """
    outModel = '/tmp/learnModel.ot'
    if outModel is not None :
        import pickle
        output = open(outModel, 'wb')
        pickle.dump(transferModel,output)    
        output.close()
    
    """
    outRaster = '/tmp/transfert_emd.tif'
    
    predictTransfer(transferModel,imageTarget,outRaster,mask=mask,NODATA=-9999,feedback=None)
    
