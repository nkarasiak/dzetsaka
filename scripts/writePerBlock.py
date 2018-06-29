#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:49:50 2018

@author: nkarasiak
"""
import numpy as np
import mainfunction as mf
initClass = mf.classifyImage()

def predictPerBlock(i,y_block_size,x_block_size,nc,nl,d,raster,mask,confidenceMapPerClass,confidenceMap,classifier,nClass,NODATA,model,\
                    out,out_confidenceMap,dst_confidenceMapPerClass,M,m):
    """
    print(i)
    lastBlock = i
    if int(lastBlock/total*100)!=int(i/total*100):
        lastBlock = i
        pushFeedback(int(i/total*100))
        
        if feedback=='gui':
            progress.addStep()
    """
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
            X[:,ind] = raster.GetRasterBand(int(ind+1)).ReadAsArray(j, i, cols, lines).reshape(cols*lines)

        # Do the prediction
        band_temp = raster.GetRasterBand(1)
        nodata_temp = band_temp.GetNoDataValue()

        if mask is None:
            band_temp = raster.GetRasterBand(1)
            mask_temp=band_temp.ReadAsArray(j, i, cols, lines).reshape(cols*lines)
            #temp_nodata = np.where(mask_temp != nodata_temp)[0]
            #t = np.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
            t = np.where(X[:,0]!=nodata_temp)[0]
            yp = np.zeros((cols*lines,))
            #K = np.zeros((cols*lines,))
            if confidenceMapPerClass or confidenceMap and classifier != 'GMM':
                K = np.zeros((cols*lines,nClass))
                K[:,:] = -1
            else:
                K = np.zeros((cols*lines))
                K[:] = -1

        else :
            mask_temp=mask.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols*lines)
            t = np.where((mask_temp!=0) & (X[:,0]!=nodata_temp))[0]
            yp = np.zeros((cols*lines,))
            yp[:] = NODATA
            #K = np.zeros((cols*lines,))
            if confidenceMapPerClass or confidenceMap and classifier != 'GMM':
                K = np.ones((cols*lines,nClass))
                K = np.negative(K)
            else:
                K = np.zeros((cols*lines))
                K = np.negative(K)
            
        
        # TODO: Change this part accorindgly ...
        if t.size > 0:
            if confidenceMap and classifier=='GMM' :
                yp[t],K[t] = model.predict(initClass.scale(X[t,:],M=M,m=m),None,confidenceMap)

            elif confidenceMap or confidenceMapPerClass and classifier !='GMM':
                yp[t] = model.predict(initClass.scale(X[t,:],M=M,m=m))                        
                K[t,:] = model.predict_proba(initClass.scale(X[t,:],M=M,m=m))*100

            else :
                yp[t] = model.predict(initClass.scale(X[t,:],M=M,m=m))

                #QgsMessageLog.logMessage('amax from predict proba is : '+str(sp.amax(model.predict.proba(self.scale(X[t,:],M=M,m=m)),axis=1)))
                

        # Write the data
        out.WriteArray(yp.reshape(lines,cols),j,i)
        out.SetNoDataValue(NODATA)
        out.FlushCache()

        if confidenceMap and classifier != 'GMM' :
            Kconf = np.amax(K,axis=1)
            out_confidenceMap.WriteArray(Kconf.reshape(lines,cols),j,i)
            out_confidenceMap.SetNoDataValue(-1)
            out_confidenceMap.FlushCache()
        
        if confidenceMapPerClass and classifier != 'GMM':
            for band in range(nClass):                        
                gdalBand = band+1
                out_confidenceMapPerClass = dst_confidenceMapPerClass.GetRasterBand(gdalBand)
                out_confidenceMapPerClass.SetNoDataValue(-1)
                out_confidenceMapPerClass.WriteArray(K[:,band].reshape(lines,cols),j,i)
                out_confidenceMapPerClass.FlushCache()
            

        del X,yp