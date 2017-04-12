# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:24:02 2017

@author: nkarasiak
"""

from sampler import DTWSampler
import scipy as sp
import gdal
import os
#from joblib import Parallel, delayed
#npy_arr = 0

def createTif(dest,nc,nl,d,geo,proj,gdaltype):
    driver = gdal.GetDriverByName('GTiff')
    #dst_ds = driver.Create(output_name, nc,nl, d, gdal.GDT_Float64) ##Float32??
    dst_ds = driver.Create(dest, nc,nl, d, gdaltype) ##Float32??
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    
    return dst_ds
    
def transformDTW(data,refList,Xf,n_color_bands,max_sz,feat,ind):
            
    #global npy_arr
    npy_arr = sp.zeros((len(data), max_sz*feat)) + sp.nan
    
    
    for idx,ts in enumerate(data):
        tempind = sp.zeros((max_sz, feat)) + sp.nan
        tempind[:refList[idx].shape[0],1:] = ts[ind,:].reshape([ts[ind,:].shape[0]/n_color_bands,n_color_bands])
        tempind[:refList[idx].shape[0],0] = refList[idx]
        
        #sz = tempind.shape[0]#/n_color_bands
        
        npy_arr[idx, :] = tempind.flatten()
    
        #sz = ts.shape[1]+(ts.shape[1]/n_color_bands)
        
        #temp[idx,:sz] = tempind[:,:] #ts[ind,:]
        
    npy_arr = npy_arr.reshape(-1, max_sz * feat)
    
    
    
    s = DTWSampler(scaling_col_idx=0, reference_idx=0, d=feat,n_samples=max_sz,save_path=True)
    global transformed_array
    transformed_array = s.fit_transform(npy_arr).flatten().reshape((len(data), -1, s.d))
    #Xf[ind,:] = transformed_array[1:,:,1:]
    
    
    return transformed_array[:,:,1:]
            
        
def getSizes(im):
    raster = gdal.Open(im,gdal.GA_ReadOnly)
    x = raster.RasterXSize
    y = raster.RasterYSize
    d = raster.RasterCount
    return raster,x,y,d
   
    
def DTW(im1,ref1,im2,ref2,outputFolder,mask=None,nodata=-10000,n_color_bands=1,scaling_col_idx=0,reference_idx=0,n_samples=100,save_path=False):
    #global npy_arr,data
    global max_sz
    # Open Mask and get additionnal information
    r1,x1,y1,d1 = getSizes(im1)
    
    if r1 is None:
        print 'Impossible to open '+im1
        exit()
    
    for r in im2:
    
        r2,x2,y2,d2 = getSizes(r)
        if r2 is None:
            print 'Impossible to open '+r
            exit()
        elif (x1 != x2) or (y1 != y2):
            print 'Image and ref should be of the same size'
            exit() 
    
    if mask:
        rm,xm,ym,dm = getSizes(mask)
        if (x1 != xm) or (y1 != ym):
            print "Ref image and mask should be the same size"
            exit()
    
    d  = d1
    nc = x1
    nl = y1
    
    feat = n_color_bands+1
#    data_3d = sp.zeros([(d),1],int)

    # Get the geoinformation    
    geo = r1.GetGeoTransform()
    proj = r1.GetProjection()

    # Get block size
    band = r1.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    del band

    
    imageList = im2[:]
    imageList.insert(0,im1)
    refList = ref2[:]
    refList.insert(0,ref1)  
    #create Tif
    dst_ds = []
    for imToCreate in imageList:
        dst_ds.append(createTif(os.path.join(outputFolder,os.path.basename(imToCreate)),nc,nl,n_samples,geo,proj,gdal.GDT_Int16))
        
    
    for i in xrange(0,nl,y_block_size):
        print i
        if i + y_block_size < nl: # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in xrange(0,nc,x_block_size): # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            data = []
            
            
            for image in imageList:
                
                # fill block for each image
                data_src = gdal.Open(image)
                d = data_src.RasterCount
                
                X = sp.empty((cols*lines,d))
                #global M
                M = sp.empty((cols*lines))

                if mask:
                    masktemp = rm.GetRasterBand(1)
                    M[:] = masktemp.ReadAsArray(j, i, cols, lines).reshape(cols*lines)
                    
                if sp.any(M==0):
    
                    for b in xrange(d):
                        
                        temp = data_src.GetRasterBand(b+1)
                        X[:,b] = temp.ReadAsArray(j, i, cols, lines).reshape(cols*lines)    

                    data.append(X)
                
            
            max_sz = max([ref.shape[0] for ref in refList])
            global Xf
            Xf = sp.empty((cols*lines,len(imageList),max_sz*n_color_bands))
            
            #Xf[:,:] = Parallel(n_jobs=-1,verbose=False)(delayed(transformDTW)(data,Xf,n_color_bands,max_sz,feat,ind) for ind in range(cols*lines))
            
            if sp.any(M==0):
                for ind in xrange(cols*lines):
                    if M[ind] >0: # if single pixel masked
                        Xf[ind,:,:] = nodata
                    else:
                        Xf[ind,:,:] = transformDTW(data,refList,Xf,n_color_bands,max_sz,feat,ind)[:,:,:].reshape((len(imageList),-1))
            else: #if all block masked
                Xf[:,:,:] = nodata
      
            for imidx,image in enumerate(imageList):
                    
                for b in xrange(n_samples):
                    out = dst_ds[imidx].GetRasterBand(b+1)
                    out.SetNoDataValue(nodata)
                    out.WriteArray(Xf[:,imidx,b].reshape(lines,cols),j,i)
                    out.FlushCache()

          
          
if __name__ == '__main__' : 
    
    
    """ sample dates """
    dates2010 = sp.loadtxt("/home/nkarasiak/Bureau/Rennes/SITS/sample_time_2010.csv",int)
    
    DJ2010 = sp.array([0.8965,29.497,71.7925,147.5515,682.9425,802.5925,875.3175,992.5475,1095.9175,1266.1745,1390.4385,1429.8005,1461.619,1469.4995])
    
    dates2012=sp.loadtxt("/home/nkarasiak/Bureau/Rennes/SITS/sample_time_2012.csv",int)
    DJ2012=sp.array([1.68,2.38,4.62,32.81,86.445,415.43,590.305,681,405,958.925,1139.145,1648.911,1682.604])
    
    DJcum2010 = sp.int0(sp.loadtxt("/home/nkarasiak/Bureau/Rennes/Météo/2010.csv",float))[:-1]
    DJcum2012 = sp.int0(sp.loadtxt("/home/nkarasiak/Bureau/Rennes/Météo/2012.csv",float))
    
    
    """ resampled at 10 days"""
    
    dates2010 = sp.loadtxt("/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/sample_time_2010.csv",int)
    
    #DJ2010 = sp.array([0.8965,29.497,71.7925,147.5515,682.9425,802.5925,875.3175,992.5475,1095.9175,1266.1745,1390.4385,1429.8005,1461.619,1469.4995])
    DJ2010plusDate = sp.loadtxt("/home/nkarasiak/Bureau/Rennes/Météo/2010_datedj.csv",object,delimiter=',')
    dates2012=sp.loadtxt("/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/sample_time_2012.csv",int)
    DJ2012plusDate = sp.loadtxt("/home/nkarasiak/Bureau/Rennes/Météo/2012_datedj.csv",object,delimiter=',')
    #DJ2012=sp.array([1.68,2.38,4.62,32.81,86.445,415.43,590.305,681,405,958.925,113x9.145,1648.911,1682.604])
    
    #aa = ["/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/sample_time_2010.csv","/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/sample_time_2012.csv"]
    #syncCsvList = [sp.loadtxt(i,float,delimiter=',') for i in aa]

    
    #DJ for 2010
    DJ2010 = sp.zeros(len(dates2010))
    for a,b in enumerate(dates2010):
        #print i
        DJ2010[a] = DJ2010plusDate[:,3][sp.where(DJ2010plusDate[:,0]==datetime.datetime.strptime(str(b), '%Y%m%d').strftime('%d/%m/%Y'))[0][0]]
    #DJ for 2012
    DJ2012 = sp.zeros(len(dates2012))
    for a,b in enumerate(dates2012):
        #print i
        DJ2012[a] = DJ2012plusDate[:,3][sp.where(DJ2012plusDate[:,0]==datetime.datetime.strptime(str(b), '%Y%m%d').strftime('%d/%m/%Y'))[0][0]]
    
    
    im1 = "/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/crop_2010.tif"
    im2 = ["/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/crop_2012.tif","/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/crop_2013.tif"]
    ref1 = DJ2010
    ref2 = [DJ2012,DJ2012]
    
    mask = "/home/nkarasiak/Bureau/Rennes/SITS_10DAYS/mask.tif"
    
    
    n_samples = max(gdal.Open(im1).RasterCount/4,gdal.Open(im2[0]).RasterCount/4)
    
    """
    for pixel in range (0,100,10):
        print("pixel is "+str(pixel))
        import time 
        t0 = time.time()
        a,b,c = DTW(im1,ref1,im2,ref2,k=pixel,n_color_bands=4,scaling_col_idx=0,reference_idx=0,n_samples=n_samples,save_path=True)
        t1 = time.time()
        print(t1-t0)
    """
#    a,b,c = 
    import time
    t0 = time.time()
    
    outputFolder = "/home/nkarasiak/Bureau/Rennes/DTW/"
    DTW(im1,ref1,im2,ref2,outputFolder,mask=mask,n_color_bands=4,n_samples=n_samples)
    print(time.time()-t0)