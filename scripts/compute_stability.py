# coding: utf-8
import scipy as sp
import function_dataraster as funraster
import os
from scipy import stats

YEAR = [2006,2007,2008,2009,2010,2011,2012,2013,2014]
CLASSIFIER = ['svm','gmm']

WDIR='/media/nkarasiak/DATA/Formosat_2006-2014/Maps/'

W,H,D=4531,4036,9

for classifier in CLASSIFIER:
    os.chdir(WDIR)
    # Initializa output
    imc = sp.empty((H,W,D),dtype='uint8')
    out = sp.zeros((H,W,2),dtype='uint8')
    # Load data        
    for i,year in enumerate(YEAR):
        imc[:,:,i],GeoTransform,Projection=funraster.open_data('map_'+classifier+'_'+str(year)+'.tif')

    # Compute the most frequent class
    t = sp.where( imc[:,:,0]> 0 )
    tx,ty=t[0],t[1]
    for tx,ty in zip(t[0],t[1]):
        tempMode = stats.mode(imc[tx,ty,:],axis=None)
        out[tx,ty,0],out[tx,ty,1] = tempMode[0],tempMode[1] # Class and number of mode

    # Save the data
    funraster.write_data('count_'+classifier+'.tif',out,GeoTransform,Projection)
    # os.system('gdal_translate -a_nodata 0 -projwin 541989.189387 6262294.656 547522.004771 6258905.18352 -of GTiff -ot Byte temp.tif temp_c.tif')
    # os.system('otbcli_ColorMapping -in temp_c.tif -out count_'+classifier+'.png uint8 -method custom -method.custom.lut lut_s.txt')
    # os.system('rm temp*.tif')
    imc,out=[],[]
    print('Finished count for file "count_'+classifier+'.tif"')
    del tempMode

