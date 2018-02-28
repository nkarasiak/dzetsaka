# -*- coding: utf-8 -*-
"""
@author: nkarasiak
www.karasiak.net
"""
import os
#import random
from osgeo import ogr
import numpy as np


class randomInSubset():

    def __init__(self,inShape,inField,outValidation,outTrain,number=50,percent=True):
        """
        inShape : str path file (e.g. '/doc/ref.shp')
        inField : string column name (e.g. 'class')
        outValidation : str path of shp output file (e.g. '/tmp/valid.shp')
        outTrain : str path of shp output file (e.g. '/tmp/train.shp')
        """
        if percent:
            number = number / 100.0
        else:
            number = int(number)
            
        lyr = ogr.Open(inShape)
        lyr1 = lyr.GetLayer()
        FIDs= np.zeros(lyr1.GetFeatureCount(),dtype=int)
        Features = []
        #unselFeat = []
        #current = 0
        
        for i,j in enumerate(lyr1):
            #print j.GetField(inField)
            FIDs[i] = j.GetField(inField)
            Features.append(j)
            #current += 1
        srs = lyr1.GetSpatialRef()
        lyr1.ResetReading()
        
        ## 
        if percent:
            validation,train = train_test_split(Features,test_size=number,train_size=1-number,stratify=FIDs)
        else:
            validation,train = train_test_split(Features,test_size=number,stratify=FIDs)
        
        self.saveToShape(validation,srs,outValidation)
        self.saveToShape(train,srs,outTrain)
    
    
    def saveToShape(self,array,srs,outShapeFile):
        # Parse a delimited text file of volcano data and create a shapefile
        # use a dictionary reader so we can access by field name
        # set up the shapefile driver
        outDriver = ogr.GetDriverByName( 'ESRI Shapefile' )
        
        # create the data source
        if os.path.exists(outShapeFile):
            outDriver.DeleteDataSource(outShapeFile)
        # Remove output shapefile if it already exists
        
        ds = outDriver.CreateDataSource(outShapeFile) #options = ['SPATIALITE=YES'])
    
        # create the spatial reference, WGS84
        
        lyrout = ds.CreateLayer('randomSubset',srs)
        fields = [array[1].GetFieldDefnRef(i).GetName() for i in range(array[1].GetFieldCount())]
        
        for f in fields:
            field_name = ogr.FieldDefn(f, ogr.OFTString)
            field_name.SetWidth(24)
            lyrout.CreateField(field_name)
            
        
        for k in array:
            lyrout.CreateFeature(k)
    
        # Save and close the data source
        ds = None

if __name__ == "__main__":
    inShape = '/mnt/DATA/demo/train.shp'
    inField = 'Class'
    number = 50
    percent = True
    
    outValidation = '/tmp/valid1.shp'
    outTrain ='/tmp/train.shp'
    
    randomInSubset(inShape,inField,outValidation,outTrain,number,percent)
    #randomInSubset('/tmp/valid.shp','level3','/tmp/processingd62a83be114a482aaa14ca317e640586/f99783a424984860ac9998b5027be604/OUTPUTVALIDATION.shp','/tmp/processingd62a83be114a482aaa14ca317e640586/1822187d819e450fa9ad9995d6757e09/OUTPUTTRAIN.shp',50,True)
