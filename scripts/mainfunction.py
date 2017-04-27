"""!@brief Interface between qgisForm and function_historical_map.py
./***************************************************************************
 HistoricalMap
                                 A QGIS plugin
 Mapping old landcover (specially forest) from historical  maps
                              -------------------
        begin                : 2016-01-26
        git sha              : $Format:%H$
        copyright            : (C) 2016 by Karasiak & Lomellini
        email                : karasiak.nicolas@gmail.com
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
# -*- coding: utf-8 -*-
import function_dataraster as dataraster
import pickle
import os
import accuracy_index as ai
import tempfile
import gmm_ridge as gmmr
import scipy as sp
from scipy import ndimage
from osgeo import gdal, ogr, osr
from PyQt4.QtGui import QProgressBar, QApplication
from PyQt4 import QtCore
from qgis.utils import iface
from qgis.core import QgsMessageLog

class learnModel():
    """!@brief Learn model with a shp file and a raster image.
    
    Input :
        inRaster : Filtered image name ('sample_filtered.tif',str).
        inVector : Name of the training shpfile ('training.shp',str).
        inField : Column name where are stored class number (str).
        inSplit : (int).
        inSeed : (int).
        outModel : Name of the model to save, will be compulsory for the 3rd step (classifying).
        outMatrix : Default the name of the file inRaster(minus the extension)_inClassifier_inSeed_confu.csv (str).
        inClassifier : GMM,KNN,SVM, or RF. (str).
        
    Output :
        Model file.
        Confusion Matrix.
        
    """
    def __init__(self,inRaster,inVector,inField='Class',outModel=None,inSplit=1,inSeed=0,outMatrix=None,inClassifier='GMM'):
          
          
        learningProgress=progressBar('Learning model...',6)
 
        # Convert vector to raster
        try:
            try:
                temp_folder = tempfile.mkdtemp()
                filename = os.path.join(temp_folder, 'temp.tif')
                
                data = gdal.Open(inRaster,gdal.GA_ReadOnly)
                shp = ogr.Open(inVector)
                
                lyr = shp.GetLayer()
            except:
                QgsMessageLog.logMessage("Problem with making tempfile or opening raster or vector")
            
            # Create temporary data set
            try:
                driver = gdal.GetDriverByName('GTiff')
                dst_ds = driver.Create(filename,data.RasterXSize,data.RasterYSize, 1,gdal.GDT_Byte)
                dst_ds.SetGeoTransform(data.GetGeoTransform())
                dst_ds.SetProjection(data.GetProjection())
                OPTIONS = 'ATTRIBUTE='+inField
                gdal.RasterizeLayer(dst_ds, [1], lyr, None,options=[OPTIONS])
                data,dst_ds,shp,lyr=None,None,None,None
            except:
                QgsMessageLog.logMessage("Cannot create temporary data set")
            
            # Load Training set
            try:
                X,Y =  dataraster.get_samples_from_roi(inRaster,filename)
            except:
                QgsMessageLog.logMessage("Problem while getting samples from ROI with"+inRaster)
                QgsMessageLog.logMessage("Are you sure to have only integer values in your "+str(inField)+" column ?")
                
            
            [n,d] = X.shape
            C = int(Y.max())
            SPLIT = inSplit
            os.remove(filename)
            os.rmdir(temp_folder)
            
            # Scale the data
            X,M,m = self.scale(X)
            
            
            learningProgress.addStep() # Add Step to ProgressBar
    
            # Learning process take split of groundthruth pixels for training and the remaining for testing
            
            
            try:
                if SPLIT < 1:
                    # progressBar, set Max to C
                    
                    # Random selection of the sample
                    x = sp.array([]).reshape(0,d)
                    y = sp.array([]).reshape(0,1)
                    xt = sp.array([]).reshape(0,d)
                    yt = sp.array([]).reshape(0,1)
                    
                    sp.random.seed(inSeed) # Set the random generator state
                    for i in range(C):            
                        t = sp.where((i+1)==Y)[0]
                        nc = t.size
                        ns = int(nc*SPLIT)
                        rp =  sp.random.permutation(nc)
                        x = sp.concatenate((X[t[rp[0:ns]],:],x))
                        xt = sp.concatenate((X[t[rp[ns:]],:],xt))
                        y = sp.concatenate((Y[t[rp[0:ns]]],y))
                        yt = sp.concatenate((Y[t[rp[ns:]]],yt))
                        #Add Pb
            except:
                QgsMessageLog.logMessage("Problem while learning if SPLIT <1")
                  
            
            x,y=X,Y
            
            learningProgress.addStep() # Add Step to ProgressBar
            # Train Classifier
            if inClassifier == 'GMM':
                try:
                    # tau=10.0**sp.arange(-8,8,0.5)
                    model = gmmr.GMMR()
                    model.learn(x,y)
                    # htau,err = model.cross_validation(x,y,tau)
                    # model.tau = htau
                except:
                    QgsMessageLog.logMessage("Cannot train with GMMM")
            else:
                try:                    
                    from sklearn import neighbors
                    from sklearn.svm import SVC
                    from sklearn.ensemble import RandomForestClassifier
                    try:
                        from sklearn.cross_validation import StratifiedKFold
                    except:
                        from sklearn.model_selection import StratifiedKFold
                        
                    from sklearn.grid_search import GridSearchCV
                    
                    try:   
                        
                        # AS Qgis in Windows doensn't manage multiprocessing, force to use 1 thread for not linux system
                        if os.name == 'posix':
                            n_jobs=-1
                        else:
                            n_jobs=1
                        
                        # 
                        if inClassifier == 'RF':
                            param_grid_rf = dict(n_estimators=3**sp.arange(1,5),max_features=sp.arange(1,4))
                            y.shape=(y.size,)    
                            cv = StratifiedKFold(y, n_folds=3)
                            grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_rf, cv=cv,n_jobs=n_jobs)
                            grid.fit(x, y)
                            model = grid.best_estimator_
                            model.fit(x,y)        
                        elif inClassifier == 'SVM':
                            param_grid_svm = dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(-2,5))
                            y.shape=(y.size,)    
                            cv = StratifiedKFold(y, n_folds=5)
                            grid = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=cv,n_jobs=n_jobs)
                            grid.fit(x, y)
                            model = grid.best_estimator_
                            model.fit(x,y)
                        elif inClassifier == 'KNN':
                            param_grid_knn = dict(n_neighbors = sp.arange(1,20,4))
                            y.shape=(y.size,)    
                            cv = StratifiedKFold(y, n_folds=3)
                            grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=param_grid_knn, cv=cv,n_jobs=n_jobs)
                            grid.fit(x, y)
                            model = grid.best_estimator_
                            model.fit(x,y)
                    except:
                        QgsMessageLog.logMessage("Cannot train with classifier "+inClassifier)
                
                except:
                    QgsMessageLog.logMessage("You must have sklearn dependencies on your computer. Please consult the documentation for installation.")
                
            learningProgress.prgBar.setValue(5) # Add Step to ProgressBar
            # Assess the quality of the model
            if SPLIT < 1 :
                # if  inClassifier == 'GMM':
                #          = model.predict(xt)[0]
                # else:
                yp = model.predict(xt)
                CONF = ai.CONFUSION_MATRIX()
                CONF.compute_confusion_matrix(yp,yt)
                sp.savetxt(outMatrix,CONF.confusion_matrix,delimiter=',',fmt='%1.4d')
                
        
            # Save Tree model
            if outModel is not None:
                output = open(outModel, 'wb')
                pickle.dump([model,M,m], output)
                output.close()
            
            learningProgress.addStep() # Add Step to ProgressBar   
            
            # Close progressBar
            learningProgress.reset()
            learningProgress=None
        except:
            learningProgress.reset()
            
    def scale(self,x,M=None,m=None):
        """!@brief Function that standardize the data.
        
            Input:
                x: the data
                M: the Max vector
                m: the Min vector
            Output:
                x: the standardize data
                M: the Max vector
                m: the Min vector
        """
        [n,d]=x.shape
        if not sp.issubdtype(x.dtype,float):
            x=x.astype('float')
    
        # Initialization of the output
        xs = sp.empty_like(x)
    
        # get the parameters of the scaling
        M,m = sp.amax(x,axis=0),sp.amin(x,axis=0)
        den = M-m
        for i in range(d):
            if den[i] != 0:
                xs[:,i] = 2*(x[:,i]-m[i])/den[i]-1
            else:
                xs[:,i]=x[:,i]
    
        return xs,M,m
        
class classifyImage():
    """!@brief Classify image with learn clasifier and learned model
    
    Create a raster file, fill hole from your give class (inClassForest), convert to a vector,
    remove parcel size which are under a certain size (defined in inMinSize) and save it to shp.

        Input :
            inRaster : Filtered image name ('sample_filtered.tif',str)
            inModel : Output name of the filtered file ('training.shp',str)
            outShpFile : Output name of vector files ('sample.shp',str)
            inMinSize : min size in acre for the forest, ex 6 means all polygons below 6000 m2 (int)
            TODO inMask : Mask size where no classification is done                                     |||| NOT YET IMPLEMENTED
            inField : Column name where are stored class number (str)
            inNODATA : if NODATA (int)
            inClassForest : Classification number of the forest class (int)
        
        Output :
            SHP file with deleted polygon below inMinSize
    
    """
            
        
    def initPredict(self,inRaster,inModel,outRaster,inMask=None,confidenceMap=None,classifier='GMM'):
        

        # Load model
        try:
            model = open(inModel,'rb') # TODO: Update to scale the data 
            if model is None:
                print "Model not load"
                QgsMessageLog.logMessage("Model : "+inModel+" is none")
            else:
                tree,M,m = pickle.load(model)
                model.close()
        except:
            QgsMessageLog.logMessage("Error while loading the model : "+inModel)
            
        # Creating temp file for saving raster classification
        try:
            temp_folder = tempfile.mkdtemp()
            rasterTemp = os.path.join(temp_folder, 'temp.tif')
        except:
            QgsMessageLog.logMessage("Cannot create temp file "+rasterTemp)
            # Process the data
        #try:
        predictedImage=self.predict_image(inRaster,outRaster,tree,inMask,confidenceMap,-10000,SCALE=[M,m],classifier=classifier)
        #except:
         #   QgsMessageLog.logMessage("Problem while predicting "+inRaster+" in temp"+rasterTemp)
        
        return predictedImage
    
                
    def scale(self,x,M=None,m=None):  # TODO:  DO IN PLACE SCALING
        """!@brief Function that standardize the data
        
            Input:
                x: the data
                M: the Max vector
                m: the Min vector
            Output:
                x: the standardize data
                M: the Max vector
                m: the Min vector
        """
        [n,d]=x.shape
        if not sp.issubdtype(x.dtype,float):
            x=x.astype('float')
    
        # Initialization of the output
        xs = sp.empty_like(x)
    
        # get the parameters of the scaling
        if M is None:
            M,m = sp.amax(x,axis=0),sp.amin(x,axis=0)
            
        den = M-m
        for i in range(d):
            if den[i] != 0:
                xs[:,i] = 2*(x[:,i]-m[i])/den[i]-1
            else:
                xs[:,i]=x[:,i]
    
        return xs
        
    def predict_image(self,inRaster,outRaster,model,inMask=None,confidenceMap=None,NODATA=-10000,SCALE=None,classifier='GMM'):
        """!@brief The function classify the whole raster image, using per block image analysis.
        
        The classifier is given in classifier and options in kwargs
        
            Input :
                inRaster : Filtered image name ('sample_filtered.tif',str)
                outRaster :Raster image name ('outputraster.tif',str)
                model : model file got from precedent step ('model', str)
                inMask : mask to 
                confidenceMap :  map of confidence per pixel
                NODATA : Default set to -10000 (int)
                SCALE : Default set to None
                classifier = Default 'GMM'
                
            Output :
                nothing but save a raster image and a confidence map if asked
        """
        # Open Raster and get additionnal information
        
        raster = gdal.Open(inRaster,gdal.GA_ReadOnly)
        if raster is None:
            print 'Impossible to open '+inRaster
            exit()
        
        if inMask is None:
            mask=None
        else:
            mask = gdal.Open(inMask,gdal.GA_ReadOnly)
            if mask is None:
                print 'Impossible to open '+inMask
                exit()
            # Check size
            if (raster.RasterXSize != mask.RasterXSize) or (raster.RasterYSize != mask.RasterYSize):
                print 'Image and mask should be of the same size'
                exit()   
        if SCALE is not None:
            M,m=sp.asarray(SCALE[0]),sp.asarray(SCALE[1])
            
        # Get the size of the image
        d  = raster.RasterCount
        nc = raster.RasterXSize
        nl = raster.RasterYSize
        
        # Get the geoinformation    
        GeoTransform = raster.GetGeoTransform()
        Projection = raster.GetProjection()
        
        # Get block size
        band = raster.GetRasterBand(1)
        block_sizes = band.GetBlockSize()
        x_block_size = block_sizes[0]
        y_block_size = block_sizes[1]
        del band
        
        ## Initialize the output
        driver = gdal.GetDriverByName('GTiff')        
        dst_ds = driver.Create(outRaster, nc,nl, 1, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(GeoTransform)
        dst_ds.SetProjection(Projection)        
        out = dst_ds.GetRasterBand(1)

        if confidenceMap : 
            dst_confidenceMap = driver.Create(confidenceMap, nc,nl, 1, gdal.GDT_Float32)
            dst_confidenceMap.SetGeoTransform(GeoTransform)
            dst_confidenceMap.SetProjection(Projection)
            out_confidenceMap = dst_confidenceMap.GetRasterBand(1)
        
        ## Perform the classification
        predictProgress=progressBar('Classifying image...',nl*y_block_size)       
        
        for i in range(0,nl,y_block_size):
            predictProgress.addStep()
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
                X = sp.empty((cols*lines,d))
                for ind in xrange(d):
                    X[:,ind] = raster.GetRasterBand(int(ind+1)).ReadAsArray(j, i, cols, lines).reshape(cols*lines)
                    
                # Do the prediction
                if mask is None:
                    mask_temp=raster.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols*lines)
                    t = sp.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
                    yp = sp.zeros((cols*lines,))
                    K = sp.zeros((cols*lines,))

                else :
                    mask_temp=mask.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols*lines)
                    t = sp.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
                    yp = sp.zeros((cols*lines,))
                    K = sp.zeros((cols*lines,))
    
                # TODO: Change this part accorindgly ...
                if t.size > 0:
                    if confidenceMap and classifier=='GMM' :
                        yp[t],K[t] = model.predict(self.scale(X[t,:],M=M,m=m),None,confidenceMap)  
                        
                    elif confidenceMap :
                        yp[t] = model.predict(self.scale(X[t,:],M=M,m=m))
                        K[t] = sp.amax(model.predict_proba(self.scale(X[t,:],M=M,m=m)),axis=1)
                        
                    else :
                        yp[t] = model.predict(self.scale(X[t,:],M=M,m=m))                    
                        
                        #QgsMessageLog.logMessage('amax from predict proba is : '+str(sp.amax(model.predict.proba(self.scale(X[t,:],M=M,m=m)),axis=1)))
                    
                    
                # Write the data
                out.WriteArray(yp.reshape(lines,cols),j,i)
                out.FlushCache()
                
                if confidenceMap :
                    out_confidenceMap.WriteArray(K.reshape(lines,cols),j,i)
                    out_confidenceMap.FlushCache()
                
        
                del X,yp
    
        # Clean/Close variables    
        predictProgress.reset()
        raster = None
        dst_ds = None
        return outRaster
        
        
class progressBar():
    """!@brief Manage progressBar and loading cursor.
    Allow to add a progressBar in Qgis and to change cursor to loading
    input:
        -inMsg : Message to show to the user (str)
        -inMax : The steps of the script (int)
    
    output:
        nothing but changing cursor and print progressBar inside Qgis
    """
    def __init__(self,inMsg=' Loading...',inMaxStep=1):
            # initialize progressBar            
            """
            """# Save reference to the QGIS interface
            QApplication.processEvents() # Help to keep UI alive
            
            widget = iface.messageBar().createMessage('Please wait  ',inMsg)            
            prgBar = QProgressBar()
            self.prgBar=prgBar
            self.iface=iface

            widget.layout().addWidget(self.prgBar)
            iface.messageBar().pushWidget(widget, iface.messageBar().WARNING)
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # if Max 0 and value 0, no progressBar, only cursor loading
            # default is set to 0
            prgBar.setValue(1)
            # set Maximum for progressBar
            prgBar.setMaximum(inMaxStep)
            
    def addStep(self,step=1):
        """!@brief Add a step to the progressBar
        addStep() simply add +1 to current value of the progressBar
        addStep(3) will add 3 steps
        """
        plusOne=self.prgBar.value()+step
        self.prgBar.setValue(plusOne)
    def reset(self):
        """!@brief Simply remove progressBar and reset cursor
        
        """
        # Remove progressBar and back to default cursor
        self.iface.messageBar().clearWidgets()
        self.iface.mapCanvas().refresh()
        QApplication.restoreOverrideCursor()

class confusionMatrix():
    
    def __init__(self):
        self.confusion_matrix= None
        self.OA= None
        self.Kappa = None
        
        
    def computeStatistics(self,inRaster,inShape,inField):
        progress = progressBar('Computing statistics...',0)
        try:
            rasterized = self.rasterize(inRaster,inShape,inField)
            Yp,Yt = dataraster.get_samples_from_roi(inRaster,rasterized)
            CONF = ai.CONFUSION_MATRIX()
            CONF.compute_confusion_matrix(Yp,Yt)
            self.confusion_matrix = CONF.confusion_matrix
            self.Kappa = CONF.Kappa
            self.OA = CONF.OA
        except:
            QgsMessageLog.logMessage('Error during statitics calculation')
        progress.reset()
        

    def rasterize(self,inRaster,inShape,inField):
        filename = tempfile.mktemp('.tif')        
        data = gdal.Open(inRaster,gdal.GA_ReadOnly)
        shp = ogr.Open(inShape)
        
        lyr = shp.GetLayer()

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(filename,data.RasterXSize,data.RasterYSize, 1,gdal.GDT_Byte)
        dst_ds.SetGeoTransform(data.GetGeoTransform())
        dst_ds.SetProjection(data.GetProjection())
        OPTIONS = 'ATTRIBUTE='+inField
        gdal.RasterizeLayer(dst_ds, [1], lyr, None,options=[OPTIONS])
        data,dst_ds,shp,lyr=None,None,None,None
        
        
        return filename
