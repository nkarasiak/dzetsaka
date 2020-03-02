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

try:
    # when using it from QGIS 3
    from qgis.core import QgsMessageLog
    from PyQt5.QtWidgets import QApplication
    from . import function_dataraster as dataraster
    from . import accuracy_index as ai
    from . import progressBar as pB
except BaseException:
    import accuracy_index as ai
    import function_dataraster as dataraster

import pickle

import os

import tempfile
import numpy as np

from osgeo import (gdal, ogr)


class learnModel:
    def __init__(self, inRaster, inVector, inField='Class', outModel=None, inSplit=100,
                 inSeed=0, outMatrix=None, inClassifier='GMM', extraParam=False, feedback=None):
        """!@brief Learn model with a shp file and a raster image.

        **********
        Parameters
        ----------
        inRaster : Filtered image name ('sample_filtered.tif',str).
        inVector : Name of the training shpfile ('training.shp',str).
        inField : Column name where are stored class number (str).
        inSplit : (int) or str 'SLOO' or 'STAND'
            if 'STAND', extraParam['SLOO'] is by default False, and extraParam['maxIter'] is 5. \n
            if 'SLOO', extraParam['distance'] must be given. extraParam['maxIter'] is False, extraParam['minTrain'] is 0.5 for 50\% \n

            Please specify a extraParam['saveDir'] to save results/confusion matrix.

        inSeed : (int).
        outModel : Name of the model to save, will be compulsory for the 3rd step (classifying).
        outMatrix : Default the name of the file inRaster(minus the extension)_inClassifier_inSeed_confu.csv (str).
        inClassifier : GMM,KNN,SVM, or RF. (str).


        Output
        ----------

        Model file.
        Confusion Matrix.

        """
        # Convert vector to raster
        needXY = True
        pushFeedback('Learning model...', feedback=feedback)
        pushFeedback(0, feedback=feedback)
        total = 100 / 10

        SPLIT = inSplit

        if feedback == 'gui':
            progress = pB.progressBar('Loading...', 6)
        try:
            if isinstance(inRaster, np.ndarray):
                needXY = False
                X = inRaster

                if isinstance(inVector, np.ndarray):
                    Y = inVector
                else:
                    msg = 'You have to give an array for label when using array for raster'
                    pushFeedback(msg, feedback=feedback)

            if extraParam:
                if 'readROIFromVector' in extraParam.keys():
                    if extraParam['readROIFromVector'] is not False:
                        try:
                            from function_vector import readROIFromVector
                            X, Y = readROIFromVector(
                                inVector, extraParam['readROIFromVector'], inField)
                            needXY = False

                        except BaseException:
                            msg = 'Problem when importing readFieldVector from functions in dzetsaka'
                            pushFeedback(msg, feedback=feedback)
                if 'saveDir' in extraParam.keys():
                    saveDir = extraParam['saveDir']
                    if not os.path.exists(saveDir):
                        os.makedirs(saveDir)
                    if not os.path.exists(os.path.join(saveDir, 'matrix/')):
                        os.makedirs(os.path.join(saveDir, 'matrix/'))

            inVectorTest = False
            if isinstance(SPLIT, str):
                if SPLIT.endswith(('.shp', '.sqlite')):
                    inVectorTest = SPLIT

            if needXY:
                ROI = rasterize(inRaster, inVector, inField)

            if inVectorTest:
                ROIt = rasterize(inRaster, inVectorTest, inField)
                X, Y = dataraster.get_samples_from_roi(inRaster, ROI)
                Xt, yt = dataraster.get_samples_from_roi(inRaster, ROIt)
                xt, N, n = self.scale(Xt)
                #x,y = dataraster.get_samples_from_roi(inRaster,ROI,getCoords=True,convertTo4326=True)
                y = Y
            
            # Create temporary data set
            if SPLIT == 'SLOO':

                from sklearn.metrics import confusion_matrix

                try:
                    from function_vector import distanceCV, distMatrix
                except BaseException:
                    from .function_vector import distanceCV, distMatrix
                from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

                """
                distanceFile = os.path.splitext(inVector)[0]+'_'+str(inField)+'_distMatrix.npy'
                if os.path.exists(distanceFile):
                    print('Distance array loaded')
                    distanceArray = np.load(distanceFile)
                    X,Y =  dataraster.get_samples_from_roi(inRaster,ROI)
                else:
                    print('Generate distance array')
                """

                if 'readROIFromVector' in extraParam.keys():
                    if extraParam['readROIFromVector'] is not False:
                        try:
                            coords = extraParam['coords']
                        except BaseException:
                            pushFeedback(
                                'Can\'t read coords array', feedback=feedback)
                    else:
                        X, Y, coords = dataraster.get_samples_from_roi(
                            inRaster, ROI, getCoords=True)
                try:
                    coords = extraParam['coords']
                except BaseException:
                    X, Y, coords = dataraster.get_samples_from_roi(
                        inRaster, ROI, getCoords=True)
                
                distanceArray = distMatrix(coords)
                # np.save(os.path.splitext(distanceFile)[0],distanceArray)

            else:
                if SPLIT == 'STAND':

                    from sklearn.metrics import confusion_matrix
                    try:
                        from .function_vector import standCV  # ,readFieldVector
                    except BaseException:
                        from function_vector import standCV  # ,readFieldVector
                    try:
                        from sklearn.metrics import accuracy_score
                    except BaseException:
                        pass

                    if 'inStand' in extraParam.keys():
                        inStand = extraParam['inStand']
                    else:
                        inStand = 'stand'
                    STAND = rasterize(inRaster, inVector, inStand)
                    X, Y, STDs = dataraster.get_samples_from_roi(
                        inRaster, ROI, STAND)
                    #ROIStand = rasterize(inRaster,inVector,inStand)
                    #temp, STDs = dataraster.get_samples_from_roi(inRaster,ROIStand)

                    # FIDs,STDs,srs=readFieldVector(inVector,inField,inStand,getFeatures=False)

                elif needXY:
                    X, Y = dataraster.get_samples_from_roi(inRaster, ROI)
                

        except BaseException:
            msg = "Problem with getting samples from ROI \n \
                Are you sure to have only integer values in your " + str(inField) + " field ?\n  "
            pushFeedback(msg, feedback=feedback)
            pB.reset()
            
        [n, d] = X.shape
        C = int(Y.max())
        SPLIT = inSplit

        try:
            #pushFeedback(str(ROI),feedback=feedback)
            os.remove(ROI)
        except BaseException:
            pass
        # os.remove(filename)
        # os.rmdir(temp_folder)

        # Scale the data
        X, M, m = self.scale(X)

        pushFeedback(int(1 * total))
        if feedback == 'gui':
            progress.addStep()  # Add Step to ProgressBar
        # Learning process take split of groundthruth pixels for training and
        # the remaining for testing

        try:
            if isinstance(SPLIT, int) or isinstance(SPLIT, float):
                if SPLIT < 100:

                    # Random selection of the sample
                    x = np.array([]).reshape(0, d)
                    y = np.array([]).reshape(0, 1)
                    xt = np.array([]).reshape(0, d)
                    yt = np.array([]).reshape(0, 1)

                    np.random.seed(inSeed)  # Set the random generator state
                    for i in range(C):
                        t = np.where((i + 1) == Y)[0]
                        nc = t.size
                        ns = int(nc * (SPLIT / float(100)))
                        rp = np.random.permutation(nc)
                        x = np.concatenate((X[t[rp[0:ns]], :], x))
                        xt = np.concatenate((X[t[rp[ns:]], :], xt))
                        y = np.concatenate((Y[t[rp[0:ns]]], y))
                        yt = np.concatenate((Y[t[rp[ns:]]], yt))

                else:
                    x, y = X, Y
                    self.x = x
                    self.y = y
            else:
                x, y = X, Y
                self.x = x
                self.y = y
        except BaseException:
            pushFeedback(
                "Problem while learning if SPLIT <1",
                feedback=feedback)

        pushFeedback(int(2 * total), feedback=feedback)
        if feedback == 'gui':
            progress.addStep()

        pushFeedback('Learning process...', feedback=feedback)
        pushFeedback(
            'This step could take a lot of time... So be patient, even if the progress bar stucks at 20% :)',
            feedback=feedback)
        
        if feedback == 'gui':
            progress.addStep()  # Add Step to ProgressBar
        # Train Classifier
        if inClassifier == 'GMM':
            try:
                from . import gmm_ridge as gmmr
            except BaseException:
                import gmm_ridge as gmmr

            try:
                # tau=10.0**sp.arange(-8,8,0.5)
                model = gmmr.GMMR()
                model.learn(x, y)
                # htau,err = model.cross_validation(x,y,tau)
                # model.tau = htau
            except BaseException:
                pushFeedback("Cannot train with GMM", feedback=feedback)
        else:

            #from sklearn import neighbors
            #from sklearn.svm import SVC
            #from sklearn.ensemble import RandomForestClassifier

            #model_selection = True
            from sklearn.model_selection import StratifiedKFold
            from sklearn.model_selection import GridSearchCV

            try:

                if extraParam:
                    if 'param_algo' in extraParam.keys():
                        param_algo = extraParam['param_algo']

                # AS Qgis in Windows doensn't manage multiprocessing, force to
                # use 1 thread for not linux system

                if SPLIT == 'STAND':
                    label = np.copy(Y)

                    if extraParam:
                        if 'SLOO' in extraParam.keys():
                            SLOO = extraParam['SLOO']
                        else:
                            SLOO = False
                        if 'maxIter' in extraParam.keys():
                            maxIter = extraParam['maxIter']
                        else:
                            maxIter = 5
                    else:
                        SLOO = False
                        maxIter = 5

                    rawCV = standCV(label, STDs, maxIter, SLOO, seed=inSeed)
                    print(rawCV)
                    cvDistance = []
                    for tr, vl in rawCV:
                        # sts.append(stat)
                        cvDistance.append((tr, vl))

                if SPLIT == 'SLOO':
                    # Compute CV for Learning later

                    label = np.copy(Y)
                    if extraParam:
                        if 'distance' in extraParam.keys():
                            distance = extraParam['distance']
                        else:
                            pushFeedback(
                                'You need distance in extraParam', feedback=feedback)

                        if 'minTrain' in extraParam.keys():
                            minTrain = float(extraParam['minTrain'])
                        else:
                            minTrain = -1

                        if 'SLOO' in extraParam.keys():
                            SLOO = extraParam['SLOO']
                        else:
                            SLOO = True

                        if 'maxIter' in extraParam.keys():
                            maxIter = extraParam['maxIter']
                        else:
                            maxIter = False

                        if 'otherLevel' in extraParam.keys():
                            otherLevel = extraParam['otherLevel']
                        else:
                            otherLevel = False
                    #sts = []
                    cvDistance = []

                    """
                    rawCV = distanceCV(distanceArray,label,distanceThresold=distance,minTrain=minTrain,SLOO=SLOO,maxIter=maxIter,verbose=False,stats=False)

                    """
                    #feedback.setProgressText('distance is '+str(extraParam['distance']))
                    pushFeedback('label is ' +
                                 str(label.shape), feedback=feedback)
                    pushFeedback('distance array shape is ' +
                                 str(distanceArray.shape), feedback=feedback)
                    pushFeedback(
                        'minTrain is ' + str(minTrain),
                        feedback=feedback)
                    pushFeedback('SLOO is ' + str(SLOO), feedback=feedback)
                    pushFeedback(
                        'maxIter is ' + str(maxIter),
                        feedback=feedback)

                    rawCV = distanceCV(
                        distanceArray,
                        label,
                        distanceThresold=distance,
                        minTrain=minTrain,
                        SLOO=SLOO,
                        maxIter=maxIter,
                        stats=False)

                    pushFeedback(
                        'Computing SLOO Cross Validation',
                        feedback=feedback)

                    for tr, vl in rawCV:

                        pushFeedback('Training size is ' +
                                     str(tr.shape), feedback=feedback)
                        pushFeedback('Validation size is ' +
                                     str(vl.shape), feedback=feedback)
                        # sts.append(stat)
                        cvDistance.append((tr, vl))
                    """
                    for tr,vl,stat in rawCV :
                        sts.append(stat)
                        cvDistance.append((tr,vl))
                    """
                    #

                if inClassifier == 'RF':

                    from sklearn.ensemble import RandomForestClassifier

                    param_grid = dict(
                        n_estimators=3**np.arange(
                            1, 5), max_features=range(
                            1, x.shape[1], int(
                                x.shape[1] / 3)))
                    if 'param_algo' in locals():
                        classifier = RandomForestClassifier(**param_algo)
                    else:
                        classifier = RandomForestClassifier()
                    n_splits = 5

                elif inClassifier == 'SVM':
                    from sklearn.svm import SVC

                    param_grid = dict(
                        gamma=2.0**np.arange(-4, 4), C=10.0**np.arange(-2, 5))

                    if 'param_algo' in locals():
                        classifier = SVC(probability=True, **param_algo)
                        print('Found param algo : ' + str(param_algo))
                    else:
                        classifier = SVC(probability=True, kernel="rbf")
                    n_splits = 5

                elif inClassifier == 'KNN':
                    from sklearn import neighbors

                    param_grid = dict(n_neighbors=np.arange(1, 20, 4))
                    if 'param_algo' in locals():
                        classifier = neighbors.KNeighborsClassifier(
                            **param_algo)
                    else:
                        classifier = neighbors.KNeighborsClassifier()

                    n_splits = 3

            except BaseException:
                pushFeedback(
                    "Cannot train with classifier " +
                    inClassifier,
                    feedback=feedback)
                pB.reset()

            if feedback == 'gui':
                progress.prgBar.setValue(5)  # Add Step to ProgressBar

            if isinstance(SPLIT, int):
                cv = StratifiedKFold(n_splits=n_splits)  # .split(x,y)
            else:
                cv = cvDistance

            y.shape = (y.size,)

            if extraParam:
                if 'param_grid' in extraParam.keys():
                    param_grid = extraParam['param_grid']

                    pushFeedback(
                        'Custom param for Grid Search CV has been found : ' +
                        str(param_grid),
                        feedback=feedback)

            grid = GridSearchCV(
                classifier,
                param_grid=param_grid,
                cv=cv)
            grid.fit(x, y)
            model = grid.best_estimator_
            model.fit(x, y)

            if isinstance(SPLIT, str):
                CM = []
                testIndex = []
                for train_index, test_index in cv:

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    X_pred = model.predict(X_test)
                    CM.append(confusion_matrix(y_test, X_pred))
                    testIndex.append(test_index)
                for i, j in enumerate(CM):
                    if SPLIT == 'SLOO':
                        # np.savetxt((saveDir+'matrix/'+str(distance)+'_'+str(inField)+'_'+str(minTrain)+'_'+str(i)+'.csv'),CM[i],delimiter=',',fmt='%.d')
                        np.savetxt(
                            os.path.join(
                                saveDir,
                                'matrix/' +
                                str(distance) +
                                '_' +
                                str(inField) +
                                '_' +
                                str(minTrain) +
                                '_' +
                                str(i) +
                                '.csv'),
                            CM[i],
                            delimiter=',',
                            fmt='%.d')
                        if otherLevel is not False:
                            otherLevelFolder = os.path.join(
                                saveDir, 'matrix/level3/')
                            if not os.path.exists(otherLevelFolder):
                                os.makedirs(otherLevelFolder)
                            bigCM = np.zeros([14, 14], dtype=np.byte)

                            arr = CM[i]
                            curLevel = otherLevel[testIndex[i]]
                            curLevel = np.sort(curLevel, axis=0)
                            for lvl in range(curLevel.shape[0]):
                                bigCM[curLevel.astype(
                                    int) - 1, curLevel[lvl].astype(int) - 1] = arr[:, lvl].reshape(-1, 1)
                            np.savetxt(
                                os.path.join(
                                    otherLevelFolder,
                                    str(distance) +
                                    '_' +
                                    str(inField) +
                                    '_' +
                                    str(minTrain) +
                                    '_' +
                                    str(i) +
                                    '.csv'),
                                bigCM,
                                delimiter=',',
                                fmt='%.d')

                    elif SPLIT == 'STAND':
                        # np.savetxt((saveDir+'matrix/stand_'+str(inField)+'_'+str(i)+'.csv'),CM[i],delimiter=',',fmt='%.d')
                        np.savetxt(
                            os.path.join(
                                saveDir,
                                'matrix/stand_' +
                                str(inField) +
                                '_' +
                                str(i) +
                                '.csv'),
                            CM[i],
                            delimiter=',',
                            fmt='%.d')

        pushFeedback(int(9 * total), feedback=feedback)

        # Assess the quality of the model
        if feedback == 'gui':
            progress.prgBar.setValue(90)

        if inVectorTest or isinstance(SPLIT, int):
            if SPLIT != 100 or inVectorTest:
                #from sklearn.metrics import cohen_kappa_score,accuracy_score,f1_score
                # if  inClassifier == 'GMM':
                #          = model.predict(xt)[0]
                # else:
                yp = model.predict(xt)
                CONF = ai.CONFUSION_MATRIX()
                CONF.compute_confusion_matrix(yp, yt)

                if outMatrix is not None:
                    if not os.path.exists(os.path.dirname(outMatrix)):
                        os.makedirs(os.path.dirname(outMatrix))
                    np.savetxt(
                        outMatrix,
                        CONF.confusion_matrix,
                        delimiter=',',
                        header='Columns=prediction,Lines=reference.',
                        fmt='%1.4d')

                if inClassifier != 'GMM':
                    for key in param_grid.keys():
                        message = 'best ' + key + ' : ' + \
                            str(grid.best_params_[key])
                        if feedback == 'gui':
                            QgsMessageLog.logMessage(message)
                        elif feedback:
                            feedback.setProgressText(message)
                        else:
                            print(message)

                """
                self.kappa = cohen_kappa_score(yp,yt)
                self.f1 = f1_score(yp,yt,average='micro')
                self.oa = accuracy_score(yp,yt)
                """
                res = {
                    'Overall Accuracy': CONF.OA,
                    'Kappa': CONF.Kappa,
                    'f1': CONF.F1mean}

                for estim in res:
                    pushFeedback(estim + ' : ' +
                                 str(res[estim]), feedback=feedback)

        # Save Tree model
        self.model = model
        self.M = M
        self.m = m
        if outModel is not None:
            output = open(outModel, 'wb')
            pickle.dump([model, M, m, inClassifier], output)
            output.close()

        pushFeedback(int(10 * total), feedback=feedback)
        if feedback == 'gui':
            progress.reset()
            progress = None

    def scale(self, x, M=None, m=None):
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
        [n, d] = x.shape
        if not np.float64 == x.dtype.type:
            x = x.astype('float')

        # Initialization of the output
        xs = np.empty_like(x)

        # get the parameters of the scaling
        M, m = np.amax(x, axis=0), np.amin(x, axis=0)
        den = M - m
        for i in range(d):
            if den[i] != 0:
                xs[:, i] = 2 * (x[:, i] - m[i]) / den[i] - 1
            else:
                xs[:, i] = x[:, i]

        return xs, M, m


class classifyImage(object):
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

    def initPredict(self, inRaster, inModel, outRaster, inMask=None, confidenceMap=None,
                    confidenceMapPerClass=None, NODATA=0, feedback=None):

        # Load model

        try:
            model = open(inModel, 'rb')  # TODO: Update to scale the data
            if model is None:
                # fix_print_with_import
                print("Model not load")
                pushFeedback("Model : " + inModel + " is none")
            else:
                tree, M, m, classifier = pickle.load(model)

                model.close()
        except BaseException:
            pushFeedback(
                "Error while loading the model : " +
                inModel,
                feedback=feedback)

        # Creating temp file for saving raster classification
        try:
            temp_folder = tempfile.mkdtemp()
            rasterTemp = os.path.join(temp_folder, 'temp.tif')
        except BaseException:
            pushFeedback(
                "Cannot create temp file " +
                rasterTemp,
                feedback=feedback)
            # Process the data
        # try:
        predictedImage = self.predict_image(
            inRaster,
            outRaster,
            tree,
            inMask,
            confidenceMap,
            confidenceMapPerClass=confidenceMapPerClass,
            NODATA=NODATA,
            SCALE=[
                M,
                m],
            classifier=classifier,
            feedback=feedback)
        # except:
        #   QgsMessageLog.logMessage("Problem while predicting "+inRaster+" in temp"+rasterTemp)

        return predictedImage

    def scale(self, x, M=None, m=None):  # TODO:  DO IN PLACE SCALING
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
        [n, d] = x.shape
        if np.float64 != x.dtype.type:
            x = x.astype('float')

        # Initialization of the output
        xs = np.empty_like(x)

        # get the parameters of the scaling
        if M is None:
            M, m = np.amax(x, axis=0), np.amin(x, axis=0)

        den = M - m
        for i in range(d):
            if den[i] != 0:
                xs[:, i] = 2 * (x[:, i] - m[i]) / den[i] - 1
            else:
                xs[:, i] = x[:, i]

        return xs

    def predict_image(self, inRaster, outRaster, model=None, inMask=None, confidenceMap=None,
                      confidenceMapPerClass=None, NODATA=0, SCALE=None, classifier='GMM', feedback=None):
        """!@brief The function classify the whole raster image, using per block image analysis.

        The classifier is given in classifier and options in kwargs

            Input :
                inRaster : Filtered image name ('sample_filtered.tif',str)
                outRaster :Raster image name ('outputraster.tif',str)
                model : model file got from precedent step ('model', str)
                inMask : mask to
                confidenceMap :  map of confidence per pixel
                NODATA : Default set to 0 (int)
                SCALE : Default set to None
                classifier = Default 'GMM'

            Output :
                nothing but save a raster image and a confidence map if asked
        """
        # Open Raster and get additionnal information

        raster = gdal.Open(inRaster, gdal.GA_ReadOnly)
        if raster is None:
            # fix_print_with_import
            print('Impossible to open ' + inRaster)
            exit()

        if inMask is None:
            mask = None
        else:
            mask = gdal.Open(inMask, gdal.GA_ReadOnly)
            if mask is None:
                # fix_print_with_import
                print('Impossible to open ' + inMask)
                exit()
            # Check size
            if (raster.RasterXSize != mask.RasterXSize) or (
                    raster.RasterYSize != mask.RasterYSize):
                # fix_print_with_import
                print('Image and mask should be of the same size')
                exit()
        if SCALE is not None:
            M, m = np.asarray(SCALE[0]), np.asarray(SCALE[1])

        # Get the size of the image
        d = raster.RasterCount
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

        # Initialize the output
        if not os.path.exists(os.path.dirname(outRaster)):
            os.makedirs(os.path.dirname(outRaster))

        driver = gdal.GetDriverByName('GTiff')
        
        if np.amax(model.classes_)>255:
            dtype = gdal.GDT_UInt16
        else:
            dtype = gdal.GDT_Byte
        
        
        dst_ds = driver.Create(outRaster, nc, nl, 1, dtype)
        dst_ds.SetGeoTransform(GeoTransform)
        dst_ds.SetProjection(Projection)
        out = dst_ds.GetRasterBand(1)

        if classifier != 'GMM':
            nClass = len(model.classes_)
        if confidenceMap:
            dst_confidenceMap = driver.Create(
                confidenceMap, nc, nl, 1, gdal.GDT_Int16)
            dst_confidenceMap.SetGeoTransform(GeoTransform)
            dst_confidenceMap.SetProjection(Projection)
            out_confidenceMap = dst_confidenceMap.GetRasterBand(1)

        if confidenceMapPerClass:
            dst_confidenceMapPerClass = driver.Create(
                confidenceMapPerClass, nc, nl, nClass, gdal.GDT_Int16)
            dst_confidenceMapPerClass.SetGeoTransform(GeoTransform)
            dst_confidenceMapPerClass.SetProjection(Projection)

        # Perform the classification

        total = nl * y_block_size

        pushFeedback('Predicting model...')

        if feedback == 'gui':
            progress = pB.progressBar('Predicting model...', total / 10)

        for i in range(0, nl, y_block_size):
            if not 'lastBlock' in locals():
                lastBlock = i
            if int(lastBlock / total * 100) != int(i / total * 100):
                lastBlock = i
                pushFeedback(int(i / total * 100))

                if feedback == 'gui':
                    progress.addStep()

            if i + y_block_size < nl:  # Check for size consistency in Y
                lines = y_block_size
            else:
                lines = nl - i
            for j in range(
                    0, nc, x_block_size):  # Check for size consistency in X
                if j + x_block_size < nc:
                    cols = x_block_size
                else:
                    cols = nc - j

                # Load the data and Do the prediction
                X = np.empty((cols * lines, d))
                for ind in range(d):
                    X[:, ind] = raster.GetRasterBand(
                        int(ind + 1)).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # Do the prediction
                band_temp = raster.GetRasterBand(1)
                nodata_temp = band_temp.GetNoDataValue()
                if nodata_temp is None:
                    nodata_temp = -9999

                if mask is None:
                    band_temp = raster.GetRasterBand(1)
                    mask_temp = band_temp.ReadAsArray(
                        j, i, cols, lines).reshape(
                        cols * lines)
                    temp_nodata = np.where(mask_temp != nodata_temp)[0]
                    #t = np.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
                    t = np.where(X[:, 0] != nodata_temp)[0]
                    yp = np.zeros((cols * lines,))
                    #K = np.zeros((cols*lines,))
                    if confidenceMapPerClass or confidenceMap and classifier != 'GMM':
                        K = np.zeros((cols * lines, nClass))
                        K[:, :] = -1
                    else:
                        K = np.zeros((cols * lines))
                        K[:] = -1

                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(
                        j, i, cols, lines).reshape(cols * lines)
                    t = np.where((mask_temp != 0) & (
                        X[:, 0] != nodata_temp))[0]
                    yp = np.zeros((cols * lines,))
                    yp[:] = NODATA
                    #K = np.zeros((cols*lines,))
                    if confidenceMapPerClass or confidenceMap and classifier != 'GMM':
                        K = np.ones((cols * lines, nClass))
                        K = np.negative(K)
                    else:
                        K = np.zeros((cols * lines))
                        K = np.negative(K)

                # TODO: Change this part accorindgly ...
                if t.size > 0:
                    
                    if confidenceMap and classifier == 'GMM':
                        yp[t], K[t] = model.predict(self.scale(
                            X[t, :], M=M, m=m), None, confidenceMap)
                        
                    elif confidenceMap or confidenceMapPerClass and classifier != 'GMM':
                        yp[t] = model.predict(self.scale(X[t, :], M=M, m=m))
                        K[t, :] = model.predict_proba(
                            self.scale(X[t, :], M=M, m=m)) * 100

                    else:
                        yp[t] = model.predict(self.scale(X[t, :], M=M, m=m))
                        

                        #QgsMessageLog.logMessage('amax from predict proba is : '+str(sp.amax(model.predict.proba(self.scale(X[t,:],M=M,m=m)),axis=1)))

                # Write the data
                out.WriteArray(yp.reshape(lines, cols), j, i)
                out.SetNoDataValue(NODATA)
                out.FlushCache()

                if confidenceMap and classifier == 'GMM':
                    K*=100
                    out_confidenceMap.WriteArray(
                        K.reshape(lines, cols), j, i)
                    out_confidenceMap.SetNoDataValue(-1)
                    out_confidenceMap.FlushCache()

                if confidenceMap and classifier != 'GMM':
                    Kconf = np.amax(K, axis=1)
                    out_confidenceMap.WriteArray(
                        Kconf.reshape(lines, cols), j, i)
                    out_confidenceMap.SetNoDataValue(-1)
                    out_confidenceMap.FlushCache()

                if confidenceMapPerClass and classifier != 'GMM':
                    for band in range(nClass):
                        gdalBand = band + 1
                        out_confidenceMapPerClass = dst_confidenceMapPerClass.GetRasterBand(
                            gdalBand)
                        out_confidenceMapPerClass.SetNoDataValue(-1)
                        out_confidenceMapPerClass.WriteArray(
                            K[:, band].reshape(lines, cols), j, i)
                        out_confidenceMapPerClass.FlushCache()

                del X, yp

        # Clean/Close variables
        if feedback == 'gui':
            progress.reset()

        raster = None
        dst_ds = None
        return outRaster


class confusionMatrix(object):

    def __init__(self):
        self.confusion_matrix = None
        self.OA = None
        self.Kappa = None

    def computeStatistics(self, inRaster, inShape, inField):
        try:
            rasterized = rasterize(inRaster, inShape, inField)
            Yp, Yt = dataraster.get_samples_from_roi(inRaster, rasterized)
            CONF = ai.CONFUSION_MATRIX()
            CONF.compute_confusion_matrix(Yp, Yt)
            self.confusion_matrix = CONF.confusion_matrix
            self.Kappa = CONF.Kappa
            self.OA = CONF.OA
        except BaseException:
            pushFeedback('Error during statitics calculation')


def rasterize(inRaster, inShape, inField):
    filename = tempfile.mktemp('.tif')
    data = gdal.Open(inRaster, gdal.GA_ReadOnly)
    shp = ogr.Open(inShape)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        filename,
        data.RasterXSize,
        data.RasterYSize,
        1,
        gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(data.GetGeoTransform())
    dst_ds.SetProjection(data.GetProjection())
    OPTIONS = 'ATTRIBUTE=' + inField
    gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=[OPTIONS])
    data, dst_ds, shp, lyr = None, None, None, None

    return filename


def pushFeedback(message, feedback=None):
    isNum = isinstance(message, (float, int))

    if feedback and feedback is not True:
        if feedback == 'gui':
            if not isNum:
                QgsMessageLog.logMessage(str(message))
        else:
            if isNum:
                feedback.setProgress(message)
            else:
                feedback.setProgressText(message)
    else:
        if not isNum:
            print(str(message))
        """
        else:
            print(52*"=")
            print(((int(message/2)-3)*'-'+(str(message)+'%')))
            print(52*"=")
        """


if __name__ == "__main__":

    INPUT_RASTER = "/mnt/DATA/demo/map.tif"
    INPUT_LAYER = "/mnt/DATA/demo/train.shp"
    INPUT_COLUMN = "Class"
    OUTPUT_MODEL = "/mnt/DATA/demo/test/model.RF"
    SPLIT_PERCENT = 50
    OUTPUT_MATRIX = '/mnt/DATA/demo/test/matrix.csv'
    SELECTED_ALGORITHM = 'RF'
    OUTPUT_CONFIDENCE = "/mnt/DATA/demo/test/confidence.tif"
    INPUT_MASK = None
    OUTPUT_RASTER = "/mnt/DATA/demo/test/class.tif"

    temp = learnModel(
        INPUT_RASTER,
        INPUT_LAYER,
        INPUT_COLUMN,
        OUTPUT_MODEL,
        SPLIT_PERCENT,
        0,
        OUTPUT_MATRIX,
        SELECTED_ALGORITHM,
        extraParam=None,
        feedback=None)
    print('learned')
    temp = classifyImage()
    temp.initPredict(
        INPUT_RASTER,
        OUTPUT_MODEL,
        OUTPUT_RASTER,
        INPUT_MASK,
        OUTPUT_CONFIDENCE)
    print('clfied')

    Test = 'SLOO'

    if Test == 'STAND':
        extraParam = {}
        extraParam['inStand'] = 'Stand'
        extraParam['saveDir'] = '/tmp/test1/'
        extraParam['maxIter'] = 5
        extraParam['SLOO'] = False
        learnModel(
            INPUT_RASTER,
            INPUT_LAYER,
            INPUT_COLUMN,
            OUTPUT_MODEL,
            inSplit='STAND',
            inSeed=0,
            outMatrix=None,
            inClassifier=SELECTED_ALGORITHM,
            feedback=None,
            extraParam=extraParam)
    if Test == 'SLOO':
        INPUT_RASTER = "/mnt/DATA/Test/DA/SITS/SITS_2013.tif"
        INPUT_LAYER = "/mnt/DATA/Test/DA/ROI_2154.sqlite"
        INPUT_COLUMN = "level1"

        extraParam = {}
        extraParam['distance'] = 100
        extraParam['maxIter'] = 5
        extraParam['saveDir'] = '/tmp/'
        learnModel(
            INPUT_RASTER,
            INPUT_LAYER,
            INPUT_COLUMN,
            OUTPUT_MODEL,
            inSplit='SLOO',
            inSeed=0,
            outMatrix=None,
            inClassifier=SELECTED_ALGORITHM,
            feedback=None,
            extraParam=extraParam)
