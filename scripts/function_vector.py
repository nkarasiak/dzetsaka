"""@author: nkarasiak.

https://github.com/nkarasiak/dzetsaka.
"""

import os

# import random
try:
    from osgeo import ogr
except ImportError:
    import ogr
import numpy as np


class RandomInSubset:
    """Generate random samples from vector data for training/validation split."""

    def __init__(self, inShape, inField, outValidation, outTrain, number=50, percent=True):
        """Initialize RandomInSubset for splitting training data.

        InShape : str path file (e.g. '/doc/ref.shp')
        inField : string column name (e.g. 'class')
        outValidation : str path of shp output file (e.g. '/tmp/valid.shp')
        outTrain : str path of shp output file (e.g. '/tmp/train.shp').
        """
        from sklearn.model_selection import train_test_split

        number = number / 100.0 if percent else int(number)

        lyr = ogr.Open(inShape)
        lyr1 = lyr.GetLayer()
        FIDs = np.zeros(lyr1.GetFeatureCount(), dtype=int)
        Features = []
        # unselFeat = []
        # current = 0

        for i, j in enumerate(lyr1):
            # print j.GetField(inField)
            FIDs[i] = j.GetField(inField)
            Features.append(j)
            # current += 1
        srs = lyr1.GetSpatialRef()
        lyr1.ResetReading()

        ##
        if percent:
            validation, train = train_test_split(Features, test_size=number, train_size=1 - number, stratify=FIDs)
        else:
            validation, train = train_test_split(Features, test_size=number, stratify=FIDs)

        self.saveToShape(validation, srs, outValidation)
        self.saveToShape(train, srs, outTrain)

    def saveToShape(self, array, srs, outShapeFile):
        """Save array data to shapefile."""
        # Parse a delimited text file of volcano data and create a shapefile
        # use a dictionary reader so we can access by field name
        # set up the shapefile driver
        outDriver = ogr.GetDriverByName("ESRI Shapefile")

        # create the data source
        if os.path.exists(outShapeFile):
            outDriver.DeleteDataSource(outShapeFile)
        # Remove output shapefile if it already exists

        # options = ['SPATIALITE=YES'])
        ds = outDriver.CreateDataSource(outShapeFile)

        # create the spatial reference, WGS84

        lyrout = ds.CreateLayer("randomSubset", srs)
        fields = [array[1].GetFieldDefnRef(i).GetName() for i in range(array[1].GetFieldCount())]

        for f in fields:
            field_name = ogr.FieldDefn(f, ogr.OFTString)
            field_name.SetWidth(24)
            lyrout.CreateField(field_name)

        for k in array:
            lyrout.CreateFeature(k)

        # Save and close the data source
        ds = None


class DistanceCV:
    """Cross-validation based on spatial distance constraints."""

    def __init__(
        self,
        distanceArray,
        Y,
        distanceThresold=1000,
        minTrain=-1,
        SLOO=True,
        maxIter=False,
        furtherSplit=False,
        onlyVaryingTrain=False,
        stats=False,
        verbose=False,
        seed=False,
        otherLevel=False,
    ):
        """Compute train/validation array with Spatial distance analysis.

        Object stops when less effective class number is reached (45 loops if your least class contains 45 ROI).

        Parameters
        ----------
        distanceArray : array
            Matrix distance

        Y : array-like
            contain classe for each ROI. Same effective as distanceArray.

        distanceThresold : int or float
            Distance(same unit of your distanceArray)

        minTrain : int or float
            if >1 : keep n ROI beyond distanceThresold
            if float <1 : minimum percent of samples to use for traning. Use 1 for use only distance Thresold.
            if -1 : same size
        SLOO : Spatial Leave One Out, keep on single validation pixel.
            SLOO=True: Skcv (if maxIter=False, skcv is SLOO from Kevin Le Rest, or SKCV from Pohjankukka)
        maxIter :
            False : as loop as min effective class
        furtherSplit : bool, optional
            Whether to perform further splitting. Default is False.
        onlyVaryingTrain : bool, optional
            Whether to only use varying training data. Default is False.
        stats : bool, optional
            Whether to compute statistics. Default is False.
        verbose : bool, optional
            Whether to print verbose output. Default is False.
        seed : bool or int, optional
            Random seed for reproducibility. Default is False.
        otherLevel : bool, optional
            Whether to use other level analysis. Default is False.

        Returns
        -------
        train : array
            List of Y selected ROI for train

        validation : array
            List of Y selected ROI for validation

        """
        self.distanceArray = distanceArray
        self.distanceThresold = distanceThresold
        self.label = np.copy(Y)
        self.T = np.copy(Y)
        self.iterPos = 0
        self.minTrain = minTrain
        self.onlyVaryingTrain = onlyVaryingTrain
        if self.onlyVaryingTrain:
            self.validation = np.array([]).astype("int")
        self.minEffectiveClass = min([len(Y[i == Y]) for i in np.unique(Y)])
        if maxIter:
            self.maxIter = maxIter
        else:
            self.maxIter = self.minEffectiveClass
        self.otherLevel = otherLevel
        self.SLOO = SLOO  # Leave One OUT
        self.verbose = verbose
        self.furtherSplit = furtherSplit
        self.stats = stats

        if seed:
            np.random.seed(seed)

    def __iter__(self):
        """Return iterator for cross-validation splits."""
        return self

    # python 3 compatibility
    def __next__(self):
        """Return next cross-validation split for Python 3 compatibility."""
        return self.next()

    def next(self):
        """Get the next cross-validation split."""
        # global CTtoRemove,trained,validate,validation,train,CT,distanceROI

        if self.iterPos < self.maxIter:
            ROItoRemove = []
            for _iterPosition in range(self.maxIter):
                if self.verbose:
                    print((53 * "=" + "\n") * 4)

                validation = np.array([]).astype("int")
                train = np.array([]).astype("int")

                # sp.random.shuffle(self.T)

                if self.stats:
                    # Cstats = sp.array([]).reshape(0,9   )
                    Cstats = []

                for C in np.unique(self.label):
                    # Y is True, where C is the unique class
                    CT = np.where(self.T == C)[0]

                    CTtemp = np.copy(CT)

                    if self.verbose:
                        print("C = " + str(C))
                        print("len total class : " + str(len(CT)))
                        fmt = "" if self.minTrain > 1 else ".0%"

                    trained = np.array([]).astype("int")
                    validate = np.array([]).astype("int")

                    while len(CTtemp) > 0:
                        # totalC = len(self.label[self.label==int(C)])
                        # uniqueTrain = 0

                        self.ROI = np.random.permutation(CTtemp)[0]  # randomize ROI choice

                        # while uniqueTrain <(self.split*totalC) :
                        # sameClass = sp.where( self.Y[CT] == C )
                        distanceROI = (self.distanceArray[int(self.ROI), :])[
                            CTtemp
                        ]  # get line of distance for specific ROI

                        if self.minTrain == -1:
                            # distToCutValid = sp.sort(distanceROI)[:self.maxIter][-1] # get distance where to split train/valid
                            # distToCutTrain =
                            # sp.sort(distanceROI)[-self.maxIter:][0] # get
                            # distance where to split train/valid

                            trainedTemp = np.array([self.ROI])

                            validateTemp = CTtemp[CTtemp != trainedTemp][
                                distanceROI[distanceROI > 0.1] >= self.distanceThresold
                            ]
                            # CTtemp[distanceROI>=self.distanceThresold] #
                            # trained > distance to cut

                        if self.maxIter == self.minEffectiveClass:
                            trainedTemp = np.array([self.ROI])

                            validateTemp = CTtemp[distanceROI > self.distanceThresold]
                            # trainedTemp = trainedTemp[trainedTemp!=self.ROI]

                        """
                        elif self.SLOO:
                            validateTemp = sp.array([self.ROI]) # validate ROI
                            trainedTemp = CTtemp[(distanceROI>=self.distanceThresold)] # Train in a buffer
                        """
                        # trainedTemp = sp.array([self.ROI]) # train is the current ROI
                        """
                        if self.SLOO is True and self.maxIter != self.minEffectiveClass:
                            #print('self.SLOO true but no LeRest')
                            print('ici')
                            CTtoRemove = np.concatenate((validateTemp,trainedTemp))

                            # Remove ROI for further selection ROI (but keep in Y list)
                            for i in np.nditer(CTtoRemove):
                                CTtemp = np.delete(CTtemp,np.where(CTtemp==i)[0])

                            #if self.verbose : print('len CTtemp is : '+str(len(CTtemp)))

                            trained = np.concatenate((trained,trainedTemp))
                            validate = np.concatenate((validate,validateTemp))


                        else:
                            trained = trainedTemp
                            validate = validateTemp



                            CTtemp = []
                        """
                        trained = trainedTemp
                        validate = validateTemp

                        CTtemp = []
                        # print len(validate)
                    initTrain = len(trained)
                    initValid = len(validate)

                    if self.minTrain > 1:
                        if len(trained) != self.minTrain:
                            # get number of ROI to keep
                            indToCut = len(CT) - int(self.minTrain)
                            # get distance where to split train/valid
                            distToCut = np.sort(distanceROI)[indToCut]
                            # trained > distance to cut
                            trained = CT[distanceROI >= distToCut]

                            if self.SLOO:  # with SLOO we keep 1 single validation ROI
                                trained = np.random.permutation(trained)[0 : self.minTrain]
                            else:
                                if self.verbose:
                                    print(
                                        "len validate before split ("
                                        + format(self.minTrain, fmt)
                                        + ") : "
                                        + str(len(validate))
                                    )
                                validate = CT[distanceROI <= distToCut]

                    elif self.onlyVaryingTrain:
                        if self.verbose:
                            print("only varying train size : First Time init")
                        if len(validate) > int(self.onlyVaryingTrain * len(CT)):
                            nValidateToRemove = int(len(validate) - self.onlyVaryingTrain * len(CT))
                            indToMove = np.random.permutation(trained)[:nValidateToRemove]
                            for i in indToMove:
                                validate = np.delete(trained, np.where(trained == i)[0])
                            trained = np.concatenate((trained, indToMove))

                        elif len(validate) < int(self.onlyVaryingTrain * len(CT)):
                            nValidToAdd = int(self.minTrain * len(CT) - len(trained))

                            indToMove = np.random.permutation(validate)[:nValidToAdd]
                            for i in indToMove:
                                trained = np.delete(validate, np.where(validate == i)[0])
                            validate = np.concatenate((trained, indToMove))

                        elif len(trained) > int(self.minTrain * (len(CT))):
                            nTrainToRemove = int(self.minTrain * len(CT) - len(trained))

                            indToMove = np.random.permutation(validate)[:nTrainToRemove]
                            for i in indToMove:
                                trained = np.delete(validate, np.where(validate == i)[0])
                            validate = np.concatenate((trained, indToMove))

                        elif len(trained) < int(self.minTrain * (len(CT))):
                            nTrainToAdd = int(self.minTrain * len(CT) - len(trained))

                            indToMove = np.random.permutation(validate)[:nTrainToAdd]
                            for i in indToMove:
                                validate = np.delete(validate, np.where(validate == i)[0])
                            trained = np.concatenate((trained, indToMove))

                    elif self.minTrain != -1 and self.minTrain != 0 and not self.onlyVaryingTrain:
                        initTrain = len(trained)
                        initValid = len(validate)
                        # if train size if less than split% of whole class
                        # (i.e. 30% for exemple)
                        if (len(trained) != self.minTrain * len(CT)) or (self.SLOO and len(trained) == 0):
                            if self.verbose:
                                print("len trained before " + format(self.minTrain, fmt) + " : " + str(len(trained)))

                            # distanceROI = (self.distanceArray[int(self.ROI),:])[CT]
                            if self.furtherSplit:
                                if len(trained) > self.minTrain * len(CT):
                                    nTrainToRemove = int(len(trained) - self.minTrain * len(CT))
                                    distanceROI = (self.distanceArray[int(np.random.permutation(trained)[0]), :])[
                                        trained
                                    ]

                                    distToMove = np.sort(distanceROI)[nTrainToRemove]
                                    # indToMove = distToMove[distanceROI]
                                    # trained > distance to cut
                                    indToMove = trained[distanceROI >= distToMove]
                                    for i in indToMove:
                                        trained = np.delete(trained, np.where(trained == i)[0])
                                    validate = np.concatenate((validate, indToMove))
                                else:
                                    nTrainToAdd = int(self.minTrain * len(CT) - len(trained))

                                    distanceROI = (self.distanceArray[int(np.random.permutation(validate)[0]), :])[
                                        validate
                                    ]
                                    distToMove = np.sort(distanceROI)[-nTrainToAdd]
                                    # indToMove = distToMove[distanceROI]
                                    # trained > distance to cut
                                    indToMove = validate[distanceROI >= distToMove]
                                    for i in indToMove:
                                        validate = np.delete(validate, np.where(validate == i)[0])
                                    trained = np.concatenate((trained, indToMove))

                            else:
                                if len(trained) > self.minTrain * len(CT):
                                    nTrainToRemove = int(len(trained) - self.minTrain * len(CT))
                                    indToMove = np.random.permutation(trained)[:nTrainToRemove]
                                    for i in indToMove:
                                        trained = np.delete(trained, np.where(trained == i)[0])
                                    validate = np.concatenate((validate, indToMove))

                                else:
                                    nTrainToAdd = int(self.minTrain * len(CT) - len(trained))

                                    indToMove = np.random.permutation(validate)[:nTrainToAdd]
                                    for i in indToMove:
                                        validate = np.delete(validate, np.where(validate == i)[0])
                                    trained = np.concatenate((trained, indToMove))

                    if self.stats:
                        # CTtemp = sp.where(self.label[trained]==C)[0]
                        CTdistTrain = np.array(self.distanceArray[trained])[:, trained]
                        if len(CTdistTrain) > 1:
                            CTdistTrain = np.mean(np.triu(CTdistTrain)[np.triu(CTdistTrain) != 0])

                        # CTtemp = sp.where(self.label[validate]==C)[0]
                        CTdistValid = np.array(self.distanceArray[validate])[:, validate]
                        CTdistValid = np.mean(np.triu(CTdistValid)[np.triu(CTdistValid) != 0])
                        Cstats.append(
                            [
                                self.distanceThresold,
                                self.minTrain,
                                C,
                                initValid,
                                initTrain,
                                len(trained) - initTrain,
                                CTdistTrain,
                                CTdistValid,
                            ]
                        )

                    if self.verbose:
                        print("len validate : " + str(len(validate)))
                        print("len trained : " + str(len(trained)))

                    validation = np.concatenate((validation, validate))
                    train = np.concatenate((train, trained))
                    # allDist[sp.where(y[allDist]==C)[0]]
                    # T = sp.searchsorted(T,currentClass)
                    # for i in sp.nditer(train):

                    # remove current validation ROI
                    ROItoRemove.append(validation)
                    ROItoRemove.append(train)

                    # Cstats = sp.vstack((Cstats,(self.distanceThresold,self.minTrain*100,C,initTrain,initValid,len(trained)-initTrain,len(validate)-initValid,meanDistTrain,meanDistValidation)))

                if self.stats is True:
                    np.savetxt(
                        self.stats,
                        Cstats,
                        fmt="%d",
                        delimiter=",",
                        header="Distance,Percent Train, Label,Init train,Init valid,Ntrain Add,Mean DisT Train,Mean Dist Valid",
                    )

                    # if not self.SLOO:
                    # validate = CT[distanceROI<distToCut]

                self.iterPos += 1

                if self.verbose:
                    print(53 * "=")
                # Remove ROI for further selection ROI (but keep in Y list)
                """
                for i in ROItoRemove:
                    self.T = sp.delete(self.T,sp.where(self.T==i)[0])
                """
                if self.stats and self.stats is True:
                    return validation, train, Cstats
                else:
                    return validation, train

        else:
            raise StopIteration()


def distMatrix(inCoords, distanceMetric=False):
    """Compute distance matrix between points.

    coords : nparray shape[nPoints,2], with first column X, and Y. Proj 4326(WGS84)
    Return matrix of distance matrix between points.
    """
    if distanceMetric:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")

        distArray = np.zeros((len(inCoords), len(inCoords)))
        for n, p in enumerate(np.nditer(inCoords.T.copy(), flags=["external_loop"], order="F")):
            for i in range(len(inCoords)):
                x1, y1 = p
                x2, y2 = inCoords[i]
                angle1, angle2, dist = geod.inv(x1, y1, x2, y2)

                distArray[n, i] = dist

    else:
        from scipy.spatial import distance

        distArray = distance.cdist(inCoords, inCoords, "euclidean")

    return distArray


def convertToDistanceMatrix(coords, sr=False, convertTo4326=False):
    """Convert coordinates to distance matrix."""
    if convertTo4326:
        from pyproj import Proj, transform

        # initProj = Proj(sr.ExportToProj4())
        # convert points coords to 4326
        # if vector
        initProj = Proj(sr.ExportToProj4())
        # http://epsg.io/4326
        destProj = Proj("+proj=longlat +datum=WGS84 +no_defs")

        coords[:, 0], coords[:, 1] = transform(initProj, destProj, coords[:, 0], coords[:, 1])

    return distMatrix(coords, distanceMetric=True)


class StandCV:
    """Stand-based cross-validation for spatial data analysis."""

    def __init__(self, Y, stand, maxIter=False, SLOO=True, seed=False):
        """Compute train/validation per stand.

        Y : array-like
            contains class for each ROI.
        Stand : array-like
            contains stand number for each ROI.
        maxIter : False or int
            if False, maxIter is the minimum stand number of all species.
        SLOO :  Bool
            True  or False. If SLOO, keep only one Y per validation stand.
        """
        self.Y = Y
        self.uniqueY = np.unique(self.Y)
        self.stand = stand
        self.SLOO = SLOO

        if isinstance(SLOO, bool):
            self.split = 0.5

        else:
            self.split = self.SLOO
        self.maxIter = maxIter
        self.iterPos = 0

        if seed:
            np.random.seed(seed)

        if maxIter:
            self.maxIter = maxIter
        else:
            maxIter = []
            for i in np.unique(Y):
                standNumber = np.unique(np.array(stand)[np.where(np.array(Y) == i)])
                maxIter.append(standNumber.shape[0])
            self.maxIter = np.amin(maxIter)

    def __iter__(self):
        """Return iterator for stand-based cross-validation splits."""
        return self

    # python 3 compatibility
    def __next__(self):
        """Return next stand-based cross-validation split for Python 3 compatibility."""
        return self.next()

    def next(self):
        """Get the next stand-based cross-validation split."""
        if self.iterPos < self.maxIter:
            StandToRemove = []
            train = np.array([], dtype=int)
            validation = np.array([], dtype=int)
            for i in self.uniqueY:
                Ycurrent = np.where(np.array(self.Y) == i)[0]
                Ystands = np.array(self.stand)[Ycurrent]
                Ystand = np.unique(Ystands)

                selectedStand = np.random.permutation(Ystand)[0]

                if self.SLOO:
                    YinSelectedStandt = np.in1d(Ystands, selectedStand)
                    YinSelectedStand = Ycurrent[YinSelectedStandt]
                    validation = np.concatenate((validation, np.asarray(YinSelectedStand)))

                    # improve code...
                    # Ycurrent[sp.where(Ystands!=selectedStand)[0]]

                    YnotInSelectedStandt = np.invert(YinSelectedStandt)
                    YnotInSelectedStand = Ycurrent[YnotInSelectedStandt]
                    train = np.concatenate((train, np.asarray(YnotInSelectedStand)))
                    StandToRemove.append(selectedStand)
                else:
                    randomYstand = np.random.permutation(Ystand)

                    Ytrain = np.in1d(Ystands, randomYstand[: int(len(Ystand) * self.split)])
                    Ytrain = Ycurrent[Ytrain]
                    Yvalidation = np.in1d(Ystands, randomYstand[int(len(Ystand) * self.split) :])
                    Yvalidation = Ycurrent[Yvalidation]

                    train = np.concatenate((train, np.asarray(Ytrain)))
                    validation = np.concatenate((validation, np.asarray(Yvalidation)))

            self.iterPos += 1
            return train, validation
        else:
            raise StopIteration()


def readFieldVector(inShape, inField, inStand=False, getFeatures=False):
    """Read field vector data from shapefile."""
    lyr = ogr.Open(inShape)
    lyr1 = lyr.GetLayer()
    FIDs = np.zeros(lyr1.GetFeatureCount(), dtype=int)

    """
    if inStand:
        STDs = sp.copy(FIDs)
    """
    Features = []
    Stands = []
    getFeaturesList = []
    # unselFeat = []
    # current = 0

    for i, j in enumerate(lyr1):
        # print j.GetField(inField)
        if inStand:
            # STDs[i] = j.GetField(inStand)
            Stands.append(j.GetField(inStand))
            Features.append(j.GetField(inField))
        else:
            FIDs[i] = j.GetField(inField)
            Features.append(j)
        if getFeatures:
            print(i)
            getFeaturesList.append(j)

        # current += 1
    srs = lyr1.GetSpatialRef()
    lyr1.ResetReading()

    if inStand:
        if getFeatures:
            return Features, Stands, srs, getFeaturesList
        else:
            return Features, Stands, srs
    else:
        if getFeatures is True:
            return Features, srs, getFeaturesList
        else:
            return Features, srs


def saveToShape(array, srs, outShapeFile):
    """Save array data to shapefile format."""
    # Parse a delimited text file of volcano data and create a shapefile
    # use a dictionary reader so we can access by field name
    # set up the shapefile driver
    import ogr

    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # create the data source
    if os.path.exists(outShapeFile):
        outDriver.DeleteDataSource(outShapeFile)
    # Remove output shapefile if it already exists

    # options = ['SPATIALITE=YES'])
    ds = outDriver.CreateDataSource(outShapeFile)

    # create the spatial reference, WGS84

    lyrout = ds.CreateLayer("randomSubset", srs, ogr.wkbPoint)
    fields = [array[1].GetFieldDefnRef(i).GetName() for i in range(array[1].GetFieldCount())]

    for i, f in enumerate(fields):
        field_name = ogr.FieldDefn(f, array[1].GetFieldDefnRef(i).GetType())
        field_name.SetWidth(array[1].GetFieldDefnRef(i).GetWidth())
        lyrout.CreateField(field_name)

    for k in array:
        lyrout.CreateFeature(k)

    # Save and close the data source
    ds = None


def readROIFromVector(vector, roiprefix, *args):
    """**********.

    Parameters.
    ----------
    vector : str
        Vector path ('myFolder/class.shp',str).
    roiprefix : str
        Common suffixof the training shpfile (i.e. 'band_',str).
    *args : str
        Field name containing your class number (i.e. 'class', str).

    Output
    ----------

    ROIvalues : array
    *args : array

    """
    file = ogr.Open(vector)
    lyr = file.GetLayer()

    roiFields = []

    # get all fields and save only roiFields
    ldefn = lyr.GetLayerDefn()

    listFields = []

    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        if fdefn.name is not listFields:
            listFields.append(fdefn.name)
        if fdefn.name.startswith(roiprefix):
            roiFields.append(fdefn.name)

    if len(roiFields) > 0:
        # fill ROI and level
        ROIvalues = np.zeros([lyr.GetFeatureCount(), len(roiFields)])
        if len(args) > 0:
            ROIlevels = np.zeros([lyr.GetFeatureCount(), len(args)])
        for i, feature in enumerate(lyr):
            for j, band in enumerate(roiFields):
                ROIvalues[i, j] = feature.GetField(band)
                if len(args) > 0:
                    for a in range(len(args)):
                        ROIlevels[i, a] = feature.GetField(args[a])

        if len(args) > 0:
            return ROIvalues, ROIlevels
        else:
            return ROIvalues
    else:
        from mainfunction import pushFeedback

        pushFeedback(f'ROI field "{roiprefix}" do not exists. These fields are available : ')
        pushFeedback(listFields)


if __name__ == "__main__":
    in_raster = "/mnt/DATA/Sentinel-2/SITS/SITS_TCJ.tif"
    in_field = "level3"
    in_vector = "/mnt/DATA/Formosat_2006-2014/v2/ROI/ROI_2154.sqlite"

    in_raster = "/mnt/DATA/Test/DA/SITS/SITS_2013.tif"
    in_vector = "/mnt/DATA/Test/DA/ROI_2154.sqlite"
    in_field = "level1"

    levels = ["level1", "level2", "level3"]
    in_stand = "spjoin_rif"
    """
    FIDs,STDs,srs,fts=readFieldVector(inVector,inField,inStand,getFeatures=True)
    StandCV(FIDs,STDs)

    for tr,vl in StandCV(FIDs,STDs,SLOO=True,maxIter=5):
        print(tr,vl)

    trFeat = [fts[i] for i in tr]
    vlFeat= [fts[i] for i in vl]
    saveToShape(trFeat,srs,'/tmp/train_{}.shp'.format(1))
    saveToShape(vlFeat,srs,'/tmp/valid_{}.shp'.format(1))
    """
    """
    inShape = '/mnt/DATA/demo/train.shp'
    inField = 'Class'
    number = 50
    percent = True

    outValidation = '/tmp/valid1.shp'
    outTrain ='/tmp/train.shp'

    RandomInSubset(inShape,inField,outValidation,outTrain,number,percent)

    """

    import function_dataraster

    function_dataraster.rasterize(in_raster, in_vector, in_field, "/tmp/roi.tif")
    X, Y, coords = function_dataraster.get_samples_from_roi(in_raster, "/tmp/roi.tif", getCoords=True)

    distance_array = distMatrix(coords)
    raw_cv = DistanceCV(distance_array, Y, 32, minTrain=-1, SLOO=True)

    # rawCV = DistanceCV(distanceArray,label,distanceThresold=distance,minTrain=minTrain,SLOO=SLOO,maxIter=maxIter,verbose=False,stats=True)

    for tr, vl in raw_cv:
        print(tr.shape)
        print(vl.shape)

    # randomInSubset('/tmp/valid.shp','level3','/tmp/processingd62a83be114a482aaa14ca317e640586/f99783a424984860ac9998b5027be604/OUTPUTVALIDATION.shp','/tmp/processingd62a83be114a482aaa14ca317e640586/1822187d819e450fa9ad9995d6757e09/OUTPUTTRAIN.shp',50,True)
