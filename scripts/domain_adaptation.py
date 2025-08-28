#!/usr/bin/env python3
"""Created on Fri Mar 23 11:24:38 2018.

@author: nkarasiak
"""

try:
    # if use in Qgis 3
    from .mainfunction import pushFeedback
except BaseException:
    from mainfunction import pushFeedback


try:
    from osgeo import gdal
except ImportError:
    import gdal
# import tempfile
# import ot
import os
from itertools import product

# from sklearn import preprocessing
import numpy as np

try:
    from sklearn.metrics import mean_squared_error
except ImportError:
    mean_squared_error = None


class RasterOT:
    """Initialize Python Optimal Transport for raster processing.

    Parameters
    ----------
    transportAlgorithm : str
        item in list : ['MappingTransport','EMDTransport','SinkhornTransport','SinkhornLpl1Transport','SinkhornL1l2Transport']
    scaler : bool
        If scaler is True, use MinMaxScaler with feature_range from -1 to 1.
    param : dict
        Target domain array.
    feedback : object
        feedback object from Qgis Processing

    """

    def __init__(
        self,
        transportAlgorithm="MappingTransport",
        scaler=False,
        params=None,
        feedback=True,
    ):
        try:
            pass
        except BaseException:
            raise ImportError("Please install itertools and scikit-learn") from None

        self.transportAlgorithm = transportAlgorithm
        self.feedback = feedback

        self.params_ = params

        if scaler:
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scalerTarget = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = scaler

    def learnTransfer(self, Xs, ys, Xt, yt=None):
        """Learn domain adaptation model.

        Parameters
        ----------
        Xs : array_like, shape (n_source_samples, n_features)
            Source domain array.
        ys : array_like, shape (n_source_samples,)
            Label source array (1d).
        Xt: array_like, shape (n_source_samples, n_features)
            Target domain array.
        yt: array_like, shape (n_source_samples,)
            Label target array (1d).

        Returns
        -------
        transportmodel : object
            The output model

        """
        # save original samples
        self.Xs_ = Xs
        self.Xt_ = Xt
        self.params = self.params_

        if self.feedback:
            pushFeedback(10, feedback=self.feedback)
            pushFeedback(
                "Learning Optimal Transport with " + str(self.transportAlgorithm) + " algorithm.",
                feedback=self.feedback,
            )

        # check if label is 1d
        if ys is not None and len(ys.shape) > 1:
            ys = ys[:, 0]
        if yt is not None and len(yt.shape) > 1:
            yt = yt[:, 0]

        # rescale Data
        if self.scaler:
            self.scaler.fit(Xs, ys)
            self.scalerTarget.fit(Xt, yt)
            Xs = self.scaler.transform(Xs)
            Xt = self.scalerTarget.transform(Xt)

        # import Domain Adaptation specific algorithm function from OT Library
        self.transportFunction = getattr(__import__("ot").da, self.transportAlgorithm)

        if self.params is None:
            self.transportModel = self.transportFunction()
        else:
            # order for reproductibility
            self.params = sorted(self.params.items())

            # if grid search
            if self.isGridSearch():
                # compute combinaison for each param
                self.findBestParameters(Xs, ys=ys, Xt=Xt, yt=yt)

                self.transportModel = self.transportFunction(**self.bestParam)

            else:
                # simply train with basic param
                self.transportModel = self.transportFunction(**self.params_)

        self.transportModel.fit(Xs, ys=ys, Xt=Xt, yt=yt)

        if self.feedback:
            pushFeedback(20, feedback=self.feedback)

        return self.transportModel

    def predictTransfer(self, imageSource, outRaster, mask=None, NODATA=-9999, feedback=None, norm=False):
        """Predict model using domain adaptation.

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
        norm : bool, optional
            Whether to normalize the data. Default is False.

        Returns
        -------
        outRaster : str
            Return the path of the predicted image.

        """
        if self.feedback:
            pushFeedback("Now transporting " + str(os.path.basename(imageSource)))

        dataSrc = gdal.Open(imageSource)
        # Get the size of the image
        d = dataSrc.RasterCount
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
        # gdal_dt = band.DataType

        # Initialize the output
        driver = gdal.GetDriverByName("GTiff")

        dst_ds = driver.Create(outRaster, nc, nl, d, 3)
        dst_ds.SetGeoTransform(GeoTransform)
        dst_ds.SetProjection(Projection)

        del band
        # Perform the classification
        if mask is not None:
            maskData = gdal.Open(mask, gdal.GA_ReadOnly)

        total = nl * y_block_size

        total = 80 / (int(nl / y_block_size))

        for i in range(0, nl, y_block_size):
            # feedback for Qgis
            if self.feedback:
                pushFeedback(int(i * total) + 20, feedback=self.feedback)
                try:
                    if self.feedback.isCanceled():
                        break
                except BaseException:
                    pass

            lines = y_block_size if i + y_block_size < nl else nl - i  # Check for size consistency in Y
            for j in range(0, nc, x_block_size):  # Check for size consistency in X
                cols = x_block_size if j + x_block_size < nc else nc - j

                # Load the data and Do the prediction
                X = np.empty((cols * lines, d))
                for ind in range(d):
                    X[:, ind] = dataSrc.GetRasterBand(int(ind + 1)).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # Do the prediction

                if mask is None:
                    mask_temp = dataSrc.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                else:
                    mask_temp = maskData.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # check if nodata
                t = np.where((mask_temp != 0) & (X[:, 0] != NODATA))[0]

                # transform array, default has nodata value
                yp = np.empty((cols * lines, d))
                yp[:, :] = NODATA
                # yp = np.nan((cols*lines,d))
                # K = np.zeros((cols*lines,))

                # TODO: Change this part accorindgly ...
                # if t.size > 0:
                if t.size > 0:
                    tempOT = X[t, :]
                    yp[t, :] = self.transportModel.transform(tempOT)

                for ind in range(d):
                    out = dst_ds.GetRasterBand(ind + 1)
                    # Write the data
                    ypTemp = yp[:, ind]
                    out.WriteArray(ypTemp.reshape(lines, cols), j, i)
                    out.SetNoDataValue(NODATA)
                    out.FlushCache()

                del X, yp

        return outRaster

    def isGridSearch(self):
        """Check if grid search parameters are configured."""
        # search for gridSearch
        paramGrid = []
        for key in self.params_:
            if isinstance(self.params_.get(key), (list, np.ndarray)):
                paramGrid.append(key)

        if paramGrid == []:
            self.paramGrid = False
        else:
            self.paramGrid = paramGrid
            self.params = self.params_.copy()

        return bool(self.paramGrid)

    def generateParamForGridSearch(self):
        """Generate parameter combinations for grid search."""
        hyperParam = {key: self.params_[key] for key in self.paramGrid}
        items = sorted(hyperParam.items())
        keys, values = zip(*items)
        for v in product(*values):
            paramsToAdd = dict(zip(keys, v))
            self.params.update(paramsToAdd)

            yield self.params

    def findBestParameters(self, Xs, ys, Xt, yt):
        """Find the best parameters using grid search."""
        self.bestScore = None
        for gridOT in self.generateParamForGridSearch():
            self.transportModel = self.transportFunction(**gridOT)
            self.transportModel.fit(Xs, ys, Xt, yt)
            # XsTransformed = self.transportModel.transform(Xs)
            # XsPredict = self.inverseTransform(XsTransformed)
            from ot.da import BaseTransport

            transp_Xt = BaseTransport.inverse_transform(self.transportModel, Xs=Xs, ys=ys, Xt=Xt, yt=yt)

            if self.feedback:
                pushFeedback("Testing params : " + str(gridOT), feedback=self.feedback)
            """
            #score = mean_squared_error(Xs,XsPredict)

            from sklearn.svm import SVC
            from sklearn.model_selection import StratifiedKFold
            from sklearn.model_selection import GridSearchCV

            param_grid = dict(gamma=2.0**np.arange(-4,1), C=10.0**np.arange(-2,3))
            classifier = SVC(probability=False)
            cv = StratifiedKFold(n_splits=5)

            grid = GridSearchCV(classifier,param_grid=param_grid, cv=cv,n_jobs=1)

            # need to rescale for hyperparameter of svm
            if self.scaler is False:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(-1,1))
                scaler.fit(Xs,ys)
                Xs = scaler.transform(Xs)
                XsPredict = scaler.transform(XsPredict)
                #XsPredict = scaler.transform(XsPredict)

            grid.fit(Xs,ys)
            model = grid.best_estimator_
            model.fit(Xs,ys)

            yp = model.predict(XsPredict)
            currentScore = dict(OA=accuracy_score(yp,ys),Kappa=cohen_kappa_score(yp,ys),F1=f1_score(yp,ys,average='micro'))


            if self.feedback:
                pushFeedback('Kappa is : '+str(currentScore.get('Kappa')))

            if self.bestScore is None or self.bestScore.get('Kappa') < currentScore.get('Kappa'):
                self.bestScore = currentScore.copy()
                self.bestParam = gridOT.copy()
            """

            currentScore = mean_squared_error(Xs, transp_Xt)

            if self.feedback:
                pushFeedback("RMSE is : " + str(currentScore), feedback=self.feedback)

            if self.bestScore is None or self.bestScore > currentScore:
                self.bestScore = currentScore
                self.bestParam = gridOT.copy()
            """

            del self.transportModel,yp
            """
        if self.feedback:
            pushFeedback("Best grid is " + str(self.bestParam), feedback=self.feedback)
            pushFeedback("Best score is " + str(self.bestScore), feedback=self.feedback)

    """
    def gridSearchCV(self):
    """

    def inverseTransform(self, Xt):
        """Transports target samples Xt onto target samples Xs.

        Parameters.
        ----------
        Xt : array-like, shape (n_source_samples, n_features)
            The training input samples.

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transport source samples.

        """
        # perform standard barycentric mapping
        transp = self.transportModel.coupling_.T / np.sum(self.transportModel.coupling_, 0)[:, None]

        # set nans to 0
        transp[~np.isfinite(transp)] = 0

        # compute transported samples
        transp_Xt = np.dot(transp, self.transportModel.xs_)

        return transp_Xt
