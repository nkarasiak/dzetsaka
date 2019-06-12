"""!@brief Manage data (opening/saving raster, get ROI...)"""
# -*- coding: utf-8 -*-

#import scipy as sp
import numpy as np
import gdal
#from osgeo import gdal_array


def convertGdalDataTypeToOTB(gdalDT):
    #availableCode = uint8/uint16/int16/uint32/int32/float/double
    code = ['uint8', 'uint16', 'int16', 'uint32', 'int32', 'float', 'double']

    return code[gdalDT]


def open_data(filename):
    '''
    The function open and load the image given its name.
    The type of the data is checked from the file and the scipy array is initialized accordingly.
    Input:
        filename: the name of the file
    Output:
        im: the data cube
        GeoTransform: the geotransform information
        Projection: the projection information
    '''
    data = gdal.Open(filename, gdal.GA_ReadOnly)
    if data is None:
        print('Impossible to open ' + filename)
        # exit()
    nc = data.RasterXSize
    nl = data.RasterYSize
    d = data.RasterCount

    # Get the type of the data
    gdal_dt = data.GetRasterBand(1).DataType
    if gdal_dt == gdal.GDT_Byte:
        dt = 'uint8'
    elif gdal_dt == gdal.GDT_Int16:
        dt = 'int16'
    elif gdal_dt == gdal.GDT_UInt16:
        dt = 'uint16'
    elif gdal_dt == gdal.GDT_Int32:
        dt = 'int32'
    elif gdal_dt == gdal.GDT_UInt32:
        dt = 'uint32'

    elif gdal_dt == gdal.GDT_Float32:
        dt = 'float32'
    elif gdal_dt == gdal.GDT_Float64:
        dt = 'float64'
    elif gdal_dt == gdal.GDT_CInt16 or gdal_dt == gdal.GDT_CInt32 or gdal_dt == gdal.GDT_CFloat32 or gdal_dt == gdal.GDT_CFloat64:
        dt = 'complex64'
    else:
        print('Data type unkown')
        # exit()

    # Initialize the array
    if d == 1:
        im = np.empty((nl, nc), dtype=dt)
    else:
        im = np.empty((nl, nc, d), dtype=dt)

    if d == 1:
        im[:, :] = data.GetRasterBand(1).ReadAsArray()
    else:
        for i in range(d):
            im[:, :, i] = data.GetRasterBand(i + 1).ReadAsArray()

    GeoTransform = data.GetGeoTransform()
    Projection = data.GetProjection()
    data = None
    return im, GeoTransform, Projection


def open_data_band(filename):
    """!@brief The function open and load the image given its name.
    The function open and load the image given its name.
    The type of the data is checked from the file and the scipy array is initialized accordingly.
        Input:
            filename: the name of the file
        Output:
            data : the opened data with gdal.Open() method
            im : empty table with right dimension (array)

    """
    data = gdal.Open(filename, gdal.GA_Update)
    if data is None:
        print('Impossible to open ' + filename)
        # exit()
    nc = data.RasterXSize
    nl = data.RasterYSize
#    d  = data.RasterCount

    # Get the type of the data
    gdal_dt = data.GetRasterBand(1).DataType
    dt = getDTfromGDAL(gdal_dt)

    # Initialize the array
    im = np.empty((nl, nc), dtype=dt)
    return data, im


'''
Old function that open all the bands
'''
#
#    for i in range(d):
#        im[:,:,i]=data.GetRasterBand(i+1).ReadAsArray()
#
#    GeoTransform = data.GetGeoTransform()
#    Projection = data.GetProjection()
#    data = None


def write_data(outname, im, GeoTransform, Projection):
    '''
    The function write the image on the  hard drive.
    Input:
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information
        Projection: the projection information
    Output:
        Nothing --
    '''
    nl = im.shape[0]
    nc = im.shape[1]
    if im.ndim == 2:
        d = 1
    else:
        d = im.shape[2]

    driver = gdal.GetDriverByName('GTiff')
    dt = im.dtype.name
    # Get the data type
    gdal_dt = getGDALGDT(dt)

    dst_ds = driver.Create(outname, nc, nl, d, gdal_dt)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)

    if d == 1:
        out = dst_ds.GetRasterBand(1)
        out.WriteArray(im)
        out.FlushCache()
    else:
        for i in range(d):
            out = dst_ds.GetRasterBand(i + 1)
            out.WriteArray(im[:, :, i])
            out.FlushCache()
    dst_ds = None


def create_empty_tiff(outname, im, d, GeoTransform, Projection):
    '''!@brief Write an empty image on the hard drive.

    Input:
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information
        Projection: the projection information
    Output:
        Nothing --
    '''
    nl = im.shape[0]
    nc = im.shape[1]

    driver = gdal.GetDriverByName('GTiff')
    dt = im.dtype.name
    # Get the data type
    gdal_dt = getGDALGDT(dt)

    dst_ds = driver.Create(outname, nc, nl, d, gdal_dt)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)

    return dst_ds

    '''
    Old function that cannot manage to write on each band outside the script
    '''
#    if d==1:
#        out = dst_ds.GetRasterBand(1)
#        out.WriteArray(im)
#        out.FlushCache()
#    else:
#        for i in range(d):
#            out = dst_ds.GetRasterBand(i+1)
#            out.WriteArray(im[:,:,i])
#            out.FlushCache()
#    dst_ds = None


def get_samples_from_roi(raster_name, roi_name,
                         stand_name=False, getCoords=False):
    '''!@brief Get the set of pixels given the thematic map.
    Get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.
        Input:
            raster_name: the name of the raster file, could be any file that GDAL can open
            roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
        Output:
            X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each
                line of the matrix is a pixel.
            Y: the label of the pixel
    Written by Mathieu Fauvel.
    '''
    # Open Raster
    raster = gdal.Open(raster_name, gdal.GA_ReadOnly)
    if raster is None:
        print('Impossible to open ' + raster_name)
        # exit()

    # Open ROI
    roi = gdal.Open(roi_name, gdal.GA_ReadOnly)
    if roi is None:
        print('Impossible to open ' + roi_name)
        # exit()

    if stand_name:
        # Open Stand
        stand = gdal.Open(stand_name, gdal.GA_ReadOnly)
        if stand is None:
            print('Impossible to open ' + stand_name)
            # exit()

    # Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (
            raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        # exit()

    # Get block size
    band = raster.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    del band

    # Get the number of variables and the size of the images
    d = raster.RasterCount
    nc = raster.RasterXSize
    nl = raster.RasterYSize

    ulx, xres, xskew, uly, yskew, yres = roi.GetGeoTransform()

    if getCoords:
        coords = np.array([], dtype=np.uint16).reshape(0, 2)
        """
    # Old function which computes metric distance...
    if getCoords :
        #list of coords
        coords = sp.array([]).reshape(0,2)

        # convert pixel position to coordinate pos
        def pixel2coord(coord):
            #Returns global coordinates from pixel x, y coords
            x,y=coord
            xp = xres * x + xskew * y + ulx
            yp = yskew * x + yres * y + uly
            return[xp, yp]
      """

    # Read block data
    X = np.array([]).reshape(0, d)
    Y = np.array([],dtype=np.uint16).reshape(0, 1)
    STD = np.array([],dtype=np.uint16).reshape(0, 1)

    for i in range(0, nl, y_block_size):
        if i + y_block_size < nl:  # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0, nc, x_block_size):  # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # Load the reference data

            ROI = roi.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            if stand_name:
                STAND = stand.GetRasterBand(1).ReadAsArray(j, i, cols, lines)

            t = np.nonzero(ROI)

            if t[0].size > 0:
                Y = np.concatenate(
                    (Y, ROI[t].reshape(
                        (t[0].shape[0], 1))))
                if stand_name:
                    STD = np.concatenate(
                        (STD, STAND[t].reshape(
                            (t[0].shape[0], 1))))
                if getCoords:
                    #coords = sp.append(coords,(i,j))
                    #coordsTp = sp.array(([[cols,lines]]))
                    #coords = sp.concatenate((coords,coordsTp))
                    # print(t[1])
                    # print(i)
                    # sp.array([[t[1],i]])
                    coordsTp = np.empty((t[0].shape[0], 2))
                    coordsTp[:, 0] = t[1]
                    coordsTp[:, 1] = [i] * t[1].shape[0]
                    """
                    for n,p in enumerate(coordsTp):
                        coordsTp[n] = pixel2coord(p)
                    """
                    coords = np.concatenate((coords, coordsTp))

                # Load the Variables
                Xtp = np.empty((t[0].shape[0], d))
                for k in range(d):
                    band = raster.GetRasterBand(
                        k +
                        1).ReadAsArray(
                        j,
                        i,
                        cols,
                        lines)
                    Xtp[:, k] = band[t]
                try:
                    X = np.concatenate((X, Xtp))
                except MemoryError:
                    print('Impossible to allocate memory: ROI too big')
                    exit()

    """
    # No conversion anymore as it computes pixel distance and not metrics
    if convertTo4326:
        import osr
        from pyproj import Proj,transform
        # convert points coords to 4326
        # if vector
        ## inShapeOp = ogr.Open(inVector)
        ## inShapeLyr = inShapeOp.GetLayer()
        ## initProj = Proj(inShapeLyr.GetSpatialRef().ExportToProj4()) # proj to Proj4

        sr = osr.SpatialReference()
        sr.ImportFromWkt(roi.GetProjection())
        initProj = Proj(sr.ExportToProj4())
        destProj = Proj("+proj=longlat +datum=WGS84 +no_defs") # http://epsg.io/4326

        coords[:,0],coords[:,1] = transform(initProj,destProj,coords[:,0],coords[:,1])
    """

    # Clean/Close variables
    del Xtp, band
    roi = None  # Close the roi file
    raster = None  # Close the raster file

    if stand_name:
        if not getCoords:
            return X, Y, STD
        else:
            return X, Y, STD, coords
    elif getCoords:
        return X, Y, coords
    else:
        return X, Y


def getDTfromGDAL(gdal_dt):
    """
    Returns datatype (numpy/scipy) from gdal_dt.

    Parameters
    ----------
    gdal_dt : datatype
        data.GetRasterBand(1).DataType

    Return
    ----------
    dt : datatype
    """
    if gdal_dt == gdal.GDT_Byte:
        dt = 'uint8'
    elif gdal_dt == gdal.GDT_Int16:
        dt = 'int16'
    elif gdal_dt == gdal.GDT_UInt16:
        dt = 'uint16'
    elif gdal_dt == gdal.GDT_Int32:
        dt = 'int32'
    elif gdal_dt == gdal.GDT_UInt32:
        dt = 'uint32'
    elif gdal_dt == gdal.GDT_Float32:
        dt = 'float32'
    elif gdal_dt == gdal.GDT_Float64:
        dt = 'float64'
    elif gdal_dt == gdal.GDT_CInt16 or gdal_dt == gdal.GDT_CInt32 or gdal_dt == gdal.GDT_CFloat32 or gdal_dt == gdal.GDT_CFloat64:
        dt = 'complex64'
    else:
        print('Data type unkown')
        # exit()
    return dt


def getGDALGDT(dt):
    """
    Need arr.dtype.name in entry.
    Returns gdal_dt from dt (numpy/scipy).

    Parameters
    ----------
    dt : datatype

    Return
    ----------
    gdal_dt : gdal datatype
    """
    if dt == 'bool' or dt == 'uint8':
        gdal_dt = gdal.GDT_Byte
    elif dt == 'int8' or dt == 'int16':
        gdal_dt = gdal.GDT_Int16
    elif dt == 'uint16':
        gdal_dt = gdal.GDT_UInt16
    elif dt == 'int32':
        gdal_dt = gdal.GDT_Int32
    elif dt == 'uint32':
        gdal_dt = gdal.GDT_UInt32
    elif dt == 'int64' or dt == 'uint64' or dt == 'float16' or dt == 'float32':
        gdal_dt = gdal.GDT_Float32
    elif dt == 'float64':
        gdal_dt = gdal.GDT_Float64
    elif dt == 'complex64':
        gdal_dt = gdal.GDT_CFloat64
    else:
        print('Data type non-suported')
        # exit()

    return gdal_dt


def predict_image(raster_name, classif_name, classifier, mask_name=None):
    """!@brief Classify the whole raster image, using per block image analysis
        The classifier is given in classifier and options in kwargs.

        Input:
            raster_name (str)
            classif_name (str)
            classifier (str)
            mask_name(str)

        Return:
            Nothing but raster written on disk
        Written by Mathieu Fauvel.
    """
    # Parameters
    block_sizes = 512

    # Open Raster and get additionnal information
    raster = gdal.Open(raster_name, gdal.GA_ReadOnly)
    if raster is None:
        print('Impossible to open ' + raster_name)
        # exit()

    # If provided, open mask
    if mask_name is None:
        mask = None
    else:
        mask = gdal.Open(mask_name, gdal.GA_ReadOnly)
        if mask is None:
            print('Impossible to open ' + mask_name)
            # exit()
        # Check size
        if (raster.RasterXSize != mask.RasterXSize) or (
                raster.RasterYSize != mask.RasterYSize):
            print('Image and mask should be of the same size')
            # exit()

    # Get the size of the image
    d = raster.RasterCount
    nc = raster.RasterXSize
    nl = raster.RasterYSize

    # Get the geoinformation
    GeoTransform = raster.GetGeoTransform()
    Projection = raster.GetProjection()

    # Set the block size
    x_block_size = block_sizes
    y_block_size = block_sizes

    # Initialize the output
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(classif_name, nc, nl, 1, gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)
    out = dst_ds.GetRasterBand(1)

    # Set the classifiers
    if classifier['name'] is 'NPFS':
        # With GMM
        model = classifier['model']
        ids = classifier['ids']
        nv = len(ids)
    elif classifier['name'] is 'GMM':
        model = classifier['model']

    # Perform the classification
    for i in range(0, nl, y_block_size):
        if i + y_block_size < nl:  # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0, nc, x_block_size):  # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # Do the prediction
            if classifier['name'] is 'NPFS':
                # Load the data
                X = np.empty((cols * lines, nv))
                for ind, v in enumerate(ids):
                    X[:, ind] = raster.GetRasterBand(
                        int(v + 1)).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # Do the prediction
                if mask is None:
                    yp = model.predict_gmm(X)[0].astype('uint16')
                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(
                        j, i, cols, lines).reshape(cols * lines)
                    t = np.where(mask_temp != 0)[0]
                    yp = np.zeros((cols * lines,))
                    yp[t] = model.predict_gmm(X[t, :])[0].astype('uint16')

            elif classifier['name'] is 'GMM':
                # Load the data
                X = np.empty((cols * lines, d))
                for ind in range(d):
                    X[:, ind] = raster.GetRasterBand(
                        int(ind + 1)).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # Do the prediction
                if mask is None:
                    yp = model.predict_gmm(X)[0].astype('uint16')
                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(
                        j, i, cols, lines).reshape(cols * lines)
                    t = np.where(mask_temp != 0)[0]
                    yp = np.zeros((cols * lines,))
                    yp[t] = model.predict_gmm(X[t, :])[0].astype('uint16')

            # Write the data
            out.WriteArray(yp.reshape(lines, cols), j, i)
            out.FlushCache()
            del X, yp

    # Clean/Close variables
    raster = None
    dst_ds = None


def create_uniquevalue_tiff(
        outname, im, d, GeoTransform, Projection, wholeValue=1, gdal_dt=False):
    '''!@brief Write an empty image on the hard drive.

    Input:
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information
        Projection: the projection information
    Output:
        Nothing --
    '''
    nl = im.shape[0]
    nc = im.shape[1]

    driver = gdal.GetDriverByName('GTiff')
    # Get the data type
    if not gdal_dt:
        gdal_dt = gdal.GDT_Byte

    dst_ds = driver.Create(outname, nc, nl, d, gdal_dt)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)

    if d == 1:
        im[:] = wholeValue
        out = dst_ds.GetRasterBand(1)
        out.WriteArray(im)
        out.FlushCache()
    else:
        for i in range(d):
            im[:, :, i] = wholeValue
            out = dst_ds.GetRasterBand(i + 1)
            out.WriteArray(im[:, :, i])
            out.FlushCache()
    dst_ds = None

    return outname


def rasterize(data, vectorSrc, field, outFile):
    dataSrc = gdal.Open(data)
    import ogr
    shp = ogr.Open(vectorSrc)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        outFile,
        dataSrc.RasterXSize,
        dataSrc.RasterYSize,
        1,
        gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(dataSrc.GetGeoTransform())
    dst_ds.SetProjection(dataSrc.GetProjection())
    if field is None:
        gdal.RasterizeLayer(dst_ds, [1], lyr, None)
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=OPTIONS)

    data, dst_ds, shp, lyr = None, None, None, None
    return outFile


def scale(x, M=None, m=None):  # TODO:  DO IN PLACE SCALING
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
    minMax = False
    if M is None:
        minMax = True
        M, m = np.amax(x, axis=0), np.amin(x, axis=0)

    den = M - m
    for i in range(d):
        if den[i] != 0:
            xs[:, i] = 2 * (x[:, i] - m[i]) / den[i] - 1
        else:
            xs[:, i] = x[:, i]

    if minMax:
        return xs, M, m
    else:
        return xs


if __name__ == "__main__":
    Raster = "/mnt/DATA/Test/dzetsaka/map.tif"
    ROI = '/home/nicolas/Bureau/train_300class.gpkg'
    rasterize(Raster, ROI, 'Class', '/tmp/roi.tif')

#    X, Y, coords = get_samples_from_roi(Raster, '/tmp/roi.tif', getCoords=True)
    X, Y = get_samples_from_roi(Raster, '/tmp/roi.tif')
    print(np.amax(Y))
    """
    import accuracy_index as ai
    print(X.shape)
    print(Y.shape)
    worker=ai.CONFUSION_MATRIX()
    worker.compute_confusion_matrix(X,Y)
    """
