"""!@brief Manage data (opening/saving raster, get ROI...)."""
# -*- coding: utf-8 -*-

# import scipy as sp
import numpy as np

try:
    from osgeo import gdal, ogr
except ImportError:
    import gdal
    import ogr
# from osgeo import gdal_array


def get_layer_source_path(layer):
    """Extract file path from a QGIS layer, handling URI formats safely.

    This function provides a robust way to get the underlying file path
    from a QGIS vector or raster layer, properly handling various URI
    formats (shapefiles, GeoPackage, etc.) that may include query
    parameters like '|layerid=0' or '|layername=table'.

    Parameters
    ----------
    layer : QgsMapLayer
        A QGIS layer (vector or raster).

    Returns
    -------
    str
        The file system path to the layer's data source.

    Notes
    -----
    For QGIS 4.0+ compatibility, this uses QgsProviderRegistry when
    available, falling back to string parsing for older versions.

    """
    try:
        # Preferred method: use QgsProviderRegistry to decode URI properly
        from qgis.core import QgsProviderRegistry

        provider_key = layer.providerType()
        uri = layer.dataProvider().dataSourceUri()
        decoded = QgsProviderRegistry.instance().decodeUri(provider_key, uri)

        # 'path' key contains the file path for most providers
        if "path" in decoded:
            return decoded["path"]
        # Fallback to full URI if no path key
        return uri.split("|")[0]

    except (ImportError, AttributeError, KeyError):
        # Fallback for environments without QgsProviderRegistry
        # or if decodeUri fails
        uri = layer.dataProvider().dataSourceUri()
        return uri.split("|")[0]


def convertGdalDataTypeToOTB(gdalDT):
    """Convert GDAL data type to OTB code."""
    # availableCode = uint8/uint16/int16/uint32/int32/float/double
    code = ["uint8", "uint16", "int16", "uint32", "int32", "float", "double"]

    return code[gdalDT]


# GDAL to NumPy datatype mapping for efficient lookup
_GDAL_TO_NUMPY_DTYPE = {
    gdal.GDT_Byte: "uint8",
    gdal.GDT_Int16: "int16",
    gdal.GDT_UInt16: "uint16",
    gdal.GDT_Int32: "int32",
    gdal.GDT_UInt32: "uint32",
    gdal.GDT_Float32: "float32",
    gdal.GDT_Float64: "float64",
    gdal.GDT_CInt16: "complex64",
    gdal.GDT_CInt32: "complex64",
    gdal.GDT_CFloat32: "complex64",
    gdal.GDT_CFloat64: "complex64",
}


def open_data(filename):
    """Open and load the image given its name.

    The type of the data is checked from the file and the numpy array is
    initialized accordingly.

    Parameters
    ----------
    filename : str
        The name/path of the raster file to open.

    Returns
    -------
    im : np.ndarray
        The data cube of shape (rows, cols) for single-band or
        (rows, cols, bands) for multi-band images.
    GeoTransform : tuple
        The geotransform information (6-element tuple).
    Projection : str
        The projection information as WKT string.

    """
    data = gdal.Open(filename, gdal.GA_ReadOnly)
    if data is None:
        print("Impossible to open " + filename)
        return None, None, None

    d = data.RasterCount

    # Get the type of the data using dict lookup
    gdal_dt = data.GetRasterBand(1).DataType
    dt = _GDAL_TO_NUMPY_DTYPE.get(gdal_dt)
    if dt is None:
        print("Data type unknown")
        dt = "float64"  # Fallback to float64

    # Read all bands at once using GDAL's ReadAsArray on the dataset
    # This is more efficient than reading band by band
    if d == 1:
        im = data.GetRasterBand(1).ReadAsArray().astype(dt)
    else:
        # ReadAsArray() on dataset returns (bands, rows, cols)
        # Transpose to (rows, cols, bands) for compatibility
        im = data.ReadAsArray().transpose(1, 2, 0).astype(dt)

    GeoTransform = data.GetGeoTransform()
    Projection = data.GetProjection()
    data = None
    return im, GeoTransform, Projection


def open_data_band(filename):
    """Open and load the image given its name.

    The function open and load the image given its name.
    The type of the data is checked from the file and the scipy array is initialized accordingly.
        Input:
            filename: the name of the file
        Output:
            data : the opened data with gdal.Open() method
            im : empty table with right dimension (array).
    """
    data = gdal.Open(filename, gdal.GA_Update)
    if data is None:
        print("Impossible to open " + filename)
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


"""
Old function that open all the bands
"""
#
#    for i in range(d):
#        im[:,:,i]=data.GetRasterBand(i+1).ReadAsArray()
#
#    GeoTransform = data.GetGeoTransform()
#    Projection = data.GetProjection()
#    data = None


def write_data(outname, im, GeoTransform, Projection):
    """Write the image to the hard drive.

    Input:
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information
        Projection: the projection information
    Output:
        Nothing --.
    """
    nl = im.shape[0]
    nc = im.shape[1]
    d = 1 if im.ndim == 2 else im.shape[2]

    driver = gdal.GetDriverByName("GTiff")
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
    """!@brief Write an empty image on the hard drive.

    Input:
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information
        Projection: the projection information
    Output:
        Nothing --
    """
    nl = im.shape[0]
    nc = im.shape[1]

    driver = gdal.GetDriverByName("GTiff")
    dt = im.dtype.name
    # Get the data type
    gdal_dt = getGDALGDT(dt)

    dst_ds = driver.Create(outname, nc, nl, d, gdal_dt)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)

    return dst_ds

    """
    Old function that cannot manage to write on each band outside the script
    """


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


def get_samples_from_roi(raster_name, roi_name, stand_name=False, getCoords=False):
    """Get the set of pixels given the thematic map.

    Get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.
        Input:
            raster_name: the name of the raster file, could be any file that GDAL can open
            roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
        Output:
            X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each
                line of the matrix is a pixel.
            Y: the label of the pixel
    Written by Mathieu Fauvel.
    """
    # Open Raster
    raster = gdal.Open(raster_name, gdal.GA_ReadOnly)
    if raster is None:
        print("Impossible to open " + raster_name)
        # exit()

    # Open ROI
    roi = gdal.Open(roi_name, gdal.GA_ReadOnly)
    if roi is None:
        print("Impossible to open " + roi_name)
        # exit()

    if stand_name:
        # Open Stand
        stand = gdal.Open(stand_name, gdal.GA_ReadOnly)
        if stand is None:
            print("Impossible to open " + stand_name)
            # exit()

    # Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (raster.RasterYSize != roi.RasterYSize):
        print("Images should be of the same size")
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
    Y = np.array([], dtype=np.uint16).reshape(0, 1)
    STD = np.array([], dtype=np.uint16).reshape(0, 1)

    for i in range(0, nl, y_block_size):
        lines = y_block_size if i + y_block_size < nl else nl - i  # Check for size consistency in Y
        for j in range(0, nc, x_block_size):  # Check for size consistency in X
            cols = x_block_size if j + x_block_size < nc else nc - j

            # Load the reference data

            ROI = roi.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            if stand_name:
                STAND = stand.GetRasterBand(1).ReadAsArray(j, i, cols, lines)

            t = np.nonzero(ROI)

            if t[0].size > 0:
                Y = np.concatenate((Y, ROI[t].reshape((t[0].shape[0], 1))))
                if stand_name:
                    STD = np.concatenate((STD, STAND[t].reshape((t[0].shape[0], 1))))
                if getCoords:
                    # coords = sp.append(coords,(i,j))
                    # coordsTp = sp.array(([[cols,lines]]))
                    # coords = sp.concatenate((coords,coordsTp))
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

                # Load all bands at once for this block, then extract ROI pixels
                # This is more efficient than reading band by band
                block_data = raster.ReadAsArray(j, i, cols, lines)  # Shape: (d, lines, cols)
                if d == 1:
                    # Single band case: ReadAsArray returns (lines, cols)
                    Xtp = block_data[t].reshape(-1, 1)
                else:
                    # Multi-band: extract pixels at ROI locations using advanced indexing
                    # block_data shape is (d, lines, cols), t is (row_indices, col_indices)
                    Xtp = block_data[:, t[0], t[1]].T  # Shape: (n_pixels, d)
                try:
                    X = np.concatenate((X, Xtp))
                except MemoryError:
                    print("Impossible to allocate memory: ROI too big")
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
    del Xtp
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


# NumPy to GDAL datatype mapping for efficient lookup
_NUMPY_TO_GDAL_DTYPE = {
    "bool": gdal.GDT_Byte,
    "uint8": gdal.GDT_Byte,
    "int8": gdal.GDT_Int16,
    "int16": gdal.GDT_Int16,
    "uint16": gdal.GDT_UInt16,
    "int32": gdal.GDT_Int32,
    "uint32": gdal.GDT_UInt32,
    "int64": gdal.GDT_Float32,
    "uint64": gdal.GDT_Float32,
    "float16": gdal.GDT_Float32,
    "float32": gdal.GDT_Float32,
    "float64": gdal.GDT_Float64,
    "complex64": gdal.GDT_CFloat64,
}


def getDTfromGDAL(gdal_dt):
    """Convert GDAL datatype to numpy datatype string.

    Parameters
    ----------
    gdal_dt : int
        GDAL datatype constant (e.g., gdal.GDT_Byte).

    Returns
    -------
    dt : str
        Numpy datatype string (e.g., 'uint8').

    """
    dt = _GDAL_TO_NUMPY_DTYPE.get(gdal_dt)
    if dt is None:
        print("Data type unknown")
        dt = "float64"  # Fallback
    return dt


def getGDALGDT(dt):
    """Convert numpy datatype string to GDAL datatype.

    Parameters
    ----------
    dt : str
        Numpy datatype string (e.g., 'float32', from arr.dtype.name).

    Returns
    -------
    gdal_dt : int
        GDAL datatype constant.

    """
    gdal_dt = _NUMPY_TO_GDAL_DTYPE.get(dt)
    if gdal_dt is None:
        print("Data type non-supported: " + str(dt))
        gdal_dt = gdal.GDT_Float64  # Fallback
    return gdal_dt


def predict_image(raster_name, classif_name, classifier, mask_name=None):
    """Classify the whole raster image using per block analysis.

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
        print("Impossible to open " + raster_name)
        # exit()

    # If provided, open mask
    if mask_name is None:
        mask = None
    else:
        mask = gdal.Open(mask_name, gdal.GA_ReadOnly)
        if mask is None:
            print("Impossible to open " + mask_name)
            # exit()
        # Check size
        if (raster.RasterXSize != mask.RasterXSize) or (raster.RasterYSize != mask.RasterYSize):
            print("Image and mask should be of the same size")
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
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(classif_name, nc, nl, 1, gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(GeoTransform)
    dst_ds.SetProjection(Projection)
    out = dst_ds.GetRasterBand(1)

    # Set the classifiers
    if classifier["name"] == "NPFS":
        # With GMM
        model = classifier["model"]
        ids = classifier["ids"]
        nv = len(ids)
    elif classifier["name"] == "GMM":
        model = classifier["model"]

    # Perform the classification
    for i in range(0, nl, y_block_size):
        lines = y_block_size if i + y_block_size < nl else nl - i  # Check for size consistency in Y
        for j in range(0, nc, x_block_size):  # Check for size consistency in X
            cols = x_block_size if j + x_block_size < nc else nc - j

            # Do the prediction
            if classifier["name"] == "NPFS":
                # Load the data
                X = np.empty((cols * lines, nv))
                for ind, v in enumerate(ids):
                    X[:, ind] = raster.GetRasterBand(int(v + 1)).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # Do the prediction
                if mask is None:
                    yp = model.predict_gmm(X)[0].astype("uint16")
                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                    t = np.where(mask_temp != 0)[0]
                    yp = np.zeros((cols * lines,))
                    yp[t] = model.predict_gmm(X[t, :])[0].astype("uint16")

            elif classifier["name"] == "GMM":
                # Load the data
                X = np.empty((cols * lines, d))
                for ind in range(d):
                    X[:, ind] = raster.GetRasterBand(int(ind + 1)).ReadAsArray(j, i, cols, lines).reshape(cols * lines)

                # Do the prediction
                if mask is None:
                    yp = model.predict_gmm(X)[0].astype("uint16")
                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                    t = np.where(mask_temp != 0)[0]
                    yp = np.zeros((cols * lines,))
                    yp[t] = model.predict_gmm(X[t, :])[0].astype("uint16")

            # Write the data
            out.WriteArray(yp.reshape(lines, cols), j, i)
            out.FlushCache()
            del X, yp

    # Clean/Close variables
    raster = None
    dst_ds = None


def create_uniquevalue_tiff(outname, im, d, GeoTransform, Projection, wholeValue=1, gdal_dt=False):
    """!@brief Write an empty image on the hard drive.

    Input:
        outname: the name of the file to be written
        im: the image cube
        GeoTransform: the geotransform information
        Projection: the projection information
    Output:
        Nothing --
    """
    nl = im.shape[0]
    nc = im.shape[1]

    driver = gdal.GetDriverByName("GTiff")
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
    """Rasterize vector data using reference raster geometry."""
    dataSrc = gdal.Open(data)

    shp = ogr.Open(vectorSrc)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(outFile, dataSrc.RasterXSize, dataSrc.RasterYSize, 1, gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(dataSrc.GetGeoTransform())
    dst_ds.SetProjection(dataSrc.GetProjection())
    if field is None:
        gdal.RasterizeLayer(dst_ds, [1], lyr, None)
    else:
        OPTIONS = ["ATTRIBUTE=" + field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=OPTIONS)

    data, dst_ds, shp, lyr = None, None, None, None
    return outFile


def scale(x, M=None, m=None):
    """Standardize the data using min-max scaling to [-1, 1] range.

    Parameters
    ----------
    x : np.ndarray
        The data array of shape (n_samples, n_features).
    M : np.ndarray, optional
        The max vector. If None, computed from x.
    m : np.ndarray, optional
        The min vector. If None, computed from x.

    Returns
    -------
    xs : np.ndarray
        The standardized data.
    M : np.ndarray
        The max vector (only if M was None on input).
    m : np.ndarray
        The min vector (only if M was None on input).

    """
    if np.float64 != x.dtype.type:
        x = x.astype("float")

    # Get the parameters of the scaling
    minMax = False
    if M is None:
        minMax = True
        M, m = np.amax(x, axis=0), np.amin(x, axis=0)

    # Vectorized scaling: avoid division by zero with safe denominator
    den = M - m
    den_safe = np.where(den != 0, den, 1.0)  # Replace zeros with 1 to avoid division by zero

    # Vectorized computation across all columns at once
    xs = 2.0 * (x - m) / den_safe - 1.0

    # Restore original values for columns with zero range
    zero_range_mask = den == 0
    if np.any(zero_range_mask):
        xs[:, zero_range_mask] = x[:, zero_range_mask]

    if minMax:
        return xs, M, m
    else:
        return xs


if __name__ == "__main__":
    Raster = "/mnt/DATA/Test/dzetsaka/map.tif"
    ROI = "/home/nicolas/Bureau/train_300class.gpkg"
    rasterize(Raster, ROI, "Class", "/tmp/roi.tif")

    #    X, Y, coords = get_samples_from_roi(Raster, '/tmp/roi.tif', getCoords=True)
    X, Y = get_samples_from_roi(Raster, "/tmp/roi.tif")
    print(np.amax(Y))
    """
    import accuracy_index as ai
    print(X.shape)
    print(Y.shape)
    worker=ai.ConfusionMatrix()
    worker.compute_confusion_matrix(X,Y)
    """
