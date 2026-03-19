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


# GDAL to NumPy datatype mapping for efficient lookup.
# Built dynamically to handle GDAL 4.x which removed the legacy GDT_* constants.
_GDAL_TO_NUMPY_DTYPE = {}
for _gdt_name, _np_dtype in [
    ("GDT_Byte", "uint8"),
    ("GDT_Int16", "int16"),
    ("GDT_UInt16", "uint16"),
    ("GDT_Int32", "int32"),
    ("GDT_UInt32", "uint32"),
    ("GDT_Float32", "float32"),
    ("GDT_Float64", "float64"),
    ("GDT_CInt16", "complex64"),
    ("GDT_CInt32", "complex64"),
    ("GDT_CFloat32", "complex64"),
    ("GDT_CFloat64", "complex64"),
]:
    _gdt_val = getattr(gdal, _gdt_name, None)
    if _gdt_val is not None:
        _GDAL_TO_NUMPY_DTYPE[_gdt_val] = _np_dtype


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
    im = data.GetRasterBand(1).ReadAsArray().astype(dt) if d == 1 else data.ReadAsArray().transpose(1, 2, 0).astype(dt)

    GeoTransform = data.GetGeoTransform()
    Projection = data.GetProjection()
    data = None
    return im, GeoTransform, Projection



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
                Xtp = (
                    block_data[t].reshape(-1, 1) if d == 1 else block_data[:, t[0], t[1]].T
                )  # Shape: (n_pixels, d) for multi-band
                try:
                    X = np.concatenate((X, Xtp))
                except MemoryError:
                    print("Impossible to allocate memory: ROI too big")
                    exit()

    # Clean/Close variables
    del Xtp
    roi = None  # Close the roi file
    raster = None  # Close the raster file

    if stand_name:
        if not getCoords:
            return X, Y, STD
        return X, Y, STD, coords
    if getCoords:
        return X, Y, coords
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
    return xs
