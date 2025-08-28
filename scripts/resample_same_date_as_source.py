#!/usr/bin/env python3
"""Created on Sat Jun 17 19:18:59 2017.

@author: nkarasiak

Only work with max 1 year SITS. Use Date Of Year, so cannot resample SITS if more on two years.

"""

try:
    from osgeo import gdal
except ImportError:
    import gdal
import datetime
import glob
import os

import numpy as np

try:
    import function_dataraster as dataraster
    from mainfunction import pushFeedback
except BaseException:
    from . import function_dataraster as dataraster
    from .mainfunction import pushFeedback


def convertToDateTime(dates, strp="%Y%m%d", DOY=False):
    """Convert dates to datetime format or Day of Year.

    Parameters
    ----------
    dates : list
        List of date strings to convert
    strp : str, optional
        Date format string (default: "%Y%m%d")
    DOY : bool, optional
        If True, return Day of Year format (default: False)

    Returns
    -------
    list
        Converted dates

    """
    datesTime = []
    for i in dates:
        cDatesTime = datetime.datetime.strptime(str(i).split(".")[0], strp)

        if DOY:
            cDatesTime = cDatesTime.timetuple().tm_yday
            datesTime.append(f"{cDatesTime:03d}")

            # print cDatesTime
        else:
            datesTime.append(cDatesTime)

    return datesTime


def listToStr(fileName, sep=" "):
    """Convert list of filenames to space-separated string.

    Parameters
    ----------
    fileName : list
        List of filenames
    sep : str, optional
        Separator character (default: " ")

    Returns
    -------
    str
        Space-separated string of filenames

    """
    strList = ""
    for file in fileName:
        strList = strList + sep + str(file)

    return strList


def resampleWithSameDateAsSource(
    sourceImage,
    targetImage,
    sourceDates,
    targetDates,
    nSpectralBands,
    resampledImage,
    feedback=None,
):
    """Resample target image to match source image dates.

    Only works with max 1 year SITS. Uses Day of Year, so cannot resample
    SITS if more than two years.

    Parameters
    ----------
    sourceImage : str
        Path to source image
    targetImage : str
        Path to target image
    sourceDates : str
        Source dates file path
    targetDates : str
        Target dates file path
    nSpectralBands : int
        Number of spectral bands
    resampledImage : str
        Output resampled image path
    feedback : object, optional
        Feedback object for progress reporting

    """
    pushFeedback(1, feedback=feedback)

    tempDir = os.path.dirname(resampledImage) + "/tmp"
    if not os.path.exists(tempDir):
        os.makedirs(tempDir)

    # RefDates = sp.loadtxt("/mnt/Data_2/Formosat/RAW/formosat_SudouestKalideos_2010/sample_time.csv")
    # ChangeDates = sp.loadtxt("/mnt/Data_2/Formosat/RAW/formosat_SudouestKalideos_2012/sample_time.csv")

    sourceDatesArr = np.loadtxt(sourceDates)
    targetDatesArr = np.loadtxt(targetDates)

    nSourceBands = len(sourceDatesArr)

    DOY = []
    for i in (sourceDatesArr, targetDatesArr):
        DOY.append(convertToDateTime(i, "%Y%m%d", DOY=True))

    sourceDOY = np.asarray(DOY[0])
    targetDOY = np.asarray(DOY[1])

    # createEmptyMaskDates = [d for d in sourceDOY if(d not in(targetDOY))]

    combineDOY = np.unique(np.sort(np.concatenate((sourceDOY, targetDOY))))

    sourceDOYidx = np.searchsorted(combineDOY, sourceDOY)

    # needmask if 1, 0 means image is in source Image
    needMask = np.ones(combineDOY.shape)
    needMask[sourceDOYidx] = 0

    # get target dtype for OTB computation
    dataTarget = gdal.Open(targetImage)
    dataTargetBand = dataTarget.GetRasterBand(1)
    targetDataType = dataraster.convertGdalDataTypeToOTB(dataTargetBand.DataType)

    # create doy vrt using false image
    dataSource = gdal.Open(sourceImage)

    GeoTransform = dataSource.GetGeoTransform()
    Projection = dataSource.GetProjection()
    # im = dataTarget.GetRasterBand(1).ReadAsArray()
    im = np.zeros([dataSource.RasterYSize, dataSource.RasterXSize])

    dataraster.create_uniquevalue_tiff(
        tempDir + "/mask1.tif",
        im,
        1,
        GeoTransform,
        Projection,
        1,
        dataTargetBand.DataType,
    )
    dataraster.create_uniquevalue_tiff(
        tempDir + "/mask0.tif",
        im,
        1,
        GeoTransform,
        Projection,
        0,
        dataTargetBand.DataType,
    )

    for i in combineDOY[needMask == 1]:
        for spectral in range(nSpectralBands):
            bashCommand = (
                "gdalbuildvrt "
                + str(tempDir)
                + "/temp_"
                + str(spectral + 1)
                + "_"
                + str(i)
                + ".tif "
                + str(tempDir)
                + "/mask1.tif"
            )
            bashCommandMask = (
                "gdalbuildvrt "
                + str(tempDir)
                + "/temp_"
                + str(spectral + 1)
                + "_"
                + str(i)
                + "_mask.tif "
                + str(tempDir)
                + "/mask1.tif"
            )
            os.system(bashCommand)
            os.system(bashCommandMask)

    # create doy vrt using real image
    for i, j in enumerate(sourceDOY):
        for spectral in range(nSpectralBands):
            bandToKeep = (spectral * nSourceBands) + int(i) + 1
            bashCommand = (
                "gdalbuildvrt -b "
                + str(bandToKeep)
                + " "
                + str(tempDir)
                + "/temp_"
                + str(spectral + 1)
                + "_"
                + str(j)
                + ".tif "
                + sourceImage
            )
            bashCommandMask = (
                "gdalbuildvrt "
                + str(tempDir)
                + "/temp_"
                + str(spectral + 1)
                + "_"
                + str(j)
                + "_mask.tif "
                + str(tempDir)
                + "/mask0.tif"
            )
            os.system(bashCommand)
            os.system(bashCommandMask)

    pushFeedback(10, feedback=feedback)
    """
    for i in convertToDateTime(createEmptyMaskDates):
        date = i.strftime('%Y%m%d')
        #bashCommand = "gdalbuildvrt "+str(WDIR)+"/mask1.tif"

        bashCommand = ("gdalbuildvrt "+str(WDIR)+'temp_'+str(date)+".tif "+str(WDIR)+"mask1.tif ")*4+' -separate'
        bashCommandMask = "gdalbuildvrt "+str(WDIR)+'temp_'+str(date)+".nuages.tif "+str(WDIR)+"mask1.tif "
        os.system(bashCommand)
        os.system(bashCommandMask)
    """
    # os.remove(WDIR+'mask1.tif')

    # nDOY = (datetime.date(2013,1,1) - datetime.date(2012, 1, 1)).days

    srcDate = convertToDateTime(sourceDatesArr)[0]
    sourceYEAR = srcDate.year

    def DOYtoDates(doys, sourceYEAR, convertToInt=False):
        dates = []
        for i in doys:
            currentDate = datetime.date(sourceYEAR, 1, 1) + datetime.timedelta(int(i) - 1)
            if convertToInt:
                currentDate = currentDate.strftime("%Y%m%d")
            dates.append(currentDate)
        return dates

    newDates = np.unique(DOYtoDates(combineDOY, sourceYEAR, convertToInt=True))

    np.savetxt(tempDir + "/sampleTime.csv", np.asarray(newDates, dtype=int), fmt="%d")

    """

    tokeep=''
    initDOY=0
    for k in range(4):
        for i in DOY[0]:
            tokeep=tokeep+'-b '+str(initDOY+i)+' '
        initDOY+=nDOY



    bashCommand='gdalbuildvrt '+tokeep+"SITS_2012_temp.tif "+WDIR+"SITS_2010.tif "

    os.system(bashCommand)

    bashCommand='gdal_translate SITS_2012_temp.tif '+WDIR+'NODTW_SameDatesAsRef/SITS_2012.tif && rm SITS_2012_temp.tif'
    os.system(bashCommand)

    bashCommand='cp /mnt/Data_2/Formosat/SITS_2/SITS_2010.tif '+WDIR+'NODTW_SameDatesAsRef/SITS_2010.tif'
    os.system(bashCommand)

    """
    # os.remove(tempDir+'/mask0.tif')
    # os.remove(tempDir+'/mask1.tif')

    # print DOY

    sourceDOYidx = np.unique(np.searchsorted(combineDOY, targetDOY))
    # bandsToKeepInVrt = listToStr(sourceDOYidx+1,sep=' -b ')

    total = 60 / nSpectralBands
    for spectral in range(nSpectralBands):
        try:
            if feedback.isCanceled():
                break
        except BaseException:
            pass
        pushFeedback(10 + total * (spectral + 1), feedback=feedback)
        pushFeedback(
            "Gap-filling timeseries (" + str(spectral + 1) + "/" + str(nSpectralBands) + ")",
            feedback=feedback,
        )

        toGapFill = glob.glob(tempDir + "/*_" + str(spectral + 1) + "_*[0-9].tif")
        toGapFillMask = glob.glob(tempDir + "/*_" + str(spectral + 1) + "_*[0-9]_mask.tif")

        vrt = "gdalbuildvrt -separate " + tempDir + "/temp.vrt" + listToStr(sorted(toGapFill))
        os.system(vrt)

        vrtmask = "gdalbuildvrt -separate " + tempDir + "/temp_mask.vrt" + listToStr(sorted(toGapFillMask))
        os.system(vrtmask)

        bashCommand = ("otbcli_ImageTimeSeriesGapFilling -in {} -mask {} -out {} {} -comp 1 -it linear -id {}").format(
            tempDir + "/temp.vrt",
            tempDir + "/temp_mask.vrt",
            tempDir + "/temp_" + str(spectral + 1) + ".tif",
            targetDataType,
            tempDir + "/sampleTime.csv",
        )

        pushFeedback("Executing gap filling : " + bashCommand, feedback=feedback)

        os.system(bashCommand)

        os.remove(tempDir + "/temp.vrt")
        os.remove(tempDir + "/temp_mask.vrt")

        tempList = []
        for i, j in enumerate(sourceDOYidx):
            currentVrt = (tempDir + "/band_temp_{}_{}.tif").format(str(spectral + 1), i)
            tempList.append(currentVrt)
            vrt = ("gdalbuildvrt -b {0} {1} " + tempDir + "/temp_{2}.tif").format(j + 1, currentVrt, str(spectral + 1))
            os.system(vrt)

        vrt = "gdalbuildvrt -separate " + tempDir + "/temp_" + str(spectral + 1) + ".vrt" + listToStr(tempList)
        os.system(vrt)

    bandList = [tempDir + "/temp_" + str(x + 1) + ".vrt" for x in range(nSpectralBands)]

    # if outfolder not exists, create

    conca = f"otbcli_ConcatenateImages -il {listToStr(bandList)} -out {resampledImage} {targetDataType}"

    pushFeedback(80, feedback=feedback)
    pushFeedback("Executing image concatenation : " + conca, feedback=feedback)

    os.system(conca)

    files = glob.glob(tempDir + "/*")
    for file in files:
        os.remove(file)
    os.removedirs(tempDir)

    return resampledImage


if __name__ == "__main__":
    # os.chdir(WDIR)
    """
    sourceImage = "/mnt/DATA/Test/DA/SITS/SITS_2013.tif"
    targetImage = "/mnt/DATA/Test/DA/SITS/SITS_2014.tif"

    sourceDates = '/mnt/DATA/Test/DA/SITS/sample_time_2013.csv'
    targetDates = '/mnt/DATA/Test/DA/SITS/sample_time_2014.csv'
    #resampledImage = WDIR+"sameDatesAsRef/SITS_2014as2013.tif"
    resampledImage = '/tmp/OUTPUT_RASTER.tif'
    """
    target_image = "/mnt/DATA/Sentinel-2/2017/SITS/T31TCJ/SITS_4bands.tif"
    source_image = "/mnt/DATA/Sentinel-2/2017/SITS/T31TDJ/SITS_4bands.tif"

    target_dates = "/mnt/DATA/Sentinel-2/2017/SITS/T31TCJ/sample_time.csv"
    source_dates = "/mnt/DATA/Sentinel-2/2017/SITS/T31TDJ/sample_time.csv"

    resampled_image = "/media/nkarasiak/Maxtor1/SITS/SITS_T31TDJasT31TCJdates.tif"
    n_spectral_bands = 4

    resampleWithSameDateAsSource(
        source_image,
        target_image,
        source_dates,
        target_dates,
        n_spectral_bands,
        resampled_image,
    )
