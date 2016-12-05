# -*- coding: utf-8 -*-

"""
/***************************************************************************
 className
                                 A QGIS plugin
 description
                              -------------------
        begin                : 2016-12-03
        copyright            : (C) 2016 by Nico
        email                : nico@nico
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

import dzetsaka.scripts.function_dataraster as dataraster

from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QIcon
from processing.core.GeoAlgorithm import GeoAlgorithm
from processing.core.parameters import ParameterRaster
from processing.core.parameters import ParameterNumber
from processing.core.outputs import OutputRaster


class closingFilterAlgorithm(GeoAlgorithm):
    """This is an example algorithm that takes a vector layer and
    creates a new one just with just those features of the input
    layer that are selected.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the GeoAlgorithm class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_RASTER = 'OUTPUT_RASTER'    
    CLOSING_SIZE = 'CLOSING_SIZE'

    def getIcon(self):
        return QIcon(":/plugins/dzetsaka/img/icon.png")

    def defineCharacteristics(self):
        """Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # The name that the user will see in the toolbox
        self.name = 'Closing filter'

        # The branch of the toolbox under which the algorithm will appear
        self.group = 'Filtering'	

        # We add the input vector layer. It can have any kind of geometry
        # It is a mandatory (not optional) one, hence the False argument
        self.addParameter(
        ParameterRaster(
            self.INPUT_RASTER,
            self.tr('Input layer'),
            False))

        # We add a vector layer as output
        self.addOutput(
        OutputRaster(
            self.OUTPUT_RASTER,
            self.tr('Output raster')))

        # add num
        self.addParameter(
        ParameterNumber(
            self.CLOSING_SIZE,
            self.tr('Window size of closing filter'),
            minValue=3,
            default=5))



    def processAlgorithm(self, progress):
        """Here is where the processing itself takes place."""

        INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_RASTER = self.getOutputValue(self.OUTPUT_RASTER)
        CLOSING_SIZE = self.getParameterValue(self.CLOSING_SIZE)



        # First we create the output layer. The output value entered by
        # the user is a string containing a filename, so we can use it
        # directly
        
        from scipy import ndimage
        data,im=dataraster.open_data_band(INPUT_RASTER)
        
        # get proj,geo and dimension (d) from data
        proj = data.GetProjection()
        geo = data.GetGeoTransform()
        d = data.RasterCount

        # add progress bar
        # And now we can process
        from PyQt4.QtGui import QProgressBar
        pb = QProgressBar()
        pb.setMaximum(d)
        pbNow = 0
        
        outFile=dataraster.create_empty_tiff(OUTPUT_RASTER,im,d,geo,proj)
        
        for i in range(d):
            # Read data from the right band
            pbNow+=1
            pb.setValue(pbNow)
            
            tempBand = data.GetRasterBand(i+1).ReadAsArray()
            tempBand = ndimage.morphology.grey_closing(tempBand,size=(CLOSING_SIZE,CLOSING_SIZE))
                
            # Save bandand outFile
            out=outFile.GetRasterBand(i+1)
            out.WriteArray(tempBand)
            out.FlushCache()
            tempBand = None
        