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

from qgis.PyQt.QtGui import QIcon
from PyQt5.QtCore import QCoreApplication

# from PyQt5.QtWidgets import QMessageBox
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterString,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterFileDestination,
)

import os

from .. import classifier_config
from ..scripts import mainfunction

pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class trainSTANDalgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = "INPUT_RASTER"
    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = classifier_config.UI_DISPLAY_NAMES
    TRAIN_ALGORITHMS_CODE = classifier_config.CLASSIFIER_CODES
    SLOO = "SLOO"
    STAND_COLUMN = "STAND_COLUMN"
    MAXITER = "MAXITER"
    PARAMGRID = "PARAMGRID"
    # MINTRAIN = "MINTRAIN"
    # SPLIT_PERCENT= 'SPLIT_PERCENT'
    OUTPUT_MODEL = "OUTPUT_MODEL"
    # OUTPUT_MATRIX = "OUTPUT_MATRIX"
    SAVEDIR = "SAVEDIR"

    def shortHelpString(self):
        return self.tr(
            "Learn with Cross Validation with Spatial Leave-One-Out stand to better learn and estimate prediction."
        )

    """
    def helpUrl(self):
        return "http://pot.readthedocs.io/en/stable/all.html#module-ot.da"
    """

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Train algorithm (CV per stand/polygon)"

    def icon(self):
        return QIcon(os.path.join(pluginPath, "icon.png"))

    def initAlgorithm(self, config=None):
        # The name that the user will see in the toolbox

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, self.tr("Input raster")
            )
        )

        # SLOO
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SLOO,
                self.tr("Check for Leave-One-Out validation. Uncheck for 50\\50."),
                defaultValue=True,
            )
        )

        # Train algorithm

        self.addParameter(
            QgsProcessingParameterEnum(
                self.TRAIN, "Select algorithm to train", self.TRAIN_ALGORITHMS, 0
            )
        )

        # ROI
        # VECTOR
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LAYER,
                "Input layer",
            )
        )
        # TABLE / COLUMN
        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_COLUMN,
                "Field (column must have classification number (e.g. '1' forest, '2' water...))",
                parentLayerParameterName=self.INPUT_LAYER,
                optional=False,
            )
        )  # save model

        self.addParameter(
            QgsProcessingParameterField(
                self.STAND_COLUMN,
                "Stand number (column must have unique id per stand)",
                parentLayerParameterName=self.INPUT_LAYER,
                optional=False,
            )
        )  # save model

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXITER,
                self.tr("Maximum iteration (default : 5)"),
                type=QgsProcessingParameterNumber.Integer,
                minValue=1,
                maxValue=99999,
                defaultValue=5,
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.PARAMGRID,
                self.tr("Parameters for the hyperparameters of the algorithm"),
                optional=True,
            )
        )
        # SAVE AS
        # SAVE MODEL
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MODEL, self.tr("Output model (to use for classifying)")
            )
        )
        """
        # SAVE CONFUSION MATRIX
        self.addParameter(
        QgsProcessingParameterFileDestination(
            self.OUTPUT_MATRIX,
            self.tr("Output confusion matrix"),
            fileFilter='csv'))#,
            #ext='csv'))
        """
        # SAVE DIR
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.SAVEDIR, self.tr("Directory to save every confusion matrix")
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        INPUT_RASTER = self.parameterAsRasterLayer(
            parameters, self.INPUT_RASTER, context
        )
        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)

        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        STAND_COLUMN = self.parameterAsFields(parameters, self.STAND_COLUMN, context)
        # SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)
        # TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
        # INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_MODEL = self.parameterAsFileOutput(
            parameters, self.OUTPUT_MODEL, context
        )
        # OUTPUT_MATRIX = self.parameterAsFileOutput(parameters, self.OUTPUT_MATRIX, context)

        SAVEDIR = self.parameterAsFileOutput(parameters, self.SAVEDIR, context)
        # Retrieve algo from code
        SLOO = self.parameterAsBool(parameters, self.SLOO, context)

        extraParam = {}

        extraParam["SLOO"] = SLOO
        # extraParam['maxIter']=False
        # extraParam['param_grid'] = dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6))

        MAXITER = self.parameterAsInt(parameters, self.MAXITER, context)

        if MAXITER == 0:
            MAXITER = False
        extraParam["maxIter"] = MAXITER

        PARAMGRID = self.parameterAsString(parameters, self.PARAMGRID, context)
        if PARAMGRID != "":
            extraParam["param_grid"] = eval(PARAMGRID)

        if not SAVEDIR.endswith("/"):
            SAVEDIR += "/"

        extraParam["saveDir"] = SAVEDIR

        extraParam["inStand"] = STAND_COLUMN[0]

        TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)

        # Retrieve algo from code
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]

        # eval(PARAM_GRID

        # learn model
        mainfunction.learnModel(
            INPUT_RASTER.source(),
            INPUT_LAYER.dataProvider().dataSourceUri().split("|")[0],
            INPUT_COLUMN[0],
            OUTPUT_MODEL,
            "STAND",
            0,
            None,
            SELECTED_ALGORITHM,
            feedback=feedback,
            extraParam=extraParam,
        )
        return {self.SAVEDIR: SAVEDIR, self.OUTPUT_MODEL: OUTPUT_MODEL}

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return trainSTANDalgorithm()

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Classification tool"
