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

from qgis.core import (
    QgsMessageLog,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterVectorDestination,
)

import os

# from PyQt5.QtWidgets import QMessageBox
from ..scripts import function_vector

pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class splitTrain(QgsProcessingAlgorithm):
    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    METHOD = "METHOD"
    METHOD_VALUES = ["Percent", "Count value"]
    VALUE = "VALUE"
    OUTPUT_VALIDATION = "OUTPUT_VALIDATION"
    OUTPUT_TRAIN = "OUTPUT_TRAIN"

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Split train and validation"

    def icon(self):
        return QIcon(os.path.join(pluginPath, "icon.png"))

    def initAlgorithm(self, config=None):
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

        # Train algorithm

        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                "Select method for splitting dataset",
                self.METHOD_VALUES,
                0,
            )
        )

        # SPLIT %

        self.addParameter(
            QgsProcessingParameterNumber(
                self.VALUE,
                self.tr(
                    "Select 50 for 50% if PERCENT method. Else, value represents whole size of test sample."
                ),
                type=QgsProcessingParameterNumber.Integer,
                minValue=1,
                maxValue=99999,
                defaultValue=50,
            )
        )

        # SAVE AS
        # SAVE MODEL
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VALIDATION, self.tr("Output validation")
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_TRAIN, self.tr("Output train")
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        METHOD = self.parameterAsEnums(parameters, self.METHOD, context)

        OUTPUT_TRAIN = self.parameterAsOutputLayer(
            parameters, self.OUTPUT_TRAIN, context
        )
        OUTPUT_VALIDATION = self.parameterAsOutputLayer(
            parameters, self.OUTPUT_VALIDATION, context
        )

        VALUE = self.parameterAsInt(parameters, self.VALUE, context)
        # Retrieve algo from code
        selectedMETHOD = self.METHOD_VALUES[METHOD[0]]

        if selectedMETHOD == "Percent":
            percent = True
        else:
            percent = False

        libOk = True

        try:
            pass
        except BaseException:
            libOk = False

        if libOk:
            function_vector.randomInSubset(
                INPUT_LAYER.dataProvider().dataSourceUri().split("|")[0],
                str(INPUT_COLUMN[0]),
                OUTPUT_VALIDATION,
                OUTPUT_TRAIN,
                VALUE,
                percent,
            )
            return {
                self.OUTPUT_TRAIN: OUTPUT_TRAIN,
                self.OUTPUT_VALIDATION: OUTPUT_VALIDATION,
            }
        else:
            # QMessageBox(None, "Please install scikit-learn library")
            QgsMessageLog.logMessage("Please install scikit-learn library")

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return splitTrain()

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
        return "Vector manipulation"
