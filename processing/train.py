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

from builtins import str

from qgis.PyQt.QtGui import QIcon
from PyQt5.QtCore import QCoreApplication
# from PyQt5.QtWidgets import QMessageBox

from qgis.core import (
    QgsMessageLog,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterFileDestination,
)

import os

from .. import classifier_config
from ..scripts.mainfunction import learnModel

pluginPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class trainAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = "INPUT_RASTER"
    INPUT_LAYER = "INPUT_LAYER"
    INPUT_COLUMN = "INPUT_COLUMN"
    TRAIN = "TRAIN"
    TRAIN_ALGORITHMS = classifier_config.UI_DISPLAY_NAMES
    TRAIN_ALGORITHMS_CODE = classifier_config.CLASSIFIER_CODES
    SPLIT_PERCENT = "SPLIT_PERCENT"
    OUTPUT_MODEL = "OUTPUT_MODEL"
    OUTPUT_MATRIX = "OUTPUT_MATRIX"
    PARAMGRID = "PARAMGRID"

    def shortHelpString(self):
        return self.tr(
            "Train classifier.\n \n \
                       Parameters for Cross Validation can be fit using a dictionnary.\n \
                       \
                       <h3>Classifier (paramgrid)</h3> \n \
                       Param grid can be fit for each algorithm : \n \
                       <h4>Random-Forest</h4> \n \
                       e.g. : dict(n_estimators=2**np.arange(4,10),max_features=[5,10,20,30,40],min_samples_split=range(2,6)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier \n \
                       \n \
                       <h4>SVM</h4> \
                       e.g. : dict(gamma=2.0**np.arange(-4,4), C=10.0**np.arange(-2,5)) \n \
                       More information : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html "
        )

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "Train algorithm"

    def icon(self):
        return QIcon(os.path.join(pluginPath, "icon.png"))

    def initAlgorithm(self, config=None):
        # The name that the user will see in the toolbox

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, self.tr("Input raster")
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

        # Train algorithm

        self.addParameter(
            QgsProcessingParameterEnum(
                self.TRAIN, "Select algorithm to train", self.TRAIN_ALGORITHMS, 0
            )
        )

        # SPLIT %

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SPLIT_PERCENT,
                self.tr("Pixels (%) to keep for validation."),
                type=QgsProcessingParameterNumber.Integer,
                minValue=0,
                maxValue=100,
                defaultValue=50,
            )
        )

        # SAVE AS
        # SAVE MODEL
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MODEL, self.tr("Output model (to use for classifying)")
            )
        )

        # SAVE CONFUSION MATRIX
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MATRIX, self.tr("Output confusion matrix"), fileFilter="csv"
            )
        )  # ,
        # ext='csv'))
        # PARAM GRID
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAMGRID,
                self.tr("Parameters for the hyperparameters of the algorithm"),
                optional=True,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        INPUT_RASTER = self.parameterAsRasterLayer(
            parameters, self.INPUT_RASTER, context
        )

        INPUT_LAYER = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)

        INPUT_COLUMN = self.parameterAsFields(parameters, self.INPUT_COLUMN, context)
        SPLIT_PERCENT = self.parameterAsInt(parameters, self.SPLIT_PERCENT, context)

        SPLIT_PERCENT = (
            100 - SPLIT_PERCENT
        )  # if 30 means 30% of valid per class, 70% of train

        TRAIN = self.parameterAsEnums(parameters, self.TRAIN, context)
        # INPUT_RASTER = self.getParameterValue(self.INPUT_RASTER)
        OUTPUT_MODEL = self.parameterAsFileOutput(
            parameters, self.OUTPUT_MODEL, context
        )
        OUTPUT_MATRIX = self.parameterAsFileOutput(
            parameters, self.OUTPUT_MATRIX, context
        )

        # Retrieve algo from code
        SELECTED_ALGORITHM = self.TRAIN_ALGORITHMS_CODE[TRAIN[0]]
        QgsMessageLog.logMessage(str(SELECTED_ALGORITHM))

        libOk = True
        PARAMGRID = self.parameterAsString(parameters, self.PARAMGRID, context)
        if PARAMGRID != "":
            extraParam = {}
            extraParam["param_grid"] = eval(PARAMGRID)
        else:
            extraParam = None

        if (
            SELECTED_ALGORITHM == "RF"
            or SELECTED_ALGORITHM == "SVM"
            or SELECTED_ALGORITHM == "KNN"
        ):
            try:
                import sklearn
                import joblib  # Test for joblib dependency
            except ImportError as e:
                if "joblib" in str(e):
                    raise ImportError("Missing dependency: joblib. Please install joblib package (e.g., pip install joblib or your system's package manager)")
                else:
                    raise ImportError("You need to install scikit-learn library and its dependencies")
                libOk = False

        # learn model
        if libOk:
            learner = learnModel(
                raster_path=INPUT_RASTER.source(),
                vector_path=INPUT_LAYER.dataProvider().dataSourceUri().split("|")[0],
                class_field=INPUT_COLUMN[0],
                model_path=OUTPUT_MODEL,
                split_config=SPLIT_PERCENT,
                random_seed=0,
                matrix_path=OUTPUT_MATRIX,
                classifier=SELECTED_ALGORITHM,
                extraParam=extraParam,
                feedback=feedback,
            )
            return {self.OUTPUT_MATRIX: OUTPUT_MATRIX, self.OUTPUT_MODEL: OUTPUT_MODEL}

        else:
            return {"Missing library": str(OUTPUT_MATRIX)}
            # QMessageBox.about(None, "Missing library", "Please install scikit-learn library to use"+str(SELECTED_ALGORITHM))

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return trainAlgorithm()

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
