# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dzetsaka_dock.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from qgis.gui import QgsCollapsibleGroupBox, QgsMapLayerComboBox
# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt import QtCore, QtGui, QtWidgets


class Ui_DockWidget(object):
    def setupUi(self, DockWidget):
        DockWidget.setObjectName("DockWidget")
        DockWidget.resize(350, 300)
        DockWidget.setMinimumSize(QtCore.QSize(366, 353))
        DockWidget.setMaximumSize(QtCore.QSize(600, 600))
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_8 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_8.setMinimumSize(QtCore.QSize(250, 0))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/parcguyane.jpg"))
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 0, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_2.setMinimumSize(QtCore.QSize(15, 15))
        self.label_2.setMaximumSize(QtCore.QSize(15, 15))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/raster.svg"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.inRaster = QgsMapLayerComboBox(self.dockWidgetContents)
        self.inRaster.setMinimumSize(QtCore.QSize(200, 0))
        self.inRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inRaster.setShowCrs(True)
        self.inRaster.setObjectName("inRaster")
        self.gridLayout.addWidget(self.inRaster, 0, 1, 1, 3)
        self.label_3 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_3.setMaximumSize(QtCore.QSize(15, 15))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/vector.svg"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.inShape = QgsMapLayerComboBox(self.dockWidgetContents)
        self.inShape.setMinimumSize(QtCore.QSize(90, 0))
        self.inShape.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inShape.setShowCrs(True)
        self.inShape.setObjectName("inShape")
        self.gridLayout.addWidget(self.inShape, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.dockWidgetContents)
        self.label.setMaximumSize(QtCore.QSize(20, 25))
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 2, 1, 1)
        self.checkInModel = QtWidgets.QCheckBox(self.dockWidgetContents)
        self.checkInModel.setMinimumSize(QtCore.QSize(110, 0))
        self.checkInModel.setMaximumSize(QtCore.QSize(110, 16777215))
        self.checkInModel.setObjectName("checkInModel")
        self.gridLayout.addWidget(self.checkInModel, 1, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_4.setMaximumSize(QtCore.QSize(15, 15))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/column.svg"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.inField = QtWidgets.QComboBox(self.dockWidgetContents)
        self.inField.setMinimumSize(QtCore.QSize(90, 0))
        self.inField.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inField.setObjectName("inField")
        self.gridLayout.addWidget(self.inField, 2, 1, 1, 1)
        self.inModel = QtWidgets.QLineEdit(self.dockWidgetContents)
        self.inModel.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inModel.sizePolicy().hasHeightForWidth())
        self.inModel.setSizePolicy(sizePolicy)
        self.inModel.setMinimumSize(QtCore.QSize(110, 0))
        self.inModel.setMaximumSize(QtCore.QSize(160, 16777215))
        self.inModel.setInputMask("")
        self.inModel.setText("")
        self.inModel.setObjectName("inModel")
        self.gridLayout.addWidget(self.inModel, 2, 2, 1, 2)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.outRaster = QtWidgets.QLineEdit(self.dockWidgetContents)
        self.outRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.outRaster.setObjectName("outRaster")
        self.gridLayout_5.addWidget(self.outRaster, 0, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(
            15,
            17,
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        self.gridLayout_5.addItem(spacerItem, 1, 0, 1, 1)
        self.performMagic = QtWidgets.QToolButton(self.dockWidgetContents)
        self.performMagic.setMinimumSize(QtCore.QSize(175, 0))
        self.performMagic.setObjectName("performMagic")
        self.gridLayout_5.addWidget(self.performMagic, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            15, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_5.addItem(spacerItem1, 1, 2, 1, 1)
        self.settingsButton = QtWidgets.QToolButton(self.dockWidgetContents)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/plugins/dzetsaka/img/settings.png"),
            QtGui.QIcon.Normal,
            QtGui.QIcon.On,
        )
        self.settingsButton.setIcon(icon)
        self.settingsButton.setObjectName("settingsButton")
        self.gridLayout_5.addWidget(self.settingsButton, 1, 3, 1, 1)
        self.outRasterButton = QtWidgets.QToolButton(self.dockWidgetContents)
        self.outRasterButton.setObjectName("outRasterButton")
        self.gridLayout_5.addWidget(self.outRasterButton, 0, 3, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_5, 3, 1, 1, 3)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            100, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout_2.addItem(spacerItem2, 3, 0, 1, 1)
        self.mGroupBox = QgsCollapsibleGroupBox(self.dockWidgetContents)
        self.mGroupBox.setEnabled(True)
        self.mGroupBox.setMaximumSize(QtCore.QSize(16777215, 23))
        self.mGroupBox.setFlat(True)
        self.mGroupBox.setCollapsed(True)
        self.mGroupBox.setScrollOnExpand(False)
        self.mGroupBox.setSaveCollapsedState(True)
        self.mGroupBox.setSaveCheckedState(False)
        self.mGroupBox.setObjectName("mGroupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.mGroupBox)
        self.gridLayout_3.setContentsMargins(0, -1, 0, -1)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_7 = QtWidgets.QLabel(self.mGroupBox)
        self.label_7.setMaximumSize(QtCore.QSize(20, 20))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/mask.svg"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 0, 1, 1)
        self.checkInMask = QtWidgets.QCheckBox(self.mGroupBox)
        self.checkInMask.setMinimumSize(QtCore.QSize(40, 0))
        self.checkInMask.setMaximumSize(QtCore.QSize(140, 16777215))
        self.checkInMask.setObjectName("checkInMask")
        self.gridLayout_3.addWidget(self.checkInMask, 0, 1, 1, 2)
        self.inMask = QtWidgets.QLineEdit(self.mGroupBox)
        self.inMask.setEnabled(False)
        self.inMask.setMinimumSize(QtCore.QSize(70, 20))
        self.inMask.setObjectName("inMask")
        self.gridLayout_3.addWidget(self.inMask, 0, 4, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.mGroupBox)
        self.label_5.setMaximumSize(QtCore.QSize(20, 20))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/confidence.png"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 1, 0, 1, 1)
        self.checkInConfidence = QtWidgets.QCheckBox(self.mGroupBox)
        self.checkInConfidence.setMinimumSize(QtCore.QSize(140, 0))
        self.checkInConfidence.setMaximumSize(QtCore.QSize(140, 16777215))
        self.checkInConfidence.setObjectName("checkInConfidence")
        self.gridLayout_3.addWidget(self.checkInConfidence, 1, 1, 1, 3)
        self.outConfidenceMap = QtWidgets.QLineEdit(self.mGroupBox)
        self.outConfidenceMap.setEnabled(False)
        self.outConfidenceMap.setMinimumSize(QtCore.QSize(70, 20))
        self.outConfidenceMap.setObjectName("outConfidenceMap")
        self.gridLayout_3.addWidget(self.outConfidenceMap, 1, 4, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.mGroupBox)
        self.label_6.setMaximumSize(QtCore.QSize(20, 20))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/model.svg"))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 2, 0, 1, 1)
        self.checkOutModel = QtWidgets.QCheckBox(self.mGroupBox)
        self.checkOutModel.setMaximumSize(QtCore.QSize(140, 16777215))
        self.checkOutModel.setObjectName("checkOutModel")
        self.gridLayout_3.addWidget(self.checkOutModel, 2, 1, 1, 3)
        self.outModel = QtWidgets.QLineEdit(self.mGroupBox)
        self.outModel.setEnabled(False)
        self.outModel.setMinimumSize(QtCore.QSize(70, 20))
        self.outModel.setObjectName("outModel")
        self.gridLayout_3.addWidget(self.outModel, 2, 4, 1, 2)
        self.label_11 = QtWidgets.QLabel(self.mGroupBox)
        self.label_11.setMaximumSize(QtCore.QSize(20, 20))
        self.label_11.setText("")
        self.label_11.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/table.png"))
        self.label_11.setScaledContents(True)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 3, 0, 1, 1)
        self.checkOutMatrix = QtWidgets.QCheckBox(self.mGroupBox)
        self.checkOutMatrix.setMaximumSize(QtCore.QSize(140, 16777215))
        self.checkOutMatrix.setObjectName("checkOutMatrix")
        self.gridLayout_3.addWidget(self.checkOutMatrix, 3, 1, 1, 3)
        self.outMatrix = QtWidgets.QLineEdit(self.mGroupBox)
        self.outMatrix.setEnabled(False)
        self.outMatrix.setMinimumSize(QtCore.QSize(70, 20))
        self.outMatrix.setObjectName("outMatrix")
        self.gridLayout_3.addWidget(self.outMatrix, 3, 4, 1, 2)
        self.inSplit = QtWidgets.QSpinBox(self.mGroupBox)
        self.inSplit.setEnabled(False)
        self.inSplit.setMinimumSize(QtCore.QSize(65, 20))
        self.inSplit.setMaximum(100)
        self.inSplit.setProperty("value", 100)
        self.inSplit.setObjectName("inSplit")
        self.gridLayout_3.addWidget(self.inSplit, 4, 4, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(
            36, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_3.addItem(spacerItem3, 4, 5, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(
            15, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_3.addItem(spacerItem4, 4, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.mGroupBox)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 4, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.mGroupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 4, 2, 1, 1)
        self.gridLayout_2.addWidget(self.mGroupBox, 2, 0, 1, 1)
        DockWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(DockWidget)
        QtCore.QMetaObject.connectSlotsByName(DockWidget)
        DockWidget.setTabOrder(self.inRaster, self.inShape)
        DockWidget.setTabOrder(self.inShape, self.inField)
        DockWidget.setTabOrder(self.inField, self.checkInModel)
        DockWidget.setTabOrder(self.checkInModel, self.inModel)
        DockWidget.setTabOrder(self.inModel, self.outRaster)
        DockWidget.setTabOrder(self.outRaster, self.performMagic)
        DockWidget.setTabOrder(self.performMagic, self.mGroupBox)
        DockWidget.setTabOrder(self.mGroupBox, self.checkInConfidence)
        DockWidget.setTabOrder(self.checkInConfidence, self.outConfidenceMap)
        DockWidget.setTabOrder(self.outConfidenceMap, self.checkOutModel)
        DockWidget.setTabOrder(self.checkOutModel, self.outModel)
        DockWidget.setTabOrder(self.outModel, self.checkOutMatrix)
        DockWidget.setTabOrder(self.checkOutMatrix, self.outMatrix)
        DockWidget.setTabOrder(self.outMatrix, self.inSplit)

    def retranslateUi(self, DockWidget):
        _translate = QtCore.QCoreApplication.translate
        DockWidget.setWindowTitle(
            _translate("DockWidget", "dzetsaka : classification tool")
        )
        self.label_2.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>The image to classify</p></body></html>",
            )
        )
        self.label_3.setToolTip(
            _translate("DockWidget", "<html><head/><body><p>Your ROI</p></body></html>")
        )
        self.label.setText(_translate("DockWidget", "or"))
        self.checkInModel.setText(_translate("DockWidget", "Load model"))
        self.label_4.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>Column name where class number is stored</p></body></html>",
            )
        )
        self.inModel.setPlaceholderText(_translate("DockWidget", "Model"))
        self.outRaster.setPlaceholderText(
            _translate("DockWidget", "Classification. Leave empty for temporary file")
        )
        self.performMagic.setText(
            _translate("DockWidget", "Perform the classification")
        )
        self.settingsButton.setText(_translate("DockWidget", "..."))
        self.outRasterButton.setText(_translate("DockWidget", "..."))
        self.mGroupBox.setTitle(_translate("DockWidget", "Optional"))
        self.label_7.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>Mask where 0 are the pixels to ignore and 1 to classify</p></body></html>",
            )
        )
        self.checkInMask.setText(_translate("DockWidget", "Mask "))
        self.inMask.setPlaceholderText(
            _translate("DockWidget", "Automatic find filename_mask.ext")
        )
        self.label_5.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>Mask where 0 are the pixels to ignore and 1 to classify</p></body></html>",
            )
        )
        self.checkInConfidence.setToolTip(
            _translate(
                "DockWidget",
                "Create a confidence map for each classified pixel. 1 is total confidence, 0 is null.",
            )
        )
        self.checkInConfidence.setText(_translate("DockWidget", "Confidence map"))
        self.outConfidenceMap.setPlaceholderText(
            _translate("DockWidget", "Map of confidence")
        )
        self.label_6.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>If you want to save the model for a further use and with another image</p></body></html>",
            )
        )
        self.checkOutModel.setText(_translate("DockWidget", "Save model"))
        self.outModel.setPlaceholderText(
            _translate("DockWidget", "To use with another image")
        )
        self.label_11.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>If you want to save the model for a further use and with another image</p></body></html>",
            )
        )
        self.checkOutMatrix.setText(_translate("DockWidget", "Save matrix"))
        self.outMatrix.setPlaceholderText(
            _translate("DockWidget", "Save confusion matrix")
        )
        self.inSplit.setSuffix(_translate("DockWidget", "%"))
        self.label_9.setText(_translate("DockWidget", "Split"))
        self.label_10.setToolTip(
            _translate(
                "DockWidget",
                "<html><head/><body><p>In percent, number of polygons used for classification and number used for stats (confusion matrix, overall accuracy and Kappa)</p></body></html>",
            )
        )
        self.label_10.setText(_translate("DockWidget", "(?)"))
