# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dzetsaka_dock.ui'
#
# Created: Tue Jun 21 16:04:56 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_DockWidget(object):
    def setupUi(self, DockWidget):
        DockWidget.setObjectName(_fromUtf8("DockWidget"))
        DockWidget.resize(305, 320)
        DockWidget.setMinimumSize(QtCore.QSize(300, 320))
        DockWidget.setMaximumSize(QtCore.QSize(524287, 1000))
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.gridLayout_2 = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_8 = QtGui.QLabel(self.dockWidgetContents)
        self.label_8.setMinimumSize(QtCore.QSize(250, 0))
        self.label_8.setText(_fromUtf8(""))
        self.label_8.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/parcguyane.jpg")))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout_2.addWidget(self.label_8, 0, 0, 1, 1)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_2 = QtGui.QLabel(self.dockWidgetContents)
        self.label_2.setMinimumSize(QtCore.QSize(15, 15))
        self.label_2.setMaximumSize(QtCore.QSize(15, 15))
        self.label_2.setText(_fromUtf8(""))
        self.label_2.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/raster.svg")))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.inRaster = gui.QgsMapLayerComboBox(self.dockWidgetContents)
        self.inRaster.setMinimumSize(QtCore.QSize(200, 0))
        self.inRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inRaster.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.RasterLayer)
        self.inRaster.setObjectName(_fromUtf8("inRaster"))
        self.gridLayout.addWidget(self.inRaster, 0, 1, 1, 3)
        self.label_3 = QtGui.QLabel(self.dockWidgetContents)
        self.label_3.setMaximumSize(QtCore.QSize(15, 15))
        self.label_3.setText(_fromUtf8(""))
        self.label_3.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/vector.svg")))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.inShape = gui.QgsMapLayerComboBox(self.dockWidgetContents)
        self.inShape.setMinimumSize(QtCore.QSize(120, 0))
        self.inShape.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inShape.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.PolygonLayer)
        self.inShape.setObjectName(_fromUtf8("inShape"))
        self.gridLayout.addWidget(self.inShape, 1, 1, 1, 1)
        self.label = QtGui.QLabel(self.dockWidgetContents)
        self.label.setMaximumSize(QtCore.QSize(20, 25))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 2, 1, 1)
        self.checkInModel = QtGui.QCheckBox(self.dockWidgetContents)
        self.checkInModel.setMinimumSize(QtCore.QSize(110, 0))
        self.checkInModel.setMaximumSize(QtCore.QSize(110, 16777215))
        self.checkInModel.setObjectName(_fromUtf8("checkInModel"))
        self.gridLayout.addWidget(self.checkInModel, 1, 3, 1, 1)
        self.label_4 = QtGui.QLabel(self.dockWidgetContents)
        self.label_4.setMaximumSize(QtCore.QSize(15, 15))
        self.label_4.setText(_fromUtf8(""))
        self.label_4.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/column.svg")))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.inField = QtGui.QComboBox(self.dockWidgetContents)
        self.inField.setMinimumSize(QtCore.QSize(120, 0))
        self.inField.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inField.setObjectName(_fromUtf8("inField"))
        self.gridLayout.addWidget(self.inField, 2, 1, 1, 1)
        self.inModel = QtGui.QLineEdit(self.dockWidgetContents)
        self.inModel.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inModel.sizePolicy().hasHeightForWidth())
        self.inModel.setSizePolicy(sizePolicy)
        self.inModel.setMinimumSize(QtCore.QSize(110, 0))
        self.inModel.setMaximumSize(QtCore.QSize(180, 16777215))
        self.inModel.setInputMask(_fromUtf8(""))
        self.inModel.setText(_fromUtf8(""))
        self.inModel.setObjectName(_fromUtf8("inModel"))
        self.gridLayout.addWidget(self.inModel, 2, 2, 1, 2)
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.outRaster = QtGui.QLineEdit(self.dockWidgetContents)
        self.outRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.outRaster.setObjectName(_fromUtf8("outRaster"))
        self.gridLayout_5.addWidget(self.outRaster, 0, 0, 1, 3)
        spacerItem = QtGui.QSpacerItem(29, 17, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem, 1, 0, 1, 1)
        self.performMagic = QtGui.QToolButton(self.dockWidgetContents)
        self.performMagic.setMinimumSize(QtCore.QSize(175, 0))
        self.performMagic.setObjectName(_fromUtf8("performMagic"))
        self.gridLayout_5.addWidget(self.performMagic, 1, 1, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(26, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem1, 1, 2, 1, 1)
        self.settingsButton = QtGui.QToolButton(self.dockWidgetContents)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/settings.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.settingsButton.setIcon(icon)
        self.settingsButton.setObjectName(_fromUtf8("settingsButton"))
        self.gridLayout_5.addWidget(self.settingsButton, 1, 3, 1, 1)
        self.outRasterButton = QtGui.QToolButton(self.dockWidgetContents)
        self.outRasterButton.setObjectName(_fromUtf8("outRasterButton"))
        self.gridLayout_5.addWidget(self.outRasterButton, 0, 3, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_5, 3, 1, 1, 3)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.mGroupBox = gui.QgsCollapsibleGroupBox(self.dockWidgetContents)
        self.mGroupBox.setEnabled(True)
        self.mGroupBox.setMaximumSize(QtCore.QSize(16777215, 20))
        self.mGroupBox.setFlat(True)
        self.mGroupBox.setCollapsed(True)
        self.mGroupBox.setScrollOnExpand(False)
        self.mGroupBox.setSaveCollapsedState(True)
        self.mGroupBox.setSaveCheckedState(False)
        self.mGroupBox.setObjectName(_fromUtf8("mGroupBox"))
        self.gridLayout_4 = QtGui.QGridLayout(self.mGroupBox)
        self.gridLayout_4.setContentsMargins(-1, 9, 0, 0)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_5 = QtGui.QLabel(self.mGroupBox)
        self.label_5.setMaximumSize(QtCore.QSize(20, 20))
        self.label_5.setText(_fromUtf8(""))
        self.label_5.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/mask.svg")))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.checkInMask = QtGui.QCheckBox(self.mGroupBox)
        self.checkInMask.setObjectName(_fromUtf8("checkInMask"))
        self.gridLayout_3.addWidget(self.checkInMask, 0, 1, 1, 2)
        self.inMask = QtGui.QLineEdit(self.mGroupBox)
        self.inMask.setEnabled(False)
        self.inMask.setMinimumSize(QtCore.QSize(80, 20))
        self.inMask.setObjectName(_fromUtf8("inMask"))
        self.gridLayout_3.addWidget(self.inMask, 0, 4, 1, 2)
        self.label_6 = QtGui.QLabel(self.mGroupBox)
        self.label_6.setMaximumSize(QtCore.QSize(20, 20))
        self.label_6.setText(_fromUtf8(""))
        self.label_6.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/model.svg")))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_3.addWidget(self.label_6, 1, 0, 1, 1)
        self.checkOutModel = QtGui.QCheckBox(self.mGroupBox)
        self.checkOutModel.setObjectName(_fromUtf8("checkOutModel"))
        self.gridLayout_3.addWidget(self.checkOutModel, 1, 1, 1, 3)
        self.outModel = QtGui.QLineEdit(self.mGroupBox)
        self.outModel.setEnabled(False)
        self.outModel.setMinimumSize(QtCore.QSize(80, 20))
        self.outModel.setObjectName(_fromUtf8("outModel"))
        self.gridLayout_3.addWidget(self.outModel, 1, 4, 1, 2)
        self.label_11 = QtGui.QLabel(self.mGroupBox)
        self.label_11.setMaximumSize(QtCore.QSize(20, 20))
        self.label_11.setText(_fromUtf8(""))
        self.label_11.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/table.png")))
        self.label_11.setScaledContents(True)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.gridLayout_3.addWidget(self.label_11, 2, 0, 1, 1)
        self.checkOutMatrix = QtGui.QCheckBox(self.mGroupBox)
        self.checkOutMatrix.setObjectName(_fromUtf8("checkOutMatrix"))
        self.gridLayout_3.addWidget(self.checkOutMatrix, 2, 1, 1, 3)
        self.outMatrix = QtGui.QLineEdit(self.mGroupBox)
        self.outMatrix.setEnabled(False)
        self.outMatrix.setMinimumSize(QtCore.QSize(129, 20))
        self.outMatrix.setObjectName(_fromUtf8("outMatrix"))
        self.gridLayout_3.addWidget(self.outMatrix, 2, 4, 1, 2)
        spacerItem2 = QtGui.QSpacerItem(60, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 3, 0, 1, 2)
        self.label_9 = QtGui.QLabel(self.mGroupBox)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.gridLayout_3.addWidget(self.label_9, 3, 2, 1, 1)
        self.label_10 = QtGui.QLabel(self.mGroupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_10.setFont(font)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.gridLayout_3.addWidget(self.label_10, 3, 3, 1, 1)
        self.inSplit = QtGui.QSpinBox(self.mGroupBox)
        self.inSplit.setEnabled(False)
        self.inSplit.setMinimumSize(QtCore.QSize(0, 20))
        self.inSplit.setMaximum(100)
        self.inSplit.setProperty("value", 100)
        self.inSplit.setObjectName(_fromUtf8("inSplit"))
        self.gridLayout_3.addWidget(self.inSplit, 3, 4, 1, 1)
        spacerItem3 = QtGui.QSpacerItem(50, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 3, 5, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.mGroupBox, 2, 0, 1, 1)
        spacerItem4 = QtGui.QSpacerItem(100, 1, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem4, 3, 0, 1, 1)
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
        DockWidget.setTabOrder(self.mGroupBox, self.checkInMask)
        DockWidget.setTabOrder(self.checkInMask, self.inMask)
        DockWidget.setTabOrder(self.inMask, self.checkOutModel)
        DockWidget.setTabOrder(self.checkOutModel, self.outModel)
        DockWidget.setTabOrder(self.outModel, self.checkOutMatrix)
        DockWidget.setTabOrder(self.checkOutMatrix, self.outMatrix)
        DockWidget.setTabOrder(self.outMatrix, self.inSplit)

    def retranslateUi(self, DockWidget):
        DockWidget.setWindowTitle(_translate("DockWidget", "dzetsaka : classification tool", None))
        self.label_2.setToolTip(_translate("DockWidget", "<html><head/><body><p>The image to classify</p></body></html>", None))
        self.label_3.setToolTip(_translate("DockWidget", "<html><head/><body><p>Your ROI</p></body></html>", None))
        self.label.setText(_translate("DockWidget", "or", None))
        self.checkInModel.setText(_translate("DockWidget", "Load model", None))
        self.label_4.setToolTip(_translate("DockWidget", "<html><head/><body><p>Column name where class number is stored</p></body></html>", None))
        self.inModel.setPlaceholderText(_translate("DockWidget", "Model", None))
        self.outRaster.setPlaceholderText(_translate("DockWidget", "Classification. Leave empty for temporary file", None))
        self.performMagic.setText(_translate("DockWidget", "Perform the classification", None))
        self.settingsButton.setText(_translate("DockWidget", "...", None))
        self.outRasterButton.setText(_translate("DockWidget", "...", None))
        self.mGroupBox.setTitle(_translate("DockWidget", "Optional", None))
        self.label_5.setToolTip(_translate("DockWidget", "<html><head/><body><p>Mask where 0 are the pixels to ignore and 1 to classify</p></body></html>", None))
        self.checkInMask.setText(_translate("DockWidget", "Mask", None))
        self.inMask.setPlaceholderText(_translate("DockWidget", "Automatic find filename_mask.ext", None))
        self.label_6.setToolTip(_translate("DockWidget", "<html><head/><body><p>If you want to save the model for a further use and with another image</p></body></html>", None))
        self.checkOutModel.setText(_translate("DockWidget", "Save model", None))
        self.outModel.setPlaceholderText(_translate("DockWidget", "To use with another image", None))
        self.label_11.setToolTip(_translate("DockWidget", "<html><head/><body><p>If you want to save the model for a further use and with another image</p></body></html>", None))
        self.checkOutMatrix.setText(_translate("DockWidget", "Save matrix", None))
        self.outMatrix.setPlaceholderText(_translate("DockWidget", "Save confusion matrix", None))
        self.label_9.setText(_translate("DockWidget", "Split", None))
        self.label_10.setToolTip(_translate("DockWidget", "<html><head/><body><p>In percent, number of polygons used for classification and number used for stats (confusion matrix, overall accuracy and Kappa)</p></body></html>", None))
        self.label_10.setText(_translate("DockWidget", "(?)", None))
        self.inSplit.setSuffix(_translate("DockWidget", "%", None))

from qgis import gui
