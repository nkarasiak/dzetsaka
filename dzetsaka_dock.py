# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dzetsaka_dock.ui'
#
# Created: Tue May 17 16:33:39 2016
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
        DockWidget.resize(400, 416)
        DockWidget.setMinimumSize(QtCore.QSize(400, 400))
        DockWidget.setMaximumSize(QtCore.QSize(524287, 500))
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.gridLayout = QtGui.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.inField = QtGui.QComboBox(self.dockWidgetContents)
        self.inField.setMinimumSize(QtCore.QSize(180, 0))
        self.inField.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inField.setObjectName(_fromUtf8("inField"))
        self.gridLayout_2.addWidget(self.inField, 3, 1, 1, 2)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 5, 4, 1, 2)
        self.label_3 = QtGui.QLabel(self.dockWidgetContents)
        self.label_3.setMaximumSize(QtCore.QSize(25, 60))
        self.label_3.setText(_fromUtf8(""))
        self.label_3.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/vector.svg")))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)
        self.inShape = gui.QgsMapLayerComboBox(self.dockWidgetContents)
        self.inShape.setMinimumSize(QtCore.QSize(180, 0))
        self.inShape.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inShape.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.PolygonLayer)
        self.inShape.setObjectName(_fromUtf8("inShape"))
        self.gridLayout_2.addWidget(self.inShape, 1, 1, 1, 2)
        self.inRaster = gui.QgsMapLayerComboBox(self.dockWidgetContents)
        self.inRaster.setMinimumSize(QtCore.QSize(200, 0))
        self.inRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inRaster.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.RasterLayer)
        self.inRaster.setObjectName(_fromUtf8("inRaster"))
        self.gridLayout_2.addWidget(self.inRaster, 0, 1, 1, 5)
        self.label_2 = QtGui.QLabel(self.dockWidgetContents)
        self.label_2.setMinimumSize(QtCore.QSize(25, 25))
        self.label_2.setMaximumSize(QtCore.QSize(25, 30))
        self.label_2.setText(_fromUtf8(""))
        self.label_2.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/raster.svg")))
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.outRaster = QtGui.QLineEdit(self.dockWidgetContents)
        self.outRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.outRaster.setObjectName(_fromUtf8("outRaster"))
        self.gridLayout_2.addWidget(self.outRaster, 4, 0, 1, 5)
        self.inModel = QtGui.QLineEdit(self.dockWidgetContents)
        self.inModel.setEnabled(False)
        self.inModel.setMaximumSize(QtCore.QSize(110, 16777215))
        self.inModel.setInputMask(_fromUtf8(""))
        self.inModel.setText(_fromUtf8(""))
        self.inModel.setObjectName(_fromUtf8("inModel"))
        self.gridLayout_2.addWidget(self.inModel, 2, 4, 2, 2)
        self.checkInModel = QtGui.QCheckBox(self.dockWidgetContents)
        self.checkInModel.setMaximumSize(QtCore.QSize(110, 16777215))
        self.checkInModel.setObjectName(_fromUtf8("checkInModel"))
        self.gridLayout_2.addWidget(self.checkInModel, 1, 4, 1, 2)
        self.outRasterButton = QtGui.QToolButton(self.dockWidgetContents)
        self.outRasterButton.setObjectName(_fromUtf8("outRasterButton"))
        self.gridLayout_2.addWidget(self.outRasterButton, 4, 5, 1, 1)
        self.label_4 = QtGui.QLabel(self.dockWidgetContents)
        self.label_4.setMaximumSize(QtCore.QSize(25, 60))
        self.label_4.setText(_fromUtf8(""))
        self.label_4.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/column.svg")))
        self.label_4.setScaledContents(False)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_2.addWidget(self.label_4, 3, 0, 1, 1)
        self.label = QtGui.QLabel(self.dockWidgetContents)
        self.label.setMaximumSize(QtCore.QSize(25, 25))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_2.addWidget(self.label, 1, 3, 2, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 17, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 5, 0, 1, 2)
        self.performMagic = QtGui.QToolButton(self.dockWidgetContents)
        self.performMagic.setMinimumSize(QtCore.QSize(184, 0))
        self.performMagic.setObjectName(_fromUtf8("performMagic"))
        self.gridLayout_2.addWidget(self.performMagic, 5, 2, 1, 2)
        self.gridLayout.addLayout(self.gridLayout_2, 1, 0, 1, 3)
        self.label_7 = QtGui.QLabel(self.dockWidgetContents)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout.addWidget(self.label_7, 2, 0, 1, 2)
        spacerItem2 = QtGui.QSpacerItem(318, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 2, 2, 1, 1)
        self.label_5 = QtGui.QLabel(self.dockWidgetContents)
        self.label_5.setMaximumSize(QtCore.QSize(25, 25))
        self.label_5.setText(_fromUtf8(""))
        self.label_5.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/mask.svg")))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.checkInMask = QtGui.QCheckBox(self.dockWidgetContents)
        self.checkInMask.setObjectName(_fromUtf8("checkInMask"))
        self.gridLayout.addWidget(self.checkInMask, 3, 1, 1, 1)
        self.inMask = QtGui.QLineEdit(self.dockWidgetContents)
        self.inMask.setEnabled(False)
        self.inMask.setMinimumSize(QtCore.QSize(200, 0))
        self.inMask.setObjectName(_fromUtf8("inMask"))
        self.gridLayout.addWidget(self.inMask, 3, 2, 1, 1)
        self.label_6 = QtGui.QLabel(self.dockWidgetContents)
        self.label_6.setMaximumSize(QtCore.QSize(25, 25))
        self.label_6.setText(_fromUtf8(""))
        self.label_6.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/model.svg")))
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.checkOutModel = QtGui.QCheckBox(self.dockWidgetContents)
        self.checkOutModel.setObjectName(_fromUtf8("checkOutModel"))
        self.gridLayout.addWidget(self.checkOutModel, 4, 1, 1, 1)
        self.outModel = QtGui.QLineEdit(self.dockWidgetContents)
        self.outModel.setEnabled(False)
        self.outModel.setMinimumSize(QtCore.QSize(200, 0))
        self.outModel.setObjectName(_fromUtf8("outModel"))
        self.gridLayout.addWidget(self.outModel, 4, 2, 1, 1)
        spacerItem3 = QtGui.QSpacerItem(20, 29, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 5, 2, 1, 1)
        self.label_8 = QtGui.QLabel(self.dockWidgetContents)
        self.label_8.setMinimumSize(QtCore.QSize(300, 0))
        self.label_8.setText(_fromUtf8(""))
        self.label_8.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/guyane.jpg")))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 3)
        DockWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(DockWidget)
        QtCore.QMetaObject.connectSlotsByName(DockWidget)

    def retranslateUi(self, DockWidget):
        DockWidget.setWindowTitle(_translate("DockWidget", "dzetsaka : classification tool", None))
        self.label_3.setToolTip(_translate("DockWidget", "<html><head/><body><p>Your ROI</p></body></html>", None))
        self.label_2.setToolTip(_translate("DockWidget", "<html><head/><body><p>The image to classify</p></body></html>", None))
        self.outRaster.setPlaceholderText(_translate("DockWidget", "Classification. Leave empty for temporary file", None))
        self.inModel.setPlaceholderText(_translate("DockWidget", "Model", None))
        self.checkInModel.setText(_translate("DockWidget", "Load model", None))
        self.outRasterButton.setText(_translate("DockWidget", "...", None))
        self.label_4.setToolTip(_translate("DockWidget", "<html><head/><body><p>Column name where class number is stored</p></body></html>", None))
        self.label.setText(_translate("DockWidget", "or", None))
        self.performMagic.setText(_translate("DockWidget", "Perform the classification", None))
        self.label_7.setText(_translate("DockWidget", "> Optional", None))
        self.label_5.setToolTip(_translate("DockWidget", "<html><head/><body><p>Mask where 0 are the pixels to ignore and 1 to classify</p></body></html>", None))
        self.checkInMask.setText(_translate("DockWidget", "Mask", None))
        self.inMask.setPlaceholderText(_translate("DockWidget", "Automatic find filename_mask.ext", None))
        self.label_6.setToolTip(_translate("DockWidget", "<html><head/><body><p>If you want to save the model for a further use and with another image</p></body></html>", None))
        self.checkOutModel.setText(_translate("DockWidget", "Save model", None))
        self.outModel.setPlaceholderText(_translate("DockWidget", "To use with another image", None))

from qgis import gui