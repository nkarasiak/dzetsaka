# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'confusion_dock.ui'
#
# Created: Sun Jun 12 16:28:12 2016
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

class Ui_confusionDock(object):
    def setupUi(self, confusionDock):
        confusionDock.setObjectName(_fromUtf8("confusionDock"))
        confusionDock.resize(488, 394)
        self.gridLayout = QtGui.QGridLayout(confusionDock)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_3 = QtGui.QLabel(confusionDock)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 2)
        self.inRaster = gui.QgsMapLayerComboBox(confusionDock)
        self.inRaster.setMinimumSize(QtCore.QSize(200, 0))
        self.inRaster.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inRaster.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.RasterLayer)
        self.inRaster.setObjectName(_fromUtf8("inRaster"))
        self.gridLayout.addWidget(self.inRaster, 0, 2, 1, 3)
        self.inShape = gui.QgsMapLayerComboBox(confusionDock)
        self.inShape.setMinimumSize(QtCore.QSize(100, 0))
        self.inShape.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inShape.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.PolygonLayer)
        self.inShape.setObjectName(_fromUtf8("inShape"))
        self.gridLayout.addWidget(self.inShape, 1, 2, 1, 3)
        self.label_5 = QtGui.QLabel(confusionDock)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 2)
        self.inField = QtGui.QComboBox(confusionDock)
        self.inField.setMinimumSize(QtCore.QSize(100, 0))
        self.inField.setMaximumSize(QtCore.QSize(16777215, 30))
        self.inField.setObjectName(_fromUtf8("inField"))
        self.gridLayout.addWidget(self.inField, 2, 2, 1, 3)
        spacerItem = QtGui.QSpacerItem(258, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 4)
        self.compare = QtGui.QPushButton(confusionDock)
        self.compare.setMinimumSize(QtCore.QSize(150, 0))
        self.compare.setMaximumSize(QtCore.QSize(999, 200))
        self.compare.setObjectName(_fromUtf8("compare"))
        self.gridLayout.addWidget(self.compare, 3, 4, 1, 1)
        self.label = QtGui.QLabel(confusionDock)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.kappa = QtGui.QLabel(confusionDock)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.kappa.setFont(font)
        self.kappa.setText(_fromUtf8(""))
        self.kappa.setObjectName(_fromUtf8("kappa"))
        self.gridLayout.addWidget(self.kappa, 4, 1, 1, 1)
        self.label_2 = QtGui.QLabel(confusionDock)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 4, 2, 1, 1)
        self.OA = QtGui.QLabel(confusionDock)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.OA.setFont(font)
        self.OA.setText(_fromUtf8(""))
        self.OA.setObjectName(_fromUtf8("OA"))
        self.gridLayout.addWidget(self.OA, 4, 3, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(200, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 4, 4, 1, 1)
        self.confusionTable = QtGui.QTableView(confusionDock)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.confusionTable.setFont(font)
        self.confusionTable.setAlternatingRowColors(True)
        self.confusionTable.setSortingEnabled(False)
        self.confusionTable.setObjectName(_fromUtf8("confusionTable"))
        self.gridLayout.addWidget(self.confusionTable, 5, 0, 1, 5)
        self.label_4 = QtGui.QLabel(confusionDock)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 2)

        self.retranslateUi(confusionDock)
        QtCore.QMetaObject.connectSlotsByName(confusionDock)

    def retranslateUi(self, confusionDock):
        confusionDock.setWindowTitle(_translate("confusionDock", "Confusion Matrix (Kappa/OA)", None))
        self.label_3.setText(_translate("confusionDock", "Prediction :", None))
        self.label_5.setText(_translate("confusionDock", "ROI column :", None))
        self.compare.setText(_translate("confusionDock", "Compare", None))
        self.label.setText(_translate("confusionDock", "Kappa :", None))
        self.label_2.setText(_translate("confusionDock", "Overall Accuracy :", None))
        self.label_4.setText(_translate("confusionDock", "ROI :", None))

from qgis import gui
