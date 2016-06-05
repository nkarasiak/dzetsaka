# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'filters_dock.ui'
#
# Created: Sat Jun  4 11:38:26 2016
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

class Ui_filterMenu(object):
    def setupUi(self, filterMenu):
        filterMenu.setObjectName(_fromUtf8("filterMenu"))
        filterMenu.setEnabled(True)
        filterMenu.resize(486, 160)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(filterMenu.sizePolicy().hasHeightForWidth())
        filterMenu.setSizePolicy(sizePolicy)
        filterMenu.setMinimumSize(QtCore.QSize(480, 160))
        filterMenu.setSizeGripEnabled(False)
        self.gridLayout = QtGui.QGridLayout(filterMenu)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.tiffImageLabel = QtGui.QLabel(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tiffImageLabel.setFont(font)
        self.tiffImageLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tiffImageLabel.setObjectName(_fromUtf8("tiffImageLabel"))
        self.gridLayout.addWidget(self.tiffImageLabel, 0, 0, 1, 1)
        self.inRaster = gui.QgsMapLayerComboBox(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.inRaster.setFont(font)
        self.inRaster.setFilters(gui.QgsMapLayerProxyModel.PluginLayer|gui.QgsMapLayerProxyModel.RasterLayer)
        self.inRaster.setObjectName(_fromUtf8("inRaster"))
        self.gridLayout.addWidget(self.inRaster, 0, 1, 1, 4)
        spacerItem = QtGui.QSpacerItem(13, 40, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 5, 2, 1)
        self.tiffImageLabel_3 = QtGui.QLabel(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tiffImageLabel_3.setFont(font)
        self.tiffImageLabel_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tiffImageLabel_3.setObjectName(_fromUtf8("tiffImageLabel_3"))
        self.gridLayout.addWidget(self.tiffImageLabel_3, 1, 0, 1, 1)
        self.inFilter = QtGui.QComboBox(filterMenu)
        self.inFilter.setObjectName(_fromUtf8("inFilter"))
        self.gridLayout.addWidget(self.inFilter, 1, 1, 1, 4)
        self.tiffImageLabel_2 = QtGui.QLabel(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tiffImageLabel_2.setFont(font)
        self.tiffImageLabel_2.setToolTip(_fromUtf8(""))
        self.tiffImageLabel_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tiffImageLabel_2.setObjectName(_fromUtf8("tiffImageLabel_2"))
        self.gridLayout.addWidget(self.tiffImageLabel_2, 2, 0, 1, 1)
        self.inFilterSize = gui.QgsSpinBox(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.inFilterSize.setFont(font)
        self.inFilterSize.setMinimum(1)
        self.inFilterSize.setMaximum(999)
        self.inFilterSize.setSingleStep(2)
        self.inFilterSize.setProperty("value", 11)
        self.inFilterSize.setObjectName(_fromUtf8("inFilterSize"))
        self.gridLayout.addWidget(self.inFilterSize, 2, 1, 1, 1)
        self.tiffImageLabel_15 = QtGui.QLabel(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tiffImageLabel_15.setFont(font)
        self.tiffImageLabel_15.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tiffImageLabel_15.setObjectName(_fromUtf8("tiffImageLabel_15"))
        self.gridLayout.addWidget(self.tiffImageLabel_15, 2, 2, 1, 1)
        self.inFilterIter = gui.QgsSpinBox(filterMenu)
        self.inFilterIter.setMinimumSize(QtCore.QSize(80, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.inFilterIter.setFont(font)
        self.inFilterIter.setMinimum(1)
        self.inFilterIter.setMaximum(99)
        self.inFilterIter.setSingleStep(1)
        self.inFilterIter.setProperty("value", 1)
        self.inFilterIter.setObjectName(_fromUtf8("inFilterIter"))
        self.gridLayout.addWidget(self.inFilterIter, 2, 3, 1, 2)
        self.tiffImageLabel_16 = QtGui.QLabel(filterMenu)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tiffImageLabel_16.setFont(font)
        self.tiffImageLabel_16.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.tiffImageLabel_16.setObjectName(_fromUtf8("tiffImageLabel_16"))
        self.gridLayout.addWidget(self.tiffImageLabel_16, 3, 0, 1, 1)
        self.outRaster = QtGui.QLineEdit(filterMenu)
        self.outRaster.setText(_fromUtf8(""))
        self.outRaster.setObjectName(_fromUtf8("outRaster"))
        self.gridLayout.addWidget(self.outRaster, 3, 1, 1, 3)
        self.outRasterButton = QtGui.QPushButton(filterMenu)
        self.outRasterButton.setMinimumSize(QtCore.QSize(15, 0))
        self.outRasterButton.setMaximumSize(QtCore.QSize(25, 16777215))
        self.outRasterButton.setAutoDefault(False)
        self.outRasterButton.setObjectName(_fromUtf8("outRasterButton"))
        self.gridLayout.addWidget(self.outRasterButton, 3, 4, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(128, 19, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 4, 0, 1, 1)
        self.runFilter = QtGui.QPushButton(filterMenu)
        self.runFilter.setMinimumSize(QtCore.QSize(150, 0))
        self.runFilter.setObjectName(_fromUtf8("runFilter"))
        self.gridLayout.addWidget(self.runFilter, 4, 1, 1, 1)
        spacerItem2 = QtGui.QSpacerItem(166, 19, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 4, 2, 1, 3)
        spacerItem3 = QtGui.QSpacerItem(20, 1, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 5, 1, 1, 1)

        self.retranslateUi(filterMenu)
        QtCore.QMetaObject.connectSlotsByName(filterMenu)

    def retranslateUi(self, filterMenu):
        filterMenu.setWindowTitle(_translate("filterMenu", "Filters | dzetska", None))
        self.tiffImageLabel.setText(_translate("filterMenu", "Image to filter :", None))
        self.tiffImageLabel_3.setText(_translate("filterMenu", "filter :", None))
        self.tiffImageLabel_2.setText(_translate("filterMenu", "filter size :", None))
        self.tiffImageLabel_15.setText(_translate("filterMenu", "iteration :", None))
        self.tiffImageLabel_16.setText(_translate("filterMenu", "Save as :", None))
        self.outRaster.setPlaceholderText(_translate("filterMenu", "Leave empty for temporary file...", None))
        self.outRasterButton.setText(_translate("filterMenu", "...", None))
        self.runFilter.setText(_translate("filterMenu", "Filter image", None))

from qgis import gui
