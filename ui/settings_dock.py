# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_dock.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
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

class Ui_settingsDock(object):
    def setupUi(self, settingsDock):
        settingsDock.setObjectName(_fromUtf8("settingsDock"))
        settingsDock.resize(304, 341)
        self.gridLayout = QtGui.QGridLayout(settingsDock)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(10, 100, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.label_8 = QtGui.QLabel(settingsDock)
        self.label_8.setMinimumSize(QtCore.QSize(250, 0))
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 100))
        self.label_8.setText(_fromUtf8(""))
        self.label_8.setPixmap(QtGui.QPixmap(_fromUtf8(":/plugins/dzetsaka/img/parcguyane.jpg")))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout.addWidget(self.label_8, 0, 1, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(10, 100, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        self.scrollArea = QtGui.QScrollArea(settingsDock)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 284, 189))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.formLayout = QtGui.QFormLayout(self.scrollAreaWidgetContents)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.settings = QtGui.QLabel(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.settings.setFont(font)
        self.settings.setAlignment(QtCore.Qt.AlignCenter)
        self.settings.setObjectName(_fromUtf8("settings"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.settings)
        self.label = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label)
        self.selectClassifier = QtGui.QComboBox(self.scrollAreaWidgetContents)
        self.selectClassifier.setObjectName(_fromUtf8("selectClassifier"))
        self.selectClassifier.addItem(_fromUtf8(""))
        self.selectClassifier.addItem(_fromUtf8(""))
        self.selectClassifier.addItem(_fromUtf8(""))
        self.selectClassifier.addItem(_fromUtf8(""))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.selectClassifier)
        self.label_2 = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_2)
        self.classSuffix = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.classSuffix.setObjectName(_fromUtf8("classSuffix"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.classSuffix)
        self.label_3 = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_3)
        self.classPrefix = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.classPrefix.setText(_fromUtf8(""))
        self.classPrefix.setObjectName(_fromUtf8("classPrefix"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.classPrefix)
        self.label_4 = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_4)
        self.maskSuffix = QtGui.QLineEdit(self.scrollAreaWidgetContents)
        self.maskSuffix.setText(_fromUtf8(""))
        self.maskSuffix.setObjectName(_fromUtf8("maskSuffix"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.maskSuffix)
        spacerItem2 = QtGui.QSpacerItem(10, 2, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.formLayout.setItem(5, QtGui.QFormLayout.FieldRole, spacerItem2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 1, 0, 1, 3)
        spacerItem3 = QtGui.QSpacerItem(20, 17, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 2, 1, 1, 1)

        self.retranslateUi(settingsDock)
        QtCore.QMetaObject.connectSlotsByName(settingsDock)

    def retranslateUi(self, settingsDock):
        settingsDock.setWindowTitle(_translate("settingsDock", "Help for dzetsaka", None))
        self.settings.setText(_translate("settingsDock", "Settings", None))
        self.label.setText(_translate("settingsDock", "Classifier :", None))
        self.selectClassifier.setItemText(0, _translate("settingsDock", "Gaussian Mixture Model", None))
        self.selectClassifier.setItemText(1, _translate("settingsDock", "Random Forest", None))
        self.selectClassifier.setItemText(2, _translate("settingsDock", "Support Vector Machines", None))
        self.selectClassifier.setItemText(3, _translate("settingsDock", "K-Nearest Neighbors", None))
        self.label_2.setText(_translate("settingsDock", "Temp suffix :", None))
        self.label_3.setText(_translate("settingsDock", "Temp prefix :", None))
        self.label_4.setText(_translate("settingsDock", "Mask suffix :", None))

