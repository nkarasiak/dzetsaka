# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_dock.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_settingsDock(object):
    def setupUi(self, settingsDock):
        settingsDock.setObjectName("settingsDock")
        settingsDock.resize(317, 202)
        settingsDock.setMinimumSize(QtCore.QSize(317, 202))
        settingsDock.setMaximumSize(QtCore.QSize(600, 600))
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.formLayout = QtWidgets.QFormLayout(self.dockWidgetContents)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.dockWidgetContents)
        self.label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.formLayout.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.selectClassifier = QtWidgets.QComboBox(self.dockWidgetContents)
        self.selectClassifier.setObjectName("selectClassifier")
        self.selectClassifier.addItem("")
        self.selectClassifier.addItem("")
        self.selectClassifier.addItem("")
        self.selectClassifier.addItem("")
        self.formLayout.setWidget(
            0,
            QtWidgets.QFormLayout.FieldRole,
            self.selectClassifier)
        self.label_2 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_2.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.classSuffix = QtWidgets.QLineEdit(self.dockWidgetContents)
        self.classSuffix.setObjectName("classSuffix")
        self.formLayout.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.classSuffix)
        self.label_3 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_3.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(
            2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.classPrefix = QtWidgets.QLineEdit(self.dockWidgetContents)
        self.classPrefix.setText("")
        self.classPrefix.setObjectName("classPrefix")
        self.formLayout.setWidget(
            2, QtWidgets.QFormLayout.FieldRole, self.classPrefix)
        self.label_4 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_4.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(
            3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.maskSuffix = QtWidgets.QLineEdit(self.dockWidgetContents)
        self.maskSuffix.setText("")
        self.maskSuffix.setObjectName("maskSuffix")
        self.formLayout.setWidget(
            3, QtWidgets.QFormLayout.FieldRole, self.maskSuffix)
        self.label_5 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_5.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(
            4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.selectProviders = QtWidgets.QComboBox(self.dockWidgetContents)
        self.selectProviders.setObjectName("selectProviders")
        self.selectProviders.addItem("")
        self.selectProviders.addItem("")
        self.formLayout.setWidget(
            4,
            QtWidgets.QFormLayout.FieldRole,
            self.selectProviders)
        settingsDock.setWidget(self.dockWidgetContents)

        self.retranslateUi(settingsDock)
        QtCore.QMetaObject.connectSlotsByName(settingsDock)

    def retranslateUi(self, settingsDock):
        _translate = QtCore.QCoreApplication.translate
        settingsDock.setWindowTitle(
            _translate(
                "settingsDock",
                "dzetsaka : settings panel"))
        self.label.setText(_translate("settingsDock", "Classifier :"))
        self.selectClassifier.setItemText(0, _translate(
            "settingsDock", "Gaussian Mixture Model"))
        self.selectClassifier.setItemText(
            1, _translate("settingsDock", "Random Forest"))
        self.selectClassifier.setItemText(2, _translate(
            "settingsDock", "Support Vector Machines"))
        self.selectClassifier.setItemText(
            3, _translate("settingsDock", "K-Nearest Neighbors"))
        self.label_2.setText(_translate("settingsDock", "Temp suffix :"))
        self.label_3.setText(_translate("settingsDock", "Temp prefix :"))
        self.label_4.setText(_translate("settingsDock", "Mask suffix :"))
        self.label_5.setText(_translate("settingsDock", "Providers : "))
        self.selectProviders.setItemText(
            0, _translate("settingsDock", "Standard"))
        self.selectProviders.setItemText(
            1, _translate("settingsDock", "Experimental"))
