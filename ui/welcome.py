# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'welcome.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DockWidget(object):
    def setupUi(self, DockWidget):
        DockWidget.setObjectName("DockWidget")
        DockWidget.resize(513, 387)
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setEnabled(True)
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.dockWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAcceptDrops(True)
        self.label.setAutoFillBackground(False)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/plugins/dzetsaka/img/parcguyane.jpg"))
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            478, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_2.setAutoFillBackground(True)
        self.label_2.setWordWrap(True)
        self.label_2.setOpenExternalLinks(True)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        DockWidget.setWidget(self.dockWidgetContents)

        self.retranslateUi(DockWidget)
        QtCore.QMetaObject.connectSlotsByName(DockWidget)

    def retranslateUi(self, DockWidget):
        _translate = QtCore.QCoreApplication.translate
        DockWidget.setWindowTitle(_translate("DockWidget", "Welcome to dzetsaka"))
        self.label_2.setText(
            _translate(
                "DockWidget",
                '<html><head/><body><p><span style=" font-weight:600;">Thanks for installing dzetsaka.</span></p><p>It looks like it is your first time using this plugin, so let me suggest you to take a look at the <a href="https://github.com/lennepkade/dzetsaka"><span style=" font-weight:600; text-decoration: underline; color:#046c08;">documentation</span></a> and to <a href="https://github.com/lennepkade/dzetsaka/archive/docs.zip"><span style=" font-weight:600; text-decoration: underline; color:#046c08;">download our demonstration dataset</span></a>.</p><p>If you want to install Random Forest or Support Vector Machine algorithms, take a look at <a href="https://github.com/lennepkade/dzetsaka#installation-of-scikit-learn"><span style=" font-weight:600; text-decoration: underline; color:#046c08;">how to install scikit-learn library in Qgis</span></a>.</p><p>Have fun,</p><p>Nicolas Karasiak (<a href="http://www.karasiak.net"><span style=" font-weight:600; text-decoration: underline; color:#046c08;">www.karasiak.net</span></a>)</p></body></html>',
            )
        )
