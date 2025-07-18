# -*- coding: utf-8 -*-

# Form implementation for dzetsaka anniversary popup
# Created for dzetsaka 10th anniversary (May 17, 2026)

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AnniversaryDialog(object):
    def setupUi(self, AnniversaryDialog):
        AnniversaryDialog.setObjectName("AnniversaryDialog")
        AnniversaryDialog.resize(500, 400)
        AnniversaryDialog.setModal(True)
        AnniversaryDialog.setWindowTitle("dzetsaka 10th Anniversary!")
        
        # Create the main layout
        self.verticalLayout = QtWidgets.QVBoxLayout(AnniversaryDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        
        # Title label with emoji
        self.titleLabel = QtWidgets.QLabel(AnniversaryDialog)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setObjectName("titleLabel")
        self.verticalLayout.addWidget(self.titleLabel)
        
        # Add some spacing
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem1)
        
        # Main message
        self.messageLabel = QtWidgets.QLabel(AnniversaryDialog)
        self.messageLabel.setWordWrap(True)
        self.messageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.messageLabel.setObjectName("messageLabel")
        self.verticalLayout.addWidget(self.messageLabel)
        
        # Add spacing
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem2)
        
        # Features request section
        self.featuresLabel = QtWidgets.QLabel(AnniversaryDialog)
        self.featuresLabel.setWordWrap(True)
        self.featuresLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.featuresLabel.setObjectName("featuresLabel")
        self.verticalLayout.addWidget(self.featuresLabel)
        
        # Buttons layout
        self.buttonsLayout = QtWidgets.QHBoxLayout()
        self.buttonsLayout.setObjectName("buttonsLayout")
        
        # GitHub link button
        self.githubButton = QtWidgets.QPushButton(AnniversaryDialog)
        self.githubButton.setObjectName("githubButton")
        self.buttonsLayout.addWidget(self.githubButton)
        
        # Poll/Survey button
        self.pollButton = QtWidgets.QPushButton(AnniversaryDialog)
        self.pollButton.setObjectName("pollButton")
        self.buttonsLayout.addWidget(self.pollButton)
        
        self.verticalLayout.addLayout(self.buttonsLayout)
        
        # Add flexible spacing
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        
        # Checkbox for "don't show again"
        self.dontShowAgainCheckBox = QtWidgets.QCheckBox(AnniversaryDialog)
        self.dontShowAgainCheckBox.setObjectName("dontShowAgainCheckBox")
        self.verticalLayout.addWidget(self.dontShowAgainCheckBox)
        
        # Button box
        self.buttonBox = QtWidgets.QDialogButtonBox(AnniversaryDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(AnniversaryDialog)
        self.buttonBox.accepted.connect(AnniversaryDialog.accept)
        self.buttonBox.rejected.connect(AnniversaryDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(AnniversaryDialog)

    def retranslateUi(self, AnniversaryDialog):
        _translate = QtCore.QCoreApplication.translate
        self.titleLabel.setText(_translate("AnniversaryDialog", "🎯 dzetsaka 10th Anniversary Preparation"))
        self.messageLabel.setText(_translate("AnniversaryDialog", 
            "Help us prepare for dzetsaka's <b>10th anniversary</b> on May 17, 2026!<br><br>"
            "Our community has grown to over <b>200,000 downloads</b>, and we want to celebrate with amazing new features!<br><br>"
            "Your input is crucial for shaping the next generation of dzetsaka."))
        self.featuresLabel.setText(_translate("AnniversaryDialog",
            "🚀 <b>What new features would you like to see in dzetsaka?</b><br>"
            "Help us prioritize development by sharing your feedback:<br>"
            "• Take our <b>quick poll</b> for structured feedback<br>"
            "• Share detailed ideas and suggestions on <b>GitHub</b><br>"
            "We're collecting feedback until the anniversary to build the best possible release!"))
        self.githubButton.setText(_translate("AnniversaryDialog", "💡 GitHub Issues"))
        self.pollButton.setText(_translate("AnniversaryDialog", "📊 Take Poll"))
        self.dontShowAgainCheckBox.setText(_translate("AnniversaryDialog", "Don't show this message again"))