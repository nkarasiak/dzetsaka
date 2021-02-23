# -*- coding: utf-8 -*-
"""
/***************************************************************************
 dzetsakaGUI
                                 A QGIS plugin
 desc
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-02-24
        git sha              : $Format:%H$
        copyright            : (C) 2018 by Nicolas Karasiak
        email                : karasiak.nicolas@gmail.com
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
# import basics
from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox, QDialog, QFileDialog, QApplication
from qgis.core import QgsMessageLog, QgsProcessingAlgorithm, QgsApplication


# import outside libraries
#import configparser
import tempfile
import os.path
try:
    from osgeo import gdal,ogr,osr
except ImportError:
    import gdal,ogr,osr

# import local libraries
from . import resources
from . import ui
from .scripts import function_dataraster as dataraster
from .scripts import mainfunction

from .dzetsaka_provider import dzetsakaProvider


class dzetsakaGUI (QDialog):
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface

        # add Processing loadAlgorithms

        # init dialog and dzetsaka dock
        QDialog.__init__(self)
        #sender = self.sender()
        self.settings = QSettings()
        self.loadConfig()

        self.provider = dzetsakaProvider(self.providerType)
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        if self.firstInstallation is True:
            self.showWelcomeWidget()

        # initialize locale
        """
        locale = self.settings.value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'dzetsaka_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)
        """
        
        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&dzetsaka')
#        # TODO: We are going to let the user set this up in a future iteration
#        self.toolbar = self.iface.addToolBar(u'dzetsaka')
#        self.toolbar.setObjectName(u'dzetsaka')
        self.pluginIsActive = False
        self.dockwidget = None
#        



        # param
        self.lastSaveDir = ''

        # run dock
        # self.run()

    def rememberLastSaveDir(self, fileName):
        """!@brief Remember last sd dir when saving or loading file"""
        if fileName != '':
            self.lastSaveDir = fileName
            self.settings.setValue('/dzetsaka/lastSaveDir', self.lastSaveDir)

    # noinspection PyMethodMayBeStatic
    def showWelcomeWidget(self):
        self.welcomeWidget = ui.welcomeWidget()
        self.welcomeWidget.show()

        self.settings.setValue('/dzetsaka/firstInstallation', False)

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('dzetsaka', message)

    def add_action(
            self,
            icon_path,
            text,
            callback,
            enabled_flag=True,
            add_to_menu=True,
            add_to_toolbar=True,
            status_tip=None,
            whats_this=None,
            parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

#        if add_to_toolbar:
#            self.toolbar.addAction(action)
#
        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        QgsApplication.processingRegistry().addProvider(self.provider)

        icon_path = ':/plugins/dzetsaka/img/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'welcome message'),
            callback=self.showWelcomeWidget,
            add_to_toolbar=False,
            parent=self.iface.mainWindow())

        icon_path = ':/plugins/dzetsaka/img/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'classification dock'),
            callback=self.run,
            parent=self.iface.mainWindow())

        icon_settings_path = ':/plugins/dzetsaka/img/dzetsaka_settings.png'
        self.add_action(
            icon_settings_path,
            text=self.tr(u'settings'),
            callback=self.loadSettings,
            add_to_toolbar=True,
            parent=self.iface.mainWindow())
        
        self.dockIcon = QAction(QIcon(':/plugins/dzetsaka/img/icon.png'), 'dzetsaka classification dock', self.iface.mainWindow())
        self.dockIcon.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.dockIcon)
        self.actions.append(self.dockIcon)
        
        self.settingsIcon = QAction(QIcon(':/plugins/dzetsaka/img/dzetsaka_settings.png'), 'dzetsaka settings', self.iface.mainWindow())
        self.settingsIcon.triggered.connect(self.loadSettings)
        self.iface.addToolBarIcon(self.settingsIcon)
        self.actions.append(self.settingsIcon)
        
    # --------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        #print "** CLOSING dzetsakaGUI"
        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # remove this statement if dockwidget is to remain
        # for reuse if plugin is reopened
        # Commented next statement since it causes QGIS crashe
        # when closing the docked window:
        # self.dockwidget = None

        self.pluginIsActive = False

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        # close dock
        self.pluginIsActive = False
        if self.dockwidget is not None:
            self.dockwidget.close()

        # Remove processing algorithms
        QgsApplication.processingRegistry().removeProvider(self.provider)
        
        
#        self.iface.removePluginMenu(self.pluginName, self.settingsIcon)
#        self.iface.removePluginMenu(self.tr(u'&dzetsaka'),self.actions[0])
        
        for action in self.actions:
            self.iface.removeToolBarIcon(action)
            self.iface.removePluginMenu(
                self.tr(u'&dzetsaka'),
                action)
#
#        # remove the toolbar
#       qg del self.toolbar
    # --------------------------------------------------------------------------

    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive or self.dockwidget is None:
            self.pluginIsActive = True

            #print "** STARTING dzetsakaGUI"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget is None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = ui.dzetsakaDockWidget()

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)

            from qgis.core import QgsProviderRegistry
            exceptRaster = QgsProviderRegistry.instance().providerList()
            exceptRaster.remove('gdal')
            self.dockwidget.inRaster.setExcludedProviders(exceptRaster)

            exceptVector = QgsProviderRegistry.instance().providerList()
            exceptVector.remove('ogr')
            self.dockwidget.inShape.setExcludedProviders(exceptVector)

            self.dockwidget.outRaster.clear()
            self.dockwidget.outRasterButton.clicked.connect(
                self.select_output_file)

            self.dockwidget.outModel.clear()
            self.dockwidget.checkOutModel.clicked.connect(self.checkbox_state)

            self.dockwidget.inModel.clear()
            self.dockwidget.checkInModel.clicked.connect(self.checkbox_state)

            self.dockwidget.inMask.clear()
            self.dockwidget.checkInMask.clicked.connect(self.checkbox_state)

            self.dockwidget.outMatrix.clear()
            self.dockwidget.checkOutMatrix.clicked.connect(self.checkbox_state)

            self.dockwidget.outConfidenceMap.clear()
            self.dockwidget.checkInConfidence.clicked.connect(
                self.checkbox_state)

            self.dockwidget.inField.clear()

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
            self.dockwidget.show()

            def onChangedLayer():
                """!@brief Update columns if vector changes"""
                # We clear combobox
                self.dockwidget.inField.clear()
                # Then we fill it with new selected Layer
                if self.dockwidget.inField.currentText() == '' and self.dockwidget.inShape.currentLayer(
                ) and self.dockwidget.inShape.currentLayer() != 'NoneType':
                    try:
                        activeLayer = self.dockwidget.inShape.currentLayer()
                        provider = activeLayer.dataProvider()
                        fields = provider.fields()
                        listFieldNames = [field.name() for field in fields]
                        self.dockwidget.inField.addItems(listFieldNames)
                    except BaseException:
                        QgsMessageLog.logMessage(
                            'dzetsaka cannot change active layer. Maybe you opened an OSM/Online background ?')

            onChangedLayer()
            self.dockwidget.inShape.currentIndexChanged[int].connect(
                onChangedLayer)

            self.dockwidget.settingsButton.clicked.connect(self.loadSettings)

            # let's run the classification !
            self.dockwidget.performMagic.clicked.connect(self.runMagic)

            # self.dockwidget.mGroupBox.toggled.connect(self.resizeDock)
            self.dockwidget.mGroupBox.collapsedStateChanged.connect(
                self.resizeDock)

    def resizeDock(self):

        if self.dockwidget.mGroupBox.isCollapsed():

            self.dockwidget.mGroupBox.setFixedHeight(20)
            self.dockwidget.setFixedHeight(350)

        else:
            self.dockwidget.setMinimumHeight(470)
            self.dockwidget.mGroupBox.setMinimumHeight(160)

    def select_output_file(self):
        """!@brief Select file to save, and gives the right extension if the user don't put it"""
        sender = self.sender()

        fileName, _filter = QFileDialog.getSaveFileName(
            self.dockwidget, "Select output file", self.lastSaveDir, "TIF (*.tif)")
        self.rememberLastSaveDir(fileName)

        if not fileName:
            return
            # If user give right file extension, we don't add it

        fileName, fileExtension = os.path.splitext(fileName)

        if sender == self.dockwidget.outRasterButton:
            if fileExtension != '.tif':
                self.dockwidget.outRaster.setText(fileName + '.tif')
            else:
                self.dockwidget.outRaster.setText(fileName + fileExtension)

        # check if historical map run
        if 'self.historicalmap' in locals():
            if sender == self.historicalmap.outRasterButton:
                if fileExtension != '.tif':
                    self.historicalmap.outRaster.setText(fileName + '.tif')
                else:
                    self.historicalmap.outRaster.setText(
                        fileName + fileExtension)
            if sender == self.historicalmap.outShpButton:
                if fileExtension != '.shp':
                    self.historicalmap.outShp.setText(fileName + '.shp')
                else:
                    self.historicalmap.outShp.setText(fileName + fileExtension)
        # check if filters_dock run
        if 'self.filters_dock' in locals():
            if sender == self.filters_dock.outRasterButton:
                if fileExtension != '.tif':
                    self.filters_dock.outRaster.setText(fileName + '.tif')
            else:
                self.filters_dock.outRaster.setText(fileName + fileExtension)

    def loadConfig(self):
        """!@brief Class that loads all saved settings from config.txt"""

        try:
            """
            dzetsakaRoot = os.path.dirname(os.path.realpath(__file__))
            self.Config = configparser.ConfigParser()
            self.configFile = os.path.join(dzetsakaRoot,'config.txt')
            self.Config.read(self.configFile)


            self.classifier = self.Config.get('Classification','classifier')


            self.classSuffix = self.Config.get('Classification','suffix')
            self.classPrefix = self.Config.get('Classification','prefix')

            self.maskSuffix = self.Config.get('Classification','maskSuffix')

            self.providerType = self.Config.get('Providers','provider')
            """
            self.classifiers = [
                'Gaussian Mixture Model',
                'Random Forest',
                'Support Vector Machines',
                'K-Nearest Neighbors']
            self.providers = ['Standard', 'Experimental']

            self.classifier = self.settings.value(
                '/dzetsaka/classifier', '', str)
            if not self.classifier:
                self.classifier = self.classifiers[0]
                self.settings.setValue('/dzetsaka/classifier', self.classifier)

            self.classSuffix = self.settings.value(
                '/dzetsaka/classSuffix', '', str)
            if not self.classSuffix:
                self.classSuffix = '_class'
                self.settings.setValue(
                    '/dzetsaka/classSuffix', self.classSuffix)

            self.classPrefix = self.settings.value(
                '/dzetsaka/classPrefix', '', str)
            if not self.classPrefix:
                self.classPrefix = ''
                self.settings.setValue(
                    '/dzetsaka/classPrefix', self.classPrefix)

            self.maskSuffix = self.settings.value(
                '/dzetsaka/maskSuffix', '', str)
            if not self.maskSuffix:
                self.maskSuffix = '_mask'
                self.settings.setValue('/dzetsaka/maskSuffix', self.maskSuffix)

            self.providerType = self.settings.value(
                '/dzetsaka/providerType', '', str)
            if not self.providerType:
                self.providerType = self.providers[0]
                self.providerType = self.settings.setValue(
                    '/dzetsaka/providerType', self.providerType)

            self.firstInstallation = self.settings.value(
                '/dzetsaka/firstInstallation', 'None', bool)
            if self.firstInstallation is None:
                self.firstInstallation = True
                self.settings.setValue('/dzetsaka/firstInstallation', True)

        except BaseException:
            QgsMessageLog.logMessage(
                'failed to open config file ' + self.configFile)

    def loadSettings(self):
        """!@brief load settings dock"""
        self.settingsdock = ui.settings_dock()
        self.settingsdock.show()

        try:
            # Reload config
            self.loadConfig()
            # Classification settings

            # classifier

            for i, cls in enumerate(self.classifiers):
                if self.classifier == cls:
                    self.settingsdock.selectClassifier.setCurrentIndex(i)

            self.settingsdock.selectClassifier.currentIndexChanged[int].connect(
                self.saveSettings)

            self.settings.setValue('/dzetsaka/classifier', self.classifier)

            # suffix
            self.settingsdock.classSuffix.setText(self.classSuffix)
            self.settingsdock.classSuffix.textChanged.connect(
                self.saveSettings)
            self.settings.setValue('/dzetsaka/classSuffix', self.classSuffix)
            # prefix
            self.settingsdock.classPrefix.setText(self.classPrefix)
            self.settingsdock.classPrefix.textChanged.connect(
                self.saveSettings)
            self.settings.setValue('/dzetsaka/classPrefix', self.classPrefix)
            # mask suffix
            self.settingsdock.maskSuffix.setText(self.maskSuffix)
            self.settingsdock.maskSuffix.textChanged.connect(self.saveSettings)
            self.settings.setValue('/dzetsaka/maskSuffix', self.maskSuffix)
            ##

            for i, prvd in enumerate(self.providers):
                if self.providerType == prvd:
                    self.settingsdock.selectProviders.setCurrentIndex(i)

            self.settingsdock.selectProviders.currentIndexChanged[int].connect(
                self.saveSettings)
            self.settings.setValue('/dzetsaka/providerType', self.providerType)
            # Reload config for further use
            self.loadConfig()

        except BaseException:
            QgsMessageLog.logMessage('Failed to load settings...')

    def runMagic(self):
        """!@brief Perform training and classification for dzetsaka"""

        """
        VERIFICATION STEP
        """
        # verif before doing the job
        message = ' '

        if self.dockwidget.inModel.text() == '':
            try:
                self.dockwidget.inShape.currentLayer().dataProvider().dataSourceUri()
            except BaseException:
                message = "\n - If you don't use a model, please specify a vector"
        try:
            self.dockwidget.inRaster.currentLayer().dataProvider().dataSourceUri()
        except BaseException:
            message = message + \
                str("\n - You need a raster to make a classification.")

        try:

            # get raster
            inRaster = self.dockwidget.inRaster.currentLayer()
            inRaster = inRaster.dataProvider().dataSourceUri()

            # get raster proj
            inRasterOp = gdal.Open(inRaster)
            inRasterProj = inRasterOp.GetProjection()
            inRasterProj = osr.SpatialReference(inRasterProj)

            if self.dockwidget.inModel.text() == '':
                # verif srs
                # get vector
                inShape = self.dockwidget.inShape.currentLayer()
                inShape = inShape.dataProvider().dataSourceUri().split('|')[
                    0]  # Remove layerid=0 from SHP Path
                # get shp proj
                inShapeOp = ogr.Open(inShape)
                inShapeLyr = inShapeOp.GetLayer()
                inShapeProj = inShapeLyr.GetSpatialRef()

                # chekc IsSame Projection
                if inShapeProj.IsSameGeogCS(inRasterProj) == 0:
                    message = message + \
                        str("\n - Raster and ROI do not have the same projection.")
        except BaseException:
            QgsMessageLog.logMessage('inShape is : ' + inShape)
            QgsMessageLog.logMessage('inRaster is : ' + inRaster)
            QgsMessageLog.logMessage(
                'inShapeProj.IsSameGeogCS(inRasterProj) : ' +
                inShapeProj.IsSameGeogCS(inRasterProj))
            message = message + \
                str('\n - Can\'t compare projection between raster and vector.')

        try:
            inMask = self.dockwidget.inMask.text()

            if inMask == '':
                inMask = None
            # check if mask with _mask.extension
            autoMask = os.path.splitext(inRaster)
            autoMask = autoMask[0] + self.maskSuffix + autoMask[1]

            if os.path.exists(autoMask):
                inMask = autoMask
                QgsMessageLog.logMessage('Mask found : ' + str(autoMask))

            if inMask is not None:
                mask = gdal.Open(inMask, gdal.GA_ReadOnly)
            # Check size
                if (inRasterOp.RasterXSize != mask.RasterXSize) or (
                        inRasterOp.RasterYSize != mask.RasterYSize):
                    message = message + \
                        str('\n - Raster image and mask do not have the same size.')

        except BaseException:
            message = message + \
                str('\n - Can\'t compare mask and raster size.')
        """ END OF VERIFICATION STEP """

        if message != ' ':

            reply = QMessageBox.question(self.iface.mainWindow(), 'Informations missing or invalid',
                                         message + '\n Would you like to continue anyway ?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                message = ' '

        # all is ok, so do the job !
        if message == ' ':
            # get config
            self.loadConfig()
            # Get model if given
            model = self.dockwidget.inModel.text()

# ==============================================================================
#             # if model not given, perform training
#             inRaster=self.dockwidget.inRaster.currentLayer()
#             inRaster=inRaster.dataProvider().dataSourceUri()
# ==============================================================================

            # create temp if not output raster
            if self.dockwidget.outRaster.text() == '':
                tempFolder = tempfile.mkdtemp()
                outRaster = os.path.join(
                    tempFolder,
                    self.classPrefix +
                    os.path.splitext(
                        os.path.basename(inRaster))[0] +
                    self.classSuffix +
                    '.tif')

            else:
                outRaster = self.dockwidget.outRaster.text()

            # Confidence map

            if self.dockwidget.checkInConfidence.isChecked():
                confidenceMap = self.dockwidget.outConfidenceMap.text()
            else:
                confidenceMap = None

            # Get Classifier
            # retrieve shortname classifier
            classifierShortName = ['GMM', 'RF', 'SVM', 'KNN']
            for i, cls in enumerate(self.classifiers):
                if self.classifier == cls:
                    inClassifier = classifierShortName[i]

            # Check if model, else perform training
            NODATA = -9999

            if model != '':
                model = self.dockwidget.inModel.text()
                QgsMessageLog.logMessage('Model is ' + str(model))
            else:
                if self.dockwidget.outModel.text() == '':
                    model = tempfile.mktemp('.' + str(inClassifier))
                else:
                    model = self.dockwidget.outModel.text()
                QgsMessageLog.logMessage('No model loaded')

            inField = self.dockwidget.inField.currentText()
            inSeed = 0
            # Perform training & classification
            if self.dockwidget.checkOutMatrix.isChecked():
                outMatrix = self.dockwidget.outMatrix.text()
                inSplit = self.dockwidget.inSplit.value()
            else:
                inSplit = 100
                outMatrix = None

            try:

                if not self.dockwidget.checkInModel.isChecked():
                    QgsMessageLog.logMessage(
                        'Begin training with ' + inClassifier + ' classifier')

                    temp = mainfunction.learnModel(
                        inRaster,
                        inShape,
                        inField,
                        model,
                        inSplit,
                        inSeed=inSeed,
                        outMatrix=outMatrix,
                        inClassifier=inClassifier,
                        extraParam=None,
                        feedback='gui')

                    # perform learning

                stop = False
            except BaseException:
                QgsMessageLog.logMessage('Raster is' + str(inRaster))
                QgsMessageLog.logMessage('Vector is' + str(inShape))
                QgsMessageLog.logMessage('Column field is ' + str(inField))
                QgsMessageLog.logMessage('Split is' + str(inSplit))
                QgsMessageLog.logMessage('Model is ' + str(model))

                message = (
                    'Something went wrong during the training. Please make sure you respect these conditions : <br> - Are you sure to have only integer values in your ' +
                    str(inField) +
                    ' column ? <br> - Do your shapefile and raster have the same projection ?')
                QMessageBox.warning(
                    self,
                    'dzetsaka has encountered a problem',
                    message,
                    QMessageBox.Ok)
                stop = True
                QApplication.restoreOverrideCursor()
                

            if not stop:
                # Begin classification
                QgsMessageLog.logMessage(
                    'Begin classification with ' + str(inClassifier))
                temp = mainfunction.classifyImage()

                temp.initPredict(
                    inRaster,
                    model,
                    outRaster,
                    inMask=inMask,
                    confidenceMap=confidenceMap,
                    confidenceMapPerClass=None,
                    NODATA=NODATA,
                    feedback='gui')
                QgsMessageLog.logMessage('Dzetsaka classification done.')
                self.iface.addRasterLayer(outRaster)

                if confidenceMap:
                    self.iface.addRasterLayer(confidenceMap)

    def checkbox_state(self):
        """!@brief Manage checkbox in main dock"""

        sender = self.sender()

        # If load model
        if sender == self.dockwidget.checkInModel and self.dockwidget.checkInModel.isChecked():
            fileName, _filter = QFileDialog.getOpenFileName(
                self.dockwidget, "Select your file", self.lastSaveDir)
            self.rememberLastSaveDir(fileName)
            if fileName != '':
                self.dockwidget.inModel.setText(fileName)
                self.dockwidget.inModel.setEnabled(True)
                # Disable training, so disable vector choise
                self.dockwidget.inShape.setEnabled(False)
                self.dockwidget.inField.setEnabled(False)

            else:
                self.dockwidget.checkInModel.setChecked(False)
                self.dockwidget.inModel.setEnabled(False)
                self.dockwidget.inShape.setEnabled(True)
                self.dockwidget.inField.setEnabled(True)

        elif sender == self.dockwidget.checkInModel:
            self.dockwidget.inModel.clear()
            self.dockwidget.inModel.setEnabled(False)
            self.dockwidget.inShape.setEnabled(True)
            self.dockwidget.inField.setEnabled(True)

        # If save model
        if sender == self.dockwidget.checkOutModel and self.dockwidget.checkOutModel.isChecked():
            fileName, _filter = QFileDialog.getSaveFileName(
                self.dockwidget, "Select output file", self.lastSaveDir)
            self.rememberLastSaveDir(fileName)
            if fileName != '':
                self.dockwidget.outModel.setText(fileName)
                self.dockwidget.outModel.setEnabled(True)

            else:
                self.dockwidget.checkOutModel.setChecked(False)
                self.dockwidget.outModel.setEnabled(False)

        elif sender == self.dockwidget.checkOutModel:
            self.dockwidget.outModel.clear()
            self.dockwidget.outModel.setEnabled(False)

        # If mask
        if sender == self.dockwidget.checkInMask and self.dockwidget.checkInMask.isChecked():
            fileName, _filter = QFileDialog.getOpenFileName(
                self.dockwidget, "Select your mask raster", self.lastSaveDir, "TIF (*.tif)")
            self.rememberLastSaveDir(fileName)
            if fileName != '':

                self.dockwidget.inMask.setText(fileName)
                self.dockwidget.inMask.setEnabled(True)
            else:
                self.dockwidget.checkInMask.setChecked(False)
                self.dockwidget.inMask.setEnabled(False)
        elif sender == self.dockwidget.checkInMask:
            self.dockwidget.inMask.clear()
            self.dockwidget.inMask.setEnabled(False)

        # If save matrix
        if sender == self.dockwidget.checkOutMatrix and self.dockwidget.checkOutMatrix.isChecked():
            fileName, _filter = QFileDialog.getSaveFileName(
                self.dockwidget, "Save to a *.csv file", self.lastSaveDir, "CSV (*.csv)")
            self.rememberLastSaveDir(fileName)
            if fileName != '':
                fileName, fileExtension = os.path.splitext(fileName)
                fileName = fileName + '.csv'
                self.dockwidget.outMatrix.setText(fileName)
                self.dockwidget.outMatrix.setEnabled(True)
                self.dockwidget.inSplit.setEnabled(True)
                self.dockwidget.inSplit.setValue(50)
            else:
                self.dockwidget.checkOutMatrix.setChecked(False)
                self.dockwidget.outMatrix.setEnabled(False)
                self.dockwidget.outMatrix.setEnabled(False)
                self.dockwidget.inSplit.setEnabled(False)
                self.dockwidget.inSplit.setValue(100)

        elif sender == self.dockwidget.checkOutMatrix:
            self.dockwidget.outMatrix.clear()
            self.dockwidget.checkOutMatrix.setChecked(False)
            self.dockwidget.outMatrix.setEnabled(False)
            self.dockwidget.outMatrix.setEnabled(False)
            self.dockwidget.inSplit.setEnabled(False)
            self.dockwidget.inSplit.setValue(100)

      # If save model
        # retrieve shortname classifier
        if sender == self.dockwidget.checkInConfidence and self.dockwidget.checkInConfidence.isChecked():
            fileName, _filter = QFileDialog.getSaveFileName(
                self.dockwidget, "Select output file (*.tif)", self.lastSaveDir, "TIF (*.tif)")
            self.rememberLastSaveDir(fileName)
            if fileName != '':
                fileName, fileExtension = os.path.splitext(fileName)
                fileName = fileName + '.tif'
                self.dockwidget.outConfidenceMap.setText(fileName)
                self.dockwidget.outConfidenceMap.setEnabled(True)

            else:
                self.dockwidget.checkInConfidence.setChecked(False)
                self.dockwidget.outConfidenceMap.setEnabled(False)

        elif sender == self.dockwidget.checkInConfidence:
            self.dockwidget.outConfidenceMap.clear()
            self.dockwidget.checkInConfidence.setChecked(False)
            self.dockwidget.outConfidenceMap.setEnabled(False)

    def saveSettings(self):
        """!@brief save settings if modifications"""
        # Change classifier
        if self.sender() == self.settingsdock.selectClassifier:
            if self.settingsdock.selectClassifier.currentText() != 'Gaussian Mixture Model':
                # try if Sklearn is installed, or force GMM
                try:
                    from sklearn import metrics
                    if self.classifier != self.settingsdock.selectClassifier.currentText():
                        # self.modifyConfig('Classification','classifier',self.settingsdock.selectClassifier.currentText())
                        self.settings.setValue(
                            '/dzetsaka/classifier',
                            self.settingsdock.selectClassifier.currentText())
                except BaseException:
                    QMessageBox.warning(
                        self,
                        'Library missing',
                        'Scikit-learn library is missing on your computer.<br><br> You must use Gaussian Mixture Model, or <a href=\'https://github.com/lennepkade/dzetsaka/#installation-of-scikit-learn\'>consult dzetsaka homepage to learn on to install the missing library</a>.',
                        QMessageBox.Ok)
                    # reset to GMM
                    self.settingsdock.selectClassifier.setCurrentIndex(0)
                    #self.modifyConfig('Classification','classifier','Gaussian Mixture Model')
                    self.settings.setValue(
                        '/dzetsaka/classifier', 'Gaussian Mixture Model')
            else:
                #self.modifyConfig('Classification','classifier','Gaussian Mixture Model')
                self.settings.setValue(
                    '/dzetsaka/classifier',
                    'Gaussian Mixture Model')

        if self.sender() == self.settingsdock.classSuffix:
            if self.classSuffix != self.settingsdock.classSuffix.text():
                # self.modifyConfig('Classification','suffix',self.settingsdock.classSuffix.text())
                self.settings.setValue(
                    '/dzetsaka/classSuffix',
                    self.settingsdock.classSuffix.text())
        if self.sender() == self.settingsdock.classPrefix:
            if self.classPrefix != self.settingsdock.classPrefix.text():
                # self.modifyConfig('Classification','prefix',self.settingsdock.classPrefix.text())
                self.settings.setValue(
                    '/dzetsaka/classPrefix',
                    self.settingsdock.classPrefix.text())
        if self.sender() == self.settingsdock.maskSuffix:
            if self.maskSuffix != self.settingsdock.maskSuffix.text():
                # self.modifyConfig('Classification','maskSuffix',self.settingsdock.maskSuffix.text())
                self.settings.setValue(
                    '/dzetsaka/maskSuffix',
                    self.settingsdock.maskSuffix.text())
        if self.sender() == self.settingsdock.selectProviders:
            self.providerType = self.settingsdock.selectProviders.currentText()

            # self.modifyConfig('Providers','provider',self.settingsdock.selectProviders.currentText())
            self.settings.setValue(
                '/dzetsaka/providerType',
                self.settingsdock.selectProviders.currentText())
            QgsApplication.processingRegistry().removeProvider(self.provider)

            from .dzetsaka_provider import dzetsakaProvider
            self.provider = dzetsakaProvider(self.providerType)
            QgsApplication.processingRegistry().addProvider(self.provider)

    def modifyConfig(self, section, option, value):
        configFile = open(self.configFile, 'w')
        self.Config.set(section, option, value)
        self.Config.write(configFile)
        configFile.close()
