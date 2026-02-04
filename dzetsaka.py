"""dzetsaka: Classification Tool for QGIS.

A powerful and fast classification plugin for QGIS that supports 12 machine learning
algorithms for remote sensing image classification. Originally based on Gaussian
Mixture Model classifier, dzetsaka now includes state-of-the-art algorithms like
XGBoost and LightGBM with automatic dependency installation.

Features:
---------
- 12 machine learning algorithms (GMM, RF, SVM, KNN, XGB, LGB, CB, ET, GBC, LR, NB, MLP)
- Automatic hyperparameter optimization using cross-validation
- Automatic dependency installation for advanced algorithms
- Support for confidence maps and per-class probability outputs
- Spatial and temporal cross-validation methods
- Model saving and loading capabilities
- Integration with QGIS Processing framework

Supported Algorithms:
--------------------
Core (built-in):
    - GMM: Gaussian Mixture Model

Scikit-learn based:
    - RF: Random Forest
    - SVM: Support Vector Machine
    - KNN: K-Nearest Neighbors
    - ET: Extra Trees
    - GBC: Gradient Boosting Classifier
    - LR: Logistic Regression
    - NB: Gaussian Naive Bayes
    - MLP: Multi-layer Perceptron

Advanced gradient boosting:
    - XGB: XGBoost (requires: pip install xgboost)
    - LGB: LightGBM (requires: pip install lightgbm)
    - CB: CatBoost (requires: pip install catboost)

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

Original GMM implementation by Mathieu Fauvel

License:
--------
GNU General Public License v2.0 or later

Citation:
---------
Karasiak, N. (2016). Dzetsaka Qgis Classification plugin.
GitHub repository: https://github.com/nkarasiak/dzetsaka
DOI: 10.5281/zenodo.2552284
"""

# import basics
import os.path

# import outside libraries
# import configparser
import tempfile

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication, QSettings, Qt
from qgis.PyQt.QtGui import QAction, QIcon
from qgis.PyQt.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from qgis.core import QgsApplication

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr

# import local libraries
import contextlib

from . import classifier_config, ui
from .dzetsaka_provider import DzetsakaProvider
from .logging_utils import QgisLogger, show_error_dialog
from .scripts import mainfunction
from .scripts.function_dataraster import get_layer_source_path

# Import resources for icons
with contextlib.suppress(ImportError):
    from . import resources


class DzetsakaGUI(QDialog):
    """Main dzetsaka plugin class for QGIS integration.

    This class provides the main interface for the dzetsaka classification plugin
    in QGIS. It manages the plugin lifecycle, GUI components, settings, and
    coordinates between the user interface and classification algorithms.

    The plugin supports both dock widget and dialog interfaces, with settings
    persistence, automatic model training, image classification, and accuracy
    assessment capabilities.

    Attributes
    ----------
    iface : QgsInterface
        Reference to the QGIS interface
    provider : dzetsakaProvider
        Processing provider for batch operations
    plugin_dir : str
        Plugin directory path
    settings : QSettings
        Qt settings object for configuration persistence
    dockwidget : QDockWidget or None
        Main plugin dock widget
    classifier : str
        Currently selected classifier name
    classifiers : list
        List of available classifier names

    Examples
    --------
    The plugin is typically instantiated by QGIS:

    >>> plugin = DzetsakaGUI(iface)
    >>> plugin.initGui()  # Called by QGIS
    >>> plugin.run()  # Opens the classification dock

    """

    def __init__(self, iface):
        """Initialize the dzetsaka plugin.

        Parameters
        ----------
        iface : QgsInterface
            QGIS interface instance that provides hooks to manipulate the QGIS
            application at runtime, including access to map canvas, layers,
            toolbars, and menus.

        Notes
        -----
        The constructor:
        1. Saves reference to QGIS interface
        2. Loads plugin configuration from settings
        3. Initializes the processing provider
        4. Shows welcome dialog on first installation
        5. Sets up plugin actions and menus

        """
        # Save reference to the QGIS interface
        self.iface = iface
        self.log = QgisLogger(tag="Dzetsaka/Core")

        # add Processing loadAlgorithms

        # init dialog and dzetsaka dock
        QDialog.__init__(self)
        # sender = self.sender()
        self.settings = QSettings()
        self.loadConfig()

        self.provider = DzetsakaProvider(self.providerType)
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
        self.menu = self.tr("&dzetsaka")
        #        # TODO: We are going to let the user set this up in a future iteration
        #        self.toolbar = self.iface.addToolBar(u'dzetsaka')
        #        self.toolbar.setObjectName(u'dzetsaka')
        self.pluginIsActive = False
        self.dockwidget = None
        #

        # param
        self.lastSaveDir = ""

        # run dock
        # self.run()

    def rememberLastSaveDir(self, fileName):
        """Remember the last directory used for saving or loading files.

        This method stores the directory path of the given filename in the
        plugin settings for use in subsequent file dialogs, providing a
        better user experience.

        Parameters
        ----------
        fileName : str
            Full path to a file. The directory portion will be extracted
            and stored as the last used directory.

        Notes
        -----
        The directory is stored in QSettings under '/dzetsaka/lastSaveDir'
        and will persist between QGIS sessions.

        """
        if fileName != "":
            self.lastSaveDir = fileName
            self.settings.setValue("/dzetsaka/lastSaveDir", self.lastSaveDir)

    # noinspection PyMethodMayBeStatic
    def showWelcomeWidget(self):
        """Display the welcome widget for first-time users.

        Shows an introduction dialog with information about dzetsaka's
        features and capabilities. This is automatically shown on the
        first plugin installation.

        Notes
        -----
        After showing the welcome widget, the firstInstallation flag
        is set to False to prevent showing it again.

        """
        self.welcomeWidget = ui.welcomeWidget()
        self.welcomeWidget.show()
        self.settings.setValue("/dzetsaka/firstInstallation", False)

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate("dzetsaka", message)

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
        parent=None,
    ):
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
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)

        return action

    def get_icon_path(self, icon_name):
        """Get icon path, trying resource first, then fallback to file path."""
        resource_path = f":/plugins/dzetsaka/img/{icon_name}"
        file_path = os.path.join(self.plugin_dir, "img", icon_name)

        # Try to create QIcon with resource path first
        icon = QIcon(resource_path)
        if not icon.isNull():
            return resource_path
        # Fallback to file path
        return file_path

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        QgsApplication.processingRegistry().addProvider(self.provider)

        icon_path = self.get_icon_path("icon.png")
        self.add_action(
            icon_path,
            text=self.tr("welcome message"),
            callback=self.showWelcomeWidget,
            add_to_toolbar=False,
            parent=self.iface.mainWindow(),
        )

        icon_path = self.get_icon_path("icon.png")
        self.add_action(
            icon_path,
            text=self.tr("classification dock"),
            callback=self.run,
            parent=self.iface.mainWindow(),
        )

        icon_settings_path = self.get_icon_path("dzetsaka_settings.png")
        self.add_action(
            icon_settings_path,
            text=self.tr("settings"),
            callback=self.loadSettings,
            add_to_toolbar=True,
            parent=self.iface.mainWindow(),
        )

        self.dockIcon = QAction(
            QIcon(self.get_icon_path("icon.png")),
            "dzetsaka classification dock",
            self.iface.mainWindow(),
        )
        self.dockIcon.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.dockIcon)
        self.actions.append(self.dockIcon)

        self.settingsIcon = QAction(
            QIcon(self.get_icon_path("dzetsaka_settings.png")),
            "dzetsaka settings",
            self.iface.mainWindow(),
        )
        self.settingsIcon.triggered.connect(self.loadSettings)
        self.iface.addToolBarIcon(self.settingsIcon)
        self.actions.append(self.settingsIcon)

        # Classification Wizard — menu only, no toolbar icon
        icon_path = self.get_icon_path("icon.png")
        self.add_action(
            icon_path,
            text=self.tr("Classification Wizard"),
            callback=self.run_wizard,
            add_to_toolbar=False,
            parent=self.iface.mainWindow(),
        )

    # --------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed."""
        # print "** CLOSING DzetsakaGUI"
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
            self.iface.removePluginMenu(self.tr("&dzetsaka"), action)

    #
    #        # remove the toolbar
    #       qg del self.toolbar
    # --------------------------------------------------------------------------

    def run(self):
        """Run method that loads and starts the plugin."""
        if not self.pluginIsActive or self.dockwidget is None:
            self.pluginIsActive = True

            # print "** STARTING DzetsakaGUI"

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
            exceptRaster.remove("gdal")
            self.dockwidget.inRaster.setExcludedProviders(exceptRaster)

            exceptVector = QgsProviderRegistry.instance().providerList()
            exceptVector.remove("ogr")
            self.dockwidget.inShape.setExcludedProviders(exceptVector)

            self.dockwidget.outRaster.clear()
            self.dockwidget.outRasterButton.clicked.connect(self.select_output_file)

            self.dockwidget.outModel.clear()
            self.dockwidget.checkOutModel.clicked.connect(self.checkbox_state)

            self.dockwidget.inModel.clear()
            self.dockwidget.checkInModel.clicked.connect(self.checkbox_state)

            self.dockwidget.inMask.clear()
            self.dockwidget.checkInMask.clicked.connect(self.checkbox_state)

            self.dockwidget.outMatrix.clear()
            self.dockwidget.checkOutMatrix.clicked.connect(self.checkbox_state)

            self.dockwidget.outConfidenceMap.clear()
            self.dockwidget.checkInConfidence.clicked.connect(self.checkbox_state)

            self.dockwidget.inField.clear()

            if hasattr(self.dockwidget, "messageBanner"):
                self.dockwidget.messageBanner.linkActivated.connect(lambda _=None: self.run_wizard())

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(_LEFT_DOCK_AREA, self.dockwidget)
            self.dockwidget.show()

            def onChangedLayer():
                """!@brief Update columns if vector changes."""
                # We clear combobox
                self.dockwidget.inField.clear()
                # Then we fill it with new selected Layer
                if (
                    self.dockwidget.inField.currentText() == ""
                    and self.dockwidget.inShape.currentLayer()
                    and self.dockwidget.inShape.currentLayer() != "NoneType"
                ):
                    try:
                        activeLayer = self.dockwidget.inShape.currentLayer()
                        provider = activeLayer.dataProvider()
                        fields = provider.fields()
                        listFieldNames = [field.name() for field in fields]
                        self.dockwidget.inField.addItems(listFieldNames)
                    except BaseException:
                        self.log.warning(
                            "dzetsaka cannot change active layer. Maybe you opened an OSM/Online background ?"
                        )

            onChangedLayer()
            self.dockwidget.inShape.currentIndexChanged[int].connect(onChangedLayer)

            self.dockwidget.settingsButton.clicked.connect(self.loadSettings)

            # let's run the classification !
            self.dockwidget.performMagic.clicked.connect(self.runMagic)

            # Force Optional section closed by default (ignore persisted state)
            self.dockwidget.mGroupBox.setSaveCollapsedState(False)
            self.dockwidget.mGroupBox.setCollapsed(True)
            self.resizeDock()

            def update_optional_title():
                title = "Optional ▼" if self.dockwidget.mGroupBox.isCollapsed() else "Optional ▲"
                self.dockwidget.mGroupBox.setTitle(title)

            update_optional_title()
            self.dockwidget.mGroupBox.collapsedStateChanged.connect(self.resizeDock)
            self.dockwidget.mGroupBox.collapsedStateChanged.connect(update_optional_title)

    def resizeDock(self):
        """Resize dock widget based on group box collapse state."""
        if self.dockwidget.mGroupBox.isCollapsed():
            self.dockwidget.mGroupBox.setFixedHeight(20)
            self.dockwidget.setFixedHeight(390)

        else:
            self.dockwidget.setMinimumHeight(520)
            self.dockwidget.mGroupBox.setMinimumHeight(160)

    def select_output_file(self):
        """!@brief Select file to save, and gives the right extension if the user don't put it."""
        sender = self.sender()

        fileName, _filter = QFileDialog.getSaveFileName(
            self.dockwidget, "Select output file", self.lastSaveDir, "TIF (*.tif)"
        )
        self.rememberLastSaveDir(fileName)

        if not fileName:
            return
            # If user give right file extension, we don't add it

        fileName, fileExtension = os.path.splitext(fileName)

        if sender == self.dockwidget.outRasterButton:
            if fileExtension != ".tif":
                self.dockwidget.outRaster.setText(fileName + ".tif")
            else:
                self.dockwidget.outRaster.setText(fileName + fileExtension)

        # check if historical map run
        if "self.historicalmap" in locals():
            if sender == self.historicalmap.outRasterButton:
                if fileExtension != ".tif":
                    self.historicalmap.outRaster.setText(fileName + ".tif")
                else:
                    self.historicalmap.outRaster.setText(fileName + fileExtension)
            if sender == self.historicalmap.outShpButton:
                if fileExtension != ".shp":
                    self.historicalmap.outShp.setText(fileName + ".shp")
                else:
                    self.historicalmap.outShp.setText(fileName + fileExtension)
        # check if filters_dock run
        if "self.filters_dock" in locals():
            if sender == self.filters_dock.outRasterButton:
                if fileExtension != ".tif":
                    self.filters_dock.outRaster.setText(fileName + ".tif")
            else:
                self.filters_dock.outRaster.setText(fileName + fileExtension)

    def loadConfig(self):
        """!@brief Class that loads all saved settings from config.txt."""
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
            self.classifiers = classifier_config.CLASSIFIER_NAMES
            self.providers = ["Standard", "Experimental"]

            self.classifier = self.settings.value("/dzetsaka/classifier", "", str)
            if not self.classifier:
                self.classifier = self.classifiers[0]
                self.settings.setValue("/dzetsaka/classifier", self.classifier)

            self.classSuffix = self.settings.value("/dzetsaka/classSuffix", "", str)
            if not self.classSuffix:
                self.classSuffix = "_class"
                self.settings.setValue("/dzetsaka/classSuffix", self.classSuffix)

            self.classPrefix = self.settings.value("/dzetsaka/classPrefix", "", str)
            if not self.classPrefix:
                self.classPrefix = ""
                self.settings.setValue("/dzetsaka/classPrefix", self.classPrefix)

            self.maskSuffix = self.settings.value("/dzetsaka/maskSuffix", "", str)
            if not self.maskSuffix:
                self.maskSuffix = "_mask"
                self.settings.setValue("/dzetsaka/maskSuffix", self.maskSuffix)

            self.providerType = self.settings.value("/dzetsaka/providerType", "", str)
            if not self.providerType:
                self.providerType = self.providers[0]
                self.providerType = self.settings.setValue("/dzetsaka/providerType", self.providerType)

            self.firstInstallation = self.settings.value("/dzetsaka/firstInstallation", "None", bool)
            if self.firstInstallation is None:
                self.firstInstallation = True
                self.settings.setValue("/dzetsaka/firstInstallation", True)

        except BaseException:
            self.log.error("Failed to open config file " + self.configFile)
            show_error_dialog(
                "dzetsaka Configuration Error",
                "Failed to load configuration. Check the QGIS log for details.",
                parent=self.iface.mainWindow(),
            )

    def loadSettings(self):
        """!@brief load settings dock."""
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

            self.settingsdock.selectClassifier.currentIndexChanged[int].connect(self.saveSettings)

            self.settings.setValue("/dzetsaka/classifier", self.classifier)

            # suffix
            self.settingsdock.classSuffix.setText(self.classSuffix)
            self.settingsdock.classSuffix.textChanged.connect(self.saveSettings)
            self.settings.setValue("/dzetsaka/classSuffix", self.classSuffix)
            # prefix
            self.settingsdock.classPrefix.setText(self.classPrefix)
            self.settingsdock.classPrefix.textChanged.connect(self.saveSettings)
            self.settings.setValue("/dzetsaka/classPrefix", self.classPrefix)
            # mask suffix
            self.settingsdock.maskSuffix.setText(self.maskSuffix)
            self.settingsdock.maskSuffix.textChanged.connect(self.saveSettings)
            self.settings.setValue("/dzetsaka/maskSuffix", self.maskSuffix)
            ##

            for i, prvd in enumerate(self.providers):
                if self.providerType == prvd:
                    self.settingsdock.selectProviders.setCurrentIndex(i)

            self.settingsdock.selectProviders.currentIndexChanged[int].connect(self.saveSettings)
            self.settings.setValue("/dzetsaka/providerType", self.providerType)
            # Reload config for further use
            self.loadConfig()

        except BaseException:
            self.log.error("Failed to load settings...")
            show_error_dialog(
                "dzetsaka Settings Error",
                "Failed to load settings. Check the QGIS log for details.",
                parent=self.iface.mainWindow(),
            )

    def runMagic(self):
        """!@brief Perform training and classification for dzetsaka."""
        """
        VERIFICATION STEP
        """
        # verif before doing the job
        message = " "

        if self.dockwidget.inModel.text() == "":
            try:
                self.dockwidget.inShape.currentLayer().dataProvider().dataSourceUri()
            except BaseException:
                message = "\n - If you don't use a model, please specify a vector"
        try:
            self.dockwidget.inRaster.currentLayer().dataProvider().dataSourceUri()
        except BaseException:
            message = message + "\n - You need a raster to make a classification."

        try:
            # get raster
            inRaster = self.dockwidget.inRaster.currentLayer()
            inRaster = inRaster.dataProvider().dataSourceUri()

            # get raster proj
            inRasterOp = gdal.Open(inRaster)
            inRasterProj = inRasterOp.GetProjection()
            inRasterProj = osr.SpatialReference(inRasterProj)

            if self.dockwidget.inModel.text() == "":
                # verif srs
                # get vector
                inShapeLayer = self.dockwidget.inShape.currentLayer()
                inShape = get_layer_source_path(inShapeLayer)
                # get shp proj
                inShapeOp = ogr.Open(inShape)
                inShapeLyr = inShapeOp.GetLayer()
                inShapeProj = inShapeLyr.GetSpatialRef()

                # chekc IsSame Projection
                if inShapeProj.IsSameGeogCS(inRasterProj) == 0:
                    message = message + "\n - Raster and ROI do not have the same projection."
        except BaseException:
            self.log.error("inShape is : " + inShape)
            self.log.error("inRaster is : " + inRaster)
            self.log.error(
                "inShapeProj.IsSameGeogCS(inRasterProj) : " + inShapeProj.IsSameGeogCS(inRasterProj)
            )
            message = message + "\n - Can't compare projection between raster and vector."

        try:
            inMask = self.dockwidget.inMask.text()

            if inMask == "":
                inMask = None
            # check if mask with _mask.extension
            autoMask = os.path.splitext(inRaster)
            autoMask = autoMask[0] + self.maskSuffix + autoMask[1]

            if os.path.exists(autoMask):
                inMask = autoMask
                self.log.info("Mask found : " + str(autoMask))

            if inMask is not None:
                mask = gdal.Open(inMask, gdal.GA_ReadOnly)
                # Check size
                if (inRasterOp.RasterXSize != mask.RasterXSize) or (inRasterOp.RasterYSize != mask.RasterYSize):
                    message = message + "\n - Raster image and mask do not have the same size."

        except BaseException:
            message = message + "\n - Can't compare mask and raster size."
        """ END OF VERIFICATION STEP """

        if message != " ":
            reply = QMessageBox.question(
                self.iface.mainWindow(),
                "Informations missing or invalid",
                message + "\n Would you like to continue anyway ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                message = " "

        # all is ok, so do the job !
        if message == " ":
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
            if self.dockwidget.outRaster.text() == "":
                tempFolder = tempfile.mkdtemp()
                outRaster = os.path.join(
                    tempFolder,
                    self.classPrefix + os.path.splitext(os.path.basename(inRaster))[0] + self.classSuffix + ".tif",
                )

            else:
                outRaster = self.dockwidget.outRaster.text()

            # Confidence map

            if self.dockwidget.checkInConfidence.isChecked():
                confidenceMap = self.dockwidget.outConfidenceMap.text()
            else:
                confidenceMap = None

            # Get Classifier
            # retrieve shortname classifier
            inClassifier = classifier_config.get_classifier_code(self.classifier)
            self.log.info(f"Selected classifier: {self.classifier} (code: {inClassifier})")

            # Ensure inClassifier is definitely a string
            inClassifier = str(inClassifier)

            # Check if model, else perform training
            NODATA = -9999

            if model != "":
                model = self.dockwidget.inModel.text()
                self.log.info(f"Using existing model: {model}")
            else:
                if self.dockwidget.outModel.text() == "":
                    model = tempfile.mktemp("." + str(inClassifier))
                else:
                    model = self.dockwidget.outModel.text()
                self.log.info("Training new model (no existing model loaded)")

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
                    self.log.info(f"Starting training phase with {inClassifier} classifier")

                    temp = mainfunction.LearnModel(
                        raster_path=inRaster,
                        vector_path=inShape,
                        class_field=inField,
                        model_path=model,
                        split_config=inSplit,
                        random_seed=inSeed,
                        matrix_path=outMatrix,
                        classifier=inClassifier,
                        extraParam=None,
                        feedback="gui",
                    )

                    # perform learning

                stop = False
            except Exception as e:
                self.log.error("Training failed with error: " + str(e))
                self.log.error("Raster: " + str(inRaster))
                self.log.error("Vector: " + str(inShape))
                self.log.error("Column field: " + str(inField))
                self.log.error("Split: " + str(inSplit))
                self.log.error("Model: " + str(model))

                # Log configuration for GitHub issue reporting
                config_info = self._get_debug_info()
                self.log.info("Configuration for issue reporting:")
                self.log.info(config_info)

                # Show specific error message if available, otherwise generic message
                if str(e).strip():
                    message = (
                        "Training failed with the following error:<br><br>"
                        + str(e).replace("\n", "<br>")
                        + "<br><br>Please check the log for more details.<br><br>"
                        + "<b>If this issue persists:</b><br>"
                        + "Please report it at <a href='https://github.com/nkarasiak/dzetsaka/issues'>github.com/nkarasiak/dzetsaka/issues</a><br>"
                        + "Include your classifier settings, QGIS version, and error details."
                    )
                else:
                    message = (
                        "Something went wrong during the training. Common issues:<br>"
                        "• Non-integer values in the '" + str(inField) + "' column<br>"
                        "• Mismatched projections between shapefile and raster<br>"
                        "• Invalid geometries in the shapefile<br>"
                        "• Insufficient training samples<br><br>"
                        "Please check the log for more details.<br><br>"
                        + "<b>If this issue persists:</b><br>"
                        + "Please report it at <a href='https://github.com/nkarasiak/dzetsaka/issues'>github.com/nkarasiak/dzetsaka/issues</a><br>"
                        + "Include your classifier settings, QGIS version, and error details."
                    )
                QMessageBox.warning(
                    self,
                    "dzetsaka Training Error",
                    message,
                    QMessageBox.StandardButton.Ok,
                )
                stop = True
                QApplication.restoreOverrideCursor()

            if not stop:
                # Begin classification
                self.log.info(f"Starting classification with {inClassifier} classifier")
                try:
                    temp = mainfunction.ClassifyImage()

                    temp.initPredict(
                        raster_path=inRaster,
                        model_path=model,
                        output_path=outRaster,
                        mask_path=inMask,
                        confidenceMap=confidenceMap,
                        confidenceMapPerClass=None,
                        NODATA=NODATA,
                        feedback="gui",
                    )
                    self.log.info("Classification completed successfully")
                    self.iface.addRasterLayer(outRaster)

                    if confidenceMap:
                        self.iface.addRasterLayer(confidenceMap)

                except Exception as e:
                    self.log.error("Classification failed with error: " + str(e))

                    # Log configuration for GitHub issue reporting
                    config_info = self._get_debug_info()
                    self.log.info("Configuration for issue reporting:")
                    self.log.info(config_info)

                    # Show error message with GitHub reporting guidance
                    message = (
                        "Classification failed with the following error:<br><br>"
                        + str(e).replace("\n", "<br>")
                        + "<br><br>Please check the log for more details.<br><br>"
                        + "<b>If this issue persists:</b><br>"
                        + "Please report it at <a href='https://github.com/nkarasiak/dzetsaka/issues'>github.com/nkarasiak/dzetsaka/issues</a><br>"
                        + "Include your classifier settings, QGIS version, and error details from the log."
                    )
                    QMessageBox.warning(
                        self,
                        "dzetsaka Classification Error",
                        message,
                        QMessageBox.StandardButton.Ok,
                    )

    def checkbox_state(self):
        """!@brief Manage checkbox in main dock."""
        sender = self.sender()

        # If load model
        if sender == self.dockwidget.checkInModel and self.dockwidget.checkInModel.isChecked():
            fileName, _filter = QFileDialog.getOpenFileName(self.dockwidget, "Select your file", self.lastSaveDir)
            self.rememberLastSaveDir(fileName)
            if fileName != "":
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
            fileName, _filter = QFileDialog.getSaveFileName(self.dockwidget, "Select output file", self.lastSaveDir)
            self.rememberLastSaveDir(fileName)
            if fileName != "":
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
                self.dockwidget,
                "Select your mask raster",
                self.lastSaveDir,
                "TIF (*.tif)",
            )
            self.rememberLastSaveDir(fileName)
            if fileName != "":
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
                self.dockwidget, "Save to a *.csv file", self.lastSaveDir, "CSV (*.csv)"
            )
            self.rememberLastSaveDir(fileName)
            if fileName != "":
                fileName, fileExtension = os.path.splitext(fileName)
                fileName = fileName + ".csv"
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
                self.dockwidget,
                "Select output file (*.tif)",
                self.lastSaveDir,
                "TIF (*.tif)",
            )
            self.rememberLastSaveDir(fileName)
            if fileName != "":
                fileName, fileExtension = os.path.splitext(fileName)
                fileName = fileName + ".tif"
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
        """!@brief save settings if modifications."""
        # Change classifier
        if self.sender() == self.settingsdock.selectClassifier:
            selected_classifier = self.settingsdock.selectClassifier.currentText()
            classifier_code = classifier_config.get_classifier_code(selected_classifier)

            # Check required dependencies
            missing_required = []
            required_message = ""

            if classifier_config.requires_sklearn(classifier_code):
                # Test sklearn availability directly
                sklearn_available = False
                try:
                    import sklearn

                    sklearn_available = True
                    self.log.info(
                        f"Sklearn detected for {selected_classifier}: version {sklearn.__version__}"
                    )
                except ImportError as e:
                    self.log.warning(f"Sklearn import failed for {selected_classifier}: {e}")

                if not sklearn_available:
                    missing_required.append("scikit-learn")
                    required_message += "Scikit-learn library is missing.<br>"
                    required_message += "Install with: <code>pip install scikit-learn</code><br><br>"

            if classifier_config.requires_xgboost(classifier_code):
                try:
                    import xgboost  # noqa: F401
                except ImportError:
                    missing_required.append("XGBoost")
                    required_message += "XGBoost library is missing.<br>"
                    required_message += "Install with: <code>pip install xgboost</code><br><br>"

            if classifier_config.requires_lightgbm(classifier_code):
                try:
                    import lightgbm  # noqa: F401
                except ImportError:
                    missing_required.append("LightGBM")
                    required_message += "LightGBM library is missing.<br>"
                    required_message += "Install with: <code>pip install lightgbm</code><br><br>"
            if classifier_config.requires_catboost(classifier_code):
                try:
                    import catboost  # noqa: F401
                except ImportError:
                    missing_required.append("CatBoost")
                    required_message += "CatBoost library is missing.<br>"
                    required_message += "Install with: <code>pip install catboost</code><br><br>"

            # Check optional enhancements (for sklearn-based classifiers)
            missing_optional = []
            optional_message = ""
            if classifier_config.requires_sklearn(classifier_code):
                try:
                    import optuna  # noqa: F401
                except ImportError:
                    missing_optional.append("Optuna")
                    optional_message += "Optuna library is missing (optional - advanced hyperparameter optimization).<br>"
                    optional_message += "Install with: <code>pip install optuna</code><br><br>"

            if missing_required:
                # Required dependencies missing — must install or fall back to GMM
                error_message = "<b>Required dependencies:</b><br>" + required_message
                if missing_optional:
                    error_message += "<b>Optional enhancements:</b><br>" + optional_message

                reply = QMessageBox.question(
                    self,
                    f"Missing Dependencies for {selected_classifier}",
                    f"{error_message}<br>"
                    f"<b>🧪 Experimental Feature:</b><br>"
                    f"Would you like dzetsaka to try installing the missing dependencies automatically?<br><br>"
                    f"<b>Note:</b> This is experimental and may not work in all QGIS environments.<br>"
                    f"Click 'Yes' to try auto-install, 'No' to install manually, or 'Cancel' to use GMM.<br><br>"
                    f"<a href='https://github.com/lennepkade/dzetsaka/#installation'>Manual installation guide</a>",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Try auto-installation (required + optional together)
                    if self._try_install_dependencies(missing_required + missing_optional):
                        # Success! Update classifier
                        self.settings.setValue("/dzetsaka/classifier", selected_classifier)
                        self.classifier = selected_classifier
                        QMessageBox.information(
                            self,
                            "Installation Successful",
                            f"Dependencies installed successfully!<br><br>"
                            f"<b>Note:</b> If {selected_classifier} doesn't work immediately, "
                            f"please restart QGIS to ensure the new libraries are properly loaded.<br><br>"
                            f"You can now try using {selected_classifier}.",
                            QMessageBox.StandardButton.Ok,
                        )
                    else:
                        # Failed, reset to GMM
                        self.settingsdock.selectClassifier.setCurrentIndex(0)
                        self.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
                        self.classifier = "Gaussian Mixture Model"
                elif reply == QMessageBox.StandardButton.Cancel:
                    # Reset to GMM
                    self.settingsdock.selectClassifier.setCurrentIndex(0)
                    self.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
                    self.classifier = "Gaussian Mixture Model"
                # If No, keep current selection but don't save (user will handle manually)

            elif missing_optional:
                # Only optional deps missing — classifier is fully usable, suggest install
                if self.classifier != selected_classifier:
                    self.settings.setValue("/dzetsaka/classifier", selected_classifier)
                    self.classifier = selected_classifier

                reply = QMessageBox.question(
                    self,
                    "Optional Enhancement Available",
                    f"{optional_message}"
                    f"<b>Note:</b> {selected_classifier} works without it using standard cross-validation.<br><br>"
                    f"Would you like to install it automatically?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self._try_install_dependencies(missing_optional)

            else:
                # All dependencies satisfied, update classifier
                if self.classifier != selected_classifier:
                    self.settings.setValue("/dzetsaka/classifier", selected_classifier)
                    self.classifier = selected_classifier

        if self.sender() == self.settingsdock.classSuffix and self.classSuffix != self.settingsdock.classSuffix.text():
            # self.modifyConfig('Classification','suffix',self.settingsdock.classSuffix.text())
            self.settings.setValue("/dzetsaka/classSuffix", self.settingsdock.classSuffix.text())
        if self.sender() == self.settingsdock.classPrefix and self.classPrefix != self.settingsdock.classPrefix.text():
            # self.modifyConfig('Classification','prefix',self.settingsdock.classPrefix.text())
            self.settings.setValue("/dzetsaka/classPrefix", self.settingsdock.classPrefix.text())
        if self.sender() == self.settingsdock.maskSuffix and self.maskSuffix != self.settingsdock.maskSuffix.text():
            # self.modifyConfig('Classification','maskSuffix',self.settingsdock.maskSuffix.text())
            self.settings.setValue("/dzetsaka/maskSuffix", self.settingsdock.maskSuffix.text())
        if self.sender() == self.settingsdock.selectProviders:
            self.providerType = self.settingsdock.selectProviders.currentText()

            # self.modifyConfig('Providers','provider',self.settingsdock.selectProviders.currentText())
            self.settings.setValue(
                "/dzetsaka/providerType",
                self.settingsdock.selectProviders.currentText(),
            )
            QgsApplication.processingRegistry().removeProvider(self.provider)

            from .dzetsaka_provider import DzetsakaProvider

            self.provider = DzetsakaProvider(self.providerType)
            QgsApplication.processingRegistry().addProvider(self.provider)

    def _get_debug_info(self):
        """Generate debug information for GitHub issue reporting."""
        try:
            import platform

            # Get QGIS version
            qgis_version = QgsApplication.applicationVersion()

            # Get Python version
            python_version = platform.python_version()

            # Get OS info
            os_info = f"{platform.system()} {platform.release()}"

            # Get classifier and settings
            classifier_code = classifier_config.get_classifier_code(self.classifier)

            # Check library availability
            sklearn_available = "No"
            try:
                import sklearn

                sklearn_available = f"Yes ({sklearn.__version__})"
            except ImportError:
                pass

            xgboost_available = "No"
            try:
                import xgboost

                xgboost_available = f"Yes ({xgboost.__version__})"
            except ImportError:
                pass

            lightgbm_available = "No"
            try:
                import lightgbm

                lightgbm_available = f"Yes ({lightgbm.__version__})"
            except ImportError:
                pass

            catboost_available = "No"
            try:
                import catboost

                catboost_available = f"Yes ({catboost.__version__})"
            except ImportError:
                pass

            debug_info = f"""
=== DZETSAKA DEBUG INFO ===
Plugin Version: 4.1.2
QGIS Version: {qgis_version}
Python Version: {python_version}
Operating System: {os_info}

Current Classifier: {self.classifier} ({classifier_code})
Available Libraries:
- Scikit-learn: {sklearn_available}
- XGBoost: {xgboost_available}
- LightGBM: {lightgbm_available}
- CatBoost: {catboost_available}

Settings:
- Class Suffix: {getattr(self, "classSuffix", "N/A")}
- Class Prefix: {getattr(self, "classPrefix", "N/A")}
- Mask Suffix: {getattr(self, "maskSuffix", "N/A")}
- Provider Type: {getattr(self, "providerType", "N/A")}
=== END DEBUG INFO ===
"""
            return debug_info.strip()
        except Exception as e:
            return f"Error generating debug info: {e!s}"

    def _try_install_dependencies(self, missing_deps):
        """Experimental feature to auto-install missing dependencies.

        Parameters
        ----------
        missing_deps : list
            List of missing dependency names

        Returns
        -------
        bool
            True if installation succeeded, False otherwise

        """

        # Package installation with proper Python executable detection
        def install_package(package, extra_args=None):
            import os
            import subprocess
            import sys

            def run_command(cmd, description):
                """Run a command and return (success, output)."""
                try:
                    self.log.info(f"Trying {description}: {' '.join(cmd)}")
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        universal_newlines=True,
                    )

                    output_lines = []
                    while True:
                        output = process.stdout.readline()
                        if output == "" and process.poll() is not None:
                            break
                        if output:
                            output_lines.append(output.strip())
                            self.log.info(f"  {output.strip()}")

                    return_code = process.poll()
                    output_text = "\n".join(output_lines)
                    return return_code == 0, output_text

                except FileNotFoundError:
                    self.log.warning(f"Command not found: {cmd[0]}")
                    return False, "Command not found"
                except Exception as e:
                    self.log.error(f"Error: {e}")
                    return False, str(e)

            # Find the correct Python executable (workaround for QGIS sys.executable issue)
            def find_python():
                if sys.platform != "win32":
                    return sys.executable

                # On Windows, sys.executable points to QGIS, not Python
                for path in sys.path:
                    assumed_path = os.path.join(path, "python.exe")
                    if os.path.isfile(assumed_path):
                        self.log.info(f"Found Python executable: {assumed_path}")
                        return assumed_path
                return "python"

            python_exe = find_python()
            self.log.info(f"Installing {package} for Python: {python_exe}")

            # Method 1: Try python -m pip (preferred)
            pip_args = ["-m", "pip", "install", package, "--user", "--no-input", "--no-deps"]
            if extra_args:
                pip_args.extend(extra_args)
            success, output = run_command(
                [python_exe, *pip_args],
                "pip module",
            )
            if success:
                self.log.info(f"Successfully installed {package}")
                return True

            # Check if pip module is missing
            pip_missing = "No module named pip" in output

            if pip_missing:
                self.log.warning("pip module not available, trying alternatives...")

                # Method 2: Try ensurepip to bootstrap pip
                self.log.info("Attempting to bootstrap pip with ensurepip...")
                success, _ = run_command(
                    [python_exe, "-m", "ensurepip", "--user"],
                    "ensurepip",
                )
                if success:
                    # Retry pip install after bootstrapping
                    success, _ = run_command(
                        [python_exe, *pip_args],
                        "pip after ensurepip",
                    )
                    if success:
                        self.log.info(f"Successfully installed {package}")
                        return True

                # Method 3: On Linux, try apt/dnf for system packages
                if sys.platform.startswith("linux"):
                    # Map pip packages to apt packages
                    apt_packages = {
                        "scikit-learn": "python3-sklearn",
                        "xgboost": "python3-xgboost",
                        "lightgbm": "python3-lightgbm",
                        "catboost": "python3-catboost",
                    }
                    apt_pkg = apt_packages.get(package.lower())

                    if apt_pkg:
                        # Check if apt is available
                        apt_path = "/usr/bin/apt"
                        if os.path.exists(apt_path):
                            self.log.info(
                                f"Trying system package manager (apt install {apt_pkg})..."
                            )
                            # Try pkexec for graphical sudo
                            success, output = run_command(
                                ["pkexec", apt_path, "install", "-y", apt_pkg],
                                "apt via pkexec",
                            )
                            if success:
                                self.log.info(f"Successfully installed {apt_pkg} via apt")
                                return True
                            else:
                                self.log.warning(f"apt install failed: {output}")

            # All methods failed
            self.log.error(
                f"Could not install {package}. Please install manually:\n"
                "  Option 1 - Install pip first:\n"
                "    sudo apt install python3-pip\n"
                "    pip3 install --user scikit-learn\n"
                "  Option 2 - Install via apt directly:\n"
                "    sudo apt install python3-sklearn\n"
                "  Then restart QGIS."
            )
            return False

        def _is_importable(module_name):
            import importlib.util

            return importlib.util.find_spec(module_name) is not None

        # Mapping of dependency names to pip packages
        pip_packages = {
            "scikit-learn": "scikit-learn",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "optuna": "optuna",
            "shap": "shap",
            "imbalanced-learn": "imbalanced-learn",
            "imblearn": "imbalanced-learn",
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "CatBoost": "catboost",
            "Optuna": "optuna",
        }

        package_deps = {
            # Absolute-minimum deps needed for import/runtime (no upgrades).
            "scikit-learn": [
                ("numpy", "numpy"),
                ("scipy", "scipy"),
                ("joblib", "joblib"),
                ("threadpoolctl", "threadpoolctl"),
            ],
            "xgboost": [("numpy", "numpy")],
            "lightgbm": [("numpy", "numpy")],
            "catboost": [("numpy", "numpy")],
            "optuna": [
                ("numpy", "numpy"),
                ("scipy", "scipy"),
                ("tqdm", "tqdm"),
                ("typing_extensions", "typing_extensions"),
            ],
            "shap": [
                ("numpy", "numpy"),
                ("scipy", "scipy"),
                ("pandas", "pandas"),
            ],
            "imbalanced-learn": [("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn")],
        }
        base_imports = {
            "scikit-learn": "sklearn",
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "optuna": "optuna",
            "shap": "shap",
            "imbalanced-learn": "imblearn",
        }

        try:
            # Show progress dialog
            from qgis.PyQt.QtWidgets import QProgressDialog

            progress = QProgressDialog("Installing dependencies...", "Cancel", 0, len(missing_deps), self)
            progress.setWindowModality(2)  # ApplicationModal
            progress.show()

            success_count = 0

            for i, dep in enumerate(missing_deps):
                if progress.wasCanceled():
                    QMessageBox.warning(
                        self,
                        "Installation Cancelled",
                        "Dependency installation was cancelled.",
                        QMessageBox.StandardButton.Ok,
                    )
                    return False

                progress.setLabelText(f"Installing {dep}...")
                progress.setValue(i)

                # Get the pip package name
                dep_key = dep.strip()
                package_name = pip_packages.get(dep_key, pip_packages.get(dep_key.lower(), dep_key.lower()))
                targets = [package_name]
                for import_name, pip_name in package_deps.get(package_name, []):
                    if not _is_importable(import_name):
                        targets.append(pip_name)
                # Deduplicate while preserving order
                seen = set()
                targets = [t for t in targets if not (t in seen or seen.add(t))]

                try:
                    dep_installed = False
                    base_import = base_imports.get(package_name, package_name)
                    if _is_importable(base_import):
                        dep_installed = True
                        success_count += 1
                        self.log.info(f"{package_name} already available; skipping install.")

                    for target in targets:
                        if dep_installed and target == package_name:
                            continue
                        # Try direct pip installation first (preferred method)
                        self.log.info(f"Attempting to install {target} using direct pip (no-deps)...")

                        if install_package(target):
                            self.log.info(f"Successfully installed {target}")
                            if target == package_name and not dep_installed:
                                success_count += 1
                                dep_installed = True

                            # Try to import to verify installation (after clearing import cache)
                            import importlib

                            try:
                                importlib.invalidate_caches()
                                imported = importlib.import_module(target)
                                if hasattr(imported, "__version__"):
                                    self.log.info(f"Verified {target} import: {imported.__version__}")
                                else:
                                    self.log.info(f"Verified {target} import.")
                            except ImportError as import_error:
                                self.log.warning(
                                    f"Package {target} installed but not immediately available: {import_error}"
                                )
                                self.log.warning("You may need to restart QGIS to use this library.")
                                # Still count as success since pip succeeded
                        else:
                            self.log.warning(f"Direct pip installation failed for {target}")

                except Exception as e:
                    self.log.error(f"Error installing {package_name}: {e!s}")

            progress.setValue(len(missing_deps))
            progress.close()

            if success_count == len(missing_deps):
                return True
            else:
                QMessageBox.warning(
                    self,
                    "Installation Incomplete",
                    f"Only {success_count} of {len(missing_deps)} dependencies were installed successfully.<br><br>"
                    f"<b>To install manually, run one of these commands in a terminal:</b><br><br>"
                    f"<b>Option 1</b> - Install via apt (recommended for Debian/Ubuntu):<br>"
                    f"<code>sudo apt install python3-sklearn</code><br><br>"
                    f"<b>Option 2</b> - Install pip first, then use pip:<br>"
                    f"<code>sudo apt install python3-pip</code><br>"
                    f"<code>pip3 install --user scikit-learn</code><br><br>"
                    f"Then restart QGIS.",
                    QMessageBox.StandardButton.Ok,
                )
                return False

        except Exception as e:
            self.log.error(f"Error during dependency installation: {e!s}")
            QMessageBox.critical(
                self,
                "Installation Error",
                f"An error occurred during installation:<br><br>{e!s}<br><br>"
                f"<b>To install manually:</b><br><br>"
                f"<code>pip install scikit-learn xgboost lightgbm optuna</code><br><br>"
                f"<code>pip install catboost</code><br><br>"
                f"On Debian/Ubuntu you can also install scikit-learn via apt:<br>"
                f"<code>sudo apt install python3-sklearn</code><br><br>"
                f"Then restart QGIS.",
                QMessageBox.StandardButton.Ok,
            )
            return False

    def run_wizard(self):
        """Open the Classification Wizard dialog."""
        self._wizard = ui.ClassificationWizard(self.iface.mainWindow(), installer=self)
        self._wizard.classificationRequested.connect(self.execute_wizard_config)
        self._wizard.show()

    def execute_wizard_config(self, config):
        """Run training and classification driven by the wizard config dict.

        Parameters
        ----------
        config : dict
            The full configuration dictionary emitted by ClassificationWizard.
            Keys: raster, vector, class_field, load_model, classifier,
            extraParam, output_raster, confidence_map, save_model,
            confusion_matrix, split_percent.

        """
        inRaster = config.get("raster", "")
        inShape = config.get("vector", "")
        inField = config.get("class_field", "")
        inClassifier = str(config.get("classifier", "GMM"))
        extraParam = config.get("extraParam", None)

        # --- output raster (temp if blank) ---
        outRaster = config.get("output_raster", "")
        if not outRaster:
            tempFolder = tempfile.mkdtemp()
            outRaster = os.path.join(tempFolder, os.path.splitext(os.path.basename(inRaster))[0] + "_class.tif")

        # --- confidence map ---
        confidenceMap = config.get("confidence_map", "") or None

        # --- model path ---
        loadModel = config.get("load_model", "")
        if loadModel:
            model = loadModel
        else:
            saveModel = config.get("save_model", "")
            model = saveModel if saveModel else tempfile.mktemp("." + inClassifier)

        # --- confusion matrix / split ---
        outMatrix = config.get("confusion_matrix", "") or None
        inSplit = config.get("split_percent", 100)
        if not outMatrix:
            inSplit = 100

        NODATA = -9999
        inSeed = 0
        stop = False

        # --- Training phase (skipped when loading an existing model) ---
        if not loadModel:
            try:
                self.log.info(f"[Wizard] Training with {inClassifier}")
                mainfunction.LearnModel(
                    raster_path=inRaster,
                    vector_path=inShape,
                    class_field=inField,
                    model_path=model,
                    split_config=inSplit,
                    random_seed=inSeed,
                    matrix_path=outMatrix,
                    classifier=inClassifier,
                    extraParam=extraParam,
                    feedback="gui",
                )
            except Exception as e:
                self.log.error("[Wizard] Training failed: " + str(e))
                message = (
                    "Training failed:<br><br>"
                    + str(e).replace("\n", "<br>")
                    + "<br><br>Check the QGIS log for details."
                )
                QMessageBox.warning(
                    self,
                    "dzetsaka Wizard — Training Error",
                    message,
                    QMessageBox.StandardButton.Ok,
                )
                stop = True

        # --- Classification phase ---
        if not stop:
            try:
                self.log.info("[Wizard] Starting classification")
                classifier_obj = mainfunction.ClassifyImage()
                classifier_obj.initPredict(
                    raster_path=inRaster,
                    model_path=model,
                    output_path=outRaster,
                    mask_path=None,
                    confidenceMap=confidenceMap,
                    confidenceMapPerClass=None,
                    NODATA=NODATA,
                    feedback="gui",
                )
                self.log.info("[Wizard] Classification completed")
                self.iface.addRasterLayer(outRaster)
                if confidenceMap:
                    self.iface.addRasterLayer(confidenceMap)
            except Exception as e:
                self.log.error("[Wizard] Classification failed: " + str(e))
                message = (
                    "Classification failed:<br><br>"
                    + str(e).replace("\n", "<br>")
                    + "<br><br>Check the QGIS log for details."
                )
                QMessageBox.warning(
                    self,
                    "dzetsaka Wizard — Classification Error",
                    message,
                    QMessageBox.StandardButton.Ok,
                )

    def modifyConfig(self, section, option, value):
        """Modify configuration file with new section/option/value.

        Parameters
        ----------
        section : str
            Configuration section name
        option : str
            Configuration option name
        value : str
            New value to set

        """
        with open(self.configFile, "w") as configFile:
            self.Config.set(section, option, value)
            self.Config.write(configFile)
# Qt6 enum compatibility (QGIS 4 / PyQt6)
try:
    _LEFT_DOCK_AREA = Qt.LeftDockWidgetArea
except AttributeError:
    _LEFT_DOCK_AREA = Qt.DockWidgetArea.LeftDockWidgetArea
