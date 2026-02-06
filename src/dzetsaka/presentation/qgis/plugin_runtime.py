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
import configparser
import os.path
from pathlib import Path

# import outside libraries
# import configparser
import tempfile

from qgis.core import QgsApplication

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import QCoreApplication, QSettings, Qt, QTimer
from qgis.PyQt.QtGui import QAction, QIcon
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QMessageBox

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr

# import local libraries
import contextlib

from dzetsaka import classifier_config, ui
from dzetsaka.dzetsaka_provider import DzetsakaProvider
from dzetsaka.logging_utils import QgisLogger, show_error_dialog
from dzetsaka.presentation.qgis.task_runner import TaskFeedbackAdapter
from dzetsaka.scripts.function_dataraster import get_layer_source_path

# Import resources for icons
with contextlib.suppress(ImportError):
    from dzetsaka import resources


_TaskFeedbackAdapter = TaskFeedbackAdapter


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
    >>> plugin.run()  # Opens the classifier UI

    """

    DEFAULT_MASK_SUFFIX = "_mask"
    DEFAULT_PROVIDER_TYPE = "Standard"

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
        self.plugin_dir = str(Path(__file__).resolve().parents[4])
        self.plugin_version = self._read_plugin_version()
        shown_version = self.settings.value("/dzetsaka/onboardingShownVersion", "", str) or ""
        should_show_onboarding = shown_version != self.plugin_version
        self._open_welcome_on_init = bool(self.firstInstallation or should_show_onboarding)
        self._open_dashboard_on_init = bool(self.firstInstallation or should_show_onboarding)

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
        self.wizarddock = None
        self._active_classification_task = None
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
            text=self.tr("classifier dashboard"),
            callback=self.run_wizard,
            parent=self.iface.mainWindow(),
        )

        self.dockIcon = QAction(
            QIcon(self.get_icon_path("icon.png")),
            "dzetsaka classifier dashboard",
            self.iface.mainWindow(),
        )
        self.dockIcon.triggered.connect(self.run_wizard)
        self.iface.addToolBarIcon(self.dockIcon)
        self.actions.append(self.dockIcon)

        if self._open_welcome_on_init:
            self._open_welcome_on_init = False
            self.settings.setValue("/dzetsaka/onboardingShownVersion", self.plugin_version)
            QTimer.singleShot(400, self.showWelcomeWidget)
        if self._open_dashboard_on_init:
            self._open_dashboard_on_init = False
            QTimer.singleShot(800, self.run_wizard)

    def _read_plugin_version(self):
        # type: () -> str
        """Read plugin version from metadata.txt, fallback to 'unknown'."""
        from dzetsaka.presentation.qgis.metadata_utils import read_plugin_version

        return read_plugin_version(self.plugin_dir, logger=self.log)

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
        if self.wizarddock is not None:
            self.wizarddock.close()

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

            self.dockwidget.settingsButton.clicked.connect(self.run_wizard)

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

            self.classifier = self.settings.value("/dzetsaka/classifier", "", str)
            if not self.classifier:
                self.classifier = self.classifiers[0]
                self.settings.setValue("/dzetsaka/classifier", self.classifier)

            # Legacy customizable naming/provider settings have been removed.
            # Keep deterministic defaults for behavior.
            self.maskSuffix = self.DEFAULT_MASK_SUFFIX
            self.providerType = self.DEFAULT_PROVIDER_TYPE

            first_install_raw = self.settings.value("/dzetsaka/firstInstallation", None)
            if first_install_raw is None:
                self.firstInstallation = True
                self.settings.setValue("/dzetsaka/firstInstallation", True)
            elif isinstance(first_install_raw, bool):
                self.firstInstallation = first_install_raw
            else:
                self.firstInstallation = str(first_install_raw).strip().lower() in ("1", "true", "yes", "on")

        except BaseException:
            self.log.error("Failed to open config file " + self.configFile)
            show_error_dialog(
                "dzetsaka Configuration Error",
                "Failed to load configuration. Check the QGIS log for details.",
                parent=self.iface.mainWindow(),
            )

    def loadSettings(self):
        """Legacy entry point retained for compatibility: open dashboard."""
        self.run_wizard()

    def runMagic(self):
        """!@brief Perform training and classification for dzetsaka."""
        """
        VERIFICATION STEP
        """
        # Mandatory checks should stop immediately (no "continue anyway").
        message = " "
        inRaster = ""
        inShape = ""
        inRasterOp = None
        inShapeProj = None
        inRasterProj = None
        model_path = self.dockwidget.inModel.text().strip()

        inRasterLayer = self.dockwidget.inRaster.currentLayer()
        if inRasterLayer is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Missing Input",
                "Please select an input raster before running classification.",
                QMessageBox.StandardButton.Ok,
            )
            return

        try:
            inRaster = get_layer_source_path(inRasterLayer)
        except Exception:
            inRaster = ""
        if not inRaster:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Invalid Raster",
                "Could not read the selected raster source. Please select a valid raster layer.",
                QMessageBox.StandardButton.Ok,
            )
            return

        # Vector is mandatory only when a model is not provided.
        if model_path == "":
            inShapeLayer = self.dockwidget.inShape.currentLayer()
            if inShapeLayer is None:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Missing Input",
                    "If you don't use an existing model, please select training data (vector).",
                    QMessageBox.StandardButton.Ok,
                )
                return
            try:
                inShape = get_layer_source_path(inShapeLayer)
            except Exception:
                inShape = ""
            if not inShape:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Invalid Vector",
                    "Could not read the selected training vector source. Please select a valid vector layer.",
                    QMessageBox.StandardButton.Ok,
                )
                return

        try:
            # get raster proj
            inRasterOp = gdal.Open(inRaster)
            inRasterProj = osr.SpatialReference(inRasterOp.GetProjection()) if inRasterOp is not None else None

            if model_path == "":
                # verif srs only when training vector is used
                inShapeOp = ogr.Open(inShape)
                inShapeLyr = inShapeOp.GetLayer() if inShapeOp is not None else None
                inShapeProj = inShapeLyr.GetSpatialRef() if inShapeLyr is not None else None

                # check IsSame Projection
                if inShapeProj is not None and inRasterProj is not None and inShapeProj.IsSameGeogCS(inRasterProj) == 0:
                    message = message + "\n - Raster and ROI do not have the same projection."
        except Exception as exc:
            self.log.error(f"Projection validation error: {exc}")
            if inShape:
                self.log.error("inShape is : " + inShape)
            if inRaster:
                self.log.error("inRaster is : " + inRaster)
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

            if inMask is not None and inRasterOp is not None:
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

            # Get classifier short code (used by processing and default output naming)
            inClassifier = classifier_config.get_classifier_code(self.classifier)
            self.log.info(f"Selected classifier: {self.classifier} (code: {inClassifier})")

            # create temp if not output raster
            if self.dockwidget.outRaster.text() == "":
                tempFolder = tempfile.mkdtemp()
                outRaster = os.path.join(
                    tempFolder,
                    self._default_output_name(inRaster, inClassifier),
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

            do_training = not self.dockwidget.checkInModel.isChecked()
            if not self._validate_classification_request(
                raster_path=inRaster,
                do_training=do_training,
                vector_path=inShape if do_training else None,
                class_field=inField if do_training else None,
                model_path=model if not do_training else None,
                source_label="Main Panel",
            ):
                return
            if not self._ensure_classifier_runtime_ready(
                inClassifier, source_label="Main Panel", fallback_to_gmm=True
            ):
                return
            self.log.info(
                f"Starting {'training and ' if do_training else ''}classification with {inClassifier} classifier"
            )
            self._start_classification_task(
                description=f"dzetsaka: {inClassifier} classification",
                do_training=do_training,
                raster_path=inRaster,
                vector_path=inShape if do_training else None,
                class_field=inField if do_training else None,
                model_path=model,
                split_config=inSplit,
                random_seed=inSeed,
                matrix_path=outMatrix,
                classifier=inClassifier,
                output_path=outRaster,
                mask_path=inMask,
                confidence_map=confidenceMap,
                nodata=NODATA,
                extra_params=None,
                error_context="Main panel classification workflow",
                success_prefix="Main",
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

    def _check_sklearn_usable(self):
        # type: () -> tuple[bool, str]
        """Return whether scikit-learn is fully usable in the current runtime."""
        try:
            import sklearn

            # Broken installs can import a namespace package named sklearn but miss core modules.
            from sklearn.base import BaseEstimator  # noqa: F401

            version = getattr(sklearn, "__version__", None)
            module_file = getattr(sklearn, "__file__", None)
            if not version:
                return False, "sklearn imported but has no __version__ (incomplete install)"
            if module_file is None:
                return False, "sklearn imported as namespace package (incomplete install)"
            return True, f"version {version}"
        except ImportError as e:
            return False, f"not importable: {e}"
        except Exception as e:
            return False, f"imported but unusable: {e}"

    def saveSettings(self):
        """!@brief save settings if modifications."""
        if not hasattr(self, "settingsdock") or self.settingsdock is None:
            return
        # Change classifier
        if self.sender() == self.settingsdock.selectClassifier:
            selected_classifier = self.settingsdock.selectClassifier.currentText()
            classifier_code = classifier_config.get_classifier_code(selected_classifier)

            # Check required dependencies
            missing_required = []

            if classifier_config.requires_sklearn(classifier_code):
                sklearn_available, sklearn_details = self._check_sklearn_usable()
                if sklearn_available:
                    self.log.info(f"Scikit-learn detected for {selected_classifier}: {sklearn_details}")
                else:
                    self.log.warning(
                        f"Scikit-learn check failed for {selected_classifier}: {sklearn_details}"
                    )
                    missing_required.append("scikit-learn")

            if classifier_config.requires_xgboost(classifier_code):
                try:
                    import xgboost  # noqa: F401
                except ImportError:
                    missing_required.append("xgboost")

            if classifier_config.requires_lightgbm(classifier_code):
                try:
                    import lightgbm  # noqa: F401
                except ImportError:
                    missing_required.append("lightgbm")
            if classifier_config.requires_catboost(classifier_code):
                try:
                    import catboost  # noqa: F401
                except ImportError:
                    missing_required.append("catboost")

            # Check optional enhancements (for sklearn-based classifiers)
            missing_optional = []
            if classifier_config.requires_sklearn(classifier_code):
                try:
                    import optuna  # noqa: F401
                except ImportError:
                    missing_optional.append("optuna")

            if missing_required:
                req_list = ", ".join(missing_required)
                optional_line = ""
                if missing_optional:
                    opt_list = ", ".join(missing_optional)
                    optional_line = f"Optional missing now: <code>{opt_list}</code><br>"
                reply = QMessageBox.question(
                    self,
                    "Dependencies Missing for dzetsaka",
                    (
                        "To fully use dzetsaka capabilities, we recommend installing all dependencies.<br><br>"
                        f"Required missing now: <code>{req_list}</code><br>"
                        f"{optional_line}<br>"
                        "Install the full dzetsaka dependency bundle now?"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Install full stack when requested (all-or-nothing mode).
                    to_install = [
                        "scikit-learn",
                        "xgboost",
                        "lightgbm",
                        "catboost",
                        "optuna",
                        "shap",
                        "imbalanced-learn",
                    ]
                    if self._try_install_dependencies(to_install):
                        # Success! Update classifier
                        self.settings.setValue("/dzetsaka/classifier", selected_classifier)
                        self.classifier = selected_classifier
                        QMessageBox.information(
                            self,
                            "Installation Successful",
                            f"Dependencies installed successfully!<br><br>"
                            "<b>Important:</b> Please restart QGIS now.<br>"
                            "Without restarting, newly installed libraries may not be loaded, "
                            f"and {selected_classifier} training/classification can fail.<br><br>"
                            f"You can now try using {selected_classifier}.",
                            QMessageBox.StandardButton.Ok,
                        )
                    else:
                        # Failed, reset to GMM
                        self.settingsdock.selectClassifier.setCurrentIndex(0)
                        self.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
                        self.classifier = "Gaussian Mixture Model"
                elif reply == QMessageBox.StandardButton.No:
                    # Reset to GMM
                    self.settingsdock.selectClassifier.setCurrentIndex(0)
                    self.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
                    self.classifier = "Gaussian Mixture Model"

            else:
                # All dependencies satisfied, update classifier
                if self.classifier != selected_classifier:
                    self.settings.setValue("/dzetsaka/classifier", selected_classifier)
                    self.classifier = selected_classifier

    def _get_debug_info(self):
        """Generate debug information for GitHub issue reporting."""
        from dzetsaka.presentation.qgis.debug_info import build_debug_info

        return build_debug_info(self)

    def _default_output_name(self, in_raster_path, classifier_code):
        """Build deterministic default output filename for temporary classifications."""
        from dzetsaka.presentation.qgis.output_naming import default_output_name

        return default_output_name(in_raster_path, classifier_code)

    def _show_github_issue_popup(self, error_title, error_type, error_message, context):
        """Show standardized compact issue popup."""
        from dzetsaka.presentation.qgis.issue_reporting import show_standard_issue_popup

        show_standard_issue_popup(
            self,
            error_title=error_title,
            error_type=error_type,
            error_message=error_message,
            context=context,
        )

    def _try_install_dependencies(self, missing_deps):
        """Compatibility wrapper around extracted dependency installer."""
        from dzetsaka.presentation.qgis.dependency_installer import try_install_dependencies

        return try_install_dependencies(self, missing_deps)

    def run_wizard(self):
        """Open the dockable classification dashboard (Quick/Advanced)."""
        if self.wizarddock is None:
            self.wizarddock = ui.ClassificationDashboardDock(self.iface.mainWindow(), installer=self)
            self.wizarddock.classificationRequested.connect(self.execute_wizard_config)
            self.wizarddock.closingPlugin.connect(self.onCloseWizardDock)
            self.iface.addDockWidget(_LEFT_DOCK_AREA, self.wizarddock)

        self.wizarddock.show()
        self.wizarddock.raise_()

    def onCloseWizardDock(self):
        """Track dashboard dock closing state."""
        if self.wizarddock is not None:
            self.wizarddock.hide()

    def _validate_classification_request(
        self,
        *,
        raster_path,
        do_training,
        vector_path=None,
        class_field=None,
        model_path=None,
        source_label="Classification",
    ):
        # type: (...) -> bool
        """Validate required inputs before launching a classification task."""
        from dzetsaka.presentation.qgis.input_validation import validate_classification_request

        return validate_classification_request(
            self,
            raster_path=raster_path,
            do_training=do_training,
            vector_path=vector_path,
            class_field=class_field,
            model_path=model_path,
            source_label=source_label,
        )

    def _is_module_importable(self, module_name):
        # type: (str) -> bool
        """Return True if a module can be imported in current runtime."""
        from dzetsaka.presentation.qgis.runtime_utils import is_module_importable

        return is_module_importable(module_name)

    def _missing_classifier_dependencies(self, classifier_code):
        # type: (str) -> list[str]
        """Return missing runtime dependencies for a classifier code."""
        from dzetsaka.presentation.qgis.classifier_runtime import missing_classifier_dependencies

        return missing_classifier_dependencies(self, classifier_code)

    def _ensure_classifier_runtime_ready(self, classifier_code, source_label="Classification", fallback_to_gmm=False):
        # type: (str, str, bool) -> bool
        """Validate runtime dependencies for selected classifier before launching task."""
        from dzetsaka.presentation.qgis.classifier_runtime import ensure_classifier_runtime_ready

        return ensure_classifier_runtime_ready(
            self,
            classifier_code=classifier_code,
            source_label=source_label,
            fallback_to_gmm=fallback_to_gmm,
        )

    def _start_classification_task(
        self,
        *,
        description,
        do_training,
        raster_path,
        vector_path,
        class_field,
        model_path,
        split_config,
        random_seed,
        matrix_path,
        classifier,
        output_path,
        mask_path,
        confidence_map,
        nodata,
        extra_params,
        error_context,
        success_prefix,
    ):
        # type: (...) -> None
        """Submit a background classification task to the QGIS task manager."""
        from dzetsaka.presentation.qgis.task_launcher import start_classification_task

        start_classification_task(
            self,
            description=description,
            do_training=do_training,
            raster_path=raster_path,
            vector_path=vector_path,
            class_field=class_field,
            model_path=model_path,
            split_config=split_config,
            random_seed=random_seed,
            matrix_path=matrix_path,
            classifier=classifier,
            output_path=output_path,
            mask_path=mask_path,
            confidence_map=confidence_map,
            nodata=nodata,
            extra_params=extra_params,
            error_context=error_context,
            success_prefix=success_prefix,
        )

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
        from dzetsaka.presentation.qgis.wizard_execution import execute_wizard_config

        execute_wizard_config(self, config)

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
        from dzetsaka.presentation.qgis.runtime_utils import write_plugin_config

        write_plugin_config(self.configFile, self.Config, section, option, value)
# Qt6 enum compatibility (QGIS 4 / PyQt6)
try:
    _LEFT_DOCK_AREA = Qt.LeftDockWidgetArea
except AttributeError:
    _LEFT_DOCK_AREA = Qt.DockWidgetArea.LeftDockWidgetArea


