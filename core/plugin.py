# -*- coding: utf-8 -*-
"""
Main Plugin Controller for dzetsaka

Refactored main plugin class with separated concerns.
This replaces the monolithic functionality from dzetsaka.py.
"""

import os.path
import tempfile

from PyQt5.QtCore import QSettings, QCoreApplication, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox, QDialog, QFileDialog, QApplication
from qgis.core import QgsMessageLog, QgsApplication

# Import resources for icons
try:
    from .. import resources
    # Force resource loading
    import os
    if hasattr(resources, 'qInitResources'):
        resources.qInitResources()
except ImportError:
    pass

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr

from .. import ui
from .. import classifier_config
from ..scripts import mainfunction

from .settings_manager import DzetsakaSettings
from .ui_controller import DzetsakaUIController
from .file_manager import DzetsakaFileManager
from ..ml.learner import ModelLearner
from ..ml.classifier import ImageClassifier


class DzetsakaPlugin:
    """Main dzetsaka plugin class with separated concerns"""
    
    def __init__(self, iface):
        """Initialize plugin
        
        Parameters
        ----------
        iface : QgsInterface
            A reference to the QgsInterface
        """
        self.iface = iface
        
        # Initialize locale (existing code preserved)
        self.plugin_dir = os.path.dirname(__file__)
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n', 
            'dzetsaka_{}.qm'.format(locale)
        )
        
        # Initialize managers
        self.settings_manager = DzetsakaSettings()
        self.ui_controller = DzetsakaUIController(iface, self.settings_manager)
        self.file_manager = DzetsakaFileManager(self.settings_manager)
        
        # Plugin state
        self.actions = []
        self.menu = self.tr("&dzetsaka")
        self.plugin_name = "dzetsaka"
        
        # Icons and actions
        self.dock_icon = None
        self.settings_icon = None
    
    def tr(self, message):
        """Get translation for a string using Qt translation API"""
        return QCoreApplication.translate("dzetsaka", message)
    
    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI"""
        self._create_actions()
        self._setup_menu()
        
        # Load settings on startup
        self.settings_manager.load_settings()
        
        # Check for first installation or anniversary
        if self.settings_manager.first_installation:
            self.ui_controller.show_welcome_widget()
        else:
            self.ui_controller.show_anniversary_popup()
    
    def _create_actions(self):
        """Create all plugin actions"""
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'img', 'icon.png')
        
        # Try Qt resource first, fallback to file path
        resource_icon = QIcon(":/plugins/dzetsaka/img/icon.png")
        if not resource_icon.isNull():
            icon = resource_icon
        else:
            icon = QIcon(icon_path)
        
        # Main dock action
        self.dock_icon = QAction(
            icon,
            "dzetsaka",
            self.iface.mainWindow()
        )
        self.dock_icon.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.dock_icon)
        self.actions.append(self.dock_icon)
        
        # Settings action
        settings_icon_path = os.path.join(os.path.dirname(__file__), '..', 'img', 'dzetsaka_settings.png')
        
        # Try Qt resource first, fallback to file path
        resource_settings_icon = QIcon(":/plugins/dzetsaka/img/dzetsaka_settings.png")
        if not resource_settings_icon.isNull():
            settings_icon = resource_settings_icon
        else:
            settings_icon = QIcon(settings_icon_path)
        
        self.settings_icon = QAction(
            settings_icon,
            "Settings",
            self.iface.mainWindow()
        )
        self.settings_icon.triggered.connect(self.show_settings)
        self.iface.addToolBarIcon(self.settings_icon)
        self.actions.append(self.settings_icon)
    
    def _setup_menu(self):
        """Setup plugin menu"""
        for action in self.actions:
            self.iface.addPluginToMenu(self.menu, action)
    
    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI"""
        for action in self.actions:
            self.iface.removePluginMenu(self.plugin_name, action)
            self.iface.removeToolBarIcon(action)
        
        # Clean up UI
        self.ui_controller.cleanup()
    
    def run(self):
        """Run method that loads and starts the plugin"""
        if not self.ui_controller.plugin_is_active or self.ui_controller.dockwidget is None:
            # Initialize and show dockwidget
            dockwidget = self.ui_controller.initialize_dockwidget()
            
            # Connect main actions
            self.ui_controller.connect_main_actions(
                self.run_classification,
                self.show_settings
            )
            
            # Connect closing signal
            self.ui_controller.connect_closing_signal(self.on_close_plugin)
            
            # Show the dockwidget
            self.ui_controller.show_dockwidget()
    
    def on_close_plugin(self):
        """Cleanup when plugin dockwidget is closed"""
        self.ui_controller.cleanup()
    
    def show_settings(self):
        """Show settings dialog"""
        available_classifiers = list(classifier_config.CODE_TO_NAME.items())
        available_providers = ["gdal", "ogr"]  # Could be made configurable
        
        self.ui_controller.show_settings_dock(available_classifiers, available_providers)
        
        # Connect settings change handlers
        if self.ui_controller.settings_dock:
            self._connect_settings_handlers()
    
    def _connect_settings_handlers(self):
        """Connect settings change event handlers"""
        settings_dock = self.ui_controller.settings_dock
        
        # Classifier selection
        settings_dock.selectClassifier.currentIndexChanged[int].connect(
            self._on_classifier_changed
        )
        
        # Text field changes
        settings_dock.classSuffix.textChanged.connect(self._on_class_suffix_changed)
        settings_dock.classPrefix.textChanged.connect(self._on_class_prefix_changed)
        settings_dock.maskSuffix.textChanged.connect(self._on_mask_suffix_changed)
        
        # Provider selection
        settings_dock.selectProviders.currentIndexChanged[int].connect(
            self._on_provider_changed
        )
    
    def _on_classifier_changed(self, index):
        """Handle classifier selection change"""
        if self.ui_controller.settings_dock:
            selected_classifier = self.ui_controller.settings_dock.selectClassifier.currentText()
            self.settings_manager.save_classifier(selected_classifier)
    
    def _on_class_suffix_changed(self):
        """Handle class suffix change"""
        if self.ui_controller.settings_dock:
            suffix = self.ui_controller.settings_dock.classSuffix.text()
            self.settings_manager.save_class_suffix(suffix)
    
    def _on_class_prefix_changed(self):
        """Handle class prefix change"""
        if self.ui_controller.settings_dock:
            prefix = self.ui_controller.settings_dock.classPrefix.text()
            self.settings_manager.save_class_prefix(prefix)
    
    def _on_mask_suffix_changed(self):
        """Handle mask suffix change"""
        if self.ui_controller.settings_dock:
            suffix = self.ui_controller.settings_dock.maskSuffix.text()
            self.settings_manager.save_mask_suffix(suffix)
    
    def _on_provider_changed(self, index):
        """Handle provider selection change"""
        if self.ui_controller.settings_dock:
            provider = self.ui_controller.settings_dock.selectProviders.currentText()
            self.settings_manager.save_provider_type(provider)
    
    def run_classification(self):
        """Run the main classification workflow"""
        try:
            # Get input parameters from UI
            params = self._get_classification_parameters()
            
            # Debug: Show parameters
            QMessageBox.information(
                self.iface.mainWindow(),
                "Debug - Parameters",
                f"Raster: {params.get('raster_path', 'None')}\nVector: {params.get('vector_path', 'None')}\nOutput: {params.get('classification_output', 'None')}"
            )
            
            if not self._validate_classification_parameters(params):
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Debug - Validation",
                    "Parameter validation failed!"
                )
                return
            
            # Determine workflow type (training, classification, or both)
            if params['model_output']:
                # Training workflow
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Debug",
                    "Running training workflow..."
                )
                self._run_training(params)
            
            if params['classification_output']:
                # Classification workflow  
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Debug",
                    "Running classification workflow..."
                )
                self._run_classification_workflow(params)
            
            QMessageBox.information(
                self.iface.mainWindow(),
                "Debug",
                "Classification workflow completed!"
            )
                
        except Exception as e:
            QMessageBox.critical(
                self.ui_controller.dockwidget,
                "Classification Error",
                f"An error occurred during classification: {str(e)}"
            )
    
    def _get_classification_parameters(self):
        """Extract parameters from UI"""
        dw = self.ui_controller.dockwidget
        
        # Helper function to get file path from QgsMapLayerComboBox
        def get_layer_path(combo_box, is_vector=False):
            layer = combo_box.currentLayer()
            if layer:
                path = layer.dataProvider().dataSourceUri()
                if is_vector and "|" in path:
                    # Remove layer ID for vector layers
                    path = path.split("|")[0]
                return path
            return None
        
        # Helper function to get file path from QLineEdit or QgsFileWidget
        def get_file_path(file_widget):
            if hasattr(file_widget, 'filePath'):
                # QgsFileWidget
                path = file_widget.filePath()
                return path if path else None
            elif hasattr(file_widget, 'text'):
                # QLineEdit
                path = file_widget.text()
                return path if path else None
            return None
        
        return {
            'raster_path': get_layer_path(dw.inRaster),
            'vector_path': get_layer_path(dw.inShape, is_vector=True),
            'class_field': dw.inField.currentText(),
            'model_input': get_file_path(dw.inModel) if dw.checkInModel.isChecked() else None,
            'model_output': get_file_path(dw.outModel) if dw.checkOutModel.isChecked() else None,
            'classification_output': get_file_path(dw.outRaster),
            'mask_path': get_layer_path(dw.inMask) if dw.checkInMask.isChecked() else None,
            'confidence_map': get_file_path(dw.outConfidenceMap) if dw.checkInConfidence.isChecked() else None,
            'matrix_output': get_file_path(dw.outMatrix) if dw.checkOutMatrix.isChecked() else None,
            'classifier': self.settings_manager.classifier
        }
    
    def _validate_classification_parameters(self, params):
        """Validate classification parameters"""
        if not params['raster_path']:
            QMessageBox.warning(
                self.ui_controller.dockwidget,
                "Missing Input",
                "Please select an input raster."
            )
            return False
        
        if params['model_output'] and not params['vector_path']:
            QMessageBox.warning(
                self.ui_controller.dockwidget,
                "Missing Input", 
                "Please select training data (vector) for model training."
            )
            return False
        
        if params['classification_output'] and not (params['model_input'] or params['model_output']):
            QMessageBox.warning(
                self.ui_controller.dockwidget,
                "Missing Model",
                "Please provide a model (input existing or output new) for classification."
            )
            return False
        
        return True
    
    def _run_training(self, params):
        """Run model training workflow"""
        learner = ModelLearner(
            raster_path=params['raster_path'],
            vector_path=params['vector_path'],
            class_field=params['class_field'],
            model_path=params['model_output'],
            matrix_path=params['matrix_output'],
            classifier=params['classifier']
        )
        
        success = learner.train_model()
        
        if success:
            QMessageBox.information(
                self.ui_controller.dockwidget,
                "Training Complete",
                f"Model training completed successfully!\nModel saved to: {params['model_output']}"
            )
    
    def _run_classification_workflow(self, params):
        """Run image classification workflow"""
        model_path = params['model_input'] or params['model_output']
        
        classifier = ImageClassifier(
            raster_path=params['raster_path'],
            model_path=model_path,
            output_path=params['classification_output'],
            mask_path=params['mask_path'],
            confidence_map=params['confidence_map']
        )
        
        success = classifier.classify()
        
        if success:
            QMessageBox.information(
                self.ui_controller.dockwidget,
                "Classification Complete", 
                f"Image classification completed successfully!\nOutput saved to: {params['classification_output']}"
            )