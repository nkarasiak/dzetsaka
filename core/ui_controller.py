# -*- coding: utf-8 -*-
"""
UI Controller for dzetsaka plugin

Manages all UI interactions and widget setup.
Extracted from the monolithic dzetsaka.py for better separation of concerns.
"""

from PyQt5.QtCore import Qt, QObject
from PyQt5.QtWidgets import QFileDialog
from qgis.core import QgsMessageLog, QgsProviderRegistry

from .. import ui


class DzetsakaUIController(QObject):
    """Manages all UI-related functionality for dzetsaka plugin"""
    
    def __init__(self, iface, settings_manager):
        super().__init__()
        self.iface = iface
        self.settings_manager = settings_manager
        self.dockwidget = None
        self.settings_dock = None
        self.welcome_widget = None
        self.plugin_is_active = False
    
    def initialize_dockwidget(self):
        """Initialize the main dockwidget with all connections"""
        if self.dockwidget is None:
            # Create the dockwidget and keep reference
            self.dockwidget = ui.dzetsakaDockWidget()
        
        # Setup provider restrictions
        self._setup_provider_restrictions()
        
        # Setup file selection widgets
        self._setup_file_widgets()
        
        # Setup field selection
        self._setup_field_selection()
        
        # Setup checkbox connections
        self._setup_checkbox_connections()
        
        return self.dockwidget
    
    def _setup_provider_restrictions(self):
        """Setup provider restrictions for raster and vector inputs"""
        # Restrict raster providers to GDAL only
        except_raster = QgsProviderRegistry.instance().providerList()
        except_raster.remove("gdal")
        self.dockwidget.inRaster.setExcludedProviders(except_raster)
        
        # Restrict vector providers to OGR only
        except_vector = QgsProviderRegistry.instance().providerList()
        except_vector.remove("ogr")
        self.dockwidget.inShape.setExcludedProviders(except_vector)
    
    def _setup_file_widgets(self):
        """Setup file input/output widgets"""
        self.dockwidget.outRaster.clear()
        self.dockwidget.outModel.clear()
        self.dockwidget.inModel.clear()
        self.dockwidget.inMask.clear()
        self.dockwidget.outMatrix.clear()
        self.dockwidget.outConfidenceMap.clear()
    
    def _setup_field_selection(self):
        """Setup field selection widget"""
        self.dockwidget.inField.clear()
        
        def on_changed_layer():
            """Update columns when vector layer changes"""
            self.dockwidget.inField.clear()
            
            if (self.dockwidget.inField.currentText() == "" and 
                self.dockwidget.inShape.currentLayer() and 
                self.dockwidget.inShape.currentLayer() != "NoneType"):
                try:
                    active_layer = self.dockwidget.inShape.currentLayer()
                    provider = active_layer.dataProvider()
                    fields = provider.fields()
                    list_field_names = [field.name() for field in fields]
                    self.dockwidget.inField.addItems(list_field_names)
                except Exception:
                    QgsMessageLog.logMessage(
                        "dzetsaka cannot change active layer. Maybe you opened an OSM/Online background?"
                    )
        
        on_changed_layer()
        self.dockwidget.inShape.currentIndexChanged[int].connect(on_changed_layer)
    
    def _setup_checkbox_connections(self):
        """Setup checkbox state change connections"""
        self.dockwidget.checkOutModel.clicked.connect(self.checkbox_state_changed)
        self.dockwidget.checkInModel.clicked.connect(self.checkbox_state_changed)
        self.dockwidget.checkInMask.clicked.connect(self.checkbox_state_changed)
        self.dockwidget.checkOutMatrix.clicked.connect(self.checkbox_state_changed)
        self.dockwidget.checkInConfidence.clicked.connect(self.checkbox_state_changed)
    
    def connect_main_actions(self, run_magic_callback, settings_callback):
        """Connect main action buttons to their callbacks"""
        if self.dockwidget:
            self.dockwidget.outRasterButton.clicked.connect(self.select_output_file)
            self.dockwidget.settingsButton.clicked.connect(settings_callback)
            self.dockwidget.performMagic.clicked.connect(run_magic_callback)
            self.dockwidget.mGroupBox.collapsedStateChanged.connect(self.resize_dock)
    
    def show_dockwidget(self):
        """Show the main dockwidget"""
        if self.dockwidget:
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
            self.plugin_is_active = True
    
    def close_dockwidget(self):
        """Close the dockwidget and cleanup"""
        if self.dockwidget:
            self.dockwidget.close()
            self.plugin_is_active = False
    
    def connect_closing_signal(self, close_callback):
        """Connect the dockwidget closing signal"""
        if self.dockwidget:
            self.dockwidget.closingPlugin.connect(close_callback)
    
    def resize_dock(self):
        """Resize dockwidget based on groupbox state"""
        if self.dockwidget and self.dockwidget.mGroupBox.isCollapsed():
            self.dockwidget.mGroupBox.setFixedHeight(20)
            self.dockwidget.setFixedHeight(350)
        elif self.dockwidget:
            self.dockwidget.setMinimumHeight(470)
            self.dockwidget.mGroupBox.setMinimumHeight(160)
    
    def select_output_file(self):
        """Select output file with proper extension handling"""
        if not self.dockwidget:
            return
        
        file_name, _filter = QFileDialog.getSaveFileName(
            self.dockwidget, 
            "Select output file", 
            self.settings_manager.last_save_dir, 
            "TIF (*.tif)"
        )
        
        if file_name:
            if not file_name.endswith('.tif'):
                file_name += '.tif'
            
            self.dockwidget.outRaster.setText(file_name)
            self.settings_manager.remember_last_save_dir(file_name)
    
    def checkbox_state_changed(self):
        """Handle checkbox state changes for optional parameters"""
        from PyQt5.QtWidgets import QFileDialog, QApplication
        
        sender = self.sender()
        
        # If load model
        if (
            sender == self.dockwidget.checkInModel
            and self.dockwidget.checkInModel.isChecked()
        ):
            fileName, _filter = QFileDialog.getOpenFileName(
                self.dockwidget, "Select your file", self.settings_manager.last_save_dir
            )
            self.settings_manager.remember_last_save_dir(fileName)
            if fileName != "":
                self.dockwidget.inModel.setText(fileName)
                self.dockwidget.inModel.setEnabled(True)
                # Disable training, so disable vector choice
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
        if (
            sender == self.dockwidget.checkOutModel
            and self.dockwidget.checkOutModel.isChecked()
        ):
            fileName, _filter = QFileDialog.getSaveFileName(
                self.dockwidget, "Select output file", self.settings_manager.last_save_dir
            )
            self.settings_manager.remember_last_save_dir(fileName)
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
        if (
            sender == self.dockwidget.checkInMask
            and self.dockwidget.checkInMask.isChecked()
        ):
            fileName, _filter = QFileDialog.getOpenFileName(
                self.dockwidget,
                "Select your mask raster",
                self.settings_manager.last_save_dir,
                "TIF (*.tif)",
            )
            self.settings_manager.remember_last_save_dir(fileName)
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
        if (
            sender == self.dockwidget.checkOutMatrix
            and self.dockwidget.checkOutMatrix.isChecked()
        ):
            fileName, _filter = QFileDialog.getSaveFileName(
                self.dockwidget, "Select output file", self.settings_manager.last_save_dir
            )
            self.settings_manager.remember_last_save_dir(fileName)
            if fileName != "":
                import os
                fileName_base, fileExtension = os.path.splitext(fileName)
                fileName = fileName_base + ".csv"
                self.dockwidget.outMatrix.setText(fileName)
                self.dockwidget.outMatrix.setEnabled(True)
                # Enable split control and set to 50%
                self.dockwidget.inSplit.setEnabled(True)
                self.dockwidget.inSplit.setValue(50)
            else:
                self.dockwidget.checkOutMatrix.setChecked(False)
                self.dockwidget.outMatrix.setEnabled(False)
                self.dockwidget.inSplit.setEnabled(False)

        elif sender == self.dockwidget.checkOutMatrix:
            self.dockwidget.outMatrix.clear()
            self.dockwidget.outMatrix.setEnabled(False)
            self.dockwidget.inSplit.setEnabled(False)

        # If confidence map
        if (
            sender == self.dockwidget.checkInConfidence
            and self.dockwidget.checkInConfidence.isChecked()
        ):
            fileName, _filter = QFileDialog.getSaveFileName(
                self.dockwidget, "Select output file", self.settings_manager.last_save_dir
            )
            self.settings_manager.remember_last_save_dir(fileName)
            if fileName != "":
                self.dockwidget.outConfidenceMap.setText(fileName)
                self.dockwidget.outConfidenceMap.setEnabled(True)
            else:
                self.dockwidget.checkInConfidence.setChecked(False)
                self.dockwidget.outConfidenceMap.setEnabled(False)

        elif sender == self.dockwidget.checkInConfidence:
            self.dockwidget.outConfidenceMap.clear()
            self.dockwidget.outConfidenceMap.setEnabled(False)
    
    def show_welcome_widget(self):
        """Show welcome widget for first-time users"""
        self.welcome_widget = ui.welcomeWidget()
        self.welcome_widget.show()
        self.settings_manager.save_first_installation(False)
    
    def show_settings_dock(self, available_classifiers, available_providers):
        """Show settings dock with current configuration"""
        self.settings_dock = ui.settings_dock()
        self.settings_dock.show()
        
        # Setup classifier selection
        for i, (code, classifier_name) in enumerate(available_classifiers):
            if code == self.settings_manager.classifier:
                self.settings_dock.selectClassifier.setCurrentIndex(i)
        
        # Setup text fields with current values
        self.settings_dock.classSuffix.setText(self.settings_manager.class_suffix)
        self.settings_dock.classPrefix.setText(self.settings_manager.class_prefix)
        self.settings_dock.maskSuffix.setText(self.settings_manager.mask_suffix)
        
        # Setup provider selection
        for i, provider in enumerate(available_providers):
            if provider == self.settings_manager.provider_type:
                self.settings_dock.selectProviders.setCurrentIndex(i)
    
    def show_anniversary_popup(self):
        """Show anniversary popup if applicable"""
        anniversary_manager = ui.AnniversaryManager()
        anniversary_manager.show_anniversary_popup(self.iface.mainWindow())
    
    def cleanup(self):
        """Cleanup UI resources"""
        if self.dockwidget:
            self.dockwidget = None
        if self.settings_dock:
            self.settings_dock = None
        if self.welcome_widget:
            self.welcome_widget = None
        self.plugin_is_active = False