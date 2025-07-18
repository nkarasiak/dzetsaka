# -*- coding: utf-8 -*-
"""
/***************************************************************************
 dzetsakaGUI
                                 A QGIS plugin
 Machine Learning classification for remote sensing
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

# Import resources for icons FIRST
try:
    from . import resources
except ImportError:
    pass

# Refactored plugin using modular architecture
# Original monolithic code has been split into core/ and ml/ modules

from .core.plugin import DzetsakaPlugin
from .dzetsaka_provider import dzetsakaProvider


class dzetsakaGUI:
    """QGIS Plugin Implementation using modular architecture."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Initialize the new modular plugin architecture
        self.plugin = DzetsakaPlugin(iface)
        
        # Initialize provider for backward compatibility
        provider_type = self.plugin.settings_manager.provider_type or "gdal"
        self.provider = dzetsakaProvider(provider_type)

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        # Register processing provider
        from qgis.core import QgsApplication
        QgsApplication.processingRegistry().addProvider(self.provider)
        
        self.plugin.initGui()

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        # Remove processing provider
        from qgis.core import QgsApplication
        QgsApplication.processingRegistry().removeProvider(self.provider)
        
        self.plugin.unload()

    # Backward compatibility methods that delegate to the modular plugin

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return self.plugin.tr(message)

    def run(self):
        """Run method that loads and starts the plugin."""
        self.plugin.run()

    @property
    def actions(self):
        """Get plugin actions for backward compatibility."""
        return self.plugin.actions

    @property 
    def menu(self):
        """Get plugin menu for backward compatibility."""
        return self.plugin.menu

    @property
    def iface(self):
        """Get QGIS interface for backward compatibility."""
        return self.plugin.iface

    @property
    def settings_manager(self):
        """Access to settings manager."""
        return self.plugin.settings_manager

    @property
    def ui_controller(self):
        """Access to UI controller."""
        return self.plugin.ui_controller

    @property
    def file_manager(self):
        """Access to file manager."""
        return self.plugin.file_manager

    # Legacy method support for backward compatibility
    def rememberLastSaveDir(self, fileName):
        """Remember last save directory when saving or loading file."""
        self.plugin.settings_manager.remember_last_save_dir(fileName)

    def showWelcomeWidget(self):
        """Show welcome widget."""
        self.plugin.ui_controller.show_welcome_widget()

    def checkAnniversary(self):
        """Check if anniversary popup should be shown."""
        self.plugin.ui_controller.show_anniversary_popup()

    def loadSettings(self):
        """Load and show settings dialog."""
        self.plugin.show_settings()

    def saveSettings(self):
        """Save settings (handled automatically by settings manager)."""
        # Settings are saved automatically when changed
        pass

    def runMagic(self):
        """Run the main classification workflow."""
        self.plugin.run_classification()