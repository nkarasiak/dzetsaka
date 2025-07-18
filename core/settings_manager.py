# -*- coding: utf-8 -*-
"""
Settings Manager for dzetsaka plugin

Centralized management of all plugin settings using QSettings.
Extracted from the monolithic dzetsaka.py for better maintainability.
"""

from PyQt5.QtCore import QSettings


class DzetsakaSettings:
    """Manages all dzetsaka plugin settings through QSettings"""
    
    def __init__(self):
        self.settings = QSettings()
        self._init_default_values()
    
    def _init_default_values(self):
        """Initialize default values for all settings"""
        self.classifier = "GMM"  # Default to GMM
        self.class_suffix = ""
        self.class_prefix = ""
        self.mask_suffix = ""
        self.provider_type = ""
        self.first_installation = True
        self.last_save_dir = ""
    
    def load_settings(self):
        """Load all settings from QSettings"""
        self.classifier = self.settings.value("/dzetsaka/classifier", "GMM", str)
        if self.classifier == "":
            self.classifier = "GMM"
            self.settings.setValue("/dzetsaka/classifier", self.classifier)
        
        # Convert any old full names to codes for backward compatibility
        from .. import classifier_config
        if self.classifier in classifier_config.NAME_TO_CODE:
            self.classifier = classifier_config.NAME_TO_CODE[self.classifier]
            self.settings.setValue("/dzetsaka/classifier", self.classifier)
        
        self.class_suffix = self.settings.value("/dzetsaka/classSuffix", "", str)
        if self.class_suffix == "":
            self.settings.setValue("/dzetsaka/classSuffix", self.class_suffix)
        
        self.class_prefix = self.settings.value("/dzetsaka/classPrefix", "", str)
        if self.class_prefix == "":
            self.settings.setValue("/dzetsaka/classPrefix", self.class_prefix)
        
        self.mask_suffix = self.settings.value("/dzetsaka/maskSuffix", "", str)
        if self.mask_suffix == "":
            self.settings.setValue("/dzetsaka/maskSuffix", self.mask_suffix)
        
        self.provider_type = self.settings.value("/dzetsaka/providerType", "", str)
        if self.provider_type == "":
            self.provider_type = self.settings.setValue("/dzetsaka/providerType", "gdal")
        
        self.first_installation = self.settings.value(
            "/dzetsaka/firstInstallation", True, bool
        )
        if self.first_installation is None:
            self.settings.setValue("/dzetsaka/firstInstallation", True)
        
        self.last_save_dir = self.settings.value("/dzetsaka/lastSaveDir", "", str)
    
    def save_classifier(self, classifier):
        """Save classifier setting"""
        self.classifier = classifier
        self.settings.setValue("/dzetsaka/classifier", classifier)
    
    def save_class_suffix(self, suffix):
        """Save class suffix setting"""
        self.class_suffix = suffix
        self.settings.setValue("/dzetsaka/classSuffix", suffix)
    
    def save_class_prefix(self, prefix):
        """Save class prefix setting"""
        self.class_prefix = prefix
        self.settings.setValue("/dzetsaka/classPrefix", prefix)
    
    def save_mask_suffix(self, suffix):
        """Save mask suffix setting"""
        self.mask_suffix = suffix
        self.settings.setValue("/dzetsaka/maskSuffix", suffix)
    
    def save_provider_type(self, provider_type):
        """Save provider type setting"""
        self.provider_type = provider_type
        self.settings.setValue("/dzetsaka/providerType", provider_type)
    
    def save_first_installation(self, is_first):
        """Save first installation flag"""
        self.first_installation = is_first
        self.settings.setValue("/dzetsaka/firstInstallation", is_first)
    
    def remember_last_save_dir(self, file_name):
        """Remember last save directory when saving or loading file"""
        if file_name != "":
            self.last_save_dir = file_name
            self.settings.setValue("/dzetsaka/lastSaveDir", self.last_save_dir)
    
    def get_locale(self):
        """Get user locale setting"""
        return self.settings.value('locale/userLocale')[0:2]