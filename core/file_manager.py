# -*- coding: utf-8 -*-
"""
File Manager for dzetsaka plugin

Handles all file I/O operations, path validation, and file format checking.
Extracted from the monolithic dzetsaka.py for better separation of concerns.
"""

import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr


class DzetsakaFileManager:
    """Manages all file I/O operations for dzetsaka plugin"""
    
    def __init__(self, settings_manager):
        self.settings_manager = settings_manager
        self.supported_raster_formats = ['.tif', '.tiff', '.img', '.bil']
        self.supported_vector_formats = ['.shp', '.gpkg', '.geojson']
    
    def select_output_raster_file(self, parent_widget, title="Select output file"):
        """Select output raster file with proper extension handling"""
        file_name, _filter = QFileDialog.getSaveFileName(
            parent_widget,
            title,
            self.settings_manager.last_save_dir,
            "TIF (*.tif);;IMG (*.img);;BIL (*.bil)"
        )
        
        if file_name:
            file_name = self._ensure_raster_extension(file_name)
            self.settings_manager.remember_last_save_dir(file_name)
            return file_name
        
        return None
    
    def select_input_raster_file(self, parent_widget, title="Select input raster"):
        """Select input raster file"""
        file_name, _filter = QFileDialog.getOpenFileName(
            parent_widget,
            title,
            self.settings_manager.last_save_dir,
            "Raster files (*.tif *.tiff *.img *.bil);;All files (*.*)"
        )
        
        if file_name:
            self.settings_manager.remember_last_save_dir(file_name)
            return file_name
        
        return None
    
    def select_input_vector_file(self, parent_widget, title="Select input vector"):
        """Select input vector file"""
        file_name, _filter = QFileDialog.getOpenFileName(
            parent_widget,
            title,
            self.settings_manager.last_save_dir,
            "Vector files (*.shp *.gpkg *.geojson);;All files (*.*)"
        )
        
        if file_name:
            self.settings_manager.remember_last_save_dir(file_name)
            return file_name
        
        return None
    
    def select_model_file(self, parent_widget, mode='open', title=None):
        """Select model file for save or load"""
        if mode == 'save':
            title = title or "Save model file"
            file_name, _filter = QFileDialog.getSaveFileName(
                parent_widget,
                title,
                self.settings_manager.last_save_dir,
                "Model files (*.model);;Pickle files (*.pkl);;All files (*.*)"
            )
        else:
            title = title or "Load model file"
            file_name, _filter = QFileDialog.getOpenFileName(
                parent_widget,
                title,
                self.settings_manager.last_save_dir,
                "Model files (*.model);;Pickle files (*.pkl);;All files (*.*)"
            )
        
        if file_name:
            self.settings_manager.remember_last_save_dir(file_name)
            return file_name
        
        return None
    
    def _ensure_raster_extension(self, file_name):
        """Ensure the file has a proper raster extension"""
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension not in self.supported_raster_formats:
            # Default to .tif if no proper extension
            file_name += '.tif'
        
        return file_name
    
    def validate_raster_file(self, file_path):
        """Validate that a raster file exists and is readable"""
        if not file_path or not os.path.exists(file_path):
            return False, "File does not exist"
        
        try:
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset is None:
                return False, "Cannot open raster file with GDAL"
            
            # Basic validation
            if dataset.RasterCount == 0:
                return False, "Raster file has no bands"
            
            dataset = None  # Close the dataset
            return True, "Valid raster file"
            
        except Exception as e:
            return False, f"Error validating raster: {str(e)}"
    
    def validate_vector_file(self, file_path):
        """Validate that a vector file exists and is readable"""
        if not file_path or not os.path.exists(file_path):
            return False, "File does not exist"
        
        try:
            driver = ogr.GetDriverByName("ESRI Shapefile")
            if file_path.endswith('.gpkg'):
                driver = ogr.GetDriverByName("GPKG")
            elif file_path.endswith('.geojson'):
                driver = ogr.GetDriverByName("GeoJSON")
            
            datasource = driver.Open(file_path, 0)  # 0 means read-only
            if datasource is None:
                return False, "Cannot open vector file with OGR"
            
            # Basic validation
            layer_count = datasource.GetLayerCount()
            if layer_count == 0:
                return False, "Vector file has no layers"
            
            datasource = None  # Close the datasource
            return True, "Valid vector file"
            
        except Exception as e:
            return False, f"Error validating vector: {str(e)}"
    
    def get_raster_info(self, file_path):
        """Get basic information about a raster file"""
        try:
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset is None:
                return None
            
            info = {
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'bands': dataset.RasterCount,
                'driver': dataset.GetDriver().ShortName,
                'projection': dataset.GetProjection(),
                'geotransform': dataset.GetGeoTransform()
            }
            
            dataset = None
            return info
            
        except Exception as e:
            return None
    
    def get_vector_info(self, file_path):
        """Get basic information about a vector file"""
        try:
            driver = ogr.GetDriverByName("ESRI Shapefile")
            if file_path.endswith('.gpkg'):
                driver = ogr.GetDriverByName("GPKG")
            elif file_path.endswith('.geojson'):
                driver = ogr.GetDriverByName("GeoJSON")
            
            datasource = driver.Open(file_path, 0)
            if datasource is None:
                return None
            
            layer = datasource.GetLayer()
            feature_count = layer.GetFeatureCount()
            
            # Get field information
            layer_defn = layer.GetLayerDefn()
            fields = []
            for i in range(layer_defn.GetFieldCount()):
                field_defn = layer_defn.GetFieldDefn(i)
                fields.append({
                    'name': field_defn.GetName(),
                    'type': field_defn.GetFieldTypeName(field_defn.GetType())
                })
            
            info = {
                'feature_count': feature_count,
                'fields': fields,
                'driver': driver.GetName()
            }
            
            datasource = None
            return info
            
        except Exception as e:
            return None
    
    def show_file_error(self, parent_widget, message, title="File Error"):
        """Show file-related error message"""
        QMessageBox.warning(parent_widget, title, message)