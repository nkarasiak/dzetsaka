# -*- coding: utf-8 -*-
"""
Machine Learning Classifier Module

Handles image classification functionality extracted from scripts/mainfunction.py
for better separation of concerns and maintainability.
"""

import pickle
import os
import tempfile
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
from osgeo import gdal, ogr

try:
    from qgis.core import QgsMessageLog
    from ..scripts import function_dataraster as dataraster
    from ..scripts import progressBar as pB
except ImportError:
    import function_dataraster as dataraster

from .utils import backward_compatible


class ImageClassifier:
    """Handles image classification using trained models"""
    
    @backward_compatible(
        inRaster="raster_path",
        inModel="model_path", 
        outRaster="output_path",
        inMask="mask_path",
    )
    def __init__(
        self,
        raster_path: str = None,
        model_path: str = None,
        output_path: str = None,
        mask_path: Optional[str] = None,
        confidence_map: Optional[str] = None,
        confidence_map_per_class: Optional[str] = None,
        nodata_value: int = 0,
        feedback=None,
    ):
        """Initialize image classifier.
        
        Parameters
        ----------
        raster_path : str
            Path to input raster image
        model_path : str  
            Path to trained model file
        output_path : str
            Path for classification output
        mask_path : str, optional
            Path to mask raster
        confidence_map : str, optional
            Path for confidence map output
        confidence_map_per_class : str, optional
            Path for per-class confidence map output
        nodata_value : int, default=0
            NoData value for output
        feedback : optional
            Feedback object for progress reporting
        """
        self.raster_path = raster_path
        self.model_path = model_path
        self.output_path = output_path
        self.mask_path = mask_path
        self.confidence_map = confidence_map
        self.confidence_map_per_class = confidence_map_per_class
        self.nodata_value = nodata_value
        self.feedback = feedback
        
        # Will be set during classification
        self.model = None
        self.model_info = None
    
    def classify(self):
        """Perform image classification"""
        try:
            # Validate inputs
            self._validate_inputs()
            
            # Load model
            self._load_model()
            
            # Load and prepare raster data
            raster_data, raster_info = self._load_raster_data()
            
            # Apply mask if provided
            if self.mask_path:
                mask_data = self._load_mask()
                raster_data = self._apply_mask(raster_data, mask_data)
            
            # Perform classification
            predictions = self._predict(raster_data)
            
            # Generate confidence maps if requested
            if self.confidence_map or self.confidence_map_per_class:
                confidence = self._calculate_confidence(raster_data)
                
                if self.confidence_map:
                    self._save_confidence_map(confidence, raster_info)
                
                if self.confidence_map_per_class:
                    self._save_per_class_confidence(confidence, raster_info)
            
            # Save classification result
            self._save_classification(predictions, raster_info)
            
            return True
            
        except Exception as e:
            if self.feedback:
                self.feedback.reportError(f"Classification failed: {str(e)}")
            raise
    
    def _validate_inputs(self):
        """Validate input parameters"""
        if not self.raster_path:
            raise ValueError("raster_path is required")
        if not self.model_path:
            raise ValueError("model_path is required")
        if not self.output_path:
            raise ValueError("output_path is required")
        
        if not os.path.exists(self.raster_path):
            raise FileNotFoundError(f"Raster file not found: {self.raster_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def _load_model(self):
        """Load trained model from file"""
        if self.feedback:
            self.feedback.setProgressText("Loading model...")
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('classifier')
                self.model_info = {
                    'classifier_type': model_data.get('classifier_type'),
                    'stats': model_data.get('stats'),
                    'parameters': model_data.get('parameters')
                }
            else:
                # Legacy model format
                self.model = model_data
                self.model_info = {'classifier_type': 'unknown'}
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _load_raster_data(self):
        """Load raster data for classification"""
        if self.feedback:
            self.feedback.setProgressText("Loading raster data...")
        
        # Open raster dataset
        dataset = gdal.Open(self.raster_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Cannot open raster: {self.raster_path}")
        
        # Get raster information
        raster_info = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'bands': dataset.RasterCount,
            'projection': dataset.GetProjection(),
            'geotransform': dataset.GetGeoTransform(),
            'datatype': dataset.GetRasterBand(1).DataType
        }
        
        # Read raster data
        raster_data = dataset.ReadAsArray()
        
        # Reshape for prediction (samples, features)
        if len(raster_data.shape) == 3:
            # Multi-band raster
            bands, height, width = raster_data.shape
            raster_data = raster_data.reshape(bands, height * width).T
        else:
            # Single band raster
            height, width = raster_data.shape
            raster_data = raster_data.reshape(height * width, 1)
        
        dataset = None  # Close dataset
        
        return raster_data, raster_info
    
    def _load_mask(self):
        """Load mask data"""
        if self.feedback:
            self.feedback.setProgressText("Loading mask...")
        
        dataset = gdal.Open(self.mask_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Cannot open mask: {self.mask_path}")
        
        mask_data = dataset.ReadAsArray()
        dataset = None
        
        return mask_data.flatten()
    
    def _apply_mask(self, raster_data, mask_data):
        """Apply mask to raster data"""
        if len(mask_data) != raster_data.shape[0]:
            raise ValueError("Mask and raster dimensions do not match")
        
        # Create boolean mask (assuming 0 = masked, non-zero = valid)
        valid_mask = mask_data != 0
        
        return raster_data[valid_mask]
    
    def _predict(self, raster_data):
        """Perform prediction on raster data"""
        if self.feedback:
            self.feedback.setProgressText("Classifying image...")
        
        # Handle large datasets by processing in chunks
        chunk_size = 100000  # Adjust based on memory constraints
        predictions = []
        
        for i in range(0, len(raster_data), chunk_size):
            chunk = raster_data[i:i + chunk_size]
            chunk_pred = self.model.predict(chunk)
            predictions.extend(chunk_pred)
        
        return np.array(predictions)
    
    def _calculate_confidence(self, raster_data):
        """Calculate prediction confidence"""
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        if self.feedback:
            self.feedback.setProgressText("Calculating confidence...")
        
        # Process in chunks like prediction
        chunk_size = 100000
        confidence = []
        
        for i in range(0, len(raster_data), chunk_size):
            chunk = raster_data[i:i + chunk_size]
            chunk_conf = self.model.predict_proba(chunk)
            confidence.extend(chunk_conf)
        
        return np.array(confidence)
    
    def _save_classification(self, predictions, raster_info):
        """Save classification results to raster file"""
        if self.feedback:
            self.feedback.setProgressText("Saving classification...")
        
        # Reshape predictions back to image dimensions
        height = raster_info['height']
        width = raster_info['width']
        
        if self.mask_path:
            # Need to handle masked data differently
            full_predictions = np.full(height * width, self.nodata_value, dtype=np.int16)
            # This would need mask handling logic
        else:
            predictions = predictions.reshape(height, width)
        
        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            self.output_path,
            width, height, 1,
            gdal.GDT_Int16
        )
        
        # Set projection and geotransform
        output_dataset.SetProjection(raster_info['projection'])
        output_dataset.SetGeoTransform(raster_info['geotransform'])
        
        # Write data
        band = output_dataset.GetRasterBand(1)
        band.WriteArray(predictions)
        band.SetNoDataValue(self.nodata_value)
        
        output_dataset = None  # Close dataset
    
    def _save_confidence_map(self, confidence, raster_info):
        """Save overall confidence map"""
        if confidence is None:
            return
        
        if self.feedback:
            self.feedback.setProgressText("Saving confidence map...")
        
        # Calculate max confidence per pixel
        max_confidence = np.max(confidence, axis=1)
        
        # Reshape and save
        height = raster_info['height']
        width = raster_info['width']
        max_confidence = max_confidence.reshape(height, width)
        
        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            self.confidence_map,
            width, height, 1,
            gdal.GDT_Float32
        )
        
        # Set projection and geotransform
        output_dataset.SetProjection(raster_info['projection'])
        output_dataset.SetGeoTransform(raster_info['geotransform'])
        
        # Write data
        band = output_dataset.GetRasterBand(1)
        band.WriteArray(max_confidence)
        
        output_dataset = None
    
    def _save_per_class_confidence(self, confidence, raster_info):
        """Save per-class confidence maps"""
        if confidence is None:
            return
        
        if self.feedback:
            self.feedback.setProgressText("Saving per-class confidence maps...")
        
        # Get number of classes
        n_classes = confidence.shape[1]
        
        # Reshape confidence data
        height = raster_info['height']
        width = raster_info['width']
        
        # Create multi-band output raster
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            self.confidence_map_per_class,
            width, height, n_classes,
            gdal.GDT_Float32
        )
        
        # Set projection and geotransform
        output_dataset.SetProjection(raster_info['projection'])
        output_dataset.SetGeoTransform(raster_info['geotransform'])
        
        # Write each class confidence as a separate band
        for class_idx in range(n_classes):
            class_confidence = confidence[:, class_idx].reshape(height, width)
            band = output_dataset.GetRasterBand(class_idx + 1)
            band.WriteArray(class_confidence)
            band.SetDescription(f"Class_{class_idx + 1}_confidence")
        
        output_dataset = None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return self.model_info