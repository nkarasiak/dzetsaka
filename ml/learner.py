# -*- coding: utf-8 -*-
"""
Machine Learning Learner Module

Handles model training functionality extracted from scripts/mainfunction.py
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
    from ..scripts import accuracy_index as ai
    from ..scripts import progressBar as pB
except ImportError:
    import accuracy_index as ai
    import function_dataraster as dataraster

from .. import classifier_config
from .utils import backward_compatible, XGBLabelWrapper, LGBLabelWrapper

# Import sklearn modules for confusion matrix
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None


class ModelLearner:
    """Handles machine learning model training and validation"""
    
    @backward_compatible(
        inRaster="raster_path",
        inVector="vector_path", 
        inField="class_field",
        outModel="model_path",
        inSplit="split_config",
        inSeed="random_seed",
        outMatrix="matrix_path",
        inClassifier="classifier",
    )
    def __init__(
        self,
        raster_path: Union[str, np.ndarray] = None,
        vector_path: Union[str, np.ndarray] = None,
        class_field: str = "Class",
        model_path: Optional[str] = None,
        split_config: Union[int, float, str] = 100,
        random_seed: int = 0,
        matrix_path: Optional[str] = None,
        classifier: str = "GMM",
        extraParam: Optional[Dict[str, Any]] = None,
        feedback=None,
    ):
        """Initialize model learner with training parameters.
        
        Parameters
        ----------
        raster_path : str or np.ndarray
            Filtered image path or numpy array
        vector_path : str or np.ndarray
            Training shapefile path or numpy array
        class_field : str, default="Class"
            Column name where class numbers are stored
        split_config : int, float, or str, default=100
            Training split percentage, 'SLOO', or 'STAND'
        random_seed : int, default=0
            Random seed for reproducibility
        model_path : str, optional
            Model output file path
        matrix_path : str, optional
            Confusion matrix output file path
        classifier : str, default="GMM"
            Classifier type from available options
        extraParam : dict, optional
            Extra parameters for the classifier
        feedback : optional
            Feedback object for progress reporting
        """
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.class_field = class_field
        self.model_path = model_path
        self.split_config = split_config
        self.random_seed = random_seed
        self.matrix_path = matrix_path
        self.classifier = classifier
        self.extraParam = extraParam or {}
        self.feedback = feedback
        
        # Initialize attributes that will be set during training
        self.model = None
        self.training_stats = None
        self.confusion_matrix_data = None
    
    def train_model(self):
        """Train the machine learning model with provided data"""
        try:
            # Validate inputs
            self._validate_inputs()
            
            # Load and prepare data
            X, y = self._load_training_data()
            
            # Split data if needed
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            
            # Initialize classifier
            classifier_instance = self._get_classifier()
            
            # Train the model
            self._fit_model(classifier_instance, X_train, y_train)
            
            # Validate model if test data available
            if X_test is not None and y_test is not None:
                self._validate_model(X_test, y_test)
            
            # Save model if path provided
            if self.model_path:
                self._save_model()
            
            # Save confusion matrix if path provided
            if self.matrix_path and self.confusion_matrix_data is not None:
                self._save_confusion_matrix()
            
            return True
            
        except Exception as e:
            if self.feedback:
                self.feedback.reportError(f"Training failed: {str(e)}")
            raise
    
    def _validate_inputs(self):
        """Validate input parameters"""
        if self.raster_path is None:
            raise ValueError("raster_path is required")
        if self.vector_path is None:
            raise ValueError("vector_path is required")
        if self.classifier not in classifier_config.CLASSIFIER_CODES:
            raise ValueError(f"Unknown classifier: {self.classifier}")
    
    def _load_training_data(self):
        """Load training data from raster and vector inputs"""
        if self.feedback:
            self.feedback.setProgressText("Loading training data...")
        
        # Use existing dataraster functionality
        if isinstance(self.raster_path, str) and isinstance(self.vector_path, str):
            X, y = dataraster.get_samples_from_roi(
                self.raster_path,
                self.vector_path, 
                self.class_field
            )
        else:
            # Handle numpy array inputs
            X = self.raster_path
            y = self.vector_path
        
        return X, y
    
    def _split_data(self, X, y):
        """Split data into training and testing sets"""
        if self.split_config == 100:
            # Use all data for training
            return X, None, y, None
        elif isinstance(self.split_config, (int, float)) and 0 < self.split_config < 100:
            # Split data by percentage
            from sklearn.model_selection import train_test_split
            return train_test_split(
                X, y, 
                train_size=self.split_config/100,
                random_state=self.random_seed,
                stratify=y
            )
        elif self.split_config == 'SLOO':
            # Spatial Leave-One-Out
            # This would need to be implemented based on spatial constraints
            return X, None, y, None
        elif self.split_config == 'STAND':
            # Standard cross-validation
            return X, None, y, None
        else:
            raise ValueError(f"Invalid split_config: {self.split_config}")
    
    def _get_classifier(self):
        """Get classifier instance based on configuration"""
        # Get classifier name for sklearn classifiers
        classifier_name = classifier_config.CODE_TO_NAME.get(self.classifier, self.classifier)
        
        if self.classifier == "GMM":
            # Built-in GMM implementation
            from ..scripts.function_dataraster import learnGMM
            return learnGMM(**self.extraParam)
        elif self.classifier == "RF":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.extraParam)
        elif self.classifier == "SVM":
            from sklearn.svm import SVC
            return SVC(**self.extraParam)
        elif self.classifier == "KNN":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**self.extraParam)
        elif self.classifier == "XGB":
            from xgboost import XGBClassifier
            return XGBLabelWrapper(XGBClassifier(**self.extraParam))
        elif self.classifier == "LGB":
            from lightgbm import LGBMClassifier
            return LGBLabelWrapper(LGBMClassifier(**self.extraParam))
        elif self.classifier == "ET":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(**self.extraParam)
        elif self.classifier == "GBC":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**self.extraParam)
        elif self.classifier == "LR":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**self.extraParam)
        elif self.classifier == "NB":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**self.extraParam)
        elif self.classifier == "MLP":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**self.extraParam)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier}")
    
    def _fit_model(self, classifier, X_train, y_train):
        """Fit the model to training data"""
        if self.feedback:
            self.feedback.setProgressText(f"Training {self.classifier} model...")
        
        classifier.fit(X_train, y_train)
        self.model = classifier
        
        # Store training statistics
        self.training_stats = {
            'classifier': self.classifier,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else None,
            'n_classes': len(np.unique(y_train))
        }
    
    def _validate_model(self, X_test, y_test):
        """Validate model on test data"""
        if self.feedback:
            self.feedback.setProgressText("Validating model...")
        
        y_pred = self.model.predict(X_test)
        
        if confusion_matrix is not None:
            cm = confusion_matrix(y_test, y_pred)
            self.confusion_matrix_data = {
                'matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred
            }
    
    def _save_model(self):
        """Save trained model to file"""
        if self.feedback:
            self.feedback.setProgressText("Saving model...")
        
        # Create model data structure
        model_data = {
            'classifier': self.model,
            'stats': self.training_stats,
            'classifier_type': self.classifier,
            'parameters': self.extraParam
        }
        
        # Save using pickle
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _save_confusion_matrix(self):
        """Save confusion matrix to file"""
        if self.confusion_matrix_data is None:
            return
        
        # Save confusion matrix using existing functionality
        if hasattr(ai, 'saveConfusionMatrix'):
            ai.saveConfusionMatrix(
                self.confusion_matrix_data['matrix'],
                self.matrix_path
            )
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            return None
        
        return {
            'classifier_type': self.classifier,
            'training_stats': self.training_stats,
            'confusion_matrix': self.confusion_matrix_data
        }