# dzetsaka Project Overview

## Purpose
dzetsaka is a fast and easy-to-use classification plugin for QGIS. It provides supervised classification capabilities for remote sensing data using various machine learning algorithms including:

- Gaussian Mixture Model (GMM)
- Random Forest (RF) - requires scikit-learn
- Support Vector Machine (SVM) - requires scikit-learn  
- K-Nearest Neighbors (KNN) - requires scikit-learn

## Key Features
- Semi-automatic classification with spectral ROI training
- Processing toolbox algorithms for batch processing
- Support for confidence maps
- Cross-validation techniques (spatial and stand-based)
- Various filtering options (median, closing filter)
- Shannon entropy calculations
- Domain adaptation capabilities

## Target Users
Remote sensing researchers, GIS professionals, and anyone working with satellite/aerial imagery classification in QGIS.

## Dependencies
- QGIS 3.0+ (main plugin)
- scipy (core dependency)
- scikit-learn (for RF, SVM, KNN algorithms)
- joblib (required by scikit-learn)
- GDAL/OGR (through QGIS)
- PyQt5 (through QGIS)

## Current Issues
- Progress bar float/int conversion error (#39)
- Training algorithm problems on newer Python versions
- SVM getting stuck at 83% on some platforms
- Website link broken (karasiak.net)