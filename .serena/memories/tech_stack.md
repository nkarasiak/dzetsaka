# dzetsaka Tech Stack

## Core Technologies
- **Language**: Python 3.6+ (supports up to Python 3.12)
- **GUI Framework**: PyQt5 (through QGIS)
- **GIS Platform**: QGIS 3.0+
- **Geospatial Libraries**: GDAL/OGR

## Required Libraries
- **scipy**: Core scientific computing (always required)
- **scikit-learn**: Machine learning algorithms (RF, SVM, KNN)
- **joblib**: Parallel processing (dependency of scikit-learn)
- **numpy**: Numerical computations (via scipy/scikit-learn)

## QGIS Integration
- **Plugin Architecture**: Standard QGIS Python plugin
- **Processing Framework**: Custom algorithms in QGIS Processing Toolbox
- **UI Components**: PyQt5 widgets integrated with QGIS interface
- **Data Access**: QGIS data providers and GDAL bindings

## File Formats Supported
- **Raster**: GeoTIFF, various GDAL-supported formats
- **Vector**: Shapefile, various OGR-supported formats
- **Models**: Custom .model files (pickled Python objects)

## Architecture
- **Main Plugin**: dzetsaka.py (GUI interface)
- **Processing Algorithms**: processing/ directory (batch operations)
- **Core Functions**: scripts/ directory (classification logic)
- **Resource Management**: resources.py/qrc (UI resources)