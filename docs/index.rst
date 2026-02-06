.. dzetsaka documentation master file

dzetsaka: Classification Tool for QGIS
========================================

**dzetsaka** is a powerful QGIS plugin for remote sensing image classification using machine learning.
It supports **12 different algorithms**, from simple Gaussian Mixture Models to advanced gradient boosting frameworks.

.. image:: https://img.shields.io/badge/version-5.0.0-blue.svg
   :target: https://github.com/nkarasiak/dzetsaka
   :alt: Version

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2552284.svg
   :target: https://doi.org/10.5281/zenodo.2552284
   :alt: DOI

Key Features
------------

* **12 Machine Learning Algorithms**

  - Built-in: Gaussian Mixture Model (GMM)
  - Scikit-learn: Random Forest, SVM, KNN, Extra Trees, Gradient Boosting, Logistic Regression, Naive Bayes, MLP
  - Advanced: XGBoost, LightGBM, CatBoost

* **Automatic Hyperparameter Optimization**

  - Grid Search with Cross-Validation
  - Optuna integration for advanced optimization
  - Algorithm-specific parameter grids

* **User-Friendly Interfaces**

  - Classic dock widget for power users
  - Modern wizard for guided workflows
  - QGIS Processing Framework integration

* **Advanced Features**

  - Confidence mapping
  - Model persistence (save/reuse models)
  - SMOTE sampling for imbalanced datasets
  - SHAP explainability
  - Spatial cross-validation (SLOO, STAND)
  - Domain adaptation

* **Auto-Dependency Installer**

  - One-click installation of optional packages
  - No manual pip commands required
  - Real-time progress feedback

Quick Start
-----------

1. **Installation**

   Install dzetsaka from the QGIS Plugin Repository:

   - Open QGIS
   - Go to Plugins → Manage and Install Plugins
   - Search for "dzetsaka"
   - Click Install

2. **Basic Classification**

   .. code-block:: python

      # 1. Load your raster and training vector layer in QGIS
      # 2. Open dzetsaka: Plugins → dzetsaka → Classification tool
      # 3. Select your data:
      #    - Input raster: Your satellite/aerial image
      #    - Training layer: Vector file with labeled polygons
      #    - Class field: Column containing class labels (1, 2, 3, etc.)
      # 4. Choose algorithm (e.g., Random Forest)
      # 5. Click "Train & Classify"

3. **Using the Wizard**

   For a guided experience:

   - Click the "Wizard" button in the dzetsaka dock
   - Follow the step-by-step interface
   - Save your configuration as a recipe for reuse

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   installation
   quickstart
   algorithms
   workflows
   troubleshooting
   faq

Developer Guide
---------------

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   api/index
   architecture
   contributing
   changelog

Examples
--------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Examples

   examples/basic_classification
   examples/hyperparameter_tuning
   examples/confidence_mapping
   examples/batch_processing
   examples/custom_recipes

Algorithm Reference
-------------------

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   algorithms/gmm
   algorithms/random_forest
   algorithms/svm
   algorithms/knn
   algorithms/xgboost
   algorithms/lightgbm
   algorithms/catboost
   algorithms/extra_trees
   algorithms/gradient_boosting
   algorithms/logistic_regression
   algorithms/naive_bayes
   algorithms/mlp

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/mainfunction
   api/classifier_config
   api/processing
   api/ui

Support & Community
-------------------

* **GitHub Repository**: https://github.com/nkarasiak/dzetsaka
* **Issue Tracker**: https://github.com/nkarasiak/dzetsaka/issues
* **Citation**: Karasiak, N. (2019). dzetsaka: classification tool. DOI: 10.5281/zenodo.2552284

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
