Installation Guide
==================

System Requirements
-------------------

**Minimum Requirements:**

* QGIS 3.0 or later
* Python 3.8 or later
* 4GB RAM (8GB recommended for large rasters)
* 500MB free disk space

**Supported Operating Systems:**

* Windows 10/11
* macOS 10.15+
* Linux (Ubuntu 20.04+, Fedora 34+, etc.)

Installing from QGIS Plugin Repository
---------------------------------------

The easiest way to install dzetsaka is through the QGIS Plugin Manager:

1. Open QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Click on **All** tab
4. Search for "**dzetsaka**"
5. Click **Install Plugin**

The plugin will be installed automatically.

Installing Dependencies
-----------------------

dzetsaka requires different Python packages depending on which algorithms you want to use:

**Built-in Algorithm (No Dependencies)**

* **GMM (Gaussian Mixture Model)** - Works out of the box

**Scikit-learn Algorithms**

For Random Forest, SVM, KNN, Extra Trees, Gradient Boosting, Logistic Regression, Naive Bayes, and MLP:

.. code-block:: bash

   pip install scikit-learn

**Advanced Gradient Boosting**

For XGBoost:

.. code-block:: bash

   pip install xgboost

For CatBoost:

.. code-block:: bash

   pip install catboost

**Optional Features**

For hyperparameter optimization with Optuna:

.. code-block:: bash

   pip install optuna

For SHAP explainability:

.. code-block:: bash

   pip install shap

For SMOTE sampling:

.. code-block:: bash

   pip install imbalanced-learn

Auto-Installer (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dzetsaka v4.2.0+ includes an **automatic dependency installer**:

1. Open the dzetsaka plugin
2. Click **Settings** (gear icon)
3. Navigate to **Dependencies** tab
4. Click **Install** next to any missing package
5. Wait for installation to complete

This method handles all pip commands automatically with real-time progress feedback.

Manual Installation (Advanced)
-------------------------------

If you prefer manual installation or the auto-installer fails:

Finding Your QGIS Python
^^^^^^^^^^^^^^^^^^^^^^^^^

**Windows:**

.. code-block:: batch

   cd C:\Program Files\QGIS 3.XX\bin
   python-qgis.bat

**macOS:**

.. code-block:: bash

   /Applications/QGIS.app/Contents/MacOS/bin/python3

**Linux:**

.. code-block:: bash

   python3  # Usually the system Python

Installing Packages
^^^^^^^^^^^^^^^^^^^

Use the QGIS Python to ensure packages are installed in the correct environment:

.. code-block:: bash

   # Windows (as Administrator)
   "C:\Program Files\QGIS 3.XX\bin\python-qgis.bat" -m pip install scikit-learn xgboost catboost

   # macOS/Linux
   /path/to/qgis/python3 -m pip install scikit-learn xgboost catboost

Installing from Source (Developers)
------------------------------------

For plugin development or testing unreleased features:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/nkarasiak/dzetsaka.git
      cd dzetsaka

2. **Create plugin package:**

   .. code-block:: bash

      make plugin-package

3. **Install in QGIS:**

   - Extract the generated `.zip` file
   - Copy to your QGIS plugins directory:

     * Windows: ``C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\``
     * macOS: ``~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/``
     * Linux: ``~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/``

4. **Enable the plugin:**

   - Open QGIS
   - Go to Plugins → Manage and Install Plugins
   - Find dzetsaka in **Installed** tab
   - Check the box to enable

Verifying Installation
-----------------------

After installation, verify dzetsaka is working:

1. **Check Plugin is Loaded:**

   - Go to **Plugins → Manage and Install Plugins → Installed**
   - Verify "dzetsaka : Classification tool" is listed and enabled

2. **Check Dependencies:**

   - Open dzetsaka (Plugins → dzetsaka → Classification tool)
   - Click the **Settings** icon
   - Go to **Dependencies** tab
   - Green checkmarks indicate installed packages

3. **Test Basic Functionality:**

   - Open Processing Toolbox (Ctrl+Alt+T)
   - Navigate to **dzetsaka**
   - You should see all dzetsaka algorithms listed

Troubleshooting Installation Issues
------------------------------------

**"Plugin not found in repository"**

* Ensure QGIS is version 3.0 or later
* Check your internet connection
* Try refreshing the plugin repository: Plugins → Manage and Install Plugins → Settings → Reload repository

**"Import Error: No module named sklearn"**

* Scikit-learn is not installed
* Use the auto-installer or install manually (see above)

**"Permission denied" when installing packages**

* **Windows**: Run QGIS as Administrator
* **macOS/Linux**: Use ``sudo`` with the pip install command, or install to user directory with ``--user`` flag

**Plugins directory not found**

* Create the directory manually if it doesn't exist
* Ensure you're using the correct QGIS3 profile path

Uninstalling
------------

To uninstall dzetsaka:

1. Go to **Plugins → Manage and Install Plugins**
2. Find "dzetsaka : Classification tool"
3. Click **Uninstall Plugin**

**Note:** This does not uninstall the Python dependencies. To remove them:

.. code-block:: bash

   pip uninstall scikit-learn xgboost catboost optuna shap imbalanced-learn

Updating
--------

To update to the latest version:

1. Go to **Plugins → Manage and Install Plugins**
2. Click **Upgradeable** tab
3. Find dzetsaka and click **Upgrade Plugin**

Alternatively, enable automatic updates:

1. Go to **Settings** tab in Plugin Manager
2. Check "Check for updates on startup"
3. Set frequency to "Once a day" or "Every time QGIS starts"

Getting Help
------------

If you encounter installation issues:

* Check the :doc:`troubleshooting` guide
* Search existing `GitHub issues <https://github.com/nkarasiak/dzetsaka/issues>`_
* Create a new issue with:

  - Your QGIS version
  - Operating system
  - Error message (full traceback)
  - Installation method used
