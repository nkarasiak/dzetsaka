"""dzetsaka: Classification Tool for QGIS.

A powerful and fast classification plugin for QGIS that supports 11 machine learning
algorithms for remote sensing image classification. Originally based on Gaussian
Mixture Model classifier, dzetsaka now includes state-of-the-art algorithms like
XGBoost and CatBoost with automatic dependency installation.

Features:
---------
- 11 machine learning algorithms (GMM, RF, SVM, KNN, XGB, CB, ET, GBC, LR, NB, MLP)
- Automatic hyperparameter optimization using cross-validation
- Automatic dependency installation for advanced algorithms
- Support for confidence maps and per-class probability outputs
- Spatial and temporal cross-validation methods
- Model saving and loading capabilities
- Integration with QGIS Processing framework

Supported Algorithms:
--------------------
Core (built-in):
    - GMM: Gaussian Mixture Model

Scikit-learn based:
    - RF: Random Forest
    - SVM: Support Vector Machine
    - KNN: K-Nearest Neighbors
    - ET: Extra Trees
    - GBC: Gradient Boosting Classifier
    - LR: Logistic Regression
    - NB: Gaussian Naive Bayes
    - MLP: Multi-layer Perceptron

Advanced gradient boosting:
    - XGB: XGBoost (requires: pip install xgboost)
    - CB: CatBoost (requires: pip install catboost)

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

Original GMM implementation by Mathieu Fauvel

License:
--------
GNU General Public License v2.0 or later

Citation:
---------
Karasiak, N. (2016). Dzetsaka Qgis Classification plugin.
GitHub repository: https://github.com/nkarasiak/dzetsaka
DOI: 10.5281/zenodo.2552284
"""

import contextlib
import traceback

# Use qgis.PyQt for forward compatibility with QGIS 4.0 (PyQt6)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QDialog

from dzetsaka.qgis.runtime_bootstrap import initialize_runtime_state
from dzetsaka.qgis.task_runner import TaskFeedbackAdapter

# Import resources for icons
with contextlib.suppress(ImportError):
    from dzetsaka import resources  # noqa: F401


_TaskFeedbackAdapter = TaskFeedbackAdapter


class DzetsakaGUI(QDialog):
    """Main dzetsaka plugin class for QGIS integration.

    This class provides the main interface for the dzetsaka classification plugin
    in QGIS. It manages the plugin lifecycle, GUI components, settings, and
    coordinates between the user interface and classification algorithms.

    The plugin relies on the guided dashboard dialog for configuration,
    persistence, and supervised learning operations.

    Attributes
    ----------
    iface : QgsInterface
        Reference to the QGIS interface
    provider : DzetsakaProvider
        Processing provider for batch operations
    plugin_dir : str
        Plugin directory path
    settings : QSettings
        Qt settings object for configuration persistence
    dashboard_dock : ClassificationDashboardDock or None
        Guided/workflow dock exposed via the toolbar action
    classifier : str
        Currently selected classifier name
    classifiers : list
        List of available classifier names

    Examples
    --------
    The plugin is typically instantiated by QGIS:

    >>> plugin = DzetsakaGUI(iface)
    >>> plugin.initGui()  # Called by QGIS
    >>> plugin.run()  # Opens the classifier UI

    """

    DEFAULT_MASK_SUFFIX = "_mask"

    def __init__(self, iface):
        """Initialize the dzetsaka plugin.

        Parameters
        ----------
        iface : QgsInterface
            QGIS interface instance that provides hooks to manipulate the QGIS
            application at runtime, including access to map canvas, layers,
            toolbars, and menus.

        Notes
        -----
        The constructor:
        1. Saves reference to QGIS interface
        2. Loads plugin configuration from settings
        3. Initializes the processing provider
        4. Sets up plugin actions and menus

        """
        initialize_runtime_state(self, iface)

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        from dzetsaka.qgis.localization import tr

        return tr(message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None,
    ):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """
        from dzetsaka.qgis.actions_utils import add_action

        return add_action(
            self,
            icon_path=icon_path,
            text=text,
            callback=callback,
            enabled_flag=enabled_flag,
            add_to_menu=add_to_menu,
            add_to_toolbar=add_to_toolbar,
            status_tip=status_tip,
            whats_this=whats_this,
            parent=parent,
        )

    def get_icon_path(self, icon_name):
        """Get icon path, trying resource first, then fallback to file path."""
        from dzetsaka.qgis.actions_utils import get_icon_path

        return get_icon_path(self.plugin_dir, icon_name)

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        from dzetsaka.qgis.ui_init import init_gui

        init_gui(self)

    def _read_plugin_version(self):
        # type: () -> str
        """Read plugin version from metadata.txt, fallback to 'unknown'."""
        from dzetsaka.qgis.metadata_utils import read_plugin_version

        return read_plugin_version(self.plugin_dir, logger=self.log)

    # --------------------------------------------------------------------------

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        from dzetsaka.qgis.unload_utils import unload_plugin

        unload_plugin(self)

    #
    #        # remove the toolbar
    #       qg del self.toolbar
    # --------------------------------------------------------------------------

    def run(self):
        """Run method that opens the guided dashboard."""
        try:
            self.open_dashboard()
        except Exception as exc:
            self._report_unhandled_exception(
                error_title="dzetsaka Runtime Error",
                error_type="Runtime Error",
                error_message="Failed to open dashboard from run()",
                context="Plugin run() entrypoint",
                exc=exc,
            )

    def loadConfig(self):
        """!@brief Class that loads all saved settings from config.txt."""
        from dzetsaka.qgis.config_runtime import load_config

        load_config(self)

    def _check_sklearn_usable(self):
        # type: () -> tuple[bool, str]
        """Return whether scikit-learn is fully usable in the current runtime."""
        from dzetsaka.qgis.runtime_checks import check_sklearn_usable

        return check_sklearn_usable()

    def saveSettings(self):
        """!@brief save settings if modifications."""
        from dzetsaka.qgis.settings_handlers import save_settings

        save_settings(self)

    def _get_debug_info(self):
        """Generate debug information for GitHub issue reporting."""
        from dzetsaka.qgis.debug_info import build_debug_info

        return build_debug_info(self)

    def _default_output_name(self, in_raster_path, classifier_code):
        """Build deterministic default output filename for temporary classifications."""
        from dzetsaka.qgis.output_naming import default_output_name

        return default_output_name(in_raster_path, classifier_code)

    def _show_github_issue_popup(self, error_title, error_type, error_message, context):
        """Show standardized compact issue popup."""
        from dzetsaka.qgis.issue_reporting import show_standard_issue_popup

        show_standard_issue_popup(
            self,
            error_title=error_title,
            error_type=error_type,
            error_message=error_message,
            context=context,
        )

    def _try_install_dependencies(self, missing_deps):
        """Compatibility wrapper around extracted dependency installer.

        Note: This is the legacy synchronous version using QEventLoop.
        For new code, prefer _try_install_dependencies_async which uses QgsTask.
        """
        from dzetsaka.qgis.dependency_installer import try_install_dependencies

        return try_install_dependencies(self, missing_deps)

    def _try_install_dependencies_async(self, missing_deps, on_complete=None):
        """Install dependencies using QgsTask (recommended - non-blocking).

        This follows QGIS best practices by running installation in a background
        task without blocking the UI.

        Parameters
        ----------
        missing_deps : list
            List of missing dependency names
        on_complete : callable, optional
            Callback function(success: bool) called when installation finishes

        """
        from dzetsaka.qgis.dependency_installer import try_install_dependencies_async

        try_install_dependencies_async(self, missing_deps, on_complete)

    def open_dashboard(self):
        """Open the dockable classification dashboard (Quick/Advanced)."""
        try:
            from dzetsaka.qgis.dashboard_dock import open_dashboard_dock

            open_dashboard_dock(self, _LEFT_DOCK_AREA)
        except Exception as exc:
            self._report_unhandled_exception(
                error_title="Dashboard Open Failed",
                error_type="Runtime Error",
                error_message="Unexpected error while opening dashboard",
                context="open_dashboard()",
                exc=exc,
            )

    def open_batch_classification(self):
        """Open the batch classification dialog."""
        try:
            try:
                from dzetsaka.ui.batch_classification_dialog import BatchClassificationDialog
            except ImportError:
                from qgis.PyQt.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Batch Classification",
                    "Batch classification dialog is not available. Please check that all files are installed correctly.",
                )
                return

            dialog = BatchClassificationDialog(parent=self.iface.mainWindow(), iface=self.iface, plugin=self)
            try:
                dialog.exec_()
            except AttributeError:
                dialog.exec()
        except Exception as exc:
            self._report_unhandled_exception(
                error_title="Batch Classification Error",
                error_type="Runtime Error",
                error_message="Unexpected error while opening batch classification dialog",
                context="open_batch_classification()",
                exc=exc,
            )

    def on_close_dashboard_dock(self):
        """Track dashboard dock closing state."""
        from dzetsaka.qgis.dashboard_dock import close_dashboard_dock

        close_dashboard_dock(self)

    def _validate_classification_request(
        self,
        *,
        raster_path,
        do_training,
        vector_path=None,
        class_field=None,
        model_path=None,
        source_label="Classification",
    ):
        # type: (...) -> bool
        """Validate required inputs before launching a classification task."""
        from dzetsaka.qgis.input_validation import validate_classification_request

        return validate_classification_request(
            self,
            raster_path=raster_path,
            do_training=do_training,
            vector_path=vector_path,
            class_field=class_field,
            model_path=model_path,
            source_label=source_label,
        )

    def _is_module_importable(self, module_name):
        # type: (str) -> bool
        """Return True if a module can be imported in current runtime."""
        from dzetsaka.qgis.runtime_utils import is_module_importable

        return is_module_importable(module_name)

    def _missing_classifier_dependencies(self, classifier_code):
        # type: (str) -> list[str]
        """Return missing runtime dependencies for a classifier code."""
        from dzetsaka.qgis.classifier_runtime import missing_classifier_dependencies

        return missing_classifier_dependencies(self, classifier_code)

    def _ensure_classifier_runtime_ready(self, classifier_code, source_label="Classification", fallback_to_gmm=False):
        # type: (str, str, bool) -> bool
        """Validate runtime dependencies for selected classifier before launching task."""
        from dzetsaka.qgis.classifier_runtime import ensure_classifier_runtime_ready

        return ensure_classifier_runtime_ready(
            self,
            classifier_code=classifier_code,
            source_label=source_label,
            fallback_to_gmm=fallback_to_gmm,
        )

    def _start_classification_task(
        self,
        *,
        description,
        do_training,
        raster_path,
        vector_path,
        class_field,
        model_path,
        split_config,
        random_seed,
        matrix_path,
        classifier,
        output_path,
        mask_path,
        confidence_map,
        nodata,
        extra_params,
        error_context,
        success_prefix,
    ):
        # type: (...) -> None
        """Submit a background classification task to the QGIS task manager."""
        from dzetsaka.qgis.task_launcher import start_classification_task

        start_classification_task(
            self,
            description=description,
            do_training=do_training,
            raster_path=raster_path,
            vector_path=vector_path,
            class_field=class_field,
            model_path=model_path,
            split_config=split_config,
            random_seed=random_seed,
            matrix_path=matrix_path,
            classifier=classifier,
            output_path=output_path,
            mask_path=mask_path,
            confidence_map=confidence_map,
            nodata=nodata,
            extra_params=extra_params,
            error_context=error_context,
            success_prefix=success_prefix,
        )

    def execute_dashboard_config(self, config):
        """Run training and classification driven by the dashboard config dict.

        Parameters
        ----------
        config : dict
            The full configuration dictionary emitted by the guided dashboard.
            Keys: raster, vector, class_field, load_model, classifier,
            extraParam, output_raster, confidence_map, save_model,
            confusion_matrix, split_percent.

        """
        try:
            from dzetsaka.qgis.dashboard_execution import execute_dashboard_config

            execute_dashboard_config(self, config)
        except Exception as exc:
            self._report_unhandled_exception(
                error_title="Dashboard Execution Error",
                error_type="Runtime Error",
                error_message="Unexpected error while starting dashboard classification",
                context="execute_dashboard_config()",
                exc=exc,
            )

    def modifyConfig(self, section, option, value):
        """Modify configuration file with new section/option/value.

        Parameters
        ----------
        section : str
            Configuration section name
        option : str
            Configuration option name
        value : str
            New value to set

        """
        from dzetsaka.qgis.runtime_utils import write_plugin_config

        write_plugin_config(self.configFile, self.Config, section, option, value)

    def _report_unhandled_exception(
        self,
        *,
        error_title: str,
        error_type: str,
        error_message: str,
        context: str,
        exc: Exception,
    ) -> None:
        """Log and show a standardized issue popup for unexpected runtime errors."""
        tb = traceback.format_exc()
        details = f"{error_message}: {exc!s}\n\nTraceback:\n{tb}"
        self.log.exception(f"{error_title}: {error_message}", exc)
        try:
            self._show_github_issue_popup(
                error_title=error_title,
                error_type=error_type,
                error_message=details,
                context=context,
            )
        except Exception:
            from dzetsaka.logging import show_error_dialog

            show_error_dialog(error_title, details)


# Qt6 enum compatibility (QGIS 4 / PyQt6)
try:
    _LEFT_DOCK_AREA = Qt.LeftDockWidgetArea
except AttributeError:
    _LEFT_DOCK_AREA = Qt.DockWidgetArea.LeftDockWidgetArea
