"""Settings dock handlers for plugin runtime."""

from __future__ import annotations

from qgis.PyQt.QtWidgets import QMessageBox

from dzetsaka import classifier_config
from dzetsaka.qgis.dependency_catalog import FULL_DEPENDENCY_BUNDLE, full_bundle_label


def save_settings(gui) -> None:
    """Save modified settings from the settings dock."""
    settings_dock = getattr(gui, "settings_dock", None)
    if settings_dock is None:
        return
    classifier_selector = getattr(settings_dock, "classifierSelector", None)
    if classifier_selector is None:
        return

    if gui.sender() == classifier_selector:
        selected_classifier = classifier_selector.currentText()
        classifier_code = classifier_config.get_classifier_code(selected_classifier)

        missing_required = []

        if classifier_config.requires_sklearn(classifier_code):
            sklearn_available, sklearn_details = gui._check_sklearn_usable()
            if sklearn_available:
                gui.log.info(f"Scikit-learn detected for {selected_classifier}: {sklearn_details}")
            else:
                gui.log.warning(f"Scikit-learn check failed for {selected_classifier}: {sklearn_details}")
                missing_required.append("scikit-learn")

        if classifier_config.requires_xgboost(classifier_code):
            try:
                import xgboost  # noqa: F401
            except ImportError:
                missing_required.append("xgboost")

        if classifier_config.requires_lightgbm(classifier_code):
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                missing_required.append("lightgbm")
        if classifier_config.requires_catboost(classifier_code):
            try:
                import catboost  # noqa: F401
            except ImportError:
                missing_required.append("catboost")

        missing_optional = []
        if classifier_config.requires_sklearn(classifier_code):
            try:
                import optuna  # noqa: F401
            except ImportError:
                missing_optional.append("optuna")

        if missing_required:
            req_list = ", ".join(missing_required)
            optional_line = ""
            if missing_optional:
                opt_list = ", ".join(missing_optional)
                optional_line = f"Optional missing now: <code>{opt_list}</code><br>"
            reply = QMessageBox.question(
                gui,
                "Dependencies Missing for dzetsaka",
                (
                    "To fully use dzetsaka capabilities, we recommend installing all dependencies.<br><br>"
                    f"Required missing now: <code>{req_list}</code><br>"
                    f"{optional_line}<br>"
                    f"Full bundle to install: <code>{full_bundle_label()}</code><br><br>"
                    "Install the full dzetsaka dependency bundle now?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                to_install = list(FULL_DEPENDENCY_BUNDLE)
                if gui._try_install_dependencies(to_install):
                    gui.settings.setValue("/dzetsaka/classifier", selected_classifier)
                    gui.classifier = selected_classifier
                    QMessageBox.information(
                        gui,
                        "Installation Successful",
                        f"Dependencies installed successfully!<br><br>"
                        "<b>Important:</b> Please restart QGIS now.<br>"
                        "Without restarting, newly installed libraries may not be loaded, "
                        f"and {selected_classifier} training/classification can fail.<br><br>"
                        f"You can now try using {selected_classifier}.",
                        QMessageBox.StandardButton.Ok,
                    )
                else:
                    classifier_selector.setCurrentIndex(0)
                    gui.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
                    gui.classifier = "Gaussian Mixture Model"
            elif reply == QMessageBox.StandardButton.No:
                classifier_selector.setCurrentIndex(0)
                gui.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
                gui.classifier = "Gaussian Mixture Model"
        else:
            if gui.classifier != selected_classifier:
                gui.settings.setValue("/dzetsaka/classifier", selected_classifier)
                gui.classifier = selected_classifier



