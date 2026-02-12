"""Classifier runtime dependency checks for QGIS UI actions."""

from __future__ import annotations

from qgis.PyQt.QtWidgets import QMessageBox

from dzetsaka import classifier_config


def missing_classifier_dependencies(plugin, classifier_code: str) -> list[str]:
    """Return missing runtime dependencies for a classifier code."""
    code = str(classifier_code or "").strip().upper()
    missing: list[str] = []

    sklearn_ok, _sklearn_details = plugin._check_sklearn_usable()
    needs_sklearn_runtime = classifier_config.requires_sklearn(code) or code in {"XGB", "LGB", "CB"}
    if needs_sklearn_runtime and not sklearn_ok:
        missing.append("scikit-learn")

    if classifier_config.requires_xgboost(code) and not plugin._is_module_importable("xgboost"):
        missing.append("xgboost")
    if classifier_config.requires_lightgbm(code) and not plugin._is_module_importable("lightgbm"):
        missing.append("lightgbm")
    if classifier_config.requires_catboost(code) and not plugin._is_module_importable("catboost"):
        missing.append("catboost")

    seen = set()
    return [d for d in missing if not (d in seen or seen.add(d))]


def ensure_classifier_runtime_ready(
    plugin,
    classifier_code: str,
    source_label: str = "Classification",
    fallback_to_gmm: bool = False,
) -> bool:
    """Validate runtime dependencies for selected classifier before launch."""
    missing = missing_classifier_dependencies(plugin, classifier_code)
    if not missing:
        return True

    classifier_name = classifier_config.get_classifier_name(str(classifier_code or "").upper())
    missing_list = ", ".join(missing)
    QMessageBox.warning(
        plugin.iface.mainWindow(),
        "Dependencies Missing for dzetsaka",
        (
            f"{source_label}: selected classifier <b>{classifier_name}</b> cannot run right now.<br><br>"
            f"Missing runtime dependencies: <code>{missing_list}</code><br><br>"
            "Please install dependencies from the dashboard installer and restart QGIS."
        ),
        QMessageBox.StandardButton.Ok,
    )
    plugin.log.warning(f"{source_label} blocked: classifier {classifier_code} missing runtime deps: {missing_list}")

    if fallback_to_gmm:
        plugin.settings.setValue("/dzetsaka/classifier", "Gaussian Mixture Model")
        plugin.classifier = "Gaussian Mixture Model"
    return False
