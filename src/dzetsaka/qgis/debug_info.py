"""Debug info helpers for QGIS issue reporting."""

from __future__ import annotations

import platform

from qgis.core import QgsApplication

from dzetsaka import classifier_config


def build_debug_info(plugin) -> str:
    """Generate debug information for GitHub issue reporting."""
    try:
        qgis_version = QgsApplication.applicationVersion()
        python_version = platform.python_version()
        os_info = f"{platform.system()} {platform.release()}"

        classifier_code = classifier_config.get_classifier_code(plugin.classifier)

        sklearn_available = "No"
        sklearn_ok, sklearn_details = plugin._check_sklearn_usable()
        sklearn_available = f"Yes ({sklearn_details})" if sklearn_ok else f"No ({sklearn_details})"

        xgboost_available = "No"
        try:
            import xgboost

            xgboost_available = f"Yes ({xgboost.__version__})"
        except ImportError:
            pass

        lightgbm_available = "No"
        try:
            import lightgbm

            lightgbm_available = f"Yes ({lightgbm.__version__})"
        except ImportError:
            pass

        catboost_available = "No"
        try:
            import catboost

            catboost_available = f"Yes ({catboost.__version__})"
        except ImportError:
            pass

        debug_info = f"""
=== DZETSAKA DEBUG INFO ===
Plugin Version: 4.1.2
QGIS Version: {qgis_version}
Python Version: {python_version}
Operating System: {os_info}

Current Classifier: {plugin.classifier} ({classifier_code})
Available Libraries:
- Scikit-learn: {sklearn_available}
- XGBoost: {xgboost_available}
- LightGBM: {lightgbm_available}
- CatBoost: {catboost_available}
=== END DEBUG INFO ===
"""
        return debug_info.strip()
    except Exception as e:
        return f"Error generating debug info: {e!s}"

