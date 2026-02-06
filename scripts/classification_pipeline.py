"""Core machine learning functions for dzetsaka classification.

This module contains the main implementation of machine learning algorithms,
model training, image classification, and accuracy assessment for dzetsaka.
It provides a comprehensive interface between the QGIS GUI and the underlying
classification algorithms.

Key Components:
--------------
- LearnModel: Train classification models with various algorithms
- ClassifyImage: Classify entire raster images using trained models
- ConfusionMatrix: Compute accuracy statistics from classifications
- Backward compatibility decorators for parameter name changes
- Advanced cross-validation methods (SLOO, STAND)
- Label encoding wrappers for XGBoost and LightGBM

Supported Algorithms:
--------------------
- GMM: Gaussian Mixture Model (built-in)
- RF: Random Forest (sklearn)
- SVM: Support Vector Machine (sklearn)
- KNN: K-Nearest Neighbors (sklearn)
- XGB: XGBoost (requires xgboost package)
- LGB: LightGBM (requires lightgbm package)
- CB: CatBoost (requires catboost package)
- ET: Extra Trees (sklearn)
- GBC: Gradient Boosting Classifier (sklearn)
- LR: Logistic Regression (sklearn)
- NB: Gaussian Naive Bayes (sklearn)
- MLP: Multi-layer Perceptron (sklearn)

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>
Based on original work by Mathieu Fauvel

License:
--------
GNU General Public License v2.0 or later

"""

try:
    # when using it from QGIS 3
    from . import accuracy_index as ai
    from . import function_dataraster as dataraster
    from . import progress_bar
except BaseException:
    import accuracy_index as ai
    import function_dataraster as dataraster
    import progress_bar

import contextlib
import base64
import html
import json
import os
import pickle
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from osgeo import gdal, ogr

# Try to import Optuna optimizer (optional)
try:
    from .optimization.optuna_optimizer import OptunaOptimizer

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None

# Try to import SHAP explainer (optional)
try:
    from .explainability.shap_explainer import ModelExplainer, SHAP_AVAILABLE
except ImportError:
    SHAP_AVAILABLE = False
    ModelExplainer = None

# Try to import sampling techniques (Phase 3)
try:
    from .sampling.smote_sampler import SMOTESampler, apply_smote_if_needed, IMBLEARN_AVAILABLE
    from .sampling.class_weights import (
        compute_class_weights,
        apply_class_weights_to_model,
        compute_sample_weights,
        recommend_strategy,
    )

    SAMPLING_AVAILABLE = True
except ImportError:
    SAMPLING_AVAILABLE = False
    IMBLEARN_AVAILABLE = False
    SMOTESampler = None
    apply_smote_if_needed = None
    compute_class_weights = None

# Try to import validation techniques (Phase 3)
try:
    from .validation.nested_cv import NestedCrossValidator, perform_nested_cv
    from .validation.metrics import ValidationMetrics, create_classification_summary

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    NestedCrossValidator = None
    ValidationMetrics = None

# Import sklearn modules (optional dependency)
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    # Create dummy classes when sklearn is not available
    class BaseEstimator:
        """Dummy BaseEstimator class when sklearn is not available."""

    class ClassifierMixin:
        """Dummy ClassifierMixin class when sklearn is not available."""

    class LabelEncoder:
        """Dummy LabelEncoder class when sklearn is not available."""

        def fit(self, y):
            """Dummy fit method."""
            return self

        def transform(self, y):
            """Dummy transform method."""
            return y

        def fit_transform(self, y):
            """Dummy fit_transform method."""
            return y

    confusion_matrix = None
    SKLEARN_AVAILABLE = False

from .. import classifier_config
from ..logging_utils import Reporter, show_issue_popup
from .wrappers.label_encoders import (
    XGBLabelWrapper,
    LGBLabelWrapper,
    CBClassifierWrapper,
    SKLEARN_AVAILABLE as WRAPPERS_SKLEARN_AVAILABLE,
)

# Update SKLEARN_AVAILABLE if needed
if not SKLEARN_AVAILABLE and WRAPPERS_SKLEARN_AVAILABLE:
    SKLEARN_AVAILABLE = WRAPPERS_SKLEARN_AVAILABLE


def _get_catboost_wrapper():
    """Return a usable CatBoost wrapper, reloading wrapper module if needed."""
    global CBClassifierWrapper
    if CBClassifierWrapper is not None:
        return CBClassifierWrapper

    try:
        import importlib

        try:
            from .wrappers import label_encoders as label_encoders_module
        except ImportError:
            import wrappers.label_encoders as label_encoders_module

        label_encoders_module = importlib.reload(label_encoders_module)
        wrapper = getattr(label_encoders_module, "CBClassifierWrapper", None)
        if wrapper is not None:
            CBClassifierWrapper = wrapper
        return wrapper
    except Exception:
        return None


def _get_xgboost_wrapper():
    """Return a usable XGBoost wrapper, refreshing runtime state if needed."""
    global XGBLabelWrapper
    if XGBLabelWrapper is None:
        _refresh_runtime_dependency_state()
    return XGBLabelWrapper


def _get_lightgbm_wrapper():
    """Return a usable LightGBM wrapper, refreshing runtime state if needed."""
    global LGBLabelWrapper
    if LGBLabelWrapper is None:
        _refresh_runtime_dependency_state()
    return LGBLabelWrapper


def _refresh_runtime_dependency_state():
    """Refresh optional dependency state after in-session installations."""
    global XGBLabelWrapper, LGBLabelWrapper, CBClassifierWrapper
    global WRAPPERS_SKLEARN_AVAILABLE, SKLEARN_AVAILABLE
    global OPTUNA_AVAILABLE, OptunaOptimizer
    global SHAP_AVAILABLE, ModelExplainer
    global SAMPLING_AVAILABLE, IMBLEARN_AVAILABLE
    global SMOTESampler, apply_smote_if_needed, compute_class_weights
    global apply_class_weights_to_model, compute_sample_weights, recommend_strategy
    global VALIDATION_AVAILABLE, NestedCrossValidator, perform_nested_cv
    global ValidationMetrics, create_classification_summary

    import importlib

    # Wrappers (XGB/LGB/CB + sklearn availability in wrapper layer)
    try:
        try:
            from .wrappers import label_encoders as label_encoders_module
        except ImportError:
            import wrappers.label_encoders as label_encoders_module
        label_encoders_module = importlib.reload(label_encoders_module)
        XGBLabelWrapper = getattr(label_encoders_module, "XGBLabelWrapper", None)
        LGBLabelWrapper = getattr(label_encoders_module, "LGBLabelWrapper", None)
        CBClassifierWrapper = getattr(label_encoders_module, "CBClassifierWrapper", None)
        WRAPPERS_SKLEARN_AVAILABLE = bool(getattr(label_encoders_module, "SKLEARN_AVAILABLE", False))
        if WRAPPERS_SKLEARN_AVAILABLE:
            SKLEARN_AVAILABLE = True
    except Exception:
        pass

    # Optuna optimizer
    try:
        try:
            from .optimization import optuna_optimizer as optuna_optimizer_module
        except ImportError:
            import optimization.optuna_optimizer as optuna_optimizer_module
        optuna_optimizer_module = importlib.reload(optuna_optimizer_module)
        OptunaOptimizer = getattr(optuna_optimizer_module, "OptunaOptimizer", None)
        OPTUNA_AVAILABLE = OptunaOptimizer is not None
    except Exception:
        OPTUNA_AVAILABLE = False
        OptunaOptimizer = None

    # SHAP explainer
    try:
        try:
            from .explainability import shap_explainer as shap_explainer_module
        except ImportError:
            import explainability.shap_explainer as shap_explainer_module
        shap_explainer_module = importlib.reload(shap_explainer_module)
        ModelExplainer = getattr(shap_explainer_module, "ModelExplainer", None)
        SHAP_AVAILABLE = bool(getattr(shap_explainer_module, "SHAP_AVAILABLE", False) and ModelExplainer is not None)
    except Exception:
        SHAP_AVAILABLE = False
        ModelExplainer = None

    # Sampling utilities
    try:
        try:
            from .sampling import smote_sampler as smote_sampler_module
            from .sampling import class_weights as class_weights_module
        except ImportError:
            import sampling.smote_sampler as smote_sampler_module
            import sampling.class_weights as class_weights_module
        smote_sampler_module = importlib.reload(smote_sampler_module)
        class_weights_module = importlib.reload(class_weights_module)
        SMOTESampler = getattr(smote_sampler_module, "SMOTESampler", None)
        apply_smote_if_needed = getattr(smote_sampler_module, "apply_smote_if_needed", None)
        IMBLEARN_AVAILABLE = bool(getattr(smote_sampler_module, "IMBLEARN_AVAILABLE", False))
        compute_class_weights = getattr(class_weights_module, "compute_class_weights", None)
        apply_class_weights_to_model = getattr(class_weights_module, "apply_class_weights_to_model", None)
        compute_sample_weights = getattr(class_weights_module, "compute_sample_weights", None)
        recommend_strategy = getattr(class_weights_module, "recommend_strategy", None)
        SAMPLING_AVAILABLE = True
    except Exception:
        SAMPLING_AVAILABLE = False
        IMBLEARN_AVAILABLE = False
        SMOTESampler = None
        apply_smote_if_needed = None
        compute_class_weights = None
        apply_class_weights_to_model = None
        compute_sample_weights = None
        recommend_strategy = None

    # Validation utilities
    try:
        try:
            from .validation import nested_cv as nested_cv_module
            from .validation import metrics as metrics_module
        except ImportError:
            import validation.nested_cv as nested_cv_module
            import validation.metrics as metrics_module
        nested_cv_module = importlib.reload(nested_cv_module)
        metrics_module = importlib.reload(metrics_module)
        NestedCrossValidator = getattr(nested_cv_module, "NestedCrossValidator", None)
        perform_nested_cv = getattr(nested_cv_module, "perform_nested_cv", None)
        ValidationMetrics = getattr(metrics_module, "ValidationMetrics", None)
        create_classification_summary = getattr(metrics_module, "create_classification_summary", None)
        VALIDATION_AVAILABLE = True
    except Exception:
        VALIDATION_AVAILABLE = False
        NestedCrossValidator = None
        perform_nested_cv = None
        ValidationMetrics = None
        create_classification_summary = None


# Note: Label encoding wrappers (XGBLabelWrapper, LGBLabelWrapper, CBClassifierWrapper)
# are now imported from .wrappers.label_encoders module (see imports at top)

# Configuration constants
CLASSIFIER_CONFIGS = {
    "RF": {
        "param_grid": {
            "n_estimators": [100],
            "max_features": lambda x_shape: range(1, max(2, x_shape), max(1, int(x_shape / 3))),
        },
        "n_splits": 3,
    },
    "SVM": {
        "param_grid": {"gamma": 2.0 ** np.arange(-2, 3), "C": 10.0 ** np.arange(-1, 3)},
        "n_splits": 3,
    },
    "KNN": {"param_grid": {"n_neighbors": [1, 3, 10]}, "n_splits": 3},
    "XGB": {
        "param_grid": {
            "n_estimators": [100],
            "max_depth": [9],
            "learning_rate": [0.01],
        },
        "n_splits": 3,
    },
    "LGB": {
        "param_grid": {
            "n_estimators": [50, 200],
            "num_leaves": [31, 100],
            "learning_rate": [0.01, 0.2],
        },
        "n_splits": 3,
    },
    "CB": {
        "param_grid": {
            "iterations": [100, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5],
            "loss_function": ["MultiClass"],
        },
        "n_splits": 3,
    },
    "ET": {
        "param_grid": {
            "n_estimators": [50, 200],
            "max_features": lambda x_shape: range(1, max(2, x_shape), max(1, int(x_shape / 2))),
        },
        "n_splits": 3,
    },
    "GBC": {
        "param_grid": {
            "n_estimators": [50, 200],
            "max_depth": [3, 7],
            "learning_rate": [0.01, 0.2],
        },
        "n_splits": 3,
    },
    "LR": {
        "param_grid": {
            "C": 10.0 ** np.arange(-2, 3, 2),  # [-2, 0, 2] -> [0.01, 1, 100]
            "solver": ["liblinear", "lbfgs"],
        },
        "n_splits": 3,
    },
    "NB": {
        "param_grid": {"var_smoothing": 10.0 ** np.arange(-9, -3, 3)},  # [-9, -6, -3]
        "n_splits": 3,
    },
    "MLP": {
        "param_grid": {
            "hidden_layer_sizes": [(50,), (100, 50)],  # Keep simple and complex
            "alpha": [0.0001, 0.01],  # Keep low and high regularization
            "learning_rate_init": [0.001, 0.01],
        },
        "n_splits": 3,
    },
}

MAX_MEMORY_MB = 512
MIN_CROSS_VALIDATION_SPLITS = 2
LOG_TAG = "Dzetsaka/Core"
FAST_MODE_MAX_SAMPLES = 15000
FAST_MODE_MAX_OPTUNA_TRIALS = 25


def _param_grid_size(param_grid: Dict[str, Any]) -> int:
    """Estimate number of combinations in a parameter grid."""
    size = 1
    for values in param_grid.values():
        if isinstance(values, (list, tuple, np.ndarray)):
            size *= max(1, len(values))
        else:
            size *= 1
    return int(size)


def _reduce_param_grid_for_fast_mode(param_grid: Dict[str, Any], classifier_code: str) -> Dict[str, Any]:
    """Return a reduced parameter grid for faster default training."""
    reduced = {}
    for key, values in param_grid.items():
        if not isinstance(values, (list, tuple, np.ndarray)):
            reduced[key] = values
            continue
        vals = list(values)
        if len(vals) <= 2:
            reduced[key] = vals
            continue

        # Keep representative values (first and last) for speed.
        reduced[key] = [vals[0], vals[-1]]

    # Algorithm-specific fast defaults.
    if classifier_code in {"RF", "ET"} and "n_estimators" in reduced:
        est_vals = reduced["n_estimators"]
        if isinstance(est_vals, list):
            reduced["n_estimators"] = [min(est_vals)]
    if classifier_code == "MLP":
        reduced = {
            "hidden_layer_sizes": [(50,)],
            "alpha": [0.0001],
            "learning_rate_init": [0.001],
        }
    return reduced


def _build_tuning_subset_indices(y: np.ndarray, max_samples: int, random_seed: int) -> np.ndarray:
    """Build stratified subset indices for faster hyperparameter search."""
    n = int(y.shape[0])
    if n <= max_samples:
        return np.arange(n, dtype=int)

    rng = np.random.RandomState(random_seed)
    classes, counts = np.unique(y, return_counts=True)
    counts_by_class = {cls: int(cnt) for cls, cnt in zip(classes, counts)}

    # Initial proportional allocation with at least 1 sample per class.
    target_per_class = {}
    total_target = 0
    for cls, cnt in zip(classes, counts):
        target = round((cnt / n) * max_samples)
        target = max(1, min(int(cnt), target))
        target_per_class[cls] = target
        total_target += target

    # Trim overflow while preserving at least one sample per class.
    if total_target > max_samples:
        overflow = total_target - max_samples
        sorted_classes = sorted(classes, key=lambda c: target_per_class[c], reverse=True)
        for cls in sorted_classes:
            if overflow <= 0:
                break
            reducible = max(0, target_per_class[cls] - 1)
            delta = min(reducible, overflow)
            target_per_class[cls] -= delta
            overflow -= delta

    total_target = sum(target_per_class.values())

    # Fill deficit from larger classes if needed.
    if total_target < max_samples:
        deficit = max_samples - total_target
        sorted_classes = sorted(classes, key=lambda cls: counts_by_class[cls], reverse=True)
        for cls in sorted_classes:
            if deficit <= 0:
                break
            cnt = counts_by_class[cls]
            available = max(0, cnt - target_per_class[cls])
            delta = min(available, deficit)
            target_per_class[cls] += delta
            deficit -= delta

    chosen = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        k = min(int(target_per_class[cls]), int(cls_idx.size))
        if k <= 0:
            continue
        if k >= cls_idx.size:
            chosen.append(cls_idx)
        else:
            chosen.append(rng.choice(cls_idx, size=k, replace=False))

    if not chosen:
        return np.arange(min(n, max_samples), dtype=int)
    return np.concatenate(chosen).astype(int)


def _report(report: Reporter, message: Any) -> None:
    """Send info/warning/error or progress updates to QGIS."""
    if isinstance(message, (int, float)):
        report.progress(message)
        return

    text = str(message)
    lowered = text.lower()
    if text.startswith("Warning:"):
        report.warning(text)
    elif lowered.startswith("error") or " failed" in lowered or "cannot" in lowered:
        report.error(text)
    else:
        report.info(text)


def _parse_label_map_text(mapping_text: str) -> Dict[str, str]:
    """Parse a user mapping like '1:Forest,2:Water' into a dict."""
    result: Dict[str, str] = {}
    text = str(mapping_text or "").strip()
    if not text:
        return result
    chunks = []
    for part in text.replace("\n", ",").replace(";", ",").split(","):
        token = part.strip()
        if token:
            chunks.append(token)
    for token in chunks:
        if ":" not in token:
            continue
        k, v = token.split(":", 1)
        key = k.strip()
        val = v.strip()
        if key and val:
            result[key] = val
    return result


def _value_key(value: Any) -> str:
    """Normalize class value key to match manual mappings."""
    with contextlib.suppress(Exception):
        fv = float(value)
        if float(int(fv)) == fv:
            return str(int(fv))
    return str(value)


def _build_label_name_map(
    vector_path: str,
    class_field: str,
    label_column: str,
    manual_map: Dict[str, str],
) -> Dict[str, str]:
    """Build class value -> display name map from manual map and optional vector label column."""
    result = dict(manual_map)
    if not vector_path or not label_column:
        return result
    ds = ogr.Open(vector_path)
    if ds is None:
        return result
    lyr = ds.GetLayer()
    if lyr is None:
        return result
    for feat in lyr:
        cls_val = feat.GetField(class_field)
        lbl_val = feat.GetField(label_column)
        if cls_val in (None, "") or lbl_val in (None, ""):
            continue
        key = _value_key(cls_val)
        if key not in result:
            result[key] = str(lbl_val)
    return result


def _compute_metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """Compute confusion-matrix metrics including F1 without sklearn helpers."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)

    total = float(cm.sum()) if cm.size else 0.0
    accuracy = float(tp.sum() / total) if total else 0.0
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0
    weighted_f1 = float(np.sum(f1 * support) / support.sum()) if support.sum() else 0.0
    micro_f1 = accuracy  # for multiclass single-label, micro-F1 == accuracy

    return {
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "f1_micro": micro_f1,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.astype(int).tolist(),
    }


def _confusion_matrix_from_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: List[Any]) -> np.ndarray:
    """Compute confusion matrix with fixed label order without relying on sklearn."""
    if confusion_matrix is not None:
        return confusion_matrix(y_true, y_pred, labels=labels)
    index = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def _labels_to_1d(values: Any) -> np.ndarray:
    """Normalize labels/predictions to a 1D vector for reporting metrics."""
    if isinstance(values, tuple) and values:
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Common case: (n, 1) or (1, n)
        if 1 in arr.shape:
            return arr.reshape(-1)
        # If matrix-like predictions are provided, use class index by argmax.
        return np.argmax(arr, axis=1)
    return arr.reshape(-1)


def _write_report_bundle(
    report_dir: str,
    cm: np.ndarray,
    class_values: List[Any],
    class_names: List[str],
    summary_metrics: Dict[str, Any],
    config_meta: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
) -> None:
    """Write a detailed classification report bundle to disk."""
    os.makedirs(report_dir, exist_ok=True)

    def _fmt_metric(value: Any) -> str:
        try:
            return f"{float(value):.6f}"
        except Exception:
            return str(value)

    def _is_temporary_path(path_value: Any) -> bool:
        path_text = str(path_value or "").strip()
        if not path_text:
            return False
        with contextlib.suppress(Exception):
            normalized = os.path.abspath(os.path.expanduser(os.path.expandvars(path_text)))
            temp_root = os.path.abspath(tempfile.gettempdir())
            return normalized == temp_root or normalized.startswith(temp_root + os.sep)
        return False

    def _sanitize_config_for_rerun(raw_config: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(raw_config)
        if _is_temporary_path(sanitized.get("matrix_path", "")):
            sanitized["matrix_path"] = ""
            sanitized["matrix_path_note"] = (
                "Temporary matrix path omitted. Set a persistent confusion matrix path in output options."
            )
        return sanitized

    def _confusion_matrix_html_table(matrix: np.ndarray, names: List[str]) -> str:
        header_cells = "".join(f"<th>{html.escape(str(n))}</th>" for n in names)
        body_rows = []
        matrix_int = matrix.astype(int)
        for i, row in enumerate(matrix_int):
            label = html.escape(str(names[i]))
            cells = "".join(f"<td>{int(v)}</td>" for v in row)
            body_rows.append(f"<tr><th>{label}</th>{cells}</tr>")
        return (
            "<table class='cm'>"
            "<thead><tr><th>true/pred</th>"
            + header_cells
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table>"
        )

    def _per_class_metrics_html_table(values: List[Any], names: List[str], metrics: Dict[str, Any]) -> str:
        rows = []
        for i, cls_val in enumerate(values):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(cls_val))}</td>"
                f"<td>{html.escape(str(names[i]))}</td>"
                f"<td>{_fmt_metric(metrics['precision_per_class'][i])}</td>"
                f"<td>{_fmt_metric(metrics['recall_per_class'][i])}</td>"
                f"<td>{_fmt_metric(metrics['f1_per_class'][i])}</td>"
                f"<td>{int(metrics['support_per_class'][i])}</td>"
                "</tr>"
            )
        return (
            "<table>"
            "<thead><tr><th>Class value</th><th>Class name</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>"
            "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    numeric_csv = os.path.join(report_dir, "confusion_matrix_numeric.csv")
    np.savetxt(numeric_csv, cm.astype(int), delimiter=",", fmt="%d")

    labeled_csv = os.path.join(report_dir, "confusion_matrix_labeled.csv")
    with open(labeled_csv, "w", encoding="utf-8") as handle:
        handle.write("true/pred," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm.astype(int)):
            handle.write(class_names[i] + "," + ",".join(str(v) for v in row) + "\n")

    per_class_csv = os.path.join(report_dir, "per_class_metrics.csv")
    with open(per_class_csv, "w", encoding="utf-8") as handle:
        handle.write("class_value,class_name,precision,recall,f1,support\n")
        for i, cls_val in enumerate(class_values):
            handle.write(
                f"{cls_val},{class_names[i]},{summary_metrics['precision_per_class'][i]:.6f},"
                f"{summary_metrics['recall_per_class'][i]:.6f},{summary_metrics['f1_per_class'][i]:.6f},"
                f"{summary_metrics['support_per_class'][i]}\n"
            )

    metrics_json = os.path.join(report_dir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(summary_metrics, handle, indent=2)

    rerun_config = _sanitize_config_for_rerun(config_meta)
    config_json = os.path.join(report_dir, "run_config.json")
    with open(config_json, "w", encoding="utf-8") as handle:
        json.dump(rerun_config, handle, indent=2, default=str)

    heatmap_png = os.path.join(report_dir, "confusion_matrix_heatmap.png")
    clf_heatmap_png = os.path.join(report_dir, "classification_report_heatmap.png")
    try:
        import matplotlib.pyplot as plt

        def _apply_text_contrast(ax, values, threshold, light="#f8fafc", dark="#111827"):
            arr = np.asarray(values, dtype=float)
            n_cols = arr.shape[1] if arr.ndim == 2 else 0
            for idx, txt in enumerate(ax.texts):
                r = idx // max(1, n_cols)
                c = idx % max(1, n_cols)
                val = arr[r, c] if r < arr.shape[0] and c < arr.shape[1] else 0.0
                txt.set_color(light if val >= threshold else dark)

        try:
            import seaborn as sns

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)
            cm_int = cm.astype(int)
            try:
                import pandas as pd

                cm_df = pd.DataFrame(cm_int, index=class_names, columns=class_names)
                # Real confusion matrix counts (no normalization), as requested.
                sns.heatmap(cm_df, annot=True, fmt="g", ax=ax)
            except Exception:
                sns.heatmap(cm_int, annot=True, fmt="g", ax=ax)
        except Exception:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            im = ax.imshow(cm.astype(int), cmap="Blues")
            fig.colorbar(im, ax=ax)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=0, ha="center")
        ax.set_yticklabels(class_names, rotation=0)
        fig.tight_layout()
        fig.savefig(heatmap_png, dpi=150)
        plt.close(fig)

        # Classification-report heatmap (precision/recall/F1)
        if y_true is not None and y_pred is not None:
            with contextlib.suppress(Exception):
                import pandas as pd
                from sklearn.metrics import classification_report as sk_classification_report

                report_dict = sk_classification_report(
                    y_true,
                    y_pred,
                    labels=class_values,
                    target_names=[str(n) for n in class_names],
                    output_dict=True,
                    zero_division=0,
                )
                # Exact style requested:
                # sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
                metric_df = pd.DataFrame(report_dict).iloc[:-1, :].T
                fig2 = plt.figure(figsize=(10, 7))
                ax2 = fig2.add_subplot(111)
                with contextlib.suppress(Exception):
                    import seaborn as sns  # noqa: F811

                    sns.heatmap(metric_df, annot=True, ax=ax2)
                ax2.set_title("Classification report heatmap")
                ax2.set_xlabel("")
                ax2.set_ylabel("")
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
                fig2.tight_layout()
                fig2.savefig(clf_heatmap_png, dpi=160)
                plt.close(fig2)
    except Exception:
        pass

    embedded_heatmap = ""
    if os.path.exists(heatmap_png):
        with contextlib.suppress(Exception):
            with open(heatmap_png, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            embedded_heatmap = (
                "<img class='heatmap' alt='Confusion matrix heatmap' "
                f"src='data:image/png;base64,{encoded}'/>"
            )
    embedded_clf_heatmap = ""
    if os.path.exists(clf_heatmap_png):
        with contextlib.suppress(Exception):
            with open(clf_heatmap_png, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            embedded_clf_heatmap = (
                "<img class='heatmap' alt='Classification report heatmap' "
                f"src='data:image/png;base64,{encoded}'/>"
            )

    summary_md = os.path.join(report_dir, "report_summary.md")
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write("# dzetsaka Classification Report\n\n")
        handle.write(
            f"- Algorithm: `{config_meta.get('classifier_name', config_meta.get('classifier_code', ''))}`\n"
        )
        handle.write(f"- Execution date: `{config_meta.get('execution_date', '')}`\n")
        handle.write(f"- Split/CV mode: `{config_meta.get('split_mode', '')}`\n")
        handle.write(f"- Train/validation setting: `{config_meta.get('split_config', '')}`\n")
        handle.write(f"- Class field: `{config_meta.get('class_field', '')}`\n")
        handle.write(f"- Best hyperparameters: `{config_meta.get('best_hyperparameters', {})}`\n")
        handle.write("\n## Global metrics\n\n")
        handle.write(f"- Accuracy: `{summary_metrics['accuracy']:.6f}`\n")
        handle.write(f"- F1 macro: `{summary_metrics['f1_macro']:.6f}`\n")
        handle.write(f"- F1 weighted: `{summary_metrics['f1_weighted']:.6f}`\n")
        handle.write(f"- F1 micro: `{summary_metrics['f1_micro']:.6f}`\n")
        handle.write("\n## Files\n\n")
        handle.write("- `confusion_matrix_numeric.csv`\n")
        handle.write("- `confusion_matrix_labeled.csv`\n")
        handle.write("- `per_class_metrics.csv`\n")
        handle.write("- `metrics.json`\n")
        handle.write("- `run_config.json`\n")
        if os.path.exists(heatmap_png):
            handle.write("- `confusion_matrix_heatmap.png`\n")
        if os.path.exists(clf_heatmap_png):
            handle.write("- `classification_report_heatmap.png`\n")
        handle.write("- `classification_report.html`\n")

    html_report = os.path.join(report_dir, "classification_report.html")
    global_metrics_rows = "".join(
        [
            f"<tr><th>Accuracy</th><td>{_fmt_metric(summary_metrics.get('accuracy', 0.0))}</td></tr>",
            f"<tr><th>F1 macro</th><td>{_fmt_metric(summary_metrics.get('f1_macro', 0.0))}</td></tr>",
            f"<tr><th>F1 weighted</th><td>{_fmt_metric(summary_metrics.get('f1_weighted', 0.0))}</td></tr>",
            f"<tr><th>F1 micro</th><td>{_fmt_metric(summary_metrics.get('f1_micro', 0.0))}</td></tr>",
            f"<tr><th>OA (CONF)</th><td>{_fmt_metric(summary_metrics.get('overall_accuracy_conf', 0.0))}</td></tr>",
            f"<tr><th>Kappa</th><td>{_fmt_metric(summary_metrics.get('kappa_conf', 0.0))}</td></tr>",
            f"<tr><th>F1 mean (CONF)</th><td>{_fmt_metric(summary_metrics.get('f1_mean_conf', 0.0))}</td></tr>",
        ]
    )
    run_config_pretty = html.escape(json.dumps(rerun_config, indent=2, default=str))
    matrix_display = str(rerun_config.get("matrix_path", "")).strip() or "<auto-generated at run time>"
    with open(html_report, "w", encoding="utf-8") as handle:
        handle.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>dzetsaka Classification Report</title>"
            "<style>"
            ":root{--bg:#f6f8fb;--fg:#111827;--muted:#475569;--card:#ffffff;--line:#d7dde8;--head:#eaf0fb;--accent:#0f766e;}"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;color:var(--fg);"
            "background:linear-gradient(180deg,#f1f5f9 0%,var(--bg) 60%,#eef2ff 100%);}"
            ".wrap{max-width:1200px;margin:20px auto;padding:0 14px;}"
            "h1,h2{margin:0 0 10px;}h2{margin-top:24px;}"
            ".muted{color:var(--muted);font-size:13px;}"
            ".hero{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;"
            "box-shadow:0 8px 22px rgba(15,23,42,.06);}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;}"
            ".card{border:1px solid var(--line);border-radius:10px;padding:12px;background:var(--card);}"
            "table{border-collapse:collapse;width:100%;font-size:13px;}"
            "th,td{border:1px solid var(--line);padding:6px 8px;text-align:center;}"
            "th{text-align:left;background:var(--head);}"
            "table.cm th{text-align:center;}"
            "pre{background:#0b1324;color:#e5e7eb;padding:12px;border-radius:8px;overflow:auto;"
            "border:1px solid rgba(148,163,184,.35);}"
            ".heatmap{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;}"
            ".pill{display:inline-block;background:#ecfeff;color:var(--accent);font-size:12px;border:1px solid #99f6e4;"
            "padding:2px 8px;border-radius:999px;margin-bottom:8px;}"
            "</style></head><body><div class='wrap'>"
        )
        handle.write("<div class='hero'><span class='pill'>dzetsaka report bundle</span>")
        handle.write("<h1>Classification Report</h1>")
        handle.write(
            "<p class='muted'>"
            f"Algorithm: <b>{html.escape(str(config_meta.get('classifier_name', config_meta.get('classifier_code', ''))))}</b> | "
            f"Execution date: <b>{html.escape(str(config_meta.get('execution_date', '')))}</b> | "
            f"Validation mode: <b>{html.escape(str(config_meta.get('split_mode', '')))}</b> | "
            f"Class field: <b>{html.escape(str(config_meta.get('class_field', '')))}</b>"
            "</p>"
        )
        handle.write("</div>")
        handle.write("<div class='grid'>")
        handle.write("<div class='card'><h2>Global Metrics</h2><table><tbody>")
        handle.write(global_metrics_rows)
        handle.write("</tbody></table></div>")
        handle.write("<div class='card'><h2>Run Metadata</h2><table><tbody>")
        for label, key in [
            ("Split config", "split_config"),
            ("Optimization method", "optimization_method"),
            ("Raster path", "raster_path"),
            ("Vector path", "vector_path"),
            ("Execution date", "execution_date"),
        ]:
            handle.write(
                f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(rerun_config.get(key, '')))}</td></tr>"
            )
        handle.write(f"<tr><th>Matrix path</th><td>{html.escape(matrix_display)}</td></tr>")
        handle.write("</tbody></table></div>")
        handle.write("</div>")

        handle.write("<h2>Confusion Matrix (NxN)</h2>")
        handle.write(_confusion_matrix_html_table(cm, class_names))
        if embedded_heatmap:
            handle.write("<h2>Heatmap</h2>")
            handle.write(embedded_heatmap)
        if embedded_clf_heatmap:
            handle.write("<h2>Classification Report Heatmap</h2>")
            handle.write(embedded_clf_heatmap)

        handle.write("<h2>Per-class Metrics</h2>")
        handle.write(_per_class_metrics_html_table(class_values, class_names, summary_metrics))

        handle.write("<h2>Run Configuration (JSON)</h2>")
        handle.write(
            "<p class='muted'>Use this JSON as reference for reproducibility. "
            "In dzetsaka, reruns are managed through Expert mode recipe tools "
            "(Save Current / Gallery / JSON import), not by pasting this block directly.</p>"
        )
        handle.write("<pre>")
        handle.write(run_config_pretty)
        handle.write("</pre>")
        handle.write("</div></body></html>")


class LearnModel:
    """Machine learning model training class for dzetsaka classification."""

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
        """Learn model with a shapefile and a raster image.

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
            Classifier type. Available options:
            - 'GMM': Gaussian Mixture Model (built-in, no dependencies)
            - 'RF': Random Forest (sklearn)
            - 'SVM': Support Vector Machine (sklearn)
            - 'KNN': K-Nearest Neighbors (sklearn)
            - 'XGB': XGBoost (requires: pip install xgboost)
            - 'LGB': LightGBM (requires: pip install lightgbm)
            - 'CB': CatBoost (requires: pip install catboost)
            - 'ET': Extra Trees (sklearn)
            - 'GBC': Gradient Boosting Classifier (sklearn)
            - 'LR': Logistic Regression (sklearn)
            - 'NB': Gaussian Naive Bayes (sklearn)
            - 'MLP': Multi-layer Perceptron (sklearn)
        extraParam : dict, optional
            Additional parameters for advanced configurations:

            Optimization (Phase 1):
            - 'USE_OPTUNA': bool, default=False
                Use Optuna for hyperparameter optimization (2-10x faster than GridSearchCV)
            - 'OPTUNA_TRIALS': int, default=100
                Number of optimization trials for Optuna

            Explainability (Phase 2):
            - 'COMPUTE_SHAP': bool, default=False
                Compute SHAP feature importance after training (requires shap>=0.41.0)
            - 'SHAP_OUTPUT': str, optional
                Path to save feature importance raster (e.g., 'importance.tif')
            - 'SHAP_SAMPLE_SIZE': int, default=1000
                Number of pixels to sample for SHAP computation

            Imbalance Handling (Phase 3):
            - 'USE_SMOTE': bool, default=False
                Apply SMOTE oversampling for imbalanced datasets (requires imbalanced-learn)
            - 'SMOTE_K_NEIGHBORS': int, default=5
                Number of neighbors for SMOTE
            - 'USE_CLASS_WEIGHTS': bool, default=False
                Apply class weights for cost-sensitive learning
            - 'CLASS_WEIGHT_STRATEGY': str, default='balanced'
                Weight strategy ('balanced', 'custom')
            - 'CUSTOM_CLASS_WEIGHTS': dict, optional
                Custom weights {class_label: weight}

            Validation (Phase 3):
            - 'USE_NESTED_CV': bool, default=False
                Use nested cross-validation for unbiased evaluation
            - 'NESTED_INNER_CV': int, default=3
                Inner CV folds for parameter tuning
            - 'NESTED_OUTER_CV': int, default=5
                Outer CV folds for evaluation

            Other:
            - 'param_grid': dict, optional
                Custom parameter grid for GridSearchCV (overrides defaults)
        feedback : object, optional
            Feedback object for progress reporting

        Returns
        -------
        None
            Stores model in self.model, scaling parameters in self.M and self.m

        """
        # Re-evaluate optional backends at runtime so newly installed
        # dependencies are picked up without restarting the Python process.
        _refresh_runtime_dependency_state()

        self.report = Reporter.from_feedback(feedback, tag=LOG_TAG)
        report = self.report
        # Validate required parameters
        if raster_path is None:
            raise ValueError("raster_path is required")
        if vector_path is None:
            raise ValueError("vector_path is required")

        # Initialize and validate parameters
        self._validate_inputs(raster_path, vector_path, classifier, feedback)
        extraParam = extraParam or {}

        # Setup progress tracking
        total = 100 / 10
        progress = self._setup_progress_feedback(feedback)

        # Load and prepare data
        try:
            X, Y, coords, distanceArray, STDs, vector_test_path = self._load_and_prepare_data(
                raster_path,
                vector_path,
                class_field,
                split_config,
                extraParam,
                feedback,
            )

        except Exception as e:
            self._handle_data_loading_error(e, class_field, feedback, progress)
            return None

        [n, d] = X.shape
        C = int(Y.max())
        SPLIT = split_config

        # Validate labels before any training
        finite_mask = np.isfinite(Y)
        if not np.all(finite_mask):
            dropped = int(np.size(Y) - np.count_nonzero(finite_mask))
            _report(
                report,
                f"Warning: Non-finite class labels detected. Dropping {dropped} samples before training.",
            )
            X = X[finite_mask]
            Y = Y[finite_mask]
            if X.size == 0 or Y.size == 0:
                _report(report, "Error: No valid training labels remain after filtering.")
                return None

        # Cleanup handled in _load_and_prepare_data method
        # os.remove(filename)
        # os.rmdir(temp_folder)

        # Phase 3: Class imbalance handling
        self._handle_class_imbalance(X, Y, classifier, extraParam)

        # Apply SMOTE oversampling if enabled
        if extraParam.get("USE_SMOTE", False):
            X, Y = self._apply_smote(X, Y, extraParam)

        # Compute class weights if enabled
        self.class_weights_ = None
        if extraParam.get("USE_CLASS_WEIGHTS", False):
            self.class_weights_ = self._compute_weights(Y, extraParam)

        [n, d] = X.shape
        C = int(Y.max())

        # Scale the data
        X, M, m = self.scale(X)

        _report(report, int(1 * total))
        if feedback == "gui":
            progress.prgBar.setValue(10)
        # Learning process take split of groundthruth pixels for training and
        # the remaining for testing

        try:
            if isinstance(SPLIT, (int, float)):
                if SPLIT < 100:
                    # Random stratified selection of samples
                    # Collect indices first, then create arrays in one operation
                    # This avoids O(n^2) memory allocation from repeated concatenation
                    np.random.seed(random_seed)

                    train_indices = []
                    test_indices = []

                    for i in range(C):
                        t = np.where((i + 1) == Y)[0]
                        nc = t.size
                        ns = int(nc * (SPLIT / float(100)))
                        rp = np.random.permutation(nc)
                        train_indices.append(t[rp[:ns]])
                        test_indices.append(t[rp[ns:]])

                    # Concatenate all indices at once
                    train_idx = np.concatenate(train_indices)
                    test_idx = np.concatenate(test_indices)

                    # Create arrays using fancy indexing (single operation)
                    x = X[train_idx, :]
                    y = Y[train_idx]
                    xt = X[test_idx, :]
                    yt = Y[test_idx]

                else:
                    x, y = X, Y
                    self.x = x
                    self.y = y
            else:
                x, y = X, Y
                self.x = x
                self.y = y
        except BaseException:
            _report(report, "Problem while learning if SPLIT <1")

        _report(report, int(2 * total))
        if feedback == "gui":
            progress.prgBar.setValue(20)

        classifier_code_input = str(classifier).upper()
        selected_hyperparameters: Dict[str, Any] = {}
        optimization_method = "none"

        _report(report, "Starting model training process...")
        _report(
            report,
            "Training phase in progress. This may take several minutes depending on your data size and classifier settings. The progress bar may appear to pause during hyperparameter optimization - this is normal.",
        )

        if feedback == "gui":
            progress.prgBar.setValue(25)
        # Train Classifier
        if classifier == "GMM":
            try:
                from . import gmm_ridge as gmmr
            except BaseException:
                import gmm_ridge as gmmr

            try:
                # tau=10.0**sp.arange(-8,8,0.5)
                model = gmmr.GMMR()
                model.learn(x, y)
                # htau,err = model.cross_validation(x,y,tau)
                # model.tau = htau
            except BaseException:
                _report(report, "Cannot train with GMM")
        else:
            # from sklearn import neighbors
            # from sklearn.svm import SVC
            # from sklearn.ensemble import RandomForestClassifier

            # model_selection = True
            try:
                from sklearn.model_selection import GridSearchCV, StratifiedKFold

                joblib = __import__("joblib")  # Test for joblib dependency
            except ImportError as e:
                if "joblib" in str(e):
                    _report(
                        report,
                        "Missing dependency: joblib. Please install with: pip install joblib",
                    )
                    return None
                else:
                    _report(
                        report,
                        "Missing scikit-learn dependency for {classifier}. Please install with: pip install scikit-learn. Error: {e}",
                    )
                    return None

            try:
                if extraParam and "param_algo" in extraParam:
                    param_algo = extraParam["param_algo"]

                # AS Qgis in Windows doensn't manage multiprocessing, force to
                # use 1 thread for not linux system

                if SPLIT == "STAND":
                    label = np.copy(Y)

                    if extraParam:
                        SLOO = extraParam.get("SLOO", False)
                        maxIter = extraParam.get("maxIter", 5)
                    else:
                        SLOO = False
                        maxIter = 5

                    try:
                        from .function_vector import StandCV
                    except ImportError:
                        from function_vector import StandCV

                    rawCV = StandCV(label, STDs, maxIter, SLOO, seed=random_seed)
                    print(rawCV)
                    cvDistance = []
                    for tr, vl in rawCV:
                        # sts.append(stat)
                        cvDistance.append((tr, vl))

                if SPLIT == "SLOO":
                    # Compute CV for Learning later

                    label = np.copy(Y)
                    if extraParam:
                        if "distance" in extraParam:
                            distance = extraParam["distance"]
                        else:
                            _report(report, "You need distance in extraParam")

                        minTrain = float(extraParam["minTrain"]) if "minTrain" in extraParam else -1

                        SLOO = extraParam.get("SLOO", True)

                        maxIter = extraParam.get("maxIter", False)

                        otherLevel = extraParam.get("otherLevel", False)
                    # sts = []
                    cvDistance = []

                    """
                    rawCV = DistanceCV(distanceArray,label,distanceThresold=distance,minTrain=minTrain,SLOO=SLOO,maxIter=maxIter,verbose=False,stats=False)

                    """
                    # feedback.setProgressText('distance is '+str(extraParam['distance']))
                    _report(report, "label is " + str(label.shape))
                    _report(report, "distance array shape is " + str(distanceArray.shape))
                    _report(report, "minTrain is " + str(minTrain))
                    _report(report, "SLOO is " + str(SLOO))
                    _report(report, "maxIter is " + str(maxIter))

                    # Import distanceCV dynamically when needed
                    try:
                        from function_vector import DistanceCV
                    except ImportError:
                        from .function_vector import DistanceCV

                    rawCV = DistanceCV(
                        distanceArray,
                        label,
                        distanceThresold=distance,
                        minTrain=minTrain,
                        SLOO=SLOO,
                        maxIter=maxIter,
                        stats=False,
                    )

                    _report(report, "Computing SLOO Cross Validation")

                    for tr, vl in rawCV:
                        _report(report, "Training size is " + str(tr.shape))
                        _report(report, "Validation size is " + str(vl.shape))
                        # sts.append(stat)
                        cvDistance.append((tr, vl))
                    """
                    for tr,vl,stat in rawCV :
                        sts.append(stat)
                        cvDistance.append((tr,vl))
                    """
                    #

                if classifier == "RF":
                    from sklearn.ensemble import RandomForestClassifier

                    config = CLASSIFIER_CONFIGS["RF"]
                    param_grid = config["param_grid"].copy()
                    # Handle lambda function for max_features
                    if callable(param_grid.get("max_features")):
                        try:
                            max_features_range = param_grid["max_features"](x.shape[1])
                            # Convert to list to avoid range object issues
                            param_grid["max_features"] = list(max_features_range)
                            _report(report, f"RF max_features range: {param_grid['max_features']}")
                        except Exception as e:
                            _report(report, f"Error generating max_features range for RF: {e}")
                            # Fallback to safe values
                            param_grid["max_features"] = [1, min(x.shape[1], 3)]

                    if "param_algo" in locals():
                        classifier = RandomForestClassifier(random_state=random_seed, **param_algo)
                    else:
                        classifier = RandomForestClassifier(random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "SVM":
                    from sklearn.svm import SVC

                    config = CLASSIFIER_CONFIGS["SVM"]
                    param_grid = config["param_grid"]

                    if "param_algo" in locals():
                        classifier = SVC(probability=True, random_state=random_seed, **param_algo)
                        print("Found param algo : " + str(param_algo))
                    else:
                        classifier = SVC(probability=True, kernel="rbf", random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "KNN":
                    from sklearn import neighbors

                    config = CLASSIFIER_CONFIGS["KNN"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = neighbors.KNeighborsClassifier(**param_algo)
                    else:
                        classifier = neighbors.KNeighborsClassifier()

                    n_splits = config["n_splits"]

                elif classifier == "XGB":
                    import importlib.util

                    if importlib.util.find_spec("xgboost") is None:
                        _report(report, "XGBoost not found. Install with: pip install xgboost")
                        return None
                    xgb_wrapper = _get_xgboost_wrapper()
                    if xgb_wrapper is None:
                        _report(
                            report,
                            "XGBoost requires a usable scikit-learn runtime for label encoding. "
                            "Install with: pip install scikit-learn and restart QGIS.",
                        )
                        return None

                    config = CLASSIFIER_CONFIGS["XGB"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = xgb_wrapper(
                            random_state=random_seed,
                            eval_metric="logloss",
                            **param_algo,
                        )
                    else:
                        classifier = xgb_wrapper(random_state=random_seed, eval_metric="logloss")
                    n_splits = config["n_splits"]

                elif classifier == "LGB":
                    import importlib.util

                    if importlib.util.find_spec("lightgbm") is None:
                        _report(report, "LightGBM not found. Install with: pip install lightgbm")
                        return None
                    lgb_wrapper = _get_lightgbm_wrapper()
                    if lgb_wrapper is None:
                        _report(
                            report,
                            "LightGBM requires a usable scikit-learn runtime for label encoding. "
                            "Install with: pip install scikit-learn and restart QGIS.",
                        )
                        return None

                    config = CLASSIFIER_CONFIGS["LGB"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = lgb_wrapper(random_state=random_seed, verbose=-1, **param_algo)
                    else:
                        classifier = lgb_wrapper(random_state=random_seed, verbose=-1)
                    n_splits = config["n_splits"]

                elif classifier == "CB":
                    import importlib.util

                    if importlib.util.find_spec("catboost") is None:
                        _report(report, "CatBoost not found. Install with: pip install catboost")
                        return None
                    cb_wrapper = _get_catboost_wrapper()
                    if cb_wrapper is None:
                        _report(
                            report,
                            "CatBoost requires a usable scikit-learn runtime for label encoding. "
                            "Install with: pip install scikit-learn and restart QGIS.",
                        )
                        return None

                    config = CLASSIFIER_CONFIGS["CB"]
                    param_grid = config["param_grid"].copy()

                    # CatBoost grid search can become very slow on large or severely imbalanced datasets.
                    # Use a reduced default search space in those situations to keep runtime practical.
                    try:
                        class_counts = np.unique(y, return_counts=True)[1]
                        min_count = int(np.min(class_counts)) if class_counts.size else 0
                        max_count = int(np.max(class_counts)) if class_counts.size else 0
                        imbalance_ratio = (max_count / max(min_count, 1)) if max_count > 0 else 0.0
                        sample_count = int(x.shape[0])

                        if sample_count > 30000 or imbalance_ratio > 100:
                            param_grid = {
                                "iterations": [100, 200],
                                "depth": [4, 6],
                                "learning_rate": [0.05, 0.1],
                                "l2_leaf_reg": [3],
                                "loss_function": ["MultiClass"],
                            }
                            _report(
                                report,
                                "Using faster CatBoost search space for large/imbalanced data "
                                f"(samples={sample_count}, imbalance_ratio={imbalance_ratio:.2f}).",
                            )
                    except Exception:
                        pass

                    if "param_algo" in locals():
                        classifier = cb_wrapper(
                            random_seed=random_seed,
                            verbose=False,
                            allow_writing_files=False,
                            **param_algo,
                        )
                    else:
                        classifier = cb_wrapper(
                            random_seed=random_seed,
                            verbose=False,
                            allow_writing_files=False,
                        )
                    n_splits = config["n_splits"]

                elif classifier == "ET":
                    from sklearn.ensemble import ExtraTreesClassifier

                    config = CLASSIFIER_CONFIGS["ET"]
                    param_grid = config["param_grid"].copy()
                    # Handle lambda function for max_features
                    if callable(param_grid.get("max_features")):
                        try:
                            max_features_range = param_grid["max_features"](x.shape[1])
                            # Convert to list to avoid range object issues
                            param_grid["max_features"] = list(max_features_range)
                            _report(report, f"ET max_features range: {param_grid['max_features']}")
                        except Exception as e:
                            _report(report, f"Error generating max_features range for ET: {e}")
                            # Fallback to safe values
                            param_grid["max_features"] = [1, min(x.shape[1], 3)]

                    if "param_algo" in locals():
                        classifier = ExtraTreesClassifier(random_state=random_seed, **param_algo)
                    else:
                        classifier = ExtraTreesClassifier(random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "GBC":
                    from sklearn.ensemble import GradientBoostingClassifier

                    config = CLASSIFIER_CONFIGS["GBC"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = GradientBoostingClassifier(random_state=random_seed, **param_algo)
                    else:
                        classifier = GradientBoostingClassifier(random_state=random_seed)
                    n_splits = config["n_splits"]

                elif classifier == "LR":
                    from sklearn.linear_model import LogisticRegression

                    config = CLASSIFIER_CONFIGS["LR"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = LogisticRegression(random_state=random_seed, max_iter=1000, **param_algo)
                    else:
                        classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
                    n_splits = config["n_splits"]

                elif classifier == "NB":
                    from sklearn.naive_bayes import GaussianNB

                    config = CLASSIFIER_CONFIGS["NB"]
                    param_grid = config["param_grid"]
                    classifier = GaussianNB(**param_algo) if "param_algo" in locals() else GaussianNB()
                    n_splits = config["n_splits"]

                elif classifier == "MLP":
                    from sklearn.neural_network import MLPClassifier

                    config = CLASSIFIER_CONFIGS["MLP"]
                    param_grid = config["param_grid"]
                    if "param_algo" in locals():
                        classifier = MLPClassifier(random_state=random_seed, max_iter=500, **param_algo)
                    else:
                        classifier = MLPClassifier(random_state=random_seed, max_iter=500)
                    n_splits = config["n_splits"]

            except ImportError as e:
                _report(report, "Import error for classifier " + classifier + ": " + str(e))
                if feedback == "gui":
                    progress.reset()
                return None
            except Exception as e:
                _report(report, "Error initializing classifier " + classifier + ": " + str(e))
                if feedback == "gui":
                    progress.reset()
                return None

            if feedback == "gui":
                progress.prgBar.setValue(30)

            y.shape = (y.size,)

            # Validate training data before proceeding
            if x.shape[0] == 0 or y.shape[0] == 0:
                _report(report, "Error: No training data found. Check your training samples.")
                if feedback == "gui":
                    progress.reset()
                return None

            if x.shape[0] != y.shape[0]:
                _report(report, "Error: Mismatch between feature data and labels. Check your training data.")
                if feedback == "gui":
                    progress.reset()
                return None

            # Check for any NaN or infinite values
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                _report(
                    report,
                    "Warning: NaN or infinite values detected in training data. These will be handled automatically.",
                )
                x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)

            # Check if all classes have sufficient samples for cross-validation
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples = np.min(class_counts)
            max_samples = np.max(class_counts)
            sample_count = int(x.shape[0])
            imbalance_ratio = float(max_samples / max(min_samples, 1))
            fast_mode_enabled = bool(extraParam.get("FAST_MODE", True))
            fast_mode_active = fast_mode_enabled and (
                sample_count > 20000 or imbalance_ratio > 50 or min_samples < 30
            )

            if fast_mode_active:
                _report(
                    report,
                    "Fast mode enabled for heavy/imbalanced data "
                    f"(samples={sample_count}, imbalance_ratio={imbalance_ratio:.2f}, min_class={min_samples}).",
                )

                if n_splits > 2:
                    n_splits = 2
                    _report(report, "Fast mode: reduced cross-validation folds to 2.")

            if min_samples < n_splits:
                n_splits = max(2, min_samples)
                _report(report, f"Adjusting cross-validation splits to {n_splits} due to small class sizes")

            # Initialize cross-validation after validation and potential n_splits adjustment
            cv = (
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
                if isinstance(SPLIT, int)
                else cvDistance
            )

            x_search = x
            y_search = y
            cv_search = cv
            if fast_mode_active and isinstance(SPLIT, int) and sample_count > FAST_MODE_MAX_SAMPLES:
                tune_idx = _build_tuning_subset_indices(y, FAST_MODE_MAX_SAMPLES, random_seed)
                x_search = x[tune_idx, :]
                y_search = y[tune_idx]
                cv_search = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed + 7)
                _report(
                    report,
                    f"Fast mode: hyperparameter search subset size {x_search.shape[0]} "
                    f"(from {sample_count}). Final model is still fit on full data.",
                )

            # Check if Optuna should be used for hyperparameter optimization
            use_optuna = extraParam.get("USE_OPTUNA", False) if extraParam else False
            optuna_trials = extraParam.get("OPTUNA_TRIALS", 100) if extraParam else 100
            if fast_mode_active and use_optuna:
                if optuna_trials > FAST_MODE_MAX_OPTUNA_TRIALS:
                    optuna_trials = FAST_MODE_MAX_OPTUNA_TRIALS
                    _report(
                        report,
                        f"Fast mode: reduced Optuna trials to {optuna_trials}.",
                    )

            if use_optuna and OPTUNA_AVAILABLE:
                # Use Optuna for faster hyperparameter optimization
                _report(
                    report,
                    f"Using Optuna optimization with {optuna_trials} trials (2-10x faster than GridSearchCV)...",
                )

                try:
                    # Get classifier code from classifier name
                    classifier_code = classifier_code_input

                    # Create Optuna optimizer
                    optimizer = OptunaOptimizer(
                        classifier_code=classifier_code,
                        n_trials=optuna_trials,
                        random_seed=random_seed,
                        verbose=False,
                    )

                    # Run optimization
                    best_params = optimizer.optimize(X=x_search, y=y_search, cv=cv_search, scoring="f1_weighted")
                    selected_hyperparameters = dict(best_params)
                    optimization_method = "optuna"

                    _report(
                        report,
                        f"Optuna optimization completed. Best score: {optimizer.study.best_value:.4f}",
                    )
                    _report(report, f"Best parameters: {best_params}")

                    # Recreate classifier with best parameters
                    if classifier_code == "GMM":
                        from . import gmm_ridge

                        model = gmm_ridge.ridge()
                    elif classifier_code == "RF":
                        from sklearn.ensemble import RandomForestClassifier

                        model = RandomForestClassifier(**best_params)
                    elif classifier_code == "SVM":
                        from sklearn.svm import SVC

                        model = SVC(**best_params, probability=True)
                    elif classifier_code == "KNN":
                        from sklearn.neighbors import KNeighborsClassifier

                        model = KNeighborsClassifier(**best_params)
                    elif classifier_code == "XGB":
                        xgb_wrapper = _get_xgboost_wrapper()
                        if xgb_wrapper is None:
                            raise ImportError(
                                "XGBoost requires a usable scikit-learn runtime for label encoding"
                            )
                        model = xgb_wrapper(**best_params)
                    elif classifier_code == "LGB":
                        lgb_wrapper = _get_lightgbm_wrapper()
                        if lgb_wrapper is None:
                            raise ImportError(
                                "LightGBM requires a usable scikit-learn runtime for label encoding"
                            )
                        model = lgb_wrapper(**best_params)
                    elif classifier_code == "CB":
                        cb_wrapper = _get_catboost_wrapper()
                        if cb_wrapper is None:
                            raise ImportError(
                                "CatBoost requires a usable scikit-learn runtime for label encoding"
                            )
                        model = cb_wrapper(**best_params)
                    elif classifier_code == "ET":
                        from sklearn.ensemble import ExtraTreesClassifier

                        model = ExtraTreesClassifier(**best_params)
                    elif classifier_code == "GBC":
                        from sklearn.ensemble import GradientBoostingClassifier

                        model = GradientBoostingClassifier(**best_params)
                    elif classifier_code == "LR":
                        from sklearn.linear_model import LogisticRegression

                        model = LogisticRegression(**best_params)
                    elif classifier_code == "NB":
                        from sklearn.naive_bayes import GaussianNB

                        model = GaussianNB(**best_params)
                    elif classifier_code == "MLP":
                        from sklearn.neural_network import MLPClassifier

                        model = MLPClassifier(**best_params)
                    else:
                        _report(
                            report,
                            f"Unknown classifier code: {classifier_code}. Falling back to GridSearchCV.",
                        )
                        use_optuna = False

                    if use_optuna:
                        # Fit final model with best parameters
                        model.fit(x, y)
                        _report(report, "Final model training completed with Optuna parameters")

                except Exception as e:
                    _report(report, f"Optuna optimization failed: {e!s}. Falling back to GridSearchCV.")
                    use_optuna = False

            elif use_optuna and not OPTUNA_AVAILABLE:
                _report(
                    report,
                    "Optuna is not installed. Falling back to GridSearchCV. "
                    "Install Optuna with: pip install optuna",
                )
                use_optuna = False

            # Use GridSearchCV if Optuna is not used
            if not use_optuna:
                custom_param_grid = bool(extraParam and "param_grid" in extraParam)
                if extraParam and "param_grid" in extraParam:
                    param_grid = extraParam["param_grid"]

                    _report(report, "Custom param for Grid Search CV has been found : " + str(param_grid))
                elif fast_mode_active:
                    before = _param_grid_size(param_grid)
                    param_grid = _reduce_param_grid_for_fast_mode(param_grid, classifier_code_input)
                    after = _param_grid_size(param_grid)
                    if after < before:
                        _report(
                            report,
                            f"Fast mode: reduced hyperparameter search from {before} to {after} combinations.",
                        )

                # Provide feedback about potentially long training time for SVM
                if classifier == "SVM":
                    _report(report, "Training SVM with GridSearchCV - this may take several minutes...")

                try:
                    grid = GridSearchCV(classifier, param_grid=param_grid, cv=cv_search)
                    grid.fit(x_search, y_search)

                    _report(report, "GridSearchCV completed, fitting final model...")
                    model = grid.best_estimator_
                    model.fit(x, y)
                    if hasattr(grid, "best_params_"):
                        selected_hyperparameters = dict(grid.best_params_)
                    optimization_method = "grid_search"
                except MemoryError:
                    _report(
                        report,
                        "Memory error during training. Try reducing the image size or using fewer training samples.",
                    )
                    if feedback == "gui":
                        progress.reset()
                    return None
                except ValueError as e:
                    _report(
                        report,
                        "Data validation error: "
                        + str(e)
                        + ". Check your training data for issues like empty classes or invalid values.",
                    )
                    if feedback == "gui":
                        progress.reset()
                    return None
                except Exception as e:
                    _report(report, "Training error: " + str(e))
                    if feedback == "gui":
                        progress.reset()
                    return None

            if isinstance(SPLIT, str):
                CM = []
                testIndex = []
                # Get saveDir from extraParam if available
                saveDir = extraParam.get("saveDir", tempfile.gettempdir())
                for train_index, test_index in cv:
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    X_pred = model.predict(X_test)
                    CM.append(confusion_matrix(y_test, X_pred))
                    testIndex.append(test_index)
                for i, _j in enumerate(CM):
                    if SPLIT == "SLOO":
                        # np.savetxt((saveDir+'matrix/'+str(distance)+'_'+str(class_field)+'_'+str(minTrain)+'_'+str(i)+'.csv'),CM[i],delimiter=',',fmt='%.d')
                        np.savetxt(
                            os.path.join(
                                saveDir,
                                "matrix/"
                                + str(distance)
                                + "_"
                                + str(class_field)
                                + "_"
                                + str(minTrain)
                                + "_"
                                + str(i)
                                + ".csv",
                            ),
                            CM[i],
                            delimiter=",",
                            fmt="%.d",
                        )
                        if otherLevel is not False:
                            otherLevelFolder = os.path.join(saveDir, "matrix/level3/")
                            if not os.path.exists(otherLevelFolder):
                                os.makedirs(otherLevelFolder)
                            bigCM = np.zeros([14, 14], dtype=np.byte)

                            arr = CM[i]
                            curLevel = otherLevel[testIndex[i]]
                            curLevel = np.sort(curLevel, axis=0)
                            for lvl in range(curLevel.shape[0]):
                                bigCM[
                                    curLevel.astype(int) - 1,
                                    curLevel[lvl].astype(int) - 1,
                                ] = arr[:, lvl].reshape(-1, 1)
                            np.savetxt(
                                os.path.join(
                                    otherLevelFolder,
                                    str(distance)
                                    + "_"
                                    + str(class_field)
                                    + "_"
                                    + str(minTrain)
                                    + "_"
                                    + str(i)
                                    + ".csv",
                                ),
                                bigCM,
                                delimiter=",",
                                fmt="%.d",
                            )

                    elif SPLIT == "STAND":
                        # np.savetxt((saveDir+'matrix/stand_'+str(class_field)+'_'+str(i)+'.csv'),CM[i],delimiter=',',fmt='%.d')
                        np.savetxt(
                            os.path.join(
                                saveDir,
                                "matrix/stand_" + str(class_field) + "_" + str(i) + ".csv",
                            ),
                            CM[i],
                            delimiter=",",
                            fmt="%.d",
                        )

        _report(report, int(9 * total))

        # Assess the quality of the model
        if feedback == "gui":
            progress.prgBar.setValue(90)

        if (vector_test_path or isinstance(SPLIT, int)) and (SPLIT != 100 or vector_test_path):
            # from sklearn.metrics import cohen_kappa_score,accuracy_score,f1_score
            # if classifier == 'GMM':
            #          = model.predict(xt)[0]
            # else:
            yp = model.predict(xt)
            CONF = ai.ConfusionMatrix()
            CONF.compute_confusion_matrix(yp, yt)

            if matrix_path is not None:
                if not os.path.exists(os.path.dirname(matrix_path)):
                    os.makedirs(os.path.dirname(matrix_path))
                np.savetxt(
                    matrix_path,
                    CONF.confusion_matrix,
                    delimiter=",",
                    header="Columns=prediction,Lines=reference.",
                    fmt="%1.4d",
                )

            if classifier != "GMM":
                for key in param_grid:
                    message = "best " + key + " : " + str(grid.best_params_[key])
                    _report(report, message)

            """
                self.kappa = cohen_kappa_score(yp,yt)
                self.f1 = f1_score(yp,yt,average='micro')
                self.oa = accuracy_score(yp,yt)
                """
            res = {
                "Overall Accuracy": CONF.OA,
                "Kappa": CONF.Kappa,
                "f1": CONF.F1mean,
            }

            for estim in res:
                _report(report, estim + " : " + str(res[estim]))

            if extraParam.get("GENERATE_REPORT_BUNDLE", False):
                report_dir = str(extraParam.get("REPORT_OUTPUT_DIR", "")).strip()
                if not report_dir:
                    report_dir = tempfile.mkdtemp(prefix="dzetsaka_report_")
                yt_report = _labels_to_1d(yt)
                yp_report = _labels_to_1d(yp)
                if yt_report.size != yp_report.size:
                    n = min(int(yt_report.size), int(yp_report.size))
                    _report(
                        report,
                        "Warning: report labels length mismatch; truncating to "
                        f"{n} samples (y_true={yt_report.size}, y_pred={yp_report.size}).",
                    )
                    yt_report = yt_report[:n]
                    yp_report = yp_report[:n]

                labels_order = np.unique(np.concatenate((yt_report, yp_report))).tolist()
                cm_report = _confusion_matrix_from_labels(yt_report, yp_report, labels_order)
                manual_map = _parse_label_map_text(str(extraParam.get("REPORT_LABEL_MAP", "")))
                label_column = str(extraParam.get("REPORT_LABEL_COLUMN", "")).strip()
                label_name_map = _build_label_name_map(vector_path, class_field, label_column, manual_map)
                class_names = [label_name_map.get(_value_key(v), str(v)) for v in labels_order]
                metrics = _compute_metrics_from_cm(cm_report)
                metrics["overall_accuracy_conf"] = CONF.OA
                metrics["kappa_conf"] = CONF.Kappa
                metrics["f1_mean_conf"] = CONF.F1mean
                config_meta = {
                    "classifier_code": classifier_code_input,
                    "classifier_name": classifier_config.get_classifier_name(classifier_code_input),
                    "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "split_mode": str(extraParam.get("CV_MODE", "RANDOM_SPLIT")),
                    "split_config": SPLIT,
                    "class_field": class_field,
                    "vector_path": vector_path,
                    "raster_path": raster_path,
                    "optimization_method": optimization_method,
                    "best_hyperparameters": selected_hyperparameters,
                    "matrix_path": matrix_path,
                }
                _write_report_bundle(
                    report_dir=report_dir,
                    cm=cm_report,
                    class_values=labels_order,
                    class_names=class_names,
                    summary_metrics=metrics,
                    config_meta=config_meta,
                    y_true=yt_report,
                    y_pred=yp_report,
                )
                _report(report, f"Detailed report bundle saved to: {report_dir}")
                open_report = bool(extraParam.get("OPEN_REPORT_IN_BROWSER", True))
                if open_report:
                    report_html = Path(report_dir) / "classification_report.html"
                    if report_html.exists():
                        with contextlib.suppress(Exception):
                            webbrowser.open(report_html.resolve().as_uri())
                            _report(report, f"Opened report in browser: {report_html}")

        # Update progress after model training completion
        if feedback == "gui":
            progress.prgBar.setValue(95)
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgress(95)

        # Save Tree model
        self.model = model
        self.M = M
        self.m = m

        # SHAP explainability computation (optional)
        if extraParam.get("COMPUTE_SHAP", False):
            self._compute_shap_importance(
                model=model,
                raster_path=raster_path,
                X_train=x if 'x' in locals() else X,
                feature_names=None,  # Will generate Band_1, Band_2, etc.
                shap_output_path=extraParam.get("SHAP_OUTPUT"),
                sample_size=extraParam.get("SHAP_SAMPLE_SIZE", 1000),
            )

        if model_path is not None:
            # Debug: log what we're saving
            _report(report, f"Saving model with classifier: {classifier} (type: {type(classifier)})")
            with open(model_path, "wb") as output:
                pickle.dump([model, M, m, str(classifier)], output)  # Ensure classifier is saved as string

        _report(report, int(10 * total))
        if feedback == "gui":
            progress.reset()
            progress = None
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgress(100)

    def scale(self, x, M=None, m=None):
        """Standardize the data using min-max scaling to [-1, 1] range.

        Parameters
        ----------
        x : np.ndarray
            The data array of shape (n_samples, n_features).
        M : np.ndarray, optional
            The max vector (unused, computed from x).
        m : np.ndarray, optional
            The min vector (unused, computed from x).

        Returns
        -------
        xs : np.ndarray
            The standardized data.
        M : np.ndarray
            The max vector.
        m : np.ndarray
            The min vector.

        """
        if np.float64 != x.dtype.type:
            x = x.astype("float")

        # Get the parameters of the scaling
        M, m = np.amax(x, axis=0), np.amin(x, axis=0)

        # Vectorized scaling: avoid division by zero with safe denominator
        den = M - m
        den_safe = np.where(den != 0, den, 1.0)

        # Vectorized computation across all columns at once
        xs = 2.0 * (x - m) / den_safe - 1.0

        # Restore original values for columns with zero range
        zero_range_mask = den == 0
        if np.any(zero_range_mask):
            xs[:, zero_range_mask] = x[:, zero_range_mask]

        return xs, M, m

    def _compute_shap_importance(
        self,
        model: Any,
        raster_path: Union[str, np.ndarray],
        X_train: np.ndarray,
        feature_names: Optional[List[str]],
        shap_output_path: Optional[str],
        sample_size: int,
    ) -> None:
        """Compute SHAP feature importance and optionally create raster output.

        Parameters
        ----------
        model : Any
            Trained classifier model
        raster_path : str or np.ndarray
            Path to raster or numpy array
        X_train : np.ndarray
            Training data for background samples
        feature_names : List[str], optional
            Names of features (bands)
        shap_output_path : str, optional
            Path to save importance raster
        sample_size : int
            Number of samples for SHAP computation
        """
        report = self.report
        if not SHAP_AVAILABLE:
            _report(report, "SHAP is not installed. Install with: pip install shap>=0.41.0")
            return

        try:
            _report(report, "Computing SHAP feature importance...")

            # Generate feature names if not provided
            n_features = X_train.shape[1]
            if feature_names is None:
                feature_names = [f"Band_{i+1}" for i in range(n_features)]

            # Create ModelExplainer
            explainer = ModelExplainer(
                model=model,
                feature_names=feature_names,
                background_data=X_train[:min(100, len(X_train))],  # Use up to 100 samples as background
            )

            # Compute feature importance on training sample
            sample_for_shap = X_train[:min(sample_size, len(X_train))]
            importance = explainer.get_feature_importance(
                X_sample=sample_for_shap,
                aggregate_method="mean_abs",
            )

            # Log importance scores
            _report(report, "\nFeature Importance (SHAP values):")
            for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
                _report(report, f"  {feat}: {score:.4f}")

            # Store importance in model metadata
            self.feature_importance = importance

            # Create importance raster if output path provided and raster_path is a file
            if shap_output_path and isinstance(raster_path, str):
                _report(report, f"Generating feature importance raster: {shap_output_path}")

                # Create progress callback for SHAP raster generation
                def shap_progress_callback(pct):
                    if self.report.feedback is not None and hasattr(self.report.feedback, "setProgress"):
                        # Map SHAP progress (0-100) to remaining progress window
                        self.report.feedback.setProgress(int(95 + pct * 0.05))

                explainer.create_importance_raster(
                    raster_path=raster_path,
                    output_path=shap_output_path,
                    sample_size=sample_size,
                    progress_callback=shap_progress_callback if self.report.feedback else None,
                )

                _report(report, f"Feature importance raster saved to: {shap_output_path}")

        except Exception as e:
            _report(report, f"Warning: SHAP computation failed: {e!s}\nContinuing without SHAP analysis.")
            # Don't raise - continue with normal workflow

    def _handle_class_imbalance(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        classifier: str,
        extraParam: Dict[str, Any],
    ) -> None:
        """Analyze class imbalance and log recommendations.

        Parameters
        ----------
        X : np.ndarray
            Training features
        Y : np.ndarray
            Training labels
        classifier : str
            Classifier code
        extraParam : dict
            Extra parameters
        """
        report = self.report
        if not SAMPLING_AVAILABLE:
            return

        # Check imbalance ratio
        unique, counts = np.unique(Y, return_counts=True)
        if len(unique) < 2:
            return

        ratio = counts.max() / counts.min()

        # Log class distribution
        _report(report, "Class distribution:")
        for cls, count in zip(unique, counts):
            pct = (count / len(Y)) * 100
            _report(report, f"  Class {int(cls)}: {int(count)} samples ({pct:.1f}%)")
        _report(report, f"  Imbalance ratio: {ratio:.2f}")

        # Recommend strategy if not already configured
        if not extraParam.get("USE_SMOTE", False) and not extraParam.get("USE_CLASS_WEIGHTS", False):
            strategy = recommend_strategy(Y)
            if strategy == "smote":
                _report(
                    report,
                    "Warning: Dataset is severely imbalanced. "
                    "Consider enabling USE_SMOTE=True or USE_CLASS_WEIGHTS=True.",
                )
            elif strategy == "class_weights":
                _report(
                    report,
                    "Note: Dataset is moderately imbalanced. "
                    "Consider enabling USE_CLASS_WEIGHTS=True for better performance.",
                )

    def _apply_smote(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        extraParam: Dict[str, Any],
    ) -> tuple:
        """Apply SMOTE oversampling if conditions are met.

        Parameters
        ----------
        X : np.ndarray
            Training features
        Y : np.ndarray
            Training labels
        extraParam : dict
            Extra parameters
        Returns
        -------
        X_resampled : np.ndarray
            Resampled features
        Y_resampled : np.ndarray
            Resampled labels

        """
        report = self.report
        if not SAMPLING_AVAILABLE or not IMBLEARN_AVAILABLE:
            _report(
                report,
                "Warning: SMOTE requires imbalanced-learn. Install with: pip install imbalanced-learn>=0.10.0",
            )
            return X, Y

        k_neighbors = extraParam.get("SMOTE_K_NEIGHBORS", 5)

        try:
            _report(report, "Applying SMOTE oversampling...")
            original_size = len(Y)

            sampler = SMOTESampler(k_neighbors=k_neighbors, random_state=0)
            X_resampled, Y_resampled = sampler.fit_resample(X, Y)

            new_size = len(Y_resampled)
            _report(
                report,
                f"SMOTE complete: {original_size} -> {new_size} samples ({new_size - original_size} synthetic)",
            )

            return X_resampled, Y_resampled

        except Exception as e:
            _report(report, f"Warning: SMOTE failed: {e!s}. Continuing with original data.")
            return X, Y

    def _compute_weights(
        self,
        Y: np.ndarray,
        extraParam: Dict[str, Any],
    ) -> Optional[Dict[int, float]]:
        """Compute class weights for cost-sensitive learning.

        Parameters
        ----------
        Y : np.ndarray
            Training labels
        extraParam : dict
            Extra parameters
        Returns
        -------
        weights : dict or None
            Class weights or None if computation fails

        """
        if not SAMPLING_AVAILABLE:
            return None

        strategy = extraParam.get("CLASS_WEIGHT_STRATEGY", "balanced")
        custom_weights = extraParam.get("CUSTOM_CLASS_WEIGHTS", None)

        try:
            weights = compute_class_weights(Y, strategy=strategy, custom_weights=custom_weights)

            _report(self.report, "Class weights computed:")
            for cls, weight in sorted(weights.items()):
                _report(self.report, f"  Class {cls}: {weight:.4f}")

            return weights

        except Exception as e:
            _report(
                self.report,
                f"Warning: Class weight computation failed: {e!s}. "
                "Continuing without class weights.",
            )
            return None

    def _validate_inputs(
        self,
        raster_path: Union[str, np.ndarray],
        vector_path: Union[str, np.ndarray],
        classifier: str,
        feedback,
    ) -> None:
        """Validate input parameters."""
        valid_classifiers = classifier_config.CLASSIFIER_CODES
        if classifier not in valid_classifiers:
            raise ValueError(f"Invalid classifier: {classifier}. Must be one of {valid_classifiers}")

        if isinstance(raster_path, np.ndarray) and not isinstance(vector_path, np.ndarray):
            msg = "You have to give an array for labels when using array for raster"
            _report(self.report, msg)
            raise ValueError(msg)

    def _setup_progress_feedback(self, feedback):
        """Setup progress feedback based on feedback type."""
        if feedback == "gui":
            return progress_bar.ProgressBar("Loading...", 100)
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgressText("Loading...")
            feedback.setProgress(0)
            return None
        return None

    def _load_and_prepare_data(
        self,
        raster_path: Union[str, np.ndarray],
        vector_path: Union[str, np.ndarray],
        class_field: str,
        split_config: Union[int, float, str],
        extraParam: Dict[str, Any],
        feedback,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[str],
    ]:
        """Load and prepare training data."""
        needXY = True
        coords = None
        distanceArray = None
        STDs = None
        vector_test_path = None

        _report(self.report, "Learning model...")
        _report(self.report, 0)

        # Handle numpy array inputs
        if isinstance(raster_path, np.ndarray):
            needXY = False
            X = raster_path
            if isinstance(vector_path, np.ndarray):
                Y = vector_path
            else:
                raise ValueError("Label array required when using raster array")
        else:
            # Handle vector inputs and extra parameters
            X, Y = None, None

            # Setup save directory if specified
            if "saveDir" in extraParam:
                self._setup_save_directory(extraParam["saveDir"])

            # Check for test vector
            if isinstance(split_config, str) and split_config.endswith((".shp", ".sqlite")):
                vector_test_path = split_config

            # Handle special ROI reading
            if extraParam.get("readROIFromVector", False):
                X, Y = self._read_roi_from_vector(vector_path, extraParam, class_field, feedback)
                needXY = False
                coords = extraParam.get("coords")

            # Standard rasterization approach
            if needXY:
                ROI = rasterize(raster_path, vector_path, class_field)

                if split_config == "SLOO":
                    X, Y, coords, distanceArray = self._prepare_sloo_data(raster_path, ROI, extraParam, feedback)
                elif split_config == "STAND":
                    X, Y, STDs = self._prepare_stand_data(
                        raster_path, vector_path, ROI, class_field, extraParam, feedback
                    )
                else:
                    X, Y = dataraster.get_samples_from_roi(raster_path, ROI)

                # Handle test vector if specified
                if vector_test_path:
                    ROIt = rasterize(raster_path, vector_test_path, class_field)
                    Xt, yt = dataraster.get_samples_from_roi(raster_path, ROIt)
                    # Store test data for later use
                    self._test_data = (Xt, yt)

        return X, Y, coords, distanceArray, STDs, vector_test_path

    def _setup_save_directory(self, saveDir: str) -> None:
        """Create save directory and subdirectories."""
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        matrix_dir = os.path.join(saveDir, "matrix/")
        if not os.path.exists(matrix_dir):
            os.makedirs(matrix_dir)

    def _read_roi_from_vector(self, vector_path, extraParam, class_field, feedback):
        """Read ROI data from vector using custom function."""
        try:
            from function_vector import readROIFromVector

            return readROIFromVector(vector_path, extraParam["readROIFromVector"], class_field)
        except ImportError:
            msg = "Problem when importing readFieldVector from functions in dzetsaka"
            _report(self.report, msg)
            raise

    def _prepare_sloo_data(self, raster_path, ROI, extraParam, feedback):
        """Prepare data for Spatial Leave-One-Out cross-validation."""
        try:
            from function_vector import distMatrix
        except ImportError:
            from .function_vector import distMatrix

        if extraParam.get("readROIFromVector", False):
            coords = extraParam.get("coords")
            if coords is None:
                _report(self.report, "Can't read coords array")
                raise ValueError("Coordinates not found in extraParam")
            X, Y = None, None  # Will be set elsewhere
        else:
            X, Y, coords = dataraster.get_samples_from_roi(raster_path, ROI, getCoords=True)

        distanceArray = distMatrix(coords)
        return X, Y, coords, distanceArray

    def _prepare_stand_data(self, raster_path, vector_path, ROI, class_field, extraParam, feedback):
        """Prepare data for stand-based cross-validation."""
        inStand = extraParam.get("inStand", "stand")
        STAND = rasterize(raster_path, vector_path, inStand)
        X, Y, STDs = dataraster.get_samples_from_roi(raster_path, ROI, STAND)
        return X, Y, STDs

    def _handle_data_loading_error(self, error: Exception, class_field: str, feedback, progress) -> None:
        """Handle data loading errors with appropriate error messages."""
        if isinstance(error, ValueError) and ("could not convert" in str(error) or "invalid literal" in str(error)):
            msg = (
                f"Data type error: Unable to convert class values to numbers.\n"
                f"Please ensure your {class_field} field contains only integer values (1, 2, 3, etc.)\n"
                f"Error details: {error}"
            )
        else:
            msg = (
                f"Problem with getting samples from ROI: {error}\n"
                "Common causes:\n"
                "- Shapefile and raster have different projections\n"
                "- Invalid geometry in shapefile\n"
                f"- Field '{class_field}' contains non-numeric values\n"
                "- Memory issues with large datasets"
            )

        _report(self.report, msg)
        if progress and hasattr(progress, "reset"):
            progress.reset()


class ClassifyImage:
    """Classify a raster image from a trained model and optional mask."""
    def initPredict(
        self,
        raster_path: Optional[str] = None,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        confidenceMap: Optional[str] = None,
        confidenceMapPerClass: Optional[str] = None,
        NODATA: int = 0,
        feedback=None,
    ) -> Optional[str]:
        """Initialize prediction with raster and model paths."""
        if not raster_path:
            raise ValueError("raster_path is required")
        if not model_path:
            raise ValueError("model_path is required")
        if not output_path:
            raise ValueError("output_path is required")

        self.report = Reporter.from_feedback(feedback, tag=LOG_TAG)

        # Load model
        try:
            tree, M, m, classifier = self._load_model(model_path)
        except FileNotFoundError as e:
            _report(
                self.report,
                f"Model file not found: {model_path}\n"
                f"Please check that the file exists and the path is correct.\n"
                f"Error details: {e}",
            )
            return None
        except (pickle.UnpicklingError, pickle.PickleError) as e:
            _report(
                self.report,
                f"Model file is corrupted or incompatible: {model_path}\n"
                f"The model file may have been created with a different version of dzetsaka or Python.\n"
                f"Try retraining your model or use a different model file.\n"
                f"Error details: {e}",
            )
            return None
        except ValueError as e:
            _report(
                self.report,
                f"Invalid model file format: {model_path}\n"
                f"The model file structure is not recognized by dzetsaka.\n"
                f"Error details: {e}",
            )
            return None
        except Exception as e:
            error_details = f"Unexpected error while loading model {model_path}\n"
            error_details += f"Error type: {type(e).__name__}\n"
            error_details += f"Error details: {e}\n"
            error_details += "Please check the QGIS log for more details and consider reporting this issue."

            _report(self.report, error_details)

            # Show GitHub issue popup for unexpected errors
            if feedback == "gui":
                self._show_github_issue_popup(
                    "Model Loading Error",
                    f"Error type: {type(e).__name__}",
                    str(e),
                    f"Model path: {model_path}",
                )
            return None

        # Create temporary directory for processing
        try:
            temp_folder = tempfile.mkdtemp()
            os.path.join(temp_folder, "temp.tif")
        except Exception as e:
            _report(self.report, f"Cannot create temp file: {e}")
            return None
            # Process the data
        # Validate model components
        if not all(var is not None for var in [tree, M, m, classifier]):
            _report(self.report, "Model variables not properly loaded")
            return None
        # try:
        predictedImage = self.predict_image(
            raster_path,
            output_path,
            tree,
            mask_path,
            confidenceMap,
            confidenceMapPerClass=confidenceMapPerClass,
            NODATA=NODATA,
            SCALE=[M, m],
            classifier=classifier,
            feedback=feedback,
        )
        # except:
        #   QgsMessageLog.logMessage("Problem while predicting "+raster_path+" in temp"+rasterTemp)

        return predictedImage

    def _load_model(self, model_path: str) -> Tuple[Any, np.ndarray, np.ndarray, str]:
        """Load pickled model with proper error handling.

        Parameters
        ----------
        model_path : str
            Path to the pickled model file
        Returns
        -------
        tuple
            (model, M, m, classifier) where M and m are scaling parameters

        Raises
        ------
        FileNotFoundError
            If model file doesn't exist
        pickle.UnpicklingError
            If model file is corrupted

        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as model_file:
                model_data = pickle.load(model_file)

            # Validate model data structure
            if not isinstance(model_data, (list, tuple)) or len(model_data) != 4:
                raise ValueError("Invalid model file format. Expected 4 components.")

            tree, M, m, classifier = model_data

            # Debug: log what we loaded
            _report(
                self.report,
                f"Loaded model data: tree={type(tree)}, M={type(M)}, m={type(m)}, classifier='{classifier}' (type: {type(classifier)})",
            )

            # Basic validation of components
            if tree is None:
                raise ValueError("Model is None")
            if not isinstance(M, np.ndarray) or not isinstance(m, np.ndarray):
                raise ValueError("Scaling parameters M and m must be numpy arrays")
            if not isinstance(classifier, str):
                raise ValueError(f"Classifier must be a string, got {type(classifier)}: {classifier}")

            return tree, M, m, classifier

        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Corrupted model file: {e}") from e
        except Exception as e:
            raise Exception(f"Failed to load model: {e}") from e

    def scale(
        self,
        x: np.ndarray,
        M: Optional[np.ndarray] = None,
        m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Standardize the data using min-max scaling to [-1, 1] range.

        Parameters
        ----------
        x : np.ndarray
            The data array of shape (n_samples, n_features).
        M : np.ndarray, optional
            The max vector. If None, computed from x.
        m : np.ndarray, optional
            The min vector. If None, computed from x.

        Returns
        -------
        xs : np.ndarray
            The standardized data.

        """
        if np.float64 != x.dtype.type:
            x = x.astype("float")

        # Get the parameters of the scaling
        if M is None:
            M, m = np.amax(x, axis=0), np.amin(x, axis=0)

        # Vectorized scaling: avoid division by zero with safe denominator
        den = M - m
        den_safe = np.where(den != 0, den, 1.0)

        # Vectorized computation across all columns at once
        xs = 2.0 * (x - m) / den_safe - 1.0

        # Restore original values for columns with zero range
        zero_range_mask = den == 0
        if np.any(zero_range_mask):
            xs[:, zero_range_mask] = x[:, zero_range_mask]

        return xs

    def predict_image(
        self,
        raster_path: str,
        output_path: str,
        model=None,
        mask_path: Optional[str] = None,
        confidenceMap: Optional[str] = None,
        confidenceMapPerClass: Optional[str] = None,
        NODATA: int = 0,
        SCALE: Optional[List[np.ndarray]] = None,
        classifier: str = "GMM",
        feedback=None,
    ) -> str:
        """Classify the whole raster image using per-block image analysis.

        Parameters
        ----------
        raster_path : str
            Input raster image path
        output_path : str
            Output classification raster path
        model : object
            Trained classification model
        mask_path : str, optional
            Mask raster path
        confidenceMap : str, optional
            Confidence map output path
        confidenceMapPerClass : str, optional
            Per-class confidence map output path
        NODATA : int, default=0
            No data value
        SCALE : list of np.ndarray, optional
            Scaling parameters [M, m]
        classifier : str, default="GMM"
            Classifier type
        feedback : object, optional
            Feedback interface

        Returns
        -------
        str
            Path to output raster

        """
        # Open Raster and get additional information
        raster = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if raster is None:
            raise RuntimeError(f"Cannot open raster: {raster_path}")

        if mask_path is None:
            mask = None
        else:
            mask = gdal.Open(mask_path, gdal.GA_ReadOnly)
            if mask is None:
                raise RuntimeError(f"Cannot open mask: {mask_path}")
            # Check size
            if (raster.RasterXSize != mask.RasterXSize) or (raster.RasterYSize != mask.RasterYSize):
                # fix_print_with_import
                print("Image and mask should be of the same size")
                exit()
        if SCALE is not None:
            M, m = np.asarray(SCALE[0]), np.asarray(SCALE[1])

        # Get the size of the image
        d = raster.RasterCount
        nc = raster.RasterXSize
        nl = raster.RasterYSize

        # Provide feedback for multi-band images
        if d > 3:
            _report(
                self.report,
                f"Processing {d}-band image. This may take longer than standard RGB images.",
            )

        # Optimize block size for memory efficiency
        x_block_size, y_block_size = self._calculate_optimal_block_size(raster, d, feedback)

        # Get the geoinformation
        GeoTransform = raster.GetGeoTransform()
        Projection = raster.GetProjection()

        # Initialize the output
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        driver = gdal.GetDriverByName("GTiff")

        dtype = gdal.GDT_UInt16 if np.amax(model.classes_) > 255 else gdal.GDT_Byte

        dst_ds = driver.Create(output_path, nc, nl, 1, dtype)
        dst_ds.SetGeoTransform(GeoTransform)
        dst_ds.SetProjection(Projection)
        out = dst_ds.GetRasterBand(1)

        if classifier != "GMM":
            nClass = len(model.classes_)
        if confidenceMap:
            dst_confidenceMap = driver.Create(confidenceMap, nc, nl, 1, gdal.GDT_Int16)
            dst_confidenceMap.SetGeoTransform(GeoTransform)
            dst_confidenceMap.SetProjection(Projection)
            out_confidenceMap = dst_confidenceMap.GetRasterBand(1)

        if confidenceMapPerClass:
            dst_confidenceMapPerClass = driver.Create(confidenceMapPerClass, nc, nl, nClass, gdal.GDT_Int16)
            dst_confidenceMapPerClass.SetGeoTransform(GeoTransform)
            dst_confidenceMapPerClass.SetProjection(Projection)

        # Perform the classification

        total = nl

        if d > 3:
            _report(self.report, f"Predicting model for {d}-band image...")
        else:
            _report(self.report, "Predicting model...")

        if feedback == "gui":
            progress_text = f"Predicting model ({d} bands)..." if d > 3 else "Predicting model..."
            progress = progress_bar.ProgressBar(progress_text, 100)
        elif feedback is not None and hasattr(feedback, "setProgress"):
            # Handle batch processing feedback
            progress_text = f"Predicting model for {d}-band image..." if d > 3 else "Predicting model..."
            feedback.setProgressText(progress_text)
            feedback.setProgress(0)

        for i in range(0, nl, y_block_size):
            if "lastBlock" not in locals():
                lastBlock = i
            if int(lastBlock / total * 100) != int(i / total * 100):
                lastBlock = i
                pct = int(i / total * 100)
                _report(self.report, pct)
                if feedback == "gui":
                    progress.prgBar.setValue(pct)

            lines = y_block_size if i + y_block_size < nl else nl - i  # Check for size consistency in Y
            for j in range(0, nc, x_block_size):  # Check for size consistency in X
                cols = x_block_size if j + x_block_size < nc else nc - j

                # Load the data efficiently
                X = self._load_block_data(raster, d, j, i, cols, lines, feedback)
                if X is None:
                    continue

                # Do the prediction
                band_temp = raster.GetRasterBand(1)
                nodata_temp = band_temp.GetNoDataValue()
                if nodata_temp is None:
                    nodata_temp = -9999

                if mask is None:
                    band_temp = raster.GetRasterBand(1)
                    mask_temp = band_temp.ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                    temp_nodata = np.where(mask_temp != nodata_temp)[0]
                    # t = np.where((mask_temp!=0) & (X[:,0]!=NODATA))[0]
                    t = np.where(X[:, 0] != nodata_temp)[0]
                    yp = np.zeros((cols * lines,))
                    # K = np.zeros((cols*lines,))
                    if confidenceMapPerClass or (confidenceMap and classifier != "GMM"):
                        K = np.zeros((cols * lines, nClass))
                        K[:, :] = -1
                    else:
                        K = np.zeros(cols * lines)
                        K[:] = -1

                else:
                    mask_temp = mask.GetRasterBand(1).ReadAsArray(j, i, cols, lines).reshape(cols * lines)
                    t = np.where((mask_temp != 0) & (X[:, 0] != nodata_temp))[0]
                    yp = np.zeros((cols * lines,))
                    yp[:] = NODATA
                    # K = np.zeros((cols*lines,))
                    if confidenceMapPerClass or (confidenceMap and classifier != "GMM"):
                        K = np.ones((cols * lines, nClass))
                        K = np.negative(K)
                    else:
                        K = np.zeros(cols * lines)
                        K = np.negative(K)

                if t.size > 0:
                    if confidenceMap and classifier == "GMM":
                        yp[t], K[t] = model.predict(self.scale(X[t, :], M=M, m=m), None, confidenceMap)

                    elif confidenceMap or (confidenceMapPerClass and classifier != "GMM"):
                        yp[t] = model.predict(self.scale(X[t, :], M=M, m=m))
                        K[t, :] = model.predict_proba(self.scale(X[t, :], M=M, m=m)) * 100

                    else:
                        yp[t] = model.predict(self.scale(X[t, :], M=M, m=m))

                        # QgsMessageLog.logMessage('amax from predict proba is : '+str(sp.amax(model.predict.proba(self.scale(X[t,:],M=M,m=m)),axis=1)))

                # Write the data
                out.WriteArray(yp.reshape(lines, cols), j, i)
                out.SetNoDataValue(NODATA)
                out.FlushCache()

                if confidenceMap and classifier == "GMM":
                    K *= 100
                    out_confidenceMap.WriteArray(K.reshape(lines, cols), j, i)
                    out_confidenceMap.SetNoDataValue(-1)
                    out_confidenceMap.FlushCache()

                if confidenceMap and classifier != "GMM":
                    Kconf = np.amax(K, axis=1)
                    out_confidenceMap.WriteArray(Kconf.reshape(lines, cols), j, i)
                    out_confidenceMap.SetNoDataValue(-1)
                    out_confidenceMap.FlushCache()

                if confidenceMapPerClass and classifier != "GMM":
                    for band in range(nClass):
                        gdalBand = band + 1
                        out_confidenceMapPerClass = dst_confidenceMapPerClass.GetRasterBand(gdalBand)
                        out_confidenceMapPerClass.SetNoDataValue(-1)
                        out_confidenceMapPerClass.WriteArray(K[:, band].reshape(lines, cols), j, i)
                        out_confidenceMapPerClass.FlushCache()

                # Explicit memory cleanup
                del X, yp
                if "K" in locals():
                    del K

        # Clean/Close variables
        if feedback == "gui":
            progress.reset()
        elif feedback is not None and hasattr(feedback, "setProgress"):
            feedback.setProgress(100)

        raster = None
        dst_ds = None
        return output_path

    def _calculate_optimal_block_size(self, raster, num_bands: int, feedback) -> Tuple[int, int]:
        """Calculate optimal block size for memory efficiency."""
        # Get default block size
        band = raster.GetRasterBand(1)
        block_sizes = band.GetBlockSize()
        x_block_size = block_sizes[0]
        y_block_size = block_sizes[1]
        del band

        # Memory optimization for large multi-band images
        if num_bands > 3:
            pixel_size_bytes = 8 * num_bands  # Assume 8 bytes per pixel per band
            max_pixels_per_block = (MAX_MEMORY_MB * 1024 * 1024) // pixel_size_bytes

            current_block_pixels = x_block_size * y_block_size
            if current_block_pixels > max_pixels_per_block:
                scale_factor = (max_pixels_per_block / current_block_pixels) ** 0.5
                x_block_size = max(32, int(x_block_size * scale_factor))
                y_block_size = max(32, int(y_block_size * scale_factor))
                _report(
                    self.report,
                    f"Adjusted block size to {x_block_size}x{y_block_size} for memory optimization",
                )

        return x_block_size, y_block_size

    def _load_block_data(
        self,
        raster,
        num_bands: int,
        x_offset: int,
        y_offset: int,
        cols: int,
        lines: int,
        feedback,
    ) -> Optional[np.ndarray]:
        """Load raster block data with memory optimization."""
        try:
            # Use memory-efficient data type for multi-band images
            dtype = np.float32 if num_bands > 3 else np.float64
            X = np.empty((cols * lines, num_bands), dtype=dtype)

            for band_idx in range(num_bands):
                band_data = raster.GetRasterBand(band_idx + 1).ReadAsArray(x_offset, y_offset, cols, lines)
                if band_data is None:
                    _report(self.report, f"Error reading band {band_idx + 1}")
                    return None
                X[:, band_idx] = band_data.reshape(cols * lines)

                # Free band_data immediately
                del band_data

            return X

        except MemoryError:
            _report(
                self.report,
                "Memory error loading block data. Consider reducing block size or using fewer bands.",
            )
            return None
        except Exception as e:
            _report(self.report, f"Error loading block data: {e}")
            return None

    def _show_github_issue_popup(self, error_title, error_type, error_message, context):
        """Show standardized compact issue popup."""
        try:
            show_issue_popup(
                error_title=error_title,
                error_type=error_type,
                error_message=error_message,
                context=context,
                parent=None,
            )
        except Exception as e:
            _report(self.report, f"Could not show GitHub issue popup: {e}")


class ConfusionMatrix:
    """Class for computing confusion matrix statistics from raster predictions."""

    def __init__(self):
        self.confusion_matrix: Optional[np.ndarray] = None
        self.OA: Optional[float] = None
        self.Kappa: Optional[float] = None

    def computeStatistics(
        self,
        raster_path: Optional[str] = None,
        shapefile_path: Optional[str] = None,
        class_field: Optional[str] = None,
        feedback=None,
    ) -> None:
        """Compute confusion matrix statistics.

        Parameters
        ----------
        raster_path : str
            Path to prediction raster
        shapefile_path : str
            Path to reference shapefile
        class_field : str
            Field name containing reference classes
        feedback : object, optional
            Feedback interface for progress reporting

        """
        report = Reporter.from_feedback(feedback, tag=LOG_TAG)
        if not raster_path:
            raise ValueError("raster_path is required")
        if not shapefile_path:
            raise ValueError("shapefile_path is required")
        if not class_field:
            raise ValueError("class_field is required")

        try:
            rasterized = rasterize(raster_path, shapefile_path, class_field)
            Yp, Yt = dataraster.get_samples_from_roi(raster_path, rasterized)
            CONF = ai.ConfusionMatrix()
            CONF.compute_confusion_matrix(Yp, Yt)
            self.confusion_matrix = CONF.confusion_matrix
            self.Kappa = CONF.Kappa
            self.OA = CONF.OA

            # Clean up temporary raster
            with contextlib.suppress(OSError):
                os.remove(rasterized)

        except Exception as e:
            error_msg = f"Error during statistics calculation: {e}"
            _report(report, error_msg)
            raise RuntimeError(error_msg) from e


def rasterize(
    raster_path: Optional[str] = None,
    shapefile_path: Optional[str] = None,
    class_field: Optional[str] = None,
) -> str:
    """Rasterize vector data to match raster extent and resolution.

    Parameters
    ----------
    raster_path : str
        Reference raster path
    shapefile_path : str
        Vector shapefile path
    class_field : str
        Attribute field to rasterize

    Returns
    -------
    str
        Path to temporary rasterized file

    """
    if not raster_path:
        raise ValueError("raster_path is required")
    if not shapefile_path:
        raise ValueError("shapefile_path is required")
    if not class_field:
        raise ValueError("class_field is required")

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    filename = tempfile.mktemp(".tif")

    try:
        data = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if data is None:
            raise RuntimeError(f"Cannot open raster: {raster_path}")

        shp = ogr.Open(shapefile_path)
        if shp is None:
            raise RuntimeError(f"Cannot open shapefile: {shapefile_path}")

        lyr = shp.GetLayer()
        if lyr is None:
            raise RuntimeError(f"Cannot access layer in shapefile: {shapefile_path}")

        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(filename, data.RasterXSize, data.RasterYSize, 1, gdal.GDT_UInt16)

        if dst_ds is None:
            raise RuntimeError(f"Cannot create output raster: {filename}")

        dst_ds.SetGeoTransform(data.GetGeoTransform())
        dst_ds.SetProjection(data.GetProjection())

        OPTIONS = f"ATTRIBUTE={class_field}"
        result = gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=[OPTIONS])

        if result != gdal.CE_None:
            raise RuntimeError(f"Rasterization failed for field {class_field}")

    except Exception as e:
        # Clean up on error
        if os.path.exists(filename):
            with contextlib.suppress(OSError):
                os.remove(filename)
        raise RuntimeError(f"Rasterization error: {e}") from e
    finally:
        # Ensure GDAL objects are properly closed
        data, dst_ds, shp, lyr = None, None, None, None

    return filename


if __name__ == "__main__":
    # Example using new parameter names
    RASTER_PATH = "/mnt/DATA/demo/map.tif"
    VECTOR_PATH = "/mnt/DATA/demo/train.shp"
    CLASS_FIELD = "Class"
    MODEL_PATH = "/mnt/DATA/demo/test/model.RF"
    SPLIT_PERCENT = 50
    MATRIX_PATH = "/mnt/DATA/demo/test/matrix.csv"
    CLASSIFIER_TYPE = "RF"
    CONFIDENCE_PATH = "/mnt/DATA/demo/test/confidence.tif"
    MASK_PATH = None
    OUTPUT_PATH = "/mnt/DATA/demo/test/class.tif"

    # Using new parameter names
    temp = LearnModel(
        raster_path=RASTER_PATH,
        vector_path=VECTOR_PATH,
        class_field=CLASS_FIELD,
        model_path=MODEL_PATH,
        split_config=SPLIT_PERCENT,
        random_seed=0,
        matrix_path=MATRIX_PATH,
        classifier=CLASSIFIER_TYPE,
        extraParam=None,
        feedback=None,
    )
    print("learned")

    temp = ClassifyImage()
    temp.initPredict(
        raster_path=RASTER_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        mask_path=MASK_PATH,
        confidenceMap=CONFIDENCE_PATH,
    )
    print("classified")

    # Advanced testing examples
    Test = "SLOO"

    if Test == "STAND":
        extra_param = {
            "inStand": "Stand",
            "saveDir": "/tmp/test1/",
            "maxIter": 5,
            "SLOO": False,
        }
        LearnModel(
            raster_path=RASTER_PATH,
            vector_path=VECTOR_PATH,
            class_field=CLASS_FIELD,
            model_path=MODEL_PATH,
            split_config="STAND",
            random_seed=0,
            matrix_path=None,
            classifier=CLASSIFIER_TYPE,
            feedback=None,
            extraParam=extra_param,
        )

    if Test == "SLOO":
        RASTER_PATH = "/mnt/DATA/Test/DA/SITS/SITS_2013.tif"
        VECTOR_PATH = "/mnt/DATA/Test/DA/ROI_2154.sqlite"
        CLASS_FIELD = "level1"

        extra_param = {"distance": 100, "maxIter": 5, "saveDir": "/tmp/"}
        LearnModel(
            raster_path=RASTER_PATH,
            vector_path=VECTOR_PATH,
            class_field=CLASS_FIELD,
            model_path=MODEL_PATH,
            split_config="SLOO",
            random_seed=0,
            matrix_path=None,
            classifier=CLASSIFIER_TYPE,
            feedback=None,
            extraParam=extra_param,
        )
