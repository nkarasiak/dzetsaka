"""Output naming helpers for classification artifacts."""

from __future__ import annotations

import os

_CLASSIFIER_CODE_TO_LABEL = {
    "XGB": "XGBoost",
    "CB": "CatBoost",
}


def default_output_name(in_raster_path: str, classifier_code: str) -> str:
    """Build deterministic default output filename for temporary classifications."""
    base_name = os.path.splitext(os.path.basename(in_raster_path))[0]
    code = str(classifier_code or "CLASS").strip().upper()
    if not code:
        code = "CLASS"
    classifier_label = _CLASSIFIER_CODE_TO_LABEL.get(code, code)
    return f"{base_name}_{classifier_label}.tif"
