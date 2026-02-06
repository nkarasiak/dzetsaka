"""Output naming helpers for classification artifacts."""

from __future__ import annotations

import os


def default_output_name(in_raster_path: str, classifier_code: str) -> str:
    """Build deterministic default output filename for temporary classifications."""
    base_name = os.path.splitext(os.path.basename(in_raster_path))[0]
    code = str(classifier_code or "CLASS").strip().upper()
    if not code:
        code = "CLASS"
    return f"{base_name}_{code}.tif"

