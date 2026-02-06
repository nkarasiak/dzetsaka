"""Raster classification use case."""

from __future__ import annotations

from dzetsaka.infrastructure.ml.ml_pipeline_adapter import classify_image


def run_classification(
    *,
    raster_path,
    model_path,
    output_path,
    mask_path,
    confidence_map,
    nodata,
    feedback,
):
    """Execute model inference via infrastructure adapter."""
    return classify_image(
        raster_path=raster_path,
        model_path=model_path,
        output_path=output_path,
        mask_path=mask_path,
        confidence_map=confidence_map,
        nodata=nodata,
        feedback=feedback,
    )
