"""Raster classification use case (Phase 1 compatibility wrapper)."""

from __future__ import annotations


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
    """Execute model inference via legacy runtime implementation."""
    from dzetsaka.scripts import mainfunction

    classifier_worker = mainfunction.ClassifyImage()
    return classifier_worker.initPredict(
        raster_path=raster_path,
        model_path=model_path,
        output_path=output_path,
        mask_path=mask_path,
        confidenceMap=confidence_map,
        confidenceMapPerClass=None,
        NODATA=nodata,
        feedback=feedback,
    )

