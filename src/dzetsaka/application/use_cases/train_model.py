"""Train model use case (Phase 1 compatibility wrapper)."""

from __future__ import annotations


def run_training(
    *,
    raster_path,
    vector_path,
    class_field,
    model_path,
    split_config,
    random_seed,
    matrix_path,
    classifier,
    extra_params,
    feedback,
):
    """Execute model training via legacy runtime implementation."""
    from dzetsaka.scripts import mainfunction

    return mainfunction.LearnModel(
        raster_path=raster_path,
        vector_path=vector_path,
        class_field=class_field,
        model_path=model_path,
        split_config=split_config,
        random_seed=random_seed,
        matrix_path=matrix_path,
        classifier=classifier,
        extraParam=extra_params,
        feedback=feedback,
    )

