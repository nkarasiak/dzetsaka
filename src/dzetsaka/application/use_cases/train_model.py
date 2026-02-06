"""Train model use case."""

from __future__ import annotations

from dzetsaka.infrastructure.ml.ml_pipeline_adapter import learn_model


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
    """Execute model training via infrastructure adapter."""
    return learn_model(
        raster_path=raster_path,
        vector_path=vector_path,
        class_field=class_field,
        model_path=model_path,
        split_config=split_config,
        random_seed=random_seed,
        matrix_path=matrix_path,
        classifier=classifier,
        extra_params=extra_params,
        feedback=feedback,
    )
