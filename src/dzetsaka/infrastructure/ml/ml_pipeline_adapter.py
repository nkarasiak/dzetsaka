"""ML adapter backed by scripts.classification_pipeline."""

from __future__ import annotations

from dzetsaka.scripts import classification_pipeline


def learn_model(
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
    """Train a model through the classification pipeline entrypoint."""
    return classification_pipeline.LearnModel(
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


def classify_image(
    *,
    raster_path,
    model_path,
    output_path,
    mask_path,
    confidence_map,
    nodata,
    feedback,
):
    """Run prediction through the classification pipeline entrypoint."""
    classifier_worker = classification_pipeline.ClassifyImage()
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


