"""Bridge to new use-case modules with safe fallback to legacy code."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dzetsaka.scripts import mainfunction


def _load_callable(file_name: str, function_name: str):
    root_dir = Path(__file__).resolve().parents[1]
    module_path = root_dir / "src" / "dzetsaka" / "application" / "use_cases" / file_name
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_dzetsaka_use_case_{function_name}", module_path)
    if spec is None or spec.loader is None:
        return None
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, function_name, None)


_RUN_TRAINING = _load_callable("train_model.py", "run_training")
_RUN_CLASSIFICATION = _load_callable("classify_raster.py", "run_classification")


def run_training(**kwargs):
    """Execute training using migrated use-case implementation when available."""
    if callable(_RUN_TRAINING):
        return _RUN_TRAINING(**kwargs)
    return mainfunction.LearnModel(
        raster_path=kwargs.get("raster_path"),
        vector_path=kwargs.get("vector_path"),
        class_field=kwargs.get("class_field"),
        model_path=kwargs.get("model_path"),
        split_config=kwargs.get("split_config"),
        random_seed=kwargs.get("random_seed"),
        matrix_path=kwargs.get("matrix_path"),
        classifier=kwargs.get("classifier"),
        extraParam=kwargs.get("extra_params"),
        feedback=kwargs.get("feedback"),
    )


def run_classification(**kwargs):
    """Execute inference using migrated use-case implementation when available."""
    if callable(_RUN_CLASSIFICATION):
        return _RUN_CLASSIFICATION(**kwargs)

    classifier_worker = mainfunction.ClassifyImage()
    return classifier_worker.initPredict(
        raster_path=kwargs.get("raster_path"),
        model_path=kwargs.get("model_path"),
        output_path=kwargs.get("output_path"),
        mask_path=kwargs.get("mask_path"),
        confidenceMap=kwargs.get("confidence_map"),
        confidenceMapPerClass=None,
        NODATA=kwargs.get("nodata"),
        feedback=kwargs.get("feedback"),
    )

