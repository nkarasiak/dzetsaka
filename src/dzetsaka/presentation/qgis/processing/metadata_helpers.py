"""Helper functions for QGIS Processing algorithm metadata."""

from __future__ import annotations


def get_help_url(algorithm_name):
    # type: (str) -> str
    """Get the documentation URL for a specific algorithm."""
    base_url = "https://github.com/nkarasiak/dzetsaka/blob/master/docs"
    doc_map = {
        "train": "USER_GUIDE.md#2-train-a-model",
        "classify": "USER_GUIDE.md#3-classify-new-images",
        "split_train_validation": "USER_GUIDE.md#2-train-a-model",
        "nested_cv": "USER_GUIDE.md#spatial-cross-validation",
        "explain_model": "USER_GUIDE.md#shap-explainability",
    }
    doc_path = doc_map.get(algorithm_name, "USER_GUIDE.md")
    return f"{base_url}/{doc_path}"


def get_common_tags():
    # type: () -> list[str]
    """Get common tags for all dzetsaka algorithms."""
    return ["classification", "machine learning", "remote sensing", "raster"]


def get_algorithm_specific_tags(algorithm_type):
    # type: (str) -> list[str]
    """Get algorithm-specific tags."""
    tag_map = {
        "training": ["training", "model", "supervised learning"],
        "classification": ["classification", "prediction", "inference"],
        "preprocessing": ["preprocessing", "filtering", "smoothing"],
        "postprocessing": ["postprocessing", "filtering", "refinement"],
        "validation": ["validation", "cross-validation", "accuracy"],
        "explainability": ["explainability", "interpretation", "shap"],
    }
    return tag_map.get(algorithm_type, [])


def get_group_id():
    # type: () -> str
    """Get the standard group ID for dzetsaka algorithms."""
    return "dzetsaka_classification"
