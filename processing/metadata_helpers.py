"""Helper functions for QGIS Processing algorithm metadata.

This module provides utilities for adding consistent metadata across all
dzetsaka processing algorithms, including help URLs, tags, and group IDs.
"""


def get_help_url(algorithm_name):
    # type: (str) -> str
    """Get the documentation URL for a specific algorithm.

    Parameters
    ----------
    algorithm_name : str
        The algorithm identifier (e.g., 'train', 'classify', 'sieve')

    Returns
    -------
    str
        The full URL to the algorithm documentation on GitHub
    """
    base_url = "https://github.com/nkarasiak/dzetsaka/blob/master/docs"

    # Map algorithm names to documentation files
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
    """Get common tags for all dzetsaka algorithms.

    Returns
    -------
    list[str]
        Common tags for searchability
    """
    return ["classification", "machine learning", "remote sensing", "raster"]


def get_algorithm_specific_tags(algorithm_type):
    # type: (str) -> list[str]
    """Get algorithm-specific tags.

    Parameters
    ----------
    algorithm_type : str
        Type of algorithm: 'training', 'classification', 'preprocessing',
        'postprocessing', 'validation', 'explainability'

    Returns
    -------
    list[str]
        Specific tags for this algorithm type
    """
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
    """Get the standard group ID for dzetsaka algorithms.

    Returns
    -------
    str
        The group ID
    """
    return "dzetsaka_classification"
