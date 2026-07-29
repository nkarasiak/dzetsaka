"""Custom exception hierarchy for dzetsaka plugin.

This module defines domain-specific exceptions with rich context information
to improve error handling, debugging, and user feedback.

Example:
    >>> try:
    ...     load_raster(path)
    ... except DataLoadError as e:
    ...     print(f"Failed to load {e.path}: {e.reason}")

Author:
    Nicolas Karasiak

"""

from typing import Optional


class DzetsakaException(Exception):  # noqa: N818
    """Base exception for all dzetsaka-specific errors.

    All custom exceptions in dzetsaka inherit from this base class,
    allowing for catch-all error handling when needed.

    Note: Named DzetsakaException (not DzetsakaError) to match plugin name
    and distinguish from specific error types.

    """


class ConfigurationError(DzetsakaException):
    """Invalid configuration or settings.

    Raised when plugin configuration, user settings, or algorithm
    parameters are invalid or incompatible.

    Parameters
    ----------
    message : str
        Description of the configuration error
    config_key : str, optional
        The specific configuration key that caused the error

    """

    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        super().__init__(message)


class DataLoadError(DzetsakaException):
    """Failed to load raster or vector data.

    Raised when GDAL/OGR cannot read input files due to corruption,
    format issues, permission errors, or missing files.

    Parameters
    ----------
    path : str
        Path to the file that failed to load
    reason : str
        Description of why loading failed

    """

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load {path}: {reason}")


class ClassificationError(DzetsakaException):
    """Image classification failed.

    Raised when applying a trained model to classify a raster fails
    due to dimension mismatches, memory issues, or I/O errors.

    Parameters
    ----------
    raster_path : str
        Path to the raster being classified
    model_path : str, optional
        Path to the model file being used
    reason : str
        Description of the failure

    """

    def __init__(self, raster_path: str, reason: str, model_path: Optional[str] = None):
        self.raster_path = raster_path
        self.model_path = model_path
        self.reason = reason

        message = f"Classification failed for {raster_path}: {reason}"

        if model_path:
            message += f"\nModel: {model_path}"

        super().__init__(message)


class DependencyError(DzetsakaException):
    """Required dependency is missing or incompatible.

    Raised when a required Python package (e.g., scikit-learn, xgboost)
    is not installed or has an incompatible version.

    Parameters
    ----------
    package_name : str
        Name of the missing package
    reason : str
        Description of the issue
    required_version : str, optional
        Required version or version range
    current_version : str, optional
        Currently installed version (if any)

    """

    def __init__(
        self,
        package_name: str,
        reason: str,
        required_version: Optional[str] = None,
        current_version: Optional[str] = None,
    ):
        self.package_name = package_name
        self.reason = reason
        self.required_version = required_version
        self.current_version = current_version

        message = f"Dependency error for '{package_name}': {reason}"

        if required_version:
            message += f"\nRequired version: {required_version}"
        if current_version:
            message += f"\nCurrent version: {current_version}"

        super().__init__(message)


class OutputError(DzetsakaException):
    """Failed to write output files.

    Raised when writing classification results, models, or other
    outputs fails due to permissions, disk space, or format issues.

    Parameters
    ----------
    output_path : str
        Path where output was attempted
    reason : str
        Description of the failure

    """

    def __init__(self, output_path: str, reason: str):
        self.output_path = output_path
        self.reason = reason
        super().__init__(f"Failed to write output to {output_path}: {reason}")
