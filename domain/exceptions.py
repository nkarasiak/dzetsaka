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


class ProjectionMismatchError(DzetsakaException):
    """CRS mismatch between raster and vector data.

    Raised when input raster and vector files have incompatible
    coordinate reference systems.

    Parameters
    ----------
    raster_crs : str
        Coordinate reference system of the raster (e.g., "EPSG:32632")
    vector_crs : str
        Coordinate reference system of the vector (e.g., "EPSG:4326")
    raster_path : str, optional
        Path to the raster file
    vector_path : str, optional
        Path to the vector file

    """

    def __init__(
        self, raster_crs: str, vector_crs: str, raster_path: Optional[str] = None, vector_path: Optional[str] = None
    ):
        self.raster_crs = raster_crs
        self.vector_crs = vector_crs
        self.raster_path = raster_path
        self.vector_path = vector_path

        message = f"CRS mismatch: Raster ({raster_crs}) != Vector ({vector_crs})"
        if raster_path or vector_path:
            message += f"\nRaster: {raster_path}\nVector: {vector_path}"

        super().__init__(message)


class InsufficientSamplesError(DzetsakaException):
    """Not enough training samples for a class.

    Raised when a class has too few training samples for reliable
    model training (typically < 5 samples).

    Parameters
    ----------
    class_id : int
        Identifier of the class with insufficient samples
    sample_count : int
        Actual number of samples found
    minimum : int, default=5
        Minimum required number of samples
    class_distribution : dict, optional
        Complete distribution of samples across all classes

    """

    def __init__(
        self, class_id: int, sample_count: int, minimum: int = 5, class_distribution: Optional[dict] = None
    ):
        self.class_id = class_id
        self.sample_count = sample_count
        self.minimum = minimum
        self.class_distribution = class_distribution

        message = f"Class {class_id} has only {sample_count} samples (minimum: {minimum})"

        if class_distribution:
            distribution_str = ", ".join(
                f"Class {k}: {v} samples" for k, v in sorted(class_distribution.items())
            )
            message += f"\nFull distribution: {distribution_str}"

        super().__init__(message)


class InvalidFieldError(DzetsakaException):
    """Invalid or missing field in vector data.

    Raised when the specified class field doesn't exist in the vector
    file, or contains invalid data (e.g., non-integer values).

    Parameters
    ----------
    field_name : str
        Name of the field that's invalid
    vector_path : str
        Path to the vector file
    reason : str
        Description of why the field is invalid
    available_fields : list, optional
        List of available fields in the vector file

    """

    def __init__(self, field_name: str, vector_path: str, reason: str, available_fields: Optional[list] = None):
        self.field_name = field_name
        self.vector_path = vector_path
        self.reason = reason
        self.available_fields = available_fields

        message = f"Invalid field '{field_name}' in {vector_path}: {reason}"

        if available_fields:
            message += f"\nAvailable fields: {', '.join(available_fields)}"

        super().__init__(message)


class ModelTrainingError(DzetsakaException):
    """Model training failed.

    Raised when the machine learning training process fails due to
    data issues, memory errors, or algorithm-specific problems.

    Parameters
    ----------
    classifier_code : str
        Classifier that failed to train (e.g., "RF", "SVM")
    reason : str
        Description of the failure
    original_exception : Exception, optional
        The original exception that caused the failure

    """

    def __init__(self, classifier_code: str, reason: str, original_exception: Optional[Exception] = None):
        self.classifier_code = classifier_code
        self.reason = reason
        self.original_exception = original_exception

        message = f"Training failed for {classifier_code}: {reason}"

        if original_exception:
            message += f"\nOriginal error: {type(original_exception).__name__}: {original_exception!s}"

        super().__init__(message)


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


class MemoryError(DzetsakaException):
    """Memory limit exceeded during processing.

    Raised when raster processing operations exceed the configured
    memory limit (typically 512MB for block-based operations).

    Parameters
    ----------
    operation : str
        Operation that caused the memory error
    required_memory_mb : float, optional
        Estimated memory required in MB
    available_memory_mb : float, optional
        Available memory in MB

    """

    def __init__(
        self, operation: str, required_memory_mb: Optional[float] = None, available_memory_mb: Optional[float] = None
    ):
        self.operation = operation
        self.required_memory_mb = required_memory_mb
        self.available_memory_mb = available_memory_mb

        message = f"Memory limit exceeded during {operation}"

        if required_memory_mb and available_memory_mb:
            message += f"\nRequired: {required_memory_mb:.1f} MB, Available: {available_memory_mb:.1f} MB"
        elif required_memory_mb:
            message += f"\nRequired: {required_memory_mb:.1f} MB"

        message += "\nTry reducing image size, using fewer training samples, or processing in smaller blocks"

        super().__init__(message)


class ValidationError(DzetsakaException):
    """Data validation failed.

    Raised when input validation detects issues with data quality,
    format, or compatibility before processing begins.

    Parameters
    ----------
    validation_type : str
        Type of validation that failed (e.g., "CRS check", "sample count")
    reason : str
        Description of the validation failure
    suggestions : list, optional
        List of suggested fixes

    """

    def __init__(self, validation_type: str, reason: str, suggestions: Optional[list] = None):
        self.validation_type = validation_type
        self.reason = reason
        self.suggestions = suggestions

        message = f"{validation_type} failed: {reason}"

        if suggestions:
            message += "\nSuggestions:"
            for suggestion in suggestions:
                message += f"\n  - {suggestion}"

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
