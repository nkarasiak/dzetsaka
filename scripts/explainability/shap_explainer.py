"""SHAP-based model explainability for dzetsaka classification.

This module provides feature importance computation and visualization using
SHAP (SHapley Additive exPlanations) values. It supports all 11 dzetsaka
algorithms with automatic explainer selection.

Key Features:
-------------
- TreeExplainer for tree-based models (RF, XGB, LGB, ET, GBC)
- KernelExplainer fallback for other models (SVM, KNN, LR, NB, MLP)
- Feature importance computation from SHAP values
- Raster importance map generation
- Memory-efficient block-based processing
- Progress callback integration

Example Usage:
--------------
    >>> from scripts.explainability.shap_explainer import ModelExplainer
    >>>
    >>> # Create explainer from trained model
    >>> explainer = ModelExplainer(
    ...     model=trained_model,
    ...     feature_names=['B1', 'B2', 'B3', 'NDVI']
    ... )
    >>>
    >>> # Get feature importance
    >>> importance = explainer.get_feature_importance(X_sample)
    >>> print(importance)  # {'B1': 0.25, 'B2': 0.15, 'B3': 0.40, 'NDVI': 0.20}
    >>>
    >>> # Generate raster importance map
    >>> explainer.create_importance_raster(
    ...     raster_path='image.tif',
    ...     output_path='importance.tif',
    ...     sample_size=1000,
    ...     progress_callback=my_callback
    ... )

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

License:
--------
GNU General Public License v2.0 or later

"""
from __future__ import annotations

import contextlib
import os
import pickle  # nosec B403
import sys
from typing import Any, Callable

import numpy as np
from osgeo import gdal

# Try to import SHAP (optional dependency)
try:
    import shap

    # Fix for QGIS: Ensure sys.stderr/stdout exist for tqdm (used by SHAP)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


@contextlib.contextmanager
def _safe_tqdm_environment():
    """Context manager to ensure tqdm has valid file handles in QGIS.

    In QGIS, sys.stderr and sys.stdout can be None, causing tqdm to crash.
    This context manager temporarily provides valid file handles.
    """
    original_stderr = sys.stderr
    original_stdout = sys.stdout

    try:
        # Ensure valid file handles for tqdm
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        yield
    finally:
        # Restore original values (even if they were None)
        sys.stderr = original_stderr
        sys.stdout = original_stdout

# Try to import from dzetsaka modules
try:
    from .. import function_dataraster as dataraster
    from ..domain.exceptions import ClassificationError, DependencyError, OutputError
except ImportError:
    # Standalone usage without QGIS
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    try:
        import function_dataraster as dataraster

        from domain.exceptions import ClassificationError, DependencyError, OutputError
    except ImportError:
        # Create minimal fallback exceptions if not available
        class DependencyError(Exception):
            """Dependency error fallback."""

        class ClassificationError(Exception):
            """Classification error fallback."""

        class OutputError(Exception):
            """Output error fallback."""

        dataraster = None


class ModelExplainer:
    """Generate feature importance using SHAP values.

    This class wraps SHAP explainers to provide feature importance computation
    for any dzetsaka classifier. It automatically selects the appropriate
    SHAP explainer based on the model type.

    Parameters
    ----------
    model : object
        Trained scikit-learn compatible model
    feature_names : List[str], optional
        Names of input features (e.g., ['B1', 'B2', 'B3', 'NDVI'])
        If not provided, generic names will be used
    background_data : np.ndarray, optional
        Background dataset for KernelExplainer (shape: [n_samples, n_features])
        Only used for non-tree models. If not provided, will be required
        when calling get_feature_importance() for non-tree models

    Attributes
    ----------
    model : object
        The trained model to explain
    feature_names : List[str]
        Names of input features
    explainer : shap.Explainer
        SHAP explainer instance (TreeExplainer or KernelExplainer)
    explainer_type : str
        Type of explainer being used ('tree' or 'kernel')

    Raises
    ------
    DependencyError
        If SHAP is not installed
    ValueError
        If model is None or invalid

    Example
    -------
    >>> explainer = ModelExplainer(rf_model, ['B1', 'B2', 'B3'])
    >>> importance = explainer.get_feature_importance(X_test[:100])
    >>> print(f"Most important: {max(importance, key=importance.get)}")

    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        background_data: np.ndarray | None = None,
    ):
        """Initialize ModelExplainer with model and feature information."""
        if not SHAP_AVAILABLE:
            raise DependencyError(
                package_name="shap",
                reason="SHAP is required for model explainability",
                required_version=">=0.41.0",
            )

        if model is None:
            raise ValueError("Model cannot be None")

        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer: Any | None = None
        self.explainer_type: str | None = None

        # Don't create explainer yet - do it lazily when needed
        # This allows background_data to be provided later for KernelExplainer

    def _create_explainer(self, background_data: np.ndarray | None = None) -> Any:
        """Create appropriate SHAP explainer based on model type.

        Parameters
        ----------
        background_data : np.ndarray, optional
            Background dataset for KernelExplainer
            Required for non-tree models if not provided at init

        Returns
        -------
        shap.Explainer
            Initialized SHAP explainer

        Raises
        ------
        ValueError
            If background_data is required but not provided

        """
        if self.explainer is not None:
            return self.explainer

        # Determine if model is tree-based
        is_tree_model = self._is_tree_based_model()

        if is_tree_model:
            # Use TreeExplainer for tree-based models (fast and exact)
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = "tree"
        else:
            # Use KernelExplainer for other models (slower but universal)
            # Requires background data for sampling
            bg_data = background_data if background_data is not None else self.background_data

            if bg_data is None:
                raise ValueError(
                    "background_data is required for KernelExplainer (non-tree models). "
                    "Provide it at init or when calling get_feature_importance()."
                )

            # Sample background data if too large (max 100 samples for performance)
            bg_sample = shap.sample(bg_data, 100, random_state=42) if len(bg_data) > 100 else bg_data

            # Create KernelExplainer with model's predict_proba
            if hasattr(self.model, "predict_proba"):
                self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, bg_sample)

            self.explainer_type = "kernel"

        return self.explainer

    def _is_tree_based_model(self) -> bool:
        """Determine if model is tree-based (supports TreeExplainer).

        Returns
        -------
        bool
            True if model is tree-based, False otherwise

        """
        # Check for common tree-based model attributes
        tree_indicators = [
            "tree_",  # Single decision tree
            "estimators_",  # Ensemble methods (RF, ET, GBC, XGB, LGB)
            "booster_",  # XGBoost
            "n_estimators",  # Most tree ensembles
        ]

        for indicator in tree_indicators:
            if hasattr(self.model, indicator):
                return True

        # Check class name for tree-based models
        model_class_name = self.model.__class__.__name__.lower()
        tree_model_names = [
            "randomforest",
            "extratrees",
            "gradientboosting",
            "xgb",
            "lgbm",
            "lightgbm",
            "decisiontree",
        ]

        return any(name in model_class_name for name in tree_model_names)

    def get_feature_importance(
        self,
        X_sample: np.ndarray,
        background_data: np.ndarray | None = None,
        aggregate_method: str = "mean_abs",
    ) -> dict[str, float]:
        """Calculate SHAP-based feature importance.

        Computes SHAP values for the given sample and aggregates them
        to produce feature importance scores.

        Parameters
        ----------
        X_sample : np.ndarray
            Sample data to compute SHAP values for (shape: [n_samples, n_features])
            For performance, recommend 100-1000 samples
        background_data : np.ndarray, optional
            Background dataset for KernelExplainer (only needed for non-tree models)
        aggregate_method : str, default='mean_abs'
            How to aggregate SHAP values into importance scores:
            - 'mean_abs': Mean absolute SHAP value (default)
            - 'mean': Mean SHAP value (can be negative)
            - 'max_abs': Maximum absolute SHAP value

        Returns
        -------
        Dict[str, float]
            Feature importance dictionary mapping feature names to importance scores
            Scores are normalized to sum to 1.0

        Raises
        ------
        ValueError
            If X_sample is invalid or aggregate_method is unknown

        Example
        -------
        >>> importance = explainer.get_feature_importance(X_test[:500])
        >>> for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        ...     print(f"{feat}: {score:.3f}")

        """
        if X_sample is None or len(X_sample) == 0:
            raise ValueError("X_sample cannot be None or empty")
        if aggregate_method not in {"mean_abs", "mean", "max_abs"}:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}. Use 'mean_abs', 'mean', or 'max_abs'")
        if hasattr(self.model, "n_features_in_") and X_sample.shape[1] != int(self.model.n_features_in_):
            raise ValueError(
                f"X_sample has {X_sample.shape[1]} features, but model expects {self.model.n_features_in_}."
            )

        # Create explainer if not already created
        if self.explainer is None:
            self._create_explainer(background_data)

        # Windows-specific stability fallback: some SHAP TreeExplainer builds can
        # crash the Python process in native code. Use model-native importances.
        if os.name == "nt" and self.explainer_type == "tree" and hasattr(self.model, "feature_importances_"):
            importance_values = np.asarray(self.model.feature_importances_, dtype=float).reshape(-1)
            if importance_values.size != X_sample.shape[1]:
                if importance_values.size > X_sample.shape[1]:
                    importance_values = importance_values[: X_sample.shape[1]]
                else:
                    importance_values = np.pad(
                        importance_values,
                        (0, X_sample.shape[1] - importance_values.size),
                        mode="constant",
                    )
            total = importance_values.sum()
            if total > 0:
                importance_values = importance_values / total
            if self.feature_names is None:
                self.feature_names = [f"Feature_{i}" for i in range(len(importance_values))]
            return {name: float(value) for name, value in zip(self.feature_names, importance_values.tolist())}

        # Compute SHAP values with protection against QGIS sys.stderr/stdout issues
        with _safe_tqdm_environment():
            # SHAP API differs by version; use silent flag only when supported.
            try:
                shap_values = self.explainer.shap_values(X_sample, silent=True)
            except TypeError:
                shap_values = self.explainer.shap_values(X_sample)

        shap_array = np.asarray(shap_values)
        n_features = X_sample.shape[1]

        def _aggregate_2d(values: np.ndarray) -> np.ndarray:
            """Aggregate per-sample SHAP values into per-feature importance."""
            if aggregate_method == "mean_abs":
                return np.abs(values).mean(axis=0)
            if aggregate_method == "mean":
                return values.mean(axis=0)
            if aggregate_method == "max_abs":
                return np.abs(values).max(axis=0)
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}. Use 'mean_abs', 'mean', or 'max_abs'")

        if shap_array.ndim == 1:
            importance_values = shap_array.astype(float)
        elif shap_array.ndim == 2:
            importance_values = _aggregate_2d(shap_array)
        elif shap_array.ndim == 3:
            # SHAP returns either [classes, samples, features] or [samples, features, classes].
            if shap_array.shape[-1] == n_features:
                per_sample_feature = np.abs(shap_array).mean(axis=0) if aggregate_method != "mean" else shap_array.mean(axis=0)
            elif shap_array.shape[1] == n_features:
                per_sample_feature = np.abs(shap_array).mean(axis=2) if aggregate_method != "mean" else shap_array.mean(axis=2)
            else:
                feature_axis = int(np.argmin(np.abs(np.array(shap_array.shape) - n_features)))
                reduce_axes = tuple(ax for ax in range(shap_array.ndim) if ax != feature_axis)
                if aggregate_method == "max_abs":
                    importance_values = np.abs(shap_array).max(axis=reduce_axes)
                elif aggregate_method == "mean":
                    importance_values = shap_array.mean(axis=reduce_axes)
                else:
                    importance_values = np.abs(shap_array).mean(axis=reduce_axes)
                importance_values = np.asarray(importance_values, dtype=float)
                per_sample_feature = None

            if per_sample_feature is not None:
                importance_values = _aggregate_2d(np.asarray(per_sample_feature))
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")

        importance_values = np.asarray(importance_values, dtype=float).reshape(-1)
        if importance_values.size != n_features:
            if importance_values.size > n_features:
                importance_values = importance_values[:n_features]
            else:
                importance_values = np.pad(importance_values, (0, n_features - importance_values.size), mode="constant")

        # Normalize to sum to 1.0
        total = importance_values.sum()
        if total > 0:
            importance_values = importance_values / total

        # Create feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(len(importance_values))]

        # Create dictionary mapping feature names to importance
        # Convert to native Python floats so sorting/logging works with scalars
        importance_dict = {
            name: float(value)
            for name, value in zip(self.feature_names, importance_values.tolist())
        }

        return importance_dict

    def create_importance_raster(
        self,
        raster_path: str,
        output_path: str,
        background_data: np.ndarray | None = None,
        sample_size: int = 1000,
        aggregate_method: str = "mean_abs",
        progress_callback: Callable[[int], None] | None = None,
    ) -> str:
        """Generate raster showing per-pixel feature importance.

        Creates a multi-band raster where each band represents the importance
        of the corresponding input band/feature. Values range from 0-100,
        where 100 indicates the most important feature for that pixel.

        Parameters
        ----------
        raster_path : str
            Path to input raster (same one used for training/classification)
        output_path : str
            Path for output importance raster (.tif)
        background_data : np.ndarray, optional
            Background dataset for KernelExplainer (only for non-tree models)
        sample_size : int, default=1000
            Number of pixels to sample for SHAP computation
            Larger values are more accurate but slower
        aggregate_method : str, default='mean_abs'
            How to aggregate SHAP values ('mean_abs', 'mean', or 'max_abs')
        progress_callback : Callable[[int], None], optional
            Callback function for progress updates (receives percentage 0-100)

        Returns
        -------
        str
            Path to created importance raster

        Raises
        ------
        ClassificationError
            If raster cannot be read or processed
        OutputError
            If output raster cannot be written

        Example
        -------
        >>> def progress(pct):
        ...     print(f"Progress: {pct}%")
        >>>
        >>> explainer.create_importance_raster(
        ...     'image.tif',
        ...     'importance.tif',
        ...     sample_size=500,
        ...     progress_callback=progress
        ... )

        """
        if not os.path.exists(raster_path):
            raise ClassificationError(
                raster_path=raster_path,
                reason=f"Raster file not found: {raster_path}",
            )

        # Open raster
        try:
            raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
            if raster_ds is None:
                raise ClassificationError(
                    raster_path=raster_path,
                    reason="Failed to open raster with GDAL",
                )
        except Exception as e:
            raise ClassificationError(
                raster_path=raster_path,
                reason=f"Error opening raster: {e!s}",
            ) from e

        try:
            # Get raster dimensions
            n_bands = raster_ds.RasterCount

            if progress_callback:
                progress_callback(5)

            # Sample pixels from raster
            X_sample = self._sample_raster_pixels(raster_ds, sample_size, n_bands)

            if progress_callback:
                progress_callback(20)

            # Compute feature importance from sample
            importance = self.get_feature_importance(
                X_sample=X_sample,
                background_data=background_data,
                aggregate_method=aggregate_method,
            )

            if progress_callback:
                progress_callback(60)

            # Create output raster with importance values
            # Each band gets its importance score as a constant value (0-100)
            self._write_importance_raster(
                output_path=output_path,
                template_ds=raster_ds,
                importance=importance,
                n_bands=n_bands,
                progress_callback=progress_callback,
            )

            if progress_callback:
                progress_callback(100)

        finally:
            raster_ds = None  # Close dataset

        return output_path

    def _sample_raster_pixels(
        self,
        raster_ds: gdal.Dataset,
        sample_size: int,
        n_bands: int,
    ) -> np.ndarray:
        """Sample random pixels from raster for SHAP computation.

        Parameters
        ----------
        raster_ds : gdal.Dataset
            Open GDAL dataset
        sample_size : int
            Number of pixels to sample
        n_bands : int
            Number of bands in raster

        Returns
        -------
        np.ndarray
            Sampled pixel values (shape: [sample_size, n_bands])

        """
        n_cols = raster_ds.RasterXSize
        n_rows = raster_ds.RasterYSize

        # Generate random sample locations
        np.random.seed(42)  # For reproducibility
        sample_rows = np.random.randint(0, n_rows, size=sample_size)
        sample_cols = np.random.randint(0, n_cols, size=sample_size)

        # Read pixel values at sample locations
        X_sample = np.zeros((sample_size, n_bands), dtype=np.float32)

        for band_idx in range(n_bands):
            band = raster_ds.GetRasterBand(band_idx + 1)  # GDAL bands are 1-indexed
            band_data = band.ReadAsArray()

            # Extract values at sample locations
            X_sample[:, band_idx] = band_data[sample_rows, sample_cols]

        return X_sample

    def _write_importance_raster(
        self,
        output_path: str,
        template_ds: gdal.Dataset,
        importance: dict[str, float],
        n_bands: int,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Write feature importance as multi-band raster.

        Creates a raster where each band contains the importance score (0-100)
        for that feature across all pixels.

        Parameters
        ----------
        output_path : str
            Path for output raster
        template_ds : gdal.Dataset
            Template dataset for georeferencing
        importance : Dict[str, float]
            Feature importance dictionary
        n_bands : int
            Number of bands to create
        progress_callback : Callable[[int], None], optional
            Progress callback function

        Raises
        ------
        OutputError
            If raster cannot be written

        """
        try:
            # Get template metadata
            n_cols = template_ds.RasterXSize
            n_rows = template_ds.RasterYSize
            projection = template_ds.GetProjection()
            geotransform = template_ds.GetGeoTransform()

            # Create output raster
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(
                output_path,
                n_cols,
                n_rows,
                n_bands,
                gdal.GDT_Float32,
                options=["COMPRESS=LZW", "TILED=YES"],
            )

            if out_ds is None:
                raise OutputError(
                    output_path=output_path,
                    reason="Failed to create output raster with GDAL",
                )

            # Set georeferencing
            out_ds.SetProjection(projection)
            out_ds.SetGeoTransform(geotransform)

            # Convert importance scores to 0-100 scale
            importance_values = np.array(list(importance.values()))
            if importance_values.max() > 0:
                importance_scaled = (importance_values * 100).astype(np.float32)
            else:
                importance_scaled = importance_values.astype(np.float32)

            # Write each band with its importance value
            for band_idx in range(n_bands):
                band = out_ds.GetRasterBand(band_idx + 1)

                # Create constant array with importance value
                importance_value = importance_scaled[band_idx] if band_idx < len(importance_scaled) else 0.0
                band_data = np.full((n_rows, n_cols), importance_value, dtype=np.float32)

                # Write band
                band.WriteArray(band_data)
                band.SetDescription(f"Importance: {self.feature_names[band_idx] if self.feature_names else f'Band {band_idx + 1}'}")
                band.FlushCache()

                if progress_callback:
                    progress_pct = 60 + int(40 * (band_idx + 1) / n_bands)
                    progress_callback(progress_pct)

            # Close and flush
            out_ds.FlushCache()
            out_ds = None

        except Exception as e:
            raise OutputError(
                output_path=output_path,
                reason=f"Error writing importance raster: {e!s}",
            ) from e

    def save_to_file(self, file_path: str) -> None:
        """Save explainer to file for later use.

        Parameters
        ----------
        file_path : str
            Path to save explainer (.pkl or .shap)

        Raises
        ------
        OutputError
            If explainer cannot be saved

        """
        try:
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "feature_names": self.feature_names,
                        "explainer_type": self.explainer_type,
                        "background_data": self.background_data,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            raise OutputError(
                output_path=file_path,
                reason=f"Failed to save explainer: {e!s}",
            ) from e

    @classmethod
    def load_from_file(cls, file_path: str) -> ModelExplainer:
        """Load explainer from file.

        Parameters
        ----------
        file_path : str
            Path to saved explainer file

        Returns
        -------
        ModelExplainer
            Loaded explainer instance

        Raises
        ------
        ValueError
            If file cannot be loaded

        """
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)  # nosec B301

            return cls(
                model=data["model"],
                feature_names=data.get("feature_names"),
                background_data=data.get("background_data"),
            )
        except Exception as e:
            raise ValueError(f"Failed to load explainer from {file_path}: {e!s}") from e


def check_shap_available() -> tuple[bool, str | None]:
    """Check if SHAP is available and return version info.

    Returns:
    -------
    Tuple[bool, Optional[str]]
        (is_available, version_string)

    Example:
    -------
    >>> available, version = check_shap_available()
    >>> if available:
    ...     print(f"SHAP {version} is available")
    ... else:
    ...     print("SHAP is not installed")

    """
    if not SHAP_AVAILABLE:
        return False, None

    try:
        version = shap.__version__
        return True, version
    except AttributeError:
        return True, "unknown"
