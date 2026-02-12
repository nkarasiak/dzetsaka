"""Runtime dependency health checks used by QGIS presentation layer."""

from __future__ import annotations


def check_sklearn_usable() -> tuple[bool, str]:
    """Return whether scikit-learn is fully usable in current runtime."""
    try:
        import sklearn

        # Broken installs can import a namespace package named sklearn but miss core modules.
        from sklearn.base import BaseEstimator  # noqa: F401

        version = getattr(sklearn, "__version__", None)
        module_file = getattr(sklearn, "__file__", None)
        if not version:
            return False, "sklearn imported but has no __version__ (incomplete install)"
        if module_file is None:
            return False, "sklearn imported as namespace package (incomplete install)"
        return True, f"version {version}"
    except ImportError as e:
        return False, f"not importable: {e}"
    except Exception as e:
        return False, f"imported but unusable: {e}"
