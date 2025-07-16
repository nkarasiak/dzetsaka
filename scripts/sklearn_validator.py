#!/usr/bin/env python3
"""
Sklearn validation utilities for dzetsaka
Provides robust checking for scikit-learn availability and specific algorithm requirements
"""

from typing import Dict, List, Tuple, Optional
from .. import classifier_config


class SklearnValidator:
    """Validator for scikit-learn availability and algorithm compatibility."""

    def __init__(self):
        self._sklearn_available = None
        self._sklearn_version = None
        self._algorithm_requirements = {
            "RF": ["sklearn.ensemble.RandomForestClassifier"],
            "SVM": ["sklearn.svm.SVC"],
            "KNN": ["sklearn.neighbors.KNeighborsClassifier"],
            "XGB": ["xgboost.XGBClassifier"],
            "LGB": ["lightgbm.LGBMClassifier"],
            "ET": ["sklearn.ensemble.ExtraTreesClassifier"],
            "GBC": ["sklearn.ensemble.GradientBoostingClassifier"],
            "LR": ["sklearn.linear_model.LogisticRegression"],
            "NB": ["sklearn.naive_bayes.GaussianNB"],
            "MLP": ["sklearn.neural_network.MLPClassifier"],
            "GMM": [],  # No sklearn required for GMM
        }
        self._sklearn_modules = {}

    def is_sklearn_available(self) -> bool:
        """Check if scikit-learn is available."""
        if self._sklearn_available is None:
            try:
                import sklearn

                self._sklearn_available = True
                self._sklearn_version = sklearn.__version__
            except ImportError:
                self._sklearn_available = False
                self._sklearn_version = None

        return self._sklearn_available

    def get_sklearn_version(self) -> Optional[str]:
        """Get scikit-learn version if available."""
        if not self.is_sklearn_available():
            return None
        return self._sklearn_version

    def validate_algorithm(self, algorithm: str) -> Tuple[bool, str]:
        """
        Validate if an algorithm is available and can be used.

        Parameters
        ----------
        algorithm : str
            Algorithm name ('GMM', 'RF', 'SVM', 'KNN', 'XGB', 'LGB', 'ET', 'GBC', 'LR', 'NB', 'MLP')

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        algorithm = algorithm.upper()

        if algorithm not in self._algorithm_requirements:
            return False, f"Unknown algorithm: {algorithm}"

        # GMM doesn't require sklearn
        if algorithm == "GMM":
            return True, ""

        # Check sklearn availability
        if not self.is_sklearn_available():
            return False, f"Scikit-learn is required for {algorithm} but not installed"

        # Check specific algorithm requirements
        required_modules = self._algorithm_requirements[algorithm]
        missing_modules = []

        for module_path in required_modules:
            if not self._check_module_import(module_path):
                missing_modules.append(module_path)

        if missing_modules:
            return (
                False,
                f"Required modules missing for {algorithm}: {', '.join(missing_modules)}",
            )

        return True, ""

    def _check_module_import(self, module_path: str) -> bool:
        """Check if a specific module can be imported."""
        if module_path in self._sklearn_modules:
            return self._sklearn_modules[module_path]

        try:
            parts = module_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]

            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)

            self._sklearn_modules[module_path] = True
            return True
        except (ImportError, AttributeError):
            self._sklearn_modules[module_path] = False
            return False

    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms based on current environment."""
        available = ["GMM"]  # GMM is always available

        if self.is_sklearn_available():
            for algorithm in ["RF", "SVM", "KNN", "ET", "GBC", "LR", "NB", "MLP"]:
                is_valid, _ = self.validate_algorithm(algorithm)
                if is_valid:
                    available.append(algorithm)

        # Check for XGBoost and LightGBM separately as they're not part of sklearn
        for algorithm in ["XGB", "LGB"]:
            is_valid, _ = self.validate_algorithm(algorithm)
            if is_valid:
                available.append(algorithm)

        return available

    def get_algorithm_info(self) -> Dict[str, Dict]:
        """Get detailed information about all algorithms."""
        info = {}

        for algorithm in [
            "GMM",
            "RF",
            "SVM",
            "KNN",
            "XGB",
            "LGB",
            "ET",
            "GBC",
            "LR",
            "NB",
            "MLP",
        ]:
            is_valid, error_msg = self.validate_algorithm(algorithm)
            info[algorithm] = {
                "available": is_valid,
                "error": error_msg,
                "requires_sklearn": algorithm not in ["GMM"],
                "requires_extra_package": algorithm in ["XGB", "LGB"],
                "full_name": self._get_algorithm_full_name(algorithm),
            }

        return info

    def _get_algorithm_full_name(self, algorithm: str) -> str:
        """Get full name for algorithm."""
        return classifier_config.get_classifier_name(algorithm)

    def get_installation_instructions(self) -> str:
        """Get installation instructions for scikit-learn."""
        return (
            "To use all available classifiers, install the following packages:\n\n"
            "Core packages (required for most classifiers):\n"
            "  pip install scikit-learn\n\n"
            "Optional packages for additional classifiers:\n"
            "  pip install xgboost        # For XGBoost (XGB)\n"
            "  pip install lightgbm       # For LightGBM (LGB)\n\n"
            "Install all at once:\n"
            "  pip install scikit-learn xgboost lightgbm\n\n"
            "Using conda:\n"
            "  conda install scikit-learn xgboost lightgbm\n\n"
            "For QGIS users:\n"
            "  Install through the OSGeo4W shell or QGIS Python environment\n\n"
            "For more detailed instructions, visit:\n"
            "https://github.com/nkarasiak/dzetsaka/#installation-of-scikit-learn"
        )


# Global validator instance
_validator = SklearnValidator()


def validate_classifier_selection(classifier: str) -> Tuple[bool, str]:
    """
    Validate a classifier selection from the UI.

    Parameters
    ----------
    classifier : str
        Classifier name (e.g., 'Random Forest', 'Support Vector Machine')

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    # Map UI names to algorithm codes
    ui_name_mapping = {
        "Gaussian Mixture Model": "GMM",
        "Random Forest": "RF",
        "Support Vector Machine": "SVM",
        "K-Nearest Neighbors": "KNN",
        "XGBoost": "XGB",
        "LightGBM": "LGB",
        "Extra Trees": "ET",
        "Gradient Boosting Classifier": "GBC",
        "Logistic Regression": "LR",
        "Gaussian Naive Bayes": "NB",
        "Multi-layer Perceptron": "MLP",
    }

    algorithm = ui_name_mapping.get(classifier, classifier.upper())
    return _validator.validate_algorithm(algorithm)


def check_sklearn_availability() -> Dict[str, any]:
    """
    Check sklearn availability and return comprehensive status.

    Returns
    -------
    dict
        Status information including availability, version, and algorithms
    """
    return {
        "sklearn_available": _validator.is_sklearn_available(),
        "sklearn_version": _validator.get_sklearn_version(),
        "available_algorithms": _validator.get_available_algorithms(),
        "algorithm_info": _validator.get_algorithm_info(),
        "installation_instructions": _validator.get_installation_instructions(),
    }


def get_sklearn_error_message(classifier: str) -> str:
    """
    Get a user-friendly error message for sklearn issues.

    Parameters
    ----------
    classifier : str
        Classifier name

    Returns
    -------
    str
        Formatted error message with installation instructions
    """
    is_valid, error = validate_classifier_selection(classifier)

    if is_valid:
        return ""

    base_message = f"Cannot use {classifier}: {error}"

    if "scikit-learn" in error or "sklearn" in error:
        installation_msg = _validator.get_installation_instructions()
        return f"{base_message}\n\n{installation_msg}"

    return base_message
