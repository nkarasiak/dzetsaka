"""Factory pattern for creating ML classifiers.

This module implements the Factory pattern to replace long if/elif chains
with a clean, extensible registry-based system.

Benefits:
- Easy to add new classifiers without modifying existing code (Open/Closed Principle)
- Clean separation of concerns
- Type-safe classifier creation
- Extensible for third-party plugins

Example:
    >>> factory = ClassifierFactory()
    >>> clf = factory.create("RF", n_estimators=100)
    >>> print(factory.get_available_classifiers())

Author:
    Nicolas Karasiak

"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type

from domain.exceptions import ConfigurationError, DependencyError


@dataclass
class ClassifierMetadata:
    """Metadata for a classifier.

    Attributes
    ----------
    code : str
        Short code (e.g., "RF", "SVM")
    name : str
        Full name (e.g., "Random Forest")
    description : str
        Human-readable description
    requires_sklearn : bool
        Whether scikit-learn is required
    requires_xgboost : bool
        Whether XGBoost is required
    requires_lightgbm : bool
        Whether LightGBM is required
    requires_catboost : bool
        Whether CatBoost is required
    supports_probability : bool
        Whether classifier supports probability estimates
    supports_feature_importance : bool
        Whether classifier provides feature importance

    """

    code: str
    name: str
    description: str
    requires_sklearn: bool = False
    requires_xgboost: bool = False
    requires_lightgbm: bool = False
    requires_catboost: bool = False
    supports_probability: bool = True
    supports_feature_importance: bool = False


class ClassifierFactory:
    """Factory for creating ML classifiers with dependency injection.

    This factory uses a registry pattern to manage classifier creation,
    replacing long if/elif chains with a clean, extensible system.

    """

    # Class-level registry
    _registry: ClassVar[Dict[str, Dict[str, Any]]] = {}

    @classmethod
    def register(
        cls,
        code: str,
        classifier_class: Type,
        metadata: ClassifierMetadata,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a classifier for the factory.

        Parameters
        ----------
        code : str
            Classifier short code (e.g., "RF", "SVM")
        classifier_class : Type
            Class or callable that creates the classifier
        metadata : ClassifierMetadata
            Metadata describing the classifier
        default_params : Optional[Dict[str, Any]]
            Default parameters for the classifier

        """
        cls._registry[code.upper()] = {
            "class": classifier_class,
            "metadata": metadata,
            "default_params": default_params or {},
        }

    @classmethod
    def create(cls, code: str, **params) -> Any:
        """Create classifier instance.

        Parameters
        ----------
        code : str
            Classifier short code (e.g., "RF", "SVM")
        **params
            Hyperparameters for the classifier

        Returns
        -------
        Any
            Classifier instance

        Raises
        ------
        ConfigurationError
            If classifier code is unknown
        DependencyError
            If required dependencies are not available

        """
        code = code.upper()

        if code not in cls._registry:
            available = ", ".join(cls.get_available_classifiers())
            raise ConfigurationError(
                f"Unknown classifier code: {code}. Available classifiers: {available}", config_key="classifier"
            )

        registry_entry = cls._registry[code]
        metadata = registry_entry["metadata"]

        # Check dependencies
        cls._check_dependencies(metadata)

        # Merge default params with user params (user params take precedence)
        final_params = {**registry_entry["default_params"], **params}

        # Create classifier
        try:
            classifier_class = registry_entry["class"]
            return classifier_class(**final_params)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create classifier {code}: {e!s}", config_key="classifier_params"
            ) from e

    @classmethod
    def get_available_classifiers(cls) -> List[str]:
        """Get list of available classifier codes.

        Returns
        -------
        List[str]
            List of classifier codes (e.g., ["GMM", "RF", "SVM"])

        """
        return sorted(cls._registry.keys())

    @classmethod
    def get_metadata(cls, code: str) -> Optional[ClassifierMetadata]:
        """Get metadata for a classifier.

        Parameters
        ----------
        code : str
            Classifier short code

        Returns
        -------
        Optional[ClassifierMetadata]
            Classifier metadata, or None if not found

        """
        code = code.upper()
        if code not in cls._registry:
            return None
        return cls._registry[code]["metadata"]

    @classmethod
    def get_all_metadata(cls) -> Dict[str, ClassifierMetadata]:
        """Get metadata for all registered classifiers.

        Returns
        -------
        Dict[str, ClassifierMetadata]
            Dictionary mapping codes to metadata

        """
        return {code: entry["metadata"] for code, entry in cls._registry.items()}

    @classmethod
    def is_available(cls, code: str) -> bool:
        """Check if a classifier is available (dependencies met).

        Parameters
        ----------
        code : str
            Classifier short code

        Returns
        -------
        bool
            True if classifier is available, False otherwise

        """
        code = code.upper()
        if code not in cls._registry:
            return False

        metadata = cls._registry[code]["metadata"]

        try:
            cls._check_dependencies(metadata)
            return True
        except DependencyError:
            return False

    @classmethod
    def _check_dependencies(cls, metadata: ClassifierMetadata) -> None:
        """Check if required dependencies are available.

        Parameters
        ----------
        metadata : ClassifierMetadata
            Classifier metadata containing dependency requirements

        Raises
        ------
        DependencyError
            If required dependencies are not available

        """
        if metadata.requires_sklearn:
            try:
                import sklearn  # noqa: F401
            except ImportError as e:
                raise DependencyError(
                    "scikit-learn",
                    f"Classifier {metadata.code} requires scikit-learn",
                    required_version=">=0.24.0",
                ) from e

        if metadata.requires_xgboost:
            try:
                import xgboost  # noqa: F401
            except ImportError as e:
                raise DependencyError(
                    "xgboost", f"Classifier {metadata.code} requires XGBoost", required_version=">=1.0.0"
                ) from e

        if metadata.requires_lightgbm:
            try:
                import lightgbm  # noqa: F401
            except ImportError as e:
                raise DependencyError(
                    "lightgbm",
                    f"Classifier {metadata.code} requires LightGBM",
                    required_version=">=3.0.0",
                ) from e

        if getattr(metadata, "requires_catboost", False):
            try:
                import catboost  # noqa: F401
            except ImportError as e:
                raise DependencyError(
                    "catboost",
                    f"Classifier {metadata.code} requires CatBoost",
                    required_version=">=1.0.0",
                ) from e

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the classifier registry.

        Useful for testing or plugin unloading.

        """
        cls._registry.clear()


# Initialize factory with built-in classifiers
def initialize_factory() -> None:
    """Initialize factory with all built-in classifiers.

    This function registers all 11 supported classifiers with their
    metadata and default parameters.

    """
    # GMM (Gaussian Mixture Model) - No dependencies
    try:
        from scripts.gmm_ridge import ridge

        ClassifierFactory.register(
            code="GMM",
            classifier_class=ridge,
            metadata=ClassifierMetadata(
                code="GMM",
                name="Gaussian Mixture Model",
                description="Fast baseline classifier with no external dependencies",
                requires_sklearn=False,
                supports_probability=True,
                supports_feature_importance=False,
            ),
            default_params={},
        )
    except ImportError:
        pass

    # Scikit-learn classifiers
    try:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC

        # Random Forest
        ClassifierFactory.register(
            code="RF",
            classifier_class=RandomForestClassifier,
            metadata=ClassifierMetadata(
                code="RF",
                name="Random Forest",
                description="Ensemble of decision trees, robust and accurate",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=True,
            ),
            default_params={"random_state": 42},
        )

        # Support Vector Machine
        ClassifierFactory.register(
            code="SVM",
            classifier_class=SVC,
            metadata=ClassifierMetadata(
                code="SVM",
                name="Support Vector Machine",
                description="Maximum margin classifier, good for complex boundaries",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=False,
            ),
            default_params={"probability": True, "random_state": 42},
        )

        # K-Nearest Neighbors
        ClassifierFactory.register(
            code="KNN",
            classifier_class=KNeighborsClassifier,
            metadata=ClassifierMetadata(
                code="KNN",
                name="K-Nearest Neighbors",
                description="Instance-based learning, simple and interpretable",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=False,
            ),
            default_params={},
        )

        # Extra Trees
        ClassifierFactory.register(
            code="ET",
            classifier_class=ExtraTreesClassifier,
            metadata=ClassifierMetadata(
                code="ET",
                name="Extra Trees",
                description="Extremely randomized trees, faster than Random Forest",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=True,
            ),
            default_params={"random_state": 42},
        )

        # Gradient Boosting Classifier
        ClassifierFactory.register(
            code="GBC",
            classifier_class=GradientBoostingClassifier,
            metadata=ClassifierMetadata(
                code="GBC",
                name="Gradient Boosting Classifier",
                description="Sequential boosting, high accuracy but slower",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=True,
            ),
            default_params={"random_state": 42},
        )

        # Logistic Regression
        ClassifierFactory.register(
            code="LR",
            classifier_class=LogisticRegression,
            metadata=ClassifierMetadata(
                code="LR",
                name="Logistic Regression",
                description="Linear probabilistic classifier, fast and interpretable",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=False,
            ),
            default_params={"random_state": 42, "max_iter": 1000},
        )

        # Gaussian Naive Bayes
        ClassifierFactory.register(
            code="NB",
            classifier_class=GaussianNB,
            metadata=ClassifierMetadata(
                code="NB",
                name="Gaussian Naive Bayes",
                description="Probabilistic classifier assuming feature independence",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=False,
            ),
            default_params={},
        )

        # Multi-layer Perceptron
        ClassifierFactory.register(
            code="MLP",
            classifier_class=MLPClassifier,
            metadata=ClassifierMetadata(
                code="MLP",
                name="Multi-layer Perceptron",
                description="Neural network, can learn complex patterns",
                requires_sklearn=True,
                supports_probability=True,
                supports_feature_importance=False,
            ),
            default_params={"random_state": 42, "max_iter": 1000},
        )

    except ImportError:
        pass

    # XGBoost
    try:
        from scripts.mainfunction import XGBClassifierWrapper

        ClassifierFactory.register(
            code="XGB",
            classifier_class=XGBClassifierWrapper,
            metadata=ClassifierMetadata(
                code="XGB",
                name="XGBoost",
                description="State-of-the-art gradient boosting, highest accuracy",
                requires_xgboost=True,
                supports_probability=True,
                supports_feature_importance=True,
            ),
            default_params={"random_state": 42, "eval_metric": "logloss"},
        )
    except ImportError:
        pass

    # LightGBM
    try:
        from scripts.mainfunction import LGBMClassifierWrapper

        ClassifierFactory.register(
            code="LGB",
            classifier_class=LGBMClassifierWrapper,
            metadata=ClassifierMetadata(
                code="LGB",
                name="LightGBM",
                description="Fast gradient boosting framework, efficient for large datasets",
                requires_lightgbm=True,
                supports_probability=True,
                supports_feature_importance=True,
            ),
            default_params={"random_state": 42, "verbose": -1},
        )
    except ImportError:
        pass

    # CatBoost
    try:
        from scripts.mainfunction import CBClassifierWrapper

        if CBClassifierWrapper is None:
            raise ImportError("CatBoost wrapper unavailable")

        ClassifierFactory.register(
            code="CB",
            classifier_class=CBClassifierWrapper,
            metadata=ClassifierMetadata(
                code="CB",
                name="CatBoost",
                description="Gradient boosting on decision trees with strong defaults",
                requires_catboost=True,
                supports_probability=True,
                supports_feature_importance=True,
            ),
            default_params={"random_seed": 42, "verbose": False},
        )
    except ImportError:
        pass


# Initialize factory on module import
initialize_factory()
