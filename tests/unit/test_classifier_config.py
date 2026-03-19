"""Unit tests for classifier configuration module.

Tests classifier codes, name mappings, dependency lookups, and validation functions.
"""

import pytest

from classifier_config import (
    CATBOOST_DEPENDENT,
    CLASSIFIER_CODES,
    CLASSIFIER_NAMES,
    CODE_TO_NAME,
    NAME_TO_CODE,
    SKLEARN_DEPENDENT,
    XGBOOST_DEPENDENT,
    get_classifier_code,
    get_classifier_name,
    is_valid_classifier,
    requires_catboost,
    requires_sklearn,
    requires_xgboost,
)


class TestClassifierCodes:
    """Test classifier code and name constants."""

    def test_all_eleven_codes_present(self):
        """Test that all 11 expected classifier codes are defined."""
        expected = ["GMM", "RF", "SVM", "KNN", "XGB", "CB", "ET", "GBC", "LR", "NB", "MLP"]
        assert CLASSIFIER_CODES == expected

    def test_codes_and_names_same_length(self):
        """Test that CLASSIFIER_CODES and CLASSIFIER_NAMES have matching lengths."""
        assert len(CLASSIFIER_CODES) == 11
        assert len(CLASSIFIER_NAMES) == 11

    def test_code_to_name_and_name_to_code_are_inverses(self):
        """Test that CODE_TO_NAME and NAME_TO_CODE are inverse mappings."""
        for code, name in CODE_TO_NAME.items():
            assert NAME_TO_CODE[name] == code

        for name, code in NAME_TO_CODE.items():
            assert CODE_TO_NAME[code] == name


class TestGetClassifierCode:
    """Test get_classifier_code() function."""

    def test_known_name_returns_correct_code(self):
        """Test that a known full name returns the correct short code."""
        assert get_classifier_code("Random Forest") == "RF"
        assert get_classifier_code("XGBoost") == "XGB"
        assert get_classifier_code("Gaussian Mixture Model") == "GMM"

    def test_unknown_name_defaults_to_gmm(self):
        """Test that an unknown name defaults to GMM."""
        assert get_classifier_code("Unknown Classifier") == "GMM"
        assert get_classifier_code("") == "GMM"


class TestGetClassifierName:
    """Test get_classifier_name() function."""

    def test_known_code_returns_correct_name(self):
        """Test that a known code returns the correct full name."""
        assert get_classifier_name("RF") == "Random Forest"
        assert get_classifier_name("CB") == "CatBoost"

    def test_unknown_code_defaults_to_gaussian_mixture_model(self):
        """Test that an unknown code defaults to Gaussian Mixture Model."""
        assert get_classifier_name("INVALID") == "Gaussian Mixture Model"
        assert get_classifier_name("ZZZ") == "Gaussian Mixture Model"

    def test_case_insensitivity(self):
        """Test that get_classifier_name is case-insensitive."""
        assert get_classifier_name("rf") == "Random Forest"
        assert get_classifier_name("Rf") == "Random Forest"
        assert get_classifier_name("xgb") == "XGBoost"


class TestIsValidClassifier:
    """Test is_valid_classifier() function."""

    def test_valid_codes(self):
        """Test that all defined codes are recognized as valid."""
        for code in CLASSIFIER_CODES:
            assert is_valid_classifier(code) is True

    def test_invalid_codes(self):
        """Test that undefined codes are recognized as invalid."""
        assert is_valid_classifier("INVALID") is False
        assert is_valid_classifier("ZZZ") is False
        assert is_valid_classifier("") is False

    def test_case_insensitivity(self):
        """Test that is_valid_classifier is case-insensitive."""
        assert is_valid_classifier("rf") is True
        assert is_valid_classifier("Svm") is True
        assert is_valid_classifier("knn") is True


class TestRequiresSklearn:
    """Test requires_sklearn() dependency check."""

    def test_sklearn_dependent_classifiers_return_true(self):
        """Test that all sklearn-dependent classifiers return True."""
        for code in SKLEARN_DEPENDENT:
            assert requires_sklearn(code) is True

    def test_gmm_does_not_require_sklearn(self):
        """Test that GMM (built-in) does not require sklearn."""
        assert requires_sklearn("GMM") is False

    def test_xgb_and_cb_do_not_require_sklearn(self):
        """Test that XGB and CB have their own dependencies, not sklearn."""
        assert requires_sklearn("XGB") is False
        assert requires_sklearn("CB") is False


class TestRequiresXgboost:
    """Test requires_xgboost() dependency check."""

    def test_xgb_returns_true(self):
        """Test that XGB requires XGBoost package."""
        assert requires_xgboost("XGB") is True

    def test_other_classifiers_return_false(self):
        """Test that non-XGB classifiers do not require XGBoost."""
        for code in CLASSIFIER_CODES:
            if code != "XGB":
                assert requires_xgboost(code) is False


class TestRequiresCatboost:
    """Test requires_catboost() dependency check."""

    def test_cb_returns_true(self):
        """Test that CB requires CatBoost package."""
        assert requires_catboost("CB") is True

    def test_other_classifiers_return_false(self):
        """Test that non-CB classifiers do not require CatBoost."""
        for code in CLASSIFIER_CODES:
            if code != "CB":
                assert requires_catboost(code) is False


class TestDependencyListConsistency:
    """Test consistency across dependency lists."""

    def test_no_overlaps_between_dependency_sets(self):
        """Test that sklearn, xgboost, and catboost sets are disjoint."""
        sklearn_set = set(SKLEARN_DEPENDENT)
        xgboost_set = set(XGBOOST_DEPENDENT)
        catboost_set = set(CATBOOST_DEPENDENT)

        assert sklearn_set & xgboost_set == set()
        assert sklearn_set & catboost_set == set()
        assert xgboost_set & catboost_set == set()

    def test_union_covers_all_except_gmm(self):
        """Test that all classifiers except GMM belong to exactly one dependency set."""
        all_dependent = set(SKLEARN_DEPENDENT) | set(XGBOOST_DEPENDENT) | set(CATBOOST_DEPENDENT)
        all_except_gmm = set(CLASSIFIER_CODES) - {"GMM"}

        assert all_dependent == all_except_gmm
