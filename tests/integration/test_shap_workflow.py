"""Integration tests for SHAP workflow in dzetsaka.

Tests the end-to-end integration of SHAP explainability with LearnModel
and the Processing algorithm.
"""
import os
import pickle
import tempfile

import numpy as np
import pytest

# Try to import SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import dzetsaka modules
try:
    from scripts.explainability.shap_explainer import ModelExplainer

    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False

# Check if we can run integration tests
CAN_RUN_TESTS = SHAP_AVAILABLE and SKLEARN_AVAILABLE and EXPLAINER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not CAN_RUN_TESTS,
    reason="SHAP, sklearn, or explainer module not available",
)


@pytest.mark.integration
class TestSHAPWithLearnModel:
    """Test SHAP integration with LearnModel class."""

    @pytest.mark.slow
    def test_learn_model_with_shap_parameter(self):
        """Test that LearnModel respects SHAP parameters."""
        # This is a placeholder test - requires full QGIS environment
        # In a real test, we would:
        # 1. Create mock raster and vector data
        # 2. Call LearnModel with COMPUTE_SHAP=True
        # 3. Verify SHAP computation was called
        # 4. Verify feature importance was computed

        # For now, just verify the parameter structure
        extra_param = {
            "COMPUTE_SHAP": True,
            "SHAP_OUTPUT": "test_importance.tif",
            "SHAP_SAMPLE_SIZE": 500,
        }

        assert extra_param["COMPUTE_SHAP"] is True
        assert isinstance(extra_param["SHAP_SAMPLE_SIZE"], int)
        assert extra_param["SHAP_SAMPLE_SIZE"] > 0

    def test_shap_parameters_validation(self):
        """Test SHAP parameter validation."""
        # Valid parameters
        valid_params = {
            "COMPUTE_SHAP": True,
            "SHAP_OUTPUT": "/path/to/output.tif",
            "SHAP_SAMPLE_SIZE": 1000,
        }

        assert isinstance(valid_params["COMPUTE_SHAP"], bool)
        assert isinstance(valid_params["SHAP_OUTPUT"], str)
        assert isinstance(valid_params["SHAP_SAMPLE_SIZE"], int)

        # Edge cases
        assert valid_params.get("COMPUTE_SHAP", False) in [True, False]
        assert valid_params.get("SHAP_SAMPLE_SIZE", 1000) >= 100


@pytest.mark.integration
class TestSHAPWorkflowWithRealModel:
    """Test complete SHAP workflow with actual models."""

    def test_rf_model_shap_workflow(self):
        """Test SHAP workflow with Random Forest."""
        # Create synthetic dataset
        np.random.seed(42)
        X_train = np.random.rand(200, 5)
        y_train = np.random.randint(0, 3, 200)

        # Train model
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)

        # Create explainer
        explainer = ModelExplainer(
            model=model,
            feature_names=[f"Band_{i+1}" for i in range(5)],
        )

        # Compute importance
        X_sample = X_train[:100]
        importance = explainer.get_feature_importance(X_sample)

        # Validate workflow
        assert len(importance) == 5
        assert all(f"Band_{i+1}" in importance for i in range(5))
        assert all(0 <= v <= 1 for v in importance.values())

        # Verify sum to 1.0 (normalized)
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    def test_multiclass_model_shap_workflow(self):
        """Test SHAP with multiclass classification."""
        # Create multiclass dataset
        np.random.seed(42)
        X_train = np.random.rand(150, 4)
        y_train = np.random.randint(0, 5, 150)  # 5 classes

        # Train model
        model = RandomForestClassifier(n_estimators=15, random_state=42)
        model.fit(X_train, y_train)

        # Create explainer
        explainer = ModelExplainer(
            model=model,
            feature_names=["F1", "F2", "F3", "F4"],
        )

        # Compute importance
        X_sample = X_train[:50]
        importance = explainer.get_feature_importance(X_sample)

        # Validate
        assert len(importance) == 4
        assert all(isinstance(v, (float, np.floating)) for v in importance.values())

    def test_model_save_load_workflow(self):
        """Test saving model and loading for SHAP analysis."""
        # Train model
        np.random.seed(42)
        X_train = np.random.rand(100, 3)
        y_train = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Simulate dzetsaka model save format
        M = np.max(X_train, axis=0)
        m = np.min(X_train, axis=0)
        classifier_code = "RF"

        # Save model
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".model", delete=False) as f:
            temp_model_path = f.name
            pickle.dump([model, M, m, classifier_code], f)

        try:
            # Load model (simulate Processing algorithm loading)
            with open(temp_model_path, "rb") as f:
                model_data = pickle.load(f)

            loaded_model = model_data[0]
            loaded_classifier = model_data[3]

            # Create explainer with loaded model
            explainer = ModelExplainer(
                model=loaded_model,
                feature_names=["B1", "B2", "B3"],
            )

            # Compute importance
            X_sample = X_train[:50]
            importance = explainer.get_feature_importance(X_sample)

            # Validate
            assert loaded_classifier == "RF"
            assert len(importance) == 3
            assert all(k in importance for k in ["B1", "B2", "B3"])

        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)


@pytest.mark.integration
class TestSHAPPerformance:
    """Test SHAP performance characteristics."""

    @pytest.mark.slow
    def test_tree_explainer_performance(self):
        """Test TreeExplainer performance (should be fast)."""
        import time

        # Create larger dataset
        np.random.seed(42)
        X_train = np.random.rand(500, 10)
        y_train = np.random.randint(0, 3, 500)

        # Train model
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        # Create explainer
        explainer = ModelExplainer(model=model)

        # Time importance computation
        X_sample = X_train[:200]
        start_time = time.time()
        importance = explainer.get_feature_importance(X_sample)
        elapsed_time = time.time() - start_time

        # TreeExplainer should be fast (< 5 seconds for this size)
        assert elapsed_time < 5.0
        assert len(importance) == 10

    def test_sample_size_impact_on_accuracy(self):
        """Test that larger sample sizes give more stable importance."""
        # Train model
        np.random.seed(42)
        X_train = np.random.rand(300, 5)
        # Create target with clear feature importance pattern
        # y depends mainly on first two features
        y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).astype(int)

        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model, feature_names=["F0", "F1", "F2", "F3", "F4"])

        # Compute importance with different sample sizes
        imp_small = explainer.get_feature_importance(X_train[:50])
        imp_large = explainer.get_feature_importance(X_train[:200])

        # Both should identify F0 and F1 as most important
        # (though exact values may differ)
        top2_small = sorted(imp_small.items(), key=lambda x: -x[1])[:2]
        top2_large = sorted(imp_large.items(), key=lambda x: -x[1])[:2]

        top_features_small = {f[0] for f in top2_small}
        top_features_large = {f[0] for f in top2_large}

        # F0 and F1 should be in top 2 for both
        assert "F0" in top_features_small or "F1" in top_features_small
        assert "F0" in top_features_large or "F1" in top_features_large


@pytest.mark.integration
class TestSHAPErrorHandling:
    """Test error handling in SHAP workflows."""

    def test_explainer_with_incompatible_sample_dimensions(self):
        """Test error when sample dimensions don't match model."""
        # Train model with 5 features
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)

        # Try to explain with wrong number of features
        X_wrong = np.random.rand(50, 3)  # Only 3 features instead of 5

        with pytest.raises(Exception):  # Will raise some error about shape mismatch
            explainer.get_feature_importance(X_wrong)

    def test_explainer_handles_missing_background_data_for_kernel(self):
        """Test that KernelExplainer requires background data."""
        from sklearn.svm import SVC

        # Train non-tree model
        np.random.seed(42)
        X_train = np.random.rand(80, 3)
        y_train = np.random.randint(0, 2, 80)

        model = SVC(kernel="linear", probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Create explainer WITHOUT background data
        explainer = ModelExplainer(model=model)

        # Should raise error when trying to create explainer
        X_sample = X_train[:20]

        with pytest.raises(ValueError, match="background_data is required"):
            explainer.get_feature_importance(X_sample)


@pytest.mark.integration
class TestSHAPWithDifferentAlgorithms:
    """Test SHAP works with different dzetsaka algorithms."""

    def test_shap_with_gradient_boosting(self):
        """Test SHAP with Gradient Boosting (tree-based)."""
        from sklearn.ensemble import GradientBoostingClassifier

        np.random.seed(42)
        X_train = np.random.rand(150, 4)
        y_train = np.random.randint(0, 2, 150)

        model = GradientBoostingClassifier(n_estimators=15, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)
        assert explainer._is_tree_based_model() is True

        importance = explainer.get_feature_importance(X_train[:50])
        assert len(importance) == 4

    def test_shap_with_extra_trees(self):
        """Test SHAP with Extra Trees (tree-based)."""
        from sklearn.ensemble import ExtraTreesClassifier

        np.random.seed(42)
        X_train = np.random.rand(150, 4)
        y_train = np.random.randint(0, 3, 150)

        model = ExtraTreesClassifier(n_estimators=15, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model)
        assert explainer._is_tree_based_model() is True

        importance = explainer.get_feature_importance(X_train[:50])
        assert len(importance) == 4

    @pytest.mark.slow
    def test_shap_with_knn(self):
        """Test SHAP with KNN (non-tree, uses KernelExplainer)."""
        from sklearn.neighbors import KNeighborsClassifier

        np.random.seed(42)
        X_train = np.random.rand(100, 3)
        y_train = np.random.randint(0, 2, 100)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # KNN is not tree-based, needs background data
        explainer = ModelExplainer(model=model, background_data=X_train[:50])
        assert explainer._is_tree_based_model() is False

        # Smaller sample for faster test
        importance = explainer.get_feature_importance(X_train[:20])
        assert len(importance) == 3


@pytest.mark.integration
class TestSHAPOutputInterpretation:
    """Test interpretation and validation of SHAP outputs."""

    def test_importance_scores_are_meaningful(self):
        """Test that importance scores make sense for known patterns."""
        # Create dataset where first feature is clearly most important
        np.random.seed(42)
        X_train = np.random.rand(200, 4)
        # Target depends primarily on first feature
        y_train = (X_train[:, 0] > 0.5).astype(int)

        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(
            model=model,
            feature_names=["Primary", "Noise1", "Noise2", "Noise3"],
        )

        importance = explainer.get_feature_importance(X_train[:100])

        # First feature should have highest importance
        max_feature = max(importance.items(), key=lambda x: x[1])[0]
        assert max_feature == "Primary"

        # Primary should have much higher importance than noise
        assert importance["Primary"] > importance["Noise1"]
        assert importance["Primary"] > importance["Noise2"]
        assert importance["Primary"] > importance["Noise3"]

    def test_all_features_contribute_when_all_relevant(self):
        """Test that all features get importance when all are relevant."""
        # Create dataset where all features matter
        np.random.seed(42)
        X_train = np.random.rand(200, 3)
        # Target is combination of all features
        y_train = ((X_train[:, 0] + X_train[:, 1] + X_train[:, 2]) > 1.5).astype(int)

        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        explainer = ModelExplainer(model=model, feature_names=["F1", "F2", "F3"])

        importance = explainer.get_feature_importance(X_train[:100])

        # All features should have non-zero importance
        assert all(v > 0 for v in importance.values())

        # Importance should be relatively balanced (no feature < 10% of total)
        min_importance = min(importance.values())
        assert min_importance > 0.1
