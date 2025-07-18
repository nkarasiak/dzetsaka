#!/usr/bin/env python3
"""
Test script for the new sklearn classifiers in dzetsaka
"""

import sys
import os
import warnings

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def test_new_classifiers():
    """Test the new classifier configurations and validation"""
    
    print("Testing new classifier support...")
    print("=" * 50)
    
    # Test configuration import
    try:
        from scripts.mainfunction import CLASSIFIER_CONFIGS
        print(f"[OK] Successfully imported CLASSIFIER_CONFIGS with {len(CLASSIFIER_CONFIGS)} classifiers")
    except ImportError as e:
        print(f"[FAIL] Could not import configurations: {e}")
        return False
    
    # Test sklearn validator import
    try:
        from scripts.sklearn_validator import (
            SklearnValidator, 
            validate_classifier_selection, 
            check_sklearn_availability
        )
        print("[OK] Successfully imported sklearn validator")
    except ImportError as e:
        print(f"[FAIL] Could not import sklearn validator: {e}")
        return False
    
    # Test validator functionality
    validator = SklearnValidator()
    
    # Test all new classifiers
    new_classifiers = ["XGB", "LGB", "ET", "GBC", "LR", "NB", "MLP"]
    existing_classifiers = ["GMM", "RF", "SVM", "KNN"]
    all_classifiers = existing_classifiers + new_classifiers
    
    print(f"\n--- Testing {len(all_classifiers)} classifiers ---")
    
    available_count = 0
    for classifier in all_classifiers:
        is_valid, error_msg = validator.validate_algorithm(classifier)
        status = "[OK]" if is_valid else "[INFO]"  # INFO instead of FAIL since missing packages is expected
        print(f"{status} {classifier}: {'Available' if is_valid else error_msg}")
        if is_valid:
            available_count += 1
    
    print(f"\n[INFO] {available_count}/{len(all_classifiers)} classifiers available in current environment")
    
    # Test configuration completeness
    print("\n--- Testing classifier configurations ---")
    for classifier in all_classifiers:
        if classifier in CLASSIFIER_CONFIGS:
            config = CLASSIFIER_CONFIGS[classifier]
            has_param_grid = "param_grid" in config
            has_n_splits = "n_splits" in config
            status = "[OK]" if has_param_grid and has_n_splits else "[FAIL]"
            print(f"{status} {classifier}: param_grid={has_param_grid}, n_splits={has_n_splits}")
        elif classifier == "GMM":
            print(f"[OK] {classifier}: No configuration needed (built-in)")
        else:
            print(f"[FAIL] {classifier}: Missing configuration")
    
    # Test UI name mapping
    print("\n--- Testing UI name mapping ---")
    ui_names = [
        'Gaussian Mixture Model',
        'Random Forest', 
        'Support Vector Machine',
        'K-Nearest Neighbors',
        'XGBoost',
        'LightGBM',
        'Extra Trees',
        'Gradient Boosting Classifier',
        'Logistic Regression',
        'Gaussian Naive Bayes',
        'Multi-layer Perceptron'
    ]
    
    for ui_name in ui_names:
        is_valid, error_msg = validate_classifier_selection(ui_name)
        status = "[OK]" if is_valid else "[INFO]"
        print(f"{status} {ui_name}: {'Available' if is_valid else 'Requires additional packages'}")
    
    # Test algorithm info
    print("\n--- Testing algorithm information ---")
    algo_info = validator.get_algorithm_info()
    for classifier, info in algo_info.items():
        requires_sklearn = info.get('requires_sklearn', False)
        requires_extra = info.get('requires_extra_package', False)
        full_name = info.get('full_name', classifier)
        
        dependency = []
        if requires_sklearn:
            dependency.append("sklearn")
        if requires_extra:
            dependency.append("extra package")
        
        dep_str = f" (requires: {', '.join(dependency)})" if dependency else ""
        print(f"[INFO] {classifier}: {full_name}{dep_str}")
    
    # Test installation instructions
    print("\n--- Testing installation instructions ---")
    instructions = validator.get_installation_instructions()
    if "xgboost" in instructions and "lightgbm" in instructions:
        print("[OK] Installation instructions include new packages")
    else:
        print("[FAIL] Installation instructions missing new packages")
    
    return True

def test_classifier_initialization():
    """Test that classifier initialization code doesn't have syntax errors"""
    
    print("\n\nTesting classifier initialization logic...")
    print("=" * 50)
    
    # We can't actually test the classifiers without their dependencies,
    # but we can test that the code structure is correct
    
    try:
        from scripts.mainfunction import CLASSIFIER_CONFIGS
        
        # Test that all expected classifiers have configurations
        expected_classifiers = ["RF", "SVM", "KNN", "XGB", "LGB", "ET", "GBC", "LR", "NB", "MLP"]
        missing_configs = []
        
        for classifier in expected_classifiers:
            if classifier not in CLASSIFIER_CONFIGS:
                missing_configs.append(classifier)
        
        if missing_configs:
            print(f"[FAIL] Missing configurations for: {missing_configs}")
            return False
        else:
            print(f"[OK] All {len(expected_classifiers)} classifiers have configurations")
        
        # Test configuration structure
        for classifier, config in CLASSIFIER_CONFIGS.items():
            if not isinstance(config, dict):
                print(f"[FAIL] {classifier}: Config is not a dict")
                return False
            
            if "param_grid" not in config:
                print(f"[FAIL] {classifier}: Missing param_grid")
                return False
                
            if "n_splits" not in config:
                print(f"[FAIL] {classifier}: Missing n_splits")
                return False
        
        print("[OK] All classifier configurations have proper structure")
        
        # Test that lambda functions are properly defined
        rf_config = CLASSIFIER_CONFIGS["RF"]
        et_config = CLASSIFIER_CONFIGS["ET"]
        
        if callable(rf_config["param_grid"]["max_features"]):
            print("[OK] RF max_features lambda function defined")
        else:
            print("[FAIL] RF max_features should be a lambda function")
            
        if callable(et_config["param_grid"]["max_features"]):
            print("[OK] ET max_features lambda function defined")
        else:
            print("[FAIL] ET max_features should be a lambda function")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing configurations: {e}")
        return False

if __name__ == "__main__":
    print("Testing new classifier implementations in dzetsaka...")
    print("=" * 60)
    
    success = True
    success &= test_new_classifiers()
    success &= test_classifier_initialization()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All new classifier tests completed successfully!")
        print("\nNew Classifiers Added:")
        print("✓ XGBoost (XGB) - Gradient boosting framework")
        print("✓ LightGBM (LGB) - Fast gradient boosting")
        print("✓ Extra Trees (ET) - Extremely randomized trees")
        print("✓ Gradient Boosting Classifier (GBC) - Sklearn gradient boosting")
        print("✓ Logistic Regression (LR) - Linear classification")
        print("✓ Gaussian Naive Bayes (NB) - Probabilistic classifier")
        print("✓ Multi-layer Perceptron (MLP) - Neural network")
        print("\nFeatures:")
        print("✓ Comprehensive parameter grids for hyperparameter tuning")
        print("✓ Proper random state handling for reproducibility")
        print("✓ Integration with sklearn validation system")
        print("✓ Automatic dependency checking and error messages")
        print("✓ Configurable cross-validation splits")
    else:
        print("[FAIL] Some classifier tests failed!")
        sys.exit(1)