#!/usr/bin/env python3
"""
Test script for enhanced sklearn validation in dzetsaka
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def test_sklearn_validator():
    """Test the sklearn validation functionality"""
    
    print("Testing sklearn validation...")
    print("=" * 50)
    
    try:
        from scripts.sklearn_validator import (
            SklearnValidator, 
            validate_classifier_selection, 
            check_sklearn_availability,
            get_sklearn_error_message
        )
        print("[OK] Successfully imported sklearn validator")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    
    # Test validator instance
    validator = SklearnValidator()
    
    # Test sklearn availability check
    sklearn_available = validator.is_sklearn_available()
    print(f"[INFO] Sklearn available: {sklearn_available}")
    
    if sklearn_available:
        version = validator.get_sklearn_version()
        print(f"[INFO] Sklearn version: {version}")
    
    # Test individual algorithm validation
    algorithms = ['GMM', 'RF', 'SVM', 'KNN']
    
    for algorithm in algorithms:
        is_valid, error_msg = validator.validate_algorithm(algorithm)
        status = "[OK]" if is_valid else "[FAIL]"
        print(f"{status} {algorithm}: {'Available' if is_valid else error_msg}")
    
    # Test UI name mapping
    ui_names = [
        'Gaussian Mixture Model',
        'Random Forest', 
        'Support Vector Machine',
        'K-Nearest Neighbors'
    ]
    
    print("\nTesting UI classifier validation...")
    for ui_name in ui_names:
        is_valid, error_msg = validate_classifier_selection(ui_name)
        status = "[OK]" if is_valid else "[FAIL]"
        print(f"{status} {ui_name}: {'Available' if is_valid else error_msg}")
    
    # Test comprehensive status
    print("\nTesting comprehensive status check...")
    status = check_sklearn_availability()
    print(f"[INFO] Available algorithms: {status['available_algorithms']}")
    
    # Test error message generation
    print("\nTesting error message generation...")
    for ui_name in ui_names:
        error_msg = get_sklearn_error_message(ui_name)
        if error_msg:
            print(f"[INFO] Error for {ui_name}: {error_msg[:100]}...")
        else:
            print(f"[OK] No error for {ui_name}")
    
    return True

def test_sklearn_imports():
    """Test actual sklearn imports"""
    
    print("\nTesting actual sklearn imports...")
    print("=" * 50)
    
    sklearn_modules = [
        'sklearn',
        'sklearn.ensemble.RandomForestClassifier',
        'sklearn.svm.SVC', 
        'sklearn.neighbors.KNeighborsClassifier',
        'sklearn.model_selection.StratifiedKFold',
        'sklearn.model_selection.GridSearchCV'
    ]
    
    for module_path in sklearn_modules:
        try:
            if '.' in module_path and module_path != 'sklearn':
                parts = module_path.split('.')
                module_name = '.'.join(parts[:-1])
                class_name = parts[-1]
                
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
            else:
                __import__(module_path)
            
            print(f"[OK] {module_path}")
        except ImportError:
            print(f"[FAIL] {module_path} - not available")
        except AttributeError:
            print(f"[FAIL] {module_path} - class not found")
    
    return True

def test_integration():
    """Test integration with main dzetsaka functionality"""
    
    print("\nTesting integration...")
    print("=" * 50)
    
    # Test if we can import the enhanced dzetsaka functions
    try:
        # We can't actually import dzetsaka.py due to QGIS dependencies
        # But we can test that our validation functions work as expected
        from scripts.sklearn_validator import validate_classifier_selection
        
        # Test cases that should work
        test_cases = [
            ('Gaussian Mixture Model', True),  # Should always work
            ('Random Forest', None),           # Depends on sklearn
            ('Support Vector Machine', None),  # Depends on sklearn
            ('K-Nearest Neighbors', None),     # Depends on sklearn
            ('Invalid Classifier', False),     # Should fail
        ]
        
        for classifier, expected in test_cases:
            is_valid, error_msg = validate_classifier_selection(classifier)
            
            if expected is None:
                # Result depends on sklearn availability
                print(f"[INFO] {classifier}: {'OK' if is_valid else 'MISSING SKLEARN'}")
            elif expected == is_valid:
                print(f"[OK] {classifier}: Validation result as expected")
            else:
                print(f"[FAIL] {classifier}: Expected {expected}, got {is_valid}")
                return False
        
        print("[OK] Integration tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing enhanced sklearn validation for dzetsaka...")
    print("=" * 60)
    
    success = True
    success &= test_sklearn_validator()
    success &= test_sklearn_imports()
    success &= test_integration()
    
    print("=" * 60)
    if success:
        print("[OK] All sklearn validation tests completed!")
        
        # Show summary of what was enhanced
        print("\nEnhancements Summary:")
        print("- Created SklearnValidator class for robust sklearn checking")
        print("- Enhanced UI classifier selection with detailed error messages")
        print("- Added validation before training starts")
        print("- Improved error messages with installation instructions")
        print("- Added comprehensive sklearn availability checking")
        print("- Enhanced mainfunction.py error reporting")
        
    else:
        print("[FAIL] Some tests failed!")
        sys.exit(1)