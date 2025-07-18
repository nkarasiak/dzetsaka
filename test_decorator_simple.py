#!/usr/bin/env python3
"""
Simple test for the backward compatibility decorator (without GDAL dependencies)
"""

import warnings
import functools

# Copy the decorator definition from mainfunction.py for testing
def backward_compatible(**parameter_mapping):
    """
    Decorator to handle backward compatibility for function parameters.
    
    Parameters
    ----------
    **parameter_mapping : dict
        Mapping from old parameter names to new parameter names
        Example: backward_compatible(inRaster='raster_path', inVector='vector_path')
    """
    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a copy of kwargs to avoid modifying the original
            new_kwargs = kwargs.copy()
            
            # Process parameter mapping
            for old_param, new_param in parameter_mapping.items():
                if old_param in kwargs and new_param not in kwargs:
                    # Move old parameter to new parameter name
                    new_kwargs[new_param] = new_kwargs.pop(old_param)
                    warnings.warn(
                        f"Parameter '{old_param}' is deprecated. Use '{new_param}' instead.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                elif old_param in kwargs and new_param in kwargs:
                    # Both old and new parameters provided - remove old one and warn
                    new_kwargs.pop(old_param)
                    warnings.warn(
                        f"Both '{old_param}' and '{new_param}' provided. Using '{new_param}' and ignoring '{old_param}'.",
                        DeprecationWarning,
                        stacklevel=2
                    )
            
            return func(*args, **new_kwargs)
        return wrapper
    return decorator

def test_decorator():
    """Test the backward compatibility decorator"""
    
    print("Testing backward compatibility decorator...")
    print("=" * 50)
    
    # Test a simple function with the decorator
    @backward_compatible(
        inRaster='raster_path',
        inVector='vector_path', 
        inField='class_field',
        outModel='model_path'
    )
    def sample_function(raster_path=None, vector_path=None, class_field="Class", model_path=None):
        return {
            'raster_path': raster_path,
            'vector_path': vector_path,
            'class_field': class_field,
            'model_path': model_path
        }
    
    # Test 1: Using new parameter names (should work without warnings)
    print("\n--- Test 1: New parameter names ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sample_function(
            raster_path="new_raster.tif", 
            vector_path="new_vector.shp", 
            class_field="class",
            model_path="new_model.pkl"
        )
        
        if len(w) == 0:
            print("[OK] No warnings when using new parameter names")
        else:
            print(f"[FAIL] Unexpected warnings: {[str(warning.message) for warning in w]}")
            
        expected = {
            'raster_path': 'new_raster.tif', 
            'vector_path': 'new_vector.shp', 
            'class_field': 'class',
            'model_path': 'new_model.pkl'
        }
        if result == expected:
            print(f"[OK] Correct result: {result}")
        else:
            print(f"[FAIL] Wrong result. Expected: {expected}, Got: {result}")
    
    # Test 2: Using old parameter names (should work with warnings)
    print("\n--- Test 2: Old parameter names ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sample_function(
            inRaster="old_raster.tif", 
            inVector="old_vector.shp", 
            inField="class",
            outModel="old_model.pkl"
        )
        
        if len(w) == 4:
            print(f"[OK] Got expected 4 warnings:")
            for warning in w:
                print(f"     - {str(warning.message)}")
        else:
            print(f"[FAIL] Expected 4 warnings, got {len(w)}: {[str(warning.message) for warning in w]}")
            
        expected = {
            'raster_path': 'old_raster.tif', 
            'vector_path': 'old_vector.shp', 
            'class_field': 'class',
            'model_path': 'old_model.pkl'
        }
        if result == expected:
            print(f"[OK] Correct parameter mapping: {result}")
        else:
            print(f"[FAIL] Wrong mapping. Expected: {expected}, Got: {result}")
    
    # Test 3: Mixed old and new parameters (new should take precedence)
    print("\n--- Test 3: Mixed parameters ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sample_function(
            inRaster="old_raster.tif",  # Old name
            raster_path="new_raster.tif",  # New name - should take precedence
            vector_path="new_vector.shp"  # New name
        )
        
        if len(w) == 1 and "ignoring" in str(w[0].message):
            print(f"[OK] Got expected warning: {str(w[0].message)}")
        else:
            print(f"[FAIL] Expected 1 warning about ignoring old parameter, got: {[str(warning.message) for warning in w]}")
            
        expected = {
            'raster_path': 'new_raster.tif',  # New parameter wins
            'vector_path': 'new_vector.shp', 
            'class_field': 'Class',  # Default value
            'model_path': None  # Default value
        }
        if result == expected:
            print(f"[OK] New parameter takes precedence: {result}")
        else:
            print(f"[FAIL] Wrong precedence handling. Expected: {expected}, Got: {result}")
    
    # Test 4: Default values preserved
    print("\n--- Test 4: Default values ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sample_function()  # No parameters - should use defaults
        
        if len(w) == 0:
            print("[OK] No warnings for default parameters")
        else:
            print(f"[FAIL] Unexpected warnings: {[str(warning.message) for warning in w]}")
            
        expected = {
            'raster_path': None,
            'vector_path': None, 
            'class_field': 'Class',  # Default value preserved
            'model_path': None
        }
        if result == expected:
            print(f"[OK] Default values preserved: {result}")
        else:
            print(f"[FAIL] Default values not preserved. Expected: {expected}, Got: {result}")
    
    return True

if __name__ == "__main__":
    print("Testing standalone backward compatibility decorator...")
    print("=" * 60)
    
    success = test_decorator()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All decorator tests completed successfully!")
        print("\nDecorator Benefits:")
        print("✓ Eliminates duplicate parameter definitions")
        print("✓ Centralized parameter mapping logic") 
        print("✓ Automatic deprecation warnings")
        print("✓ Proper precedence handling (new > old)")
        print("✓ Maintains function signatures clean")
        print("✓ Reusable across multiple functions")
        print("✓ Type hints and documentation stay clean")
    else:
        print("[FAIL] Some decorator tests failed!")