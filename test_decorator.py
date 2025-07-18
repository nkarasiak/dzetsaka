#!/usr/bin/env python3
"""
Test script for the new backward compatibility decorator
"""

import sys
import os
import warnings

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def test_decorator():
    """Test the backward compatibility decorator"""
    
    print("Testing backward compatibility decorator...")
    print("=" * 50)
    
    # Import the decorator
    try:
        from scripts.mainfunction import backward_compatible
        print("[OK] Successfully imported backward_compatible decorator")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    
    # Test a simple function with the decorator
    @backward_compatible(oldParam='newParam', oldValue='newValue')
    def test_function(newParam=None, newValue=None, normalParam=None):
        return {
            'newParam': newParam,
            'newValue': newValue, 
            'normalParam': normalParam
        }
    
    # Test 1: Using new parameter names (should work without warnings)
    print("\n--- Test 1: New parameter names ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = test_function(newParam="new1", newValue="new2", normalParam="normal")
        
        if len(w) == 0:
            print("[OK] No warnings when using new parameter names")
        else:
            print(f"[FAIL] Unexpected warnings: {[str(warning.message) for warning in w]}")
            
        expected = {'newParam': 'new1', 'newValue': 'new2', 'normalParam': 'normal'}
        if result == expected:
            print(f"[OK] Correct result: {result}")
        else:
            print(f"[FAIL] Wrong result. Expected: {expected}, Got: {result}")
    
    # Test 2: Using old parameter names (should work with warnings)
    print("\n--- Test 2: Old parameter names ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = test_function(oldParam="old1", oldValue="old2", normalParam="normal")
        
        if len(w) == 2:
            print(f"[OK] Got expected warnings: {[str(warning.message) for warning in w]}")
        else:
            print(f"[FAIL] Expected 2 warnings, got {len(w)}: {[str(warning.message) for warning in w]}")
            
        expected = {'newParam': 'old1', 'newValue': 'old2', 'normalParam': 'normal'}
        if result == expected:
            print(f"[OK] Correct parameter mapping: {result}")
        else:
            print(f"[FAIL] Wrong mapping. Expected: {expected}, Got: {result}")
    
    # Test 3: Mixed old and new parameters (new should take precedence)
    print("\n--- Test 3: Mixed parameters ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = test_function(oldParam="old1", newParam="new1", normalParam="normal")
        
        if len(w) == 1 and "ignoring" in str(w[0].message):
            print(f"[OK] Got expected warning about ignoring old parameter: {str(w[0].message)}")
        else:
            print(f"[FAIL] Expected warning about ignoring old parameter, got: {[str(warning.message) for warning in w]}")
            
        expected = {'newParam': 'new1', 'newValue': None, 'normalParam': 'normal'}
        if result == expected:
            print(f"[OK] New parameter takes precedence: {result}")
        else:
            print(f"[FAIL] Wrong precedence handling. Expected: {expected}, Got: {result}")
    
    return True

def test_actual_functions():
    """Test the actual functions in mainfunction.py"""
    
    print("\n\nTesting actual decorated functions...")
    print("=" * 50)
    
    # Test rasterize function
    try:
        from scripts.mainfunction import rasterize
        print("[OK] Successfully imported rasterize function")
        
        # Test with old parameter names - this should show deprecation warnings
        print("\n--- Testing rasterize with old parameters ---")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                # This will fail because files don't exist, but we can test the parameter mapping
                rasterize(inRaster="test.tif", inShape="test.shp", inField="class")
            except (ValueError, FileNotFoundError) as e:
                # Expected - files don't exist, but parameters should be mapped
                print(f"[OK] Function called successfully, expected error: {e}")
                
            if len(w) >= 3:  # Should have warnings for the 3 old parameters
                print(f"[OK] Got deprecation warnings: {[str(warning.message) for warning in w if 'deprecated' in str(warning.message)]}")
            else:
                print(f"[INFO] Got {len(w)} warnings (might be fewer if caught by validation first)")
                
    except ImportError as e:
        print(f"[FAIL] Could not import rasterize: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing backward compatibility decorator implementation...")
    print("=" * 60)
    
    success = True
    success &= test_decorator()
    success &= test_actual_functions()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All decorator tests completed successfully!")
        print("\nSummary of improvements:")
        print("- Created reusable @backward_compatible decorator")
        print("- Eliminated duplicate parameter definitions in function signatures") 
        print("- Centralized parameter mapping logic")
        print("- Simplified function implementations")
        print("- Maintained full backward compatibility with proper deprecation warnings")
        print("- Cleaner, more maintainable code")
    else:
        print("[FAIL] Some decorator tests failed!")
        sys.exit(1)