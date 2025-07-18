#!/usr/bin/env python3
"""
Script to regenerate Python UI files from .ui files after adding new classifiers.

This script helps regenerate the Python UI files when changes are made to the .ui files.
The UI files now include all 11 classifiers instead of just the original 4.

Requirements:
- PyQt5 development tools (pyuic5)
- Install with: pip install PyQt5-tools

Usage:
    python regenerate_ui.py
"""

import os
import subprocess
import sys

def regenerate_ui_files():
    """Regenerate Python UI files from Qt Designer .ui files"""
    
    ui_files = [
        ('ui/dzetsaka_dock.ui', 'ui/dzetsaka_dock.py'),
        ('ui/settings_dock.ui', 'ui/settings_dock.py')
    ]
    
    print("Regenerating Python UI files from .ui files...")
    print("=" * 50)
    
    # Check if pyuic5 is available
    try:
        result = subprocess.run(['pyuic5', '--version'], capture_output=True, text=True)
        print(f"Found pyuic5: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: pyuic5 not found!")
        print("Install PyQt5 development tools with:")
        print("  pip install PyQt5-tools")
        print("\nAlternatively, you can use:")
        print("  python -m PyQt5.uic.pyuic ui_file.ui -o output_file.py")
        return False
    
    success = True
    
    for ui_file, py_file in ui_files:
        if not os.path.exists(ui_file):
            print(f"WARNING: {ui_file} not found, skipping...")
            continue
            
        print(f"Regenerating {py_file} from {ui_file}...")
        
        try:
            # Backup existing Python file
            if os.path.exists(py_file):
                backup_file = py_file + '.backup'
                print(f"  Creating backup: {backup_file}")
                with open(py_file, 'r') as f:
                    content = f.read()
                with open(backup_file, 'w') as f:
                    f.write(content)
            
            # Regenerate Python file
            result = subprocess.run([
                'pyuic5', 
                ui_file, 
                '-o', py_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ Successfully generated {py_file}")
            else:
                print(f"  ✗ Error generating {py_file}:")
                print(f"    {result.stderr}")
                success = False
                
        except Exception as e:
            print(f"  ✗ Exception while processing {ui_file}: {e}")
            success = False
    
    return success

def verify_classifiers():
    """Verify that the UI files contain all expected classifiers"""
    
    expected_classifiers = [
        "Gaussian Mixture Model",
        "Random Forest", 
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "XGBoost",
        "LightGBM",
        "Extra Trees",
        "Gradient Boosting Classifier",
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "Multi-layer Perceptron"
    ]
    
    print("\nVerifying classifier coverage in UI files...")
    print("=" * 50)
    
    ui_files = ['ui/dzetsaka_dock.ui', 'ui/settings_dock.ui']
    
    for ui_file in ui_files:
        if not os.path.exists(ui_file):
            print(f"WARNING: {ui_file} not found")
            continue
            
        with open(ui_file, 'r') as f:
            content = f.read()
        
        print(f"\nChecking {ui_file}:")
        missing = []
        
        for classifier in expected_classifiers:
            if classifier in content:
                print(f"  ✓ {classifier}")
            else:
                print(f"  ✗ {classifier} - MISSING")
                missing.append(classifier)
        
        if missing:
            print(f"  ERROR: {len(missing)} classifiers missing from {ui_file}")
        else:
            print(f"  SUCCESS: All {len(expected_classifiers)} classifiers found!")

def main():
    """Main function"""
    print("dzetsaka UI Regeneration Tool")
    print("Adding support for 7 new machine learning classifiers")
    print("=" * 60)
    
    # Verify classifier coverage first
    verify_classifiers()
    
    # Regenerate UI files
    if regenerate_ui_files():
        print("\n" + "=" * 60)
        print("SUCCESS: UI files regenerated successfully!")
        print("\nNext steps:")
        print("1. Test the UI in QGIS to ensure all classifiers appear")
        print("2. Try selecting different classifiers to test validation")
        print("3. Install additional packages if needed:")
        print("   pip install xgboost lightgbm")
    else:
        print("\n" + "=" * 60)
        print("FAILED: Some UI files could not be regenerated")
        print("You may need to:")
        print("1. Install PyQt5 tools: pip install PyQt5-tools")
        print("2. Check that .ui files are properly formatted")
        print("3. Manually regenerate with: pyuic5 ui_file.ui -o output.py")

if __name__ == "__main__":
    main()