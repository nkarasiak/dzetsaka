#!/usr/bin/env python3
"""QGIS Plugin Packaging Script for dzetsaka.

Creates a clean ZIP package for publishing to QGIS Plugin Repository.
"""

import fnmatch
import os
import re
import sys
import zipfile

# Fix encoding issues on Windows
if sys.platform.startswith("win"):
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


def get_version_from_metadata(metadata_file="metadata.txt"):
    """Extract version from metadata.txt file."""
    try:
        with open(metadata_file, encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"version=(.+)", content)
            if match:
                return match.group(1).strip()
    except FileNotFoundError:
        print(f"Error: {metadata_file} not found")
        return None
    return None


def should_exclude(path):
    """Check if a file or directory should be excluded from the package."""
    # readme.md is used by the QGIS Plugin Repository as the long description
    if os.path.basename(path).lower() == "readme.md":
        return False

    exclude_patterns = [
        # Build / cache artefacts
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.py~",
        # Qt source files (generated .py files ARE included)
        "*.qrc",
        "*.ui",
        # Assets (loaded at runtime via resources.py)
        "img/*",
        # Dev tooling & scripts
        "Makefile",
        "conftest.py",
        "zip_file.py",
        "*.sh",
        "*.bat",
        # Dev / CI config
        "pyproject.toml",
        ".git*",
        # Dev-only directories
        "tests",
        # Documentation (readme.md whitelisted above)
        "*.md",
        # Coverage artefacts
        "htmlcov",
        "coverage.xml",
        "coverage.*",
        # Packaging output
        "*.zip",
        # Windows reserved device names
        "nul",
    ]

    # Exclude any folder or file starting with a dot
    if any(part.startswith(".") for part in path.split(os.sep)):
        return True

    # Check against exclude patterns
    return any(fnmatch.fnmatch(path, pattern) or pattern.rstrip("/*") in path for pattern in exclude_patterns)


def create_plugin_package(output_dir=".."):
    """Create the QGIS plugin package."""
    # Get version from metadata
    version = get_version_from_metadata()
    if not version:
        print("Error: Could not determine version from metadata.txt")
        return False

    # Create output filename
    zip_filename = f"dzetsaka_{version}.zip"
    zip_path = os.path.join(output_dir, zip_filename)

    print(f"Creating package for version: {version}")
    print(f"Output file: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            file_count = 0

            for root, dirs, files in os.walk("."):
                # Remove excluded directories from dirs list to prevent recursion
                dirs[:] = [d for d in dirs if not should_exclude(d)]

                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, ".")

                    if not should_exclude(rel_path):
                        # Store files inside dzetsaka folder
                        zip_path_in_archive = os.path.join("dzetsaka", rel_path)
                        zipf.write(file_path, zip_path_in_archive)
                        file_count += 1
                        print(f"  Added: {zip_path_in_archive}")

            print("\nPackage created successfully!")
            print(f"Files included: {file_count}")
            print(f"Output: {zip_path}")

        # Show file size
        file_size = os.path.getsize(zip_path)
        size_mb = file_size / (1024 * 1024)
        print(f"Package size: {file_size:,} bytes ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        print(f"Error creating package: {e}")
        return False


def main():
    """Main function."""
    print("QGIS Plugin Packager for dzetsaka")
    print("=" * 40)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create the package
    success = create_plugin_package()

    if success:
        print("\n✅ Package ready for QGIS Plugin Repository upload!")
    else:
        print("\n❌ Package creation failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
