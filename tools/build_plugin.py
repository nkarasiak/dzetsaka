#!/usr/bin/env python3
"""Build dzetsaka QGIS plugin zip artifact."""

from __future__ import annotations

import argparse
import fnmatch
import re
import zipfile
from pathlib import Path

DEFAULT_OUTPUT = "dzetsaka.zip"
PLUGIN_DIR_NAME = "dzetsaka"

INCLUDED_TOP_LEVEL_DIRS = {
    "config",
    "domain",
    "factories",
    "img",
    "processing",
    "scripts",
    "services",
    "src",
    "ui",
}

INCLUDED_TOP_LEVEL_FILES = {
    "__init__.py",
    "classifier_config.py",
    "config.txt",
    "constants.py",
    "dzetsaka.py",
    "dzetsaka_provider.py",
    "icon.png",
    "LICENSE",
    "metadata.txt",
    "resources.py",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "docs",
    "htmlcov",
    "tests",
    "tools",
}

EXCLUDED_FILE_PATTERNS = [
    "*.md",
    "*.qrc",
    "*.ui",
    "*.pyc",
    "*.pyo",
    ".coverage",
    ".pre-commit-config.yaml",
    "coverage.*",
    "Makefile",
    "pyproject.toml",
    "zip_file.py",
]

EXCLUDED_PATH_PATTERNS = [
    "*.egg-info/*",
]


def _resource_image_paths(repo_root: Path) -> set[str]:
    qrc_file = repo_root / "resources.qrc"
    if not qrc_file.exists():
        return set()
    content = qrc_file.read_text(encoding="utf-8")
    return {
        match.group(1)
        for match in re.finditer(r"<file>([^<]+)</file>", content)
        if match.group(1).startswith("img/")
    }


def _is_included(rel_path: Path, qrc_images: set[str]) -> bool:
    parts = rel_path.parts
    if not parts:
        return False

    top_level = parts[0]
    if len(parts) == 1:
        return top_level in INCLUDED_TOP_LEVEL_FILES
    if top_level not in INCLUDED_TOP_LEVEL_DIRS:
        return False

    path_posix = rel_path.as_posix()
    if top_level != "img":
        return True

    # Keep modern UI icons on disk as a fallback for direct file loading.
    if path_posix.startswith("img/modern/"):
        return True

    # Keep images embedded via resources.py (resources.qrc source of truth).
    return path_posix in qrc_images


def _is_excluded(rel_path: Path, qrc_images: set[str]) -> bool:
    if not _is_included(rel_path, qrc_images):
        return True
    parts = rel_path.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts[:-1]):
        return True
    if any(part.startswith(".") and part not in (".", "..") for part in parts):
        return True
    path_posix = rel_path.as_posix()
    if any(fnmatch.fnmatch(path_posix, pattern) for pattern in EXCLUDED_PATH_PATTERNS):
        return True
    filename = rel_path.name
    if any(fnmatch.fnmatch(filename, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
        return filename.lower() != "readme.md"
    return False


def build_plugin_zip(repo_root: Path, output_zip: Path) -> int:
    qrc_images = _resource_image_paths(repo_root)
    files = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        if _is_excluded(rel, qrc_images):
            continue
        files.append(rel)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(files):
            zf.write(repo_root / rel, f"{PLUGIN_DIR_NAME}/{rel.as_posix()}")

    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dzetsaka plugin zip")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output zip path (default: dzetsaka.zip)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_zip = (repo_root / args.output).resolve()
    count = build_plugin_zip(repo_root=repo_root, output_zip=output_zip)
    print(f"Created {output_zip} with {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
