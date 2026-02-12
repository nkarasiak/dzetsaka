#!/usr/bin/env python3
"""Compatibility wrapper for plugin packaging."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tools.build_plugin import build_plugin_zip


def _get_version_from_metadata(metadata_file: Path) -> str | None:
    if not metadata_file.exists():
        return None
    content = metadata_file.read_text(encoding="utf-8")
    match = re.search(r"^version=(.+)$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _validate_resources(repo_root: Path) -> tuple[bool, str]:
    qrc_file = repo_root / "resources.qrc"
    py_file = repo_root / "resources.py"
    if not qrc_file.exists():
        return False, "Error: resources.qrc not found"
    if not py_file.exists():
        return False, "Error: resources.py not found. Compile resources.qrc before packaging."
    if qrc_file.stat().st_mtime > py_file.stat().st_mtime:
        return False, "Error: resources.py is older than resources.qrc. Recompile resources before packaging."
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build versioned dzetsaka QGIS plugin zip")
    parser.add_argument(
        "--output",
        help="Optional output zip path (default: ../dzetsaka_<version>.zip)",
    )
    parser.add_argument(
        "--skip-resource-check",
        action="store_true",
        help="Skip validation that resources.py is up to date with resources.qrc",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    version = _get_version_from_metadata(repo_root / "metadata.txt")
    if not version:
        print("Error: could not determine version from metadata.txt")
        return 1

    if not args.skip_resource_check:
        ok, message = _validate_resources(repo_root)
        if not ok:
            print(message)
            return 1

    output_zip = (Path(args.output).resolve() if args.output else repo_root.parent / f"dzetsaka_{version}.zip")
    count = build_plugin_zip(repo_root=repo_root, output_zip=output_zip)
    print(f"Created {output_zip} with {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
