#!/usr/bin/env python3
"""Compatibility wrapper for plugin packaging."""

from __future__ import annotations

import re
from pathlib import Path

from tools.build_plugin import build_plugin_zip


def _get_version_from_metadata(metadata_file: Path) -> str | None:
    if not metadata_file.exists():
        return None
    content = metadata_file.read_text(encoding="utf-8")
    match = re.search(r"^version=(.+)$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    version = _get_version_from_metadata(repo_root / "metadata.txt")
    if not version:
        print("Error: could not determine version from metadata.txt")
        return 1

    output_zip = repo_root.parent / f"dzetsaka_{version}.zip"
    count = build_plugin_zip(repo_root=repo_root, output_zip=output_zip)
    print(f"Created {output_zip} with {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
