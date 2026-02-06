"""Helpers for reading plugin metadata values."""

from __future__ import annotations

import configparser
import os


def read_plugin_version(plugin_dir: str, logger=None) -> str:
    """Read plugin version from metadata.txt, fallback to 'unknown'."""
    metadata_path = os.path.join(plugin_dir, "metadata.txt")
    parser = configparser.ConfigParser()
    try:
        if parser.read(metadata_path, encoding="utf-8") and parser.has_section("general"):
            version = parser.get("general", "version", fallback="unknown").strip()
            return version or "unknown"
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Unable to read plugin version from metadata.txt: {exc}")
    return "unknown"

