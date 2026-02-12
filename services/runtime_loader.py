"""Utilities for loading Python modules from explicit file paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module_from_path(module_name: str, module_path: Path):
    """Load and return a module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
