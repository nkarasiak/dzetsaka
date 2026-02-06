"""Compatibility shim for migrated processing metadata helpers."""

from __future__ import annotations

from dzetsaka.presentation.qgis.processing.metadata_helpers import (  # noqa: F401
    get_algorithm_specific_tags,
    get_common_tags,
    get_group_id,
    get_help_url,
)
