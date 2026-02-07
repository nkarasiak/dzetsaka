"""Recipe schema v2 helpers with backward-compatible migration from legacy recipes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

SCHEMA_VERSION = 2
LEGACY_VERSION = 1

_DEFAULT_PROVENANCE = {
    "source": "local",
    "author": "",
    "created_at": "",
    "updated_at": "",
}

_DEFAULT_CONSTRAINTS = {
    "offline_compatible": True,
    "requires_gpu": False,
}

_DEFAULT_COMPAT = {
    "min_plugin_version": "",
    "max_plugin_version": "",
}


def _ensure_dict(value: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    """Return dict(value) when possible, else a deep copy of default."""
    if isinstance(value, dict):
        return dict(value)
    return deepcopy(default)


def upgrade_recipe_to_v2(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a recipe dictionary to schema v2 while preserving legacy keys."""
    upgraded = dict(recipe or {})

    # Keep legacy version marker for backward compatibility with older payloads.
    upgraded["version"] = int(upgraded.get("version", LEGACY_VERSION))
    upgraded["schema_version"] = int(upgraded.get("schema_version", SCHEMA_VERSION))

    upgraded["provenance"] = _ensure_dict(upgraded.get("provenance"), _DEFAULT_PROVENANCE)
    upgraded["constraints"] = _ensure_dict(upgraded.get("constraints"), _DEFAULT_CONSTRAINTS)
    upgraded["compat"] = _ensure_dict(upgraded.get("compat"), _DEFAULT_COMPAT)

    upgraded.setdefault("expected_runtime_class", "medium")
    upgraded.setdefault("expected_accuracy_class", "high")
    upgraded.setdefault("dataset_fingerprint", "")
    upgraded.setdefault("signature", "")

    upgraded["schema_version"] = SCHEMA_VERSION
    return upgraded
