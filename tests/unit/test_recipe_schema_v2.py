"""Unit tests for recipe schema v2 migration helpers."""

import importlib.util
import os

_MODULE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "dzetsaka",
        "domain",
        "value_objects",
        "recipe_schema_v2.py",
    ),
)
_SPEC = importlib.util.spec_from_file_location("_recipe_schema_v2_test", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

SCHEMA_VERSION = _MOD.SCHEMA_VERSION
upgrade_recipe_to_v2 = _MOD.upgrade_recipe_to_v2


def test_upgrade_adds_v2_fields_to_legacy_recipe() -> None:
    legacy = {
        "version": 1,
        "name": "Legacy RF",
        "classifier": {"code": "RF"},
    }
    upgraded = upgrade_recipe_to_v2(legacy)

    assert upgraded["schema_version"] == SCHEMA_VERSION
    assert upgraded["version"] == 1
    assert upgraded["provenance"]["source"] == "local"
    assert upgraded["constraints"]["offline_compatible"] is True
    assert upgraded["compat"]["min_plugin_version"] == ""
    assert upgraded["expected_runtime_class"] == "medium"
    assert upgraded["expected_accuracy_class"] == "high"
    assert upgraded["dataset_fingerprint"] == ""
    assert upgraded["signature"] == ""


def test_upgrade_preserves_existing_v2_metadata() -> None:
    recipe = {
        "version": 2,
        "schema_version": 2,
        "name": "Signed recipe",
        "provenance": {"source": "remote", "author": "alice", "created_at": "2026-01-01", "updated_at": ""},
        "constraints": {"offline_compatible": False, "requires_gpu": True},
        "compat": {"min_plugin_version": "5.1.0", "max_plugin_version": ""},
        "expected_runtime_class": "slow",
        "expected_accuracy_class": "very_high",
        "dataset_fingerprint": "abc123",
        "signature": "deadbeef",
    }
    upgraded = upgrade_recipe_to_v2(recipe)

    assert upgraded["schema_version"] == 2
    assert upgraded["provenance"]["source"] == "remote"
    assert upgraded["constraints"]["requires_gpu"] is True
    assert upgraded["compat"]["min_plugin_version"] == "5.1.0"
    assert upgraded["expected_runtime_class"] == "slow"
    assert upgraded["expected_accuracy_class"] == "very_high"
    assert upgraded["dataset_fingerprint"] == "abc123"
    assert upgraded["signature"] == "deadbeef"
