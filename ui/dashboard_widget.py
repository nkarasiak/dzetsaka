"""Preferred dashboard widget exports."""

from __future__ import annotations

from .classification_workflow_ui import (
    ClassificationSetupDialog,
    ClassificationDashboardDock,
    check_dependency_availability,
)

__all__ = [
    "ClassificationSetupDialog",
    "ClassificationDashboardDock",
    "check_dependency_availability",
]

