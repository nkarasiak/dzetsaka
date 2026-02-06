"""Preferred dashboard widget exports."""

from __future__ import annotations

from .guided_workflow_widget import (
    ClassificationDashboardDock,
    GuidedClassificationDialog,
    check_dependency_availability,
)

__all__ = ["ClassificationDashboardDock", "GuidedClassificationDialog", "check_dependency_availability"]
