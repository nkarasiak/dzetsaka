"""Issue reporting helpers for dzetsaka QGIS presentation layer."""

from __future__ import annotations

from dzetsaka.logging import show_issue_popup


def show_standard_issue_popup(
    plugin,
    *,
    error_title: str,
    error_type: str,
    error_message: str,
    context: str,
) -> None:
    """Show standardized compact issue popup."""
    show_issue_popup(
        error_title=error_title,
        error_type=error_type,
        error_message=error_message,
        context=context,
        parent=plugin,
    )

