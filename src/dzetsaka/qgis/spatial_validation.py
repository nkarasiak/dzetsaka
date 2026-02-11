"""Spatial validation helpers for dzetsaka QGIS UI.

This module keeps the dashboard/processing code dry when working with
polygon-based cross-validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from dzetsaka.infrastructure.geo.vector_split import count_polygons_per_class

if TYPE_CHECKING:
    from qgis.PyQt.QtWidgets import QWidget


def find_classes_with_insufficient_polygons(
    class_counts: Dict[Any, int],
    min_polygons: int = 2,
) -> Dict[Any, int]:
    """Return classes that appear in fewer than ``min_polygons`` samples."""
    return {
        class_label: polygons
        for class_label, polygons in class_counts.items()
        if polygons < min_polygons
    }


def confirm_polygon_group_split(
    parent: Optional["QWidget"],
    *,
    vector_path: str,
    class_field: str,
    min_polygons: int = 2,
    log = None,
) -> Tuple[bool, bool]:
    """Prompt the user before running polygon-based cross-validation.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget for the confirmation dialog.
    vector_path : str
        Path to the training vector layer.
    class_field : str
        Field name holding class labels.
    min_polygons : int
        Minimum number of polygons required per class.
    log : Any, optional
        Optional logger for reporting what happened.

    Returns
    -------
    Tuple[bool, bool]
        ``(continue_with_spatial_cv, fallback_to_random_split)``.
    """

    class_counts = count_polygons_per_class(vector_path, class_field)

    if not class_counts:
        if log:
            log.warning("Unable to inspect polygons for spatial CV; falling back to random split.")
        return True, True

    insufficient = find_classes_with_insufficient_polygons(class_counts, min_polygons)

    if not insufficient:
        return True, False

    from qgis.PyQt.QtWidgets import QMessageBox

    classes_text = ", ".join(f"{cls} ({cnt})" for cls, cnt in sorted(insufficient.items()))
    message = (
        f"Polygon-based cross-validation requires at least {min_polygons} polygons per class.\n"
        f"The following classes do not meet that threshold: {classes_text}.\n\n"
        "Continue with a random split instead?"
    )
    reply = QMessageBox.question(
        parent,
        "Insufficient polygons for spatial CV",
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )

    if reply == QMessageBox.StandardButton.Yes:
        if log:
            log.warning(
                "Spatial CV disabled because some classes lack enough polygons; falling back to random split."
            )
        return True, True

    if log:
        log.info("Spatial CV canceled by user because polygon coverage is insufficient.")
    return False, False
