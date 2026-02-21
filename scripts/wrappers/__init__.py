"""Label encoding wrappers for gradient boosting algorithms.

This module provides wrapper classes that handle sparse label encoding/decoding
for XGBoost and CatBoost. These wrappers ensure that models work
correctly with non-continuous class labels (e.g., 0, 1, 3, 5 instead of 0, 1, 2, 3).
"""

from .label_encoders import (
    SKLEARN_AVAILABLE,
    CBClassifierWrapper,
    XGBLabelWrapper,
)

__all__ = [
    "SKLEARN_AVAILABLE",
    "CBClassifierWrapper",
    "XGBLabelWrapper",
]
