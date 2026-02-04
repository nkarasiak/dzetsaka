"""Root conftest — ensure the repository root is on sys.path.

pytest's default ``prepend`` import mode inserts the *test* directory into
``sys.path``, which can hide top-level packages like ``scripts`` and ``ui``.
This conftest explicitly adds the repo root so that all tests can resolve
those packages regardless of collection order or import mode.
"""

import os
import sys

# Insert repo root at position 0 so it takes priority
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
