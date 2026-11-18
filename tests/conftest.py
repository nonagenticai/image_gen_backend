"""Test bootstrap — ensure `src/` is on sys.path before tests import the app.

image_gen_backend is configured with ``package-dir = {"" = "src"}`` (PEP 621
``[tool.setuptools]`` in pyproject.toml), so the runtime imports are bare
(``from main import app``, ``from models import …``). pytest's rootdir is the
repo root, so we need to add ``src/`` to ``sys.path`` for collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
