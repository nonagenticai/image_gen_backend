"""mflux imports ``PIL._typing`` at runtime; pillow 13 made it fail on Python 3.14.

Run in a fresh interpreter so an earlier import of ``PIL._typing`` (or the shim
itself) in this process cannot make the test pass vacuously.
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=SRC,
        check=False,
    )


def test_pil_typing_imports_after_the_shim() -> None:
    result = _run(
        "from services.flux_service import _ensure_pillow_typing_importable\n"
        "_ensure_pillow_typing_importable()\n"
        "from PIL._typing import StrOrBytesPath  # what mflux image_util imports\n"
    )
    assert result.returncode == 0, result.stderr


def test_shim_leaves_an_existing_name_alone() -> None:
    result = _run(
        "import numbers\n"
        "sentinel = object()\n"
        "numbers._IntegralLike = sentinel\n"
        "from services.flux_service import _ensure_pillow_typing_importable\n"
        "_ensure_pillow_typing_importable()\n"
        "assert numbers._IntegralLike is sentinel\n"
    )
    assert result.returncode == 0, result.stderr
