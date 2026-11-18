"""`ImageGenRequest.seed` is bounded to int32 — the service's own contract.

Every sibling field carries bounds (``width``/``height``: 256–2048 multiple of
64; ``steps``: 1–50; ``guidance``: 0.0–20.0); ``seed`` was unbounded. The
service's generated default is ``random.randint(0, 2**31 - 1)``
(``flux_service.generate``), so int32 IS the contract.

Verified against ``routers/image_routes.py``: the route maps only
``ServiceUnavailableError`` to a 503. An out-of-range seed reaches
``pipeline.generate_image(seed=seed_used, ...)`` and raises inside the MLX
executor as something else, which ``flux_service`` re-raises unchanged and the
``@app.exception_handler(Exception)`` catch-all in ``main.py`` turns into an
opaque 500. Bounding the field turns that into a 422 at the edge.

No mlx/mflux import — pure schema validation, runs on any machine.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from models import ImageGenRequest

_INT32_MAX = 2**31 - 1


class TestSeedBound:
    @pytest.mark.parametrize("seed", [2**31, 2**63, -1, -(2**31)])
    def test_rejects_out_of_int32_range(self, seed: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ImageGenRequest(prompt="a cat", seed=seed)
        assert "seed" in str(exc_info.value)

    @pytest.mark.parametrize("seed", [None, 0, 1, _INT32_MAX])
    def test_accepts_in_range_and_none(self, seed: int | None) -> None:
        assert ImageGenRequest(prompt="a cat", seed=seed).seed == seed

    def test_seed_defaults_to_none(self) -> None:
        assert ImageGenRequest(prompt="a cat").seed is None

    def test_generated_default_stays_inside_the_field_bound(self) -> None:
        """The service's own generated seed must satisfy the new field bound.

        Guards against the bound and ``flux_service``'s ``random.randint``
        range drifting apart: a generated seed the schema would reject is a
        contradiction the API could never round-trip.
        """
        for _ in range(200):
            generated = random.randint(0, _INT32_MAX)  # noqa: S311 - mirrors flux_service
            assert ImageGenRequest(prompt="a cat", seed=generated).seed == generated

    def test_flux_service_default_range_matches_the_bound(self) -> None:
        """Pin the literal range in flux_service to this schema's bound.

        Source-level assertion on purpose: importing ``flux_service`` would
        drag in mflux/mlx, which are Apple-silicon-only.
        """
        src = Path(__file__).resolve().parent.parent / "src" / "services" / "flux_service.py"
        assert "random.randint(0, 2**31 - 1)" in src.read_text(encoding="utf-8")
