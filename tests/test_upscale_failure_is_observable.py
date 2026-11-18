"""A failed upscale is OBSERVABLE — to the operator and to the caller.

Issue #59, follow-up to #58. #58 established that a failed post-render upscale
must not discard the committed render: ``FluxService.generate`` logs and returns
the image at its real size. What it left behind is that this is the *only*
trace. The route answers 200 with an image smaller than the 1536 contract, the
cost event bills the smaller size, and nothing in the response or in /metrics
says a degradation happened — so a host that fails every upscale (MemoryError
under pressure is the documented mode on this 16 GB box) looks perfectly healthy
from outside while silently shipping sub-contract assets.

Two signals close that, and these tests pin both:

  1. ``image_gen_upscale_failures_total`` — a module-level Counter on
     ``prometheus_client.REGISTRY``, which is the registry ``src/main.py``
     exposes at /metrics via ``Instrumentator().instrument(app).expose(app)``;
  2. ``ImageGenResponse.upscale_failed`` — the caller's half, so a client can
     tell a degraded 200 from a healthy one without knowing the target.

The status code stays 200 and the fallback image is unchanged: this adds
visibility, it does not change behaviour. ``test_healthy_upscale_is_not_marked``
is the control — it fails if the marker or the counter fires on the happy path,
which is what a naive "always set it" implementation would do.

Counter deltas are measured, never absolutes: pytest may run this module after
anything else, and the Counter is process-global.

No mlx/mflux import anywhere — the pipeline is stubbed, so this runs on any
machine. Stubs mirror tests/test_cost_event_survives_postprocessing.py.
"""

from __future__ import annotations

import io
import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import routers.image_routes as image_routes
import services.flux_service as flux_service
from models import ImageGenRequest

TENANT = uuid.UUID("3f2a1b4c-5d6e-4f70-8912-a3b4c5d6e7f8")
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(64))


class _StubImage:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def save(self, buf: io.BytesIO, format: str) -> None:
        assert format == "PNG"
        buf.write(_PNG_BYTES)

    def resize(self, size: tuple[int, int], resample: object) -> _StubImage:
        return _StubImage(size)


class _InstantPipeline:
    def generate_image(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(image=_StubImage((512, 512)))


def _real_service_with_stub_pipeline() -> flux_service.FluxService:
    """A REAL ``FluxService`` with the mflux pipeline stubbed — no mlx import.

    ``Any`` on the local deliberately: ``_pipeline`` is declared ``Flux1 | None``
    and the whole point is to install a stand-in without importing mflux."""
    svc: Any = flux_service.FluxService()
    svc._pipeline = _InstantPipeline()
    svc.ready = True
    return svc


def _redis() -> AsyncMock:
    redis_mock = AsyncMock()
    redis_mock.xadd = AsyncMock(return_value=b"1-0")
    return redis_mock


def _route_request(svc: object) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(tenant_id=str(TENANT)),
        # The route passes the request to the admission gate (#69), which asks
        # whether the client is still there before starting the render.
        is_disconnected=_still_connected,
        headers={"X-App-Id": "img"},
        app=SimpleNamespace(
            state=SimpleNamespace(flux_service=svc, metering_redis=_redis(), metering_stream="billing:costs")
        ),
    )


def _body() -> ImageGenRequest:
    return ImageGenRequest(prompt="a cat", width=1024, height=1024, steps=4, guidance=0.0)


def _failure_count() -> float:
    """Current value of the counter.

    ``_value.get()`` is the documented-enough prometheus_client internal the
    fleet uses for this; ``REGISTRY.get_sample_value`` is the public path but
    resolves by NAME against the global registry, which silently returns None
    if the collector is ever moved to a custom registry — a None that would read
    as "no failures" and make this test vacuous. Reading the object we imported
    cannot drift from the object under test.
    """
    return cast(Any, flux_service.UPSCALE_FAILURES)._value.get()


def _upscale_raises(_img: object, _target: int) -> object:
    raise MemoryError("injected: LANCZOS upscale failed after the render committed")


async def test_failed_upscale_increments_the_counter_and_marks_the_response(monkeypatch) -> None:
    """The defect (#59). On origin/main the counter does not exist and the
    response carries no marker: a degraded 200 is indistinguishable from a
    healthy one."""
    monkeypatch.setattr(flux_service, "_upscale", _upscale_raises)
    monkeypatch.delenv("IMAGE_GEN_UPSCALE", raising=False)
    before = _failure_count()

    svc = _real_service_with_stub_pipeline()
    req = _route_request(svc)
    try:
        resp = await image_routes.generate(cast(Any, req), _body())
    finally:
        await svc.close()

    # (a) the operator's signal
    assert _failure_count() - before == 1.0, "image_gen_upscale_failures_total did not increment exactly once"
    # (b) the caller's signal
    assert resp.upscale_failed is True
    # (c) still a 200 — generate() returns a model, not an HTTPException, and
    #     the fallback image is unchanged at its real un-upscaled size.
    assert (resp.width, resp.height) == (512, 512)
    assert resp.image_base64


async def test_healthy_upscale_is_not_marked(monkeypatch) -> None:
    """Control. Fails if the marker or the counter fires on the happy path —
    which is what makes the assertions above non-vacuous."""
    monkeypatch.delenv("IMAGE_GEN_UPSCALE", raising=False)
    before = _failure_count()

    svc = _real_service_with_stub_pipeline()
    req = _route_request(svc)
    try:
        resp = await image_routes.generate(cast(Any, req), _body())
    finally:
        await svc.close()

    assert _failure_count() - before == 0.0
    assert resp.upscale_failed is False
    assert (resp.width, resp.height) == (1536, 1536)


async def _still_connected() -> bool:
    return False
