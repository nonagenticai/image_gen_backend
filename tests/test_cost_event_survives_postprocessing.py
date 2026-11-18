"""A committed render is billed even when the post-processing after it fails.

``svc.generate`` returns only after mflux has run: the GPU time is spent and
the render is complete. Two pure-CPU passes follow it, and either can raise on
a 16 GB host with a documented swap-out history:

  * ``_upscale`` (LANCZOS to 1536) inside ``FluxService.generate``;
  * ``_encode_png_b64`` (~7 MB zlib + ~9 MB base64) in the route.

On ``origin/main`` before this fix the cost event was emitted AFTER the encode,
so an exception in either window reached ``global_exception_handler`` as a 500
with NO cost event — a paid render, unbilled. That is the sibling of the bug
``flux_service.py`` documents at the upscale call site (*"returning an opaque
500 and discarding a multi-second render that had already completed"*), one
step later. These tests pin the fix for both windows:

  1. the route emits the cost event as soon as the render is committed, before
     the encode — a failed encode still 500s the caller, but exactly ONE event
     lands, billed on the dims the service produced;
  2. a failed upscale does not discard the render: ``FluxService.generate``
     returns the un-upscaled image at its real size, the response reports that
     size, and the event bills that size.

The count assertions are ``assert_awaited_once`` on ``xadd``: zero (the defect)
and two (a naive ``finally`` on top of the existing emit) both fail them.

No mlx/mflux import anywhere — the pipeline is stubbed, so this runs on any
machine.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import w2v_common.metering.cost_bus as cost_bus
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

import routers.image_routes as image_routes
import services.flux_service as flux_service
from main import global_exception_handler
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


class _StubFluxService:
    """Render already done: ``generate`` hands back a finished 1536x1536 image."""

    ready = True
    model_repo = "flux-schnell"

    async def generate(self, **_kwargs: object) -> tuple[_StubImage, int]:
        return _StubImage((1536, 1536)), 4242


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


async def _drain_background() -> None:
    # ``schedule_cost_event`` parks its in-flight task in w2v_common's module-
    # level set; drain it so the XADD lands before we count.
    if cost_bus._background_tasks:
        await asyncio.gather(*list(cost_bus._background_tasks))


def _raise_memory_error(*_args: object, **_kwargs: object) -> str:
    raise MemoryError("injected: PNG encode failed after the render committed")


def _one_event(redis_mock: AsyncMock) -> dict[str, str]:
    redis_mock.xadd.assert_awaited_once()
    return redis_mock.xadd.call_args.args[1]


# ---------------------------------------------------------------------------
# Window 1: the encode
# ---------------------------------------------------------------------------


async def test_failed_encode_still_bills_the_committed_render_exactly_once(monkeypatch) -> None:
    """The defect. On origin/main this test fails with ``xadd`` awaited 0 times."""
    monkeypatch.setattr(image_routes, "_encode_png_b64", _raise_memory_error)
    req = _route_request(_StubFluxService())

    with pytest.raises(MemoryError):
        await image_routes.generate(cast(Any, req), _body())
    await _drain_background()

    fields = _one_event(req.app.state.metering_redis)
    assert fields["tenant_id"] == str(TENANT)
    assert fields["app_id"] == "img"
    assert fields["provider"] == "mflux"
    assert fields["model"] == "flux-schnell"
    # Billed on what the service PRODUCED (1536x1536 -> 2 MP), not the request.
    assert fields["input_tokens"] == "2"
    usage_extra = json.loads(fields["usage_extra"])
    assert usage_extra == {"unit": "megapixels", "width": 1536, "height": 1536, "steps": 4}


async def test_failed_encode_is_a_500_to_the_caller_with_one_event(monkeypatch) -> None:
    """Through the REAL catch-all handler: the caller still gets the 500
    envelope (there is no payload to return), and the event has landed."""
    monkeypatch.setattr(image_routes, "_encode_png_b64", _raise_memory_error)

    app = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = str(TENANT)
            return await call_next(request)

    app.add_middleware(_StampIdentity)
    app.add_exception_handler(Exception, global_exception_handler)
    app.include_router(image_routes.router)
    app.state.flux_service = _StubFluxService()
    app.state.metering_redis = _redis()
    app.state.metering_stream = "billing:costs"

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/image/generate", json=_body().model_dump())
    await _drain_background()

    assert response.status_code == 500, response.text
    assert response.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    fields = _one_event(app.state.metering_redis)
    assert fields["tenant_id"] == str(TENANT)


async def test_happy_path_emits_exactly_one_event() -> None:
    """No double emit: moving the emit earlier must not leave the old one behind."""
    req = _route_request(_StubFluxService())

    resp = await image_routes.generate(cast(Any, req), _body())
    await _drain_background()

    assert (resp.width, resp.height) == (1536, 1536)
    fields = _one_event(req.app.state.metering_redis)
    assert fields["input_tokens"] == "2"


# ---------------------------------------------------------------------------
# Window 2: the upscale inside FluxService.generate
# ---------------------------------------------------------------------------


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


def _upscale_raises(_img: object, _target: int) -> object:
    raise MemoryError("injected: LANCZOS upscale failed after the render committed")


async def test_failed_upscale_returns_the_committed_render_at_its_real_size(monkeypatch) -> None:
    """The service half: a failed upscale is logged, not raised. The finished
    512x512 render comes back at its real size instead of being discarded."""
    monkeypatch.setattr(flux_service, "_upscale", _upscale_raises)
    monkeypatch.delenv("IMAGE_GEN_UPSCALE", raising=False)
    svc = _real_service_with_stub_pipeline()
    try:
        image, seed = await svc.generate(prompt="a cat", width=512, height=512, steps=4, guidance=0.0, seed=7)
    finally:
        await svc.close()

    assert seed == 7
    assert image.size == (512, 512)


async def test_failed_upscale_bills_the_real_size_exactly_once(monkeypatch) -> None:
    """End to end through the route: 200, honest dims, ONE event billed on the
    un-upscaled size the GPU actually produced (512x512 -> floor 1 MP)."""
    monkeypatch.setattr(flux_service, "_upscale", _upscale_raises)
    monkeypatch.delenv("IMAGE_GEN_UPSCALE", raising=False)
    svc = _real_service_with_stub_pipeline()
    req = _route_request(svc)
    try:
        resp = await image_routes.generate(cast(Any, req), _body())
    finally:
        await svc.close()
    await _drain_background()

    assert (resp.width, resp.height) == (512, 512)
    fields = _one_event(req.app.state.metering_redis)
    assert fields["input_tokens"] == "1"
    assert json.loads(fields["usage_extra"])["width"] == 512


async def test_upscale_still_runs_when_it_does_not_fail(monkeypatch) -> None:
    """Guard against 'fixing' window 2 by skipping the upscale: the 1536
    contract still holds on the normal path."""
    monkeypatch.delenv("IMAGE_GEN_UPSCALE", raising=False)
    svc = _real_service_with_stub_pipeline()
    req = _route_request(svc)
    try:
        resp = await image_routes.generate(cast(Any, req), _body())
    finally:
        await svc.close()
    await _drain_background()

    assert (resp.width, resp.height) == (1536, 1536)
    _one_event(req.app.state.metering_redis)


async def _still_connected() -> bool:
    return False
