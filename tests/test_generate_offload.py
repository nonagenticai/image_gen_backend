"""Off-loading rules for the generate path — WHICH THREAD does the CPU work,
and what happens when the executor is shut down mid-request.

Part 1 (encode): the PNG + base64 pass must not run on the asyncio event loop.
Part 2 (shutdown window): ``close()`` landing during an in-flight generation
must never surface as an opaque 500 — see the section header further down.

``svc.generate`` is correctly off-loaded to the pinned MLX thread, but the
route then PNG-encoded and base64'd the result inline. At the service's
1536px upscale target that is a zlib pass over ~7 MB plus a ~9 MB base64
pass — hundreds of milliseconds with the event loop blocked, stalling
``/health`` and every concurrent generate.

The assertion is about WHICH THREAD does the work, not about the status code:
a 200 proves nothing here. ``_ThreadRecordingImage.save`` records the thread it
runs on, and the test asserts that thread is not the event-loop thread.

No mlx/mflux import anywhere — the service is stubbed, so this runs on any
machine.
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException

import routers.image_routes as image_routes
from models import ImageGenRequest

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(64))


class _ThreadRecordingImage:
    """Minimal PIL-Image stand-in that records the thread ``save`` ran on."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.save_thread: threading.Thread | None = None

    def save(self, buf: io.BytesIO, format: str) -> None:
        # `format` mirrors PIL's own kwarg name — the route passes format="PNG".
        assert format == "PNG"
        self.save_thread = threading.current_thread()
        buf.write(_PNG_BYTES)


class _StubFluxService:
    def __init__(self, image: _ThreadRecordingImage) -> None:
        self.ready = True
        self.model_repo = "schnell"
        self._image = image

    async def generate(self, **_kwargs: object) -> tuple[_ThreadRecordingImage, int]:
        return self._image, 4242


def _fake_request(svc: _StubFluxService) -> SimpleNamespace:
    # metering_redis=None → _emit_cost_event no-ops, keeping this test on the
    # route's encode path only. The tenant is mandatory since w2v-common 0.67.0:
    # ``generate`` refuses an unattributable request with 401 BEFORE the GPU
    # work, so these encode/shutdown tests must present a real one.
    return SimpleNamespace(
        state=SimpleNamespace(tenant_id="11111111-2222-4333-8444-555555555555"),
        # The route passes the request to the admission gate (#69), which asks
        # whether the client is still there before starting the render.
        is_disconnected=_still_connected,
        headers={},
        app=SimpleNamespace(state=SimpleNamespace(flux_service=svc, metering_redis=None)),
    )


def _body() -> ImageGenRequest:
    return ImageGenRequest(prompt="a cat", width=1024, height=1024, steps=4, guidance=0.0)


async def test_png_encode_does_not_run_on_the_event_loop() -> None:
    image = _ThreadRecordingImage((1536, 1536))
    req = _fake_request(_StubFluxService(image))

    loop_thread = threading.current_thread()
    await image_routes.generate(cast(Any, req), _body())

    assert image.save_thread is not None, "Image.save was never called"
    assert image.save_thread is not loop_thread, (
        "PNG encode ran on the asyncio event-loop thread "
        f"({loop_thread.name}) — a 1536x1536 encode blocks /health and every "
        "concurrent generate for hundreds of ms. Off-load it to a thread."
    )


async def test_encoded_payload_is_the_produced_png() -> None:
    """Off-loading must not change the payload."""
    image = _ThreadRecordingImage((1536, 1536))
    req = _fake_request(_StubFluxService(image))

    resp = await image_routes.generate(cast(Any, req), _body())

    assert base64.b64decode(resp.image_base64) == _PNG_BYTES
    assert resp.seed == 4242
    assert resp.model_repo == "schnell"


async def test_response_dims_still_come_from_the_produced_image() -> None:
    """Regression guard on the existing billing hardening: dims are read from
    ``pil_image.size`` (what the GPU produced), never the request's
    width/height. Off-loading the encode must not disturb that."""
    image = _ThreadRecordingImage((896, 512))
    req = _fake_request(_StubFluxService(image))

    resp = await image_routes.generate(cast(Any, req), _body())

    assert (resp.width, resp.height) == (896, 512)


# ────────────────────────────────────────────────────────────────────────────
# close() during an in-flight generation
#
# ``generate()`` used to submit to the pinned MLX executor TWICE — the mflux
# call, then the LANCZOS upscale — with an ``await`` in between. ``close()``
# lands in exactly that window on any launchd SIGTERM with traffic in flight,
# and the second submit raised ``RuntimeError: cannot schedule new futures
# after shutdown`` *inside* ``generate()``. The route only checks ``svc.ready``
# BEFORE calling, so nothing caught it: the client got an opaque 500 and a
# completed multi-second GPU render was thrown away.
#
# The fix has two halves and these tests pin both:
#   1. the upscale is pure-PIL CPU work, so it moved OFF the MLX executor onto
#      a general worker thread — a render that already finished is returned;
#   2. the remaining window (close landing before the mflux submit) raises a
#      typed ``ServiceUnavailableError`` that the route renders as a clean 503.
# ────────────────────────────────────────────────────────────────────────────


# The upscaled image never leaves the route, so the thread `resize` ran on is
# recorded here rather than on the returned object.
_resize_threads: list[threading.Thread] = []


class _StubImage:
    """PIL-Image stand-in supporting the save + resize the path exercises."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def save(self, buf: io.BytesIO, format: str) -> None:
        assert format == "PNG"
        buf.write(_PNG_BYTES)

    def resize(self, size: tuple[int, int], resample: object) -> _StubImage:
        _resize_threads.append(threading.current_thread())
        return _StubImage(size)


class _BlockingPipeline:
    """mflux pipeline stand-in that parks on the MLX thread until released."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self._entered = entered
        self._release = release
        self.calls = 0

    def generate_image(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        self._entered.set()
        self._release.wait(10)
        return SimpleNamespace(image=_StubImage((512, 512)))


def _real_service_with_stub_pipeline(
    entered: threading.Event, release: threading.Event
) -> tuple[Any, _BlockingPipeline]:
    """A REAL ``FluxService`` (real executor, real close) + a stub pipeline.

    No mflux/mlx import anywhere — ``_build_pipeline`` is bypassed by assigning
    ``_pipeline`` directly, so this runs on any machine.
    """
    from services.flux_service import FluxService

    # `Any` deliberately: `_pipeline` is declared as `Flux1 | None`, and the whole
    # point is to install a stand-in without importing mflux.
    svc: Any = FluxService()
    pipeline = _BlockingPipeline(entered, release)
    svc._pipeline = pipeline
    svc.ready = True
    return svc, pipeline


async def test_close_midflight_returns_the_completed_render() -> None:
    """close() landing AFTER the GPU work finished must not discard it.

    Before the fix this raised ``RuntimeError: cannot schedule new futures
    after shutdown`` out of ``generate()`` on the upscale submit → opaque 500.
    """
    entered, release = threading.Event(), threading.Event()
    svc, _pipeline = _real_service_with_stub_pipeline(entered, release)
    req = _fake_request(svc)

    task = asyncio.create_task(image_routes.generate(cast(Any, req), _body()))
    assert await asyncio.to_thread(entered.wait, 10), "stub pipeline never started"

    # close() flips _closed and begins shutting the MLX executor down while the
    # generation is parked on it — the exact SIGTERM-with-traffic window.
    close_task = asyncio.create_task(svc.close())
    await asyncio.sleep(0.05)
    release.set()

    resp = await task
    await close_task

    assert base64.b64decode(resp.image_base64) == _PNG_BYTES
    assert resp.seed is not None


async def test_upscale_runs_off_the_mlx_executor_and_off_the_loop() -> None:
    """The LANCZOS upscale is pure-PIL CPU work: it must not occupy the single
    MLX-pinned thread (which would serialise it against the next generation),
    and it must not run inline on the event loop."""
    entered, release = threading.Event(), threading.Event()
    release.set()
    svc, _pipeline = _real_service_with_stub_pipeline(entered, release)
    req = _fake_request(svc)

    loop_thread = threading.current_thread()
    _resize_threads.clear()
    resp = await image_routes.generate(cast(Any, req), _body())
    await svc.close()

    # The route reports the produced dims, so a 1536 long edge proves the
    # upscale actually ran (stub renders 512x512).
    assert (resp.width, resp.height) == (1536, 1536)
    assert len(_resize_threads) == 1, "upscale never ran"
    thread = _resize_threads[0]
    assert thread is not loop_thread, "LANCZOS upscale ran on the event-loop thread"
    assert not thread.name.startswith("gpu-lane-render"), (
        f"LANCZOS upscale ran on the pinned MLX thread ({thread.name}) — pure-PIL "
        "CPU work must not occupy the single GPU-stream thread."
    )


async def test_generate_after_shutdown_is_a_clean_503_not_a_500() -> None:
    """The residual window: close() completes between the route's ``svc.ready``
    check and the mflux submit. The stale-ready caller must get the same clean
    503 the pre-check promises, not ``RuntimeError`` → 500."""
    entered, release = threading.Event(), threading.Event()
    release.set()
    svc, pipeline = _real_service_with_stub_pipeline(entered, release)

    await svc.close()
    # Simulate the route having read ``ready`` just before close() flipped it.
    svc.ready = True
    req = _fake_request(svc)

    with pytest.raises(HTTPException) as excinfo:
        await image_routes.generate(cast(Any, req), _body())

    assert excinfo.value.status_code == 503
    assert pipeline.calls == 0, "no GPU work should have been attempted"


async def _still_connected() -> bool:
    return False
