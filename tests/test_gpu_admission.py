"""GPU admission control on the render path (#69).

Before: one ``ThreadPoolExecutor(1)`` with an UNBOUNDED queue and no timeout.
20 concurrent renders at ~28.6 s each meant the last caller waited ~9.5 min,
and a caller that had already hung up still had its render run.

Now every render goes through the w2v_common ``AdmissionGate`` lane ``render``
(code defaults 1 active / 4 waiting / 30 s): the 6th+ concurrent request gets
an immediate 503 + Retry-After, a queued render whose client disconnected is
skipped, and ``/health`` stays fast because inference never runs on the loop.

Every test drives a REAL ``FluxService`` (real gate, real lane executor, real
close) with only the mflux pipeline stubbed — no mlx/mflux import, so this
runs on any machine.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import MutableMapping
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.admission import ClientGone, GateClosedError, GpuBusy, install_gpu_busy_handler

import main
import routers.image_routes as image_routes
from models import ImageGenRequest
from services.flux_service import RENDER_LANE, FluxService, ServiceUnavailableError

TENANT = str(uuid.UUID("3f2a1b4c-5d6e-4f70-8912-a3b4c5d6e7f8"))
BODY: dict[str, Any] = {"prompt": "a cat", "width": 512, "height": 512, "steps": 4}

MAX_ACTIVE, MAX_WAITING = 1, 4  # the issue's lane numbers = the code defaults


class _BlockingPipeline:
    """mflux stand-in: every render parks on the lane thread until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def generate_image(self, **_kwargs: object) -> SimpleNamespace:
        with self._lock:
            self.calls += 1
        self.release.wait(20)
        return SimpleNamespace(image=Image.new("RGB", (512, 512)))


def _pipe(service: FluxService) -> _BlockingPipeline:
    return cast("_BlockingPipeline", service._pipeline)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for suffix in ("MAX_ACTIVE", "MAX_WAITING", "MAX_WAIT_S", "SEED_S"):
        monkeypatch.delenv(f"GPU_LANE_RENDER_{suffix}", raising=False)
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "none")
    monkeypatch.setenv("IMAGE_GEN_MLX_CACHE_TRIM", "0")


@pytest.fixture
async def svc():
    service: Any = FluxService()
    service._pipeline = _BlockingPipeline()
    service.ready = True
    yield service
    await service.close()


def _app(service: FluxService) -> FastAPI:
    """The REAL image router + the REAL /health handler + the gate handlers,
    with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TENANT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(image_routes.router)
    application.add_api_route("/health", main.health, methods=["GET"])
    install_gpu_busy_handler(application)
    application.state.flux_service = service
    application.state.metering_redis = None
    application.state.metering_stream = "billing:costs"
    return application


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


async def test_twenty_parallel_renders_one_active_four_queued_rest_503(svc: FluxService) -> None:
    pipeline = _pipe(svc)
    lane = svc._gate.lane(RENDER_LANE)
    assert (lane.config.max_active, lane.config.max_waiting) == (MAX_ACTIVE, MAX_WAITING)
    app = _app(svc)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30) as client:
        tasks = [asyncio.create_task(client.post("/api/image/generate", json=BODY)) for _ in range(20)]
        try:
            # The 15 overflow requests must be answered WHILE the GPU is busy.
            await _until(lambda: sum(t.done() for t in tasks) == 20 - MAX_ACTIVE - MAX_WAITING)
            await _until(lambda: pipeline.calls == MAX_ACTIVE and lane.waiting == MAX_WAITING)
            assert lane.active == MAX_ACTIVE
            assert pipeline.calls == MAX_ACTIVE, "more renders reached the GPU than max_active"

            rejected = [t.result() for t in tasks if t.done()]
            assert len(rejected) == 15
            for r in rejected:
                assert r.status_code == 503, r.text
                assert r.json() == {"error": "gpu_busy", "lane": "render", "retry_after": r.json()["retry_after"]}
                retry_after = int(r.headers["Retry-After"])
                assert 1 <= retry_after <= 120
                assert retry_after == r.json()["retry_after"]

            # /health while the lane is saturated: answered, fast.
            t0 = time.perf_counter()
            health = await client.get("/health")
            elapsed = time.perf_counter() - t0
            assert health.status_code == 200, health.text
            assert elapsed < 0.2, f"/health took {elapsed * 1000:.0f} ms under full load"
        finally:
            pipeline.release.set()
        rest = await asyncio.gather(*tasks)

    ok = [r for r in rest if r.status_code == 200]
    assert len(ok) == MAX_ACTIVE + MAX_WAITING, [r.status_code for r in rest]
    assert pipeline.calls == MAX_ACTIVE + MAX_WAITING


async def test_queued_render_is_skipped_when_the_client_disconnected(svc: FluxService) -> None:
    """A render still waiting for the lane when its client hangs up never
    reaches the GPU (ClientGone → 499), and the route passes the request on."""
    pipeline = _pipe(svc)
    lane = svc._gate.lane(RENDER_LANE)
    body = ImageGenRequest(**BODY)

    def _request(gone: bool) -> SimpleNamespace:
        async def is_disconnected() -> bool:
            return gone

        return SimpleNamespace(
            state=SimpleNamespace(tenant_id=TENANT),
            headers={},
            is_disconnected=is_disconnected,
            app=SimpleNamespace(state=SimpleNamespace(flux_service=svc, metering_redis=None)),
        )

    first = asyncio.create_task(image_routes.generate(cast(Any, _request(False)), body))
    await _until(lambda: pipeline.calls == 1)
    queued = asyncio.create_task(image_routes.generate(cast(Any, _request(True)), body))
    await _until(lambda: lane.waiting == 1)

    pipeline.release.set()
    await first
    with pytest.raises(ClientGone):
        await queued
    assert pipeline.calls == 1, "the disconnected client's render still ran on the GPU"


async def test_queued_render_times_out_with_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_LANE_RENDER_MAX_WAIT_S", "0.2")
    service: Any = FluxService()
    service._pipeline = pipeline = _BlockingPipeline()
    service.ready = True
    try:
        first = asyncio.create_task(service.generate("a", 512, 512, 4, 0.0, 1))
        await _until(lambda: pipeline.calls == 1)
        with pytest.raises(GpuBusy) as info:
            await service.generate("b", 512, 512, 4, 0.0, 1)
        assert info.value.reason == "timeout"
        pipeline.release.set()
        await first
    finally:
        pipeline.release.set()
        await service.close()


async def test_close_turns_queued_renders_into_clean_503s(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renders queued at SIGTERM must answer 503 (ServiceUnavailableError), not 500."""
    monkeypatch.setenv("IMAGE_GEN_SHUTDOWN_TIMEOUT_S", "0.3")
    service: Any = FluxService()
    service._pipeline = pipeline = _BlockingPipeline()
    service.ready = True
    lane = service._gate.lane(RENDER_LANE)
    first = asyncio.create_task(service.generate("a", 512, 512, 4, 0.0, 1))
    await _until(lambda: pipeline.calls == 1)
    queued = asyncio.create_task(service.generate("b", 512, 512, 4, 0.0, 1))
    await _until(lambda: lane.waiting == 1)
    try:
        await service.close()
        pipeline.release.set()
        await first
        with pytest.raises(ServiceUnavailableError) as info:
            await queued
    finally:
        pipeline.release.set()
    # The 503 must come from the GATE refusing admission — not from a submit to
    # an executor that happened to be shut down already.
    assert isinstance(info.value.__cause__, GateClosedError), repr(info.value.__cause__)
    assert pipeline.calls == 1


async def test_slot_is_held_until_the_render_thread_finishes(svc: FluxService) -> None:
    """Cancelling the awaiting coroutine (client gone mid-render, handler
    timeout) does NOT stop the MLX thread, so it must NOT free the lane: the
    next request has to queue until the thread really ends. A ``slot()``
    wrapped around ``run_in_executor`` releases on cancellation and fails this.
    """
    pipeline = _pipe(svc)
    lane = svc._gate.lane(RENDER_LANE)
    first = asyncio.create_task(svc.generate("a", 512, 512, 4, 0.0, 1))
    await _until(lambda: pipeline.calls == 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert lane.active == 1, "the slot was released while the render thread is still running"

    second = asyncio.create_task(svc.generate("b", 512, 512, 4, 0.0, 1))
    await _until(lambda: lane.waiting == 1)
    await asyncio.sleep(0.05)
    assert pipeline.calls == 1 and lane.waiting == 1, "second render admitted while the first thread is on the GPU"

    pipeline.release.set()
    await second
    assert pipeline.calls == 2
    await _until(lambda: lane.active == 0)


def test_render_lane_rejects_max_active_above_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MLX pipeline is bound to ONE lane thread; a second worker would 500."""
    monkeypatch.setenv("GPU_LANE_RENDER_MAX_ACTIVE", "2")
    with pytest.raises(ValueError, match="max_active=1"):
        FluxService()


def test_env_overrides_waiting_and_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_LANE_RENDER_MAX_WAITING", "64")
    monkeypatch.setenv("GPU_LANE_RENDER_MAX_WAIT_S", "120")
    cfg = FluxService()._gate.lane(RENDER_LANE).config
    assert (cfg.max_active, cfg.max_waiting, cfg.max_wait_s) == (1, 64, 120.0)


def test_steps_default_is_three() -> None:
    # Owner decision 2026-10-22: 3 steps is visually ~identical to 4 on schnell
    # at ~24% less latency (mini1 benchmark: 14.67 s vs 19.32 s at 512x320).
    assert ImageGenRequest(prompt="x").steps == 3
    assert ImageGenRequest.model_fields["steps"].default == 3


@pytest.mark.parametrize("steps", [1, 3, 4, 8])
def test_steps_up_to_eight_accepted(steps: int) -> None:
    assert ImageGenRequest(prompt="x", steps=steps).steps == steps


@pytest.mark.parametrize("steps", [9, 50])
def test_steps_above_eight_rejected(steps: int) -> None:
    with pytest.raises(ValidationError):
        ImageGenRequest(prompt="x", steps=steps)


async def test_steps_above_eight_is_a_422_before_the_gpu(svc: FluxService) -> None:
    async with AsyncClient(transport=ASGITransport(app=_app(svc)), base_url="http://test") as client:
        r = await client.post("/api/image/generate", json={**BODY, "steps": 9})
    assert r.status_code == 422, r.text
    assert _pipe(svc).calls == 0


def test_main_app_installs_the_gpu_busy_handlers() -> None:
    """Without it a GpuBusy reaches the catch-all handler as a 500."""
    assert GpuBusy in main.app.exception_handlers
    assert ClientGone in main.app.exception_handlers


async def test_admission_metrics_are_on_the_service_metrics_endpoint(svc: FluxService) -> None:
    async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as client:
        text = (await client.get("/metrics")).text
    for name in ("gpu_admission_active", "gpu_admission_waiting", "gpu_admission_rejected_total"):
        assert f'{name}{{lane="render"' in text, f"{name} missing from /metrics"


async def test_disconnect_while_queued_is_skipped_through_the_real_app_stack(
    svc: FluxService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``main.app`` as deployed — CORS, AuthMiddleware (a
    BaseHTTPMiddleware), Instrumentator, LICENSE_GATE router — driven at the raw
    ASGI level so the client can really hang up. Behind a BaseHTTPMiddleware
    ``Request.is_disconnected()`` never sees the disconnect; only the outermost
    DisconnectProbeMiddleware makes the gate's skip work. Remove that line from
    main.py and this test fails (the abandoned render runs).
    """

    async def _introspect(_token: str) -> dict[str, object]:
        return {"active": True, "tenant_id": TENANT, "subject_id": "user-under-test"}

    monkeypatch.setattr("w2v_common.auth.middleware._introspect_license", _introspect)
    for key, value in (("flux_service", svc), ("metering_redis", None), ("metering_stream", "billing:costs")):
        monkeypatch.setattr(main.app.state, key, value, raising=False)

    pipeline = _pipe(svc)
    lane = svc._gate.lane(RENDER_LANE)
    payload = json.dumps(BODY).encode()

    async def _call(hang_up: asyncio.Event) -> list[dict[str, Any]]:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/image/generate",
            "raw_path": b"/api/image/generate",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"x-tenant-id", b"Bearer test-licence-token"),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("test", 80),
            "state": {},
        }
        body_sent = False

        async def receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": payload, "more_body": False}
            if not hang_up.is_set():
                await hang_up.wait()
            return {"type": "http.disconnect"}

        sent: list[dict[str, Any]] = []

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(dict(message))

        await main.app(scope, receive, send)
        return sent

    stays, leaves = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(_call(stays))
    await _until(lambda: pipeline.calls == 1)
    queued = asyncio.create_task(_call(leaves))
    await _until(lambda: lane.waiting == 1)

    leaves.set()  # the queued client hangs up
    await asyncio.sleep(0.05)
    pipeline.release.set()
    first_sent = await asyncio.wait_for(first, 10)
    queued_sent = await asyncio.wait_for(queued, 10)
    stays.set()

    assert first_sent[0]["status"] == 200, first_sent
    assert queued_sent[0]["status"] == 499, queued_sent
    assert pipeline.calls == 1, "the hung-up client's render still ran on the GPU"
