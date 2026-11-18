"""Lifespan teardown tests — the MLX executor must actually be shut down.

``FluxService`` pins the loaded FLUX model to a single-thread
``ThreadPoolExecutor``. Nothing ever shut that executor down: the app's
lifespan closed the metering Redis client and returned, leaking the pool
thread and the multi-GB model it holds for the lifetime of the process
(and, in-test, of every ``TestClient`` that starts the app).

These tests pin three properties:

1. ``FluxService.close()`` exists and really shuts the executor down.
2. It is idempotent — a second close is a no-op, not a crash.
3. Lifespan shutdown is DEFENSIVE — a failure closing one resource must not
   prevent the other from being closed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient


class _FakeFluxService:
    """Stand-in for FluxService — never imports mlx/mflux, so this runs anywhere."""

    def __init__(self, *, close_raises: bool = False) -> None:
        self.model_repo = "schnell"
        self.ready = True
        self.close_calls = 0
        self._close_raises = close_raises

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_raises:
            raise RuntimeError("boom: executor shutdown failed")


def _install_fake_flux(monkeypatch: pytest.MonkeyPatch, fake: _FakeFluxService) -> None:
    import main

    monkeypatch.setattr(main, "get_flux_service", lambda: fake)


async def test_flux_service_close_shuts_down_executor() -> None:
    from services.flux_service import FluxService

    svc = FluxService()
    executor: concurrent.futures.ThreadPoolExecutor = svc._executor

    await svc.close()

    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)
    assert svc.ready is False


async def test_flux_service_close_is_idempotent() -> None:
    from services.flux_service import FluxService

    svc = FluxService()
    await svc.close()
    await svc.close()  # must not raise


def test_lifespan_shutdown_closes_flux_service(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    fake = _FakeFluxService()
    _install_fake_flux(monkeypatch, fake)

    with TestClient(main.app):
        pass

    assert fake.close_calls == 1, "lifespan shutdown never closed the FLUX service (executor + model leaked)"


def test_redis_close_failure_still_closes_flux(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    fake = _FakeFluxService()
    _install_fake_flux(monkeypatch, fake)

    async def _boom(_client: object) -> None:
        raise RuntimeError("redis close failed")

    monkeypatch.setattr(main, "close_metering_redis", _boom)

    with TestClient(main.app):
        pass

    assert fake.close_calls == 1, "a failing Redis close prevented the FLUX executor from being shut down"


def test_flux_close_failure_still_closes_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    fake = _FakeFluxService(close_raises=True)
    _install_fake_flux(monkeypatch, fake)

    seen: list[Any] = []

    async def _record(client: object) -> None:
        seen.append(client)

    monkeypatch.setattr(main, "close_metering_redis", _record)

    with TestClient(main.app):
        pass

    assert fake.close_calls == 1
    assert len(seen) == 1, "a failing FLUX close prevented the metering Redis client from being closed"


def test_no_flux_executor_threads_leak_after_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: after a full app lifespan with a REAL FluxService (real
    executor, stubbed model load — no mflux/mlx, no network), no ``gpu-lane-render``
    worker thread is left alive."""
    import main
    from services.flux_service import FluxService

    svc = FluxService()

    async def _no_model_load() -> None:
        # Keep the real executor; skip the mflux pipeline build (gated HF repo).
        # ThreadPoolExecutor spawns its worker lazily, so submit a no-op to
        # materialise the `gpu-lane-render` thread — otherwise this test is vacuous.
        svc._executor.submit(lambda: None).result()
        svc.ready = True

    monkeypatch.setattr(svc, "initialize", _no_model_load)
    monkeypatch.setattr(main, "get_flux_service", lambda: svc)

    with TestClient(main.app):
        pass

    alive = [t.name for t in threading.enumerate() if t.name.startswith("gpu-lane-render") and t.is_alive()]
    assert alive == [], f"MLX executor threads still alive after shutdown: {alive}"


# ────────────────────────────────────────────────────────────────────────────
# Shutdown must be BOUNDED
#
# ``close()`` called ``shutdown(wait=True)`` with no timeout and no
# ``cancel_futures``. ``main.lifespan`` awaits it inline, so uvicorn's graceful
# shutdown blocked for the whole remaining FLUX render (30s to several minutes
# on M-series) PLUS every queued job in the backlog. launchd's SIGTERM→SIGKILL
# window is ~20s, so the process was killed mid-drain and "model released"
# never logged.
# ────────────────────────────────────────────────────────────────────────────


def _blocking_job(started: threading.Event, release: threading.Event):
    def _run() -> None:
        started.set()
        release.wait(10)

    return _run


async def test_close_is_bounded_when_a_slow_job_is_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long render in flight must not hold shutdown open indefinitely."""
    from services.flux_service import FluxService

    monkeypatch.setenv("IMAGE_GEN_SHUTDOWN_TIMEOUT_S", "0.5")
    svc = FluxService()
    started, release = threading.Event(), threading.Event()
    svc._executor.submit(_blocking_job(started, release))
    assert await asyncio.to_thread(started.wait, 10), "job never started"

    t0 = time.perf_counter()
    try:
        await svc.close()
        elapsed = time.perf_counter() - t0
    finally:
        release.set()

    assert elapsed < 3.0, (
        f"close() took {elapsed:.2f}s with a slow job in flight — lifespan shutdown "
        "must be bounded well under launchd's ~20s SIGTERM→SIGKILL window."
    )


async def test_close_cancels_the_queued_backlog() -> None:
    """``cancel_futures=True``: shutdown must not also drain everything QUEUED
    behind the running job — that multiplies the stall by the backlog depth."""
    from services.flux_service import FluxService

    svc = FluxService()
    started, release = threading.Event(), threading.Event()
    ran: list[int] = []
    svc._executor.submit(_blocking_job(started, release))
    for i in range(3):
        svc._executor.submit(lambda i=i: ran.append(i))
    assert await asyncio.to_thread(started.wait, 10), "job never started"

    close_task = asyncio.create_task(svc.close())
    await asyncio.sleep(0.05)  # let close() reach shutdown(cancel_futures=True)
    release.set()
    await close_task

    assert ran == [], f"queued jobs were drained instead of cancelled: {ran}"


async def test_close_logs_when_the_executor_does_not_drain(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """On timeout, shutdown must complete with a clear log — not raise."""
    from services.flux_service import FluxService

    monkeypatch.setenv("IMAGE_GEN_SHUTDOWN_TIMEOUT_S", "0.3")
    svc = FluxService()
    started, release = threading.Event(), threading.Event()
    svc._executor.submit(_blocking_job(started, release))
    assert await asyncio.to_thread(started.wait, 10), "job never started"

    try:
        with caplog.at_level("WARNING", logger="services.flux_service"):
            await svc.close()  # must not raise
    finally:
        release.set()

    messages = [r.getMessage() for r in caplog.records]
    assert any("did not drain" in m for m in messages), f"no warning logged about the undrained executor: {messages}"
