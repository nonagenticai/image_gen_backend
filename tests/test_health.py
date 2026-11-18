"""Smoke test for /health — exercises the real lifespan without loading FLUX.

This docstring used to claim the test "runs without loading FLUX (skips on
non-Darwin)". Neither half was true. `with TestClient(app)` runs the app's
lifespan, and `lifespan` calls `FluxService.initialize()` — the real mflux/mlx
import plus a 4-bit quantised model load. Measured on an M4: **61.7s of fixture
setup** for two assertions that never look at the model. Nothing skipped on
non-Darwin either — the mflux import is inside `initialize()`, so every platform
paid for it.

`initialize()` is stubbed rather than the lifespan being skipped. That keeps all
the wiring under test — `app.state` assignment, metering-Redis init, the
teardown path — while dropping the model load. `TestClient(app)` without the
context manager would be faster still, but it would silently stop exercising the
lifespan at all, and this repo has no CI (per its CLAUDE.md): these tests and the
on-device deploy check are the only gates it has.

Real model loading is covered where it actually matters — the documented deploy
check (`git pull` → `launchctl kickstart -k` on mini1 → `/health` → watch logs).
Exercised there 2026-08-03: clean SIGTERM logged "FLUX executor shut down; model
released" in 54ms and the unit came back `model_loaded: true`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")


@pytest.fixture(autouse=True)
def _reset_flux_singleton() -> Iterator[None]:
    """`get_flux_service` caches a module-level singleton, so `ready` LEAKS.

    Without this, `ready_client` setting `ready=True` persists into every later
    test in the file, and the degraded assertions below would pass on whatever
    the previous test happened to leave behind rather than on the code under
    test — order-dependent and vacuous. Measured: the lifespan log test failed
    for exactly this reason before the reset was added.
    """
    from services import flux_service  # type: ignore[import-not-found]

    flux_service._singleton = None
    yield
    flux_service._singleton = None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from main import app  # type: ignore[import-not-found]
    from services.flux_service import FluxService  # type: ignore[import-not-found]

    async def _no_model_load(self: FluxService) -> None:
        """Stand in for the mflux import + quantised model load.

        Returning cleanly (rather than raising) exercises the lifespan's SUCCESS
        branch. `ready` stays False because nothing sets it, so /health reports
        `model_loaded: false` — the shape a GPU-less box produces.
        """

    monkeypatch.setattr(FluxService, "initialize", _no_model_load)

    with TestClient(app) as c:
        yield c


@pytest.fixture
def ready_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A service whose model DID load — the false-alarm control.

    Without this, a fix that made /health non-2xx unconditionally would pass the
    degraded test and turn every healthy probe red on mini1.
    """
    from main import app  # type: ignore[import-not-found]
    from services.flux_service import FluxService  # type: ignore[import-not-found]

    async def _model_loaded(self: FluxService) -> None:
        self.ready = True

    monkeypatch.setattr(FluxService, "initialize", _model_loaded)

    with TestClient(app) as c:
        yield c


def test_health_live(client: TestClient) -> None:
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_health_is_not_2xx_when_the_model_failed_to_load(client: TestClient) -> None:
    """THE regression this file exists for.

    `service.yml`'s `healthcheck: curl -f http://localhost:9204/health` and
    gitops-production's `repo-services.txt` probe BOTH hit /health with `curl -f`,
    which fails only on HTTP >= 400. While /health answered 200 unconditionally, a
    service whose mflux/mlx import broke — `flux_service.py`'s `except ImportError`
    leaves `ready=False` and `main.lifespan` swallows any init failure — was
    byte-indistinguishable from a working one to every supervisor, while every
    POST /api/image/generate returned 503 forever.

    This test previously asserted `status_code == 200`, i.e. it PINNED the defect,
    justified by "it has to answer even in degraded mode". It does still answer —
    the body below is intact — it just must not answer *successfully*.
    """
    resp = client.get("/health")
    assert resp.status_code == 503, (
        f"a degraded service reported HTTP {resp.status_code}; `curl -f` cannot "
        "distinguish that from healthy, which is the whole defect"
    )


def test_health_still_returns_a_readable_degraded_body(client: TestClient) -> None:
    """The status code changed; the body contract must NOT.

    launchd-liveness-cronjob.yaml greps the body for `"status":"(healthy|ok)"`,
    and the response_model is still HealthResponse. A 503 that dropped or
    reshaped the payload would break that consumer instead of informing it.
    """
    resp = client.get("/health")
    body = resp.json()
    assert body["service"] == "image-gen-backend"
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_health_is_200_when_the_model_is_loaded(ready_client: TestClient) -> None:
    """The false-alarm control: a healthy service must still pass `curl -f`.

    Guards against "fix" the supervisor would read as permanently unhealthy.
    """
    resp = ready_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_lifespan_does_not_log_ready_when_the_model_failed_to_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The log line a human reads first must not claim a readiness it lacks.

    `lifespan` catches ANY init failure and then fell through to
    `logger.info("%s ready", SERVICE_NAME)` — so a service that failed to load
    FLUX announced itself ready in the one place an operator looks before
    anything else. Built inline rather than via the `client` fixture so caplog
    is installed before the lifespan runs.
    """
    from main import app  # type: ignore[import-not-found]
    from services.flux_service import FluxService  # type: ignore[import-not-found]

    async def _init_explodes(self: FluxService) -> None:
        raise ImportError("no mflux on this box")

    monkeypatch.setattr(FluxService, "initialize", _init_explodes)

    with caplog.at_level(logging.INFO), TestClient(app):
        pass

    messages = [r.getMessage() for r in caplog.records]
    assert not any(m == "image-gen-backend ready" for m in messages), (
        f"a service that failed FLUX init announced itself ready: {messages}"
    )
    assert any("DEGRADED" in m for m in messages), f"a degraded start must say so loudly; got {messages}"


def test_lifespan_leaves_a_flux_service_on_app_state(client: TestClient) -> None:
    """Shutdown's executor release depends on this assignment existing.

    `lifespan` sets `app.state.flux_service` on BOTH the success and the
    exception path, and shutdown calls `.close()` on whatever it finds there. If
    the assignment were ever dropped from a branch, teardown would silently skip
    releasing the MLX executor — the leak #28 fixed. Nothing asserted it.

    The `client` fixture is required so the assertion runs inside a started
    lifespan; the state is read off the app object itself, which is typed.
    """
    from main import app  # type: ignore[import-not-found]

    assert client is not None
    assert app.state.flux_service is not None
