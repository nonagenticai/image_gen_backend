"""`/health` must not call a cold process warm.

`model_loaded` mirrors `FluxService.ready`, which flips the moment
`_build_pipeline` returns — and that only CONSTRUCTS `Flux1(...)`. mflux
lazy-loads the weights on first render, so a process that has never generated
anything still answers `model_loaded: true`. Measured live on mini1 2026-09-11:
pid 45824, footprint 283 MB, zero generations since start, `model_loaded: true`.

`model_loaded` is deliberately left lying (the liveness CronJob and
`service.yml`'s `curl -f` both consume its current meaning). These tests pin the
honest fields added beside it: `pipeline_constructed`, `warm`, `rss_mb`.

The generation is MOCKED — no real model is loaded. The flag is set inside
`_generate_and_trim`, which is the seam the real executor calls, so driving that
directly exercises the production line rather than a stand-in for it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")

_RENDER_KWARGS: dict[str, Any] = {
    "seed": 1,
    "prompt": "a test prompt",
    "num_inference_steps": 4,
    "height": 512,
    "width": 512,
    "guidance": 0.0,
    "negative_prompt": None,
}


class _GoodPipeline:
    """Stands in for mflux's `Flux1`; returns instead of rendering."""

    def generate_image(self, **_kwargs: object) -> object:
        return object()


class _ExplodingPipeline:
    """A render that fails — the case that must NOT mark the process warm."""

    def generate_image(self, **_kwargs: object) -> object:
        raise RuntimeError("metal allocator refused the request")


@pytest.fixture(autouse=True)
def _reset_flux_singleton() -> Iterator[None]:
    """`get_flux_service` caches a module-level singleton, so `warm` LEAKS.

    Without this, the first test to set `warm=True` would make every later
    `warm is False` assertion pass on leftover state instead of on the code
    under test — order-dependent and vacuous. Same reason the sibling fixture in
    `test_health.py` exists for `ready`.
    """
    from services import flux_service  # type: ignore[import-not-found]

    flux_service._singleton = None
    yield
    flux_service._singleton = None


@pytest.fixture
def ready_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A service whose pipeline was CONSTRUCTED but which has never rendered.

    This is exactly the live mini1 shape: `ready=True`, weights not yet paged in.
    """
    from main import app  # type: ignore[import-not-found]
    from services.flux_service import FluxService  # type: ignore[import-not-found]

    async def _model_constructed(self: FluxService) -> None:
        self.ready = True

    monkeypatch.setattr(FluxService, "initialize", _model_constructed)

    with TestClient(app) as c:
        yield c


def test_a_constructed_but_never_rendered_service_reports_warm_false(ready_client: TestClient) -> None:
    """THE regression: construction is not warmth.

    `model_loaded` and `pipeline_constructed` both say true here and both are
    honest about what they mean. `warm` is the field that must disagree.
    """
    from main import app  # type: ignore[import-not-found]

    body = ready_client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["pipeline_constructed"] is True
    assert body["warm"] is False, (
        "a process that has never generated an image reported warm; that is the "
        "283 MB cold-process lie this field exists to expose"
    )
    assert app.state.flux_service.warm is False


def test_warm_flips_true_after_a_successful_generation(ready_client: TestClient) -> None:
    from main import app  # type: ignore[import-not-found]

    svc = app.state.flux_service
    assert ready_client.get("/health").json()["warm"] is False

    svc._generate_and_trim(_GoodPipeline(), **_RENDER_KWARGS)

    assert ready_client.get("/health").json()["warm"] is True, (
        "a render returned successfully but /health still reports the process cold"
    )


def test_warm_stays_false_after_a_FAILED_generation(ready_client: TestClient) -> None:
    """The false-alarm control, and the reason the flag is not set in `finally`.

    `_generate_and_trim` wraps the render in try/finally (the MLX cache trim).
    Setting `warm` in that `finally` would mark the process warm on the failure
    path too — this test is what distinguishes the two placements.
    """
    from main import app  # type: ignore[import-not-found]

    svc = app.state.flux_service
    with pytest.raises(RuntimeError, match="metal allocator refused"):
        svc._generate_and_trim(_ExplodingPipeline(), **_RENDER_KWARGS)

    assert ready_client.get("/health").json()["warm"] is False, (
        "a FAILED render marked the process warm — the flag is being set in the "
        "`finally` block rather than on the success path"
    )


def test_a_failed_render_does_not_undo_an_earlier_success(ready_client: TestClient) -> None:
    """`warm` is a latch: it answers "have the weights EVER been resident?"."""
    from main import app  # type: ignore[import-not-found]

    svc = app.state.flux_service
    svc._generate_and_trim(_GoodPipeline(), **_RENDER_KWARGS)
    with pytest.raises(RuntimeError):
        svc._generate_and_trim(_ExplodingPipeline(), **_RENDER_KWARGS)

    assert ready_client.get("/health").json()["warm"] is True


def test_health_reports_rss_in_the_right_unit_for_this_platform(ready_client: TestClient) -> None:
    """Guards the macOS-bytes / Linux-kilobytes `ru_maxrss` unit split.

    Getting it wrong is a 1024x error in one direction or the other, so the band
    has to be tight enough to catch that — an arbitrary "< 1 TiB" ceiling does
    NOT (measured here: the correct answer is 67 MiB, the bytes-read-as-KiB
    answer is 69184, and both sit under any loose ceiling).

    Physical RAM is the objective anchor: a process's PEAK RSS cannot exceed the
    machine's memory. `os.sysconf` supplies it on both macOS and Linux with no
    new dependency. On this box that is 69184 > 49152 MiB — caught; on mini1's
    16 GB box, caught by a wider margin.
    """
    total_ram_mb = (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // (1024 * 1024)

    rss_mb = ready_client.get("/health").json()["rss_mb"]

    assert isinstance(rss_mb, int)
    assert rss_mb > 8, f"rss_mb={rss_mb}; a live Python+FastAPI process cannot peak this low — units read too small"
    assert rss_mb < total_ram_mb, (
        f"rss_mb={rss_mb} exceeds this machine's {total_ram_mb} MiB of RAM — "
        "`ru_maxrss` is being read as KiB on a platform that reports bytes"
    )
