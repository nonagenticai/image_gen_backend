"""The MLX Metal allocator cache must be BOUNDED, and bounding it must never
be able to fail a render.

Measured on mini1 (M4 16 GB, schnell @ quantize=4, 512x512, 4 steps,
2026-09-11): MLX retains a flat 2453.7 MB of allocator cache from the first
generation onward and never gives it back — the idle unit sat on 11.2 GB of
IOAccelerator mappings after 22 h of zero traffic. ``set_cache_limit`` bounds
it to ~1.12 GB; the per-render ``clear_cache`` takes it to 0.

No real mlx/mflux is needed — a stand-in ``mlx.core`` module is installed into
``sys.modules``, so this runs on Linux CI as well as on-device. That is also
what lets the fail-safe arms assert the *raising* API, which is unreachable
against the real library.
"""

from __future__ import annotations

import os
import sys
import textwrap
import threading
from types import ModuleType
from typing import Any, cast

import pytest

from services.flux_service import DEFAULT_MLX_CACHE_LIMIT_MB, FluxService


class _MxRecorder:
    """Records every allocator call, in order, with the thread it ran on."""

    def __init__(self, raise_on: str | None = None) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.threads: list[threading.Thread] = []
        self._raise_on = raise_on

    def _record(self, name: str, arg: int | None = None) -> None:
        if self._raise_on == name:
            raise RuntimeError(f"simulated mlx failure in {name}")
        self.calls.append((name, arg))
        self.threads.append(threading.current_thread())

    def set_cache_limit(self, n: int) -> None:
        self._record("set_cache_limit", n)

    def clear_cache(self) -> None:
        self._record("clear_cache")

    def reset_peak_memory(self) -> None:
        self._record("reset_peak_memory")


class _StubMlxCore(ModuleType):
    """Stand-in for ``mlx.core``, carrying the three symbols the service calls.

    A ModuleType SUBCLASS with declared attributes (rather than attributes
    bolted onto a bare ModuleType) — the same shape ``test_native_import_thread``
    uses, so the attributes are statically visible.
    """

    def __init__(self, recorder: _MxRecorder) -> None:
        super().__init__("mlx.core")
        self.set_cache_limit = recorder.set_cache_limit
        self.clear_cache = recorder.clear_cache
        self.reset_peak_memory = recorder.reset_peak_memory


class _StubMlxPackage(ModuleType):
    def __init__(self, core: _StubMlxCore) -> None:
        super().__init__("mlx")
        self.core = core


def _install_mx(monkeypatch: pytest.MonkeyPatch, recorder: _MxRecorder) -> _MxRecorder:
    """Install ``mlx`` / ``mlx.core`` into sys.modules for the test's duration.

    The service imports ``mlx.core`` INSIDE the function (deliberately — it must
    not import on non-Darwin at module load), so patching sys.modules is what
    intercepts it.
    """
    core = _StubMlxCore(recorder)
    monkeypatch.setitem(sys.modules, "mlx", _StubMlxPackage(core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return recorder


@pytest.fixture
def fake_mx(monkeypatch: pytest.MonkeyPatch) -> _MxRecorder:
    return _install_mx(monkeypatch, _MxRecorder())


# --------------------------------------------------------------------------
# The limit itself
# --------------------------------------------------------------------------


def test_default_limit_is_applied_in_bytes(fake_mx: _MxRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGE_GEN_MLX_CACHE_LIMIT_MB", raising=False)
    FluxService()._apply_cache_limit()

    # MB in, BYTES out — a unit slip here is a 1 MB cache (catastrophic for
    # latency) or a 1 TB one (no bound at all), and both would still satisfy an
    # assertion that only checked the call happened.
    assert ("set_cache_limit", DEFAULT_MLX_CACHE_LIMIT_MB * 1024 * 1024) in fake_mx.calls
    assert DEFAULT_MLX_CACHE_LIMIT_MB == 1024


def test_limit_is_env_tunable_without_a_deploy(fake_mx: _MxRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_GEN_MLX_CACHE_LIMIT_MB", "4096")
    FluxService()._apply_cache_limit()
    assert ("set_cache_limit", 4096 * 1024 * 1024) in fake_mx.calls


def test_set_limit_is_paired_with_a_clear(fake_mx: _MxRecorder) -> None:
    """``set_cache_limit`` alone reclaims NOTHING.

    Measured: a 2453.7 MB cache was still exactly 2453.7 MB immediately after
    the call — it only bounds future growth. The one-shot clear is what hands
    the existing pool back, and it must come AFTER the limit is in place.
    """
    FluxService()._apply_cache_limit()
    names = [n for n, _ in fake_mx.calls]
    assert names.index("set_cache_limit") < names.index("clear_cache")


def test_negative_limit_leaves_mlx_alone(fake_mx: _MxRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_GEN_MLX_CACHE_LIMIT_MB", "-1")
    FluxService()._apply_cache_limit()
    assert fake_mx.calls == []


def test_garbage_limit_falls_back_to_the_default(fake_mx: _MxRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in a host env file must not stop the service from booting."""
    monkeypatch.setenv("IMAGE_GEN_MLX_CACHE_LIMIT_MB", "1gb")
    FluxService()._apply_cache_limit()
    assert ("set_cache_limit", DEFAULT_MLX_CACHE_LIMIT_MB * 1024 * 1024) in fake_mx.calls


@pytest.mark.parametrize("broken", ["set_cache_limit", "clear_cache", "reset_peak_memory"])
def test_a_raising_mlx_api_does_not_break_startup(
    broken: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_mx(monkeypatch, _MxRecorder(raise_on=broken))
    FluxService()._apply_cache_limit()  # must not raise
    assert "Could not apply MLX cache limit" in caplog.text


def test_a_missing_mlx_does_not_break_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux/dev boxes have no mlx at all — the ImportError must be swallowed."""
    monkeypatch.setitem(sys.modules, "mlx", cast("Any", None))
    monkeypatch.setitem(sys.modules, "mlx.core", cast("Any", None))
    FluxService()._apply_cache_limit()  # must not raise


# --------------------------------------------------------------------------
# The per-render trim
# --------------------------------------------------------------------------


class _Rendered:
    """Stand-in for mflux's ``GeneratedImage`` (it carries ``.image``)."""

    def __init__(self) -> None:
        self.image = "PIL"


class _StubPipeline:
    """Stand-in for ``mflux…Flux1``; records kwargs and the thread it ran on."""

    def __init__(self, boom: bool = False) -> None:
        self.boom = boom
        self.seen: dict[str, object] | None = None
        self.render_threads: list[threading.Thread] = []

    def generate_image(self, **kwargs: object) -> _Rendered:
        self.seen = kwargs
        self.render_threads.append(threading.current_thread())
        if self.boom:
            raise ValueError("render exploded")
        return _Rendered()


_RENDER_KWARGS: dict[str, object] = {
    "seed": 7,
    "prompt": "x",
    "num_inference_steps": 4,
    "height": 512,
    "width": 512,
    "guidance": 3.5,
    "negative_prompt": None,
}


def _trim(svc: FluxService, pipe: _StubPipeline) -> object:
    """Call the real ``_generate_and_trim`` with the real (keyword-only) signature.

    ``cast`` only on the pipeline: the stub stands in for ``Flux1``, which is a
    darwin-only import. Everything else is checked for real, so a signature
    change in the service breaks this test rather than silently passing.
    """
    return svc._generate_and_trim(cast("Any", pipe), **cast("Any", _RENDER_KWARGS))


def test_trim_runs_after_a_render_and_returns_the_result(fake_mx: _MxRecorder) -> None:
    pipe = _StubPipeline()
    out = cast("_Rendered", _trim(FluxService(), pipe))

    assert out.image == "PIL"
    assert pipe.seen == _RENDER_KWARGS  # every generation param passes through untouched
    assert ("clear_cache", None) in fake_mx.calls


def test_trim_runs_even_when_the_render_raises(fake_mx: _MxRecorder) -> None:
    """The failed render is the case that most needs the pool handed back."""
    with pytest.raises(ValueError, match="render exploded"):
        _trim(FluxService(), _StubPipeline(boom=True))
    assert ("clear_cache", None) in fake_mx.calls


def test_trim_is_switchable_off(fake_mx: _MxRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling can carry the load alone if the clear ever costs latency at
    a larger resolution than the one measured."""
    monkeypatch.setenv("IMAGE_GEN_MLX_CACHE_TRIM", "0")
    _trim(FluxService(), _StubPipeline())
    assert fake_mx.calls == []


def test_a_raising_trim_does_not_lose_a_finished_render(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_mx(monkeypatch, _MxRecorder(raise_on="clear_cache"))
    out = cast("_Rendered", _trim(FluxService(), _StubPipeline()))
    assert out.image == "PIL"
    assert "MLX cache trim after generation failed" in caplog.text


# --------------------------------------------------------------------------
# Thread affinity — the whole reason this lives where it does
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_trim_shares_the_pinned_mlx_thread_with_the_render(
    fake_mx: _MxRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLX's GPU stream is thread-bound: the allocator call must happen on the
    same ``gpu-lane-render`` worker that owns the pipeline, not on the event loop and
    not on a second executor submit (which re-opens the close()-between-submits
    race the upscale path documents).
    """
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "none")
    svc = FluxService()
    svc.ready = True
    pipe = _StubPipeline()
    svc._pipeline = cast("Any", pipe)

    try:
        await svc.generate(prompt="x", width=512, height=512, steps=4, guidance=3.5, seed=1)
    finally:
        await svc.close()

    assert len(fake_mx.threads) == 1, "expected exactly one allocator call — one render, one trim"
    assert fake_mx.threads[0] is pipe.render_threads[0]
    assert fake_mx.threads[0].name.startswith("gpu-lane-render")
    assert fake_mx.threads[0] is not threading.current_thread()


def test_the_native_import_hook_names_mlx_core_explicitly() -> None:
    """``_import_native_runtime`` must import ``mlx.core`` BY NAME on the main
    thread, not rely on it arriving as an mflux transitive — otherwise the
    first ``import mlx.core`` is the one inside ``_apply_cache_limit``, which
    runs on the pinned worker.

    Asserted over the AST, NOT over ``getsource`` text: the function carries a
    comment block explaining the rule, and a substring check matched THAT and
    stayed green with the import statement deleted. A guard that its own
    explanation satisfies is not a guard.
    """
    import ast
    import inspect

    from services import flux_service

    tree = ast.parse(textwrap.dedent(inspect.getsource(flux_service._import_native_runtime)))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "mlx.core" in imported, f"_import_native_runtime imports {sorted(imported)} — mlx.core must be named"


def test_the_limit_is_actually_wired_into_pipeline_construction() -> None:
    """``_apply_cache_limit`` must be CALLED from ``_build_pipeline``.

    Everything else here tests the helper in isolation, so deleting the single
    call site leaves the whole feature inert with a fully green suite. Asserted
    over the AST of ``_build_pipeline`` — a real ``initialize()`` run cannot
    reach it without mflux, and this is the one edge that makes the rest matter.
    """
    import ast
    import inspect

    from services import flux_service

    tree = ast.parse(textwrap.dedent(inspect.getsource(flux_service.FluxService._build_pipeline)))
    called = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_apply_cache_limit" in called, f"_build_pipeline calls {sorted(called)} — the cache limit is never applied"


def test_env_knobs_are_read_at_call_time_not_import_time() -> None:
    """Both knobs must be resolvable without a restart — an operator tunes them
    in ``~/.config/image-gen-backend/env`` and kickstarts, and a module-level
    read would bake the boot-time value into the process.
    """
    os.environ["IMAGE_GEN_MLX_CACHE_LIMIT_MB"] = "77"
    try:
        assert FluxService._cache_limit_mb() == 77
        os.environ["IMAGE_GEN_MLX_CACHE_LIMIT_MB"] = "88"
        assert FluxService._cache_limit_mb() == 88
    finally:
        os.environ.pop("IMAGE_GEN_MLX_CACHE_LIMIT_MB", None)
