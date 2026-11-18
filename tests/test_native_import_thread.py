"""WHICH THREAD first imports the native runtime (mflux → mlx).

`mflux` and `mlx` ship single-phase-init C extensions. Importing those off the
main thread is a documented CPython fragility, and it is what made a module
eviction lethal in tts_service (libtorch re-ran ``initModule()`` → SIGSEGV,
PR #92); the main-thread-import convention was established in tts_service #96,
and tts_backend / asr_backend already follow it.

Before the fix, the ONLY import of mflux in this repo lived inside
``FluxService._build_pipeline``, which ``initialize()`` hands to a pinned
``gpu-lane-render`` ThreadPoolExecutor(1) — so the first import of the native runtime
landed on a worker thread. These tests fail if it moves back there.

The counterpart property, which must NOT break: MLX's GPU stream is
thread-bound, so the pipeline CONSTRUCTION still has to happen on the pinned
executor thread. That is asserted here too — a "fix" that hoisted ``Flux1(...)``
onto the main thread would be a real regression.

Mechanism: ``builtins.__import__`` is wrapped for the duration of the test and
records the thread on which each ``mflux*``/``mlx*`` import statement executes.
``IMPORT_NAME`` calls ``__import__`` even when the module is already in
``sys.modules``, so this observes the statement itself without evicting or
re-creating any module — deliberately NOT ``patch.dict("sys.modules")``, whose
eviction is what caused the #92 segfault.

Fully hermetic: the wrapper serves stand-in modules, so no real MLX stack is
required and this runs on any machine.
"""

from __future__ import annotations

import builtins
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from services.flux_service import FluxService

_NATIVE_PREFIXES = ("mflux", "mlx")


class _StubModelConfig:
    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_name(cls, name: str) -> _StubModelConfig:
        return cls(name)


class _StubFlux1:
    """Stand-in for ``mflux…Flux1``; records the thread it was CONSTRUCTED on."""

    build_threads: list[threading.Thread] = []
    generate_threads: list[threading.Thread] = []

    def __init__(self, model_config: object, quantize: int) -> None:
        self.model_config = model_config
        self.quantize = quantize
        type(self).build_threads.append(threading.current_thread())

    def generate_image(self, **_kwargs: object) -> SimpleNamespace:
        type(self).generate_threads.append(threading.current_thread())
        return SimpleNamespace(image=Image.new("RGB", (64, 64)))


class _StubNativeModule(ModuleType):
    """Stand-in for an ``mflux…`` submodule, carrying the two symbols the
    service imports from it. A ModuleType subclass (rather than attributes
    bolted onto a bare ModuleType) so the attributes are statically declared.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.ModelConfig = _StubModelConfig  # noqa: N803 - mirrors the mflux symbol
        self.Flux1 = _StubFlux1  # noqa: N803 - mirrors the mflux symbol


@contextmanager
def _recording_native_imports(
    records: list[tuple[str, threading.Thread]],
    *,
    fail_with: ImportError | None = None,
) -> Iterator[None]:
    """Wrap ``builtins.__import__`` to record native-runtime import statements.

    Native names are served from a stand-in module (or raise ``fail_with``);
    every other import is delegated to the real ``__import__`` untouched.
    """
    real_import = builtins.__import__

    def _tracking_import(
        name: str,
        globals: Mapping[str, object] | None = None,  # noqa: A002 - mirrors __import__
        locals: Mapping[str, object] | None = None,  # noqa: A002 - mirrors __import__
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in _NATIVE_PREFIXES:
            records.append((name, threading.current_thread()))
            if fail_with is not None:
                raise fail_with
            return _StubNativeModule(name)
        return real_import(name, globals, locals, fromlist, level)

    # Installed via MonkeyPatch (rather than a bare attribute assignment, which
    # ty reads as an implicit shadow of the `__import__` builtin) so the
    # original is restored even if the body raises.
    patcher = pytest.MonkeyPatch()
    patcher.setattr(builtins, "__import__", _tracking_import)
    try:
        yield
    finally:
        patcher.undo()


def _fresh_service(monkeypatch: pytest.MonkeyPatch) -> FluxService:
    """A real ``FluxService`` on a pretend-Darwin host (the non-Darwin guard in
    ``initialize`` returns before any import, so CI would otherwise no-op)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    _StubFlux1.build_threads.clear()
    _StubFlux1.generate_threads.clear()
    return FluxService()


async def test_native_runtime_is_imported_on_the_main_thread() -> None:
    """The first ``mflux``/``mlx`` import must execute on MainThread."""
    monkeypatch = pytest.MonkeyPatch()
    records: list[tuple[str, threading.Thread]] = []
    try:
        svc = _fresh_service(monkeypatch)
        assert threading.current_thread() is threading.main_thread(), (
            "test harness precondition: the lifespan/event-loop thread under test must be MainThread"
        )
        with _recording_native_imports(records):
            await svc.initialize()
        await svc.close()
    finally:
        monkeypatch.undo()

    assert records, "no mflux/mlx import was observed — initialize() never imported the native runtime"

    # For every distinct native module, the FIRST import statement to touch it
    # (the one that would initialise its C extension) must be on MainThread.
    first_by_name: dict[str, threading.Thread] = {}
    for name, thread in records:
        first_by_name.setdefault(name, thread)
    offenders = {n: t.name for n, t in first_by_name.items() if t is not threading.main_thread()}
    assert not offenders, (
        f"native runtime first imported off the main thread: {offenders}. "
        "mflux/mlx are single-phase-init C extensions; import them on the main thread in "
        "FluxService.initialize() BEFORE the run_in_executor hand-off."
    )


async def test_pipeline_is_still_built_on_the_pinned_mlx_thread() -> None:
    """The property the fix must not break: MLX's GPU stream is thread-bound,
    so ``Flux1(...)`` construction stays on the pinned ``gpu-lane-render`` executor."""
    monkeypatch = pytest.MonkeyPatch()
    records: list[tuple[str, threading.Thread]] = []
    try:
        svc = _fresh_service(monkeypatch)
        with _recording_native_imports(records):
            await svc.initialize()
        # Read before close() — close() flips `ready` back to False.
        ready_after_init = svc.ready
        await svc.close()
    finally:
        monkeypatch.undo()

    assert ready_after_init is True, "initialize() did not reach the ready state"
    assert len(_StubFlux1.build_threads) == 1, "the pipeline was never constructed"
    build_thread = _StubFlux1.build_threads[0]
    assert build_thread is not threading.main_thread(), (
        "pipeline construction moved onto the main thread — it must stay on the pinned MLX executor"
    )
    assert build_thread.name.startswith("gpu-lane-render"), (
        f"pipeline built on {build_thread.name!r}, not the pinned gpu-lane-render executor thread"
    )


async def test_generate_runs_on_the_very_thread_that_built_the_pipeline() -> None:
    """MLX's GPU stream is bound to a THREAD OBJECT, not a name prefix: the
    render must run on the identical thread the pipeline was constructed on
    (the admission lane's single worker), or MLX raises ``no Stream(gpu, ...)``.
    """
    monkeypatch = pytest.MonkeyPatch()
    records: list[tuple[str, threading.Thread]] = []
    try:
        monkeypatch.setenv("IMAGE_GEN_UPSCALE", "none")
        monkeypatch.setenv("IMAGE_GEN_MLX_CACHE_TRIM", "0")
        svc = _fresh_service(monkeypatch)
        with _recording_native_imports(records):
            await svc.initialize()
            assert svc.ready is True
            for _ in range(2):
                await svc.generate(prompt="x", width=512, height=512, steps=4, guidance=0.0, seed=1)
        await svc.close()
    finally:
        monkeypatch.undo()

    assert len(_StubFlux1.build_threads) == 1, "the pipeline was never constructed"
    assert len(_StubFlux1.generate_threads) == 2, "generate never reached the pipeline"
    build_thread = _StubFlux1.build_threads[0]
    for gen_thread in _StubFlux1.generate_threads:
        assert gen_thread is build_thread, (
            f"render ran on {gen_thread!r}, pipeline was built on {build_thread!r} — not the same thread object"
        )


async def test_missing_mflux_still_degrades_instead_of_raising() -> None:
    """Preserved failure behaviour: an ImportError from the native import is
    logged and the service starts in degraded mode — moving the import to the
    main thread must not change that."""
    monkeypatch = pytest.MonkeyPatch()
    records: list[tuple[str, threading.Thread]] = []
    try:
        svc = _fresh_service(monkeypatch)
        with _recording_native_imports(records, fail_with=ImportError("No module named 'mflux'")):
            await svc.initialize()
        await svc.close()
    finally:
        monkeypatch.undo()

    assert records, "the native import was never attempted"
    assert svc.ready is False
    assert _StubFlux1.build_threads == []
