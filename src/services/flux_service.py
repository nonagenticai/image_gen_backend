"""FLUX inference service backed by mflux (Apple Silicon MLX).

Lazy-loads the model on first init() call. All MLX work — both pipeline
construction and generation — runs on a single dedicated worker thread.
MLX's GPU stream is thread-bound, so building the pipeline on one thread and
generating on another raises `RuntimeError: no Stream(gpu, ...)`. A pinned
single-thread executor keeps every MLX call on the same thread while leaving
FastAPI's event loop responsive (and serialises generation, which is correct:
there is one GPU).

Since #69 that single thread is OWNED BY THE w2v_common ADMISSION GATE, lane
``render`` (``ThreadPoolExecutor(max_workers=max_active)``, threads named
``gpu-lane-render``). Renders go through ``gate.run`` — bounded FIFO queue,
503 + Retry-After when full or waited past ``max_wait_s``, and a queued render
is skipped when its client has already disconnected. Pipeline construction is
submitted to the SAME lane executor (outside admission — there is no traffic
before ``ready``) so the thread affinity above is preserved. ``max_active`` of
the render lane is therefore pinned to 1: a second worker thread would be a
second MLX stream (see ``_render_lane_config``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import random
import sys
from typing import TYPE_CHECKING

from prometheus_client import Counter
from w2v_common.admission import AdmissionGate, GateClosedError, LaneConfig

if TYPE_CHECKING:
    from w2v_common.admission import SupportsDisconnect

logger = logging.getLogger(__name__)

# A failed upscale is otherwise INVISIBLE. The except arm below deliberately
# swallows the exception so a committed render is never discarded (see the long
# comment at the call site), which means a degraded 200 — an image smaller than
# the 1536 contract — is indistinguishable from a healthy one in every signal
# this service emits. ``ImageGenResponse.upscale_failed`` is the caller's half
# of that fact; this counter is the operator's.
#
# Default registry ON PURPOSE: ``src/main.py`` exposes /metrics via
# ``Instrumentator().instrument(app).expose(app)``, which serves
# ``prometheus_client.REGISTRY``. This repo declares no custom registry, so a
# bare module-level Counter is already on the scraped one. Module level (not
# per-call) because a collector may be registered exactly once — constructing
# it inside generate() raises "Duplicated timeseries" on the second request.
UPSCALE_FAILURES = Counter(
    "image_gen_upscale_failures_total",
    "Post-render LANCZOS upscales that failed, returning the render at its un-upscaled size.",
)

if TYPE_CHECKING:
    from mflux.models.flux.variants.txt2img.flux import Flux1  # noqa: F401
    from mflux.utils.generated_image import GeneratedImage  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401


# mflux-supported FLUX variants. schnell = 4-step distilled (fastest, no CFG);
# dev = 20-50 step (best quality, uses CFG). schnell is the default for
# interactive workflows on mini1's M4 16 GB.
DEFAULT_MODEL = os.getenv("IMAGE_GEN_MODEL", "schnell")

# Upper bound on how long lifespan shutdown may wait for the MLX executor to
# drain. launchd sends SIGTERM then SIGKILLs ~20s later, so an unbounded wait
# (a FLUX render is 30s to several minutes on M-series) guarantees the process
# is killed mid-drain and the teardown log never lands. Read at call time so
# the host env / tests can tune it.
DEFAULT_SHUTDOWN_TIMEOUT_S = 5.0

# GPU admission lane for renders (#69, owner contract 2026-10-15). Code
# defaults = the issue's initial numbers: one render on the GPU, four queued,
# a queued render gives up (503 + Retry-After) after 30 s. The gitops
# host-config ConfigMap overrides them via GPU_LANE_RENDER_MAX_WAITING /
# GPU_LANE_RENDER_MAX_WAIT_S (read once, when FluxService is constructed).
RENDER_LANE = "render"
RENDER_LANE_DEFAULT = LaneConfig(max_active=1, max_waiting=4, max_wait_s=30.0)


def _render_lane_config() -> dict[str, LaneConfig]:
    """The gate's lane map. Fails LOUD on ``GPU_LANE_RENDER_MAX_ACTIVE != 1``.

    The lane executor has ``max_active`` worker threads. The pipeline is built
    on one of them and MLX's GPU stream is bound to that thread, so a render
    scheduled on a second worker raises ``no Stream(gpu, ...)`` — and one
    FLUX render already saturates this GPU. An env override that cannot work
    must stop the unit at startup, not 500 every second concurrent request.
    """
    cfg = LaneConfig.from_env(RENDER_LANE, RENDER_LANE_DEFAULT)
    if cfg.max_active != 1:
        raise ValueError(
            f"GPU_LANE_RENDER_MAX_ACTIVE={cfg.max_active}: the render lane must have max_active=1 — "
            "the MLX pipeline is bound to the single lane thread it was built on"
        )
    return {RENDER_LANE: cfg}


# Ceiling on MLX's Metal allocator CACHE — the pool of freed-but-retained GPU
# buffers MLX keeps around so the next allocation of the same shape is free.
# Measured on mini1 (M4, 16 GB, schnell @ quantize=4, 512x512, 4 steps,
# 2026-09-11): MLX retains a FLAT 2453.7 MB of cache from the first generation
# onward — it is not a per-request leak, it is a one-time high-water allocator
# reservation that is then never given back. On a 16 GB box co-hosting
# mlx-backend and tts_backend that 2.4 GB is the difference between fitting and
# swapping; the live unit had 11.2 GB of IOAccelerator mappings SWAPPED OUT
# after 22 h of zero traffic.
#
# 1 GB matches mflux's own default (``MemorySaver(cache_limit_bytes=1000**3)``
# and ``CallbackManager._register_memory_saver``'s non-low_ram branch). Env-
# tunable so an operator on a bigger host can raise it without a deploy:
#   >0  -> that many MB
#    0  -> no cache at all (every buffer returned immediately; slowest)
#   <0  -> leave MLX's default alone (feature off)
DEFAULT_MLX_CACHE_LIMIT_MB = 1024


class ServiceUnavailableError(RuntimeError):
    """The service cannot serve this request — it is not ready, or is closing.

    Distinct from a genuine internal fault: the route maps it to a clean 503
    (the same answer its ``svc.ready`` pre-check gives) instead of letting a
    bare ``RuntimeError`` fall through to the 500 envelope. This is the
    close()-during-an-in-flight-generation window, which is the NORMAL case on
    a launchd SIGTERM with traffic in flight.
    """


def _snap_to_long_edge(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Rescale ``(width, height)`` so its LONG edge lands on ``long_edge``,
    holding the aspect ratio as closely as the 64-px grid allows.

    mflux requires both dimensions to be multiples of 64, so exact aspect is
    rarely representable and the ROUNDING RULE is what decides the shape. The
    long edge is FLOORED (it must never exceed the caller's ceiling); the short
    edge is rounded to the NEAREST multiple of 64, because flooring it too is
    what broke aspect preservation — see ``_aspect_clamp``.

    Both dims keep the historical 256 floor, and the short edge is additionally
    bounded by the long edge so rounding can never swap which side is longer.
    """
    longest = max(width, height)
    shortest = min(width, height)
    scale = long_edge / longest
    long_px = max(256, (long_edge // 64) * 64)
    # +0.5 rather than round(): round() is banker's, so a short edge landing
    # exactly halfway (768x1024 → 672 → 10.5 grid units) would round DOWN on
    # even units and UP on odd ones. Half-up is boring and reproducible.
    short_px = max(256, min(long_px, int(shortest * scale / 64.0 + 0.5) * 64))
    return (long_px, short_px) if width >= height else (short_px, long_px)


def _aspect_clamp(width: int, height: int, long_edge_cap: int) -> tuple[int, int]:
    """Scale ``(width, height)`` DOWN so the LONG edge fits ``long_edge_cap``,
    preserving aspect ratio as closely as the 64-px grid allows. Requests whose
    long edge already fits are returned unchanged. A per-dimension clamp would
    collapse 896x512 → 512x512; this keeps the 16:9 shape.

    ⚠️ FIXED 2026-09-15 — it did NOT preserve aspect, despite this docstring
    having promised it since the function was written. It floored BOTH dims
    independently to a multiple of 64, and flooring the SHORT edge throws away
    up to 63 px of a number that is already small, which is a large fraction of
    it. Measured against the live 896 cap, before the fix:

        request      old result     old aspect   ordered aspect
        1280x720  →  896x448            2.000          1.778     (16:9)
        1920x1080 →  896x448            2.000          1.778     (16:9)
        1792x1024 →  896x512            1.750          1.750     (ok)
        1024x1024 →  896x896            1.000          1.000     (ok)

    So EVERY 16:9 order — `course/thumbnail.py`'s 1280x720, and anything
    authored at 1080p/1440p — was rendered at 2.0:1 and then handed to a 16:9
    canvas, where ken_burns ``contain`` letterboxes the mismatch. It also cost
    pixels: 896x448 is 401k, 896x512 is 458k. The gitops ConfigMap comment
    claiming the cap yields "896x504 for 16:9" describes arithmetic this
    function never performed.

    Rounding the short edge to the NEAREST 64 gives 896x512 (1.75) for a 16:9
    order — 0.028 off instead of 0.222, and more real pixels. 16:9 is not
    exactly representable on a 64-px grid at this cap (504 is 7.875 units), so
    1.75 is the closest honest answer, not a residual bug.
    """
    if max(width, height) <= long_edge_cap:
        return width, height
    return _snap_to_long_edge(width, height, long_edge_cap)


def _render_to_cap(width: int, height: int, long_edge_cap: int) -> tuple[int, int]:
    """Scale a BELOW-cap request UP to ``long_edge_cap`` before rendering.

    The inverse of ``_aspect_clamp``, and the answer to the double-resample:
    the output is LANCZOS-upscaled to ``IMAGE_GEN_UPSCALE_LONG_EDGE`` (1536)
    regardless, so a request rendered at 512x320 has that 1536 asset
    interpolated out of 164k real pixels — a 3x blow-up of invented detail —
    while the host's budget allows 896 and the GPU is idle either way. Asking
    the model for the pixels instead of asking LANCZOS to guess them changes
    NO output dimension: the same 1536 asset comes back, carrying ~3x the real
    information. Requests already at or above the cap are returned unchanged
    (``_aspect_clamp`` owns those).

    Off by default (``IMAGE_GEN_RENDER_TO_CAP``) — it is a real GPU cost on a
    16 GB box that co-hosts other MLX services, and that cost is UNMEASURED at
    896. See the call site.
    """
    if max(width, height) >= long_edge_cap:
        return width, height
    return _snap_to_long_edge(width, height, long_edge_cap)


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean env var. Accepts 1/true/yes/on (case-insensitive)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _upscale(img: PILImage.Image, target_long_edge: int) -> PILImage.Image:
    """Enlarge ``img`` so its long edge reaches ``target_long_edge`` (LANCZOS),
    preserving aspect ratio. Never downscales — a no-op when already large
    enough. Pure-PIL CPU work; call it off the event loop.
    """
    from PIL import Image as _Image

    w, h = img.size
    longest = max(w, h)
    if longest >= target_long_edge:
        return img
    scale = target_long_edge / longest
    return img.resize((round(w * scale), round(h * scale)), _Image.Resampling.LANCZOS)


def _ensure_pillow_typing_importable() -> None:
    """Let ``PIL._typing`` import on Python 3.14 under pillow 13.

    pillow 13.0.0's ``PIL/_typing.py`` does ``from numbers import _IntegralLike``,
    a name CPython 3.14 does not have, so importing that module raises
    ``ImportError``. pillow itself only reaches it under ``TYPE_CHECKING``, but
    mflux (``mflux/utils/image_util.py``) imports ``PIL._typing`` at runtime, so
    on 2026-10-21 the pillow 13 security bump (#72) left the model unloadable
    and ``/health`` at 503. The name is used for type hints only, so aliasing it
    to ``numbers.Integral`` changes no behaviour. It is a no-op once a Python or
    pillow release provides the name itself.
    """
    import numbers

    if not hasattr(numbers, "_IntegralLike"):
        numbers._IntegralLike = numbers.Integral  # type: ignore[attr-defined]


def _import_native_runtime() -> None:
    """Import mflux (and transitively mlx) — MUST run on the MAIN THREAD.

    Kept as a module-level function purely so the import statements have one
    home; the thread it runs on is the point. See the call site in
    ``FluxService.initialize`` for the full rationale. Raises ``ImportError``
    exactly as the executor-side import used to, so a missing/broken mflux is
    still handled by the caller's existing ``except ImportError`` (degraded
    mode), unchanged.
    """
    _ensure_pillow_typing_importable()
    import mlx.core  # noqa: F401 - see below; NOT redundant with the mflux imports
    from mflux.models.common.config.model_config import ModelConfig  # noqa: F401
    from mflux.models.flux.variants.txt2img.flux import Flux1  # noqa: F401

    # ``mlx.core`` is named EXPLICITLY, not left to arrive as an mflux transitive.
    # ``_apply_cache_limit`` / ``_generate_and_trim`` both do their own
    # ``import mlx.core`` on the pinned ``gpu-lane-render`` thread, and an IMPORT_NAME
    # for a module that is not yet in sys.modules is exactly the off-main-thread
    # single-phase C-extension init this function exists to prevent.
    # ``test_native_import_thread`` fails without this line.


class FluxService:
    """Wrapper around mflux's Flux1 pipeline."""

    def __init__(self) -> None:
        self.model_repo: str = DEFAULT_MODEL
        self.ready: bool = False
        # ``ready`` means "Flux1 was CONSTRUCTED". mflux lazy-loads the weights,
        # so it says nothing about whether this process can actually render.
        # ``warm`` is the honest signal: flipped only by a render that RETURNED
        # (see _generate_and_trim), never by one that raised. Surfaced on
        # /health as ``warm``.
        self.warm: bool = False
        self._pipeline = None  # mflux.Flux1 instance once loaded
        self._lock = asyncio.Lock()
        self._closed: bool = False
        # All MLX work runs on the render lane's single thread — see module
        # docstring. The gate owns that executor; the reference is kept only so
        # close() can JOIN the worker after the gate has shut it down (the gate
        # itself never blocks on its executor).
        self._gate = AdmissionGate(_render_lane_config())
        self._executor: concurrent.futures.ThreadPoolExecutor = self._gate.lane(RENDER_LANE).executor

    async def initialize(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("FluxService is closed — it cannot be re-initialized.")
            if self.ready:
                return
            if sys.platform != "darwin":
                logger.warning("Not on Darwin — FLUX service runs in degraded mode (no model loaded).")
                return

            loop = asyncio.get_running_loop()
            try:
                # ⚠️ Import the native runtime HERE, on the MAIN THREAD, BEFORE the
                # executor hand-off below — do not "tidy" this away as a duplicate
                # of _build_pipeline's imports.
                #
                # `initialize()` is awaited from the FastAPI lifespan, which runs on
                # the event loop that uvicorn starts with `asyncio.run()` in the
                # process's main thread — so this line executes on MainThread, while
                # `_build_pipeline` (correctly) does not. mflux pulls in `mlx`, and
                # both ship single-phase-init C extensions; importing those off the
                # main thread is a documented CPython fragility. It is what made a
                # module eviction lethal in tts_service (libtorch re-ran
                # initModule() → SIGSEGV, PR #92) and the main-thread-import
                # convention was established there in PR #96; tts_backend and
                # asr_backend already follow it.
                #
                # By the time the executor runs _build_pipeline these modules are in
                # sys.modules, so its own imports are cache hits and no C extension
                # initialises off-thread. The pipeline CONSTRUCTION stays on the
                # pinned `gpu-lane-render` thread — MLX's GPU stream is thread-bound (see
                # the module docstring), so it must not move.
                _import_native_runtime()
                await loop.run_in_executor(self._executor, self._build_pipeline)
            except ImportError:
                logger.exception("mflux not installed — install via `uv sync`")
                return

            self.ready = True
            logger.info(
                "FLUX pipeline initialized: model=%s quantize=%s",
                self.model_repo,
                os.getenv("IMAGE_GEN_QUANTIZE", "8"),
            )

    def _build_pipeline(self) -> None:
        """Construct the mflux pipeline. Runs on the pinned MLX thread.

        Imported lazily so non-Darwin (e.g. CI) never imports mflux. mflux
        0.10+ moved Flux1/ModelConfig out of the top-level __init__; import
        from the canonical submodule paths.
        """
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

        self._model_config = ModelConfig.from_name(self.model_repo)
        self._pipeline = Flux1(
            model_config=self._model_config,
            quantize=int(os.getenv("IMAGE_GEN_QUANTIZE", "8")),
        )
        self._apply_cache_limit()

    def _apply_cache_limit(self) -> None:
        """Bound MLX's Metal allocator cache. Runs on the pinned MLX thread.

        WHY NOT mflux's ``MemorySaver`` callback, which is the obvious seam:
        it is a CLI-only construct. ``CallbackManager._register_memory_saver``
        builds it from an argparse ``Namespace`` gated on ``args.low_ram``, and
        its ``call_before_loop`` runs ``_delete_text_encoders()`` on EVERY
        generation — nulling ``clip_text_encoder`` / ``t5_text_encoder``, with
        mflux's own comment reading "repeated image generation only works with
        the same prompt (cache)". That is fine for a one-shot CLI render and
        wrong for a long-lived multi-prompt HTTP service: every request would
        pay a full text-encoder reload, or worse, reuse the previous prompt's
        embedding. We want the memory bound, not the model dismantled.

        So we call the same two mlx primitives MemorySaver itself calls, which
        is also exactly what mflux does in its OWN non-low_ram branch
        (``mx.set_cache_limit(n); mx.clear_cache(); mx.reset_peak_memory()``).
        This is the library's supported seam for "bound the cache without
        dropping the model" — MemorySaver is the seam for "drop the model".

        FAIL-SAFE: an image must never fail because a memory hint failed. Any
        missing/renamed mlx API or raising call is logged and swallowed.
        """
        limit_mb = self._cache_limit_mb()
        if limit_mb < 0:
            logger.info("MLX cache limit disabled (IMAGE_GEN_MLX_CACHE_LIMIT_MB<0) — MLX default in effect.")
            return
        try:
            import mlx.core as mx

            mx.set_cache_limit(limit_mb * 1024 * 1024)
            # set_cache_limit only bounds FUTURE growth: measured on mini1, a
            # 2453.7 MB cache already in hand was still 2453.7 MB immediately
            # after the call. The one-shot clear is what actually hands it back,
            # and mflux pairs the two calls for the same reason.
            mx.clear_cache()
            mx.reset_peak_memory()
        except Exception:  # noqa: BLE001 - a memory hint must never break generation
            logger.warning("Could not apply MLX cache limit (%d MB) — continuing unbounded.", limit_mb, exc_info=True)
        else:
            logger.info("MLX Metal allocator cache limited to %d MB.", limit_mb)

    @staticmethod
    def _cache_limit_mb() -> int:
        """Resolve ``IMAGE_GEN_MLX_CACHE_LIMIT_MB``, read at call time.

        A non-numeric value is a misconfiguration, not a reason to crash the
        service on boot: log it and fall back to the default.
        """
        raw = os.getenv("IMAGE_GEN_MLX_CACHE_LIMIT_MB")
        if raw is None or raw.strip() == "":
            return DEFAULT_MLX_CACHE_LIMIT_MB
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "IMAGE_GEN_MLX_CACHE_LIMIT_MB=%r is not an integer — using the %d MB default.",
                raw,
                DEFAULT_MLX_CACHE_LIMIT_MB,
            )
            return DEFAULT_MLX_CACHE_LIMIT_MB

    def _generate_and_trim(
        self,
        pipeline: Flux1,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance: float,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        """Render, then hand MLX's retained buffers back. On the pinned MLX thread.

        The trim MUST share the thread that owns the pipeline (the ``gpu-lane-render``
        single-worker executor, see the module docstring) — that is why this is
        one callable submitted to the executor rather than a second submit, which
        would also re-open the close()-between-submits race documented below on
        the upscale path.

        MEASURED, not assumed (mini1 M4 16 GB, schnell quantize=4, 512x512,
        4 steps, n=3 per arm, 2026-09-11, warm generations only):

          arm                          latency s        cache MB   footprint
          A  unbounded (today)         28.59 / 28.58     2453.7      12 GB
          B  set_cache_limit(1 GB)     30.15/30.37/28.52 1119-1127   10-11 GB
          C  B + clear_cache() each    28.74/28.72/28.76     0.0     10-11 GB

        So the "clearing the cache costs latency" worry does not hold for this
        workload: C is +0.16 s (+0.54%) against A's warm mean, inside A's own
        run-to-run spread, and it returns a further ~1.1 GB. B's 30.1/30.4 are
        the ONE-OFF cost of handing 1.3 GB back on the first bounded render, not
        a steady state — its third render is already 28.52 s. At 512x512/4 steps
        the retained pool buys essentially nothing because each step reallocates
        anyway. That may not hold at larger resolutions or step counts, which is
        why the trim is a switch (``IMAGE_GEN_MLX_CACHE_TRIM=0`` turns it off and
        leaves the ``set_cache_limit`` ceiling doing the work on its own).
        """
        try:
            rendered = pipeline.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance=guidance,
                negative_prompt=negative_prompt,
            )
            # HERE, not in the ``finally`` below: ``finally`` also runs when the
            # render RAISED, and a failed render is exactly the case that must
            # NOT mark the process warm. Placed after the call returns so the
            # flag means "weights are resident and produced an image".
            self.warm = True
            return rendered
        finally:
            # ``finally``, not a trailing statement: a render that RAISED is
            # precisely the case that leaves the retained pool at its high-water
            # mark with nothing to show for it.
            if os.getenv("IMAGE_GEN_MLX_CACHE_TRIM", "1").strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:  # noqa: BLE001 - a memory hint must never lose a finished render
                    logger.warning("MLX cache trim after generation failed — continuing.", exc_info=True)

    async def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int | None,
        negative_prompt: str | None = None,
        *,
        request: SupportsDisconnect | None = None,
    ) -> tuple[PILImage.Image, int]:
        """Render under the ``render`` admission lane.

        Raises ``GpuBusy`` (queue full / waited past ``max_wait_s`` → 503 +
        Retry-After) and ``ClientGone`` (``request`` disconnected while queued →
        render skipped, 499); both are mapped by ``install_gpu_busy_handler``.
        """
        pipeline = self._pipeline
        if self._closed or not self.ready or pipeline is None:
            # 503, not 500: "not ready" and "shutting down" are both availability
            # answers. The route pre-checks ``svc.ready``, but close() can flip it
            # in the window between that check and here, so this must give the
            # caller the same clean 503 rather than an opaque internal error.
            raise ServiceUnavailableError("FLUX service is not ready (initializing or shutting down).")

        # Resolution cap. mini1 (16 GB M4) co-hosts mlx-backend; FLUX saturates
        # the unified-memory pool near ~1 MP (≈1024²). Cap the LONG EDGE and
        # scale both dims proportionally so non-square (e.g. 16:9 896x512)
        # requests keep their aspect instead of collapsing to a square — the
        # old per-dimension clamp turned 896x512 into 512x512. 896 long-edge
        # ≈458k px is well under the saturation point. Tune via env on bigger
        # hosts; IMAGE_GEN_MAX_DIM stays a back-compat alias — but it is NOT
        # set in production any more (removed from the ConfigMap 2026-09-12
        # when the cap was raised 512 → 896), so the live value arrives via
        # IMAGE_GEN_MAX_LONG_EDGE. Verified against the ConfigMap AND mini1's
        # running process env, 2026-09-15.
        long_edge_cap = int(os.getenv("IMAGE_GEN_MAX_LONG_EDGE", os.getenv("IMAGE_GEN_MAX_DIM", "896")))
        clamped_w, clamped_h = _aspect_clamp(width, height, long_edge_cap)
        if (clamped_w, clamped_h) != (width, height):
            logger.info(
                "Clamping %dx%d → %dx%d (long-edge cap %dpx, aspect preserved)",
                width,
                height,
                clamped_w,
                clamped_h,
                long_edge_cap,
            )
            width, height = clamped_w, clamped_h

        # Render AT the cap rather than upscaling into it. OFF by default.
        #
        # The live pixel path for a lesson still is: the orchestrator orders
        # 512x320 (its DEFAULT_LOCAL_IMAGE_SIZE, chosen back when this host's
        # cap really was 512), FLUX renders 512x320, LANCZOS invents its way to
        # 1536x960 below, ken_burns records that into a 2560x1440 Chromium
        # frame, and the compiler scales THAT to the master. Four resamples,
        # and only the first carries information. The 512 → 896 cap raise on
        # 2026-09-12 bought exactly nothing for those stills, because the
        # ORDER is still 512 and a cap only ever scales down.
        #
        # This flag closes that gap from the service side without waiting on a
        # caller change: the same 1536 asset is returned, so no downstream
        # contract moves (and w2v_orchestrator scripts/e2e_validate.py:3866
        # hard-asserts `long_edge == 1536`, so none may) — it just carries ~3x
        # the real pixels. Deliberately NOT the default: rendering 896x576
        # instead of 512x320 is ~3x the latent area on a 16 GB box whose
        # memory envelope was measured at 512x512, and that cost is UNMEASURED.
        # Production should turn it on via the ConfigMap once latency and
        # phys_footprint have been read off the device at 896.
        if _env_flag("IMAGE_GEN_RENDER_TO_CAP", default=False):
            grown_w, grown_h = _render_to_cap(width, height, long_edge_cap)
            if (grown_w, grown_h) != (width, height):
                logger.info(
                    "Rendering at cap: %dx%d → %dx%d (long-edge cap %dpx) — the output "
                    "upscale interpolates from real pixels instead of inventing them",
                    width,
                    height,
                    grown_w,
                    grown_h,
                    long_edge_cap,
                )
                width, height = grown_w, grown_h

        seed_used = seed if seed is not None else random.randint(0, 2**31 - 1)

        # mflux's generate_image is synchronous and CPU/GPU-bound. Run on the
        # render lane's single thread (see module docstring) so the event loop
        # stays responsive and every MLX call shares one GPU stream. mflux 0.17+
        # takes generation params as flat kwargs (no `config=` argument).
        # gate.run holds the slot until the THREAD ends, even if this coroutine
        # is cancelled, so an abandoned in-flight render still blocks the next.
        try:
            result = await self._gate.run(
                RENDER_LANE,
                lambda: self._generate_and_trim(
                    pipeline,
                    seed=seed_used,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    negative_prompt=negative_prompt,
                ),
                request=request,
            )
        except GateClosedError as exc:
            # close() shut the gate down between the route's readiness check
            # and admission (or while this render was queued). No GPU work was
            # done, so answer 503 rather than 500. This is the ONLY shutdown
            # path: close() closes the gate before it touches the executor, and
            # gate.run submits synchronously after admission (no await between
            # the closed-check and the submit), so the old "cannot schedule new
            # futures after shutdown" RuntimeError fallback is unreachable.
            raise ServiceUnavailableError("FLUX service is shutting down.") from exc

        # mflux returns a GeneratedImage with `.image` PIL attribute.
        pil_image = result.image if hasattr(result, "image") else result

        # Upscale on the way out so callers get a frame-fillable asset (~1536px
        # long edge) without paying the memory cost of generating that large.
        # IMAGE_GEN_UPSCALE=none disables it per host.
        #
        # ⛔ DO NOT set IMAGE_GEN_UPSCALE=none (or move IMAGE_GEN_UPSCALE_LONG_EDGE)
        # in production to "avoid the double resample". It is a real double
        # resample, but the 1536 output is a CONTRACT, not an accident:
        # ``w2v_orchestrator/scripts/e2e_validate.py:3866`` fails the panel-image
        # E2E on `long_edge != 1536` — an exact equality, not a floor, made a
        # deliberate product decision on 2026-08-26. Turning the upscale off
        # here reds that gate and hands ken_burns a still that Chromium then
        # blows up 5x in CSS, which is the WORSE scaler of the two. The lever
        # that actually recovers detail without touching the contract is
        # IMAGE_GEN_RENDER_TO_CAP above — feed this upscale more real pixels
        # rather than deleting it.
        #
        # NOT on self._executor. LANCZOS is pure-PIL CPU work with no MLX
        # involvement, so it has no reason to occupy the single GPU-stream
        # thread — and a SECOND submit to that pool re-opens a shutdown race:
        # close() landing in the await between the two submits made this raise
        # "cannot schedule new futures after shutdown" INSIDE generate(),
        # returning an opaque 500 and discarding a multi-second render that had
        # already completed. A general worker thread (asyncio.to_thread) keeps
        # the event loop free, leaves the MLX thread available for the next
        # generation, and survives executor teardown so the finished render is
        # still returned.
        #
        # And it must not RAISE past this point either. The same reasoning as
        # the executor move above, one failure mode later: the render is
        # already committed, and LANCZOS to 1536 is a fresh ~7 MB allocation
        # that can fail (MemoryError) on this 16 GB host under pressure. Letting
        # that propagate turned a finished multi-second render into an opaque
        # 500 — and, because the route bills only on a return, an UNBILLED one.
        # Degrade instead: log loudly and return the render at its real size.
        # The route reports ``pil_image.size``, so the caller sees the honest
        # dimensions and is billed on them; the 1536 contract is missed on this
        # path only, which the orchestrator's E2E gate catches rather than a 500
        # the caller would answer by paying for a second render.
        if os.getenv("IMAGE_GEN_UPSCALE", "lanczos").lower() == "lanczos":
            target = int(os.getenv("IMAGE_GEN_UPSCALE_LONG_EDGE", "1536"))
            try:
                pil_image = await asyncio.to_thread(_upscale, pil_image, target)
            except Exception:  # noqa: BLE001 - post-processing must never discard a finished render
                UPSCALE_FAILURES.inc()
                # Mark the image itself so the ROUTE can tell the caller. The
                # image is the only object that already crosses this seam;
                # widening the return tuple would break the four stub
                # ``generate`` implementations in tests/ for no behavioural
                # gain. The route reads it with a ``getattr`` default, so the
                # attribute exists only on the degraded path.
                pil_image.upscale_failed = True  # ty: ignore[missing-attribute]
                logger.error(
                    "Upscale to %dpx failed after the render completed — returning the %dx%d render as-is.",
                    target,
                    *pil_image.size,
                    exc_info=True,
                )

        return pil_image, seed_used

    async def close(self) -> None:
        """Shut the pinned MLX executor down, releasing the loaded model.

        Nothing used to do this: the executor thread (and the multi-GB FLUX
        pipeline pinned to it) outlived the app's lifespan. Called from
        ``main.lifespan``'s shutdown half.

        Idempotent — a second call is a no-op, so a double shutdown (lifespan
        + an explicit teardown, or two TestClient exits) cannot raise. The
        blocking ``shutdown(wait=True)`` runs in a helper thread so an
        in-flight generation does not block the event loop while draining.

        BOUNDED, and cancels the backlog. ``main.lifespan`` awaits this inline,
        so an unbounded ``shutdown(wait=True)`` stalls uvicorn's graceful
        shutdown for the whole remaining render — 30s to several minutes for
        FLUX on M-series — PLUS every job still queued behind it. launchd
        SIGKILLs ~20s after SIGTERM, so that guaranteed a kill mid-drain.
        ``cancel_futures=True`` drops the queued backlog (those requests are
        already lost to the restart) and the wait is capped at
        ``IMAGE_GEN_SHUTDOWN_TIMEOUT_S``. On timeout we log and return: a slow
        MLX thread must not fail shutdown, and the process is exiting anyway.
        """
        if self._closed:
            return
        self._closed = True
        self.ready = False
        self._pipeline = None
        timeout = float(os.getenv("IMAGE_GEN_SHUTDOWN_TIMEOUT_S", str(DEFAULT_SHUTDOWN_TIMEOUT_S)))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            # Gate first: stop admitting (queued renders wake into
            # GateClosedError → 503), wait for the in-flight render within the
            # budget, then shut the lane executor down with cancel_futures.
            await self._gate.shutdown(timeout=timeout)
            # The gate's executor shutdown does not block; JOIN the worker in
            # whatever budget is left so the thread (and the pinned model) is
            # really gone when lifespan returns.
            await asyncio.wait_for(
                asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True),
                timeout=max(deadline - loop.time(), 0.0),
            )
        except TimeoutError:
            logger.warning(
                "FLUX executor did not drain within %.1fs — abandoning the wait and continuing shutdown. "
                "An in-flight generation is still running on the MLX thread; the model is NOT released "
                "until the process exits.",
                timeout,
            )
        else:
            logger.info("FLUX executor shut down; model released")


_singleton: FluxService | None = None


def get_flux_service() -> FluxService:
    global _singleton
    if _singleton is None:
        _singleton = FluxService()
    return _singleton
