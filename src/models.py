"""Pydantic models for image-gen-backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    # ⚠️ `model_loaded` OVERSTATES readiness and is kept that way ON PURPOSE.
    # It mirrors `FluxService.ready`, which is set the instant `_build_pipeline`
    # returns — and that only CONSTRUCTS `Flux1(...)`. mflux lazy-loads the
    # weights on first render, so a 283 MB cold process that has never generated
    # anything reports `model_loaded: true` (measured live on mini1 2026-09-11,
    # pid 45824). It is NOT corrected here because
    # `launchd-liveness-cronjob.yaml` and `service.yml`'s `curl -f` both consume
    # this field's current meaning; narrowing it would red every healthy probe.
    # The three fields below are the honest answer, added alongside it.
    model_loaded: bool
    model_repo: str | None = None
    timestamp: float
    # Pipeline OBJECT exists (today's `svc.ready`) — construction, not weights.
    pipeline_constructed: bool = False
    # A render has actually SUCCEEDED in this process, so the weights are
    # resident. This is the field that distinguishes a cold process from a hot
    # one; `pipeline_constructed` cannot.
    warm: bool = False
    # Peak RSS in MiB. Corroborates `warm`: a process that has never touched the
    # weights cannot have peaked above a few hundred MB.
    rss_mb: int = 0


class ImageGenRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt for image generation",
    )
    # 512 here is only this REQUEST FIELD's default, not the service's cap.
    # The cap applies to the LONG EDGE (aspect preserved) in
    # flux_service.generate.
    #
    # ⛔ DO NOT WRITE THE LIVE CAP VALUE HERE. This comment has now gone stale
    # THREE TIMES in a single day (2026-09-15): it said 512, was corrected to
    # 896, and 896 was reverted to 512 hours later. Each version presented
    # itself as verified ground truth — one of them even warned that "guessing
    # has misled twice already" — and each was wrong within hours, because the
    # value lives somewhere this file cannot see and changes without touching
    # this repo. A number here is not documentation, it is a trap with a date
    # on it.
    #
    # WHERE THE VALUE ACTUALLY LIVES (read it, every time):
    #   ConfigMap image-gen-backend-env, ns external-services — SOURCE OF TRUTH
    #     gitops-production/clusters/strato/external-services/image-gen-backend.yaml
    #   key IMAGE_GEN_MAX_LONG_EDGE (IMAGE_GEN_MAX_DIM is a back-compat alias,
    #   consulted ONLY when the former is unset).
    #
    # HOW TO READ IT, in the two places that can disagree:
    #   kubectl get cm image-gen-backend-env -n external-services -o yaml
    #   ps eww <uvicorn pid> on mini1        # the value actually in effect
    # host-config-sync.sh renders that ConfigMap to ~/.config/<service>/env on
    # mini1, which start.sh sources, so the two disagree for up to ~5 min after
    # a change AND until the unit restarts. That device file is a GENERATED
    # CACHE — editing it there is silently reverted; change the ConfigMap.
    #
    # The cap only ever scales DOWN (_snap_to_long_edge / _aspect_clamp), so an
    # order larger than the cap is reshaped, never honoured — which is why
    # raising a caller's requested size without first raising the cap cannot
    # buy detail. Raising the cap is a COST decision (latency + peak footprint
    # on a 16 GB box), not a code decision: benchmark before proposing it.
    width: int = Field(512, ge=256, le=2048, multiple_of=64)
    height: int = Field(512, ge=256, le=2048, multiple_of=64)
    # Capped at 8 (#69). ~28.6 s per warm 512px/4-step schnell render on
    # mini1; the old ``le=50`` let one caller hold the single GPU lane for ~6
    # minutes and starve every request queued behind it. schnell is a 4-step
    # distilled model, so 8 is already twice its design point. >8 → 422.
    #
    # Default is 3, not 4 (owner decision 2026-10-22). Benchmark on mini1
    # (schnell @ quantize=4, 512x320, n=3, warm): 2 steps 11.25 s, 3 steps
    # 14.67 s, 4 steps 19.32 s (~4.3 s/step + ~2.4 s fixed). 3 is visually
    # ~identical to 4; 2 is softer on faces and UI text. Callers that want
    # schnell's 4-step design point still pass ``steps=4`` explicitly.
    steps: int = Field(
        3,
        ge=1,
        le=8,
        description=(
            "Number of inference steps (1–8, default 3). FLUX.1-schnell is distilled "
            "for 4; 3 is visually ~identical at ~24% less latency (benchmark 2026-10-22)."
        ),
    )
    guidance: float = Field(
        0.0,
        ge=0.0,
        le=20.0,
        description=(
            "Classifier-free-guidance scale. FLUX.1-schnell is distilled and "
            "ignores this (use 0); FLUX.1-dev uses 3.5–7.0."
        ),
    )
    # Bounded to int32, matching the service's own generated default
    # (``random.randint(0, 2**31 - 1)`` in flux_service.generate). An
    # out-of-range seed reaches ``pipeline.generate_image`` and raises inside
    # the MLX executor — not a ServiceUnavailableError, so it bypasses the 503
    # mapping in routers/image_routes.py and surfaces as an opaque 500.
    seed: int | None = Field(
        None,
        ge=0,
        le=2**31 - 1,
        description="Optional random seed for reproducibility.",
    )
    negative_prompt: str | None = None


class ImageGenResponse(BaseModel):
    image_base64: str = Field(..., description="PNG image encoded as base64.")
    width: int
    height: int
    seed: int
    steps: int
    inference_time_ms: float
    model_repo: str
    # The ONLY caller-visible signal that this 200 is degraded. ``flux_service``
    # treats a failed post-render upscale as a soft failure — it logs and
    # returns the render at its real size rather than discarding a committed
    # multi-second render (see the call-site comment there) — so before this
    # field the response was indistinguishable from a healthy one apart from
    # width/height the caller has no target to compare against.
    #
    # Optional with a False default, so it is additive: existing clients that
    # ignore it are unaffected, and the status code stays 200 by design. There
    # is no metadata/warnings dict on this model to reuse, and adding one would
    # be a ``dict[str, Any]`` — which ``tests/test_any_ratchet.py`` caps at 3
    # and is meant to ratchet DOWN, never up. A single typed bool is the
    # cheaper and more honest shape.
    upscale_failed: bool = Field(
        False,
        description=(
            "True when the post-render upscale failed and the image is returned at its "
            "un-upscaled size. The render itself succeeded and is billed; only the "
            "long-edge target was missed."
        ),
    )
