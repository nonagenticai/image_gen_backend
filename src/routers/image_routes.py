"""image-gen routes. POST /api/image/generate is the canonical endpoint."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request
from w2v_common.auth import LICENSE_GATE, get_license_id_from_context
from w2v_common.auth.identity_context import get_attribution_subject_id_from_context
from w2v_common.metering import cost_event_key, schedule_cost_event
from w2v_common.tenant import InvalidTenantIdError, TenantError, parse_tenant_id

from models import ImageGenRequest, ImageGenResponse
from services.flux_service import ServiceUnavailableError
from services.metering import COST_SOURCE, FUNDING, SERVICE_KEY

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.76.2), mounted router-wide.
# Until now this service authenticated the licence token for tenant IDENTITY
# only: ``AuthMiddleware`` in ``src/main.py`` answers "who are you", and its own
# comment says so — "attribution for billing — NOT a deny gate". A tenant whose
# licence is ``expired`` or ``revoked`` still carries a syntactically valid
# token, so mflux burned GPU minutes for them and metered a cost event nobody
# could collect on. LICENSE_GATE closes that: none/expired/revoked -> 402
# license_<state>, an unestablishable verdict -> 503 license_unavailable +
# Retry-After, active/expiring_soon -> pass. Same mount as tts_service
# (``src/api/tts.py``), which is the pattern this copies.
#
# Router-wide is correct: the only route under /api/image is the tenant-scoped,
# GPU-consuming, cost-metered POST /generate.
#
# DELIBERATELY NOT GATED: the /health, /health/live, /health/ready, /status and
# /metrics endpoints, which are declared on ``app`` directly in src/main.py and
# never reach this router. A 402/503 on a probe would make launchd restart the
# unit, and an unlicensed tenant cannot be billed by killing our own service.
# w2v_common also refuses to enforce on those paths by name
# (``is_license_exempt_path``) — belt and braces, not a single point of failure.
#
# No WebSocket surface exists here, so LICENSE_GATE_WS is not mounted.
#
# ⛔ DORMANT ON MERGE: ``require_license`` is a no-op unless
# ``W2V_LICENSE_GATE_ENABLED`` is true, read per-call from os.environ via
# ``license_gate_enabled()``. It defaults to False and this PR adds the key to
# NO manifest and to NO env file, so merging changes no behaviour. Arming is a
# SEPARATE change and requires modality_gateway (the only caller,
# IMAGE_BACKEND_URL=http://image-gen.test:9204) to forward a real licence
# token first. Preserve this default.
router = APIRouter(prefix="/api/image", tags=["image-generation"], dependencies=list(LICENSE_GATE))


def _encode_png_b64(pil_image: PILImage.Image) -> str:
    """PNG-encode an image and base64 it. Pure CPU — never call inline.

    At the service's 1536px upscale target this is a zlib pass over ~7 MB plus
    a ~9 MB base64 pass: hundreds of milliseconds. Run via ``asyncio.to_thread``
    (below) so the event loop stays responsive.
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def require_request_tenant(request: Request) -> uuid.UUID:
    """Resolve the request's tenant to a real ``uuid.UUID``, or refuse with 401.

    This used to be ``request.state.tenant_id or DEFAULT_TENANT_ID``, where the
    default was the all-zeros UUID. That is a *valid* UUID, so it passed every
    type check and every dirty-row audit while attributing a commercial tenant's
    GPU minutes to nobody. w2v-common 0.67.0 deleted the sentinel outright
    (see ``w2v_common.tenant``); there is no fallback tenant by design.

    Refusing here — at the boundary, before the GPU does any work — is the fleet
    pattern (content_service, video_compiler_service): emitting an unattributed
    cost event is silently unbilled, which is worse than a loud failure. 401 is
    correct rather than 400: auth ran and resolved no identity.

    The nil check is by VALUE, not by spelling: ``uuid.UUID`` accepts the
    undashed, braced and ``urn:uuid:`` forms as the same nil value, so a caller
    that still sends the retired sentinel in ``X-Tenant-Id`` is rejected in every
    spelling. Mirrors ``content_service`` ``src/tenant.py``.

    The 401 detail deliberately does NOT echo the offending value —
    ``X-Tenant-Id`` carries a ``Bearer <license token>`` transport form, and
    ``w2v_common.tenant`` omits it from its own messages for the same reason.
    """
    try:
        tenant = parse_tenant_id(getattr(request.state, "tenant_id", None), source="request")
        if tenant.int == 0:
            raise InvalidTenantIdError("request: nil UUID is not a tenant — every cost event must carry a real tenant")
    except TenantError as exc:
        logger.warning("generate refused — %s", exc)
        raise HTTPException(status_code=401, detail="request has no valid tenant") from exc
    return tenant


@router.post(
    "/generate",
    response_model=ImageGenResponse,
    operation_id="generate_image",
    summary="Generate an image from a text prompt via FLUX (mflux on Apple Silicon)",
)
async def generate(request: Request, body: ImageGenRequest) -> ImageGenResponse:
    # Identity FIRST: an unattributable request must not consume a GPU minute,
    # and must not learn the service's readiness state either.
    tenant_id = require_request_tenant(request)
    # Identity of THIS physical render, minted once per request: the render is
    # synchronous (no queue, no redelivery), so the request IS the call and the
    # cost event's idempotency key derives from it (see ``_emit_cost_event``).
    render_id = str(uuid.uuid7())

    svc = getattr(request.app.state, "flux_service", None)
    if not svc or not svc.ready:
        raise HTTPException(status_code=503, detail="flux service not ready")

    started = time.perf_counter()
    try:
        pil_image, seed_used = await svc.generate(
            prompt=body.prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            guidance=body.guidance,
            seed=body.seed,
            negative_prompt=body.negative_prompt,
            # Admission lane ``render`` (#69): a render still queued when this
            # client disconnects is skipped (499) instead of burning the GPU.
            # GpuBusy (queue full / waited too long) → 503 + Retry-After via
            # install_gpu_busy_handler in main.py.
            request=request,
        )
    except ServiceUnavailableError as exc:
        # The pre-check above is inherently racy: on a launchd SIGTERM with
        # traffic in flight, close() lands between it and the submit. Answer
        # with the same clean 503 the pre-check gives rather than letting a
        # RuntimeError reach the catch-all handler as an opaque 500.
        logger.info("generate rejected — FLUX service unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="flux service not ready") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Report the dimensions actually produced — the service may clamp the
    # request down to its host memory budget (IMAGE_GEN_MAX_DIM). Bill on these
    # too: the request dims can be far larger than what the GPU actually
    # rendered, so metering off body.width/height overbills clamped requests.
    actual_width, actual_height = pil_image.size

    # Bill HERE — the render is committed — and BEFORE the encode below. The
    # GPU time is spent the moment ``svc.generate`` returns; the PNG/base64 pass
    # is pure-CPU post-processing that can still raise (~7 MB zlib + ~9 MB
    # base64 on a 16 GB host with a swap-out history). This used to sit after
    # the encode, so an encode failure reached ``global_exception_handler`` as
    # a 500 with NO cost event: a paid render, unbilled. That is the sibling of
    # the shutdown-window bug ``flux_service.generate`` documents at its
    # upscale — a completed render discarded by a step that ran after it — one
    # step later. A failed encode still 500s the caller (there is nothing to
    # return), but it is billed exactly once, on what was produced.
    _emit_cost_event(
        request,
        tenant_id=tenant_id,
        body=body,
        produced_width=actual_width,
        produced_height=actual_height,
        model_repo=svc.model_repo,
        latency_ms=int(elapsed_ms),
        render_id=render_id,
    )

    # Off-load the encode. NOT onto FluxService's executor: that pool is the
    # single MLX-pinned thread, and occupying it with zlib would serialise the
    # encode against the NEXT generation for no reason. A general worker thread
    # (asyncio.to_thread) frees the loop without touching the GPU stream.
    image_b64 = await asyncio.to_thread(_encode_png_b64, pil_image)

    return ImageGenResponse(
        image_base64=image_b64,
        width=actual_width,
        height=actual_height,
        seed=seed_used,
        steps=body.steps,
        inference_time_ms=elapsed_ms,
        model_repo=svc.model_repo,
        # Surface the service's soft upscale failure. Set by
        # ``FluxService.generate``'s except arm and ONLY there, so the getattr
        # default is the healthy path. Deliberately does NOT change the status
        # code: the render is committed, billed and returned: it is a degraded
        # success, not an error.
        upscale_failed=bool(getattr(pil_image, "upscale_failed", False)),
    )


def _emit_cost_event(
    request: Request,
    *,
    tenant_id: uuid.UUID,
    body: ImageGenRequest,
    produced_width: int,
    produced_height: int,
    model_repo: str | None,
    latency_ms: int,
    render_id: str,
) -> None:
    """Fire-and-forget metering emission. No-ops when METERING_REDIS_URL unset.

    G5 full attribution (every field stated, none left for billing to guess):
    ``app_id`` is the CALLING app from ``X-App-Id`` — image-gen is a platform
    backend reached through modality-gateway, which relays the caller's app, so
    hard-coding one app here would mislabel every other app's renders.
    ``funding`` is ``local_gpu`` (mflux on mini1). ``event_key`` derives from
    ``render_id``, one per physical render. Under ``METERING_REQUIRE_ATTRIBUTION``
    a missing licence/subject/app raises instead of emitting a partial event.

    ``tenant_id`` is already a real ``uuid.UUID``, resolved and validated by
    :func:`require_request_tenant` at the request boundary. There is no
    substitution path left here: a request that could not produce a tenant was
    refused with 401 before the image was generated, so this function is never
    reached without one. Do NOT reintroduce a fallback — an unattributed cost
    event is silently unbilled, which is worse than a loud failure.

    license/subject come by a DIFFERENT channel: AuthMiddleware sets
    ``request.state.tenant_id`` but leaves license/subject only in the identity
    ContextVars it binds alongside (empty strings there are acceptable per
    EMITTER_GUIDE.md).
    """
    metering_redis = getattr(request.app.state, "metering_redis", None)
    if metering_redis is None:
        return
    # Bill on the dimensions actually produced, not the raw request. The service
    # clamps the long edge to its host memory budget (IMAGE_GEN_MAX_LONG_EDGE),
    # so a 2048x2048 request that renders at ~896px must be billed for what the
    # GPU produced (~1 MP), not the 4 MP the caller nominally asked for.
    width = produced_width or 0
    height = produced_height or 0
    megapixels = max(1, (width * height) // 1_000_000)
    # AuthMiddleware sets request.state.tenant_id, but NOT license_id/subject_id —
    # those live only in the identity ContextVars it binds alongside
    # (w2v_common.auth.middleware). Reading request.state for them was therefore
    # always None, and the X-License-Id/X-Subject-Id fallback is empty east-west
    # (no in-mesh caller sends those headers), so every image cost event landed
    # with a NULL licence. The resolved values were in scope the whole time.
    license_id = get_license_id_from_context()
    subject_id = get_attribution_subject_id_from_context()
    # GC-safe fire-and-forget: w2v_common retains a strong ref to the spawned
    # task until the XADD lands (a bare asyncio.create_task could be collected
    # mid-flight and silently drop the cost event).
    schedule_cost_event(
        cast(Any, metering_redis),
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id=tenant_id,
        license_id=license_id,
        subject_id=subject_id,
        app_id=request.headers.get("X-App-Id", "").strip(),
        provider="mflux",
        model=model_repo or "mflux",
        input_tokens=megapixels,
        output_tokens=0,
        latency_ms=latency_ms,
        cost_usd=0.0,  # local model, no external quota
        stream=request.app.state.metering_stream,
        usage_extra={
            "unit": "megapixels",
            "width": width,
            "height": height,
            "steps": body.steps,
        },
        funding=FUNDING,
        event_key=cost_event_key(SERVICE_KEY, "render", render_id),
    )
