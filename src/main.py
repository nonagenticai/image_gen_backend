"""image-gen-backend — FastAPI entrypoint. FLUX image generation on Apple Silicon via mflux."""

from __future__ import annotations

import logging
import os
import resource
import sys
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException
from w2v_common.admission import DisconnectProbeMiddleware, install_gpu_busy_handler

from models import HealthResponse
from routers import image_routes
from services.flux_service import get_flux_service
from services.metering import DEFAULT_STREAM, close_metering_redis, init_metering_redis

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_NAME = os.getenv("SERVICE_NAME", "image-gen-backend")
VERSION = os.getenv("VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}


def _rss_mb() -> int:
    """Peak resident-set size of this process, in MiB. No new dependency.

    PEAK, not current: ``ru_maxrss`` is a high-water mark and never falls back
    after a render. That is the property wanted here — it corroborates ``warm``.
    A process reporting ``model_loaded: true`` at ~283 MB has demonstrably never
    paged the FLUX weights in (a 4-bit schnell render peaks around 9–12 GB on
    mini1), which is the exact lie this health check was telling.

    ⚠️ UNITS DIFFER BY PLATFORM and the stdlib does not normalise them: macOS
    (and the BSDs) report ``ru_maxrss`` in BYTES, Linux reports it in KILOBYTES.
    This service only ever runs on macOS (mini1, Apple-silicon-only via mflux),
    but the tests run wherever they are run, so branch EXPLICITLY rather than
    assume either — assuming bytes on Linux under-reports by 1024x, and
    assuming kilobytes on macOS over-reports by the same factor.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin" or sys.platform.startswith(("freebsd", "openbsd", "netbsd")):
        return int(peak // (1024 * 1024))  # bytes -> MiB
    return int(peak // 1024)  # kilobytes -> MiB


# ============================================================================
# Public-path allow-list (used by custom_openapi to strip security on these)
# ============================================================================

_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/healthz",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
        "/metrics",
    }
)
_PUBLIC_PREFIXES: tuple[str, ...] = ("/docs", "/redoc", "/health/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s (%s)", SERVICE_NAME, VERSION, ENVIRONMENT)
    svc = get_flux_service()
    try:
        await svc.initialize()
        app.state.flux_service = svc
        logger.info("FLUX model loaded: %s", svc.model_repo)
    except Exception:
        logger.exception("FLUX init failed — service starts in degraded mode")
        app.state.flux_service = svc
    app.state.metering_redis = await init_metering_redis(os.getenv("METERING_REDIS_URL", ""))
    app.state.metering_stream = DEFAULT_STREAM
    # Do NOT log an unqualified "ready" here: this line is reached on the failure
    # branch above too, so it used to assert readiness the service did not have —
    # the same false claim /health was making, in the log a human reads first.
    if svc.ready:
        logger.info("%s ready", SERVICE_NAME)
    else:
        logger.error(
            "%s started DEGRADED — FLUX model not loaded; /health will answer 503 "
            "and image generation will fail until this is resolved",
            SERVICE_NAME,
        )
    yield
    logger.info("Shutting down %s", SERVICE_NAME)
    # Defensive teardown: each resource is closed in its own try/except so a
    # failure closing one cannot leak the other. Both closers are idempotent,
    # so running this twice (e.g. nested lifespans in tests) is safe.
    try:
        flux = getattr(app.state, "flux_service", None)
        if flux is not None:
            await flux.close()
    except Exception:
        logger.exception("Error shutting down the FLUX service")
    try:
        await close_metering_redis(getattr(app.state, "metering_redis", None))
    except Exception:
        logger.exception("Error closing the metering Redis client")


app = FastAPI(
    title="image-gen-backend",
    description=(
        "FLUX image generation on Apple Silicon via mflux. "
        "Mirrors vlm_backend's deployment shape "
        "(launchd on mini1, ServiceEntry image-gen.test:9204)."
    ),
    version=VERSION,
    lifespan=lifespan,
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-id generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a future route
    # that forgets the kwarg degrades to `<tag>_<func>` instead of FastAPI's
    # default `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: f"{route.tags[0] if route.tags else 'default'}_{route.name}",
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, and basic service-status probes."},
        {
            "name": "image-generation",
            "description": "FLUX text-to-image generation on Apple Silicon (mflux).",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# License-token / trusted-network auth so cost-metering events are attributed
# to the real tenant (resolved onto request.state) rather than the raw header.
# This is attribution for billing — NOT a deny gate: the shared middleware
# carries no byo_mode logic, /health* + /docs + /metrics stay public, and
# in-cluster callers are identified via their forwarded X-Tenant-Id.
# NOTE: AUTH_TRUST_INCLUSTER_IDENTITY is false on the live mini1 host (verified
# 2026-08-31), so a bare X-Tenant-Id DOES 401 here -- callers must forward a
# real license token as "X-Tenant-Id: Bearer <token>".
_AUTH_ENABLED = os.getenv("AUTH_ENABLED", os.getenv("ENABLE_AUTH", "true")).lower() in {
    "1",
    "true",
    "yes",
}
if _AUTH_ENABLED:
    from w2v_common.auth.middleware import AuthMiddleware

    app.add_middleware(AuthMiddleware)
else:
    logger.warning("image-gen-backend running without auth (development mode)")

Instrumentator().instrument(app).expose(app)
# OUTERMOST middleware — keep this the LAST add_middleware (Instrumentator's
# .instrument() above adds one too). Behind AuthMiddleware (a
# BaseHTTPMiddleware) Request.is_disconnected() never turns True, so the
# admission gate could not skip a queued render whose client hung up. The
# probe watches the server's receive channel directly (w2v_common >=0.99.0).
app.add_middleware(DisconnectProbeMiddleware)

app.include_router(image_routes.router)
# GPU admission (#69): GpuBusy → 503 + Retry-After + {"error":"gpu_busy",...},
# ClientGone → bodyless 499. Registered per exception class, so these win over
# the catch-all ``Exception`` handler below (Starlette resolves by MRO).
install_gpu_busy_handler(app)


@app.get(
    "/health",
    response_model=HealthResponse,
    operation_id="health_check_basic",
    tags=["health"],
    summary="Service status, version, and FLUX model-loaded state",
    responses={503: {"model": HealthResponse, "description": "FLUX model not loaded"}},
)
async def health(request: Request, response: Response) -> HealthResponse:
    """200 while the FLUX model is loaded, 503 with the same body when it is not.

    ⚠️ The status code is load-bearing, and it is what makes this endpoint honest.
    THREE supervisors probe THIS path, and every one of them was blind:

      * `service.yml`  -> `healthcheck: curl -f http://localhost:9204/health`
      * gitops-production `scripts/host-configs/mini1/repo-services.txt`
        -> host-config-sync's `health_probe`, `curl -fsS`
      * gitops-production `clusters/strato/.../launchd-liveness-cronjob.yaml`
        -> `probe image-gen http://image-gen.test:9204/health`

    `curl -f` fails only on HTTP >= 400, so while this returned 200
    unconditionally a `degraded` service was indistinguishable from a healthy
    one. That state is reachable and silent: `flux_service.initialize` swallows a
    missing/broken mflux with `except ImportError` (leaving `ready=False`), and
    `lifespan` catches ANY init failure, still assigns `app.state.flux_service`,
    and carries on. Meanwhile every `POST /api/image/generate` answers
    `503 flux service not ready` forever. `start.sh` runs `uv run`, which
    re-syncs uv.lock on every start, so a bad lock is exactly how that arrives.

    Fixing THIS endpoint rather than repointing `service.yml` at `/health/ready`
    is deliberate: two of the three probes above live in gitops-production and
    cannot be changed from this repo, so a `service.yml` edit alone would leave
    the live probes blind. One honest endpoint fixes all three at once.

    The BODY is deliberately unchanged — `response_model` still serialises the
    full `HealthResponse` on both paths, because the liveness CronJob greps it
    for `"status":"(healthy|ok)"`. Pure liveness ("is the process up at all?")
    has its own probe and is unaffected: see `/health/live`.
    """
    svc = getattr(request.app.state, "flux_service", None)
    if not (svc and svc.ready):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if (svc and svc.ready) else "degraded",
        service=SERVICE_NAME,
        version=VERSION,
        environment=ENVIRONMENT,
        model_loaded=bool(svc and svc.ready),
        model_repo=(svc.model_repo if svc else None),
        timestamp=time.time(),
        pipeline_constructed=bool(svc and svc.ready),
        warm=bool(svc and svc.warm),
        rss_mb=_rss_mb(),
    )


@app.get(
    "/health/ready",
    operation_id="health_check_ready",
    tags=["health"],
    summary="Readiness probe — 200 once the FLUX model is loaded, 503 otherwise",
)
async def health_ready(request: Request):
    svc = getattr(request.app.state, "flux_service", None)
    if svc and svc.ready:
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not ready"}, status_code=503)


@app.get(
    "/health/live",
    operation_id="health_check_live",
    tags=["health"],
    summary="Liveness probe — 200 if the process is up",
)
async def health_live():
    return JSONResponse({"status": "alive"})


# ============================================================================
# OpenAPI customisation — BearerAuth scheme + TenantIdHeader parameter
# ============================================================================
#
# Mirrors the analysis_service / kag / worker_management / nlp_service pattern:
# every protected operation advertises ``security: [{BearerAuth: []}]`` and a
# reusable ``TenantIdHeader`` parameter so Swagger UI exposes the license-token
# transport correctly.
#
# ⚠️ CORRECTED 2026-08-26. This comment used to say image_gen_backend "does not
# currently mount the w2v_common AuthMiddleware". That is FALSE and was worth
# real money: it reads as "auth is not enforced here, so auth changes do not
# apply to this service", which is exactly the reasoning that left this repo
# three w2v_common releases behind on a SECURITY fix. `AUTH_ENABLED` /
# `ENABLE_AUTH` default to "true" (see ~line 129), so `AuthMiddleware` IS
# mounted on every normal start — verified on the live mini1 launchd unit, whose
# plist sets neither variable.
#
# The block below is spec-side only, so downstream client generators emit
# consistent code; it does not gate anything at runtime.
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "License",
        "description": (
            "License token. Transport: `X-Tenant-Id: Bearer <token>` "
            "(not the Authorization header — that is reserved for caller-side auth)."
        ),
    }
    components.setdefault("parameters", {})["TenantIdHeader"] = {
        "name": "X-Tenant-Id",
        "in": "header",
        "required": False,
        "description": "License token transport: `Bearer <token>`.",
        "schema": {"type": "string"},
    }

    for path, item in schema.get("paths", {}).items():
        clean = path.rstrip("/") or "/"
        is_public = clean in _PUBLIC_PATHS or any(clean.startswith(p) for p in _PUBLIC_PREFIXES)
        for operation in item.values():
            if not isinstance(operation, dict):
                continue
            if is_public:
                operation.pop("security", None)
            else:
                operation.setdefault("security", [{"BearerAuth": []}])
            params = operation.setdefault("parameters", [])
            if not any(p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header" for p in params):
                params.append({"$ref": "#/components/parameters/TenantIdHeader"})

    app.openapi_schema = schema
    return app.openapi_schema


object.__setattr__(app, "openapi", custom_openapi)


# ============================================================================
# Error Handlers — standardized {error: {code, message, status, ...}} envelope
# ============================================================================


def _request_id_from(request: Request) -> str:
    rid = getattr(request.state, "request_id", "")
    return rid or str(_uuid.uuid7())


def _error_envelope(
    code: str,
    message: str,
    http_status: int,
    *,
    request_id: str,
    details: str | None = None,
) -> dict[str, Any]:
    """Build the standardized error envelope.

    Shape matches the org standard ``{error: {code, message, status, …}}`` used by
    nlp_service / worker_management / analysis_service. ``details`` is only present
    when populated (Pydantic-side it's optional).
    """
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "status": http_status,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def _http_exception_to_response(exc: StarletteHTTPException, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(f"HTTP_{exc.status_code}", str(exc.detail), exc.status_code, request_id=request_id),
        headers=exc.headers,
    )


def _find_http_exception(exc: BaseException) -> StarletteHTTPException | None:
    """Unwrap nested ExceptionGroups to find the first HTTPException."""
    if isinstance(exc, StarletteHTTPException):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = _find_http_exception(sub)
            if found is not None:
                return found
    return None


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render HTTPException (incl. FastAPI's 404 for unknown routes) into the envelope."""
    return _http_exception_to_response(exc, _request_id_from(request))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render pydantic request-validation errors into the envelope (HTTP 422)."""
    return JSONResponse(
        status_code=422,
        content=_error_envelope(
            "VALIDATION_ERROR",
            "Request validation failed",
            422,
            request_id=_request_id_from(request),
            details=str(exc.errors()) if DEBUG else None,
        ),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler returning the standardized envelope."""
    request_id = _request_id_from(request)
    http_exc = _find_http_exception(exc)
    if http_exc is not None:
        return _http_exception_to_response(http_exc, request_id)
    logger.error(
        "Unhandled exception in request %s: %s",
        request_id,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=_error_envelope(
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred",
            500,
            request_id=request_id,
            details=str(exc) if DEBUG else None,
        ),
    )
