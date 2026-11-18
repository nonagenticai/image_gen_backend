# Image Gen Backend

Generates images from text prompts using FLUX (via mflux) on Apple Silicon, exposed to the cluster as a launchd host process. Port 9204.

| | |
|---|---|
| **Stack** | Python 3.14 · FastAPI · uvicorn · mflux (FLUX on MLX) |
| **Model** | FLUX.1-schnell (4-step distilled), 4-bit quantize on the live host; FLUX.1-dev optional |
| **Resolution** | long edge clamped to 896 px (aspect preserved on a 64-px grid), then LANCZOS-upscaled to a 1536 px long edge |
| **Runs on** | the host Mac — macOS launchd unit `com.nonagentic.image-gen-backend` (16 GB Apple Silicon) |
| **Endpoint** | `image-gen.test:9204` (Istio ServiceEntry + CoreDNS hostAlias) |
| **CI/CD** | none (heavy-GPU bare-metal rule) — deploy-tested on-device |

## What Image Gen Backend Does

Image Gen Backend is a dedicated text-to-image inference host. A caller POSTs a prompt (plus optional width, height, steps, guidance, seed, and negative prompt) to `/api/image/generate`, and the service runs it through a FLUX pipeline built on `mflux` — the MLX-native FLUX runtime for Apple Silicon. In production the sole caller is `ai_content_generation_service`'s `local_image` provider, which reaches this service across the mesh; the orchestrator's image-producing stages ultimately funnel through that provider. Unlike `analysis_service`, which runs diffusers in-process on the MPS backend, this service is a standalone MLX process purpose-built so its heavy unified-memory footprint never competes with an application pod.

Internally the model loads once. On startup the FastAPI lifespan builds the FLUX pipeline on a pinned single-thread executor and marks the service `ready`; until that finishes, `/health/ready` returns 503 and generation requests are rejected with 503 rather than queued against an unloaded model. All MLX work — both pipeline construction and every generation — runs on that one pinned thread because MLX's GPU stream is thread-bound: moving generation to another thread raises `RuntimeError: no Stream(gpu, 1)`. The single-thread executor also serialises generations, which is correct since there is exactly one GPU. Each request scales its resolution down so the long edge fits the host memory budget (default 896 px, aspect preserved so a 16:9 request does not collapse to a square), generates the image, then optionally upscales it with LANCZOS to a frame-fillable ~1536 px long edge on the way out.

The response carries the rendered image as a base64-encoded PNG plus metadata: the seed actually used, the produced dimensions (which may differ from the request after clamping), the step count, inference time in milliseconds, and the model repo. The consumer decodes the base64 PNG and uses it as a generated asset. Because the model is local there is no external API quota to bill, but every successful generation still emits a fire-and-forget cost event (cost 0 USD) to a Redis cost-bus stream so GPU usage is attributed to the real tenant for metering.

The service is not a Kubernetes Deployment. It runs as a macOS launchd unit on the host Mac and joins the mesh through an Istio ServiceEntry plus a CoreDNS hostAlias resolving `image-gen.test:9204`; in-cluster reachability additionally depends on a NetworkPolicy egress `/32` and a ServiceEntry/DestinationRule maintained on the platform side. Configuration is delivered by rendering a cluster ConfigMap into a host env file, while the gated FLUX weights' `HF_TOKEN` lives only in a host-local secret file that the ConfigMap render never touches.

## Features

- **FLUX text-to-image on Apple Silicon** — runs FLUX.1-schnell/dev through `mflux` (MLX-native) so image generation uses the Mac's unified-memory GPU instead of a CUDA box, keeping the workload off the K8s cluster
- **Load-once, serve-many** — the pipeline is built a single time at startup and reused for every request, so callers pay model-load latency only on cold start, not per generation
- **Thread-pinned MLX execution** — pipeline build and generation share one dedicated worker thread because MLX's GPU stream is thread-bound, avoiding the `no Stream(gpu, 1)` crash while keeping the event loop responsive
- **Aspect-preserving resolution clamp** — scales the long edge down to a host memory budget (default 896 px) proportionally, so non-square requests keep their shape instead of collapsing to a square. The short edge rounds to the NEAREST multiple of 64, not down: flooring it squashed every 16:9 order to 2.0:1 (fixed 2026-09-15)
- **LANCZOS upscale to a 1536 px long edge** — enlarges the output after generation so consumers get a frame-fillable asset without paying the memory cost of generating at that size. The 1536 is a contract, not a convenience: `w2v_orchestrator/scripts/e2e_validate.py:3866` hard-asserts it
- **Optional render-at-cap** (`IMAGE_GEN_RENDER_TO_CAP`, off by default) — renders a below-cap request AT the cap so that upscale interpolates real pixels rather than inventing them, without moving any output dimension
- **schnell / dev model switch** — defaults to the 4-step distilled FLUX.1-schnell for speed on a 16 GB box, with FLUX.1-dev (20–50 step, higher quality) selectable via env
- **Tunable quantization** — `IMAGE_GEN_QUANTIZE` trades memory for quality (the live 16 GB host runs 4-bit) so the model fits alongside other MLX services
- **Billing attribution + cost metering** — resolves the real tenant onto request state and emits a fire-and-forget cost event per generation to the Redis cost-bus, no-op when the cost bus is unset
- **Standard error envelope** — HTTP, validation, and unhandled errors all render the org-standard `{error: {code, message, status, request_id, timestamp}}` shape used across the fleet
- **Prometheus metrics + health probes** — exposes `/metrics` plus liveness/readiness/status endpoints for on-device monitoring and mesh health checks

## Authentication

The service mounts `w2v_common.auth.middleware.AuthMiddleware` when `AUTH_ENABLED` (default `true`). This is **attribution for billing, not a deny gate**: it resolves the real tenant from the forwarded license token onto `request.state` so cost-metering events are attributed to the correct tenant, rather than gating access. The transport is `X-Tenant-Id: Bearer <license_token>` — note the license token travels in `X-Tenant-Id`, not `Authorization` (which is reserved for caller-side Keycloak auth).

⚠️ `AUTH_TRUST_INCLUSTER_IDENTITY` is **`false` on the live the host Mac host** (verified live 2026-08-31 from the running process environment). This section previously said `=true` "must be set in the deploy env"; it is not set that way in production. A bare `X-Tenant-Id` is therefore **rejected with 401** — the `ai_content_generation_service` caller must forward a real license token.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected and resolved onto `request.state` for metering |
| 2 | Trusted in-cluster identity | bare `X-Tenant-Id` honoured only when `AUTH_TRUST_INCLUSTER_IDENTITY=true` — **not enabled on the live host** |

Public paths — `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics` — are always exempt. When no tenant resolves, metering falls back to the nil-UUID sentinel (`__unattributed__`, excluded from billing) rather than a phantom tenant.

> The OpenAPI-customisation comment in `main.py` claiming the service "does not currently mount the w2v_common AuthMiddleware" is stale — it does mount it (attribution only, as above).

## Quick Start

```bash
uv sync
IMAGE_GEN_QUANTIZE=4 ./start.sh                                        # binds 0.0.0.0:9204 (sources env + secret.env)
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .
```

`start.sh` runs `PYTHONPATH=src uv run uvicorn main:app --host 0.0.0.0 --port 9204`. First run downloads the gated FLUX.1-schnell weights (the HF account must accept the licence once; `HF_TOKEN` required — see `DEPLOY.md`).

## API Endpoints

### Image Generation `/api/image`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Generate an image from a text prompt via FLUX; returns base64 PNG + metadata |

Request body: `prompt` (required, 1–2000 chars), `width`/`height` (256–2048, multiple of 64, default 512), `steps` (1–8, default 3; 4 is schnell's design point and still accepted), `guidance` (0.0–20.0, default 0.0), `seed` (optional), `negative_prompt` (optional). Response: `image_base64` (PNG), actual `width`/`height`, `seed`, `steps`, `inference_time_ms`, `model_repo`.

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, version, and FLUX model-loaded state |
| GET | `/health/live` | Liveness probe — 200 if the process is up |
| GET | `/health/ready` | Readiness probe — 200 once the FLUX model is loaded, 503 otherwise |
| GET | `/metrics` | Prometheus metrics |

## Configuration

Configuration is via environment variables — read directly with `os.getenv` in `main.py`, `services/flux_service.py`, and `services/metering.py` (there is no `config.py`). The `HF_TOKEN` lives only in a host-local `secret.env`.

On-device values are rendered from the gitops ConfigMap `image-gen-backend-env` (namespace `external-services`, label `host-config-target: the host Mac`), defined in **`gitops-production/clusters/strato/external-services/image-gen-backend.yaml`**. `host-config-sync.sh` writes it to `~/.config/image-gen-backend/env` and kickstarts the launchd unit when it changes; `start.sh` sources that file.

> ⛔ `~/.config/image-gen-backend/env` is a **generated cache**. Editing it on the host Mac is silently reverted within ~5 minutes — change the gitops ConfigMap via a PR instead. The `Default` column below is the **code** default; where the live host differs, the cell says so.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_GEN_PORT` | `9204` | Listen port (read by `start.sh`) |
| `SERVICE_NAME` | `image-gen-backend` | Service name in health/logs |
| `VERSION` | `0.1.0` | Version string |
| `ENVIRONMENT` | `production` | Environment label |
| `DEBUG` | `false` | Enables error `details` in the envelope |
| `LOG_LEVEL` | `INFO` | Logging level |
| `AUTH_ENABLED` (alias `ENABLE_AUTH`) | `true` | Mount the billing-attribution auth middleware |
| `AUTH_TRUST_INCLUSTER_IDENTITY` | `false` (live the host Mac host) | Honour a bare `X-Tenant-Id` from a trusted peer; **off in prod**, so a bare header 401s |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `IMAGE_GEN_MODEL` | `schnell` | FLUX variant (`schnell` = 4-step distilled; `dev` = 20–50 step) |
| `IMAGE_GEN_QUANTIZE` | `8` (live host: `4`) | Model quantization bits; 4-bit on the 16 GB host |
| `IMAGE_GEN_MAX_LONG_EDGE` (alias `IMAGE_GEN_MAX_DIM`) | `896` (live the host Mac: **896**) | Long-edge clamp in px (aspect preserved). ⚠️ Corrected 2026-09-15: this row said "live the host Mac: 512 via `IMAGE_GEN_MAX_DIM`" and that has been wrong since 2026-09-12, when the ConfigMap was raised 512 → 896 and the `IMAGE_GEN_MAX_DIM` key was removed outright. Verified against both the ConfigMap and the host Mac's running process env |
| `IMAGE_GEN_UPSCALE` | `lanczos` | Output upscale mode (`none` to disable). ⛔ **Do not set `none` in production** — `w2v_orchestrator/scripts/e2e_validate.py:3866` hard-asserts `long_edge == 1536`, so disabling the upscale reds the panel-image E2E. See the note under `IMAGE_GEN_RENDER_TO_CAP` |
| `IMAGE_GEN_UPSCALE_LONG_EDGE` | `1536` | Target long edge for the LANCZOS upscale. Pinned by the same downstream assertion — changing it reds the E2E too |
| `IMAGE_GEN_RENDER_TO_CAP` | `0` (off) | When `1`, a request whose long edge is BELOW `IMAGE_GEN_MAX_LONG_EDGE` is scaled UP to the cap **before** rendering, so the LANCZOS upscale interpolates from real generated pixels instead of inventing them. Output dimensions are unchanged (still the 1536 upscale target), so no downstream contract moves. Costs GPU time — see the note below |
| `IMAGE_GEN_MLX_CACHE_LIMIT_MB` | `1024` | Ceiling on MLX's Metal allocator cache. `0` = no cache; negative = leave MLX's default (unbounded). Measured: unbounded retains a flat 2453.7 MB forever |
| `IMAGE_GEN_MLX_CACHE_TRIM` | `1` | Hand retained GPU buffers back after every render. Measured cost at 512×512/4 steps: +0.16 s (+0.54%). Set `0` if the cache ever earns its keep at a larger resolution |
| `METERING_REDIS_URL` | — (unset) | Redis cost-bus URL; emission disabled when unset |
| `METERING_STREAM` | `improvement-loop:costs` | Cost-bus stream name |
| `DEFAULT_TENANT_ID` | nil-UUID sentinel | Fail-safe tenant when none resolves (excluded from billing) |
| `HF_TOKEN` | — (host secret file) | Token for the gated `black-forest-labs/FLUX.1-schnell` HF repo |

## Architecture

```
src/
├── main.py                     # FastAPI app, lifespan (load FLUX), CORS + auth middleware,
│                               #   health endpoints, custom OpenAPI, error-envelope handlers
├── models.py                   # Pydantic models: HealthResponse, ImageGenRequest, ImageGenResponse
├── routers/
│   └── image_routes.py         # POST /api/image/generate + fire-and-forget cost emission
└── services/
    ├── flux_service.py         # mflux pipeline build + generate on the pinned MLX thread; clamp/upscale
    └── metering.py             # cost-metering identity + thin delegation to w2v_common.metering

start.sh                        # env + secret.env sourcing, port kill, uvicorn launch
service.yml                     # launchd service descriptor (host paths)
DEPLOY.md                       # full on-device deploy: plist, sync-env ConfigMap render, secret handling
```

## Testing

```bash
uv run pytest tests/ -v                     # all tests
uv run ruff check .                         # lint
uv run ruff format --check .                # format check
uv run ty check                             # type check
```

Tests are lightweight and Darwin-independent (mflux is `sys_platform`-gated, so the model never loads under CI/Linux): `test_health.py`, `test_auth_middleware.py`, `test_error_envelope.py`, `test_openapi.py`, `test_metering.py`, `test_metering_attribution.py`. No BDD suite and no coverage gate is configured in `pyproject.toml`.

## Deployment

This service is not containerised: it runs as a launchd unit on a Mac host with an
on-device venv and `uvicorn` (`./start.sh`). An MLX/mflux container would be built
but never pulled by a consumer, so there is no image and no rollout pipeline.

Update the venv, restart the unit, then check `/health` and the unit's logs. The
operator's own launchd plist and env file are provisioned separately and are not
part of this repository (secrets live in a host-only file outside the tree).

## Docker

There is **no Dockerfile**. The service runs as a launchd unit executing `uvicorn` directly against the on-device venv, not a container — an MLX/mflux image would be built but never pulled by any consumer. Local development uses `uv sync` + `./start.sh` (see Quick Start).

## Dependencies

- **ai_content_generation_service** — sole production caller; its `local_image` provider POSTs `/api/image/generate` here (across the mesh at `image-gen.test:9204`)
- **w2v_orchestrator** — drives image-producing stages that funnel through the `local_image` provider
- **mflux** — MLX-native FLUX runtime (Apple-silicon only, `sys_platform`-gated)
- **MLX** — Apple's array/GPU framework backing mflux (Apple-silicon only)
- **FLUX.1-schnell / FLUX.1-dev weights** — gated Hugging Face repos (`HF_TOKEN` required)
- **transformers** — model/tokenizer support for mflux (Apple-silicon only)
- **Redis** — cost-bus stream target for metering (via Tailnet-reachable cluster Redis; optional)
- **w2v-common** (`>=0.25.0`) — shared auth middleware + cost-metering emitter
- **FastAPI** / **uvicorn** — web framework and ASGI server
- **Pydantic** / **pydantic-settings** — request/response validation
- **Pillow** — PNG encoding and LANCZOS upscale
- **prometheus-fastapi-instrumentator** — `/metrics` exposition
- **httpx** / **structlog** — HTTP client and structured logging

## Known Issues

- **MLX GPU stream is thread-bound** — pipeline build and generation must run on the same pinned thread; moving generation off it raises `RuntimeError: no Stream(gpu, 1)` and can abort uvicorn. The single-thread executor enforces this and serialises generations (one GPU).
- **16 GB unified-memory ceiling** — the host Mac co-hosts other MLX services; FLUX saturates the unified-memory pool near ~1 MP (≈1024²). The 4-bit quantize + 896 px long-edge clamp keep it under the saturation point. Requesting larger just gets clamped; raising the cap risks OOM.
- **Cold-start "degraded" window** — `/health` reports `degraded` and `/health/ready` returns 503 until FLUX finishes loading (first run also downloads ~12 GB of gated weights). This is expected, not a fault.
- **Gated weights need `HF_TOKEN`** — FLUX.1-schnell is a gated HF repo; the token must be in the host-only `secret.env`, never the ConfigMap (plaintext-in-git). A missing/expired token leaves the service stuck `degraded`.
- **Stale in-code comment** — `main.py`'s custom-OpenAPI comment says the service doesn't mount `AuthMiddleware`; it does (attribution-only). Ignore the comment; trust the code path and CLAUDE.md.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. It runs with no licence switch set; unset means off.
