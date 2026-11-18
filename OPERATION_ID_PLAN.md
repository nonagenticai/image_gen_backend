# Operation ID Plan: image-gen-backend

Canonical registry of `operation_id` values for every route in
`image-gen-backend`. These IDs feed the org-wide n8n custom-node generator
and the `ultrareview` cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (nlp_service,
  worker_management, kag, analysis_service),
- **globally unique** — disambiguated against the rest of the fleet's verbs.

This service is a thin FLUX wrapper (Apple-silicon-only, launchd-deployed)
with a single business endpoint plus the three standard health probes.
Generic health verbs are suffixed (`health_check_*`) so they do not collide
with sibling services' `health_check` operation_id.

## Files to edit

- `src/main.py` (health surface)
- `src/routers/image_routes.py` (image generation)

## Mapping Tables

### `src/main.py` (tag: `health`)

| Method | Function       | operation_id          | Rationale                                              |
|--------|----------------|-----------------------|--------------------------------------------------------|
| GET    | `health`       | `health_check_basic`  | Status + version + model_loaded; service banner probe. |
| GET    | `health_ready` | `health_check_ready`  | Readiness — gated on FLUX model load.                  |
| GET    | `health_live`  | `health_check_live`   | Liveness — process-up only.                            |

### `src/routers/image_routes.py` (tag: `image-generation`)

| Method | Function   | operation_id     | Rationale                                                                  |
|--------|------------|------------------|----------------------------------------------------------------------------|
| POST   | `generate` | `generate_image` | Sole business endpoint. Verb chosen to namespace cleanly against future video/audio gen services in the fleet. |
