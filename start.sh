#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${IMAGE_GEN_PORT:-9204}"
ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/image-gen-backend/env"
if [ ! -f "$ENV_FILE" ] && [ -f /etc/image-gen-backend/env ]; then ENV_FILE=/etc/image-gen-backend/env; fi
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
# Host-only secrets (e.g. HF_TOKEN for the gated FLUX repo) live in a separate
# file that ConfigMap-render tooling (sync-env.sh) never touches. Sourced AFTER
# the env file so secrets layer on top and survive a sync-env re-render. This
# file is created once per host (chmod 600) and is NOT in gitops — per-service
# secret. See DEPLOY.md.
SECRET_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/image-gen-backend/secret.env"
if [ -f "$SECRET_FILE" ]; then set -a; . "$SECRET_FILE"; set +a; fi
lsof -ti :"${PORT}" | xargs kill -9 2>/dev/null || true
sleep 1
mkdir -p logs
export SSL_CERT_FILE="${SSL_CERT_FILE:-/opt/homebrew/etc/ca-certificates/cert.pem}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/opt/homebrew/etc/ca-certificates/cert.pem}"
echo "Starting image-gen-backend on port ${PORT}..."
PYTHONPATH=src exec uv run uvicorn main:app --host 0.0.0.0 --port "${PORT}"
