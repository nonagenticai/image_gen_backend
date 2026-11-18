"""Guardrail tests for the auth surface of image_gen_backend.

This service is the Apple-silicon-only mflux launchd unit; it does NOT
currently mount ``w2v_common.auth.middleware.AuthMiddleware`` — network
ingress is restricted by the launchd unit's bind address and by upstream
istio ServiceEntry policy (see CLAUDE.md "No CI for heavy-GPU bare-metal
services"). If/when AuthMiddleware is added the analysis_service-style
401-envelope assertion will live here.

For now the canonical guardrail is the **stub** assertion that the
service's own exception handlers emit the org-standard
``{error: {code, message, status, request_id, timestamp}}`` envelope when a
401 HTTPException flows through them. That covers the case where an
ingress/proxy layer raises a 401 to clients, and ensures the envelope shape
is locked-in before the middleware is ever wired.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from main import (
    global_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)

UUID7_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _assert_envelope(body: dict[str, Any], *, expected_status: int) -> None:
    assert set(body.keys()) == {"error"}, body
    err = body["error"]
    assert isinstance(err.get("code"), str) and err["code"], err
    assert isinstance(err.get("message"), str) and err["message"], err
    assert err.get("status") == expected_status, err
    assert UUID7_RE.match(err.get("request_id", "")), err
    assert TIMESTAMP_RE.match(err.get("timestamp", "")), err


def _build_minimal_app() -> FastAPI:
    """Standalone app wiring the service's three exception handlers + a 401 route."""
    # Pass handlers via the FastAPI constructor; ty narrows
    # ``app.add_exception_handler``'s ``handler`` param to Starlette's
    # ``ExceptionHandler`` union and rejects explicitly-typed handlers.
    app = FastAPI(
        exception_handlers={
            StarletteHTTPException: http_exception_handler,
            RequestValidationError: validation_exception_handler,
            Exception: global_exception_handler,
        },
    )

    @app.get("/api/protected/echo")
    async def _echo() -> None:
        # Simulates the future AuthMiddleware -> raises 401 on missing/invalid token.
        raise HTTPException(status_code=401, detail="Invalid or expired license token")

    return app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=_build_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_401_uses_standard_envelope(client: httpx.AsyncClient) -> None:
    """An HTTPException(401) must surface as the standard error envelope."""
    resp = await client.get("/api/protected/echo")
    assert resp.status_code == 401
    body = resp.json()
    _assert_envelope(body, expected_status=401)
    assert body["error"]["code"] == "HTTP_401"
    assert "Invalid or expired license token" in body["error"]["message"]


def test_main_module_exposes_required_handlers() -> None:
    """``src/main.py`` must export the three handlers wired on the asgi app.

    This is the regression-against-false-merge assertion: if the handlers are
    renamed or removed, this import + symbol check fails loudly.
    """
    from main import (  # noqa: F401  — import-check only
        global_exception_handler as gh,
    )
    from main import (
        http_exception_handler as hh,
    )
    from main import (
        validation_exception_handler as vh,
    )

    assert callable(hh)
    assert callable(vh)
    assert callable(gh)


def test_main_module_registers_three_exception_handlers() -> None:
    """The live app must have handlers registered for the three standard exceptions."""
    from main import app

    handled = set(app.exception_handlers.keys())
    assert StarletteHTTPException in handled, "Missing StarletteHTTPException handler"
    assert RequestValidationError in handled, "Missing RequestValidationError handler"
    assert Exception in handled, "Missing catch-all Exception handler"
