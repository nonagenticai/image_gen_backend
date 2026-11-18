"""Guardrail tests for the error-envelope shape.

Every error response produced by an image_gen_backend-owned handler must use
the standard envelope:

    {
        "error": {
            "code": "...",
            "message": "...",
            "status": <int>,
            "request_id": "<uuid7>",
            "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
        }
    }

The ``details`` key is optional and only present when ``DEBUG=true``.

This file asserts the envelope for the three handlers registered in
``src/main.py``: ``StarletteHTTPException``, ``RequestValidationError``, and
the catch-all ``Exception``.
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


def _build_minimal_app_with_handlers() -> FastAPI:
    """Mount only the three main-module exception handlers on a fresh FastAPI app."""
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

    @app.get("/boom")
    async def _boom() -> None:
        raise HTTPException(status_code=418, detail="i am a teapot")

    return app


@pytest.fixture
async def client():
    app = _build_minimal_app_with_handlers()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_handled_http_exception_uses_envelope(client: httpx.AsyncClient) -> None:
    """A raised HTTPException flows through the envelope handler."""
    resp = await client.get("/boom")
    assert resp.status_code == 418
    body = resp.json()
    _assert_envelope(body, expected_status=418)
    assert body["error"]["message"] == "i am a teapot"
    assert body["error"]["code"] == "HTTP_418"


async def test_404_uses_envelope(client: httpx.AsyncClient) -> None:
    """A request to an unknown path returns 404 in the envelope shape."""
    resp = await client.get("/this-path-does-not-exist")
    assert resp.status_code == 404
    _assert_envelope(resp.json(), expected_status=404)
