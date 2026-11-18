"""Identity-attribution tests for the mflux cost-metering caller.

These pin Part 3 of the BYO-LLM Phase 4 plan: ``_emit_cost_event`` must
attribute the identity resolved by AuthMiddleware, never the raw
``X-Tenant-Id`` header (which may carry a ``Bearer <token>`` transport form).
They also assert the AuthMiddleware is actually mounted on the app.

⚠️ The tenant and the licence/subject arrive by DIFFERENT channels, and getting
that wrong is what made every image cost event unattributed in prod.
AuthMiddleware sets ``request.state.tenant_id`` but **never**
``request.state.license_id`` / ``request.state.subject_id`` — those exist only
in the identity ContextVars it binds via ``set_identity_context``
(verified against the installed w2v_common 0.31.1 and the 0.44.0 source).

These tests used to fabricate ``request.state.license_id`` and assert it came
back, which passed forever while prod emitted 42 of 42 image cost events with a
NULL licence. They now bind the real ContextVar instead — the precondition the
test name always claimed.

⚠️ Since w2v-common 0.67.0 there is **no fallback tenant**. The nil-UUID
sentinel (``UNATTRIBUTED_TENANT_ID``) and ``DEFAULT_TENANT_ID`` are deleted, and
a request that cannot produce a real tenant is refused at the boundary by
``require_request_tenant`` with HTTP 401 — before the GPU renders anything. The
old ``test_no_tenant_defaults_to_nil_sentinel`` therefore pinned the DEFECT as
correct and is replaced by the 401 tests below.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import w2v_common.metering.cost_bus as cost_bus
from fastapi import HTTPException
from w2v_common.auth import clear_identity_context, set_identity_context

import routers.image_routes as image_routes
from models import ImageGenRequest

TENANT = uuid.UUID("3f2a1b4c-5d6e-4f70-8912-a3b4c5d6e7f8")
BEARER = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.super-secret-license-token"


@pytest.fixture(autouse=True)
def _clean_identity_context():
    """Identity ContextVars are process-global; never let one test leak."""
    clear_identity_context()
    yield
    clear_identity_context()


def _fake_request(*, state: dict, headers: dict) -> SimpleNamespace:
    redis_mock = AsyncMock()
    redis_mock.xadd = AsyncMock(return_value=b"1-0")
    app_state = SimpleNamespace(metering_redis=redis_mock, metering_stream="billing:costs")
    return SimpleNamespace(
        state=SimpleNamespace(**state),
        headers=headers,
        app=SimpleNamespace(state=app_state),
    )


async def _drain_background() -> None:
    # ``schedule_cost_event`` parks its in-flight task in w2v_common's module-
    # level set; drain that so the XADD lands before we assert.
    if cost_bus._background_tasks:
        await asyncio.gather(*list(cost_bus._background_tasks))


async def test_attributes_resolved_identity() -> None:
    # tenant comes from request.state; licence + subject come from the identity
    # ContextVars — exactly as AuthMiddleware populates them.
    set_identity_context(tenant_id=str(TENANT), license_id="lic-9", subject_id="sub-9")
    req = _fake_request(
        state={"tenant_id": str(TENANT)},
        # A Bearer transport header must be IGNORED in favour of the resolved identity.
        headers={"X-Tenant-Id": BEARER, "X-App-Id": "img"},
    )
    body = SimpleNamespace(width=1024, height=1024, steps=4)

    image_routes._emit_cost_event(
        cast(Any, req),
        tenant_id=image_routes.require_request_tenant(cast(Any, req)),
        body=cast(Any, body),
        produced_width=1024,
        produced_height=1024,
        model_repo="flux-schnell",
        latency_ms=4200,
        render_id="0199e5a0-0000-7000-8000-000000000001",
    )
    await _drain_background()

    fields = req.app.state.metering_redis.xadd.call_args.args[1]
    assert fields["tenant_id"] == str(TENANT)
    assert fields["license_id"] == "lic-9"
    assert fields["subject_id"] == "sub-9"
    assert fields["app_id"] == "img"
    assert fields["provider"] == "mflux"


async def test_trusted_network_bare_tenant_attributed() -> None:
    # Trusted-network path: AuthMiddleware sets state.tenant_id from the bare
    # X-Tenant-Id, with no license/subject. Emitter should use it directly.
    set_identity_context(tenant_id=str(TENANT), license_id=None, subject_id=None)
    req = _fake_request(
        state={"tenant_id": str(TENANT)},
        headers={"X-Tenant-Id": str(TENANT)},
    )
    body = SimpleNamespace(width=512, height=512, steps=2)

    image_routes._emit_cost_event(
        cast(Any, req),
        tenant_id=image_routes.require_request_tenant(cast(Any, req)),
        body=cast(Any, body),
        produced_width=512,
        produced_height=512,
        model_repo="flux-schnell",
        latency_ms=900,
        render_id="0199e5a0-0000-7000-8000-000000000001",
    )
    await _drain_background()

    fields = req.app.state.metering_redis.xadd.call_args.args[1]
    assert fields["tenant_id"] == str(TENANT)
    assert fields["license_id"] == ""
    assert fields["subject_id"] == ""


class _ReadyFluxService:
    ready = True
    model_repo = "flux-schnell"

    async def generate(self, **_kwargs: object) -> tuple[Any, int]:  # pragma: no cover
        raise AssertionError("generate must never run for an unattributable request")


def _route_request(*, state: dict) -> SimpleNamespace:
    redis_mock = AsyncMock()
    redis_mock.xadd = AsyncMock(return_value=b"1-0")
    return SimpleNamespace(
        state=SimpleNamespace(**state),
        headers={},
        app=SimpleNamespace(
            state=SimpleNamespace(
                flux_service=_ReadyFluxService(),
                metering_redis=redis_mock,
                metering_stream="billing:costs",
            )
        ),
    )


async def test_missing_tenant_is_401_and_emits_no_cost_event() -> None:
    """No tenant at all → 401, and NOTHING is billed.

    The predecessor of this test asserted the nil-UUID sentinel was emitted.
    That pinned the defect: the nil UUID is a *valid* UUID and is registered in
    ``licenses.tenants`` as ``default-anon``, so it passed every audit while the
    spend went unattributed. w2v-common 0.67.0 deleted it; the correct behaviour
    is a loud refusal at the boundary, before the GPU does any work.
    """
    req = _route_request(state={})
    body = ImageGenRequest(prompt="a cat", width=512, height=512, steps=2, guidance=0.0)

    with pytest.raises(HTTPException) as excinfo:
        await image_routes.generate(cast(Any, req), body)

    assert excinfo.value.status_code == 401
    await _drain_background()
    req.app.state.metering_redis.xadd.assert_not_awaited()


async def test_invalid_tenant_is_401_and_does_not_leak_the_value() -> None:
    """A non-UUID tenant → 401, and the offending value never reaches the client.

    ``X-Tenant-Id`` carries a ``Bearer <license token>`` transport form, so
    echoing the rejected value into the response detail would leak a credential.
    ``w2v_common.tenant`` omits it from its own message for exactly this reason;
    this asserts the route does not put it back.
    """
    req = _route_request(state={"tenant_id": BEARER})
    body = ImageGenRequest(prompt="a cat", width=512, height=512, steps=2, guidance=0.0)

    with pytest.raises(HTTPException) as excinfo:
        await image_routes.generate(cast(Any, req), body)

    assert excinfo.value.status_code == 401
    detail = str(excinfo.value.detail)
    assert BEARER not in detail
    assert "super-secret-license-token" not in detail
    req.app.state.metering_redis.xadd.assert_not_awaited()


async def test_nil_uuid_tenant_is_401_in_every_spelling() -> None:
    """The retired sentinel is rejected BY VALUE, not by spelling.

    ``uuid.UUID`` accepts the dashed, undashed, braced and ``urn:uuid:`` forms as
    the same nil value. A ``None``-only guard would let a caller that still sends
    the old ``DEFAULT_TENANT_ID`` straight back through.
    """
    for spelling in (
        "00000000-0000-0000-0000-000000000000",
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
    ):
        req = _route_request(state={"tenant_id": spelling})
        body = ImageGenRequest(prompt="a cat", width=512, height=512, steps=2, guidance=0.0)
        with pytest.raises(HTTPException) as excinfo:
            await image_routes.generate(cast(Any, req), body)
        assert excinfo.value.status_code == 401, spelling
        req.app.state.metering_redis.xadd.assert_not_awaited()


async def test_valid_tenant_is_metered_as_a_uuid_object() -> None:
    """The happy path: a real tenant renders and meters, typed as ``uuid.UUID``.

    ``require_request_tenant`` must hand ``schedule_cost_event`` a real
    ``uuid.UUID`` (0.67.0 retyped the parameter ``uuid.UUID | str``), not the
    request-state string.
    """
    resolved = image_routes.require_request_tenant(cast(Any, _route_request(state={"tenant_id": str(TENANT)})))
    assert isinstance(resolved, uuid.UUID)
    assert resolved == TENANT

    req = _fake_request(state={"tenant_id": str(TENANT)}, headers={"X-App-Id": "img"})
    image_routes._emit_cost_event(
        cast(Any, req),
        tenant_id=resolved,
        body=cast(Any, SimpleNamespace(width=512, height=512, steps=2)),
        produced_width=512,
        produced_height=512,
        model_repo="flux-schnell",
        latency_ms=700,
        render_id="0199e5a0-0000-7000-8000-000000000001",
    )
    await _drain_background()

    fields = req.app.state.metering_redis.xadd.call_args.args[1]
    assert fields["tenant_id"] == str(TENANT)
    assert fields["app_id"] == "img"


async def test_bills_produced_megapixels_not_request_dims() -> None:
    # Overbilling regression: a 2048x2048 request is clamped by the FLUX service
    # to ~896px long edge (~0.8 MP). Billing MUST follow the produced image, not
    # the raw request. Old behaviour billed 2048*2048 // 1_000_000 == 4 MP for an
    # image the GPU never rendered; the fix bills the produced ~1 MP.
    req = _fake_request(
        state={"tenant_id": str(TENANT), "license_id": "lic-1", "subject_id": "sub-1"},
        headers={},
    )
    body = SimpleNamespace(width=2048, height=2048, steps=4)

    image_routes._emit_cost_event(
        cast(Any, req),
        tenant_id=TENANT,
        body=cast(Any, body),
        produced_width=896,
        produced_height=896,
        model_repo="flux-schnell",
        latency_ms=1000,
        render_id="0199e5a0-0000-7000-8000-000000000001",
    )
    await _drain_background()

    fields = req.app.state.metering_redis.xadd.call_args.args[1]
    # 896*896 = 802_816 → max(1, 0) = 1, NOT 2048*2048 // 1e6 = 4.
    assert fields["input_tokens"] == "1"
    usage_extra = json.loads(fields["usage_extra"])
    assert usage_extra["width"] == 896
    assert usage_extra["height"] == 896
    assert usage_extra["unit"] == "megapixels"


def test_auth_middleware_is_mounted() -> None:
    # Default env (AUTH_ENABLED unset → true) must mount the shared middleware.
    from w2v_common.auth.middleware import AuthMiddleware

    from main import app

    assert any(m.cls is AuthMiddleware for m in app.user_middleware), (
        "w2v_common AuthMiddleware not mounted on image-gen app"
    )


def test_default_stream_resolves_from_env(monkeypatch) -> None:
    """DEFAULT_STREAM honours $METERING_STREAM, defaulting to the canonical
    cost-bus stream (``metering:costs`` was renamed away on 2026-06-24 and has
    no consumer — events there never reach billing)."""
    import importlib

    import services.metering as metering_module

    monkeypatch.delenv("METERING_STREAM", raising=False)
    importlib.reload(metering_module)
    assert metering_module.DEFAULT_STREAM == "billing:costs"

    monkeypatch.setenv("METERING_STREAM", "custom:costs")
    importlib.reload(metering_module)
    assert metering_module.DEFAULT_STREAM == "custom:costs"

    monkeypatch.delenv("METERING_STREAM", raising=False)
    importlib.reload(metering_module)


# --- G5: full attribution on every image cost event ---------------------------

LIC = "0199e5a0-1111-7000-8000-000000000001"
SUBJ = "0199e5a0-2222-7000-8000-000000000002"
RENDER_A = "0199e5a0-aaaa-7000-8000-00000000000a"
RENDER_B = "0199e5a0-bbbb-7000-8000-00000000000b"


async def _emit_fields(*, render_id: str, headers: dict | None = None) -> dict[str, str]:
    req = _fake_request(state={"tenant_id": str(TENANT)}, headers={"X-App-Id": "w2v"} if headers is None else headers)
    image_routes._emit_cost_event(
        cast(Any, req),
        tenant_id=TENANT,
        body=cast(Any, SimpleNamespace(width=1024, height=1024, steps=4)),
        produced_width=1024,
        produced_height=1024,
        model_repo="flux-schnell",
        latency_ms=1000,
        render_id=render_id,
    )
    await _drain_background()
    return req.app.state.metering_redis.xadd.call_args.args[1]


async def test_g5_emits_full_attribution() -> None:
    set_identity_context(tenant_id=str(TENANT), license_id=LIC, subject_id=SUBJ)
    fields = await _emit_fields(render_id=RENDER_A)
    assert fields["license_id"] == LIC
    assert fields["subject_id"] == SUBJ
    assert fields["app_id"] == "w2v"
    assert fields["funding"] == "local_gpu"
    assert fields["event_key"] == cost_bus.cost_event_key("w2v-image", "render", RENDER_A)


async def test_g5_event_key_is_deterministic_per_render() -> None:
    set_identity_context(tenant_id=str(TENANT), license_id=LIC, subject_id=SUBJ)
    first = (await _emit_fields(render_id=RENDER_A))["event_key"]
    again = (await _emit_fields(render_id=RENDER_A))["event_key"]
    other = (await _emit_fields(render_id=RENDER_B))["event_key"]
    assert first == again
    assert first != other


async def test_g5_strict_full_attribution_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(tenant_id=str(TENANT), license_id=LIC, subject_id=SUBJ)
    fields = await _emit_fields(render_id=RENDER_A)
    assert fields["funding"] == "local_gpu"


async def test_g5_strict_missing_licence_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(tenant_id=str(TENANT), license_id=None, subject_id=SUBJ)
    with pytest.raises(cost_bus.MissingAttributionError, match="license_id"):
        await _emit_fields(render_id=RENDER_A)


async def test_g5_strict_missing_app_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(tenant_id=str(TENANT), license_id=LIC, subject_id=SUBJ)
    with pytest.raises(cost_bus.MissingAttributionError, match="app_id"):
        await _emit_fields(render_id=RENDER_A, headers={})


async def test_g5_generate_mints_one_render_id_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route hands every render its own id, so two renders never share a key."""
    seen: list[str] = []
    monkeypatch.setattr(image_routes, "_emit_cost_event", lambda *a, **kw: seen.append(kw["render_id"]))

    class _Flux:
        ready = True
        model_repo = "flux-schnell"

        async def generate(self, **_kwargs: object) -> tuple[Any, int]:
            from PIL import Image

            return Image.new("RGB", (8, 8)), 1

    for _ in range(2):
        req = _route_request(state={"tenant_id": str(TENANT)})
        req.app.state.flux_service = _Flux()
        body = ImageGenRequest(prompt="a cat", width=512, height=512, steps=2, guidance=0.0)
        await image_routes.generate(cast(Any, req), body)
    assert len(seen) == 2
    assert len(set(seen)) == 2
