"""The entitlement gate on /api/image actually refuses an unlicensed tenant.

Until now this service authenticated the licence token for tenant IDENTITY and
stopped there. ``AuthMiddleware`` (mounted in ``src/main.py``) answers "who are
you", never "may you" — its own comment says so: *"attribution for billing —
NOT a deny gate"*. A tenant whose licence is ``expired`` or ``revoked`` still
presents a syntactically valid token, so ``POST /api/image/generate`` burned
GPU minutes for them and emitted a cost event nobody could collect on.
``LICENSE_GATE`` is mounted router-wide on ``routers.image_routes.router`` to
close that, copying ``tts_service`` ``src/api/tts.py``.

Every test here drives a REAL HTTP request through the REAL router, because the
interesting question is not whether ``require_license`` works — w2v_common owns
that, and has its own tests — but whether THIS service's router actually
carries it. FastAPI applies a parent router's ``dependencies=`` when a request
is MATCHED, so static introspection of ``route.dependant.dependencies`` can be
a FALSE NEGATIVE. A request is the only honest probe.

The 200s (entitled tenant; flag off) are what stop this file from passing
vacuously: if the gate were mounted wrongly, or the fake FLUX service broken,
those cases would not be 200s either.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from routers.image_routes import router as image_router

TENANT = str(uuid.UUID("3f2a1b4c-5d6e-4f70-8912-a3b4c5d6e7f8"))
SUBJECT = "user-under-test"

#: The only route under the gated router, and the whole point of gating it:
#: tenant-scoped, GPU-consuming, cost-metered generation.
GATED_PATH = "/api/image/generate"

BODY: dict[str, Any] = {"prompt": "a test prompt", "width": 256, "height": 256, "steps": 1}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TENANT,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


class _FakeFlux:
    """A FLUX service that renders instantly.

    Real generation needs MLX and an Apple GPU. What the ALLOW assertions need
    is only that the request reached the handler and completed, so this returns
    a 1x1 image. ``metering_redis`` is left unset on app.state, which makes
    ``_emit_cost_event`` a documented no-op — no redis in this test.
    """

    ready = True
    model_repo = "fake/flux-test"

    async def generate(self, **kwargs: object) -> tuple[Image.Image, int]:
        return Image.new("RGB", (1, 1), (0, 0, 0)), 4242


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture
def app() -> FastAPI:
    """The REAL router, with identity stamped the way ``AuthMiddleware`` stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TENANT
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(image_router)
    application.state.flux_service = _FakeFlux()
    application.state.metering_redis = None
    application.state.metering_stream = "billing:costs"
    return application


async def _post(app: FastAPI, path: str) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json=BODY)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr("w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake)


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _post(app, GATED_PATH)

    assert response.status_code == 402, response.text
    # A bare FastAPI app here, so the default HTTPException handler renders the
    # gate's structured detail verbatim (src/main.py's envelope handlers are not
    # mounted on this test app, deliberately — this asserts the GATE, not main).
    assert response.json()["detail"]["error"] == code, response.text


async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _post(app, GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable", response.text
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _post(app, GATED_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["seed"] == 4242


# ---------------------------------------------------------------------------
# Dormant on merge — this is what ships
# ---------------------------------------------------------------------------


async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy.

    Arming also requires modality_gateway (the only caller) to forward a real
    licence token first; arming blind took this service down at 1am once."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post(app, GATED_PATH)

    assert response.status_code == 200, response.text


async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post(app, GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    endpoints, w2v_common would still refuse to enforce on these paths. A 402 on
    a probe makes launchd restart the unit — an unlicensed tenant cannot be
    billed by killing our own service."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)


def test_the_gate_is_actually_mounted_on_the_image_router():
    """A direct structural check on the router object, paired with the HTTP
    tests above so a FastAPI change that moves where dependencies are applied
    cannot leave BOTH silent."""
    from w2v_common.auth import require_license

    mounted = [getattr(dep, "dependency", None) for dep in (image_router.dependencies or []) if dep is not None]
    assert require_license in mounted, mounted


def test_no_manifest_or_env_file_arms_the_gate():
    """⛔ This PR ships the gate DORMANT. If a future change adds
    W2V_LICENSE_GATE_ENABLED to service.yml or start.sh, it must be a
    deliberate, separately-reviewed arming — not a drive-by."""
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    for name in ("service.yml", "start.sh"):
        candidate = repo / name
        if candidate.exists():
            assert "W2V_LICENSE_GATE_ENABLED" not in candidate.read_text(), name
