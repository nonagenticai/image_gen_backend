"""Guardrail tests for the OpenAPI surface of image_gen_backend.

Locks in the org compliance standard:

- every route has an explicit ``operation_id="…"`` per
  ``OPERATION_ID_PLAN.md`` (no reliance on the
  ``generate_unique_id_function`` fallback — that is defence-in-depth only);
- the standard OpenAPI metadata (BearerAuth security scheme, openapi_tags
  grouping by domain) is present;
- protected ``/api/...`` paths advertise the security requirement;
- public health paths do NOT.

If a new route lands, update ``OPERATION_ID_PLAN.md`` *and* the
``EXPECTED_OPERATION_IDS`` set below — they should move together.
"""

from __future__ import annotations

import pytest

from main import app  # noqa: E402  — sys.path bootstrap in conftest.py

# Source of truth — keep in sync with OPERATION_ID_PLAN.md.
EXPECTED_OPERATION_IDS: frozenset[str] = frozenset(
    {
        "health_check_basic",
        "health_check_ready",
        "health_check_live",
        "generate_image",
    }
)


@pytest.fixture(scope="module")
def openapi_schema() -> dict:
    """Build the OpenAPI schema once per module."""
    app.openapi_schema = None
    return app.openapi()


def test_every_route_has_explicit_operation_id(openapi_schema: dict) -> None:
    """No route may rely on FastAPI's auto-generated operationId."""
    auto_suffixes = ("_post", "_get", "_put", "_delete", "_patch")
    offenders: list[tuple[str, str, str]] = []
    for path, item in openapi_schema.get("paths", {}).items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId", "")
            if not op_id:
                offenders.append((method.upper(), path, "<missing>"))
                continue
            if op_id.endswith(auto_suffixes) and op_id not in EXPECTED_OPERATION_IDS:
                offenders.append((method.upper(), path, op_id))
    assert not offenders, f"Routes without explicit operation_id (auto-generated): {offenders}"


def test_operation_ids_match_plan(openapi_schema: dict) -> None:
    """The set of operationIds in the schema must equal the plan.

    The prometheus-fastapi-instrumentator mounts ``/metrics`` which is excluded
    here because it is not part of the service's business surface and is not
    in OPERATION_ID_PLAN.md.
    """
    actual: set[str] = set()
    for path, item in openapi_schema.get("paths", {}).items():
        if path == "/metrics":
            continue
        for op in item.values():
            if isinstance(op, dict) and op.get("operationId"):
                actual.add(op["operationId"])

    missing = EXPECTED_OPERATION_IDS - actual
    unexpected = actual - EXPECTED_OPERATION_IDS
    assert not missing and not unexpected, (
        f"operationId drift — missing: {sorted(missing)}, unexpected: {sorted(unexpected)}"
    )


def test_operation_ids_are_unique(openapi_schema: dict) -> None:
    """Every operationId must be unique across the whole schema."""
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    for path, item in openapi_schema.get("paths", {}).items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            if op_id in seen:
                duplicates.append((op_id, seen[op_id], (method.upper(), path)))
            else:
                seen[op_id] = (method.upper(), path)
    assert not duplicates, f"Duplicate operationIds: {duplicates}"


def test_bearer_auth_scheme_is_declared(openapi_schema: dict) -> None:
    """The BearerAuth security scheme must be present in components."""
    schemes = openapi_schema.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes, "BearerAuth scheme missing from OpenAPI"
    scheme = schemes["BearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"


def test_tenant_id_header_parameter_is_declared(openapi_schema: dict) -> None:
    """The reusable TenantIdHeader parameter must be in components.parameters."""
    parameters = openapi_schema.get("components", {}).get("parameters", {})
    assert "TenantIdHeader" in parameters, "TenantIdHeader parameter missing"
    param = parameters["TenantIdHeader"]
    assert param["in"] == "header"
    assert param["name"].lower() == "x-tenant-id"


def test_protected_routes_have_security_requirement(openapi_schema: dict) -> None:
    """Every /api/... operation must advertise BearerAuth."""
    api_paths = [p for p in openapi_schema.get("paths", {}) if p.startswith("/api/")]
    assert api_paths, "No /api/* paths found in schema"
    offenders: list[tuple[str, str]] = []
    for path in api_paths:
        for method, op in openapi_schema["paths"][path].items():
            if not isinstance(op, dict):
                continue
            security = op.get("security", [])
            if not any("BearerAuth" in entry for entry in security):
                offenders.append((method.upper(), path))
    assert not offenders, f"Protected routes missing BearerAuth: {offenders}"


def test_public_routes_have_no_security_requirement(openapi_schema: dict) -> None:
    """Health paths must not require auth in the schema."""
    public = {"/health", "/health/ready", "/health/live"}
    for path in public:
        item = openapi_schema.get("paths", {}).get(path)
        if item is None:
            continue
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            assert not op.get("security"), (
                f"Public route {method.upper()} {path} unexpectedly has security={op.get('security')}"
            )


def test_tag_groups_are_defined(openapi_schema: dict) -> None:
    """``openapi_tags`` must define a group per domain."""
    tag_names = {t["name"] for t in openapi_schema.get("tags", [])}
    required = {"health", "image-generation"}
    assert required <= tag_names, f"Missing required tag groups; have {sorted(tag_names)}"
