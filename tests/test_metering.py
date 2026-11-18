"""Unit tests for the mflux cost-metering emitter.

The emit / field-marshalling / Redis-lifecycle logic now lives in the shared
``w2v_common.metering`` cost-bus emitter. These tests pin image-gen's
service-specific identity + tenant fail-safe and verify that emitting through
the shared emitter with those constants produces the EMITTER_GUIDE field shape.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock

from w2v_common.metering import emit_cost_event, init_metering_redis

import services.metering as metering_module
from services.metering import (
    COST_SOURCE,
    DEFAULT_STREAM,
    SERVICE_KEY,
)

# w2v-common 0.67.0 made tenant_id a real UUID end to end: the emitter parses it
# and raises InvalidTenantIdError on a slug. The old "tenant-a" / "t" literals
# are no longer representable, which is the point of the change.
TENANT = uuid.UUID("3f2a1b4c-5d6e-4f70-8912-a3b4c5d6e7f8")


class TestServiceIdentity:
    """image-gen's routing identity — and the absence of any tenant fail-safe."""

    def test_routing_slugs(self) -> None:
        # cost_source stays platform-specific; service_key is the shared PRODUCT
        # slug that must match the modality-gateway image route + licenses App.key
        # so one markup rule / offering prices both paths.
        assert COST_SOURCE == "w2v-image-gen"
        assert SERVICE_KEY == "w2v-image"  # pragma: allowlist secret -- product slug

    def test_module_exposes_no_default_tenant(self) -> None:
        """There is no fallback tenant, and no env knob to reintroduce one.

        This asserted ``DEFAULT_TENANT_ID == "000...0"`` until w2v-common 0.67.0
        deleted the sentinel. Keeping a ``DEFAULT_TENANT_ID`` env override would
        not be a milder version of it — a caller able to supply a default can put
        the sentinel straight back — so the whole name is gone. Unattributable
        requests are refused with 401 in ``routers.image_routes``.
        """
        assert not hasattr(metering_module, "DEFAULT_TENANT_ID")
        assert not hasattr(metering_module, "UNATTRIBUTED_TENANT_ID")
        assert "DEFAULT_TENANT_ID" not in metering_module.__all__

    def test_default_stream_is_canonical(self) -> None:
        assert DEFAULT_STREAM == "billing:costs"


class TestEmitCostEvent:
    """One XADD per generation, EMITTER_GUIDE field shape, via the shared emitter."""

    async def test_xadd_called_with_required_fields(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"1234567890-0")

        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=TENANT,
            license_id="lic-1",
            subject_id="sub-1",
            app_id="app-1",
            provider="mflux",
            model="black-forest-labs/FLUX.1-schnell",
            input_tokens=1,
            output_tokens=0,
            latency_ms=4500,
            cost_usd=0.0,
            stream=DEFAULT_STREAM,
            usage_extra={
                "unit": "megapixels",
                "width": 1024,
                "height": 1024,
                "steps": 4,
            },
        )

        redis_mock.xadd.assert_awaited_once()
        call_args = redis_mock.xadd.call_args
        assert call_args.args[0] == DEFAULT_STREAM
        fields = call_args.args[1]
        assert fields["tenant_id"] == str(TENANT)
        assert fields["license_id"] == "lic-1"
        assert fields["subject_id"] == "sub-1"
        assert fields["app_id"] == "app-1"
        assert fields["service_key"] == SERVICE_KEY == "w2v-image"  # pragma: allowlist secret
        assert fields["cost_source"] == COST_SOURCE == "w2v-image-gen"
        assert fields["provider"] == "mflux"
        assert fields["model"].endswith("FLUX.1-schnell")
        assert fields["input_tokens"] == "1"
        assert fields["output_tokens"] == "0"
        assert fields["latency_ms"] == "4500"
        assert fields["cost_usd"] == "0.00000000"
        usage_extra = json.loads(fields["usage_extra"])
        assert usage_extra == {
            "height": 1024,
            "steps": 4,
            "unit": "megapixels",
            "width": 1024,
        }
        assert call_args.kwargs["maxlen"] == 100_000
        assert call_args.kwargs["approximate"] is True

    async def test_xadd_failure_is_swallowed(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(side_effect=ConnectionError("redis down"))

        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=TENANT,
            license_id="l",
            subject_id="s",
            app_id="a",
            provider="mflux",
            model="flux-schnell",
            input_tokens=1,
            output_tokens=0,
            latency_ms=1000,
            stream=DEFAULT_STREAM,
        )

    async def test_default_usage_extra_is_empty_dict(self) -> None:
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(return_value=b"id")

        await emit_cost_event(
            redis_mock,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=TENANT,
            license_id="l",
            subject_id="s",
            app_id="a",
            provider="mflux",
            model="flux-schnell",
            input_tokens=1,
            output_tokens=0,
            latency_ms=500,
            stream=DEFAULT_STREAM,
        )

        fields = redis_mock.xadd.call_args.args[1]
        assert fields["usage_extra"] == "{}"


class TestInitMeteringRedis:
    """Returns None when URL is empty / unreachable; never raises."""

    async def test_empty_url_returns_none(self) -> None:
        client = await init_metering_redis("")
        assert client is None

    async def test_connect_failure_returns_none(self) -> None:
        client = await init_metering_redis("redis://127.0.0.1:1/0")
        assert client is None
