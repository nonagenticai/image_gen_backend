"""image-gen cost-metering identity + thin delegation to ``w2v_common.metering``.

The emit / stream-default / field-marshalling logic now lives in the shared
cost-bus emitter ``w2v_common.metering`` (one source of truth for every w2v
service — see that module's docstring for the rename-safety rationale). This
module keeps only what is image-gen-specific:

* the routing identity (``COST_SOURCE`` / ``SERVICE_KEY``), and
* a single import site that re-exports the Redis lifecycle + stream helpers.

There is deliberately **no default tenant here any more**. w2v-common 0.67.0
deleted ``UNATTRIBUTED_TENANT_ID`` / ``default_tenant_id``: a cost event with
no real tenant is refused at the request boundary (HTTP 401 in
``routers.image_routes``), never substituted. A ``DEFAULT_TENANT_ID`` env knob
is not a milder version of that — it is the same sentinel with a different
owner, so it is gone too.

image_gen_backend runs as an Apple-silicon launchd unit on mini1 — there is no
in-cluster Redis route. The launchd plist provides ``METERING_REDIS_URL``
pointing at the Tailnet-reachable cluster Redis. When unset, emission is
disabled silently and the service runs normally.
"""

from __future__ import annotations

from typing import Final

from w2v_common.metering import (
    close_metering_redis,
    init_metering_redis,
    resolve_stream,
)

__all__ = [
    "COST_SOURCE",
    "DEFAULT_STREAM",
    "FUNDING",
    "SERVICE_KEY",
    "close_metering_redis",
    "init_metering_redis",
    "resolve_stream",
]

# COST_SOURCE identifies WHERE the spend came from (platform mflux on-device) —
# kept distinct so billing can tell platform-local image gen apart from the
# BYO-cloud path within the same product.
COST_SOURCE = "w2v-image-gen"
# SERVICE_KEY is the billable PRODUCT slug. It MUST match the modality-gateway's
# image route (`w2v-image`) and the licenses `App.key` for the image product, so
# a single per-(tenant, product) markup rule / offering prices BOTH the BYO-cloud
# path (gateway) and this platform-local path. A divergent slug would silently
# leave one path unpriced. See w2v emit-coverage audit.
SERVICE_KEY = "w2v-image"  # pragma: allowlist secret -- routing slug, not a credential

#: WHO FUNDS a render: mflux runs on the platform's own Apple-silicon GPU, so
#: every image-gen cost event is ``local_gpu`` (billing FUNDING_VALUES). There is
#: no external provider and no tenant key on this path.
FUNDING: Final = "local_gpu"


# Resolved cost-bus stream name (``METERING_STREAM`` env, else the canonical
# ``billing:costs``). Captured at import so it can be stashed on
# ``app.state`` once at startup; ``resolve_stream()`` is the live accessor.
DEFAULT_STREAM = resolve_stream()
