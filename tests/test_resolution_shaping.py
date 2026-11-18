"""Resolution shaping: what pixels the model is actually asked to render.

Two separate rules live here and they pull in opposite directions:

* ``_aspect_clamp`` scales an OVER-cap request DOWN to the host's memory budget.
  It promised aspect preservation and did not deliver it — flooring BOTH dims to
  a multiple of 64 turned every 16:9 order into 2.0:1. These tests pin the
  rounding rule, not just the cap, because the rounding rule is the bug.
* ``_render_to_cap`` scales a BELOW-cap request UP so the LANCZOS upscale on the
  way out interpolates from real pixels. Off by default; the tests assert both
  the off-by-default contract and that the flag actually moves the request dims.

No mlx/mflux import anywhere — the pipeline is stubbed, so this runs on any
machine.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from services.flux_service import (
    FluxService,
    _aspect_clamp,
    _env_flag,
    _render_to_cap,
    _snap_to_long_edge,
)

# The live host cap: IMAGE_GEN_MAX_LONG_EDGE=896 in ConfigMap
# image-gen-backend-env (ns external-services), confirmed against mini1's
# running process env on 2026-09-15.
_LIVE_CAP = 896


def _aspect(dims: tuple[int, int]) -> float:
    return dims[0] / dims[1]


# ── _aspect_clamp: the 16:9 regression ───────────────────────────────────────


@pytest.mark.parametrize(
    ("request_dims", "expected"),
    [
        # The regression. Flooring the short edge gave 896x448 (2.0:1) for every
        # one of these; the ordered shape is 16:9 (1.778).
        ((1280, 720), (896, 512)),  # course/thumbnail.py _DEFAULT_SIZE
        ((1920, 1080), (896, 512)),
        ((1536, 864), (896, 512)),
        # Portrait: the long edge is the HEIGHT, so the tuple must come back
        # the other way round. 672 lands exactly on a half-unit (10.5), which is
        # why the rounding is half-up rather than round().
        ((768, 1024), (704, 896)),
        # Already-correct cases must not move.
        ((1792, 1024), (896, 512)),
        ((1024, 1024), (896, 896)),
        # Under the cap: returned untouched, no snapping at all.
        ((512, 320), (512, 320)),
        ((896, 512), (896, 512)),
    ],
)
def test_aspect_clamp_shapes(request_dims: tuple[int, int], expected: tuple[int, int]) -> None:
    assert _aspect_clamp(*request_dims, _LIVE_CAP) == expected


def test_aspect_clamp_no_longer_squashes_sixteen_by_nine_to_two_to_one() -> None:
    """The headline defect, asserted on the PROPERTY rather than the number.

    Before the fix a 16:9 order came back at exactly 2.0:1 — a 0.222 aspect
    error that ken_burns then letterboxes against a 16:9 canvas. 16:9 is not
    representable on a 64-px grid at this cap (896/1.778 = 504 = 7.875 units),
    so the bar is "close", not "exact": 896x512 is 0.028 off.
    """
    got = _aspect_clamp(1280, 720, _LIVE_CAP)
    assert abs(_aspect(got) - 16 / 9) < 0.05, f"{got} is {_aspect(got):.3f}, ordered 1.778"
    assert _aspect(got) != pytest.approx(2.0), "the squash regressed"


def test_aspect_clamp_never_exceeds_the_cap_and_keeps_the_64_grid() -> None:
    """Rounding the short edge UP must not breach either invariant the model
    depends on: mflux needs multiples of 64, and the cap is a memory budget."""
    for w in range(256, 2049, 64):
        for h in range(256, 2049, 64):
            got_w, got_h = _aspect_clamp(w, h, _LIVE_CAP)
            assert max(got_w, got_h) <= max(_LIVE_CAP, max(w, h))
            assert got_w % 64 == 0 and got_h % 64 == 0
            assert got_w >= 256 and got_h >= 256


def test_aspect_clamp_preserves_which_side_is_longer() -> None:
    """Rounding must never flip a portrait order into a landscape render."""
    assert _aspect_clamp(1024, 1920, _LIVE_CAP)[1] > _aspect_clamp(1024, 1920, _LIVE_CAP)[0]
    assert _aspect_clamp(1920, 1024, _LIVE_CAP)[0] > _aspect_clamp(1920, 1024, _LIVE_CAP)[1]


def test_snap_rounds_the_short_edge_to_nearest_not_down() -> None:
    """The rounding rule in isolation — this is the single line that was wrong.

    720 scaled by 896/1280 is 504, i.e. 7.875 grid units. Flooring gives 7
    (448); nearest gives 8 (512).
    """
    assert _snap_to_long_edge(1280, 720, _LIVE_CAP) == (896, 512)


# ── _render_to_cap: feed the upscale real pixels ─────────────────────────────


def test_render_to_cap_grows_a_below_cap_request() -> None:
    """512x320 is the orchestrator's live default for lesson stills, chosen when
    this host's cap really was 512. At the 896 cap it leaves ~3x the available
    pixels unrendered and hands LANCZOS a 3x blow-up to invent."""
    grown = _render_to_cap(512, 320, _LIVE_CAP)
    assert grown == (896, 576)
    assert grown[0] * grown[1] > 3 * (512 * 320)


def test_render_to_cap_leaves_at_or_above_cap_requests_alone() -> None:
    """``_aspect_clamp`` owns the down direction; this must not fight it."""
    assert _render_to_cap(896, 512, _LIVE_CAP) == (896, 512)
    assert _render_to_cap(1920, 1080, _LIVE_CAP) == (1920, 1080)


# ── _env_flag ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True), (" 1 ", True)]
    + [("0", False), ("false", False), ("no", False), ("", False), ("lanczos", False)],
)
def test_env_flag_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("IMAGE_GEN_RENDER_TO_CAP", raw)
    assert _env_flag("IMAGE_GEN_RENDER_TO_CAP", default=False) is expected


def test_env_flag_unset_returns_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGE_GEN_RENDER_TO_CAP", raising=False)
    assert _env_flag("IMAGE_GEN_RENDER_TO_CAP", default=False) is False
    assert _env_flag("IMAGE_GEN_RENDER_TO_CAP", default=True) is True


# ── end to end through generate(): which dims reach the model ────────────────


class _StubImage:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def save(self, buf: io.BytesIO, format: str) -> None:  # pragma: no cover - unused here
        buf.write(b"\x89PNG\r\n\x1a\n")

    def resize(self, size: tuple[int, int], resample: object) -> _StubImage:
        return _StubImage(size)


class _DimRecordingPipeline:
    """mflux pipeline stand-in that records the dims it was ASKED to render.

    This is the assertion that matters: the whole change is about what the model
    is told to produce, and the returned image is a stub, so nothing downstream
    of the submit can prove it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def generate_image(self, *, width: int, height: int, **_rest: object) -> SimpleNamespace:
        self.calls.append((width, height))
        return SimpleNamespace(image=_StubImage((width, height)))


def _service_with_recorder() -> tuple[Any, _DimRecordingPipeline]:
    # `Any` deliberately, as in test_generate_offload: `_pipeline` is declared
    # `Flux1 | None` and the whole point is to install a stand-in without
    # importing mflux.
    svc: Any = FluxService()
    pipeline = _DimRecordingPipeline()
    svc._pipeline = pipeline
    svc.ready = True
    return svc, pipeline


async def _generate(svc: FluxService, width: int, height: int) -> tuple[int, int]:
    image, _seed = await svc.generate(prompt="a cat", width=width, height=height, steps=4, guidance=0.0, seed=7)
    return image.size


async def test_generate_asks_the_model_for_a_sixteen_by_nine_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_GEN_MAX_LONG_EDGE", str(_LIVE_CAP))
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "none")
    svc, pipeline = _service_with_recorder()
    try:
        await _generate(svc, 1280, 720)
    finally:
        await svc.close()
    assert pipeline.calls == [(896, 512)], "the model was asked for the squashed 2.0:1 shape"


async def test_render_to_cap_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped default must be behaviour-identical to before this change —
    a 512x320 order still renders at 512x320 until an operator opts in."""
    monkeypatch.setenv("IMAGE_GEN_MAX_LONG_EDGE", str(_LIVE_CAP))
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "none")
    monkeypatch.delenv("IMAGE_GEN_RENDER_TO_CAP", raising=False)
    svc, pipeline = _service_with_recorder()
    try:
        await _generate(svc, 512, 320)
    finally:
        await svc.close()
    assert pipeline.calls == [(512, 320)]


async def test_render_to_cap_flag_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_GEN_MAX_LONG_EDGE", str(_LIVE_CAP))
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "none")
    monkeypatch.setenv("IMAGE_GEN_RENDER_TO_CAP", "1")
    svc, pipeline = _service_with_recorder()
    try:
        await _generate(svc, 512, 320)
    finally:
        await svc.close()
    assert pipeline.calls == [(896, 576)], "the flag did not reach the render call"


async def test_render_to_cap_does_not_change_the_delivered_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract guard. ``w2v_orchestrator scripts/e2e_validate.py:3866``
    hard-asserts ``long_edge == 1536`` on the delivered PNG, so turning this
    flag on must move the RENDER size and nothing else."""
    monkeypatch.setenv("IMAGE_GEN_MAX_LONG_EDGE", str(_LIVE_CAP))
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "lanczos")
    monkeypatch.setenv("IMAGE_GEN_UPSCALE_LONG_EDGE", "1536")

    monkeypatch.delenv("IMAGE_GEN_RENDER_TO_CAP", raising=False)
    svc, _ = _service_with_recorder()
    try:
        off_size = await _generate(svc, 1024, 1024)
    finally:
        await svc.close()

    monkeypatch.setenv("IMAGE_GEN_RENDER_TO_CAP", "1")
    svc, _ = _service_with_recorder()
    try:
        on_size = await _generate(svc, 1024, 1024)
    finally:
        await svc.close()

    assert max(off_size) == 1536
    assert on_size == off_size, "the delivered size moved — this breaks the 1536 E2E assertion"


async def test_upscale_target_is_still_reached_from_a_grown_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 512x320 order with the flag ON must still deliver a 1536 long edge."""
    monkeypatch.setenv("IMAGE_GEN_MAX_LONG_EDGE", str(_LIVE_CAP))
    monkeypatch.setenv("IMAGE_GEN_UPSCALE", "lanczos")
    monkeypatch.setenv("IMAGE_GEN_UPSCALE_LONG_EDGE", "1536")
    monkeypatch.setenv("IMAGE_GEN_RENDER_TO_CAP", "1")
    svc, _ = _service_with_recorder()
    try:
        size = await _generate(svc, 512, 320)
    finally:
        await svc.close()
    assert max(size) == 1536


def test_the_shaping_helpers_need_no_native_runtime() -> None:
    """These tests must run on a Linux CI box as well as on mini1: the shaping
    is pure arithmetic and must stay importable without mflux/mlx present.

    Asserting ``"mlx" not in sys.modules`` would NOT prove this — on a mac dev
    box some other collected test has already imported it, so the assertion
    passes or fails for reasons that have nothing to do with this module.
    Re-importing in a clean interpreter is the honest check.
    """
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]);"
            " from services.flux_service import _aspect_clamp, _render_to_cap;"
            " assert _aspect_clamp(1280, 720, 896) == (896, 512);"
            " assert 'mlx' not in sys.modules and 'mflux' not in sys.modules",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
