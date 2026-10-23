"""Immutable operational slowdown baselines; no scientific payloads are read."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

from .exports import _verify, publish_json
from .slowdown_guard import SlowdownGuard
from .w12_runtime import _canonical_series_state_dirs

SCHEMA = "bl-labels-f0-slowdown-baseline/v1"
RULING_SHA256 = "e50a573e69e6aeea1de7ddde590bac36e62a28a46c3399f5bce0182ae4109551"


def _path(state: Path, stream: str, width: int) -> Path:
    identity = hashlib.sha256(stream.encode()).hexdigest()[:16]
    return state / f"slowdown-baseline-{identity}-w{width}.json"


def _read(path: Path, pin: str) -> dict:
    _verify(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != SCHEMA
        or value.get("source_commit") != pin
        or value.get("ruling_sha256") != RULING_SHA256
    ):
        raise ValueError("slowdown baseline source identity differs; HOLD")
    return value


def build_guards(
    control: dict, workers: int, started: float, terminals: list[dict], state: Path | None
) -> dict[str, SlowdownGuard]:
    """Restore frozen medians and collection, with fresh attempt windows only."""
    streams = sorted({item["stream"] for item in control["allocation"]})
    pin, namespace = control.get("source_commit"), control["namespace"]
    saved = terminals[-1].get("slowdown_state_by_stream", {}) if terminals else {}
    width_wall: dict[str, float] = {}
    for terminal in terminals:
        key = str(terminal["workers"])
        width_wall[key] = width_wall.get(key, 0.0) + terminal["elapsed_seconds"]
    refs: dict[int, float] = {}
    latest: dict[int, float] = {}
    if state is not None:
        for related in _canonical_series_state_dirs(state):
            for path in related.glob("slowdown-baseline-*.json"):
                _verify(path)
                raw = json.loads(path.read_text())
                if raw.get("source_commit") != pin:
                    continue
                value = _read(path, pin)
                width, epoch = value["width"], value["created_epoch"]
                if epoch >= latest.get(width, 0):
                    refs[width], latest[width] = value["baseline"]["median"], epoch
    # The ruling explicitly freezes the original single fresh D256 W6 hour.
    # Never take the later resume's replay-contaminated first-hour statistic.
    legacy: dict | None = None
    if terminals and control.get("mac_gate", {}).get("stage") == "d256":
        first = terminals[0]
        samples = first.get("first_hour_life_seconds", [])
        if first["workers"] == 6 and first["elapsed_seconds"] >= 3600 and len(samples) >= 60:
            legacy_path = _path(state, "watch6", 6) if state is not None else None
            if legacy_path is not None and legacy_path.exists():
                legacy = _read(legacy_path, pin)["baseline"]
            else:
                legacy = saved.get("watch6", {}).get("baselines", {}).get("6")
            if legacy is None:
                legacy = {
                    "median": statistics.median(samples),
                    "sample_count": len(samples),
                    "frozen": True,
                }
            refs.setdefault(6, legacy["median"])
    d256_w6 = legacy["median"] if legacy is not None else None
    if d256_w6 is None:
        from . import w12_calibration
        from .w12_runtime import _attempt_pairs

        canonical = w12_calibration.PACK / "d256-state-mac-01"
        baseline_path = _path(canonical, "watch6", 6)
        if baseline_path.exists():
            from .frozen_factory import OLD_PIN

            d256_w6 = _read(baseline_path, OLD_PIN)["baseline"]["median"]
        elif canonical.is_dir():
            pairs = _attempt_pairs(canonical)
            if pairs:
                first = pairs[0][3]
                samples = first.get("first_hour_life_seconds", [])
                if (
                    first["workers"] == 6
                    and first["elapsed_seconds"] >= 3600
                    and len(samples) >= 60
                ):
                    d256_w6 = statistics.median(samples)
    guards = {}
    for stream in streams:
        restored = saved.get(stream)
        if restored is None:
            restored = {
                "schema": "bl-labels-f0-slowdown-guard/v1",
                "stream": stream,
                "width_wall_seconds": {},
                "baselines": {},
                "candidates": {},
            }
            if legacy is not None and stream == "watch6":
                restored["baselines"]["6"] = dict(legacy)
                restored["width_wall_seconds"] = dict(width_wall)
        # Independently persisted baselines are authoritative after freeze,
        # including a cooperative stop between freeze and terminal publication.
        if state is not None:
            for width in range(1, 13):
                path = _path(state, stream, width)
                if any(
                    Path(str(path) + suffix).exists() for suffix in ("", ".sha256", ".complete")
                ):
                    value = _read(path, pin)
                    if (
                        value.get("namespace") != namespace
                        or value.get("stream") != stream
                        or value.get("width") != width
                    ):
                        raise ValueError("slowdown baseline namespace/width differs; HOLD")
                    existing = restored["baselines"].get(str(width))
                    if existing is not None and existing != value["baseline"]:
                        raise ValueError("frozen slowdown baseline was recomputed; HOLD")
                    restored["baselines"][str(width)] = value["baseline"]
        guards[stream] = SlowdownGuard(
            stream,
            workers,
            started,
            persisted=restored,
            reference_baselines=refs,
            d256_w6_baseline=d256_w6,
        )
    return guards


def persist_baselines(
    guards: dict[str, SlowdownGuard], state: Path | None, control: dict, now: float
) -> None:
    """Publish a baseline exactly once, checking every reuse against its bytes."""
    if state is None:
        return
    latest: dict[int, tuple[float, float]] = {}
    for stream, guard in guards.items():
        for raw_width, baseline in guard.snapshot(now)["baselines"].items():
            if not baseline["frozen"]:
                continue
            width = int(raw_width)
            path = _path(state, stream, width)
            if any(Path(str(path) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
                value = _read(path, control.get("source_commit"))
                if (
                    value.get("namespace") != control["namespace"]
                    or value.get("stream") != stream
                    or value.get("width") != width
                    or value.get("baseline") != baseline
                ):
                    raise ValueError("frozen slowdown baseline differs; HOLD")
            else:
                value = {
                    "schema": SCHEMA,
                    "ruling_sha256": RULING_SHA256,
                    "namespace": control["namespace"],
                    "source_commit": control.get("source_commit"),
                    "stream": stream,
                    "width": width,
                    "baseline": baseline,
                    "created_epoch": time.time(),
                }
                publish_json(path, value)
            epoch = value["created_epoch"]
            if width not in latest or epoch >= latest[width][0]:
                latest[width] = (epoch, baseline["median"])
    for guard in guards.values():
        for width, (_epoch, baseline) in latest.items():
            guard.update_reference(width, baseline)
