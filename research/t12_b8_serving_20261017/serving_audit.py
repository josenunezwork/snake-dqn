#!/usr/bin/env python3
"""Independent, stdlib-only audit for the T12 B8 serving qualification.

The producer writes result envelopes; this module deliberately does not import
the producer or any runtime implementation.  It re-derives the qualification
criteria from the raw evidence retained in those envelopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "protocol.md"
PROTOCOL_SHA256 = "3d8b51675a2cdecbc476af56c48cfb4fa3ac6566eac1d4faf21c1870174bd406"
CHECKPOINT_NAME = "b8-s3"
CHECKPOINT_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
SCHEMA = "t12-serving-result/v1"
AUDIT_SCHEMA = "t12-serving-audit/v1"


class Invalid(RuntimeError):
    """Evidence is absent, unreadable, or cannot be bound to one run."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Invalid(f"unreadable JSON {path}: {exc}") from exc


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _bound_sources(intent: Mapping[str, Any]) -> List[str]:
    """Verify the source bytes the producer committed to before episodes began."""
    sources = intent.get("sources")
    out: List[str] = []
    if not isinstance(sources, Mapping) or not sources:
        return ["intent.sources is empty"]
    for rel, digest in sources.items():
        path = Path(str(rel))
        if path.is_absolute() or ".." in path.parts or not _is_hex(digest):
            out.append(f"unsafe/unhashed source binding: {rel}")
        elif not (REPO / path).is_file() or _sha(REPO / path) != digest:
            out.append(f"registered source changed: {rel}")
    return out


def _intent_bindings(intent: Mapping[str, Any]) -> List[str]:
    """Check immutable candidate, fresh-bank, and strict-pin evidence bindings."""
    out = _bound_sources(intent)
    checkpoint = intent.get("checkpoint", {})
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("name") != CHECKPOINT_NAME
        or checkpoint.get("sha256") != CHECKPOINT_SHA256
    ):
        out.append("intent checkpoint choice is not b8-s3")
    if intent.get("released_default") != "frp3-s12":
        out.append("qualification changed CHECKPOINT_RELEASED_DEFAULT")
    audit = intent.get("seed_audit", {})
    if (
        not isinstance(audit, Mapping)
        or not isinstance(audit.get("path"), str)
        or not _is_hex(audit.get("sha256"))
    ):
        out.append("intent seed audit binding is invalid")
    else:
        path = Path(audit["path"])
        if not path.is_file() or _sha(path) != audit["sha256"]:
            out.append("seed freshness report bytes differ from intent")
        else:
            report = _json(path)
            if (
                not isinstance(report, Mapping)
                or report.get("pass") is not True
                or report.get("banks") != intent.get("banks")
            ):
                out.append("seed freshness report does not pass/bind exact banks")
    pin = intent.get("strict_pin", {})
    receipt = pin.get("strict_receipt") if isinstance(pin, Mapping) else None
    if (
        not isinstance(pin, Mapping)
        or pin.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or pin.get("veto_variant") != "none"
    ):
        out.append("intent strict pin does not bind B8 no-veto")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("verdict") != "STRICT_PASS"
        or not isinstance(receipt.get("root"), str)
    ):
        out.append("intent strict receipt is not STRICT_PASS")
    elif Path(receipt["root"]).is_absolute():
        for key in ("decision", "audit_report", "independent_review"):
            item = receipt.get(key, {})
            path = Path(receipt["root"]) / str(item.get("path", ""))
            if (
                not isinstance(item, Mapping)
                or not _is_hex(item.get("sha256"))
                or not path.is_file()
                or _sha(path) != item["sha256"]
            ):
                out.append(f"strict pin evidence mismatch: {key}")
    else:
        out.append("strict receipt root is not absolute")
    return out


def _num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentile(values: Sequence[float], q: float) -> float:
    """The interpolation used by the measurement producer, recomputed from samples."""
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return (
        values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)
    )


def _expected_files() -> set[str]:
    names = {"intent.json"}
    names |= {f"records/watch-{i:03d}.json" for i in range(25)}
    names |= {f"records/play-{i:03d}.json" for i in range(25)}
    names |= {f"parity-{i:03d}.json" for i in range(2)}
    names |= {
        "p2_h5000.json",
        "p2_h10000.json",
        "controls.json",
        "defaults.json",
        "confidence.json",
        "latency.json",
        "ui.json",
        "validation.json",
    }
    return names


def _result(root: Path, rel: str, intent_sha: str, intent: Mapping[str, Any]) -> Mapping[str, Any]:
    doc = _json(root / rel)
    if not isinstance(doc, Mapping):
        raise Invalid(f"{rel} is not an object")
    required = {
        "schema",
        "intent_sha256",
        "kind",
        "index",
        "seed_or_seeds",
        "checkpoint_sha256",
        "started_utc",
        "wall_seconds",
        "status",
        "error",
        "data",
    }
    if set(doc) != required:
        raise Invalid(f"{rel} wrapper keys differ: {sorted(set(doc) ^ required)}")
    if doc["schema"] != SCHEMA or doc["intent_sha256"] != intent_sha:
        raise Invalid(f"{rel} is not bound to this intent/schema")
    if doc["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise Invalid(f"{rel} is not bound to b8-s3 bytes")
    if (
        doc["status"] != "complete"
        or doc["error"] is not None
        or not isinstance(doc["data"], Mapping)
    ):
        raise Invalid(f"{rel} is incomplete or errored")
    if not _num(doc["wall_seconds"]) or float(doc["wall_seconds"]) < 0:
        raise Invalid(f"{rel} has invalid wall_seconds")
    stem = Path(rel).stem
    if rel.startswith("records/"):
        kind, index = stem.rsplit("-", 1)
        expected_seed = ((intent.get("banks") or {}).get(kind) or {}).get("seeds", [])
        if (
            doc["kind"] != kind
            or doc["index"] != int(index)
            or len(expected_seed) <= int(index)
            or doc["seed_or_seeds"] != expected_seed[int(index)]
        ):
            raise Invalid(f"{rel} kind/index/seed does not bind registered bank")
        if doc["data"].get("seed") != doc["seed_or_seeds"]:
            raise Invalid(f"{rel} nested data.seed does not bind envelope seed")
    elif stem.startswith("parity-"):
        index = int(stem.rsplit("-", 1)[1])
        seeds = ((intent.get("banks") or {}).get("parity") or {}).get("seeds", [])
        if (
            doc["kind"] != "parity"
            or doc["index"] != index
            or len(seeds) <= index
            or doc["seed_or_seeds"] != seeds[index]
        ):
            raise Invalid(f"{rel} kind/index/seed does not bind registered bank")
        if doc["data"].get("world_seed") != doc["seed_or_seeds"]:
            raise Invalid(f"{rel} nested data.world_seed does not bind envelope seed")
    elif stem.startswith("p2_"):
        purpose = stem
        seeds = ((intent.get("banks") or {}).get(purpose) or {}).get("seeds", [])
        if doc["kind"] != purpose or doc["seed_or_seeds"] != seeds:
            raise Invalid(f"{rel} seed list does not bind registered bank")
        data_seeds = doc["data"].get("actual_world_seeds")
        worlds = doc["data"].get("worlds")
        expected_mixes = (
            (["frozen"] * 3 + ["scripted"] * 3 + ["mixed"] * 3)
            if purpose == "p2_h5000"
            else ["frozen", "scripted", "mixed"]
        )
        if data_seeds != seeds or not isinstance(worlds, list) or len(worlds) != len(seeds):
            raise Invalid(f"{rel} P2 worlds do not bind registered seed list")
        for row, seed, mix in zip(worlds, seeds, expected_mixes):
            if (
                not isinstance(row, Mapping)
                or row.get("world_seed") != seed
                or row.get("mix") != mix
            ):
                raise Invalid(f"{rel} P2 world seed/mix assignment is wrong")
    else:
        purpose = stem
        if doc["kind"] != purpose:
            raise Invalid(f"{rel} wrapper kind does not match filename")
        banks = intent.get("banks") or {}
        bank_purpose = "rollback" if purpose == "defaults" else purpose
        registered = (banks.get(bank_purpose) or {}).get("seeds", [])
        expected_seed = registered[0] if purpose in {"controls", "confidence", "ui"} else registered
        if bank_purpose in banks and doc["seed_or_seeds"] != expected_seed:
            raise Invalid(f"{rel} wrapper seeds do not bind registered bank")
    return doc


def _failure(pass_: bool, *items: str) -> Dict[str, Any]:
    return {"pass": bool(pass_), "failures": [item for item in items if item]}


def _identity(data: Mapping[str, Any]) -> List[str]:
    served = data.get("served") or data.get("identity") or {}
    registry = served.get("registry", served) if isinstance(served, Mapping) else {}
    veto = served.get("veto", {}) if isinstance(served, Mapping) else {}
    failures = []
    if registry.get("name") != CHECKPOINT_NAME or registry.get("sha256") != CHECKPOINT_SHA256:
        failures.append("served registry is not b8-s3 pinned bytes")
    if (
        not isinstance(veto, Mapping)
        or veto.get("active") is not False
        or veto.get("wrapped_snake_ids", [])
    ):
        failures.append("served candidate has an active/wrapped veto")
    return failures


def _p2_view(value: Any) -> Any:
    """The registered P2 comparison: remove only the documented SIMD provenance."""
    simd_only = {
        "world_runtime_spec",
        "world_runtime_spec_digest",
        "vector61_policy",
        "veto_diagnostics",
        "ego2s_hero",
    }
    if isinstance(value, Mapping):
        return {
            key: _p2_view(item)
            for key, item in value.items()
            if key not in simd_only and key != "wall"
        }
    if isinstance(value, list):
        return [_p2_view(item) for item in value]
    return value


def _criterion_1(
    results: Mapping[str, Mapping[str, Any]], validation: Mapping[str, Any]
) -> Dict[str, Any]:
    failures: List[str] = []
    for name, expected_count, horizon in (("p2_h5000.json", 9, 5000), ("p2_h10000.json", 3, 10000)):
        data = results[name]["data"]
        worlds = data.get("worlds")
        if not isinstance(worlds, list) or len(worlds) != expected_count:
            failures.append(f"{name} lacks {expected_count} raw worlds")
            continue
        for world in worlds:
            if (
                not isinstance(world, Mapping)
                or world.get("identical") is not True
                or world.get("horizon") != horizon
            ):
                failures.append(f"{name} has non-identical or wrong-horizon P2 world")
                break
            comparison = world.get("comparison")
            if not isinstance(comparison, Mapping) or comparison.get("identical") is not True:
                failures.append(f"{name} lacks raw record comparison")
                break
            live, simd = world.get("live"), world.get("simd")
            if (
                not isinstance(live, Mapping)
                or not isinstance(simd, Mapping)
                or "record" not in live
            ):
                failures.append(f"{name} lacks full raw live/SIMD records")
                break
            if _canonical(_p2_view(live["record"])) != _canonical(_p2_view(simd)):
                failures.append(f"{name} raw P2 records differ after only SIMD exclusions")
                break
    for name in ("parity-000.json", "parity-001.json"):
        data = results[name]["data"]
        if data.get("pass") is not True or data.get("horizon") != 5000:
            failures.append(f"{name} session parity did not pass at H5000")
            continue
        if (
            data.get("frames_compared") != 5000
            or data.get("session_frame_count") != 5000
            or data.get("simd_frame_count") != 5000
        ):
            failures.append(f"{name} lacks 5000-frame trace counts")
        session, simd = data.get("session", {}), data.get("simd", {})
        left = session.get("frame_digests") if isinstance(session, Mapping) else None
        right = simd.get("frame_digests") if isinstance(simd, Mapping) else None
        if (
            not isinstance(left, list)
            or not isinstance(right, list)
            or len(left) != 5000
            or len(right) != 5000
        ):
            failures.append(f"{name} lacks retained 5000-cell traces")
            continue
        first = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
        left_sha = hashlib.sha256(_canonical(left).encode("utf-8")).hexdigest()
        right_sha = hashlib.sha256(_canonical(right).encode("utf-8")).hexdigest()
        if (
            first is not None
            or data.get("first_differing_frame") is not None
            or data.get("session_trace_sha256") != left_sha
            or data.get("simd_trace_sha256") != right_sha
        ):
            failures.append(f"{name} trace digest diverged")
    tests = validation.get("tests", validation)
    for required in ("tests/test_ego_live_bv8.py", "tests/test_ego_live_identity.py"):
        row = tests.get(required) if isinstance(tests, Mapping) else None
        if not isinstance(row, Mapping) or not _verified_log(row) or row.get("exit_code") != 0:
            failures.append(f"lockstep test missing/failed: {required}")
    commands = validation.get("commands")
    expected_commands = {"coretests", "harnesstests", "flake8", "frontendbuild", "frontendtests"}
    if not isinstance(commands, list) or not expected_commands <= {
        row.get("name") for row in commands if isinstance(row, Mapping)
    }:
        failures.append("validation lacks required core/harness/lint/frontend commands")
    else:
        for row in commands:
            if isinstance(row, Mapping) and row.get("name") in expected_commands:
                if row.get("exit_code") != 0 or not _verified_log(row):
                    failures.append(f"validation command failed/unbound: {row.get('name')}")
    return _failure(not failures, *failures)


def _verified_log(row: Mapping[str, Any]) -> bool:
    """A command result only counts when its recorded log bytes still match."""
    path_value, digest = row.get("log_path"), row.get("log_sha256")
    if not isinstance(path_value, str) or not _is_hex(digest):
        return False
    path = Path(path_value)
    return path.is_file() and _sha(path) == digest


def _criterion_2(
    results: Mapping[str, Mapping[str, Any]], intent: Mapping[str, Any]
) -> Dict[str, Any]:
    failures: List[str] = []
    pin = intent.get("strict_pin", {})
    if not isinstance(pin, Mapping) or pin.get("veto_variant") != "none":
        failures.append("intent pin does not bind veto_variant none")
    watch = [results[f"records/watch-{i:03d}.json"]["data"] for i in range(25)]
    for record in watch:
        failures.extend(_identity(record))
        veto = (record.get("served") or {}).get("veto") or {}
        if "applies only to a vector61 Apex policy" not in str(veto.get("reason", "")):
            failures.append("Watch no-veto reason is absent")
            break
    defaults = results["defaults.json"]["data"]
    evidence = defaults
    if not isinstance(evidence, Mapping) or CHECKPOINT_SHA256 in evidence.get(
        "v8_gated_sha256s", []
    ):
        failures.append("v8 gated-checkpoint guard is not proven absent")
    return _failure(not failures, *failures)


def _criterion_3(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    data = results["defaults.json"]["data"]
    matrix = data.get("cases", data)
    failures: List[str] = []
    required = {
        "empty_env",
        "champion",
        "frp3-s12",
        "b8-s3",
        "unknown",
        "unfilled",
        "both_unfilled",
        "corrupt_bytes",
        "v7_no_name",
    }
    if not isinstance(matrix, Mapping):
        return _failure(False, "defaults matrix missing")
    if not required <= set(matrix):
        failures.append("defaults matrix omits required registry/fallback case")
    expected = {
        "empty_env": ("frp3-s12", True),
        "champion": ("champion", True),
        "frp3-s12": ("frp3-s12", True),
        "b8-s3": ("b8-s3", False),
        "unknown": ("frp3-s12", True),
        "unfilled": ("frp3-s12", True),
        "both_unfilled": ("champion", True),
        "corrupt_bytes": ("frp3-s12", True),
        "v7_no_name": ("champion", True),
    }
    for name, (checkpoint, veto_active) in expected.items():
        row = matrix.get(name, {})
        watch = row.get("watch", {}) if isinstance(row, Mapping) else {}
        choice = watch.get("choice", {}) if isinstance(watch, Mapping) else {}
        veto = watch.get("veto", {}) if isinstance(watch, Mapping) else {}
        if (
            choice.get("name") != checkpoint
            or not isinstance(veto, Mapping)
            or veto.get("active") is not veto_active
        ):
            failures.append(f"defaults case has wrong checkpoint/veto: {name}")
        if name in ("champion", "frp3-s12") and watch.get("frames_stepped") != 100:
            failures.append(f"live rollback lacks 100 frames: {name}")
    return _failure(not failures, *failures)


def _criterion_4(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    data = results["latency.json"]["data"]
    failures: List[str] = []
    pre = data.get("preconditions", {})
    tmux = pre.get("tmux", {}) if isinstance(pre, Mapping) else {}
    complete_tmux = (
        isinstance(tmux, Mapping)
        and tmux.get("present") is True
        and tmux.get("workload_panes_exited") is True
        and isinstance(tmux.get("panes"), list)
        and bool(tmux["panes"])
        and all(
            isinstance(pane, Mapping) and str(pane.get("dead")) == "1" for pane in tmux["panes"]
        )
    )
    if not isinstance(tmux, Mapping) or (tmux.get("present") is not False and not complete_tmux):
        failures.append("quiet-machine tmux evidence is absent")
    cases = data.get("cases")
    expected = {"Watch6", "Watch12", "Play6", "Play12"}
    if (
        not isinstance(cases, list)
        or {row.get("case") for row in cases if isinstance(row, Mapping)} != expected
    ):
        failures.append("latency cases are incomplete")
    else:
        for row in cases:
            samples = row.get("samples", {})
            update = samples.get("update_ms", []) if isinstance(samples, Mapping) else []
            tick_values = samples.get("tick_ms", []) if isinstance(samples, Mapping) else []
            if (
                not isinstance(update, list)
                or not isinstance(tick_values, list)
                or len(update) < 1500
                or len(tick_values) < 1500
            ):
                failures.append(f"{row.get('case')} raw samples incomplete")
                continue
            ordered = sorted(float(x) for x in update if _num(x))
            if len(ordered) != len(update) or _percentile(ordered, 0.95) >= 66.667:
                failures.append(f"{row.get('case')} p95 fails 66.667ms")
            failures.extend(
                f"{row.get('case')} {item}" for item in _pacing_failures(row, tick_values)
            )
    return _failure(not failures, *failures)


def _speed_bound(samples: Sequence[object]) -> int:
    maximum = max(float(value) for value in samples)
    return max(fps for fps in range(1, 61) if maximum < 1000.0 / fps)


def _paced_samples(attempt: Mapping[str, Any]) -> Any:
    return attempt.get("tick_samples_ms", attempt.get("raw_tick_ms", attempt.get("samples", [])))


def _pacing_failures(row: Mapping[str, Any], unpaced: Sequence[object]) -> List[str]:
    """Re-derive the permitted one-time pacing retry from raw tick samples."""
    attempts = row.get("pacing_attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) not in (1, 2)
        or not all(isinstance(a, Mapping) for a in attempts)
    ):
        return ["pacing attempts missing"]
    initial = _speed_bound(unpaced)
    expected = initial
    for index, attempt in enumerate(attempts):
        samples = _paced_samples(attempt)
        requested = attempt.get("requested_fps")
        if (
            requested != expected
            or attempt.get("ticks") != 120
            or not isinstance(samples, list)
            or len(samples) != 120
            or not all(_num(v) for v in samples)
        ):
            return ["paced attempt does not bind raw request/samples"]
        actual = sum(float(value) >= 1000.0 / float(requested) for value in samples)
        if attempt.get("overruns") != actual:
            return ["paced overrun count is not raw-derived"]
        if actual == 0:
            if index != len(attempts) - 1:
                return ["successful paced attempt was retried"]
            if row.get("ui_speed_bound_fps") != requested:
                return ["final UI speed bound is not the validated request"]
            return []
        if index != 0 or len(attempts) != 2:
            return ["paced retry count exceeds the one permitted retry"]
        expected = _speed_bound(samples)
        if expected >= requested:
            return ["paced retry did not lower from the observed overrun"]
    return ["final paced attempt still overruns"]


def _criterion_5(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    decisions = deaths = frames = 0
    for i in range(25):
        data = results[f"records/watch-{i:03d}.json"]["data"]
        failures.extend(_identity(data))
        hero, roster = data.get("hero", {}), data.get("roster", {})
        if data.get("frames_stepped") != 5000 or data.get("last_error") is not None:
            failures.append(f"watch {i} did not complete H5000 cleanly")
        if not isinstance(roster, Mapping) or roster.get("all_snakes_b8") is not True:
            failures.append(f"watch {i} roster is not all-B8")
        if not isinstance(hero, Mapping):
            failures.append(f"watch {i} lacks hero monitoring")
            continue
        decisions += int(hero.get("decisions", 0))
        deaths += int(hero.get("deaths", 0))
        frames += 5000
        if not isinstance(hero.get("decision_length_histogram"), Mapping):
            failures.append(f"watch {i} lacks length histogram")
    if frames != 125000 or decisions <= 0:
        failures.append("watch aggregate lacks 125000 stepped frames/decisions")
    return _failure(not failures, *failures)


def _criterion_6(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    for i in range(25):
        data = results[f"records/play-{i:03d}.json"]["data"]
        failures.extend(_identity(data))
        play, dispatch = data.get("play", {}), data.get("dispatch", {})
        if (
            data.get("last_error") is not None
            or not isinstance(play, Mapping)
            or play.get("opponents_unwrapped") is not True
        ):
            failures.append(f"play {i} has error/wrapped opponent")
        if (
            not isinstance(dispatch, Mapping)
            or dispatch.get("human_dispatched")
            or dispatch.get("human_observed")
        ):
            failures.append(f"play {i} human entered AI dispatch/observation")
    controls = results["controls.json"]["data"]
    if not all(controls.get("play_opponents", {}).get(str(n)) == n for n in (1, 5, 11)):
        failures.append("play opponent controls missing")
    score = controls.get("score_path", controls.get("score", {}))
    if (
        not isinstance(score, Mapping)
        or score.get("submit_status") != 200
        or score.get("player_status") != 200
    ):
        failures.append("isolated score endpoint evidence failed")
    if CHECKPOINT_NAME not in _canonical(score):
        failures.append("scored row does not retain b8-s3 identity")
    return _failure(not failures, *failures)


def _criterion_7(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    data = results["confidence.json"]["data"]
    buckets = data.get("buckets", {})
    vectors = data.get("q_values", data.get("q_vectors", []))
    recomputed = {"confident": 0, "moderate": 0, "close": 0}
    if not isinstance(vectors, list) or len(vectors) != 1500:
        failures.append("confidence raw Q vectors are absent")
    else:
        for vector in vectors:
            finite = (
                sorted((float(x) for x in vector if _num(x)), reverse=True)
                if isinstance(vector, list)
                else []
            )
            relative = (
                0.0
                if len(finite) < 2 or finite[0] - finite[-1] <= 1e-9
                else (finite[0] - finite[1]) / (finite[0] - finite[-1])
            )
            recomputed[
                "confident" if relative > 0.25 else "moderate" if relative > 0.10 else "close"
            ] += 1
    if (
        data.get("frames_with_q_values") != 1500
        or not isinstance(buckets, Mapping)
        or sum(int(v) for v in buckets.values()) != 1500
    ):
        failures.append("confidence sample is not exactly 1500 usable Q vectors")
    if sum(int(v) > 0 for v in buckets.values()) < 2:
        failures.append("confidence split has fewer than two populated buckets")
    if isinstance(vectors, list) and buckets != recomputed:
        failures.append("confidence buckets do not rederive from raw Q vectors")
    no_veto = data.get("no_veto", {})
    if not isinstance(no_veto, Mapping) or no_veto.get("absent") is not True:
        failures.append("confidence session observed a veto")
    ui = results["ui.json"]["data"]
    screenshot = Path(str(ui.get("screenshot_path", ""))) if isinstance(ui, Mapping) else Path()
    if (
        not isinstance(ui, Mapping)
        or ui.get("browser_observed") is not True
        or not _is_hex(ui.get("screenshot_sha256"))
        or not screenshot.is_file()
        or _sha(screenshot) != ui.get("screenshot_sha256")
    ):
        failures.append("external browser/screenshot evidence missing")
    backend = Path(str(ui.get("backend_path", ""))) if isinstance(ui, Mapping) else Path()
    if (
        not isinstance(ui, Mapping)
        or not _is_hex(ui.get("backend_sha256"))
        or not backend.is_file()
        or _sha(backend) != ui.get("backend_sha256")
        or ui.get("intent_sha256") != results["ui.json"].get("intent_sha256")
        or ui.get("seed") != results["ui.json"].get("seed_or_seeds")
        or ui.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or ui.get("veto_active") is not False
    ):
        failures.append("browser backend identity evidence missing or mismatched")
    checks = ui.get("checks", {}) if isinstance(ui, Mapping) else {}
    if not isinstance(checks, Mapping) or not all(
        checks.get(key) is True
        for key in ("narrator", "steering_wheel", "ego2s_bv8_groups", "no_veto_counters")
    ):
        failures.append("browser UI checks incomplete")
    return _failure(not failures, *failures)


def _criterion_8(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    controls = results["controls.json"]["data"]
    train = controls.get("train", controls.get("train_refusal", {}))
    ok = (
        isinstance(train, Mapping)
        and train.get("mode") == "watch"
        and train.get("policy_training") is False
        and "forward" in str(train.get("last_error", "")).lower()
    )
    return _failure(ok, "B8 Train was not refused to forward-only Watch" if not ok else "")


def audit(root: str | Path) -> Dict[str, Any]:
    """Audit one completed T12 run directory without running a workload."""
    root = Path(root)
    try:
        if _sha(PROTOCOL) != PROTOCOL_SHA256:
            raise Invalid("local protocol bytes do not match the audit pin")
        intent_path = root / "intent.json"
        intent = _json(intent_path)
        if not isinstance(intent, Mapping):
            raise Invalid("intent.json is not an object")
        intent_sha = _sha(intent_path)
        if intent.get("protocol_sha256") != PROTOCOL_SHA256:
            raise Invalid("intent does not pin the protocol")
        if intent.get("smoke") is True:
            return {
                "schema": AUDIT_SCHEMA,
                "verdict": "SMOKE_ONLY",
                "criteria": {},
                "serving_path_qualified": False,
            }
        if intent.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            # The current producer records this under intent.checkpoint; retain this
            # compatibility only while still requiring the exact nested binding below.
            if (intent.get("checkpoint") or {}).get("sha256") != CHECKPOINT_SHA256:
                raise Invalid("intent does not pin b8-s3")
        receipt = _json(root / "receipt.json")
        if not isinstance(receipt, Mapping) or receipt.get("intent_sha256") != intent_sha:
            raise Invalid("receipt does not bind intent bytes")
        files = receipt.get("files")
        if not isinstance(files, Mapping):
            raise Invalid("receipt lacks file inventory")
        expected = _expected_files()
        if set(files) != expected:
            raise Invalid("receipt inventory has omissions or unexpected files")
        for rel, digest in files.items():
            path = root / rel
            if not _is_hex(digest) or not path.is_file() or _sha(path) != digest:
                raise Invalid(f"receipt hash mismatch: {rel}")
        binding_problems = _intent_bindings(intent)
        if binding_problems:
            raise Invalid("; ".join(binding_problems))
        # validation is a hash-bound command/log manifest, not a gameplay result.
        validation = _json(root / "validation.json")
        if not isinstance(validation, Mapping):
            raise Invalid("validation.json is not an object")
        if intent.get("validation_sha256") != _sha(root / "validation.json"):
            raise Invalid("intent does not bind validation.json bytes")
        results = {
            rel: _result(root, rel, intent_sha, intent)
            for rel in expected
            if rel not in {"intent.json", "validation.json"}
        }
        criteria = {
            "1": _criterion_1(results, validation),
            "2": _criterion_2(results, intent),
            "3": _criterion_3(results),
            "4": _criterion_4(results),
            "5": _criterion_5(results),
            "6": _criterion_6(results),
            "7": _criterion_7(results),
            "8": _criterion_8(results),
        }
        verdict = (
            "SERVING_PASS" if all(row["pass"] for row in criteria.values()) else "SERVING_FAIL"
        )
        return {
            "schema": AUDIT_SCHEMA,
            "verdict": verdict,
            "intent_sha256": intent_sha,
            "protocol_sha256": PROTOCOL_SHA256,
            "criteria": criteria,
            "serving_path_qualified": verdict == "SERVING_PASS",
        }
    except (Invalid, TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
        return {
            "schema": AUDIT_SCHEMA,
            "verdict": "INVALID",
            "criteria": {},
            "error": str(exc),
            "serving_path_qualified": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return {"SERVING_PASS": 0, "SERVING_FAIL": 1, "INVALID": 2, "SMOKE_ONLY": 0}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
