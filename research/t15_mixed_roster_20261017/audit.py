"""Independent, create-only audit for the T15 mixed Watch qualification.

The producer is intentionally not imported here.  This module consumes the
intent, result envelopes, and receipt written by :mod:`run`, then re-derives
the six qualification checks from the retained values.  Missing, corrupt, or
unbound evidence is ``INVALID``; a completed criterion with a bad value is
``SERVING_FAIL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "protocol.md"
PROTOCOL_SHA256 = "0b882fc2a11e47ce67a0ec541f560d157bd32c6e62c95ce9554683633603428d"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
NAMESPACE = "t15-mixed-roster-20261017/v1"
RESULT_SCHEMA = "t15-mixed-roster-result/v1"
INTENT_SCHEMA = "t15-mixed-roster-intent/v1"
RECEIPT_SCHEMA = "t15-mixed-roster-receipt/v1"
AUDIT_SCHEMA = "t15-mixed-roster-audit/v1"
P95_LIMIT_MS = 66.667
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Invalid(RuntimeError):
    """The retained evidence cannot be bound to a T15 run."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Invalid(f"unreadable JSON {path}: {exc}") from exc


def _hex(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def _num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _failure(*items: str) -> dict[str, Any]:
    return {"pass": not items, "failures": list(items)}


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _same_number(left: Any, right: float) -> bool:
    return _num(left) and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-9)


def _seed(namespace: str, purpose: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{namespace}|{purpose}|{index}".encode()).digest()[:4], "big"
    )


def _seed_digest(values: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in sorted(set(values))).encode()
    ).hexdigest()


def _expected_banks() -> dict[str, dict[str, Any]]:
    counts = {"watch": 25, "parity": 3, "latency": 3, "ui": 1, "controls": 1}
    return {
        purpose: {
            "count": count,
            "seeds": [_seed(NAMESPACE, purpose, i) for i in range(count)],
            "seed_digest": _seed_digest([_seed(NAMESPACE, purpose, i) for i in range(count)]),
        }
        for purpose, count in counts.items()
    }


def _path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Invalid(f"{label} path is absent")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _source_bindings(intent: Mapping[str, Any]) -> None:
    sources = intent.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise Invalid("intent.sources is empty")
    required = {
        "web/backend/served_checkpoint.py",
        "web/backend/safety_veto_serving.py",
        "web/backend/session.py",
        "web/backend/serving_speed.py",
        "research/t15_mixed_roster_20261017/run.py",
        "research/t15_mixed_roster_20261017/episodes.py",
        "research/t15_mixed_roster_20261017/measurements.py",
        "research/t15_mixed_roster_20261017/parity.py",
        "research/t15_mixed_roster_20261017/seed_audit.py",
        "research/t15_mixed_roster_20261017/audit.py",
        "research/t15_mixed_roster_20261017/controls.py",
        "research/t15_mixed_roster_20261017/ui_server.py",
    }
    missing = required - set(sources)
    if missing:
        raise Invalid(f"intent source closure omits: {sorted(missing)}")
    dist_sources = {str(path) for path in sources if str(path).startswith("web/frontend/dist/")}
    if "web/frontend/dist/index.html" not in dist_sources or not dist_sources:
        raise Invalid("intent source closure omits the served frontend dist/index.html")
    for relative, digest in sources.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts or not _hex(digest):
            raise Invalid(f"unsafe or unhashed source binding: {relative}")
        path = REPO / relative_path
        if not path.is_file() or _sha(path) != digest:
            raise Invalid(f"registered source changed: {relative}")


def _intent(root: Path) -> tuple[Mapping[str, Any], str]:
    path = root / "intent.json"
    intent = _json(path)
    if not isinstance(intent, Mapping):
        raise Invalid("intent.json is not an object")
    intent_sha = _sha(path)
    if intent.get("schema") != INTENT_SCHEMA or intent.get("namespace") != NAMESPACE:
        raise Invalid("intent schema or namespace is wrong")
    if intent.get("smoke") is True:
        return intent, intent_sha
    if intent.get("protocol_sha256") != PROTOCOL_SHA256 or _sha(PROTOCOL) != PROTOCOL_SHA256:
        raise Invalid("protocol bytes are not independently pinned")
    git = intent.get("git")
    if not isinstance(git, Mapping) or not re.fullmatch(
        r"[0-9a-f]{40}", str(git.get("commit", ""))
    ):
        raise Invalid("intent lacks a valid git commit")
    if git.get("dirty") not in (False, "", []):
        raise Invalid("intent was created from a dirty tracked tree")
    _source_bindings(intent)
    producer = intent.get("producer")
    if not isinstance(producer, Mapping):
        raise Invalid("intent producer binding is absent")
    producer_path = _path(producer.get("path"), REPO, "producer")
    if (
        not _hex(producer.get("sha256"))
        or not producer_path.is_file()
        or _sha(producer_path) != producer["sha256"]
    ):
        raise Invalid("producer bytes differ from intent pin")
    if intent.get("protocol_sha256") != PROTOCOL_SHA256:
        raise Invalid("intent protocol pin is wrong")
    release_env = intent.get("release_env")
    if (
        not isinstance(release_env, Mapping)
        or release_env.get("SNAKE_SERVE_CHECKPOINT") not in (None, "b8-s3")
        or release_env.get("SNAKE_SERVE_WATCH_ROSTER") != "mixed"
        or any(
            release_env.get(key) is not None
            for key in (
                "SNAKE_SERVE_VETO_VARIANT",
                "SNAKE_SERVE_VETO_WATCH_HERO",
                "SNAKE_SERVE_VETO_PLAY_AI",
            )
        )
    ):
        raise Invalid("intent release environment changed")
    if release_env != {
        "SNAKE_SERVE_CHECKPOINT": release_env.get("SNAKE_SERVE_CHECKPOINT"),
        "SNAKE_SERVE_WATCH_ROSTER": "mixed",
        "SNAKE_SERVE_VETO_VARIANT": None,
        "SNAKE_SERVE_VETO_WATCH_HERO": None,
        "SNAKE_SERVE_VETO_PLAY_AI": None,
    }:
        raise Invalid("intent release environment has unexpected keys")
    registry = intent.get("registry")
    if not isinstance(registry, Mapping):
        raise Invalid("intent registry binding is absent")
    if (
        registry.get("released_default") != "b8-s3"
        or registry.get("watch_roster_default") != "all_b8"
    ):
        raise Invalid("serving default or Watch default changed")
    if not _hex(registry.get("pins_sha256")):
        raise Invalid("registry pins hash is absent")
    for label, expected_name, expected_sha in (
        ("b8", "b8-s3", B8_SHA256),
        ("frp3", "frp3-s12", FRP3_SHA256),
    ):
        entry = registry.get(label)
        if (
            not isinstance(entry, Mapping)
            or entry.get("name") != expected_name
            or entry.get("sha256") != expected_sha
        ):
            raise Invalid(f"intent {label} checkpoint pin is wrong")
        checkpoint = _path(entry.get("path"), REPO, f"{label} checkpoint")
        if _sha(checkpoint) != expected_sha:
            raise Invalid(f"{label} checkpoint bytes differ from pin")
    banks = intent.get("banks")
    if banks != _expected_banks() or intent.get("counts") != {
        "watch": 25,
        "parity": 3,
        "latency": 3,
        "ui": 1,
        "controls": 1,
    }:
        raise Invalid("intent seed banks or counts do not match protocol")
    seed_audit = intent.get("seed_audit")
    if not isinstance(seed_audit, Mapping) or not _hex(seed_audit.get("sha256")):
        raise Invalid("seed audit binding is absent")
    seed_path = _path(seed_audit.get("path"), REPO, "seed audit")
    if not seed_path.is_file() or _sha(seed_path) != seed_audit["sha256"]:
        raise Invalid("seed audit bytes differ from intent")
    seed_doc = _json(seed_path)
    expected_banks = _expected_banks()
    expected_allocation = {
        "namespace": NAMESPACE,
        "recipe": "uint32_be(sha256(f'{namespace}|{purpose}|{i}')[:4])",
        "banks": expected_banks,
        "count": 33,
        "seed_digest": _seed_digest(
            [value for row in expected_banks.values() for value in row["seeds"]]
        ),
    }
    if (
        not isinstance(seed_doc, Mapping)
        or seed_doc.get("pass") is not True
        or seed_doc.get("allocation") != expected_allocation
    ):
        raise Invalid("seed freshness evidence does not bind exact registered banks")
    validation = root / "validation.json"
    if (
        not validation.is_file()
        or not _hex(intent.get("validation_sha256"))
        or _sha(validation) != intent["validation_sha256"]
    ):
        raise Invalid("validation evidence is absent or changed")
    return intent, intent_sha


def _receipt(root: Path, intent_sha: str) -> None:
    receipt = _json(root / "receipt.json")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("intent_sha256") != intent_sha
    ):
        raise Invalid("receipt does not bind the intent")
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise Invalid("receipt file inventory is absent")
    required = {
        "intent.json",
        "validation.json",
        "records/controls-000.json",
        "records/ui-000.json",
    }
    required |= {f"records/watch-{i:03d}.json" for i in range(25)}
    required |= {f"records/parity-{i:03d}.json" for i in range(3)}
    required |= {f"records/latency-{i:03d}.json" for i in range(3)}
    if not required <= set(files):
        raise Invalid(f"receipt omits required evidence: {sorted(required - set(files))}")
    for relative, digest in files.items():
        relative_path = Path(str(relative))
        path = root / relative_path
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not _hex(digest)
            or not path.is_file()
            or _sha(path) != digest
        ):
            raise Invalid(f"receipt hash mismatch: {relative}")


def _envelope(
    root: Path, relative: str, intent_sha: str, kind: str, index: int, seed: int
) -> Mapping[str, Any]:
    doc = _json(root / relative)
    if (
        not isinstance(doc, Mapping)
        or doc.get("schema") != RESULT_SCHEMA
        or doc.get("intent_sha256") != intent_sha
    ):
        raise Invalid(f"{relative} envelope is not bound to this intent")
    required = {
        "schema",
        "intent_sha256",
        "kind",
        "index",
        "seed",
        "status",
        "error",
        "started_utc",
        "wall_seconds",
        "data",
    }
    if set(doc) != required:
        raise Invalid(f"{relative} envelope keys differ")
    if doc.get("kind") != kind or doc.get("index") != index or doc.get("seed") != seed:
        raise Invalid(f"{relative} kind/index/seed is wrong")
    if (
        doc.get("status") != "complete"
        or doc.get("error") is not None
        or not isinstance(doc.get("data"), Mapping)
    ):
        raise Invalid(f"{relative} is incomplete or errored")
    if not _num(doc.get("wall_seconds")) or float(doc["wall_seconds"]) < 0:
        raise Invalid(f"{relative} wall_seconds is invalid")
    return doc


def _roster_failures(data: Mapping[str, Any], snakes: int) -> list[str]:
    failures: list[str] = []
    roster = data.get("roster")
    served = roster.get("served", {}) if isinstance(roster, Mapping) else {}
    opponent = roster.get("opponent_checkpoint", {}) if isinstance(roster, Mapping) else {}
    veto = roster.get("veto", {}) if isinstance(roster, Mapping) else {}
    if not isinstance(roster, Mapping) or roster.get("watch_roster") != "mixed":
        failures.append("mixed Watch selector is absent")
    if (
        not isinstance(served, Mapping)
        or served.get("name") != "b8-s3"
        or served.get("sha256") != B8_SHA256
    ):
        failures.append("hero registry identity is not b8-s3")
    if (
        not isinstance(opponent, Mapping)
        or opponent.get("name") != "frp3-s12"
        or opponent.get("sha256") != FRP3_SHA256
    ):
        failures.append("opponent registry identity is not frp3-s12")
    expected = list(range(1, snakes))
    if (
        not isinstance(veto, Mapping)
        or veto.get("active") is not True
        or veto.get("scope") != "watch_opponents"
        or veto.get("wrapped_snake_ids") != expected
    ):
        failures.append("veto scope is not opponent-only")
    counters = veto.get("counters") if isinstance(veto, Mapping) else None
    if (
        not isinstance(counters, Mapping)
        or set(counters) != {str(i) for i in expected}
        or any(str(i) in counters for i in (0,))
    ):
        failures.append("veto counters include the hero or omit an opponent")
    snakes_data = roster.get("snakes") if isinstance(roster, Mapping) else None
    if not isinstance(snakes_data, list) or len(snakes_data) != snakes:
        failures.append("roster snake records are incomplete")
    else:
        hero, opponents = snakes_data[0], snakes_data[1:]
        if hero.get("safety_veto") is not None or hero.get("veto_method") is not None:
            failures.append("hero is wrapped")
        if hero.get("policy_is_hero") is not True or hero.get("policy_is_opponent") is not False:
            failures.append("hero policy identity is not independently recorded")
        if any(
            row.get("policy_is_hero") is not False or row.get("policy_is_opponent") is not True
            for row in opponents
        ):
            failures.append("opponent policy identity is not independently recorded")
        if any(
            row.get("veto_method") != V8_METHOD
            or row.get("veto_reference_lambda") is not None
            or row.get("veto_head_avoidance") is not True
            for row in opponents
        ):
            failures.append("opponent wrapper is not exact v8")
    wrapper = veto.get("wrapper") if isinstance(veto, Mapping) else None
    if (
        not isinstance(wrapper, Mapping)
        or veto.get("strict_receipt_checkpoint_match") is not True
        or veto.get("strict_receipt_wrapper_match") is not True
        or wrapper.get("method") != V8_METHOD
        or not _hex(wrapper.get("source_sha256"))
    ):
        failures.append("opponent wrapper source or strict receipt identity is absent")
    return failures


def _check_identity(root: Path, intent: Mapping[str, Any], intent_sha: str) -> dict[str, Any]:
    failures: list[str] = []
    banks = intent["banks"]
    for index, (horizon, snakes) in enumerate(((5000, 6), (5000, 8), (10000, 12))):
        row = _envelope(
            root,
            f"records/parity-{index:03d}.json",
            intent_sha,
            "parity",
            index,
            banks["parity"]["seeds"][index],
        )
        data = row["data"]
        if (
            data.get("schema_version") != "t15-mixed-watch-identity/v1"
            or data.get("horizon") != horizon
            or data.get("num_snakes") != snakes
        ):
            raise Invalid(f"parity-{index:03d} identity metadata is wrong")
        checks = data.get("checks")
        if not isinstance(checks, Mapping):
            raise Invalid(f"parity-{index:03d} lacks role comparisons")
        evidence = data.get("evidence")
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("live"), Mapping)
            or not isinstance(evidence.get("eval"), Mapping)
        ):
            raise Invalid(f"parity-{index:03d} lacks raw live/eval streams")
        expected_slots = set(range(1, snakes))
        for side_name in ("live", "eval"):
            side = evidence[side_name]
            hero_stream = side.get("hero_ego2s")
            opponent_stream = side.get("opponent_vector61")
            hero_active = side.get("hero_active")
            active_opponent_slots = side.get("active_opponent_slots")
            if (
                not isinstance(hero_stream, list)
                or not isinstance(opponent_stream, list)
                or not isinstance(hero_active, list)
                or not isinstance(active_opponent_slots, list)
                or len(hero_active) != horizon
                or len(active_opponent_slots) != horizon
            ):
                raise Invalid(f"parity-{index:03d} {side_name} active-slot evidence is absent")
            for frame, (hero_packet, is_hero_active, opponent_frame, declared_slots) in enumerate(
                zip(hero_stream, hero_active, opponent_stream, active_opponent_slots)
            ):
                if not isinstance(is_hero_active, bool) or is_hero_active is not (
                    hero_packet is not None
                ):
                    failures.append(
                        f"parity-{index:03d} {side_name} hero active flag "
                        f"disagrees at frame {frame}"
                    )
                if not isinstance(opponent_frame, list) or not isinstance(declared_slots, list):
                    failures.append(
                        f"parity-{index:03d} {side_name} opponent active slots are malformed"
                    )
                    continue
                packet_slots = [
                    packet.get("slot") for packet in opponent_frame if isinstance(packet, Mapping)
                ]
                packet_slots_valid = all(type(slot) is int for slot in packet_slots)
                slot_mismatch = not packet_slots_valid or any(
                    type(slot) is not int for slot in declared_slots
                )
                if not slot_mismatch:
                    slot_mismatch = (
                        declared_slots != sorted(set(declared_slots))
                        or not set(declared_slots) <= expected_slots
                        or declared_slots != sorted(packet_slots)
                    )
                if slot_mismatch:
                    failures.append(
                        f"parity-{index:03d} {side_name} opponent active slots "
                        f"disagree at frame {frame}"
                    )
        for role in ("world", "hero_ego2s", "opponent_vector61"):
            value = checks.get(role)
            if not isinstance(value, Mapping):
                raise Invalid(f"parity-{index:03d} lacks {role} comparison")
            live = evidence["live"].get(role if role != "world" else "frames")
            evaluated = evidence["eval"].get(role if role != "world" else "frames")
            if (
                not isinstance(live, list)
                or not isinstance(evaluated, list)
                or len(live) != horizon
                or len(evaluated) != horizon
            ):
                raise Invalid(f"parity-{index:03d} {role} raw stream length is wrong")
            first = next(
                (
                    position
                    for position, pair in enumerate(zip(live, evaluated))
                    if pair[0] != pair[1]
                ),
                None,
            )
            if first is not None or value.get("first_difference") is not None:
                failures.append(f"parity-{index:03d} {role} differs at frame {first}")
            if (
                not _hex(value.get("live_sha256"))
                or value.get("live_sha256") != _canonical_digest(live)
                or not _hex(value.get("eval_sha256"))
                or value.get("eval_sha256") != _canonical_digest(evaluated)
            ):
                failures.append(f"parity-{index:03d} {role} hash is not raw-derived")
            if (
                value.get("live_count") != len(live)
                or value.get("eval_count") != len(evaluated)
                or value.get("identical")
                is not (first is None and value.get("live_sha256") == value.get("eval_sha256"))
            ):
                failures.append(f"parity-{index:03d} {role} differs")
            if role == "world" and (
                not all(_hex(item) for item in live) or not all(_hex(item) for item in evaluated)
            ):
                failures.append(f"parity-{index:03d} world signatures are not hashes")
            if role == "hero_ego2s":
                for stream in (live, evaluated):
                    packets = [item for item in stream if item is not None]
                    if not packets:
                        raise Invalid(f"parity-{index:03d} hero stream has no decision packets")
                    if any(
                        not isinstance(item, Mapping)
                        or item.get("slot") != 0
                        or not isinstance(item.get("action"), int)
                        or not _hex(item.get("state_sha256"))
                        or not _hex(item.get("q_sha256"))
                        for item in packets
                    ):
                        failures.append(f"parity-{index:03d} hero packet identity is malformed")
            if role == "opponent_vector61":
                if not any(frame for frame in live) or not any(frame for frame in evaluated):
                    raise Invalid(f"parity-{index:03d} opponent stream has no decision packets")
                seen_slots: set[int] = set()
                for frame in live + evaluated:
                    if frame == []:
                        continue
                    if frame is None or not isinstance(frame, list):
                        raise Invalid(f"parity-{index:03d} opponent stream has invalid frame")
                    if any(not isinstance(packet, Mapping) for packet in frame):
                        failures.append(f"parity-{index:03d} opponent packet is malformed")
                        break
                    slots_list = [
                        packet.get("slot") for packet in frame if isinstance(packet, Mapping)
                    ]
                    slots = set(slots_list)
                    if len(slots_list) != len(slots) or not slots <= expected_slots:
                        failures.append(f"parity-{index:03d} opponent slot count differs")
                        break
                    seen_slots |= slots
                    if any(
                        not isinstance(packet.get("action"), int)
                        or not _hex(packet.get("state_sha256"))
                        or not _hex(packet.get("q_sha256"))
                        for packet in frame
                    ):
                        failures.append(f"parity-{index:03d} opponent packet identity is malformed")
                if seen_slots != expected_slots:
                    failures.append(f"parity-{index:03d} opponent stream omits live slots")
        parity_config = data.get("parity_config")
        if not isinstance(parity_config, Mapping) or not _hex(parity_config.get("sha256")):
            raise Invalid(f"parity-{index:03d} parity config binding is absent")
        config_path = _path(parity_config.get("path"), REPO, "parity config")
        if not config_path.is_file() or _sha(config_path) != parity_config["sha256"]:
            raise Invalid(f"parity-{index:03d} parity config bytes changed")
        session_config = _path(parity_config.get("session_config_path"), REPO, "session config")
        if not session_config.is_file():
            raise Invalid(f"parity-{index:03d} session config is absent")
        profile = data.get("eval_profile")
        metadata = data.get("eval_record_metadata")
        if (
            not isinstance(profile, Mapping)
            or not isinstance(profile.get("name"), str)
            or not _hex(profile.get("digest"))
            or not isinstance(profile.get("world"), Mapping)
            or not isinstance(metadata, Mapping)
            or not metadata
        ):
            raise Invalid(f"parity-{index:03d} evaluator profile/record metadata is absent")
        instrumentation = data.get("eval_instrumentation")
        if (
            not isinstance(instrumentation, Mapping)
            or instrumentation.get("source_mutation") is not False
            or "slots 1..N-1 v8" not in str(instrumentation.get("opponent_veto", ""))
        ):
            failures.append(f"parity-{index:03d} evaluator adapter binding is absent")
    return _failure(*failures)


def _watch_rows(root: Path, intent: Mapping[str, Any], intent_sha: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for index in range(25):
        rows.append(
            _envelope(
                root,
                f"records/watch-{index:03d}.json",
                intent_sha,
                "watch",
                index,
                intent["banks"]["watch"]["seeds"][index],
            )["data"]
        )
    return rows


def _check_scope(root: Path, intent: Mapping[str, Any], intent_sha: str) -> dict[str, Any]:
    failures: list[str] = []
    for index, data in enumerate(_watch_rows(root, intent, intent_sha)):
        if data.get("schema_version") != "t15-mixed-watch-episode/v1":
            raise Invalid(f"watch-{index:03d} frame/seed evidence is incomplete")
        failures.extend(f"watch-{index:03d}: {item}" for item in _roster_failures(data, 6))
    return _failure(*failures)


def _check_continuous(root: Path, intent: Mapping[str, Any], intent_sha: str) -> dict[str, Any]:
    failures: list[str] = []
    total_frames = 0
    for index, data in enumerate(_watch_rows(root, intent, intent_sha)):
        if (
            data.get("seed") != intent["banks"]["watch"]["seeds"][index]
            or data.get("horizon") != 5000
            or data.get("frames_stepped") != 5000
        ):
            raise Invalid(f"watch-{index:03d} frame/seed evidence is incomplete")
        total_frames += int(data["frames_stepped"])
        if (
            data.get("last_error") is not None
            or data.get("error") is not None
            or data.get("no_errors") is not True
            or data.get("identity_checked_frames") != 5000
        ):
            failures.append(f"watch-{index:03d}: session reported an error")
        hero = data.get("hero")
        dispatch = data.get("dispatch")
        if (
            not isinstance(hero, Mapping)
            or not isinstance(hero.get("decision_length_histogram"), Mapping)
            or not isinstance(hero.get("decision_length_quantiles"), Mapping)
        ):
            failures.append(f"watch-{index:03d}: hero monitoring baseline is incomplete")
        else:
            histogram = hero["decision_length_histogram"]
            histogram_valid = not any(
                not isinstance(key, str)
                or not key.isdigit()
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in histogram.items()
            )
            if not histogram_valid:
                failures.append(f"watch-{index:03d}: decision histogram has invalid counts")
            decisions = hero.get("decisions")
            deaths = hero.get("deaths")
            alive_frames = hero.get("alive_frames")
            valid_counts = (
                isinstance(decisions, int)
                and not isinstance(decisions, bool)
                and decisions >= 0
                and isinstance(deaths, int)
                and not isinstance(deaths, bool)
                and 0 <= deaths <= decisions
                and isinstance(alive_frames, int)
                and not isinstance(alive_frames, bool)
                and 0 <= alive_frames <= 5000
            )
            if histogram_valid and valid_counts and sum(histogram.values()) != decisions:
                failures.append(f"watch-{index:03d}: decision histogram does not sum to decisions")
            if not valid_counts:
                failures.append(f"watch-{index:03d}: hero counts are invalid")
            elif histogram_valid and valid_counts:
                lengths = [int(length) for length, count in histogram.items() for _ in range(count)]
                if alive_frames != decisions - deaths:
                    failures.append(f"watch-{index:03d}: alive-frame count is not death-derived")
                quantiles = hero["decision_length_quantiles"]
                if not isinstance(quantiles, Mapping):
                    failures.append(f"watch-{index:03d}: decision quantiles are malformed")
                else:
                    ordered = sorted(lengths)
                    expected_quantiles = (
                        {"p50": None, "p95": None, "min": None, "max": None}
                        if not ordered
                        else {
                            "p50": float(ordered[round((len(ordered) - 1) * 0.5)]),
                            "p95": float(ordered[round((len(ordered) - 1) * 0.95)]),
                            "min": float(ordered[0]),
                            "max": float(ordered[-1]),
                        }
                    )
                    if quantiles != expected_quantiles:
                        failures.append(
                            f"watch-{index:03d}: decision quantiles are not histogram-derived"
                        )
        if not isinstance(dispatch, Mapping):
            failures.append(f"watch-{index:03d}: dispatch evidence is absent")
        else:
            ids = dispatch.get("ids")
            observations = dispatch.get("observation_ids")
            alive = dispatch.get("observation_alive")
            decisions = hero.get("decisions") if isinstance(hero, Mapping) else None
            if not isinstance(ids, list) or any(item != 0 for item in ids) or len(ids) != decisions:
                failures.append(f"watch-{index:03d}: dispatch ids are not raw hero decisions")
            if (
                not isinstance(observations, list)
                or not isinstance(alive, list)
                or len(observations) != 5000
                or len(alive) != 5000
            ):
                failures.append(f"watch-{index:03d}: observation lifecycle evidence is incomplete")
            elif (
                any(not isinstance(item, bool) for item in alive)
                or sum(alive) != (hero.get("decisions") if isinstance(hero, Mapping) else -1)
                or any(
                    item != ([0] if is_alive else []) for item, is_alive in zip(observations, alive)
                )
            ):
                failures.append(f"watch-{index:03d}: observation rows contradict hero liveness")
    if total_frames != 125000:
        failures.append("Watch aggregate does not contain exactly 125000 advancing frames")
    return _failure(*failures)


def _paced_failures(
    attempt: Mapping[str, Any], expected_speed: float, expected_ceiling: float, snakes: int
) -> list[str]:
    failures: list[str] = []
    requested = attempt.get("requested_fps")
    actual = attempt.get("actual_speed", attempt.get("accepted_speed"))
    ceiling = attempt.get("speed_ceiling")
    ai_count = attempt.get("total_ai_count")
    raw = attempt.get("raw_tick_ms")
    if attempt.get("warmup") != 200:
        failures.append("paced probe does not retain its 200-frame warmup")
    if (
        requested not in (12, 120)
        or not _same_number(actual, expected_speed)
        or not _same_number(ceiling, expected_ceiling)
        or ai_count != snakes
        or not isinstance(raw, list)
        or len(raw) != 120
        or not all(_num(v) and float(v) >= 0 for v in raw)
    ):
        return [
            "paced evidence does not bind request, accepted speed, ceiling, "
            "AI count, and 120 raw ticks"
        ]
    overrun = sum(float(value) >= 1000.0 / float(actual) for value in raw)
    if overrun:
        failures.append(f"paced request {requested} has {overrun} budget overruns")
    if attempt.get("overruns") != overrun:
        failures.append(f"paced request {requested} overrun count is not raw-derived")
    if attempt.get("zero_overrun") is not (overrun == 0):
        failures.append(f"paced request {requested} zero_overrun is not raw-derived")
    elapsed = attempt.get("elapsed_seconds")
    achieved = attempt.get("actual_achieved_fps")
    if (
        not _num(elapsed)
        or float(elapsed) <= 0
        or not _num(achieved)
        or not math.isclose(float(achieved), 120.0 / float(elapsed), rel_tol=0.0, abs_tol=1e-6)
    ):
        failures.append(f"paced request {requested} cadence is not raw-derived")
    roster = attempt.get("roster")
    if (
        not isinstance(roster, Mapping)
        or roster.get("watch_roster") != "mixed"
        or roster.get("hero_id") != 0
    ):
        failures.append(f"paced request {requested} roster identity is absent")
    return failures


def _check_latency(root: Path, intent: Mapping[str, Any], intent_sha: str) -> dict[str, Any]:
    failures: list[str] = []
    banks = intent["banks"]
    latencies: list[dict[str, Any]] = []
    for index, snakes in enumerate((6, 8, 12)):
        row = _envelope(
            root,
            f"records/latency-{index:03d}.json",
            intent_sha,
            "latency",
            index,
            banks["latency"]["seeds"][index],
        )
        data = row["data"]
        measurement = data.get("measurement", data)
        if (
            not isinstance(measurement, Mapping)
            or measurement.get("schema_version") != "t15-mixed-watch-latency/v1"
            or measurement.get("num_snakes") != snakes
            or measurement.get("warmup") != 200
        ):
            raise Invalid(f"latency-{index:03d} metadata is incomplete")
        samples = measurement.get("samples")
        if (
            not isinstance(samples, Mapping)
            or not isinstance(samples.get("update_ms"), list)
            or not isinstance(samples.get("tick_ms"), list)
            or len(samples["update_ms"]) != 1500
            or len(samples["tick_ms"]) != 1500
        ):
            raise Invalid(f"latency-{index:03d} does not retain 1500 raw samples")
        updates = [float(value) for value in samples["update_ms"]]
        ticks = [float(value) for value in samples["tick_ms"]]
        if not all(_num(value) and value >= 0 for value in updates) or not all(
            _num(value) and value >= 0 for value in ticks
        ):
            raise Invalid(f"latency-{index:03d} update samples are invalid")
        p95 = _p95(updates)
        if p95 >= P95_LIMIT_MS:
            failures.append(f"Watch{snakes} update p95 {p95:.6f} >= {P95_LIMIT_MS}")
        reported = (
            measurement.get("update_ms", {}).get("p95")
            if isinstance(measurement.get("update_ms"), Mapping)
            else None
        )
        if not _same_number(reported, p95):
            failures.append(f"Watch{snakes} reported p95 is not recomputed from raw samples")
        expected_ceiling = 24.0 if snakes <= 6 else 12.0
        paced = measurement.get("paced")
        if (
            not isinstance(paced, list)
            or len(paced) != 2
            or {item.get("requested_fps") for item in paced if isinstance(item, Mapping)}
            != {12, 120}
        ):
            raise Invalid(f"latency-{index:03d} must retain 12 and 120 FPS probes")
        for attempt in paced:
            if not isinstance(attempt, Mapping):
                raise Invalid(f"latency-{index:03d} paced row is not an object")
            expected_speed = min(float(attempt["requested_fps"]), expected_ceiling)
            failures.extend(
                f"Watch{snakes}: {item}"
                for item in _paced_failures(attempt, expected_speed, expected_ceiling, snakes)
            )
        for side in ("quiet_before", "quiet_after"):
            quiet = data.get(side)
            power = quiet.get("power", {}) if isinstance(quiet, Mapping) else {}
            processes = quiet.get("processes", {}) if isinstance(quiet, Mapping) else {}
            if (
                not isinstance(quiet, Mapping)
                or quiet.get("quiet") is not True
                or power.get("ac_power") is not True
                or power.get("lid_closed") is not False
                or power.get("pmset_returncode") != 0
                or power.get("ioreg_returncode") != 0
                or quiet.get("foreign_active_compute") != []
                or processes.get("returncode") != 0
            ):
                failures.append(f"Watch{snakes} lacks quiet-Mac {side} evidence")
        latencies.append({"snakes": snakes, "p95_ms": p95, "ceiling_fps": expected_ceiling})
    return {**_failure(*failures), "latency": latencies}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _check_controls(root: Path, intent: Mapping[str, Any], intent_sha: str) -> dict[str, Any]:
    row = _envelope(
        root,
        "records/controls-000.json",
        intent_sha,
        "controls",
        0,
        intent["banks"]["controls"]["seeds"][0],
    )
    data = row["data"]
    matrix = data.get("cases", data.get("matrix", data.get("controls")))
    if not isinstance(matrix, list) or not matrix:
        raise Invalid("controls matrix is absent")
    expected_names = [
        "roster_unset",
        "roster_all_b8",
        "roster_mixed",
        "roster_unknown",
        "checkpoint_b8-s3",
        "checkpoint_frp3-s12",
        "checkpoint_champion",
        "checkpoint_not-a-checkpoint",
        "mixed_variant_unset",
        "mixed_variant_v2",
        "mixed_variant_v5",
        "mixed_variant_v7",
        "mixed_variant_v8",
        "mixed_variant_unknown",
        "mixed_watch_hero_off",
        "mixed_missing_pin",
        "mixed_unfilled_pin",
        "mixed_wrong_pin",
        "mixed_missing_bytes",
        "mixed_corrupt_bytes",
        "mixed_changed_wrapper_source",
        "mode_play_refuses_mixed",
        "train_refusal_b8",
        "mode_watch_rebuilds_mixed",
        "checkpoint_replacement_refused",
        "mixed_roster_count_6",
        "mixed_roster_count_8",
        "mixed_roster_count_12",
        "b8_fallback_chain",
        "checkpoint_replacement_success_b8",
        "play_opt_in_veto_champion",
        "play_opt_in_veto_frp3",
    ]
    by_name = {str(item.get("name")): item for item in matrix if isinstance(item, Mapping)}
    failures = [
        f"controls matrix missing case {name}" for name in expected_names if name not in by_name
    ]
    if len(by_name) != len(matrix):
        failures.append("controls matrix has duplicate or malformed case names")

    def observed_case(name: str) -> tuple[Mapping[str, Any], list[str]]:
        row = by_name.get(name)
        if not isinstance(row, Mapping):
            return {}, [f"{name}: malformed row"]
        observed = row.get("observed")
        errors: list[str] = []
        if not isinstance(row.get("expected"), Mapping) or not isinstance(observed, Mapping):
            errors.append(f"{name}: expected/observed raw mappings are required")
        if not isinstance(row.get("steps"), int) or row["steps"] < 1:
            errors.append(f"{name}: bounded steps are absent")
        if row.get("no_errors") is not True or observed.get("errors") != []:
            errors.append(f"{name}: retained errors are not empty")
        return observed, errors

    def check(name: str, predicate: Any) -> None:
        observed, errors = observed_case(name)
        if not errors and not predicate(observed):
            errors.append(f"{name}: observed state contradicts its registered scenario")
        failures.extend(errors)

    def all_b8(row: Mapping[str, Any]) -> bool:
        return row.get("watch_roster") == "all_b8" and row.get("hero_wrapped") is False

    def mixed(row: Mapping[str, Any]) -> bool:
        return (
            row.get("watch_roster") == "mixed"
            and row.get("hero_wrapped") is False
            and (row.get("veto") or {}).get("active") is True
            and (row.get("veto") or {}).get("scope") == "watch_opponents"
            and (row.get("veto") or {}).get("wrapped_snake_ids") == (row.get("snake_ids") or [])[1:]
            and set((row.get("veto") or {}).get("counters", {}))
            == {str(i) for i in (row.get("snake_ids") or [])[1:]}
            and any("active=True scope=watch_opponents" in line for line in row.get("stderr", []))
        )

    check("roster_unset", all_b8)
    check("roster_all_b8", all_b8)
    check("roster_mixed", mixed)
    check(
        "roster_unknown",
        lambda row: all_b8(row) and "unknown" in str(row.get("watch_roster_reason")).lower(),
    )
    for name, checkpoint, obs_spec, wrapped in (
        ("checkpoint_b8-s3", "b8-s3", "ego2s-bv8", False),
        ("checkpoint_frp3-s12", "frp3-s12", "vector61", True),
        ("checkpoint_champion", "champion", "vector61", True),
    ):
        check(
            name,
            lambda row, n=checkpoint, spec=obs_spec, wrap=wrapped: (
                row.get("served_checkpoint") or {}
            ).get("requested")
            == n
            and (row.get("served_checkpoint") or {}).get("name") == n
            and row.get("obs_spec") == spec
            and row.get("hero_wrapped") is wrap,
        )
    check(
        "checkpoint_not-a-checkpoint",
        lambda row: (row.get("served_checkpoint") or {}).get("name") == "b8-s3"
        and "unknown" in str((row.get("served_checkpoint") or {}).get("reason")).lower(),
    )
    for name in expected_names:
        if name.startswith("mixed_variant_") or name == "mixed_watch_hero_off":
            check(name, mixed)
    for name in (
        "mixed_missing_pin",
        "mixed_unfilled_pin",
        "mixed_wrong_pin",
        "mixed_missing_bytes",
        "mixed_corrupt_bytes",
        "mixed_changed_wrapper_source",
    ):
        check(
            name,
            lambda row: all_b8(row)
            and not (row.get("veto") or {}).get("wrapped_snake_ids")
            and (
                "wrapper" in str(row.get("watch_roster_reason")).lower()
                or "frp3" in str(row.get("watch_roster_reason")).lower()
                or "pin" in str(row.get("watch_roster_reason")).lower()
                or "bytes" in str(row.get("watch_roster_reason")).lower()
            ),
        )
    check("mode_play_refuses_mixed", lambda row: row.get("mode") == "play" and all_b8(row))
    check(
        "train_refusal_b8",
        lambda row: row.get("mode") == "watch"
        and "forward-only" in str(row.get("last_error")).lower(),
    )
    check("mode_watch_rebuilds_mixed", mixed)
    check(
        "checkpoint_replacement_refused",
        lambda row: (row.get("served_checkpoint") or {}).get("name") == "b8-s3"
        and "unknown checkpoint" in str(row.get("last_error")).lower(),
    )
    check(
        "b8_fallback_chain",
        lambda row: (
            (
                (row.get("served_checkpoint") or {}).get("name") == "frp3-s12"
                and row.get("watch_roster") == "all_b8"
                and "fallback" in _canonical(row).lower()
            )
            or (
                (row.get("first") or {}).get("name") == "frp3-s12"
                and (row.get("second") or {}).get("name") == "champion"
                and row.get("first", {}).get("sha256") == FRP3_SHA256
                and "fallback" in _canonical(row).lower()
            )
        ),
    )
    check(
        "checkpoint_replacement_success_b8",
        lambda row: row.get("obs_spec") == "ego2s-bv8" and row.get("last_error") is None,
    )
    check(
        "play_opt_in_veto_champion",
        lambda row: row.get("mode") == "play"
        and (row.get("served_checkpoint") or {}).get("name") == "champion"
        and (row.get("veto") or {}).get("active") is True
        and (row.get("veto") or {}).get("scope") == "play_ai"
        and row.get("hero_wrapped") is False,
    )
    check(
        "play_opt_in_veto_frp3",
        lambda row: row.get("mode") == "play"
        and (row.get("served_checkpoint") or {}).get("name") == "frp3-s12"
        and (row.get("veto") or {}).get("active") is False
        and (row.get("veto") or {}).get("scope") == "play_ai"
        and "no strict-gate" in str((row.get("veto") or {}).get("reason")).lower(),
    )
    for size in (6, 8, 12):
        check(
            f"mixed_roster_count_{size}",
            lambda row, n=size: row.get("total_ai_count") == n
            and row.get("speed_ceiling") == (24.0 if n == 6 else 12.0),
        )
    return _failure(*failures)


def _artifact(value: Any, base: Path, label: str) -> tuple[Path, str]:
    if isinstance(value, Mapping):
        path, digest = value.get("path"), value.get("sha256")
    else:
        path, digest = value, None
    resolved = _path(path, base, label)
    if not _hex(digest) or not resolved.is_file() or _sha(resolved) != digest:
        raise Invalid(f"{label} is absent or hash-mismatched")
    return resolved, digest


def _check_ui(root: Path, intent: Mapping[str, Any], intent_sha: str) -> dict[str, Any]:
    row = _envelope(
        root,
        "records/ui-000.json",
        intent_sha,
        "ui",
        0,
        intent["banks"]["ui"]["seeds"][0],
    )
    data = row["data"]
    browser_url = data.get("url", data.get("browser_url"))
    if (
        data.get("browser_observed") is not True
        or not isinstance(browser_url, str)
        or not browser_url.startswith("http://127.0.0.1")
    ):
        return _failure("UI browser observation or loopback URL is absent")
    screenshot_value = data.get("screenshot", data.get("screenshot_path"))
    dom_value = data.get("dom", data.get("dom_path"))
    backend_value = data.get("backend", data.get("backend_path"))
    if isinstance(screenshot_value, str):
        screenshot_value = {"path": screenshot_value, "sha256": data.get("screenshot_sha256")}
    if isinstance(dom_value, str):
        dom_value = {"path": dom_value, "sha256": data.get("dom_sha256")}
    if isinstance(backend_value, str):
        backend_value = {"path": backend_value, "sha256": data.get("backend_sha256")}
    screenshot, _ = _artifact(screenshot_value, root, "UI screenshot")
    dom_path, _ = _artifact(dom_value, root, "UI DOM")
    backend_path, _ = _artifact(backend_value, root, "UI backend identity")
    dom_text = dom_path.read_text(encoding="utf-8", errors="replace").lower()
    backend = _json(backend_path)
    if not isinstance(backend, Mapping):
        raise Invalid("UI backend identity is not JSON")
    failures: list[str] = []
    counter_region = dom_text[max(0, dom_text.find("vetoes") - 400) : dom_text.find("vetoes") + 800]
    if any(
        token not in dom_text
        for token in ("opponents v8", "vetoes", "decisions", "#1", "#2", "#3", "#4", "#5")
    ):
        failures.append("DOM does not show the visible opponents v8 veto counters")
    if "#0" in counter_region:
        failures.append("DOM counter region includes a hero #0 veto counter")
    assertions = data.get("assertions", data.get("checks", {}))
    if not isinstance(assertions, Mapping) or assertions.get("speed_controls") is not True:
        failures.append("UI speed-control evidence is incomplete")
    if backend.get("intent_sha256") != intent_sha or backend.get("watch_roster") != "mixed":
        failures.append("UI backend identity is not bound to this mixed run")
    veto = backend.get("veto", {})
    if (
        not isinstance(veto, Mapping)
        or veto.get("active") is not True
        or veto.get("scope") != "watch_opponents"
        or veto.get("wrapped_snake_ids") != [1, 2, 3, 4, 5]
        or "0" in (veto.get("counters") or {})
    ):
        failures.append("UI backend identity does not prove opponent-only veto")
    return _failure(*failures)


def audit(root: str | Path) -> dict[str, Any]:
    """Audit one retained T15 result directory without running a workload."""
    root = Path(root)
    try:
        intent, intent_sha = _intent(root)
        if intent.get("smoke") is True:
            return {
                "schema": AUDIT_SCHEMA,
                "verdict": "SMOKE_ONLY",
                "criteria": {},
                "serving_path_qualified": False,
            }
        _receipt(root, intent_sha)
        criteria: dict[str, Any] = {
            "1": _check_identity(root, intent, intent_sha),
            "2": _check_scope(root, intent, intent_sha),
            "3": _check_controls(root, intent, intent_sha),
            "4": _check_latency(root, intent, intent_sha),
            "5": _check_continuous(root, intent, intent_sha),
            "6": _check_ui(root, intent, intent_sha),
        }
        verdict = (
            "SERVING_PASS"
            if all(value.get("pass") is True for value in criteria.values())
            else "SERVING_FAIL"
        )
        return {
            "schema": AUDIT_SCHEMA,
            "verdict": verdict,
            "intent_sha256": intent_sha,
            "protocol_sha256": PROTOCOL_SHA256,
            "criteria": criteria,
            "serving_path_qualified": verdict == "SERVING_PASS",
        }
    except (Invalid, OSError, TypeError, ValueError, KeyError, IndexError) as exc:
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"verdict": result["verdict"], "out": str(args.out)}))
    return {"SERVING_PASS": 0, "SERVING_FAIL": 1, "INVALID": 2, "SMOKE_ONLY": 0}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
