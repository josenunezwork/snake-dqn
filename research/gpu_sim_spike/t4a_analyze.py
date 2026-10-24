"""Frozen Mac-only T4A reducer; remote summaries cannot grant a premise PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import t4a_units as units

LAYOUTS = ("canonical", "reversed", "interspersed")
EVENTS = (
    "ring_wrap",
    "simultaneous_death_respawn",
    "contested_food_refill",
    "clone_restore",
    "stale_cache",
)
CONTROLS = ("q", "state", "cursor", "row_map")
SEEDS = {"main": 9402401, "canary": 9402411, "smoke": 9402421}
SIM_FIELDS = (
    "bodies head_ptr seg_count length alive direction boost_frames frames_since_food "
    "respawn_timer _reward_prev_length frame rng_counter rng_key food_cells food_present "
    "corpse_food food_count _last_reward _last_done _last_death_cause _last_kills "
    "_last_food_ate _last_ambient_food_ate _last_corpse_food_ate _last_transition_valid "
    "_boosted_this_step terminal_slots capacity_overflow"
).split()
RUNTIME_FIELDS = (
    "controlled prev_dist prev_id _carry_state _carry_mask _carry_valid _alive_post "
    "sel_state sel_mask sel_ready sel_fresh"
).split()
MANDATORY_FIELDS = frozenset(
    [f"sim.{f}" for f in SIM_FIELDS]
    + [f"runtime.{f}" for f in RUNTIME_FIELDS]
    + [f"collector.{f}" for f in ("length", "alive", "action", "qmax")]
)


def required_fields(mix: str) -> frozenset[str]:
    slots = {"scripted": (0,), "mixed": (0, 1, 3, 5), "frozen": tuple(range(6))}[mix]
    v8_fields = (
        "mask action base v5_action v7_action outcome v5_reason counts cap need "
        "area_cap area landing head_risky head_waived v5_eligible"
    ).split()
    return MANDATORY_FIELDS | frozenset(
        [
            "output.actions",
            "accounting.reconciled",
            "lifecycle.sim_prepared",
            "lifecycle.vector_prepared",
            "configuration.sha256",
            "output.obs.0.1",
            "output.obs.0.2",
        ]
        + [f"output.q.{s}" for s in slots]
        + [f"output.obs.{s}.0" for s in slots]
        + [f"output.v8.{i}.{name}" for i in range(len(slots)) for name in v8_fields]
        + [f"output.advisory.{i}" for i in range(3 + 6 - len(slots))]
    )


def _json(path: Path) -> dict[str, Any]:
    units._verify(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("T4A evidence must be an object")
    return value


def _content_errors(content: dict, mix: str, lane: str) -> tuple[list[str], list[str], list[Any]]:
    invalid: list[str] = []
    missing: list[str] = []
    mismatches: list[Any] = []
    horizon = {"main": 256, "canary": 2, "smoke": 1}[lane]
    if (
        content.get("roster"),
        content.get("mode"),
        content.get("horizon"),
        content.get("root_seed"),
    ) != (mix, lane, horizon, SEEDS[lane] + units.MIXES.index(mix)):
        invalid.append(f"{mix}: root/mode/horizon differs")
    controls = content.get("controls", {})
    if not isinstance(controls, dict) or any(controls.get(k) is not True for k in CONTROLS):
        invalid.append(f"{mix}: negative controls failed/missing")
    witnesses = content.get("control_witnesses", {})
    for key, field in {
        "q": "output.q.0",
        "state": "sim.bodies",
        "cursor": "sim.rng_counter",
        "row_map": "output.q.0",
    }.items():
        witness = witnesses.get(key, {}) if isinstance(witnesses, dict) else {}
        if (
            witness.get("field") != field
            or not isinstance(witness.get("byte_offset"), int)
            or witness["byte_offset"] < 0
            or witness.get("expected") == witness.get("actual")
            or not all(
                isinstance(witness.get(n), int) and 0 <= witness[n] <= 255
                for n in ("expected", "actual")
            )
        ):
            invalid.append(f"{mix}: invalid comparison control witness {key}")
    fields = content.get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or len(set(fields)) != len(fields)
        or any(not isinstance(x, str) or not x for x in fields)
    ):
        invalid.append(f"{mix}: field inventory missing or duplicate")
    else:
        mandatory = required_fields(mix)
        if not mandatory.issubset(fields):
            invalid.append(f"{mix}: mandatory fields omitted: {sorted(mandatory - set(fields))}")
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
        if content.get("field_inventory_sha256") != digest:
            invalid.append(f"{mix}: field inventory binding differs")
    sizes = (4,) if lane == "smoke" else (4, 64, 1024)
    expected = {(e, layout, r) for e in sizes for layout in LAYOUTS for r in (0, 1)}
    seen: set[tuple] = set()
    tapes: list[list] = []
    digests: list[list] = []
    records = content.get("records")
    if not isinstance(records, list):
        return invalid + [f"{mix}: malformed replay records"], missing, mismatches
    for row in records:
        if not isinstance(row, dict):
            invalid.append(f"{mix}: malformed replay")
            continue
        key = (row.get("worlds"), row.get("layout"), row.get("replay"))
        if key not in expected or key in seen:
            invalid.append(f"{mix}: foreign/duplicate replay {key}")
            continue
        seen.add(key)
        if row.get("ticks") != horizon or row.get("matched_ticks") != horizon:
            missing.append(f"{mix}: incomplete replay {key}")
        expected_calls = horizon * {"frozen": 6, "mixed": 4, "scripted": 1}[mix]
        if (
            not isinstance(row.get("comparison_count"), int)
            or row["comparison_count"] != horizon * len(fields or [])
            or row.get("forward_rows") != 1024
            or not isinstance(row.get("forward_calls"), int)
            or row["forward_calls"] != expected_calls
            or row.get("field_inventory_sha256") != content.get("field_inventory_sha256")
        ):
            invalid.append(f"{mix}: comparison/forward/inventory omitted in {key}")
        shapes = row.get("forward_shapes")
        if not isinstance(shapes, list) or len(shapes) != expected_calls:
            invalid.append(f"{mix}: missing physical forward shape proof in {key}")
        else:
            for call in shapes:
                try:
                    padded, output = call["padded_shapes"], call["output_shape"]
                    if (
                        not padded
                        or any(s[0] != 1024 for s in padded)
                        or output != [1024, 6]
                        or call["real_rows"] != row["worlds"]
                    ):
                        raise ValueError("native forward shape differs")
                except (KeyError, TypeError, IndexError, ValueError):
                    invalid.append(f"{mix}: variable-shaped learned forward in {key}")
                    break
        tape = row.get("action_q_tape")
        field_digests = row.get("field_digests")
        if not isinstance(field_digests, list) or len(field_digests) != horizon:
            invalid.append(f"{mix}: per-field/tick evidence omitted in {key}")
        else:
            digests.append(field_digests)
            for tick_fields in field_digests:
                if not isinstance(tick_fields, dict) or set(tick_fields) != set(fields or []):
                    invalid.append(f"{mix}: per-tick field inventory differs in {key}")
                    break
                for descriptor in tick_fields.values():
                    if (
                        not isinstance(descriptor, dict)
                        or not isinstance(descriptor.get("shape"), list)
                        or not isinstance(descriptor.get("dtype"), str)
                        or not isinstance(descriptor.get("sha256"), str)
                        or len(descriptor["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in descriptor["sha256"])
                    ):
                        invalid.append(f"{mix}: malformed typed field digest in {key}")
                        break
        if not isinstance(tape, list) or len(tape) != horizon:
            invalid.append(f"{mix}: missing exact action/Q tape in {key}")
        else:
            tapes.append(tape)
            for tick_index, tick in enumerate(tape):
                try:
                    if not isinstance(tick, dict) or not isinstance(tick.get("q_hex"), dict):
                        raise ValueError("tape tick is not an exact typed action/Q object")
                    actions = bytes.fromhex(tick["actions_hex"])
                    if len(actions) != 4 * 6 * 8:
                        raise ValueError("action tape must contain all four worlds and six seats")
                    if any(a not in range(6) for a in struct.unpack("<24q", actions)):
                        raise ValueError("action outside six-action space")
                    descriptors = (
                        field_digests[tick_index]
                        if isinstance(field_digests, list) and tick_index < len(field_digests)
                        else {}
                    )
                    if not isinstance(descriptors, dict):
                        raise ValueError("typed field tick container is not an object")
                    action_desc = descriptors.get("output.actions", {})
                    if (
                        not isinstance(action_desc, dict)
                        or action_desc.get("shape") != [4, 6]
                        or action_desc.get("dtype") != "torch.int64"
                        or action_desc.get("sha256") != hashlib.sha256(actions).hexdigest()
                    ):
                        raise ValueError("exact action tape is not bound to typed field digest")
                    slots = {
                        "scripted": {"0"},
                        "mixed": {"0", "1", "3", "5"},
                        "frozen": {str(s) for s in range(6)},
                    }[mix]
                    if set(tick["q_hex"]) != slots:
                        raise ValueError("learned-seat Q tape inventory differs")
                    if "0" not in tick["q_hex"]:
                        raise ValueError("hero Q tape omitted")
                    for slot, qhex in tick["q_hex"].items():
                        q = bytes.fromhex(qhex)
                        if len(q) != 4 * 6 * 4 or not all(
                            math.isfinite(v) for v in struct.unpack("<24f", q)
                        ):
                            raise ValueError("raw fp32 Q bytes malformed or nonfinite")
                        desc = descriptors.get(f"output.q.{slot}", {})
                        if (
                            not isinstance(desc, dict)
                            or desc.get("shape") != [4, 6]
                            or desc.get("dtype") != "torch.float32"
                            or desc.get("sha256") != hashlib.sha256(q).hexdigest()
                        ):
                            raise ValueError("exact Q tape is not bound to typed field digest")
                except (KeyError, ValueError, TypeError, struct.error) as exc:
                    invalid.append(f"{mix}: invalid exact action/Q tape in {key}: {exc}")
                    break
    if expected != seen:
        missing.append(f"{mix}: missing replays {sorted(expected - seen)}")
    if tapes and any(tape != tapes[0] for tape in tapes[1:]):
        mismatches.append(f"{mix}: exact action/Q tapes differ across configuration replays")
    if digests and any(tape != digests[0] for tape in digests[1:]):
        mismatches.append(f"{mix}: field/tick digests differ across configuration replays")
    coverage = content.get("coverage", {})
    if not isinstance(coverage, dict) or any(coverage.get(k) is not True for k in EVENTS):
        missing.append(f"{mix}: event coverage incomplete")
    event_rows = content.get("event_records")
    expected_events = {(event, e) for event in EVENTS for e in sizes}
    seen_events: set[tuple] = set()
    event_tapes: dict[str, Any] = {}
    if not isinstance(event_rows, list):
        missing.append(f"{mix}: executed event witnesses absent")
    else:
        for row in event_rows:
            if not isinstance(row, dict):
                invalid.append(f"{mix}: malformed event record")
                continue
            key = (row.get("event"), row.get("worlds"))
            if key not in expected_events or key in seen_events:
                invalid.append(f"{mix}: foreign/duplicate event record")
                continue
            seen_events.add(key)
            if row.get("status") != "PASS" or row.get("ticks") != 2:
                missing.append(f"{mix}: unexercised event {key}")
            witness = row.get("witness")
            if not isinstance(witness, dict) or not witness:
                missing.append(f"{mix}: absent event predicate witness {key}")
            else:
                if row.get("status") == "INVALID_SELF_CONSISTENCY":
                    mismatches.append(witness.get("first_witness", {"event": key}))
                digest = hashlib.sha256(
                    json.dumps(witness, sort_keys=True, default=str).encode()
                ).hexdigest()
                if row.get("witness_sha256") != digest:
                    invalid.append(f"{mix}: event witness digest differs {key}")
                mapped = witness.get("mapped_tape")
                raw = witness.get("action_q_tape")
                event_fields = (
                    required_fields(mix)
                    - {f"collector.{n}" for n in ("length", "alive", "action", "qmax")}
                ) | {"event.supplied_actions"}
                if (
                    witness.get("ticks") != 2
                    or not isinstance(mapped, list)
                    or len(mapped) != 2
                    or any(not isinstance(t, dict) or not event_fields.issubset(t) for t in mapped)
                    or not isinstance(raw, list)
                    or len(raw) != 2
                ):
                    missing.append(f"{mix}: event policy/state/tick proof omitted {key}")
                else:
                    data = {"mapped_tape": mapped, "action_q_tape": raw}
                    if key[0] in event_tapes and event_tapes[key[0]] != data:
                        mismatches.append(f"{mix}: event {key[0]} mapped tape differs across E")
                    event_tapes.setdefault(key[0], data)
        if seen_events != expected_events:
            missing.append(f"{mix}: required event/E cells missing")
    return invalid, missing, mismatches


def aggregate(
    exports: Path, commit: str, *, canary: bool = False, smoke: bool = False
) -> dict[str, Any]:
    """Reduce all verified units once, without trusting producer verdicts."""
    lane = units.mode(smoke=smoke, canary=canary)
    invalid: list[str] = []
    missing: list[str] = []
    mismatches: list[Any] = []
    result: dict[str, Any] = {
        "schema": "gpu-sim-spike-t4a-analysis/v1",
        "commit": commit,
        "mode": lane,
        "claim": "standalone dev batch-invariance premise only; no CPU/label/release authority",
        "invalid_evidence": invalid,
        "incomplete": missing,
        "mismatches": mismatches,
    }
    try:
        native_path = exports / "native-parity.json"
        native = _json(native_path) if native_path.is_file() else {}
        expected = "PREFLIGHT" if smoke else "PASS"
        if not native:
            missing.append("native proof absent")
        elif native.get("commit") != commit or native.get("status") != expected:
            invalid.append("current-pin native proof absent or failed")
        if native and not smoke and native.get("native_cuda_executed") is not True:
            invalid.append("native CUDA execution not proved")
        failure = exports / "t4a-failure.json"
        if failure.exists():
            fail = _json(failure)
            if fail.get("commit") != commit:
                invalid.append("foreign failure record")
            if fail.get("status") == "INVALID_SELF_CONSISTENCY":
                witness = fail.get("first_witness")
                if not isinstance(witness, dict) or not witness.get("field"):
                    invalid.append("behavioral failure omitted first witness")
                else:
                    mismatches.append(witness)
            elif fail.get("status") == "INVALID_EVIDENCE":
                invalid.append(str(fail.get("reason", "producer evidence failure")))
            else:
                missing.append(str(fail.get("reason", "producer interrupted")))
        try:
            if not (exports / "verified-ledger.json").is_file():
                raise FileNotFoundError("verified ledger absent")
            rows = units.load_ledger(
                exports / "verified-ledger.json",
                units.namespace(commit, smoke=smoke, canary=canary),
                units.allocation(),
                exports=exports,
            )
        except FileNotFoundError:
            rows = {}
            missing.append("verified ledger absent")
        expected_paths = {f"{u}.json" for u in rows}
        actual_paths = {
            p.name
            for p in (exports / units.UNIT_DIRECTORY).glob("*.json")
            if Path(str(p) + ".complete").exists()
        }
        if actual_paths != expected_paths:
            invalid.append("unaccounted completed export or duplicate source")
        for unit in units.allocation():
            if unit not in rows:
                missing.append(f"missing unit: {unit}")
                continue
            content = rows[unit]
            status = content.get("status")
            if status == "INVALID_EVIDENCE":
                invalid.append(f"{unit}: producer invalid evidence")
            elif status == "INVALID_SELF_CONSISTENCY":
                mismatches.append(content.get("first_witness", unit))
            elif status != "COMPLETE":
                missing.append(f"{unit}: {status}")
            errs, gaps, differences = _content_errors(content, units.parse(unit), lane)
            invalid.extend(errs)
            missing.extend(gaps)
            mismatches.extend(differences)
            if content.get("first_witness") is not None:
                mismatches.append(content["first_witness"])
    except FileNotFoundError as exc:
        missing.append(str(exc))
    except (ValueError, TypeError, KeyError) as exc:
        invalid.append(str(exc))
    result["verdict"] = (
        "INVALID_EVIDENCE"
        if invalid
        else (
            "INVALID_SELF_CONSISTENCY"
            if mismatches
            else (
                "INCOMPLETE"
                if missing
                else "PREFLIGHT" if smoke else "PASS_CANARY" if canary else "PASS_T4A_DEV"
            )
        )
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    result = aggregate(a.exports, a.commit, canary=a.canary, smoke=a.smoke)
    units.publish_json(a.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
