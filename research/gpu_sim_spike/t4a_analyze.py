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


def _json(path: Path) -> dict[str, Any]:
    units._verify(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("T4A evidence must be an object")
    return value


def _content_errors(content: dict, mix: str, lane: str) -> tuple[list[str], list[str]]:
    invalid: list[str] = []
    missing: list[str] = []
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
    fields = content.get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or len(set(fields)) != len(fields)
        or any(not isinstance(x, str) or not x for x in fields)
    ):
        invalid.append(f"{mix}: field inventory missing or duplicate")
    else:
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
        if content.get("field_inventory_sha256") != digest:
            invalid.append(f"{mix}: field inventory binding differs")
    sizes = (4,) if lane == "smoke" else (4, 64, 1024)
    expected = {(e, layout, r) for e in sizes for layout in LAYOUTS for r in (0, 1)}
    seen: set[tuple] = set()
    tapes: list[list] = []
    records = content.get("records")
    if not isinstance(records, list):
        return invalid + [f"{mix}: malformed replay records"], missing
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
        if (
            not isinstance(row.get("comparison_count"), int)
            or row["comparison_count"] <= 0
            or row.get("forward_rows") != 1024
            or not isinstance(row.get("forward_calls"), int)
            or row["forward_calls"] <= 0
            or row.get("field_inventory_sha256") != content.get("field_inventory_sha256")
        ):
            invalid.append(f"{mix}: comparison/forward/inventory omitted in {key}")
        tape = row.get("action_q_tape")
        if not isinstance(tape, list) or len(tape) != horizon:
            invalid.append(f"{mix}: missing exact action/Q tape in {key}")
        else:
            tapes.append(tape)
            for tick in tape:
                try:
                    if not isinstance(tick, dict) or not isinstance(tick.get("q_hex"), dict):
                        raise ValueError("tape tick is not an exact typed action/Q object")
                    actions = bytes.fromhex(tick["actions_hex"])
                    if len(actions) != 4 * 6 * 8:
                        raise ValueError("action tape must contain all four worlds and six seats")
                    if any(a not in range(6) for a in struct.unpack("<24q", actions)):
                        raise ValueError("action outside six-action space")
                    slots = {
                        "scripted": {"0"},
                        "mixed": {"0", "1", "3", "5"},
                        "frozen": {str(s) for s in range(6)},
                    }[mix]
                    if set(tick["q_hex"]) != slots:
                        raise ValueError("learned-seat Q tape inventory differs")
                    if "0" not in tick["q_hex"]:
                        raise ValueError("hero Q tape omitted")
                    for qhex in tick["q_hex"].values():
                        q = bytes.fromhex(qhex)
                        if len(q) != 4 * 6 * 4 or not all(
                            math.isfinite(v) for v in struct.unpack("<24f", q)
                        ):
                            raise ValueError("raw fp32 Q bytes malformed or nonfinite")
                except (KeyError, ValueError, TypeError, struct.error) as exc:
                    invalid.append(f"{mix}: invalid exact action/Q tape in {key}: {exc}")
                    break
    if expected != seen:
        missing.append(f"{mix}: missing replays {sorted(expected - seen)}")
    if tapes and any(tape != tapes[0] for tape in tapes[1:]):
        # A producer cannot certify matching bytes while exporting contradictory tapes.
        invalid.append(f"{mix}: exact action/Q tapes contradict claimed comparisons")
    coverage = content.get("coverage", {})
    if not isinstance(coverage, dict) or any(coverage.get(k) is not True for k in EVENTS):
        missing.append(f"{mix}: event coverage incomplete")
    return invalid, missing


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
        native = _json(exports / "native-parity.json")
        expected = "PREFLIGHT" if smoke else "PASS"
        if native.get("commit") != commit or native.get("status") != expected:
            invalid.append("current-pin native proof absent or failed")
        if not smoke and native.get("native_cuda_executed") is not True:
            invalid.append("native CUDA execution not proved")
        rows = units.load_ledger(
            exports / "verified-ledger.json",
            units.namespace(commit, smoke=smoke, canary=canary),
            units.allocation(),
            exports=exports,
        )
        expected_paths = {f"{u}.json" for u in rows}
        actual_paths = {
            p.name
            for p in (exports / units.UNIT_DIRECTORY).glob("*.json")
            if Path(str(p) + ".complete").exists()
        }
        if actual_paths != expected_paths:
            invalid.append("unaccounted completed export or duplicate source")
        failure = exports / "t4a-failure.json"
        if failure.exists():
            fail = _json(failure)
            if fail.get("commit") != commit:
                invalid.append("foreign failure record")
            if fail.get("status") == "INVALID_SELF_CONSISTENCY":
                mismatches.append(fail.get("first_witness", fail.get("reason")))
            elif fail.get("status") == "INVALID_EVIDENCE":
                invalid.append(str(fail.get("reason", "producer evidence failure")))
            else:
                missing.append(str(fail.get("reason", "producer interrupted")))
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
            errs, gaps = _content_errors(content, units.parse(unit), lane)
            invalid.extend(errs)
            missing.extend(gaps)
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
