"""Pre-registered R2 reducer (frozen before data; engineering evidence only).

Acceptance (``JUDGMENT_POST_I0S_DIRECTION_20261024.md`` step 4, frozen here):

(a) Pure-optimization identity on every cell and exception fixture: native
    MOVE2/geometry parity PASS, R2 CUDA twin tests PASS, every unit's debug-mode
    identity proof PASS, one plain-tick state digest shared by both arms and all
    repeats, and equal I/I' final digests for every continuous block.  Any
    failure, foreign/duplicate unit or ``r2-failure.json`` is ``INVALID``.
(b) R2 advances (``R2_ADVANCE``) only if, on every E1024 roster, the median
    reset-inclusive plain tick improves by at least 15%:
    ``1 - median(I') / median(I) >= 0.15``; and the exception-heavy fixture
    packet (three ``exc-*-e256`` cells) regresses by at most 10%:
    ``sum_roster median(I') / sum_roster median(I) <= 1.10``.  Else ``R2_STOP``.
(c) Q4 speed decision: ``PASS_SPEED`` only if every I' continuous 32-tick block
    on every E1024 roster completes at least 2,000 world-steps/s; else
    ``STOP_SPEED``.

Precedence: INVALID > INCOMPLETE (missing gating cell) > decision.  The E6
floor and the per-world slope ``(t(E1024) - t(E64)) / 960`` are reported
separately and are not bars.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import r2_units as units

ARMS = ("I", "I_prime")
R2_GAIN = 0.15
EXCEPTION_REGRESSION = 1.10
Q4_RATE = 2000.0
MIXES = units.MIXES


def _json(path: Path) -> dict[str, Any]:
    units._verify(path)
    return json.loads(path.read_text())


def _identity_errors(unit: str, content: dict[str, Any]) -> list[str]:
    errors = []
    proof = content.get("identity_proof", {})
    if content.get("pure_optimization_identity") is not True or proof.get("status") != "PASS":
        errors.append(f"{unit}: identity proof missing or failed")
    if not proof.get("candidate_debug", {}).get("debug_checks"):
        errors.append(f"{unit}: R2 debug recomputation did not run")
    digests = {d for arm in ARMS for d in content.get("arms", {}).get(arm, {}).get("digests", [])}
    if len(digests) != 1:
        errors.append(f"{unit}: plain digests differ across arms/repeats")
    continuous = content.get("continuous", {})
    if continuous.get("status") == "COMPLETE":
        left = [r["final_digest"] for r in continuous["arms"]["I"]["repeats"]]
        right = [r["final_digest"] for r in continuous["arms"]["I_prime"]["repeats"]]
        if left != right or len(set(left)) != 1:
            errors.append(f"{unit}: continuous final digests differ")
    return errors


def _median(content: dict[str, Any], arm: str) -> float:
    return float(statistics.median(content["arms"][arm]["plain_seconds"]))


def aggregate(exports: list[Path], commit: str, *, canary: bool = False) -> dict[str, Any]:
    if len(exports) != 1:
        raise ValueError("R2 reduces exactly one scored attempt")
    root = Path(exports[0])
    cells = units.allocation(canary=canary)
    ns = units.namespace(commit, canary=canary)
    errors: list[str] = []
    if (root / "r2-failure.json").exists():
        errors.append("producer reported R2_PURE_OPTIMIZATION_MISMATCH")
    native = _json(root / "native-parity.json") if (root / "native-parity.json").exists() else {}
    r2_native = (
        _json(root / "r2-native-parity.json") if (root / "r2-native-parity.json").exists() else {}
    )
    if native.get("status") != "PASS" or native.get("native_cuda_executed") is not True:
        errors.append("I0 native MOVE2/geometry parity did not PASS on CUDA")
    if r2_native.get("status") != "PASS" or r2_native.get("device") != "cuda":
        errors.append("R2 CUDA twin-identity tests did not PASS")
    for name, body in (("native-parity", native), ("r2-native-parity", r2_native)):
        if body and body.get("commit") != commit:
            errors.append(f"{name} commit differs")
    rows = units.load_ledger(root / "verified-ledger.json", ns, cells, exports=root)
    for unit, content in rows.items():
        errors.extend(_identity_errors(unit, content))
    missing = [unit for unit in cells if unit not in rows]
    result: dict[str, Any] = {
        "schema": "gpu-sim-spike-r2-analysis/v1",
        "commit": commit,
        "mode": "canary" if canary else "main",
        "bars": {
            "r2_min_complete_tick_gain": R2_GAIN,
            "exception_packet_max_ratio": EXCEPTION_REGRESSION,
            "q4_min_continuous_world_steps_per_second": Q4_RATE,
        },
        "identity_errors": errors,
        "missing_units": missing,
        "cells": {
            unit: {arm: _median(content, arm) for arm in ARMS} for unit, content in rows.items()
        },
        "claim": "engineering evidence only; no hero, label, release or serving claim",
    }
    if errors:
        result["verdict"] = "INVALID"
        return result
    if canary:
        result["verdict"] = "PASS_CANARY" if not missing else "INCOMPLETE"
        return result
    gating = [f"cell-{mix}-e1024" for mix in MIXES] + [f"exc-{mix}-e256" for mix in MIXES]
    if any(unit not in rows for unit in gating):
        result["verdict"] = "INCOMPLETE"
        result["q4_speed"] = "INCOMPLETE"
        return result
    gains, rates = {}, {}
    for mix in MIXES:
        content = rows[f"cell-{mix}-e1024"]
        gains[mix] = 1.0 - _median(content, "I_prime") / _median(content, "I")
        blocks = content["continuous"]["arms"]["I_prime"]["repeats"]
        rates[mix] = [float(b["complete_world_steps_per_second"]) for b in blocks]
    exception = {arm: sum(_median(rows[f"exc-{mix}-e256"], arm) for mix in MIXES) for arm in ARMS}
    ratio = exception["I_prime"] / exception["I"]
    advance = all(gain >= R2_GAIN for gain in gains.values()) and ratio <= EXCEPTION_REGRESSION
    q4 = all(len(v) == 3 and min(v) >= Q4_RATE for v in rates.values())
    slopes, floors = {}, {}
    for mix in MIXES:
        for arm in ARMS:
            if f"cell-{mix}-e64" in rows:
                slopes[f"{mix}/{arm}"] = (
                    _median(rows[f"cell-{mix}-e1024"], arm) - _median(rows[f"cell-{mix}-e64"], arm)
                ) / 960.0
            if f"cell-{mix}-e6" in rows:
                floors[f"{mix}/{arm}"] = _median(rows[f"cell-{mix}-e6"], arm)
    result.update(
        {
            "verdict": "R2_ADVANCE" if advance else "R2_STOP",
            "q4_speed": "PASS_SPEED" if q4 else "STOP_SPEED",
            "e1024_complete_tick_gain": gains,
            "e1024_i_prime_continuous_world_steps_per_second": rates,
            "exception_packet_seconds": exception,
            "exception_packet_ratio": ratio,
            "descriptive_e6_floor_seconds": floors,
            "descriptive_per_world_slope_seconds": slopes,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = aggregate([args.exports], args.commit, canary=args.canary)
    units.publish_json(args.output, value)
    print(json.dumps({"verdict": value["verdict"], "q4_speed": value.get("q4_speed")}))


if __name__ == "__main__":
    main()
