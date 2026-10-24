"""Synthetic, outcome-free I0S export trees and registers for offline tests.

Shapes mirror the frozen producer (``integration_bench.measure``/``run``) and the
qualified supervisor's output layout (``receipt.json`` + ``exports/`` with sidecars).
Run as a script under the qualified PYTHONPATH to reduce a synthetic rc-2
TimeoutError tree with the frozen validators; it prints the reduction JSON.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools import i0s_attempt as i0s
from tools import small_bench_register as register

ORDERS = [["R", "I"], ["I", "R"], ["R", "I"]]
NOW = datetime(2026, 10, 25, 0, 0, tzinfo=timezone.utc)
HISTORY = [
    {
        "id": i0s.I0_HISTORY_IDS[0],
        "pair": {"units": 1.4921040233777603, "wall_minutes": 8.952624140266561},
        "basis": "test",
    },
    {
        "id": i0s.I0_HISTORY_IDS[1],
        "pair": {"units": 0.6878387509374685, "wall_minutes": 13.756775018749371},
        "basis": "test",
    },
]
SOURCE = [{"path": "/synthetic/source", "sha256": "0" * 64}]


def units_module() -> Any:
    from research.gpu_sim_spike import integration_units

    return integration_units


def cell_row(unit: str, median: float, blocks: list[float] | None = None) -> dict[str, Any]:
    """A frozen-valid cell whose I-arm median and continuous rates are chosen."""
    mix, worlds = unit.split("-")[1], int(unit.rsplit("e", 1)[1])
    plain = [median * 0.99, median, median * 1.01]

    def arm(values: list[float]) -> dict[str, Any]:
        return {
            "plain_seconds": list(values),
            "digests": ["a" * 64] * 3,
            "stages": [
                {
                    "repeat": index,
                    "wall_seconds": wall,
                    "exclusive_cuda_elapsed_seconds": {"x": wall},
                }
                for index, wall in enumerate(values)
            ],
        }

    row: dict[str, Any] = {
        "mix": mix,
        "worlds": worlds,
        "repeats": 3,
        "warmups": 3,
        "orders": ORDERS,
        "arms": {"R": arm(plain), "I": arm(plain)},
        "capacity": 10400,
        "lengths": [800, 900, 1000, 1100, 1300, 1500],
        "device": "cuda",
        "native_parity": "PASS",
        "frozen_device_identity": True,
        "cpu_geometry_carry": True,
        "combined_parity": {
            "frozen_device_identity": True,
            "cpu_geometry_carry": True,
            "frozen_move": {"R": {"match": True}, "I": {"match": True}},
            "feature_calls": [{"arm": "I", "cpu_match": True}],
            "state_inventory": {"R": {}, "I": {}},
            "action_tape": {"shape": [worlds, 6]},
        },
        "cpu_closed_loop_identity": "UNPROVED",
        "trace": {"status": "NOT_RUN"},
    }
    if worlds == 1024:

        def repeats(rates: list[float]) -> list[dict[str, Any]]:
            out = []
            for index, rate in enumerate(rates):
                wall = 32 * 1024 / rate
                out.append(
                    {
                        "repeat": index,
                        "completed_world_steps": 32 * 1024,
                        "wall_seconds": wall,
                        "tick_seconds": [wall / 32 * 0.99] * 32,
                        "complete_world_steps_per_second": 32 * 1024 / wall,
                        "final_digest": "b" * 64,
                        "collector_positions": list(range(32)),
                    }
                )
            return out

        row["continuous"] = {
            "status": "COMPLETE",
            "ticks": 32,
            "orders": ORDERS,
            "arms": {"R": {"repeats": repeats([1500.0] * 3)}, "I": {"repeats": repeats(blocks)}},
            "reset_charged_once_per_repeat": True,
            "warmups": 3,
        }
    else:
        row["continuous"] = {"status": "NOT_REQUIRED"}
    return row


def native_report(full: bool = False) -> dict[str, Any]:
    if not full:
        return {"status": "PASS", "commit": i0s.PIN}
    from research.gpu_sim_spike.benchmark import source_hashes

    return {
        "schema": "gpu-sim-spike-i0-native-parity/v1",
        "status": "PASS",
        "commit": i0s.PIN,
        "source_sha256": source_hashes(),
        "native_cuda_executed": True,
        "fallback": False,
        "rows": [{"outcome": "passed"}] * 14,
        "geometry_native_calls": {"indexed_body_view": 1, "indexed_features58": 1},
        "move2_events": {"status": "PASS", "fallback": False, "native_calls": {"move": 1}},
        "compiled_binaries": {name: {} for name in ("geometry", "move", "events", "flood")},
    }


def receipt(output: Path, **fields: Any) -> dict[str, Any]:
    exports = output / "exports"
    listed = []
    if exports.is_dir():
        for path in sorted(exports.rglob("*")):
            if path.is_file() and not path.name.endswith((".sha256", ".complete")):
                listed.append(
                    {
                        "path": path.relative_to(exports).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size": path.stat().st_size,
                    }
                )
    value = {
        "job_id": fields.pop("job_id", i0s.job_id(1)),
        "generation": "g" * 32,
        "endpoint": "gpu-a100-synthetic",
        "elapsed_seconds": 6000.0,
        "usage_before": {"balance": 1000.0},
        "usage_after": {"balance": 990.0},
        "status": "INCOMPLETE",
        "failure_class": "CHILD_NONZERO_EXIT",
        "stop_verified": True,
        "exports": listed,
    }
    value.update(fields)
    output.mkdir(parents=True, exist_ok=True)
    (output / "receipt.json").write_text(json.dumps(value, indent=2, sort_keys=True))
    return value


def timeout_tree(
    output: Path,
    cells: dict[str, tuple[float, list[float] | None]],
    *,
    full_native: bool = False,
    failure: str | None = None,
    complete_ledger: bool = False,
    namespace: str | None = None,
    **receipt_fields: Any,
) -> Path:
    """rc-2 producer tree: sidecar-complete units, no verified-ledger.json."""
    units = units_module()
    exports = output / "exports"
    exports.mkdir(parents=True)
    ns, allocation = namespace or units.namespace(i0s.PIN), units.allocation()
    units.publish_json(exports / "native-parity.json", native_report(full_native))
    units.publish_json(
        exports / "discrepancy-register.json",
        {"commit": i0s.PIN, "cpu_component_register": {"status": "DIAGNOSTIC"}},
    )
    for unit, (median, blocks) in cells.items():
        units.publish_unit(exports, ns, allocation, unit, cell_row(unit, median, blocks))
    status = {"status": "INCOMPLETE", "reason": "I0 deadline during continuous block"}
    if failure:
        status = {"status": "INVALID", "commit": i0s.PIN, "failure_class": failure}
        units.publish_json(exports / "integration-failure.json", status)
    units.publish_json(exports / "integration-bench.json", status)
    if complete_ledger:
        units.build_ledger(exports, exports / "verified-ledger.json", commit=i0s.PIN)
    receipt(output, **receipt_fields)
    return output


def init_row(o1: tuple[float, float] = (48.0, 288.0)) -> dict[str, Any]:
    return {
        "type": "init",
        "id": "init",
        "scopes": {
            "global": {
                "ceiling": {"units": 1100, "wall_minutes": 7200},
                "spent": {"units": 34.37, "wall_minutes": 248.28},
            },
            "q4": {
                "ceiling": {"units": 192, "wall_minutes": 1152},
                "spent": {"units": 25.53, "wall_minutes": 195.25},
            },
        },
        "holds": [
            {
                "id": "I0",
                "scopes": ["global", "q4"],
                "pair": {"units": 24, "wall_minutes": 144},
                "account_units": 24,
                "kind": "protected_required_work",
            },
            {
                "id": "O1",
                "scopes": ["global"],
                "pair": {"units": o1[0], "wall_minutes": o1[1]},
                "account_units": o1[0],
                "kind": "protected_required_work",
            },
        ],
        "external_commitments_units": 0,
        "unsettled_debit_holds_units": 0,
        "completeness_attestation": "synthetic test register",
        "sources": SOURCE,
        "historical_charges": HISTORY,
    }


def make_register(path: Path, o1: tuple[float, float] = (8.0, 48.0)) -> list[dict[str, Any]]:
    register.append(path, init_row(o1))
    register.append(path, i0s.scopes_delta(register.read(path)))
    return register.read(path)


def package(attempt: int, now: datetime = NOW, balance: float = 2000.0) -> dict[str, Any]:
    return {
        "job_id": i0s.job_id(attempt),
        "manifest_sha256": "1" * 64,
        "packet_id": i0s.PACKET_ID,
        "reservation_id": i0s.reservation_id(attempt),
        "deadline_utc": i0s.SPEED_DEADLINE,
        "capture_utc": (now - timedelta(minutes=1)).isoformat(),
        "operator": "test",
        "scopes": list(i0s.SCOPES),
        "cap": {"units": i0s.UNIT_CAP, "wall_minutes": float(i0s.TIMEOUT_MINUTES)},
        "outside_tail": dict(i0s.OUTSIDE_TAIL),
        "live_balance_units": balance,
        "bindings": SOURCE,
        "allocation_wait_minutes": i0s.ALLOCATION_WAIT_MINUTES,
    }


def admit(
    path: Path, attempt: int, packet: Path, now: datetime = NOW, **kwargs: Any
) -> dict[str, Any]:
    rows = register.read(path)
    event = i0s.admission_event(rows, package(attempt, now, **kwargs), attempt, now, packet=packet)
    return register.append(path, event)


def settle(path: Path, attempt: int, charged: float = 18.4) -> dict[str, Any]:
    return register.append(
        path,
        {
            "type": "settlement",
            "id": f"settlement:{i0s.reservation_id(attempt)}",
            "reservation_id": i0s.reservation_id(attempt),
            "sources": SOURCE,
            "charged": {"units": charged, "wall_minutes": 110.0},
            "remaining_hold": {"units": 0.0, "wall_minutes": 0.0},
            "remaining_account_units": charged,
        },
    )


def frozen_rc2_reduction(work: Path) -> dict[str, Any]:
    """Qualified-checkout end to end: rc-2 tree -> settle -> frozen-validator reduction."""
    from tools import i0s_reduce

    packet = work / "packet"
    path = work / "register.jsonl"
    make_register(path)
    admit(path, 1, packet)
    timeout_tree(
        i0s.attempt_dir(1, packet) / "main-output",
        {"cell-frozen-e1024": (0.40, [2600.0] * 3), "cell-mixed-e1024": (0.41, [2500.0] * 3)},
        full_native=True,
    )
    settle(path, 1)
    attempts = {1: i0s.attempt_outputs(1, packet)}
    return i0s_reduce.final(attempts, register.read(path), work / "reduction-work")


if __name__ == "__main__":
    result = frozen_rc2_reduction(Path(sys.argv[1]))
    print(json.dumps(result, sort_keys=True, default=str))
