"""T4A-DEV-01 standalone batch-invariance harness.

This module performs no dispatch and deliberately emits per-roster sealed
diagnostics only.  ``t4a_analyze`` owns the cross-roster scientific verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import integration_native
from research.gpu_sim_spike import t4a_compare as compare
from research.gpu_sim_spike import t4a_fixtures as fixtures
from research.gpu_sim_spike import t4a_units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes
from research.gpu_sim_spike.t4a_capture import Capture

SCHEMA = "gpu-sim-spike-t4a-bench/v1"
WORLD_SIZES = (4, 64, 1024)


class T4AError(RuntimeError):
    """A harness precondition or an exact comparison failed."""


class T4AMismatch(RuntimeError):
    """A valid-prefix byte mismatch is terminal and cannot become retryable."""

    def __init__(self, witness: dict[str, Any]) -> None:
        self.witness = witness
        super().__init__(str(witness))


def _determinism(device: str) -> dict[str, Any]:
    required_workspace = ":4096:8"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_workspace:
        raise T4AError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise T4AError("CUDA requested but unavailable")
    return {
        "deterministic_algorithms": True,
        "tf32": False,
        "cudnn_benchmark": False,
        "padded_forward_rows": fixtures.FORWARD_ROWS,
    }


def _target_inventory(case: tuple, rows: tuple[int, int, int, int]) -> dict[str, torch.Tensor]:
    sim, policy, _template, _ptemplate, collection, _arm = case
    rows_t = torch.tensor(rows, device=sim.device, dtype=torch.long)
    fields: dict[str, torch.Tensor] = {}
    for owner, prefix in ((sim, "sim"), (policy.vector_runtime, "runtime")):
        for name, value in compare.world_tensors(owner, prefix, int(sim.E)).items():
            fields[name] = value.index_select(0, rows_t).contiguous()
    for name, value in collection.items():
        # Only written collector row is admitted (never torch.empty bytes).
        fields[f"collector.{name}"] = value[-1].index_select(0, rows_t).contiguous()
    return fields


def _permutation(worlds: int, layout: str) -> tuple[int, int, int, int]:
    return fixtures.layout_rows(worlds, layout)


def _run_record(
    args: argparse.Namespace, roster: str, worlds: int, layout: str, replay: int, horizon: int
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]], list[Any]]:
    built_at = time.monotonic()
    case, wrappers, target_rows = fixtures.build_mapped_case(
        args.checkpoint,
        roster,
        worlds,
        layout,
        args.device,
        smoke=args.smoke,
        canary=args.canary,
    )
    # Four independent lifecycle offsets are exercised sequentially; a real
    # R2 construction occurs for every matrix replay, preventing aliasing.
    snapshots: list[dict[str, torch.Tensor]] = []
    tapes: list[Any] = []
    matched_ticks = 0
    action_q_tape: list[dict[str, str]] = []
    tick_seconds: list[float] = []
    sim, policy, _template, _ptemplate, _collection, _arm = case
    with Capture(case) as capture:
        with engine._arm("I"), torch.inference_mode():
            # The offsets are a frozen seam schedule.  Building fresh records
            # per layout/replay makes the series independent, and the target
            # lanes remain in their physical positions during every forward.
            for tick in range(horizon):
                if time.monotonic() >= args.deadline:
                    raise TimeoutError("T4A deadline during matrix tick")
                tick_started = time.monotonic()
                collection = ruler._collection(1, worlds, sim.device)
                capture.begin_tick()
                ruler._full_tick(sim, policy, collection, 0)
                state = _target_inventory((*case[:4], collection, case[5]), target_rows)
                state.update(capture.fields(target_rows))
                actions = capture.values["output.actions"]
                action_q_tape.append(capture.q_tape(actions, target_rows))
                snapshots.append(state)
                tapes.append(compare.compact_inventory(state))
                matched_ticks += 1
                tick_seconds.append(time.monotonic() - tick_started)
    calls = [call.__dict__ for wrapper in wrappers for call in wrapper.calls]
    if not calls or any(call["padded_shapes"][0][0] != fixtures.FORWARD_ROWS for call in calls):
        raise fixtures.T4ANotReady("a learned forward was not genuinely 1024 rows")
    inventory = snapshots[-1] if snapshots else {}
    fields = sorted(inventory)
    return (
        {
            "worlds": worlds,
            "layout": layout,
            "replay": replay,
            "ticks": horizon,
            "build_seconds": tick_seconds[0] * 0
            + (time.monotonic() - built_at - sum(tick_seconds)),
            "tick_seconds": tick_seconds,
            "matched_ticks": matched_ticks,
            "comparison_count": 0,
            "field_inventory_sha256": hashlib.sha256(
                json.dumps(fields, sort_keys=True).encode()
            ).hexdigest(),
            "forward_calls": len(calls),
            "forward_shapes": calls,
            "forward_rows": fixtures.FORWARD_ROWS,
            "action_q_tape": action_q_tape,
            "field_digests": tapes,
        },
        snapshots,
        tapes,
    )


def _compare_tapes(
    reference: list[dict[str, torch.Tensor]],
    candidate: list[dict[str, torch.Tensor]],
    *,
    roster: str,
    worlds: int,
    layout: str,
    replay: int,
) -> tuple[int, dict[str, Any] | None]:
    if len(reference) != len(candidate):
        return 0, {
            "roster": roster,
            "worlds": worlds,
            "layout": layout,
            "replay": replay,
            "field": "ticks",
            "byte_offset": 0,
        }
    total = 0
    for tick, (left, right) in enumerate(zip(reference, candidate)):
        total += len(left)
        mismatch = compare.compare_inventory(left, right)
        if mismatch is not None:
            detail = {
                "roster": roster,
                "worlds": worlds,
                "layout": layout,
                "replay": replay,
                "tick": tick,
                "field": mismatch.field,
                "byte_offset": mismatch.byte_offset,
                "expected": mismatch.expected,
                "actual": mismatch.actual,
            }
            value = left.get(mismatch.field)
            if value is not None and value.ndim and value.shape[0] == 4:
                width = value[0].numel() * value.element_size()
                detail["logical_id"] = mismatch.byte_offset // max(1, width)
                if value.ndim > 1 and value.shape[1] == 6:
                    detail["seat"] = (mismatch.byte_offset % max(1, width)) // max(1, width // 6)
            return total, detail
    return total, None


def _compare_action_q(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]], **where: Any
) -> dict[str, Any] | None:
    if len(reference) != len(candidate):
        return {**where, "field": "action_q_tape.length", "byte_offset": 0}
    for tick, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return {**where, "tick": tick, "field": "action_q_tape", "byte_offset": 0}
    return None


def _controls(device: str = "cpu") -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    """Disposable comparator controls; these do not inspect a matrix prefix."""
    base = {
        "output.q.0": torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6),
        "sim.bodies": torch.arange(32, dtype=torch.long, device=device).reshape(4, 2, 2, 2),
        "sim.rng_counter": torch.arange(4, dtype=torch.long, device=device),
    }

    def fault(field: str) -> dict[str, torch.Tensor]:
        copied = {name: value.clone() for name, value in base.items()}
        raw = copied[field].reshape(-1).view(torch.uint8)
        raw[0] ^= 1
        return copied

    witnesses: dict[str, dict[str, Any]] = {}
    statuses: dict[str, bool] = {}
    for label, field in (
        ("q", "output.q.0"),
        ("state", "sim.bodies"),
        ("cursor", "sim.rng_counter"),
    ):
        mismatch = compare.compare_inventory(base, fault(field))
        statuses[label] = mismatch is not None and mismatch.field == field
        witnesses[label] = {} if mismatch is None else mismatch.__dict__
    # This is an actual inverse-map fault: logical targets 0/1 are swapped.
    # It changes which captured physical-row bytes are compared.
    row_map = torch.tensor((3, 2, 1, 0), dtype=torch.long, device=device)
    wrong = {name: value.index_select(0, row_map) for name, value in base.items()}
    mismatch = compare.compare_inventory(base, wrong)
    statuses["row_map"] = mismatch is not None and mismatch.field == "output.q.0"
    witnesses["row_map"] = {} if mismatch is None else mismatch.__dict__
    return statuses, witnesses


def _coverage(args: argparse.Namespace, roster: str) -> dict[str, bool]:
    """Delegate independent two-tick forced fixtures to the event owner."""
    from research.gpu_sim_spike.t4a_events import run_events

    return run_events(
        args.checkpoint,
        roster,
        args.device,
        fixtures.mode(smoke=args.smoke, canary=args.canary),
        args.deadline,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    lane = fixtures.mode(smoke=args.smoke, canary=args.canary)
    determinism = _determinism(args.device)
    if (
        not math.isfinite(args.seconds)
        or not 0 < args.seconds <= {"smoke": 600, "canary": 720, "main": 6000}[lane]
        or args.smoke
        and args.canary
    ):
        raise ValueError("invalid T4A execution arguments")
    if args.device == "cpu" and lane != "smoke":
        raise ValueError("CPU is permitted only for the T4A tiny smoke")
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("T4A CUDA requires an A100")
    controls, control_witnesses = _controls(args.device)
    if not all(controls.values()):
        raise T4AError("injected comparison control did not fail in the expected field")
    commit = source_commit()
    namespace = t4a_units.namespace(commit, smoke=args.smoke, canary=args.canary)
    allocation = t4a_units.allocation(smoke=args.smoke, canary=args.canary)
    completed = t4a_units.load_ledger(args.ledger, namespace, allocation)
    started = time.monotonic()
    args.deadline = started + args.seconds
    horizon = 1 if args.smoke else 2 if args.canary else 256
    sizes = (4,) if args.smoke else WORLD_SIZES
    output: dict[str, Any] = {
        "schema": SCHEMA,
        "commit": commit,
        "source_sha256": source_hashes(),
        "namespace": namespace,
        "allocation": list(allocation),
        "determinism": determinism,
        "completed_units": [],
        "units": [],
    }
    native_started = time.monotonic()
    native = integration_native.run(args.output_dir / "native-parity.json", device=args.device)
    native_elapsed_seconds = time.monotonic() - native_started
    if native.get("status") not in ("PASS", "PREFLIGHT"):
        output.update(status="INCOMPLETE", reason="NATIVE_PARITY_REQUIRED")
        t4a_units.publish_json(args.output_dir / "t4a-bench.json", output)
        return output
    for unit in allocation:
        if unit in completed:
            source = t4a_units.payload_path(args.ledger.parent, unit)
            destination = t4a_units.payload_path(args.output_dir, unit)
            destination.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", ".sha256", ".complete"):
                src, dst = Path(str(source) + suffix), Path(str(destination) + suffix)
                if dst.exists() and dst.read_bytes() != src.read_bytes():
                    raise T4AError("immutable resumed unit differs")
                if not dst.exists():
                    shutil.copyfile(src, dst)
            output["completed_units"].append(unit)
            output["units"].append({"unit_id": unit, "skipped_verified": True})
            continue
        if time.monotonic() >= args.deadline:
            break
        roster = t4a_units.parse(unit)
        records: list[dict[str, Any]] = []
        reference: list[dict[str, torch.Tensor]] | None = None
        reference_actions: list[dict[str, Any]] | None = None
        first_witness = None
        comparisons = 0
        for worlds in sizes:
            for layout in fixtures.LAYOUTS:
                for replay in range(2):
                    record, tape, _unused = _run_record(
                        args, roster, worlds, layout, replay, horizon
                    )
                    if worlds == 4 and layout == "canonical" and replay == 0:
                        reference = tape
                        reference_actions = record["action_q_tape"]
                    if reference is None:
                        raise T4AError("canonical E4 reference was not created first")
                    comparison_started = time.monotonic()
                    count, witness = _compare_tapes(
                        reference, tape, roster=roster, worlds=worlds, layout=layout, replay=replay
                    )
                    comparisons += count
                    record["comparison_count"] = count
                    records.append(record)
                    first_witness = first_witness or witness
                    if reference_actions is None:
                        raise T4AError("canonical E4 action/Q reference was not created")
                    first_witness = first_witness or _compare_action_q(
                        reference_actions,
                        record["action_q_tape"],
                        roster=roster,
                        worlds=worlds,
                        layout=layout,
                        replay=replay,
                    )
                    record["comparison_seconds"] = time.monotonic() - comparison_started
                    if first_witness:
                        t4a_units.publish_json(
                            args.output_dir / "diagnostics/first-mismatch-record.json",
                            {
                                "commit": commit,
                                "roster": roster,
                                "first_witness": first_witness,
                                "reference": records[0],
                                "candidate": record,
                            },
                        )
                        raise T4AMismatch(first_witness)
        events_started = time.monotonic()
        event_result = _coverage(args, roster)
        event_seconds = time.monotonic() - events_started
        for event in event_result.get("evidence", []):
            witness = event.get("witness", {}).get("first_witness")
            if event.get("status") == "INVALID_SELF_CONSISTENCY" and witness:
                raise T4AMismatch(
                    {
                        "roster": roster,
                        "event": event["event"],
                        "worlds": event["worlds"],
                        **witness,
                    }
                )
        coverage_names = (
            "ring_wrap",
            "simultaneous_death_respawn",
            "contested_food_refill",
            "clone_restore",
            "stale_cache",
        )
        coverage = {name: event_result.get(name) is True for name in coverage_names}
        for event in event_result.get("evidence", []):
            if isinstance(event, dict) and event.get("first_witness"):
                first_witness = first_witness or {
                    "event": event.get("event"),
                    **event["first_witness"],
                }
        status = "COMPLETE"
        if first_witness:
            status = "INVALID_SELF_CONSISTENCY"
        elif not all(coverage.values()) or len(records) != len(sizes) * 3 * 2:
            status = "INCOMPLETE"
        fields = sorted(reference[-1]) if reference else []
        field_types = {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted((reference[-1] if reference else {}).items())
        }
        content = {
            "status": status,
            "roster": roster,
            "mode": fixtures.mode(smoke=args.smoke, canary=args.canary),
            "horizon": horizon,
            "seeds": {"root_seed": fixtures.seed_for(roster, smoke=args.smoke, canary=args.canary)},
            "root_seed": fixtures.seed_for(roster, smoke=args.smoke, canary=args.canary),
            "records": records,
            "matrix": {
                "expected": len(sizes) * 3 * 2,
                "completed": len(records),
                "worlds": list(sizes),
            },
            "controls": controls,
            "control_witnesses": control_witnesses,
            "coverage": coverage,
            "event_records": event_result.get("evidence", []),
            "event_seconds": event_seconds,
            "first_witness": first_witness,
            "fields": fields,
            "field_types": field_types,
            "field_inventory_sha256": hashlib.sha256(
                json.dumps(fields, sort_keys=True).encode()
            ).hexdigest(),
            "elapsed_seconds": time.monotonic() - started,
        }
        if status != "COMPLETE":
            diagnostic = args.output_dir / "diagnostics" / f"{unit}.json"
            t4a_units.publish_json(diagnostic, content)
            output["diagnostic_units"] = output.get("diagnostic_units", []) + [unit]
            output["status"] = status
            output["reason"] = "behavioral mismatch" if first_witness else "incomplete roster unit"
            if first_witness:
                t4a_units.publish_json(
                    args.output_dir / "t4a-failure.json",
                    {
                        "schema": SCHEMA,
                        "commit": commit,
                        "namespace": namespace,
                        "source_sha256": output["source_sha256"],
                        "status": status,
                        "unit": unit,
                        "first_witness": first_witness,
                        "native_parity": "native-parity.json",
                        "fields": fields,
                        "field_inventory_sha256": content["field_inventory_sha256"],
                    },
                )
            break
        t4a_units.publish_unit(args.output_dir, namespace, allocation, unit, content)
        output["completed_units"].append(unit)
        output["units"].append({"unit_id": unit, "skipped_verified": False})
    t4a_units.build_ledger(
        args.output_dir,
        args.output_dir / "verified-ledger.json",
        commit=commit,
        smoke=args.smoke,
        canary=args.canary,
    )
    output.setdefault("status", "MEASURED_PENDING_MAC_REVIEW")
    output["native_parity"] = "native-parity.json"
    output["native_elapsed_seconds"] = native_elapsed_seconds
    output["ledger"] = "verified-ledger.json"
    t4a_units.publish_json(args.output_dir / "t4a-bench.json", output)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Bound telemetry and retain verified work on every mechanical failure."""
    torch.set_num_threads(1)
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(110):
            print(
                json.dumps(
                    {"event": "t4a_heartbeat", "elapsed_seconds": time.monotonic() - started}
                ),
                flush=True,
            )

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        return _run(args)
    except Exception as exc:
        mechanical = isinstance(exc, (TimeoutError, torch.cuda.OutOfMemoryError))
        status = (
            "INVALID_SELF_CONSISTENCY"
            if isinstance(exc, T4AMismatch)
            else "INCOMPLETE" if mechanical else "INVALID_EVIDENCE"
        )
        commit = source_commit()
        failure = {
            "schema": SCHEMA,
            "status": status,
            "commit": commit,
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
            "source_sha256": source_hashes(),
        }
        if isinstance(exc, T4AMismatch):
            failure["first_witness"] = exc.witness
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "t4a-failure.json"
        if not failure_path.exists():
            t4a_units.publish_json(failure_path, failure)
        ledger = args.output_dir / "verified-ledger.json"
        if not ledger.exists():
            t4a_units.build_ledger(
                args.output_dir, ledger, commit=commit, smoke=args.smoke, canary=args.canary
            )
        summary = args.output_dir / "t4a-bench.json"
        if not summary.exists():
            t4a_units.publish_json(summary, failure)
        return failure
    finally:
        stopped.set()
        worker.join(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    if result.get("status") != "MEASURED_PENDING_MAC_REVIEW":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
