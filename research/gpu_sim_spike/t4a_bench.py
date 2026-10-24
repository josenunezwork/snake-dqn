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

    def __init__(
        self,
        witness: dict[str, Any],
        *,
        fields: list[str] | None = None,
        field_types: dict[str, Any] | None = None,
        comparison_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.witness = witness
        self.fields = fields
        self.field_types = field_types
        self.comparison_records = comparison_records
        super().__init__(str(witness))


class T4AEvidenceError(T4AError):
    """A capture cannot be compared as equal-shape, equal-dtype byte evidence."""


def _event_exception(witness: dict[str, Any], **context: Any) -> T4AMismatch:
    """Preserve the exact event seam's schema on the terminal failure path."""
    try:
        records = witness["comparison_records"]
        binding = witness["comparison"]["reference"]
        record = next(
            row
            for row in records
            if all(row[key] == binding[key] for key in ("worlds", "layout", "replay"))
        )
        descriptors = record["field_digests"][binding["tick"]]
        if witness["field"] not in descriptors:
            raise KeyError(witness["field"])
        return T4AMismatch(
            {**context, **witness},
            fields=sorted(descriptors),
            field_types={
                name: {"dtype": descriptor["dtype"], "shape": descriptor["shape"]}
                for name, descriptor in descriptors.items()
            },
            comparison_records=records,
        )
    except (IndexError, KeyError, StopIteration, TypeError) as exc:
        raise T4AEvidenceError("event mismatch lacks its exact compared seam") from exc


def _require_valid_event_evidence(evidence: list[dict[str, Any]], output_dir: Path) -> None:
    """Retain completed harness faults and send them through the failure writer."""
    invalid = [row for row in evidence if row.get("status") == "INVALID_EVIDENCE"]
    if invalid:
        t4a_units.publish_json(
            output_dir / "diagnostics/invalid-event-evidence.json", {"event_records": invalid}
        )
        raise T4AEvidenceError(
            "completed event witness is INVALID_EVIDENCE; see retained diagnostics"
        )


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
    """Compatibility accessor for the separately checked extraction order."""
    return fixtures.row_map_proof(worlds, layout).extraction_rows


def _extract_capture(
    physical_fields: dict[str, torch.Tensor], extraction_rows: tuple[int, int, int, int]
) -> dict[str, torch.Tensor]:
    """Extract captured physical rows into frozen logical-ID order.

    Matrix records and disposable controls both traverse this helper before
    reaching ``_compare_tapes``.  That prevents controls from validating only
    the leaf comparator while bypassing the real row-map boundary.
    """
    if len(extraction_rows) != 4 or len(set(extraction_rows)) != 4:
        raise T4AError("invalid extraction row map")
    indices: torch.Tensor | None = None
    extracted: dict[str, torch.Tensor] = {}
    for name, value in sorted(physical_fields.items()):
        if value.ndim < 1:
            raise T4AError(f"captured field lacks physical row axis: {name}")
        if indices is None:
            indices = torch.tensor(extraction_rows, dtype=torch.long, device=value.device)
        if int(value.shape[0]) <= max(extraction_rows):
            raise T4AError(f"captured field cannot satisfy extraction map: {name}")
        extracted[name] = value.index_select(0, indices).contiguous()
    return extracted


def _run_record(
    args: argparse.Namespace,
    roster: str,
    worlds: int,
    layout: str,
    replay: int,
    horizon: int,
    *,
    extraction_rows: tuple[int, int, int, int] | None = None,
    control_fault: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]], list[Any]]:
    built_at = time.monotonic()
    case, wrappers, row_proof = fixtures.build_mapped_case(
        args.checkpoint,
        roster,
        worlds,
        layout,
        args.device,
        smoke=args.smoke,
        canary=args.canary,
    )
    selected_rows = extraction_rows or row_proof.extraction_rows
    if len(selected_rows) != 4 or len(set(selected_rows)) != 4:
        raise T4AError("invalid control extraction map")
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
                if control_fault is not None:
                    fields = {
                        "q": capture.values.get("output.q.0"),
                        "state": sim.bodies,
                        "cursor": sim.rng_counter,
                    }
                    value = fields.get(control_fault)
                    if not isinstance(value, torch.Tensor):
                        raise T4AError("unknown actual-capture control fault")
                    # Mutate the captured physical row before normal
                    # extraction, never an already logical-order snapshot.
                    raw = value[selected_rows[0]].reshape(-1).view(torch.uint8)
                    raw[0] ^= 1
                # Both inventories select physical rows using the independent
                # extraction map; their resulting leading axis is logical-ID
                # order 0..3 for comparison.
                state = _target_inventory((*case[:4], collection, case[5]), selected_rows)
                state.update(capture.fields(selected_rows))
                actions = capture.values["output.actions"]
                action_q_tape.append(capture.q_tape(actions, selected_rows))
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
            "map_proof": row_proof.exported(),
            "extracted_rows": list(selected_rows),
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
        raise T4AEvidenceError("comparison tape tick inventory differs")
    for left, right in zip(reference, candidate):
        if set(left) != set(right):
            raise T4AEvidenceError("comparison tape field inventory differs")
        for field in left:
            if left[field].dtype != right[field].dtype or tuple(left[field].shape) != tuple(
                right[field].shape
            ):
                raise T4AEvidenceError(f"comparison tape schema differs for {field}")
    total = 0
    for tick, (left, right) in enumerate(zip(reference, candidate)):
        total += len(left)
        mismatch = compare.compare_inventory(left, right)
        if isinstance(mismatch, compare.SchemaIncompatibility):
            raise T4AEvidenceError(
                "incompatible comparison evidence "
                f"({mismatch.field}: {mismatch.reason}; "
                f"expected={mismatch.expected_dtype}/{mismatch.expected_shape}; "
                f"actual={mismatch.actual_dtype}/{mismatch.actual_shape})"
            )
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
                # These descriptors bind the claimed unequal byte to the two
                # actual tensor captures.  The reducer independently checks
                # their schema and the offset bounds before accepting a
                # self-consistency failure.
                "expected_descriptor": compare.compact_inventory(
                    {mismatch.field: left[mismatch.field]}
                )[mismatch.field],
                "actual_descriptor": compare.compact_inventory(
                    {mismatch.field: right[mismatch.field]}
                )[mismatch.field],
                "comparison": {
                    "reference": {
                        "worlds": 4,
                        "layout": "canonical",
                        "replay": 0,
                        "tick": tick,
                    },
                    "candidate": {
                        "worlds": worlds,
                        "layout": layout,
                        "replay": replay,
                        "tick": tick,
                    },
                },
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
        raise T4AEvidenceError("action/Q capture tick inventory differs")
    for tick, (left, right) in enumerate(zip(reference, candidate)):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise T4AEvidenceError("action/Q capture is not an object")
        for key, field, dtype, shape in (("actions_hex", "output.actions", "torch.int64", [4, 6]),):
            expected_hex, actual_hex = left.get(key), right.get(key)
            if not isinstance(expected_hex, str) or not isinstance(actual_hex, str):
                raise T4AEvidenceError(f"action/Q capture omits {key}")
            try:
                expected_raw, actual_raw = bytes.fromhex(expected_hex), bytes.fromhex(actual_hex)
            except ValueError as exc:
                raise T4AEvidenceError(f"action/Q capture has invalid hex for {key}") from exc
            if len(expected_raw) != len(actual_raw):
                raise T4AEvidenceError(f"action/Q capture shape differs for {key}")
            if expected_raw != actual_raw:
                offset = next(
                    i for i, pair in enumerate(zip(expected_raw, actual_raw)) if pair[0] != pair[1]
                )
                return {
                    **where,
                    "tick": tick,
                    "field": field,
                    "byte_offset": offset,
                    "expected": expected_raw[offset],
                    "actual": actual_raw[offset],
                    "expected_descriptor": {
                        "dtype": dtype,
                        "shape": shape,
                        "sha256": hashlib.sha256(expected_raw).hexdigest(),
                    },
                    "actual_descriptor": {
                        "dtype": dtype,
                        "shape": shape,
                        "sha256": hashlib.sha256(actual_raw).hexdigest(),
                    },
                    "comparison": {
                        "reference": {
                            "worlds": where["reference_worlds"],
                            "layout": where["reference_layout"],
                            "replay": where["reference_replay"],
                            "tick": tick,
                        },
                        "candidate": {
                            "worlds": where["worlds"],
                            "layout": where["layout"],
                            "replay": where["replay"],
                            "tick": tick,
                        },
                    },
                }
        expected_q, actual_q = left.get("q_hex"), right.get("q_hex")
        if not isinstance(expected_q, dict) or not isinstance(actual_q, dict):
            raise T4AEvidenceError("action/Q capture omits Q inventory")
        if set(expected_q) != set(actual_q):
            raise T4AEvidenceError("action/Q capture Q inventory differs")
        for slot in sorted(expected_q):
            expected_hex, actual_hex = expected_q[slot], actual_q[slot]
            if not isinstance(expected_hex, str) or not isinstance(actual_hex, str):
                raise T4AEvidenceError("action/Q capture Q bytes are malformed")
            try:
                expected_raw, actual_raw = bytes.fromhex(expected_hex), bytes.fromhex(actual_hex)
            except ValueError as exc:
                raise T4AEvidenceError("action/Q capture Q bytes are malformed") from exc
            if len(expected_raw) != len(actual_raw):
                raise T4AEvidenceError("action/Q capture Q shape differs")
            if expected_raw != actual_raw:
                offset = next(
                    i for i, pair in enumerate(zip(expected_raw, actual_raw)) if pair[0] != pair[1]
                )
                return {
                    **where,
                    "tick": tick,
                    "field": f"output.q.{slot}",
                    "byte_offset": offset,
                    "expected": expected_raw[offset],
                    "actual": actual_raw[offset],
                    "expected_descriptor": {
                        "dtype": "torch.float32",
                        "shape": [4, 6],
                        "sha256": hashlib.sha256(expected_raw).hexdigest(),
                    },
                    "actual_descriptor": {
                        "dtype": "torch.float32",
                        "shape": [4, 6],
                        "sha256": hashlib.sha256(actual_raw).hexdigest(),
                    },
                    "comparison": {
                        "reference": {
                            "worlds": where["reference_worlds"],
                            "layout": where["reference_layout"],
                            "replay": where["reference_replay"],
                            "tick": tick,
                        },
                        "candidate": {
                            "worlds": where["worlds"],
                            "layout": where["layout"],
                            "replay": where["replay"],
                            "tick": tick,
                        },
                    },
                }
    return None


def _controls(device: str = "cpu") -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    """Inject faults through captured physical rows and normal extraction."""
    physical_rows = 8
    extraction_rows = (7, 5, 3, 1)
    base = {
        "output.q.0": torch.arange(physical_rows * 6, dtype=torch.float32, device=device).reshape(
            physical_rows, 6
        ),
        "sim.bodies": torch.arange(physical_rows * 8, dtype=torch.long, device=device).reshape(
            physical_rows, 2, 2, 2
        ),
        "sim.rng_counter": torch.arange(physical_rows, dtype=torch.long, device=device),
    }
    reference = [_extract_capture(base, extraction_rows)]

    def fault(field: str) -> dict[str, torch.Tensor]:
        copied = {name: value.clone() for name, value in base.items()}
        # Mutate a physical row selected by the same extraction map, rather
        # than a convenient but unobserved row zero.
        raw = copied[field][extraction_rows[0]].reshape(-1).view(torch.uint8)
        raw[0] ^= 1
        return copied

    witnesses: dict[str, dict[str, Any]] = {}
    statuses: dict[str, bool] = {}
    for label, field in (
        ("q", "output.q.0"),
        ("state", "sim.bodies"),
        ("cursor", "sim.rng_counter"),
    ):
        _count, mismatch = _compare_tapes(
            reference,
            [_extract_capture(fault(field), extraction_rows)],
            roster="control",
            worlds=physical_rows,
            layout="reversed",
            replay=0,
        )
        statuses[label] = mismatch is not None and mismatch.get("field") == field
        witnesses[label] = {} if mismatch is None else mismatch
    # A wrong extraction map is applied to the same captured physical rows.
    # It cannot pass by only reordering a handmade already-extracted tape.
    _count, mismatch = _compare_tapes(
        reference,
        [_extract_capture(base, tuple(reversed(extraction_rows)))],
        roster="control",
        worlds=physical_rows,
        layout="reversed",
        replay=0,
    )
    statuses["row_map"] = mismatch is not None and mismatch.get("field") == "output.q.0"
    witnesses["row_map"] = {} if mismatch is None else mismatch
    return statuses, witnesses


def _actual_capture_controls(
    args: argparse.Namespace,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    """Run disposable faults through a real Capture -> extraction -> comparison.

    These records are never added to the registered matrix or evidence units.
    They validate only that the capture and row-map boundary cannot conceal a
    deliberately injected Q/state/cursor/extraction defect.
    """
    _record, reference, _digests = _run_record(args, "frozen", 4, "canonical", -1, 1)
    statuses: dict[str, bool] = {}
    witnesses: dict[str, dict[str, Any]] = {}
    for label, expected_field in (
        ("q", "output.q.0"),
        ("state", "sim.bodies"),
        ("cursor", "sim.rng_counter"),
    ):
        _record, candidate, _digests = _run_record(
            args, "frozen", 4, "canonical", -1, 1, control_fault=label
        )
        _count, mismatch = _compare_tapes(
            reference,
            candidate,
            roster="control",
            worlds=4,
            layout="canonical",
            replay=-1,
        )
        statuses[label] = mismatch is not None and mismatch.get("field") == expected_field
        witnesses[label] = {} if mismatch is None else mismatch
    _record, candidate, _digests = _run_record(
        args, "frozen", 4, "canonical", -1, 1, extraction_rows=(3, 2, 1, 0)
    )
    _count, mismatch = _compare_tapes(
        reference,
        candidate,
        roster="control",
        worlds=4,
        layout="canonical",
        replay=-1,
    )
    statuses["row_map"] = mismatch is not None
    witnesses["row_map"] = {} if mismatch is None else mismatch
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
    commit = source_commit()
    if args.device == "cuda":
        # Preload before disposable controls: controls exercise the same real
        # policy path and must not trigger an unreceipted lazy native build.
        from research.gpu_sim_spike.t4a_native import preload_from_environment

        preload_from_environment(commit)
    started = time.monotonic()
    args.deadline = started + args.seconds
    controls, control_witnesses = _actual_capture_controls(args)
    if not all(controls.values()):
        raise T4AError("injected comparison control did not fail in the expected field")
    namespace = t4a_units.namespace(commit, smoke=args.smoke, canary=args.canary)
    allocation = t4a_units.allocation(smoke=args.smoke, canary=args.canary)
    completed = t4a_units.load_ledger(args.ledger, namespace, allocation)
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
                        reference_worlds=4,
                        reference_layout="canonical",
                        reference_replay=0,
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
                        reference_fields = sorted(reference[-1]) if reference else []
                        raise T4AMismatch(
                            first_witness,
                            fields=reference_fields,
                            field_types={
                                name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                                for name, value in sorted(
                                    (reference[-1] if reference else {}).items()
                                )
                            },
                            comparison_records=records,
                        )
        events_started = time.monotonic()
        event_result = _coverage(args, roster)
        event_seconds = time.monotonic() - events_started
        for event in event_result.get("evidence", []):
            witness = event.get("witness", {}).get("first_witness")
            if event.get("status") == "INVALID_SELF_CONSISTENCY" and witness:
                raise _event_exception(
                    witness, roster=roster, event=event["event"], worlds=event["worlds"]
                )
        _require_valid_event_evidence(event_result.get("evidence", []), args.output_dir)
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
                        "field_types": field_types,
                        "comparison_records": records,
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
    cuda_memory = (
        str(getattr(args, "device", "cpu")).startswith("cuda") and torch.cuda.is_available()
    )
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
        if cuda_memory:
            torch.cuda.reset_peak_memory_stats(args.device)
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
            if exc.fields is not None:
                failure["fields"] = exc.fields
            if exc.field_types is not None:
                failure["field_types"] = exc.field_types
            if exc.comparison_records is not None:
                failure["comparison_records"] = exc.comparison_records
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
        # Separate resource telemetry, never a replacement for any comparison
        # or verdict. Measure the complete invocation, including all E1024
        # fixtures, rather than extrapolating from CPU or four-target captures.
        if cuda_memory:
            try:
                memory = {
                    "status": "MEASURED",
                    "scope": "whole T4A invocation since reset_peak_memory_stats",
                    "device": str(args.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                    "total_device_bytes": torch.cuda.get_device_properties(
                        args.device
                    ).total_memory,
                    "body_digest_chunk_bytes": compare.BODY_DIGEST_CHUNK_BYTES,
                    "body_digest_tensor_scratch_bound": compare.BODY_DIGEST_SCRATCH_BYTES,
                }
            except Exception as exc:
                memory = {"status": "UNAVAILABLE", "reason": type(exc).__name__}
            try:
                t4a_units.publish_json(args.output_dir / "device-memory.json", memory)
            except Exception as exc:
                # A telemetry I/O failure must never replace a sealed mismatch
                # or change the existing scientific verdict. The packet's
                # required-export gate will refuse missing memory evidence.
                print(
                    json.dumps(
                        {"event": "device_memory_export_failed", "reason": type(exc).__name__}
                    ),
                    flush=True,
                )


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
