"""Runtime-only executor for the registered P1 fixed 100M PPO endpoint.

This module has no provider calls.  The qualified Colab supervisor supplies a
small immutable control document and any transitive snapshots; this runner
validates that document, executes exactly one phase, and writes create-only
receipts.  E2 remains on :func:`research.bl_ppo.run.run_train`'s original
admission path.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from research.bl_ppo import p1_contract
from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.recovery import _IMMUTABLE_REF, read_snapshot
from research.bl_ppo.run import run_train

TARGET = 100_000_000
QUAL_PAUSE = 1_000_064
CONTROL_SCHEMA = "bl-p1-control/v1"


def _load_control(path: Path, phase: str) -> dict[str, Any]:
    """Load a complete control document and reject a template or wrong phase."""
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != CONTROL_SCHEMA:
        raise ValueError("unrecognized P1 control document")
    if value.get("phase") != phase:
        raise ValueError("control phase does not match runtime command")
    if value.get("scientific_admission") is not True:
        raise ValueError("P1 manifest is a template or has unresolved dispatch blockers")
    fixed = {
        "target": p1_contract.SHAPE.target,
        "seed": 20261021,
        "arm": "F",
        "worlds": p1_contract.SHAPE.worlds,
        "rollout_steps": p1_contract.SHAPE.rollout_steps,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise ValueError(f"P1 control must freeze {key}={expected}")
    if (
        value.get("processes") != p1_contract.SHAPE.processes
        or value.get("roster", "frozen-equal-mixes") != "frozen-equal-mixes"
    ):
        raise ValueError("P1 control must freeze the 12-worker equal-mix collector")
    commit = os.environ.get("BL_PPO_COMMIT")
    if not commit or value.get("source_commit") != commit:
        raise ValueError("runtime source commit differs from the hash-bound P1 control")
    registration = value.get("registration_sha256")
    if not isinstance(registration, str) or len(registration) != 64:
        raise ValueError("P1 control lacks the frozen registration digest")
    registration_path = Path(__file__).with_name("REGISTRATION_P1.md")
    if not registration_path.is_file() or digest(registration_path) != registration:
        raise ValueError("runtime registration bytes differ from the frozen control digest")
    excluded = value.get("evaluation_excluded_seeds", ())
    bank_sha = value.get("evaluation_bank_sha256")
    if (
        not isinstance(excluded, list)
        or len(excluded) != 384
        or excluded != sorted(set(excluded))
        or not all(isinstance(seed, int) for seed in excluded)
    ):
        raise ValueError("P1 control lacks the fresh-evaluation exclusion inventory")
    if not isinstance(bank_sha, str) or len(bank_sha) != 64:
        raise ValueError("P1 control lacks the frozen fresh-evaluation bank digest")
    audit_sha = value.get("evaluation_audit_sha256")
    audit_path = path.parent / "bank-audit.json"
    if (
        not isinstance(audit_sha, str)
        or len(audit_sha) != 64
        or verify_export(audit_path) != audit_sha
    ):
        raise ValueError("P1 control lacks its verified pre-data bank audit")
    audit = json.loads(audit_path.read_text())
    from research.bl_ppo.p1_eval import check_audit, materialize

    allocation = materialize(value.get("evaluation_round"))
    check_audit(allocation, audit)
    if (
        allocation["allocation_sha256"] != bank_sha
        or sorted(int(row["world_seed"]) for row in allocation["rows"]) != excluded
    ):
        raise ValueError("P1 evaluation seeds differ from the deterministic frozen bank")
    pin_path = Path(__file__).with_name("P1_BANK_PIN.json")
    if not pin_path.is_file():
        raise ValueError("P1 runtime lacks the committed fresh-bank pin")
    pin = json.loads(pin_path.read_text())
    if (
        pin.get("round") != value.get("evaluation_round")
        or pin.get("allocation_sha256") != bank_sha
    ):
        raise ValueError("P1 control evaluation bank differs from the committed fresh-bank pin")
    if phase in {"qual-1m", "tranche"}:
        admission = path.parent / "admission.json"
        verify_export(admission)
        row = json.loads(admission.read_text())
        required = "bl-p1-y3/v1" if phase == "qual-1m" else "bl-p1-complete-job-qualification/v1"
        if row.get("schema") != required or row.get("status") != "PASS":
            raise ValueError("P1 phase lacks its immutable PASS admission evidence")
        if (
            row.get("source_commit") != commit
            or row.get("registration_sha256", registration) != registration
        ):
            raise ValueError("P1 admission identity differs from frozen control")
        expected = value.get("prerequisites", {}).get(
            "y3" if phase == "qual-1m" else "qualification", {}
        )
        if verify_export(admission) != expected.get("sha256"):
            raise ValueError("P1 admission digest differs from the frozen prerequisite")
        if phase == "tranche":
            from research.bl_ppo.p1_manifests import measured_plan

            measured_plan(row)
    return value


def _assignment_endpoint() -> str:
    """Read only the non-secret endpoint identity from the supervisor spec."""
    root = os.environ.get("COLAB_CLI_ROOT")
    if not root:
        raise ValueError("P1 runs only under the qualified supervisor")
    spec = json.loads((Path(root) / "spec.json").read_text())
    endpoint = spec.get("release_endpoint")
    generation = spec.get("generation")
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or not isinstance(generation, str)
        or not generation
    ):
        raise ValueError("supervisor spec lacks assignment endpoint/generation")
    return f"{endpoint}#generation={generation}"


def _snapshot_fingerprint(path: Path) -> str:
    """Hash complete state bytes after sidecar verification.

    The P1 Y3 check intentionally compares the whole persisted snapshot, not
    weights alone: policy, optimizer/PopArt, collector, adapter and RNG all
    contribute to the serialized payload.
    """
    verify_export(path)
    snapshot = read_snapshot(path)
    required = {"policy", "learner", "collector", "adapter", "rng", "counters"}
    if not required.issubset(snapshot):
        raise ValueError("Y3 snapshot is missing a transitive resume component")
    # Do not hash raw pickle bytes: memo/storage ids can vary despite equal
    # state.  This canonical walk covers all semantic snapshot components.
    import dataclasses

    import numpy as np
    import torch

    hasher = hashlib.sha256()
    seen: set[int] = set()

    def put(value: Any) -> None:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            hasher.update(type(value).__qualname__.encode() + b":" + repr(value).encode() + b";")
        elif isinstance(value, torch.Tensor):
            hasher.update(b"tensor" + str(value.dtype).encode() + repr(tuple(value.shape)).encode())
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, enum.Enum):
            hasher.update(
                b"enum:" + type(value).__qualname__.encode() + b":" + str(value.value).encode()
            )
        elif isinstance(value, (torch.device, torch.dtype)):
            hasher.update(b"torch-meta:" + str(value).encode())
        elif isinstance(value, ModuleType):
            hasher.update(b"imported-module:" + value.__name__.encode())
        elif isinstance(value, np.ndarray):
            hasher.update(b"array" + str(value.dtype).encode() + repr(value.shape).encode())
            if value.dtype.hasobject:
                # Object-array bytes are process-local PyObject addresses,
                # not values (the simulator's kill metadata uses this dtype).
                put(value.tolist())
            else:
                hasher.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            hasher.update(b"dict")
            for key in sorted(value, key=lambda item: repr(item)):
                put(key)
                put(value[key])
        elif isinstance(value, (list, tuple)):
            hasher.update(type(value).__name__.encode())
            for item in value:
                put(item)
        elif isinstance(value, deque):
            hasher.update(b"deque" + repr(value.maxlen).encode())
            for item in value:
                put(item)
        elif isinstance(value, random.Random):
            hasher.update(b"random.Random")
            put(value.getstate())
        elif isinstance(value, np.random.Generator):
            hasher.update(b"numpy.Generator")
            put(value.bit_generator.state)
        elif isinstance(value, np.random.RandomState):
            hasher.update(b"numpy.RandomState")
            put(value.get_state())
        elif isinstance(value, np.generic):
            hasher.update(b"numpy.scalar" + str(value.dtype).encode())
            if value.dtype.hasobject:
                put(value.tolist())
            else:
                hasher.update(value.tobytes())
        elif isinstance(value, (set, frozenset)):
            hasher.update(type(value).__name__.encode())
            for item in sorted(value, key=repr):
                put(item)
        elif dataclasses.is_dataclass(value):
            put(dataclasses.asdict(value))
        elif hasattr(value, "__dict__"):
            marker = id(value)
            if marker in seen:
                hasher.update(b"cycle")
                return
            seen.add(marker)
            hasher.update(type(value).__qualname__.encode())
            state = value.__getstate__() if "__getstate__" in type(value).__dict__ else vars(value)
            put(state)
        else:
            raise TypeError(f"cannot canonically fingerprint {type(value)!r}")

    def dereference(value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {_IMMUTABLE_REF}:
            reference = value[_IMMUTABLE_REF]
            target = path.parent / str(reference["name"])
            if verify_export(target) != reference["sha256"]:
                raise ValueError("immutable snapshot component digest mismatch")
            return dereference(torch.load(target, map_location="cpu", weights_only=False))
        if isinstance(value, dict):
            return {key: dereference(item) for key, item in value.items()}
        if isinstance(value, list):
            return [dereference(item) for item in value]
        if isinstance(value, tuple):
            return tuple(dereference(item) for item in value)
        return value

    semantic = {key: dereference(snapshot[key]) for key in sorted(required)}
    # Host/runtime timing differs by assignment and does not influence a next
    # action.  All actor/update/episode counters remain part of the witness.
    semantic["counters"] = {
        key: value
        for key, value in semantic["counters"].items()
        if key not in {"collection_seconds", "learner_seconds"}
    }
    put(semantic)
    return hasher.hexdigest()


def _snapshot_catalog(path: Path) -> list[dict[str, Any]]:
    """List the complete snapshot and every hash-verified immutable sibling."""
    snapshot = read_snapshot(path)
    paths = [path]

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {_IMMUTABLE_REF}:
                target = path.parent / str(value[_IMMUTABLE_REF]["name"])
                if verify_export(target) != value[_IMMUTABLE_REF]["sha256"]:
                    raise ValueError("snapshot immutable sibling digest mismatch")
                paths.append(target)
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(snapshot["adapter"])
    return [
        {"name": item.name, "sha256": verify_export(item), "size_bytes": item.stat().st_size}
        for item in paths
    ]


def _train_args(
    control: dict[str, Any],
    *,
    output: Path,
    checkpoint: Path,
    resume: Path | None,
    durability: bool,
    snapshot_label: str,
) -> SimpleNamespace:
    """Build the explicit P1-only invocation of the shared training loop."""
    pause_at = int(control["pause_at"])
    if control["phase"] == "qual-1m" and pause_at != p1_contract.SHAPE.qualification_pause:
        raise ValueError("qualification must stop at registered 1000064 boundary")
    if control["phase"] == "tranche" and pause_at not in {
        index * p1_contract.SHAPE.tranche_size for index in range(1, 6)
    }:
        raise ValueError("tranche pause is not a frozen 20M boundary")
    return SimpleNamespace(
        command="train",
        output=output,
        b8_checkpoint=checkpoint,
        temperature=None,
        device=os.environ.get("BL_PPO_DEVICE", "cuda"),
        worlds=p1_contract.SHAPE.worlds,
        seed=20261021,
        roster="frozen-equal-mixes",
        # Long-body injection belongs solely to the E4 ruler.  P1's registered
        # F recipe uses ordinary natural resets at every life boundary.
        long_bodies=False,
        native_v8=True,
        rollout_steps=int(control["rollout_steps"]),
        preflight=False,
        allow_tiny_preflight=False,
        arm="F",
        processes=int(control["processes"]),
        e4_receipt=None,
        e9_receipt=None,
        transitions=p1_contract.SHAPE.target,
        checkpoint_every=float(control.get("checkpoint_every_seconds", 1800)),
        horizon=5000,
        resume=resume,
        scale_forecast_approved=False,
        p1_runtime=True,
        p1_registration_sha256=control["registration_sha256"],
        p1_evaluation_bank_sha256=control["evaluation_bank_sha256"],
        p1_excluded_seeds=tuple(control["evaluation_excluded_seeds"]),
        pause_at=pause_at,
        requested_pause_at=int(control.get("requested_pause_at", pause_at)),
        p1_durability=durability,
        p1_trace_actions=1000 if durability else 0,
        p1_snapshot_label=snapshot_label,
    )


def _run_one(
    control: dict[str, Any],
    *,
    output: Path,
    checkpoint: Path,
    resume: Path | None,
    durability: bool,
    snapshot_label: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    return run_train(
        _train_args(
            control,
            output=output,
            checkpoint=checkpoint,
            resume=resume,
            durability=durability,
            snapshot_label=snapshot_label,
        )
    )


def _require_pause(value: dict[str, Any], *, pause: int, durability: bool = False) -> Path:
    if value.get("critic_qualified") is not True:
        critic = value.get("critic_refit") or {}
        raise ValueError(
            f"P1 NOT_READY: E8 held-out critic gate did not pass; "
            f"status={value.get('status')}, stop_reason={value.get('stop_reason')}, "
            f"tripwires={value.get('tripwires', [])}, actor_transitions="
            f"{value.get('counters', {}).get('actor_transitions')}, critic_refit={critic}"
        )
    terminal = pause == p1_contract.SHAPE.target and value.get("status") == "COMPLETE"
    if (value.get("status") != "PAUSED" and not terminal) or value.get("pause_at") != pause:
        raise ValueError(
            f"P1 phase did not stop at its registered pause boundary: "
            f"status={value.get('status')}, expected={pause}, observed={value.get('pause_at')}"
        )
    actor = value.get("counters", {}).get("actor_transitions")
    if (
        not isinstance(actor, int)
        or actor % p1_contract.SHAPE.worlds
        or (actor < pause if durability else actor != pause)
        or value.get("actual_pause_at") != actor
    ):
        raise ValueError("P1 actual actor counter differs from the registered pause boundary")
    snapshot = value.get("snapshot", {})
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str):
        raise ValueError("P1 pause is missing its complete recovery snapshot")
    path = Path(value["_output"]) / snapshot["path"]
    if verify_export(path) != snapshot.get("sha256"):
        raise ValueError("P1 status snapshot digest mismatch")
    state = read_snapshot(path)
    boundary = state.get("boundary", {})
    requested = value.get("requested_pause_at")
    if (
        not isinstance(requested, int)
        or p1_contract.whole_world_pause(requested, p1_contract.SHAPE.worlds) != pause
        or boundary
        != {
            "requested_pause_at": requested,
            "pause_at": pause,
            "actual_pause_at": actor,
            "kind": "durability" if durability else "exact",
        }
        or state["counters"] != value["counters"]
    ):
        raise ValueError("P1 snapshot boundary metadata differs from the run status")
    if not durability and pause == 0 and state["counters"].get("actor_updates") != 0:
        raise ValueError("P1 origin snapshot must be pre-actor")
    if durability:
        trace = value.get("action_trace", {})
        if not isinstance(trace, dict) or trace.get("size", 0) <= 0:
            raise ValueError("Y3 continuation is missing an immutable action trace")
        counters = value["counters"]
        if counters.get("actor_updates", 0) < 3 or counters.get("actor_episode_boundaries", 0) < 1:
            raise ValueError("Y3 did not reach three actor updates and an episode boundary")
    return path


def _read_trace(path: Path) -> dict[str, Any]:
    verify_export(path)
    value = json.loads(path.read_text())
    if value.get("schema") != "bl-ppo-p1-action-trace/v1" or value.get("actions_recorded") != 1000:
        raise ValueError("Y3 trace must contain exactly 1000 recorded actor actions")
    if value.get("full_tape_actions", 0) < 1000 or not isinstance(
        value.get("full_tape_sha256"), str
    ):
        raise ValueError("Y3 trace lacks the complete continuation action tape")
    return value


def _source_continuation_proof(path: Path) -> tuple[dict[str, Any], Path, Path]:
    """Resolve the hash-bound source receipt into trace and final snapshot bytes."""
    verify_export(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "bl-p1-y3-source/v1"
        or value.get("status") != "COMPLETE_SOURCE_CONTINUATION"
    ):
        raise ValueError("Y3 replacement needs the immutable source-continuation receipt")
    trace_meta, snapshot_meta = value.get("source_trace"), value.get("source_continuation_snapshot")
    if not isinstance(trace_meta, dict) or not isinstance(snapshot_meta, dict):
        raise ValueError("source Y3 receipt lacks transitive trace/snapshot references")
    # `p1_inputs` stores hash-verified transport artifacts as flat names.  P1
    # makes the source-origin and source-continuation snapshot basenames unique
    # before they leave the runtime, so this lookup never relies on directories.
    trace = path.parent / str(trace_meta.get("name"))
    snapshot = path.parent / str(snapshot_meta.get("name"))
    if verify_export(trace) != trace_meta.get("sha256") or verify_export(
        snapshot
    ) != snapshot_meta.get("sha256"):
        raise ValueError("source Y3 transitive artifact digest mismatch")
    return value, trace, snapshot


def _receipt(output: Path, value: dict[str, Any]) -> Path:
    return Path(export_json(output / "receipt.json", value)["path"])


def _validate_resume(control: dict[str, Any], resume: Path, input_dir: Path) -> dict[str, Any]:
    """Bind the actual complete state to this phase's exact predecessor."""
    snapshot = read_snapshot(resume)
    identity = snapshot["identity"]
    expected_identity = {
        "p1_runtime": True,
        "arm": "F",
        "seed": 20261021,
        "target_transitions": p1_contract.SHAPE.target,
        "source_commit": control["source_commit"],
        "p1_registration_sha256": control["registration_sha256"],
        "p1_evaluation_bank_sha256": control["evaluation_bank_sha256"],
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("P1 resume identity differs from the frozen trajectory")
    phase = control["phase"]
    start = control.get("start_at", 0)
    if snapshot["counters"].get("actor_transitions") != start:
        raise ValueError("P1 resume is not the exact registered predecessor boundary")
    boundary = snapshot.get("boundary", {})
    requested = boundary.get("requested_pause_at")
    if (
        not isinstance(requested, int)
        or p1_contract.whole_world_pause(requested, p1_contract.SHAPE.worlds) != start
        or boundary.get("pause_at") != start
        or boundary.get("actual_pause_at") != start
        or boundary.get("kind") != "exact"
    ):
        raise ValueError("P1 resume metadata differs from its exact predecessor boundary")
    if phase == "qual-1m":
        admission = json.loads((input_dir / "admission.json").read_text())
        expected_sha = admission.get("origin_snapshot_sha256")
    elif phase == "tranche":
        index = control.get("chunk_index")
        expected_start = (
            p1_contract.SHAPE.qualification_pause
            if index == 1
            else (index - 1) * p1_contract.SHAPE.tranche_size
        )
        if start != expected_start or control["pause_at"] != index * p1_contract.SHAPE.tranche_size:
            raise ValueError("P1 tranche boundary differs from the fixed chunk plan")
        receipt = input_dir / ("admission.json" if index == 1 else "previous.json")
        checksum = verify_export(receipt)
        prior = json.loads(receipt.read_text())
        if index != 1 and (
            checksum != control.get("previous_receipt_sha256")
            or prior.get("schema") != "bl-p1-tranche/v1"
            or prior.get("status") != "PAUSED"
            or prior.get("actor_transitions") != start
            or prior.get("source_commit") != control["source_commit"]
        ):
            raise ValueError("P1 preceding tranche receipt identity differs")
        expected_sha = prior.get("snapshot", {}).get("sha256")
    else:
        expected_sha = verify_export(resume)  # Y3 source proof is checked separately below.
    if not expected_sha or verify_export(resume) != expected_sha:
        raise ValueError("P1 resume digest differs from its required predecessor receipt")
    # Check referenced ordinary/refit banks before allocating model/worker state.
    _snapshot_catalog(resume)
    return snapshot


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Execute exactly one P1 runtime phase and publish its immutable receipt."""
    control = _load_control(args.control, args.phase)
    output = args.output.resolve()
    started = time.monotonic()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("missing hash-verified B8 checkpoint")
    from research.bl_ppo.p1_eval import B8_SHA256

    if digest(checkpoint) != B8_SHA256:
        raise ValueError("P1 requires the exact released B8 s3 checkpoint")
    assignment_endpoint = _assignment_endpoint()
    if args.phase == "y3-source":
        # The origin is post-E8-refit and pre-actor.  Both the source and
        # replacement continuations begin from these exact bytes.
        origin_control = dict(control, pause_at=0, requested_pause_at=0)
        origin = _run_one(
            origin_control,
            output=output / "origin",
            checkpoint=checkpoint,
            resume=None,
            durability=False,
            snapshot_label="p1-y3-origin",
        )
        origin["_output"] = str(output / "origin")
        origin_snapshot = _require_pause(origin, pause=0)
        continuation = _run_one(
            control,
            output=output / "source-continuation",
            checkpoint=checkpoint,
            resume=origin_snapshot,
            durability=True,
            snapshot_label="p1-y3-source-continuation",
        )
        continuation["_output"] = str(output / "source-continuation")
        continuation_snapshot = _require_pause(continuation, pause=0, durability=True)
        trace_path = output / "source-continuation" / "action-trace.json"
        trace = _read_trace(trace_path)
        result = {
            "schema": "bl-p1-y3-source/v1",
            "status": "COMPLETE_SOURCE_CONTINUATION",
            "phase": args.phase,
            "assignment_endpoint": assignment_endpoint,
            "origin_snapshot": {
                "name": origin_snapshot.name,
                "relative_path": str(origin_snapshot.relative_to(output)),
                "sha256": verify_export(origin_snapshot),
                "state_fingerprint": _snapshot_fingerprint(origin_snapshot),
            },
            "source_continuation_snapshot": {
                "name": continuation_snapshot.name,
                "relative_path": str(continuation_snapshot.relative_to(output)),
                "sha256": verify_export(continuation_snapshot),
                "state_fingerprint": _snapshot_fingerprint(continuation_snapshot),
            },
            "source_trace": {
                "name": trace_path.name,
                "relative_path": str(trace_path.relative_to(output)),
                "sha256": verify_export(trace_path),
                "rows": trace["actions_recorded"],
                "full_tape_actions": trace["full_tape_actions"],
                "full_tape_sha256": trace["full_tape_sha256"],
            },
            "e2_status": {
                "origin_snapshot": origin_snapshot.name,
                "origin_status": origin["status"],
                "continuation_snapshot": continuation_snapshot.name,
                "continuation_status": continuation["status"],
            },
            "critic_qualified": True,
            "transitive_snapshot_catalog": _snapshot_catalog(origin_snapshot)
            + _snapshot_catalog(continuation_snapshot),
        }
    else:
        if args.resume is None:
            raise ValueError(f"{args.phase} requires a hash-verified transitive resume snapshot")
        resume = args.resume.resolve()
        _validate_resume(control, resume, args.control.parent)
        durability = args.phase == "y3-replacement"
        proof = expected_path = expected_snapshot = expected = None
        source_continuation_fingerprint = None
        if durability:
            if args.expected_trace is None:
                raise ValueError("replacement requires the source continuation receipt")
            proof, expected_path, expected_snapshot = _source_continuation_proof(
                args.expected_trace.resolve()
            )
            expected = _read_trace(expected_path)
            source_continuation_fingerprint = _snapshot_fingerprint(expected_snapshot)
            if proof.get("assignment_endpoint") == assignment_endpoint:
                raise ValueError("Y3 replacement must use a different L4 assignment")
            if verify_export(resume) != proof.get("origin_snapshot", {}).get("sha256"):
                raise ValueError("replacement resume is not the source Y3 origin snapshot")
        value = _run_one(
            control,
            output=output / args.phase,
            checkpoint=checkpoint,
            resume=resume,
            durability=durability,
            snapshot_label=f"p1-{args.phase}",
        )
        value["_output"] = str(output / args.phase)
        snapshot = _require_pause(value, pause=int(control["pause_at"]), durability=durability)
        if args.phase == "y3-replacement":
            observed_path = output / args.phase / "action-trace.json"
            observed = _read_trace(observed_path)
            source_fingerprint = _snapshot_fingerprint(resume)
            replacement_fingerprint = _snapshot_fingerprint(snapshot)
            if (
                expected["rows"] != observed["rows"]
                or expected["full_tape_sha256"] != observed["full_tape_sha256"]
            ):
                raise ValueError("Y3 replacement action trace differs from source continuation")
            if source_continuation_fingerprint != replacement_fingerprint:
                raise ValueError("Y3 replacement full state differs after the matched continuation")
            result = {
                "schema": "bl-p1-y3/v1",
                "status": "PASS",
                "phase": args.phase,
                "critic_qualified": True,
                "origin_snapshot_sha256": verify_export(resume),
                "origin_state_fingerprint": source_fingerprint,
                "replacement_snapshot_sha256": verify_export(snapshot),
                "replacement_state_fingerprint": replacement_fingerprint,
                "source_receipt_sha256": verify_export(args.expected_trace),
                "source_assignment_endpoint": proof["assignment_endpoint"],
                "replacement_assignment_endpoint": assignment_endpoint,
                "source_endpoint": str(proof["assignment_endpoint"]).split("#generation=", 1)[0],
                "replacement_endpoint": assignment_endpoint.split("#generation=", 1)[0],
                "expected_trace_sha256": verify_export(expected_path),
                "actual_trace_sha256": verify_export(observed_path),
                "source_continuation_state_fingerprint": source_continuation_fingerprint,
                "actions_compared": 1000,
                "full_tape_actions_compared": observed["full_tape_actions"],
                "actor_updates": value["counters"]["actor_updates"],
                "episode_boundaries": value["counters"]["actor_episode_boundaries"],
                "transitive_snapshot_components": [
                    "model",
                    "optimizer",
                    "popart",
                    "collector",
                    "adapter",
                    "rng",
                ],
                "origin_snapshot_files": _snapshot_catalog(resume),
                "replacement_snapshot_files": _snapshot_catalog(snapshot),
            }
        elif args.phase == "qual-1m":
            actor = int(value["counters"]["actor_transitions"])
            wall = float(value["wall_seconds"])
            result = {
                "schema": "bl-p1-qualification-runtime/v1",
                # Only the Mac-side finalizer can append the supervisor's
                # transport, terminal export and verified-release evidence.
                "status": "PAUSED",
                "phase": args.phase,
                "critic_qualified": True,
                "target": p1_contract.SHAPE.target,
                "requested_pause_at": p1_contract.SHAPE.qualification_requested,
                "actual_pause_at": actor,
                "actor_transitions": actor,
                "boundary_overshoot": actor - p1_contract.SHAPE.qualification_requested,
                "actor_phase_seconds": wall,
                "run_wall_seconds": wall,
                "collection_seconds": value["counters"]["collection_seconds"],
                "learner_seconds": value["counters"]["learner_seconds"],
                "probe_seconds": max(
                    0.0,
                    wall
                    - float(value["counters"]["collection_seconds"])
                    - float(value["counters"]["learner_seconds"]),
                ),
                "timing_includes_collection_updates_probes": True,
                "snapshot": {"name": snapshot.name, "sha256": verify_export(snapshot)},
                "origin_snapshot_sha256": verify_export(resume),
            }
        else:
            result = {
                "schema": "bl-p1-tranche/v1",
                "status": "COMPLETE" if value["status"] == "COMPLETE" else "PAUSED",
                "phase": args.phase,
                "critic_qualified": True,
                "target": p1_contract.SHAPE.target,
                "pause_at": control["pause_at"],
                "actor_transitions": value["counters"]["actor_transitions"],
                "runner_wall_seconds": value["wall_seconds"],
                "collection_seconds": value["counters"]["collection_seconds"],
                "learner_seconds": value["counters"]["learner_seconds"],
                "snapshot": {"name": snapshot.name, "sha256": verify_export(snapshot)},
                "endpoint_sha256": value.get("checkpoint_sha256"),
            }
    result.update(
        control_sha256=verify_export(args.control),
        source_commit=control["source_commit"],
        registration_sha256=control["registration_sha256"],
        runtime_seconds=time.monotonic() - started,
    )
    _receipt(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("y3-source", "y3-replacement", "qual-1m", "tranche"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--expected-trace", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    value = execute(build_parser().parse_args(argv))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
