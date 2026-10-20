"""Create-only paired Round 0 trace producer.

The real controller is evaluated twice on every allocated world: first as bare
B8 and then with the Amendment-2 shadow/LAD teacher acting at beta=1.  The two
traces are retained for identity replay, but teacher labels are deliberately
removed before the public Round 0 export is written.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
from research.t34_exit_b8.exports import load_world, state_hash, write_world
from research.t34_exit_b8.policy import B8Policy
from research.t34_exit_b8.teacher_v2 import _step, run_worlds
from research.t34_exit_b8.trajectory import build_world, prepare_world
from research.t34_round0.faults import (
    CapExpiry,
    ExportSerializationError,
    IllegalActionFault,
    NonFiniteQLossFault,
    OneActionShareFault,
)

ROOT = Path(__file__).resolve().parents[2]
PAIR_SCHEMA = "t34-round0-pair/v1"
CHUNK_SCHEMA = "t34-round0-chunk/v1"
WORLD_SCHEMA = "t34-exit-world/v1"
HORIZON = 5000
PRODUCTION_WORKERS = 12


class Round0Error(RuntimeError):
    """A fail-closed Round 0 producer error."""


class Round0PairError(Round0Error):
    """A worker failure with enough context to preserve a KILL record."""

    def __init__(
        self,
        arm: str,
        world_id: str,
        cause: BaseException | str,
        marker: str | None = None,
        operational_kind: str | None = None,
    ) -> None:
        """Retain only primitive fields so spawn-pool exception transport is safe."""
        super().__init__(str(cause))
        self.arm = arm
        self.world_id = world_id
        self.marker = marker if marker is not None else _fault_marker(cause)
        self.operational_kind = (
            operational_kind if operational_kind is not None else _operational_kind(cause)
        )

    def __reduce__(self) -> tuple[Any, tuple[str, str, str, str | None, str | None]]:
        """Reconstruct contextual fault data when ProcessPoolExecutor unpickles us."""
        return (
            type(self),
            (self.arm, self.world_id, str(self), self.marker, self.operational_kind),
        )


def _fault_marker(exc: BaseException) -> str | None:
    """Return the registered marker for an explicit controller-fault type only."""
    markers: dict[type[BaseException], str] = {
        NonFiniteQLossFault: "nonfinite_q",
        IllegalActionFault: "illegal_action",
        OneActionShareFault: "action_collapse_100k",
    }
    return markers.get(type(exc))


def _operational_kind(exc: BaseException) -> str | None:
    """Identify explicitly wrapped operational failures without text inspection."""
    if type(exc) is ExportSerializationError:
        return "export_or_serialization"
    if type(exc) is CapExpiry:
        return "cap_expired"
    return None


class _ActionWindow:
    """O(1) exact 100k-action collapse guard for one controller arm."""

    def __init__(self) -> None:
        self.actions: deque[int] = deque()
        self.counts = [0] * 6

    def add(self, action: int) -> bool:
        if not 0 <= action < 6:
            raise IllegalActionFault("illegal hero action")
        if len(self.actions) == 100000:
            self.counts[self.actions.popleft()] -= 1
        self.actions.append(action)
        self.counts[action] += 1
        return len(self.actions) == 100000 and max(self.counts) > 90000


def sha256(path: Path) -> str:
    """Return a byte SHA-256 without accepting directories."""
    if not path.is_file():
        raise Round0Error(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    digest = sha256(path)
    with path.with_name(f"{path.name}.sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {path.name}\n")
    path.with_name(f"{path.name}.complete").touch(exist_ok=False)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Round0Error(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Round0Error(f"JSON object required: {path}")
    return value


def _safe_world_id(world_id: str) -> str:
    return hashlib.sha256(world_id.encode("utf-8")).hexdigest()[:24]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _validate_bank(bank: Mapping[str, Any], *, preflight: bool) -> list[dict[str, Any]]:
    if bank.get("schema") != "t34-round0-bank/v1":
        raise Round0Error("unknown Round 0 bank schema")
    rows = bank.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Round0Error("bank requires nonempty rows")
    if bool(bank.get("preflight", False)) != preflight:
        raise Round0Error("bank preflight status differs from invocation")
    expected = 2 if preflight else 240
    if len(rows) != expected:
        raise Round0Error(
            f"{'preflight' if preflight else 'production'} bank requires {expected} rows"
        )
    ids: set[str] = set()
    clean: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise Round0Error("bank row must be an object")
        world_id, mix = row.get("world_id"), row.get("mix")
        if not isinstance(world_id, str) or not world_id or world_id in ids:
            raise Round0Error("bank world ids must be unique nonempty strings")
        if mix not in ("scripted", "frozen", "mixed"):
            raise Round0Error("bank mix is invalid")
        if int(row.get("horizon", -1)) != (500 if preflight else HORIZON):
            raise Round0Error("bank horizon differs from governed mode")
        if row.get("opponent_veto") != "v8" or not isinstance(row.get("opponents"), list):
            raise Round0Error("bank requires v8 opponents")
        int(row["world_seed"])
        ids.add(world_id)
        clean.append(dict(row))
    return clean


def _validate_audit(audit_path: Path, bank_path: Path) -> dict[str, Any]:
    audit = _load(audit_path)
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != sha256(bank_path):
        raise Round0Error("seed audit is not CLEAR for these bank bytes")
    return audit


def _registered_bank(registration: Path, bank: Path, audit: Path) -> list[dict[str, Any]]:
    """Use the common admission guard once its sibling is present.

    The fallback is intentionally strict enough for a standalone producer test;
    the shared validator remains the production identity authority.
    """
    try:
        from research.t34_round0.registration import require_registration
    except ImportError as exc:  # pragma: no cover - integration ordering guard
        raise Round0Error("Round 0 registration validator is unavailable") from exc
    value = require_registration(_load(registration), bank, audit, repo=ROOT)
    if not isinstance(value, Mapping):
        raise Round0Error("registration validator did not return a bank")
    return _validate_bank(value, preflight=False)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    """Hash an admission object with the history ledger's canonical JSON encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_dispatch_admission(
    admission: Path | None, ratification: Path, expected_world_ids: list[str]
) -> dict[str, Any]:
    """Require the remote create-only claim copied into this frozen runtime."""
    if admission is None:
        raise Round0Error("production requires a dispatch admission receipt")
    value = _load(admission)
    registration = _load(ratification)
    phase_path = value.get("phase_path")
    if (
        value.get("schema") != "t34-round0-admission/v1"
        or value.get("registration_sha256") != _canonical_sha(registration)
        or not isinstance(phase_path, str)
        or not re.search(r"(?:^|/)chunks/chunk-\d{3}/attempt-0[12]$", phase_path)
        or not isinstance(value.get("claim_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", value["claim_commit"])
        or value.get("world_ids") != expected_world_ids
    ):
        raise Round0Error("dispatch admission receipt is not a fixed registered chunk attempt")
    return value


def _validate_execution(*, preflight: bool, device: str, threads: int, seconds: float) -> None:
    if not math.isfinite(seconds) or not 0 < seconds <= 21000:
        raise Round0Error("operational wall bound must be in (0, 21000]")
    if preflight:
        if platform.system() != "Darwin" or device != "cpu" or threads != 1:
            raise Round0Error("preflight requires Darwin, cpu, and exactly one thread")
        return
    if platform.system() == "Darwin" or device != "cpu" or threads != 1:
        raise Round0Error("production requires non-Darwin CPU inference and exactly one thread")


def _metrics_from_trace(exported: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Replay a trace and calculate terminal-hero metrics from physical state."""
    if exported.get("schema") != WORLD_SCHEMA:
        raise Round0Error("unexpected trace schema")
    if exported.get("complete") is not True:
        raise Round0Error("trace is incomplete")
    if exported.get("checkpoint_sha256") != sha256(checkpoint):
        raise Round0Error("trace checkpoint identity differs")
    world = build_world(exported["row"], checkpoint)
    expected = {int(frame): digest for frame, digest in exported["hashes"].items()}
    if state_hash(world.sim) != expected.get(0):
        raise Round0Error("trace initial state hash does not replay")
    horizon = int(exported["frames"])
    trace_actions = exported["actions"]
    final = len(trace_actions)
    required_hashes = {0, final, *range(250, final + 1, 250)}
    if set(expected) != required_hashes:
        raise Round0Error("trace lacks required initial, interval, or final state hashes")
    masses: list[int] = []
    deaths = boosts = 0
    counts = [0] * 6
    was_alive = bool(world.sim.alive[0, 0])
    for decision, row in enumerate(trace_actions):
        if len(row) != 6:
            raise Round0Error("trace action row width is not six")
        action = int(row[0])
        if not 0 <= action < 6:
            raise Round0Error("illegal hero action in trace")
        prepare_world(world)
        actions = np.asarray(row, dtype=np.int64).reshape(1, 6)
        world.sim.step(actions)
        world.runtime.observe_step(world.sim)
        alive = bool(world.sim.alive[0, 0])
        masses.append(int(world.sim.length[0, 0]) if alive else 0)
        counts[action] += 1
        boosts += int(action >= 3)
        deaths += int(was_alive and not alive)
        was_alive = alive
        frame = decision + 1
        if frame in expected and state_hash(world.sim) != expected[frame]:
            raise Round0Error(f"trace state hash mismatch at frame {frame}")
    if len(masses) > horizon:
        raise Round0Error("trace has more actions than its registered horizon")
    if len(masses) < horizon and was_alive:
        raise Round0Error("short trace ended while terminal hero was alive")
    # Terminal heroes stop producing decisions.  Dead frames still contribute
    # zero to the full-horizon endpoint and must not shorten the denominator.
    masses.extend([0] * (horizon - len(masses)))
    decisions = sum(counts)
    if not decisions:
        raise Round0Error("trace has no hero decisions")
    return {
        "mass_integral": sum(masses) / horizon,
        "survival_fraction": sum(value > 0 for value in masses) / horizon,
        "no_death": int(deaths == 0),
        "deaths": deaths,
        "decisions": decisions,
        "boost_decisions": boosts,
        "action_counts": counts,
        "boost_share": boosts / decisions,
        "one_action_share": max(counts) / decisions,
        "unconditional_faults": [],
    }


def _bare_trace(
    row: Mapping[str, Any],
    checkpoint: Path,
    *,
    frames: int,
    device: str,
    on_action: Any = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Independently execute greedy bare B8 without teacher search or labels."""
    teacher = ExitTeacher(
        ExitTeacherConfig(horizon=300, beta=1.0), device=device, policy=B8Policy(device)
    )
    world = build_world(row, checkpoint)
    actions: list[list[int]] = []
    hashes = {"0": state_hash(world.sim)}
    for decision in range(frames):
        if on_progress:
            on_progress()
        if not bool(world.sim.alive[0, 0]):
            break
        prepare_world(world)
        evaluation = teacher.evaluate(world)
        if not np.isfinite(np.asarray(evaluation.q, dtype=np.float64)).all():
            raise NonFiniteQLossFault("bare policy produced non-finite Q")
        action = int(evaluation.greedy_action())
        if not 0 <= action < 6:
            raise IllegalActionFault("bare policy returned illegal action")
        actions.append(_step(world, teacher, action))
        if on_action:
            on_action(action)
        count = decision + 1
        if count % 250 == 0:
            hashes[str(count)] = state_hash(world.sim)
    hashes[str(len(actions))] = state_hash(world.sim)
    return {
        "schema": WORLD_SCHEMA,
        "world_id": row["world_id"],
        "seed": int(row["world_seed"]),
        "mix": row["mix"],
        "row": dict(row),
        "checkpoint_sha256": sha256(checkpoint),
        "rollout_policy_sha256": sha256(checkpoint),
        "actions": actions,
        "hashes": hashes,
        "labels": [],
        "rewinds": 0,
        "complete": True,
        "frames": frames,
        "horizon": 300,
        "beta": 0.0,
        "profile": {"decisions": len(actions)},
    }


def _strip_labels(exported: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(exported)
    result["labels"] = []
    result.pop("teacher_v2", None)
    result["round0_role"] = "trace_only"
    return result


def _trace_meta(
    root: Path, path: Path, metrics: Mapping[str, Any], exported: Mapping[str, Any]
) -> dict[str, Any]:
    return _trace_meta_from_final_hash(
        root,
        path,
        metrics,
        str(exported["hashes"][str(len(exported["actions"]))]),
    )


def _trace_meta_from_final_hash(
    root: Path, path: Path, metrics: Mapping[str, Any], final_state_hash: str
) -> dict[str, Any]:
    """Build receipt metadata without retaining a full trace in the parent process."""
    return {
        **dict(metrics),
        "trace": {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "final_state_hash": final_state_hash,
        },
    }


def _validated_metrics(value: Any) -> dict[str, Any]:
    """Reject malformed worker metrics before they can enter scientific receipts."""
    if not isinstance(value, Mapping):
        raise Round0Error("worker metrics must be an object")
    result = dict(value)
    counts = result.get("action_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != 6
        or any(type(count) is not int or count < 0 for count in counts)
        or type(result.get("decisions")) is not int
        or result["decisions"] < 1
        or sum(counts) != result["decisions"]
    ):
        raise Round0Error("worker metrics have invalid action counts")
    for key in ("deaths", "boost_decisions"):
        if type(result.get(key)) is not int or result[key] < 0:
            raise Round0Error(f"worker metrics have invalid {key}")
    if result["boost_decisions"] > result["decisions"]:
        raise Round0Error("worker metrics have invalid boost_decisions")
    if type(result.get("no_death")) is not int or result["no_death"] not in (0, 1):
        raise Round0Error("worker metrics have invalid no_death")
    for key in ("mass_integral", "survival_fraction", "boost_share", "one_action_share"):
        if type(result.get(key)) not in (int, float) or not math.isfinite(float(result[key])):
            raise Round0Error(f"worker metrics have invalid {key}")
    if (
        any(
            not 0.0 <= float(result[key]) <= 1.0
            for key in ("survival_fraction", "boost_share", "one_action_share")
        )
        or float(result["mass_integral"]) < 0.0
    ):
        raise Round0Error("worker metrics are out of range")
    return result


def _run_identity(
    *,
    runtime_id: str,
    bank: Path,
    audit: Path,
    ratification: Path | None,
    checkpoint: Path,
    preflight: bool,
) -> dict[str, Any]:
    result = {
        "runtime_id": runtime_id,
        "bank_sha256": sha256(bank),
        "audit_sha256": sha256(audit),
        "ratification_sha256": sha256(ratification) if ratification else None,
        "checkpoint_sha256": sha256(checkpoint),
        "commit": _git_commit(),
    }
    return result


def _tracked_source_is_clean() -> bool:
    """Return whether tracked source has no uncommitted changes.

    Untracked runtime output is intentionally allowed: the qualified supervisor
    materializes inputs and exports beside the frozen checkout.
    """
    return not subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_is_detached() -> bool:
    """Return detached-HEAD status, refusing ambiguous Git failures."""
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1:
        return True
    if result.returncode == 0:
        return False
    raise Round0Error("could not verify detached frozen source checkout")


def _source_fence(*, preflight: bool, ratification: Path | None) -> dict[str, Any]:
    """Bind the running producer to bytes checked before, during, and after work."""
    from research.t34_round0.common import source_hashes

    snapshot = {"commit": _git_commit(), "source_hashes": source_hashes(ROOT)}
    if preflight:
        return snapshot
    if ratification is None:
        raise Round0Error("production requires a ratification source fence")
    registered = _load(ratification)
    if registered.get("commit") != snapshot["commit"]:
        raise Round0Error("checked-out source differs from the ratified commit")
    if registered.get("source_hashes") != snapshot["source_hashes"]:
        raise Round0Error("checked-out source differs from the ratified source closure")
    if not _source_is_detached():
        raise Round0Error("production requires a detached frozen source checkout")
    if not _tracked_source_is_clean():
        raise Round0Error("production requires a clean tracked source commit")
    return snapshot


def _check_source_fence(snapshot: Mapping[str, Any], *, preflight: bool) -> None:
    """Fail closed if source bytes or production tracked state drift mid-chunk."""
    from research.t34_round0.common import source_hashes

    if _git_commit() != snapshot.get("commit") or source_hashes(ROOT) != snapshot.get(
        "source_hashes"
    ):
        raise Round0Error("source bytes changed during Round 0 collection")
    if not preflight and not _tracked_source_is_clean():
        raise Round0Error("tracked source changed during Round 0 collection")
    if not preflight and not _source_is_detached():
        raise Round0Error("source checkout is no longer detached during Round 0 collection")


def _worker_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise CapExpiry("operational chunk wall bound reached")


def _write_export(path: Path, exported: Mapping[str, Any]) -> None:
    """Write a trace while marking any serialization failure as operational."""
    try:
        write_world(path, exported)
    except (OSError, TypeError, ValueError) as exc:
        raise ExportSerializationError("Round 0 trace export failed") from exc


def _load_export(path: Path) -> dict[str, Any]:
    """Load a trace while marking incomplete or malformed export bytes as operational."""
    try:
        return load_world(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExportSerializationError("Round 0 trace export could not be loaded") from exc


def _write_json_export(path: Path, value: Mapping[str, Any]) -> None:
    """Write a non-terminal Round 0 JSON export as an operational operation."""
    try:
        _write_new(path, value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExportSerializationError("Round 0 JSON export failed") from exc


def _submit_available(
    pool: Any,
    pending: dict[Any, int],
    *,
    next_index: int,
    stop_index: int,
    workers: int,
    request_for_index: Any,
) -> int:
    """Keep every worker occupied; canonical receipt ordering is independent."""
    while next_index < stop_index and len(pending) < workers:
        pending[pool.submit(_produce_pair, request_for_index(next_index))] = next_index
        next_index += 1
    return next_index


def _produce_pair(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run both canonical arms in one single-threaded spawned worker process."""
    from research.t34_exit_b8.label import _cpu_ceiling

    _cpu_ceiling(1)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import numba
    import torch

    if numba.get_num_threads() != 1:
        raise Round0Error("worker Numba thread ceiling did not take effect")
    torch.set_num_threads(1)
    row = dict(request["row"])
    checkpoint = Path(str(request["checkpoint"]))
    trace_dir = Path(str(request["trace_dir"]))
    deadline = float(request["deadline"])
    world_id = str(row["world_id"])
    trace_dir.mkdir(parents=True, exist_ok=False)
    try:
        bare_started = time.monotonic()
        bare = _bare_trace(
            row,
            checkpoint,
            frames=int(row["horizon"]),
            device="cpu",
            on_progress=lambda: _worker_deadline(deadline),
        )
        bare_elapsed = time.monotonic() - bare_started
        bare_path = trace_dir / "bare.json"
        _write_export(bare_path, bare)
        metric_started = time.monotonic()
        _worker_deadline(deadline)
        bare_metrics = _metrics_from_trace(bare, checkpoint)
        _worker_deadline(deadline)
        bare_metric_elapsed = time.monotonic() - metric_started
    except BaseException as exc:
        raise Round0PairError("bare", world_id, exc) from exc
    try:
        teacher_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="t34-round0-teacher-") as temporary:
            teacher_out = Path(temporary) / "teacher"
            result = run_worlds(
                {
                    "checkpoint": checkpoint,
                    "output_dir": teacher_out,
                    "rows": [row],
                    "frames": int(row["horizon"]),
                    "seconds": max(1.0, deadline - time.monotonic()),
                    "beta": 1.0,
                    "device": "cpu",
                    "batch_device_vectors": False,
                    "worlds_per_process": 1,
                    "branch_bucket_size": 6,
                }
            )
            _worker_deadline(deadline)
            if not result.get("complete") or result.get("worlds") != 1:
                raise Round0Error("teacher did not complete a world")
            teacher = _strip_labels(_load_export(teacher_out / "world-000000.json"))
        teacher_elapsed = time.monotonic() - teacher_started
        teacher_path = trace_dir / "teacher.json"
        _write_export(teacher_path, teacher)
        metric_started = time.monotonic()
        _worker_deadline(deadline)
        teacher_metrics = _metrics_from_trace(teacher, checkpoint)
        _worker_deadline(deadline)
        teacher_metric_elapsed = time.monotonic() - metric_started
    except BaseException as exc:
        raise Round0PairError("teacher", world_id, exc) from exc
    return {
        "index": int(request["index"]),
        "row": row,
        "arms": {
            "bare": {
                "path": str(bare_path),
                "metrics": bare_metrics,
                "final_state_hash": str(bare["hashes"][str(len(bare["actions"]))]),
                "action_stream": [int(actions[0]) for actions in bare["actions"]],
                "rollout_elapsed_seconds": bare_elapsed,
                "metric_elapsed_seconds": bare_metric_elapsed,
            },
            "teacher": {
                "path": str(teacher_path),
                "metrics": teacher_metrics,
                "final_state_hash": str(teacher["hashes"][str(len(teacher["actions"]))]),
                "action_stream": [int(actions[0]) for actions in teacher["actions"]],
                "rollout_elapsed_seconds": teacher_elapsed,
                "metric_elapsed_seconds": teacher_metric_elapsed,
            },
        },
        "worker": {"threads": 1, "device": "cpu", "worlds_per_process": 1},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank_path, audit_path = Path(args.bank), Path(args.audit)
    ratification = Path(args.ratification) if args.ratification else None
    dispatch_value = getattr(args, "dispatch_admission", None)
    dispatch_admission = Path(dispatch_value) if dispatch_value else None
    checkpoint, out = Path(args.checkpoint), Path(args.output_dir)
    preflight = bool(args.preflight)
    _validate_execution(
        preflight=preflight, device=args.device, threads=args.threads, seconds=args.seconds
    )
    from research.t34_round0.common import B8_SHA

    bank_data = _load(bank_path)
    rows = _validate_bank(bank_data, preflight=preflight)
    _validate_audit(audit_path, bank_path)
    admission: dict[str, Any] | None = None
    if preflight:
        if ratification is not None or dispatch_admission is not None:
            raise Round0Error(
                "preflight must not carry a campaign ratification or dispatch admission"
            )
    else:
        if ratification is None:
            raise Round0Error("production requires a ratification")
        rows = _registered_bank(ratification, bank_path, audit_path)
        from research.t34_round0.bank import ordered_rows

        rows = ordered_rows(rows)
    if not checkpoint.is_file():
        raise Round0Error("B8 checkpoint is missing")
    if sha256(checkpoint) != B8_SHA:
        raise Round0Error("checkpoint differs from released B8 s3")
    if args.start < 0 or args.count < 1 or args.start + args.count > len(rows):
        raise Round0Error("requested contiguous bank slice is outside the immutable bank")
    selected = rows[args.start : args.start + args.count]
    if not preflight:
        assert ratification is not None
        admission = _validate_dispatch_admission(
            dispatch_admission, ratification, [str(row["world_id"]) for row in selected]
        )
    if out.exists():
        raise Round0Error("output directory must be fresh")
    fence = _source_fence(preflight=preflight, ratification=ratification)
    out.mkdir(parents=True)
    identity = _run_identity(
        runtime_id=args.runtime_id,
        bank=bank_path,
        audit=audit_path,
        ratification=ratification,
        checkpoint=checkpoint,
        preflight=preflight,
    )
    identity["source_hashes"] = fence["source_hashes"]
    if not preflight:
        identity["dispatch_admission_sha256"] = sha256(dispatch_admission)
        assert admission is not None
        identity["phase_path"] = admission["phase_path"]
        identity["claim_commit"] = admission["claim_commit"]
    started = time.monotonic()
    heartbeat = started
    pairs: list[dict[str, str]] = []
    windows = {"bare": _ActionWindow(), "teacher": _ActionWindow()}
    arm_summary = {
        name: {
            "rollout_elapsed_seconds": 0.0,
            "metric_elapsed_seconds": 0.0,
            "total_elapsed_seconds": 0.0,
            "decisions": 0,
            "action_counts": [0] * 6,
        }
        for name in ("bare", "teacher")
    }
    selected_by_index = {args.start + offset: row for offset, row in enumerate(selected)}
    completed: set[int] = set()
    ready: dict[int, dict[str, Any]] = {}
    deadline = started + args.seconds
    workers = 1 if preflight else min(PRODUCTION_WORKERS, len(selected))

    def fault(arm: str, marker: str, world_id: str, reason: str) -> None:
        _write_new(
            out / "fault.json",
            {
                "schema": "t34-round0-fault/v1",
                "preflight": preflight,
                "run_identity": identity,
                "unconditional": True,
                "arm": arm,
                "world_id": world_id,
                "fault": marker,
                "operational_reason": reason,
            },
        )

    def kill(reason: str, *, error: BaseException | None = None) -> None:
        """Leave create-only operational evidence for every abort or interruption."""
        dropped = len(selected) - len(completed)
        payload: dict[str, Any] = {
            "schema": "t34-round0-kill/v1",
            "status": "KILL",
            "preflight": preflight,
            "run_identity": identity,
            "operational_reason": reason,
            "completed_pairs": len(completed),
            "dropped_pairs": dropped,
            "requested_pairs": len(selected),
            "dropped_fraction": dropped / len(selected),
            "drop_limit_fraction": 0.02,
            "one_look": "no outcome analysis was performed by the producer",
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)
        try:
            _write_new(out / "kill.json", payload)
        except FileExistsError:
            pass

    def incomplete(reason: str, *, error: BaseException) -> None:
        """Seal a recoverable export/serialization interruption without a STOP."""
        payload = {
            "schema": "t34-round0-incomplete/v1",
            "status": "INCOMPLETE",
            "preflight": preflight,
            "run_identity": identity,
            "operational_reason": reason,
            "mechanical_rerun_eligible": True,
            "completed_pairs": len(completed),
            "requested_pairs": len(selected),
            "error_type": type(error).__name__,
            "error": str(error),
            "one_look": "no outcome analysis was performed by the producer",
        }
        try:
            _write_new(out / "incomplete.json", payload)
        except FileExistsError:
            pass

    def cap_expired(*, error: BaseException) -> None:
        """Seal the first registered wall-cap expiry as non-extendable INCONCLUSIVE."""
        payload = {
            "schema": "t34-round0-cap-expiry/v1",
            "status": "INCONCLUSIVE",
            "preflight": preflight,
            "run_identity": identity,
            "operational_reason": "registered chunk wall cap expired",
            "no_extension": True,
            "completed_pairs": len(completed),
            "requested_pairs": len(selected),
            "error_type": type(error).__name__,
            "error": str(error),
            "one_look": "no outcome analysis was performed by the producer",
        }
        try:
            _write_new(out / "cap_expiry.json", payload)
        except FileExistsError:
            pass

    def operational() -> None:
        nonlocal heartbeat
        now = time.monotonic()
        if now >= deadline:
            raise CapExpiry("operational chunk wall bound reached")
        _check_source_fence(fence, preflight=preflight)
        if now - heartbeat >= 110:
            print(
                json.dumps(
                    {
                        "event": "t34_round0_operational_progress",
                        "elapsed_seconds": now - started,
                        "completed_pairs": len(pairs),
                        "requested_pairs": len(selected),
                        "workers": workers,
                    }
                ),
                flush=True,
            )
            heartbeat = now

    def publish(result: Mapping[str, Any]) -> None:
        index = int(result["index"])
        row = selected_by_index[index]
        arms: dict[str, Any] = {}
        worker_arms = result.get("arms")
        if not isinstance(worker_arms, Mapping):
            raise Round0Error("worker result lacks arm inventory")
        for name in ("bare", "teacher"):
            worker_arm = worker_arms.get(name)
            if not isinstance(worker_arm, Mapping):
                raise Round0Error(f"worker result lacks {name} arm")
            metrics = _validated_metrics(worker_arm.get("metrics"))
            actions = worker_arm.get("action_stream")
            if (
                not isinstance(actions, list)
                or len(actions) != metrics["decisions"]
                or any(type(action) is not int or not 0 <= action < 6 for action in actions)
            ):
                raise Round0Error(f"worker {name} action stream is invalid")
            for action in actions:
                if windows[name].add(action):
                    fault_exception = OneActionShareFault(
                        f"{name} exceeded the one-action-collapse tripwire"
                    )
                    fault(
                        name,
                        _fault_marker(fault_exception),
                        str(row["world_id"]),
                        "one action exceeded 90 percent",
                    )
                    raise fault_exception
            path, final_state_hash = Path(str(worker_arm.get("path", ""))), worker_arm.get(
                "final_state_hash"
            )
            if not path.is_file() or not isinstance(final_state_hash, str) or not final_state_hash:
                raise Round0Error(f"worker {name} trace metadata is invalid")
            arms[name] = _trace_meta_from_final_hash(out, path, metrics, final_state_hash)
            for key in ("rollout_elapsed_seconds", "metric_elapsed_seconds"):
                elapsed = worker_arm.get(key)
                if (
                    type(elapsed) not in (int, float)
                    or not math.isfinite(float(elapsed))
                    or elapsed < 0
                ):
                    raise Round0Error(f"worker {name} {key} is invalid")
                arms[name][key] = float(elapsed)
                arm_summary[name][key] += float(elapsed)
            arms[name]["total_elapsed_seconds"] = (
                arms[name]["rollout_elapsed_seconds"] + arms[name]["metric_elapsed_seconds"]
            )
            arm_summary[name]["total_elapsed_seconds"] += arms[name]["total_elapsed_seconds"]
            arm_summary[name]["decisions"] += int(metrics["decisions"])
            for action, count in enumerate(metrics["action_counts"]):
                arm_summary[name]["action_counts"][action] += int(count)
        tag = _safe_world_id(str(row["world_id"]))
        pair_path = out / "pairs" / f"{index:03d}-{tag}.json"
        _write_json_export(
            pair_path,
            {
                "schema": PAIR_SCHEMA,
                "status": "complete",
                "preflight": preflight,
                "run_identity": identity,
                "world": dict(row),
                "worker": dict(result["worker"]),
                "arms": arms,
            },
        )
        completed.add(index)
        pairs.append({"path": str(pair_path.relative_to(out)), "sha256": sha256(pair_path)})

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    pending: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    next_index = args.start
    next_publish = args.start
    try:
        while next_index < args.start + len(selected) or pending:
            operational()

            def request_for_index(index: int) -> dict[str, Any]:
                row = selected_by_index[index]
                tag = _safe_world_id(str(row["world_id"]))
                return {
                    "index": index,
                    "row": row,
                    "checkpoint": str(checkpoint),
                    "trace_dir": str(out / "traces" / tag),
                    "deadline": deadline,
                }

            next_index = _submit_available(
                pool,
                pending,
                next_index=next_index,
                stop_index=args.start + len(selected),
                workers=workers,
                request_for_index=request_for_index,
            )
            done, _ = concurrent.futures.wait(
                pending,
                timeout=min(5.0, max(0.0, deadline - time.monotonic())),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in sorted(done, key=lambda item: pending[item]):
                pending.pop(future)
                result = future.result()
                _check_source_fence(fence, preflight=preflight)
                ready[int(result["index"])] = result
            while next_publish in ready:
                publish(ready.pop(next_publish))
                next_publish += 1
        _check_source_fence(fence, preflight=preflight)
    except BaseException as exc:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        marker = exc.marker if isinstance(exc, Round0PairError) else _fault_marker(exc)
        operational_kind = (
            exc.operational_kind if isinstance(exc, Round0PairError) else _operational_kind(exc)
        )
        if isinstance(exc, Round0PairError) and marker:
            fault(exc.arm, marker, exc.world_id, str(exc))
        if operational_kind == "export_or_serialization":
            incomplete("export or serialization failure", error=exc)
        elif operational_kind == "cap_expired":
            cap_expired(error=exc)
        else:
            kill("operator abort or unexplained interruption", error=exc)
        if isinstance(exc, Round0Error):
            raise
        if operational_kind == "export_or_serialization":
            raise Round0Error(
                "Round 0 export/serialization failure; INCOMPLETE evidence was preserved"
            ) from exc
        if operational_kind == "cap_expired":
            raise Round0Error("Round 0 cap expired; INCONCLUSIVE evidence was preserved") from exc
        raise Round0Error("Round 0 worker aborted; KILL evidence was preserved") from exc
    else:
        pool.shutdown(wait=True)
    receipt = {
        "schema": CHUNK_SCHEMA,
        "status": "complete",
        "preflight": preflight,
        "run_identity": identity,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "slice": {
            "start": args.start,
            "count": args.count,
            "world_ids": [row["world_id"] for row in selected],
        },
        "pairs": pairs,
        "arms": arm_summary,
        "elapsed_seconds": time.monotonic() - started,
        "operational": {
            "seconds_cap": args.seconds,
            "threads": args.threads,
            "device": "cpu",
            "workers": workers,
            "worlds_per_process": 1,
            "inference": "cpu",
            "dropped_pairs": 0,
            "drop_limit_fraction": 0.02,
        },
    }
    try:
        _write_json_export(out / "chunk.json", receipt)
    except ExportSerializationError as exc:
        incomplete("export or serialization failure", error=exc)
        raise Round0Error(
            "Round 0 export/serialization failure; INCOMPLETE evidence was preserved"
        ) from exc
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded paired T34 Round 0 worlds")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--ratification")
    parser.add_argument("--dispatch-admission")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=21000)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = run(args)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "preflight": receipt["preflight"],
                    "pairs": len(receipt["pairs"]),
                    "elapsed_seconds": receipt["elapsed_seconds"],
                },
                sort_keys=True,
            )
        )
    except Round0Error as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
