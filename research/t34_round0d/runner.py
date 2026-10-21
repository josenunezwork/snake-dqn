"""Create-only paired Round 0-D trace producer (bare B8 s3 vs the exact T3-dense rule).

Both arms of each pair run sequentially in one single-threaded spawned worker on
one runtime, exactly like Round 0. The bare arm, trace metrics, typed faults and
export helpers are imported unchanged from ``research.t34_round0.runner``.
Differences from Round 0 (frozen in PREREGISTRATION_ROUND0D.md):

* the teacher arm is :class:`research.t34_round0d.controller.DenseT3Controller`;
* the public teacher trace has its labels stripped (scored exactly like Round 0),
  while a second, sealed teacher export keeps the per-action labels in the
  ``research.t34_exit_b8.exports`` schema under ``sealed/``. No outcome consumer
  opens ``sealed/``; ``labels.py`` admits it to R1 fitting only after a GO.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
from research.t34_exit_b8.exports import state_hash
from research.t34_exit_b8.faults import CapExpiry, ExportSerializationError, OneActionShareFault
from research.t34_exit_b8.policy import B8Policy
from research.t34_exit_b8.teacher_v2 import _step
from research.t34_exit_b8.trajectory import build_world, prepare_world
from research.t34_round0.runner import (
    Round0Error,
    Round0PairError,
    _ActionWindow,
    _bare_trace,
    _canonical_sha,
    _fault_marker,
    _load,
    _load_export,
    _metrics_from_trace,
    _operational_kind,
    _safe_world_id,
    _source_is_detached,
    _tracked_source_is_clean,
    _validated_metrics,
    _worker_deadline,
    _write_export,
    _write_json_export,
    _write_new,
    sha256,
)
from research.t34_round0d.controller import MIN_LENGTH, RULE_ID, DenseT3Controller

ROOT = Path(__file__).resolve().parents[2]
PAIR_SCHEMA = "t34-round0d-pair/v1"
CHUNK_SCHEMA = "t34-round0d-chunk/v1"
WORLD_SCHEMA = "t34-exit-world/v1"
HORIZON = 5000
PREFLIGHT_HORIZON = 500
PRODUCTION_WORKERS = 12
OUTPUT_NAME = "round0d"


class Round0DError(Round0Error):
    """A fail-closed Round 0-D producer error (subclass keeps Round 0 handling)."""


def dense_trace(
    row: Mapping[str, Any],
    checkpoint: Path,
    *,
    frames: int,
    device: str,
    on_progress: Any = None,
    min_length: int = MIN_LENGTH,
) -> dict[str, Any]:
    """Execute the T3-dense controller for one terminal-hero episode, with labels."""
    teacher = ExitTeacher(
        ExitTeacherConfig(horizon=300, beta=1.0), device=device, policy=B8Policy(device)
    )
    if on_progress is not None:
        teacher.progress = on_progress
    world = build_world(row, checkpoint)
    controller = DenseT3Controller(teacher, horizon=frames, min_length=min_length)
    actions: list[list[int]] = []
    labels: list[dict[str, Any]] = []
    hashes = {"0": state_hash(world.sim)}
    for decision in range(frames):
        if on_progress:
            on_progress()
        if not bool(world.sim.alive[0, 0]):
            break
        prepare_world(world)
        action, label = controller.decide(world, decision)
        actions.append(_step(world, teacher, action))
        if label is not None:
            labels.append(label)
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
        "labels": labels,
        "rewinds": 0,
        "complete": True,
        "frames": frames,
        "horizon": 300,
        "beta": 1.0,
        "profile": {"decisions": len(actions)},
        "teacher_v2": {"controller": RULE_ID, "min_length": min_length, **controller.stats},
    }


def public_trace(exported: Mapping[str, Any]) -> dict[str, Any]:
    """Strip labels for scoring/replay; the sealed copy keeps them."""
    result = dict(exported)
    result["labels"] = []
    result.pop("teacher_v2", None)
    result["round0_role"] = "trace_only"
    return result


def _validate_bank(bank: Mapping[str, Any], *, preflight: bool) -> list[dict[str, Any]]:
    if bank.get("schema") != "t34-round0d-bank/v1":
        raise Round0DError("unknown Round 0-D bank schema")
    rows = bank.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Round0DError("bank requires nonempty rows")
    if preflight != bool(bank.get("excluded", False)):
        raise Round0DError("preflight requires an excluded fixture bank (and only then)")
    if not preflight and len(rows) != 240:
        raise Round0DError("production bank requires 240 rows")
    ids: set[str] = set()
    for row in rows:
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or not world_id or world_id in ids:
            raise Round0DError("bank world ids must be unique nonempty strings")
        if row.get("mix") not in ("scripted", "frozen", "mixed"):
            raise Round0DError("bank mix is invalid")
        if not preflight and int(row.get("horizon", -1)) != HORIZON:
            raise Round0DError("bank horizon differs from governed mode")
        if row.get("opponent_veto") != "v8" or not isinstance(row.get("opponents"), list):
            raise Round0DError("bank requires v8 opponents")
        int(row["world_seed"])
        ids.add(world_id)
    return [dict(row) for row in rows]


def _validate_execution(*, preflight: bool, device: str, threads: int, seconds: float) -> None:
    if not math.isfinite(seconds) or not 0 < seconds <= 21000:
        raise Round0DError("operational wall bound must be in (0, 21000]")
    if preflight:
        if platform.system() != "Darwin" or device != "cpu" or threads != 1:
            raise Round0DError("preflight requires Darwin, cpu, and exactly one thread")
        return
    if platform.system() == "Darwin" or device != "cpu" or threads != 1:
        raise Round0DError("production requires non-Darwin CPU inference and exactly one thread")


def _validate_dispatch_admission(
    admission: Path | None, ratification: Path, expected_world_ids: list[str]
) -> dict[str, Any]:
    """Require the remote create-only Round 0-D claim copied into this runtime."""
    if admission is None:
        raise Round0DError("production requires a dispatch admission receipt")
    value = _load(admission)
    phase_path = value.get("phase_path")
    if (
        value.get("schema") != "t34-round0d-admission/v1"
        or value.get("registration_sha256") != _canonical_sha(_load(ratification))
        or not isinstance(phase_path, str)
        or not re.search(r"(?:^|/)chunks/chunk-\d{3}/attempt-0[12]$", phase_path)
        or not isinstance(value.get("claim_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", value["claim_commit"])
        or value.get("world_ids") != expected_world_ids
    ):
        raise Round0DError("dispatch admission receipt is not a fixed registered chunk attempt")
    return value


def _source_fence(*, preflight: bool, ratification: Path | None) -> dict[str, Any]:
    from research.t34_round0d.common import head, source_hashes

    snapshot = {"commit": head(ROOT), "source_hashes": source_hashes(ROOT)}
    if preflight:
        return snapshot
    if ratification is None:
        raise Round0DError("production requires a ratification source fence")
    registered = _load(ratification)
    if registered.get("commit") != snapshot["commit"]:
        raise Round0DError("checked-out source differs from the ratified commit")
    if registered.get("source_hashes") != snapshot["source_hashes"]:
        raise Round0DError("checked-out source differs from the ratified source closure")
    if not _source_is_detached():
        raise Round0DError("production requires a detached frozen source checkout")
    if not _tracked_source_is_clean():
        raise Round0DError("production requires a clean tracked source commit")
    return snapshot


def _check_source_fence(snapshot: Mapping[str, Any], *, preflight: bool) -> None:
    from research.t34_round0d.common import head, source_hashes

    if head(ROOT) != snapshot.get("commit") or source_hashes(ROOT) != snapshot.get("source_hashes"):
        raise Round0DError("source bytes changed during Round 0-D collection")
    if not preflight and (not _tracked_source_is_clean() or not _source_is_detached()):
        raise Round0DError("tracked source or detached state changed during collection")


def _produce_pair(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run both canonical arms in one single-threaded spawned worker process."""
    from research.t34_exit_b8.label import _cpu_ceiling

    _cpu_ceiling(1)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import numba
    import torch

    if numba.get_num_threads() != 1:
        raise Round0DError("worker Numba thread ceiling did not take effect")
    torch.set_num_threads(1)
    row = dict(request["row"])
    checkpoint = Path(str(request["checkpoint"]))
    trace_dir = Path(str(request["trace_dir"]))
    sealed_dir = Path(str(request["sealed_dir"]))
    deadline = float(request["deadline"])
    world_id = str(row["world_id"])
    trace_dir.mkdir(parents=True, exist_ok=False)
    sealed_dir.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        bare = _bare_trace(
            row,
            checkpoint,
            frames=int(row["horizon"]),
            device="cpu",
            on_progress=lambda: _worker_deadline(deadline),
        )
        bare_elapsed = time.monotonic() - started
        bare_path = trace_dir / "bare.json"
        _write_export(bare_path, bare)
        started = time.monotonic()
        _worker_deadline(deadline)
        bare_metrics = _metrics_from_trace(_load_export(bare_path), checkpoint)
        bare_metric_elapsed = time.monotonic() - started
    except BaseException as exc:
        raise Round0PairError("bare", world_id, exc) from exc
    try:
        started = time.monotonic()
        full = dense_trace(
            row,
            checkpoint,
            frames=int(row["horizon"]),
            device="cpu",
            on_progress=lambda: _worker_deadline(deadline),
        )
        teacher_elapsed = time.monotonic() - started
        sealed_path = sealed_dir / "teacher_labels.json"
        _write_export(sealed_path, full)
        teacher_path = trace_dir / "teacher.json"
        _write_export(teacher_path, public_trace(full))
        started = time.monotonic()
        _worker_deadline(deadline)
        teacher_metrics = _metrics_from_trace(_load_export(teacher_path), checkpoint)
        teacher_metric_elapsed = time.monotonic() - started
    except BaseException as exc:
        raise Round0PairError("teacher", world_id, exc) from exc
    dense = {
        key: int(full["teacher_v2"][key])
        for key in ("batches", "action_changes", "rescue_changes", "delay_changes")
    }
    return {
        "index": int(request["index"]),
        "row": row,
        "arms": {
            "bare": {
                "path": str(bare_path),
                "metrics": bare_metrics,
                "final_state_hash": str(bare["hashes"][str(len(bare["actions"]))]),
                "action_stream": [int(a[0]) for a in bare["actions"]],
                "rollout_elapsed_seconds": bare_elapsed,
                "metric_elapsed_seconds": bare_metric_elapsed,
            },
            "teacher": {
                "path": str(teacher_path),
                "metrics": teacher_metrics,
                "final_state_hash": str(full["hashes"][str(len(full["actions"]))]),
                "action_stream": [int(a[0]) for a in full["actions"]],
                "rollout_elapsed_seconds": teacher_elapsed,
                "metric_elapsed_seconds": teacher_metric_elapsed,
                "dense": dense,
            },
        },
        "sealed": {"path": str(sealed_path), "sha256": sha256(sealed_path)},
        "worker": {"threads": 1, "device": "cpu", "worlds_per_process": 1},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901 - mirrors Round 0's run()
    bank_path, audit_path = Path(args.bank), Path(args.audit)
    ratification = Path(args.ratification) if args.ratification else None
    dispatch_value = getattr(args, "dispatch_admission", None)
    dispatch_admission = Path(dispatch_value) if dispatch_value else None
    checkpoint, out = Path(args.checkpoint), Path(args.output_dir)
    preflight = bool(args.preflight)
    _validate_execution(
        preflight=preflight, device=args.device, threads=args.threads, seconds=args.seconds
    )
    from research.t34_round0d.common import B8_SHA

    rows = _validate_bank(_load(bank_path), preflight=preflight)
    audit = _load(audit_path)
    if audit.get("status") != "CLEAR" or audit.get("schema") != "t34-round0d-seed-audit/v1":
        raise Round0DError("Round 0-D seed audit is not CLEAR")
    admission: dict[str, Any] | None = None
    if preflight:
        if ratification is not None or dispatch_admission is not None:
            raise Round0DError("preflight must not carry a ratification or dispatch admission")
        if any(int(row["horizon"]) != PREFLIGHT_HORIZON for row in rows):
            raise Round0DError("preflight fixtures must be H500")
    else:
        if ratification is None:
            raise Round0DError("production requires a ratification")
        from research.t34_round0d.bank import ordered_rows
        from research.t34_round0d.registration import require_registration

        bank = require_registration(_load(ratification), bank_path, audit_path, repo=ROOT)
        rows = ordered_rows(_validate_bank(bank, preflight=False))
    if not checkpoint.is_file() or sha256(checkpoint) != B8_SHA:
        raise Round0DError("checkpoint differs from released B8 s3")
    if args.start < 0 or args.count < 1 or args.start + args.count > len(rows):
        raise Round0DError("requested contiguous bank slice is outside the immutable bank")
    selected = rows[args.start : args.start + args.count]
    if not preflight:
        assert ratification is not None
        admission = _validate_dispatch_admission(
            dispatch_admission, ratification, [str(row["world_id"]) for row in selected]
        )
    if out.exists():
        raise Round0DError("output directory must be fresh")
    fence = _source_fence(preflight=preflight, ratification=ratification)
    out.mkdir(parents=True)
    identity: dict[str, Any] = {
        "runtime_id": args.runtime_id,
        "bank_sha256": sha256(bank_path),
        "audit_sha256": sha256(audit_path),
        "ratification_sha256": sha256(ratification) if ratification else None,
        "checkpoint_sha256": sha256(checkpoint),
        "commit": fence["commit"],
        "source_hashes": fence["source_hashes"],
        "controller": RULE_ID,
    }
    if not preflight:
        assert admission is not None and dispatch_admission is not None
        identity["dispatch_admission_sha256"] = sha256(dispatch_admission)
        identity["phase_path"] = admission["phase_path"]
        identity["claim_commit"] = admission["claim_commit"]
    started = time.monotonic()
    heartbeat = started
    pairs: list[dict[str, str]] = []
    sealed: list[dict[str, str]] = []
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

    def seal(name: str, schema: str, payload: dict[str, Any]) -> None:
        try:
            _write_new(
                out / name,
                {
                    "schema": schema,
                    "preflight": preflight,
                    "run_identity": identity,
                    "completed_pairs": len(completed),
                    "requested_pairs": len(selected),
                    "one_look": "no outcome analysis was performed by the producer",
                    **payload,
                },
            )
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
                        "event": "t34_round0d_operational_progress",
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
        for name in ("bare", "teacher"):
            worker_arm = result["arms"][name]
            metrics = _validated_metrics(worker_arm.get("metrics"))
            actions = worker_arm.get("action_stream")
            if (
                not isinstance(actions, list)
                or len(actions) != metrics["decisions"]
                or any(type(a) is not int or not 0 <= a < 6 for a in actions)
            ):
                raise Round0DError(f"worker {name} action stream is invalid")
            for action in actions:
                if windows[name].add(action):
                    fault_exception = OneActionShareFault(
                        f"{name} exceeded the one-action-collapse tripwire"
                    )
                    seal(
                        "fault.json",
                        "t34-round0d-fault/v1",
                        {
                            "unconditional": True,
                            "arm": name,
                            "world_id": str(row["world_id"]),
                            "fault": _fault_marker(fault_exception),
                            "operational_reason": "one action exceeded 90 percent",
                        },
                    )
                    raise fault_exception
            path = Path(str(worker_arm["path"]))
            if not path.is_file():
                raise Round0DError(f"worker {name} trace is missing")
            arms[name] = {
                **metrics,
                "trace": {
                    "path": str(path.relative_to(out)),
                    "sha256": sha256(path),
                    "final_state_hash": str(worker_arm["final_state_hash"]),
                },
            }
            if name == "teacher":
                arms[name]["dense"] = dict(worker_arm["dense"])
            for key in ("rollout_elapsed_seconds", "metric_elapsed_seconds"):
                elapsed = float(worker_arm[key])
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise Round0DError(f"worker {name} {key} is invalid")
                arms[name][key] = elapsed
                arm_summary[name][key] += elapsed
            total = arms[name]["rollout_elapsed_seconds"] + arms[name]["metric_elapsed_seconds"]
            arms[name]["total_elapsed_seconds"] = total
            arm_summary[name]["total_elapsed_seconds"] += total
            arm_summary[name]["decisions"] += int(metrics["decisions"])
            for action, count in enumerate(metrics["action_counts"]):
                arm_summary[name]["action_counts"][action] += int(count)
        sealed_path = Path(str(result["sealed"]["path"]))
        if sha256(sealed_path) != result["sealed"]["sha256"]:
            raise Round0DError("sealed teacher label export changed after the worker wrote it")
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
        sealed.append(
            {
                "world_id": str(row["world_id"]),
                "path": str(sealed_path.relative_to(out)),
                "sha256": result["sealed"]["sha256"],
            }
        )

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    pending: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    next_index = args.start
    next_publish = args.start
    stop_index = args.start + len(selected)
    try:
        while next_index < stop_index or pending:
            operational()
            while next_index < stop_index and len(pending) < workers:
                row = selected_by_index[next_index]
                tag = _safe_world_id(str(row["world_id"]))
                request = {
                    "index": next_index,
                    "row": row,
                    "checkpoint": str(checkpoint),
                    "trace_dir": str(out / "traces" / tag),
                    "sealed_dir": str(out / "sealed" / tag),
                    "deadline": deadline,
                }
                pending[pool.submit(_produce_pair, request)] = next_index
                next_index += 1
            done, _ = concurrent.futures.wait(
                pending,
                timeout=min(5.0, max(0.0, deadline - time.monotonic())),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
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
        kind = exc.operational_kind if isinstance(exc, Round0PairError) else _operational_kind(exc)
        if isinstance(exc, Round0PairError) and marker:
            seal(
                "fault.json",
                "t34-round0d-fault/v1",
                {
                    "unconditional": True,
                    "arm": exc.arm,
                    "world_id": exc.world_id,
                    "fault": marker,
                    "operational_reason": str(exc),
                },
            )
        if kind == "export_or_serialization":
            seal(
                "incomplete.json",
                "t34-round0d-incomplete/v1",
                {
                    "status": "INCOMPLETE",
                    "operational_reason": "export or serialization failure",
                    "mechanical_rerun_eligible": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        elif kind == "cap_expired":
            seal(
                "cap_expiry.json",
                "t34-round0d-cap-expiry/v1",
                {
                    "status": "INCONCLUSIVE",
                    "operational_reason": "registered chunk wall cap expired",
                    "no_extension": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        else:
            dropped = len(selected) - len(completed)
            seal(
                "kill.json",
                "t34-round0d-kill/v1",
                {
                    "status": "KILL",
                    "operational_reason": "operator abort or unexplained interruption",
                    "dropped_pairs": dropped,
                    "dropped_fraction": dropped / len(selected),
                    "drop_limit_fraction": 0.02,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        if isinstance(exc, Round0Error):
            raise
        raise Round0DError(
            f"Round 0-D worker stopped ({kind or 'kill'}); evidence preserved"
        ) from exc
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
        "sealed_labels": sealed,
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
        seal(
            "incomplete.json",
            "t34-round0d-incomplete/v1",
            {
                "status": "INCOMPLETE",
                "operational_reason": "export or serialization failure",
                "mechanical_rerun_eligible": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise Round0DError("Round 0-D chunk receipt export failed; INCOMPLETE preserved") from exc
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded paired T34 Round 0-D worlds")
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
    except Round0Error as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "preflight": receipt["preflight"],
                "pairs": len(receipt["pairs"]),
                "elapsed_seconds": receipt["elapsed_seconds"],
                "teacher_rollout_seconds": receipt["arms"]["teacher"]["rollout_elapsed_seconds"],
                "bare_rollout_seconds": receipt["arms"]["bare"]["rollout_elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
