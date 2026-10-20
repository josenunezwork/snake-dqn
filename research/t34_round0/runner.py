"""Create-only paired Round 0 trace producer.

The real controller is evaluated twice on every allocated world: first as bare
B8 and then with the Amendment-2 shadow/LAD teacher acting at beta=1.  The two
traces are retained for identity replay, but teacher labels are deliberately
removed before the public Round 0 export is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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

ROOT = Path(__file__).resolve().parents[2]
PAIR_SCHEMA = "t34-round0-pair/v1"
CHUNK_SCHEMA = "t34-round0-chunk/v1"
WORLD_SCHEMA = "t34-exit-world/v1"
HORIZON = 5000


class Round0Error(RuntimeError):
    """A fail-closed Round 0 producer error."""


class _ActionWindow:
    """O(1) exact 100k-action collapse guard for one controller arm."""

    def __init__(self) -> None:
        self.actions: deque[int] = deque()
        self.counts = [0] * 6

    def add(self, action: int) -> bool:
        if not 0 <= action < 6:
            raise Round0Error("illegal hero action")
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


def _validate_execution(*, preflight: bool, device: str, threads: int, seconds: float) -> None:
    if not math.isfinite(seconds) or not 0 < seconds <= 21000:
        raise Round0Error("operational wall bound must be in (0, 21000]")
    if preflight:
        if platform.system() != "Darwin" or device != "cpu" or threads != 1:
            raise Round0Error("preflight requires Darwin, cpu, and exactly one thread")
        return
    if platform.system() == "Darwin" or device != "cuda" or threads != 5:
        raise Round0Error("production requires non-Darwin CUDA with exactly five threads")


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
        try:
            prepare_world(world)
            evaluation = teacher.evaluate(world)
        except ValueError as exc:
            if type(exc).__name__ == "NonFiniteQ" or "non-finite" in str(exc).lower():
                raise Round0Error("bare policy produced non-finite Q") from exc
            raise
        if not np.isfinite(np.asarray(evaluation.q, dtype=np.float64)).all():
            raise Round0Error("bare policy produced non-finite Q")
        action = int(evaluation.greedy_action())
        if not 0 <= action < 6:
            raise Round0Error("bare policy returned illegal action")
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
    return {
        **dict(metrics),
        "trace": {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "final_state_hash": str(exported["hashes"][str(len(exported["actions"]))]),
        },
    }


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
    if preflight:
        from research.t34_round0.common import source_hashes

        result["source_hashes"] = source_hashes(ROOT)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank_path, audit_path = Path(args.bank), Path(args.audit)
    ratification = Path(args.ratification) if args.ratification else None
    checkpoint, out = Path(args.checkpoint), Path(args.output_dir)
    preflight = bool(args.preflight)
    _validate_execution(
        preflight=preflight, device=args.device, threads=args.threads, seconds=args.seconds
    )
    from research.t34_exit_b8.label import _cpu_ceiling
    from research.t34_round0.common import B8_SHA

    _cpu_ceiling(args.threads)
    os.environ["SNAKE_DQN_DEVICE"] = args.device
    import numba
    import torch

    if numba.get_num_threads() != args.threads:
        raise Round0Error("configured Numba thread ceiling did not take effect")
    torch.set_num_threads(1)
    bank_data = _load(bank_path)
    rows = _validate_bank(bank_data, preflight=preflight)
    _validate_audit(audit_path, bank_path)
    if preflight:
        if ratification is not None:
            raise Round0Error("preflight must not carry a campaign ratification")
    else:
        if ratification is None:
            raise Round0Error("production requires a ratification")
        rows = _registered_bank(ratification, bank_path, audit_path)
    if not checkpoint.is_file():
        raise Round0Error("B8 checkpoint is missing")
    if sha256(checkpoint) != B8_SHA:
        raise Round0Error("checkpoint differs from released B8 s3")
    if args.start < 0 or args.count < 1 or args.start + args.count > len(rows):
        raise Round0Error("requested contiguous bank slice is outside the immutable bank")
    selected = rows[args.start : args.start + args.count]
    if out.exists():
        raise Round0Error("output directory must be fresh")
    out.mkdir(parents=True)
    identity = _run_identity(
        runtime_id=args.runtime_id,
        bank=bank_path,
        audit=audit_path,
        ratification=ratification,
        checkpoint=checkpoint,
        preflight=preflight,
    )
    started = time.monotonic()
    heartbeat = started
    pairs: list[dict[str, str]] = []
    windows = {"bare": _ActionWindow(), "teacher": _ActionWindow()}

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

    def operational() -> None:
        nonlocal heartbeat
        now = time.monotonic()
        if now - started >= args.seconds:
            raise Round0Error("operational chunk wall bound reached; no complete receipt written")
        if now - heartbeat >= 110:
            print(
                json.dumps(
                    {
                        "event": "t34_round0_operational_progress",
                        "elapsed_seconds": now - started,
                        "completed_pairs": len(pairs),
                    }
                ),
                flush=True,
            )
            heartbeat = now

    def action_callback(arm: str, world_id: str) -> Any:
        def record(action: int) -> None:
            operational()
            if windows[arm].add(int(action)):
                fault(arm, "action_collapse_100k", world_id, "one action exceeded 90 percent")
                raise Round0Error(f"{arm} exceeded the one-action-collapse tripwire")

        return record

    def fault_marker(exc: BaseException) -> str | None:
        text = str(exc).lower()
        if type(exc).__name__ == "NonFiniteQ" or "non-finite" in text:
            return "nonfinite_q"
        if "illegal action" in text:
            return "illegal_action"
        return None

    for offset, row in enumerate(selected):
        operational()
        tag = _safe_world_id(str(row["world_id"]))
        trace_dir = out / "traces" / tag
        trace_dir.mkdir(parents=True)
        try:
            bare = _bare_trace(
                row,
                checkpoint,
                frames=int(row["horizon"]),
                device=args.device,
                on_action=action_callback("bare", str(row["world_id"])),
                on_progress=operational,
            )
        except (Round0Error, ValueError) as exc:
            marker = fault_marker(exc)
            if marker:
                fault("bare", marker, str(row["world_id"]), str(exc))
            raise
        bare_path = trace_dir / "bare.json"
        write_world(bare_path, bare)
        with tempfile.TemporaryDirectory(prefix="t34-round0-teacher-") as temp:
            teacher_out = Path(temp) / "teacher"
            teacher_callback = action_callback("teacher", str(row["world_id"]))
            try:
                result = run_worlds(
                    {
                        "checkpoint": checkpoint,
                        "output_dir": teacher_out,
                        "rows": [row],
                        "frames": int(row["horizon"]),
                        "seconds": max(1.0, args.seconds - (time.monotonic() - started)),
                        "beta": 1.0,
                        "device": args.device,
                        "worlds_per_process": 1,
                        "branch_bucket_size": 6,
                        "on_action": lambda _row, action: teacher_callback(action),
                    }
                )
            except (Round0Error, ValueError) as exc:
                marker = fault_marker(exc)
                if marker:
                    fault("teacher", marker, str(row["world_id"]), str(exc))
                raise
            if not result.get("complete") or result.get("worlds") != 1:
                raise Round0Error("teacher did not complete a world")
            teacher_raw = load_world(teacher_out / "world-000000.json")
        teacher = _strip_labels(teacher_raw)
        teacher_path = trace_dir / "teacher.json"
        write_world(teacher_path, teacher)
        arms = {}
        for name, trace, path in (("bare", bare, bare_path), ("teacher", teacher, teacher_path)):
            metrics = _metrics_from_trace(trace, checkpoint)
            arms[name] = _trace_meta(out, path, metrics, trace)
        pair = {
            "schema": PAIR_SCHEMA,
            "status": "complete",
            "preflight": preflight,
            "run_identity": identity,
            "world": dict(row),
            "arms": arms,
        }
        pair_path = out / "pairs" / f"{args.start + offset:03d}-{tag}.json"
        _write_new(pair_path, pair)
        pairs.append({"path": str(pair_path.relative_to(out)), "sha256": sha256(pair_path)})
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
        "elapsed_seconds": time.monotonic() - started,
        "operational": {
            "seconds_cap": args.seconds,
            "threads": args.threads,
            "device": args.device,
        },
    }
    _write_new(out / "chunk.json", receipt)
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded paired T34 Round 0 worlds")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--ratification")
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
        print(json.dumps(run(args), sort_keys=True))
    except Round0Error as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
