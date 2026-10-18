#!/usr/bin/env python3
"""Run Q0b's bounded four-cell CUDA qualification packet.

This is deliberately a measurement harness, not a synthetic receipt builder.  It
starts the registered multi-actor learner for every cell and accepts a packet only
when the learner wrote all required counters, two *actual* 180-second score-window
snapshots, and the CUDA parity/oracle subprocesses passed.  A missing boundary
snapshot is INCOMPLETE; aggregate end-of-run counters are never split or guessed.

The M3-C trainer must be invoked through a command template that writes
``q0b-boundaries.json`` into ``{out}``.  The file has three monotonically ordered
records named ``warmup``, ``window1`` and ``window2``.  This small explicit
instrumentation boundary keeps timing ownership with the learner process and avoids
using wall-clock polling as a proxy for learner decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.q0q1_20261017 import qualification

SCHEMA = qualification.PACKET_SCHEMA
APPROVED = "APPROVED_Q0Q1_20261017"
WARMUP_SECONDS = 120.0
WINDOW_SECONDS = 180.0
TOTAL_CELL_SECONDS = WARMUP_SECONDS + 2 * WINDOW_SECONDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _environment_identity() -> dict[str, str]:
    """Bind the Python/Torch stack without claiming a package lock exists."""
    try:
        import numpy
        import torch

        details = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "numpy": numpy.__version__,
        }
        if torch.cuda.is_available():
            details["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        details = {"python": sys.version, "platform": platform.platform()}
    return {"runtime": canonical_sha(details)}


def _command(template: Iterable[str], **values: str) -> list[str]:
    return [part.format(**values) for part in template]


def _phase_snapshot(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if value.get("phase") != name:
        raise ValueError(f"expected {name} boundary snapshot")
    if _finite(value.get("elapsed_seconds"), f"{name}.elapsed_seconds") < 0:
        raise ValueError(f"invalid {name} elapsed time")
    return value


def _delta(end: Mapping[str, Any], start: Mapping[str, Any], key: str) -> int:
    value = _positive_int(end.get(key), f"end.{key}") - _positive_int(
        start.get(key), f"start.{key}"
    )
    if value <= 0:
        raise ValueError(f"{key} did not advance in scored window")
    return value


def cell_from_boundaries(
    *,
    schedule: str,
    collector: str,
    actor_count: int,
    boundaries: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert learner-owned timing snapshots into the validator's one-cell schema."""
    warmup = _phase_snapshot(boundaries["warmup"], "warmup")
    first = _phase_snapshot(boundaries["window1"], "window1")
    second = _phase_snapshot(boundaries["window2"], "window2")
    warmup_elapsed = _finite(warmup["elapsed_seconds"], "warmup.elapsed_seconds")
    first_elapsed = _finite(first["elapsed_seconds"], "window1.elapsed_seconds")
    second_elapsed = _finite(second["elapsed_seconds"], "window2.elapsed_seconds")
    if warmup_elapsed < WARMUP_SECONDS:
        raise ValueError("warmup boundary is shorter than 120 seconds")
    if first_elapsed - warmup_elapsed < WINDOW_SECONDS:
        raise ValueError("first scored window is shorter than 180 seconds")
    if second_elapsed - first_elapsed < WINDOW_SECONDS:
        raise ValueError("second scored window is shorter than 180 seconds")
    actor_exitcodes = summary.get("actor_exitcodes")
    if summary.get("stopped_for_time") is not True or summary.get("halt") is not None:
        raise ValueError("learner did not cleanly finish its bounded qualification cell")
    if not isinstance(actor_exitcodes, list) or any(code != 0 for code in actor_exitcodes):
        raise ValueError("learner did not cleanly exit every actor")
    windows = []
    for before, after in ((warmup, first), (first, second)):
        decisions = _delta(after, before, "learner_decisions")
        windows.append({"seconds": WINDOW_SECONDS, "learner_decisions": decisions})
    final = second
    lag = final.get("policy_lag")
    control = final.get("matched_control")
    if not isinstance(lag, Mapping) or not isinstance(control, Mapping):
        raise ValueError("boundary snapshot lacks measured policy lag and matched control")
    counters = {
        "actual_learner_decisions": _positive_int(
            final.get("learner_decisions"), "learner_decisions"
        ),
        "world_frames": _positive_int(final.get("world_frames"), "world_frames"),
        "pool_decisions": _positive_int(final.get("pool_decisions"), "pool_decisions"),
        "emitted_replay_rows": _positive_int(
            final.get("emitted_replay_rows"), "emitted_replay_rows"
        ),
        "completed_updates": _positive_int(final.get("completed_updates"), "completed_updates"),
        "pending_update_debt": int(final.get("pending_update_debt", -1)),
    }
    if counters["pending_update_debt"] < 0:
        raise ValueError("pending_update_debt must be nonnegative")
    return {
        "schedule": schedule,
        "collector": collector,
        "actor_count": actor_count,
        "warmup_seconds": WARMUP_SECONDS,
        "compile_ready": final.get("compile_ready") is True,
        "replay_ready": final.get("replay_ready") is True,
        "scored_windows": windows,
        "clean_exit": True,
        "finite_updates": final.get("finite_updates") is True,
        "dropped_transitions": final.get("dropped_transitions"),
        "ring_overruns": final.get("ring_overruns"),
        "counters": counters,
        "policy_lag": {**lag, "matched_control": dict(control)},
        "long_body_exposure": final.get("long_body_exposure"),
        "memory": final.get("memory"),
    }


def make_packet(
    args: argparse.Namespace,
    cells: list[dict[str, Any]],
    parity: Mapping[str, Any],
    stages: Mapping[str, float],
) -> dict[str, Any]:
    """Build an auditable packet from measured values; no defaulted price or duration."""
    packet_seconds = sum(stages.values()) + len(cells) * TOTAL_CELL_SECONDS
    return {
        "schema": SCHEMA,
        "hardware": args.hardware,
        "packet_seconds": packet_seconds,
        "unit_rate_per_hour": args.unit_rate_per_hour,
        "runtime": {
            "precision": "fp32",
            "batch_scale": 2,
            "raw_local_rings": True,
            "queue_depth": 2,
            "emitted_rows_per_update": 256,
            "ram_gb": args.ram_gb,
            "cpu_cores": args.cpu_cores,
        },
        "stages": dict(stages),
        "source": {
            "q0b_measure": sha256_file(Path(__file__)),
            "train_m3c": sha256_file(
                args.repo_root / "research/redesign_m3c_20261008/train_m3c.py"
            ),
        },
        "config": {"trainer_command": canonical_sha(args.trainer_command)},
        "environment": _environment_identity(),
        "hashes": {
            "b8_initializer": sha256_file(args.b8),
            "t22_receipt": sha256_file(args.t22_receipt),
            "t24_receipt": sha256_file(args.t24_receipt),
        },
        "cells": cells,
        "parity": dict(parity),
    }


def validate_current_runtime(
    packet: Mapping[str, Any], *, expected_hardware: str
) -> dict[str, Any]:
    """Fail closed before a Q1 learner uses a qualified Q0b packet.

    This deliberately verifies the *current* worker's GPU declaration and the
    exact Python/Torch software identity recorded by Q0b.  It also requires the
    packet's CUDA T22/T24 fixture results; callers must run the package's own
    fresh CUDA fixture command before replacing this packet with another one.
    """
    errors: list[str] = []
    outcome = qualification.validate_packet(packet)
    if outcome["status"] != "QUALIFIED":
        errors.append("Q0b packet is not qualified")
    if expected_hardware not in {"L4", "A100"} or packet.get("hardware") != expected_hardware:
        errors.append("current job hardware does not match its qualified Q0b packet")
    try:
        import torch

        if (
            not torch.cuda.is_available()
            or expected_hardware not in torch.cuda.get_device_name(0).upper()
        ):
            errors.append("current worker does not expose the expected CUDA GPU class")
    except ImportError:
        errors.append("current worker cannot inspect CUDA hardware")
    environment = packet.get("environment")
    if not isinstance(environment, Mapping) or environment != _environment_identity():
        errors.append("current Python/Torch runtime differs from the Q0b qualified software stack")
    parity = packet.get("parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("target_runtime") != "cuda"
        or any(parity.get(name) != "PASS" for name in ("t22", "t24"))
    ):
        errors.append("qualified packet lacks passing target-CUDA T22/T24 fixtures")
    return {"pass": not errors, "errors": errors, "hardware": expected_hardware}


def _run(
    command: list[str], *, env: Mapping[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, env=dict(env), text=True, capture_output=True, check=False
    )


def run_cell(
    args: argparse.Namespace, schedule: str, collector: str, actor_count: int, out: Path
) -> dict[str, Any]:
    """Execute one exact 480-second learner measurement and read its owned snapshots."""
    cell_out = out / f"{schedule}-{collector}"
    trainer_collector = "task-aligned-n1" if collector == "n1" else "legacy"
    command = _command(
        args.trainer_command,
        out=str(cell_out),
        schedule=schedule,
        collector=trainer_collector,
        actors=str(actor_count),
        max_wall_seconds="3600",
        b8=str(args.b8),
        demo_data=str(args.demo_data),
        learner_seats="1" if collector == "n1" else "3",
    )
    required = (
        "--max-wall-seconds",
        "--batch-scale",
        "--actor-queue-frames",
        "--learner-precision",
    )
    if any(flag not in command for flag in required):
        raise ValueError(
            "trainer command must explicitly bind time, fp32 batch-scale, and queue depth"
        )
    started = time.monotonic()
    result = _run(
        command,
        env=os.environ
        | {"Q0Q1_MEASUREMENT": "1", "Q0Q1_BOUNDARY_OUT": str(cell_out / "q0b-boundaries.json")},
        cwd=args.repo_root,
    )
    elapsed = time.monotonic() - started
    (cell_out / "launcher.stdout").write_text(result.stdout)
    (cell_out / "launcher.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"trainer exited {result.returncode} for {schedule}/{collector}")
    boundaries = _load_json(cell_out / "q0b-boundaries.json")
    summary = _load_json(cell_out / "summary.json")
    if not isinstance(boundaries, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("trainer did not produce Q0b boundary/summary objects")
    cell = cell_from_boundaries(
        schedule=schedule,
        collector=collector,
        actor_count=actor_count,
        boundaries=boundaries,
        summary=summary,
    )
    cell["observed_launcher_seconds"] = elapsed
    return cell


def _parity(args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], float]:
    t22, t24 = _load_json(args.t22_receipt), _load_json(args.t24_receipt)
    for name, receipt in (("t22", t22), ("t24", t24)):
        if (
            receipt.get("schema") != "q0b-cuda-fixture/v1"
            or receipt.get("fixture") != name
            or receipt.get("status") != "PASS"
            or receipt.get("target_runtime") != "cuda"
            or receipt.get("short_body") is not True
            or receipt.get("long_body") is not True
            or not isinstance(receipt.get("source_sha256"), str)
            or len(receipt["source_sha256"]) != 64
        ):
            raise ValueError(f"{name.upper()} must be a completed hash-bound CUDA fixture receipt")
    receipt = out / "runtime-parity.json"
    command = [
        sys.executable,
        "-m",
        "research.q0q1_20261017.q1_parity",
        "--verify-reference",
        str(args.mac_parity_reference),
        "--bank",
        str(args.bank),
        "--b8",
        str(args.b8),
        "--out",
        str(receipt),
    ]
    started = time.monotonic()
    result = _run(command, env=os.environ, cwd=args.repo_root)
    elapsed = time.monotonic() - started
    (out / "runtime-parity.stdout").write_text(result.stdout)
    (out / "runtime-parity.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Q1 runtime parity verification failed")
    value = _load_json(receipt)
    if value.get("schema") != "q1-runtime-parity-receipt/v1" or value.get("pass") is not True:
        raise ValueError("runtime parity receipt is not a passing Q1 receipt")
    # ``q1_parity`` always runs exactly the four reserved H5000 worlds on one
    # CPU thread.  This is a direct per-episode observation, not a projection.
    return (
        {
            "target_runtime": "cuda",
            "short_body": True,
            "long_body": True,
            "t22": "PASS",
            "t24": "PASS",
        },
        elapsed / 4.0,
    )


def _oracles(args: argparse.Namespace, out: Path) -> None:
    """Run the registered non-timed correctness suite before any capacity cell."""
    result = _run(args.oracle_command, env=os.environ, cwd=args.repo_root)
    (out / "oracles.stdout").write_text(result.stdout)
    (out / "oracles.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Q0b oracle suite exited {result.returncode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--hardware", choices=("L4", "A100"), required=True)
    parser.add_argument("--unit-rate-per-hour", type=float, required=True)
    parser.add_argument("--observed-cpu-unit-rate-per-hour", type=float, required=True)
    parser.add_argument("--cpu-worker-efficiency", type=float, required=True)
    parser.add_argument("--ram-gb", type=float, required=True)
    parser.add_argument("--cpu-cores", type=float, required=True)
    parser.add_argument("--b8", required=True, type=Path)
    parser.add_argument("--demo-data", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--mac-parity-reference", required=True, type=Path)
    parser.add_argument("--t22-receipt", required=True, type=Path)
    parser.add_argument("--t24-receipt", required=True, type=Path)
    parser.add_argument(
        "--oracle-command", nargs="+", help="override registered Q0b oracle/parity argv"
    )
    parser.add_argument(
        "--trainer-command",
        nargs="+",
        help="override argv template with {out}, {schedule}, {collector}, {actors}, {max_wall_seconds}, {b8}, {demo_data}",
    )
    parser.add_argument("--n1-actors", type=int, required=True)
    parser.add_argument("--n3-actors", type=int, required=True)
    return parser


def _registered_defaults(args: argparse.Namespace) -> None:
    """Supply the fixed Q1 recipe so normal queue jobs need no shell template."""
    if args.oracle_command is None:
        args.oracle_command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_q1_trainer.py",
            "tests/test_q1_collector_oracles.py",
            "tests/test_q1_qualification.py",
            "tests/test_t22_random_safe.py",
            "tests/test_t24_v8_bodymap.py",
        ]
    if args.trainer_command is None:
        args.trainer_command = [
            sys.executable,
            "-m",
            "research.redesign_m3c_20261008.train_m3c",
            "--seed",
            "261701",
            "--init",
            "{b8}",
            "--demo-data",
            "{demo_data}",
            "--out",
            "{out}",
            "--transitions",
            "999999999",
            "--device",
            "cuda",
            "--actor-device",
            "cuda",
            "--max-wall-seconds",
            "{max_wall_seconds}",
            "--worlds",
            "128",
            "--actors",
            "{actors}",
            "--checkpoint-every",
            "999999999",
            "--reward",
            "mass",
            "--mass-kappa",
            "1.0",
            "--kill-scale",
            "0",
            "--gamma-end",
            "0.998",
            "--gamma-anneal-end",
            "8000000",
            "--mixed-frac",
            "0.25",
            "--anchor-mode",
            "v8margin",
            "--onpolicy-labels",
            "--label-frac",
            "1",
            "--obs-version",
            "bv8",
            "--v61-backend",
            "numba",
            "--learner-fast",
            "--learner-precision",
            "fp32",
            "--batch-scale",
            "2",
            "--ring-dir",
            "/dev/shm",
            "--compile-learner",
            "reduce-overhead",
            "--gather-threads",
            "3",
            "--actor-queue-frames",
            "2",
            "--schedule-package",
            "{schedule}",
            "--collection-profile",
            "{collector}",
            "--learner-seats",
            "{learner_seats}",
            "--namespace",
            "q0b-measurement-20261018/{collector}/{schedule}",
            "--league-b8-checkpoint",
            "{b8}",
        ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _registered_defaults(args)
    if (
        os.environ.get("Q0Q1_REGISTRATION_STATUS") != APPROVED
        or os.environ.get("Q0Q1_OWNER_RELEASE") != APPROVED
    ):
        raise SystemExit(
            "Q0b measurement requires the approved registration and owner-release environment"
        )
    if args.out.exists():
        raise SystemExit("Q0b output directory must be create-only")
    if any(
        _positive_int(value, name) < 1
        for value, name in ((args.n1_actors, "n1 actors"), (args.n3_actors, "n3 actors"))
    ):
        raise SystemExit("actor counts must be positive")
    args.out.mkdir(parents=True)
    began = time.monotonic()
    try:
        _oracles(args, args.out)
        parity, cpu_seconds_per_episode = _parity(args, args.out)
        cells = [
            run_cell(
                args,
                schedule,
                collector,
                args.n1_actors if collector == "n1" else args.n3_actors,
                args.out,
            )
            for collector in ("n1", "n3")
            for schedule in ("restart", "mature")
        ]
        elapsed = time.monotonic() - began
        stages = {
            "setup_seconds": max(0.0, elapsed - len(cells) * TOTAL_CELL_SECONDS),
            "cleanup_seconds": 0.0,
        }
        packet = make_packet(args, cells, parity, stages)
        packet["observed_units"] = packet["packet_seconds"] / 3600.0 * args.unit_rate_per_hour
        packet["observed_gpu_hours"] = packet["packet_seconds"] / 3600.0
        packet["observed_cpu_hours"] = elapsed / 3600.0
        packet["observed_cpu_unit_rate_per_hour"] = args.observed_cpu_unit_rate_per_hour
        packet["cpu_seconds_per_episode"] = cpu_seconds_per_episode
        packet["cpu_worker_efficiency"] = args.cpu_worker_efficiency
        packet["runtime_parity_receipt"] = "runtime-parity.json"
        packet["runtime_parity_receipt_sha256"] = sha256_file(args.out / "runtime-parity.json")
        status = qualification.validate_packet(packet)
        packet["qualification"] = status
    except Exception as exc:  # preserve an honest incomplete receipt
        packet = {"schema": SCHEMA, "status": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}"}
    (args.out / "receipt.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return 0 if packet.get("qualification", {}).get("status") == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
