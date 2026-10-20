#!/usr/bin/env python3
"""Runtime fixtures for the bounded Colab CLI acceptance qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_STOP_REQUESTED = False
_ACTIVE_STATE: dict[str, Any] | None = None


def request_checkpoint(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def checkpoint_if_requested() -> None:
    """Handle SIGUSR1 at a completed toy update, never mid-state mutation."""
    if _STOP_REQUESTED:
        output = Path(os.environ["COLAB_OUTPUT_DIR"])
        state = _ACTIVE_STATE if _ACTIVE_STATE is not None else initial_state()
        publish(output / "lease-checkpoint.pkl", pickle.dumps(state, protocol=5))
        raise SystemExit(0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish an immutable artifact in the runtime supervisor's export format."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    digest = sha256(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    path.with_name(path.name + ".complete").write_text("complete\n")
    return {"path": path.name, "sha256": digest, "bytes": path.stat().st_size}


def publish_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    return publish(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def full_snapshot(path: Path, size_bytes: int, *, incompressible: bool = False) -> dict[str, Any]:
    block = hashlib.sha256(b"snake-dqn-colab-cli-acceptance-v1").digest() * (1024 * 1024 // 32)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as destination:
        remaining = size_bytes
        while remaining:
            count = min(remaining, len(block))
            chunk = os.urandom(count) if incompressible else block[:count]
            destination.write(chunk)
            remaining -= len(chunk)
    os.replace(temporary, path)
    digest = sha256(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    path.with_name(path.name + ".complete").write_text("complete\n")
    return {"path": path.name, "sha256": digest, "bytes": size_bytes}


def toy_step(state: dict[str, Any]) -> None:
    """Mutate every toy full-state component deterministically once."""
    rng = random.Random()
    rng.setstate(state["rng"])
    draw = rng.randrange(1 << 30)
    state["rng"] = rng.getstate()
    state["weights"] = [value + (draw % 17) for value in state["weights"]]
    state["optimizer_momentum"] = [value * 3 + draw % 5 for value in state["optimizer_momentum"]]
    state["replay"].append((state["counters"]["steps"], draw))
    state["replay"] = state["replay"][-8:]
    state["target"] = list(state["weights"])
    state["actors"]["actor-0"] = draw
    state["counters"]["steps"] += 1


def initial_state() -> dict[str, Any]:
    rng = random.Random(20261018)
    return {
        "weights": [1, 2, 3],
        "optimizer_momentum": [0, 0, 0],
        "replay": [],
        "target": [1, 2, 3],
        "actors": {"actor-0": 0},
        "counters": {"steps": 0},
        "rng": rng.getstate(),
    }


def toy_resume(output: Path) -> dict[str, Any]:
    uninterrupted = initial_state()
    resumed = initial_state()
    for _ in range(9):
        toy_step(uninterrupted)
    for _ in range(4):
        toy_step(resumed)
    checkpoint = pickle.dumps(resumed, protocol=5)
    publish(output / "toy-checkpoint.pkl", checkpoint)
    resumed = pickle.loads(checkpoint)
    for _ in range(5):
        toy_step(resumed)
    left, right = pickle.dumps(uninterrupted, protocol=5), pickle.dumps(resumed, protocol=5)
    return publish_json(
        output / "toy-resume.json",
        {
            "scope": "synthetic toy exact-resume only; not evidence for a trainer",
            "uninterrupted_sha256": hashlib.sha256(left).hexdigest(),
            "resumed_sha256": hashlib.sha256(right).hexdigest(),
            "exact": left == right,
            "state_components": [
                "weights",
                "optimizer_momentum",
                "replay",
                "target",
                "actors",
                "counters",
                "rng",
            ],
        },
    )


def benchmark(iterations: int) -> float:
    """Fixed useful toy updates; repeat counts, algorithm, and seed match."""
    global _ACTIVE_STATE
    state = initial_state()
    _ACTIVE_STATE = state
    started = time.monotonic()
    next_progress = started + 15
    for index in range(iterations):
        if index % 100 == 0:
            state = initial_state()
            _ACTIVE_STATE = state
        toy_step(state)
        if index % 100 == 0:
            checkpoint_if_requested()
            now = time.monotonic()
            if now >= next_progress:
                print(
                    json.dumps(
                        {
                            "event": "benchmark",
                            "done": index,
                            "total": iterations,
                            "elapsed": now - started,
                        }
                    ),
                    flush=True,
                )
                next_progress = now + 15
    return time.monotonic() - started


def control(root: Path, value: dict[str, Any]) -> None:
    temporary = root / ".acceptance-state.tmp"
    temporary.write_text(json.dumps(value))
    temporary.replace(root / "acceptance-state.json")


def await_control(root: Path, name: str) -> None:
    while not (root / name).exists():
        checkpoint_if_requested()
        time.sleep(0.2)


def suite(
    output: Path, *, silence_seconds: int, snapshot_bytes: int, profile: str = "final-only"
) -> list[dict[str, Any]]:
    # runtime.supervise creates COLAB_OUTPUT_DIR before it invokes its child.
    # A pre-existing artifact remains a foreign/reused attempt and is refused.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    suite_started = time.monotonic()
    root = Path(os.environ.get("COLAB_CLI_ROOT", str(output.parent)))
    identity = {"pid": os.getpid(), "execution_id": os.urandom(16).hex()}
    publish_json(output / "execution.json", identity)
    snapshot_elapsed = 0.0
    overhead = None
    if profile == "final-only":
        arm = os.environ.get("COLAB_ACCEPTANCE_ARM", "cpu")
        name, size = (
            ("final-artifact.bin", 64 * 1024 * 1024) if arm == "L4" else ("client-job.bin", 4096)
        )
        evidence = [full_snapshot(output / name, size, incompressible=arm == "L4")]
        control(root, {"phase": "transfer_ready", "snapshot": evidence[0], **identity})
        await_control(root, "acceptance-transfer-start")
        await_control(root, "acceptance-transfer-done")
        control(root, {"phase": "transfer_finished", **identity})
        await_control(root, "acceptance-finish")
    elif profile == "full-state":
        # Calibrate without an accelerator claim; identical fixed work is repeated.
        calibration = [benchmark(1000) for _ in range(3)]
        target = float(os.environ.get("COLAB_ACCEPTANCE_BENCH_SECONDS", "90"))
        iterations = max(1000, int(1000 * target / (sum(calibration) / len(calibration))))
        control(root, {"phase": "baseline", **identity})
        baseline = benchmark(iterations)
        snapshot_started = time.monotonic()
        evidence = [full_snapshot(output / "busy-snapshot.bin", snapshot_bytes)]
        snapshot_elapsed = time.monotonic() - snapshot_started
        snapshot_record = evidence[0]
        control(
            root,
            {
                "phase": "transfer_ready",
                "baseline_seconds": baseline,
                "snapshot_write_seconds": snapshot_elapsed,
                "iterations": iterations,
                "snapshot": snapshot_record,
                **identity,
            },
        )
        await_control(root, "acceptance-transfer-start")
        measured_started = time.monotonic()
        benchmark(iterations)
        # The export is asynchronous to useful work. Any transfer tail that delays
        # the next checkpoint is still charged, rather than hidden after the timer.
        await_control(root, "acceptance-transfer-done")
        measured = time.monotonic() - measured_started
        overhead = (measured + snapshot_elapsed - baseline) / baseline * 100
        measurement = {
            "scope": "synthetic useful-work transport benchmark; not trainer throughput",
            "baseline_seconds": baseline,
            "with_checkpoint_seconds": measured,
            "checkpoint_write_seconds": snapshot_elapsed,
            "iterations": iterations,
            "calibration_seconds": calibration,
            "overhead_percent": overhead,
            "threshold_percent": 2.0,
        }
        publish_json(output / "checkpoint-overhead.json", measurement)
        control(root, {"phase": "transfer_finished", "measurement": measurement, **identity})
        await_control(root, "acceptance-finish")
    else:
        raise ValueError(f"unknown acceptance profile: {profile}")
    started = time.monotonic()
    print(json.dumps({"event": "silence_start", "seconds": silence_seconds}), flush=True)
    time.sleep(silence_seconds)
    evidence.append(
        publish_json(
            output / "silence.json",
            {"silence_seconds": silence_seconds, "elapsed_seconds": time.monotonic() - started},
        )
    )
    child = subprocess.run([sys.executable, "-c", "import sys; sys.exit(23)"], check=False)
    evidence.append(
        publish_json(
            output / "intentional-child-failure.json",
            {"exit_code": child.returncode, "expected": 23},
        )
    )
    if profile == "full-state":
        evidence.append(toy_resume(output))
    evidence.append(
        publish_json(
            output / "completion.json",
            {
                "status": "PASS",
                "evidence": evidence,
                "measured": {
                    "snapshot_write_seconds": snapshot_elapsed,
                    "suite_seconds": time.monotonic() - suite_started,
                    "checkpoint_overhead_percent": overhead,
                },
            },
        )
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        choices=("final-only", "full-state"),
        default=os.environ.get("COLAB_ACCEPTANCE_PROFILE", "final-only"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume-input", type=Path)
    parser.add_argument("--resume-output", type=Path)
    parser.add_argument(
        "--silence-seconds",
        type=int,
        default=int(os.environ.get("COLAB_ACCEPTANCE_SILENCE_SECONDS", "65")),
    )
    parser.add_argument(
        "--snapshot-bytes",
        type=int,
        default=int(os.environ.get("COLAB_ACCEPTANCE_SNAPSHOT_BYTES", str(600 * 1024 * 1024))),
    )
    args = parser.parse_args(argv)
    if args.resume_input:
        if args.resume_output is None:
            parser.error("--resume-output is required with --resume-input")
        state = pickle.loads(args.resume_input.read_bytes())
        for _ in range(5):
            toy_step(state)
        args.resume_output.parent.mkdir(parents=True, exist_ok=True)
        publish_json(
            args.resume_output,
            {
                "resumed_sha256": hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest(),
                "steps": state["counters"]["steps"],
            },
        )
        return 0
    output = args.output or (
        Path(os.environ["COLAB_OUTPUT_DIR"]) if "COLAB_OUTPUT_DIR" in os.environ else None
    )
    if output is None:
        parser.error("--output or COLAB_OUTPUT_DIR is required")
    if args.silence_seconds <= 60 or args.snapshot_bytes <= 0:
        parser.error("silence must exceed 60 seconds and snapshot size must be positive")
    try:
        signal.signal(signal.SIGUSR1, request_checkpoint)
        evidence = suite(
            output.resolve(),
            silence_seconds=args.silence_seconds,
            snapshot_bytes=args.snapshot_bytes,
            profile=args.profile,
        )
    except FileExistsError:
        print("refusing prior/foreign acceptance output directory", file=sys.stderr)
        return 2
    print(json.dumps({"event": "complete", "artifacts": evidence}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
