"""Bounded Drive-queue prelude for the approved Q0b measurement packets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017.run_queue_job import (
    _bound_json_bytes,
    q0_admission_checks,
    require_owner_release,
    sha256_file,
)


def bound_input(root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    """Read an operator input only after checking its synchronized SHA sidecar."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("Q0b input escapes Drive root")
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text().split()
    if not fields or len(fields[0]) != 64:
        raise RuntimeError(f"Q0b input lacks a SHA-256 sidecar: {path}")
    return path, _bound_json_bytes(str(path), fields[0])


def observed_pricing(value: Mapping[str, Any], hardware: str) -> tuple[float, float]:
    if value.get("schema") != "q0q1-observed-pricing/v1":
        raise RuntimeError("pricing must be a q0q1-observed-pricing/v1 observation")
    if not value.get("observed_at") or not value.get("evidence"):
        raise RuntimeError("pricing must identify its observation time and evidence")
    rates = []
    for device in (hardware, "cpu"):
        rate = value.get("rates", {}).get(device)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise RuntimeError(f"missing observed {device} debit rate")
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError(f"observed {device} debit rate must be positive")
        rates.append(float(rate))
    return rates[0], rates[1]


def attempt_age(env: Mapping[str, str], now: float) -> float:
    """Conservatively include arbitration and worker setup from the owning claim."""
    from research.colab_queue.core import Queue

    claims = Queue(Path(env["QUEUE_DRIVE_ROOT"]) / "queue").claims(
        env["QUEUE_JOB_ID"], int(env["QUEUE_GENERATION"])
    )
    matching = [
        claim
        for claim in claims
        if claim.get("token") == env["QUEUE_CLAIM_TOKEN"]
        and claim.get("worker_id") == env["QUEUE_WORKER_ID"]
    ]
    if len(matching) != 1:
        raise RuntimeError("Q0b cannot bind its owning queue claim/start time")
    started = dt.datetime.fromisoformat(matching[0]["created_at"].replace("Z", "+00:00"))
    elapsed = now - started.timestamp()
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("queue claim timestamp is in the future")
    return elapsed


def run_bounded(command: list[str], *, seconds: float, env: Mapping[str, str]) -> None:
    """Terminate the owned subprocess group, including spawned actors, on timeout."""
    if seconds <= 0:
        raise RuntimeError("Q0b's inclusive 60-minute packet cap is exhausted")
    label = command[2] if len(command) > 2 else command[0]
    started = time.monotonic()
    print(f"Q0b progress: started {label}; packet budget={seconds:.0f}s", flush=True)
    process = subprocess.Popen(command, env=dict(env), start_new_session=True)
    try:
        while True:
            elapsed = time.monotonic() - started
            remaining = seconds - elapsed
            if remaining <= 0:
                raise RuntimeError(f"Q0b child exceeded its bounded packet time: {label}")
            try:
                result = process.wait(timeout=min(120.0, remaining))
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                remaining = seconds - elapsed
                print(
                    f"Q0b progress: {label} still running; elapsed={elapsed:.0f}s "
                    f"remaining={max(0.0, remaining):.0f}s",
                    flush=True,
                )
        if result:
            raise RuntimeError(f"Q0b child exited {result}: {label}")
        print(
            f"Q0b progress: completed {label}; elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )
    finally:
        # A completed parent can leave failed actor descendants. Reap this group only.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hardware", choices=("L4", "A100"), required=True)
    args = parser.parse_args(argv)
    require_owner_release(os.environ)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    error = None
    try:
        q0_admission_checks(os.environ)
        import torch

        root = Path(os.environ["QUEUE_DRIVE_ROOT"])
        claim_age = attempt_age(os.environ, time.time())
        deadline = began + 3600 - claim_age - 60  # retain time for durable export/cleanup
        pricing_path, pricing = bound_input(
            root,
            os.environ.get("Q0Q1_PRICING_FILE", "q0q1-20261017/admission/observed-pricing.json"),
        )
        rate, cpu_rate = observed_pricing(pricing, args.hardware)
        reference, _ = bound_input(
            root,
            os.environ.get(
                "Q1_MAC_PARITY_REFERENCE", "q0q1-20261017/admission/mac-parity-reference.json"
            ),
        )
        if not torch.cuda.is_available():
            raise RuntimeError("Q0b requires the requested CUDA GPU")
        name = torch.cuda.get_device_name(0)
        if args.hardware not in name:
            raise RuntimeError(f"Q0b requested {args.hardware}, got {name}")
        inputs = Path(os.environ["Q0Q1_INPUT_ROOT"])
        child_env = dict(os.environ)
        # setup_runtime populates the checkpoint paths consumed by the pinned pool.
        child_env["SNAKE_DQN_DEVICE"] = "cpu"
        for fixture in ("t22", "t24"):
            run_bounded(
                [
                    sys.executable,
                    "-m",
                    "research.q0q1_20261017.q0b_cuda_fixtures",
                    "--fixture",
                    fixture,
                    "--out",
                    str(out / f"{fixture}-cuda.json"),
                ],
                seconds=deadline - time.monotonic() - 1920,
                env=child_env,
            )
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        measurement = out / "measurement"
        command = [
            sys.executable,
            "-m",
            "research.q0q1_20261017.q0b_measure",
            "--out",
            str(measurement),
            "--hardware",
            args.hardware,
            "--unit-rate-per-hour",
            str(rate),
            "--observed-cpu-unit-rate-per-hour",
            str(cpu_rate),
            "--cpu-worker-efficiency",
            "1",
            "--ram-gb",
            str(ram_gb),
            "--cpu-cores",
            str(cpus),
            "--b8",
            str(inputs / "init/B8_s3_final.pth"),
            "--demo-data",
            str(inputs / "demo"),
            "--bank",
            "research/q0q1_20261017/Q1_WORLD_BANK.json",
            "--mac-parity-reference",
            str(reference),
            "--t22-receipt",
            str(out / "t22-cuda.json"),
            "--t24-receipt",
            str(out / "t24-cuda.json"),
            "--n1-actors",
            os.environ.get("Q0Q1_N1_ACTORS", "3"),
            "--n3-actors",
            os.environ.get("Q0Q1_N3_ACTORS", "6"),
        ]
        prelude_seconds = time.monotonic() - began + claim_age
        run_bounded(command, seconds=deadline - time.monotonic(), env=child_env)
        receipt = json.loads((measurement / "receipt.json").read_text())
        if receipt.get("qualification", {}).get("status") != "QUALIFIED":
            raise RuntimeError("measurement did not produce a qualified packet")
        elapsed = time.monotonic() - began + claim_age
        receipt["packet_seconds"] = elapsed
        receipt["stages"]["setup_seconds"] += claim_age + max(
            0.0, elapsed - claim_age - receipt["observed_gpu_hours"] * 3600
        )
        receipt["observed_gpu_hours"] = elapsed / 3600
        receipt["observed_units"] = rate * elapsed / 3600
        receipt["observed_cpu_hours"] = 0.0  # this attempt owns a GPU allocation
        receipt["hashes"]["observed_pricing"] = sha256_file(pricing_path)
        receipt["hashes"]["mac_reference"] = sha256_file(reference)
        receipt["queue_setup_includes_arbitration"] = True
        cell_overheads = [
            max(
                0.0,
                cell["observed_launcher_seconds"]
                - cell["warmup_seconds"]
                - sum(window["seconds"] for window in cell["scored_windows"]),
            )
            for cell in receipt["cells"]
        ]
        receipt["q1_attempt_overhead_seconds"] = (
            prelude_seconds + 4 * receipt["cpu_seconds_per_episode"] + max(cell_overheads) + 60
        )
        receipt["stages"]["export_reserve_seconds"] = 60.0
        receipt["packet_seconds"] += 60.0
        receipt["stages"]["cleanup_seconds"] += 60.0
        receipt["packet_seconds_includes_export_reserve"] = True
        from research.q0q1_20261017.qualification import validate_packet

        receipt["qualification"] = validate_packet(receipt)
        if receipt["qualification"]["status"] != "QUALIFIED":
            raise RuntimeError(f"inclusive packet accounting failed: {receipt['qualification']}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        receipt = {"schema": "q0q1-qualification-packet/v1", "status": "INCOMPLETE", "error": error}
    (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return 2 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
