"""Excluded-fixture C/T CUDA capture, never a science or continuation attempt.

Runs the real ct_run CLI to completion, then signals it during a large CUDA
checkpoint publication. The parent records signal-to-exit independently of the
child's marker. No producer, torch operation, or progress emitter is replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from research.sandbox_a_resident import ct_json as json
from research.sandbox_a_resident.ct_liveness import (
    finish_status,
    owns_output,
    print_status,
    set_phase,
)
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.ct_units import empty_ledger
from research.sandbox_a_resident.units import digest, publish

FIXTURE_OFFSET = 950_000_000
PADDING_BYTES = 255 * 1024 * 1024


def fixture_control(commit: str, *, signal_probe: bool = False) -> CTControl:
    """Small random-checkpoint worlds reserved exclusively for this canary."""
    return CTControl(
        source_commit=commit,
        seed_offset=FIXTURE_OFFSET + (1_000_000 if signal_probe else 0),
        allocation_index=80 + int(signal_probe),
        worlds=6,
        collection_ticks=12,
        max_body_capacity=128,
        max_deaths=2,
        lookbacks=(2, 4),
        max_ordinary_roots=4,
        anchor_rows=16,
        min_death_length=1,
        min_held_out_deaths=0,
        min_held_out_death_worlds=0,
        min_held_out_ordinary_roots=1,
        min_held_out_anchor_rows=1,
        branch_horizon=3,
        minibatches=2,
        searched_batch_size=2,
        anchor_batch_size=4,
        fit_seed=31,
        bootstrap_seed=131,
        bootstrap_resamples=50,
        scoring_chunks=2,
        branch_batch_size=16,
        snapshot_store_max_bytes=16 * 1024 * 1024,
        tiny=True,
    )


def run_capture(inputs: Path, output: Path, commit: str, *, device: str = "cuda") -> dict[str, Any]:
    """Publish immutable raw captures and a strict fixture-only qualification report."""
    import torch

    from research.sandbox_a_resident.ct_progress import (
        progress_inventory,
        validate_heartbeat,
        validate_progress,
        verify_cadence,
    )

    if output.exists() and not owns_output(output):
        raise ValueError("canary output already exists")
    output.mkdir(parents=True, exist_ok=owns_output(output))
    set_phase("probe-configuration")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required")
    probes = []
    for name, signal_probe in (("complete", False), ("signal", True)):
        ctrl = fixture_control(commit, signal_probe=signal_probe)
        control_path, ledger_path = output / f"{name}-control.json", output / f"{name}-ledger.json"
        control_path.write_text(json.dumps(ctrl.to_dict(), sort_keys=True) + "\n")
        ledger_path.write_text(json.dumps(empty_ledger(ctrl.to_dict()), sort_keys=True) + "\n")
        child_output = output / name
        command = [
            sys.executable,
            "-m",
            "research.sandbox_a_resident.ct_entrypoint",
            "runner",
            "--checkpoint",
            str(inputs / "b8-fixture.pth"),
            "--frp3",
            str(inputs / "frp3-fixture.pth"),
            "--champion",
            str(inputs / "champion-fixture.pth"),
            "--control",
            str(control_path),
            "--ledger",
            str(ledger_path),
            "--input-root",
            str(inputs),
            "--output",
            str(child_output),
            "--device",
            device,
            "--progress-interval-seconds",
            "0",
        ]
        if signal_probe:
            command += ["--fixture-stage-padding-bytes", str(PADDING_BYTES)]
        set_phase("probe-execution")
        started = time.monotonic()
        progress_times = [started]
        heartbeat_times = [started]
        observations = []
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        signalled: list[float] = []
        controller_errors: list[str] = []

        def controller() -> None:
            deadline = started + 600
            while child.poll() is None:
                if signal_probe and (child_output / "fixture-checkpoint-start").is_file():
                    signalled.append(time.monotonic())
                    child.send_signal(signal.SIGUSR1)
                    return
                if time.monotonic() >= deadline:
                    controller_errors.append("fixture child exceeded 600-second bound")
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    return
                time.sleep(0.002)

        thread = threading.Thread(target=controller, daemon=True)
        thread.start()
        log_path = output / f"{name}-entrypoint.log"
        with log_path.open("x") as log:
            assert child.stdout is not None
            for line in child.stdout:
                if line.startswith(("CT_PROGRESS ", "CT_HEARTBEAT ", "CT_SIGUSR1_CLEAN_EXIT ")):
                    received = time.monotonic()
                    progress_times.append(received)
                    if line.startswith("CT_HEARTBEAT "):
                        heartbeat_times.append(received)
                    observations.append(
                        {
                            "received_seconds": received - started,
                            "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                        }
                    )
                log.write(line)
                log.flush()
                print_status(line.rstrip("\n"), output)
        rc = child.wait(timeout=15)
        exited = time.monotonic()
        thread.join(timeout=1)
        progress_times.append(exited)
        heartbeat_times.append(exited)
        max_gap = max(b - a for a, b in zip(progress_times, progress_times[1:]))
        heartbeat_gap = max(b - a for a, b in zip(heartbeat_times, heartbeat_times[1:]))
        if max_gap > 60:
            raise RuntimeError("fixture progress gap exceeds unchanged 60-second limit")
        if heartbeat_gap > 30:
            raise RuntimeError("fixture heartbeat gap exceeds 30-second startup requirement")
        cadence = {
            "line_count": len(progress_times) - 2,
            "max_gap_seconds": max_gap,
            "limit_seconds": 60,
            "heartbeat_max_gap_seconds": heartbeat_gap,
            "heartbeat_limit_seconds": 30,
            "clock": "parent-monotonic-receive-including-start-and-exit",
            "exit_seconds": exited - started,
            "observations": observations,
        }
        verify_cadence(log_path.read_text(), cadence)
        if controller_errors:
            raise RuntimeError(controller_errors[0])
        if rc != (138 if signal_probe else 0):
            raise RuntimeError(f"fixture {name} returned unexpected status {rc}")
        rows = [
            json.loads(line.partition(" ")[2])
            for line in log_path.read_text().splitlines()
            if line.startswith("CT_PROGRESS ")
        ]
        heartbeat_rows = [
            json.loads(line.partition(" ")[2])
            for line in log_path.read_text().splitlines()
            if line.startswith("CT_HEARTBEAT ")
        ]
        for row in rows:
            validate_progress(row, fixture_completion=True)
        for row in heartbeat_rows:
            validate_heartbeat(row)
        if not heartbeat_rows:
            raise RuntimeError("fixture did not use the status-heartbeat entrypoint")
        if signal_probe:
            marker = json.loads((child_output / "sigusr1-marker.json").read_text())
            elapsed = exited - signalled[0] if len(signalled) == 1 else float("inf")
            size = (child_output / "stage-0001.pt").stat().st_size
            if not (
                marker["within_15s_sla"] is True
                and marker["latest_durable_stage"] == 1
                and elapsed < 15
                and PADDING_BYTES <= size <= ctrl.max_stage_export_bytes
                and (child_output / "stage-0001.json.complete").read_text() == "complete\n"
            ):
                raise RuntimeError("large-checkpoint SIGUSR1 qualification failed")
            probes.append(
                {
                    "name": name,
                    "progress_cadence": cadence,
                    "heartbeat_phases": sorted({r["phase"] for r in heartbeat_rows}),
                    "returncode": rc,
                    "runtime": json.loads((child_output / "fixture-runtime.json").read_text()),
                    "checkpoint_bytes": size,
                    "signal_to_exit_seconds": elapsed,
                    "marker": marker,
                }
            )
        else:
            stages = {row["stage"] for row in rows}
            required = {
                "initializing",
                "startup-self-checks-passed",
                "collection-progress",
                "scoring-progress",
                "fit-progress",
                "stage-0004-fit-start",
                "completed",
            }
            if not required <= stages:
                raise RuntimeError("fixture did not exercise all required progress phases")
            # Every static or formatted emitter seen here is inventory-covered.
            import re

            if any(
                not any(re.fullmatch(p, stage) for p in progress_inventory()) for stage in stages
            ):
                raise RuntimeError("fixture emitted a stage outside the AST inventory")
            probes.append(
                {
                    "name": name,
                    "progress_cadence": cadence,
                    "heartbeat_phases": sorted({r["phase"] for r in heartbeat_rows}),
                    "returncode": rc,
                    "runtime": json.loads((child_output / "fixture-runtime.json").read_text()),
                    "stages": sorted(stages),
                }
            )
    report = {
        "schema": "ct-cuda-fixture-capture/v1",
        "science_data": False,
        "device": device,
        "real_cuda": device == "cuda",
        "source_commit": commit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_allocated_bytes": max(p["runtime"].get("peak_allocated_bytes", 0) for p in probes),
        "probes": probes,
        "qualification": "AWAITING_OPERATOR_CAPTURE_REVIEW" if device == "cuda" else "CPU_ONLY",
    }
    set_phase("export-sealing")
    # Seal all raw evidence with the same completion-last publication primitive.
    for path in sorted(output.rglob("*")):
        if path.name == "progress-history.log" and path.parent == output and owns_output(output):
            continue  # Active parent history is sealed only after its thread joins.
        if path.is_file() and not path.name.endswith((".sha256", ".complete")):
            sidecar = Path(str(path) + ".sha256")
            if not sidecar.exists():
                sidecar.write_text(f"{digest(path)}  {path.name}\n")
                Path(str(path) + ".complete").write_text("complete\n")
    status_cadence = finish_status()
    if status_cadence is not None:
        if status_cadence["max_gap_seconds"] > 30:
            raise RuntimeError("canary parent heartbeat gap exceeds 30 seconds")
        report["startup_heartbeat_cadence"] = status_cadence
        history = output / "progress-history.log"
        history.with_suffix(".log.sha256").write_text(f"{digest(history)}  {history.name}\n")
        history.with_suffix(".log.complete").write_text("complete\n")
    publish(
        output / "capture.json",
        lambda p: p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run_capture(args.inputs, args.output, args.commit, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
