"""Bounded, owner-run CUDA throughput smoke. No gates or serving changes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "t16-colab-throughput/v1"
CONFIGS = [
    {"name": f"hero_w128_a{actors}", "worlds": 128, "actors": actors, "self_play": False}
    for actors in (2, 4, 8)
] + [
    {"name": f"selfplay_w{worlds}_a2", "worlds": worlds, "actors": 2, "self_play": True}
    for worlds in (64, 128)
]
PARITY_TESTS = [
    "tests/test_grid_sim_parity.py::test_lockstep_small_arena[jit-True-False-2]",
    "tests/test_ego_live_identity.py::test_ego2s_live_equals_sim_every_frame[train-mid]",
]


def sha256(path: Path) -> str:
    """Hash an artifact without loading it into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Replace a result only after its complete temporary file is written."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def interrupt_on_signal(signum: int, frame: object) -> None:
    """Turn an owner deadline into normal Python child-process cleanup."""
    raise KeyboardInterrupt(f"received signal {signum}")


def bounded_command(command: list[str], log: Path, timeout: float, env: dict) -> int:
    """Bound only the subprocess group created here, including its own actors."""
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=max(1, timeout))
            # Also clean descendants left by an early trainer error or compiler shutdown.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return code
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            # A private process group from start_new_session; never discover/kill other jobs.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The leader may exit before its actors/compiler workers. Kill remaining
            # members of this owned group independently of the leader's wait status.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            if isinstance(exc, KeyboardInterrupt):
                raise
            return 124


def trainer_command(
    config: dict,
    inputs: Path,
    out: Path,
    init: Path,
    seconds: float,
    compile_mode: str,
    ring_dir: Path,
) -> list[str]:
    """B8 recipe with adopted batch-scale 2 and an explicit smoke time limit."""
    command = [
        sys.executable,
        "research/redesign_m3c_20261008/train_m3c.py",
        "--seed",
        "3",
        "--init",
        str(init),
        "--demo-data",
        str(inputs / "demo"),
        "--out",
        str(out),
        "--device",
        "cuda",
        "--actor-device",
        "cuda",
        "--worlds",
        str(config["worlds"]),
        "--actors",
        str(config["actors"]),
        "--transitions",
        "1000000000",
        "--max-wall-seconds",
        str(seconds),
        "--checkpoint-every",
        "100000",
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
        "--batch-scale",
        "2",
        "--obs-pack",
        "--namespace",
        "t16-colab-throughput/v1",
        "--ring-dir",
        str(ring_dir),
    ]
    if compile_mode != "off":
        command += ["--compile-learner", compile_mode]
    if config["self_play"]:
        command += ["--all-snakes-learner"]
    return command


def measured_rate(rows: list[dict], warm_seconds: float = 30) -> dict:
    """Use an observed post-warmup interval, never extrapolate from a progress counter."""
    rows = [row for row in rows if row.get("wall", 0) >= warm_seconds]
    if len(rows) < 2:
        raise ValueError("fewer than two post-warmup telemetry samples")
    first, last = rows[0], rows[-1]
    elapsed = last["wall"] - first["wall"]
    transitions = last["transitions"] - first["transitions"]
    if elapsed < 30 or transitions <= 0 or last["updates"] <= first["updates"]:
        raise ValueError("need >=30 seconds of transitions and learner updates after warmup")
    rate = transitions / elapsed
    if not math.isfinite(rate):
        raise ValueError("nonfinite throughput")
    return {
        "transitions_per_second": rate,
        "measured_transitions": transitions,
        "measured_seconds": elapsed,
        "first_transition": first["transitions"],
        "last_transition": last["transitions"],
        "warmup_seconds": warm_seconds,
    }


def latest_checkpoint(directory: Path) -> Path | None:
    """Find a hash-verified checkpoint, excluding explicitly aborted attempts."""
    candidates = []
    for manifest in sorted(directory.glob("attempt-*/manifest.jsonl")):
        summary = manifest.parent / "summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("halt"):
            continue
        for line in manifest.read_text().splitlines():
            try:
                row = json.loads(line)
                path = manifest.parent / Path(row["file"]).name
                if path.name != "aborted.pth" and path.is_file() and sha256(path) == row["sha256"]:
                    candidates.append(path)
            except (ValueError, KeyError):
                continue  # an interrupted final manifest line is not a checkpoint receipt
    return candidates[-1] if candidates else None


def completed_artifacts_match(row: dict) -> bool:
    """A resumable completion needs its summary and final checkpoint receipts."""
    artifacts = row.get("artifacts", [])
    return len(artifacts) == 2 and all(
        Path(item["path"]).is_file() and sha256(Path(item["path"])) == item["sha256"]
        for item in artifacts
    )


def environment_info() -> dict:
    """Check actual CUDA execution and record the environment used for timings."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime required; choose H100/A100/L4, then Run all")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "numba", "llvmlite", "pytest", "psutil", "pyyaml")
    }
    for name, expected in {
        "numpy": "1.26.4",
        "numba": "0.68.0",
        "llvmlite": "0.50.0",
        "torch": "2.9.1",
    }.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {versions[name]}")
    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cpu_count": os.cpu_count(),
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "cuda": torch.version.cuda,
        "c_compiler": shutil.which("gcc") or shutil.which("clang"),
    }


def run(inputs: Path, output: Path, commit: str, seconds: float, budget: float) -> int:
    """Run preflight and each timed configuration, retaining resumable Drive artifacts."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "colab_throughput_results.json"
    save_path = result_path
    result = {
        "schema": SCHEMA,
        "commit": commit,
        "configs": [],
        "status": "preflight",
        "authority": "throughput smoke only; not gate, quality, or serving evidence",
    }
    # Preserve the completion index through a failed preflight on reconnect. A row is
    # reused only after the identity check and artifact hash checks succeed below.
    env = {
        **os.environ,
        "SNAKE_CKPT_DIR": str(inputs / "checkpoints"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "SNAKE_DQN_DEVICE": "cpu",
    }
    try:
        if result_path.exists():
            result["configs"] = json.loads(result_path.read_text()).get("configs", [])
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if head != commit or len(commit) != 40:
            raise ValueError(f"pinned commit mismatch: requested {commit}, checkout {head}")
        result.update(environment_info())
        manifest_sha = sha256(inputs / "manifest.json")
        result["input_manifest_sha256"] = manifest_sha
        mode = "reduce-overhead" if result["c_compiler"] else "off"
        identity = {
            "commit": commit,
            "input_manifest_sha256": manifest_sha,
            "gpu": result["gpu"],
            "versions": result["versions"],
            "cpu_count": result["cpu_count"],
            "compile_mode": mode,
            "seconds": seconds,
            "configs": CONFIGS,
        }
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            save_path = output / "refused_identity.json"
            raise ValueError("output belongs to different inputs/code/runtime; use a new RUN_NAME")
        write_json(identity_path, identity)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            result["configs"] = previous.get("configs", [])
        print(json.dumps(result, indent=2), flush=True)
        code = bounded_command(
            [sys.executable, "-m", "pytest", "-q", *PARITY_TESTS],
            output / "parity.log",
            min(150, budget),
            env,
        )
        result["parity"] = {"returncode": code, "tests": PARITY_TESTS}
        if code:
            raise RuntimeError("environment parity failed/timed out; see parity.log")
        write_json(result_path, result)
        for config in CONFIGS:
            old = next((r for r in result["configs"] if r["name"] == config["name"]), None)
            if old and old["status"] == "completed" and completed_artifacts_match(old):
                print(f"Already complete: {config['name']}", flush=True)
                continue
            row = {**config, "status": "running", "transitions_per_second": None}
            result["configs"] = [r for r in result["configs"] if r["name"] != config["name"]]
            result["configs"].append(row)
            remaining = budget - (time.monotonic() - started)
            if remaining < seconds + 50:
                row.update(status="budget_exhausted", reason="rerun Run all to continue")
                write_json(result_path, result)
                continue
            directory = output / config["name"]
            directory.mkdir(exist_ok=True)
            init = latest_checkpoint(directory) or inputs / "init" / "B8_s3_final.pth"
            attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:03d}"
            row.update(
                init_sha256=sha256(init),
                attempt=str(attempt),
                continuation="weights-only; optimizer/replay/actors/RNG/schedules restart",
                compile_requested=mode,
                ring_storage="local file-backed mmap",
            )
            with tempfile.TemporaryDirectory(prefix="t16-rings-") as ring:
                command = trainer_command(config, inputs, attempt, init, seconds, mode, Path(ring))
                row["command"] = command
                write_json(result_path, result)
                print(f"Timing {config['name']} ({seconds:g}s)", flush=True)
                code = bounded_command(
                    command,
                    directory / f"{attempt.name}.log",
                    min(seconds + 45, remaining - 5),
                    env,
                )
            row["returncode"] = code
            summary_path = attempt / "summary.json"
            if code or not summary_path.exists():
                row.update(status="failed", reason="trainer failed/timed out; see attempt log")
            else:
                summary = json.loads(summary_path.read_text())
                row["summary"] = summary
                try:
                    if summary["halt"] or summary.get("ring_overrun_rows", 0):
                        raise ValueError("trainer halt or replay ring overrun")
                    if any(code != 0 for code in summary["actor_exitcodes"]):
                        raise ValueError("actor did not exit cleanly")
                    rows = [
                        json.loads(line)
                        for line in (attempt / "log.jsonl").read_text().splitlines()
                    ]
                    row.update(measured_rate(rows), status="completed")
                    row["end_to_end_rate"] = summary["rate"]
                    row["compile_actual"] = summary.get("compile_learner", {"mode": "off"})
                    row["artifacts"] = [
                        {"path": str(path), "sha256": sha256(path)}
                        for path in (summary_path, attempt / "final.pth")
                    ]
                except (ValueError, KeyError, OSError) as exc:
                    row.update(status="invalid_sample", reason=str(exc))
            write_json(result_path, result)
        result["status"] = (
            "completed"
            if all(row["status"] == "completed" for row in result["configs"])
            else "incomplete"
        )
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["invocation_wall_seconds"] = time.monotonic() - started
        write_json(save_path, result)
        write_json(output / "invocation_result.json", result)
        print(save_path, flush=True)
    return 0 if result["status"] == "completed" else 1


def main() -> int:
    """CLI used by the notebook's fresh Python environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seconds", type=float, default=150)
    parser.add_argument("--budget-seconds", type=float, default=1050)
    args = parser.parse_args()
    if not 120 <= args.seconds <= 180 or not 0 < args.budget_seconds <= 1200:
        parser.error("seconds must be 120..180 and budget must be >0 and <=1200")
    # The notebook owns this runner, while bounded_command owns separate trainer
    # groups. Propagate an outer deadline through Python cleanup of those groups.
    signal.signal(signal.SIGTERM, interrupt_on_signal)
    return run(
        args.inputs.resolve(), args.output.resolve(), args.commit, args.seconds, args.budget_seconds
    )


if __name__ == "__main__":
    raise SystemExit(main())
