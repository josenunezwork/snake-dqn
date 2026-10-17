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
CONFIGS = (
    [
        {"name": f"hero_w128_a{actors}", "worlds": 128, "actors": actors, "self_play": False}
        for actors in (2, 4, 8)
    ]
    + [
        {"name": f"selfplay_w{worlds}_a2", "worlds": worlds, "actors": 2, "self_play": True}
        for worlds in (64, 128)
    ]
    + [{"name": "selfplay_w128_a4", "worlds": 128, "actors": 4, "self_play": True}]
)
# One unscored run before the matrix: it populates the numba on-disk kernel cache and the
# inductor FX-graph cache, so no scored row pays first-process compilation.
WARMUP = {"name": "warmup_w128_a2", "worlds": 128, "actors": 2, "self_play": False}
WARMUP_SECONDS = 60
# Trainer startup (CUDA contexts, numba imports, ring allocation, K actor spawns) and the
# final checkpoint/summary are outside --max-wall-seconds; allow them before killing.
TRAINER_GRACE_SECONDS = 150
# A smoke never needs intermediate checkpoints; each one is a synchronous torch.save in the
# learner loop. The final checkpoint is still written and hash-receipted.
SMOKE_CHECKPOINT_EVERY = 10_000_000
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
        str(SMOKE_CHECKPOINT_EVERY),
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


def measured_rate(
    rows: list[dict],
    warm_seconds: float = 30,
    settle_seconds: float = 20,
    min_window_seconds: float = 30,
    tail_seconds: float = 60,
) -> dict:
    """Use an observed steady-state interval, never extrapolate from a progress counter.

    The window opens at the later of ``warm_seconds`` and ``settle_seconds`` after the
    first telemetry row with a learner update. Before the first update the learner is
    filling replay or compiling (``reduce-overhead`` can take a minute on a cold inductor
    cache) while actors run free or stall on back-pressure, so transitions in that span say
    nothing about the trained rate. The tail rate over the last ``tail_seconds`` is also
    reported; a large gap between the two means the window still held a transient.
    """
    updated = [row for row in rows if row.get("updates", 0) > 0]
    if not updated:
        raise ValueError("no telemetry row with a learner update; run too short or stalled")
    first_update_wall = updated[0]["wall"]
    start = max(warm_seconds, first_update_wall + settle_seconds)
    window = [row for row in rows if row.get("wall", 0) >= start]
    if len(window) < 2:
        raise ValueError(
            f"fewer than two telemetry samples after the steady-state start ({start:.0f}s)"
        )
    first, last = window[0], window[-1]
    elapsed = last["wall"] - first["wall"]
    transitions = last["transitions"] - first["transitions"]
    if elapsed < min_window_seconds or transitions <= 0 or last["updates"] <= first["updates"]:
        raise ValueError(
            f"need >={min_window_seconds:.0f} seconds of transitions and learner updates "
            "in the steady-state window"
        )
    rate = transitions / elapsed
    if not math.isfinite(rate):
        raise ValueError("nonfinite throughput")
    tail = [row for row in window if row["wall"] >= last["wall"] - tail_seconds]
    tail_rate = None
    if len(tail) >= 2 and tail[-1]["wall"] > tail[0]["wall"]:
        tail_rate = (tail[-1]["transitions"] - tail[0]["transitions"]) / (
            tail[-1]["wall"] - tail[0]["wall"]
        )
    return {
        "transitions_per_second": rate,
        "tail_transitions_per_second": tail_rate,
        "updates_per_second": (last["updates"] - first["updates"]) / elapsed,
        "measured_transitions": transitions,
        "measured_seconds": elapsed,
        "first_transition": first["transitions"],
        "last_transition": last["transitions"],
        "first_update_wall": first_update_wall,
        "window_start_wall": first["wall"],
        "warmup_seconds": warm_seconds,
        "settle_seconds": settle_seconds,
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
    }.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {versions[name]}")
    # Preserve the existing torch 2.9.x smoke contract; record the exact patch version.
    if not versions["torch"].split("+")[0].startswith("2.9."):
        raise RuntimeError(f"torch: expected 2.9.x, got {versions['torch']}")
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - informational only
        physical = None
    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_physical": physical,
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "cuda": torch.version.cuda,
        "c_compiler": shutil.which("gcc") or shutil.which("clang"),
    }


def default_scratch() -> Path:
    """Local-disk scratch for trainer outputs (never the FUSE-mounted Drive)."""
    content = Path("/content")
    base = content if content.is_dir() else Path(tempfile.gettempdir())
    return base / "t16-colab-scratch"


def copy_attempt(source: Path, destination: Path) -> None:
    """Publish a finished local attempt directory to the resumable (Drive) run directory."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def start_sampler(out: Path, env: dict) -> subprocess.Popen:
    """Start the owned GPU/CPU utilization sampler writing JSON lines to ``out``."""
    return subprocess.Popen(
        [sys.executable, "-m", "research.t16_colab.sampler", "--out", str(out)],
        cwd=REPO,
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_sampler(process: subprocess.Popen, out: Path) -> dict:
    """Stop the sampler and summarize its rows (empty when nothing was sampled)."""
    from research.t16_colab.sampler import summarize

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    rows = []
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return summarize(rows)


def run(
    inputs: Path,
    output: Path,
    commit: str,
    seconds: float,
    budget: float,
    scratch: Path | None = None,
) -> int:
    """Run preflight and each timed configuration, retaining resumable Drive artifacts.

    Trainer outputs (checkpoints, telemetry, logs) go to local ``scratch`` while a run is
    timed and are copied to ``output`` afterwards, so Drive FUSE latency never sits in the
    learner loop. Results JSON is written to ``output`` directly (small, infrequent).
    """
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    scratch = (scratch or default_scratch()).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    result_path = output / "colab_throughput_results.json"
    save_path = result_path
    result = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
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
        # Shared on-disk caches so the warm-up run's numba kernels and inductor FX graphs
        # are reused by every later trainer process in this session.
        "NUMBA_CACHE_DIR": str(scratch / "numba-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(scratch / "inductor-cache"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
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
            "python": result["python"],
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
        warm_marker = scratch / "warmup-done"
        if not warm_marker.exists():  # caches are session-local: redo after a VM reset
            remaining = budget - (time.monotonic() - started)
            warm_seconds = min(WARMUP_SECONDS, max(20, remaining - 600))
            warm_dir = scratch / WARMUP["name"]
            shutil.rmtree(warm_dir, ignore_errors=True)
            warm_dir.mkdir(parents=True)
            with tempfile.TemporaryDirectory(prefix="t16-rings-") as ring:
                command = trainer_command(
                    WARMUP,
                    inputs,
                    warm_dir / "attempt-001",
                    inputs / "init" / "B8_s3_final.pth",
                    warm_seconds,
                    mode,
                    Path(ring),
                )
                print(f"Warm-up {WARMUP['name']} ({warm_seconds:g}s, unscored)", flush=True)
                t_warm = time.monotonic()
                code = bounded_command(
                    command,
                    warm_dir / "attempt-001.log",
                    min(warm_seconds + TRAINER_GRACE_SECONDS, remaining - 5),
                    env,
                )
            result["warmup"] = {
                "name": WARMUP["name"],
                "seconds": warm_seconds,
                "returncode": code,
                "wall_seconds": time.monotonic() - t_warm,
                "completed": code == 0,
                "purpose": "numba kernel cache and inductor FX-graph cache; not scored",
            }
            if code == 0:
                warm_marker.write_text(json.dumps(result["warmup"]) + "\n")
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
            if remaining < seconds + TRAINER_GRACE_SECONDS + 30:
                row.update(status="budget_exhausted", reason="rerun Run all to continue")
                write_json(result_path, result)
                continue
            directory = output / config["name"]
            directory.mkdir(exist_ok=True)
            init = latest_checkpoint(directory) or inputs / "init" / "B8_s3_final.pth"
            attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:03d}"
            local_attempt = scratch / config["name"] / attempt.name
            shutil.rmtree(local_attempt, ignore_errors=True)
            local_attempt.parent.mkdir(parents=True, exist_ok=True)
            row.update(
                init_sha256=sha256(init),
                attempt=str(attempt),
                local_attempt=str(local_attempt),
                continuation="weights-only; optimizer/replay/actors/RNG/schedules restart",
                compile_requested=mode,
                ring_storage="local file-backed mmap",
                trainer_output="local scratch during the run, copied to Drive afterwards",
                oversubscription=(config["actors"] + 1) / max(1, result.get("cpu_affinity") or 1),
            )
            samples_path = local_attempt.parent / f"{attempt.name}.resources.jsonl"
            with tempfile.TemporaryDirectory(prefix="t16-rings-") as ring:
                command = trainer_command(
                    config, inputs, local_attempt, init, seconds, mode, Path(ring)
                )
                row["command"] = command
                write_json(result_path, result)
                print(f"Timing {config['name']} ({seconds:g}s)", flush=True)
                sampler = start_sampler(samples_path, env)
                t_run = time.monotonic()
                try:
                    code = bounded_command(
                        command,
                        local_attempt.parent / f"{attempt.name}.log",
                        min(seconds + TRAINER_GRACE_SECONDS, remaining - 5),
                        env,
                    )
                finally:
                    row["resources"] = stop_sampler(sampler, samples_path)
                    row["trainer_wall_seconds"] = time.monotonic() - t_run
            # Publish the attempt (checkpoints, log.jsonl, summary, trainer log) to Drive.
            copy_attempt(local_attempt, attempt)
            for extra in (f"{attempt.name}.log", f"{attempt.name}.resources.jsonl"):
                if (local_attempt.parent / extra).exists():
                    shutil.copyfile(local_attempt.parent / extra, directory / extra)
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
    parser.add_argument("--budget-seconds", type=float, default=2700)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="local-disk directory for trainer outputs during timing (default: /content)",
    )
    args = parser.parse_args()
    if not 120 <= args.seconds <= 180 or not 0 < args.budget_seconds <= 3600:
        parser.error("seconds must be 120..180 and budget must be >0 and <=3600")
    # The notebook owns this runner, while bounded_command owns separate trainer
    # groups. Propagate an outer deadline through Python cleanup of those groups.
    signal.signal(signal.SIGTERM, interrupt_on_signal)
    return run(
        args.inputs.resolve(),
        args.output.resolve(),
        args.commit,
        args.seconds,
        args.budget_seconds,
        args.scratch_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
