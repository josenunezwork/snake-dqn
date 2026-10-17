"""Bounded, owner-run CUDA throughput smoke. No gates or serving changes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The notebook launches this file by path, so the repo root is not on sys.path; the in-process
# `from research.t16_colab.sampler import ...` below needs it (Colab smoke-01 failed here).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCHEMA = "t16-colab-throughput/v1"
COMMIT_SHA1 = re.compile(r"[0-9a-f]{40}")
FROZEN_POLICY_IDS = frozenset(
    {
        "b8-s3",
        "frp3-s12",
        "frp3-s12-v8",
        "champion-v8",
        "scripted:greedy_food",
        "scripted:random_safe",
    }
)
T16_CONFIGS = (
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
# Kept as the public/default matrix for the smoke-02 reproduction tests and old callers.
CONFIGS = T16_CONFIGS
SMOKE03_CONFIGS = (
    {
        "name": "sp_w128_a2_packed_g0_bs2_fp32",
        "worlds": 128,
        "actors": 2,
        "self_play": True,
        "gather_threads": 0,
        "actor_queue_frames": 4,
    },
    {
        "name": "sp_w128_a2_raw_g0_bs2_fp32",
        "worlds": 128,
        "actors": 2,
        "self_play": True,
        "gather_threads": 0,
        "actor_queue_frames": 4,
        "obs_pack": False,
    },
    {
        "name": "sp_w128_a3_raw_g3_bs2_fp32",
        "worlds": 128,
        "actors": 3,
        "self_play": True,
        "gather_threads": 3,
        "actor_queue_frames": 4,
        "obs_pack": False,
    },
    {
        "name": "sp_w128_a4_raw_g3_bs2_fp32",
        "worlds": 128,
        "actors": 4,
        "self_play": True,
        "gather_threads": 3,
        "actor_queue_frames": 4,
        "obs_pack": False,
    },
    {
        "name": "sp_w128_a6_raw_g3_bs2_fp32",
        "worlds": 128,
        "actors": 6,
        "self_play": True,
        "gather_threads": 3,
        "actor_queue_frames": 4,
        "obs_pack": False,
    },
    {
        "name": "sp_w128_a4_raw_g3_bs4_fp32",
        "worlds": 128,
        "actors": 4,
        "self_play": True,
        "gather_threads": 3,
        "actor_queue_frames": 4,
        "obs_pack": False,
        "batch_scale": 4,
        "throughput_only": True,
    },
    {
        "name": "sp_w128_a4_raw_g3_bs2_bf16",
        "worlds": 128,
        "actors": 4,
        "self_play": True,
        "gather_threads": 3,
        "actor_queue_frames": 4,
        "obs_pack": False,
        "precision": "bf16",
        "throughput_only": True,
    },
)
SMOKE03_PROFILE_CONFIG = "sp_w128_a4_raw_g3_bs2_fp32"
SMOKE03_PROFILE_WARM_SECONDS = 60
SMOKE03_PROFILE_SECONDS = 60
SMOKE03_PROFILE_STOP_GRACE_SECONDS = 5
# The first league-throughput receipt: same fixed recipe in both rows, with exactly three
# learner seats and the registered static frozen pool.  It deliberately excludes PFSP,
# scheduler bursts and payoff feedback; those need their own qualified implementation.
N3_CONFIGS = (
    {
        "name": "n3_w128_a3_raw_g3_bs2_fp32_static_pool",
        "worlds": 128,
        "actors": 3,
        "learner_seats": 3,
        "static_pool": True,
        "gather_threads": 3,
        "actor_queue_frames": 2,
        "obs_pack": False,
    },
    {
        "name": "n3_w128_a4_raw_g3_bs2_fp32_static_pool",
        "worlds": 128,
        "actors": 4,
        "learner_seats": 3,
        "static_pool": True,
        "gather_threads": 3,
        "actor_queue_frames": 2,
        "obs_pack": False,
    },
)
# One unscored run before the matrix: it populates the numba on-disk kernel cache and the
# inductor FX-graph cache, so no scored row pays first-process compilation.
WARMUP = {"name": "warmup_w128_a2", "worlds": 128, "actors": 2, "self_play": False}
SMOKE03_WARMUP = {
    "name": "warmup_selfplay_w128_a4_raw_g3",
    "worlds": 128,
    "actors": 4,
    "self_play": True,
    "gather_threads": 3,
    "actor_queue_frames": 4,
    "obs_pack": False,
}
N3_WARMUP = {
    "name": "warmup_n3_w128_a3_raw_g3_static_pool",
    "worlds": 128,
    "actors": 3,
    "learner_seats": 3,
    "static_pool": True,
    "gather_threads": 3,
    "actor_queue_frames": 2,
    "obs_pack": False,
}
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


def source_archive_commit(source_archive: Path) -> str:
    """Read the commit recorded in a gzip git archive's global PAX header."""
    try:
        with tarfile.open(source_archive, "r:gz") as archive:
            commit = archive.pax_headers.get("comment")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"cannot read source archive {source_archive}: {exc}") from exc
    if not isinstance(commit, str) or not COMMIT_SHA1.fullmatch(commit):
        raise ValueError("source archive is missing an exact 40-hex PAX comment commit")
    return commit


def verify_source_identity(commit: str, source_archive: Path | None = None) -> str | None:
    """Verify the requested commit against a checkout or its transported git archive."""
    if not COMMIT_SHA1.fullmatch(commit):
        raise ValueError("requested commit must be an exact 40-hex SHA-1")
    if source_archive is not None:
        archive_commit = source_archive_commit(source_archive)
        if archive_commit != commit:
            raise ValueError(
                f"pinned commit mismatch: requested {commit}, source archive {archive_commit}"
            )
        return sha256(source_archive)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != commit:
        raise ValueError(f"pinned commit mismatch: requested {commit}, checkout {head}")
    return None


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
        str(int(config.get("batch_scale", 2))),
        "--namespace",
        "t16-colab-throughput/v1",
        "--ring-dir",
        str(ring_dir),
    ]
    if config.get("obs_pack", True):
        command.append("--obs-pack")
    if config.get("precision", "fp32") != "fp32":
        command += ["--learner-precision", str(config["precision"])]
    if compile_mode != "off":
        command += ["--compile-learner", compile_mode]
    if config.get("self_play"):
        command += ["--all-snakes-learner"]
    if "learner_seats" in config:
        command += ["--learner-seats", str(int(config["learner_seats"]))]
    if config.get("static_pool"):
        command += ["--league-b8-checkpoint", str(inputs / "init" / "B8_s3_final.pth")]
    if "gather_threads" in config:
        command += ["--gather-threads", str(config["gather_threads"])]
    if config.get("actor_queue_frames") is not None:
        command += ["--actor-queue-frames", str(config["actor_queue_frames"])]
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
    # ``transitions`` remains the legacy D/R alias. League rows make the ledger
    # explicit: D is learner decisions, R is emitted replay rows and U is updates.
    decisions_key = "learner_decisions" if "learner_decisions" in last else "transitions"
    replay_key = "replay_rows" if "replay_rows" in last else "transitions"
    decisions = last[decisions_key] - first[decisions_key]
    replay_rows = last[replay_key] - first[replay_key]
    transitions = replay_rows
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
        tail_rate = (tail[-1][replay_key] - tail[0][replay_key]) / (
            tail[-1]["wall"] - tail[0]["wall"]
        )
    updates = last["updates"] - first["updates"]
    learn_seconds = last.get("learn_s", 0.0) - first.get("learn_s", 0.0)
    # Counter deltas use ``last - first``.  The telemetry row at ``first`` describes
    # work before that boundary, so only later interval summaries belong to the scored
    # window.  Legacy logs have no histogram and retain their historical mean behavior.
    scored_intervals = window[1:]
    histogram_rows = [
        row for row in scored_intervals if isinstance(row.get("policy_lag_histogram"), dict)
    ]
    # Historical suites only carried interval means. Preserve their established
    # aggregation while using exact boundary-aligned histograms for league rows.
    lag_rows = scored_intervals if histogram_rows else window
    lag_values = [row.get("policy_lag_mean") for row in lag_rows]
    finite_lags = [
        float(value)
        for value in lag_values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    p95_values = [
        float(row["policy_lag_p95"])
        for row in lag_rows
        if isinstance(row.get("policy_lag_p95"), (int, float))
        and math.isfinite(row["policy_lag_p95"])
    ]
    lag_histogram: dict[int, int] = {}
    for row in histogram_rows:
        histogram = row.get("policy_lag_histogram")
        if not isinstance(histogram, dict):
            continue
        for lag, count in histogram.items():
            if isinstance(count, int) and count >= 0:
                try:
                    lag_histogram[int(lag)] = lag_histogram.get(int(lag), 0) + count
                except (TypeError, ValueError):
                    pass
    total_lag_samples = sum(lag_histogram.values())
    lag_p95 = max(p95_values) if p95_values else None
    if total_lag_samples:
        lag_mean = sum(lag * count for lag, count in lag_histogram.items()) / total_lag_samples
        threshold = math.ceil(total_lag_samples * 0.95)
        cumulative = 0
        for lag in sorted(lag_histogram):
            cumulative += lag_histogram[lag]
            if cumulative >= threshold:
                lag_p95 = float(lag)
                break
    else:
        lag_mean = sum(finite_lags) / len(finite_lags) if finite_lags else None
    frozen_gpu_seconds = None
    if all(isinstance(row.get("frozen_forward_gpu_s"), (int, float)) for row in (first, last)):
        frozen_gpu_seconds = last["frozen_forward_gpu_s"] - first["frozen_forward_gpu_s"]
    first_live = first.get("per_policy_live_rows", first.get("live_rows_by_policy"))
    last_live = last.get("per_policy_live_rows", last.get("live_rows_by_policy"))
    live_rows_by_policy = None
    if isinstance(first_live, dict) and isinstance(last_live, dict):
        keys = set(first_live) | set(last_live)
        live_rows_by_policy = {
            str(key): int(last_live.get(key, 0)) - int(first_live.get(key, 0))
            for key in sorted(keys)
        }
    return {
        "transitions_per_second": rate,
        "tail_transitions_per_second": tail_rate,
        "updates_per_second": updates / elapsed,
        "updates": updates,
        "measured_updates": updates,
        "transitions_per_update": transitions / updates,
        "learner_decisions": decisions,
        "learner_decisions_per_second": decisions / elapsed,
        "replay_rows": replay_rows,
        "replay_rows_per_second": replay_rows / elapsed,
        "replay_rows_per_update": replay_rows / updates,
        "learn_seconds_per_update": learn_seconds / updates if updates else None,
        "policy_lag_mean": lag_mean,
        # Each log row is an interval summary; retaining the maximum protects the
        # tail while avoiding a false claim that concatenated interval p95s are raw samples.
        "policy_lag_p95": lag_p95,
        "policy_lag_histogram_samples": total_lag_samples,
        "live_rows_by_policy": live_rows_by_policy,
        "frozen_forward_gpu_seconds": frozen_gpu_seconds,
        "frozen_forward_gpu_seconds_per_second": (
            frozen_gpu_seconds / elapsed if frozen_gpu_seconds is not None else None
        ),
        "measured_transitions": transitions,
        "measured_seconds": elapsed,
        "first_transition": first[replay_key],
        "last_transition": last[replay_key],
        "first_update_wall": first_update_wall,
        "window_start_wall": first["wall"],
        "warmup_seconds": warm_seconds,
        "settle_seconds": settle_seconds,
    }


def require_league_metrics(measured: dict) -> None:
    """Reject an N=3 receipt that omits any counters needed to compare league cost."""
    numeric = (
        "learner_decisions",
        "replay_rows",
        "updates",
        "updates_per_second",
        "replay_rows_per_update",
        "policy_lag_mean",
        "policy_lag_p95",
        "policy_lag_histogram_samples",
        "frozen_forward_gpu_seconds",
    )
    missing = [
        key
        for key in numeric
        if isinstance(measured.get(key), bool)
        or not isinstance(measured.get(key), (int, float))
        or not math.isfinite(float(measured[key]))
        or measured[key] < 0
    ]
    live_rows = measured.get("live_rows_by_policy")
    if not isinstance(live_rows, dict) or not live_rows:
        missing.append("live_rows_by_policy")
    elif not any(
        policy in FROZEN_POLICY_IDS
        and isinstance(rows, int)
        and not isinstance(rows, bool)
        and rows > 0
        for policy, rows in live_rows.items()
    ):
        missing.append("frozen per-policy live rows")
    histogram_samples = measured.get("policy_lag_histogram_samples")
    if (
        isinstance(histogram_samples, bool)
        or not isinstance(histogram_samples, (int, float))
        or not math.isfinite(float(histogram_samples))
        or histogram_samples <= 0
    ):
        missing.append("policy_lag_histogram_samples")
    if missing:
        raise ValueError("missing league telemetry: " + ", ".join(missing))


def merge_telemetry_rows(log_rows: list[dict], telemetry_rows: list[dict]) -> list[dict]:
    """Join side-channel timing/lag fields to their matching trainer log interval."""
    by_transition = {
        row.get("transitions"): row
        for row in telemetry_rows
        if isinstance(row.get("transitions"), int)
    }
    merged = []
    for row in log_rows:
        item = dict(row)
        telemetry = by_transition.get(row.get("transitions"))
        if telemetry is not None:
            item.update(
                {
                    key: telemetry[key]
                    for key in (
                        "policy_lag_mean",
                        "policy_lag_p95",
                        "policy_lag_histogram",
                        "frozen_forward_gpu_s",
                        "frozen_forward_s",
                    )
                    if key in telemetry
                }
            )
        merged.append(item)
    return merged


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


def completed_profile_artifacts_match(profile: object) -> bool:
    """Accept only a complete, hash-receipted smoke-03 learner profile on resume."""
    if not isinstance(profile, dict):
        return False
    if (
        profile.get("returncode") != 0
        or profile.get("published") is not True
        or profile.get("mode") != "cprofile"
        or profile.get("seconds") != SMOKE03_PROFILE_SECONDS
        or profile.get("warmup_seconds") != SMOKE03_PROFILE_WARM_SECONDS
    ):
        return False
    metadata = profile.get("profile_metadata")
    learner = metadata.get("learner.meta.json") if isinstance(metadata, dict) else None
    profiled_seconds = learner.get("profiled_s") if isinstance(learner, dict) else None
    requested_duration = learner.get("requested_duration_s") if isinstance(learner, dict) else None
    recorded_warm_seconds = learner.get("warm_s") if isinstance(learner, dict) else None
    if (
        learner is None
        or learner.get("mode") != "cprofile"
        or learner.get("warm_from_first_tick") is not True
        or not isinstance(profiled_seconds, (int, float))
        or not math.isfinite(profiled_seconds)
        or profiled_seconds < SMOKE03_PROFILE_SECONDS - 0.1
        or requested_duration != SMOKE03_PROFILE_SECONDS
        or not isinstance(recorded_warm_seconds, (int, float))
        or not math.isclose(
            recorded_warm_seconds, SMOKE03_PROFILE_WARM_SECONDS, rel_tol=0.0, abs_tol=1e-6
        )
    ):
        return False
    artifacts = profile.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    required = {"learner.pstats", "learner.meta.json"}
    present = {
        Path(item["path"]).name for item in artifacts if isinstance(item, dict) and "path" in item
    }
    if not required <= present:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
        and Path(item["path"]).is_file()
        and sha256(Path(item["path"])) == item["sha256"]
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


def copy_attempt(source: Path, destination: Path, retries: int = 2) -> bool:
    """Publish a finished local attempt directory to the resumable (Drive) run directory.

    Best effort: a missing source (the trainer died before creating its output directory)
    or a Drive FUSE error must not abort the smoke, because the measurement itself lives in
    the local copy. Returns ``True`` when the destination holds a complete copy.
    """
    if not source.is_dir():
        return False
    for attempt in range(retries + 1):
        try:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            return True
        except OSError:
            if attempt == retries:
                return False
            time.sleep(2)
    return False


def finite_json(value):
    """Return ``value`` with non-finite floats replaced by ``None`` so strict JSON accepts it."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


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


def configs_for_suite(suite: str, omit_smoke03_row5: bool = False) -> tuple[dict, ...]:
    """Return the fixed config matrix for a named smoke suite."""
    if suite == "t16":
        return CONFIGS
    if suite == "smoke03":
        if omit_smoke03_row5:
            return tuple(
                config for index, config in enumerate(SMOKE03_CONFIGS, start=1) if index != 5
            )
        return SMOKE03_CONFIGS
    if suite == "n3":
        return N3_CONFIGS
    raise ValueError(f"unknown suite {suite!r}")


def matrix_configs(matrix: str) -> list[dict]:
    """Compatibility spelling for ``--suite``; both select the same fixed matrices."""
    return list(configs_for_suite("t16" if matrix == "smoke02" else matrix))


def warmup_for_suite(suite: str) -> dict:
    """Return an unscored cache warm-up matching the selected learner path."""
    if suite == "smoke03":
        return SMOKE03_WARMUP
    if suite == "n3":
        return N3_WARMUP
    return WARMUP


def run(
    inputs: Path,
    output: Path,
    commit: str,
    seconds: float,
    budget: float,
    scratch: Path | None = None,
    suite: str = "t16",
    omit_smoke03_row5: bool = False,
    source_archive: Path | None = None,
) -> int:
    """Run preflight and each timed configuration, retaining resumable Drive artifacts.

    Trainer outputs (checkpoints, telemetry, logs) go to local ``scratch`` while a run is
    timed and are copied to ``output`` afterwards, so Drive FUSE latency never sits in the
    learner loop. Results JSON is written to ``output`` directly (small, infrequent).
    """
    started = time.monotonic()
    # JSON round-trips result identity on Drive; do not retain tuple containers here.
    configs = [dict(config) for config in configs_for_suite(suite, omit_smoke03_row5)]
    warmup = warmup_for_suite(suite)
    output.mkdir(parents=True, exist_ok=True)
    scratch = (scratch or default_scratch()).resolve()
    source_archive = source_archive.resolve() if source_archive is not None else None
    scratch.mkdir(parents=True, exist_ok=True)
    result_path = output / "colab_throughput_results.json"
    save_path = result_path
    result = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "schema": SCHEMA,
        "commit": commit,
        "suite": suite,
        "omitted_smoke03_row5": omit_smoke03_row5,
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
            previous = json.loads(result_path.read_text())
            result["configs"] = previous.get("configs", [])
            if "profile" in previous:
                result["profile"] = previous["profile"]
        source_archive_sha256 = verify_source_identity(commit, source_archive)
        if source_archive_sha256 is not None:
            result["source_archive_sha256"] = source_archive_sha256
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
            "suite": suite,
            "omitted_smoke03_row5": omit_smoke03_row5,
            "configs": configs,
        }
        if source_archive_sha256 is not None:
            identity["source_archive_sha256"] = source_archive_sha256
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            save_path = output / "refused_identity.json"
            raise ValueError("output belongs to different inputs/code/runtime; use a new RUN_NAME")
        write_json(identity_path, identity)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            result["configs"] = previous.get("configs", [])
            if "profile" in previous:
                result["profile"] = previous["profile"]
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
        warm_marker = scratch / ("warmup-done" if suite == "t16" else f"warmup-{suite}-done")
        if not warm_marker.exists():  # caches are session-local: redo after a VM reset
            remaining = budget - (time.monotonic() - started)
            warm_seconds = min(WARMUP_SECONDS, max(20, remaining - seconds))
            warm_dir = scratch / warmup["name"]
            shutil.rmtree(warm_dir, ignore_errors=True)
            warm_dir.mkdir(parents=True)
            with tempfile.TemporaryDirectory(prefix="t16-rings-") as ring:
                command = trainer_command(
                    warmup,
                    inputs,
                    warm_dir / "attempt-001",
                    inputs / "init" / "B8_s3_final.pth",
                    warm_seconds,
                    mode,
                    Path(ring),
                )
                print(f"Warm-up {warmup['name']} ({warm_seconds:g}s, unscored)", flush=True)
                t_warm = time.monotonic()
                code = bounded_command(
                    command,
                    warm_dir / "attempt-001.log",
                    min(warm_seconds + TRAINER_GRACE_SECONDS, remaining - 5),
                    env,
                )
            result["warmup"] = {
                "name": warmup["name"],
                "seconds": warm_seconds,
                "returncode": code,
                "wall_seconds": time.monotonic() - t_warm,
                "completed": code == 0,
                "purpose": "numba kernel cache and inductor FX-graph cache; not scored",
            }
            if code == 0:
                warm_marker.write_text(json.dumps(result["warmup"]) + "\n")
            write_json(result_path, result)
        if suite == "smoke03" and not completed_profile_artifacts_match(result.get("profile")):
            profile_config = next(c for c in configs if c["name"] == SMOKE03_PROFILE_CONFIG)
            remaining = budget - (time.monotonic() - started)
            profile_wall_seconds = (
                SMOKE03_PROFILE_WARM_SECONDS
                + SMOKE03_PROFILE_SECONDS
                + SMOKE03_PROFILE_STOP_GRACE_SECONDS
            )
            if remaining < profile_wall_seconds + TRAINER_GRACE_SECONDS + 5:
                raise RuntimeError("budget exhausted before smoke-03 cProfile run")
            profile_dir = scratch / "profile-smoke03" / "attempt-001"
            shutil.rmtree(profile_dir, ignore_errors=True)
            profile_dir.parent.mkdir(parents=True, exist_ok=True)
            profile_env = {
                **env,
                "M3C_PROFILE": "cprofile",
                "M3C_PROFILE_WARM_S": str(SMOKE03_PROFILE_WARM_SECONDS),
                "M3C_PROFILE_WARM_FROM_FIRST_TICK": "1",
                "M3C_PROFILE_DURATION_S": str(SMOKE03_PROFILE_SECONDS),
                "M3C_PROFILE_DIR": str(profile_dir / "profile"),
            }
            with tempfile.TemporaryDirectory(prefix="t16-rings-") as ring:
                command = trainer_command(
                    profile_config,
                    inputs,
                    profile_dir,
                    inputs / "init" / "B8_s3_final.pth",
                    profile_wall_seconds,
                    mode,
                    Path(ring),
                )
                code = bounded_command(
                    command,
                    profile_dir.parent / "attempt-001.log",
                    min(profile_wall_seconds + TRAINER_GRACE_SECONDS, remaining - 5),
                    profile_env,
                )
            profile_destination = output / "profile-smoke03" / "attempt-001"
            published = copy_attempt(profile_dir, profile_destination)
            profile_files = sorted(profile_dir.glob("profile/*.meta.json"))
            profile_metadata = {
                path.name: finite_json(json.loads(path.read_text())) for path in profile_files
            }
            result["profile"] = {
                "config": profile_config["name"],
                "seconds": SMOKE03_PROFILE_SECONDS,
                "warmup_seconds": SMOKE03_PROFILE_WARM_SECONDS,
                "max_wall_seconds": profile_wall_seconds,
                "stop_grace_seconds": SMOKE03_PROFILE_STOP_GRACE_SECONDS,
                "mode": "cprofile",
                "timed_after_warmup": True,
                "returncode": code,
                "attempt": str(profile_destination),
                "published": published,
                "profile_files": [str(path.relative_to(profile_dir)) for path in profile_files],
                "profile_metadata": profile_metadata,
                "artifacts": (
                    [
                        {"path": str(path), "sha256": sha256(path)}
                        for path in sorted((profile_destination / "profile").glob("learner.*"))
                    ]
                    if published
                    else []
                ),
            }
            if not completed_profile_artifacts_match(result["profile"]):
                raise RuntimeError("smoke-03 cProfile run did not publish profile artifacts")
            write_json(result_path, result)
        for config in configs:
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
            # Publish the attempt (checkpoints, log.jsonl, summary, trainer log) to Drive. The
            # measurement is read from the local copy, so a Drive failure only costs the
            # resumable artifacts, never the row.
            row["published"] = copy_attempt(local_attempt, attempt)
            for extra in (f"{attempt.name}.log", f"{attempt.name}.resources.jsonl"):
                source = local_attempt.parent / extra
                if source.exists():
                    try:
                        shutil.copyfile(source, directory / extra)
                    except OSError:
                        row["published"] = False
            measured_dir = attempt if row["published"] else local_attempt
            row["returncode"] = code
            summary_path = measured_dir / "summary.json"
            if code or not summary_path.exists():
                row.update(status="failed", reason="trainer failed/timed out; see attempt log")
            else:
                summary = finite_json(json.loads(summary_path.read_text()))
                row["summary"] = summary
                row["actor_exitcodes"] = summary.get("actor_exitcodes")
                try:
                    if summary["halt"] or summary.get("ring_overrun_rows", 0):
                        raise ValueError("trainer halt or replay ring overrun")
                    if any(code != 0 for code in summary["actor_exitcodes"]):
                        raise ValueError("actor did not exit cleanly")
                    rows = [
                        json.loads(line)
                        for line in (measured_dir / "log.jsonl").read_text().splitlines()
                    ]
                    telemetry_path = measured_dir / "telemetry.jsonl"
                    if telemetry_path.exists():
                        telemetry_rows = [
                            json.loads(line) for line in telemetry_path.read_text().splitlines()
                        ]
                        rows = merge_telemetry_rows(rows, telemetry_rows)
                    measured = measured_rate(rows)
                    if suite == "n3":
                        require_league_metrics(measured)
                    row.update(measured, status="completed")
                    row["end_to_end_rate"] = summary["rate"]
                    row["compile_actual"] = summary.get("compile_learner", {"mode": "off"})
                    row["artifacts"] = [
                        {"path": str(path), "sha256": sha256(path)}
                        for path in (summary_path, measured_dir / "final.pth")
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
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help="gzip git archive whose PAX commit must match --commit (for archive-only runners)",
    )
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--budget-seconds", type=float, default=2700)
    parser.add_argument(
        "--suite",
        "--matrix",
        dest="suite",
        choices=("t16", "smoke02", "smoke03", "n3"),
        default="t16",
        help="pre-registered smoke suite; --matrix is a compatibility alias",
    )
    parser.add_argument(
        "--omit-smoke03-row5",
        action="store_true",
        help="explicitly omit only smoke-03 row 5 (a6 raw g3) when the notebook budget requires it",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="local-disk directory for trainer outputs during timing (default: /content)",
    )
    args = parser.parse_args()
    if args.suite == "smoke02":
        args.suite = "t16"
    if args.seconds is None:
        args.seconds = 300 if args.suite == "smoke03" else 150
    if not 120 <= args.seconds <= 600 or not 0 < args.budget_seconds <= 3600:
        parser.error("seconds must be 120..600 and budget must be >0 and <=3600")
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
        args.suite,
        args.omit_smoke03_row5,
        args.source_archive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
