"""Bank-v2 paired dev screen: exact-stop-l1 vs exact-stop-l1-os, resumable inside units.

Allocation: ``BANK_V2_SCREEN_NAMESPACE`` worlds 0..47 (dev only; never training, tuning
or release evidence) x arms (exact-stop-l1, exact-stop-l1-os), H=6000, same seed per
pair.  Unit index = 2 * world + arm_index, so each arm of a world is its own work unit
(better load balance on a 40-worker host than world-pair units; pairs are joined by
``analyze_screen``).

Resume (owner rule: <= ~45 min of work lost):

* completed units are immutable ``units/<index>.json`` (+ .sha256 + .complete); they
  are skipped when verified locally or listed in the uploaded verified ledger;
* inside a unit, a checkpoint ``checkpoints/<index>/frame-<f>.json`` (immutable,
  sidecars) is published every ``--checkpoint-seconds`` of compute (default 1200 s) and
  on SIGUSR1/SIGTERM.  It holds the frame, a rolling SHA-256 over the canonical
  recorded Watch frames 1..f, the root counters as of frame f and the sparse list of
  action overrides (frame -> chosen).  Resuming re-simulates frames 1..f with search
  replaced by the recorded choice (chosen = bare except at logged overrides), which is
  ~12 ms/frame on the Mac, then verifies the rolling digest at frame f before any new
  search; a mismatch discards the checkpoint and restarts the unit from frame 1.
  Search never mutates the parent world (``_root_bytes`` checks) and the Bank-v2 build
  canary showed byte-identical frames with and without extra searches, which is what
  makes action-log replay exact; ``tests/test_screen_l1_os.py`` checks resumed ==
  uninterrupted on a real world.
"""

from __future__ import annotations

import argparse
import faulthandler
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from research.watch_replay.seed_audit import (
    BANK_V2_CONFIRM_COUNT,
    BANK_V2_CONFIRM_NAMESPACE,
    BANK_V2_SCREEN_COUNT,
    BANK_V2_SCREEN_NAMESPACE,
    seed_for,
)
from tools.ops.package_spec import (
    PackageRefused,
    publish_json,
    read_ledger,
    unit_id,
    verify_immutable,
)

SCHEMA = "t34-bank-v2-screen-unit/v1"
CHECKPOINT_SCHEMA = "t34-bank-v2-screen-checkpoint/v1"
ALLOCATION = "screen-units"
ARMS: tuple[str, str] = ("exact-stop-l1", "exact-stop-l1-os")
COUNT = 48
HORIZON = 6000
DEFAULT_CHECKPOINT_SECONDS = 1200.0
# Studies sharing this runner (allocation parameters only; same arms and metrics).
STUDIES: dict[str, dict[str, Any]] = {
    "screen": {
        "namespace": BANK_V2_SCREEN_NAMESPACE,
        "count": BANK_V2_SCREEN_COUNT,
        "horizon": 6000,
    },
    "confirm": {
        "namespace": BANK_V2_CONFIRM_NAMESPACE,
        "count": BANK_V2_CONFIRM_COUNT,
        "horizon": 14400,
    },
}
MAX_HORIZON = 14400
# Pool safety: a unit with no recorded frame for STALL_SECONDS checkpoints and fails
# itself; the parent kills a unit process whose heartbeat is older than HANG_SECONDS
# (covers hangs that never return to Python); the in-child faulthandler hard exit at
# 900 s only ever ends that unit's own process.
STALL_SECONDS = 840.0
HANG_SECONDS = 1200.0
HEARTBEAT_SECONDS = 10.0
STOP_GRACE_SECONDS = 60.0
MAX_RESUME_TRIES = 2
DEFAULT_RUNNER = "research.watch_replay.screen_l1_os:run_unit"
COUNTERS = (
    "exact_roots",
    "exact_overrides",
    "exact_branches",
    "scaled_horizon_roots",
    "open_stop_roots",
    "open_stop_fires",
    "open_stop_branches",
)
_STOP = None


class CheckpointMismatch(RuntimeError):
    """A resumed re-simulation did not reproduce the checkpointed frames."""


class UnitStalled(InterruptedError):
    """No recorded frame for STALL_SECONDS: checkpoint and fail only this unit."""


def hardware_fingerprint() -> str:
    """Coarse host identity recorded in checkpoints (same-hardware resumes preferred)."""
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        try:
            model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            model = ""
    return f"{platform.system()}/{platform.machine()}/{model}"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)).encode()


def allocation(
    namespace: str = BANK_V2_SCREEN_NAMESPACE, count: int = COUNT, horizon: int = HORIZON
) -> dict[str, Any]:
    """Frozen outcome-blind paired allocation (unit = one arm of one world)."""
    unit_id(namespace, ALLOCATION, 0)
    if not 1 <= count <= 128 or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError("invalid screen allocation")
    units = [
        {"index": 2 * world + a, "world": world, "arm": arm, "seed": seed_for(namespace, world)}
        for world in range(count)
        for a, arm in enumerate(ARMS)
    ]
    return {
        "schema": "t34-bank-v2-screen-allocation/v1",
        "namespace": namespace,
        "allocation": ALLOCATION,
        "count": count,
        "arms": list(ARMS),
        "horizon": horizon,
        "units": units,
    }


def identity(contract: dict) -> dict[str, str]:
    return {"contract_sha256": hashlib.sha256(canonical(contract)).hexdigest()}


def _unit_path(out: Path, index: int) -> Path:
    return out / "units" / f"{index:06d}.json"


def _verified(path: Path) -> bool:
    try:
        verify_immutable(path)
        return True
    except (PackageRefused, OSError):
        return False


def _set_aside(path: Path) -> None:
    """Move an unverified partial publication out of the way (never delete)."""
    for suffix in ("", ".sha256", ".complete"):
        item = Path(f"{path}{suffix}")
        if item.exists() or item.is_symlink():
            n = 0
            while Path(f"{item}.stale-{n}").exists():
                n += 1
            item.rename(Path(f"{item}.stale-{n}"))


def _rejected_keys(dirs: list[Path], index: int) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for root in dirs:
        for path in sorted((root / f"{index:06d}").glob("rejected-*.json")):
            if _verified(path):
                value = json.loads(path.read_text())
                keys.add((int(value["frame"]), str(value["frames_sha256"])))
    return keys


# Frozen allowlist of predecessor executable commits whose checkpoints a contract may
# resume (Bank-v2 confirmation: attempt 1 ran at f52d2827; attempt 2 needs a new commit
# for canary checkpoint handling).  Only the "commit" field may differ; every other
# contract field must be identical, and the rolling frame-hash replay check still runs.
PREDECESSOR_COMMITS: tuple[str, ...] = ("f52d2827096784a993bc40c226efdd863eaacd9e",)


def admitted_identities(contract: dict) -> list[dict[str, str]]:
    """identity(contract) plus identities of the same contract at allowlisted predecessors."""
    identities = [identity(contract)]
    if "commit" in contract:
        identities += [
            identity({**contract, "commit": commit})
            for commit in PREDECESSOR_COMMITS
            if commit != contract["commit"]
        ]
    return identities


def checkpoint_candidates(
    dirs: list[Path], index: int, contract: dict, *, foreign: str = "refuse"
) -> list[dict]:
    """Verified, non-rejected checkpoints for ``index``: same hardware first, then newest.

    A checkpoint bound to another contract is refused (``foreign="refuse"``, main runs)
    or skipped and counted (``foreign="skip"``, family canaries that upload the main's
    checkpoint archive only to cover its input bytes).
    """
    if foreign not in ("refuse", "skip"):
        raise ValueError("foreign must be 'refuse' or 'skip'")
    admitted = admitted_identities(contract)
    skipped = 0
    rejected = _rejected_keys(dirs, index)
    here = hardware_fingerprint()
    found: dict[tuple[int, str], dict] = {}
    for root in dirs:
        unit_dir = root / f"{index:06d}"
        if not unit_dir.is_dir():
            continue
        for path in sorted(unit_dir.glob("frame-*.json")):
            if not _verified(path):
                continue
            value = json.loads(path.read_text())
            if (
                value.get("schema") != CHECKPOINT_SCHEMA
                or value.get("allocation") not in admitted
                or value.get("index") != index
            ):
                if foreign == "skip":
                    skipped += 1
                    continue
                raise PackageRefused(f"foreign checkpoint: {path}")
            key = (int(value["frame"]), str(value["frames_sha256"]))
            if key not in rejected:
                found.setdefault(key, value)
    if skipped:
        print(f"screen unit={index}: skipped {skipped} foreign checkpoint(s)", flush=True)
    return sorted(
        found.values(), key=lambda v: (v.get("hardware") == here, v["frame"]), reverse=True
    )


def latest_checkpoint(dirs: list[Path], index: int, contract: dict) -> dict | None:
    """Preferred resumable checkpoint for ``index`` (see checkpoint_candidates), else None."""
    candidates = checkpoint_candidates(dirs, index, contract)
    return candidates[0] if candidates else None


def reject_checkpoint(out: Path, index: int, value: dict, reason: str) -> None:
    """Mark a checkpoint as non-reproducing so later resumes and packs skip it."""
    path = (
        out
        / "checkpoints"
        / f"{index:06d}"
        / f"rejected-{int(value['frame']):06d}-{value['frames_sha256'][:16]}.json"
    )
    if not path.exists():
        publish_json(
            path,
            {
                "index": index,
                "frame": int(value["frame"]),
                "frames_sha256": value["frames_sha256"],
                "checkpoint_hardware": value.get("hardware"),
                "rejecting_hardware": hardware_fingerprint(),
                "reason": reason,
            },
        )


class _RootLog:
    """Search interception: live search with counters, or forced replay of a checkpoint."""

    def __init__(self, forced_until: int, overrides: dict[int, int], counters: dict) -> None:
        self.forced_until = forced_until
        self.overrides = overrides  # frame -> chosen (only where chosen != bare)
        self.counters = dict(counters)
        self.new_overrides: dict[int, int] = {}

    def wrap(self, fn: Callable, kind: str) -> Callable:
        def wrapper(world: Any, decision: int, q: np.ndarray, legal: np.ndarray, *rest) -> dict:
            frame = int(decision) + 1
            bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
            if frame <= self.forced_until:
                chosen = self.overrides.get(frame, bare)
                return {"chosen": chosen, "bare": bare, "branches_evaluated": 0, "forced": True}
            result = fn(world, decision, q, legal, *rest)
            chosen = int(result["chosen"])
            branches = int(result.get("branches_evaluated", 0))
            c = self.counters
            if kind == "open_stop":
                c["open_stop_roots"] += 1
                c["open_stop_fires"] += int(chosen != bare)
                c["open_stop_branches"] += branches
            else:
                c["exact_roots"] += 1
                c["exact_overrides"] += int(chosen != bare)
                c["exact_branches"] += branches
                c["scaled_horizon_roots"] += int(kind == "scaled")
            if chosen != bare:
                self.new_overrides[frame] = chosen
            return result

        return wrapper


def _normalize(snapshot: dict) -> bytes:
    # Same normalisation as research/watch_replay/run.py (drop v8 wall diagnostics).
    veto = snapshot["session"].get("safety_veto")
    if isinstance(veto, dict):
        veto.pop("diagnostics", None)
    from web.backend.watch_replay import canonical as replay_canonical

    return replay_canonical(snapshot)


def run_unit(
    contract: dict,
    unit: dict,
    out: Path,
    check: Callable[[], None],
    *,
    checkpoint_dirs: list[Path] | None = None,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    checkpoint_frames: frozenset[int] = frozenset(),
    stall_seconds: float = STALL_SECONDS,
    heartbeat: Path | None = None,
    hard_watchdog: bool = False,
    foreign: str = "refuse",
) -> dict:
    """Run (or resume) one arm of one world and publish its immutable result.

    Up to MAX_RESUME_TRIES checkpoints are tried (same hardware first, then newest); a
    checkpoint whose replay does not reproduce its frame digest is marked rejected (and
    never offered again, here or in a packed resume archive) before the next candidate
    or a restart from frame 1.
    """
    from research.watch_replay import engine, engine_l1_os

    index, arm = unit["index"], unit["arm"]
    dirs = [out / "checkpoints", *(checkpoint_dirs or [])]
    candidates = checkpoint_candidates(dirs, index, contract, foreign=foreign)[:MAX_RESUME_TRIES]
    rejected = 0
    while True:
        resumed = candidates.pop(0) if candidates else None
        try:
            return _attempt(
                engine,
                engine_l1_os,
                contract,
                unit,
                out,
                check,
                resumed,
                rejected,
                checkpoint_seconds,
                checkpoint_frames,
                stall_seconds,
                heartbeat,
                hard_watchdog,
            )
        except CheckpointMismatch as exc:
            if resumed is None:
                raise
            reject_checkpoint(out, index, resumed, str(exc))
            print(f"screen unit={index} arm={arm} checkpoint rejected: {exc}", flush=True)
            rejected += 1


def _attempt(
    engine: Any,
    engine_l1_os: Any,
    contract: dict,
    unit: dict,
    out: Path,
    check: Callable[[], None],
    resumed: dict | None,
    rejected: int,
    checkpoint_seconds: float,
    checkpoint_frames: frozenset[int],
    stall_seconds: float,
    heartbeat: Path | None,
    hard_watchdog: bool,
) -> dict:
    index, arm, horizon = unit["index"], unit["arm"], contract["horizon"]
    forced_until = int(resumed["frame"]) if resumed else 0
    log = _RootLog(
        forced_until,
        {int(k): int(v) for k, v in (resumed or {}).get("overrides", {}).items()},
        (resumed or {}).get("counters", {name: 0 for name in COUNTERS}),
    )
    prior_compute = float((resumed or {}).get("compute_seconds", 0.0))
    digest = hashlib.sha256()
    state: dict[str, Any] = {"frame": 0}
    started = time.monotonic()
    replay_seconds = 0.0 if resumed else None
    last_ckpt = time.monotonic()
    last_print = time.monotonic()
    state["progress"] = time.monotonic()
    state["beat"] = 0.0
    hardware = hardware_fingerprint()

    def beat() -> None:
        if heartbeat is not None and time.monotonic() - state["beat"] >= HEARTBEAT_SECONDS:
            heartbeat.parent.mkdir(parents=True, exist_ok=True)
            heartbeat.touch()
            state["beat"] = time.monotonic()

    def publish_checkpoint(frame: int, frame_digest: str, counters: dict, overrides: dict) -> None:
        path = out / "checkpoints" / f"{index:06d}" / f"frame-{frame:06d}.json"
        if path.exists():
            return
        publish_json(
            path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "allocation": identity(contract),
                "index": index,
                "arm": arm,
                "seed": unit["seed"],
                "frame": frame,
                "frames_sha256": frame_digest,
                "counters": counters,
                "overrides": {str(k): v for k, v in sorted(overrides.items())},
                "compute_seconds": prior_compute + (time.monotonic() - started),
                "hardware": hardware,
            },
        )

    def record(snapshot: dict) -> None:
        nonlocal last_ckpt, last_print, replay_seconds
        frame = state["frame"] + 1
        digest.update(_normalize(snapshot))
        state["frame"] = frame
        if resumed and frame == forced_until:
            if digest.hexdigest() != resumed["frames_sha256"]:
                raise CheckpointMismatch(f"unit {index}: frames 1..{frame} differ")
            replay_seconds = time.monotonic() - started
        if frame > forced_until:
            overrides = {**log.overrides, **log.new_overrides}
            state["snapshot"] = (frame, digest.hexdigest(), dict(log.counters), overrides)
            now = time.monotonic()
            if frame < horizon and (
                frame in checkpoint_frames or now - last_ckpt >= checkpoint_seconds
            ):
                publish_checkpoint(*state["snapshot"])
                last_ckpt = now
        state["progress"] = time.monotonic()
        beat()
        faulthandler.dump_traceback_later(900, exit=hard_watchdog)
        if time.monotonic() - last_print >= 60:
            print(f"screen unit={index} arm={arm} frame={frame}/{horizon}", flush=True)
            last_print = time.monotonic()

    def guarded_check() -> None:
        try:
            check()
            beat()
            if time.monotonic() - state["progress"] > stall_seconds:
                raise UnitStalled(f"unit {index}: no frame for {stall_seconds:.0f} s")
        except InterruptedError:
            if "snapshot" in state:
                publish_checkpoint(*state["snapshot"])
            raise

    patches = [patch.object(engine, "dense_batch", log.wrap(engine.dense_batch, "exact"))]
    if arm == "exact-stop-l1-os":
        patches += [
            patch.object(
                engine_l1_os, "dense_batch_h", log.wrap(engine_l1_os.dense_batch_h, "scaled")
            ),
            patch.object(
                engine_l1_os,
                "open_stop_batch",
                log.wrap(engine_l1_os.open_stop_batch, "open_stop"),
            ),
        ]
    elif arm != "exact-stop-l1":
        raise ValueError(f"unknown screen arm {arm!r}")
    faulthandler.dump_traceback_later(900, exit=hard_watchdog)
    try:
        for item in patches:
            item.start()
        try:
            raw = engine_l1_os.run_controller(
                unit["seed"], arm, guarded_check, horizon=horizon, record=record
            )
        finally:
            for item in reversed(patches):
                item.stop()
    finally:
        faulthandler.cancel_dump_traceback_later()
    frames = raw["frames"]
    if len(frames) != horizon or state["frame"] != horizon:
        raise RuntimeError("incomplete screen unit")
    deaths = [f for f in frames if f["death"]]
    compute = prior_compute + (time.monotonic() - started)
    result = {
        "schema": SCHEMA,
        "unit_id": unit_id(contract["namespace"], ALLOCATION, index),
        "allocation": identity(contract),
        "index": index,
        "world": unit["world"],
        "arm": arm,
        "seed": unit["seed"],
        "horizon": horizon,
        "hero_deaths": len(deaths),
        "death_frames": [int(f["frame"]) for f in deaths],
        "death_lengths": [int(f["death_length"]) for f in deaths],
        "deaths_per_1k_ticks": 1000.0 * len(deaths) / horizon,
        # Same definition as run.py header stats.hero_mass_integral.
        "hero_mass_integral": sum(f["mass"] for f in frames) / horizon,
        "max_hero_length": max(f["mass"] for f in frames),
        "decisions": len(raw["decisions"]),
        **log.counters,
        "frames_sha256": digest.hexdigest(),
        "compute_seconds": compute,
        "replay_seconds": replay_seconds,
        "resumed_from_frame": forced_until or None,
        "checkpoints_rejected": rejected,
    }
    path = _unit_path(out, index)
    if path.exists() and not _verified(path):
        _set_aside(path)
    publish_json(path, result)
    return result


def _load_runner(spec: str) -> Callable[..., dict]:
    module, name = spec.split(":")
    return getattr(__import__(module, fromlist=[name]), name)


def work(task: dict) -> dict:
    """One unit with stop fence (+ checkpoint-on-signal); used in- and out-of-process."""
    stop = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    if task.get("isolated"):
        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def check() -> None:
        if stop or (_STOP is not None and _STOP.is_set()):
            raise InterruptedError("mechanical stop; checkpoint published")

    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    out = Path(task["out"])
    return _load_runner(task.get("runner", DEFAULT_RUNNER))(
        task["contract"],
        task["unit"],
        out,
        check,
        checkpoint_dirs=[Path(p) for p in task["checkpoint_dirs"]],
        checkpoint_seconds=task["checkpoint_seconds"],
        stall_seconds=task.get("stall_seconds", STALL_SECONDS),
        heartbeat=out / "heartbeats" / f"{task['unit']['index']:06d}",
        hard_watchdog=bool(task.get("isolated")),
        foreign=task.get("foreign", "refuse"),
    )


def _child(task: dict, stop_event: object) -> None:
    """Isolated unit process: any crash/hang ends only this process, never the pool."""
    global _STOP
    _STOP = stop_event
    try:
        work(task)
    except BaseException as exc:  # noqa: BLE001 - recorded for the parent, outcome-blind
        errors = Path(task["out"]) / "errors"
        errors.mkdir(parents=True, exist_ok=True)
        (errors / f"{task['unit']['index']:06d}-{os.getpid()}.txt").write_text(repr(exc)[:2000])
        sys.exit(3 if isinstance(exc, InterruptedError) else 1)


def pending_units(
    contract: dict,
    out: Path,
    ledger: Path | None,
    checkpoint_dirs: list[Path] | None = None,
    foreign: str = "refuse",
) -> tuple[list[dict], list[int]]:
    """Units to run and indices skipped as verified.

    Order: units with a resumable checkpoint first (finish in-flight work), then
    l1-os before l1 (longest first), then index.
    """
    skip: set[int] = set()
    if ledger is not None:
        skip |= {
            row["index"]
            for row in read_ledger(ledger, contract["namespace"], ALLOCATION, identity(contract))
        }
        if any(not 0 <= i < len(contract["units"]) for i in skip):
            raise PackageRefused("foreign allocation index in ledger")
    for unit in contract["units"]:
        path = _unit_path(out, unit["index"])
        if _verified(path):
            value = json.loads(path.read_text())
            if value.get("allocation") != identity(contract) or value.get("index") != unit["index"]:
                raise PackageRefused(f"foreign completed unit: {path}")
            skip.add(unit["index"])
    dirs = [out / "checkpoints", *(checkpoint_dirs or [])]
    pending = [u for u in contract["units"] if u["index"] not in skip]
    resumable = {
        u["index"]
        for u in pending
        if checkpoint_candidates(dirs, u["index"], contract, foreign=foreign)
    }
    pending.sort(
        key=lambda u: (u["index"] not in resumable, u["arm"] != "exact-stop-l1-os", u["index"])
    )
    return pending, sorted(skip)


def run(
    contract: dict,
    out: Path,
    workers: int,
    *,
    ledger: Path | None = None,
    checkpoint_dirs: list[Path] | None = None,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    session_seconds: float | None = None,
    isolate: bool | None = None,
    runner: str = DEFAULT_RUNNER,
    stall_seconds: float = STALL_SECONDS,
    hang_seconds: float = HANG_SECONDS,
    poll_seconds: float = 1.0,
    shard: tuple[int, int] | None = None,
    foreign: str = "refuse",
) -> dict:
    """Resume by default: skip verified units, continue checkpointed ones.

    ``isolate`` (default: workers > 1) runs every unit in its own spawned process, so a
    crash, faulthandler exit or hang fails only that unit.  ``session_seconds`` bounds
    one host session: when it elapses no new unit starts and running units publish a
    checkpoint and stop (completed next session from that checkpoint).
    """
    frozen = allocation(contract["namespace"], contract["count"], contract["horizon"])
    if {k: v for k, v in contract.items() if k != "commit"} != frozen:
        raise PackageRefused("contract differs from the frozen allocation")
    if not 1 <= workers <= max(1, min(os.cpu_count() or 1, 48)):
        raise ValueError("invalid worker count")
    isolate = workers > 1 if isolate is None else isolate
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise PackageRefused("output directory belongs to a different contract")
    else:
        publish_json(contract_path, contract)
    started = time.monotonic()
    pending, skipped = pending_units(contract, out, ledger, checkpoint_dirs, foreign)
    if shard is not None:
        k, n = shard
        if not 0 <= k < n:
            raise ValueError("shard must be k/n with 0 <= k < n")
        pending = [u for u in pending if u["world"] % n == k]
    tasks = [
        {
            "contract": contract,
            "unit": u,
            "out": str(out),
            "checkpoint_dirs": [str(p) for p in (checkpoint_dirs or [])],
            "checkpoint_seconds": checkpoint_seconds,
            "stall_seconds": stall_seconds,
            "runner": runner,
            "isolated": isolate,
            "foreign": foreign,
        }
        for u in pending
    ]
    completed: list[int] = []
    exclusions: list[dict] = []
    not_started: list[int] = []
    stopped: str | None = None
    try:
        if not isolate:
            for i, task in enumerate(tasks):
                if session_seconds is not None and time.monotonic() - started >= session_seconds:
                    stopped = "session budget"
                    not_started = [t["unit"]["index"] for t in tasks[i:]]
                    break
                try:
                    completed.append(work(task)["index"])
                except Exception as exc:  # noqa: BLE001 - recorded, outcome-blind
                    exclusions.append({"index": task["unit"]["index"], "error": repr(exc)[:400]})
                    if isinstance(exc, InterruptedError) and not isinstance(exc, UnitStalled):
                        stopped = "interrupted"
                        not_started = [t["unit"]["index"] for t in tasks[i + 1 :]]
                        break
        else:
            stopped = _schedule(
                tasks,
                workers,
                out,
                completed,
                exclusions,
                not_started,
                session_seconds,
                started,
                hang_seconds,
                poll_seconds,
            )
    finally:
        operational = {
            "allocation": identity(contract),
            "completed_now": sorted(completed),
            "skipped_verified": skipped,
            "exclusions": sorted(exclusions, key=lambda r: r["index"]),
            "not_started": sorted(not_started),
            "stopped": stopped,
            "session_seconds": session_seconds,
            "total_units": len(contract["units"]),
            "shard": list(shard) if shard else None,
            "hardware": hardware_fingerprint(),
            "generated_at": time.time(),
            "elapsed_seconds": time.monotonic() - started,
        }
        n = 0
        while (out / f"operational-{n:03d}.json").exists():
            n += 1
        publish_json(out / f"operational-{n:03d}.json", operational)
    return operational


def _schedule(
    tasks: list[dict],
    workers: int,
    out: Path,
    completed: list[int],
    exclusions: list[dict],
    not_started: list[int],
    session_seconds: float | None,
    started: float,
    hang_seconds: float,
    poll_seconds: float,
) -> str | None:
    """Process-per-unit scheduler with heartbeat hang kill and a session budget."""
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    reason: dict[str, Any] = {"why": None, "at": None}

    def request_stop(why: str) -> None:
        if not stop_event.is_set():
            stop_event.set()
            reason.update(why=why, at=time.monotonic())

    previous = {
        sig: signal.signal(sig, lambda s, f: request_stop("signal"))
        for sig in (signal.SIGUSR1, signal.SIGTERM)
    }
    queue = list(tasks)
    running: dict[int, tuple[Any, float]] = {}
    try:
        while queue or running:
            now = time.monotonic()
            if session_seconds is not None and now - started >= session_seconds:
                request_stop("session budget")
            while queue and len(running) < workers and not stop_event.is_set():
                task = queue.pop(0)
                proc = context.Process(target=_child, args=(task, stop_event))
                proc.start()
                running[task["unit"]["index"]] = (proc, time.monotonic())
            for index, (proc, t0) in list(running.items()):
                if not proc.is_alive():
                    proc.join()
                    del running[index]
                    if _verified(_unit_path(out, index)):
                        completed.append(index)
                    else:
                        errs = sorted((out / "errors").glob(f"{index:06d}-*.txt"))
                        exclusions.append(
                            {
                                "index": index,
                                "exitcode": proc.exitcode,
                                "error": errs[-1].read_text()[:400] if errs else None,
                            }
                        )
                    continue
                beat = out / "heartbeats" / f"{index:06d}"
                last = max(t0, beat.stat().st_mtime - time.time() + now if beat.exists() else t0)
                hung = now - last > hang_seconds
                late = stop_event.is_set() and now - reason["at"] > STOP_GRACE_SECONDS
                if hung or late:
                    proc.terminate()
                    proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    del running[index]
                    exclusions.append(
                        {
                            "index": index,
                            "exitcode": proc.exitcode,
                            "error": "hang: heartbeat stale" if hung else "stop grace exceeded",
                        }
                    )
            if stop_event.is_set() and not running:
                break
            if queue or running:
                time.sleep(poll_seconds)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for proc, _ in running.values():
            proc.kill()
    not_started.extend(t["unit"]["index"] for t in queue)
    return reason["why"]


def stage_checkpoint_resume(
    input_dir: Path, archive_name: str = "checkpoints-resume.tar.gz"
) -> Path:
    """Unpack an uploaded checkpoint archive (create-only) into input_dir/checkpoints-resume."""
    archive = input_dir / archive_name
    verify_immutable(archive)
    target = input_dir / "checkpoints-resume"
    target.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if (
                not member.isfile()
                or member.name.startswith("/")
                or ".." in parts
                or len(parts) != 2
            ):
                raise PackageRefused(f"unsafe checkpoint archive member: {member.name}")
            dest = target / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(member)
            assert data is not None
            with dest.open("xb") as handle:
                handle.write(data.read())
    return target


def pack_checkpoint_resume(sources: list[Path], archive: Path, exclude: set[int]) -> int:
    """Pack the newest verified checkpoint per unfinished unit (with sidecars)."""
    chosen: dict[int, Path] = {}
    for root in sources:
        for unit_dir in sorted(p for p in root.glob("[0-9]" * 6) if p.is_dir()):
            index = int(unit_dir.name)
            if index in exclude:
                continue
            rejected = _rejected_keys([root], index)
            for path in sorted(unit_dir.glob("frame-*.json")):
                if not _verified(path):
                    continue
                value = json.loads(path.read_text())
                if (int(value["frame"]), str(value["frames_sha256"])) in rejected:
                    continue
                if index not in chosen or path.name > chosen[index].name:
                    chosen[index] = path
    with (
        archive.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for index, path in sorted(chosen.items()):
            for suffix in ("", ".sha256", ".complete"):
                item = Path(f"{path}{suffix}")
                data = item.read_bytes()
                info = tarfile.TarInfo(f"{index:06d}/{item.name}")
                info.size, info.mtime, info.mode = len(data), 0, 0o444
                tar.addfile(info, io.BytesIO(data))
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--checkpoint-in", type=Path, nargs="*", default=[])
    parser.add_argument("--checkpoint-seconds", type=float, default=DEFAULT_CHECKPOINT_SECONDS)
    parser.add_argument("--shard", help="k/n: only worlds with world %% n == k")
    parser.add_argument(
        "--foreign-checkpoints",
        choices=("refuse", "skip"),
        default="refuse",
        help="skip: family canary carrying the main's checkpoint archive for input coverage",
    )
    parser.add_argument(
        "--session-seconds",
        type=float,
        help="host-session budget: afterwards running units checkpoint and stop",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="local dry run: 1 dev pair at H=120 (BANK_V2_DEV_NAMESPACE world 0), 2 workers",
    )
    args = parser.parse_args()
    if args.smoke:
        from research.watch_replay.seed_audit import BANK_V2_DEV_NAMESPACE

        contract = allocation(BANK_V2_DEV_NAMESPACE, 1, 120)
        started = time.monotonic()
        report = run(
            contract,
            args.out,
            2,
            checkpoint_seconds=args.checkpoint_seconds,
            session_seconds=args.session_seconds,
        )
        print(json.dumps({**report, "smoke_wall_seconds": time.monotonic() - started}))
        return
    if args.contract is None:
        parser.error("--contract is required unless --smoke")
    contract = json.loads(args.contract.read_text())
    report = run(
        contract,
        args.out,
        args.workers,
        ledger=args.ledger,
        checkpoint_dirs=args.checkpoint_in,
        checkpoint_seconds=args.checkpoint_seconds,
        session_seconds=args.session_seconds,
        shard=tuple(int(x) for x in args.shard.split("/")) if args.shard else None,
        foreign=args.foreign_checkpoints,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "completed_now",
                    "skipped_verified",
                    "exclusions",
                    "not_started",
                    "stopped",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
