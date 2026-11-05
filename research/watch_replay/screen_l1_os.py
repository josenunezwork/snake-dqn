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
import signal
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from research.watch_replay.seed_audit import BANK_V2_SCREEN_NAMESPACE, seed_for
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


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)).encode()


def allocation(
    namespace: str = BANK_V2_SCREEN_NAMESPACE, count: int = COUNT, horizon: int = HORIZON
) -> dict[str, Any]:
    """Frozen outcome-blind paired allocation (unit = one arm of one world)."""
    unit_id(namespace, ALLOCATION, 0)
    if not 1 <= count <= 128 or not 1 <= horizon <= 6000:
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


def latest_checkpoint(dirs: list[Path], index: int, contract: dict) -> dict | None:
    """Newest verified checkpoint for ``index`` bound to this contract, else None."""
    best: dict | None = None
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
                or value.get("allocation") != identity(contract)
                or value.get("index") != index
            ):
                raise PackageRefused(f"foreign checkpoint: {path}")
            if best is None or value["frame"] > best["frame"]:
                best = value
    return best


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
) -> dict:
    """Run (or resume) one arm of one world and publish its immutable result."""
    from research.watch_replay import engine, engine_l1_os

    index, arm = unit["index"], unit["arm"]
    ckpt_root = out / "checkpoints"
    dirs = [ckpt_root, *(checkpoint_dirs or [])]
    resumed = latest_checkpoint(dirs, index, contract)
    rejected = 0
    while True:
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
            )
        except CheckpointMismatch:
            if resumed is None:
                raise
            print(f"screen unit={index} arm={arm} checkpoint rejected; restarting", flush=True)
            resumed, rejected = None, rejected + 1


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
        faulthandler.dump_traceback_later(900, exit=True)
        if time.monotonic() - last_print >= 60:
            print(f"screen unit={index} arm={arm} frame={frame}/{horizon}", flush=True)
            last_print = time.monotonic()

    def guarded_check() -> None:
        try:
            check()
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
    faulthandler.dump_traceback_later(900, exit=True)
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


def initialize_worker(stop_event: object) -> None:
    global _STOP
    _STOP = stop_event


def work(task: tuple) -> dict:
    """Process-pool entry: one unit with stop fence + checkpoint-on-signal."""
    contract, unit, out, checkpoint_dirs, checkpoint_seconds = task
    stop = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def check() -> None:
        if stop or (_STOP is not None and _STOP.is_set()):
            raise InterruptedError("mechanical stop; checkpoint published")

    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    return run_unit(
        contract,
        unit,
        Path(out),
        check,
        checkpoint_dirs=[Path(p) for p in checkpoint_dirs],
        checkpoint_seconds=checkpoint_seconds,
    )


def pending_units(contract: dict, out: Path, ledger: Path | None) -> tuple[list[dict], list[int]]:
    """Units to run (l1-os first: longest first), and indices skipped as verified."""
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
    pending = [u for u in contract["units"] if u["index"] not in skip]
    pending.sort(key=lambda u: (u["arm"] != "exact-stop-l1-os", u["index"]))
    return pending, sorted(skip)


def run(
    contract: dict,
    out: Path,
    workers: int,
    *,
    ledger: Path | None = None,
    checkpoint_dirs: list[Path] | None = None,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
) -> dict:
    """Resume by default: skip verified units, continue checkpointed ones."""
    frozen = allocation(contract["namespace"], contract["count"], contract["horizon"])
    if {k: v for k, v in contract.items() if k != "commit"} != frozen:
        raise PackageRefused("contract differs from the frozen allocation")
    if not 1 <= workers <= max(1, min(os.cpu_count() or 1, 48)):
        raise ValueError("invalid worker count")
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise PackageRefused("output directory belongs to a different contract")
    else:
        publish_json(contract_path, contract)
    pending, skipped = pending_units(contract, out, ledger)
    dirs = [str(p) for p in (checkpoint_dirs or [])]
    tasks = [(contract, u, str(out), dirs, checkpoint_seconds) for u in pending]
    rows, exclusions = [], []
    try:
        if workers == 1:
            for task in tasks:
                try:
                    rows.append(work(task))
                except Exception as exc:  # noqa: BLE001 - recorded, outcome-blind
                    exclusions.append({"index": task[1]["index"], "error": repr(exc)[:400]})
                    if isinstance(exc, InterruptedError):
                        break
        else:
            context = multiprocessing.get_context("spawn")
            stop_event = context.Event()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(stop_event,),
            ) as pool:
                signal.signal(signal.SIGUSR1, lambda s, f: stop_event.set())
                signal.signal(signal.SIGTERM, lambda s, f: stop_event.set())
                futures = {pool.submit(work, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        rows.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        exclusions.append(
                            {"index": futures[future][1]["index"], "error": repr(exc)[:400]}
                        )
    finally:
        operational = {
            "allocation": identity(contract),
            "completed_now": sorted(r["index"] for r in rows),
            "skipped_verified": skipped,
            "exclusions": sorted(exclusions, key=lambda r: r["index"]),
            "total_units": len(contract["units"]),
            "generated_at": time.time(),
        }
        n = 0
        while (out / f"operational-{n:03d}.json").exists():
            n += 1
        publish_json(out / f"operational-{n:03d}.json", operational)
    return operational


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
            for path in sorted(unit_dir.glob("frame-*.json")):
                if _verified(path) and (index not in chosen or path.name > chosen[index].name):
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
        report = run(contract, args.out, 2, checkpoint_seconds=args.checkpoint_seconds)
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
    )
    print(json.dumps({k: report[k] for k in ("completed_now", "skipped_verified", "exclusions")}))


if __name__ == "__main__":
    main()
