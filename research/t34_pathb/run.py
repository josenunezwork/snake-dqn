"""Mac-only T34 development banks, real paced comparisons and timing probes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

# Before importing any numerical library, including the bank's evaluator imports.
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.t34_pathb import bank  # noqa: E402

ARMS = ("bare", "async", "sync")
FPS = (12, 24)
DEFAULTS = {
    "blocks": 5,
    "scripted_worlds": 96,
    "frozen_worlds": 48,
    "mixed_worlds": 48,
    "product_worlds": 25,
    "horizon": 5000,
    "workers": 1,
    "max_hours": 24.0,
    "warmup": 200,
    "ticks": 1500,
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _guard() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("T34 execution is Mac CPU only")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _power_guard() -> None:
    """Long real banks refuse battery, closed lid or unknown thermal status."""
    from research.perf_sim_20261007 import with_slots

    if not with_slots.on_ac() or not with_slots.lid_open():
        raise RuntimeError("real T34 banks require AC power and an open lid")
    from research.compute.thermal_guard import parse_pmset_therm, read_pmset_therm, thermal_reasons

    reading = read_pmset_therm()
    reasons = thermal_reasons(parse_pmset_therm(reading.get("text") or ""))
    if not reading.get("available") or reasons:
        raise RuntimeError(f"thermal admission refused: {reasons}")


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "horizon": args.horizon,
        "fps": list(FPS),
        "workers": args.workers,
        "max_hours": args.max_hours,
        "smoke": bool(args.smoke),
        "timing_samples": args.ticks,
    }


def _source_files() -> dict[str, str]:
    names = _git("ls-files").splitlines()
    selected = [
        name
        for name in names
        if (name.endswith(".py") and name.startswith(("src/", "research/", "web/backend/")))
        or (name.endswith(".yaml") and name.startswith(("configs/", "research/")))
    ]
    selected.extend(
        [
            "research/t34_pathb/PROTOCOL.md",
            "research/t34_pathb/PROTOCOL_TIMING_ERRATUM.md",
            "research/t34_pathb/PROTOCOL_ANALYSIS_ERRATUM.md",
            "research/t34_pathb/smoke.json",
        ]
    )
    return {name: sha(REPO / name) for name in sorted(set(selected))}


def _manifest(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from web.backend.served_checkpoint import B8_S3_SHA256, resolve_served_checkpoint

    choice = resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "b8-s3"})
    checkpoint = {"path": str(Path(choice.path).resolve()), "sha256": sha(Path(choice.path))}
    if checkpoint["sha256"] != B8_S3_SHA256 or choice.reason:
        raise RuntimeError("exact pinned B8 s3 required")
    context = gid._context(1)
    paths = {checkpoint["path"]}
    paths.update(
        str(Path(spec[1]).resolve())
        for spec in context["lookup"].values()
        if spec[0] == "checkpoint"
    )
    return {
        "schema": "t34-pathb-manifest/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "timing_erratum_sha256": sha(HERE / "PROTOCOL_TIMING_ERRATUM.md"),
        "analysis_erratum_sha256": sha(HERE / "PROTOCOL_ANALYSIS_ERRATUM.md"),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_files": _source_files(),
        "namespace": plan["namespace"],
        "checkpoint": checkpoint,
        "checkpoints": [{"path": path, "sha256": sha(Path(path))} for path in sorted(paths)],
        "deployment_profile": context["profile"].descriptor(),
        "bank": plan,
        "settings": _settings(args),
        "timing_settings": {"warmup": args.warmup, "ticks": args.ticks},
        "expected_records": sum(
            (len(b["terminal"]) + len(b["product"])) * len(FPS) for b in plan["blocks"]
        ),
        "mac_only": True,
        "no_tmux": True,
        "development_only": True,
        "decision_logs_required": True,
    }


def create_bank(args: argparse.Namespace) -> Path:
    _guard()
    if not args.out.is_absolute() or args.out.exists():
        raise RuntimeError("--out must name a fresh absolute directory")
    if args.workers != 1 or args.horizon < 1 or args.max_hours <= 0:
        raise RuntimeError("require workers=1, positive horizon and wall cap")
    if not args.smoke and (
        args.horizon != 5000
        or min(args.scripted_worlds, args.frozen_worlds, args.mixed_worlds, args.product_worlds) < 1
    ):
        raise RuntimeError("real development bank requires H5000 and all four world groups")
    if _git("status", "--porcelain"):
        raise RuntimeError("commit the implementation before preparing its immutable bank")
    plan = bank.allocation(
        blocks=args.blocks,
        scripted=args.scripted_worlds,
        frozen=args.frozen_worlds,
        mixed=args.mixed_worlds,
        product=args.product_worlds,
        smoke=args.smoke,
        repo=REPO,
        smoke_key=args.smoke_key,
    )
    manifest = _manifest(args, plan)
    _write_new(args.out / "manifest.json", manifest)
    _write_new(args.out / "seed-audit.json", plan["seed_audit"])
    return args.out


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _guard()
    from research.t34_pathb.analyze import _verify_identity

    manifest = json.loads((args.out / "manifest.json").read_text())
    _verify_identity(manifest)
    for name in ("horizon", "workers", "max_hours", "smoke"):
        supplied = getattr(args, name)
        registered = manifest["settings"][name]
        if supplied is not None and supplied != registered:
            raise RuntimeError(f"CLI {name} differs from the frozen manifest")
        setattr(args, name, registered)
    if args.workers != 1:
        raise RuntimeError("one assistance worker only")
    if not args.smoke:
        _power_guard()
    return manifest


def _arm_session(seed: int, fps: int, mode: str, smoke: bool) -> tuple[Any, Any]:
    from research.t15_mixed_roster_20261017.episodes import new_session
    from web.backend.pathb import install

    # An inherited experimental flag must not affect the bare comparator.
    with _flag_off():
        session = new_session(seed, 6)
    session.set_speed(fps)
    obj = install(session, mode=mode, smoke=smoke) if mode != "bare" else None
    return session, obj


@contextlib.contextmanager
def _flag_off() -> Iterator[None]:
    old = os.environ.pop("SNAKE_SERVE_PATHB", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["SNAKE_SERVE_PATHB"] = old


def _terminal_session(row: dict[str, Any], fps: int) -> Any:
    """Canonical live evaluator construction, including carried opponent policies."""
    from research.apex_veto_serving_20261001.serving_run import set_world_seed
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.scripts.tournament_eval import (
        _attach_agent,
        _evaluation_world_from_config,
        configure_eval_game_state,
        create_training_game_state,
    )
    from web.backend.served_checkpoint import resolve_served_checkpoint

    context = gid._context(1)  # Installs the hash-pinned deployment config.
    profile = context["profile"]
    set_world_seed(int(row["seed"]))
    game = create_training_game_state(eval_mode=False)
    if _evaluation_world_from_config().digest != profile.world.digest:
        raise RuntimeError("historical terminal profile differs from configured live world")
    choice = resolve_served_checkpoint({"SNAKE_SERVE_CHECKPOINT": "b8-s3"})
    cache: dict[str, Any] = {}
    _attach_agent(game, 0, ("checkpoint", str(choice.path)), row["seed"], cache, profile)
    for slot, spec in enumerate(row["roster"]["specs"], 1):
        _attach_agent(game, slot, tuple(spec), row["seed"], cache, profile)
    game._shared_policy = None
    configure_eval_game_state(game)
    game.snakes[0].auto_respawn = False
    session = SimpleNamespace(
        game=game,
        policy=game.snakes[0].policy,
        hero_id=game.snakes[0].id,
        speed=float(fps),
        pathb=None,
    )

    def step() -> None:
        game.update(
            train_mode=profile.runtime.training, learn=False, allow_respawn=profile.runtime.respawn
        )
        game.food_manager.trim_ambient(profile.world.max_food)
        if session.pathb is not None:
            session.pathb.after_step()

    session.step, session.snapshot = step, lambda: {}
    return session


def _check_cap(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("T34 wall cap reached; preserve partial evidence as INCOMPLETE")


def _metrics(
    session: Any, horizon: int, fps: int, obj: Any, log: Any, deadline: float
) -> dict[str, Any]:
    hero = session.game.snakes[0]
    masses, lengths = [], []
    deaths = decisions = boosts = large = late = overlap = 0
    original = hero.update
    pending_log: dict[str, Any] | None = None

    def capture(value: dict[str, Any]) -> None:
        nonlocal pending_log
        pending_log = dict(value)

    if obj is not None:
        obj.log_sink = capture
        obj.cap_deadline = deadline

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal decisions, boosts, pending_log
        decisions += 1
        value = original(*args, **kwargs)
        action = int(hero._pre_collision_action)
        boosts += int(action >= 3)
        entry = pending_log or {
            "hero_id": int(hero.id),
            "frame": int(session.game.frame),
            "requested": False,
            "applied": False,
            "stale_discarded": False,
            "deadline_miss": False,
            "fallback_reason": "bare_comparator",
            "bare_action": action,
            "proposed_action": action,
            "executed_action": action,
        }
        log.write(json.dumps(entry, sort_keys=True) + "\n")
        pending_log = None
        return value

    counted._pathb_measurement = True
    hero.update = counted
    started, cpu_before = time.monotonic(), time.process_time()
    try:
        for index in range(horizon):
            _check_cap(deadline)
            session.step()
            session.snapshot()
            alive = bool(hero.is_alive)
            masses.append(int(hero.length) if alive else 0)
            lengths.append(int(hero.length) if alive else 0)
            died = int(hero.id) in session.game.frame_death_causes
            deaths += int(died)
            fatal_length = int(hero.length)
            large += int(died and fatal_length >= 800)
            late += int(died and index + 1 >= 2500)
            overlap += int(died and fatal_length >= 800 and index + 1 >= 2500)
            _check_cap(deadline)
            time.sleep(min(1.0 / fps, max(0, deadline - time.monotonic())))
    finally:
        hero.update = original
    if decisions < 1:
        raise RuntimeError("world had no actual hero decisions")
    ordered = sorted(lengths)
    return {
        "mass_integral": sum(masses) / horizon,
        "deaths": deaths,
        "decisions": decisions,
        "survival_fraction": sum(value > 0 for value in masses) / horizon,
        "no_death": int(deaths == 0),
        "deaths_per_1k": deaths * 1000.0 / decisions,
        "median_length": ordered[len(ordered) // 2],
        "p95_length": ordered[int(0.95 * (len(ordered) - 1))],
        "deaths_length_ge_800": large,
        "late_deaths": late,
        "late_large_overlap": overlap,
        "boost_decisions": boosts,
        "boost_share": boosts / decisions,
        "elapsed_seconds": time.monotonic() - started,
        "parent_cpu_seconds": time.process_time() - cpu_before,
        "controller": None if obj is None else obj.state(),
    }


def _record(
    root: Path,
    block: int,
    row: dict[str, Any],
    kind: str,
    fps: int,
    args: argparse.Namespace,
    deadline: float,
) -> None:
    from research.t15_mixed_roster_20261017.episodes import roster_identity
    from web.backend.pathb import install

    name = f"{kind}-b{block:02d}-{row['mix']}-{row['seed']}-{fps}"
    arms, logs = {}, {}
    for mode in ARMS:
        _check_cap(deadline)
        session = obj = None
        log_path = root / "logs" / f"{name}-{mode}.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        try:
            if kind == "terminal":
                session = _terminal_session(row, fps)
                obj = None if mode == "bare" else install(session, mode=mode, smoke=args.smoke)
                identity = row["roster"]
            else:
                session, obj = _arm_session(row["seed"], fps, mode, args.smoke)
                identity = roster_identity(session)
            with log_path.open("x") as handle:
                arms[mode] = _metrics(session, args.horizon, fps, obj, handle, deadline)
            logs[mode] = {
                "path": str(log_path.relative_to(root)),
                "sha256": sha(log_path),
                "rows": arms[mode]["decisions"],
            }
        finally:
            if obj is not None:
                obj.close(wait=True)
    _write_new(
        root / "records" / f"{name}.json",
        {
            "schema": f"t34-pathb-{kind}/v1",
            "status": "complete",
            "kind": kind,
            "block": block,
            "seed": row["seed"],
            "mix": row["mix"],
            "fps": fps,
            "arms": arms,
            "roster": identity,
            "decision_logs": logs,
            "manifest_sha256": sha(root / "manifest.json"),
        },
    )


def efficacy(args: argparse.Namespace) -> None:
    manifest = _preflight(args)
    _write_new(
        args.out / "start.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha(args.out / "manifest.json"),
        },
    )
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    try:
        for block in manifest["bank"]["blocks"]:
            for kind in ("terminal", "product"):
                for row in block[kind]:
                    for fps in FPS:
                        _record(args.out, block["block"], row, kind, fps, args, deadline)
                        print(
                            json.dumps(
                                {
                                    "progress": "world_complete",
                                    "kind": kind,
                                    "block": block["block"],
                                    "fps": fps,
                                }
                            ),
                            flush=True,
                        )
        _write_new(
            args.out / "complete.json",
            {
                "status": "complete",
                "manifest_sha256": sha(args.out / "manifest.json"),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    except Exception as exc:
        _write_new(
            args.out / "incomplete.json",
            {"status": "INCOMPLETE", "reason": f"{type(exc).__name__}: {exc}"},
        )
        raise


def timing(args: argparse.Namespace) -> None:
    from research.t15_mixed_roster_20261017.measurements import _timed_tick

    manifest = _preflight(args)
    settings = manifest["timing_settings"]
    for key in ("warmup", "ticks"):
        supplied = getattr(args, key)
        if supplied is not None and supplied != settings[key]:
            raise RuntimeError(f"timing {key} differs from frozen manifest")
        setattr(args, key, settings[key])
    seed = manifest["bank"]["blocks"][0]["product"][0]["seed"]
    deadline = time.monotonic() + args.max_hours * 3600
    for mode in ("bare", "async"):
        for fps in FPS:
            session, obj = _arm_session(seed, fps, mode, args.smoke)
            samples, overruns = [], 0
            try:
                for _ in range(args.warmup):
                    _check_cap(deadline)
                    _timed_tick(session)
                    time.sleep(1.0 / fps)
                started = time.monotonic()
                for _ in range(args.ticks):
                    _check_cap(deadline)
                    length = int(session.game.snakes[0].length)
                    busy = bool(obj and obj.controller.computing)
                    update_ms, tick_ms = _timed_tick(session)
                    samples.append(
                        {
                            "update_ms": update_ms,
                            "tick_ms": tick_ms,
                            "hero_length": length,
                            "worker_busy": busy,
                        }
                    )
                    overruns += int(tick_ms >= 1000.0 / fps)
                    time.sleep(1.0 / fps)
                elapsed = time.monotonic() - started
            finally:
                if obj is not None:
                    obj.close(wait=True)
            _write_new(
                args.out / "timing" / f"{mode}-{fps}.json",
                {
                    "status": "complete",
                    "mode": mode,
                    "fps": fps,
                    "samples": samples,
                    "ticks": len(samples),
                    "warmup": args.warmup,
                    "elapsed_seconds": elapsed,
                    "actual_speed": session.speed,
                    "speed_ceiling": session.speed_ceiling,
                    "overruns": overruns,
                    "client_latency": "NOT_MEASURED",
                    "manifest_sha256": sha(args.out / "manifest.json"),
                    "controller": None if obj is None else obj.state(),
                },
            )


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    worlds = args.blocks * (
        args.scripted_worlds + args.frozen_worlds + args.mixed_worlds + args.product_worlds
    )
    paced_world_hours = worlds * args.horizon * sum(1 / fps for fps in FPS) / 3600
    value = {
        "worlds": worlds,
        "paced_wall_hours_lower_bound": 3 * paced_world_hours,
        "reserved_math_thread_hours_lower_bound": 4 * paced_world_hours,
        "active_cpu_core_hours": "NOT_MEASURED",
        "limitations": "sleep-only lower bound; add setup, clone, tick and unlimited-search work",
    }
    print(json.dumps(value))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bank", "efficacy", "timing", "smoke", "estimate"))
    parser.add_argument("--out", type=Path, required=True)
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--smoke-key", default="owner-smoke-01")
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error("--out must be absolute")
    if args.action in {"bank", "smoke", "estimate"}:
        for key, value in DEFAULTS.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        if args.action == "smoke":
            args.smoke = True
            for key, value in json.loads((HERE / "smoke.json").read_text()).items():
                setattr(args, key, value)
        args.smoke = bool(args.smoke)
    if args.action == "estimate":
        estimate(args)
    elif args.action == "bank":
        create_bank(args)
    elif args.action == "smoke":
        create_bank(args)
        efficacy(args)
        timing(args)
        from research.t34_pathb.analyze import analyze

        _write_new(args.out / "analysis.json", analyze(args.out))
        print(json.dumps({"smoke": "complete", "artifact_root": str(args.out)}))
    elif args.action == "efficacy":
        efficacy(args)
    else:
        timing(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
