#!/usr/bin/env python3
"""Create-only, resumable paired runner for the Phase-0 search-teacher study.

This module deliberately has no allocation, training, serving, or remote-control code.
It turns a frozen bank into one immutable result JSON per world.  A later operator may
use the same worker protocol on Linux, but the Mac guard and thread cap are enforced here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.compute.thermal_guard import ThermalGuard  # noqa: E402
from research.phase0_search_20261019 import bank as phase0_bank  # noqa: E402

STUDY = "phase0-search-teacher/v1"
RESULT_VERSION = 1
MAC_MAX_WORKERS = 6
POD_MAX_WORKERS = 32
CHECK_INTERVAL_SECONDS = 120.0
DEFAULT_WALL_SECONDS = 4 * 60 * 60
EXPECTED_B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
PREREG = Path("research/phase0_search_20261019/PREREGISTRATION.md")
BOUND_SOURCE_ROOTS = (
    Path("src"),
    Path("research/phase0_search_20261019"),
    Path("research/t3_rollout_labels_20261015"),
    Path("research/b8_death_census_20261014"),
    Path("research/redesign_scope_20261007"),
    Path("research/apex_safety_20260926"),
    Path("research/compute"),
)


class RunError(RuntimeError):
    """A fail-closed study execution error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RunError(f"expected JSON object: {path}")
    return value


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically write canonical JSON once; a valid existing result is never replaced."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"create-only evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"create-only evidence already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def source_closure() -> dict[str, str]:
    paths = {PREREG}
    for root in BOUND_SOURCE_ROOTS:
        absolute = REPO / root
        if not absolute.is_dir():
            raise RunError(f"source closure root is missing: {root}")
        paths.update(path.relative_to(REPO) for path in absolute.rglob("*.py") if path.is_file())
    return {str(path): sha256_file(REPO / path) for path in sorted(paths)}


def _git_head() -> str:
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return done.stdout.strip()


def on_ac_power() -> bool:
    if sys.platform != "darwin":
        return True
    try:
        done = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "AC Power" in done.stdout


def lid_open() -> bool:
    if sys.platform != "darwin":
        return True
    try:
        done = subprocess.run(
            ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "AppleClamshellState" in done.stdout and "= No" in done.stdout


def require_mac_admission() -> None:
    if sys.platform == "darwin" and not on_ac_power():
        raise RunError("refusing local run on battery")
    if sys.platform == "darwin" and not lid_open():
        raise RunError("refusing local run with lid closed or unreadable")


def operational_guard() -> None:
    """Fail closed before a local preflight or run admits simulator work."""
    require_mac_admission()
    status = ThermalGuard(ac_reader=on_ac_power).check()
    if not status["ok"]:
        raise RunError("thermal: " + ", ".join(status["reasons"]))


def worker_limit(platform: str, requested: int) -> int:
    if platform not in {"mac", "cpu-pod"}:
        raise RunError("platform must be mac or cpu-pod")
    ceiling = (
        MAC_MAX_WORKERS
        if sys.platform == "darwin"
        else (POD_MAX_WORKERS if platform == "cpu-pod" else MAC_MAX_WORKERS)
    )
    if requested < 1 or requested > ceiling:
        raise RunError(f"{platform} worker count must be in 1..{ceiling}")
    return requested


def _prereg_is_approved(prereg: Path) -> bool:
    text = prereg.read_text()
    return "Status: **APPROVED" in text or "Status: APPROVED" in text


def _bank_worlds(bank: Mapping[str, Any], *, kind: str = "selection") -> list[dict[str, Any]]:
    worlds = bank.get("worlds")
    if worlds is None:
        worlds = (
            bank.get("selection") if kind == "selection" else (bank.get("fixtures") or {}).get(kind)
        )
    if not isinstance(worlds, list) or not worlds:
        raise RunError("bank must contain a non-empty worlds list")
    result = []
    seen: set[str] = set()
    for row in worlds:
        if not isinstance(row, dict):
            raise RunError("bank world must be an object")
        world_id = str(row.get("world_id", ""))
        if not world_id or world_id in seen:
            raise RunError("bank has missing or duplicate world_id")
        if "world_seed" not in row or "mix" not in row or "opponents" not in row:
            raise RunError(f"world {world_id} is missing required identity fields")
        seen.add(world_id)
        result.append(dict(row))
    return result


def prepare_plan(
    *,
    bank_path: Path,
    audit_path: Path,
    config_path: Path,
    out: Path,
    kind: str = "selection",
    input_root: Path | None = None,
) -> dict[str, Any]:
    """Bind inputs into one immutable plan before any world can be admitted."""
    bank, audit, config = read_json(bank_path), read_json(audit_path), read_json(config_path)
    if kind not in {"selection", "smoke", "parity"}:
        raise RunError("plan kind must be selection, smoke, or parity")
    worlds = _bank_worlds(bank, kind=kind)
    if audit.get("status") not in {"PASS", "pass"} and audit.get("pass") is not True:
        raise RunError("world-bank audit is not PASS")
    prereg = REPO / PREREG
    if kind == "selection" and not _prereg_is_approved(prereg):
        raise RunError("selection is refused while the pre-registration is draft")
    expected_alloc = phase0_bank.digest(
        {"selection": bank.get("selection"), "fixtures": bank.get("fixtures")}
    )
    if (
        bank.get("allocation_semantic_sha256") != expected_alloc
        or audit.get("world_bank_semantic_sha256") != expected_alloc
    ):
        raise RunError("bank/audit semantic binding mismatch")
    if kind == "selection":
        counts = {mix: sum(row["mix"] == mix for row in worlds) for mix in phase0_bank.MIX_COUNTS}
        if (
            len(worlds) != 160
            or counts != phase0_bank.MIX_COUNTS
            or any(int(row.get("horizon", 0)) != 5000 for row in worlds)
        ):
            raise RunError("selection bank does not match registered 96/32/32 H5000 design")
    execution_horizon = (
        int(config.get("fixture_horizon", config.get("smoke_horizon", 32)))
        if kind != "selection"
        else None
    )
    if kind != "selection" and not 1 <= execution_horizon <= 32:
        raise RunError("fixture horizon must be in 1..32")
    checkpoint = Path(str(config.get("checkpoint", ""))).expanduser()
    if not checkpoint.is_file() and input_root is not None:
        checkpoint = Path(input_root) / checkpoint.name
    if not checkpoint.is_file():
        raise RunError("configured B8 checkpoint is missing")
    checkpoint_sha = sha256_file(checkpoint)
    expected_checkpoint_sha = (
        str(config.get("checkpoint_sha256", "")) if kind != "selection" else EXPECTED_B8_SHA256
    )
    if kind != "selection" and not expected_checkpoint_sha:
        raise RunError("fixture config must explicitly bind its checkpoint SHA-256")
    if checkpoint_sha != expected_checkpoint_sha:
        raise RunError("configured B8 checkpoint digest does not match the registered pin")
    plan = {
        "version": RESULT_VERSION,
        "study": STUDY,
        "kind": kind,
        "created_at_unix": time.time(),
        "git_head": _git_head(),
        "world_count": len(worlds),
        "execution_horizon": execution_horizon,
        "world_ids": [row["world_id"] for row in worlds],
        "input_sha256": {
            "bank": sha256_file(bank_path),
            "audit": sha256_file(audit_path),
            "config": sha256_file(config_path),
            "preregistration": sha256_file(prereg),
            "checkpoint": checkpoint_sha,
        },
        "input_paths": {
            "bank": str(Path(bank_path).resolve()),
            "audit": str(Path(audit_path).resolve()),
            "config": str(Path(config_path).resolve()),
            "preregistration": str(prereg.resolve()),
            "checkpoint": str(checkpoint.resolve()),
        },
        "config": config,
        "source_closure_sha256": source_closure(),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    write_create_only(out, plan)
    return plan


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    claimed = plan.pop("plan_sha256", None)
    if claimed != canonical_sha256(plan):
        raise RunError("plan canonical digest mismatches")
    plan["plan_sha256"] = claimed
    if plan.get("study") != STUDY or plan.get("version") != RESULT_VERSION:
        raise RunError("not a Phase-0 runner plan")
    if plan.get("git_head") != _git_head():
        raise RunError("git commit differs from the immutable plan")
    for label, raw in plan.get("input_paths", {}).items():
        if label == "checkpoint" and not Path(raw).is_file() and os.environ.get("SNAKE_CKPT_DIR"):
            raw = str(Path(os.environ["SNAKE_CKPT_DIR"]) / Path(raw).name)
            plan["input_paths"][label] = raw
        expected = plan.get("input_sha256", {}).get(label)
        if not expected or not Path(raw).is_file() or sha256_file(Path(raw)) != expected:
            raise RunError(f"bound input drift or absence: {label}")
    if source_closure() != plan.get("source_closure_sha256"):
        raise RunError("source closure drift")
    if plan["kind"] == "selection" and not _prereg_is_approved(
        Path(plan["input_paths"]["preregistration"])
    ):
        raise RunError("selection remains blocked pending pre-registration approval")
    return plan


def run_world(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    trace: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run B8 and search from freshly built, identical world rows; no output is written here."""
    from research.phase0_search_20261019 import teacher

    checkpoint = Path(plan["input_paths"]["checkpoint"])
    config = teacher.SearchConfig(**dict(plan["config"].get("search", plan["config"])))
    search = teacher.SearchTeacher(config)
    teacher.configure_cpu()
    if trace is not None:
        trace({"phase": "before_b8_world_build"})
    baseline_world = teacher.build_world(dict(row), checkpoint)
    if trace is not None:
        trace({"phase": "before_search_world_build"})
    search_world = teacher.build_world(dict(row), checkpoint)

    def world_digest(world: Any) -> str:
        from research.redesign_scope_20261007.grid_h5000_identity import env_digest

        return hashlib.sha256(env_digest(world.sim, 0).encode()).hexdigest()

    start_a, start_b = world_digest(baseline_world), world_digest(search_world)
    if start_a != start_b:
        raise RunError("paired arms did not begin from identical state")
    horizon = int(plan.get("execution_horizon") or row.get("horizon", 5000))

    def episode(world: Any, policy: Any, arm: str) -> dict[str, Any]:
        mass, actions = 0.0, []
        alive = bool(world.sim.alive[0, 0])
        executed = 0
        for _ in range(horizon):
            if trace is not None:
                trace({"phase": "before_decision", "arm": arm, "frame": executed})
            pre_digest = world_digest(world)

            def trace_step(diag: Mapping[str, Any]) -> None:
                item = {
                    "frame": int(diag["frame"]),
                    "action": int(diag["action"]),
                    "b8_action": int(diag["b8_action"]),
                    "mask": [bool(x) for x in diag["mask"]],
                    "selection_mask": [bool(x) for x in diag["selection_mask"]],
                    "q": [float(x) for x in diag["q"]],
                    "searched": bool(diag["searched"]),
                    "pre_world_digest": pre_digest,
                    "post_world_digest": world_digest(world),
                }
                if trace is not None:
                    trace({"arm": arm, **item})
                actions.append(item)

            teacher.step_world(world, teacher=policy, trace=trace_step)
            executed += 1
            alive = bool(world.sim.alive[0, 0])
            if alive:
                mass += float(world.sim.length[0, 0])
            else:
                break
        trace_digest = canonical_sha256(actions)
        return {
            "mass_integral": mass / horizon,
            "mass_sum": mass,
            "survived": alive,
            "frames_scored": horizon,
            "frames_executed": executed,
            "start_world_digest": start_a,
            "final_world_digest": world_digest(world),
            "trace": actions,
            "trace_sha256": trace_digest,
        }

    baseline = episode(baseline_world, None, "b8")
    candidate = episode(search_world, search, "search")
    return {
        "version": RESULT_VERSION,
        "study": STUDY,
        "world_id": str(row["world_id"]),
        "world_seed": int(row["world_seed"]),
        "mix": str(row["mix"]),
        "bank_sha256": plan["input_sha256"]["bank"],
        "world_digest": str(row.get("world_semantic_sha256", "")),
        "world_identity": {
            key: row[key]
            for key in ("opponents", "horizon", "world_semantic_sha256", "opponent_semantic_sha256")
            if key in row
        },
        "arms": {"b8": baseline, "search": candidate},
        "paired": {
            "mass_delta": candidate["mass_integral"] - baseline["mass_integral"],
            "survival_delta": int(candidate["survived"]) - int(baseline["survived"]),
        },
        "complete": True,
    }


def _semantic_result(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"elapsed_seconds", "platform", "result_sha256"}
    }


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _valid_existing(path: Path, plan: Mapping[str, Any], world: Mapping[str, Any]) -> bool:
    try:
        row = read_json(path)
        expected_identity = {
            key: world[key]
            for key in ("opponents", "horizon", "world_semantic_sha256", "opponent_semantic_sha256")
            if key in world
        }
        if not (
            bool(row.get("complete"))
            and row.get("plan_sha256") == plan["plan_sha256"]
            and row.get("world_id") == world["world_id"]
            and row.get("world_identity") == expected_identity
            and row.get("world_seed") == world["world_seed"]
            and row.get("mix") == world["mix"]
            and row.get("bank_sha256") == plan.get("input_sha256", {}).get("bank")
            and row.get("world_digest") == world.get("world_semantic_sha256", "")
        ):
            return False
        if row.get("result_sha256") != canonical_sha256(_semantic_result(row)):
            return False
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"b8", "search"}:
            return False
        horizon = int(plan.get("execution_horizon") or world["horizon"])
        starts: set[str] = set()
        for arm in ("b8", "search"):
            evidence = arms[arm]
            if not isinstance(evidence, Mapping):
                return False
            mass, integral = evidence.get("mass_sum"), evidence.get("mass_integral")
            if not isinstance(mass, (int, float)) or not isinstance(integral, (int, float)):
                return False
            if (
                not math.isfinite(float(mass))
                or not math.isfinite(float(integral))
                or min(float(mass), float(integral)) < 0
            ):
                return False
            if not math.isclose(float(integral), float(mass) / horizon, rel_tol=0.0, abs_tol=1e-12):
                return False
            executed = int(evidence.get("frames_executed", -1))
            if evidence.get("frames_scored") != horizon or not isinstance(
                evidence.get("survived"), bool
            ):
                return False
            if not 1 <= executed <= horizon or (evidence["survived"] and executed != horizon):
                return False
            trace = evidence.get("trace")
            if (
                not isinstance(trace, list)
                or len(trace) != executed
                or evidence.get("trace_sha256") != canonical_sha256(trace)
            ):
                return False
            if not _sha256_text(evidence.get("start_world_digest")) or not _sha256_text(
                evidence.get("final_world_digest")
            ):
                return False
            starts.add(evidence["start_world_digest"])
            previous = None
            for item in trace:
                if not isinstance(item, Mapping) or not isinstance(item.get("frame"), int):
                    return False
                if previous is not None and item["frame"] <= previous:
                    return False
                previous = item["frame"]
                if not isinstance(item.get("action"), int) or not 0 <= item["action"] < 6:
                    return False
                if not isinstance(item.get("b8_action"), int) or not 0 <= item["b8_action"] < 6:
                    return False
                if not isinstance(item.get("searched"), bool) or len(item.get("q", [])) != 6:
                    return False
                if len(item.get("mask", [])) != 6 or len(item.get("selection_mask", [])) != 6:
                    return False
                if not _sha256_text(item.get("pre_world_digest")) or not _sha256_text(
                    item.get("post_world_digest")
                ):
                    return False
        if len(starts) != 1:
            return False
        return True
    except (OSError, ValueError, RunError, KeyError, TypeError):
        return False


def _run_one_task(
    task: tuple[dict[str, Any], dict[str, Any], str, str, float],
) -> tuple[str, str | None]:
    """Process-pool unit: one paired world, its own periodic Mac guard and one receipt."""
    plan, row, destination_text, platform, deadline = task
    destination = Path(destination_text)
    if time.monotonic() >= deadline:
        raise RunError("wall_cap")
    guard = ThermalGuard(ac_reader=on_ac_power if sys.platform == "darwin" else None)
    last_check = time.monotonic()
    if sys.platform == "darwin":
        operational_guard()

    def trace(_event: Mapping[str, Any]) -> None:
        nonlocal last_check
        if time.monotonic() >= deadline:
            raise RunError("wall_cap")
        if sys.platform == "darwin" and time.monotonic() - last_check >= CHECK_INTERVAL_SECONDS:
            operational_guard()
            last_check = time.monotonic()

    before = time.monotonic()
    result = run_world(row, plan, trace=trace)
    elapsed = time.monotonic() - before
    guard.record_episode(str(row["mix"]), elapsed)
    result.update(
        {"plan_sha256": plan["plan_sha256"], "platform": platform, "elapsed_seconds": elapsed}
    )
    result["result_sha256"] = canonical_sha256(_semantic_result(result))
    write_create_only(destination, result)
    return str(row["world_id"]), None


def run_plan(
    plan_path: Path,
    out: Path,
    *,
    workers: int = 1,
    platform: str = "mac",
    wall_seconds: float = DEFAULT_WALL_SECONDS,
) -> dict[str, Any]:
    """Execute unfinished worlds in fixed bank order.

    Parallel dispatch belongs to a future launcher.

    Keeping this worker serial gives an identity reference and respects the
    one-math-thread rule;
    external process fan-out may invoke disjoint ``--world-id`` units later.
    """
    plan = validate_plan(plan_path)
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise RunError("wall_seconds must be finite and positive")
    worker_limit(platform, workers)
    if sys.platform == "darwin":
        operational_guard()
    bank = read_json(Path(plan["input_paths"]["bank"]))
    worlds = _bank_worlds(bank, kind=plan["kind"])
    if [row["world_id"] for row in worlds] != plan["world_ids"]:
        raise RunError("bank world order drift")
    start, completed, stopped = time.monotonic(), [], None
    out = Path(out)
    pending = []
    for row in worlds:
        destination = out / "worlds" / f"{row['world_id']}.json"
        if _valid_existing(destination, plan, row):
            completed.append(str(row["world_id"]))
        else:
            pending.append(
                (dict(plan), dict(row), str(destination), platform, start + wall_seconds)
            )
    if pending:
        try:
            if workers == 1:
                for task in pending:
                    if time.monotonic() >= start + wall_seconds:
                        raise RunError("wall_cap")
                    completed.append(_run_one_task(task)[0])
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                    iterator = iter(pending)
                    active = {pool.submit(_run_one_task, task) for task in list(pending)[:workers]}
                    for _ in range(min(workers, len(pending))):
                        next(iterator)
                    while active:
                        seconds_left = start + wall_seconds - time.monotonic()
                        if seconds_left <= 0:
                            raise RunError("wall_cap")
                        done, active = concurrent.futures.wait(
                            active,
                            timeout=seconds_left,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        if not done:
                            raise RunError("wall_cap")
                        for future in done:
                            completed.append(future.result()[0])
                            if time.monotonic() < start + wall_seconds:
                                try:
                                    active.add(pool.submit(_run_one_task, next(iterator)))
                                except StopIteration:
                                    pass
        except (RunError, OSError, ValueError, concurrent.futures.process.BrokenProcessPool) as exc:
            stopped = str(exc)
    return {
        "complete": stopped is None and len(completed) == len(worlds),
        "completed_world_ids": completed,
        "stop_reason": stopped,
    }


def merge_runs(plan_path: Path, roots: Iterable[Path], out: Path) -> dict[str, Any]:
    """Merge only a complete, identity-valid bank in frozen bank order."""
    plan = validate_plan(plan_path)
    bank = read_json(Path(plan["input_paths"]["bank"]))
    rows = []
    for world in _bank_worlds(bank, kind=plan["kind"]):
        candidates = [Path(root) / "worlds" / f"{world['world_id']}.json" for root in roots]
        found = [path for path in candidates if path.is_file()]
        if len(found) != 1:
            raise RunError(f"world {world['world_id']} must appear exactly once across shards")
        result = read_json(found[0])
        if not _valid_existing(found[0], plan, world):
            raise RunError(f"invalid or incomplete result: {found[0]}")
        rows.append(_semantic_result(result))
    merged = {
        "schema": "phase0-search-teacher-analysis-input/v1",
        "version": RESULT_VERSION,
        "study": STUDY,
        "plan_sha256": plan["plan_sha256"],
        "plan_binding": {
            "kind": plan["kind"],
            "config": plan["config"],
            "input_sha256": plan["input_sha256"],
            "source_closure_sha256": plan["source_closure_sha256"],
        },
        "bank_file_sha256": plan["input_sha256"]["bank"],
        "bank": bank,
        "bank_audit": read_json(Path(plan["input_paths"]["audit"])),
        "complete": True,
        "worlds": rows,
    }
    merged["merged_sha256"] = canonical_sha256(merged)
    write_create_only(out, merged)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--parity", action="store_true")
    p.add_argument("--input-root", type=Path)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--platform", default="mac")
    p.add_argument("--wall-seconds", type=float, default=DEFAULT_WALL_SECONDS)
    p = sub.add_parser("merge")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_plan(
                bank_path=args.bank,
                audit_path=args.audit,
                config_path=args.config,
                out=args.out,
                kind="parity" if args.parity else "smoke" if args.smoke else "selection",
                input_root=args.input_root,
            )
        elif args.command == "run":
            result = run_plan(
                args.plan,
                args.out,
                workers=args.workers,
                platform=args.platform,
                wall_seconds=args.wall_seconds,
            )
        else:
            result = merge_runs(args.plan, args.root, args.out)
    except (RunError, FileExistsError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
