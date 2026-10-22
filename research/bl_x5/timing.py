"""Price bare banks with the T7 evaluator, without opening a strict family.

Only this module's fresh development outputs are written. There are no signals,
Colab commands, serving changes, scientific decisions, or automatic retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from research.mac_lease import DEFAULT_LEASE_PATH, held

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
DEFAULT_ROOT = ARTIFACTS / "big-lever/exp/week1-20261021/mac"
B8 = REPO.parent / "snake-dqn/saved_snakes/b8_s3_t7_20261016.pth"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
MIXES = ("scripted", "frozen", "mixed")
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def command(argv: list[str]) -> str:
    return subprocess.run(argv, cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path: Path, payload: dict[str, Any]) -> str:
    """Publish JSON followed by its hash and completion marker, all create-only."""
    from research.sequential_phase_r.receipts import write_once

    if any(p.exists() for p in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))):
        raise FileExistsError(path)
    digest = write_once(path, payload)
    for suffix, value in ((".sha256", f"{digest}  {path.name}\n"), (".complete", digest + "\n")):
        with Path(str(path) + suffix).open("x") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    return digest


def read_sealed(path: Path) -> dict[str, Any]:
    digest = sha(path)
    if Path(str(path) + ".sha256").read_text() != f"{digest}  {path.name}\n":
        raise ValueError(f"hash mismatch: {path}")
    if Path(str(path) + ".complete").read_text() != digest + "\n":
        raise ValueError(f"incomplete: {path}")
    return json.loads(path.read_text())


def host() -> dict[str, Any]:
    from research.compute.thermal_guard import parse_pmset_therm, thermal_reasons

    batt = command(["pmset", "-g", "batt"])
    lid = command(["ioreg", "-r", "-k", "AppleClamshellState"])
    therm = command(["pmset", "-g", "therm"])
    parsed = parse_pmset_therm(therm)
    reasons = thermal_reasons(parsed)
    if "AC Power" not in batt:
        reasons.append("AC_required")
    if '"AppleClamshellState" = No' not in lid:
        reasons.append("lid_open_required")
    processes = command(["ps", "-axo", "pid,ppid,pcpu,comm"])
    foreign = []
    owned_parents = {os.getpid(), os.getppid()}
    for line in processes.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        pid, ppid, cpu, executable = parts
        if (
            Path(executable).name.lower().startswith("python")
            and int(pid) not in owned_parents
            and int(ppid) not in owned_parents
            and float(cpu) >= 25
        ):
            foreign.append(line.strip())
    if foreign:
        reasons.append("foreign_python_cpu_lane")
    return {
        "time_pt": now(),
        "loadavg": os.getloadavg(),
        "ac_raw": batt,
        "lid_open": '"AppleClamshellState" = No' in lid,
        "thermal_raw": therm,
        "thermal": parsed,
        "reasons": reasons,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "top": command(["top", "-l", "1", "-n", "0"]),
        "processes": processes,
        "foreign_python_cpu_lanes": foreign,
    }


def sources() -> dict[str, str]:
    files = command(["git", "ls-files"]).splitlines()
    return {
        p: sha(REPO / p)
        for p in files
        if p.endswith((".py", ".yaml", ".yml")) and (REPO / p).is_file()
    }


def audit_worlds(worlds: list[int], root: Path) -> dict[str, Any]:
    """Scan visible text inventories, explicitly excluding the sealed study."""
    from research.redesign_t7_phase_r_20261016.seed_audit import _registered

    registered = _registered(REPO)
    historical = sorted(ARTIFACTS.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(historical) < 7:
        raise ValueError("historical training exclusions missing")
    inventory = {}
    for path in historical:
        registered.update(json.loads(path.read_text()))
        inventory[str(path)] = sha(path)
    collisions = sorted(set(worlds) & registered)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".x5-seeds") as patterns:
        patterns.write("\n".join(map(str, worlds)) + "\n")
        patterns.flush()
        argv = [
            "rg",
            "--hidden",
            "--no-ignore",
            "--fixed-strings",
            "--word-regexp",
            "--files-with-matches",
            "--glob",
            "*.json",
            "--glob",
            "*.jsonl",
            "--glob",
            "*.md",
            "--glob",
            "!**/t34-round0-study-01/**",
            "--glob",
            f"!**/{root.relative_to(ARTIFACTS).as_posix()}/**",
            "-f",
            patterns.name,
            str(REPO / "research"),
            str(ARTIFACTS),
        ]
        result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"freshness scan failed: {result.stderr}")
    matches = result.stdout.splitlines()
    return {
        "pass": not collisions and not matches and len(set(worlds)) == 48,
        "candidate_count": len(worlds),
        "registered_count": len(registered),
        "registered_collisions": collisions,
        "text_match_paths": matches,
        "historical_inventory_hashes": inventory,
        "scan_excluded": ["**/t34-round0-study-01/**", str(root)],
        "coverage_limit": (
            "visible JSON/JSONL/Markdown and generated historical banks; "
            "opaque binaries and sealed study excluded"
        ),
    }


def prepare(root: Path, allocation_key: str = "attempt-01") -> None:
    from research.apex_safety_20260926 import dev_screen as ds
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.evaluation.strict_promotion import _expected_world_identity

    if command(["git", "status", "--porcelain"]):
        raise ValueError("commit code before prepare")
    if platform.system() != "Darwin" or sha(B8) != B8_SHA:
        raise ValueError("requires this Mac and pinned B8 s3")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "x5-manifest.json").exists() or (root / "x5.json").exists():
        raise FileExistsError("X5 already prepared/executed; no automatic retry")
    idle = host()
    if idle["reasons"]:
        raise ValueError(idle["reasons"])
    seal(root / "x5-idle.json", idle)
    worlds = ds.screen_seeds(48, "bl-x5-development-timing/20261021/v1", allocation_key)
    audit = audit_worlds(worlds, root)
    seal(root / "x5-seed-audit.json", audit)
    if not audit["pass"]:
        raise ValueError("whole development allocation rejected; do not repair individual worlds")
    designs = {(r["mix"], r["world_seed"]): r for r in ds._design_rows(worlds)}
    banks = {mix: worlds[i * 16 : (i + 1) * 16] for i, mix in enumerate(MIXES)}
    rows = {
        mix: {
            str(w): {
                "design": designs[mix, w],
                "identity": _expected_world_identity(designs[mix, w]),
            }
            for w in bank
        }
        for mix, bank in banks.items()
    }
    checkpoints = {str(B8): B8_SHA}
    for name, digest in (*ds.POOL, gid.FRP3_S12):
        p = gid.CHECKPOINT_DIR / name
        if sha(p) != digest:
            raise ValueError(f"checkpoint drift: {p}")
        checkpoints[str(p)] = digest
    plan_root = ARTIFACTS / "big-lever"
    seal(
        root / "x5-manifest.json",
        {
            "schema": "bl-x5-manifest/v1",
            "created_pt": now(),
            "authority": "owner: run now; development timing only, no registered study/family",
            "schedule_override": "owner requested immediate run instead of Wed 19:00-21:00 PT",
            "builder": "Codex root / Sol-X5",
            "reviewer": "Codex-V / x5_review",
            "commit": command(["git", "rev-parse", "HEAD"]),
            "sources": sources(),
            "authority_hashes": {
                str(p): sha(p)
                for p in (
                    plan_root / "PLAN_v2.md",
                    plan_root / "round-11/JUDGMENT.md",
                    plan_root / "round-07/JUDGMENT.md",
                )
            },
            "banks": banks,
            "allocation_key": allocation_key,
            "rows": rows,
            "checkpoints": checkpoints,
            "horizon": 5000,
            "batch_size": 8,
            "workers": 3,
            "threads_per_worker": 1,
            "arms": ["B8-repeat-1", "B8-repeat-2"],
            "parent": "no F parent supplied; B8 twice per original X5 allowance",
            "world_count": 48,
            "episodes": 96,
            "paced": False,
            "wall_cap_seconds": 7200,
            "evaluation_cap_seconds": 7000,
            "worker_cpu_cap_seconds": 2300,
            "aggregate_worker_cpu_cap_seconds": 6900,
            "kill_rules": (
                "no efficacy kill; guard/deadline/STOP file cooperatively abort "
                "at frame boundary; no process signals/retry"
            ),
            "forecast_before_measurement": {
                "status": "UNPRICED",
                "wall_upper_bound_hours": 2,
                "cpu_upper_bound_hours": 2,
                "units": 0,
            },
            "upload_preflight": {
                "status": "NOT_APPLICABLE",
                "reason": "Mac-only; no transport/Colab manifest or allocation",
            },
            "idle_sha256": sha(root / "x5-idle.json"),
            "seed_audit_sha256": sha(root / "x5-seed-audit.json"),
        },
    )


def worker(root: Path, mix: str) -> int:
    """Run four fixed eight-world units; stop only through cooperative checks."""
    start_cpu, start_wall = time.process_time(), time.monotonic()
    plan = read_sealed(root / "x5-manifest.json")
    start = read_sealed(root / "x5-start.json")
    samples, units = [], []
    error = None
    last_check = 0.0

    def guard(_payload: Any = None) -> None:
        nonlocal last_check
        if (
            time.monotonic() >= start["deadline_monotonic"]
            or time.process_time() - start_cpu >= 2300
        ):
            raise RuntimeError("development timing cap reached")
        if (root / "STOP").exists():
            raise RuntimeError("operator STOP file")
        if time.monotonic() - last_check >= 30:
            snapshot = host()
            samples.append(snapshot)
            last_check = time.monotonic()
            if snapshot["reasons"]:
                raise RuntimeError(f"host guard: {snapshot['reasons']}")

    try:
        for p, digest in plan["sources"].items():
            if sha(REPO / p) != digest:
                raise ValueError(f"source changed: {p}")
        from research.redesign_scope_20261007 import grid_h5000_identity as gid
        from src.simd_env.eval_engine import run_simd_eval

        ctx = gid._context(1)
        for arm in plan["arms"]:
            for lo in (0, 8):
                guard()
                seeds = plan["banks"][mix][lo : lo + 8]
                rows = plan["rows"][mix]
                rosters = {
                    w: [ctx["lookup"][s["member_sha256"]] for s in rows[str(w)]["design"]["slots"]]
                    for w in seeds
                }
                identities = {w: rows[str(w)]["identity"] for w in seeds}
                cpu, wall = time.process_time(), time.monotonic()
                records = run_simd_eval(
                    ctx["lookup"][gid.FRP3_S12[1]],
                    rosters[seeds[0]],
                    5000,
                    seeds,
                    profile=ctx["profile"],
                    opponent_specs_by_world=rosters,
                    world_identities=identities,
                    mix_id=mix,
                    frame_observer=guard,
                    vector61=True,
                    vector61_forward="rowwise",
                    sim_engine="grid",
                    hero_ego2s=str(B8),
                    ego2s_forward="rowwise",
                    ego2s_mask="advisory",
                )
                unit = {
                    "mix": mix,
                    "arm": arm,
                    "offset": lo,
                    "worlds": seeds,
                    "cpu_seconds": time.process_time() - cpu,
                    "wall_seconds": time.monotonic() - wall,
                    "records": json.loads(json.dumps(records)),
                }
                name = f"x5-{mix}-{arm}-{lo}.json"
                seal(root / "units" / name, unit)
                units.append(
                    {
                        "path": str(root / "units" / name),
                        "sha256": sha(root / "units" / name),
                        **{k: v for k, v in unit.items() if k != "records"},
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "unit_complete",
                            "mix": mix,
                            "arm": arm,
                            "offset": lo,
                            "seconds": unit["wall_seconds"],
                            "episodes_per_hour": 8 * 3600 / unit["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    seal(
        root / f"x5-worker-{mix}.json",
        {
            "schema": "bl-x5-worker/v1",
            "mix": mix,
            "pid": os.getpid(),
            "manifest_sha256": sha(root / "x5-manifest.json"),
            "status": "COMPLETE" if len(units) == 4 and error is None else "INCOMPLETE",
            "units": units,
            "host_samples": samples,
            "error": error,
            "process_cpu_seconds": time.process_time() - start_cpu,
            "process_wall_seconds": time.monotonic() - start_wall,
        },
    )
    return 0 if error is None else 2


def forecasts(workers: list[dict[str, Any]]) -> dict[str, Any]:
    rates = {}
    for row in workers:
        units = row["units"]
        episodes = sum(len(u["worlds"]) for u in units)
        cpu = sum(u["cpu_seconds"] for u in units)
        wall = sum(u["wall_seconds"] for u in units)
        rates[row["mix"]] = {
            "episodes": episodes,
            "cpu_seconds": cpu,
            "cpu_seconds_per_episode": cpu / episodes,
            "episodes_per_cpu_hour": episodes * 3600 / cpu,
            "unit_wall_seconds": wall,
            "measured_episodes_per_worker_wall_hour": episodes * 3600 / wall,
            "including_worker_startup_cpu_seconds_per_episode": row["process_cpu_seconds"]
            / episodes,
        }
    out = {}
    for name, counts, arms in (
        ("bare-192-96-96", (192, 96, 96), 1),
        ("paired-192-96-96", (192, 96, 96), 2),
        ("bare-384-128-128", (384, 128, 128), 1),
        ("paired-384-128-128", (384, 128, 128), 2),
        ("e13-768-episodes", (128, 128, 128), 2),
    ):
        cpu_h = (
            sum(n * arms * rates[m]["cpu_seconds_per_episode"] for m, n in zip(MIXES, counts))
            / 3600
        )
        incl_h = (
            sum(
                n * arms * rates[m]["including_worker_startup_cpu_seconds_per_episode"]
                for m, n in zip(MIXES, counts)
            )
            / 3600
        )
        out[name] = {
            "worlds_per_mix": dict(zip(MIXES, counts)),
            "arms": arms,
            "episodes": sum(counts) * arms,
            "horizon": 5000,
            "projected_cpu_hours_from_measured_rate": cpu_h,
            "projected_cpu_hours_including_amortized_startup": incl_h,
            "ideal_wall_hours_at_measured_3_workers": cpu_h / 3,
            "ideal_wall_hours_at_T7_6_workers_UNMEASURED": cpu_h / 6,
            "basis": (
                "mix-weighted process CPU; B8 replacement actor cost, variance/power, "
                "controls/audit and scheduler scaling unmeasured"
            ),
        }
    return {"rates_by_mix": rates, "banks": out}


def validate_units(root: Path, plan: dict[str, Any], workers: list[dict[str, Any]]) -> None:
    """Re-read all sealed units and check the frozen identities before pricing."""
    if sorted(w["mix"] for w in workers) != sorted(MIXES):
        raise ValueError("worker mix coverage")
    manifest_sha = sha(root / "x5-manifest.json")
    for w in workers:
        mix = w["mix"]
        if w["manifest_sha256"] != manifest_sha or len(w["units"]) != 4:
            raise ValueError("worker manifest/units")
        seen = set()
        for ref in w["units"]:
            unit = read_sealed(Path(ref["path"]))
            if sha(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("unit hash")
            for key in ("mix", "arm", "offset", "worlds", "cpu_seconds", "wall_seconds"):
                if unit[key] != ref[key]:
                    raise ValueError("unit timing/identity differs")
            key = (unit["arm"], unit["offset"])
            if key in seen or key[0] not in plan["arms"] or key[1] not in (0, 8):
                raise ValueError("duplicate/unexpected unit")
            seen.add(key)
            expected = plan["banks"][mix][key[1] : key[1] + 8]
            if unit["worlds"] != expected or unit["mix"] != mix or len(unit["records"]) != 8:
                raise ValueError("world coverage")
            for seed, record in zip(expected, unit["records"]):
                if (
                    record["seed"] != seed
                    or record["world_identity"] != plan["rows"][mix][str(seed)]["identity"]
                ):
                    raise ValueError("record world identity")


def drain_owned(root: Path, procs: list[tuple[Any, Any, Any]]) -> None:
    """Retain locks until every started child exits; never send a signal."""
    if any(p.poll() is None for _, p, _ in procs) and not (root / "STOP").exists():
        with (root / "STOP").open("x") as stream:
            stream.write("coordinator cleanup: cooperative stop, no retry\n")
    for _, p, log in procs:
        p.wait()
        log.close()


def run(root: Path, *, lease_path: Path = DEFAULT_LEASE_PATH) -> int:
    from research.apex_safety_20260926 import dev_screen as ds

    plan = read_sealed(root / "x5-manifest.json")
    if (root / "x5-start.json").exists() or (root / "x5.json").exists():
        raise FileExistsError("attempt already started; no automatic retry")
    for p, digest in plan["checkpoints"].items():
        if sha(Path(p)) != digest:
            raise ValueError(f"checkpoint changed: {p}")
    with held("bl-x5-timing", "bl-x5 Mac timing", seconds=7200, path=lease_path) as lease:
        slots = ds.acquire_cpu_slots(ds.DEFAULT_SLOT_LOCK_ROOT, 3, 0, pool=3)
        procs = []
        try:
            snapshot = host()
            if snapshot["reasons"]:
                raise ValueError(snapshot["reasons"])
            seal(root / "x5-launch-host.json", snapshot)
            started = time.monotonic()
            seal(
                root / "x5-start.json",
                {
                    "time_pt": now(),
                    "deadline_epoch": time.time() + 7000,
                    "deadline_monotonic": time.monotonic() + 7000,
                    "manifest_sha256": sha(root / "x5-manifest.json"),
                    "pid": os.getpid(),
                    "mac_lease": lease.receipt,
                    "slots": [s.name for s in slots],
                    "tmux_inventory": command(["tmux", "ls"]),
                },
            )
            for mix in MIXES:
                log = (root / f"x5-{mix}.log").open("x")
                try:
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "research.bl_x5.timing",
                            "worker",
                            "--root",
                            str(root),
                            "--mix",
                            mix,
                        ],
                        cwd=REPO,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                procs.append((mix, p, log))
            while any(p.poll() is None for _, p, _ in procs):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "time_pt": now(),
                            "elapsed_seconds": time.monotonic() - started,
                            "active": [m for m, p, _ in procs if p.poll() is None],
                        }
                    ),
                    flush=True,
                )
                time.sleep(5)
            workers, errors = [], []
            for mix, p, log in procs:
                log.close()
                path = root / f"x5-worker-{mix}.json"
                if path.exists():
                    workers.append(read_sealed(path))
                if p.returncode != 0:
                    errors.append({"mix": mix, "exit": p.returncode})
            complete = (
                len(workers) == 3 and all(w["status"] == "COMPLETE" for w in workers) and not errors
            )
            if complete:
                validate_units(root, plan, workers)
            receipt = {
                "schema": "bl-x5-timing/v1",
                "status": "MEASURED" if complete else "INCOMPLETE",
                "completed_pt": now(),
                "development_only": True,
                "strict_family_opened": False,
                "serving_changed": False,
                "paced": False,
                "units": 0,
                "manifest": {
                    "path": str(root / "x5-manifest.json"),
                    "sha256": sha(root / "x5-manifest.json"),
                },
                "workers": [
                    {
                        "path": str(root / f"x5-worker-{w['mix']}.json"),
                        "sha256": sha(root / f"x5-worker-{w['mix']}.json"),
                    }
                    for w in workers
                ],
                "errors": errors,
                "mac_lease": lease.receipt,
                "elapsed_wall_seconds": time.monotonic() - started,
                "children_rusage": resource.getrusage(resource.RUSAGE_CHILDREN)[:2],
                "process_cpu_seconds": sum(w["process_cpu_seconds"] for w in workers),
                "final_host": host(),
                "caveat": (
                    "small development timing, no skill/promotion claim; "
                    "H5000 only; no fixed strict N is authorized"
                ),
            }
            if complete:
                receipt.update(forecasts(workers))
                receipt["completed_episodes"] = 96
                receipt["measured_episodes_per_elapsed_wall_hour"] = (
                    96 * 3600 / receipt["elapsed_wall_seconds"]
                )
            seal(root / "x5.json", receipt)
            return 0 if complete else 2
        except BaseException as exc:
            seal(
                root / "x5-coordinator-error.json",
                {
                    "status": "INCOMPLETE",
                    "time_pt": now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "started_children": [{"mix": m, "pid": p.pid} for m, p, _ in procs],
                    "mac_lease": lease.receipt,
                },
            )
            raise
        finally:
            drain_owned(root, procs)
            ds.release_cpu_slots(slots)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mix", choices=MIXES)
    parser.add_argument("--allocation-key", default="attempt-01")
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    args = parser.parse_args()
    for name in THREADS:
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    os.environ["SNAKE_V61_BACKEND"] = "python"
    if args.action == "check":
        print(
            json.dumps(
                {
                    "root": str(args.root),
                    "host": host(),
                    "B8_pin_ok": sha(B8) == B8_SHA,
                    "argv": [sys.executable, "-m", "research.bl_x5.timing", "prepare"],
                    "no_dispatch": True,
                },
                indent=2,
            )
        )
        return 0
    if args.action == "prepare":
        prepare(args.root, args.allocation_key)
        return 0
    if args.action == "worker":
        return worker(args.root, args.mix)
    return run(args.root, lease_path=args.lease_path)


if __name__ == "__main__":
    raise SystemExit(main())
