"""Excluded S1 v4 digest diagnostics; never import this from a registered study.

The candidate runs only in private module copies.  In particular, loading this
file does not replace ``research.t34_shadow.shadow.full_digest`` or
``research.t34_exit_b8.teacher_v2.full_digest``.  ``bench`` is a small,
interleaved v3/v4 production-shape probe; it is deliberately not the S0/D2
adoption receipt.  ``identity`` runs the established M1 live/Grid/shadow
comparison with a private ``ShadowTrajectory`` bound to v4.

Example (run only during an allowed two-core diagnostic window)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
      nice -n 10 ./venv/bin/python -m research.speed_s1_digest_v4.validate \\
      --mode bench --checkpoint /absolute/hero.pth --out /tmp/s1-bench.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from research.speed_s1_digest_v4.digest import FULL_DIGEST_SCHEMA as V4_SCHEMA
from research.speed_s1_digest_v4.digest import DigestV4

ROOT = Path(__file__).resolve().parents[2]
BASELINE_REVISION = "d62070e51759ccf8bfbd3ee38c947132777aa392"
D2_STARTED_AT = datetime(2026, 10, 21, 4, 32, 28, tzinfo=UTC)
D2_DEADLINE = datetime(2026, 10, 23, 4, 32, 28, tzinfo=UTC)
MIXES = ("scripted", "frozen", "mixed")
BODY_REGIMES = ("short", "long")
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_sha(relative: str) -> str:
    """Hash the ratified pre-S1 oracle source without trusting the worktree."""
    content = subprocess.run(
        ["git", "show", f"{BASELINE_REVISION}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _assert_oracle_unchanged(relative: str) -> None:
    """Refuse an oracle run if its current source is no longer baseline bytes."""
    actual = _sha(ROOT / relative)
    expected = _baseline_sha(relative)
    if actual != expected:
        raise RuntimeError(f"oracle source drifted from {BASELINE_REVISION}: {relative}")


def _source_receipt() -> dict[str, Any]:
    paths = {
        "shadow": "research/t34_shadow/shadow.py",
        "teacher": "research/t34_exit_b8/teacher_v2.py",
        "identity": "research/t34_shadow/identity.py",
        "digest_v4": "research/speed_s1_digest_v4/digest.py",
        "validate": "research/speed_s1_digest_v4/validate.py",
    }
    oracle_paths = {
        name: relative
        for name, relative in paths.items()
        if name in {"shadow", "teacher", "identity"}
    }
    return {
        "baseline_revision": BASELINE_REVISION,
        "v3_shadow_oracle_sha256": (
            "9346b8ff447dd7322c78b0f87aa01e7603870e95b6a394e92104889880019058"
        ),
        "baseline_sha256": {
            name: _baseline_sha(relative) for name, relative in oracle_paths.items()
        },
        "worktree_sha256": {name: _sha(ROOT / relative) for name, relative in paths.items()},
    }


def _cpu_ceiling() -> None:
    """Set the required single-thread limits before candidate imports/JIT work."""
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import numba
    import torch

    numba.set_num_threads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _exec_private(name: str, path: Path, source: str | None = None) -> ModuleType:
    """Execute ``path`` under a non-production module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise RuntimeError(f"could not build private spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        code = compile(source if source is not None else path.read_text(), str(path), "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _v4_digest_seam() -> Callable[..., str]:
    """Keep one lane-local serializer without assuming a checkpoint type."""
    lane = DigestV4()

    def digest(
        world: Any, *, checkpoint: Any = None, global_rng_state: object | None = None
    ) -> str:
        return lane(world, checkpoint=checkpoint, global_rng_state=global_rng_state)

    return digest


def load_private_v4_modules() -> tuple[ModuleType, ModuleType]:
    """Return isolated shadow/teacher copies whose only digest seam is v4.

    Normal modules retain their globals.  The teacher source has one explicit
    import rewrite so its direct imports resolve to the isolated shadow copy.
    """
    nonce = f"{os.getpid()}_{time.monotonic_ns()}"
    shadow_name = f"_s1_v4_shadow_{nonce}"
    teacher_name = f"_s1_v4_teacher_{nonce}"
    shadow_path = ROOT / "research/t34_shadow/shadow.py"
    teacher_path = ROOT / "research/t34_exit_b8/teacher_v2.py"
    _assert_oracle_unchanged("research/t34_shadow/shadow.py")
    _assert_oracle_unchanged("research/t34_exit_b8/teacher_v2.py")
    shadow = _exec_private(shadow_name, shadow_path)
    shadow.full_digest = _v4_digest_seam()
    shadow.FULL_DIGEST_SCHEMA = V4_SCHEMA
    teacher_source = teacher_path.read_text()
    original = (
        "from research.t34_shadow.shadow import ShadowFrame, ShadowTrajectory, "
        "clone_world_exact, full_digest"
    )
    replacement = (
        f"from {shadow_name} import ShadowFrame, ShadowTrajectory, clone_world_exact, full_digest"
    )
    if teacher_source.count(original) != 1:
        raise RuntimeError("teacher private-import seam changed")
    teacher = _exec_private(
        teacher_name, teacher_path, teacher_source.replace(original, replacement)
    )
    return shadow, teacher


def private_v4_identity() -> ModuleType:
    """Load M1 identity with its rolling shadow object redirected to private v4."""
    shadow, _teacher = load_private_v4_modules()
    path = ROOT / "research/t34_shadow/identity.py"
    _assert_oracle_unchanged("research/t34_shadow/identity.py")
    source = path.read_text()
    original = "from research.t34_shadow.shadow import ShadowTrajectory"
    replacement = f"from {shadow.__name__} import ShadowTrajectory"
    if source.count(original) != 1:
        raise RuntimeError("identity private-import seam changed")
    return _exec_private(
        f"_s1_v4_identity_{os.getpid()}_{time.monotonic_ns()}",
        path,
        source.replace(original, replacement),
    )


def _rows() -> list[dict[str, Any]]:
    from research.t34_exit_b8.trajectory import make_world_rows

    return make_world_rows({"namespace": "s1-v4-excluded-fixtures/v1", "mixes": MIXES}, len(MIXES))


def _short_builder(row: Mapping[str, Any], checkpoint: Path) -> Any:
    from research.t34_exit_b8.trajectory import build_world, prepare_world
    from research.t34_shadow.shadow import clone_world_exact

    world = build_world(row, checkpoint)
    prepare_world(world)
    return clone_world_exact(world)


def _long_builder(row: Mapping[str, Any], checkpoint: Path) -> Any:
    from research.t34_exit_b8.trajectory import prepare_world
    from src.simd_env.grid_scenarios import inject_serpentines

    world = _short_builder(row, checkpoint)
    lengths = [900, 600, 400, 300, 150, 150]
    if int(world.sim.S) != len(lengths) or int(world.sim.cap) <= max(lengths) + 3:
        raise ValueError("long excluded fixture requires six slots and ring capacity above 903")
    inject_serpentines([world.sim], lengths, box_height=8)
    controlled = world.runtime.controlled.copy()
    world.runtime.bind(world.sim, controlled)
    world.sim._policy_phase_prepared = False
    prepare_world(world)
    return world


def _export_hashes(path: Path) -> dict[str, str]:
    return {item.name: _sha(item) for item in sorted(path.glob("world-*.json*"))}


def _config(
    checkpoint: Path, output: Path, row: Mapping[str, Any], body: str, frames: int, seconds: float
) -> dict[str, Any]:
    return {
        "kernel": "v2",
        "checkpoint": checkpoint,
        "output_dir": output,
        "rows": [dict(row)],
        "max_worlds": 1,
        "worlds": 1,
        "worlds_per_process": 1,
        "frames": frames,
        "seconds": seconds,
        "steady_state_seconds": 0,
        "horizon": 300,
        "shadow_depth": 300,
        "threads": 1,
        "device": "cpu",
        "namespace": "s1-v4-excluded-fixtures/v1",
        "mixes": MIXES,
        "world_builder": _long_builder if body == "long" else _short_builder,
    }


def _one_run(
    runner: Callable[[Mapping[str, Any]], dict[str, Any]], config: dict[str, Any], output: Path
) -> dict[str, Any]:
    started = time.monotonic()
    result = runner(config)
    elapsed = time.monotonic() - started
    active = int(result.get("active_row_steps", 0))
    exports = _export_hashes(output)
    return {
        "elapsed_seconds": elapsed,
        "active_row_steps": active,
        "active_row_steps_per_second": active / elapsed if elapsed else None,
        "complete": bool(result.get("complete", False)),
        "worlds": int(result.get("worlds", 0)),
        "result": result,
        "exports": exports,
        "export_file_count": len(exports),
    }


def _identity_eligible(run: Mapping[str, Any]) -> bool:
    """An incomplete or exportless run is never eligible for a byte claim."""
    files = run.get("exports", {})
    payloads = [name for name in files if name.endswith(".json")]
    return (
        bool(run.get("complete"))
        and int(run.get("worlds", 0)) == 1
        and len(payloads) == 1
        and all(name + suffix in files for name in payloads for suffix in (".sha256", ".complete"))
    )


def _same_export_bytes(first: Path, second: Path) -> bool:
    """Compare every payload and sidecar directly, with names checked separately."""
    left = {path.name: path.read_bytes() for path in first.glob("world-*.json*")}
    right = {path.name: path.read_bytes() for path in second.glob("world-*.json*")}
    return bool(left) and left == right


def _case_seconds(deadline: float, remaining_cases: int) -> float:
    """Divide the unexpired global cap; each run still receives an explicit bound."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("S1 diagnostic wall cap reached")
    return max(0.01, remaining / max(1, remaining_cases))


def run_bench(checkpoint: Path, workdir: Path, *, frames: int, deadline: float) -> dict[str, Any]:
    """Run >=3 interleaved v3/v4 short/long x three-mix fixture pairs."""
    from research.t34_exit_b8.teacher_v2 import run_worlds as v3_run_worlds

    if not 1 <= frames <= 5000:
        raise ValueError("--frames must be in 1..5000")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _shadow, v4_teacher = load_private_v4_modules()
    rows = {str(row["mix"]): row for row in _rows()}
    arms_to_warm = (
        (body, mix, arm) for body in BODY_REGIMES for mix in MIXES for arm in ("v3", "v4")
    )
    warmups: list[dict[str, Any]] = []
    for warm_index, (body, mix, arm) in enumerate(arms_to_warm):
        output = workdir / f"warm-{body}-{mix}-{arm}"
        runner = v3_run_worlds if arm == "v3" else v4_teacher.run_worlds
        warm = _one_run(
            runner,
            _config(
                checkpoint,
                output,
                rows[mix],
                body,
                frames=min(2, frames),
                seconds=_case_seconds(deadline, 12 - warm_index + 36),
            ),
            output,
        )
        warmups.append({"body_regime": body, "mix": mix, "arm": arm, **warm})
        if not _identity_eligible(warm):
            raise RuntimeError(f"warmup incomplete: {body}/{mix}/{arm}")
    cases: list[dict[str, Any]] = []
    # A fixed alternating first arm makes every repetition an actual interleave.
    for repeat in range(3):
        for body in BODY_REGIMES:
            for mix in MIXES:
                arms = ("v3", "v4") if repeat % 2 == 0 else ("v4", "v3")
                runs: dict[str, dict[str, Any]] = {}
                for arm_index, arm in enumerate(arms):
                    output = workdir / f"bench-r{repeat}-{body}-{mix}-{arm}"
                    runner = v3_run_worlds if arm == "v3" else v4_teacher.run_worlds
                    remaining_runs = 2 * (18 - len(cases)) - arm_index
                    runs[arm] = _one_run(
                        runner,
                        _config(
                            checkpoint,
                            output,
                            rows[mix],
                            body,
                            frames,
                            _case_seconds(deadline, remaining_runs),
                        ),
                        output,
                    )
                identical = (
                    _identity_eligible(runs["v3"])
                    and _identity_eligible(runs["v4"])
                    and runs["v3"]["exports"] == runs["v4"]["exports"]
                    and _same_export_bytes(
                        workdir / f"bench-r{repeat}-{body}-{mix}-v3",
                        workdir / f"bench-r{repeat}-{body}-{mix}-v4",
                    )
                )
                v3_elapsed, v4_elapsed = (
                    runs["v3"]["elapsed_seconds"],
                    runs["v4"]["elapsed_seconds"],
                )
                cases.append(
                    {
                        "repeat": repeat,
                        "mix": mix,
                        "body_regime": body,
                        "order": list(arms),
                        "v3": runs["v3"],
                        "v4": runs["v4"],
                        "exports_and_sidecars_byte_identical": identical,
                        "whole_workload_speedup_v3_over_v4": (
                            v3_elapsed / v4_elapsed if v4_elapsed else None
                        ),
                    }
                )
    comparable = [case for case in cases if case["v3"]["complete"] and case["v4"]["complete"]]
    v3_seconds = sum(float(case["v3"]["elapsed_seconds"]) for case in comparable)
    v4_seconds = sum(float(case["v4"]["elapsed_seconds"]) for case in comparable)
    return {
        "schema": "t34-speed-s1-excluded-fixture-probe/v1",
        "mode": "bench",
        "d2_adoption_receipt": False,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)},
        "fixtures": {"mixes": list(MIXES), "body_regimes": list(BODY_REGIMES), "repetitions": 3},
        "frames": frames,
        "warmup_excluded": True,
        "warmups": warmups,
        "cases": cases,
        "complete": len(comparable) == len(cases),
        "all_exports_and_sidecars_byte_identical": bool(cases)
        and all(bool(case["exports_and_sidecars_byte_identical"]) for case in cases),
        "matched_whole_workload_speedup_v3_over_v4": (
            v3_seconds / v4_seconds if v4_seconds else None
        ),
        "matched_completed_cases": len(comparable),
        "fixture_events": {
            "cold_forks": "exercised: every warmup and timed arm builds a fresh teacher/world",
            "death_respawn_refork": (
                "not forced by this short excluded probe; pending S0 fixture evidence"
            ),
        },
    }


def run_identity(*, frames: int, deadline: float) -> dict[str, Any]:
    """Execute ≥5,000-frame M1 checks in isolated v4 shadow copies per mix."""
    if not 5000 <= frames <= 5000:
        raise ValueError("identity requires exactly 5000 frames per mix")
    module = private_v4_identity()
    results: list[dict[str, Any]] = []
    for offset, mix in enumerate(MIXES):
        result = module.run_m1_mix(2026101934 + offset, mix, frames=frames, deadline=deadline)
        results.append(result)
    return {
        "schema": "t34-speed-s1-v4-m1-identity/v1",
        "mode": "identity",
        "d2_adoption_receipt": False,
        "frames_per_mix": frames,
        "mixes": list(MIXES),
        "results": results,
        "complete": all(result.get("status") == "COMPLETE" for result in results),
        "zero_physical_carry_v8_action_shadow_differences": all(
            bool(result.get("zero_unexplained")) for result in results
        ),
    }


def _mac_scheduling_guard() -> dict[str, Any]:
    """Keep the excluded probe out of an admitted paced/retention quiet window."""
    if sys.platform != "darwin":
        return {"platform": sys.platform, "mac_guard": "not_applicable"}
    power = subprocess.run(
        ["pmset", "-g", "batt"], check=True, capture_output=True, text=True
    ).stdout
    lid = subprocess.run(
        ["ioreg", "-r", "-k", "AppleClamshellState"], check=True, capture_output=True, text=True
    ).stdout
    if "AC Power" not in power or '"AppleClamshellState" = No' not in lid:
        raise RuntimeError("S1 diagnostics require AC power and an open lid")
    tmux = subprocess.run(["tmux", "list-sessions", "-F", "#S"], capture_output=True, text=True)
    sessions = [] if tmux.returncode else [line for line in tmux.stdout.splitlines() if line]
    quiet = [
        name
        for name in sessions
        if any(token in name.lower() for token in ("retention", "pathb", "paced"))
    ]
    if quiet:
        raise RuntimeError(f"S1 diagnostics prohibited during quiet retention session(s): {quiet}")
    return {"platform": sys.platform, "ac_power": True, "open_lid": True, "tmux_sessions": sessions}


def _admit() -> dict[str, Any]:
    """Refuse late S1 execution; D2 closes the line unless a new profile opens it."""
    now = datetime.now(UTC)
    admission = {
        "first_implementation_utc": D2_STARTED_AT.isoformat(),
        "d2_deadline_utc": D2_DEADLINE.isoformat(),
        "checked_at_utc": now.isoformat(),
        "auto_adopt": False,
        "requires_s0_m1_mutations_and_matched_1_25x": True,
    }
    if now >= D2_DEADLINE:
        raise RuntimeError("D2 deadline passed: retain v3 and close S1 pending a new profile")
    return admission


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
    with path.with_name(path.name + ".sha256").open("x") as handle:
        handle.write(hashlib.sha256(payload).hexdigest() + "\n")
    path.with_name(path.name + ".complete").touch(exist_ok=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bench", "identity", "all"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="create-only JSON receipt path")
    parser.add_argument("--checkpoint", type=Path, help="hero checkpoint; required for bench")
    parser.add_argument("--frames", type=int, default=32, help="bench horizon (1..5000)")
    parser.add_argument("--identity-frames", type=int, default=5000)
    parser.add_argument(
        "--seconds", type=float, default=1800, help="whole invocation cap, at most 1800"
    )
    args = parser.parse_args()
    if args.out.exists() or args.out.with_name(args.out.name + ".complete").exists():
        parser.error("--out receipt must be create-only")
    if not 0 < args.seconds <= 1800:
        parser.error("--seconds must be in (0, 1800]")
    if args.mode in {"bench", "all"} and args.checkpoint is None:
        parser.error("--checkpoint is required for bench")
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "t34-speed-s1-v4-validation/v1",
        "status": "INCOMPLETE",
        "argv": sys.argv,
        "thread_caps": None,
        "scheduling_guard": None,
        "d2_admission": None,
        "source": _source_receipt(),
        "mode": args.mode,
        "start_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    try:
        _cpu_ceiling()
        receipt["thread_caps"] = {name: os.environ.get(name) for name in THREAD_ENV}
        receipt["scheduling_guard"] = _mac_scheduling_guard()
        receipt["d2_admission"] = _admit()
        if args.mode in {"bench", "all"}:
            receipt["bench"] = run_bench(
                args.checkpoint.resolve(),
                args.out.with_suffix(".bench-work"),
                frames=args.frames,
                deadline=started + args.seconds,
            )
        if args.mode in {"identity", "all"}:
            receipt["identity"] = run_identity(
                frames=args.identity_frames, deadline=started + args.seconds
            )
        expected = [key for key in ("bench", "identity") if args.mode in {key, "all"}]
        if all(bool(receipt[key].get("complete")) for key in expected):
            receipt["status"] = "COMPLETE"
        else:
            receipt["error"] = "one or more diagnostic modes were incomplete"
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["end_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt["final_source"] = _source_receipt()
    receipt["source_unchanged_during_run"] = receipt["source"] == receipt["final_source"]
    if not receipt["source_unchanged_during_run"]:
        receipt["status"] = "INCOMPLETE"
        receipt["error"] = "source changed during diagnostic"
    _write_once(args.out, receipt)
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key not in {"bench", "identity"}},
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
