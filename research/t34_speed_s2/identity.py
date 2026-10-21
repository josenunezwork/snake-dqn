"""S2 identity fixtures and matched speed measurement (excluded diagnostic).

For each (body regime, mix) fixture it runs the *stock* ``teacher_v2.run_worlds`` and the
opt-in native-branch collector on the same deterministic workload (a fixed frame budget,
no wall-clock cut-off), then requires:

* byte-identical world exports and ``.sha256`` sidecars;
* identical deterministic receipt fields (worlds, decisions, row-step counters by kind);
* with ``--audit``, a second native run in which every branch v8 row also runs the stock
  Python hook and must agree on action, base, v2 counts and need (raises otherwise).

``--q-ties`` floors every frozen opponent's branch Q to integers in *both* arms (a
fixture, never a production recipe) so exact float32 ties reach the v8 normalization and
tie-break rules on many rows.

Timing: the stock and native arms do identical work (asserted), so the wall-clock ratio
is the per-row-step speed-up for that fixture.  Arms alternate order per repeat.  This is
a contended-host number on the Mac whenever other jobs run; the receipt records load.

Usage (Mac: threads = 1, nice 10)::

    nice -n 10 ./venv/bin/python -m research.t34_speed_s2.identity \
        --checkpoint .../B8_s3_final.pth --output-dir /fresh/dir --frames 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

DIAGNOSTIC_NAMESPACE = "t34-exit-cpu-profile-diagnostic/v1"
MIXES = ("scripted", "frozen", "mixed")
BODIES = ("short", "long", "xlong")
NATIVE_MODES = ("non_root", "all", "branch")
NATIVE_MODE = "non_root"  # set by --native-mode
DETERMINISTIC_KEYS = (
    "worlds",
    "requested_worlds",
    "decisions",
    "complete",
    "invalid_reason",
    "executed_row_steps",
    "completed_world_row_steps",
    "export_bytes",
    "mixes",
)
PROFILE_KEYS = (
    "root_row_steps",
    "shadow_row_steps",
    "reconstruction_row_steps",
    "search_row_steps",
    "active_row_steps",
    "nominal_row_steps",
    "compactions",
    "peak_active_rows",
)


def _single_thread() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)


def _config(checkpoint: Path, output: Path, mix: str, body: str, frames: int) -> dict[str, Any]:
    from research.t34_exit_b8 import profile_cpu

    config = profile_cpu._case_config(
        checkpoint, output, 3600.0, mix, "long" if body == "xlong" else body
    )
    if body == "xlong":
        config["world_builder"] = _xlong_world_builder
    config.update(frames=int(frames), seconds=3600.0, steady_state_seconds=0.0)
    return config


def _floor_q_install(policy: Any) -> None:
    """Fixture only: integer-floored opponent Q (many exact float32 ties)."""
    import torch

    if getattr(policy, "_t34_s2_floor_q", False) or not hasattr(policy, "q_values"):
        return
    original = type(policy).q_values.__get__(policy)

    def floored(states: Any) -> Any:
        return torch.floor(original(states))

    policy.q_values = floored
    policy._t34_s2_floor_q = True


def _tie_builder(builder: Callable[..., Any]) -> Callable[..., Any]:
    """Floor every frozen opponent's Q in the root world (clones inherit it)."""

    def build(row: Mapping[str, Any], checkpoint: Any) -> Any:
        world = builder(row, checkpoint)
        for policy in world.opponents:
            if getattr(policy, "veto_slots", frozenset()):
                _floor_q_install(policy)
        return world

    return build


class _Events:
    """Physical event coverage on every stepped world (root, shadow, doom, branch)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self._inside = 0

    def _observe(self, sim: Any, before: Any, prefix: str) -> None:
        after = sim.alive[:, 1:]
        self.counts[f"{prefix}_ticks"] += 1
        self.counts[f"{prefix}_opponent_deaths"] += int((before & ~after).sum())
        self.counts[f"{prefix}_opponent_respawns"] += int((~before & after).sum())
        self.counts[f"{prefix}_rows_opponent_len_ge_800"] += int(
            ((sim.length[:, 1:] >= 800) & after).any(axis=1).sum()
        )

    def _contested(self, sim: Any, prefix: str) -> None:
        """Envs where >= 2 live heads are within Manhattan 2 of one food cell."""
        heads = sim.heads()
        alive = np.asarray(sim.alive, dtype=bool)
        for env in range(int(sim.E)):
            live = heads[env][alive[env]]
            if len(live) < 2 or not sim.food_cells[env]:
                continue
            food = np.asarray(sim.food_cells[env], dtype=np.int64).reshape(-1, 2)
            near = (np.abs(food[:, None, :] - live[None, :, :]).sum(axis=2) <= 2).sum(axis=1)
            self.counts[f"{prefix}_contested_food_envs"] += int((near >= 2).any())

    def install(self) -> Callable[[], None]:
        from research.t34_exit_b8.engine import ExitTeacher
        from src.simd_env.batch_sim import BatchSim

        events, original_rollout, original_step = self, ExitTeacher._rollout_merged, BatchSim.step

        def rollout(teacher: Any, *args: Any, **kwargs: Any) -> Any:
            events._inside += 1
            try:
                return original_rollout(teacher, *args, **kwargs)
            finally:
                events._inside -= 1

        def step(sim: Any, *args: Any, **kwargs: Any) -> Any:
            prefix = "branch" if events._inside else "world"
            before = sim.alive[:, 1:].copy()
            events._contested(sim, prefix)
            result = original_step(sim, *args, **kwargs)
            events._observe(sim, before, prefix)
            return result

        ExitTeacher._rollout_merged = rollout
        BatchSim.step = step

        def restore() -> None:
            ExitTeacher._rollout_merged = original_rollout
            BatchSim.step = original_step

        return restore


def _xlong_world_builder(row: Mapping[str, Any], checkpoint: Path) -> Any:
    """``profile_cpu`` long bodies, but with three opponents at >= 800 cells."""
    from research.t34_exit_b8 import profile_cpu
    from research.t34_exit_b8.trajectory import prepare_world
    from src.simd_env.grid_scenarios import inject_serpentines

    world = profile_cpu._reset_world_builder(row, checkpoint)
    lengths = [400, 900, 850, 800, 300, 150]
    if int(world.sim.S) != len(lengths) or int(world.sim.cap) <= max(lengths) + 3:
        raise ValueError("xlong fixture requires six slots and ring capacity above 903")
    inject_serpentines([world.sim], lengths, box_height=8)
    controlled = world.runtime.controlled.copy()
    world.runtime.bind(world.sim, controlled)
    world.sim._policy_phase_prepared = False
    prepare_world(world)
    return world


def _exports(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _deterministic(receipt: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: receipt.get(key) for key in DETERMINISTIC_KEYS}
    profile = receipt.get("teacher_profile", {})
    out["teacher_profile"] = {key: profile.get(key) for key in PROFILE_KEYS}
    return out


def _run(
    kind: str,
    checkpoint: Path,
    output: Path,
    mix: str,
    body: str,
    frames: int,
    *,
    q_ties: bool,
    audit: bool = False,
    events: _Events | None = None,
) -> dict[str, Any]:
    from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
    from research.t34_exit_b8.teacher_v2 import run_worlds
    from research.t34_speed_s2.native_v8 import make_native_teacher, native_config

    config = _config(checkpoint, output, mix, body, frames)
    if q_ties:
        config["world_builder"] = _tie_builder(config["world_builder"])
    if kind == "stock":
        teacher = ExitTeacher(
            ExitTeacherConfig(horizon=int(config.get("horizon", 300)), beta=1.0), device="cpu"
        )
        config["teacher"] = teacher
    else:
        teacher = make_native_teacher(
            config,
            audit=audit,
            branch_only=NATIVE_MODE == "branch",
            include_roots=NATIVE_MODE == "all",
        )
        config = native_config(config, teacher)
    restore = events.install() if events is not None else None
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    try:
        receipt = run_worlds(config)
    finally:
        if restore is not None:
            restore()
    wall, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
    if not receipt.get("complete"):
        raise RuntimeError(f"{kind} {body}/{mix} did not complete: {receipt.get('invalid_reason')}")
    result: dict[str, Any] = {
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "host_stage_seconds": receipt.get("host_stage_seconds"),
        "deterministic": _deterministic(receipt),
        "exports": _exports(output),
    }
    if kind != "stock":
        result["native_stats"] = teacher.native_stats.as_dict()
        teacher.release_roots()
    return result


def run_fixture(
    checkpoint: Path,
    output: Path,
    mix: str,
    body: str,
    frames: int,
    *,
    repeats: int,
    q_ties: bool,
    audit: bool,
) -> dict[str, Any]:
    """Stock vs native (timed, alternating), plus an optional audited native run."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    tag = f"{body}-{mix}{'-qties' if q_ties else ''}"
    timings: dict[str, list[float]] = {"stock": [], "native": []}
    cpu: dict[str, list[float]] = {"stock": [], "native": []}
    reference: dict[str, Any] | None = None
    failures: list[str] = []
    native_stats: dict[str, int] = {}
    stages: dict[str, Any] = {}
    for repeat in range(repeats):
        order = ("stock", "native") if repeat % 2 == 0 else ("native", "stock")
        for kind in order:
            result = _run(
                kind,
                checkpoint,
                output / f"{tag}-{kind}-r{repeat}",
                mix,
                body,
                frames,
                q_ties=q_ties,
            )
            timings[kind].append(result["wall_seconds"])
            cpu[kind].append(result["process_cpu_seconds"])
            stages.setdefault(kind, result["host_stage_seconds"])
            if kind == "native":
                native_stats = result["native_stats"]
            observed = {"deterministic": result["deterministic"], "exports": result["exports"]}
            if reference is None:
                reference = observed
            elif observed != reference:
                failures.append(f"{kind} r{repeat} differs from the first run")
    audit_result = None
    if audit:
        events = _Events()
        result = _run(
            "native",
            checkpoint,
            output / f"{tag}-native-audit",
            mix,
            body,
            frames,
            q_ties=q_ties,
            audit=True,
            events=events,
        )
        observed = {"deterministic": result["deterministic"], "exports": result["exports"]}
        if observed != reference:
            failures.append("audited native run differs from the stock run")
        audit_result = {**result["native_stats"], **dict(events.counts)}
    assert reference is not None
    stock = float(statistics.median(timings["stock"]))
    native = float(statistics.median(timings["native"]))
    rows = int(reference["deterministic"]["executed_row_steps"] or 0)
    return {
        "fixture": tag,
        "mix": mix,
        "body": body,
        "frames": frames,
        "q_ties": q_ties,
        "identical": not failures,
        "failures": failures,
        "exports": len(reference["exports"]),
        "export_sha256": reference["exports"],
        "deterministic": reference["deterministic"],
        "native_stats": native_stats,
        "audit": audit_result,
        "wall_seconds": timings,
        "process_cpu_seconds": cpu,
        "host_stage_seconds_first": stages,
        "row_steps": rows,
        "ms_per_row_step": {
            "stock": 1000.0 * stock / rows if rows else None,
            "native": 1000.0 * native / rows if rows else None,
        },
        "speedup_median_wall": stock / native if native else None,
        "speedup_min_wall": min(timings["stock"]) / min(timings["native"]),
    }


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "argv": list(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "source_revision": revision,
        "load_average_start": os.getloadavg(),
        "authority": "excluded diagnostic (speed program S2); not A5/Round 0/gate evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mixes", default=",".join(MIXES))
    parser.add_argument("--bodies", default=",".join(BODIES))
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--q-ties", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--native-mode", choices=NATIVE_MODES, default="non_root")
    args = parser.parse_args()
    global NATIVE_MODE
    NATIVE_MODE = args.native_mode
    _single_thread()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Untimed warm-up: imports, torch, numba compilation/cache load and the v8 kernel
    # on both arms, so neither timed arm pays first-use costs.
    warm_started = time.perf_counter()
    for kind in ("stock", "native"):
        _run(
            kind,
            args.checkpoint.resolve(),
            args.output_dir / "warmup" / kind,
            "frozen",
            "long",
            args.warmup_frames,
            q_ties=False,
        )
    warmup_seconds = time.perf_counter() - warm_started
    fixtures = []
    for body in args.bodies.split(","):
        for mix in args.mixes.split(","):
            fixture = run_fixture(
                args.checkpoint.resolve(),
                args.output_dir / "runs",
                mix,
                body,
                args.frames,
                repeats=args.repeats,
                q_ties=args.q_ties,
                audit=args.audit,
            )
            fixtures.append(fixture)
            print(
                json.dumps(
                    {
                        key: fixture[key]
                        for key in (
                            "fixture",
                            "identical",
                            "row_steps",
                            "ms_per_row_step",
                            "speedup_median_wall",
                            "audit",
                        )
                    },
                    default=float,
                ),
                flush=True,
            )
    total_stock = sum(min(f["wall_seconds"]["stock"]) for f in fixtures)
    total_native = sum(min(f["wall_seconds"]["native"]) for f in fixtures)
    median_stock = sum(statistics.median(f["wall_seconds"]["stock"]) for f in fixtures)
    median_native = sum(statistics.median(f["wall_seconds"]["native"]) for f in fixtures)
    receipt = {
        "schema": "t34-speed-s2-identity/v2",
        "provenance": {**_provenance(), "load_average_end": os.getloadavg()},
        "recipe": {
            "threads": 1,
            "device": "cpu",
            "frames": args.frames,
            "repeats": args.repeats,
            "warmup_frames": args.warmup_frames,
            "warmup_seconds": warmup_seconds,
            "q_ties": args.q_ties,
            "audit": args.audit,
            "native_mode": args.native_mode,
            "namespace": DIAGNOSTIC_NAMESPACE,
            "timing_aggregation": "median per fixture; pooled sum of fixture medians",
        },
        "all_identical": all(f["identical"] for f in fixtures),
        "total_row_steps": sum(f["row_steps"] for f in fixtures),
        "total_min_wall_seconds": {"stock": total_stock, "native": total_native},
        "total_median_wall_seconds": {"stock": median_stock, "native": median_native},
        "pooled_speedup": median_stock / median_native if median_native else None,
        "pooled_speedup_min_wall": total_stock / total_native if total_native else None,
        "fixtures": fixtures,
    }
    target = args.output_dir / "identity.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"receipt": str(target), "all_identical": receipt["all_identical"],
                      "pooled_speedup": receipt["pooled_speedup"]}))  # fmt: skip
    if not receipt["all_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
