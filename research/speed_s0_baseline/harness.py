"""S0 matched baseline harness for the T34 ExIt v2 teacher engine (speed program).

The ruler for every later engine speed change (speed_vs_constraints JUDGMENT §3 S0,
§5 D1-D3).  It runs a fixed, excluded fixture through the unmodified production
collector ``research.t34_exit_b8.teacher_v2.run_worlds``:

* three opponent mixes (scripted, frozen, mixed) x two body regimes (production
  short bodies; injected long serpentine bodies, hero 900, all >= 150, three >= 400);
* fixed work: one world per case, at most ``FRAMES`` decisions, production horizon and
  shadow depth (H300 / D300, beta 1.0, one resident world, CPU, one thread);
* every case starts cold (fresh build, clone/reset, first shadow fork) and the fixture
  must exercise opponent deaths and respawns, re-forks after invalidation and at least
  one doom search, checked by a separate counting round.

Round kinds, each one full pass over the six cases:

``coverage``      uninstrumented + event counters (deaths, respawns, forks).
``plain``         uninstrumented: the whole-workload wall used for matched speed-ups.
``instrumented``  wraps hot functions to get exclusive per-stage seconds.

Every round's exports and checksum sidecars are compared byte-for-byte with the
committed frozen reference (``REFERENCE_MAC.json``) and with the run's own first
round.  This is an excluded diagnostic: it is not A5, Round 0, gate, retention or
serving evidence and its seeds sit in a private namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

SCHEMA = "t34-speed-s0-baseline/v1"
REFERENCE_SCHEMA = "t34-speed-s0-reference/v1"
NAMESPACE = "t34-speed-s0-baseline/v1"
MIXES = ("scripted", "frozen", "mixed")
BODIES = ("short", "long")
FRAMES = 300
HORIZON = 300
SHADOW_DEPTH = 300
BETA = 1.0
CASE_SECONDS = 3600.0
LONG_LENGTHS = (900, 600, 400, 300, 150, 150)
STAGES = (
    "digest",
    "v8",
    "v8_hero_features",
    "raster",
    "sim",
    "inference",
    "clone",
    "hero_evaluate",
    "opponent_actions",
    "audit_hash",
    "export",
    "build",
)
EXPORT_SUFFIXES = ("", ".sha256", ".complete")
ROUND_KINDS = ("coverage", "plain", "instrumented")
PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
DEFAULT_REFERENCE = PACKAGE / "REFERENCE_MAC.json"
# Engine import closure whose bytes define "the engine" for identity purposes.
ENGINE_SOURCES = (
    "research/t34_exit_b8",
    "research/t34_shadow",
    "research/t33_batched_teacher",
    "research/phase0_search_20261019",
    "research/t3_rollout_labels_20261015",
    "src/simd_env",
    "src/model",
)


# --------------------------------------------------------------------------- fixture


def fixture_cases() -> list[dict[str, Any]]:
    """Six deterministic cases; short bodies use rows 0-2, long bodies rows 3-5."""
    from research.t34_exit_b8.trajectory import make_world_rows

    rows = make_world_rows({"namespace": NAMESPACE, "mixes": MIXES}, len(MIXES) * len(BODIES))
    cases = []
    for body_index, body in enumerate(BODIES):
        for mix_index, mix in enumerate(MIXES):
            row = rows[body_index * len(MIXES) + mix_index]
            if row["mix"] != mix:
                raise AssertionError("fixture row/mix order changed")
            cases.append({"case": f"{body}-{mix}", "mix": mix, "body": body, "row": row})
    return cases


def _reset_world(row: Mapping[str, Any], checkpoint: Path) -> Any:
    """Production builder, prepared, then deep-cloned/reset (a cold refork source)."""
    from research.t34_exit_b8.trajectory import build_world, prepare_world
    from research.t34_shadow.shadow import clone_world_exact

    world = build_world(row, checkpoint)
    prepare_world(world)
    return clone_world_exact(world)


def _long_world(row: Mapping[str, Any], checkpoint: Path) -> Any:
    """Inject deterministic long serpentine bodies (no RNG), then rebind features."""
    from research.t34_exit_b8.trajectory import prepare_world
    from src.simd_env.grid_scenarios import inject_serpentines

    world = _reset_world(row, checkpoint)
    if int(world.sim.S) != len(LONG_LENGTHS) or int(world.sim.cap) <= max(LONG_LENGTHS) + 3:
        raise ValueError("long-body fixture requires six slots and ring capacity above 903")
    inject_serpentines([world.sim], list(LONG_LENGTHS), box_height=8)
    controlled = world.runtime.controlled.copy()
    world.runtime.bind(world.sim, controlled)
    world.sim._policy_phase_prepared = False
    prepare_world(world)
    return world


def _builder(body: str) -> Any:
    return _long_world if body == "long" else _reset_world


def case_config(
    case: Mapping[str, Any], checkpoint: Path, output_dir: Path, *, frames: int = FRAMES
) -> dict[str, Any]:
    return {
        "kernel": "v2",
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "rows": [case["row"]],
        "max_worlds": 1,
        "worlds": 1,
        "frames": frames,
        "horizon": HORIZON,
        "shadow_depth": SHADOW_DEPTH,
        "beta": BETA,
        "seconds": CASE_SECONDS,
        "steady_state_seconds": 0.0,
        "threads": 1,
        "device": "cpu",
        "namespace": NAMESPACE,
        "mixes": MIXES,
        "world_builder": _builder(str(case["body"])),
    }


# --------------------------------------------------------------------------- timing


class ExclusiveStages:
    """Nested wall-clock accounting where each stage excludes timed children."""

    def __init__(self, names: Sequence[str] = STAGES) -> None:
        self.names = tuple(names)
        self.inclusive: dict[str, float] = defaultdict(float)
        self.exclusive: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._stack: list[list[Any]] = []

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        entry: list[Any] = [name, 0.0]
        self._stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._stack.pop()
            self.inclusive[name] += elapsed
            self.exclusive[name] += max(0.0, elapsed - float(entry[1]))
            self.calls[name] += 1
            if self._stack:
                self._stack[-1][1] += elapsed

    def receipt(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "inclusive_seconds": float(self.inclusive[name]),
                "exclusive_seconds": float(self.exclusive[name]),
                "calls": int(self.calls[name]),
            }
            for name in self.names
        }


@contextmanager
def _patched(patches: list[tuple[Any, str, Any]]) -> Iterator[None]:
    originals = [(owner, attr, getattr(owner, attr)) for owner, attr, _ in patches]
    try:
        for owner, attr, replacement in patches:
            setattr(owner, attr, replacement)
        yield
    finally:
        for owner, attr, original in reversed(originals):
            setattr(owner, attr, original)


def _timed(stages: ExclusiveStages, stage: str, original: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stages.measure(stage):
            return original(*args, **kwargs)

    return wrapper


@contextmanager
def instrument(stages: ExclusiveStages) -> Iterator[None]:
    """Wrap production hot-path callables at the references the collector uses."""
    import torch

    from research.t3_rollout_labels_20261015.bank_engine import CapturedWorld
    from research.t34_exit_b8 import teacher_v2
    from research.t34_exit_b8.engine import ExitTeacher
    from research.t34_shadow import shadow
    from src.simd_env import ego_raster, v8_nb
    from src.simd_env.batch_sim import BatchSim
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    # ``v8`` is the opponents' veto hook (S2's target, D1); ``v8_hero_features`` is the
    # hero's bv8 observation kernel, called inside the raster build (not D1's v8).
    targets = [
        (shadow, "full_digest", "digest"),
        (teacher_v2, "full_digest", "digest"),
        (Vector61SimdPolicy, "_apply_veto", "v8"),
        (ego_raster, "build_ego_raster", "raster"),
        (v8_nb, "v8_decide", "v8_hero_features"),
        (BatchSim, "step", "sim"),
        (torch.nn.Module, "_call_impl", "inference"),
        (CapturedWorld, "clone", "clone"),
        (ExitTeacher, "evaluate", "hero_evaluate"),
        (ExitTeacher, "actions", "opponent_actions"),
        (teacher_v2, "state_hash", "audit_hash"),
        (teacher_v2, "write_world", "export"),
    ]
    with _patched([(o, a, _timed(stages, s, getattr(o, a))) for o, a, s in targets]):
        yield


@contextmanager
def count_events(counters: dict[str, int]) -> Iterator[None]:
    """Count root-world opponent/hero deaths and respawns plus shadow (re)forks.

    Only the root ("live") world steps are counted: the collector advances roots
    through module-level ``teacher_v2._step`` with the world it built, whose ids the
    caller registers in ``counters['_roots']`` via :func:`_tracking_builder`.
    """
    import numpy as np

    from research.t34_exit_b8 import teacher_v2
    from research.t34_shadow.shadow import ShadowTrajectory

    step, fork = teacher_v2._step, ShadowTrajectory.fork
    roots: set[int] = counters.setdefault("_roots", set())  # type: ignore[assignment]
    last: dict[int, Any] = {}

    def counted_step(world: Any, teacher: Any, hero_action: int) -> Any:
        if id(world) not in roots:
            return step(world, teacher, hero_action)
        # Respawns happen in the selector-prepare phase, which the collector runs
        # between root steps; compare with the alive vector left by the last step.
        before = np.asarray(world.sim.alive[0], dtype=bool).copy()
        previous = last.get(id(world))
        if previous is not None:
            born_between = ~previous & before
            counters["opponent_respawns"] += int(born_between[1:].sum())
        result = step(world, teacher, hero_action)
        after = np.asarray(world.sim.alive[0], dtype=bool).copy()
        died, born = before & ~after, ~before & after
        last[id(world)] = after
        counters["root_steps"] += 1
        counters["hero_deaths"] += int(died[0])
        counters["opponent_deaths"] += int(died[1:].sum())
        counters["opponent_respawns"] += int(born[1:].sum())
        return result

    def counted_fork(self: Any, source: Any, **kwargs: Any) -> Any:
        counters["shadow_forks"] += 1
        return fork(self, source, **kwargs)

    with _patched([(teacher_v2, "_step", counted_step), (ShadowTrajectory, "fork", counted_fork)]):
        yield


def _tracking_builder(builder: Any, counters: dict[str, Any]) -> Any:
    def build(row: Mapping[str, Any], checkpoint: Path) -> Any:
        world = builder(row, checkpoint)
        counters.setdefault("_roots", set()).add(id(world))
        return world

    return build


# --------------------------------------------------------------------------- rounds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_hashes(case_dir: Path, case: str) -> dict[str, str]:
    """SHA-256 of every export payload, checksum sidecar and completion marker."""
    hashes: dict[str, str] = {}
    for payload in sorted(case_dir.glob("world-*.json")):
        for suffix in EXPORT_SUFFIXES:
            path = payload.with_name(payload.name + suffix)
            hashes[f"{case}/{path.name}"] = _sha256(path)
    return hashes


def _cpu_ceiling() -> None:
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
    torch.set_num_interop_threads(1)


def warm_up(checkpoint: Path, output_dir: Path) -> float:
    """Load models and compile kernels for every case outside timed rounds."""
    from research.t34_exit_b8.teacher_v2 import run_worlds

    started = time.perf_counter()
    for case in fixture_cases():
        config = case_config(case, checkpoint, output_dir / case["case"], frames=2)
        config.update(horizon=2, shadow_depth=2)
        receipt = run_worlds(config)
        if not receipt.get("complete"):
            raise RuntimeError(f"warm-up failed for {case['case']}")
    return time.perf_counter() - started


def run_round(checkpoint: Path, output_dir: Path, kind: str) -> dict[str, Any]:
    """Run all six cases once; returns per-case timings, work, counters and hashes."""
    from research.t34_exit_b8.teacher_v2 import run_worlds

    if kind not in ROUND_KINDS:
        raise ValueError(f"unknown round kind {kind}")
    cases_out: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    cpu_started, round_started = time.process_time(), time.perf_counter()
    for case in fixture_cases():
        case_dir = output_dir / case["case"]
        config = case_config(case, checkpoint, case_dir)
        stages = ExclusiveStages()
        counters: dict[str, Any] = defaultdict(int)
        if kind == "coverage":
            config["world_builder"] = _tracking_builder(config["world_builder"], counters)
            context = count_events(counters)
        elif kind == "instrumented":
            config["world_builder"] = _timed(stages, "build", config["world_builder"])
            context = instrument(stages)
        else:
            context = _patched([])
        started = time.perf_counter()
        with context:
            receipt = run_worlds(config)
        wall = time.perf_counter() - started
        if not receipt.get("complete") or receipt.get("invalid_reason"):
            raise RuntimeError(f"case {case['case']} incomplete: {receipt.get('invalid_reason')}")
        case_hashes = export_hashes(case_dir, case["case"])
        hashes.update(case_hashes)
        export = json.loads((case_dir / "world-000000.json").read_text())
        teacher = export["teacher_v2"]
        entry: dict[str, Any] = {
            "case": case["case"],
            "mix": case["mix"],
            "body": case["body"],
            "wall_seconds": wall,
            "decisions": int(receipt["decisions"]),
            "active_row_steps": int(receipt["active_row_steps"]),
            "work": dict(teacher["work"]),
            "shadow_searches": int(teacher["shadow_searches"]),
            "divergences": int(teacher["divergences"]),
            "interventions": int(teacher["interventions"]),
            "labels": len(export["labels"]),
            "hero_survived_to_cap": int(receipt["decisions"]) >= FRAMES,
            "host_stage_seconds": dict(receipt["host_stage_seconds"]),
            "export_bytes": int(receipt["export_bytes"]),
        }
        if kind == "coverage":
            entry["events"] = {k: int(v) for k, v in counters.items() if not k.startswith("_")}
        if kind == "instrumented":
            entry["exclusive_stages"] = stages.receipt()
        cases_out.append(entry)
    wall = time.perf_counter() - round_started
    return {
        "kind": kind,
        "wall_seconds": wall,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "worlds": len(cases_out),
        "cases": cases_out,
        "hashes": hashes,
    }


def _worker(
    rank: int, checkpoint: str, output_dir: str, kinds: list[str], result_queue: Any
) -> None:
    try:
        _cpu_ceiling()
        root = Path(output_dir)
        warm = warm_up(Path(checkpoint), root / "warmup")
        rounds = []
        for index, kind in enumerate(kinds):
            rounds.append(run_round(Path(checkpoint), root / f"round-{index:02d}-{kind}", kind))
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result_queue.put(
            (
                rank,
                {
                    "rank": rank,
                    "warmup_seconds": warm,
                    "rounds": rounds,
                    "rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                },
                None,
            )
        )
    except BaseException as exc:  # reported to the parent; never swallowed
        result_queue.put((rank, None, f"{type(exc).__name__}: {exc}"))


def schedule(processes: int, plain: int, instrumented: int) -> list[list[list[str]]]:
    """Return phases of per-process round lists.

    Serial (one process): a single phase whose process runs the coverage round and
    then the interleaved timed rounds.  Parallel: phase 1 runs the coverage round
    alone (it also fills the on-disk Numba cache for every path, including doom
    search); phase 2 runs one timed round per concurrent single-thread process.
    """
    if processes < 1 or plain < 0 or instrumented < 0 or plain + instrumented < 1:
        raise ValueError("need >= 1 process and >= 1 timed round")
    timed: list[str] = []
    for index in range(max(plain, instrumented)):
        if index < plain:
            timed.append("plain")
        if index < instrumented:
            timed.append("instrumented")
    if processes == 1:
        return [[["coverage", *timed]]]
    if len(timed) != processes:
        raise ValueError("parallel mode needs exactly one timed round per process")
    return [[["coverage"]], [[kind] for kind in timed]]


def run_processes(
    checkpoint: Path, output_dir: Path, plan: list[list[str]], *, deadline_seconds: float
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    results = context.Queue()
    children = [
        context.Process(
            target=_worker,
            args=(rank, str(checkpoint), str(output_dir / f"proc-{rank:02d}"), kinds, results),
        )
        for rank, kinds in enumerate(plan)
    ]
    reports: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + deadline_seconds
    last_progress = time.monotonic()
    try:
        for child in children:
            child.start()
        while len(reports) < len(children):
            if time.monotonic() - last_progress >= 110:
                print(json.dumps({"event": "s0_progress", "done": len(reports)}), flush=True)
                last_progress = time.monotonic()
            try:
                rank, report, failure = results.get(timeout=1)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise RuntimeError("S0 harness exceeded its wall deadline")
                dead = [
                    i for i, c in enumerate(children) if i not in reports and c.exitcode is not None
                ]
                if dead:
                    raise RuntimeError(f"S0 worker crashed before reporting: {dead}")
                continue
            if failure:
                raise RuntimeError(f"S0 worker {rank} failed: {failure}")
            reports[int(rank)] = dict(report)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()  # owned child only
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
    return [reports[i] for i in range(len(children))]


# --------------------------------------------------------------------------- report


def engine_source_digest(repo: Path = REPO) -> dict[str, Any]:
    """Per-file SHA-256 of the engine import closure plus one combined digest."""
    files: dict[str, str] = {}
    for relative in ENGINE_SOURCES:
        for path in sorted((repo / relative).rglob("*.py")):
            files[str(path.relative_to(repo))] = _sha256(path)
    combined = hashlib.sha256(
        "".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    return {"combined_sha256": combined, "files": len(files), "roots": list(ENGINE_SOURCES)}


def compare_hashes(observed: Mapping[str, str], reference: Mapping[str, str]) -> dict[str, Any]:
    keys = sorted(set(observed) | set(reference))
    differing = [k for k in keys if observed.get(k) != reference.get(k)]
    return {
        "files_compared": len(keys),
        "files_equal": len(keys) - len(differing),
        "identical": not differing,
        "differing": differing,
    }


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize(reports: list[dict[str, Any]], reference: Mapping[str, str] | None) -> dict[str, Any]:
    rounds = [r for report in reports for r in report["rounds"]]
    anchor = rounds[0]["hashes"]
    identity = {
        "vs_reference": [
            (
                {"round": i, "kind": r["kind"], **compare_hashes(r["hashes"], reference)}
                if reference is not None
                else None
            )
            for i, r in enumerate(rounds)
        ],
        "vs_first_round": [
            {"round": i, "kind": r["kind"], **compare_hashes(r["hashes"], anchor)}
            for i, r in enumerate(rounds)
        ],
    }
    identity["all_rounds_self_identical"] = all(x["identical"] for x in identity["vs_first_round"])
    identity["all_rounds_match_reference"] = (
        None if reference is None else all(x["identical"] for x in identity["vs_reference"])
    )
    plain = [r for r in rounds if r["kind"] == "plain"]
    instr = [r for r in rounds if r["kind"] == "instrumented"]
    coverage = [r for r in rounds if r["kind"] == "coverage"]
    cases = [c["case"] for c in rounds[0]["cases"]]
    worlds = rounds[0]["worlds"]
    rows = sum(c["active_row_steps"] for c in rounds[0]["cases"])
    plain_walls = [r["wall_seconds"] for r in plain]
    median_plain = _median(plain_walls)
    per_case: dict[str, Any] = {}
    for name in cases:

        def pick(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [c for r in group for c in r["cases"] if c["case"] == name]

        plain_c, instr_c, cov_c = pick(plain), pick(instr), pick(coverage)
        base = cov_c[0] if cov_c else pick(rounds)[0]
        wall = _median([c["wall_seconds"] for c in plain_c])
        entry: dict[str, Any] = {
            "mix": base["mix"],
            "body": base["body"],
            "decisions": base["decisions"],
            "active_row_steps": base["active_row_steps"],
            "work": base["work"],
            "shadow_searches": base["shadow_searches"],
            "divergences": base["divergences"],
            "interventions": base["interventions"],
            "hero_survived_to_cap": base["hero_survived_to_cap"],
            "events": base.get("events"),
            "plain_wall_seconds_median": wall,
            "plain_wall_seconds": [c["wall_seconds"] for c in plain_c],
            "ms_per_active_row_step": (1000 * wall / base["active_row_steps"]) if wall else None,
        }
        if instr_c:
            walls = [c["wall_seconds"] for c in instr_c]
            shares: dict[str, list[float]] = defaultdict(list)
            seconds: dict[str, list[float]] = defaultdict(list)
            for c in instr_c:
                total = 0.0
                for stage, value in c["exclusive_stages"].items():
                    total += value["exclusive_seconds"]
                    seconds[stage].append(value["exclusive_seconds"])
                    shares[stage].append(value["exclusive_seconds"] / c["wall_seconds"])
                seconds["other"].append(c["wall_seconds"] - total)
                shares["other"].append((c["wall_seconds"] - total) / c["wall_seconds"])
            entry["instrumented_wall_seconds_median"] = _median(walls)
            entry["exclusive_stage_seconds_median"] = {k: _median(v) for k, v in seconds.items()}
            entry["exclusive_stage_share_median"] = {k: _median(v) for k, v in shares.items()}
            share = entry["exclusive_stage_share_median"]
            entry["digest_plus_v8_share"] = (share.get("digest") or 0.0) + (share.get("v8") or 0.0)
            entry["profiler_overhead_ratio"] = (_median(walls) / wall) if wall else None
        per_case[name] = entry
    pooled_stage: dict[str, Any] = {}
    if instr:
        totals: dict[str, list[float]] = defaultdict(list)
        for r in instr:
            sums: dict[str, float] = defaultdict(float)
            wall_sum = sum(c["wall_seconds"] for c in r["cases"])
            for c in r["cases"]:
                for stage, value in c["exclusive_stages"].items():
                    sums[stage] += value["exclusive_seconds"]
            sums["other"] = wall_sum - sum(sums.values())
            for stage, value in sums.items():
                totals[stage].append(value / wall_sum)
        pooled_stage = {k: _median(v) for k, v in totals.items()}
    long_cases = [v for v in per_case.values() if v["body"] == "long"]
    long_dv8 = [v.get("digest_plus_v8_share") for v in long_cases]
    digest_long = [(v.get("exclusive_stage_share_median") or {}).get("digest") for v in long_cases]
    d1 = None
    if instr and all(x is not None for x in long_dv8):
        if all(x >= 0.40 for x in long_dv8):
            verdict = "S1_AND_S2_CONTINUE"
        elif all(x is not None and x >= 0.25 for x in digest_long):
            verdict = "S2_CUT_S1_CONTINUES"
        else:
            verdict = "S2_CUT_S1_CUT"
        d1 = {
            "rule": "digest+v8 >= 40% of worker time on all mixes with long bodies; "
            "else S2 cut and S1 continues only if digest alone >= 25%",
            "long_body_digest_plus_v8_share": {
                v["mix"]: v.get("digest_plus_v8_share") for v in long_cases
            },
            "long_body_digest_share": {
                v["mix"]: (v.get("exclusive_stage_share_median") or {}).get("digest")
                for v in long_cases
            },
            "verdict_on_this_host": verdict,
        }
    events: dict[str, int] = defaultdict(int)
    for c in coverage[0]["cases"] if coverage else []:
        for k, v in (c.get("events") or {}).items():
            events[k] += v
    searches = sum(v["shadow_searches"] for v in per_case.values())
    return {
        "identity": identity,
        "fixture_coverage": {
            "events_total": dict(events),
            "doom_searches_total": searches,
            "mixes": sorted({v["mix"] for v in per_case.values()}),
            "long_body_cases": [k for k, v in per_case.items() if v["body"] == "long"],
            "requirements_met": {
                "three_mixes": len({v["mix"] for v in per_case.values()}) == 3,
                "long_bodies_ge_800": max(LONG_LENGTHS) >= 800,
                "opponent_deaths": events.get("opponent_deaths", 0) > 0,
                "opponent_respawns": events.get("opponent_respawns", 0) > 0,
                "cold_forks_and_reforks": events.get("shadow_forks", 0) > len(per_case),
                "doom_search": searches > 0,
            },
        },
        "whole_workload": {
            "worlds_per_round": worlds,
            "active_row_steps_per_round": rows,
            "plain_round_wall_seconds": plain_walls,
            "plain_round_wall_seconds_median": median_plain,
            "completed_worlds_per_hour": (worlds * 3600 / median_plain) if median_plain else None,
            "active_row_steps_per_second": (rows / median_plain) if median_plain else None,
            "ms_per_active_row_step": (1000 * median_plain / rows) if median_plain else None,
            "coverage_round_wall_seconds": [r["wall_seconds"] for r in coverage],
            "instrumented_round_wall_seconds": [r["wall_seconds"] for r in instr],
        },
        "pooled_exclusive_stage_share_median": pooled_stage,
        "per_case": per_case,
        "d1": d1,
    }


def runtime_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "nproc": os.cpu_count(),
    }
    try:
        if sys.platform == "darwin":
            info["cpu_brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            info["load_average"] = list(os.getloadavg())
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu_brand"] = line.split(":", 1)[1].strip()
                    break
            info["load_average"] = list(os.getloadavg())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("S0_SOURCE_COMMIT")


def write_completed(path: Path, value: Mapping[str, Any]) -> None:
    """Create-only payload, SHA-256 sidecar, then completion marker."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
    with path.with_name(path.name + ".sha256").open("x") as handle:
        handle.write(f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n")
    with path.with_name(path.name + ".complete").open("x") as handle:
        handle.write("complete\n")


def load_reference(path: Path) -> dict[str, Any]:
    reference = json.loads(path.read_text())
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"not an S0 reference: {path}")
    return reference


def fixture_identity() -> dict[str, Any]:
    return {
        "namespace": NAMESPACE,
        "mixes": list(MIXES),
        "bodies": list(BODIES),
        "frames": FRAMES,
        "horizon": HORIZON,
        "shadow_depth": SHADOW_DEPTH,
        "beta": BETA,
        "long_lengths": list(LONG_LENGTHS),
        "world_ids": [c["row"]["world_id"] for c in fixture_cases()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-label", required=True, help="e.g. mac-contended-phase0, colab-l4")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--plain-rounds", type=int, default=3)
    parser.add_argument("--instrumented-rounds", type=int, default=2)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=3 * 3600.0)
    parser.add_argument("--work-dir", type=Path, help="world exports (default OUTPUT/work)")
    args = parser.parse_args()
    if sys.platform == "darwin" and args.processes > 2:  # Mac rule: <= 2 threads
        parser.error("Mac runs are capped at two single-thread processes")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference_doc = None if args.no_reference else load_reference(args.reference)
    if reference_doc is not None and reference_doc["fixture"] != fixture_identity():
        parser.error("reference fixture differs from this harness fixture")
    checkpoint = args.checkpoint.resolve()
    work = args.work_dir or output / "work"
    if work.exists() and any(work.iterdir()):
        parser.error(f"work directory must be fresh: {work}")
    plan = schedule(args.processes, args.plain_rounds, args.instrumented_rounds)
    started = time.time()
    reports = []
    for phase, processes in enumerate(plan):
        remaining = args.deadline_seconds - (time.time() - started)
        for report in run_processes(
            checkpoint, work / f"phase-{phase}", processes, deadline_seconds=remaining
        ):
            reports.append({"phase": phase, "concurrent_processes": len(processes), **report})
    summary = summarize(reports, None if reference_doc is None else reference_doc["files"])
    receipt = {
        "schema": SCHEMA,
        "authority": "excluded diagnostic (speed program S0 ruler); not A5, Round 0, gate, "
        "retention or serving evidence",
        "host_label": args.host_label,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "source_commit": _git_revision(),
        "engine_source": engine_source_digest(),
        "checkpoint_sha256": _sha256(checkpoint),
        "hardware": runtime_hardware(),
        "recipe": {
            "device": "cpu",
            "threads_per_process": 1,
            "processes": args.processes,
            "plan": plan,
            "resident_worlds_per_process": 1,
            "warmup_excluded": True,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
        },
        "fixture": fixture_identity(),
        "reference": (
            None
            if reference_doc is None
            else {
                "path": str(args.reference.name),
                "sha256": _sha256(args.reference),
                "source_commit": reference_doc.get("source_commit"),
                "engine_source": reference_doc.get("engine_source"),
            }
        ),
        "summary": summary,
        "processes": reports,
    }
    write_completed(output / "s0-baseline.json", receipt)
    if args.write_reference is not None:
        if not summary["identity"]["all_rounds_self_identical"]:
            raise SystemExit("refusing to write a reference from non-identical rounds")
        reference_out = {
            "schema": REFERENCE_SCHEMA,
            "fixture": fixture_identity(),
            "source_commit": receipt["source_commit"],
            "engine_source": receipt["engine_source"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "host_label": args.host_label,
            "hardware": receipt["hardware"],
            "files": reports[0]["rounds"][0]["hashes"],
        }
        args.write_reference.write_text(json.dumps(reference_out, indent=2, sort_keys=True) + "\n")
    whole = summary["whole_workload"]
    print(
        json.dumps(
            {
                "receipt": str(output / "s0-baseline.json"),
                "plain_round_wall_seconds_median": whole["plain_round_wall_seconds_median"],
                "completed_worlds_per_hour": whole["completed_worlds_per_hour"],
                "self_identical": summary["identity"]["all_rounds_self_identical"],
                "match_reference": summary["identity"]["all_rounds_match_reference"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
