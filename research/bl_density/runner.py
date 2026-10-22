"""Development-only density measurement collector.

This deliberately reuses the Watch selector seam used by the dense-search
fixtures.  It is neither a gate nor a source of scientific conclusions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from research.bl_ppo.exports import export_json

NAMESPACE = "bl-density-development-20261021"
HORIZON, DEPTH = 5000, 300
MIXES = ("scripted", "frozen", "mixed")
ARMS = ("A", "B50", "B70", "C")
FEATURE_NAMES = tuple(
    f"{action}_{name}"
    for action in range(6)
    for name in ("area_g", "head_risky", "landing_ok", "v5_eligible", "spacious")
) + ("q_gap", "q_max", "q_min", "q_mean", "q_std")


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cheap_prefilter(counts: np.ndarray, advisory: np.ndarray) -> bool:
    """Frozen broad screen used only to avoid unnecessary H300 continuations."""
    values = np.asarray(counts, dtype=np.int64).reshape(-1)
    mask = np.asarray(advisory, dtype=bool).reshape(-1)
    if values.shape != (3,) or mask.shape != (6,):
        raise ValueError("malformed free-space counts or advisory mask")
    return bool(values.min() < 160 or mask[:3].sum() <= 2)


def effective_q_features(q: np.ndarray, advisory: np.ndarray) -> tuple[int, list[float]]:
    """Return B8's fallback action and the frozen five Q summaries."""
    q = np.asarray(q, dtype=np.float64)
    advisory = np.asarray(advisory, dtype=bool)
    if q.shape != (6,) or advisory.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("malformed Q or advisory mask")
    allowed = advisory.copy()
    if not allowed.any():
        allowed[:3] = True
    values = q[allowed]
    ordered = np.sort(values)[::-1]
    gap = float(ordered[0] - ordered[1]) if ordered.size > 1 else 0.0
    action = int(np.where(allowed, q, -np.inf).argmax())
    return action, [
        gap,
        float(values.max()),
        float(values.min()),
        float(values.mean()),
        float(values.std()),
    ]


def retrospective_doom(death_decision: int | None, decision: int) -> bool:
    """Whether a natural fatal transition lies inside the contract's H300 window."""
    return death_decision is not None and 0 <= int(death_decision) - int(decision) < DEPTH


def _seed(arm: str, mix: str, index: int) -> int:
    raw = f"{NAMESPACE}/{arm}/{mix}/{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _expanded_specs(specs: Sequence[Sequence[str]], count: int) -> list[list[str]]:
    return [list(specs[index % len(specs)]) for index in range(count)]


@contextlib.contextmanager
def _profile_override(*, area: float | None, snakes: int) -> Iterator[None]:
    """Temporarily retain the standard profile knobs while changing only density."""
    from src.simd_env import eval_engine

    original = eval_engine._config_from_game_config

    def replacement(num_snakes: int, gamma: float, profile: Any = None) -> Any:
        base_count = int(profile.world.num_snakes) if profile is not None else num_snakes
        cfg = original(base_count, gamma, profile)
        if area is None:
            return dataclasses.replace(cfg, num_snakes=snakes)
        factor = float(area) ** 0.5
        cell = int(cfg.segment_size)
        width = max(cell, int(cfg.game_width * factor) // cell * cell)
        height = max(cell, int(cfg.game_height * factor) // cell * cell)
        return dataclasses.replace(cfg, num_snakes=snakes, game_width=width, game_height=height)

    eval_engine._config_from_game_config = replacement
    try:
        yield
    finally:
        eval_engine._config_from_game_config = original


def _geometry_distances(sim: Any) -> dict[str, int]:
    hero = sim.heads()[0, 0]
    heads = sim.heads()[0, 1:]
    head_distance = min(int(np.abs(hero - other).sum()) for other in heads)
    body_distance = 1 << 30
    for slot in range(1, sim.S):
        for offset in range(int(sim.seg_count[0, slot])):
            cell = sim._segments_at(np.full((sim.E, sim.S), offset, dtype=np.int64))[0, slot]
            body_distance = min(body_distance, int(np.abs(hero - cell).sum()))
    return {
        "nearest_opponent_head_manhattan": head_distance,
        "nearest_opponent_body_manhattan": body_distance,
    }


def _build(
    plan: dict[str, Any], b8: Path, arm: str, target: int | None
) -> tuple[Any, dict[str, Any]]:
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    specs = list(plan["specs"])
    area = {"B50": 0.5, "B70": 0.7}.get(arm)
    if area is not None:
        plan = {**plan, "specs": _expanded_specs(specs, 7)}
    with _profile_override(area=area, snakes=8 if area is not None else 6):
        world = build_world("b8_s3", plan, hero_checkpoint=b8)
    details: dict[str, Any] = {
        "sim_config": {
            key: getattr(world.sim.cfg, key)
            for key in (
                "game_width",
                "game_height",
                "segment_size",
                "num_snakes",
                "initial_food",
                "max_food",
            )
        },
        "roster_specs": plan["specs"],
    }
    cfg = details["sim_config"]
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    standard = gid._context(1)["profile"].world
    details["realized_density"] = {
        "area_fraction": (cfg["game_width"] * cfg["game_height"])
        / (standard.width * standard.height),
        "snakes_per_map_cell": cfg["num_snakes"]
        / (
            (cfg["game_width"] // cfg["segment_size"]) * (cfg["game_height"] // cfg["segment_size"])
        ),
    }
    if arm == "C":
        from research.bl_served_search.fixture_geometry import install

        details["fixture"] = install(world, int(target))
        details["fixture"].update(_geometry_distances(world.sim))
    return world, details


def _step(
    world: Any,
    decision: int,
    horizon: int,
    rows: list[dict[str, Any]],
    lengths: list[int],
    h300: list[dict[str, Any]],
) -> bool:
    """One exact prepare/observe/dispatch transition, optionally screening a root."""
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q, rollout_all
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        alive = bool(prepared.alive[0, 0])
        length = int(prepared.length[0, 0]) if alive else 0
        if alive:
            lengths.append(length)
        advisory = prepared.get_advisory_action_mask()[0, 0].astype(bool)
        obs = build_ego_raster(
            prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
        bare, q_stats = effective_q_features(q, advisory)
        stride = 50 if horizon < HORIZON else 200
        sampled = (
            alive
            and length >= 800
            and decision % stride == 0
            and decision <= min(4700, horizon - 1)
        )
        record: dict[str, Any] | None = None
        if sampled:
            counts = world.runtime.decision_free_space(
                prepared, np.array([[0, 0]], dtype=np.int64)
            )[0]
            passed = cheap_prefilter(counts, advisory)
            record = {
                "decision": decision,
                "length": length,
                "prefilter": passed,
                "free_space_counts": [int(x) for x in counts],
            }
            if passed:
                branch_cpu = time.process_time()
                outcome = rollout_all(world.clone(), [bare], DEPTH, decision, record_outcomes=True)[
                    bare
                ]
                doom = not bool(outcome["survives"])
                record.update(
                    h300_doom=doom,
                    h300_death_decision=outcome.get("death_decision"),
                    h300_cpu_seconds=time.process_time() - branch_cpu,
                    h300_frames_executed=int(outcome.get("frames_executed", DEPTH)),
                    features=[float(x) for x in obs["scalars"][0, -30:].tolist() + q_stats],
                )
                h300.append(record)
            rows.append(record)
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        if int(actions[0, 0]) != bare:
            raise RuntimeError(f"B8 dispatch/Q mismatch at decision {decision}")
        selected["alive"] = alive
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return bool(selected.get("alive", False))


def run_episode(job: dict[str, Any]) -> dict[str, Any]:
    """Run one deterministic arm/mix episode and preserve every sampled root."""
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    b8, arm, mix = Path(job["b8"]), str(job["arm"]), str(job["mix"])
    horizon = int(job["horizon"])
    target = (800, 1200, 1500)[int(job["index"]) % 3] if arm == "C" else None
    plan = dict(job["plan"])
    world, construction = _build(plan, b8, arm, target)
    setup_cpu = time.process_time() - start_cpu
    sampled: list[dict[str, Any]] = []
    h300: list[dict[str, Any]] = []
    lengths: list[int] = []
    decision = 0
    step_cpu: list[float] = []
    deadline = job.get("deadline")
    while decision < horizon and bool(world.sim.alive[0, 0]):
        if deadline is not None and time.monotonic() >= float(deadline):
            break
        tick = time.process_time()
        _step(world, decision, horizon, sampled, lengths, h300)
        step_cpu.append(time.process_time() - tick)
        decision += 1
    death = decision - 1 if not bool(world.sim.alive[0, 0]) else None
    for row in sampled:
        # A tiny horizon and a cooperative deadline have the same evidence
        # boundary: an alive original trajectory cannot label an unobserved
        # 300-transition suffix as non-doom.
        unknown = death is None and decision - int(row["decision"]) < DEPTH
        row["natural_doom"] = None if unknown else retrospective_doom(death, row["decision"])
        if (
            row["prefilter"]
            and row["natural_doom"] is not None
            and bool(row["h300_doom"]) != bool(row["natural_doom"])
        ):
            raise RuntimeError("H300 continuation disagrees with uninterrupted natural trajectory")
    cpu_seconds = time.process_time() - start_cpu
    branch_cpu = sum(float(row.get("h300_cpu_seconds", 0.0)) for row in h300)
    complete = decision >= horizon or death is not None
    live_cpu = cpu_seconds - branch_cpu - setup_cpu
    later_branch_cpu = sum(
        float(row.get("h300_cpu_seconds", 0.0)) for row in h300 if row["decision"] > 0
    )
    first_branch_cpu = sum(
        float(row.get("h300_cpu_seconds", 0.0)) for row in h300 if row["decision"] == 0
    )
    steady = sum(step_cpu[1:]) - later_branch_cpu
    return {
        "schema": "bl-density-episode/v1",
        "episode_id": f"{NAMESPACE}/{arm}/{mix}/{job['index']:03d}",
        "namespace": NAMESPACE,
        "arm": arm,
        "mix": mix,
        "seed": plan["seed"],
        "plan": plan,
        "sample_stride": 50 if horizon < HORIZON else 200,
        "horizon": horizon,
        "complete": complete,
        "decisions": decision,
        "hero_decisions": decision,
        "death_decision": death,
        "lengths": lengths,
        "sampled": sampled,
        "h300_calls": len(h300),
        "h300_branch_decisions": sum(int(row.get("h300_frames_executed", DEPTH)) for row in h300),
        "h300_cpu_seconds": branch_cpu,
        "world_setup_cpu_seconds": setup_cpu,
        "live_collection_cpu_seconds": live_cpu,
        "live_step_cpu_seconds": live_cpu,
        "first_step_cpu_seconds": max(0.0, step_cpu[0] - first_branch_cpu) if step_cpu else 0.0,
        "steady_live_step_cpu_seconds": max(0.0, steady),
        "steady_live_decisions": max(0, len(step_cpu) - 1),
        "construction": construction,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": time.perf_counter() - start_wall,
    }


def _verify_metrics() -> dict[str, str]:
    metric = Path(__file__).with_name("METRICS.md")
    digest = _sha(metric)
    expected = os.environ.get("BL_DENSITY_METRICS_SHA256")
    commit = os.environ.get("BL_DENSITY_COMMIT", "HEAD")
    if expected and expected != digest:
        raise RuntimeError("METRICS.md SHA does not match supervisor binding")
    try:
        frozen = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS.md"], text=False
        )
    except subprocess.CalledProcessError as exc:
        if expected:
            frozen = metric.read_bytes()
        else:
            raise RuntimeError("committed METRICS.md unavailable before gameplay") from exc
    if hashlib.sha256(frozen).hexdigest() != digest:
        raise RuntimeError("working metric contract differs from committed binding")
    amendment = Path(__file__).with_name("METRICS_AMENDMENT_01.md")
    amendment_digest = _sha(amendment)
    expected_amendment = os.environ.get("BL_DENSITY_AMENDMENT_SHA256")
    if expected_amendment and expected_amendment != amendment_digest:
        raise RuntimeError("METRICS_AMENDMENT_01.md SHA does not match supervisor binding")
    try:
        frozen_amendment = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS_AMENDMENT_01.md"], text=False
        )
    except subprocess.CalledProcessError:
        if expected_amendment:
            frozen_amendment = amendment.read_bytes()
        else:
            raise RuntimeError("committed metric amendment unavailable before gameplay")
    if hashlib.sha256(frozen_amendment).hexdigest() != amendment_digest:
        raise RuntimeError("working metric amendment differs from committed binding")
    capacity = Path(__file__).with_name("METRICS_AMENDMENT_02.md")
    capacity_digest = _sha(capacity)
    expected_capacity = os.environ.get("BL_DENSITY_CAPACITY_AMENDMENT_SHA256")
    if expected_capacity and expected_capacity != capacity_digest:
        raise RuntimeError("METRICS_AMENDMENT_02.md SHA does not match supervisor binding")
    try:
        frozen_capacity = subprocess.check_output(
            ["git", "show", f"{commit}:research/bl_density/METRICS_AMENDMENT_02.md"], text=False
        )
    except subprocess.CalledProcessError:
        if expected_capacity:
            frozen_capacity = capacity.read_bytes()
        else:
            raise RuntimeError("committed capacity amendment unavailable before gameplay")
    if hashlib.sha256(frozen_capacity).hexdigest() != capacity_digest:
        raise RuntimeError("working capacity amendment differs from committed binding")
    return {
        "metrics_sha256": digest,
        "metrics_amendment_sha256": amendment_digest,
        "metrics_capacity_amendment_sha256": capacity_digest,
        "metrics_commit": commit,
    }


def planned_jobs(
    context: dict[str, Any],
    *,
    b8: Path,
    arms: Sequence[str],
    mixes: Sequence[str],
    episodes: int,
    horizon: int,
    tiny: bool,
    deadline: float | None,
) -> list[dict[str, Any]]:
    """Materialize each entire cell before execution, preserving pool rotation."""
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    jobs = []
    for arm in arms:
        for mix in mixes:
            seeds = [_seed(arm, mix, index) for index in range(episodes)]
            plans = planned_rosters(context, seeds, [mix] * len(seeds))
            for index, plan in enumerate(plans):
                jobs.append(
                    {
                        "b8": str(b8),
                        "arm": arm,
                        "mix": mix,
                        "index": index,
                        "plan": plan,
                        "horizon": horizon,
                        "tiny": tiny,
                        "deadline": deadline,
                    }
                )
    if len({job["plan"]["seed"] for job in jobs}) != len(jobs):
        raise RuntimeError("density namespace has an internal seed collision")
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b8", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes-per-cell", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--mixes", default=",".join(MIXES))
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    if args.tiny:
        args.horizon, args.episodes_per_cell, args.workers = (
            min(args.horizon, 8),
            min(args.episodes_per_cell, 1),
            1,
        )
    elif args.horizon != HORIZON or args.episodes_per_cell != 6:
        raise SystemExit("production allocation is frozen at H5000 and six episodes per cell")
    if args.workers < 1 or args.workers > 6 or args.episodes_per_cell < 1:
        raise SystemExit("invalid bounded worker/allocation request")
    if (
        args.horizon < 1
        or args.max_seconds is not None
        and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)
    ):
        raise SystemExit("horizon and work bound must be finite and positive")
    binding = _verify_metrics()
    arms, mixes = tuple(args.arms.split(",")), tuple(args.mixes.split(","))
    if _sha(args.b8) != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise SystemExit("--b8 does not match the frozen B8 s3 checkpoint digest")
    if set(arms) - set(ARMS) or set(mixes) - set(MIXES):
        raise SystemExit("unknown arm or mix")
    if len(set(arms)) != len(arms) or len(set(mixes)) != len(mixes):
        raise SystemExit("duplicate arm or mix")
    if not args.tiny and (arms != ARMS or mixes != MIXES):
        raise SystemExit("production requires all frozen arms and mixes")
    if args.out.exists():
        raise SystemExit("--out must be a fresh directory")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    parent_cpu = time.process_time()
    records = []
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    jobs = planned_jobs(
        gid._context(1),
        b8=args.b8,
        arms=arms,
        mixes=mixes,
        episodes=args.episodes_per_cell,
        horizon=args.horizon,
        tiny=bool(args.tiny),
        deadline=(started + args.max_seconds) if args.max_seconds is not None else None,
    )
    child_before = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as pool:
        pending: dict[Any, dict[str, Any]] = {}
        cursor = 0
        while cursor < len(jobs) or pending:
            while cursor < len(jobs) and len(pending) < args.workers:
                if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds:
                    cursor = len(jobs)
                    break
                future = pool.submit(run_episode, jobs[cursor])
                pending[future] = jobs[cursor]
                cursor += 1
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED
            )
            if not done:
                print(
                    json.dumps(
                        {
                            "event": "bl_density_progress",
                            "completed": len(records),
                            "scheduled": len(jobs),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
                continue
            for future in done:
                pending.pop(future)
                record = future.result()
                record.update(binding)
                records.append(record)
                export_json(
                    args.out / "episodes" / f"{record['episode_id'].replace('/', '_')}.json", record
                )
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "event": "bl_density_progress",
                            "completed": len(records),
                            "scheduled": len(jobs),
                            "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed / len(records) * (len(jobs) - len(records)),
                        }
                    ),
                    flush=True,
                )
    # Completion order is deliberately unconstrained by process scheduling;
    # every downstream descriptive calculation receives the fixed job order.
    records.sort(key=lambda record: str(record["episode_id"]))
    status = (
        "COMPLETE"
        if len(records) == len(jobs) and all(r["complete"] for r in records)
        else "INCOMPLETE"
    )
    child_after = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    )
    analysis_started = time.process_time()
    from research.bl_density.analysis import summarize

    analysis = summarize(records, FEATURE_NAMES)
    analysis_cpu = time.process_time() - analysis_started
    episode_cpu = sum(float(r["cpu_seconds"]) for r in records)
    parent_total_cpu = time.process_time()
    end_to_end_cpu = parent_total_cpu + child_after
    complete_cpu = analysis["aggregate_complete_episode_metrics"]["process_cpu_seconds"]
    complete_count = analysis["aggregate_complete_episode_metrics"]["episodes"]
    overhead = max(0.0, end_to_end_cpu - complete_cpu)
    for arm_cells in analysis["cells"].values():
        for cell in arm_cells.values():
            allocated = overhead * cell["episodes"] / complete_count if complete_count else 0.0
            denominator = cell["process_cpu_seconds"] + allocated
            cell["attributed_end_to_end_cpu_seconds"] = denominator
            cell["doom_moments_per_end_to_end_cpu_second"] = (
                cell["doom_moments"] / denominator if denominator else None
            )
    summary = {
        "schema": "bl-density-run/v1",
        "status": status,
        "development_only": True,
        "tiny_preflight": bool(args.tiny),
        **binding,
        "b8_sha256": _sha(args.b8),
        "scheduled_episodes": len(jobs),
        "completed_episodes": len(records),
        "records": [r["episode_id"] for r in records],
        "parent_analysis_cpu_seconds": analysis_cpu,
        "parent_cpu_seconds": parent_total_cpu,
        "parent_precollection_cpu_seconds": parent_cpu,
        "end_to_end_cpu_seconds": end_to_end_cpu,
        "doom_moments_per_end_to_end_cpu_second": (
            analysis["aggregate_complete_episode_metrics"]["doom_moments"] / end_to_end_cpu
            if end_to_end_cpu
            else None
        ),
        "overhead_attribution": (
            "non-episode and partial CPU allocated equally per completed episode"
        ),
        "child_cpu_seconds": child_after - child_before,
        "worker_startup_import_cpu_seconds": max(0.0, child_after - child_before - episode_cpu),
        **analysis,
    }
    export_json(args.out / "summary.json", summary)
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
