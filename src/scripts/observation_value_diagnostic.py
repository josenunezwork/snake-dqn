#!/usr/bin/env python3
"""Create-only H1-A geometric remote-heading observation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _thread_key in _THREAD_KEYS:
    os.environ.setdefault(_thread_key, "1")

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from src.core.seeding import derive_seed  # noqa: E402

SCHEMA, RUN_SCHEMA = "observation-value-freeze/v1", "observation-value-run/v1"
WORLD_COUNT, DEVELOPMENT_WORLDS, MAX_STEPS, MAX_PAIRS = 24, 8, 256, 72
FRAMES, HORIZONS, PRIMARY_HORIZON, SNAKES, TAPE_STEPS = (16, 80, 160), (1, 8, 16, 32), 16, 6, 32
LIMITS = {
    "wall_seconds": 840.0,
    "rss_bytes": 2 * 1024**3,
    "available_bytes": 6 * 1024**3,
    "agent_slots": 202752,
    "cpu_threads": 1,
}


class ResourceStop(RuntimeError):
    """A local resource or wall limit was reached before completion."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical(_jsonable(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_closure(repo: Path) -> dict[str, str]:
    return {
        str(path.relative_to(repo)): sha256_file(path)
        for path in sorted([*repo.glob("src/**/*.py"), *repo.glob("configs/*.yaml")])
        if path.is_file()
    }


def digest_without(value: Mapping[str, Any], key: str) -> str:
    copied = dict(value)
    copied.pop(key, None)
    return hashlib.sha256(canonical(copied)).hexdigest()


def _seed_series(root: int, label: str, count: int) -> list[int]:
    return [derive_seed(root, f"H1-A/{label}/{index}") for index in range(count)]


def resolved_protocol(source_revision: str, seed_root: int, repo: Path) -> dict[str, Any]:
    if isinstance(seed_root, bool) or not isinstance(seed_root, int) or not 0 <= seed_root < 2**64:
        raise ValueError("seed root must be uint64")
    from src.evaluation.observation_probe import ProbeConfig
    from src.simd_env.batch_sim import BatchSimConfig

    world = _seed_series(seed_root, "world", WORLD_COUNT)
    policy = [
        _seed_series(seed_root, f"policy/world-{index:02d}", SNAKES) for index in range(WORLD_COUNT)
    ]
    pair_keys = [
        f"world-{index:02d}/frame-{frame}" for index in range(WORLD_COUNT) for frame in FRAMES
    ]
    tapes = {key: derive_seed(seed_root, f"H1-A/tape/{key}") for key in pair_keys}
    bootstrap = derive_seed(seed_root, "H1-A/bootstrap")
    values = [
        seed_root,
        bootstrap,
        *world,
        *(seed for row in policy for seed in row),
        *tapes.values(),
    ]
    if len(values) != len(set(values)):
        raise RuntimeError("C0 seed derivation collision")
    config = BatchSimConfig(num_envs=1, num_snakes=SNAKES, mechanics_version=2, gamma=0.99)
    return {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "repo_root": str(repo.resolve()),
        "source_closure": source_closure(repo),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "probe_config": asdict(
            ProbeConfig(max_frames=5000, starvation_max=500, max_length=400, gamma=0.99)
        ),
        "batch_config": asdict(config),
        "collection": {
            "world_count": WORLD_COUNT,
            "development_world_count": DEVELOPMENT_WORLDS,
            "frames": list(FRAMES),
            "max_steps_per_world": MAX_STEPS,
            "max_pairs": MAX_PAIRS,
            "horizons": list(HORIZONS),
            "primary_horizon": PRIMARY_HORIZON,
            "train_mode": True,
            "allow_respawn": False,
            "roster": ["RandomSafeSimdPolicy", "GreedyFoodSimdPolicy"] * 3,
            "hero_policy": "RandomSafeSimdPolicy",
            "normal_relative_action_probabilities": [0.1, 0.8, 0.1],
            "observation_spec": "raster31v3",
            "require_equal_observation_and_resolved_mask": True,
            "geometric_twin_only": True,
        },
        "statistics": {
            "tie_margin": 0.01,
            "common_action_regret_threshold": 0.01,
            "bootstrap_samples": 2000,
            "min_eligible_worlds": 8,
            "min_disjoint_worlds": 3,
            "holdout_gate": "heldout_primary_horizon_world_clustered_only",
        },
        "limits": LIMITS,
        "thread_environment": {key: os.environ.get(key) for key in _THREAD_KEYS},
        "seeds": {
            "root": seed_root,
            "world": world,
            "policy": policy,
            "tape": tapes,
            "bootstrap": bootstrap,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def _clean_revision(repo: Path, revision: str) -> bool:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip() == revision and not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )


def freeze(out_dir: Path, source_revision: str, seed_root: int) -> Path:
    if not out_dir.is_absolute() or out_dir.exists():
        raise FileExistsError("freeze output must be a new absolute directory")
    if not _clean_revision(_REPO_ROOT, source_revision):
        raise RuntimeError("source revision is not the clean frozen checkout")
    if any(os.environ.get(key) != "1" for key in _THREAD_KEYS):
        raise RuntimeError("diagnostic requires one numerical thread")
    manifest = resolved_protocol(source_revision, seed_root, _REPO_ROOT)
    manifest["manifest_digest"] = digest_without(manifest, "manifest_digest")
    out_dir.mkdir(parents=True)
    path = out_dir / "manifest.json"
    _atomic_json(path, manifest)
    return path


def verify_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise RuntimeError("manifest path or SHA differs from command pin")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("manifest_digest") != digest_without(
        manifest, "manifest_digest"
    ):
        raise RuntimeError("manifest schema or digest mismatch")
    if manifest.get("script", {}).get("path") != str(Path(__file__).resolve()) or manifest[
        "script"
    ].get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("manifest executable script identity differs")
    repo = Path(manifest["repo_root"])
    if repo.resolve() != _REPO_ROOT or not _clean_revision(repo, manifest["source_revision"]):
        raise RuntimeError("source revision is not the clean frozen checkout")
    if (
        manifest.get("python") != platform.python_version()
        or manifest.get("numpy") != np.__version__
    ):
        raise RuntimeError("runtime version drift")
    if any(os.environ.get(key) != "1" for key in _THREAD_KEYS):
        raise RuntimeError("diagnostic requires one numerical thread")
    if source_closure(repo) != manifest["source_closure"]:
        raise RuntimeError("frozen source closure drift")
    expected = resolved_protocol(manifest["source_revision"], manifest["seeds"]["root"], repo)
    for key in (
        "probe_config",
        "batch_config",
        "collection",
        "statistics",
        "limits",
        "seeds",
        "thread_environment",
    ):
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"approved semantic projection drift: {key}")
    return manifest


def _obs_digest(obs: Mapping[str, np.ndarray]) -> str:
    return hashlib.sha256(canonical(_jsonable(dict(obs)))).hexdigest()


def _tape(seed: int, steps: int, snakes: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(
        np.array([0, 1, 2], dtype=np.int64), size=(steps, 1, snakes), p=(0.1, 0.8, 0.1)
    )


def _persist_new(path: Path, value: Any) -> str:
    with path.open("xb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _roster_actions(sim: Any, policies: list[Any]) -> np.ndarray:
    masks, actions = sim.get_resolved_action_mask()[0], np.ones((1, sim.S), dtype=np.int64)
    for snake, policy in enumerate(policies):
        actions[0, snake] = int(
            policy.actions(masks[snake : snake + 1], sim, np.array([[0, snake]], dtype=np.int64))[0]
        )
    return actions


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _available_bytes() -> int:
    try:
        return int(psutil.virtual_memory().available)
    except (psutil.Error, OSError) as exc:
        raise ResourceStop(f"available_memory_unavailable:{type(exc).__name__}") from exc


def _resource_snapshot(started: float, counters: Mapping[str, int]) -> dict[str, Any]:
    return {
        "elapsed_seconds": time.monotonic() - started,
        "rss_bytes": _rss_bytes(),
        "available_bytes": _available_bytes(),
        "counters": dict(counters),
    }


def _check_resources(started: float, counters: Mapping[str, int]) -> dict[str, Any]:
    snapshot = _resource_snapshot(started, counters)
    if snapshot["elapsed_seconds"] > LIMITS["wall_seconds"]:
        raise ResourceStop("wall_seconds")
    if snapshot["rss_bytes"] > LIMITS["rss_bytes"]:
        raise ResourceStop("rss_bytes")
    if snapshot["available_bytes"] < LIMITS["available_bytes"]:
        raise ResourceStop("available_bytes")
    if (
        counters["natural_agent_slots"] + counters["counterfactual_agent_slots"]
        > LIMITS["agent_slots"]
    ):
        raise ResourceStop("agent_slots")
    return snapshot


def _candidate(world_id: str, frame: int, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "candidate",
        "world_id": world_id,
        "frame": frame,
        "status": "rejected",
        "reason": reason,
        **extra,
    }


def _values(result: Mapping[str, Any]) -> list[float | None]:
    return [
        (
            None
            if result["actions"][action] is None
            else result["actions"][action]["discounted_return_by_horizon"][PRIMARY_HORIZON]
        )
        for action in range(6)
    ]


def run(manifest_path: Path, manifest_sha256: str) -> Path:
    """Collect exact-frame candidates; a partial terminal makes ``main`` fail."""
    manifest = verify_manifest(manifest_path, manifest_sha256)
    from src.evaluation.observation_probe import (
        ProbeConfig,
        clone_sim,
        finite_action_returns,
        full_state_digest,
        heading_twin,
        observe_hero,
    )
    from src.evaluation.observation_probe_stats import (
        aggregate_world_metrics,
        pair_metrics,
    )
    from src.simd_env.batch_sim import BatchSim, BatchSimConfig
    from src.simd_env.eval_engine import GreedyFoodSimdPolicy, RandomSafeSimdPolicy

    out = manifest_path.parent / "run"
    if out.exists() or (manifest_path.parent / "terminal.json").exists():
        raise FileExistsError("run output or terminal already exists")
    out.mkdir()
    raw, heart = out / "raw.jsonl", out / "heartbeat.jsonl"
    counters = {
        "worlds": 0,
        "natural_world_ticks": 0,
        "natural_agent_slots": 0,
        "natural_valid_transition_agent_slots": 0,
        "counterfactual_world_ticks": 0,
        "counterfactual_agent_slots": 0,
        "counterfactual_valid_transition_agent_slots": 0,
        "pairs": 0,
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    active_world: str | None = None
    active_reached: set[int] = set()
    started, status, cause = time.monotonic(), "completed", "natural_collection_complete"
    try:
        probe = ProbeConfig(**manifest["probe_config"])
        config = BatchSimConfig(**manifest["batch_config"])
        for index, world_seed in enumerate(manifest["seeds"]["world"]):
            verify_manifest(manifest_path, manifest_sha256)
            _check_resources(started, counters)
            world_id, reached, end_reason = f"world-{index:02d}", set(), "step_cap"
            active_world, active_reached = world_id, reached
            sim = BatchSim(config, seeds=[world_seed], train_mode=True, allow_respawn=False)
            policies = [
                (
                    RandomSafeSimdPolicy(manifest["seeds"]["policy"][index][slot])
                    if slot % 2 == 0
                    else GreedyFoodSimdPolicy()
                )
                for slot in range(SNAKES)
            ]
            for _ in range(MAX_STEPS):
                _check_resources(started, counters)
                frame = int(sim.frame[0])
                if frame in FRAMES:
                    reached.add(frame)
                    hero = 0
                    enemy = next(
                        (
                            slot
                            for slot in range(1, SNAKES)
                            if sim.alive[0, slot] and int(sim.seg_count[0, slot]) == 1
                        ),
                        None,
                    )
                    old = int(sim.direction[0, enemy]) if enemy is not None else None
                    row = _candidate(
                        world_id,
                        frame,
                        "unknown",
                        hero_alive=bool(sim.alive[0, hero]),
                        enemy=enemy,
                        old_heading=old,
                    )
                    if not sim.alive[0, hero]:
                        row["reason"] = "hero_dead"
                    elif sim.population_floor_reached()[0]:
                        row["reason"] = "initial_population_floor"
                    elif frame >= probe.max_frames:
                        row["reason"] = "initial_max_frames"
                    elif enemy is None:
                        row["reason"] = "no_lowest_alive_single_segment_enemy"
                    else:
                        twin = heading_twin(sim, hero, enemy, (old + 1) % 4, probe)
                        if twin is None:
                            row["reason"] = "heading_twin_observation_or_mask_changed"
                        elif counters["pairs"] >= MAX_PAIRS:
                            raise ResourceStop("pair_budget")
                        else:
                            _check_resources(started, counters)
                            pair_id = f"{world_id}-f{frame}"
                            tape_seed = manifest["seeds"]["tape"][f"{world_id}/frame-{frame}"]
                            tape = _tape(tape_seed, TAPE_STEPS, SNAKES)
                            pair_dir = out / "pairs"
                            pair_dir.mkdir(exist_ok=True)
                            left_path, right_path, tape_path = (
                                pair_dir / f"{pair_id}-{kind}.pkl"
                                for kind in ("left", "right", "tape")
                            )
                            left_hash = _persist_new(left_path, clone_sim(sim))
                            right_hash = _persist_new(right_path, clone_sim(twin))
                            tape_hash = _persist_new(tape_path, tape)
                            _append_jsonl(
                                raw,
                                {
                                    "kind": "branch_inputs",
                                    "world_id": world_id,
                                    "pair_id": pair_id,
                                    "status": "persisted",
                                    "left_snapshot_path": str(left_path),
                                    "left_snapshot_sha256": left_hash,
                                    "right_snapshot_path": str(right_path),
                                    "right_snapshot_sha256": right_hash,
                                    "tape_path": str(tape_path),
                                    "tape_sha256": tape_hash,
                                },
                            )
                            left, right = finite_action_returns(
                                sim, hero, tape, HORIZONS, probe
                            ), finite_action_returns(twin, hero, tape, HORIZONS, probe)
                            left_values, right_values = _values(left), _values(right)
                            steps = sum(
                                item["actual_steps"]
                                for item in left["actions"].values()
                                if item is not None
                            ) + sum(
                                item["actual_steps"]
                                for item in right["actions"].values()
                                if item is not None
                            )
                            counters["counterfactual_world_ticks"] += steps
                            counters["counterfactual_agent_slots"] += steps * SNAKES
                            counters["counterfactual_valid_transition_agent_slots"] += sum(
                                item["valid_agent_transitions"]
                                for result in (left, right)
                                for item in result["actions"].values()
                                if item is not None
                            )
                            _check_resources(started, counters)
                            row.update(
                                {
                                    "status": "accepted",
                                    "reason": None,
                                    "pair_id": pair_id,
                                    "new_heading": (old + 1) % 4,
                                    "geometric_twin_only": True,
                                    "left_snapshot_path": str(left_path),
                                    "left_snapshot_sha256": left_hash,
                                    "right_snapshot_path": str(right_path),
                                    "right_snapshot_sha256": right_hash,
                                    "tape_path": str(tape_path),
                                    "tape_sha256": tape_hash,
                                    "tape_seed": tape_seed,
                                    "left_obs_digest": _obs_digest(observe_hero(sim, hero, probe)),
                                    "right_obs_digest": _obs_digest(
                                        observe_hero(twin, hero, probe)
                                    ),
                                    "left_state_digest": full_state_digest(sim),
                                    "right_state_digest": full_state_digest(twin),
                                    "initial_mask": left["initial_mask"],
                                    "left_returns": left,
                                    "right_returns": right,
                                    "metrics": pair_metrics(
                                        left_values,
                                        right_values,
                                        tie_margin=manifest["statistics"]["tie_margin"],
                                    ),
                                }
                            )
                            records.append(
                                {
                                    "world_id": world_id,
                                    "pair_id": pair_id,
                                    "metrics": row["metrics"],
                                }
                            )
                            counters["pairs"] += 1
                    _append_jsonl(raw, row)
                    _append_jsonl(
                        heart,
                        {
                            "world_id": world_id,
                            "frame": frame,
                            **_resource_snapshot(started, counters),
                        },
                    )
                if sim.population_floor_reached()[0]:
                    end_reason = "population_floor"
                    break
                if frame >= probe.max_frames:
                    end_reason = "max_frames"
                    break
                sim.step(_roster_actions(sim, policies))
                counters["natural_valid_transition_agent_slots"] += int(
                    sim.get_transition_valid()[0].sum()
                )
                counters["natural_world_ticks"] += 1
                counters["natural_agent_slots"] += SNAKES
            for frame in FRAMES:
                if frame not in reached:
                    _append_jsonl(
                        raw, _candidate(world_id, frame, "not_reached", world_end_reason=end_reason)
                    )
            counters["worlds"] += 1
            seen.add(world_id)
            active_world = None
            _append_jsonl(
                heart,
                {
                    "world_id": world_id,
                    "event": "world_complete",
                    "world_end_reason": end_reason,
                    **_resource_snapshot(started, counters),
                },
            )
        if len(seen) != WORLD_COUNT:
            raise RuntimeError("missing world completion")
        expected = [f"world-{index:02d}" for index in range(WORLD_COUNT)]
        stat_args = {
            "bootstrap_seed": manifest["seeds"]["bootstrap"],
            "bootstrap_samples": manifest["statistics"]["bootstrap_samples"],
            "regret_threshold": manifest["statistics"]["common_action_regret_threshold"],
            "min_eligible_worlds": manifest["statistics"]["min_eligible_worlds"],
            "min_disjoint_worlds": manifest["statistics"]["min_disjoint_worlds"],
        }

        def aggregate_partition(world_ids: list[str]) -> dict[str, Any]:
            if not world_ids:
                return {"status": "not_applicable_empty_partition"}
            return aggregate_world_metrics(
                [item for item in records if item["world_id"] in world_ids],
                expected_world_ids=world_ids,
                **stat_args,
            )

        development = aggregate_partition(expected[:DEVELOPMENT_WORLDS])
        holdout = aggregate_partition(expected[DEVELOPMENT_WORLDS:])
        verify_manifest(manifest_path, manifest_sha256)
    except ResourceStop as exc:
        status, cause, development, holdout = "partial", str(exc), None, None
    except BaseException as exc:
        status, cause, development, holdout = "failed", f"{type(exc).__name__}: {exc}", None, None
    if status != "completed":
        for index in range(WORLD_COUNT):
            world_id = f"world-{index:02d}"
            if world_id in seen:
                continue
            current = world_id == active_world
            _append_jsonl(
                raw,
                {
                    "kind": "world",
                    "world_id": world_id,
                    "status": "partial" if current else "unrun",
                    "reason": status,
                },
            )
            for frame in FRAMES:
                if current and frame in active_reached:
                    continue
                _append_jsonl(raw, _candidate(world_id, frame, "unrun", run_status=status))
    try:
        final_resource = _resource_snapshot(started, counters)
    except ResourceStop as exc:
        final_resource = {"resource_error": str(exc), "counters": dict(counters)}
    terminal = {
        "schema": RUN_SCHEMA,
        "status": status,
        "cause": cause,
        "manifest_sha256": manifest_sha256,
        "counters": counters,
        "raw_path": str(raw),
        "raw_sha256": sha256_file(raw) if raw.exists() else None,
        "development": development,
        "holdout": holdout,
        "evaluation_started": False,
        "promotion_authority": False,
        "completed_world_count": len(seen),
        "resource": final_resource,
    }
    terminal["terminal_digest"] = digest_without(terminal, "terminal_digest")
    terminal_path = out / "terminal.json"
    _atomic_json(terminal_path, terminal)
    return terminal_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--out-dir", type=Path, required=True)
    freeze_parser.add_argument("--source-revision", required=True)
    freeze_parser.add_argument("--seed-root", type=int, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            print(freeze(args.out_dir, args.source_revision, args.seed_root))
            return 0
        terminal = run(args.manifest, args.manifest_sha256)
        print(terminal)
        return 0 if json.loads(terminal.read_text())["status"] == "completed" else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"observation-value diagnostic failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
