#!/usr/bin/env python3
"""Create-only, finite controlled observation-value diagnostic.

The command intentionally has no learner, checkpoint, restart, or promotion
path.  It records conditional counterfactual returns under a common action tape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "observation-value-freeze/v1"
RUN_SCHEMA = "observation-value-run/v1"
WORLD_COUNT, DEVELOPMENT_WORLDS, MAX_STEPS, MAX_PAIRS = 24, 8, 256, 72
HORIZONS = (1, 8, 16, 32)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(seed: int, namespace: str) -> int:
    return int.from_bytes(
        hashlib.sha256(seed.to_bytes(8, "big") + b"\0" + namespace.encode()).digest()[:8], "big"
    )


def source_closure(repo: Path) -> dict[str, str]:
    paths = [*repo.glob("src/**/*.py"), *repo.glob("configs/*.yaml")]
    script = Path(__file__).resolve()
    if script not in paths:
        paths.append(script)
    return {
        str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path): sha256_file(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def digest_without(value: Mapping[str, Any], key: str) -> str:
    copy = dict(value)
    copy.pop(key, None)
    return hashlib.sha256(canonical(copy)).hexdigest()


def resolved_protocol(source_revision: str, seed_root: int, repo: Path) -> dict[str, Any]:
    if isinstance(seed_root, bool) or not 0 <= seed_root < 2**64:
        raise ValueError("seed root must be uint64")
    worlds = [derive_seed(seed_root, f"H1-A/world/{index}") for index in range(WORLD_COUNT)]
    policy = [derive_seed(seed_root, f"H1-A/policy/{index}") for index in range(WORLD_COUNT)]
    tape = [derive_seed(seed_root, f"H1-A/tape/{index}") for index in range(WORLD_COUNT)]
    bootstrap = derive_seed(seed_root, "H1-A/bootstrap")
    values = [seed_root, bootstrap, *worlds, *policy, *tape]
    if len(values) != len(set(values)):
        raise RuntimeError("C0 seed derivation collision")
    from src.evaluation.observation_probe import ProbeConfig
    from src.simd_env.batch_sim import BatchSimConfig

    config = BatchSimConfig(num_envs=1, num_snakes=6, mechanics_version=2, gamma=0.99)
    return {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "repo_root": str(repo),
        "source_closure": source_closure(repo),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "probe_config": asdict(ProbeConfig()),
        "batch_config": asdict(config),
        "collection": {
            "world_count": WORLD_COUNT,
            "development_world_count": DEVELOPMENT_WORLDS,
            "max_steps_per_world": MAX_STEPS,
            "max_pairs": MAX_PAIRS,
            "horizons": list(HORIZONS),
            "train_mode": True,
            "allow_respawn": False,
            "roster": ["random_safe", "greedy_food"] * 3,
        },
        "seeds": {
            "root": seed_root,
            "world": worlds,
            "policy": policy,
            "tape": tape,
            "bootstrap": bootstrap,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def freeze(out_dir: Path, source_revision: str, seed_root: int) -> Path:
    if not out_dir.is_absolute() or out_dir.exists():
        raise FileExistsError("freeze output must be a new absolute directory")
    repo = Path(__file__).resolve().parents[2]
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if current != source_revision or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ):
        raise RuntimeError("source revision is not the clean frozen checkout")
    manifest = resolved_protocol(source_revision, seed_root, repo)
    manifest["manifest_digest"] = digest_without(manifest, "manifest_digest")
    out_dir.mkdir(parents=True)
    path = out_dir / "manifest.json"
    path.write_bytes(canonical(manifest) + b"\n")
    return path


def verify_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or sha256_file(path) != expected_sha256:
        raise RuntimeError("manifest path or SHA differs from command pin")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("manifest_digest") != digest_without(
        manifest, "manifest_digest"
    ):
        raise RuntimeError("manifest schema or digest mismatch")
    repo = Path(manifest["repo_root"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        != manifest["source_revision"]
    ):
        raise RuntimeError("source revision drift")
    if (
        source_closure(repo) != manifest["source_closure"]
        or sha256_file(Path(manifest["script"]["path"])) != manifest["script"]["sha256"]
    ):
        raise RuntimeError("frozen source closure drift")
    return manifest


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


def _obs_digest(obs: Mapping[str, np.ndarray]) -> str:
    return hashlib.sha256(canonical(_jsonable(dict(obs)))).hexdigest()


def _tape(seed: int, steps: int, snakes: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(
        np.array([0, 1, 2], dtype=np.int64), size=(steps, 1, snakes), p=(0.1, 0.8, 0.1)
    )


def _roster_actions(sim: Any, policies: list[Any]) -> np.ndarray:
    masks = sim.get_resolved_action_mask()[0]
    actions = np.ones((1, sim.S), dtype=np.int64)
    for snake, policy in enumerate(policies):
        slot = np.array([[0, snake]], dtype=np.int64)
        actions[0, snake] = int(policy.actions(masks[snake : snake + 1], sim, slot)[0])
    return actions


def _persist(path: Path, value: Any) -> str:
    path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    return sha256_file(path)


def run(manifest_path: Path, manifest_sha256: str) -> Path:
    """Collect finite controlled twins; no reset, replacement, retry, or learner."""
    manifest = verify_manifest(manifest_path, manifest_sha256)
    from src.evaluation.observation_probe import (
        ProbeConfig,
        clone_sim,
        finite_action_returns,
        full_state_digest,
        heading_twin,
        observe_hero,
    )
    from src.evaluation.observation_probe_stats import aggregate_world_metrics, pair_metrics
    from src.simd_env.batch_sim import BatchSim, BatchSimConfig
    from src.simd_env.eval_engine import GreedyFoodSimdPolicy, RandomSafeSimdPolicy

    out = manifest_path.parent / "run"
    if out.exists() or (manifest_path.parent / "terminal.json").exists():
        raise FileExistsError("run output or terminal already exists")
    out.mkdir()
    raw_path = out / "raw.jsonl"
    heartbeat = out / "heartbeat.jsonl"
    probe = ProbeConfig(**manifest["probe_config"])
    config = BatchSimConfig(**manifest["batch_config"])
    records: list[dict[str, Any]] = []
    counters = {"worlds": 0, "steps": 0, "pairs": 0}
    started = time.monotonic()
    status, cause = "completed", "natural_collection_complete"
    try:
        for index, world_seed in enumerate(manifest["seeds"]["world"]):
            if counters["steps"] >= WORLD_COUNT * MAX_STEPS or counters["pairs"] >= MAX_PAIRS:
                status, cause = "partial", "budget_cap"
                break
            world_id = f"world-{index:02d}"
            sim = BatchSim(config, seeds=[world_seed], train_mode=True, allow_respawn=False)
            policy_seed = manifest["seeds"]["policy"][index]
            policies = [
                (
                    RandomSafeSimdPolicy(policy_seed + slot)
                    if slot % 2 == 0
                    else GreedyFoodSimdPolicy()
                )
                for slot in range(6)
            ]
            for _ in range(MAX_STEPS):
                frame = int(sim.frame[0])
                if frame in (16, 80, 160):
                    hero = 0
                    enemy = next(
                        (
                            slot
                            for slot in range(1, 6)
                            if sim.alive[0, slot] and int(sim.seg_count[0, slot]) == 1
                        ),
                        None,
                    )
                    old = int(sim.direction[0, enemy]) if enemy is not None else None
                    twin = (
                        None
                        if enemy is None
                        else heading_twin(sim, hero, enemy, (old + 1) % 4, probe)
                    )
                    row: dict[str, Any] = {
                        "kind": "candidate",
                        "world_id": world_id,
                        "frame": frame,
                        "hero_alive": bool(sim.alive[0, hero]),
                        "enemy": enemy,
                        "old_heading": old,
                        "status": "rejected",
                    }
                    if not sim.alive[0, hero]:
                        row["reason"] = "hero_dead"
                    elif sim.population_floor_reached()[0]:
                        row["reason"] = "initial_population_floor"
                    elif frame >= probe.max_frames:
                        row["reason"] = "initial_max_frames"
                    elif enemy is None:
                        row["reason"] = "no_lowest_alive_single_segment_enemy"
                    elif twin is None:
                        row["reason"] = "heading_twin_observation_or_mask_changed"
                    else:
                        pair_id = f"{world_id}-f{frame}"
                        tape = _tape(manifest["seeds"]["tape"][index], 32, 6)
                        left, right = finite_action_returns(
                            sim, hero, tape, HORIZONS, probe
                        ), finite_action_returns(twin, hero, tape, HORIZONS, probe)
                        left_values = [
                            (
                                None
                                if left["actions"][action] is None
                                else left["actions"][action]["discounted_return_by_horizon"][16]
                            )
                            for action in range(6)
                        ]
                        right_values = [
                            (
                                None
                                if right["actions"][action] is None
                                else right["actions"][action]["discounted_return_by_horizon"][16]
                            )
                            for action in range(6)
                        ]
                        pair_dir = out / "pairs"
                        pair_dir.mkdir(exist_ok=True)
                        left_path, right_path, tape_path = (
                            pair_dir / f"{pair_id}-{kind}.pkl" for kind in ("left", "right", "tape")
                        )
                        row.update(
                            {
                                "status": "accepted",
                                "pair_id": pair_id,
                                "new_heading": (old + 1) % 4,
                                "left_snapshot_sha256": _persist(left_path, clone_sim(sim)),
                                "right_snapshot_sha256": _persist(right_path, clone_sim(twin)),
                                "tape_sha256": _persist(tape_path, tape),
                                "left_obs_digest": _obs_digest(observe_hero(sim, hero, probe)),
                                "right_obs_digest": _obs_digest(observe_hero(twin, hero, probe)),
                                "left_state_digest": full_state_digest(sim),
                                "right_state_digest": full_state_digest(twin),
                                "metrics": pair_metrics(left_values, right_values),
                                "left_returns": left_values,
                                "right_returns": right_values,
                            }
                        )
                        records.append(
                            {"world_id": world_id, "pair_id": pair_id, "metrics": row["metrics"]}
                        )
                        counters["pairs"] += 1
                    with raw_path.open("a") as handle:
                        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                if (
                    not sim.alive[0, 0]
                    or sim.population_floor_reached()[0]
                    or int(sim.frame[0]) >= probe.max_frames
                ):
                    break
                sim.step(_roster_actions(sim, policies))
                counters["steps"] += 6
            counters["worlds"] += 1
            with heartbeat.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "world_id": world_id,
                            "counters": counters,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
        expected = [f"world-{index:02d}" for index in range(WORLD_COUNT)]
        development = aggregate_world_metrics(
            [r for r in records if int(r["world_id"].split("-")[-1]) < DEVELOPMENT_WORLDS],
            expected_world_ids=expected[:DEVELOPMENT_WORLDS],
            bootstrap_seed=manifest["seeds"]["bootstrap"],
        )
        holdout = aggregate_world_metrics(
            [r for r in records if int(r["world_id"].split("-")[-1]) >= DEVELOPMENT_WORLDS],
            expected_world_ids=expected[DEVELOPMENT_WORLDS:],
            bootstrap_seed=manifest["seeds"]["bootstrap"],
        )
    except BaseException as exc:
        status, cause = "failed", f"{type(exc).__name__}: {exc}"
        development = holdout = None
    terminal = {
        "schema": RUN_SCHEMA,
        "status": status,
        "cause": cause,
        "manifest_sha256": manifest_sha256,
        "counters": counters,
        "raw_sha256": sha256_file(raw_path) if raw_path.exists() else None,
        "development": development,
        "holdout": holdout,
        "evaluation_started": False,
        "promotion_authority": False,
    }
    terminal["terminal_digest"] = digest_without(terminal, "terminal_digest")
    (out / "terminal.json").write_bytes(canonical(_jsonable(terminal)) + b"\n")
    return out / "terminal.json"


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
    if args.command == "freeze":
        print(freeze(args.out_dir, args.source_revision, args.seed_root))
        return 0
    print(run(args.manifest, args.manifest_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
