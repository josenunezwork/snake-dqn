#!/usr/bin/env python3
"""Run immutable, bounded, non-promoting corrected-PQN diagnostics.

Each invocation creates a new manifest and runs every planned learner once in
its own process group. The parent monitors resources and heartbeats, requires a
terminal receipt, freezes completed checkpoint bytes, and only then starts a
separately monitored CPU evaluator. This tool never gates, retries, or extends.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# isort: off
from src.core.runtime_contract import (  # noqa: E402
    EffectiveWorldConfig,
    RuntimeModeContract,
    canonical_digest,
)
from src.core.seeding import derive_seed, initialize_run_seed  # noqa: E402
from src.evaluation.protocol import promotion_v2_watch_rect  # noqa: E402
from src.model.checkpoint_io import atomic_torch_save  # noqa: E402
from src.model.obs_spec import RASTER31V3  # noqa: E402
from src.training.pqn_trainer import (  # noqa: E402
    PQNConfig,
    PQNTrainer,
    TripwireError,
    pqn_action_mask_contract,
    pqn_optimizer_contract,
    pqn_reward_contract,
    pqn_sampler_contract,
    pqn_target_contract,
)

# isort: on

EXPERIMENTAL_STATUS = "EXPERIMENTAL_NOT_PROMOTED"
MANIFEST_SCHEMA = "h0-correctness-diagnostic/v2"
WORKER_SCHEMA = "h0-worker/v2"
TERMINAL_SCHEMA = "h0-worker-terminal/v2"
SHA256_SCHEMA = "sha256/v1"
SHAKE_RECEIPT_SCHEMA = "h0-g0-shakedown/v1"
GIB = 1024**3
EXIT_TRIPWIRE = 70
EXIT_RESOURCE = 71
EXIT_PROTOCOL = 72
EXIT_FAILURE = 73
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
OUTCOMES = frozenset(
    {
        "completed",
        "tripwire",
        "resource_stop",
        "wall_stop",
        "learner_exit_failure",
        "source_drift",
        "protocol_drift",
        "watchdog_timeout",
        "evaluation_failure",
        "unrun",
    }
)


@dataclass(frozen=True)
class DiagnosticLimits:
    """Frozen limits; RSS, system available, and MPS are separate metrics."""

    rss_bytes: int = 4 * GIB
    available_bytes: int = 6 * GIB
    mps_driver_bytes: int = 8 * GIB
    wall_seconds: float = 600.0
    evaluation_wall_seconds: float = 600.0
    no_heartbeat_seconds: float = 60.0
    poll_seconds: float = 1.0
    term_grace_seconds: float = 5.0


@dataclass(frozen=True)
class Arm:
    """One immutable learner attempt."""

    name: str
    training_seed: int
    requested_steps: int


@dataclass(frozen=True)
class DiagnosticPlan:
    """Complete public plan, frozen before any child starts."""

    mode: str
    output_dir: Path
    requested_seed: int
    device: str
    num_envs: int
    rollout_len: int
    arms: tuple[Arm, ...]
    limits: DiagnosticLimits = field(default_factory=DiagnosticLimits)
    evaluation_seeds: tuple[int, ...] = tuple(range(12))
    mixes: tuple[str, ...] = ("scripted", "mixed")
    opponents: tuple[str, ...] = ("scripted:greedy_food", "scripted:random_safe")
    projection: Mapping[str, Any] | None = None


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any], *, immutable: bool = True) -> None:
    """Atomically publish JSON and never overwrite immutable artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if immutable:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    first = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if first:
        _fsync_directory(path.parent)


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    copy = dict(value)
    copy.pop(key, None)
    return hashlib.sha256(_json_bytes(copy)).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_identity() -> dict[str, Any]:
    root = _repository_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"commit": "unavailable", "dirty": None, "error": str(exc)}


def _source_closure() -> dict[str, str]:
    """Hash complete Python training/evaluation/serving source plus promotion YAML."""
    root = _repository_root()
    paths = set((root / "src").rglob("*.py"))
    backend = root / "web/backend"
    if backend.exists():
        paths.update(backend.rglob("*.py"))
    paths.add(root / "configs/promotion_mechanics_v2.yaml")
    return {
        str(path.relative_to(root)): sha256_file(path) for path in sorted(paths) if path.is_file()
    }


def _dependency_identity() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "psutil", "PyYAML", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def corrected_world() -> EffectiveWorldConfig:
    """Return the complete 23-field E1 promotion world targeted by H0."""
    return EffectiveWorldConfig(
        width=1450,
        height=830,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=6,
        max_frames=5000,
        initial_food=250,
        max_food=300,
        min_boost_length=5,
        boost_length_cost_frames=3,
        frame_rate=1,
        max_length=150,
        starvation_max_frames=500,
        arena_radius=400,
        arena_center_x=0,
        arena_center_y=0,
        max_capacity=400,
        kill_scale=0.3,
        death_value=-3.0,
        normalization={
            "max_frames": 5000.0,
            "starvation_max": 500.0,
            "max_length": 150.0,
        },
    )


def _world_mapping(world: EffectiveWorldConfig) -> dict[str, Any]:
    return {
        item.name: (
            dict(getattr(world, item.name))
            if item.name == "normalization"
            else getattr(world, item.name)
        )
        for item in fields(world)
    }


def _protocol_descriptor() -> dict[str, Any]:
    root = _repository_root()
    anchors = [root / "src/evaluation/anchors.py", root / "src/game/scripted_snake.py"]
    return {
        "recipe": "corrected-v3",
        "obs_spec": RASTER31V3,
        "reward_version": 2,
        "mechanics_version": 2,
        "flip_augment": False,
        "sampler": "legacy-fixed-minibatches",
        "optimizer": "Adam+Huber",
        "pool": "snapshot_pool",
        "epsilon_decay_agent_steps": 500_000,
        "evaluator": "promotion-v2-watch-rect",
        "scripted_anchors": {
            "version": "scripted-anchor/v1",
            "agents": ["scripted:greedy_food", "scripted:random_safe"],
            "source_hashes": {
                str(path.relative_to(root)): sha256_file(path) for path in anchors if path.is_file()
            },
        },
        "seed_namespaces": {
            "training_arm": "h0/training-arm/{index}",
            "global": "global",
            "sgd": "h0/sgd",
            "evaluation_world": "h0/evaluation-world/{raw_seed}",
        },
    }


def build_config(plan: DiagnosticPlan, arm: Arm) -> PQNConfig:
    """Build every corrected-PQN field explicitly with truthful provenance."""
    world = corrected_world()
    sources = {
        name: "h0_fixed_protocol"
        for name in PQNConfig.__dataclass_fields__
        if name != "field_sources"
    }
    sources.update(
        {
            "num_envs": "cli",
            "rollout_len": "cli",
            "seed": "derived:h0/training-arm",
            "sgd_seed": "derived:h0/sgd",
            "eps_decay_steps": "h0_fixed_protocol",
            "source_revision": "git_rev_parse_head",
            "requested_device": "cli",
            "effective_device": "cli",
        }
    )
    return PQNConfig(
        num_envs=plan.num_envs,
        num_snakes=world.num_snakes,
        rollout_len=plan.rollout_len,
        gamma=0.997,
        lambda_=0.65,
        lr=5e-4,
        adam_eps=1.5e-4,
        grad_clip=10.0,
        minibatches=4,
        minibatch_size=256,
        sgd_epochs=None,
        pad_sgd_batches=False,
        sgd_seed=derive_seed(arm.training_seed, "h0/sgd"),
        action_collapse_patience=1,
        action_collapse_min_samples=0,
        action_collapse_raw_actions=False,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay_steps=500_000,
        hero_frac=0.8,
        pool_capacity=10,
        pool_add_interval=50,
        rollout_policy_mode="snapshot_pool",
        fixed_policy_identity=None,
        death_value=world.death_value,
        kill_scale=world.kill_scale,
        flip_augment=False,
        max_frames=world.max_frames,
        max_abs_q_alarm=1e3,
        seed=arm.training_seed,
        arena_type=world.arena_type,
        mechanics_version=world.mechanics_version,
        reward_version=2,
        profile=False,
        game_width=world.width,
        game_height=world.height,
        segment_size=world.segment_size,
        wall_thickness=world.wall_thickness,
        initial_food=world.initial_food,
        max_food=world.max_food,
        min_boost_length=world.min_boost_length,
        boost_length_cost_frames=world.boost_length_cost_frames,
        frame_rate=world.frame_rate,
        max_capacity=world.max_capacity,
        starvation_max=world.starvation_max_frames,
        max_length=world.max_length,
        obs_spec=RASTER31V3,
        recipe="corrected-v3",
        field_sources=sources,
        source_revision=_git_identity()["commit"],
        requested_device=plan.device,
        effective_device=plan.device,
    )


def expected_arms(mode: str, seed: int, steps: int) -> tuple[Arm, ...]:
    """Return one smoke/shakedown arm or five unique screen arms."""
    counts = {"smoke": 1, "shakedown": 1, "screen": 5}
    if mode not in counts:
        raise ValueError("mode must be smoke, shakedown, or screen")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("requested steps must be a positive integer")
    return tuple(
        Arm(
            f"{mode}-{index + 1}",
            derive_seed(seed, f"h0/training-arm/{index}"),
            steps,
        )
        for index in range(counts[mode])
    )


def _validate_plan(plan: DiagnosticPlan) -> None:
    expected_count = {"smoke": 1, "shakedown": 1, "screen": 5}.get(plan.mode)
    if expected_count is None or len(plan.arms) != expected_count:
        raise ValueError(f"{plan.mode!r} requires exactly {expected_count} arms")
    if plan.device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (plan.num_envs, plan.rollout_len)
    ):
        raise ValueError("num_envs and rollout_len must be positive integers")
    names = [f"{plan.mode}-{index + 1}" for index in range(expected_count)]
    if [arm.name for arm in plan.arms] != names:
        raise ValueError("arm names must match the exact mode sequence")
    seeds = [arm.training_seed for arm in plan.arms]
    if len(seeds) != len(set(seeds)):
        raise ValueError("arm seeds must be unique")
    for arm in plan.arms:
        if (
            isinstance(arm.requested_steps, bool)
            or not isinstance(arm.requested_steps, int)
            or arm.requested_steps <= 0
        ):
            raise ValueError("every arm requires a positive requested step budget")
        if (
            isinstance(arm.training_seed, bool)
            or not isinstance(arm.training_seed, int)
            or not 0 <= arm.training_seed < 2**64
        ):
            raise ValueError("arm seeds must be unsigned 64-bit integers")
    if not plan.evaluation_seeds or len(set(plan.evaluation_seeds)) != len(plan.evaluation_seeds):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
        for seed in plan.evaluation_seeds
    ):
        raise ValueError("evaluation seeds must be unsigned 64-bit integers")
    for item in fields(plan.limits):
        value = getattr(plan.limits, item.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"limit {item.name} must be positive")
    if plan.mode == "smoke" and plan.num_envs != 1:
        raise ValueError("smoke mode requires exactly one environment")
    if plan.mode == "screen" and plan.projection is None:
        raise ValueError("screen mode requires a validated shakedown projection")


def _manifest_runtime(manifest: dict[str, Any], path: Path, sidecar: Path) -> dict[str, Any]:
    value = dict(manifest)
    value["_manifest_path"] = str(path)
    value["_manifest_file_sha256"] = sha256_file(path)
    value["_manifest_sidecar_path"] = str(sidecar)
    value["_manifest_sidecar_sha256"] = sha256_file(sidecar)
    return value


def freeze_manifest(plan: DiagnosticPlan) -> dict[str, Any]:
    """Create one new immutable manifest; prior manifests are never reused."""
    _validate_plan(plan)
    destination = plan.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty diagnostic destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    world = corrected_world()
    source_hashes = _source_closure()
    protocol = _protocol_descriptor()
    actual_eval_seeds = {
        str(seed): derive_seed(plan.requested_seed, f"h0/evaluation-world/{seed}")
        for seed in plan.evaluation_seeds
    }
    configs = {arm.name: asdict(build_config(plan, arm)) for arm in plan.arms}
    profile = promotion_v2_watch_rect(world)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "experiment": "pqn_correctness_diagnostic",
        "evaluation_status": EXPERIMENTAL_STATUS,
        "mode": plan.mode,
        "output_dir": str(destination),
        "requested_seed": plan.requested_seed,
        "device": plan.device,
        "threading": {
            "torch_intraop": 2,
            "torch_interop": 1,
            "native_env": {name: "2" for name in THREAD_ENV},
        },
        "arms": [asdict(arm) for arm in plan.arms],
        "limits": asdict(plan.limits),
        "world": _world_mapping(world),
        "world_digest": world.digest,
        "e1_promotion_profile": {
            "descriptor": profile.descriptor(),
            "digest": profile.digest,
        },
        "configs": configs,
        "source_hashes": source_hashes,
        "source_digest": canonical_digest(source_hashes),
        "git": _git_identity(),
        "protocol": protocol,
        "protocol_digest": canonical_digest(protocol),
        "seed_namespaces": {
            "requested": plan.requested_seed,
            "training": {arm.name: arm.training_seed for arm in plan.arms},
            "global": {arm.name: derive_seed(arm.training_seed, "global") for arm in plan.arms},
            "sgd": {arm.name: derive_seed(arm.training_seed, "h0/sgd") for arm in plan.arms},
            "batch_sim": {
                arm.name: [arm.training_seed + index for index in range(plan.num_envs)]
                for arm in plan.arms
            },
            "evaluation_world": {
                "raw": list(plan.evaluation_seeds),
                "actual": actual_eval_seeds,
                "namespace": "h0/evaluation-world/{raw_seed}",
            },
        },
        "evaluation": {
            "profile": "promotion-v2-watch-rect",
            "frames": 5000,
            "raw_seeds": list(plan.evaluation_seeds),
            "actual_seeds": [actual_eval_seeds[str(seed)] for seed in plan.evaluation_seeds],
            "mixes": list(plan.mixes),
            "opponents": list(plan.opponents),
            "gate": False,
            "config_path": str(
                (_repository_root() / "configs/promotion_mechanics_v2.yaml").resolve()
            ),
            "config_sha256": sha256_file(
                _repository_root() / "configs/promotion_mechanics_v2.yaml"
            ),
        },
        "projection": dict(plan.projection) if plan.projection is not None else None,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _dependency_identity(),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_rng_state": "not_captured_no_exact_resume",
        },
    }
    manifest["manifest_digest"] = _digest_without(manifest, "manifest_digest")
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    file_sha = sha256_file(manifest_path)
    sidecar_path = destination / "manifest.sha256"
    _write_json(
        sidecar_path,
        {"schema": SHA256_SCHEMA, "path": "manifest.json", "sha256": file_sha},
    )
    return _manifest_runtime(manifest, manifest_path, sidecar_path)


def _verify_boundary(manifest: Mapping[str, Any]) -> str | None:
    """Verify original manifest/sidecar bytes and the frozen source/protocol."""
    path = Path(str(manifest.get("_manifest_path", "")))
    sidecar = Path(str(manifest.get("_manifest_sidecar_path", "")))
    try:
        if sha256_file(path) != manifest.get("_manifest_file_sha256"):
            return "protocol_drift"
        if sha256_file(sidecar) != manifest.get("_manifest_sidecar_sha256"):
            return "protocol_drift"
        disk = json.loads(path.read_text())
        side = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return "protocol_drift"
    expected_sidecar = {
        "schema": SHA256_SCHEMA,
        "path": "manifest.json",
        "sha256": manifest["_manifest_file_sha256"],
    }
    if side != expected_sidecar:
        return "protocol_drift"
    if disk.get("manifest_digest") != _digest_without(disk, "manifest_digest"):
        return "protocol_drift"
    if disk.get("manifest_digest") != manifest.get("manifest_digest"):
        return "protocol_drift"
    if canonical_digest(manifest["protocol"]) != manifest["protocol_digest"]:
        return "protocol_drift"
    try:
        current_source = _source_closure()
    except OSError:
        return "source_drift"
    if current_source != manifest["source_hashes"]:
        return "source_drift"
    return None


def _expected_worker_spec(manifest: Mapping[str, Any], arm_name: str) -> dict[str, Any]:
    arm = next((item for item in manifest["arms"] if item["name"] == arm_name), None)
    if arm is None:
        raise ValueError(f"unknown arm {arm_name!r}")
    output = Path(manifest["output_dir"])
    return {
        "schema": WORKER_SCHEMA,
        "arm": arm,
        "arm_dir": str((output / arm_name).resolve()),
        "config": manifest["configs"][arm_name],
        "device": manifest["device"],
        "requested_steps": arm["requested_steps"],
        "limits": manifest["limits"],
        "manifest_digest": manifest["manifest_digest"],
        "manifest_path": str((output / "manifest.json").resolve()),
        "manifest_file_sha256": manifest["_manifest_file_sha256"],
        "manifest_sidecar_path": str((output / "manifest.sha256").resolve()),
        "manifest_sidecar_sha256": manifest["_manifest_sidecar_sha256"],
        "stdout_path": str((output / arm_name / "learner.stdout.log").resolve()),
        "stderr_path": str((output / arm_name / "learner.stderr.log").resolve()),
    }


def _save_child_checkpoint(trainer: PQNTrainer, path: Path, manifest_digest: str) -> None:
    state = trainer.checkpoint_state()
    state.update(
        {
            "evaluation_status": EXPERIMENTAL_STATUS,
            "h0_manifest_digest": manifest_digest,
        }
    )
    atomic_torch_save(state, str(path))


def _checkpoint_contract(
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    arm_name: str,
    *,
    initial: bool,
) -> str | None:
    expected_world_keys = {item.name for item in fields(EffectiveWorldConfig)}
    raw_world = state.get("effective_world")
    if not isinstance(raw_world, Mapping) or set(raw_world) != expected_world_keys:
        return "incomplete_effective_world"
    try:
        world = EffectiveWorldConfig(**dict(raw_world))
    except (TypeError, ValueError):
        return "invalid_effective_world"
    profile_world = manifest["e1_promotion_profile"]["descriptor"]["world"]
    if (
        dict(raw_world) != profile_world
        or world.digest != manifest["world_digest"]
        or state.get("effective_world_digest") != world.digest
    ):
        return "world_lineage_mismatch"
    config = manifest["configs"][arm_name]
    if state.get("effective_seed") != config["seed"]:
        return "seed_lineage_mismatch"
    if state.get("source_revision") != manifest["git"]["commit"]:
        return "source_revision_mismatch"
    try:
        typed_config = PQNConfig(**config)
        optimizer_state = state.get("optimizer_state_dict")
        expected_contracts = {
            "action_mask_contract": pqn_action_mask_contract(typed_config),
            "reward_contract": pqn_reward_contract(typed_config),
            "target_contract": pqn_target_contract(typed_config),
            "sampler_contract": pqn_sampler_contract(typed_config),
            "optimizer_contract": pqn_optimizer_contract(
                typed_config,
                optimizer_state if isinstance(optimizer_state, dict) else None,
            ),
            "runtime_contract": asdict(
                RuntimeModeContract(
                    mode="pqn_train",
                    training=True,
                    respawn=False,
                    hero_terminal=True,
                    population_floor=(
                        typed_config.mechanics_version == 2 and typed_config.num_snakes >= 3
                    ),
                    reset_strategy="batch_episode",
                )
            ),
        }
    except (TypeError, ValueError):
        return "invalid_recipe_contract"
    scalar_crosslinks = {
        "obs_spec": typed_config.obs_spec,
        "recipe": typed_config.recipe,
        "mechanics_version": typed_config.mechanics_version,
        "reward_version": typed_config.reward_version,
        "gamma": typed_config.gamma,
        "lambda": typed_config.lambda_,
        "lr": typed_config.lr,
        "adam_eps": typed_config.adam_eps,
        "eps_start": typed_config.eps_start,
        "eps_end": typed_config.eps_end,
        "eps_decay_steps": typed_config.eps_decay_steps,
        "sgd_epochs": typed_config.sgd_epochs,
        "pad_sgd_batches": typed_config.pad_sgd_batches,
        "sgd_seed": typed_config.sgd_seed,
        "field_sources": typed_config.field_sources,
    }
    if any(state.get(key) != value for key, value in scalar_crosslinks.items()):
        return "recipe_crosslink_mismatch"
    for key, expected in expected_contracts.items():
        if state.get(key) != expected:
            return f"{key}_mismatch"
        digest_key = f"{key}_digest"
        if state.get(digest_key) != canonical_digest(expected):
            return f"{digest_key}_mismatch"
    if initial and (state.get("agent_steps") != 0 or state.get("update_counter") != 0):
        return "nonzero_initial_clocks"
    return None


def _mps_metrics(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "mps":
        return None, None
    torch.mps.synchronize()
    return (
        int(torch.mps.driver_allocated_memory()),
        int(torch.mps.current_allocated_memory()),
    )


def _worker_terminal(
    arm_dir: Path,
    *,
    status: str,
    cause: str,
    requested_steps: int,
    actual_steps: int,
    update_count: int,
    started: float,
    resources: Mapping[str, Any],
    exception: Mapping[str, Any] | None = None,
) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError):
            pass
    artifacts: dict[str, Any] = {}
    names = (
        "initial.pth",
        "final.pth",
        "telemetry.jsonl",
        "heartbeat.jsonl",
        "learner.stdout.log",
        "learner.stderr.log",
        "incident.pth",
    )
    for name in names:
        path = arm_dir / name
        artifacts[name] = (
            {"path": str(path), "sha256": sha256_file(path)} if path.is_file() else None
        )
    receipt: dict[str, Any] = {
        "schema": TERMINAL_SCHEMA,
        "status": status,
        "cause": cause,
        "evaluation_status": EXPERIMENTAL_STATUS,
        "exception_or_tripwire": (dict(exception) if exception is not None else None),
        "actual_clocks": {
            "requested_agent_steps": requested_steps,
            "actual_agent_steps": actual_steps,
            "last_accepted_step": actual_steps,
            "overshoot_steps": max(0, actual_steps - requested_steps),
            "update_count": update_count,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
        },
        "resources": dict(resources),
        "artifacts": artifacts,
    }
    receipt["receipt_digest"] = _digest_without(receipt, "receipt_digest")
    _write_json(arm_dir / "terminal.json", receipt)


def worker_main(spec_path: Path) -> int:
    """Execute one validated learner and always attempt a terminal receipt."""
    started = time.monotonic()
    try:
        spec = json.loads(spec_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        arm_name = spec_path.name.removesuffix(".worker.json")
        arm_dir = (spec_path.parent / arm_name).resolve()
        arm_dir.mkdir(parents=True, exist_ok=True)
        _worker_terminal(
            arm_dir,
            status="failed",
            cause="protocol_drift",
            requested_steps=0,
            actual_steps=0,
            update_count=0,
            started=started,
            resources={"thresholds": None},
            exception={"type": type(exc).__name__, "message": str(exc)},
        )
        return EXIT_PROTOCOL
    arm_dir = Path(str(spec.get("arm_dir", spec_path.parent / "invalid-worker"))).resolve()
    arm_dir.mkdir(parents=True, exist_ok=True)
    requested = (
        int(spec.get("requested_steps", 0)) if isinstance(spec.get("requested_steps"), int) else 0
    )
    observed: dict[str, Any] = {
        "thresholds": dict(spec.get("limits", {})),
        "max_rss_bytes": None,
        "min_available_bytes": None,
        "max_mps_driver_bytes": None,
        "max_mps_current_bytes": None,
    }
    trainer: Any | None = None

    def fail(cause: str, detail: str, code: int = EXIT_PROTOCOL) -> int:
        _worker_terminal(
            arm_dir,
            status="failed",
            cause=cause,
            requested_steps=requested,
            actual_steps=getattr(trainer, "agent_steps", 0),
            update_count=getattr(trainer, "update_idx", 0),
            started=started,
            resources=observed,
            exception={"type": cause, "message": detail},
        )
        return code

    if spec.get("worker_spec_digest") != _digest_without(spec, "worker_spec_digest"):
        return fail("protocol_drift", "worker descriptor digest mismatch")
    try:
        manifest_path = Path(spec["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        sidecar_path = Path(spec["manifest_sidecar_path"])
        if (
            sha256_file(manifest_path) != spec["manifest_file_sha256"]
            or sha256_file(sidecar_path) != spec["manifest_sidecar_sha256"]
        ):
            return fail("protocol_drift", "manifest or sidecar bytes changed")
        runtime_manifest = _manifest_runtime(manifest, manifest_path, sidecar_path)
        expected = _expected_worker_spec(runtime_manifest, spec["arm"]["name"])
        actual = dict(spec)
        actual.pop("worker_spec_digest", None)
        if actual != expected:
            return fail("protocol_drift", "worker descriptor differs from frozen manifest")
        try:
            current_source = _source_closure()
        except OSError as exc:
            return fail("source_drift", f"source closure unreadable: {exc}")
        if current_source != manifest["source_hashes"]:
            return fail("source_drift", "source closure differs from frozen manifest")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return fail("protocol_drift", str(exc))

    try:
        for name in THREAD_ENV:
            if os.environ.get(name) != "2":
                return fail("protocol_drift", f"{name} must equal frozen value 2")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        config = PQNConfig(**spec["config"])
        seed_context = initialize_run_seed(config.seed)
        expected_global = manifest["seed_namespaces"]["global"][spec["arm"]["name"]]
        if seed_context.stream_seed("global") != expected_global:
            return fail("protocol_drift", "actual global seed differs from manifest")
        device = torch.device(spec["device"])
        trainer = PQNTrainer(config, device=device)
        if trainer.agent_steps != 0 or trainer.update_idx != 0:
            return fail("protocol_drift", "trainer clocks must start at zero", EXIT_FAILURE)
        initial_state = trainer.checkpoint_state()
        problem = _checkpoint_contract(initial_state, manifest, spec["arm"]["name"], initial=True)
        if problem:
            return fail("protocol_drift", problem, EXIT_FAILURE)
        _save_child_checkpoint(trainer, arm_dir / "initial.pth", spec["manifest_digest"])
        driver, current = _mps_metrics(device)
        observed["max_mps_driver_bytes"] = driver
        observed["max_mps_current_bytes"] = current
        _append_jsonl(
            arm_dir / "heartbeat.jsonl",
            {
                "monotonic": time.monotonic(),
                "agent_steps": 0,
                "update_idx": 0,
                "mps_driver_bytes": driver,
                "mps_current_bytes": current,
            },
        )
        if driver is not None and driver > spec["limits"]["mps_driver_bytes"]:
            _worker_terminal(
                arm_dir,
                status="resource_stop",
                cause="mps_driver_cap",
                requested_steps=requested,
                actual_steps=0,
                update_count=0,
                started=started,
                resources=observed,
            )
            return EXIT_RESOURCE

        while trainer.agent_steps < requested:
            before = trainer.agent_steps
            update_started = time.monotonic()
            telemetry = trainer.update()
            driver, current = _mps_metrics(device)
            finished = time.monotonic()  # includes the preceding MPS synchronization
            record = asdict(telemetry)
            record.update(
                {
                    "monotonic": finished,
                    "update_wall_seconds": finished - update_started,
                    "before_hero_steps": before,
                    "after_hero_steps": trainer.agent_steps,
                    "actual_update_index": trainer.update_idx,
                    "useful_hero_steps": trainer.agent_steps - before,
                    "requested_agent_steps": requested,
                    "overshoot_steps": max(0, trainer.agent_steps - requested),
                }
            )
            _append_jsonl(arm_dir / "telemetry.jsonl", record)
            driver_values = [
                value for value in (observed["max_mps_driver_bytes"], driver) if value is not None
            ]
            current_values = [
                value for value in (observed["max_mps_current_bytes"], current) if value is not None
            ]
            observed["max_mps_driver_bytes"] = max(driver_values) if driver_values else None
            observed["max_mps_current_bytes"] = max(current_values) if current_values else None
            _append_jsonl(
                arm_dir / "heartbeat.jsonl",
                {
                    "monotonic": finished,
                    "agent_steps": trainer.agent_steps,
                    "update_idx": trainer.update_idx,
                    "mps_driver_bytes": driver,
                    "mps_current_bytes": current,
                },
            )
            if trainer.agent_steps <= before:
                return fail("protocol_drift", "update made no positive progress", EXIT_FAILURE)
            if driver is not None and driver > spec["limits"]["mps_driver_bytes"]:
                _worker_terminal(
                    arm_dir,
                    status="resource_stop",
                    cause="mps_driver_cap",
                    requested_steps=requested,
                    actual_steps=trainer.agent_steps,
                    update_count=trainer.update_idx,
                    started=started,
                    resources=observed,
                )
                return EXIT_RESOURCE
        final_state = trainer.checkpoint_state()
        problem = _checkpoint_contract(final_state, manifest, spec["arm"]["name"], initial=False)
        if problem or trainer.agent_steps < requested:
            return fail(
                "protocol_drift",
                problem or "final steps below requested budget",
                EXIT_FAILURE,
            )
        _save_child_checkpoint(trainer, arm_dir / "final.pth", spec["manifest_digest"])
        _worker_terminal(
            arm_dir,
            status="completed",
            cause="budget_reached",
            requested_steps=requested,
            actual_steps=trainer.agent_steps,
            update_count=trainer.update_idx,
            started=started,
            resources=observed,
        )
        return 0
    except TripwireError as exc:
        incident = {
            "class": exc.incident_class,
            "details": exc.incident,
            "finite": exc.is_finite_alarm,
            "message": str(exc),
        }
        _write_json(arm_dir / "incident.json", incident)
        if exc.is_finite_alarm and trainer is not None:
            trainer.save_incident_checkpoint(str(arm_dir / "incident.pth"), exc)
        _worker_terminal(
            arm_dir,
            status="tripwire",
            cause="tripwire",
            requested_steps=requested,
            actual_steps=getattr(trainer, "agent_steps", 0),
            update_count=getattr(trainer, "update_idx", 0),
            started=started,
            resources=observed,
            exception=incident,
        )
        return EXIT_TRIPWIRE
    except Exception as exc:
        _worker_terminal(
            arm_dir,
            status="failed",
            cause="worker_exception",
            requested_steps=requested,
            actual_steps=getattr(trainer, "agent_steps", 0),
            update_count=getattr(trainer, "update_idx", 0),
            started=started,
            resources=observed,
            exception={"type": type(exc).__name__, "message": str(exc)},
        )
        return EXIT_FAILURE


def _last_heartbeat(path: Path) -> dict[str, Any] | None:
    """Return the latest complete heartbeat, ignoring a partial crash tail."""
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("monotonic"), (int, float)):
            return value
    return None


def _read_terminal(arm_dir: Path) -> tuple[bool, dict[str, Any]]:
    path = arm_dir / "terminal.json"
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, {
            "reason": "missing_or_invalid_terminal_receipt",
            "error": str(exc),
        }
    if receipt.get("schema") != TERMINAL_SCHEMA or receipt.get("receipt_digest") != _digest_without(
        receipt, "receipt_digest"
    ):
        return False, {"reason": "invalid_terminal_receipt_digest"}
    for artifact in receipt.get("artifacts", {}).values():
        if artifact is None:
            continue
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            return False, {
                "reason": "terminal_artifact_hash_mismatch",
                "path": str(artifact_path),
            }
    return True, receipt


def checkpoint_lineage(arm_dir: Path, manifest: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Validate pair lineage, complete clocks, and the worker terminal receipt."""
    paths = {"initial": arm_dir / "initial.pth", "final": arm_dir / "final.pth"}
    if any(not path.is_file() for path in paths.values()):
        return False, {"reason": "missing_initial_or_final_checkpoint"}
    try:
        states = {
            name: torch.load(path, map_location="cpu", weights_only=False)
            for name, path in paths.items()
        }
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        return False, {"reason": "unreadable_checkpoint", "error": str(exc)}
    for name, state in states.items():
        if (
            not isinstance(state, Mapping)
            or state.get("h0_manifest_digest") != manifest["manifest_digest"]
        ):
            return False, {
                "reason": "manifest_lineage_mismatch",
                "checkpoint": name,
            }
        problem = _checkpoint_contract(state, manifest, arm_dir.name, initial=name == "initial")
        if problem:
            return False, {"reason": problem, "checkpoint": name}
    terminal_ok, terminal = _read_terminal(arm_dir)
    if not terminal_ok:
        return False, terminal
    clocks = terminal["actual_clocks"]
    requested = next(
        item["requested_steps"] for item in manifest["arms"] if item["name"] == arm_dir.name
    )
    actual = states["final"].get("agent_steps")
    update_count = states["final"].get("update_counter")
    if terminal.get("status") != "completed" or not isinstance(actual, int) or actual < requested:
        return False, {"reason": "incomplete_terminal_clocks"}
    if (
        clocks.get("requested_agent_steps") != requested
        or clocks.get("actual_agent_steps") != actual
        or clocks.get("last_accepted_step") != actual
        or clocks.get("update_count") != update_count
        or clocks.get("overshoot_steps") != actual - requested
    ):
        return False, {"reason": "terminal_clock_mismatch"}
    return True, {
        "initial_sha256": sha256_file(paths["initial"]),
        "final_sha256": sha256_file(paths["final"]),
        "actual_agent_steps": actual,
        "requested_agent_steps": requested,
        "overshoot_steps": actual - requested,
        "update_count": update_count,
        "terminal_receipt_sha256": sha256_file(arm_dir / "terminal.json"),
        "elapsed_seconds": clocks["elapsed_seconds"],
    }


def stop_process(
    process: Any,
    grace_seconds: float,
    now: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> str:
    """TERM then KILL a process group and distinguish an unconfirmed exit."""
    terminated = False
    killed = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        terminated = True
    except ProcessLookupError:
        pass
    except OSError:
        return "unconfirmed_exit"
    deadline = now() + grace_seconds
    while process.poll() is None and now() < deadline:
        sleeper(min(0.05, max(0.0, deadline - now())))
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
        except ProcessLookupError:
            pass
        except OSError:
            return "unconfirmed_exit"
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except (subprocess.TimeoutExpired, TypeError, OSError):
        return "unconfirmed_exit"
    if process.poll() is None:
        return "unconfirmed_exit"
    if killed:
        return "killed"
    return "terminated" if terminated else "already_exited"


def _monitor_process(
    process: Any,
    limits: Mapping[str, Any],
    *,
    wall_seconds: float,
    heartbeat_path: Path | None,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    process_factory: Callable[[int], Any],
) -> dict[str, Any]:
    """Monitor one learner or evaluator and always reap or report unconfirmed."""
    started = now()
    child = None
    last_heartbeat: dict[str, Any] | None = None
    observations: dict[str, Any] = {
        "max_rss_bytes": 0,
        "min_available_bytes": None,
        "latest_heartbeat": None,
    }
    while process.poll() is None:
        current = now()
        if heartbeat_path is not None:
            latest = _last_heartbeat(heartbeat_path)
            if latest is not None:
                last_heartbeat = latest
                observations["latest_heartbeat"] = latest
        try:
            if child is None:
                child = process_factory(process.pid)
            rss = int(child.memory_info().rss)
            available = int(psutil.virtual_memory().available)
        except psutil.NoSuchProcess as exc:
            if process.poll() is not None:
                break
            termination = stop_process(process, limits["term_grace_seconds"], now, sleep)
            return {
                "cause": "process_monitor_race",
                "error": str(exc),
                "termination": termination,
                "confirmed_exit": termination != "unconfirmed_exit",
                **observations,
            }
        except (psutil.AccessDenied, psutil.Error, OSError) as exc:
            termination = stop_process(process, limits["term_grace_seconds"], now, sleep)
            return {
                "cause": "process_monitor_error",
                "error": str(exc),
                "termination": termination,
                "confirmed_exit": termination != "unconfirmed_exit",
                **observations,
            }
        observations["max_rss_bytes"] = max(observations["max_rss_bytes"], rss)
        previous_available = observations["min_available_bytes"]
        observations["min_available_bytes"] = (
            available if previous_available is None else min(previous_available, available)
        )
        cause = None
        if rss > limits["rss_bytes"] or available < limits["available_bytes"]:
            cause = "resource_stop"
        elif current - started > wall_seconds:
            cause = "wall_stop"
        elif heartbeat_path is not None:
            reference = started if last_heartbeat is None else float(last_heartbeat["monotonic"])
            if current - reference > limits["no_heartbeat_seconds"]:
                cause = "watchdog_timeout"
            elif (
                last_heartbeat
                and isinstance(last_heartbeat.get("mps_driver_bytes"), int)
                and last_heartbeat["mps_driver_bytes"] > limits["mps_driver_bytes"]
            ):
                cause = "resource_stop"
        if cause:
            breach = now()
            termination = stop_process(process, limits["term_grace_seconds"], now, sleep)
            return {
                "cause": cause,
                "termination": termination,
                "confirmed_exit": termination != "unconfirmed_exit",
                "elapsed_at_detection_seconds": breach - started,
                "monitor_overshoot_seconds": (
                    max(0.0, breach - started - wall_seconds) if cause == "wall_stop" else None
                ),
                **observations,
            }
        sleep(limits["poll_seconds"])
    try:
        process.wait(timeout=max(1.0, limits["term_grace_seconds"]))
    except (subprocess.TimeoutExpired, TypeError, OSError) as exc:
        return {
            "cause": "exit_unconfirmed",
            "error": str(exc),
            "termination": "unconfirmed_exit",
            "confirmed_exit": False,
            **observations,
        }
    return {
        "cause": None,
        "termination": "natural_exit",
        "confirmed_exit": process.poll() is not None,
        "elapsed_seconds": max(0.0, now() - started),
        **observations,
    }


def _freeze_checkpoint_pair(arm_dir: Path, lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze content-addressed checkpoint copies from stable file identities."""
    root = arm_dir / "accepted_checkpoints" / "sha256"
    root.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "h0-accepted-checkpoints/v1",
        "checkpoints": {},
    }
    for name in ("initial", "final"):
        source = arm_dir / f"{name}.pth"
        expected = lineage[f"{name}_sha256"]
        before = source.stat()
        with (
            source.open("rb") as input_handle,
            tempfile.NamedTemporaryFile(
                mode="wb", dir=root, prefix=".copy-", delete=False
            ) as output_handle,
        ):
            temporary = Path(output_handle.name)
            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        after = source.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if (
            before_identity != after_identity
            or sha256_file(source) != expected
            or sha256_file(temporary) != expected
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"{name} checkpoint changed while freezing")
        destination = root / f"{expected}.pth"
        if destination.exists():
            if sha256_file(destination) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("content-addressed destination is corrupt")
            temporary.unlink()
        else:
            os.link(temporary, destination)
            temporary.unlink()
            destination.chmod(0o444)
            _fsync_directory(root)
        receipt["checkpoints"][name] = {
            "source": str(source),
            "path": str(destination),
            "sha256": expected,
            "size_bytes": destination.stat().st_size,
        }
    receipt["receipt_digest"] = _digest_without(receipt, "receipt_digest")
    _write_json(arm_dir / "accepted_checkpoints.json", receipt)
    return receipt


def _run_evaluation(
    arm_dir: Path, manifest: Mapping[str, Any], timeout: float
) -> tuple[bool, dict[str, Any]]:
    """Evaluate immutable checkpoint copies under the shared process monitor."""
    accepted_path = arm_dir / "accepted_checkpoints.json"
    try:
        accepted = json.loads(accepted_path.read_text())
        if accepted.get("receipt_digest") != _digest_without(accepted, "receipt_digest"):
            raise ValueError("accepted checkpoint receipt digest mismatch")
        initial = accepted["checkpoints"]["initial"]
        final = accepted["checkpoints"]["final"]
        for item in (initial, final):
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("accepted checkpoint hash mismatch")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return False, {
            "reason": "invalid_accepted_checkpoints",
            "error": str(exc),
        }
    output = arm_dir / "evaluation.json"
    snapshots = arm_dir / "evaluation_snapshots"
    stdout_path = arm_dir / "evaluation.stdout.log"
    stderr_path = arm_dir / "evaluation.stderr.log"
    evaluation = manifest["evaluation"]
    command = [
        sys.executable,
        "-u",
        str((_repository_root() / "src/scripts/tournament_eval.py").resolve()),
        final["path"],
        "--baseline",
        initial["path"],
        "--engine",
        "simd",
        "--evaluation-profile",
        evaluation["profile"],
        "--frames",
        str(evaluation["frames"]),
        "--config",
        evaluation["config_path"],
        "--seeds",
        ",".join(map(str, evaluation["actual_seeds"])),
        "--mixes",
        ",".join(evaluation["mixes"]),
        "--opponents",
        ",".join(evaluation["opponents"]),
        "--json-output",
        str(output.resolve()),
        "--snapshot-dir",
        str(snapshots.resolve()),
    ]
    if "--gate" in command:
        raise AssertionError("H0 evaluation must never invoke promotion gate")
    child_env = os.environ.copy()
    child_env.update({name: "2" for name in THREAD_ENV})
    child_env["SNAKE_DQN_DEVICE"] = "cpu"
    process = None
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                command,
                start_new_session=True,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=child_env,
            )
            monitor = _monitor_process(
                process,
                manifest["limits"],
                wall_seconds=timeout,
                heartbeat_path=None,
                now=time.monotonic,
                sleep=time.sleep,
                process_factory=psutil.Process,
            )
    except OSError as exc:
        if process is not None and process.poll() is None:
            termination = stop_process(process, manifest["limits"]["term_grace_seconds"])
        else:
            termination = "not_started"
        return False, {
            "reason": "evaluation_launch_or_monitor_failure",
            "error": str(exc),
            "termination": termination,
            "command": command,
        }
    if not monitor["confirmed_exit"]:
        return False, {
            "reason": "evaluation_exit_unconfirmed",
            "monitor": monitor,
            "command": command,
        }
    if monitor["cause"] or process.returncode != 0:
        return False, {
            "reason": monitor["cause"] or "evaluation_nonzero_exit",
            "monitor": monitor,
            "returncode": process.returncode,
            "command": command,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        }
    try:
        payload = json.loads(output.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("evaluation output must be a JSON object")
        receipt_path = Path(payload["evaluation_inputs"]["receipt"])
        evaluator_receipt = json.loads(receipt_path.read_text())
        if not isinstance(evaluator_receipt, Mapping):
            raise ValueError("E1 receipt must be a JSON object")
        config_receipt = evaluator_receipt.get("config")
        if not isinstance(config_receipt, Mapping):
            raise ValueError("E1 receipt is missing its config snapshot")
        config_snapshot = Path(str(config_receipt.get("snapshot_path", "")))
        if (
            config_receipt.get("sha256") != evaluation["config_sha256"]
            or not config_snapshot.is_file()
            or sha256_file(config_snapshot) != evaluation["config_sha256"]
        ):
            raise ValueError("E1 config snapshot differs from frozen config")
        expected_output = {
            "mode": "eval",
            "config": evaluation["config_path"],
            "engine": "simd",
            "frames": evaluation["frames"],
            "seeds": evaluation["actual_seeds"],
            "baseline": initial["path"],
            "mixes": evaluation["mixes"],
            "opponent_pool": evaluation["opponents"],
        }
        if any(payload.get(key) != value for key, value in expected_output.items()):
            raise ValueError("evaluation output descriptor differs from manifest")
        candidates = payload.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 1
            or not isinstance(candidates[0], Mapping)
            or candidates[0].get("candidate") != final["path"]
            or "error" in candidates[0]
        ):
            raise ValueError("evaluation candidate result differs from accepted final")
        if evaluator_receipt.get("evaluation_profile") != manifest["e1_promotion_profile"]:
            raise ValueError("E1 evaluation profile receipt differs from manifest")
        checkpoint_snapshots = evaluator_receipt.get("checkpoint_snapshots")
        if not isinstance(checkpoint_snapshots, list) or any(
            not isinstance(item, Mapping) for item in checkpoint_snapshots
        ):
            raise ValueError("E1 checkpoint snapshots must be a list of objects")
        snapshot_hashes = {item["sha256"] for item in checkpoint_snapshots}
        expected_hashes = {initial["sha256"], final["sha256"]}
        if not expected_hashes.issubset(snapshot_hashes):
            raise ValueError("E1 snapshot receipt does not match accepted checkpoint lineage")
        for item in checkpoint_snapshots:
            if sha256_file(Path(item["snapshot_path"])) != item["sha256"]:
                raise ValueError("E1 snapshot bytes changed")
        for item in (initial, final):
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("accepted checkpoint bytes changed during evaluation")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, {
            "reason": "evaluation_artifact_validation",
            "error": str(exc),
            "command": command,
        }
    receipt: dict[str, Any] = {
        "schema": "h0-evaluation-terminal/v1",
        "status": "completed",
        "command": command,
        "returncode": process.returncode,
        "monitor": monitor,
        "evaluation_json": {
            "path": str(output),
            "sha256": sha256_file(output),
        },
        "evaluation_inputs_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        "accepted_checkpoints_receipt": {
            "path": str(accepted_path),
            "sha256": sha256_file(accepted_path),
        },
        "stdout": {"path": str(stdout_path), "sha256": sha256_file(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": sha256_file(stderr_path)},
    }
    receipt["receipt_digest"] = _digest_without(receipt, "receipt_digest")
    h0_receipt_path = arm_dir / "evaluation_terminal.json"
    _write_json(h0_receipt_path, receipt)
    receipt["path"] = str(h0_receipt_path)
    receipt["sha256"] = sha256_file(h0_receipt_path)
    return True, receipt


def _valid_evaluation_receipt(receipt: Mapping[str, Any]) -> bool:
    path_value = receipt.get("path")
    digest = receipt.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        return False
    path = Path(path_value)
    if not path.is_file() or sha256_file(path) != digest:
        return False
    try:
        disk = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        disk.get("receipt_digest") == _digest_without(disk, "receipt_digest")
        and disk.get("status") == "completed"
    )


def _result_for_unrun(arm: Arm, cause: str) -> dict[str, Any]:
    return {
        "arm": arm.name,
        "outcome": "unrun",
        "cause": cause,
        "evaluation_status": EXPERIMENTAL_STATUS,
    }


def _write_results(plan: DiagnosticPlan, results: Sequence[Mapping[str, Any]]) -> None:
    _write_json(
        plan.output_dir.resolve() / "results.json",
        {"evaluation_status": EXPERIMENTAL_STATUS, "results": list(results)},
    )


def _write_shakedown_receipt(
    plan: DiagnosticPlan,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if plan.mode != "shakedown" or result.get("outcome") != "completed":
        return
    lineage = result["checkpoint_lineage"]
    learner_elapsed = result.get("monitor", {}).get("elapsed_seconds", lineage["elapsed_seconds"])
    if (
        lineage["requested_agent_steps"] != 100_000
        or lineage["actual_agent_steps"] < 100_000
        or not isinstance(learner_elapsed, (int, float))
        or learner_elapsed <= 0
        or learner_elapsed > 600
    ):
        return
    receipt: dict[str, Any] = {
        "schema": SHAKE_RECEIPT_SCHEMA,
        "status": "completed",
        "mode": "shakedown",
        "device": plan.device,
        "requested_steps": 100_000,
        "actual_steps": lineage["actual_agent_steps"],
        "elapsed_seconds": learner_elapsed,
        "source_digest": manifest["source_digest"],
        "protocol_digest": manifest["protocol_digest"],
        "manifest_sha256": manifest["_manifest_file_sha256"],
        "num_envs": plan.num_envs,
        "rollout_len": plan.rollout_len,
    }
    receipt["receipt_digest"] = _digest_without(receipt, "receipt_digest")
    path = plan.output_dir.resolve() / "shakedown_projection_receipt.json"
    _write_json(path, receipt)
    _write_json(
        path.with_suffix(".sha256"),
        {"schema": SHA256_SCHEMA, "path": path.name, "sha256": sha256_file(path)},
    )


def run_plan(
    plan: DiagnosticPlan,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    evaluator: Callable[
        [Path, Mapping[str, Any], float], tuple[bool, dict[str, Any]]
    ] = _run_evaluation,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    process_factory: Callable[[int], Any] = psutil.Process,
    checkpoint_inspector: Callable[
        [Path, Mapping[str, Any]], tuple[bool, dict[str, Any]]
    ] = checkpoint_lineage,
) -> list[dict[str, Any]]:
    """Freeze a new plan and retain exactly one terminal result per arm."""
    manifest = freeze_manifest(plan)
    results: list[dict[str, Any]] = []
    abort_remaining = False
    for arm in plan.arms:
        if abort_remaining:
            result = _result_for_unrun(arm, "prior_child_exit_unconfirmed")
            arm_dir = (plan.output_dir / arm.name).resolve()
            arm_dir.mkdir(parents=False, exist_ok=False)
            _write_json(arm_dir / "result.json", result)
            results.append(result)
            continue
        boundary = _verify_boundary(manifest)
        if boundary:
            arm_dir = (plan.output_dir / arm.name).resolve()
            arm_dir.mkdir(parents=False, exist_ok=False)
            result = {
                "arm": arm.name,
                "outcome": boundary,
                "cause": boundary,
                "evaluation_status": EXPERIMENTAL_STATUS,
            }
            _write_json(arm_dir / "result.json", result)
            results.append(result)
            continue
        arm_dir = (plan.output_dir / arm.name).resolve()
        arm_dir.mkdir(parents=False, exist_ok=False)
        spec = _expected_worker_spec(manifest, arm.name)
        spec["worker_spec_digest"] = _digest_without(spec, "worker_spec_digest")
        spec_path = plan.output_dir.resolve() / f"{arm.name}.worker.json"
        _write_json(spec_path, spec)
        child_env = os.environ.copy()
        child_env.update({name: "2" for name in THREAD_ENV})
        stdout_path = arm_dir / "learner.stdout.log"
        stderr_path = arm_dir / "learner.stderr.log"
        process = None
        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--worker-spec",
                        str(spec_path),
                    ],
                    start_new_session=True,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    env=child_env,
                )
                monitor = _monitor_process(
                    process,
                    manifest["limits"],
                    wall_seconds=plan.limits.wall_seconds,
                    heartbeat_path=arm_dir / "heartbeat.jsonl",
                    now=now,
                    sleep=sleep,
                    process_factory=process_factory,
                )
        except OSError as exc:
            result = {
                "arm": arm.name,
                "outcome": "learner_exit_failure",
                "cause": "launch_failure",
                "error": str(exc),
                "evaluation_status": EXPERIMENTAL_STATUS,
            }
            _write_json(arm_dir / "result.json", result)
            results.append(result)
            continue
        if not monitor["confirmed_exit"]:
            result = {
                "arm": arm.name,
                "outcome": "learner_exit_failure",
                "cause": "child_exit_unconfirmed",
                "returncode": process.returncode,
                "monitor": monitor,
                "evaluation_status": EXPERIMENTAL_STATUS,
            }
            _write_json(arm_dir / "result.json", result)
            results.append(result)
            abort_remaining = True
            continue
        terminal_ok, terminal = _read_terminal(arm_dir)
        cause: str | None
        if monitor["cause"] in {
            "resource_stop",
            "wall_stop",
            "watchdog_timeout",
        }:
            outcome = monitor["cause"]
            cause = monitor["cause"]
        elif not terminal_ok:
            outcome = "learner_exit_failure"
            cause = terminal["reason"]
        elif process.returncode == EXIT_PROTOCOL:
            cause = terminal.get("cause", "protocol_drift")
            outcome = "source_drift" if cause == "source_drift" else "protocol_drift"
        elif monitor["cause"] is not None:
            outcome = "learner_exit_failure"
            cause = monitor["cause"]
        elif process.returncode == 0 and terminal.get("status") == "completed":
            outcome = "completed"
            cause = terminal.get("cause")
        elif process.returncode == EXIT_TRIPWIRE and terminal.get("status") == "tripwire":
            outcome = "tripwire"
            cause = "tripwire"
        elif process.returncode == EXIT_RESOURCE and terminal.get("status") == "resource_stop":
            outcome = "resource_stop"
            cause = terminal.get("cause")
        else:
            outcome = "learner_exit_failure"
            cause = terminal.get("cause")
        lineage: dict[str, Any] = {}
        accepted: dict[str, Any] | None = None
        if outcome == "completed":
            complete, lineage = checkpoint_inspector(arm_dir, manifest)
            if not complete:
                outcome = "learner_exit_failure"
                cause = lineage.get("reason")
            else:
                try:
                    accepted = _freeze_checkpoint_pair(arm_dir, lineage)
                except (OSError, RuntimeError, FileExistsError) as exc:
                    outcome = "learner_exit_failure"
                    cause = "checkpoint_freeze_failure"
                    lineage["freeze_error"] = str(exc)
        boundary = _verify_boundary(manifest)
        if boundary:
            outcome, cause = boundary, boundary
        result: dict[str, Any] = {
            "arm": arm.name,
            "outcome": outcome,
            "cause": cause,
            "returncode": process.returncode,
            "monitor": monitor,
            "terminal_receipt": terminal,
            "checkpoint_lineage": lineage,
            "accepted_checkpoints": accepted,
            "evaluation_status": EXPERIMENTAL_STATUS,
        }
        if outcome == "completed":
            before_eval = _verify_boundary(manifest)
            if before_eval:
                result["outcome"] = before_eval
                result["cause"] = before_eval
            else:
                ok, evaluation_receipt = evaluator(
                    arm_dir, manifest, plan.limits.evaluation_wall_seconds
                )
                result["evaluation"] = evaluation_receipt
                if ok and not _valid_evaluation_receipt(evaluation_receipt):
                    ok = False
                    evaluation_receipt["validation_error"] = (
                        "missing_or_invalid_evaluation_terminal_receipt"
                    )
                post_eval = _verify_boundary(manifest)
                result["outcome"] = (
                    post_eval if post_eval else ("completed" if ok else "evaluation_failure")
                )
                result["cause"] = (
                    post_eval
                    if post_eval
                    else (None if ok else evaluation_receipt.get("reason", "evaluation_failure"))
                )
        if result["outcome"] not in OUTCOMES:
            raise RuntimeError(f"unknown H0 outcome {result['outcome']!r}")
        _write_json(arm_dir / "result.json", result)
        results.append(result)
    _write_results(plan, results)
    if len(results) == 1:
        _write_shakedown_receipt(plan, manifest, results[0])
    return results


def load_shakedown_projection(
    path: Path,
    *,
    device: str,
    requested_steps: int,
    num_envs: int,
    rollout_len: int,
) -> dict[str, Any]:
    """Validate a compatible 100k G0 receipt and compute the screen wall cap."""
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable receipt: {exc}") from exc
    required = {
        "schema": SHAKE_RECEIPT_SCHEMA,
        "status": "completed",
        "mode": "shakedown",
        "device": device,
        "requested_steps": 100_000,
        "source_digest": canonical_digest(_source_closure()),
        "protocol_digest": canonical_digest(_protocol_descriptor()),
        "num_envs": num_envs,
        "rollout_len": rollout_len,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"incompatible shakedown receipt field {key}")
    if receipt.get("receipt_digest") != _digest_without(receipt, "receipt_digest"):
        raise ValueError("shakedown receipt digest mismatch")
    actual = receipt.get("actual_steps")
    elapsed = receipt.get("elapsed_seconds")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 100_000:
        raise ValueError("shakedown actual_steps must be at least 100000")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 600:
        raise ValueError("shakedown elapsed_seconds must be in (0, 600]")
    uncapped = float(elapsed) / actual * requested_steps * 1.25
    return {
        "schema": "h0-screen-projection/v1",
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256_file(path),
        "observed_seconds": float(elapsed),
        "observed_actual_steps": actual,
        "requested_screen_steps": requested_steps,
        "headroom_multiplier": 1.25,
        "formula": ("observed_seconds / observed_actual_steps * " "requested_screen_steps * 1.25"),
        "uncapped_seconds": uncapped,
        "cap_seconds": 1800.0,
        "projected_wall_seconds": min(1800.0, uncapped),
        "source_digest": receipt["source_digest"],
        "protocol_digest": receipt["protocol_digest"],
        "device": device,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "shakedown", "screen"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--shakedown-receipt", type=Path)
    parser.add_argument("--worker-spec", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.worker_spec is None:
        missing = [
            name
            for name in ("mode", "out_dir", "seed", "total_steps")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "required outside --worker-spec: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        if args.mode == "shakedown" and args.total_steps != 100_000:
            parser.error("shakedown requires --total-steps 100000")
        if args.mode == "screen" and args.total_steps != 500_000:
            parser.error("screen requires --total-steps 500000")
        if args.mode == "screen" and args.shakedown_receipt is None:
            parser.error("screen requires --shakedown-receipt")
        if args.mode == "smoke" and args.envs not in {None, 1}:
            parser.error("smoke requires --envs 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_spec is not None:
        return worker_main(args.worker_spec)
    num_envs = 1 if args.envs is None and args.mode == "smoke" else (args.envs or 16)
    rollout_len = (
        2 if args.rollout_len is None and args.mode == "smoke" else (args.rollout_len or 16)
    )
    projection = None
    limits = DiagnosticLimits()
    if args.mode == "screen":
        try:
            projection = load_shakedown_projection(
                args.shakedown_receipt,
                device=args.device,
                requested_steps=args.total_steps,
                num_envs=num_envs,
                rollout_len=rollout_len,
            )
        except ValueError as exc:
            raise SystemExit(f"invalid --shakedown-receipt: {exc}") from exc
        limits = DiagnosticLimits(wall_seconds=projection["projected_wall_seconds"])
    evaluation_seeds = (0,) if args.mode in {"smoke", "shakedown"} else tuple(range(12))
    mixes = ("scripted",) if args.mode in {"smoke", "shakedown"} else ("scripted", "mixed")
    plan = DiagnosticPlan(
        mode=args.mode,
        output_dir=args.out_dir,
        requested_seed=args.seed,
        device=args.device,
        num_envs=num_envs,
        rollout_len=rollout_len,
        arms=expected_arms(args.mode, args.seed, args.total_steps),
        limits=limits,
        evaluation_seeds=evaluation_seeds,
        mixes=mixes,
        projection=projection,
    )
    outcomes = run_plan(plan)
    return 0 if all(item["outcome"] == "completed" for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
