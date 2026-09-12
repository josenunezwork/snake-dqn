"""Adversarial synthetic receipts for the strict promotion artifact boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pytest
import torch

from src.core.runtime_contract import (
    EffectiveWorldConfig,
    ModelHeadContract,
    RuntimeModeContract,
    canonical_digest,
)
from src.core.seeding import derive_seed
from src.evaluation.protocol import promotion_v2_watch_rect
from src.evaluation.serving_episode import ServingEpisodeSpec, run_serving_episode
from src.evaluation.strict_promotion import (
    STRICT_CALIBRATION_VERSION,
    STRICT_FINAL_RECEIPT_VERSION,
    STRICT_PILOT_VERSION,
    STRICT_RAW_WORLD_VERSION,
    STRICT_REQUEST_VERSION,
    STRICT_SERVING_BUNDLE_VERSION,
    STRICT_SERVING_EPISODE_VERSION,
    StrictPromotionArtifactError,
    bind_strict_raw_world_artifact,
    build_strict_final_receipt,
    freeze_strict_request,
    materialize_rosters,
    scripted_agent,
    validate_materialized_rosters,
    validate_strict_final_receipt,
    write_strict_final_receipt,
)
from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V3, RASTER31V3_CONTRACT
from src.model.raster_network import RasterDuelingNetwork
from src.scripts.eval_stats import PAIRED_DELTA_PILOT_METHOD
from tests.pqn_lifecycle_fixtures import corrected_v3_pre_lifecycle_metadata
from web.backend.session import V3_ACTION_MASK_CONTRACT, _deployment_target_manifest


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _snapshot(path: Path, kind: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "artifact_kind": kind,
        "source_path": str(path),
        "source_identity": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "sha256": _sha_bytes(path.read_bytes()),
        "size_bytes": stat.st_size,
        "snapshot_path": str(path),
    }


def _world_identity(row: dict[str, object]) -> dict[str, object]:
    slots = [slot["member_sha256"] for slot in row["slots"]]
    return {
        "seed_namespace": "evaluation-world/v1",
        "seed": row["world_seed"],
        "mix_id": row["mix"],
        "ordered_slot_content_hashes": slots,
        "roster_id": canonical_digest({"slots": slots}),
    }


def _plain_contract_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_contract_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_contract_value(item) for item in value]
    return value


def _promotion_world() -> EffectiveWorldConfig:
    return EffectiveWorldConfig(
        width=400,
        height=300,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=3,
        max_frames=5000,
        initial_food=2,
        max_food=3,
        min_boost_length=5,
        boost_length_cost_frames=5,
        frame_rate=1,
        max_length=100,
        starvation_max_frames=500,
        max_capacity=400,
        kill_scale=0.3,
        death_value=-3.0,
        normalization={"max_frames": 5000.0, "starvation_max": 500.0, "max_length": 100.0},
    )


def _candidate_contracts_from_metadata(checkpoint: dict[str, object]) -> dict[str, object]:
    from src.training.pqn_lifecycle import validate_pqn_episode_lifecycle_metadata

    lifecycle = validate_pqn_episode_lifecycle_metadata(
        checkpoint,
        allow_corrected_v3_adapter=True,
    )
    return {
        "model_head_digest": checkpoint["model_head_digest"],
        "run_provenance_digest": checkpoint["run_provenance_digest"],
        "effective_world_digest": checkpoint["effective_world_digest"],
        "source_runtime": {
            "descriptor": copy.deepcopy(checkpoint["runtime_contract"]),
            "digest": checkpoint["runtime_contract_digest"],
        },
        "episode_lifecycle": {
            "descriptor": _plain_contract_value(lifecycle.descriptor),
            "digest": lifecycle.digest,
            "compatibility": (
                None
                if lifecycle.compatibility is None
                else _plain_contract_value(lifecycle.compatibility)
            ),
        },
        "policy_source": {
            "descriptor": _plain_contract_value(lifecycle.policy_source_descriptor),
            "digest": lifecycle.policy_source_digest,
        },
    }


@lru_cache(maxsize=1)
def _actual_candidate_payload() -> tuple[bytes, dict[str, object]]:
    """Create a real raster31v3 checkpoint and its frozen serving identities."""
    world = _promotion_world()
    runtime = RuntimeModeContract(
        mode="pqn_train",
        training=True,
        respawn=False,
        hero_terminal=True,
        population_floor=True,
        reset_strategy="batch_episode",
    )
    head = ModelHeadContract("pqn", "dueling_q", 6)
    world_descriptor = {
        "width": world.width,
        "height": world.height,
        "segment_size": world.segment_size,
        "wall_thickness": world.wall_thickness,
        "arena_type": world.arena_type,
        "mechanics_version": world.mechanics_version,
        "num_snakes": world.num_snakes,
        "max_frames": world.max_frames,
        "initial_food": world.initial_food,
        "max_food": world.max_food,
        "min_boost_length": world.min_boost_length,
        "boost_length_cost_frames": world.boost_length_cost_frames,
        "frame_rate": world.frame_rate,
        "max_length": world.max_length,
        "starvation_max_frames": world.starvation_max_frames,
        "arena_radius": world.arena_radius,
        "arena_center_x": world.arena_center_x,
        "arena_center_y": world.arena_center_y,
        "max_capacity": world.max_capacity,
        "kill_scale": world.kill_scale,
        "death_value": world.death_value,
        "normalization": dict(world.normalization),
    }
    runtime_descriptor = {
        "mode": runtime.mode,
        "training": runtime.training,
        "respawn": runtime.respawn,
        "hero_terminal": runtime.hero_terminal,
        "population_floor": runtime.population_floor,
        "reset_strategy": runtime.reset_strategy,
    }
    checkpoint = {
        "dqn_state_dict": RasterDuelingNetwork().state_dict(),
        "output_size": 6,
        OBS_SPEC_KEY: RASTER31V3,
        **RASTER31V3_CONTRACT.to_metadata(),
        **head.to_metadata(),
        "effective_world": world_descriptor,
        "effective_world_digest": world.digest,
        "runtime_contract": runtime_descriptor,
        "runtime_contract_digest": runtime.digest,
        "action_mask_contract": copy.deepcopy(V3_ACTION_MASK_CONTRACT),
        "action_mask_contract_digest": canonical_digest(V3_ACTION_MASK_CONTRACT),
        **corrected_v3_pre_lifecycle_metadata(
            runtime=runtime,
            observation_digest=RASTER31V3_CONTRACT.digest,
            world_digest=world.digest,
            model_head_digest=head.digest,
            action_mask_contract=V3_ACTION_MASK_CONTRACT,
            source_revision="strict-artifact-test",
        ),
    }
    payload = BytesIO()
    torch.save(checkpoint, payload)
    return payload.getvalue(), _candidate_contracts_from_metadata(checkpoint)


@lru_cache(maxsize=2)
def _native_candidate_payload(reset_mode: str) -> tuple[bytes, dict[str, object]]:
    """Build bytes from the actual B3T native checkpoint writer for one lifecycle mode."""
    from src.training.pqn_trainer import PQNConfig, PQNTrainer

    world = _promotion_world()
    trainer = PQNTrainer(
        PQNConfig(
            num_envs=1,
            num_snakes=world.num_snakes,
            rollout_len=2,
            max_frames=world.max_frames,
            game_width=world.width,
            game_height=world.height,
            segment_size=world.segment_size,
            wall_thickness=world.wall_thickness,
            arena_type=world.arena_type,
            initial_food=world.initial_food,
            max_food=world.max_food,
            min_boost_length=world.min_boost_length,
            boost_length_cost_frames=world.boost_length_cost_frames,
            frame_rate=world.frame_rate,
            max_length=world.max_length,
            starvation_max=world.starvation_max_frames,
            max_capacity=world.max_capacity,
            kill_scale=world.kill_scale,
            death_value=world.death_value,
            mechanics_version=world.mechanics_version,
            reward_version=2,
            recipe="corrected-v3",
            obs_spec=RASTER31V3,
            flip_augment=False,
            episode_reset_mode=reset_mode,
            episode_seed_mode="derived_env_episode_v1",
            source_revision=f"strict-native-{reset_mode}-test",
        )
    )
    try:
        checkpoint = trainer.checkpoint_state()
    finally:
        trainer.close()
    payload = BytesIO()
    torch.save(checkpoint, payload)
    return payload.getvalue(), _candidate_contracts_from_metadata(checkpoint)


class ArtifactFixture:
    def __init__(
        self,
        tmp_path: Path,
        *,
        candidate_payload: bytes | None = None,
        candidate_contracts: dict[str, object] | None = None,
    ) -> None:
        self.root = tmp_path
        if candidate_payload is None:
            candidate_payload, actual_contracts = _actual_candidate_payload()
            if candidate_contracts is None:
                candidate_contracts = copy.deepcopy(actual_contracts)
        self.repo = tmp_path / "repo"
        required_sources = (
            "src/scripts/tournament_eval.py",
            "src/scripts/eval_cli.py",
            "src/scripts/eval_stats.py",
            "src/core/config_loader.py",
            "src/core/game_config.py",
            "src/core/runtime_contract.py",
            "src/core/seeding.py",
            "src/core/mechanics_constants.py",
            "src/core/reward_events.py",
            "src/evaluation/strict_promotion.py",
            "src/evaluation/artifacts.py",
            "src/evaluation/protocol.py",
            "src/evaluation/metrics.py",
            "src/evaluation/anchors.py",
            "src/evaluation/serving_episode.py",
            "src/game/game_state_factory.py",
            "src/game/game_state.py",
            "src/game/game_logic.py",
            "src/game/food_manager.py",
            "src/game/ai_snake.py",
            "src/game/snake.py",
            "src/game/scripted_snake.py",
            "src/game/snake_factory.py",
            "src/game/snake_reward.py",
            "src/game/snake_state.py",
            "src/training/behavior_probes.py",
            "src/training/action_mask.py",
            "src/training/apex_policy.py",
            "src/training/base_dqn_policy.py",
            "src/training/checkpoint_contract.py",
            "src/training/pqn_lifecycle.py",
            "src/training/td_targets.py",
            "src/model/apex_network.py",
            "src/model/checkpoint_manager.py",
            "src/model/inference_agent.py",
            "src/model/obs_spec.py",
            "src/model/raster_network.py",
            "src/simd_env/eval_engine.py",
            "src/simd_env/live_adapter.py",
            "src/simd_env/batch_sim.py",
            "src/simd_env/featurizer.py",
            "web/backend/raster_policy.py",
            "web/backend/session.py",
            "web/backend/serialize.py",
            "web/backend/checkpoints.py",
            "web/backend/state_labels.py",
            "src/scripts/serving_spot_check.py",
        )
        for index, relative in enumerate(required_sources):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen source {index}\n")
        self.source_manifest = {
            relative: _sha_bytes((self.repo / relative).read_bytes())
            for relative in required_sources
        }
        self.source_closure = canonical_digest(self.source_manifest)
        self.evaluator = self.repo / "src/scripts/tournament_eval.py"
        self.source = {
            "evaluator_sha256": _sha_bytes(self.evaluator.read_bytes()),
            "source_manifest": self.source_manifest,
            "closure_sha256": self.source_closure,
            "git_commit": "a" * 40,
        }

        self.candidate_file = self._blob("candidate.pth", candidate_payload)
        self.incumbent_file = self._blob("incumbent.pth", b"incumbent")
        self.pool_files = [self._blob("pool-a.pth", b"pool-a"), self._blob("pool-b.pth", b"pool-b")]
        self.candidate = {"kind": "checkpoint", "sha256": _sha_bytes(candidate_payload)}
        self.incumbent = {"kind": "checkpoint", "sha256": _sha_bytes(b"incumbent")}
        self.pool = [
            {"kind": "checkpoint", "sha256": _sha_bytes(b"pool-a")},
            {"kind": "checkpoint", "sha256": _sha_bytes(b"pool-b")},
        ]
        self.scripts = [scripted_agent("greedy_food"), scripted_agent("random_safe")]

        world = _promotion_world()
        profile = promotion_v2_watch_rect(world)
        self.profile = {"descriptor": profile.descriptor(), "digest": profile.digest}
        self.namespaces = {
            "training": [1],
            "development": [2, 3],
            "shakedown": [4],
            "pilot": [5, 6, 7],
            "final": list(range(100, 141)),
            "serving": list(range(200, 250)),
        }
        mixed_rules = [
            {
                "slot": 1,
                "eligible_member_sha256s": [agent["sha256"] for agent in self.pool],
            },
            {
                "slot": 2,
                "eligible_member_sha256s": [scripted_agent("random_safe")["sha256"]],
            },
        ]
        self.roster_design = {
            "mixes": ["frozen", "scripted", "mixed"],
            "roster_width": 2,
            "mixed_slot_rules": mixed_rules,
        }
        self.rosters = materialize_rosters(
            final_world_seeds=self.namespaces["final"],
            roster_width=2,
            checkpoint_pool=self.pool,
            scripted_anchors=self.scripts,
            mixed_slot_rules=mixed_rules,
        )
        self.band = {
            "name": "candidate-food",
            "metric": "probes.food_eaten",
            "subject": "candidate",
            "mix_scope": ["frozen", "scripted", "mixed"],
            "reducer": "mean",
            "denominator": "candidate_final_world_records",
            "lower": 1.0,
            "upper": 3.0,
            "bound_source_sha256": "0" * 64,
            "calibration_method": "reference-mean-plus-offsets/v1",
            "lower_offset": -1.0,
            "upper_offset": 1.0,
        }
        self.calibration_decl = {
            "absolute_delta_ni": 0.5,
            "mde_by_mix": {"frozen": 10.0, "scripted": 10.0, "mixed": 10.0},
            "behavioral_bands": [self.band],
        }
        self.e0_path = self._e0()
        self.pilot_path = self._pilot()
        self.calibration_path = self._calibration()
        self.action_contract = {
            "descriptor": {
                "version": "legal-advisory-resolved-v1",
                "action_count": 6,
                "resolution": "row_local_intersection_else_legal",
                "dead_rows": "all_false",
            },
        }
        self.action_contract["digest"] = canonical_digest(self.action_contract["descriptor"])
        self.observation_contract = {"descriptor": RASTER31V3_CONTRACT.semantic_dict()}
        self.observation_contract["digest"] = canonical_digest(
            self.observation_contract["descriptor"]
        )
        self.serving_plan = {
            "schema_version": "strict-serving-plan/v1",
            "frame_limit": 5000,
            "watch_completion": "external-horizon",
            "play_completion": "human-terminal-or-external-horizon",
            "episodes": [
                {
                    "episode_id": f"serving-{index:02d}",
                    "serving_seed": seed,
                    "mode": "watch" if index == 0 else "play",
                }
                for index, seed in enumerate(self.namespaces["serving"])
            ],
        }
        if candidate_contracts is None:
            raise AssertionError("candidate lifecycle contracts were not materialized")
        self.candidate_contracts = copy.deepcopy(candidate_contracts)
        self.serving_path = self._serving()
        self.request = {
            "schema_version": STRICT_REQUEST_VERSION,
            "engine": "live",
            "candidate": self.candidate,
            "incumbent": self.incumbent,
            "checkpoint_pool": self.pool,
            "scripted_anchors": self.scripts,
            "evaluator_source": self.source,
            "profile": self.profile,
            "seed_namespaces": self.namespaces,
            "artifact_bindings": {
                "e0": _sha_bytes(self.e0_path.read_bytes()),
                "pilot": _sha_bytes(self.pilot_path.read_bytes()),
                "calibration": _sha_bytes(self.calibration_path.read_bytes()),
                "serving": _sha_bytes(self.serving_path.read_bytes()),
            },
            "calibration": self.calibration_decl,
            "serving_contract": {
                "observation": self.observation_contract,
                "action": self.action_contract,
                "deployment_profile_digest": self.profile["digest"],
                "source_closure_sha256": self.source_closure,
                "plan": self.serving_plan,
                "candidate_contracts": self.candidate_contracts,
            },
            "roster_design": self.roster_design,
            "materialized_rosters": self.rosters,
        }
        self.request_path = _write_json(self.root / "request.json", self.request)
        self.raw_path = self._raw(candidate_mass=11.0, incumbent_mass=10.0)

    def _blob(self, name: str, value: bytes) -> Path:
        path = self.root / "snapshots" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def _e0(self) -> Path:
        config = self._blob("promotion.yaml", b"game: {}\n")
        snapshots = [
            _snapshot(self.candidate_file, "checkpoints"),
            _snapshot(self.incumbent_file, "checkpoints"),
            *[_snapshot(path, "checkpoints") for path in self.pool_files],
        ]
        payload = {
            "artifact_version": "evaluation-input-snapshot-v1",
            "checkpoint_digest_algorithm": "sha256",
            "checkpoint_snapshots": snapshots,
            "config": {**_snapshot(config, "configs"), "effective": {"game": "resolved"}},
            "evaluator": {
                "source_path": str(self.evaluator),
                "sha256": self.source["evaluator_sha256"],
                "source_manifest": self.source_manifest,
                "git": {"commit": "a" * 40, "dirty": False, "diff_sha256": _sha_bytes(b"")},
            },
            "source_agents": [
                {"kind": "checkpoint", "reference": str(self.incumbent_file)},
                *[{"kind": "checkpoint", "reference": str(path)} for path in self.pool_files],
                {"kind": "checkpoint", "reference": str(self.candidate_file)},
            ],
            "evaluation_profile": self.profile,
        }
        return _write_json(self.root / "e0.json", payload)

    def _pilot(self) -> Path:
        pilot_rosters = materialize_rosters(
            final_world_seeds=self.namespaces["pilot"],
            roster_width=2,
            checkpoint_pool=self.pool,
            scripted_anchors=self.scripts,
            mixed_slot_rules=self.roster_design["mixed_slot_rules"],
        )
        pairs = [
            {
                "mix": roster["mix"],
                "world_seed": roster["world_seed"],
                "world_identity": _world_identity(roster),
                "candidate_mass_integral": 11.0,
                "incumbent_mass_integral": 10.0,
            }
            for roster in pilot_rosters
        ]
        return _write_json(
            self.root / "pilot.json",
            {
                "schema_version": STRICT_PILOT_VERSION,
                "candidate_sha256": self.candidate["sha256"],
                "incumbent_sha256": self.incumbent["sha256"],
                "profile_digest": self.profile["digest"],
                "source_closure_sha256": self.source_closure,
                "metric_version": self.profile["descriptor"]["metric_version"],
                "sizing_method": PAIRED_DELTA_PILOT_METHOD,
                "declared_per_mix_power": 0.8,
                "family_alpha": 0.05,
                "mde_by_mix": self.calibration_decl["mde_by_mix"],
                "pairs": pairs,
            },
        )

    def _calibration(self) -> Path:
        development_rosters = materialize_rosters(
            final_world_seeds=self.namespaces["development"],
            roster_width=2,
            checkpoint_pool=self.pool,
            scripted_anchors=self.scripts,
            mixed_slot_rules=self.roster_design["mixed_slot_rules"],
        )
        reference_records = [
            {
                "mix": roster["mix"],
                "world_seed": roster["world_seed"],
                "checkpoint_sha256": self.incumbent["sha256"],
                "record": self._record(roster, 100.0),
            }
            for roster in development_rosters
        ]
        reference_source = canonical_digest({"records": reference_records})
        self.band["bound_source_sha256"] = reference_source
        return _write_json(
            self.root / "calibration.json",
            {
                "schema_version": STRICT_CALIBRATION_VERSION,
                "reference_source_sha256": reference_source,
                "reference_checkpoint_sha256": self.incumbent["sha256"],
                "profile_digest": self.profile["digest"],
                "source_closure_sha256": self.source_closure,
                "seed_namespace": "development",
                "seeds": self.namespaces["development"],
                "effect_derivation_method": "positive-reference-mean-fraction/v1",
                "absolute_delta_ni_fraction": 0.005,
                "mde_fraction_by_mix": {
                    "frozen": 0.1,
                    "scripted": 0.1,
                    "mixed": 0.1,
                },
                "reference_records": reference_records,
                **self.calibration_decl,
            },
        )

    def _serving(self) -> Path:
        action = self.action_contract["digest"]
        observation = self.observation_contract["digest"]
        descriptor = self.profile["descriptor"]
        lifecycle_fields = {
            "episode_lifecycle_contract": self.candidate_contracts["episode_lifecycle"][
                "descriptor"
            ],
            "episode_lifecycle_contract_digest": self.candidate_contracts["episode_lifecycle"][
                "digest"
            ],
            "episode_lifecycle_compatibility": self.candidate_contracts["episode_lifecycle"][
                "compatibility"
            ],
            "policy_source_contract": self.candidate_contracts["policy_source"]["descriptor"],
            "policy_source_contract_digest": self.candidate_contracts["policy_source"]["digest"],
        }
        episodes = []
        for index, seed in enumerate(self.namespaces["serving"]):
            episode_id = f"serving-{index:02d}"
            runtime_mode = "watch" if index == 0 else "play"
            target = _deployment_target_manifest(
                EffectiveWorldConfig(**descriptor["world"]),
                RuntimeModeContract(**self.candidate_contracts["source_runtime"]["descriptor"]),
                self.candidate["sha256"],
                runtime_mode,
                lifecycle_fields=lifecycle_fields,
            )
            serving_contract = {
                "obs_contract_digest": observation,
                "model_head_digest": self.candidate_contracts["model_head_digest"],
                "effective_world_digest": descriptor["world_digest"],
                "runtime_contract_digest": self.candidate_contracts["source_runtime"]["digest"],
                "action_mask_contract_digest": action,
                "run_provenance_digest": self.candidate_contracts["run_provenance_digest"],
                **lifecycle_fields,
                "deployment_profile": descriptor["name"],
                "deployment_target_manifest": target,
                "deployment_target_manifest_digest": canonical_digest(target),
                "checkpoint_sha256": self.candidate["sha256"],
            }
            participants = [
                {
                    "slot": slot,
                    "snake_id": slot,
                    "controller": (
                        "human" if runtime_mode == "play" and slot == 0 else "candidate_ai"
                    ),
                    "checkpoint_sha256": (
                        None if runtime_mode == "play" and slot == 0 else self.candidate["sha256"]
                    ),
                    "controller_class": (
                        "HumanSnake" if runtime_mode == "play" and slot == 0 else "AISnake"
                    ),
                    "policy_binding": (
                        None if runtime_mode == "play" and slot == 0 else "session_shared_policy"
                    ),
                }
                for slot in range(3)
            ]
            initial_descriptor = {
                "mode": runtime_mode,
                "frame": 0,
                "ordered_snakes": [
                    {
                        "slot": participant["slot"],
                        "snake_id": participant["snake_id"],
                        "controller_class": participant["controller_class"],
                        "head": [20 + 20 * participant["slot"], 20],
                        "segments": [[20 + 20 * participant["slot"], 20]],
                        "direction": [1, 0],
                        "length": 1,
                        "alive": True,
                        "auto_respawn": participant["slot"] != 0,
                    }
                    for participant in participants
                ],
                "food_positions": [[100, 100], [120, 100]],
                "ordered_slot_to_snake_id": [
                    participant["snake_id"] for participant in participants
                ],
            }
            candidate_slots = list(range(3)) if runtime_mode == "watch" else [1, 2]
            receipt_path = _write_json(
                self.root / "serving" / f"{episode_id}.json",
                {
                    "schema_version": STRICT_SERVING_EPISODE_VERSION,
                    "episode_id": episode_id,
                    "serving_seed": seed,
                    "mode": runtime_mode,
                    "checkpoint_path": str(self.candidate_file),
                    "candidate_sha256": self.candidate["sha256"],
                    "profile_digest": self.profile["digest"],
                    "source_closure_sha256": self.source_closure,
                    "frame_limit": 5000,
                    "start_frame": 0,
                    "frames_completed": 5000,
                    "end_frame": 5000,
                    "completion": {
                        "owner": "external-evaluator",
                        "reason": "external-horizon",
                        "backend_run_over": False,
                        "backend_run_frames": 0 if runtime_mode == "watch" else 5000,
                    },
                    "hero_lifecycle": {
                        "snake_id": 0,
                        "initial_alive": True,
                        "final_alive": True,
                        "alive_to_dead_transitions": 0,
                        "dead_to_alive_transitions": 0,
                        "frame_rewind_count": 0,
                        "game_instance_change_count": 0,
                    },
                    "seed_application": {
                        "api": "src.core.seeding.initialize_run_seed/v1",
                        "requested_seed": seed,
                        "effective_seed": seed,
                        "global_stream_seed": derive_seed(seed, "global"),
                        "evaluated_build_hook": (
                            "GameSession.__init__"
                            if runtime_mode == "watch"
                            else "GameSession.set_mode(play)"
                        ),
                    },
                    "initial_layout": initial_descriptor,
                    "initial_layout_digest": canonical_digest(initial_descriptor),
                    "participants": participants,
                    "dispatch_observation": {
                        "human_control_events": (
                            []
                            if runtime_mode == "watch"
                            else [
                                {
                                    "direction_name": "down",
                                    "direction_before": [1, 0],
                                    "direction_after": [0, 1],
                                    "run_started_before": False,
                                    "run_started_after": True,
                                    "accepted_observed": True,
                                }
                            ]
                        ),
                        "human_update_calls": 0 if runtime_mode == "watch" else 5000,
                        "candidate_action_context_calls": [
                            {"slot": slot, "snake_id": slot, "calls": 5000}
                            for slot in candidate_slots
                        ],
                        "context_id_mismatch_count": 0,
                        "unknown_context_id_count": 0,
                    },
                    "obs_spec": "raster31v3",
                    "serving_contract": serving_contract,
                    "status": "completed",
                    "error": None,
                },
            )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "serving_seed": seed,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": _sha_bytes(receipt_path.read_bytes()),
                }
            )
        return _write_json(
            self.root / "serving.json",
            {
                "schema_version": STRICT_SERVING_BUNDLE_VERSION,
                "candidate_sha256": self.candidate["sha256"],
                "profile_digest": self.profile["digest"],
                "action_contract_digest": action,
                "observation_contract_digest": observation,
                "source_closure_sha256": self.source_closure,
                "episodes": episodes,
            },
        )

    def _record(self, roster: dict[str, object], mass: float) -> dict[str, object]:
        return {
            "seed": roster["world_seed"],
            "evaluation_profile": self.profile["descriptor"],
            "evaluation_profile_digest": self.profile["digest"],
            "world_identity": _world_identity(roster),
            "mass_integral": mass,
            "max_mass": mass,
            "mean_mass_alive": mass,
            "survival_fraction": 1.0,
            "kills": 0,
            "deaths": 0,
            "probes": {
                "food_eaten": 2,
                "boost_frame_fraction": 0.1,
                "death_cause": None,
                "peak_length": mass,
                "kill_opportunity_count": None,
                "entrapment_event": None,
            },
            "probe_unavailable_reasons": {
                "kill_opportunity_count": "not supplied by this transition adapter",
                "entrapment_event": "temporal probe is not integrated into this accumulator",
            },
            "denominators": {
                "scored_frames": 5000,
                "decision_frames": 5000,
                "alive_frames": 5000,
                "food_event_frames": 2,
                "boost_executed_frames": 500,
            },
        }

    def raw_payload(self, candidate_mass: float, incumbent_mass: float) -> dict[str, object]:
        records = []
        for roster in self.rosters:
            for role, agent, mass in (
                ("candidate", self.candidate, candidate_mass),
                ("incumbent", self.incumbent, incumbent_mass),
            ):
                records.append(
                    {
                        "role": role,
                        "mix": roster["mix"],
                        "world_seed": roster["world_seed"],
                        "checkpoint_sha256": agent["sha256"],
                        "record": self._record(roster, mass),
                    }
                )
        return {
            "schema_version": STRICT_RAW_WORLD_VERSION,
            "engine": "live",
            "strict_request_semantic_digest": canonical_digest(self.request),
            "e0_receipt_sha256": _sha_bytes(self.e0_path.read_bytes()),
            "profile_digest": self.profile["digest"],
            "source_closure_sha256": self.source_closure,
            "materialized_rosters_digest": canonical_digest({"rows": self.rosters}),
            "records": records,
        }

    def _raw(self, candidate_mass: float, incumbent_mass: float) -> Path:
        return _write_json(self.root / "raw.json", self.raw_payload(candidate_mass, incumbent_mass))

    def freeze(self):
        return freeze_strict_request(
            self.request_path,
            e0_receipt_path=self.e0_path,
            pilot_artifact_path=self.pilot_path,
            calibration_artifact_path=self.calibration_path,
            serving_bundle_path=self.serving_path,
        )

    def build(self):
        token = self.freeze()
        raw_token = bind_strict_raw_world_artifact(token, self.raw_path)
        paths = {
            "request_path": self.request_path,
            "e0_receipt_path": self.e0_path,
            "raw_world_artifact_path": self.raw_path,
            "pilot_artifact_path": self.pilot_path,
            "calibration_artifact_path": self.calibration_path,
            "serving_bundle_path": self.serving_path,
        }
        return token, raw_token, paths, build_strict_final_receipt(token, raw_token, **paths)


def test_real_positive_synthetic_artifact_pipeline_and_archived_revalidation(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    token, raw_token, paths, receipt = fixture.build()

    assert token.required_final_worlds == 40
    assert receipt["schema_version"] == STRICT_FINAL_RECEIPT_VERSION
    assert receipt["strict_authority"] is True
    assert receipt["release_action_authorized"] is False
    assert receipt["paired_deltas_by_mix"] == {
        "frozen": [1.0] * 41,
        "scripted": [1.0] * 41,
        "mixed": [1.0] * 41,
    }
    assert len(receipt["serving_episode_receipt_sha256s"]) == 50

    archived = tmp_path / "final.json"
    write_strict_final_receipt(archived, token, raw_token, **paths)
    assert (
        validate_strict_final_receipt(archived, token, raw_token, **paths)["strict_authority"]
        is True
    )
    with pytest.raises(FileExistsError):
        write_strict_final_receipt(archived, token, raw_token, **paths)


def test_nested_request_is_immutable_and_request_file_swap_fails(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    with pytest.raises(TypeError):
        token.request["candidate"]["sha256"] = "0" * 64
    fixture.request["candidate"]["sha256"] = "0" * 64
    assert token.request["candidate"]["sha256"] != "0" * 64

    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="request bytes changed"):
        bind_strict_raw_world_artifact(token, fixture.raw_path)


def test_e0_and_raw_replacement_after_binding_fail(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    raw_token = bind_strict_raw_world_artifact(token, fixture.raw_path)
    paths = {
        "request_path": fixture.request_path,
        "e0_receipt_path": fixture.e0_path,
        "raw_world_artifact_path": fixture.raw_path,
        "pilot_artifact_path": fixture.pilot_path,
        "calibration_artifact_path": fixture.calibration_path,
        "serving_bundle_path": fixture.serving_path,
    }
    fixture.e0_path.write_bytes(fixture.e0_path.read_bytes() + b" ")
    with pytest.raises(StrictPromotionArtifactError, match="artifact bytes differ"):
        build_strict_final_receipt(token, raw_token, **paths)

    fixture = ArtifactFixture(tmp_path / "raw-swap")
    token = fixture.freeze()
    raw_token = bind_strict_raw_world_artifact(token, fixture.raw_path)
    fixture.raw_path.write_bytes(fixture.raw_path.read_bytes() + b" ")
    paths.update(
        request_path=fixture.request_path,
        e0_receipt_path=fixture.e0_path,
        raw_world_artifact_path=fixture.raw_path,
        pilot_artifact_path=fixture.pilot_path,
        calibration_artifact_path=fixture.calibration_path,
        serving_bundle_path=fixture.serving_path,
    )
    with pytest.raises(StrictPromotionArtifactError, match="request/raw artifact bytes changed"):
        build_strict_final_receipt(token, raw_token, **paths)


def test_forged_true_dictionary_cannot_grant_authority_and_raw_deltas_control(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    _write_json(fixture.raw_path, fixture.raw_payload(candidate_mass=9.0, incumbent_mass=10.0))
    raw_token = bind_strict_raw_world_artifact(token, fixture.raw_path)
    paths = {
        "request_path": fixture.request_path,
        "e0_receipt_path": fixture.e0_path,
        "raw_world_artifact_path": fixture.raw_path,
        "pilot_artifact_path": fixture.pilot_path,
        "calibration_artifact_path": fixture.calibration_path,
        "serving_bundle_path": fixture.serving_path,
    }
    receipt = build_strict_final_receipt(token, raw_token, **paths)
    assert receipt["strict_authority"] is False
    assert receipt["paired_deltas_by_mix"]["scripted"] == [-1.0] * 41

    forged = copy.deepcopy(fixture.raw_payload(11.0, 10.0))
    forged["statistics"] = {"valid": True, "passes": True, "strict_authority": True}
    _write_json(fixture.raw_path, forged)
    with pytest.raises(StrictPromotionArtifactError, match="extra=.*statistics"):
        bind_strict_raw_world_artifact(token, fixture.raw_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["records"][0].update(role="incumbent"), "role/checkpoint"),
        (lambda raw: raw["records"][0].update(world_seed=999), "duplicate/undeclared"),
        (
            lambda raw: raw["records"][0]["record"].update(evaluation_profile_digest="0" * 64),
            "identity differs",
        ),
        (lambda raw: raw["records"].pop(), "lacks full"),
    ],
)
def test_wrong_role_seed_profile_or_full_cardinality_fails(tmp_path: Path, mutation, message):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    raw = fixture.raw_payload(11.0, 10.0)
    mutation(raw)
    _write_json(fixture.raw_path, raw)
    with pytest.raises(StrictPromotionArtifactError, match=message):
        bind_strict_raw_world_artifact(token, fixture.raw_path)


def test_wrong_or_incomplete_source_closure_fails(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    request = copy.deepcopy(fixture.request)
    request["evaluator_source"]["source_manifest"] = {
        "src/scripts/tournament_eval.py": fixture.source["evaluator_sha256"]
    }
    request["evaluator_source"]["closure_sha256"] = canonical_digest(
        request["evaluator_source"]["source_manifest"]
    )
    _write_json(fixture.request_path, request)
    with pytest.raises(StrictPromotionArtifactError, match="closure is incomplete"):
        fixture.freeze()


def test_e0_ordered_roles_cannot_swap_candidate_and_incumbent(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    e0 = json.loads(fixture.e0_path.read_text())
    e0["source_agents"][0], e0["source_agents"][-1] = (
        e0["source_agents"][-1],
        e0["source_agents"][0],
    )
    _write_json(fixture.e0_path, e0)
    fixture.request["artifact_bindings"]["e0"] = _sha_bytes(fixture.e0_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="ordered source roles"):
        fixture.freeze()


def test_pilot_uses_canonical_mix_and_slot_rosters(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    pilot = json.loads(fixture.pilot_path.read_text())
    scripted = next(pair for pair in pilot["pairs"] if pair["mix"] == "scripted")
    scripted["world_identity"]["ordered_slot_content_hashes"][0] = fixture.pool[0]["sha256"]
    scripted["world_identity"]["roster_id"] = canonical_digest(
        {"slots": scripted["world_identity"]["ordered_slot_content_hashes"]}
    )
    _write_json(fixture.pilot_path, pilot)
    fixture.request["artifact_bindings"]["pilot"] = _sha_bytes(fixture.pilot_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="pilot world identity"):
        fixture.freeze()


def test_mixed_roster_rules_are_canonical_not_arbitrary(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    request = copy.deepcopy(fixture.request)
    request["roster_design"]["mixed_slot_rules"][1]["eligible_member_sha256s"] = [
        fixture.pool[0]["sha256"]
    ]
    _write_json(fixture.request_path, request)
    with pytest.raises(StrictPromotionArtifactError, match="must alternate"):
        fixture.freeze()


def test_pilot_requirement_is_recomputed_not_claimed(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    pilot = json.loads(fixture.pilot_path.read_text())
    for index, pair in enumerate(pilot["pairs"]):
        pair["candidate_mass_integral"] = 100.0 if index % 3 == 0 else 0.0
        pair["incumbent_mass_integral"] = 0.0
    _write_json(fixture.pilot_path, pilot)
    fixture.request["artifact_bindings"]["pilot"] = _sha_bytes(fixture.pilot_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="recomputed pilot requirement"):
        fixture.freeze()


def test_roster_balance_is_per_slot_for_nondivisible_world_count(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    validate_materialized_rosters(
        fixture.rosters,
        final_world_seeds=fixture.namespaces["final"],
        roster_design=fixture.roster_design,
        checkpoint_pool=fixture.pool,
        scripted_anchors=fixture.scripts,
    )
    frozen = [row for row in fixture.rosters if row["mix"] == "frozen"]
    # Keep aggregate totals unchanged but force slot 1 heavily toward one member.
    for index, row in enumerate(frozen):
        row["slots"][0]["member_sha256"] = (
            fixture.pool[0]["sha256"] if index < 30 else fixture.pool[1]["sha256"]
        )
        row["slots"][1]["member_sha256"] = (
            fixture.pool[1]["sha256"] if index < 30 else fixture.pool[0]["sha256"]
        )
    with pytest.raises(StrictPromotionArtifactError, match="canonical slot-phased|unbalanced"):
        validate_materialized_rosters(
            fixture.rosters,
            final_world_seeds=fixture.namespaces["final"],
            roster_design=fixture.roster_design,
            checkpoint_pool=fixture.pool,
            scripted_anchors=fixture.scripts,
        )


def test_mixed_checkpoint_slots_use_distinct_slot_phases():
    pool = [
        {"kind": "checkpoint", "sha256": _sha_bytes(b"pool-a")},
        {"kind": "checkpoint", "sha256": _sha_bytes(b"pool-b")},
    ]
    scripts = [scripted_agent("greedy_food"), scripted_agent("random_safe")]
    random_safe = scripts[1]["sha256"]
    rules = [
        {
            "slot": slot,
            "eligible_member_sha256s": (
                [member["sha256"] for member in pool] if slot % 2 else [random_safe]
            ),
        }
        for slot in range(1, 6)
    ]
    rows = materialize_rosters(
        final_world_seeds=[11, 12],
        roster_width=5,
        checkpoint_pool=pool,
        scripted_anchors=scripts,
        mixed_slot_rules=rules,
    )
    mixed = [row for row in rows if row["mix"] == "mixed"]
    assert [slot["member_sha256"] for slot in mixed[0]["slots"][::2]] == [
        pool[0]["sha256"],
        pool[1]["sha256"],
        pool[0]["sha256"],
    ]
    assert [slot["member_sha256"] for slot in mixed[1]["slots"][::2]] == [
        pool[1]["sha256"],
        pool[0]["sha256"],
        pool[1]["sha256"],
    ]


def test_null_unimplemented_required_band_is_denied_before_final_worlds(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    calibration = json.loads(fixture.calibration_path.read_text())
    calibration["behavioral_bands"][0]["metric"] = "probes.kill_opportunity_count"
    fixture.request["calibration"]["behavioral_bands"][0][
        "metric"
    ] = "probes.kill_opportunity_count"
    _write_json(fixture.calibration_path, calibration)
    fixture.request["artifact_bindings"]["calibration"] = _sha_bytes(
        fixture.calibration_path.read_bytes()
    )
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="unavailable/unimplemented"):
        fixture.freeze()


def test_serving_requires_50_unique_actual_evidenced_episodes(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    serving = json.loads(fixture.serving_path.read_text())
    serving["episodes"][1]["serving_seed"] = serving["episodes"][0]["serving_seed"]
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="unique IDs and declared seeds"):
        fixture.freeze()

    fixture = ArtifactFixture(tmp_path / "status-only")
    serving = json.loads(fixture.serving_path.read_text())
    serving["episodes"] = [{"episode_id": str(i), "status": True} for i in range(50)]
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="serving episode reference"):
        fixture.freeze()


def test_legacy_operability_spot_check_is_not_a_strict_episode(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    serving = json.loads(fixture.serving_path.read_text())
    episode = serving["episodes"][0]
    receipt_path = Path(episode["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    legacy = {
        "checkpoint": receipt["checkpoint_path"],
        "frames_requested": 100,
        "frame": 100,
        "obs_spec": receipt["obs_spec"],
        "serving_contract": receipt["serving_contract"],
        "error": None,
    }
    _write_json(receipt_path, legacy)
    episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="serving episode receipt"):
        fixture.freeze()


def test_serving_plan_requires_one_watch_and_49_play_rows(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    fixture.request["serving_contract"]["plan"]["episodes"][0]["mode"] = "play"
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="one Watch and 49 Play"):
        fixture.freeze()


def test_frozen_request_rejects_simultaneous_serving_bundle_and_receipt_rewrite(
    tmp_path: Path,
):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    serving = json.loads(fixture.serving_path.read_text())
    forged_action = _sha_bytes(b"forged-action")
    serving["action_contract_digest"] = forged_action
    for episode in serving["episodes"]:
        receipt_path = Path(episode["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["serving_contract"]["action_mask_contract_digest"] = forged_action
        _write_json(receipt_path, receipt)
        episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    raw_token = bind_strict_raw_world_artifact(token, fixture.raw_path)
    paths = {
        "request_path": fixture.request_path,
        "e0_receipt_path": fixture.e0_path,
        "raw_world_artifact_path": fixture.raw_path,
        "pilot_artifact_path": fixture.pilot_path,
        "calibration_artifact_path": fixture.calibration_path,
        "serving_bundle_path": fixture.serving_path,
    }
    with pytest.raises(StrictPromotionArtifactError, match="artifact bytes differ"):
        build_strict_final_receipt(token, raw_token, **paths)


def test_coherent_preflight_serving_contract_substitution_is_not_accepted(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    serving = json.loads(fixture.serving_path.read_text())
    forged_descriptor = {
        "version": "invented-mask/v9",
        "action_count": 6,
        "resolution": "caller_claimed",
        "dead_rows": "all_false",
    }
    forged_digest = canonical_digest(forged_descriptor)
    fixture.request["serving_contract"]["action"] = {
        "descriptor": forged_descriptor,
        "digest": forged_digest,
    }
    serving["action_contract_digest"] = forged_digest
    for episode in serving["episodes"]:
        receipt_path = Path(episode["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["serving_contract"]["action_mask_contract_digest"] = forged_digest
        _write_json(receipt_path, receipt)
        episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="accepted typed contract"):
        fixture.freeze()


def test_frozen_lifecycle_claim_must_match_opened_e0_candidate_bytes(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    lifecycle = fixture.request["serving_contract"]["candidate_contracts"]["episode_lifecycle"]
    lifecycle["descriptor"]["episode_reset_mode"] = "per_env_autoreset_v1"
    lifecycle["digest"] = canonical_digest(lifecycle["descriptor"])
    lifecycle["compatibility"] = None
    policy_source = fixture.request["serving_contract"]["candidate_contracts"]["policy_source"]
    policy_source["descriptor"]["effective_episode_lifecycle_contract_digest"] = lifecycle["digest"]
    policy_source["digest"] = canonical_digest(policy_source["descriptor"])
    _write_json(fixture.request_path, fixture.request)

    with pytest.raises(
        StrictPromotionArtifactError,
        match="checkpoint contracts differ",
    ):
        fixture.freeze()


def test_recognized_reset_strategy_without_matching_checkpoint_lifecycle_fails(
    tmp_path: Path,
):
    fixture = ArtifactFixture(tmp_path)
    source_runtime = fixture.request["serving_contract"]["candidate_contracts"]["source_runtime"]
    source_runtime["descriptor"]["reset_strategy"] = "per_env_rollout_boundary"
    source_runtime["digest"] = canonical_digest(source_runtime["descriptor"])
    _write_json(fixture.request_path, fixture.request)

    with pytest.raises(
        StrictPromotionArtifactError,
        match="checkpoint contracts differ",
    ):
        fixture.freeze()


@pytest.mark.parametrize("completed_frames", [1, 100])
def test_serving_spot_check_or_partial_episode_cannot_claim_readiness(
    tmp_path: Path, completed_frames: int
):
    fixture = ArtifactFixture(tmp_path)
    serving = json.loads(fixture.serving_path.read_text())
    episode = serving["episodes"][0]
    receipt_path = Path(episode["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(
        frames_completed=completed_frames,
        end_frame=completed_frames,
    )
    for decision in receipt["dispatch_observation"]["candidate_action_context_calls"]:
        decision["calls"] = completed_frames
    _write_json(receipt_path, receipt)
    episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="terminal or external-horizon"):
        fixture.freeze()


def test_play_episode_may_complete_at_observed_human_terminal(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    serving = json.loads(fixture.serving_path.read_text())
    episode = serving["episodes"][1]
    receipt_path = Path(episode["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update(frames_completed=100, end_frame=100)
    receipt["completion"] = {
        "owner": "GameSession._step_play",
        "reason": "human-terminal",
        "backend_run_over": True,
        "backend_run_frames": 99,
    }
    receipt["hero_lifecycle"].update(final_alive=False, alive_to_dead_transitions=1)
    receipt["dispatch_observation"]["human_update_calls"] = 100
    for decision in receipt["dispatch_observation"]["candidate_action_context_calls"]:
        decision["calls"] = 100
    _write_json(receipt_path, receipt)
    episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    fixture.freeze()


def test_actual_producer_terminal_play_receipt_is_accepted(tmp_path: Path):
    """Prove the strict consumer accepts bytes emitted by the real producer."""
    from src.core import game_config

    candidate_payload, candidate_contracts = _actual_candidate_payload()
    fixture = ArtifactFixture(
        tmp_path,
        candidate_payload=candidate_payload,
        candidate_contracts=candidate_contracts,
    )
    serving = json.loads(fixture.serving_path.read_text())
    episode = serving["episodes"][1]
    spec = ServingEpisodeSpec(
        episode_id=episode["episode_id"],
        mode="play",
        serving_seed=episode["serving_seed"],
        checkpoint_path=fixture.candidate_file,
        expected_candidate_sha256=fixture.candidate["sha256"],
        profile_digest=fixture.profile["digest"],
        source_closure_sha256=fixture.source_closure,
        frame_limit=5000,
    )
    previous_config = game_config._current_config
    try:
        receipt = run_serving_episode(spec)
    finally:
        game_config._current_config = previous_config

    assert receipt["schema_version"] == STRICT_SERVING_EPISODE_VERSION
    assert receipt["completion"]["reason"] == "human-terminal"
    assert 1 <= receipt["frames_completed"] < 5000
    receipt_path = _write_json(Path(episode["receipt_path"]), receipt)
    episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)

    fixture.freeze()


def test_native_per_env_candidate_lifecycle_freezes_from_actual_checkpoint_bytes(
    tmp_path: Path,
):
    candidate_payload, candidate_contracts = _native_candidate_payload("per_env_autoreset_v1")
    fixture = ArtifactFixture(
        tmp_path,
        candidate_payload=candidate_payload,
        candidate_contracts=candidate_contracts,
    )

    token = fixture.freeze()

    frozen = token.request["serving_contract"]["candidate_contracts"]
    assert frozen["source_runtime"]["descriptor"]["reset_strategy"] == ("per_env_rollout_boundary")
    assert frozen["episode_lifecycle"]["descriptor"]["episode_reset_mode"] == (
        "per_env_autoreset_v1"
    )
    assert frozen["episode_lifecycle"]["compatibility"] is None


@pytest.mark.parametrize(
    ("reset_mode", "source_reset"),
    [
        ("batch_barrier_v1", "batch_episode"),
        ("per_env_autoreset_v1", "per_env_rollout_boundary"),
    ],
)
def test_native_writer_checkpoint_reaches_real_play_receipt_and_strict_final_consumer(
    tmp_path: Path, reset_mode: str, source_reset: str
):
    """Exercise native writer bytes through the real Play producer and strict reopen path."""
    candidate_payload, candidate_contracts = _native_candidate_payload(reset_mode)
    fixture = ArtifactFixture(
        tmp_path,
        candidate_payload=candidate_payload,
        candidate_contracts=candidate_contracts,
    )
    serving = json.loads(fixture.serving_path.read_text())
    episode = serving["episodes"][1]
    receipt = run_serving_episode(
        ServingEpisodeSpec(
            episode_id=episode["episode_id"],
            mode="play",
            serving_seed=episode["serving_seed"],
            checkpoint_path=fixture.candidate_file,
            expected_candidate_sha256=fixture.candidate["sha256"],
            profile_digest=fixture.profile["digest"],
            source_closure_sha256=fixture.source_closure,
            frame_limit=5000,
        )
    )
    receipt_path = _write_json(Path(episode["receipt_path"]), receipt)
    episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    fixture.raw_path = fixture._raw(candidate_mass=11.0, incumbent_mass=10.0)

    token, raw_token, paths, final = fixture.build()
    assert final["strict_authority"] is True
    assert final["readiness"]["artifact_validation"] == "passed"
    assert (
        token.request["serving_contract"]["candidate_contracts"]["source_runtime"]["descriptor"][
            "reset_strategy"
        ]
        == source_reset
    )
    assert (
        token.request["serving_contract"]["candidate_contracts"]["episode_lifecycle"][
            "compatibility"
        ]
        is None
    )
    assert bind_strict_raw_world_artifact(token, fixture.raw_path) == raw_token
    assert (
        validate_strict_final_receipt(
            _write_json(tmp_path / "final.json", final), token, raw_token, **paths
        )["strict_authority"]
        is True
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(error="policy failed"), "matching executed evidence"),
        (lambda receipt: receipt.update(end_frame=4999), "frame counters"),
        (
            lambda receipt: receipt["serving_contract"].update(obs_contract_digest="0" * 64),
            "runtime contract differs",
        ),
        (
            lambda receipt: receipt["serving_contract"].update(
                model_head_digest=_sha_bytes(b"other-model-head")
            ),
            "runtime contract differs",
        ),
        (
            lambda receipt: receipt["seed_application"].update(effective_seed=999),
            "application of its frozen seed",
        ),
        (
            lambda receipt: receipt["participants"][1].update(controller="unbound_ai"),
            "human/AI dispatch",
        ),
    ],
)
def test_serving_raw_s1_error_frame_contract_seed_and_dispatch_are_validated(
    tmp_path: Path, mutation, message: str
):
    fixture = ArtifactFixture(tmp_path)
    serving = json.loads(fixture.serving_path.read_text())
    episode = serving["episodes"][0]
    receipt_path = Path(episode["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    _write_json(receipt_path, receipt)
    episode["receipt_sha256"] = _sha_bytes(receipt_path.read_bytes())
    _write_json(fixture.serving_path, serving)
    fixture.request["artifact_bindings"]["serving"] = _sha_bytes(fixture.serving_path.read_bytes())
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match=message):
        fixture.freeze()


@pytest.mark.parametrize("source_path", ["src/game/game_logic.py", "web/backend/session.py"])
def test_mutating_live_or_serving_source_after_freeze_is_detected(tmp_path: Path, source_path: str):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    raw_token = bind_strict_raw_world_artifact(token, fixture.raw_path)
    (fixture.repo / source_path).write_text("# changed authoritative source\n")
    paths = {
        "request_path": fixture.request_path,
        "e0_receipt_path": fixture.e0_path,
        "raw_world_artifact_path": fixture.raw_path,
        "pilot_artifact_path": fixture.pilot_path,
        "calibration_artifact_path": fixture.calibration_path,
        "serving_bundle_path": fixture.serving_path,
    }
    with pytest.raises(StrictPromotionArtifactError, match="evaluator source changed"):
        build_strict_final_receipt(token, raw_token, **paths)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record["probes"].update(entrapment_event="false"),
        lambda record: record.update(survival_fraction=0.5),
        lambda record: record["denominators"].update(food_event_frames=99),
        lambda record: record["probe_unavailable_reasons"].update(food_eaten="missing"),
        lambda record: record.update(max_mass=1.0),
    ],
)
def test_malformed_or_contradictory_e1_probe_evidence_fails(tmp_path: Path, mutation):
    fixture = ArtifactFixture(tmp_path)
    token = fixture.freeze()
    raw = fixture.raw_payload(11.0, 10.0)
    mutation(raw["records"][0]["record"])
    _write_json(fixture.raw_path, raw)
    with pytest.raises(StrictPromotionArtifactError):
        bind_strict_raw_world_artifact(token, fixture.raw_path)


def test_calibration_numbers_are_recomputed_from_raw_reference_records(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    calibration = json.loads(fixture.calibration_path.read_text())
    calibration["absolute_delta_ni"] = 99.0
    fixture.request["calibration"]["absolute_delta_ni"] = 99.0
    _write_json(fixture.calibration_path, calibration)
    fixture.request["artifact_bindings"]["calibration"] = _sha_bytes(
        fixture.calibration_path.read_bytes()
    )
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="calibration values differ"):
        fixture.freeze()


def test_calibration_source_hash_binds_actual_reference_records(tmp_path: Path):
    fixture = ArtifactFixture(tmp_path)
    calibration = json.loads(fixture.calibration_path.read_text())
    calibration["reference_records"][0]["record"]["mass_integral"] = 0.0
    calibration["reference_records"][0]["record"]["mean_mass_alive"] = 0.0
    _write_json(fixture.calibration_path, calibration)
    fixture.request["artifact_bindings"]["calibration"] = _sha_bytes(
        fixture.calibration_path.read_bytes()
    )
    _write_json(fixture.request_path, fixture.request)
    with pytest.raises(StrictPromotionArtifactError, match="source hash does not bind"):
        fixture.freeze()
