"""Public-path evidence tests for bounded real raster serving episodes."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.core import runtime_contract
from src.core.seeding import derive_seed
from src.evaluation import serving_episode
from src.model.inference_agent import InferenceAgent
from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V3, RASTER31V3_CONTRACT
from src.model.raster_network import RasterDuelingNetwork
from src.training.pqn_trainer import PQNConfig, PQNTrainer
from tests.pqn_lifecycle_fixtures import corrected_v3_pre_lifecycle_metadata
from web.backend.session import V3_ACTION_MASK_CONTRACT


@pytest.fixture(autouse=True)
def _restore_global_config():
    """GameSession owns a process-global deployment config."""
    from src.core import game_config

    before = game_config._current_config
    yield
    game_config._current_config = before


def _metadata() -> dict:
    world = runtime_contract.EffectiveWorldConfig(
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
        kill_scale=0.3,
        death_value=-3.0,
        normalization={"max_frames": 5000.0, "starvation_max": 500.0, "max_length": 150.0},
    )
    runtime = runtime_contract.RuntimeModeContract(
        mode="pqn_train",
        training=True,
        respawn=False,
        hero_terminal=True,
        population_floor=True,
        reset_strategy="batch_episode",
    )
    head = runtime_contract.ModelHeadContract("pqn", "dueling_q", 6)
    return {
        OBS_SPEC_KEY: RASTER31V3,
        **RASTER31V3_CONTRACT.to_metadata(),
        **head.to_metadata(),
        "effective_world": {
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
        },
        "effective_world_digest": world.digest,
        "runtime_contract": {
            "mode": runtime.mode,
            "training": runtime.training,
            "respawn": runtime.respawn,
            "hero_terminal": runtime.hero_terminal,
            "population_floor": runtime.population_floor,
            "reset_strategy": runtime.reset_strategy,
        },
        "runtime_contract_digest": runtime.digest,
        "action_mask_contract": copy.deepcopy(V3_ACTION_MASK_CONTRACT),
        "action_mask_contract_digest": runtime_contract.canonical_digest(V3_ACTION_MASK_CONTRACT),
        **corrected_v3_pre_lifecycle_metadata(
            runtime=runtime,
            observation_digest=RASTER31V3_CONTRACT.digest,
            world_digest=world.digest,
            model_head_digest=head.digest,
            action_mask_contract=V3_ACTION_MASK_CONTRACT,
        ),
    }


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.pth"
    torch.save(
        {"dqn_state_dict": RasterDuelingNetwork().state_dict(), "output_size": 6, **_metadata()},
        path,
    )
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(checkpoint: Path, *, mode: str, seed: int = 31, frames: int = 3):
    return serving_episode.ServingEpisodeSpec(
        episode_id=f"episode-{mode}-{seed}",
        mode=mode,
        serving_seed=seed,
        checkpoint_path=checkpoint,
        expected_candidate_sha256=_digest(checkpoint),
        profile_digest="a" * 64,
        source_closure_sha256="b" * 64,
        frame_limit=frames,
    )


def _native_checkpoint(tmp_path: Path, reset_mode: str) -> Path:
    """Save a checkpoint from the current corrected-v3 writer, not hand-built metadata."""
    trainer = PQNTrainer(
        PQNConfig(
            num_envs=1,
            num_snakes=3,
            rollout_len=2,
            max_frames=5000,
            game_width=400,
            game_height=300,
            initial_food=2,
            max_food=3,
            recipe="corrected-v3",
            obs_spec=RASTER31V3,
            flip_augment=False,
            mechanics_version=2,
            reward_version=2,
            episode_reset_mode=reset_mode,
            episode_seed_mode="derived_env_episode_v1",
            source_revision=f"serving-native-{reset_mode}",
        )
    )
    try:
        state = trainer.checkpoint_state()
    finally:
        trainer.close()
    path = tmp_path / f"{reset_mode}.pth"
    torch.save(state, path)
    return path


@pytest.mark.parametrize(
    ("reset_mode", "source_reset"),
    [
        ("batch_barrier_v1", "batch_episode"),
        ("per_env_autoreset_v1", "per_env_rollout_boundary"),
    ],
)
def test_native_writer_checkpoint_reaches_inference_watch_and_play(
    tmp_path: Path, reset_mode: str, source_reset: str
):
    checkpoint = _native_checkpoint(tmp_path, reset_mode)
    assert (
        InferenceAgent.from_checkpoint(checkpoint, device=torch.device("cpu")).obs_spec
        == RASTER31V3
    )

    for mode in ("watch", "play"):
        receipt = serving_episode.run_serving_episode(_spec(checkpoint, mode=mode, frames=2))
        contract = receipt["serving_contract"]
        manifest = contract["deployment_target_manifest"]
        assert receipt["candidate_sha256"] == _digest(checkpoint)
        assert contract["episode_lifecycle_compatibility"] is None
        assert manifest["source_runtime"]["reset_strategy"] == source_reset
        assert manifest["deployed_runtime"]["reset_strategy"] == "manual"
        for name in (
            "episode_lifecycle_contract",
            "episode_lifecycle_contract_digest",
            "policy_source_contract",
            "policy_source_contract_digest",
        ):
            assert contract[name] == manifest[name]


def test_actual_watch_and_play_observe_seeded_real_dispatch(checkpoint: Path):
    watch = serving_episode.run_serving_episode(_spec(checkpoint, mode="watch"))
    play = serving_episode.run_serving_episode(_spec(checkpoint, mode="play"))

    for receipt in (watch, play):
        assert receipt["schema_version"] == "strict-serving-episode/v3"
        assert receipt["status"] == "completed"
        assert receipt["error"] is None
        assert receipt["frames_completed"] == 3
        assert receipt["end_frame"] - receipt["start_frame"] == 3
        assert receipt["initial_layout_digest"] == runtime_contract.canonical_digest(
            receipt["initial_layout"]
        )
        assert receipt["candidate_sha256"] == _digest(checkpoint)
        assert receipt["seed_application"]["global_stream_seed"] == derive_seed(
            receipt["serving_seed"], "global"
        )
        assert receipt["hero_lifecycle"]["dead_to_alive_transitions"] == 0
        assert receipt["hero_lifecycle"]["game_instance_change_count"] == 0
        assert all(
            row["calls"] > 0
            for row in receipt["dispatch_observation"]["candidate_action_context_calls"]
        )
        assert receipt["dispatch_observation"]["context_id_mismatch_count"] == 0
        assert receipt["dispatch_observation"]["unknown_context_id_count"] == 0
        assert receipt["serving_contract"]["checkpoint_sha256"] == receipt["candidate_sha256"]

    assert watch["completion"]["reason"] == "external-horizon"
    assert watch["seed_application"]["evaluated_build_hook"] == "GameSession.__init__"
    assert watch["dispatch_observation"]["human_control_events"] == []
    assert watch["dispatch_observation"]["human_update_calls"] == 0
    assert all(row["controller"] == "candidate_ai" for row in watch["participants"])

    assert play["seed_application"]["evaluated_build_hook"] == "GameSession.set_mode(play)"
    assert len(play["dispatch_observation"]["human_control_events"]) == 1
    assert play["dispatch_observation"]["human_update_calls"] == 3
    assert play["participants"][0]["controller"] == "human"
    assert play["participants"][0]["checkpoint_sha256"] is None
    assert all(row["controller"] == "candidate_ai" for row in play["participants"][1:])


def test_seed_is_applied_at_the_actual_evaluated_build_hook(checkpoint: Path, monkeypatch):
    events: list[str] = []
    original_seed = serving_episode.initialize_run_seed
    original_session = serving_episode.GameSession
    original_set_mode = original_session.set_mode

    def observe_seed(seed: int):
        events.append("seed")
        return original_seed(seed)

    def observe_constructor(*args, **kwargs):
        events.append("constructor")
        return original_session(*args, **kwargs)

    def observe_set_mode(self, mode, *args, **kwargs):
        events.append(f"mode:{mode}")
        return original_set_mode(self, mode, *args, **kwargs)

    monkeypatch.setattr(serving_episode, "initialize_run_seed", observe_seed)
    monkeypatch.setattr(serving_episode, "GameSession", observe_constructor)
    monkeypatch.setattr(original_session, "set_mode", observe_set_mode)
    serving_episode.run_serving_episode(_spec(checkpoint, mode="watch", frames=1))
    assert events[:2] == ["seed", "constructor"]

    events.clear()
    serving_episode.run_serving_episode(_spec(checkpoint, mode="play", frames=1))
    assert events[:3] == ["constructor", "seed", "mode:play"]


def test_same_seed_reproduces_layout_and_a_changed_seed_changes_it(checkpoint: Path):
    first = serving_episode.run_serving_episode(_spec(checkpoint, mode="play", seed=71, frames=1))
    second = serving_episode.run_serving_episode(_spec(checkpoint, mode="play", seed=71, frames=1))
    changed = serving_episode.run_serving_episode(_spec(checkpoint, mode="play", seed=72, frames=1))

    assert first["initial_layout_digest"] == second["initial_layout_digest"]
    assert first["initial_layout_digest"] != changed["initial_layout_digest"]


def test_rejects_shifted_raster_context_identity(checkpoint: Path, monkeypatch):
    from web.backend.raster_policy import RasterServingPolicy

    original = RasterServingPolicy.action_context_for

    def shifted(self, snake_id):
        context = original(self, snake_id)
        return replace(context, snake_id=int(snake_id) + 1000)

    monkeypatch.setattr(RasterServingPolicy, "action_context_for", shifted)
    with pytest.raises(RuntimeError, match="identity"):
        serving_episode.run_serving_episode(_spec(checkpoint, mode="watch", frames=1))


def test_actual_play_never_claims_or_writes_a_user_score(
    checkpoint: Path, tmp_path: Path, monkeypatch
):
    from web.backend.session import GameSession

    score_path = tmp_path / "user-scores.db"
    monkeypatch.setenv("SNAKE_SCORES_DB", str(score_path))

    def forbidden_claim(self):
        raise AssertionError("serving evidence must not claim a user score")

    monkeypatch.setattr(GameSession, "claim_submission", forbidden_claim)
    receipt = serving_episode.run_serving_episode(_spec(checkpoint, mode="play", frames=1))
    assert receipt["status"] == "completed"
    assert not score_path.exists()


def test_actual_play_human_terminal_keeps_score_store_isolated(
    checkpoint: Path, tmp_path: Path, monkeypatch
):
    """The real public input reaches a terminal human run without submission."""
    from web.backend.session import GameSession

    score_path = tmp_path / "terminal-user-scores.db"
    monkeypatch.setenv("SNAKE_SCORES_DB", str(score_path))

    def forbidden_claim(self):
        raise AssertionError("serving evidence must not claim a user score")

    monkeypatch.setattr(GameSession, "claim_submission", forbidden_claim)
    receipt = serving_episode.run_serving_episode(
        _spec(checkpoint, mode="play", seed=31, frames=120)
    )
    assert receipt["status"] == "completed"
    assert receipt["completion"] == {
        "owner": "GameSession._step_play",
        "reason": "human-terminal",
        "backend_run_over": True,
        "backend_run_frames": receipt["frames_completed"] - 1,
    }
    assert receipt["hero_lifecycle"]["alive_to_dead_transitions"] == 1
    assert receipt["hero_lifecycle"]["dead_to_alive_transitions"] == 0
    assert receipt["hero_lifecycle"]["final_alive"] is False
    assert receipt["hero_lifecycle"]["game_instance_change_count"] == 0
    assert not score_path.exists()


def test_create_only_writer_and_failure_artifact_preserve_attempt(checkpoint: Path, tmp_path: Path):
    spec = _spec(checkpoint, mode="play", frames=1)
    receipt_path = tmp_path / "episode.json"
    assert serving_episode.write_serving_episode_receipt(receipt_path, spec) == 0
    original = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        serving_episode.write_serving_episode_receipt(receipt_path, spec)
    assert receipt_path.read_bytes() == original

    bad_args = [
        "--episode-id",
        "bad",
        "--mode",
        "watch",
        "--serving-seed",
        "1",
        "--checkpoint",
        str(checkpoint),
        "--candidate-sha256",
        "0" * 64,
        "--profile-digest",
        "a" * 64,
        "--source-closure-sha256",
        "b" * 64,
        "--out",
        str(tmp_path / "bad.json"),
        "--frame-limit",
        "1",
    ]
    assert serving_episode.main(bad_args) == 1
    failure = tmp_path / "bad.json"
    assert failure.exists()
    assert "RuntimeError" in failure.read_text(encoding="utf-8")
    assert '"status":"failed"' in failure.read_text(encoding="utf-8")


def test_writer_preflights_existing_path_before_constructing_a_world(
    checkpoint: Path, tmp_path: Path, monkeypatch
):
    path = tmp_path / "already-exists.json"
    path.write_text("prior receipt", encoding="utf-8")
    spec = _spec(checkpoint, mode="watch", frames=1)

    def no_world(_spec):
        raise AssertionError("existing output must not consume a serving world")

    monkeypatch.setattr(serving_episode, "run_serving_episode", no_world)
    with pytest.raises(FileExistsError):
        serving_episode.write_serving_episode_receipt(path, spec)
    assert path.read_text(encoding="utf-8") == "prior receipt"


def test_invalid_preflight_spec_does_not_claim_an_episode(checkpoint: Path, tmp_path: Path):
    out = tmp_path / "invalid.json"
    args = [
        "--episode-id",
        "invalid",
        "--mode",
        "invalid-mode",
        "--serving-seed",
        "-1",
        "--checkpoint",
        str(checkpoint),
        "--candidate-sha256",
        "bad",
        "--profile-digest",
        "a" * 64,
        "--source-closure-sha256",
        "b" * 64,
        "--out",
        str(out),
        "--frame-limit",
        "0",
    ]
    assert serving_episode.main(args) != 0
    assert not out.exists()
