"""Serving gates for the corrected raster31v3 checkpoint contract."""

import copy
from dataclasses import fields

import pytest
import torch

pytest.importorskip("fastapi")

from src.core.runtime_contract import EffectiveWorldConfig  # noqa: E402
from src.core.runtime_contract import ModelHeadContract  # noqa: E402
from src.core.runtime_contract import RunProvenance  # noqa: E402
from src.core.runtime_contract import RuntimeModeContract, canonical_digest  # noqa: E402
from src.model.inference_agent import InferenceAgent  # noqa: E402
from src.model.obs_spec import RASTER31V3  # noqa: E402
from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V3_CONTRACT  # noqa: E402
from src.model.raster_network import RasterDuelingNetwork  # noqa: E402
from src.simd_env.featurizer import build_observations  # noqa: E402
from src.simd_env.live_adapter import game_state_to_obs_inputs  # noqa: E402
from web.backend.raster_policy import RasterServingPolicy  # noqa: E402
from web.backend.session import V3_ACTION_MASK_CONTRACT, GameSession  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_config():
    """Do not leak the mechanics-v2 deployment profile to later test modules."""
    from src.core import game_config

    previous = game_config._current_config
    yield
    game_config._current_config = previous


def _v3_metadata(normalization=None) -> dict:
    normalization = normalization or {
        "max_frames": 5000.0,
        "starvation_max": 500.0,
        "max_length": 150.0,
    }
    world = EffectiveWorldConfig(
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
        normalization=normalization,
    )
    runtime = RuntimeModeContract(
        mode="train", training=True, respawn=False, hero_terminal=True, population_floor=True
    )
    head = ModelHeadContract("pqn", "dueling_q", 6)
    provenance = RunProvenance(
        effective_seed=7,
        observation_digest=RASTER31V3_CONTRACT.digest,
        world_digest=world.digest,
        runtime_digest=runtime.digest,
        reward_digest="reward",
        target_digest="target",
        sampler_digest="sampler",
        optimizer_digest="optimizer",
        model_head_digest=head.digest,
        source_revision="test",
    )
    metadata = {
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
        "action_mask_contract_digest": canonical_digest(V3_ACTION_MASK_CONTRACT),
        **provenance.to_metadata(),
    }
    return metadata


@pytest.fixture()
def v3_checkpoint(tmp_path):
    path = tmp_path / "v3.pth"
    net = RasterDuelingNetwork()
    torch.save({"dqn_state_dict": net.state_dict(), "output_size": 6, **_v3_metadata()}, path)
    return str(path)


def test_v3_loads_the_rectangular_mechanics_v2_serving_profile(v3_checkpoint):
    session = GameSession(checkpoint=v3_checkpoint)
    assert session.obs_spec == RASTER31V3
    assert session.config_path == "promotion-v2-watch-rect"
    assert session.serving_contract["obs_contract_digest"] == RASTER31V3_CONTRACT.digest
    assert len(session.game.snakes) == 6
    manifest = session.serving_contract["deployment_target_manifest"]
    assert manifest["deployed_world"]["engine"] == "live"
    assert manifest["deployed_world"]["max_capacity"] is None
    with pytest.raises(ValueError, match="food target"):
        session.set_food(42)
    assert session.game.food_manager.max_food == 300
    with pytest.raises(ValueError, match="Play roster"):
        session.set_play_opponents(2)
    profile_manifest = session.serving_contract["deployment_target_manifest"]
    profile_hero = session.hero_id
    with pytest.raises(ValueError, match="hero is pinned"):
        session.set_hero(session.game.snakes[1].id)
    assert session.hero_id == profile_hero
    assert not session.game.snakes[0].auto_respawn
    assert session.serving_contract["deployment_target_manifest"] == profile_manifest
    session.set_mode("play")
    assert len(session.game.snakes) == 6
    with pytest.raises(ValueError, match="hero is pinned"):
        session.set_hero(session.game.snakes[1].id)
    assert session.hero_id == session.human_id
    assert not session.game.snakes[0].auto_respawn
    assert (
        session.serving_contract["deployment_target_manifest"]["deployed_runtime"]["mode"] == "play"
    )


def test_v3_watch_hero_is_terminal_while_opponents_respawn(v3_checkpoint):
    session = GameSession(checkpoint=v3_checkpoint)
    hero, opponent = session.game.snakes[:2]
    hero.is_alive = False
    hero.respawn_timer = 0
    opponent.is_alive = False
    opponent.respawn_timer = 0
    session.game.update(train_mode=False, learn=False, allow_respawn=True)
    assert not hero.is_alive
    assert opponent.is_alive
    assert session.serving_contract["deployment_target_manifest"]["deployed_runtime"][
        "hero_terminal"
    ]


def test_v3_direct_raster_policy_requires_explicit_normalization(v3_checkpoint):
    agent = InferenceAgent.from_checkpoint(v3_checkpoint, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="explicit validated normalization"):
        RasterServingPolicy(agent)
    policy = RasterServingPolicy(
        agent, normalization={"max_frames": 5000, "starvation_max": 500, "max_length": 150}
    )
    assert policy.obs_spec == RASTER31V3


def test_bad_v3_descriptor_rejects_without_replacing_existing_session(v3_checkpoint, tmp_path):
    session = GameSession(checkpoint=v3_checkpoint)
    before_game = session.game
    before_path = session.checkpoint_path
    bad = tmp_path / "bad.pth"
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["obs_contract_digest"] = "bad"
    torch.save(blob, bad)

    with pytest.raises(ValueError, match="observation contract"):
        session._build(str(bad), mode="watch")
    assert session.game is before_game
    assert session.checkpoint_path == before_path


def test_corrupt_v3_weights_restore_global_config_and_existing_session(v3_checkpoint, tmp_path):
    """A failure after profile initialization is fully transactional."""
    from src.core.game_config import get_config

    session = GameSession(checkpoint=v3_checkpoint)
    before_game = session.game
    before_config = get_config()
    corrupt = tmp_path / "corrupt.pth"
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["dqn_state_dict"].pop("tactical_conv.0.weight")
    torch.save(blob, corrupt)

    with pytest.raises(RuntimeError):
        session._build(str(corrupt), mode="watch")
    assert get_config() is before_config
    assert session.game is before_game


def test_unknown_explicit_obs_spec_is_not_silently_treated_as_vector(tmp_path):
    path = tmp_path / "unknown.pth"
    torch.save({OBS_SPEC_KEY: "future-raster", "dqn_state_dict": {}}, path)
    with pytest.raises(ValueError, match="unknown obs_spec"):
        GameSession(checkpoint=str(path))


@pytest.mark.parametrize(
    "normalization",
    [
        {"max_frames": 1.0, "starvation_max": 1.0},
        {"max_frames": 0.0, "starvation_max": 1.0, "max_length": 1.0},
        {"max_frames": 1.5, "starvation_max": 1.0, "max_length": 1.0},
        {"max_frames": 1.0, "starvation_max": 1.0, "max_length": 1.0, "extra": 1.0},
    ],
)
def test_bad_v3_normalization_rejects_before_mutating_session(
    v3_checkpoint, tmp_path, normalization
):
    session = GameSession(checkpoint=v3_checkpoint)
    before_game = session.game
    from src.core.game_config import get_config

    before_config = get_config()
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["effective_world"]["normalization"] = normalization
    world = EffectiveWorldConfig(**blob["effective_world"])
    blob["effective_world_digest"] = world.digest
    provenance = RunProvenance.from_metadata(blob)
    blob["run_provenance"] = {**provenance.__dict__, "world_digest": world.digest}
    blob["run_provenance_digest"] = canonical_digest(blob["run_provenance"])
    path = tmp_path / "bad-normalization.pth"
    torch.save(blob, path)

    with pytest.raises(ValueError, match="normalization"):
        session._build(str(path), mode="watch")
    assert session.game is before_game
    assert get_config() is before_config


def test_v3_advisory_empty_uses_legal_boost_and_id_keyed_row(v3_checkpoint):
    """The human does not consume row 0 and an empty advisory keeps legal boost."""
    session = GameSession(checkpoint=v3_checkpoint)
    session.set_play_opponents(5)
    session.set_mode("play")
    human, first_ai, second_ai = session.game.snakes[:3]
    first_ai.length = 6
    first_ai._get_safe_actions = lambda *_args, **_kwargs: []

    class RowLogits:
        def __call__(self, tensors):
            q = torch.full((tensors["tactical"].shape[0], 6), -100.0)
            q[1, 4] = 100.0
            q[2, 2] = 100.0
            return q

    # Non-consuming inspector work before the update cannot shift either AI.
    assert session.policy.hero_observation(second_ai.id) is not None
    assert session.policy.hero_activations(first_ai.id) is not None
    session.policy.agent.network = RowLogits()
    session.game.update(train_mode=False, learn=False, allow_respawn=True)

    assert human.is_alive
    assert first_ai.last_action == 4
    assert second_ai.last_action == 2
    context = session.policy.action_context_for(first_ai.id)
    assert context.resolved.tolist() == [True] * 6


def test_v3_live_adapter_uses_checkpoint_normalization_bytes(tmp_path):
    """Serving must never silently fall back to live-adapter scalar defaults."""
    path = tmp_path / "custom-normalization.pth"
    net = RasterDuelingNetwork()
    normalization = {"max_frames": 1234.0, "starvation_max": 77.0, "max_length": 66.0}
    torch.save(
        {"dqn_state_dict": net.state_dict(), "output_size": 6, **_v3_metadata(normalization)}, path
    )
    session = GameSession(checkpoint=str(path))
    snake = session.game.snakes[0]
    actual = session.policy.hero_observation(snake.id)
    expected = build_observations(
        game_state_to_obs_inputs(session.game, max_frames=1234, starvation_max=77, max_length=66),
        obs_spec=RASTER31V3,
    )
    assert actual is not None
    assert actual["scalars"].tobytes() == expected["scalars"][0, 0].tobytes()


def test_v3_manifest_binds_actual_manual_runtime_and_external_horizon(v3_checkpoint):
    session = GameSession(checkpoint=v3_checkpoint)
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    for mode in ("watch", "play"):
        session.set_mode(mode)
        manifest = session.serving_contract["deployment_target_manifest"]
        assert manifest["source_runtime"] == blob["runtime_contract"]
        assert manifest["source_runtime_digest"] == blob["runtime_contract_digest"]
        assert (
            manifest["source_runtime_digest"] == session.serving_contract["runtime_contract_digest"]
        )
        assert manifest["source_runtime_digest"] == canonical_digest(manifest["source_runtime"])
        assert "frame_rate" not in manifest["distribution_differences"]
        deployed = manifest["deployed_runtime"]
        assert deployed == dict(
            mode=mode,
            training=False,
            respawn=True,
            hero_terminal=True,
            population_floor=False,
            reset_strategy="manual",
        )
        assert manifest["deployed_runtime_digest"] == canonical_digest(deployed)
        assert manifest["deployed_world_digest"] == canonical_digest(manifest["deployed_world"])
        assert manifest["source_world"]["max_frames"] == 5000
        assert "max_frames" not in manifest["deployed_world"]
        assert manifest["deployed_world"]["observation_progress_normalizer_frames"] == 5000
        assert manifest["episode_horizon_frames"] is None
        assert manifest["deployed_world"]["episode_horizon_frames"] is None
        assert manifest["evaluation_horizon_frames"] == 5000
        assert manifest["evaluation_horizon_owner"] == "external_evaluator"
        assert manifest["serving_enforced"] is False
        assert manifest["distribution_differences"]["runtime"] == {
            key: {"source": value, "deployed": deployed[key]}
            for key, value in blob["runtime_contract"].items()
            if deployed[key] != value
        }
        for prefix in ("source", "deployed"):
            assert manifest[f"{prefix}_normalization_digest"] == canonical_digest(
                manifest[f"{prefix}_normalization"]
            )
        # Reach the advertised evaluation horizon and prove live serving does
        # not implement that external cutoff or reset its world implicitly.
        game = session.game
        game.frame = 5000
        session.set_playing(True)
        session.run_started = True
        session.step()
        assert session.game is game
        assert game.frame == 5001


@pytest.mark.parametrize(
    "field,value", [("kill_scale", 0.3000000001), ("death_value", -3.000000001)]
)
def test_digest_valid_noncanonical_reward_rejects_before_mutation(
    v3_checkpoint, tmp_path, field, value
):
    from src.core.game_config import get_config

    session = GameSession(checkpoint=v3_checkpoint)
    before_game, before_config = session.game, get_config()
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["effective_world"][field] = value
    world = EffectiveWorldConfig(**blob["effective_world"])
    blob["effective_world_digest"] = world.digest
    provenance = RunProvenance.from_metadata(blob)
    blob["run_provenance"] = {**provenance.__dict__, "world_digest": world.digest}
    blob["run_provenance_digest"] = canonical_digest(blob["run_provenance"])
    path = tmp_path / "close-but-unequal.pth"
    torch.save(blob, path)
    with pytest.raises(ValueError, match="canonical kill_scale"):
        session._build(str(path), mode="watch")
    assert session.game is before_game
    assert get_config() is before_config


@pytest.mark.parametrize(
    "safe,length,expected",
    [
        ([], 4, [True] * 3 + [False] * 3),
        ([], 6, [True] * 6),
        ([2, 4], 4, [False, False, True, False, False, False]),
        ([2, 4], 6, [False, False, True, False, True, False]),
    ],
)
def test_v3_collected_successor_mask_matches_fresh_inspection(
    v3_checkpoint, safe, length, expected, monkeypatch
):
    session = GameSession(checkpoint=v3_checkpoint)
    snake = session.game.snakes[0]
    snake.length = length
    snake._get_safe_actions = lambda *_args, **_kwargs: [1]
    # Prime an action-time cache, then collect a distinct successor mask. A2
    # consumes this persisted tag alongside the mask; serving itself is forward-only.
    old = session.policy.action_context_for(snake.id)
    snake._get_safe_actions = lambda *_args, **_kwargs: safe
    assert old.resolved.tolist() != expected
    snake._pre_collision_state = snake.get_state(session.game.snakes, session.game.food)
    snake._pre_collision_action = 1
    monkeypatch.setattr(snake, "calculate_reward", lambda *_a, **_kw: 0.0)
    snake.compute_reward_and_train(session.game.snakes, session.game.food)
    assert snake.last_next_action_mask.tolist() == expected
    assert snake.last_next_action_mask_semantics == "raster_resolved_v3"
    session.policy._invalidate_cache()
    fresh = session.policy.action_context_for(snake.id)
    assert fresh.resolved.tolist() == expected
    assert session.policy.hero_observation(snake.id)["resolved_mask"].tolist() == expected
    assert old is not fresh
    snake._pre_collision_state = snake.last_next_state
    snake._pre_collision_action = 1
    snake.compute_reward_and_train(session.game.snakes, session.game.food, collided=True)
    assert snake.last_next_action_mask is None
    assert snake.last_next_action_mask_semantics == "terminal_no_successor"
    snake.soft_reset((200, 200))
    assert snake.last_next_action_mask is None
    assert snake.last_next_action_mask_semantics is None


@pytest.mark.parametrize(
    "missing",
    ["mode", "training", "respawn", "hero_terminal", "population_floor", "reset_strategy"],
)
def test_incomplete_resigned_runtime_rejects_before_session_mutation(
    v3_checkpoint, tmp_path, missing
):
    from src.core.game_config import get_config

    session = GameSession(checkpoint=v3_checkpoint)
    before_game, before_config = session.game, get_config()
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["runtime_contract"].pop(missing)
    runtime_digest = canonical_digest(blob["runtime_contract"])
    blob["runtime_contract_digest"] = runtime_digest
    provenance = RunProvenance.from_metadata(blob)
    blob["run_provenance"] = {**provenance.__dict__, "runtime_digest": runtime_digest}
    blob["run_provenance_digest"] = canonical_digest(blob["run_provenance"])
    path = tmp_path / "incomplete-runtime.pth"
    torch.save(blob, path)
    with pytest.raises(ValueError, match="runtime_contract"):
        session._build(str(path), mode="watch")
    assert session.game is before_game
    assert get_config() is before_config


@pytest.mark.parametrize("missing", [field.name for field in fields(EffectiveWorldConfig)])
def test_incomplete_resigned_world_rejects_before_session_mutation(
    v3_checkpoint, tmp_path, missing
):
    from src.core.game_config import get_config

    session = GameSession(checkpoint=v3_checkpoint)
    before_game, before_config = session.game, get_config()
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["effective_world"].pop(missing)
    try:
        world_digest = EffectiveWorldConfig(**blob["effective_world"]).digest
    except TypeError:
        # Required constructor fields already cannot be synthesized. They still
        # fail through the same explicit loader contract before any mutation.
        world_digest = blob["effective_world_digest"]
    blob["effective_world_digest"] = world_digest
    provenance = RunProvenance.from_metadata(blob)
    blob["run_provenance"] = {**provenance.__dict__, "world_digest": world_digest}
    blob["run_provenance_digest"] = canonical_digest(blob["run_provenance"])
    path = tmp_path / "incomplete-world.pth"
    torch.save(blob, path)
    with pytest.raises(ValueError, match="complete effective_world"):
        session._build(str(path), mode="watch")
    assert session.game is before_game
    assert get_config() is before_config


def test_v3_missing_model_head_default_is_not_synthesized(v3_checkpoint, tmp_path):
    session = GameSession(checkpoint=v3_checkpoint)
    before_game = session.game
    blob = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
    blob["model_head"].pop("critic_outputs")
    # The class would default this to zero, preserving its expanded digest.
    path = tmp_path / "incomplete-head.pth"
    torch.save(blob, path)
    with pytest.raises(ValueError, match="complete model_head"):
        session._build(str(path), mode="watch")
    assert session.game is before_game
