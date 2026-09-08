"""Actual public checkpoint load/save oracles for fresh-runtime Apex resumes."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from src.core.game_config import GameConfig
from src.game.snake_factory import SnakeFactory
from src.main import (
    load_checkpoint_into_game_state,
    run_learning_health_smoke,
    save_training_checkpoint,
)
from src.scripts.apex_train import load_validated_apex_resume_checkpoint
from src.scripts.offline_train import load_checkpoint
from src.training.apex_policy import ApexPolicy
from src.training.resume_lineage import load_checkpoint_snapshot

pytestmark = pytest.mark.usefixtures("setup_config")


def _policy(seed: int) -> ApexPolicy:
    return ApexPolicy(
        GameConfig.INPUT_SIZE,
        GameConfig.HIDDEN_SIZE,
        6,
        device=torch.device("cpu"),
        seed_context={"requested_seed": None, "effective_seed": seed, "namespace": "test"},
    )


@pytest.mark.parametrize("mode", ["weights-only", "continuation"])
@pytest.mark.parametrize("entrypoint", ["main", "offline", "snake"])
def test_public_resume_save_retains_parent_and_fresh_seed(tmp_path, monkeypatch, mode, entrypoint):
    source = _policy(789)
    source.total_reward = 123.0
    checkpoint = source.get_state_dict()
    checkpoint["run_seed_manifest"] = {"requested_seed": None, "effective_seed": 789}
    checkpoint["memories"] = [("archival", "must not enter new replay")]
    checkpoint["frame"] = 999
    checkpoint["episode_reward"] = 9999.0
    source_path = tmp_path / "source.pth"
    torch.save(checkpoint, source_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    receiver = _policy(987)
    frame = [0]
    snake = SnakeFactory.create_ai_snake(
        snake_id=0,
        color=(1, 2, 3),
        start_pos=(100, 100),
        policy=receiver,
        get_frame=lambda: frame[0],
        set_frame=lambda value: frame.__setitem__(0, value),
    )
    game = SimpleNamespace(_shared_policy=receiver, snakes=[snake])
    if entrypoint == "main":
        assert load_checkpoint_into_game_state(game, str(source_path), resume_mode=mode)
    elif entrypoint == "offline":
        assert load_checkpoint(receiver, str(source_path), resume_mode=mode)
    else:
        assert snake.load_state(str(source_path), resume_mode=mode)
    assert len(receiver.memory) == 0
    assert frame[0] == 0
    assert snake.total_reward == 0.0
    assert receiver.total_reward == (123.0 if mode == "continuation" else 0.0)
    monkeypatch.setattr("src.main.get_checkpoint_path", lambda name: tmp_path / name)
    saved = save_training_checkpoint(game, "descendant.pth")
    descendant = torch.load(saved, map_location="cpu", weights_only=False)
    parent = descendant["resume_parent"]
    assert parent["source_content_sha256"] == source_hash
    assert parent["source_run_seed_manifest"]["effective_seed"] == 789
    assert parent["source_recipe_runtime_seed_identity"]["effective_seed"] == 789
    assert parent["resume_mode"] == mode
    assert parent["rng_state_restored"] is False
    assert parent["replay_state_restored"] is False
    assert descendant["apex_recipe_runtime"]["seed_identity"]["effective_seed"] == 987
    assert descendant["memories"] == []
    assert descendant["episode_reward"] == 0.0
    assert descendant["total_reward"] == (123.0 if mode == "continuation" else 0.0)


def test_snapshot_binds_loaded_bytes_despite_path_replacement(tmp_path, monkeypatch):
    path = tmp_path / "rolling.pth"
    torch.save({"generation": 1}, path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    real_load = torch.load

    def replace_during_load(snapshot, **kwargs):
        # The reader has closed its source and will deserialize a private file.
        torch.save({"generation": 2}, path)
        return real_load(snapshot, **kwargs)

    monkeypatch.setattr(torch, "load", replace_during_load)
    checkpoint, parent = load_checkpoint_snapshot(path)
    assert checkpoint == {"generation": 1}
    assert parent["source_content_sha256"] == expected_hash


def test_distributed_weights_loader_parent_is_bound_to_snapshot(tmp_path):
    path = tmp_path / "legacy-online.pth"
    payload = {
        "dqn_state_dict": {"weight": torch.ones(1)},
        "run_seed_manifest": {"effective_seed": 789},
    }
    torch.save(payload, path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_validated_apex_resume_checkpoint(str(path), {}, resume_mode="weights-only")
    parent = loaded["_loaded_resume_parent"]
    assert parent["source_content_sha256"] == expected_hash
    assert parent["source_run_seed_manifest"] == {"effective_seed": 789}
    assert parent["resume_mode"] == "weights-only"


def test_health_smoke_checkpoint_carries_the_cli_seed_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("src.main.get_checkpoint_path", lambda name: tmp_path / name)
    identity = {"requested_seed": 0, "effective_seed": 0, "namespace": "global"}
    result = run_learning_health_smoke(
        max_frames=1, checkpoint_filename="smoke.pth", seed_identity=identity
    )
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["apex_recipe_runtime"]["seed_identity"] == identity
