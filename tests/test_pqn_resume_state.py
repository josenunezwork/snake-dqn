"""P1 resume modes keep run identity and mutable training state honest."""

import pytest
import torch

from src.model.obs_spec import RASTER31V3
from src.scripts import train_pqn
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def _config():
    return PQNConfig(num_envs=1, num_snakes=2, rollout_len=2, max_frames=20)


def test_legacy_checkpoint_requires_explicit_weights_only(tmp_path):
    path = tmp_path / "legacy.pth"
    legacy = PQNTrainer(_config()).checkpoint_state()
    del legacy["recipe"]
    torch.save(legacy, path)

    with pytest.raises(RuntimeError, match="continuation provenance"):
        train_pqn.load_pqn_resume_checkpoint(str(path), _config(), mode="continuation")

    blob = train_pqn.load_pqn_resume_checkpoint(str(path), _config(), mode="weights-only")
    assert blob is not None and blob["_p1_parent_hash"]


def test_exact_resume_is_rejected_and_weights_only_starts_fresh_world(tmp_path):
    original = PQNTrainer(_config())
    path = tmp_path / "checkpoint.pth"
    original.save_checkpoint(str(path))

    with pytest.raises(RuntimeError, match="exact PQN resume is unsupported"):
        train_pqn.load_pqn_resume_checkpoint(str(path), _config(), mode="exact")

    fresh = PQNTrainer(_config())
    blob = train_pqn.load_pqn_resume_checkpoint(str(path), fresh.cfg, mode="weights-only")
    train_pqn.apply_resume_checkpoint(fresh, blob, mode="weights-only")

    assert fresh.agent_steps == 0
    assert fresh.update_idx == 0
    assert fresh.checkpoint_state()["resume_mode"] == "weights-only"
    assert fresh.checkpoint_state()["resume_state"]["environment"] == "fresh"


@pytest.mark.parametrize("mode", ["weights-only", "continuation"])
@pytest.mark.parametrize("mutation", ["world", "episode", "rng"])
def test_both_resume_modes_require_a_pristine_runtime_before_mutation(tmp_path, mode, mutation):
    """Neither resume flavor may overwrite a trainer with hidden live state."""
    original = PQNTrainer(_config())
    path = tmp_path / "checkpoint.pth"
    original.save_checkpoint(str(path))
    target = PQNTrainer(_config())
    if mutation == "world":
        target.sim.frame[0] = 1
    elif mutation == "episode":
        target._episode_ids[0] = 1
    else:
        target.rng.random()

    blob = train_pqn.load_pqn_resume_checkpoint(str(path), target.cfg, mode=mode)
    with pytest.raises(ValueError, match="fresh trainer with pristine runtime"):
        train_pqn.apply_resume_checkpoint(target, blob, mode=mode)


def test_v3_checkpoint_continues_only_with_identical_world_and_optimizer(tmp_path):
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=2,
        max_frames=100,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
    )
    path = tmp_path / "v3.pth"
    PQNTrainer(config).save_checkpoint(str(path))

    assert train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="continuation")
    conflicting = PQNConfig(**{**config.__dict__, "max_frames": 120})
    with pytest.raises(RuntimeError, match="effective_world conflicts"):
        train_pqn.load_pqn_resume_checkpoint(str(path), conflicting, mode="continuation")


def test_derived_or_per_environment_mode_rejects_optimizer_continuation_before_loading(tmp_path):
    """Fresh worlds/episodes are intentional: only weights-only may cross this boundary."""
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=2,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        episode_reset_mode="per_env_autoreset_v1",
        episode_seed_mode="derived_env_episode_v1",
    )
    path = tmp_path / "derived.pth"
    PQNTrainer(config).save_checkpoint(str(path))
    with pytest.raises(RuntimeError, match="optimizer continuation is unsupported"):
        train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="continuation")
    assert train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="weights-only")


def test_resume_requires_real_counters_and_optimizer_group_settings(tmp_path):
    config = _config()
    state = PQNTrainer(config).checkpoint_state()
    state["agent_steps"] = -1
    path = tmp_path / "bad_counter.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="non-negative integer agent_steps"):
        train_pqn.load_pqn_resume_checkpoint(str(path), config, mode="continuation")
